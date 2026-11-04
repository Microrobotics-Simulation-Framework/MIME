"""Loopback servers for an adapter's tests: nothing leaves the machine.

:class:`LoopbackStore` is an HTTP server on a loopback address, with TLS where it is given a context. It serves the
files it is given, answers ``Range`` (and ``If-Range`` on the ``ETag`` it sends) unless told to ignore it, redirects
where told, refuses a file without the token it is told to want (401, or 403 for a wrong one), cuts a stream short once
where told, serves more bytes than a file has where told, answers a scripted run of statuses before a file (a busy
answer with the ``Retry-After`` it is told, ``0`` otherwise), and keeps every request it saw with its headers.
:class:`ConnectProxy` is an HTTP proxy that tunnels ``CONNECT`` to one loopback port, and counts what it tunnelled.
:func:`certificates` makes a certificate authority and a server certificate for the loopback names with ``openssl``, so
a test of an institution's CA bundle needs no package of its own.
"""

from __future__ import annotations

import hashlib
import http.server
import select
import shutil
import socket
import ssl
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, override


@dataclass(frozen=True)
class Seen:
    """A request the store saw: its method, path and headers (as sent: a test reads them to see what a host was given)."""

    method: str
    path: str
    headers: dict[str, str]


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def store(self) -> LoopbackStore:
        return cast(LoopbackStore, self.server)

    def do_GET(self) -> None:  # noqa: N802 - the standard library's name
        store = self.store
        headers = {key: value for key, value in self.headers.items()}
        store.seen.append(Seen("GET", self.path, headers))
        scripted = store.statuses.get(self.path)
        if scripted:
            self._answer(scripted.pop(0), b"")
            return
        target = store.redirects.get(self.path)
        if target is not None:
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        wanted = store.tokens.get(self.path)
        if wanted is not None:
            given = self.headers.get("Authorization")
            if given is None:
                self._answer(401, b"")
                return
            if given != f"Bearer {wanted}":
                self._answer(403, b"")
                return
        body = store.files.get(self.path)
        if body is None:
            self._answer(404, b"")
            return
        body = body + store.extra.get(self.path, b"")
        tag = '"' + hashlib.sha256(body).hexdigest()[:16] + '"'
        asked = self.headers.get("Range")
        condition = self.headers.get("If-Range")
        if asked and not store.ignore_range and (condition is None or condition == tag):
            first, _, last = asked.removeprefix("bytes=").partition("-")
            start = int(first)
            end = int(last) + 1 if last else len(body)
            part = body[start:end]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{start + len(part) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(part)))
            self.send_header("ETag", tag)
            self.end_headers()
            self.wfile.write(part)
            return
        cut = store.cuts.pop(self.path, None)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", tag)
        self.end_headers()
        if cut is not None:
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        self.wfile.write(body)

    def _answer(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status in (429, 503):
            self.send_header("Retry-After", self.store.retry_after.get(self.path, "0"))
        self.end_headers()
        self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the standard library's name
        pass


class LoopbackStore(http.server.ThreadingHTTPServer):
    """A store on ``host`` (a loopback address), over TLS where ``tls`` is given: see the module's docstring."""

    daemon_threads = True

    def __init__(self, host: str = "127.0.0.1", tls: ssl.SSLContext | None = None) -> None:
        super().__init__((host, 0), _Handler)
        self.host = host
        self.files: dict[str, bytes] = {}
        self.redirects: dict[str, str] = {}
        self.tokens: dict[str, str] = {}
        self.cuts: dict[str, int] = {}
        self.extra: dict[str, bytes] = {}
        self.statuses: dict[str, list[int]] = {}
        self.retry_after: dict[str, str] = {}
        self.ignore_range = False
        self.seen: list[Seen] = []
        if tls is not None:
            self.socket = tls.wrap_socket(self.socket, server_side=True)
        self.scheme = "https" if tls is not None else "http"
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def url(self, path: str, host: str | None = None) -> str:
        return f"{self.scheme}://{host or self.host}:{self.port}{path}"

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join()

    @override
    def __enter__(self) -> LoopbackStore:
        return self

    @override
    def __exit__(self, *exc: object) -> None:
        self.close()


class ConnectProxy:
    """An HTTP proxy on a loopback port that tunnels every ``CONNECT`` to ``target`` (a loopback port), whatever host it
    names, and records each host it was asked for."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.asked: list[str] = []
        self._socket = socket.create_server(("127.0.0.1", 0))
        self._closed = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._socket.getsockname()[1]}"

    def _serve(self) -> None:
        while not self._closed:
            try:
                client, _ = self._socket.accept()
            except OSError:
                return
            threading.Thread(target=self._tunnel, args=(client,), daemon=True).start()

    def _tunnel(self, client: socket.socket) -> None:
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = client.recv(4096)
            if not chunk:
                client.close()
                return
            head += chunk
        line = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        if len(line) < 2 or line[0] != "CONNECT":
            client.sendall(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")
            client.close()
            return
        self.asked.append(line[1])
        upstream = socket.create_connection(("127.0.0.1", self.target))
        client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        sockets = [client, upstream]
        try:
            while True:
                ready, _, _ = select.select(sockets, [], [], 5)
                if not ready:
                    break
                for each in ready:
                    data = each.recv(65536)
                    if not data:
                        return
                    (upstream if each is client else client).sendall(data)
        finally:
            client.close()
            upstream.close()

    def close(self) -> None:
        self._closed = True
        self._socket.close()


@dataclass(frozen=True)
class Certificates:
    """A certificate authority (``ca``, PEM) and a server certificate and key it signed for ``names``."""

    ca: Path
    certificate: Path
    key: Path
    names: tuple[str, ...] = field(default=())

    def server_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(self.certificate), str(self.key))
        return context


def certificates(folder: Path, names: tuple[str, ...] = ("localhost", "127.0.0.1")) -> Certificates:
    """A new certificate authority and a server certificate for ``names`` (host names, or IP addresses), made with
    ``openssl`` in ``folder``. Raises ``RuntimeError`` where ``openssl`` is not on the path: a test that needs a CA is
    then not run, and says so, rather than passing."""
    tool = shutil.which("openssl")
    if tool is None:
        raise RuntimeError("openssl is not on the path, and the CA bundle's test needs it to make its certificates")
    folder.mkdir(parents=True, exist_ok=True)
    ca_key, ca, key, request, certificate = (folder / name for name in
                                             ("ca.key", "ca.pem", "server.key", "server.csr", "server.pem"))
    subject = ",".join(f"{'IP' if name.replace('.', '').isdigit() else 'DNS'}:{name}" for name in names)
    extensions = folder / "server.ext"
    extensions.write_text(f"basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\n"
                          f"extendedKeyUsage=serverAuth\nsubjectAltName={subject}\n"
                          f"subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n", encoding="utf-8")

    def run(*arguments: str) -> None:
        subprocess.run([tool, *arguments], check=True, capture_output=True, timeout=60)

    run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(ca_key), "-out", str(ca), "-days", "2",
        "-subj", "/CN=MIME test CA", "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    run("req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(request), "-subj", f"/CN={names[0]}")
    run("x509", "-req", "-in", str(request), "-CA", str(ca), "-CAkey", str(ca_key), "-CAcreateserial", "-out",
        str(certificate), "-days", "2", "-extfile", str(extensions))
    return Certificates(ca, certificate, key, names)


__all__ = ["Certificates", "ConnectProxy", "LoopbackStore", "Seen", "certificates"]
