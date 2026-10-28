"""A local HTTP server in a thread, a cache directory of its own, and the environment pointing at both: the tests of
``mime.assets`` never leave the machine. The server counts every request it gets, so "no request" is a number."""

import hashlib
import http.server
import socket
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from mime.assets import AssetRef


class _Handler(http.server.BaseHTTPRequestHandler):
    server: "Server"

    def do_GET(self) -> None:  # noqa: N802
        self.server.requests.append(f"GET {self.path}")
        body = self.server.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class Server(http.server.ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.005}, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}{path}"

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join()


def dead_url(path: str = "/blob") -> str:
    """A location nothing listens at: a port that was free a moment ago."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}{path}"


def reference(payload: bytes, *locations: str, name: str = "blob") -> AssetRef:
    return AssetRef(name, hashlib.sha256(payload).hexdigest(), len(payload), "CC0-1.0", locations)


class Rig:
    """A server and a cache, with ``MIME_ASSETS_DIR`` set to the cache and ``MIME_OFFLINE`` unset."""

    def __init__(self, directory: Path, patch: pytest.MonkeyPatch) -> None:
        self.server = Server()
        self.cache = directory / "cache"
        patch.setenv("MIME_ASSETS_DIR", str(self.cache))
        patch.delenv("MIME_OFFLINE", raising=False)
        patch.setenv("NO_PROXY", "127.0.0.1,localhost")
        patch.setenv("no_proxy", "127.0.0.1,localhost")

    @property
    def requests(self) -> int:
        return len(self.server.requests)


def run_with_rig(scenario: Callable[[Rig], None], directory: Path) -> None:
    """``scenario`` on a fresh rig, closed afterwards, the environment restored."""
    with pytest.MonkeyPatch.context() as patch:
        rig = Rig(directory, patch)
        try:
            scenario(rig)
        finally:
            rig.server.close()
