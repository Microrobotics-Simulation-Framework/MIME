"""ZMQ's wire protocol, ZMTP 3.0 with the NULL mechanism (RFC 23), spoken by MIME itself: the two sockets ``mime
serve`` binds, a PUB for frames and a REP for commands, and the matching SUB and REQ for Python clients and the tests.
A libzmq peer (pyzmq, cppzmq, MICROROBOTICA's) connects as it would to libzmq's own sockets: the greeting, the READY
handshake with each side's socket type, and frames of one or more parts, the last without its MORE flag.

What it does not do: CURVE or PLAIN security (a session is reached through an SSH tunnel, which authenticates and
encrypts), ZMTP 3.1's heartbeats (a PING is answered with a PONG all the same), and any socket type but these four. A
PUB never waits on a subscriber: a subscriber that falls more than :data:`QUEUE_LIMIT` messages behind loses the oldest
it has not taken, as libzmq's high-water mark drops them.

The server never stops for one peer: a peer whose bytes are not ZMTP, that does not finish its handshake within
:data:`HANDSHAKE_SECONDS`, that sends a message over its socket's limit (:data:`REQUEST_LIMIT` for a request, as a
command is a small JSON object; :data:`SUBSCRIPTION_LIMIT` for a subscription), or that sends requests without reading
the replies (:data:`PENDING_LIMIT`, :data:`REPLY_BUFFER_LIMIT`) is closed, and the others are served on. A connection
that arrives when the process has no descriptor left is accepted on a spare and closed at once, and the listener rests a
moment rather than spin."""

from __future__ import annotations

import errno
import ipaddress
import os
import selectors
import socket
import struct
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

SIGNATURE = b"\xff" + b"\x00" * 8 + b"\x7f"
#: Our greeting: the signature, version 3.0, the NULL mechanism, not as-server, and the filler.
GREETING = SIGNATURE + b"\x03\x00" + b"NULL".ljust(20, b"\x00") + b"\x00" + b"\x00" * 31
MORE, LONG, COMMAND = 0x01, 0x02, 0x04
#: The messages a subscriber may fall behind by before its oldest are dropped.
QUEUE_LIMIT = 1000
#: The peer socket types each of ours speaks with.
PEERS = {"PUB": {"SUB", "XSUB"}, "REP": {"REQ", "DEALER"}, "SUB": {"PUB", "XPUB"}, "REQ": {"REP", "ROUTER"}}
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
#: The most bytes a request (its parts together) may hold, and a subscriber's message: commands are small JSON objects,
#: a subscription a topic.
REQUEST_LIMIT = 1 << 20
SUBSCRIPTION_LIMIT = 1 << 16
#: The most parts a message may have: a request's envelope, its delimiter and its body.
PARTS_LIMIT = 64
#: The requests a peer may send ahead of their replies, and the reply bytes it may leave unread, before it is closed.
PENDING_LIMIT = 256
REPLY_BUFFER_LIMIT = 16 << 20
#: The seconds a peer has from connecting to finishing its handshake.
HANDSHAKE_SECONDS = 10.0
#: The seconds a listener rests when the process has no descriptor left for a new connection.
ACCEPT_REST_SECONDS = 0.1


class ProtocolError(ConnectionError):
    """A peer that does not speak ZMTP 3 with the NULL mechanism, or a socket type ours cannot speak with."""


def frame(body: bytes, *, more: bool = False, command: bool = False) -> bytes:
    flags = (MORE if more else 0) | (COMMAND if command else 0)
    if len(body) > 255:
        return bytes([flags | LONG]) + struct.pack(">Q", len(body)) + body
    return bytes([flags, len(body)]) + body


def message(parts: Sequence[bytes]) -> bytes:
    """A message of ``parts``, each but the last with its MORE flag."""
    return b"".join(frame(part, more=index < len(parts) - 1) for index, part in enumerate(parts))


def ready(socket_type: str) -> bytes:
    """The NULL mechanism's READY command, naming our socket type."""
    name, value = b"Socket-Type", socket_type.encode("ascii")
    body = bytes([5]) + b"READY" + bytes([len(name)]) + name + struct.pack(">I", len(value)) + value
    return frame(body, command=True)


def _properties(body: bytes) -> dict[str, bytes]:
    """A command's properties, each a name (a length byte, then the name) and a value (a four-byte length, then the
    value): refused when one is cut short or runs past the command."""
    found: dict[str, bytes] = {}
    at = 0
    while at < len(body):
        size = body[at]
        if at + 1 + size + 4 > len(body):
            raise ProtocolError("a command's property is cut short")
        name = body[at + 1:at + 1 + size].decode("ascii", "replace")
        at += 1 + size
        (length,) = struct.unpack_from(">I", body, at)
        if at + 4 + length > len(body):
            raise ProtocolError(f"a command's property {name!r} runs past the command")
        found[name] = body[at + 4:at + 4 + length]
        at += 4 + length
    return found


@dataclass
class _Reader:
    """A peer's bytes as they arrive, parsed into its greeting, its commands and its messages."""

    buffer: bytearray = field(default_factory=bytearray)
    greeted: bool = False
    signed: bool = False
    peer_type: str | None = None
    parts: list[bytes] = field(default_factory=list[bytes])
    #: The most bytes one frame may declare (``None``: no limit, a client's).
    limit: int | None = None

    def feed(self, data: bytes) -> None:
        self.buffer += data

    def signature(self) -> bool:
        """Whether the peer's signature has arrived (its first ten bytes), refusing one that is not ZMTP's."""
        if self.signed:
            return True
        if len(self.buffer) < 10:
            return False
        if self.buffer[0] != 0xFF or not self.buffer[9] & 0x01:
            raise ProtocolError("the peer does not speak ZMTP: its first bytes are not a ZMTP signature")
        self.signed = True
        return True

    def greeting(self) -> bool:
        if self.greeted:
            return True
        if len(self.buffer) < 64:
            return False
        held, self.buffer = bytes(self.buffer[:64]), self.buffer[64:]
        if held[10] < 3:
            raise ProtocolError(f"the peer speaks ZMTP {held[10]}.{held[11]}; this speaks 3")
        mechanism = held[12:32].rstrip(b"\x00")
        if mechanism != b"NULL":
            raise ProtocolError(f"the peer asks for the {mechanism.decode('ascii', 'replace')} mechanism; this offers "
                                "NULL (reach a remote session through an SSH tunnel)")
        self.greeted = True
        return True

    def frames(self) -> list[tuple[int, bytes]]:
        found: list[tuple[int, bytes]] = []
        while len(self.buffer) >= 2:
            flags = self.buffer[0]
            if flags & LONG:
                if len(self.buffer) < 9:
                    break
                (size,) = struct.unpack_from(">Q", self.buffer, 1)
                start = 9
            else:
                size, start = self.buffer[1], 2
            if self.limit is not None and size > self.limit:
                raise ProtocolError(f"a frame of {size} bytes, over the {self.limit} this socket takes")
            if len(self.buffer) < start + size:
                break
            found.append((flags, bytes(self.buffer[start:start + size])))
            del self.buffer[:start + size]
        return found


@dataclass
class _Peer:
    sock: socket.socket
    kind: str
    reader: _Reader = field(default_factory=_Reader)
    out: bytearray = field(default_factory=bytearray)
    queue: deque[bytes] = field(default_factory=deque[bytes])
    sent_greeting: bool = False
    ready: bool = False
    topics: set[bytes] = field(default_factory=set[bytes])
    requests: deque[list[bytes]] = field(default_factory=deque[list[bytes]])
    envelope: list[bytes] | None = None
    dropped: int = 0
    opened: float = field(default_factory=time.monotonic)

    def wants(self) -> bool:
        return bool(self.out or self.queue)


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """``tcp://<host>:<port>`` as a host and a port (``*`` or ``0``: any free port)."""
    if not endpoint.startswith("tcp://"):
        raise ValueError(f"{endpoint!r} is not a tcp:// endpoint")
    host, _, port = endpoint.removeprefix("tcp://").rpartition(":")
    host = host.strip("[]")
    if host == "*":
        host = "0.0.0.0"
    return host, 0 if port in ("*", "") else int(port)


def is_loopback(host: str) -> bool:
    if host in LOOPBACK:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Server:
    """The two sockets a live session is served on, in one thread: a PUB that sends each frame to every subscriber
    whose topic it begins with, and a REP that takes one request at a time from each peer and sends its reply back
    on the envelope it came on. :meth:`poll` moves bytes both ways and returns the requests that arrived, or returns
    early when another thread calls :meth:`wake`."""

    def __init__(self, commands: str, frames: str) -> None:
        self.selector = selectors.DefaultSelector()
        self.peers: dict[int, _Peer] = {}
        self.listeners: dict[int, tuple[socket.socket, str]] = {}
        #: Listeners resting after the process ran out of descriptors, with when they listen again.
        self.resting: dict[int, float] = {}
        self.endpoints: dict[str, str] = {}
        self._spare = _spare()
        self._waker, self._wakes = socket.socketpair()
        for end in (self._waker, self._wakes):
            end.setblocking(False)
        self.selector.register(self._waker, selectors.EVENT_READ)
        for kind, endpoint in (("REP", commands), ("PUB", frames)):
            host, port = parse_endpoint(endpoint)
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            listener = socket.socket(family, socket.SOCK_STREAM)
            if sys.platform != "win32":  # Windows' SO_REUSEADDR would let a second server take a port in use
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind((host, port))
            except OSError:
                listener.close()
                self.close()
                raise
            listener.listen(16)
            listener.setblocking(False)
            self.selector.register(listener, selectors.EVENT_READ)
            self.listeners[listener.fileno()] = (listener, kind)
            bound = listener.getsockname()
            shown = f"[{bound[0]}]" if family == socket.AF_INET6 else bound[0]
            self.endpoints[kind] = f"tcp://{shown}:{bound[1]}"

    def wake(self) -> None:
        """Make the :meth:`poll` in progress, or the next, return at once: the one call another thread may make (a
        chunk that has ended, on the thread that ran it)."""
        try:
            self._wakes.send(b"\0")
        except OSError:  # its buffer full (a wake is waiting already) or the server closed
            pass

    def _accept(self, listener: socket.socket, kind: str) -> None:
        try:
            sock, _ = listener.accept()
        except (BlockingIOError, InterruptedError):
            return
        except OSError as error:
            if error.errno in (errno.EMFILE, errno.ENFILE) and self._spare is not None:
                os.close(self._spare)  # the spare descriptor takes the connection, which is closed: the peer refused
                self._spare = None
                try:
                    listener.accept()[0].close()
                except OSError:
                    pass
                self._spare = _spare()
            self._rest(listener)
            return
        try:
            sock.setblocking(False)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            peer = _Peer(sock, kind, _Reader(limit=REQUEST_LIMIT if kind == "REP" else SUBSCRIPTION_LIMIT))
            self.selector.register(sock, selectors.EVENT_READ)
        except (OSError, ValueError):
            sock.close()
            return
        self.peers[sock.fileno()] = peer

    def _rest(self, listener: socket.socket) -> None:
        """Stop listening on ``listener`` for a moment, so that a listener the process cannot accept on does not
        spin the loop."""
        fileno = listener.fileno()
        if fileno in self.resting:
            return
        try:
            self.selector.unregister(listener)
        except (KeyError, ValueError):
            pass
        self.resting[fileno] = time.monotonic() + ACCEPT_REST_SECONDS

    def _housekeeping(self) -> None:
        """Listen again on rested listeners whose rest is over, and close each peer whose handshake ran out of
        time."""
        now = time.monotonic()
        for fileno, until in list(self.resting.items()):
            if now >= until:
                del self.resting[fileno]
                self.selector.register(self.listeners[fileno][0], selectors.EVENT_READ)
        for peer in list(self.peers.values()):
            if not peer.ready and now - peer.opened > HANDSHAKE_SECONDS:
                self._close(peer)

    def _close(self, peer: _Peer) -> None:
        self.peers.pop(peer.sock.fileno(), None)
        try:
            self.selector.unregister(peer.sock)
        except (KeyError, ValueError):
            pass
        peer.sock.close()

    def _read(self, peer: _Peer) -> None:
        try:
            data = peer.sock.recv(1 << 16)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._close(peer)
            return
        if not data:
            self._close(peer)
            return
        reader = peer.reader
        reader.feed(data)
        try:
            if not reader.signature():
                return
            if not peer.sent_greeting:
                peer.out += GREETING
                peer.sent_greeting = True
            if not reader.greeting():
                return
            for flags, body in reader.frames():
                self._frame(peer, flags, body)
        except Exception:  # noqa: BLE001 - a peer's bytes never stop the server: that peer is closed
            self._close(peer)

    def _frame(self, peer: _Peer, flags: int, body: bytes) -> None:
        if flags & COMMAND:
            size = body[0] if body else 0
            name = body[1:1 + size]
            if name == b"READY":
                peer_type = _properties(body[1 + size:]).get("Socket-Type", b"").decode("ascii", "replace")
                if peer_type not in PEERS[peer.kind]:
                    raise ProtocolError(f"a {peer_type} socket cannot speak with this {peer.kind}")
                peer.reader.peer_type = peer_type
                peer.out += ready(peer.kind)
                peer.ready = True
            elif name == b"PING":
                context = body[1 + size + 2:] if len(body) >= 3 + size else b""
                peer.out += frame(bytes([4]) + b"PONG" + context, command=True)
            elif name == b"SUBSCRIBE" and peer.kind == "PUB":
                peer.topics.add(body[1 + size:])
            elif name == b"CANCEL" and peer.kind == "PUB":
                peer.topics.discard(body[1 + size:])
            return
        if not peer.ready:
            raise ProtocolError("a message before the READY handshake")
        if peer.kind == "PUB":
            if body[:1] == b"\x01":
                peer.topics.add(body[1:])
            elif body[:1] == b"\x00":
                peer.topics.discard(body[1:])
            return
        peer.reader.parts.append(body)
        if len(peer.reader.parts) > PARTS_LIMIT or sum(map(len, peer.reader.parts)) > REQUEST_LIMIT:
            raise ProtocolError(f"a message of more than {PARTS_LIMIT} parts or {REQUEST_LIMIT} bytes")
        if not flags & MORE:
            parts, peer.reader.parts = peer.reader.parts, []
            delimiter = next((index for index, part in enumerate(parts) if part == b""), None)
            if delimiter is None:
                return  # a REP drops a request with no envelope, as libzmq's does
            if len(peer.requests) >= PENDING_LIMIT:
                raise ProtocolError(f"more than {PENDING_LIMIT} requests sent ahead of their replies")
            peer.requests.append(parts)

    def _write(self, peer: _Peer) -> None:
        while peer.queue and len(peer.out) < (1 << 20):
            peer.out += peer.queue.popleft()
        if not peer.out:
            return
        try:
            sent = peer.sock.send(peer.out)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._close(peer)
            return
        del peer.out[:sent]

    def _requests(self) -> list[tuple[int, bytes]]:
        """The next request of each peer that is not waiting for a reply, with its envelope kept for the reply."""
        found: list[tuple[int, bytes]] = []
        for fileno, peer in list(self.peers.items()):
            if peer.kind == "REP" and peer.envelope is None and peer.requests:
                parts = peer.requests.popleft()
                delimiter = parts.index(b"")
                peer.envelope = parts[:delimiter + 1]
                found.append((fileno, b"".join(parts[delimiter + 1:])))
        return found

    def poll(self, timeout: float) -> list[tuple[int, bytes]]:
        """Move bytes for up to ``timeout`` seconds (0: once, without waiting) and return each request that arrived,
        with the peer to reply to, at most one per peer until it is answered. A request already waiting is returned
        without waiting for more bytes."""
        deadline = time.monotonic() + max(0.0, timeout)
        found = self._requests()
        while True:
            self._housekeeping()
            for key, peer in list(self.peers.items()):
                if key in self.peers:
                    events = selectors.EVENT_READ | (selectors.EVENT_WRITE if peer.wants() else 0)
                    self.selector.modify(peer.sock, events)
            wait = 0.0 if found else max(0.0, deadline - time.monotonic())
            if self.resting:
                wait = min(wait, max(0.0, min(self.resting.values()) - time.monotonic()))
            if not self.selector.get_map():  # nothing to wait on (Windows' select refuses an empty set)
                time.sleep(wait)
                ready: list[tuple[selectors.SelectorKey, int]] = []
            else:
                ready = self.selector.select(wait)
            woken = False
            for key, events in ready:
                held = key.fileobj
                if held is self._waker:
                    woken = True
                    try:
                        while self._waker.recv(4096):
                            pass
                    except OSError:  # drained
                        pass
                    continue
                fileno = held.fileno() if not isinstance(held, int) else held
                if fileno in self.listeners:
                    self._accept(*self.listeners[fileno])
                    continue
                peer = self.peers.get(fileno)
                if peer is None:
                    continue
                if events & selectors.EVENT_READ:
                    self._read(peer)
                if events & selectors.EVENT_WRITE and fileno in self.peers:
                    self._write(peer)
            found += self._requests()
            if found or woken or time.monotonic() >= deadline:
                return found

    def reply(self, peer_id: int, body: bytes) -> None:
        """Answer the request ``peer_id`` sent, on its envelope."""
        peer = self.peers.get(peer_id)
        if peer is None or peer.envelope is None:
            return
        peer.out += message([*peer.envelope, body])
        peer.envelope = None
        if len(peer.out) > REPLY_BUFFER_LIMIT:  # a peer that sends requests and does not read the replies
            self._close(peer)
            return
        self._write(peer)

    def publish(self, body: bytes) -> None:
        """Send one single-part message to every subscriber whose topic it begins with; a subscriber too far behind
        loses its oldest message."""
        held = frame(body)
        for peer in list(self.peers.values()):
            if peer.kind != "PUB" or not peer.ready or not any(body.startswith(topic) for topic in peer.topics):
                continue
            if len(peer.queue) >= QUEUE_LIMIT:
                peer.queue.popleft()
                peer.dropped += 1
            peer.queue.append(held)
            self._write(peer)

    def flush(self, timeout: float) -> None:
        """Send what is queued, for up to ``timeout`` seconds: before the server closes, so that the last frame and
        the last reply leave."""
        deadline = time.monotonic() + timeout
        while any(peer.wants() for peer in self.peers.values()) and time.monotonic() < deadline:
            self.poll(0.01)

    def close(self) -> None:
        for peer in list(self.peers.values()):
            self._close(peer)
        for fileno, (listener, _) in self.listeners.items():
            if fileno not in self.resting:
                self.selector.unregister(listener)
            listener.close()
        self.listeners.clear()
        self.resting.clear()
        try:
            self.selector.unregister(self._waker)
        except (KeyError, ValueError):  # closed already
            pass
        self._waker.close()
        self._wakes.close()
        self.selector.close()
        if self._spare is not None:
            os.close(self._spare)
            self._spare = None


def _spare() -> int | None:
    """A descriptor held in reserve, to accept and close a connection with when the process has none left."""
    try:
        return os.open(os.devnull, os.O_RDONLY)
    except OSError:
        return None


class Client:
    """A blocking ZMTP 3.0 client socket: a REQ (:meth:`request`) or a SUB (:meth:`receive`, every topic)."""

    def __init__(self, endpoint: str, kind: str, *, timeout: float = 10.0) -> None:
        host, port = parse_endpoint(endpoint)
        self.kind = kind
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = _Reader()
        self._pending: deque[tuple[int, bytes]] = deque()
        self.sock.sendall(GREETING + ready(kind))
        while not (self.reader.signature() and self.reader.greeting()):
            self._fill()
        while True:
            flags, body = self._next()
            if flags & COMMAND and body[1:1 + body[0]] == b"READY":
                peer = _properties(body[1 + body[0]:]).get("Socket-Type", b"").decode("ascii", "replace")
                if peer not in PEERS[kind]:
                    raise ProtocolError(f"a {kind} cannot speak with a {peer}")
                break
        if kind == "SUB":
            self.sock.sendall(frame(b"\x01"))

    def _fill(self) -> None:
        data = self.sock.recv(1 << 16)
        if not data:
            raise ConnectionError("the server closed the connection")
        self.reader.feed(data)

    def _next(self) -> tuple[int, bytes]:
        while not self._pending:
            self._pending.extend(self.reader.frames())
            if not self._pending:
                self._fill()
        return self._pending.popleft()

    def _message(self) -> list[bytes]:
        parts: list[bytes] = []
        while True:
            flags, body = self._next()
            if flags & COMMAND:
                continue
            parts.append(body)
            if not flags & MORE:
                return parts

    def request(self, body: bytes) -> bytes:
        self.sock.sendall(message([b"", body]))
        parts = self._message()
        return b"".join(parts[parts.index(b"") + 1:]) if b"" in parts else b"".join(parts)

    def receive(self) -> bytes:
        return b"".join(self._message())

    def close(self) -> None:
        self.sock.close()


__all__ = ["ACCEPT_REST_SECONDS", "GREETING", "HANDSHAKE_SECONDS", "LOOPBACK", "PARTS_LIMIT", "PENDING_LIMIT",
           "QUEUE_LIMIT", "REPLY_BUFFER_LIMIT", "REQUEST_LIMIT", "SUBSCRIPTION_LIMIT", "Client", "ProtocolError", "Server",
           "frame", "is_loopback", "message", "parse_endpoint", "ready"]
