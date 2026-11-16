"""ZMQ's wire protocol, ZMTP 3.0 with the NULL mechanism (RFC 23), spoken by MIME itself: the two sockets ``mime
serve`` binds, a PUB for frames and a REP for commands, and the matching SUB and REQ for Python clients and the tests.
A libzmq peer (pyzmq, cppzmq, MICROROBOTICA's) connects as it would to libzmq's own sockets: the greeting, the READY
handshake with each side's socket type, and frames of one or more parts, the last without its MORE flag.

What it does not do: CURVE or PLAIN security (a session is reached through an SSH tunnel, which authenticates and
encrypts), ZMTP 3.1's heartbeats (a PING is answered with a PONG all the same), and any socket type but these four. A
PUB never waits on a subscriber: a subscriber that falls more than :data:`QUEUE_LIMIT` messages behind loses the oldest
it has not taken, as libzmq's high-water mark drops them.

The server never stops for one peer: a peer whose bytes are not ZMTP, that does not finish its handshake within
:data:`HANDSHAKE_SECONDS`, or that sends requests without reading the replies (:data:`PENDING_LIMIT`,
:data:`REPLY_BUFFER_LIMIT`) is closed, and the others are served on. A connection that arrives when the process has no
descriptor left is accepted on a spare and closed at once, and the listener rests a moment rather than spin.

**Every message received is held to its socket's limit** (:data:`LIMITS`), checked on each frame's header, before its
bytes are read (:class:`MessageTooLarge`): a request on the server's REP, a subscription on its PUB, a reply on a
client's REQ, a frame on a client's SUB. A request over it is answered with the server's coded refusal where the server
was given one (``too_large``, ``mime serve``'s ``live.message-too-large``) and its bytes are dropped as they arrive, so
the peer is served on; any other message over it closes the peer. A client raises :class:`MessageTooLarge`.

**A server binds loopback only** unless it is allowed otherwise, resolving ``localhost`` once and binding the address it
resolved (:func:`loopback_address`), and sets on its listeners only the options that keep a second server off their
ports on its platform (:func:`listener_options`)."""

from __future__ import annotations

import errno
import ipaddress
import itertools
import os
import selectors
import socket
import struct
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

SIGNATURE = b"\xff" + b"\x00" * 8 + b"\x7f"
#: Our greeting: the signature, version 3.0, the NULL mechanism, not as-server, and the filler.
GREETING = SIGNATURE + b"\x03\x00" + b"NULL".ljust(20, b"\x00") + b"\x00" + b"\x00" * 31
MORE, LONG, COMMAND = 0x01, 0x02, 0x04
#: The messages a subscriber may fall behind by before its oldest are dropped.
QUEUE_LIMIT = 1000
#: The peer socket types each of ours speaks with.
PEERS = {"PUB": {"SUB", "XSUB"}, "REP": {"REQ", "DEALER"}, "SUB": {"PUB", "XPUB"}, "REQ": {"REP", "ROUTER"}}
#: The most bytes a message (its parts together) may hold, by the socket that receives it. Measured through the real
#: session on the repository's rigs (docs/live.md, "Message sizes"): the largest command is 93 bytes (A0's ``set`` of
#: its live input at the longest float text), the largest ``hello`` 0.71 MB and the largest frame 5.3 MB (each a sheet
#: of 40,000 vertices with its points and two fields selected, the frame as JSON at the longest float text; 0.80 MB
#: binary). A request may be 700 times the largest command (a ``set`` of some 2,500 numbers), a reply 90 times the
#: largest ``hello`` and a frame 48 times the largest frame. A subscription is a topic, a byte for MIME's client.
REQUEST_LIMIT = 1 << 16
SUBSCRIPTION_LIMIT = 1 << 16
REPLY_LIMIT = 1 << 26
FRAME_LIMIT = 1 << 28
#: Each socket type's limit on what it receives.
LIMITS = {"REP": REQUEST_LIMIT, "PUB": SUBSCRIPTION_LIMIT, "REQ": REPLY_LIMIT, "SUB": FRAME_LIMIT}
#: The most parts a message may have: a request's envelope, its delimiter and its body.
PARTS_LIMIT = 64
#: The requests a peer may send ahead of their replies, and the reply bytes it may leave unread, before it is closed.
PENDING_LIMIT = 256
REPLY_BUFFER_LIMIT = 16 << 20
#: The seconds a peer has from connecting to finishing its handshake.
HANDSHAKE_SECONDS = 10.0
#: The seconds a listener rests when the process has no descriptor left for a new connection.
ACCEPT_REST_SECONDS = 0.1


#: What a peer of each of our socket types is called to a person watching: a REP's peer sends commands, a PUB's receives
#: frames.
_ROLE = {"REP": "commands", "PUB": "frames"}


def _address(sock: socket.socket) -> str:
    """Where a connection came from, as ``host:port``; empty if the connection is already gone."""
    try:
        peer = sock.getpeername()
    except OSError:
        return ""
    host, port = str(peer[0]), peer[1]
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


class ProtocolError(ConnectionError):
    """A peer that does not speak ZMTP 3 with the NULL mechanism, or a socket type ours cannot speak with."""


class MessageTooLarge(ProtocolError):
    """A message over its socket's limit, refused from the header of the frame that takes it over (``size``: the bytes
    of its parts so far and that frame's), before that frame's bytes are read. The reader drops its bytes, and those of
    the message's later parts, as they arrive, so the stream goes on with the next message."""

    code = "live.message-too-large"

    def __init__(self, size: int, limit: int, *, part: bool = True) -> None:
        what = "a message" if part else "a command"
        super().__init__(f"{what} of at least {size} bytes, over the {limit} this socket takes")
        self.size, self.limit, self.part = size, limit, part


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
    #: The most bytes a message (its parts together), or a command, may declare (``None``: no limit).
    limit: int | None = None
    #: The bytes of the message's parts so far, which the limit holds together with its next part's.
    held: int = 0
    #: The bytes still to drop of a message refused as too large, and whether parts of it are still to come.
    skip: int = 0
    skipping: bool = False

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

    def frames(self) -> Iterator[tuple[int, bytes]]:
        """Each frame whole in the buffer, taken out as it is returned. A frame that takes its message over the limit
        raises :class:`MessageTooLarge` from its header, before its bytes are read; they, and the message's later parts,
        are dropped as they arrive, so the next call goes on with the next message."""
        while True:
            if self.skip:
                dropped = min(self.skip, len(self.buffer))
                del self.buffer[:dropped]
                self.skip -= dropped
                if self.skip:
                    return
            if len(self.buffer) < 2:
                return
            flags = self.buffer[0]
            if flags & LONG:
                if len(self.buffer) < 9:
                    return
                (size,) = struct.unpack_from(">Q", self.buffer, 1)
                start = 9
            else:
                size, start = self.buffer[1], 2
            part = not flags & COMMAND
            if part and self.skipping:  # a later part of a message refused as too large
                del self.buffer[:start]
                self.skip, self.skipping = size, bool(flags & MORE)
                continue
            total = size + (self.held if part else 0)
            if self.limit is not None and total > self.limit:
                del self.buffer[:start]
                self.skip, self.skipping = size, part and bool(flags & MORE)
                if part:
                    self.held = 0
                raise MessageTooLarge(total, self.limit, part=part)
            if len(self.buffer) < start + size:
                return
            body = bytes(self.buffer[start:start + size])
            del self.buffer[:start + size]
            if part:
                self.held = self.held + size if flags & MORE else 0
            yield flags, body


@dataclass(frozen=True)
class PeerEvent:
    """A peer arriving or leaving, for a view of the server (``mime serve``'s): ``what`` is ``connected`` (its handshake
    finished), ``left`` (a peer that had connected, closed by either side) or ``refused`` (a peer that closed without
    finishing its handshake: not ZMTP, too slow, or a socket type this one cannot speak with); ``role`` is ``commands``
    (a requester) or ``frames`` (a subscriber); ``address`` is where its connection came from, which the transport gave
    and the peer did not write; ``dropped`` the frames it lost, for a subscriber that left."""

    what: str
    role: str
    address: str
    dropped: int = 0


#: The peer events a server holds for its caller to take before the oldest are dropped.
EVENT_LIMIT = 1024


@dataclass(frozen=True)
class _Refused:
    """A request over the limit, answered by the server's refusal in its turn: its envelope, its size and the limit."""

    envelope: list[bytes]
    size: int
    limit: int


@dataclass
class _Peer:
    sock: socket.socket
    kind: str
    #: Who the peer is to the server's caller: never reused, unlike its socket's descriptor, which the next connection
    #: takes as soon as this one closes.
    ident: int = 0
    reader: _Reader = field(default_factory=_Reader)
    out: bytearray = field(default_factory=bytearray)
    queue: deque[bytes] = field(default_factory=deque[bytes])
    sent_greeting: bool = False
    ready: bool = False
    topics: set[bytes] = field(default_factory=set[bytes])
    requests: deque[list[bytes] | _Refused] = field(default_factory=deque[list[bytes] | _Refused])
    envelope: list[bytes] | None = None
    dropped: int = 0
    opened: float = field(default_factory=time.monotonic)
    address: str = ""

    def wants(self) -> bool:
        return bool(self.out or self.queue)


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """``tcp://<host>:<port>`` as a host and a port (``*`` or ``0``: any free port); a port that is not a whole number
    from 0 to 65535 is refused."""
    if not endpoint.startswith("tcp://"):
        raise ValueError(f"{endpoint!r} is not a tcp:// endpoint")
    host, _, port = endpoint.removeprefix("tcp://").rpartition(":")
    host = host.strip("[]")
    if host == "*":
        host = "0.0.0.0"
    number = 0 if port in ("*", "") else int(port) if port.isdecimal() else -1
    if not 0 <= number <= 65535:
        raise ValueError(f"{endpoint!r} names the port {port!r}, not a whole number from 0 to 65535")
    return host, number


def loopback_address(host: str) -> str | None:
    """The address a socket asked to bind ``host`` binds when that is reached from this machine only: ``host`` itself,
    an address in 127.0.0.0/8 or ``::1``; for ``localhost``, the first IPv4 address it resolves to, where every one it
    resolves to is loopback (resolved once, here, so that the bind takes the address this checked); ``None`` for any
    other host. Any other name is not resolved: what it resolves to when bound may differ."""
    if host.lower() == "localhost":
        try:
            found = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
        except OSError:
            return None
        addresses = [str(info[4][0]) for info in found]
        return addresses[0] if addresses and all(ipaddress.ip_address(each).is_loopback for each in addresses) else None
    try:
        return host if ipaddress.ip_address(host).is_loopback else None
    except ValueError:
        return None


def is_loopback(host: str) -> bool:
    """Whether a socket bound to ``host`` is reached from this machine only (:func:`loopback_address`)."""
    return loopback_address(host) is not None


def listener_options(platform: str) -> tuple[str, ...]:
    """The socket options a listener sets to 1 on ``platform`` (as ``sys.platform`` names it), none of which lets a
    second server take its port. Linux: ``SO_REUSEADDR``, so that a server restarted on its port binds it while the
    last one's connections wait out TIME_WAIT; Linux refuses a second listener on an overlapping address with it all
    the same. macOS and the BSDs: none, since there ``SO_REUSEADDR`` lets a second socket bind a specific address
    (``127.0.0.1:P``) on a port a wildcard listener holds (``0.0.0.0:P``), and the connections to that address go to
    the second; a restart on the same port may then be refused until TIME_WAIT ends (``live.address-in-use``).
    Windows: none, since there ``SO_REUSEADDR`` lets a second socket bind the very address and port held."""
    return ("SO_REUSEADDR",) if platform.startswith("linux") else ()


class ListenError(OSError):
    """A socket of :class:`Server`'s that could not listen: which (``REP`` or ``PUB``), on what endpoint, and the
    system's error (its ``errno``, ``EADDRINUSE`` for a port in use); ``malformed`` for an endpoint that is no
    ``tcp://<host>:<port>``, and ``not_loopback`` for an address off loopback that the server was not allowed."""

    def __init__(self, kind: str, endpoint: str, error: OSError, *, malformed: bool = False,
                 not_loopback: bool = False) -> None:
        super().__init__(error.errno, error.strerror or str(error))
        self.kind, self.endpoint = kind, endpoint
        self.malformed, self.not_loopback = malformed, not_loopback


class Server:
    """The two sockets a live session is served on, in one thread: a PUB that sends each frame to every subscriber
    whose topic it begins with, and a REP that takes one request at a time from each peer and sends its reply back
    on the envelope it came on. :meth:`poll` moves bytes both ways and returns the requests that arrived, or returns
    early when another thread calls :meth:`wake`.

    Both sockets bind loopback (:func:`loopback_address`) unless ``allow_remote``; an endpoint off it, or malformed, is
    refused (:class:`ListenError`) with every socket opened so far closed. ``too_large`` makes the reply to a request
    over :data:`REQUEST_LIMIT`, from its size and the limit, sent on its envelope in its turn; without it the peer is
    closed."""

    def __init__(self, commands: str, frames: str, *, allow_remote: bool = False,
                 too_large: Callable[[int, int], bytes] | None = None) -> None:
        self.too_large = too_large
        self.selector = selectors.DefaultSelector()
        self.peers: dict[int, _Peer] = {}  # by socket descriptor, for the selector
        self.identified: dict[int, _Peer] = {}  # by identity, for the caller's replies
        self._identities = itertools.count(1)
        self.listeners: dict[int, tuple[socket.socket, str]] = {}
        #: Listeners resting after the process ran out of descriptors, with when they listen again.
        self.resting: dict[int, float] = {}
        self.endpoints: dict[str, str] = {}
        #: Who arrived and who left since the caller last took them (:meth:`take_events`).
        self.events: deque[PeerEvent] = deque(maxlen=EVENT_LIMIT)
        self._spare = _spare()
        self._waker, self._wakes = socket.socketpair()
        for end in (self._waker, self._wakes):
            end.setblocking(False)
        self.selector.register(self._waker, selectors.EVENT_READ)
        for kind, endpoint in (("REP", commands), ("PUB", frames)):
            listener: socket.socket | None = None
            try:
                try:
                    host, port = parse_endpoint(endpoint)
                except ValueError as error:
                    raise ListenError(kind, endpoint, OSError(errno.EINVAL, str(error)), malformed=True) from error
                address = loopback_address(host)
                if address is None and not allow_remote:
                    raise ListenError(kind, endpoint, OSError(errno.EACCES, f"{host or 'every interface'} is not a "
                                                                           "loopback address"), not_loopback=True)
                bound = address if address is not None else host
                family = socket.AF_INET6 if ":" in bound else socket.AF_INET
                listener = socket.socket(family, socket.SOCK_STREAM)
                for option in listener_options(sys.platform):
                    listener.setsockopt(socket.SOL_SOCKET, getattr(socket, option), 1)
                listener.bind((bound, port))
                listener.listen(16)
            except ListenError:
                self.close()
                raise
            except OSError as error:
                if listener is not None:
                    listener.close()
                self.close()
                raise ListenError(kind, endpoint, error) from error
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

    def take_events(self) -> list[PeerEvent]:
        """The peers that connected, left or were refused since the last call, oldest first."""
        taken = list(self.events)
        self.events.clear()
        return taken

    def counts(self) -> tuple[int, int]:
        """The peers connected now (handshake finished): requesters, then subscribers."""
        ready = [peer.kind for peer in self.peers.values() if peer.ready]
        return ready.count("REP"), ready.count("PUB")

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
            peer = _Peer(sock, kind, self._identify(sock), _Reader(
                limit=REQUEST_LIMIT if kind == "REP" else SUBSCRIPTION_LIMIT), address=_address(sock))
            self.selector.register(sock, selectors.EVENT_READ)
        except (OSError, ValueError):
            sock.close()
            return
        self.peers[sock.fileno()] = peer
        self.identified[peer.ident] = peer

    def _identify(self, sock: socket.socket) -> int:
        """A peer's identity: a count, never reused (its socket's descriptor is, by the next connection)."""
        return next(self._identities)

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
        if self.identified.pop(peer.ident, None) is not None:  # a peer is closed once, whoever closes it
            self.events.append(PeerEvent("left" if peer.ready else "refused", _ROLE[peer.kind], peer.address,
                                         peer.dropped))
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
            while True:
                try:
                    for flags, body in reader.frames():
                        self._frame(peer, flags, body)
                    return
                except MessageTooLarge as error:
                    self._refuse(peer, error)
        except Exception:  # noqa: BLE001 - a peer's bytes never stop the server: that peer is closed
            self._close(peer)

    def _refuse(self, peer: _Peer, error: MessageTooLarge) -> None:
        """A request over the limit, refused in its turn on its envelope, its bytes dropped as they arrive; any other
        message over its limit (a subscription, a command, or a request with no envelope yet) closes the peer."""
        parts, peer.reader.parts = peer.reader.parts, []
        delimiter = next((index for index, part in enumerate(parts) if part == b""), None)
        if peer.kind != "REP" or not peer.ready or self.too_large is None or delimiter is None or not error.part:
            raise error
        if len(peer.requests) >= PENDING_LIMIT:
            raise ProtocolError(f"more than {PENDING_LIMIT} requests sent ahead of their replies")
        peer.requests.append(_Refused(parts[:delimiter + 1], error.size, error.limit))

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
                if not peer.ready:
                    self.events.append(PeerEvent("connected", _ROLE[peer.kind], peer.address))
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
        if len(peer.reader.parts) > PARTS_LIMIT:
            raise ProtocolError(f"a message of more than {PARTS_LIMIT} parts")
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
        """The next request of each peer that is not waiting for a reply, with its envelope kept for the reply; a
        request refused as too large before it is answered here, in its turn."""
        found: list[tuple[int, bytes]] = []
        for peer in list(self.peers.values()):
            while peer.kind == "REP" and peer.envelope is None and peer.requests and peer.ident in self.identified:
                parts = peer.requests.popleft()
                if isinstance(parts, _Refused):
                    assert self.too_large is not None
                    peer.envelope = parts.envelope
                    self.reply(peer.ident, self.too_large(parts.size, parts.limit))  # closes one that reads no reply
                    continue
                delimiter = parts.index(b"")
                peer.envelope = parts[:delimiter + 1]
                found.append((peer.ident, b"".join(parts[delimiter + 1:])))
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
        """Answer the request the peer ``peer_id`` sent, on its envelope; a peer that has gone is not answered, and
        neither is whoever has its descriptor now."""
        peer = self.identified.get(peer_id)
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
        self.reader = _Reader(limit=LIMITS[kind])
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
        """The next frame; a message over this socket's limit raises :class:`MessageTooLarge`, its parts dropped."""
        while not self._pending:
            try:
                for held in self.reader.frames():
                    self._pending.append(held)
            except MessageTooLarge:
                while self._pending and self._pending[-1][0] & MORE and not self._pending[-1][0] & COMMAND:
                    self._pending.pop()  # the refused message's earlier parts
                raise
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


__all__ = ["ACCEPT_REST_SECONDS", "FRAME_LIMIT", "GREETING", "HANDSHAKE_SECONDS", "LIMITS", "PARTS_LIMIT",
           "PENDING_LIMIT", "QUEUE_LIMIT", "REPLY_BUFFER_LIMIT", "REPLY_LIMIT", "REQUEST_LIMIT", "SUBSCRIPTION_LIMIT", "Client",
           "ListenError", "MessageTooLarge", "PeerEvent", "ProtocolError", "Server", "frame", "is_loopback",
           "listener_options", "loopback_address", "message", "parse_endpoint", "ready"]
