"""``mime serve``'s transport never stops for one peer's bytes (``mime.live._zmtp.Server``): whatever a peer sends, on
the commands' socket or the frames', the server's loop raises nothing and a well-behaved REQ beside it is served.

The streams are drawn as ZMTP is framed, so that they reach every parser: a greeting (ZMTP's own, one byte of it
changed, cut short, or none), then frames, each a command (READY, PING, SUBSCRIBE, CANCEL or another name, with
properties whose lengths may be cut short or run past the command), a message part with any flags, a frame declaring
more bytes than it holds or more than the socket takes (up to 2^62), or raw bytes; the whole cut at any point. The
known hostile peers of the review (a READY whose property is cut short, on either socket; a READY whose name length
overruns; a frame declaring 2^62 bytes) are examples run every time. The control: the transport as it was before the
review (its property reader unchecked, its read loop letting an error escape) stops on those examples, every run.

Every message received is held to its socket's limit, from the header of the frame that takes it over, before that
frame's bytes are read: a request over the server's limit is answered with the server's refusal in its turn, its bytes
dropped as they arrive, and the same peer is served on; the limit holds a message's parts together; a client refuses a
reply over its own limit. Before, a request under the old 1 MiB limit per frame was buffered whole, the parts were added
up only after each was held, and a client took any size."""

import os
import socket
import struct
import threading
import time

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mime.live import _zmtp
from mime.live._zmtp import GREETING, LONG, MORE, REQUEST_LIMIT, Client, MessageTooLarge, Server, frame, message

NAMES = [b"READY", b"PING", b"SUBSCRIBE", b"CANCEL", b"ERROR", b"", b"X" * 40]
SOCKET_TYPES = [b"REQ", b"DEALER", b"SUB", b"XSUB", b"PUB", b"REP", b"ROUTER", b"CLIENT", b""]


@st.composite
def a_property(draw) -> bytes:
    name = draw(st.sampled_from([b"Socket-Type", b"Identity", b"Resource", b""]) | st.binary(max_size=20))
    value = draw(st.sampled_from(SOCKET_TYPES) | st.binary(max_size=40))
    name_size = draw(st.sampled_from([len(name)]) | st.integers(0, 255))
    value_size = draw(st.sampled_from([len(value)]) | st.integers(0, 2 ** 32 - 1))
    held = bytes([name_size]) + name + struct.pack(">I", value_size) + value
    return held[:draw(st.integers(0, len(held)))] if draw(st.booleans()) else held


@st.composite
def a_command(draw) -> bytes:
    name = draw(st.sampled_from(NAMES))
    body = bytes([len(name)]) + name + b"".join(draw(st.lists(a_property(), max_size=3)))
    return frame(body, command=True)


@st.composite
def a_lying_frame(draw) -> bytes:
    flags = draw(st.integers(0, 255))
    declared = draw(st.sampled_from([2 ** 62, 2 ** 32, _zmtp.REQUEST_LIMIT + 1]) | st.integers(0, 4096))
    body = draw(st.binary(max_size=200))
    if flags & LONG:
        return bytes([flags]) + struct.pack(">Q", declared) + body
    return bytes([flags, declared % 256]) + body


a_part = st.builds(lambda body, more: frame(body, more=more), st.binary(max_size=300), st.booleans())
a_frame = st.one_of(a_command(), a_part, a_lying_frame(), st.binary(min_size=1, max_size=64))


@st.composite
def a_greeting(draw) -> bytes:
    choice = draw(st.sampled_from(["zmtp", "changed", "cut", "none", "random"]))
    if choice == "zmtp":
        return GREETING
    if choice == "changed":
        at = draw(st.integers(0, len(GREETING) - 1))
        return GREETING[:at] + bytes([draw(st.integers(0, 255))]) + GREETING[at + 1:]
    if choice == "cut":
        return GREETING[:draw(st.integers(0, len(GREETING) - 1))]
    return b"" if choice == "none" else draw(st.binary(max_size=80))


@st.composite
def a_stream(draw) -> bytes:
    held = draw(a_greeting()) + b"".join(draw(st.lists(a_frame, max_size=8)))
    return held[:draw(st.integers(0, len(held)))] if draw(st.booleans()) else held


def ready_cut_short(socket_type: bytes) -> bytes:
    """A READY whose one property is cut short (20 bytes of the command, the review's probe)."""
    body = bytes([5]) + b"READY" + bytes([11]) + b"Socket-Type" + struct.pack(">I", len(socket_type)) + socket_type
    return GREETING + frame(body[:20], command=True)


NAME_OVERRUNS = GREETING + frame(bytes([5]) + b"READY" + bytes([200]) + b"Socket-Type", command=True)
DECLARES_TOO_MUCH = GREETING + _zmtp.ready("REQ") + bytes([LONG]) + struct.pack(">Q", 2 ** 62) + b"\x00" * 64


def endpoint(server: Server, kind: str) -> tuple[str, int]:
    host, port = _zmtp.parse_endpoint(server.endpoints[kind])
    return host, port


def served(server: Server) -> bool:
    """Whether a well-behaved REQ gets its reply from ``server`` while the loop is run here."""
    nonce = b"ping-" + os.urandom(8).hex().encode()
    answers: list[bytes] = []

    def ask() -> None:
        client = Client(server.endpoints["REP"], "REQ", timeout=10)
        try:
            answers.append(client.request(nonce))
        finally:
            client.close()

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    for _ in range(2000):
        for peer, body in server.poll(0.005):
            if body == nonce:
                server.reply(peer, b"pong")
        if answers or not thread.is_alive():
            break
    thread.join(timeout=10)
    return answers == [b"pong"]


def survives(stream: bytes, kind: str) -> None:
    server = Server("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    hostile = socket.create_connection(endpoint(server, kind))
    try:
        hostile.sendall(stream)
        for _ in range(5):
            server.poll(0.0)
        assert served(server)
        hostile.close()
        for _ in range(3):
            server.poll(0.0)
    finally:
        hostile.close()
        server.close()


@settings(deadline=None)
@given(stream=a_stream(), kind=st.sampled_from(["REP", "PUB"]))
@example(stream=ready_cut_short(b"REQ"), kind="REP")
@example(stream=ready_cut_short(b"SUB"), kind="PUB")
@example(stream=NAME_OVERRUNS, kind="REP")
@example(stream=DECLARES_TOO_MUCH, kind="REP")
def test_the_server_survives_any_bytes_a_peer_sends(stream, kind):
    survives(stream, kind)


def unchecked_properties(body: bytes) -> dict[str, bytes]:
    """The property reader as it was before the review: no check that a property fits its command."""
    found: dict[str, bytes] = {}
    at = 0
    while at < len(body):
        size = body[at]
        name = body[at + 1:at + 1 + size].decode("ascii", "replace")
        at += 1 + size
        (length,) = struct.unpack_from(">I", body, at)
        found[name] = body[at + 4:at + 4 + length]
        at += 4 + length
    return found


def test_the_control_a_cut_short_property_reaches_the_reader_and_is_caught(monkeypatch):
    """Planted: the unchecked reader, and a read loop that lets its error escape, as the server was before the fix.
    The examples above then stop the server, on both sockets; restored, they are refused peer by peer."""
    monkeypatch.setattr(_zmtp, "_properties", unchecked_properties)
    def escaping(self, peer):
        data = peer.sock.recv(1 << 16)
        if not data:
            self._close(peer)
            return
        peer.reader.feed(data)
        if peer.reader.signature():
            if not peer.sent_greeting:
                peer.out += GREETING
                peer.sent_greeting = True
            if peer.reader.greeting():
                for flags, body in peer.reader.frames():
                    self._frame(peer, flags, body)

    monkeypatch.setattr(Server, "_read", escaping)
    for stream, kind in ((ready_cut_short(b"REQ"), "REP"), (ready_cut_short(b"SUB"), "PUB"), (NAME_OVERRUNS, "REP")):
        with pytest.raises(struct.error):
            survives(stream, kind)
    monkeypatch.undo()
    for stream, kind in ((ready_cut_short(b"REQ"), "REP"), (ready_cut_short(b"SUB"), "PUB"), (NAME_OVERRUNS, "REP")):
        survives(stream, kind)



class Pumped:
    """A server whose loop runs on a thread of its own, answering each request ``pong:<request>`` and each request over
    the limit ``refused:<size>``, so that the test's own thread can be a raw peer."""

    def __init__(self) -> None:
        self.server = Server("tcp://127.0.0.1:0", "tcp://127.0.0.1:0",
                             too_large=lambda size, limit: b"refused:%d" % size)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self) -> None:
        while not self.stopping.is_set():
            for peer, body in self.server.poll(0.01):
                self.server.reply(peer, b"pong:" + body)

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=10)


class RawRequester:
    """A REQ peer written by hand, so that it can send a frame's header without its bytes."""

    def __init__(self, server: Server) -> None:
        self.sock = socket.create_connection(endpoint(server, "REP"), timeout=10)
        self.reader = _zmtp._Reader()
        self.sock.sendall(GREETING + _zmtp.ready("REQ"))
        while not (self.reader.signature() and self.reader.greeting()):
            self.fill()
        assert self.parts() is None  # the server's READY

    def fill(self) -> None:
        data = self.sock.recv(1 << 16)
        if not data:
            raise ConnectionError("the server closed the connection")
        self.reader.feed(data)

    def parts(self) -> list[bytes] | None:
        """The next message's parts, or None for a command."""
        held: list[bytes] = []
        while True:
            for flags, body in self.reader.frames():
                if flags & _zmtp.COMMAND:
                    return None
                held.append(body)
                if not flags & MORE:
                    return held
            self.fill()


def test_a_request_over_the_limit_is_refused_from_its_header_its_bytes_dropped_and_the_peer_served_on():
    pumped = Pumped()
    peer = RawRequester(pumped.server)
    try:
        over = REQUEST_LIMIT + 1
        peer.sock.sendall(frame(b"", more=True) + bytes([LONG]) + struct.pack(">Q", over))  # the header, no bytes
        assert peer.parts() == [b"", b"refused:%d" % over], "refused from the header, before any of its bytes came"
        peer.sock.sendall(b"\0" * over + message([b"", b"next"]))  # its bytes, dropped as they arrive; then a request
        assert peer.parts() == [b"", b"pong:next"], "the same peer is served on"
        assert served_beside(pumped.server)
    finally:
        pumped.stop()
        peer.sock.close()
    assert max((len(held.reader.buffer) for held in pumped.server.peers.values()), default=0) < 1 << 16
    pumped.server.close()


def served_beside(server: Server) -> bool:
    """Whether a well-behaved REQ is answered by a server whose loop runs elsewhere."""
    client = Client(server.endpoints["REP"], "REQ", timeout=10)
    try:
        return client.request(b"hi") == b"pong:hi"
    finally:
        client.close()


def test_the_limit_holds_a_messages_parts_together_and_refuses_the_part_over_it_before_reading_it():
    reader = _zmtp._Reader(limit=100)
    reader.feed(frame(b"a" * 60, more=True) + frame(b"b" * 60)[:2])  # the second part's header alone
    frames = reader.frames()
    assert next(frames) == (MORE, b"a" * 60)
    with pytest.raises(MessageTooLarge) as refused:
        next(frames)
    assert (refused.value.size, refused.value.limit, len(reader.buffer)) == (120, 100, 0)
    reader.feed(b"b" * 60 + frame(b"c" * 60, more=True) + frame(b"x" * 300, more=True) + frame(b"d") + frame(b"ok"))
    with pytest.raises(MessageTooLarge):  # the next message's second part, of 300 bytes alone
        list(reader.frames())
    assert list(reader.frames()) == [(0, b"ok")], "the refused messages' bytes dropped, the next message whole"


def test_a_client_refuses_a_reply_over_its_limit_from_its_header():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    accepted: list[socket.socket] = []

    def lying_server() -> None:
        conn, _ = listener.accept()
        accepted.append(conn)
        conn.sendall(GREETING + _zmtp.ready("REP"))
        conn.recv(1 << 16)
        time.sleep(0.2)  # the request, whatever it is
        conn.sendall(frame(b"", more=True) + bytes([LONG]) + struct.pack(">Q", _zmtp.REPLY_LIMIT + 1))

    thread = threading.Thread(target=lying_server, daemon=True)
    thread.start()
    client = Client(f"tcp://127.0.0.1:{listener.getsockname()[1]}", "REQ", timeout=10)
    try:
        with pytest.raises(MessageTooLarge) as refused:
            client.request(b"hello")
        assert refused.value.limit == _zmtp.REPLY_LIMIT and refused.value.code == "live.message-too-large"
        assert len(client.reader.buffer) == 0
    finally:
        client.close()
        thread.join(timeout=10)
        for conn in accepted:
            conn.close()
        listener.close()
