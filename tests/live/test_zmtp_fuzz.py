"""``mime serve``'s transport never stops for one peer's bytes (``mime.live._zmtp.Server``): whatever a peer sends, on
the commands' socket or the frames', the server's loop raises nothing and a well-behaved REQ beside it is served.

The streams are drawn as ZMTP is framed, so that they reach every parser: a greeting (ZMTP's own, one byte of it
changed, cut short, or none), then frames, each a command (READY, PING, SUBSCRIBE, CANCEL or another name, with
properties whose lengths may be cut short or run past the command), a message part with any flags, a frame declaring
more bytes than it holds or more than the socket takes (up to 2^62), or raw bytes; the whole cut at any point. The
known hostile peers of the review (a READY whose property is cut short, on either socket; a READY whose name length
overruns; a frame declaring 2^62 bytes) are examples run every time. The control: the transport as it was before the
review (its property reader unchecked, its read loop letting an error escape) stops on those examples, every run."""

import os
import socket
import struct
import threading

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mime.live import _zmtp
from mime.live._zmtp import GREETING, LONG, Client, Server, frame

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

