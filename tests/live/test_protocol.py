"""``mime.live/1``'s messages and MIME's own ZMTP, without a session: a frame's binary and JSON forms read back to the
same arrays in their own dtypes, a frame of another schema or length refused, the frame order checker seeing a frame
dropped or delivered out of order, and the handshakes libzmq's own sockets sent (captured from libzmq 4.3.5 through
pyzmq, ``libzmq_handshakes.json``) answered by MIME's server as libzmq's would answer them, with the controls: a peer
asking for another mechanism, or of a socket type a REP cannot speak with, is closed."""

import json
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from mime.live import FINAL, HEADER, PROTOCOL, Frame, FrameOrder, FrameSchema, frame_from_json, is_frame
from mime.live._zmtp import GREETING, Client, Server, frame, message, ready

CAPTURED = json.loads((Path(__file__).parent / "libzmq_handshakes.json").read_text())


def schema() -> FrameSchema:
    return FrameSchema.of([("umr", "pose", np.arange(7, dtype=np.float32), "m; 1"),
                           ("rpm.arm", "frame_poses", np.ones((27, 7), dtype=np.float32), "m; 1"),
                           ("swim_speed:umr", "speed", np.float32(0.5), "m/s"),
                           ("swim_speed:umr", "steps", np.int32(3), None),
                           ("drag:umr", "tilt_fired", np.float64(1.0), "1")])


def values(seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    return {"umr.pose": generator.standard_normal(7).astype(np.float32),
            "rpm.arm.frame_poses": generator.standard_normal((27, 7)).astype(np.float32),
            "swim_speed:umr.speed": np.float32(generator.standard_normal()),
            "swim_speed:umr.steps": np.int32(generator.integers(0, 1000)),
            "drag:umr.tilt_fired": np.float64(generator.standard_normal())}


def test_a_binary_frame_reads_back_bit_for_bit_in_its_own_dtypes():
    held = schema()
    sent = values(1)
    data = held.encode(4, 132, 0.066, sent, final=True)
    assert len(data) == held.frame_bytes == HEADER.size + held.payload_bytes and HEADER.size == 32 and is_frame(data)
    back = held.decode(data)
    assert (back.number, back.step, back.time, back.final) == (4, 132, 0.066, True)
    for path, value in sent.items():
        assert back.values[path].dtype == np.asarray(value).dtype and back.values[path].tobytes() == value.tobytes()


def test_a_json_frame_reads_back_the_same_values():
    held = schema()
    sent = values(2)
    message = json.loads(json.dumps(held.json_frame(7, 231, 0.1155, sent)))
    assert message["protocol"] == PROTOCOL and message["schema"] == held.id
    back = frame_from_json(message, held)
    assert all(back.values[path].tobytes() == value.tobytes() for path, value in sent.items())


def test_a_frame_of_another_schema_length_or_version_is_refused():
    held = schema()
    data = held.encode(0, 0, 0.0, values(3))
    other = FrameSchema.of([(spec.node, spec.field, np.zeros(spec.shape, spec.dtype), "other unit")
                            for spec in held.fields])
    assert other.id != held.id and other.frame_bytes == held.frame_bytes
    with pytest.raises(ValueError, match="schema"):
        other.decode(data)
    with pytest.raises(ValueError, match="bytes"):
        held.decode(data[:-1])
    with pytest.raises(ValueError, match="frame"):
        held.decode(b"XXXX" + data[4:])
    assert FrameSchema.from_record(held.record()).id == held.id
    with pytest.raises(ValueError, match="id"):
        FrameSchema.from_record({**held.record(), "id": held.id ^ 1})


def numbered(numbers):
    return [Frame(n, 33 * n, 0.0165 * n, {}) for n in numbers]


def test_the_order_checker_sees_every_dropped_and_every_reordered_frame():
    clean = FrameOrder()
    assert [clean.see(f) for f in numbered(range(50))] == [None] * 50 and clean.problems == []
    dropped = FrameOrder()
    for f in numbered([0, 1, 2, 4, 5]):
        dropped.see(f)
    assert dropped.problems == ["frames 3 to 3 were dropped before frame 4"]
    swapped = FrameOrder()
    for f in numbered([0, 1, 3, 2, 4]):
        swapped.see(f)
    assert len(swapped.problems) == 2 and "dropped" in swapped.problems[0] and "out of order" in swapped.problems[1]


def raw(endpoint: str) -> socket.socket:
    host, _, port = endpoint.removeprefix("tcp://").rpartition(":")
    held = socket.create_connection((host, int(port)), timeout=5)
    held.settimeout(5)
    return held


def received(server: Server, sock: socket.socket, at_least: int) -> bytes:
    data = b""
    deadline = time.monotonic() + 5
    while len(data) < at_least and time.monotonic() < deadline:
        server.poll(0.02)
        sock.setblocking(False)
        try:
            data += sock.recv(1 << 16)
        except BlockingIOError:
            pass
    return data


@pytest.fixture
def server():
    held = Server("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    yield held
    held.close()


def test_libzmqs_subscriber_is_greeted_and_then_sent_every_frame(server):
    sub = raw(server.endpoints["PUB"])
    sub.sendall(bytes.fromhex(CAPTURED["SUB"]))
    answer = received(server, sub, len(GREETING) + len(ready("PUB")))
    assert answer == GREETING + ready("PUB")
    server.poll(0.05)
    server.publish(b"MIMF-a-frame")
    assert received(server, sub, 2 + len(b"MIMF-a-frame")) == frame(b"MIMF-a-frame")
    sub.close()


def test_libzmqs_request_is_taken_and_answered_on_its_envelope(server):
    req = raw(server.endpoints["REP"])
    req.sendall(bytes.fromhex(CAPTURED["REQ"]))
    deadline, found = time.monotonic() + 5, []
    while not found and time.monotonic() < deadline:
        found = server.poll(0.05)
    assert [body for _, body in found] == [b'{"command": "hello"}']
    req.setblocking(True)
    assert req.recv(len(GREETING) + len(ready("REP"))) == GREETING + ready("REP")
    server.reply(found[0][0], b'{"ok": true}')
    assert received(server, req, 2 + 2 + 12) == message([b"", b'{"ok": true}'])
    req.close()


@pytest.mark.parametrize("planted", ["mechanism", "socket type"])
def test_a_peer_of_another_mechanism_or_socket_type_is_closed(server, planted):
    """The controls: the captured REQ handshake with its mechanism made PLAIN, or its socket type made PUSH."""
    data = bytes.fromhex(CAPTURED["REQ"])
    if planted == "mechanism":
        data = data[:12] + b"PLAIN".ljust(20, b"\x00") + data[32:]
    else:
        data = data.replace(b"\x00\x00\x00\x03REQ", b"\x00\x00\x00\x04PUSH")
        data = data[:65] + bytes([data[65] + 1]) + data[66:]
    req = raw(server.endpoints["REP"])
    req.sendall(data)
    deadline, closed, requests = time.monotonic() + 5, False, []
    while not closed and time.monotonic() < deadline:
        requests += server.poll(0.05)
        closed = not any(peer.kind == "REP" for peer in server.peers.values())
    assert closed and requests == []
    req.close()


def test_mimes_own_client_speaks_with_its_server(server):
    sub = Client(server.endpoints["PUB"], "SUB")
    req_ready: list[Client] = []
    server.poll(0.05)
    import threading

    thread = threading.Thread(target=lambda: req_ready.append(Client(server.endpoints["REP"], "REQ")))
    thread.start()
    while thread.is_alive():
        server.poll(0.02)
    req = req_ready[0]
    server.poll(0.05)
    for n in range(3):
        server.publish(f"frame {n}".encode())
    server.flush(1.0)
    assert [sub.receive() for _ in range(3)] == [b"frame 0", b"frame 1", b"frame 2"]
    answers: list[bytes] = []
    thread = threading.Thread(target=lambda: answers.append(req.request(b"ping")))
    thread.start()
    while thread.is_alive():
        for peer, body in server.poll(0.02):
            server.reply(peer, body + b" back")
    assert answers == [b"ping back"]
    sub.close()
    req.close()
