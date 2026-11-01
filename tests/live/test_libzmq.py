"""MIME's own ZMTP (``mime.live._zmtp``) against libzmq itself (pyzmq, in the development group), both ways: a libzmq
REQ, SUB and DEALER drive a real ``mime serve`` (hello, start, frames decoded by the schema, set, a step answered when
its chunks are done, stop and the final frame; three requests pipelined on a DEALER, answered in order); and MIME's
REQ and SUB talk to a libzmq REP and PUB (requests of every length class echoed, frames received). The controls: a
frame of another schema is refused by the decoder, and a request libzmq's REP gets is the bytes MIME sent."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zmq
from live_cases import saved

from mime.live import PROTOCOL, FrameSchema
from mime.live._zmtp import Client

MIME = [sys.executable, "-m", "mime.cli"]
TIMEOUT_MS = 60_000


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("libzmq"))


def served(launch, folder: Path, tmp_path: Path) -> tuple[subprocess.Popen, dict]:
    """``mime serve`` in a process group of its own, stopped with it when the test ends (``launch``)."""
    ready = tmp_path / "ready.json"
    process = launch([*MIME, "serve", str(folder), "--commands", "tcp://127.0.0.1:0", "--frames",
                      "tcp://127.0.0.1:0", "--ready-file", str(ready), "--out", str(tmp_path / "out"), "--unpaced"],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 600  # a guard against a hang, not a measure of speed
    while not ready.exists() or not ready.read_text():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return process, json.loads(ready.read_text())


def asked(socket: zmq.Socket, command: str, **arguments) -> dict:
    socket.send_json({"protocol": PROTOCOL, "command": command, **arguments})
    return socket.recv_json()


def test_libzmq_peers_drive_mime_serve(launch, folder, tmp_path):
    process, endpoints = served(launch, folder, tmp_path)
    context = zmq.Context()
    try:
        frames = context.socket(zmq.SUB)
        # Heartbeats every 100 ms, each answered between chunks; the timeout is longer than the first chunk's compile,
        # during which the server moves no bytes (a shorter one drops the connection then, and libzmq reconnects).
        frames.setsockopt(zmq.HEARTBEAT_IVL, 100)
        frames.setsockopt(zmq.HEARTBEAT_TIMEOUT, 30_000)
        frames.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
        frames.setsockopt(zmq.SUBSCRIBE, b"")
        frames.connect(endpoints["frames"])
        commands = context.socket(zmq.REQ)
        commands.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
        commands.connect(endpoints["commands"])
        hello = asked(commands, "hello")
        assert hello["ok"] and hello["protocol"] == PROTOCOL
        schema = FrameSchema.from_record(hello["schema"])
        time.sleep(0.5)  # libzmq's subscription reaches the server before the first frame (its slow joiner)
        assert asked(commands, "start")["state"] == "running"
        received = [schema.decode(frames.recv()) for _ in range(20)]
        numbers = [held.number for held in received]
        assert numbers == list(range(numbers[0], numbers[0] + 20))
        assert all(held.step == held.number * hello["chunk_steps"] for held in received)
        assert asked(commands, "pause")["state"] == "paused"
        assert asked(commands, "set", values={"offset": 0.002})["ok"]
        stepped = asked(commands, "step", chunks=2)  # answered once both chunks are stepped
        assert stepped["ok"] and stepped["stepping"] == 0
        dealer = context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
        dealer.connect(endpoints["commands"])
        for _ in range(3):
            dealer.send_multipart([b"", json.dumps({"command": "status"}).encode()])
        answers = [json.loads(dealer.recv_multipart()[-1]) for _ in range(3)]
        assert [answer["step"] for answer in answers] == [stepped["step"]] * 3
        assert asked(commands, "stop")["state"] == "stopped"
        last = None
        while last is None or not last.final:
            last = schema.decode(frames.recv())
        assert last.step == stepped["step"] and process.wait(timeout=60) == 0
        # The control: a frame of another schema is refused, so the decoding above could fail.
        other = FrameSchema.of([("x", "y", np.zeros(3, dtype=np.float32), None)])
        with pytest.raises(ValueError):
            other.decode(schema.encode(0, 0, 0.0, received[0].values))
    finally:
        context.destroy(linger=0)


def test_mime_clients_talk_to_libzmq_sockets():
    context = zmq.Context()
    try:
        rep = context.socket(zmq.REP)
        rep.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
        port = rep.bind_to_random_port("tcp://127.0.0.1")
        pub = context.socket(zmq.PUB)
        frames_port = pub.bind_to_random_port("tcp://127.0.0.1")
        sizes = [0, 1, 255, 256, 65_536, 1 << 20]
        seen: list[bytes] = []

        def echo() -> None:
            for _ in sizes:
                held = rep.recv()
                seen.append(held)
                rep.send(b"echo:" + held)

        server = threading.Thread(target=echo, daemon=True)
        server.start()
        requester = Client(f"tcp://127.0.0.1:{port}", "REQ", timeout=60)
        for size in sizes:
            body = bytes(range(256)) * (size // 256) + bytes(size % 256)
            assert requester.request(body) == b"echo:" + body
        server.join(timeout=30)
        # The control: what libzmq's REP received is what MIME's REQ sent, byte for byte.
        assert [len(held) for held in seen] == sizes
        subscriber = Client(f"tcp://127.0.0.1:{frames_port}", "SUB", timeout=60)
        sending = threading.Event()

        def publish() -> None:
            while not sending.is_set():
                pub.send(b"frame:" + b"\x01" * 300_000)
                time.sleep(0.01)

        publisher = threading.Thread(target=publish, daemon=True)
        publisher.start()
        assert subscriber.receive() == b"frame:" + b"\x01" * 300_000
        sending.set()
        publisher.join(timeout=10)
        requester.close()
        subscriber.close()
    finally:
        context.destroy(linger=0)
