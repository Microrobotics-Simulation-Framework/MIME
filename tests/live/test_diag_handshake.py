"""DIAGNOSTIC, NOT TO BE MERGED: the sequence of test_a_reply_held_across_a_chunk... repeated against one gated server, with
the server's closes and accepts logged, to find the cause of one ConnectionResetError in a new client's handshake."""

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from live_cases import saved

from mime.live import PROTOCOL, LiveClient
from mime.live._zmtp import message

MIME = [sys.executable, "-m", "mime.cli"]
HANG = 600
import os
ROUNDS = int(os.environ.get("DIAG_ROUNDS", "500"))

GATED = """
import pathlib, sys, time, traceback, threading, itertools, os
import mime.control._run as run
import mime.live._zmtp as zmtp

gate, mode, chunk = pathlib.Path(sys.argv[1]), sys.argv[2], run.ChunkedRun.run
logf = open(str(gate) + ".log", "a", buffering=1)

def log(*a):
    logf.write(f"{time.monotonic():.4f} " + " ".join(str(x) for x in a) + "\\n")

def held(self, *args, **kwargs):
    pathlib.Path(str(gate) + ".held").touch()
    while not gate.exists():
        sum(range(10_000))
    return chunk(self, *args, **kwargs)

if mode != "idle":
    run.ChunkedRun.run = held

if mode == "fd":
    def reused(self, sock):
        kind = next(kind for listener, kind in self.listeners.values()
                    if listener.getsockname()[1] == sock.getsockname()[1])
        slot = next(slot for slot in itertools.count() if 2 * slot + (kind == "PUB") not in self.identified)
        return 2 * slot + (kind == "PUB")
    zmtp.Server._identify = reused

orig_close = zmtp.Server._close
def _close(self, peer):
    where = "|".join(f"{f.name}:{f.lineno}" for f in traceback.extract_stack(limit=5)[:-1])
    log("close", peer.kind, peer.address, "fd", peer.sock.fileno(), "ready", peer.ready, "ident", peer.ident,
        "inbuf", len(peer.reader.buffer), "out", len(peer.out), "age", round(time.monotonic() - peer.opened, 3), where)
    orig_close(self, peer)
zmtp.Server._close = _close

orig_accept = zmtp.Server._accept
def _accept(self, listener, kind):
    before = set(self.identified)
    try:
        orig_accept(self, listener, kind)
    except BaseException:
        log("accept RAISED", kind, traceback.format_exc().replace("\\n", " | "))
        raise
    new = [p for i, p in self.identified.items() if i not in before]
    log("accept", kind, [(p.address, p.sock.fileno(), p.ident) for p in new], "peers", len(self.peers))
zmtp.Server._accept = _accept

orig_poll = zmtp.Server.poll
last_exit = [time.monotonic()]
def poll(self, timeout):
    entered = time.monotonic()
    if entered - last_exit[0] > 0.5:
        log("STALL between polls", round(entered - last_exit[0], 3))
    try:
        found = orig_poll(self, timeout)
        if time.monotonic() - entered > timeout + 0.5:
            log("STALL inside poll", round(time.monotonic() - entered - timeout, 3))
        last_exit[0] = time.monotonic()
        return found
    except BaseException:
        log("poll RAISED", traceback.format_exc().replace("\\n", " | "))
        raise
zmtp.Server.poll = poll

orig_read = zmtp.Server._read
def _read(self, peer):
    n = len(peer.reader.buffer)
    orig_read(self, peer)
    if peer.sock.fileno() == -1 and n == 0 and not peer.ready:
        log("read closed an unready peer", peer.kind, peer.address)
zmtp.Server._read = _read

import atexit
atexit.register(lambda: log("process exit"))
threading.excepthook = lambda args: log("thread raised", args.exc_type, args.exc_value)

from mime.cli import main
try:
    code = main(sys.argv[3:])
except BaseException:
    log("main RAISED", traceback.format_exc().replace("\\n", " | "))
    raise
log("main returned", code)
raise SystemExit(code)
"""


def client(ready: Path, timeout: float = HANG) -> LiveClient:
    endpoints = json.loads(ready.read_text())
    return LiveClient(endpoints["commands"], endpoints["frames"], timeout=timeout)


def sent(live: LiveClient, command: str, **arguments) -> None:
    live.commands.sock.sendall(message([b"", json.dumps({"protocol": PROTOCOL, "command": command,
                                                          **arguments}).encode("utf-8")]))


def hung_up(live: LiveClient) -> None:
    for part in (live.commands, live.frames):
        try:
            part.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        part.close()


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("served"))


def asked(live, command, **arguments):
    try:
        live.request(command, **arguments)
    except (ConnectionError, OSError):
        pass


@pytest.mark.parametrize("mode", ["threaded", "fd", "idle"])
def test_the_new_client_after_a_hang_up_during_a_held_chunk_is_greeted(launch, folder, tmp_path, mode, monkeypatch):
    gate, ready, out = tmp_path / "gate", tmp_path / "ready.json", tmp_path / "out"
    process = launch([sys.executable, "-c", GATED, str(gate), mode, "serve", str(folder), "--commands",
                      "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0", "--ready-file", str(ready), "--out",
                      str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + HANG
    while not ready.exists() or not ready.read_text():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    ports: list[tuple[str, int]] = []
    real = socket.create_connection

    def recording(address, *args, **kwargs):
        sock = real(address, *args, **kwargs)
        ports.append((f"{address[1]}", sock.getsockname()[1]))
        return sock

    monkeypatch.setattr(socket, "create_connection", recording)
    stepper, probe = client(ready), client(ready)
    if mode != "idle":
        thread = threading.Thread(target=asked, args=(stepper, "step"), kwargs={"chunks": 2}, daemon=True)
        thread.start()
        while not Path(str(gate) + ".held").exists():
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
    failures: list[str] = []
    for round_ in range(ROUNDS):
        first = client(ready)
        sent(first, "record", on=True)
        if mode != "idle":
            assert probe.request("status")["activity"] == "compiling"
        else:
            first.commands.receive()
        hung_up(first)
        assert probe.request("status")["ok"]
        mark = len(ports)
        try:
            second = client(ready)
        except ConnectionError as error:
            failures.append(f"round {round_}: {error!r}; the client's connections (server port, local port): "
                            f"{ports[mark:]}; server alive: {process.poll() is None}")
            if process.poll() is not None:
                failures.append("server output: " + repr(process.communicate()))
                break
            continue
        sent(second, "set", values={"offset": 0.25})
        assert probe.request("status")["ok"]
        hung_up(second)
    log = Path(str(gate) + ".log")
    text = log.read_text() if log.exists() else "(no log)"
    if failures:
        shown = [line for line in text.splitlines() if "accept" not in line or "RAISED" in line]
        raise AssertionError(f"{len(failures)} of {ROUNDS} rounds failed in mode {mode}:\n" + "\n".join(failures[:5])
                             + "\nthe server's log, closes only (last 60):\n" + "\n".join(shown[-60:]))
    stalls = [line for line in text.splitlines() if "STALL" in line]
    odd = [line for line in text.splitlines() if "close" in line and "ready False" in line]
    print(f"mode {mode}: {ROUNDS} rounds, no failure; log lines: {len(text.splitlines())}; stalls: {stalls[:5]}; "
          f"closes of unready peers: {odd[:5]}")
