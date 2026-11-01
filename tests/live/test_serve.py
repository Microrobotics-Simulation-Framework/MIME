"""``mime serve`` in a process of its own, the live rig's saved folder served over ZMQ on loopback ports of the system's
choosing, driven by MIME's client: ``hello`` describes the session; every frame is delivered, in order, and each is the
batch run's state at its steps bit for bit; a live input set over the wire moves the angle; ``stop`` stops it from every
state and writes its run folder; SIGTERM stops it as cleanly. The controls: a frame dropped on the client's side is
seen by the order checker, a request that is not JSON is refused in the coded shape, and an address that is not loopback
is refused before anything is built."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from live_cases import saved, steered

from mime.live import COMMAND_LOG, FrameOrder, LiveClient
from mime.units import ms

MIME = [sys.executable, "-m", "mime.cli"]


def started(folder: Path, out: Path, ready: Path, *extra: str) -> subprocess.Popen:
    process = subprocess.Popen([*MIME, "serve", str(folder), "--commands", "tcp://127.0.0.1:0", "--frames",
                                "tcp://127.0.0.1:0", "--ready-file", str(ready), "--out", str(out), *extra],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 240
    while not ready.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(0.05)
    return process


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("served"))


def client(ready: Path) -> LiveClient:
    endpoints = json.loads(ready.read_text())
    return LiveClient(endpoints["commands"], endpoints["frames"])


def test_every_frame_is_delivered_in_order_and_is_the_batch_runs_state(folder, tmp_path):
    process = started(folder, tmp_path / "out", tmp_path / "ready.json", "--unpaced")
    live = client(tmp_path / "ready.json")
    hello = live.hello
    assert hello["ok"] and hello["state"] == "ready" and hello["stage"].endswith("scene.usda")
    assert hello["visuals"] == {"overlay": None, "drawn": 0, "missing": {}, "notice": []}
    assert live.request("start")["state"] == "running"
    frames = [live.frame() for _ in range(40)]
    assert live.request("pause")["state"] == "paused"
    assert live.request("set", values={"offset": 0.5})["ok"]
    stepped = live.request("step", chunks=3)
    frames += [live.frame() for _ in range(3)]
    assert stepped["step"] == frames[-1].step
    stopped = live.request("stop")
    last = live.frame()
    assert stopped["state"] == "stopped" and last.final and process.wait(timeout=60) == 0
    order = FrameOrder()
    assert [order.see(f) for f in frames + [last]] == [None] * (len(frames) + 1)
    history = steered().build().run(ms(frames[39].step), history=True).history
    for held in frames[1:40]:
        for path, value in held.values.items():
            node, _, field = path.rpartition(".")
            assert value.tobytes() == np.asarray(history[node][field][held.step - 1]).tobytes(), (path, held.step)
    assert float(frames[-1].values["spin.motor.angle"]) != float(np.asarray(
        history["spin.motor"]["angle"][-1]))
    out = tmp_path / "out"
    assert {"commands.jsonl", "outputs.npz", "outputs.json", "run.json", "topology.json", "experiment"} <= \
        {path.name for path in out.iterdir()}
    log = [json.loads(line) for line in (out / COMMAND_LOG).read_text().splitlines()]
    assert [line["command"] for line in log[1:]] == ["start", "pause", "set", "step", "stop"]
    # The control: a frame the client drops is seen.
    order = FrameOrder()
    for held in frames[:5] + frames[6:9]:
        order.see(held)
    assert order.problems == [f"frames {frames[5].number} to {frames[5].number} were dropped before frame "
                              f"{frames[6].number}"]
    live.close()


@pytest.mark.parametrize("state", ["ready", "running", "paused"])
def test_stop_stops_the_server_in_every_state(folder, tmp_path, state):
    process = started(folder, tmp_path / "out", tmp_path / "ready.json", "--pace", "0.2")
    live = client(tmp_path / "ready.json")
    if state in ("running", "paused"):
        live.request("start")
        live.frame()
        live.frame()
    if state == "paused":
        live.request("pause")
    reply = live.request("stop")
    assert reply["ok"] and reply["state"] == "stopped" and process.wait(timeout=60) == 0
    assert (tmp_path / "out" / COMMAND_LOG).is_file()
    live.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGTERM to send a process")
def test_sigterm_stops_it_cleanly_and_writes_its_folder(folder, tmp_path):
    process = started(folder, tmp_path / "out", tmp_path / "ready.json", "--start", "--pace", "0.5")
    live = client(tmp_path / "ready.json")
    live.frame()
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 0
    assert (tmp_path / "out" / "outputs.json").is_file()
    live.close()


def test_a_request_that_is_not_json_is_refused_in_the_coded_shape(folder, tmp_path):
    process = started(folder, tmp_path / "out", tmp_path / "ready.json")
    live = client(tmp_path / "ready.json")
    answer = json.loads(live.commands.request(b"stop please").decode())
    assert not answer["ok"] and answer["errors"][0]["code"] == "live.command"
    assert set(answer["errors"][0]) == {"code", "class", "message", "why", "action", "where", "page"}
    assert live.request("stop")["ok"] and process.wait(timeout=60) == 0
    live.close()


def test_an_address_that_is_not_loopback_is_refused_before_the_build(folder, tmp_path):
    done = subprocess.run([*MIME, "serve", str(folder), "--commands", "tcp://0.0.0.0:0"], capture_output=True,
                          text=True, timeout=120, env={**os.environ, "MIME_TRACEBACK": ""})
    assert done.returncode == 2 and "live.remote-bind" in done.stderr and "load report" not in done.stdout


def test_a_port_in_use_and_an_address_not_of_this_machine_are_refused(folder):
    from mime.build import BuildRefused
    from mime.live import serve
    from mime.live._zmtp import Server

    held = Server("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    try:
        with pytest.raises(BuildRefused, match="live.address-in-use"):
            serve(folder, commands=held.endpoints["REP"], frames="tcp://127.0.0.1:0", say=lambda line: None)
    finally:
        held.close()
    with pytest.raises(BuildRefused, match="live.transport"):
        serve(folder, commands="tcp://192.0.2.1:0", frames="tcp://127.0.0.1:0", remote=True, say=lambda line: None)
