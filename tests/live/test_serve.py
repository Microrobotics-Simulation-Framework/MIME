"""``mime serve`` in a process of its own, A0's apparatus closed by ``ArmTracking`` (``live_cases.tracked``) served over ZMQ on loopback ports of the system's
choosing, driven by MIME's client: ``hello`` describes the session; every frame is delivered, in order, and each is the
batch run's state at its steps bit for bit; a live input set over the wire moves the angle; ``stop`` stops it from every
state and writes its run folder, a long step included at its next chunk's end; SIGTERM stops it as cleanly, and so does
the death of the process that launched it; and while a chunk runs, its first chunk's compile held open, ``status``,
``hello``, ``pause`` and ``stop`` are answered at once. The controls: a frame dropped on the client's side is seen by
the order checker, a request that is not JSON is refused in the coded shape, an address that is not loopback is refused
before anything is built, a server told to outlive its launcher serves on, and a server that steps its chunks on its
loop's own thread answers nothing while one runs. Every server is started in a process group of
its own and stopped with it when its test ends (``conftest.launch``); no check here measures time, and every wait has a
guard against a hang (``HANG``) that is no bound on speed."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from live_cases import built, saved

from mime.live import COMMAND_LOG, FrameOrder, LiveClient
from mime.units import ms

MIME = [sys.executable, "-m", "mime.cli"]
#: How long a wait may last before the test calls it a hang: a guard against waiting for ever, never a measure of speed
#: (a loaded runner takes seconds where a laptop takes milliseconds).
HANG = 600


def started(launch, folder: Path, out: Path, ready: Path, *extra: str) -> subprocess.Popen:
    """``mime serve`` of ``folder`` in a process group of its own (``launch``: stopped with its group when the test
    ends), once its ready file is written."""
    process = launch([*MIME, "serve", str(folder), "--commands", "tcp://127.0.0.1:0", "--frames",
                      "tcp://127.0.0.1:0", "--ready-file", str(ready), "--out", str(out), *extra],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + HANG
    while not ready.exists() or not ready.read_text():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "mime serve never wrote its ready file"
        time.sleep(0.05)
    return process


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("served"))


def client(ready: Path, timeout: float = 30.0) -> LiveClient:
    endpoints = json.loads(ready.read_text())
    return LiveClient(endpoints["commands"], endpoints["frames"], timeout=timeout)


def acts_at(answer: dict, chunk_steps: int) -> int:
    """The step a ``stop`` or a ``pause`` acts at: where its reply says the session is, or, taken while a chunk ran
    (its command pending), that chunk's end."""
    return answer["step"] + (chunk_steps if answer["command"] in answer["pending"] else 0)


def test_every_frame_is_delivered_in_order_and_is_the_batch_runs_state(launch, folder, tmp_path):
    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json", "--unpaced")
    live = client(tmp_path / "ready.json")
    hello = live.hello
    assert hello["ok"] and hello["state"] == "ready" and hello["stage"].endswith("scene.usda")
    from mime.scene import visual_sets

    # Nothing baked: no overlay, every mesh missing, and the set's notice (its credit and terms) to show beside the
    # simple shapes, with how to fetch the meshes.
    assert hello["visuals"] == {"overlay": None, "drawn": 0, "missing": {"ar4_mk5": 18},
                                "notice": [visual_sets()["ar4_mk5"].text()]}
    assert "Annin Robotics" in hello["visuals"]["notice"][0]
    assert live.request("start")["state"] == "running"
    frames = [live.frame() for _ in range(40)]
    paused = live.request("pause")
    assert paused["state"] == "paused" or paused["pending"] == ["pause"]
    while frames[-1].step < acts_at(paused, hello["chunk_steps"]):  # the frames sent before the pause took effect
        frames.append(live.frame())
    assert live.request("status")["state"] == "paused"
    before = len(frames)
    assert live.request("set", values={"offset": 0.002})["ok"]
    stepped = live.request("step", chunks=3)
    frames += [live.frame() for _ in range(3)]
    assert stepped["step"] == frames[-1].step
    stopped = live.request("stop")
    last = live.frame()
    assert stopped["state"] == "stopped" and last.final and process.wait(timeout=60) == 0
    order = FrameOrder()
    assert [order.see(f) for f in frames + [last]] == [None] * (len(frames) + 1)
    history = built().run(ms(frames[-1].step), history=True).history
    for held in frames[1:before]:
        for path, value in held.values.items():
            node, _, field = path.rpartition(".")
            assert value.tobytes() == np.asarray(history[node][field][held.step - 1]).tobytes(), (path, held.step)
    # The live input took effect: the arm the batch run left alone stands elsewhere at the session's last step.
    assert frames[-1].values["rpm.arm.frame_poses"].tobytes() != np.asarray(
        history["rpm.arm"]["frame_poses"][frames[-1].step - 1]).tobytes()
    out = tmp_path / "out"
    assert {"commands.jsonl", "outputs.npz", "outputs.json", "run.json", "topology.json", "experiment"} <= \
        {path.name for path in out.iterdir()}
    log = [json.loads(line) for line in (out / COMMAND_LOG).read_text().splitlines()]
    assert [line["command"] for line in log[1:] if "command" in line] == ["start", "pause", "set", "step", "stop"]
    # After each chunk the step it reached, before its frame was sent: every frame's step is in the log.
    assert {f.step for f in frames[1:] + [last]} <= {line["reached"] for line in log if "reached" in line}
    # The control: a frame the client drops is seen.
    order = FrameOrder()
    for held in frames[:5] + frames[6:9]:
        order.see(held)
    assert order.problems == [f"frames {frames[5].number} to {frames[5].number} were dropped before frame "
                              f"{frames[6].number}"]
    live.close()


@pytest.mark.parametrize("state", ["ready", "running", "paused"])
def test_stop_stops_the_server_in_every_state(launch, folder, tmp_path, state):
    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json")
    live = client(tmp_path / "ready.json")
    if state in ("running", "paused"):
        live.request("start")
        live.frame()
        live.frame()
    if state == "paused":
        live.request("pause")
    reply = live.request("stop")
    assert reply["ok"] and (reply["state"] == "stopped" or reply["pending"][-1:] == ["stop"])
    assert process.wait(timeout=60) == 0
    assert (tmp_path / "out" / COMMAND_LOG).is_file()
    live.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGTERM to send a process")
def test_sigterm_stops_it_cleanly_and_writes_its_folder(launch, folder, tmp_path):
    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json", "--start", "--pace", "0.5")
    live = client(tmp_path / "ready.json")
    live.frame()
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 0
    assert (tmp_path / "out" / "outputs.json").is_file()
    live.close()


def test_a_request_that_is_not_json_is_refused_in_the_coded_shape(launch, folder, tmp_path):
    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json")
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


def test_a_malformed_argument_is_refused_over_the_wire_and_the_server_goes_on(launch, folder, tmp_path):
    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json")
    live = client(tmp_path / "ready.json")
    codes = [live.request("step", chunks="x"), live.request("step", chunks=None),
             live.request("set", values={"offset": "abc"}), live.request("record", on="off")]
    assert [answer["errors"][0]["code"] for answer in codes] == ["live.step", "live.step", "live.input-value",
                                                                 "live.record"]
    nested = json.loads(live.commands.request(b"[" * 100_000 + b"]" * 100_000).decode())
    assert nested["errors"][0]["code"] == "live.command"
    assert live.request("status")["ok"] and live.request("stop")["ok"] and process.wait(timeout=60) == 0
    log = [json.loads(line) for line in (tmp_path / "out" / COMMAND_LOG).read_text().splitlines()]
    assert [line.get("command") for line in log[1:]] == ["stop"]
    live.close()


def test_a_killed_server_leaves_a_log_that_replays_every_frame_it_sent(launch, folder, tmp_path):
    from mime.experiment import Experiment
    from mime.live import replay

    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json", "--unpaced")
    live = client(tmp_path / "ready.json")
    live.request("step", chunks=4)
    live.request("set", values={"offset": 0.002})
    live.request("start")
    received = [live.frame() for _ in range(30)]
    process.kill()
    process.wait(timeout=60)
    live.close()
    out = tmp_path / "out"
    assert {path.name for path in out.iterdir()} == {"experiment", COMMAND_LOG}  # it died: nothing else was written
    replayed: dict[int, object] = {}
    session = replay(Experiment.load(out / "experiment").build(), out / COMMAND_LOG, tmp_path / "again",
                     subscriber=lambda held: replayed.setdefault(held.step, held))
    reached = max(json.loads(line).get("reached", 0) for line in (out / COMMAND_LOG).read_text().splitlines())
    assert session.steps == reached >= received[-1].step
    for held in received:
        again = replayed[held.step]
        assert all(value.tobytes() == again.values[path].tobytes() for path, value in held.values.items()), held.step
    # The control: the live input mattered, so a replay that dropped it would differ.
    plain = built().live(pace=None)
    plain.step(received[-1].step // plain.chunk_steps)
    assert plain._final["rpm.arm"]["frame_poses"].tobytes() != \
        received[-1].values["rpm.arm.frame_poses"].tobytes()


def logged_stop(out: Path) -> tuple[list[str], int, list[int]]:
    """The commands a run folder's log holds, the step its stop took effect at, and the steps its chunks reached."""
    log = [json.loads(line) for line in (out / COMMAND_LOG).read_text().splitlines()][1:]
    commands = [line for line in log if "command" in line]
    assert log[-1] == commands[-1] and commands[-1]["command"] == "stop"  # nothing was stepped after the stop
    return [line["command"] for line in commands], int(commands[-1]["step"]), [line["reached"] for line in log
                                                                              if "reached" in line]


def test_stop_ends_a_long_step_at_the_next_chunk_boundary(launch, folder, tmp_path):
    """A ``step`` of 10^9 chunks (days of stepping) and, from a second client, a ``status`` answered while it runs and
    a ``stop``: the session stops at a chunk's end (the end of the chunk the stop came during, when it came during
    one), its step cut short, and the step's reply says so. Machine-free: the log holds the step, each chunk's step,
    then the stop at the last chunk's step; no time is measured."""
    import threading

    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json", "--unpaced")
    stepper, stopper = client(tmp_path / "ready.json"), client(tmp_path / "ready.json")
    answers: list[dict] = []
    thread = threading.Thread(target=lambda: answers.append(stepper.request("step", chunks=10 ** 9)), daemon=True)
    thread.start()
    stepper.frame()
    stepper.frame()  # the first chunk compiled: the step is under way
    assert stopper.request("status")["stepping"] > 0  # the server answers while the step's chunks run
    stopped = stopper.request("stop")
    thread.join(timeout=HANG)
    assert stopped["ok"] and answers and answers[0]["ok"] and answers[0]["state"] == "stopped"
    assert answers[0]["stepping"] == 0 and process.wait(timeout=HANG) == 0
    commands, at, reached = logged_stop(tmp_path / "out")
    assert commands == ["step", "stop"] and reached[-1] == at == acts_at(stopped, stepper.hello["chunk_steps"])
    assert at < 10 ** 9 * stepper.hello["chunk_steps"]  # the step cut short
    stepper.close()
    stopper.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGTERM to send a process")
def test_sigterm_ends_a_long_step_at_the_next_chunk_boundary(launch, folder, tmp_path):
    import threading

    process = started(launch, folder, tmp_path / "out", tmp_path / "ready.json", "--unpaced")
    live = client(tmp_path / "ready.json")
    thread = threading.Thread(target=lambda: live.request("step", chunks=10 ** 9), daemon=True)
    thread.start()
    live.frame()
    live.frame()
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=HANG) == 0 and (tmp_path / "out" / "outputs.json").is_file()
    commands, at, reached = logged_stop(tmp_path / "out")
    assert commands == ["step", "stop"] and reached[-1] == at < 10 ** 9 * live.hello["chunk_steps"]
    thread.join(timeout=HANG)
    live.close()


def test_the_server_stops_when_the_process_that_launched_it_is_gone(launch, folder, tmp_path):
    """A launcher (a Python process) starts ``mime serve`` and is killed outright, as a pytest worker can be: the
    server stops as on SIGTERM, writing its run folder, and exits. The control: the same with ``--outlive-launcher``
    serves on after its launcher is gone (stopped here by its process group)."""
    for outlive in (False, True):
        out, ready = tmp_path / f"out_{outlive}", tmp_path / f"ready_{outlive}.json"
        argv = [*MIME, "serve", str(folder), "--commands", "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0",
                "--ready-file", str(ready), "--out", str(out), *(["--outlive-launcher"] if outlive else [])]
        launcher = launch([sys.executable, "-c", "import subprocess, sys, time; "
                           f"server = subprocess.Popen({argv!r}); print(server.pid, flush=True); time.sleep(10 ** 6)"],
                          stdout=subprocess.PIPE, text=True)
        assert launcher.stdout is not None
        pid = int(launcher.stdout.readline())
        deadline = time.monotonic() + HANG
        while not ready.exists() or not ready.read_text():
            assert time.monotonic() < deadline, "mime serve never wrote its ready file"
            time.sleep(0.05)
        live = client(ready)
        launcher.kill()  # SIGKILL on POSIX: the launcher runs no handler
        launcher.wait(timeout=HANG)
        if outlive:
            assert live.request("status")["ok"]  # it serves on, its launcher gone
            live.close()
            stop_pid(pid)
            continue
        while running(pid):
            assert time.monotonic() < deadline, "mime serve outlived its launcher"
            time.sleep(0.05)
        assert (out / "run.json").is_file() and json.loads((out / COMMAND_LOG).read_text().splitlines()[-1]) == \
            {"command": "stop", "step": 0}
        live.close()


def running(pid: int) -> bool:
    """Whether the process ``pid`` still runs (a zombie, its parent gone and not yet reaped, does not)."""
    if sys.platform == "win32":
        done = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False)
        return str(pid) in done.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/stat")
    return not (status.is_file() and status.read_text().split(") ")[-1].startswith("Z"))


def stop_pid(pid: int) -> None:
    """Kill the server of the control, which outlived its launcher, so that nothing is left running."""
    try:
        os.kill(pid, signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + HANG
    while running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)


def test_a_run_folder_served_or_replayed_into_itself_is_refused_and_left_whole(folder, tmp_path):
    from mime.experiment import Experiment

    run = Experiment.load(folder).build().live(pace=None, out=tmp_path / "run")  # stopped as it opens: nothing compiled
    run.stop()
    before = {str(p.relative_to(tmp_path / "run")): p.read_bytes() for p in sorted((tmp_path / "run").rglob("*"))
              if p.is_file()}
    env = {**os.environ, "MIME_TRACEBACK": ""}
    for argv in (["serve", str(tmp_path / "run" / "experiment"), "--out", str(tmp_path / "run")],
                 ["run", str(tmp_path / "run" / "experiment"), "--commands", str(tmp_path / "run" / COMMAND_LOG),
                  "--out", str(tmp_path / "run")]):
        done = subprocess.run([*MIME, *argv, *(["--commands", "tcp://127.0.0.1:0", "--frames",
                                                 "tcp://127.0.0.1:0"] if argv[0] == "serve" else [])],
                              capture_output=True, text=True, timeout=240, env=env)
        assert done.returncode == 2 and "run.out-over-source" in done.stderr, done.stderr
    after = {str(p.relative_to(tmp_path / "run")): p.read_bytes() for p in sorted((tmp_path / "run").rglob("*"))
             if p.is_file()}
    assert after == before


#: ``mime serve`` with each chunk held until the file ``argv[1]`` exists (a long first compile, without the clock: the
#: process is kept busy meanwhile, as tracing keeps it), the file ``<argv[1]>.held`` written when one is; and with
#: ``argv[2]`` "inline" each chunk stepped on the loop's own thread, as before 7n-b. Then ``mime`` with the rest.
GATED = """
import concurrent.futures, pathlib, sys
import mime.control._run as run
import mime.live._serve as serve

gate, inline, chunk = pathlib.Path(sys.argv[1]), sys.argv[2] == "inline", run.ChunkedRun.run

def held(self, *args, **kwargs):
    pathlib.Path(str(gate) + ".held").touch()
    while not gate.exists():
        sum(range(10_000))
    return chunk(self, *args, **kwargs)

class Inline(concurrent.futures.ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        done = concurrent.futures.Future()
        done.set_result(fn(*args, **kwargs))
        return done

run.ChunkedRun.run = held
if inline:
    serve.ThreadPoolExecutor = Inline
from mime.cli import main
raise SystemExit(main(sys.argv[3:]))
"""


def gated_server(launch, folder: Path, tmp_path: Path, mode: str) -> tuple[subprocess.Popen, Path, Path]:
    gate, ready, out = tmp_path / f"gate_{mode}", tmp_path / f"ready_{mode}.json", tmp_path / f"out_{mode}"
    process = launch([sys.executable, "-c", GATED, str(gate), mode, "serve", str(folder), "--commands",
                      "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0", "--ready-file", str(ready), "--out",
                      str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + HANG
    while not ready.exists() or not ready.read_text():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "mime serve never wrote its ready file"
        time.sleep(0.05)
    return process, gate, out


def held_open(gate: Path, thread) -> None:
    """Wait until a chunk is held at ``gate``, while ``thread`` (the step's request) waits for its reply."""
    deadline = time.monotonic() + HANG
    while not Path(str(gate) + ".held").exists():
        assert thread.is_alive() and time.monotonic() < deadline, "no chunk was held"
        time.sleep(0.01)


def test_the_server_answers_while_its_first_chunk_compiles(launch, folder, tmp_path):
    """The client's stall: a ``status`` sent while a chunk compiled was answered when the chunk ended. A step of two
    chunks, its first held open (``GATED``, until the test lets it go): ``status`` and ``hello`` from a second client
    are answered meanwhile, saying the session is compiling at step 0 with two chunks in hand, and so is a ``pause``,
    taken to act at the chunk's end (``pending``); let go, the chunk ends, the pause ends the step there, and the step
    is answered with the step it reached. Machine-free: the chunk cannot end before the test lets it, so a reply that
    comes before is one that did not wait for it; each wait is a guard against a hang. The control: a server that
    steps each chunk on its loop's own thread, as before, answers no ``status`` while the chunk is held."""
    import threading

    process, gate, out = gated_server(launch, folder, tmp_path, "threaded")
    stepper, asker = client(tmp_path / "ready_threaded.json"), client(tmp_path / "ready_threaded.json")
    answers: list[dict] = []
    thread = threading.Thread(target=lambda: answers.append(stepper.request("step", chunks=2)), daemon=True)
    thread.start()
    held_open(gate, thread)
    status, hello = asker.request("status"), asker.request("hello")
    assert (status["activity"], status["step"], status["stepping"], status["state"]) == ("compiling", 0, 2, "paused")
    assert hello["activity"] == "compiling" and hello["report"]["format"] == "mime.report/1"
    paused = asker.request("pause")
    assert paused["ok"] and paused["pending"] == ["pause"] and paused["activity"] == "compiling"
    assert not answers  # the step's reply waits for its chunks
    gate.touch()
    thread.join(timeout=HANG)
    chunk = hello["chunk_steps"]
    assert answers and answers[0]["ok"] and (answers[0]["step"], answers[0]["stepping"]) == (chunk, 0)
    after = asker.request("status")
    assert (after["activity"], after["state"], after["pending"]) == ("idle", "paused", [])
    assert asker.request("stop")["state"] == "stopped" and process.wait(timeout=HANG) == 0
    log = [json.loads(line) for line in (out / COMMAND_LOG).read_text().splitlines()][1:]
    assert log == [{"command": "step", "chunks": 2, "step": 0}, {"reached": chunk}, {"command": "pause", "step": chunk},
                   {"command": "stop", "step": chunk}]
    stepper.close()
    asker.close()
    # The control: the chunk on the loop's own thread, held; a status waits for it, and is not answered in 3 s (no
    # load could answer it sooner: the gate stays shut). The server is killed with its group when the test ends.
    process, gate, _ = gated_server(launch, folder, tmp_path, "inline")
    stepper, asker = client(tmp_path / "ready_inline.json"), client(tmp_path / "ready_inline.json", timeout=3.0)
    thread = threading.Thread(target=given_up, args=(stepper,), daemon=True)
    thread.start()
    held_open(gate, thread)
    with pytest.raises(TimeoutError):
        asker.request("status")
    asker.close()
    stepper.commands.sock.shutdown(socket.SHUT_RDWR)  # the step's reply given up on
    thread.join(timeout=HANG)
    stepper.close()


def given_up(stepper: LiveClient) -> None:
    """A step of one chunk, whose reply the control gives up on."""
    with contextlib.suppress(ConnectionError, OSError):
        stepper.request("step", chunks=1)
