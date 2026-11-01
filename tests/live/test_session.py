"""The live session in process (``built.live()``), on the live rig (``live_cases.steered``): its frames are a batch
run's states at the same steps, bit for bit; its live input moves the motor's angle from the next tick and nothing
before it; replaying its command log reproduces its run folder; and ``stop`` stops it in every state. Each with its
control: frames compared one step off are not the batch's, a session that sets nothing moves nothing, and a log with
one value changed replays to another folder."""

import json
import threading
import time

import numpy as np
import pytest
from live_cases import built, files_of, in_graph, saved, steered

from mime.build import ArgumentRefused
from mime.experiment import Experiment
from mime.live import COMMAND_LOG, PAUSED, READY, RUNNING, STOPPED, Frame, FrameOrder, replay
from mime.units import ms

RATE = 0.1  # Steered's angle per tick


def batch_history(steps: int) -> dict:
    """Every step's state of the rig run in batch from its start, ``steps`` steps."""
    return steered().build().run(ms(steps), history=True).history


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def mismatches(frames: list[Frame], history: dict, shift: int = 0) -> list[str]:
    """Each frame's field that is not the batch's state after the frame's steps (history row ``step - 1``)."""
    found = []
    for held in frames:
        if held.step == 0:
            continue
        for path, value in held.values.items():
            node, _, field = path.rpartition(".")
            if not same(value, history[node][field][held.step - 1 + shift]):
                found.append(f"{path} at step {held.step}")
    return found


def test_frames_are_the_batch_runs_states_at_the_same_steps_bit_for_bit():
    session = steered().build().live(pace=None)
    frames: list[Frame] = []
    session.subscribe(frames.append)
    session.step(20)
    order = FrameOrder()
    assert [order.see(f) for f in frames] == [None] * 21 and frames[-1].step == 20 * session.chunk_steps
    history = batch_history(frames[-1].step)
    assert mismatches(frames, history) == []
    # The control: the same frames held to the batch's states one step later differ, so the comparison can fail.
    off = {line.rsplit(" ", 1)[-1] for line in mismatches(frames[:-1], history, shift=1)}
    assert off == {str(f.step) for f in frames[1:-1]}


def angle(held: Frame) -> float:
    return float(held.values["spin.motor.angle"])


def test_a_live_input_moves_the_angle_from_the_next_tick_and_nothing_before_it():
    session = steered().build().live(pace=None, select=["spin.motor.angle"])
    frames: list[Frame] = []
    session.subscribe(frames.append)
    session.step(4)
    session.set("offset", 0.5)
    at = session.steps
    session.step(4)
    period = 3  # Steered's rate, 3 ms at the rig's 1 ms step
    for held in frames[1:]:
        ticks = (held.step - 1) // period
        expected = RATE * ticks + (0.5 if (held.step - 1) // period * period >= at else 0.0)
        assert angle(held) == pytest.approx(expected, rel=0, abs=1e-6), held.step
    # The control: a session that sets nothing writes the angle alone at those steps.
    plain = steered().build().live(pace=None, select=["spin.motor.angle"])
    held: list[Frame] = []
    plain.subscribe(held.append)
    plain.step(8)
    assert [angle(f) for f in held[5:]] != [angle(f) for f in frames[5:]]
    assert all(angle(f) == pytest.approx(RATE * ((f.step - 1) // period), abs=1e-6) for f in held[1:])


def test_replaying_the_command_log_reproduces_the_sessions_run_folder(tmp_path):
    folder = saved(tmp_path)
    live = Experiment.load(folder).build().live(pace=None, out=tmp_path / "live")
    live.step(3)
    live.set("offset", 0.002)
    live.record(True)
    live.step(5)
    live.record(False)
    live.set("offset", -0.003)
    live.step(2)
    written = live.stop()
    assert written == tmp_path / "live"
    log = written / COMMAND_LOG
    replay(Experiment.load(folder).build(), log, tmp_path / "again")
    first, second = files_of(written), files_of(tmp_path / "again")
    assert set(first) == set(second) and "recording.usdc" in first
    differ = [name for name in first if first[name] != second[name] and name not in ("run.json", "outputs.npz")]
    assert differ == []
    for name in ("outputs.npz",):
        a, b = np.load(written / name), np.load(tmp_path / "again" / name)
        assert sorted(a.files) == sorted(b.files) and all(same(a[k], b[k]) for k in a.files)
    plain = {k: v for k, v in json.loads(first["run.json"])["live"].items() if k != "wall_seconds"}
    assert plain == {k: v for k, v in json.loads(second["run.json"])["live"].items() if k != "wall_seconds"}
    # The control: the log with one live input's value changed replays to another folder.
    lines = log.read_text().splitlines()
    planted = [json.dumps({**json.loads(line), "value": 0.004}) if '"set"' in line and "0.002" in line else line
               for line in lines]
    (tmp_path / "planted.jsonl").write_text("\n".join(planted) + "\n")
    replay(Experiment.load(folder).build(), tmp_path / "planted.jsonl", tmp_path / "planted")
    changed = files_of(tmp_path / "planted")
    assert changed["recording.usdc"] != first["recording.usdc"]


def threaded(session):
    session.start()
    deadline = time.monotonic() + 30
    while session.frames < 3 and time.monotonic() < deadline:
        time.sleep(0.01)


@pytest.mark.parametrize("state", [READY, RUNNING, PAUSED, "stepping"])
def test_stop_stops_in_every_state(tmp_path, state):
    session = built().live(pace=None, out=tmp_path / "out")
    frames: list[Frame] = []
    session.subscribe(frames.append)
    if state == RUNNING:
        threaded(session)
        assert session.state == RUNNING
    elif state == PAUSED:
        threaded(session)
        session.pause()
    elif state == "stepping":
        stepping = threading.Thread(target=session.step, args=(200,))
        stepping.start()
        while session.frames < 2:
            time.sleep(0.005)
    written = session.stop()
    if state == "stepping":
        stepping.join(timeout=30)
        assert not stepping.is_alive()
    assert session.state == STOPPED and written == tmp_path / "out" and (written / COMMAND_LOG).is_file()
    assert frames[-1].final and (session._thread is None or not session._thread.is_alive())
    steps = session.steps
    time.sleep(0.05)
    assert session.steps == steps and session.stop() == written
    with pytest.raises(ArgumentRefused, match="live.stopped"):
        session.start()
    assert session.command({"command": "step"})["errors"][0]["code"] == "live.stopped"


def test_what_a_session_refuses():
    session = steered().build().live(pace=None)
    codes = {
        "set unknown": session.command({"command": "set", "values": {"gain": 1.0}}),
        "set shape": session.command({"command": "set", "values": {"offset": [1.0, 2.0]}}),
        "set nothing": session.command({"command": "set"}),
        "no command": session.command({"command": "jump"}),
        "protocol": session.command({"protocol": "mime.live/2", "command": "hello"}),
    }
    assert {name: reply["errors"][0]["code"] for name, reply in codes.items()} == {
        "set unknown": "live.input-unknown", "set shape": "live.input-value", "set nothing": "live.input-value",
        "no command": "live.command", "protocol": "live.protocol"}
    assert all(not reply["ok"] and reply["errors"][0]["class"] == "user" for reply in codes.values())
    with pytest.raises(ArgumentRefused, match="live.input-mode"):
        in_graph().build().live(pace=None).set("offset", 1.0)
    with pytest.raises(ArgumentRefused, match="live.chunk"):
        steered().build().live(chunk=ms(4))
    with pytest.raises(ArgumentRefused, match="live.pace"):
        steered().build().live(pace=-1.0)
    with pytest.raises(ArgumentRefused, match="live.select"):
        steered().build().live(select=["spin.motor.anglee"])
    with pytest.raises(ArgumentRefused, match="live.step"):
        session.step(0)
    threaded(session)
    with pytest.raises(ArgumentRefused, match="live.state"):
        session.step(1)
    session.stop()


def test_a_log_of_another_experiment_or_no_log_at_all_is_refused(tmp_path):
    session = built().live(pace=None, out=tmp_path / "live")
    session.step(1)
    log = session.stop() / COMMAND_LOG
    with pytest.raises(ArgumentRefused, match="live.log"):
        replay(in_graph().build(), log, tmp_path / "other")
    (tmp_path / "not_a_log.jsonl").write_text('{"step": 0}\n')
    with pytest.raises(ArgumentRefused, match="live.log"):
        replay(steered().build(), tmp_path / "not_a_log.jsonl", None)


def test_hello_describes_the_session():
    session = steered().build().live(pace=None)
    hello = session.command({"command": "hello"})
    assert hello["ok"] and hello["protocol"] == "mime.live/1" and hello["state"] == READY
    assert hello["chunk_steps"] % 3 == 0 and hello["controller"]["mode"] == "callback"
    assert hello["live_inputs"] == {"offset": {"units": "1", "shape": [], "meaning": "added to every angle it writes"}}
    assert {e["name"] for e in hello["binding"]["entries"]} >= {"umr", "spin.motor.base", "spin.motor.rotor"}
    assert [f["path"] for f in hello["schema"]["fields"]][:1] == ["umr.pose"]
    assert json.loads(json.dumps(hello)) == hello


def test_a_run_for_steps_whole_chunks_and_a_slow_pace_waits():
    session = steered().build().live(pace=None)
    frames = session.run_for(ms(60))
    assert len(frames) * session.chunk_steps * session.dt == pytest.approx(0.060, abs=session.chunk_steps * 0.001)
    paced = steered().build().live(pace=0.5)
    paced.step(1)
    paced.start()
    began = time.monotonic()
    while paced.steps < paced.chunk_steps * 6:
        time.sleep(0.005)
    took = time.monotonic() - began
    paced.stop()
    assert took >= 0.8 * (5 * paced.chunk_steps * paced.dt) / 0.5


def test_a_frame_carries_the_selected_fields_in_the_schemas_order():
    session = steered().build().live(pace=None, select=["umr.twist", "spin.motor.angle"])
    held = session.step(1)[-1]
    assert list(held.values) == ["umr.twist", "spin.motor.angle"] == [f.path for f in session.schema.fields]
