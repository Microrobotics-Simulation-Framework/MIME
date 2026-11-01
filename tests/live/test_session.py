"""The live session in process (``built.live()``), on the live rig (``live_cases.steered``): its frames are a batch
run's states at the same steps, bit for bit; its live input moves the motor's angle from the next tick and nothing
before it; replaying its command log reproduces its run folder; and ``stop`` stops it in every state. Each with its
control: frames compared one step off are not the batch's, a session that sets nothing moves nothing, and a log with
one value changed replays to another folder. No set of a live input compiles the step again (its control: the cause put
back compiles at the first set); ``status`` and ``hello`` are answered while a chunk runs, saying so (their control: a
command that changes the run waits for the chunk); and ``hello`` and every run folder carry the build report's summary,
which ``RunResult.read`` reads back (its control: a report of another format is refused)."""

import json
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from jax import monitoring
from live_cases import built, files_of, in_graph, jit_steered, saved, steered

from mime.build import ArgumentRefused
from mime.experiment import Experiment
from mime.live import COMMAND_LOG, PAUSED, READY, RUNNING, STOPPED, Frame, FrameOrder, replay
from mime.results import REPORT_FORMAT, RUN_REPORT, RunResult
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
    with compilations() as seen:  # A0's controller, a jax.jit, has read its live input since the first tick
        live.set("offset", 0.002)
        live.step(1)
    assert seen == []
    live.record(True)
    live.step(4)
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
    assert hello["live_inputs"] == {"offset": {"units": "1", "shape": [], "meaning": "added to every angle it writes",
                                               "default": 0.0}}
    assert hello["activity"] == "idle" and hello["pending"] == []
    assert {e["name"] for e in hello["binding"]["entries"]} >= {"umr", "spin.motor.base", "spin.motor.rotor"}
    assert [f["path"] for f in hello["schema"]["fields"]][:1] == ["umr.pose"]
    assert json.loads(json.dumps(hello)) == hello


def test_a_run_for_steps_whole_chunks_and_a_slow_pace_waits():
    session = steered().build().live(pace=None)
    frames = session.run_for(ms(60))
    assert len(frames) * session.chunk_steps * session.dt == pytest.approx(0.060, abs=session.chunk_steps * 0.001)
    paced = steered().build().live(pace=0.5)
    paced.step(1)  # the compile, before pacing starts
    began = time.monotonic()  # before the start: the session's clock starts no earlier
    paced.start()
    while paced.steps < paced.chunk_steps * 6:
        time.sleep(0.005)
    took = time.monotonic() - began
    paced.stop()
    # Five chunks after the start, the fifth begun no earlier than four chunks' time at half real time from the start
    # (a session may run a chunk behind and catch up, never ahead), on any machine: load only makes it later. 5 ms for
    # the clocks' readings.
    assert took >= 4 * paced.chunk_steps * paced.dt / 0.5 - 0.005


def test_a_frame_carries_the_selected_fields_in_the_schemas_order():
    session = steered().build().live(pace=None, select=["umr.twist", "spin.motor.angle"])
    held = session.step(1)[-1]
    assert list(held.values) == ["umr.twist", "spin.motor.angle"] == [f.path for f in session.schema.fields]


def test_a_malformed_argument_is_refused_with_its_code_and_the_session_goes_on():
    session = steered().build().live(pace=None, select=["spin.motor.angle"])
    malformed = {
        "step text": ({"command": "step", "chunks": "x"}, "live.step"),
        "step null": ({"command": "step", "chunks": None}, "live.step"),
        "step true": ({"command": "step", "chunks": True}, "live.step"),
        "step half": ({"command": "step", "chunks": 1.5}, "live.step"),
        "set text": ({"command": "set", "values": {"offset": "abc"}}, "live.input-value"),
        "set null": ({"command": "set", "values": {"offset": None}}, "live.input-value"),
        "set true": ({"command": "set", "values": {"offset": True}}, "live.input-value"),
        "set object": ({"command": "set", "values": {"offset": {"a": 1}}}, "live.input-value"),
        "set ragged": ({"command": "set", "values": {"offset": [[1.0], [1.0, 2.0]]}}, "live.input-value"),
        "record text": ({"command": "record", "on": "off"}, "live.record"),
        "record number": ({"command": "record", "on": 1}, "live.record"),
        "pace text": ({"command": "pace", "rtf": "fast"}, "live.pace"),
        "pace true": ({"command": "pace", "rtf": True}, "live.pace"),
    }
    for name, (request, code) in malformed.items():
        answer = session.command(request)
        assert not answer["ok"] and answer["errors"][0]["code"] == code, name
    # Nothing a refused request asked for was done or logged: the session is where it began, and goes on.
    assert session.steps == 0 and session.log == [] and session.command({"command": "status"})["state"] == READY
    # Two values, the second refused: neither is set.
    assert not session.command({"command": "set", "values": {"offset": 0.5, "gain": 1.0}})["ok"]
    assert session.log == []
    assert session.command({"command": "step", "chunks": 2})["ok"] and session.steps == 2 * session.chunk_steps


def test_an_error_inside_a_command_is_answered_in_the_coded_shape_and_the_session_goes_on(monkeypatch):
    session = steered().build().live(pace=None)

    def broken() -> dict:
        raise RuntimeError("a fault planted in hello")

    monkeypatch.setattr(session, "hello", broken)
    answer = session.command({"command": "hello"})
    assert not answer["ok"] and answer["errors"][0]["class"] in ("internal", "unrecognised")
    assert "a fault planted in hello" in answer["errors"][0]["message"]
    assert session.command({"command": "step"})["ok"]


def test_a_live_input_never_outlives_its_session(tmp_path):
    held = steered().build()
    first = held.live(pace=None, select=["spin.motor.angle"])
    first.step(2)
    first.set("offset", 0.5)
    moved = first.step(2)
    first.stop()
    period = 3
    # The control: in the session that set it, the input moved the angle, so a leak would show.
    assert angle(moved[-1]) == pytest.approx(RATE * ((moved[-1].step - 1) // period) + 0.5, abs=1e-6)
    second = held.live(pace=None, select=["spin.motor.angle"])
    frames = second.step(4)
    assert all(angle(f) == pytest.approx(RATE * ((f.step - 1) // period), abs=1e-6) for f in frames[1:])
    # A session left open, then a batch run of the same built: the run starts from the declared inputs too.
    third = held.live(pace=None, select=["spin.motor.angle"])
    third.set("offset", 0.5)
    result = held.run(ms(3 * period))
    steps = result.start_step + result.steps
    assert float(result.final["spin.motor"]["angle"]) == pytest.approx(RATE * ((steps - 1) // period), abs=1e-6)


def test_the_command_log_is_written_as_the_session_goes(tmp_path):
    session = built().live(pace=None, out=tmp_path / "live")
    log = tmp_path / "live" / COMMAND_LOG
    assert [json.loads(line)["format"] for line in log.read_text().splitlines()] == ["mime.commands/1"]
    session.step(2)
    session.set("offset", 0.002)
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    chunk = session.chunk_steps
    assert lines[1:] == [{"command": "step", "chunks": 2, "step": 0}, {"reached": chunk}, {"reached": 2 * chunk},
                         {"command": "set", "label": "offset", "step": 2 * chunk, "value": 0.002}]
    session.stop()
    assert json.loads(log.read_text().splitlines()[-1]) == {"command": "stop", "step": 2 * chunk}


@pytest.fixture(scope="module")
def tracked_folder(tmp_path_factory):
    """The tracked rig saved once for the module's checks that read a saved folder and write no run beside it."""
    return saved(tmp_path_factory.mktemp("tracked"))


def test_a_run_folder_is_never_written_over_what_it_reads(tracked_folder, tmp_path):
    folder = tracked_folder
    loaded = Experiment.load(folder).build()
    for out in (folder, folder / "inside", folder.parent):
        with pytest.raises(ArgumentRefused, match="run.out-over-source"):
            loaded.live(pace=None, out=out)
        with pytest.raises(ArgumentRefused, match="run.out-over-source"):
            loaded.run_to(out, ms(2))
    run = loaded.live(pace=None, out=tmp_path / "run")  # stopped as it opens: a run folder, with nothing compiled
    run.stop()
    before = files_of(tmp_path / "run")
    rebuilt = Experiment.load(tmp_path / "run" / "experiment").build()
    with pytest.raises(ArgumentRefused, match="run.out-over-source"):
        replay(rebuilt, tmp_path / "run" / COMMAND_LOG, tmp_path / "run")
    with pytest.raises(ArgumentRefused, match="run.out-over-source"):
        rebuilt.live(pace=None, out=tmp_path / "run")
    assert files_of(tmp_path / "run") == before
    # The control: a folder of its own is written.
    replay(rebuilt, tmp_path / "run" / COMMAND_LOG, tmp_path / "again")
    assert (tmp_path / "again" / "experiment").is_dir()


#: JAX's events for a trace and for a compile (or a read of the persistent compilation cache, which JAX times as one).
TRACE, COMPILE = "/jax/core/compile/jaxpr_trace_duration", "/jax/core/compile/backend_compile_duration"


@contextmanager
def compilations():
    """Every trace and compile while the block runs, as JAX's own monitoring reports them."""
    seen: list[str] = []

    def listen(event: str, duration: float, **_: object) -> None:
        if event in (TRACE, COMPILE):
            seen.append(event)

    monitoring.register_event_duration_secs_listener(listen)
    try:
        yield seen
    finally:
        monitoring.unregister_event_duration_listener(listen)


#: A scalar live input in every form a client gives one: JSON's integers and floats, NumPy's scalars, 0-d arrays.
VALUES = [1, -2, 0.5, -0.25, 3.0e-3, np.float32(0.125), np.float64(-1.5), np.int32(2), np.int64(-4),
          np.asarray(0.75, dtype=np.float32), np.asarray(-0.5)]


def test_no_set_of_a_live_input_compiles_the_step_again():
    """The client's stall after its first set: a step compiled once (A0's ``ArmTracking`` is a ``jax.jit``) read a live
    input under its label only once one was set, so the first set compiled it again, 2 to 5 s in which the session
    answered nothing. A session now hands every live input to the step at its declared default from its first tick:
    after the first chunk, 33 sets in every form a client gives, each with a chunk after it, trace and compile nothing,
    and the angle is the last value's. The control: the same rig with the cause put back (the defaults cleared before
    the first chunk, so that the step's first calls read no offset, as before) compiles at its first set and at no
    other."""
    for control in (False, True):
        session = jit_steered().build().live(pace=None, select=["spin.motor.angle"])
        if control:
            session._callback.live.clear()
        with compilations() as first:
            session.step(1)
        assert first  # the counter hears the first chunk's compile
        each = []
        for value in VALUES * 3:
            with compilations() as seen:
                session.set("offset", value)
                held = session.step(1)[-1]
            each.append(len(seen))
        ticks = (held.step - 1) // 3
        assert angle(held) == pytest.approx(RATE * ticks + float(VALUES[-1]), rel=1e-6)  # float32's, on its scale
        if control:
            assert each[0] > 0 and each[1:] == [0] * (len(each) - 1)
        else:
            assert each == [0] * len(each)


def gated(session):
    """The session's chunks held open until the event returned is set: a long first compile, without the clock."""
    gate = threading.Event()
    run = session.built._runs.run

    def held(*args, **kwargs):
        assert gate.wait(timeout=600)  # a guard against a hang, no bound on speed
        return run(*args, **kwargs)

    session.built._runs.run = held
    return gate


def test_status_and_hello_are_answered_while_a_chunk_runs():
    """A chunk held open on another thread (its first, so compiling; then a later one, stepping): ``status`` and
    ``hello`` answer at once, with the step the chunk started at, and a ``set`` waits for the chunk's end, where it
    acts. Machine-free: the chunk cannot end before the test lets it, so an answer during it is an answer that did not
    wait for it. The control: the set is logged at the chunk's end step, after it, not at the step it was sent at."""
    session = steered().build().live(pace=None, select=["spin.motor.angle"])
    gate = gated(session)
    for activity in ("compiling", "stepping"):
        at = session.steps
        stepping = threading.Thread(target=session.step, args=(1,), daemon=True)
        stepping.start()
        while session.activity != activity:
            assert stepping.is_alive()
            time.sleep(0.001)
        status, hello = session.command({"command": "status"}), session.command({"command": "hello"})
        assert (status["activity"], status["step"], status["stepping"]) == (activity, at, 1)
        assert hello["activity"] == activity and hello["report"]["format"] == REPORT_FORMAT
        setting = threading.Thread(target=session.set, args=("offset", 0.5), daemon=True)
        setting.start()
        setting.join(timeout=0.05)
        assert setting.is_alive() and [line["command"] for line in session.log][-1] == "step"  # it waits
        gate.set()
        stepping.join(timeout=600)
        setting.join(timeout=600)
        assert session.log[-1] == {"step": at + session.chunk_steps, "command": "set", "label": "offset",
                                   "value": 0.5}
        gate.clear()
    gate.set()
    assert session.command({"command": "status"})["activity"] == "idle"
    session.stop()


def test_hello_and_every_run_folder_carry_the_build_reports_summary(tmp_path):
    """``hello``'s ``report`` is the build report's summary, the validity lines with it (each with its kind: declared,
    computed at the build, or stated by a record), the flags a run may raise, each setting's role and default, and the
    profile's stamp; a live session's run folder and ``run_to``'s each hold it as ``report.json``, which
    ``RunResult.read`` reads. The controls: a run folder without one (written before) reads ``None``, and a report of
    another format (a newer one) is marked unreadable with the format it states, the run read all the same."""
    held = built()
    summary = held.report_summary()
    session = held.live(pace=None, out=tmp_path / "live")  # stopped as it opens: nothing compiled
    hello = session.hello()
    assert hello["report"] == summary and summary["format"] == REPORT_FORMAT
    validity = {line["line"] for line in summary["validity"]}
    assert validity and all(any(line in text for text in summary["summary"]) for line in validity)
    assert {line["kind"] for line in summary["validity"]} <= {"declared", "computed", "record"}
    assert [flag["path"] for flag in summary["flags"]] == [flag.path for flag in held.layout.flags]
    settings = {entry["path"]: entry for entry in summary["settings"]}
    damping = settings["controller.damping"]
    assert (damping["role"], damping["default"], damping["pending"]) == (
        "numerical", "numerical", ["controller.damping (a numerical default)"])
    assert damping["taken"].startswith("controller.damping = 0.05, a numerical default")
    assert settings["controller.end_effector"]["role"] == "design" and settings["controller.end_effector"]["taken"] is None
    assert summary["profile"]["stamp"] == held.layout.stamp.text() and "controller.damping (a numerical default)" in \
        summary["profile"]["pending"]
    written = session.stop()
    run = held.run_to(tmp_path / "batch", ms(8))
    for folder in (written, run.outputs.parent):
        assert json.loads((folder / RUN_REPORT).read_text()) == summary == RunResult.read(folder).report
    (written / RUN_REPORT).unlink()
    older = RunResult.read(written)
    assert older.report is None and older.report_unreadable is None  # none written: not unreadable
    readable = RunResult.read(run.outputs.parent)
    assert readable.report == summary and readable.report_unreadable is None
    (run.outputs.parent / RUN_REPORT).write_text(json.dumps({**summary, "format": "mime.report/9"}))
    newer = RunResult.read(run.outputs.parent)
    assert newer.report is None and newer.report_unreadable == "mime.report/9"
    assert newer.steps == readable.steps and newer.final.keys() == readable.final.keys()  # the run is read
    (run.outputs.parent / RUN_REPORT).write_text("{not json")
    assert RunResult.read(run.outputs.parent).report_unreadable == "not JSON"


def test_an_open_that_is_refused_leaves_no_live_input_installed(tmp_path, monkeypatch):
    """A session installs its live inputs at their defaults in the built's callback only once it has opened: an open
    refused by the stage check, or failed by an ``out`` that is not a folder, leaves the callback as it was (none), so
    nothing of a session that never was reaches a later batch run. The control: an open that succeeds installs them."""
    from mime.build._refusals import Refusal, RefusalClass
    from mime.live import _session

    held = jit_steered().build()
    callback = held._runs.callback  # pyright: ignore[reportPrivateUsage]
    assert callback.live == {}

    def refuse(self) -> None:
        raise ArgumentRefused([Refusal("live.stage-writer", RefusalClass.USER, "planted", "none")])

    with monkeypatch.context() as patch:
        patch.setattr(_session.LiveSession, "_refuse_an_older_stage", refuse)
        with pytest.raises(ArgumentRefused, match="live.stage-writer"):
            held.live(pace=None)
    assert callback.live == {}
    (tmp_path / "afile").write_text("x")
    with pytest.raises(Exception):  # noqa: B017, PT011 - whichever error an out that is a file raises, the callback is clean
        held.live(pace=None, out=tmp_path / "afile")
    assert callback.live == {}
    opened = held.live(pace=None)
    assert set(callback.live) == {"offset"}
    opened.stop()
