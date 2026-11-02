"""Ctrl+C on ``mime run``. A run is compiled scans, and Python runs a signal handler between bytecodes, so a SIGINT used
to wait for the scan to return, which for a run with one extra zero in its duration is hours. Now a run is split into
about 64 compiled chunks (``mime.control._run.automatic_chunks``), which change nothing it computes; the first Ctrl+C
is acknowledged at once by a thread reading the signal's wakeup descriptor (``mime._interrupt``) and stops the run
at its next chunk, written as the steps reached, marked stopped, exit 130; the second quits within a second wherever
the main thread is, inside a compiled chunk too, leaving no ``outputs.json``.

Each with its control. The chunking: the same run compared against the steps one off. The stop: a subprocess running
the same experiment as one compiled scan under Python's own handler (what ``mime run`` was) has not stopped within the
bound after one press, nor after two. The subprocess tests send SIGINT to a child process and so need POSIX; the tests
of the machinery in process raise it with ``signal.raise_signal`` and run on every platform."""

import json
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import jax
import numpy as np
import pytest
from _platform_support import needs_posix_sigint
from hypothesis import example, given
from hypothesis import strategies as st
from process_groups import stop_group

from mime._interrupt import INTERRUPTED, USER_STOP
from mime.control._run import MIN_CHUNK, TARGET_CHUNKS, automatic_chunks
from mime.experiment import Experiment
from mime.results import RunResult
from mime.saved import read_folder
from mime.testing.toys import full_toy
from mime.units import s

#: The bound on a run that is stopped by one press, from the press to the exit: at most one chunk (about a second on a
#: developer's machine, here a toy's), the compile and eight steps of the saved form, and a loaded runner's slack.
STOP_BOUND = 60.0
#: The bound on the second press, from the press to the exit ("about a second"; the rest is a loaded runner's slack).
QUIT_BOUND = 10.0
#: How long a process may take to get to its run: imports, load, build. A guard against a hang, not a measure.
START_BOUND = 180.0


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("toy")
    full_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    held = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", held)


def dt_of(folder: Path) -> float:
    return read_folder(folder).lock["resolved"]["run"]["dt"]


class Child:
    """A child process in a process group of its own, its standard output read by a thread so that a test can wait for a
    line with a deadline; killed with its group when the test ends."""

    def __init__(self, args: list[str]) -> None:
        options: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" \
            else {"start_new_session": True}
        self.process = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, **options)
        self.lines: queue.Queue[str] = queue.Queue()
        self.seen: list[str] = []
        self.errors: list[str] = []
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read_out(self) -> None:
        for line in self.process.stdout:  # type: ignore[union-attr]
            self.lines.put(line)

    def _read_err(self) -> None:
        for line in self.process.stderr:  # type: ignore[union-attr]
            self.errors.append(line)

    def wait_for_line(self, text: str, bound: float = START_BOUND) -> None:
        deadline = time.monotonic() + bound
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.1)
            except queue.Empty:
                assert self.process.poll() is None, f"the process exited before printing {text!r}: {self.stderr()}"
                continue
            self.seen.append(line)
            if text in line:
                return
        raise AssertionError(f"no line with {text!r} within {bound} s; saw {self.seen}")

    def stderr(self) -> str:
        time.sleep(0.2)  # the reader thread drains the pipe after the exit
        return "".join(self.errors)

    def stop(self) -> None:
        stop_group(self.process)


def run_command(folder: Path, out: Path, duration: float) -> list[str]:
    return ["-m", "mime.cli", "run", str(folder), "--duration", repr(duration), "--out", str(out)]


#: A run of the toy rig long enough that its first chunk is about a second or more on any runner, and its whole run
#: minutes to hours: stopping it any other way than at a chunk or by the second press would show.
LONG_STEPS = 64 * 4 * 10**6
#: And one whose chunks are long on purpose, so that the second press arrives inside a compiled chunk (a chunk of
#: 2e8 steps is minutes of the toy's 0.4 us steps).
HUGE_STEPS = 5 * 10**10


# -- the first press, in a process of its own ---------------------------------------------------------------------------


@needs_posix_sigint
def test_the_first_ctrl_c_stops_the_run_at_a_chunk_and_writes_it_as_stopped_exit_130(saved, tmp_path):
    out = tmp_path / "run"
    child = Child(run_command(saved, out, LONG_STEPS * dt_of(saved)))
    try:
        child.wait_for_line("running")
        time.sleep(1.0)
        pressed = time.monotonic()
        child.process.send_signal(signal.SIGINT)
        code = child.process.wait(timeout=STOP_BOUND)
        took = time.monotonic() - pressed
        errors = child.stderr()
    finally:
        child.stop()
    assert code == INTERRUPTED == 130 and took < STOP_BOUND
    assert errors.count("Ctrl+C: stopping at the end of this chunk") == 1 and "Traceback" not in errors
    assert "stopped by the user (Ctrl+C)" in "".join(child.seen + list(child.lines.queue))
    chunk = automatic_chunks(LONG_STEPS)[0]
    manifest = json.loads((out / "outputs.json").read_text(encoding="utf-8"))
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    for stopped in (manifest["stopped"], record["stopped"]):
        assert stopped["reason"] == USER_STOP and stopped["steps_asked"] == LONG_STEPS
        assert stopped["time_asked"] == LONG_STEPS * dt_of(saved)
        assert 0 < stopped["steps_reached"] < LONG_STEPS and stopped["steps_reached"] % chunk == 0
    assert manifest["steps"] == record["steps"] == manifest["stopped"]["steps_reached"]
    result = RunResult.read(out)
    assert result.steps == manifest["steps"] and result.stopped.steps_asked == LONG_STEPS
    assert "STOPPED" in result.text()
    assert (out / "experiment").is_dir() and read_folder(out / "experiment").lock


@needs_posix_sigint
def test_two_ctrl_cs_inside_a_compiled_chunk_quit_at_once_and_leave_no_finished_run(saved, tmp_path):
    out = tmp_path / "run"
    child = Child(run_command(saved, out, HUGE_STEPS * dt_of(saved)))
    try:
        child.wait_for_line("running")
        time.sleep(3.0)  # past the compile, inside the first chunk: minutes long
        assert child.process.poll() is None
        child.process.send_signal(signal.SIGINT)
        time.sleep(0.5)
        assert child.process.poll() is None, "one press stops a run at its next chunk, not at once"
        pressed = time.monotonic()
        child.process.send_signal(signal.SIGINT)
        code = child.process.wait(timeout=QUIT_BOUND)
        took = time.monotonic() - pressed
        errors = child.stderr()
    finally:
        child.stop()
    assert code == INTERRUPTED and took < QUIT_BOUND
    assert "quitting now, nothing written" in errors and "Traceback" not in errors
    assert not (out / "outputs.json").exists() and not (out / "run.json").exists()


CONTROL = """
import sys

import numpy as np

from mime.build._maddening import fed_inputs, scan_chunk
from mime.env import apply_lock_precision
from mime.experiment import Experiment

folder, steps = sys.argv[1], int(sys.argv[2])
apply_lock_precision(folder)
graph = Experiment.load(folder).build()._runs.graph
fed_inputs(graph, {})
for _ in range(6):  # MADDENING waits for a graph's first few stored states, a wait a second press can interrupt
    scan_chunk(graph, 2, {})
print("ready", flush=True)
final = scan_chunk(graph, steps, {})  # one compiled scan, then the state on the host: what mime run did, whole
{node: {name: np.asarray(value) for name, value in fields.items()} for node, fields in final.items()}
"""


@needs_posix_sigint
@pytest.mark.parametrize("presses", [1, 2])
def test_control_without_the_handling_the_process_has_not_stopped_within_the_bound(saved, presses):
    """The same experiment as one compiled scan under Python's own handler (what ``mime run`` was) is still running
    ``QUIT_BOUND`` seconds after one press and after two, so the tests above can fail: they do not pass on a process that
    stops by itself."""
    child = Child(["-c", CONTROL, str(saved), str(10**9)])
    try:
        child.wait_for_line("ready")
        time.sleep(2.0)
        for _ in range(presses):
            child.process.send_signal(signal.SIGINT)
            time.sleep(0.5)
        with pytest.raises(subprocess.TimeoutExpired):
            child.process.wait(timeout=QUIT_BOUND)
    finally:
        child.stop()


# -- the chunks change nothing it computes -------------------------------------------------------------------------------


def digest(state) -> dict[str, bytes]:
    return {f"{node}.{name}": np.asarray(value).tobytes() for node, fields in state.items()
            for name, value in fields.items()}


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_a_run_in_chunks_is_the_one_scan_bit_for_bit_in_its_state_and_its_history(x64):
    """The toy rig and the live rig (a controller inside the compiled step), run for 60 steps in the chunks the run
    takes (30 of two steps) and as one compiled scan: every field of the final state, and of every step, equal in every
    byte. The control: the one scan one step longer is not the chunked run's state."""
    from live_cases import in_graph

    jax.config.update("jax_enable_x64", x64)
    steps = 60
    for make in (full_toy, in_graph):
        built = make().build()
        assert automatic_chunks(steps) == (MIN_CHUNK,) * (steps // MIN_CHUNK)
        chunked = built.run(s(steps * built.plans.dt), history=True)
        oracle_final, oracle_history = make().build()._runs.run(steps, history=True)  # pyright: ignore[reportPrivateUsage]
        assert digest(chunked.final) == digest(oracle_final)
        assert digest(chunked.history) == digest(oracle_history)
        chunked_final = make().build().run(s(steps * built.plans.dt)).final  # no history: the scans' own state
        assert digest(chunked_final) == digest(oracle_final)
        longer_final, _ = make().build()._runs.run(steps + 1, history=False)  # pyright: ignore[reportPrivateUsage]
        assert digest(chunked_final) != digest(longer_final)


def test_a_run_stopped_at_a_chunk_is_the_one_scan_at_that_step_bit_for_bit(saved):
    built = Experiment.load(saved).build()
    steps, calls = 600, []

    def stop(progress) -> str | None:
        calls.append(progress)
        return "enough" if len(calls) == 7 else None

    chunk = automatic_chunks(steps)[0]
    result = built.run(s(steps * built.plans.dt), history=True, stop=stop)
    reached = 7 * chunk
    assert result.steps == reached and result.stopped is not None and result.stopped.reason == "enough"
    assert result.stopped.steps_asked == steps and built._runs.steps == reached  # pyright: ignore[reportPrivateUsage]
    assert [p.index for p in calls] == list(range(7)) and calls[-1].steps == reached
    assert calls[-1].time == reached * built.plans.dt and {p.end for p in calls} == {steps}
    _, oracle = Experiment.load(saved).build()._runs.run(steps, history=True)  # pyright: ignore[reportPrivateUsage]
    assert digest(result.history) == digest({n: {f: v[:reached] for f, v in fields.items()}
                                             for n, fields in oracle.items()})
    again = Experiment.load(saved).build().run(s(steps * built.plans.dt), history=True, stop=lambda progress: None)
    assert again.stopped is None and again.steps == steps


def plan_failures(steps: int, plan: tuple[int, ...]) -> list[str]:
    """What is wrong with the lengths of the chunks a run of ``steps`` steps takes, by the rules the plan states."""
    found = []
    if sum(plan) != steps:
        found.append(f"the chunks add to {sum(plan)}, not {steps}")
    if len(set(plan)) > 2:
        found.append(f"{len(set(plan))} compiled lengths")
    if steps >= 2 * MIN_CHUNK and min(plan) < MIN_CHUNK:
        found.append(f"a chunk of {min(plan)} step")
    aim = max(MIN_CHUNK, round(steps / TARGET_CHUNKS))
    if steps >= 2 * TARGET_CHUNKS * MIN_CHUNK and not steps / (2 * aim) <= len(plan) <= 2 * steps / aim + 1:
        found.append(f"{len(plan)} chunks, where the aim is {steps / aim:.1f} and a length may be half to twice it")
    near = [d for d in range(max(MIN_CHUNK, -(-aim // 2)), 2 * aim + 1) if steps % d == 0 and d <= steps // 2]
    if near and len(set(plan)) > 1:
        found.append(f"two lengths where {near[0]} divides the steps")
    return found


@example(1114)  # 557 chunks of two: a length at half the aim, the count at twice the aim (stress run, seed 238387334)
@given(st.integers(min_value=0, max_value=3 * 10**6))
def test_the_chunks_of_a_run_add_to_its_steps_in_at_most_two_lengths_and_one_where_a_divisor_lies_near(steps):
    assert plan_failures(steps, automatic_chunks(steps)) == []


def test_control_the_plan_check_finds_each_fault_planted_in_a_plan():
    assert any("add to" in line for line in plan_failures(1000, (4,) * 249))  # four steps lost
    assert any("compiled lengths" in line for line in plan_failures(1000, (2, 8) + (4,) * 247 + (2,)))
    assert any("a chunk of 1 step" in line for line in plan_failures(1001, (1,) + (4,) * 250))
    assert any("two lengths where" in line for line in plan_failures(1000, (8,) + (4,) * 248))
    assert any("chunks, where the aim" in line for line in plan_failures(10**6, (2,) * 500000))
    assert any("chunks, where the aim" in line for line in plan_failures(10**6, (500000,) * 2))
