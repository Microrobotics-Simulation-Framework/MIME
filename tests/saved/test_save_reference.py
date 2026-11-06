"""The save's reference outputs taken from the run's own graph (saved form 8), and the usable CPUs and the BLAS threads
in the lock's environment.

A save runs the reference steps on the build's own graph, reset to its initial state and put back where the runs left
it: by the history scan of one step where a run chunked by one step compiled it (the lock names it, ``one-step``, and
nothing is built or compiled), else by the history scan of whole control periods (nothing is built). Where the compiled
step reads anything the build did not give it, the graph is built afresh, as every save did before.

**What carries the contract** is that ``verify`` reruns, on a graph built afresh, the program the lock names: the rerun
and the save then run one program, and agree bit for bit within one environment whatever another program computes.
``test_reference_table.py`` measures it case by case.

**The controls.** The compile listener sees a compile: a save after a run chunked by two compiles, and one with the
guard tripped builds a graph. Saving then running on is bit for bit against a run that never saved, and the same with
the state left where the reference steps put it (the restore planted away) differs on every case it is run on; so does
the same with only the internal step counters left (``_meta``), after a step count the multi-rate observation's
divider does not divide, which four steps hid. The
guard's plants move the reference (a replaced parameter leaf after a run chunked by one; a static array edited in place
after a run chunked by two, which the history scan then traces): a reference taken from the stale graph differs from a
fresh graph's on each, so the test can fail; an edit in place is seen only by the bytes of a NumPy array, not by its
identity (planted away, the guard misses it). An edit in place after a run chunked by one cannot be told: the cached
program holds the value it was traced with, and its reference is the fresh graph's. A one-ulp change at any one array of
the saved reference fails ``verify``, every array tried. The lock's program cannot be planted wrong on a case where the
two programs agree; it is planted on one where they differ (script A, float64), and ``verify`` then fails."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest
from maddening import GraphManager
from _platform_support import needs_readable_cpu_flags
from chain_cases import tracked_rpm
from script_a import script_a

from mime.build import construct
from mime.build import _maddening
from mime.build._maddening import state_put_back, step_reads
from mime.experiment import BuiltExperiment, Experiment
from mime.experiment import _experiment
from mime.saved import LOCK_FILE, REFERENCE_FILE, read_folder
from mime.saved import _folder, _record
from mime.saved._folder import arrays_hash
from mime.testing._saved import _compilations
from mime.testing.toys import full_toy
from mime.units import s

#: An affinity to set in a child process: ``os.sched_setaffinity``, which Linux has.
needs_affinity = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"),
                                    reason="sets a child process's affinity (os.sched_setaffinity, Linux only)")
HERE = Path(__file__).parent
TESTS = HERE.parent
STEPS = 8
#: Steps run before a save, and after it: not ``STEPS``, so that a state the reference steps left behind is not the
#: state the run reached.
BEFORE, AFTER = 4, 4

type History = dict[str, dict[str, np.ndarray]]


def dt_of(built: BuiltExperiment) -> float:
    return float(built.report.dt)  # type: ignore[arg-type]


def run_chunked_by_one(built: BuiltExperiment, steps: int) -> Any:
    """``steps`` steps chunked by one, keeping every step's state: the one-step history program is compiled."""
    if built._runs.chunk is not None:  # noqa: SLF001  a controller sets the chunks
        return built.run(s(steps * dt_of(built)), history=True)
    return built.run(s(steps * dt_of(built)), history=True, hook=lambda chunk: None, chunk=s(dt_of(built)))


def run_unchunked(built: BuiltExperiment, steps: int) -> Any:
    """``steps`` steps in the automatic chunks of two or more, keeping every step's state."""
    return built.run(s(steps * dt_of(built)), history=True)


def fresh_reference(built: BuiltExperiment, program: str | None = None) -> History:
    return built._history_on(construct(built._plans, built._law)[0], STEPS, program)  # noqa: SLF001


def stale_reference(built: BuiltExperiment, program: str | None) -> History:
    """The reference taken from the run's graph whatever the guard says: what a save without the guard would write."""
    with state_put_back(built.graph):
        built.graph.reset_state()
        return built._history_on(built.graph, STEPS, program)  # noqa: SLF001


def saved_reference(folder: Path) -> History:
    found: History = {}
    for key, value in read_folder(folder).reference.items():
        if key.startswith("history/"):
            _, node, field = key.split("/", 2)
            found.setdefault(node, {})[field] = value
    return found


def differing(a: History, b: History) -> list[str]:
    return sorted(f"{node}.{field}" for node, fields in b.items() for field, value in fields.items()
                  if not np.array_equal(np.asarray(a[node][field]), np.asarray(value)))


@contextmanager
def counted_constructs() -> Iterator[list[int]]:
    """How many graphs ``BuiltExperiment`` constructs while the block runs."""
    calls: list[int] = []
    real = _experiment.construct

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_experiment, "construct", counted)
        yield calls


# --- nothing compiled, and the listener sees a compile --------------------------------------------------------------


def compiles_in_save(built: BuiltExperiment, folder: Path) -> tuple[int, int, list[int]]:
    traces = built.graph.scan_trace_count
    with counted_constructs() as constructed, _compilations() as compiled:
        built.save(folder)
    return len(compiled), built.graph.scan_trace_count - traces, constructed


@pytest.mark.parametrize("make", [full_toy, script_a], ids=["full_toy", "script_a"])
def test_a_save_after_a_run_chunked_by_one_step_builds_and_compiles_nothing(make: Callable[[], Experiment],
                                                                           tmp_path: Path):
    built = make().build()
    run_chunked_by_one(built, BEFORE)
    compiled, traced, constructed = compiles_in_save(built, tmp_path / "saved")
    assert (compiled, traced, constructed) == (0, 0, [])
    assert read_folder(tmp_path / "saved").lock["reference"]["program"] == "one-step"


def test_control_the_listener_sees_the_history_scan_compiled_and_a_build_with_the_guard_tripped(tmp_path: Path):
    """A run chunked by two holds no one-step program: the save compiles the history scan on the run's graph (and
    builds nothing). With a parameter leaf replaced by an equal one (a new object: the reference is the same either
    way), the guard sends the save to a fresh graph, which is built, and its history scan compiled. The toy's build
    compiles nothing of its own (an MRS drag's compiles its held matrix), so the build is counted, not its compiles."""
    built = full_toy().build()
    run_unchunked(built, BEFORE)
    compiled, traced, constructed = compiles_in_save(built, tmp_path / "chunked_by_two")
    assert compiled >= 1 and traced == 1 and constructed == []
    assert "program" not in read_folder(tmp_path / "chunked_by_two").lock["reference"]
    tripped = full_toy().build()
    run_chunked_by_one(tripped, BEFORE)
    leaves = tripped.graph.params["nodes"]["toy.spring:probe"]
    leaves["k"] = jax.device_put(np.asarray(leaves["k"]))
    compiled_tripped, _, constructed = compiles_in_save(tripped, tmp_path / "tripped")
    assert constructed == [1] and compiled_tripped >= 1


# --- saving then running on is bit for bit -------------------------------------------------------------------------

#: A coupling group (its predictor history in ``_meta``) and an observation at half the base rate (multi-rate); a
#: controller called back every two steps, its writes one step late (its ``Callback`` state); and a run chunked by two,
#: whose save takes the history scan.
CONTINUED = {"full_toy_chunked_by_one": (full_toy, run_chunked_by_one),
             "tracked_rpm": (tracked_rpm, run_chunked_by_one),
             "full_toy_chunked_by_two": (full_toy, run_unchunked)}


def runs_on_after(make: Callable[[], Experiment], run: Callable[[BuiltExperiment, int], Any], folder: Path | None,
                  restore: Callable[[GraphManager], AbstractContextManager[None]] | None = None,
                  steps: tuple[int, int] = (BEFORE, AFTER)) -> tuple[History, History]:
    """``steps[0]`` steps, a save into ``folder`` (none where it is ``None``), ``steps[1]`` steps: their history and the
    final state. ``restore`` replaces the save's put-back of the graph's state where it is given."""
    built = make().build()
    run(built, steps[0])
    if folder is not None:
        with pytest.MonkeyPatch.context() as patch:
            if restore is not None:
                patch.setattr(_experiment, "state_put_back", restore)
            built.save(folder)
    after = run(built, steps[1])
    assert after.history is not None
    return after.history, after.final


def nothing_put_back(graph: GraphManager) -> AbstractContextManager[None]:
    """The whole put-back planted away."""
    return nullcontext()


@contextmanager
def put_back_but_the_counters(graph: GraphManager) -> Iterator[None]:
    """The put-back with the graph's internal counters (``_meta``: the step count the multi-rate observations and the
    predictor histories read) left where the reference steps put them, and everything else put back."""
    state = cast(dict[str, Any], cast(Any, graph)._state)  # noqa: SLF001
    with state_put_back(graph):
        yield
        left = dict(state["_meta"])
    state["_meta"].clear()
    state["_meta"].update(left)


@pytest.mark.parametrize("case", list(CONTINUED))
def test_saving_then_running_on_is_bit_for_bit_a_run_that_never_saved(case: str, tmp_path: Path):
    make, run = CONTINUED[case]
    saved_history, saved_final = runs_on_after(make, run, tmp_path / "saved")
    history, final = runs_on_after(make, run, None)
    assert differing(saved_history, history) == [] and differing(saved_final, final) == []
    assert read_folder(tmp_path / "saved").lock["reference"].get("program") == (
        None if run is run_unchunked else "one-step")


@pytest.mark.parametrize("case", list(CONTINUED))
def test_control_the_state_left_where_the_reference_steps_put_it_differs_on_every_case(case: str, tmp_path: Path):
    make, run = CONTINUED[case]
    planted_history, _ = runs_on_after(make, run, tmp_path / "saved", restore=nothing_put_back)
    history, _ = runs_on_after(make, run, None)
    assert differing(planted_history, history) != []


#: Step counts that are not multiples of the multi-rate observation's divider (the toy's recorder reads every second
#: step): a step counter put back wrong, or not at all, lands in the same phase after a count that is one, and the
#: observation's phase is read from it. The cases above run four and four, which hides that.
ODD_STEPS = [(3, 3), (5, 4)]
MULTI_RATE = {name: CONTINUED[name] for name in ("full_toy_chunked_by_one", "full_toy_chunked_by_two")}


@pytest.mark.parametrize("steps", ODD_STEPS, ids=str)
@pytest.mark.parametrize("case", list(MULTI_RATE))
def test_saving_after_a_step_count_the_observations_divider_does_not_divide_is_bit_for_bit_a_run_that_never_saved(
        case: str, steps: tuple[int, int], tmp_path: Path):
    make, run = MULTI_RATE[case]
    saved_history, saved_final = runs_on_after(make, run, tmp_path / "saved", steps=steps)
    history, final = runs_on_after(make, run, None, steps=steps)
    assert differing(saved_history, history) == [] and differing(saved_final, final) == []


@pytest.mark.parametrize("steps", ODD_STEPS, ids=str)
@pytest.mark.parametrize("case", list(MULTI_RATE))
def test_control_the_counters_left_where_the_reference_steps_put_them_differ_after_an_odd_step_count(
        case: str, steps: tuple[int, int], tmp_path: Path):
    """Only the ``_meta`` counters planted away from the put-back: the run on differs from one that never saved, on
    every case here. Nothing is drawn, so it fails on every run."""
    make, run = MULTI_RATE[case]
    planted_history, planted_final = runs_on_after(make, run, tmp_path / "saved", restore=put_back_but_the_counters,
                                                   steps=steps)
    history, final = runs_on_after(make, run, None, steps=steps)
    assert differing(planted_history, history) != [] or differing(planted_final, final) != []


# --- the guard -----------------------------------------------------------------------------------------------------


def replace_a_parameter(built: BuiltExperiment) -> None:
    leaves = built.graph.params["nodes"]["toy.spring:probe"]
    leaves["k"] = jax.device_put(np.asarray(leaves["k"]) * 1.5 + 0.25)


def edit_a_static_array_in_place(built: BuiltExperiment) -> None:
    static = built.graph.get_node("source.push").static_data["phase"]
    static.value[...] += 0.5


#: Each plant, and the run before it: a parameter leaf is read by the compiled program at every call, so it moves a
#: reference the cached one-step program takes; a static array is a constant of the program traced after it, the
#: history scan of a run chunked by two.
PLANTS = {"replaced_parameter_leaf": (replace_a_parameter, run_chunked_by_one, "one-step"),
          "static_array_edited_in_place": (edit_a_static_array_in_place, run_unchunked, None)}


@pytest.mark.parametrize("plant", list(PLANTS))
def test_a_graph_edited_after_the_build_is_saved_from_a_fresh_graph(plant: str, tmp_path: Path):
    edit, run, program = PLANTS[plant]
    built = full_toy().build()
    run(built, BEFORE)
    edit(built)
    with counted_constructs() as constructed:
        built.save(tmp_path / "saved")
    assert constructed == [1]
    assert "program" not in read_folder(tmp_path / "saved").lock["reference"]
    assert differing(saved_reference(tmp_path / "saved"), fresh_reference(built)) == []
    assert differing(stale_reference(built, program), fresh_reference(built)) != [], \
        "the plant moves the reference: a save that took it from the edited graph would write another"


def test_control_an_edit_in_place_is_seen_by_the_bytes_of_the_array_and_not_by_its_identity(
        monkeypatch: pytest.MonkeyPatch):
    built = full_toy().build()
    run_unchunked(built, BEFORE)
    edit_a_static_array_in_place(built)
    assert not step_reads(built.graph).unchanged_since(built._built_reads)  # noqa: SLF001
    monkeypatch.setattr(_maddening, "_content", lambda value: None)
    unhashed = full_toy().build()
    run_unchunked(unhashed, BEFORE)
    edit_a_static_array_in_place(unhashed)
    assert step_reads(unhashed.graph).unchanged_since(unhashed._built_reads)  # noqa: SLF001


# --- verify reruns the lock's program on a fresh graph ---------------------------------------------------------------


@pytest.fixture
def one_step_folder(tmp_path: Path) -> Path:
    built = full_toy().build()
    run_chunked_by_one(built, BEFORE)
    built.save(tmp_path / "one_step")
    assert read_folder(tmp_path / "one_step").lock["reference"]["program"] == "one-step"
    return tmp_path / "one_step"


@needs_readable_cpu_flags
def test_verify_builds_one_fresh_graph_and_reproduces_a_one_step_reference_bit_for_bit(one_step_folder: Path):
    built = Experiment.load(one_step_folder).build()
    with counted_constructs() as constructed:
        report = built.verify()
    assert constructed == [1] and report.reproduced and report.same_environment


@needs_readable_cpu_flags
def test_mime_verify_in_a_fresh_process_reproduces_a_one_step_reference_bit_for_bit(one_step_folder: Path):
    done = subprocess.run([sys.executable, "-c", "import sys; from mime.cli import main; sys.exit(main(sys.argv[1:]))",
                           "verify", str(one_step_folder)], capture_output=True, text=True, check=False,
                          env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verify: reproduced 8 steps bit for bit" in done.stdout


def with_reference(folder: Path, change: Callable[[dict[str, np.ndarray]], None]) -> None:
    """``folder``'s reference outputs changed by ``change``, written back with the lock's hash of them, so the load
    reads them and only ``verify`` can tell."""
    with np.load(folder / REFERENCE_FILE, allow_pickle=False) as held:
        arrays = {key: np.array(held[key]) for key in held.files}
    change(arrays)
    np.savez(folder / REFERENCE_FILE, allow_pickle=False, **arrays)
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    lock["reference"]["sha256"] = arrays_hash(arrays)
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")


@needs_readable_cpu_flags
def test_control_a_one_ulp_change_at_any_array_of_the_reference_fails_verify(one_step_folder: Path, tmp_path: Path):
    """One element of one array at step 3 moved by one ulp, for every array of floating type in turn: ``verify``'s
    resolution is the last bit of one element, and each plant is caught."""
    import shutil

    keys = [key for key, value in read_folder(one_step_folder).reference.items()
            if key.startswith("history/") and np.issubdtype(value.dtype, np.floating) and value.size]
    caught = []
    for index, key in enumerate(keys):
        folder = Path(shutil.copytree(one_step_folder, tmp_path / str(index)))

        def nudge(arrays: dict[str, np.ndarray], key: str = key) -> None:
            flat = arrays[key].reshape(arrays[key].shape[0], -1)
            flat[3, 0] = np.nextafter(flat[3, 0], np.inf)

        with_reference(folder, nudge)
        report = Experiment.load(folder).build().verify()
        label = key.removeprefix("history/").replace("/", ".", 1)
        caught.append(not report.reproduced and any(line.startswith(f"{label}:") and "first at step 3" in line
                                                    for line in report.moved))
    assert keys and all(caught), dict(zip(keys, caught, strict=True))


def script_a_folder(folder: Path, run: Callable[[BuiltExperiment, int], Any]) -> Path:
    built = script_a().build()
    run(built, BEFORE)
    built.save(folder)
    return folder


@needs_readable_cpu_flags
@pytest.mark.parametrize("planted", ["one_step_named_none", "history_named_one_step"])
def test_control_the_lock_naming_the_other_program_fails_verify_where_the_two_differ(planted: str, tmp_path: Path):
    """Script A in float64, where a scan of one step and the history scan differ in their last bits (measured: the
    drag's resistance from step 0, 1.3e-18): a folder whose lock names the program its reference was not run by fails
    ``verify``. Unedited, each verifies (the test's arm before the plant)."""
    run = run_chunked_by_one if planted == "one_step_named_none" else run_unchunked
    folder = script_a_folder(tmp_path / planted, run)
    assert Experiment.load(folder).build().verify().reproduced
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    if planted == "one_step_named_none":
        del lock["reference"]["program"]
    else:
        lock["reference"]["program"] = "one-step"
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    assert not Experiment.load(folder).build().verify().reproduced


# --- the usable CPUs: another count is another environment ----------------------------------------------------------


@pytest.fixture
def toy_folder(tmp_path: Path) -> Path:
    full_toy().save(tmp_path / "toy")
    return tmp_path / "toy"


@needs_readable_cpu_flags
def test_a_folder_verified_at_the_count_it_was_saved_at_is_the_same_environment_bit_for_bit(toy_folder: Path):
    lock = read_folder(toy_folder).lock
    assert lock["environment"]["usable_cpus"] == _record.usable_cpus()
    loaded = Experiment.load(toy_folder)
    assert loaded.load_report is not None and "environment: the one the lock was written in" in \
        loaded.load_report.text()
    report = loaded.build().verify()
    assert report.same_environment and report.reproduced and "bit for bit" in report.text()


def test_a_folder_verified_at_another_count_is_another_environment(toy_folder: Path,
                                                                   monkeypatch: pytest.MonkeyPatch):
    """Not asserted: that the comparison within the tolerance then passes (the tolerance is absolute below its scale,
    MIME-ANO-141)."""
    saved = read_folder(toy_folder).lock["environment"]["usable_cpus"]
    monkeypatch.setattr(_record, "usable_cpus", lambda: saved + 1)
    loaded = Experiment.load(toy_folder)
    assert loaded.load_report is not None and not loaded.load_report.same_environment
    assert f"usable_cpus: built under {saved!r}, loaded under {saved + 1!r}" in loaded.load_report.environment
    report = loaded.build().verify()
    assert not report.same_environment and report.tolerance.startswith("within rtol")


@needs_affinity
def test_mime_verify_under_another_affinity_in_a_fresh_process_reports_another_environment(toy_folder: Path):
    """The process's affinity set to one CPU before JAX starts, as ``taskset -c`` would: XLA's thread pools are sized
    by it, and the lock's count is another."""
    usable = sorted(os.sched_getaffinity(0))
    assert len(usable) >= 2, "this test needs two usable CPUs to set another count"
    saved = read_folder(toy_folder).lock["environment"]["usable_cpus"]
    script = ("import os, sys; os.sched_setaffinity(0, {int(sys.argv[1])}); from mime.cli import main; "
              "sys.exit(main(['verify', sys.argv[2]]))")
    done = subprocess.run([sys.executable, "-c", script, str(usable[0]), str(toy_folder)], capture_output=True,
                          text=True, check=False, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert done.returncode in (0, 1), done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert "environment: built under another; results are not guaranteed to match: run verify" in lines
    assert f"  usable_cpus: built under {saved!r}, loaded under 1" in lines
    assert "bit for bit" not in done.stdout and (" within rtol " in done.stdout or "moved over" in done.stdout)


@pytest.mark.parametrize("count", ["usable_cpus", "blas_threads"])
def test_a_count_either_side_could_not_read_is_never_the_same(count: str):
    assert _folder._differences_of(count, None, None) == [  # noqa: SLF001
        f"{count}: built under None, loaded under None (unreadable, so not the same)"]
    assert _folder._differences_of(count, 4, 4) == []  # noqa: SLF001


# --- the BLAS threads: another count is another environment --------------------------------------------------------
#
# The compiled steps of the MRS drag solve on JAX's CPU LAPACK kernels, which run on the OpenBLAS SciPy ships; a routine
# that large splits its sums over that pool's threads, and another count gives other last bits (MIME-ANO-140). The
# count is OpenBLAS's own: ``OPENBLAS_NUM_THREADS`` sets it, and the affinity does where that is unset.


def test_the_lock_records_the_blas_threads_and_a_folder_loaded_at_that_count_is_the_same_environment(toy_folder: Path):
    lock = read_folder(toy_folder).lock
    assert lock["environment"]["blas_threads"] == _record.blas_threads()
    assert isinstance(lock["environment"]["blas_threads"], int) and lock["environment"]["blas_threads"] >= 1
    assert not any("blas_threads" in line for line in _folder.environment_differences(lock))


def another_blas_count_is_another_environment(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved = read_folder(folder).lock["environment"]["blas_threads"]
    monkeypatch.setattr(_record, "blas_threads", lambda: saved + 1)
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None and not loaded.load_report.same_environment
    assert f"blas_threads: built under {saved!r}, loaded under {saved + 1!r}" in loaded.load_report.environment
    report = loaded.build().verify()
    assert not report.same_environment and report.tolerance.startswith("within rtol")


def test_a_folder_loaded_at_another_blas_count_is_another_environment(toy_folder: Path,
                                                                      monkeypatch: pytest.MonkeyPatch):
    another_blas_count_is_another_environment(toy_folder, monkeypatch)


def test_control_the_blas_count_dropped_from_the_comparison_fails_the_test_above(toy_folder: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch):
    """The count planted away from the comparison (read as information only, as the runner's information is): the
    assertion that another count is another environment then fails. Nothing here is drawn, so it fails on every run."""
    monkeypatch.setattr(_folder, "INFORMATION_ONLY", _folder.INFORMATION_ONLY | {"blas_threads"})
    with pytest.raises(AssertionError):
        another_blas_count_is_another_environment(toy_folder, monkeypatch)


#: A child process that saves the MRS point-force case under its own BLAS pool and verifies it there, or
#: verifies a folder; it prints the pool's size and the verify report. ``OPENBLAS_NUM_THREADS`` is set by the parent.
BLAS_CHILD = """
import sys
import os
sys.path[:0] = sys.argv[2].split(os.pathsep)
from mime.env._host import blas_threads
print('blas_threads', blas_threads())
if sys.argv[1] == 'save':
    from cases import CASES
    from mime.experiment import Experiment
    next(case.make for case in CASES if case.name == 'mrs_point_forces')().save(sys.argv[3])
    loaded = Experiment.load(sys.argv[3])
    print(loaded.load_report.text())
    print(loaded.build().verify().text())
else:
    from mime.cli import main
    sys.exit(main(['verify', sys.argv[3]]))
"""


def run_blas_child(mode: str, threads: int, folder: Path) -> tuple[int, list[str]]:
    done = subprocess.run([sys.executable, "-c", BLAS_CHILD, mode, os.pathsep.join([str(HERE), str(TESTS)]),
                           str(folder)], capture_output=True,
                          text=True, check=False, env={**os.environ, "OPENBLAS_NUM_THREADS": str(threads)})
    assert done.returncode in ((0,) if mode == "save" else (0, 1)), done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == f"blas_threads {threads}", done.stdout + done.stderr
    return done.returncode, lines


@needs_readable_cpu_flags
def test_the_mrs_case_saved_at_one_blas_count_reproduces_at_it_and_is_another_environment_at_another(tmp_path: Path):
    """A fresh process per count, the pool set by ``OPENBLAS_NUM_THREADS`` as a user would. At the saving count the
    folder loads as the same environment and verifies bit for bit; at another the environment names the count and the
    comparison is the tolerance's. Not asserted: that the tolerance then passes (MIME-ANO-141)."""
    folder = tmp_path / "mrs"
    _, saved = run_blas_child("save", 1, folder)
    assert "environment: the one the lock was written in" in saved and any("bit for bit" in line for line in saved)
    assert read_folder(folder).lock["environment"]["blas_threads"] == 1
    _, other = run_blas_child("verify", 2, folder)
    assert "environment: built under another; results are not guaranteed to match: run verify" in other
    assert "  blas_threads: built under 1, loaded under 2" in other
    assert not any("bit for bit" in line for line in other)
