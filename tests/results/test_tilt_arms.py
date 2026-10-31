"""Friction 4c's arms (the first-hour walk's screw in its tube, ``scripts/inertia_arms.py``) against the surrogate's
``bore_fired``: the flag a run reports when the body's axis leaves the tube's by more than the bore allows.

**What these bound.** For the arms whose axial moment is the smallest (as a real screw's is, and including the one that
leaves the transverse moments at the isotropic value and makes only the axial one the smallest), which turn broadside
within half a second and leave the bore, that the flag is raised at the steps the body's recorded poses put it beyond
the bound, and at no other; for the arms that stay along the axis (the isotropic body, and the anisotropic bodies whose
axial moment is the largest, the transverse moments raised to the diagonal's or left isotropic), that it is never raised and the body's axis never leaves the tube's by a degree. The
expected flag is computed here from the run's own recorded poses and the closed form of the bound, in float64, one
step behind the pose (the drag reads the pose at the start of its step), and compared step by step except at the few
steps whose extent lies within 1e-4 of the bore, where float32's rounding may decide either way. That the flag reaches
``text()``, ``outputs.json`` and the ``.mat`` file as 7k's flags do. They do not bound what the body does after it has
left the bore (the surrogate does not describe it), nor the arms' physics in any other regime.

**Controls.** The comparison itself is tried on three wrong flags, each computed from the same recorded poses, and must
disagree with the run's flag: the tube's axis read as the world's up (the body along the tube then reads as across
it) and the bore read as a radius, in every arm; the bore 10% too wide, in every arm that turns (in the others no
bore wider than the real one is exceeded either), where it moves the first raised step by at least three. The runs are
deterministic, so a plant's detection is a fact of the recording and not a rate: it is asserted arm by arm, and a
planted flag that is not caught in an arm fails the test.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mime.results import RUN_MANIFEST
from mime.units import s

ROOT = Path(__file__).resolve().parents[2]
FLAG = "drag:umr.bore_fired"
RANGE_FLAG = "drag:umr.tilt_fired"
LENGTH, OUTER, BORE = 7.47e-3, 1.33 * 1.56e-3, 2 * 3.175e-3
RUN_S = 1.0
#: The arms whose axial moment is the smallest, which turn broadside, and those that stay along the axis.
TURNING = ("diagonal", "cylinder", "isotropic axial, diagonal transverse", "transverse moments isotropic, axial the smallest")
STAYING = ("isotropic", "diagonal axial, isotropic transverse",
           "transverse moments raised to the diagonal's, axial raised past them")


@pytest.fixture(scope="module")
def arms():
    spec = importlib.util.spec_from_file_location("inertia_arms_for_the_flag", ROOT / "scripts" / "inertia_arms.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runs(arms):
    """Each arm run once for a second with its history."""
    return {name: arms.experiment(arms.ARMS[name]).build().run(s(RUN_S), history=True) for name in (*TURNING, *STAYING)}


def starts_of_steps(result) -> np.ndarray:
    """The body's orientation at the start of each step: the initial one, then each step's end but the last."""
    ends = np.asarray(result.history["umr"]["pose"], dtype=np.float64)[:, 3:]
    return np.concatenate([[[1.0, 0.0, 0.0, 0.0]], ends[:-1]])


def body_axis(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternions.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)


def extent_of(result, tube_axis=(1.0, 0.0, 0.0), bore_scale: float = 1.0) -> np.ndarray:
    """The body's extent across the tube at the start of each step, ``L sin + 2 r cos`` of its angle to the tube's axis,
    less the bore, relative to the bore (positive: beyond it)."""
    along = body_axis(starts_of_steps(result))
    cosine = np.abs(along @ np.asarray(tube_axis))
    angle = np.arccos(np.clip(cosine, 0.0, 1.0))
    return (LENGTH * np.sin(angle) + 2.0 * OUTER * np.cos(angle) - BORE * bore_scale) / (BORE * bore_scale)


def raised_in(result, flag: str = FLAG) -> np.ndarray:
    return np.asarray(result.tracks[flag].raised)


def tilt_beyond_the_limit(result, limit_degrees: float) -> np.ndarray:
    """Whether the body's angle to the tube's axis at the start of each step exceeds the declared limit, and by how
    much (positive: beyond), in the fraction of the limit."""
    along = body_axis(starts_of_steps(result))
    angle = np.degrees(np.arccos(np.clip(np.abs(along[:, 0]), 0.0, 1.0)))
    return (angle - limit_degrees) / limit_degrees


def disagreements(result, expected_extent: np.ndarray) -> int:
    """How many steps the run's flag differs from the flag the recorded poses give, apart from the steps whose extent
    lies within 1e-4 of the bore."""
    decided = np.abs(expected_extent) > 1e-4
    return int(np.sum((expected_extent > 0.0)[decided] != raised_in(result)[decided]))


@pytest.mark.parametrize("name", TURNING)
def test_an_arm_whose_axial_moment_is_the_smallest_raises_the_flag_where_its_poses_leave_the_bore(runs, name):
    result = runs[name]
    expected = extent_of(result)
    assert disagreements(result, expected) == 0
    raised = raised_in(result)
    assert raised.any() and not raised[:50].any(), "raised, and not in the first 25 ms, when the body is along the axis"
    first = float(next(r for r in result.flags if r.path == FLAG).first)
    assert 0.3 < first < 0.7, "the body turns broadside within half a second, as the arms are known to"
    assert np.degrees(np.arccos(np.clip(np.abs(body_axis(starts_of_steps(result))[-1] @ [1, 0, 0]), 0, 1))) > 45.0


@pytest.mark.parametrize("name", TURNING)
def test_the_declared_tilt_range_is_passed_before_the_bore_is_and_the_flag_follows_the_poses(runs, name):
    from mime.nodes.hydro import TILT_VALIDITY

    result = runs[name]
    expected = tilt_beyond_the_limit(result, math.degrees(TILT_VALIDITY))
    decided = np.abs(expected) > 1e-4
    assert np.array_equal((expected > 0.0)[decided], raised_in(result, RANGE_FLAG)[decided])
    first_range, first_bore = (int(np.argmax(raised_in(result, flag))) for flag in (RANGE_FLAG, FLAG))
    assert 0 < first_range < first_bore, "the declared range is passed first, the bore's limit some steps after"
    # the control: a limit a tenth of a degree too far is a different first step
    moved = tilt_beyond_the_limit(result, math.degrees(TILT_VALIDITY) + 0.5) > 0.0
    assert int(np.argmax(moved)) != first_range


@pytest.mark.parametrize("name", (*TURNING, *STAYING))
def test_every_arm_is_far_outside_the_stokes_rungs_range_by_its_turning_reynolds_number(runs, name):
    """Body inertia without the fluid's is not a consistent model at these runs: the watch flags each arm, at a turning
    Reynolds number in the hundreds, whether or not the body leaves the axis."""
    (watch,) = runs[name].watches
    assert watch.raised and watch.turning is not None and 100.0 < watch.turning < 1000.0


@pytest.mark.parametrize("name", STAYING)
def test_an_arm_that_stays_along_the_axis_never_raises_it_and_its_axis_never_leaves_the_tubes_by_a_degree(runs, name):
    result = runs[name]
    axis = body_axis(starts_of_steps(result))
    assert np.degrees(np.arccos(np.clip(np.abs(axis @ [1, 0, 0]), 0, 1))).max() < 1.0
    assert not raised_in(result).any() and not raised_in(result, RANGE_FLAG).any() and result.flags == ()
    assert disagreements(result, extent_of(result)) == 0


PLANTS = {
    "the tube's axis read as the world's up": lambda r: extent_of(r, tube_axis=(0.0, 0.0, 1.0)),
    "the bore read as a radius": lambda r: extent_of(r, bore_scale=0.5),
    "the bore 10% too wide": lambda r: extent_of(r, bore_scale=1.1),
}


@pytest.mark.parametrize("plant", PLANTS)
def test_each_planted_wrong_flag_disagrees_with_the_runs_flag_in_every_arm(runs, plant):
    for arm, result in runs.items():
        if plant == "the bore 10% too wide" and arm in STAYING:
            continue  # a body that never leaves the axis is raised by no bore wider than the real one: not a test there
        assert disagreements(result, PLANTS[plant](result)) > 0, f"{plant}: not caught in {arm}"


def test_a_bore_a_tenth_too_wide_moves_the_first_raised_step_by_at_least_three_in_every_turning_arm(runs):
    for arm in TURNING:
        result = runs[arm]
        planted = extent_of(result, bore_scale=1.1) > 0.0
        assert int(np.argmax(planted)) - int(np.argmax(raised_in(result))) >= 3, arm


def test_the_flag_reaches_text_outputs_json_and_the_mat_file_as_a_7k_flag_does(arms, tmp_path):
    from scipy.io import loadmat

    built = arms.experiment(arms.DIAGONAL).build()
    written = built.run_to(tmp_path / "run", s(0.8), record=["swim_speed:umr.speed"])
    result = written.result
    read = next(r for r in result.flags if r.path == FLAG)
    assert [r.path for r in result.flags] == [RANGE_FLAG, FLAG] and read.raised and read.first is not None
    assert FLAG in result.text() and RANGE_FLAG in result.text() and "no longer fits the tube's bore" in result.text()
    manifest = json.loads((tmp_path / "run" / RUN_MANIFEST).read_text())
    assert [f["path"] for f in manifest["flags"]] == ["drag:umr.clamp_fired", RANGE_FLAG, FLAG]
    run_flag = next(f for f in manifest["run_flags"] if f["flag"] == FLAG)
    assert run_flag["flag"] == FLAG and run_flag["raised_steps"] == read.raised and run_flag["first"] == read.first
    result.to_mat(tmp_path / "run.mat")
    held = loadmat(str(tmp_path / "run.mat"), squeeze_me=True, struct_as_record=False)
    flag = held["flags"].drag_umr_bore_fired
    assert flag.path == FLAG and bool(flag.was_raised) and len(flag.raised_steps) == read.raised
    assert float(flag.raised_times[0]) == pytest.approx(read.first)
