"""The containment watch: a body the centre-held contact (0.1.0) holds, whose hull leaves the region it is held in, is
flagged after the run.

**What these bound.** That the watch reads the surface's signed distance at the hull's points, from the body's pose, as
stated (against a second computation of the same distance over the same points, to rounding); that it agrees with an
independent computation of the distance of the body (``hull_cases``: the samples' radial distance from the tube's axis,
at a spacing a fourth of the watch's) to the samples' shortfall; that the step it names is the step of the kept pose, in
a first run and in a run that continues one; that a part placed off the body's origin or turned in the body is placed as
it is; that a body the watch cannot check is said to be unwatched; that the case a probe found (script A with FL-9's own
diagonal inertia, whose axis turns across the tube within 0.35 s under that contact) is flagged on every run, with the
check removed it passes silently, and script A as it is (its isotropic effective inertia) and the record arm are never
flagged. They do not bound the physics of the body going through the wall (nothing here says what the real body would
do), and the watch reads the hull's samples, which fall short of the body by up to the plan's ``shortfall`` (an estimate
over finitely many directions).

**Controls.** The reproduced case is deterministic, so it is flagged on every run by construction; the check removed
(``read_containments`` replaced by nothing) leaves the same run with no flag of the contact's (the surrogate drag's own
validity flags, ``tilt_fired`` and ``bore_fired``, do rise, later, and say nothing of the wall) while an independent
first-principles bound (the screw's axis end, a point of its body, past the tube's wall) shows the body through it. The
mutants are run against the same recorded poses: the depth's sign flipped, the hull replaced by the centre, and the check
run at step 0 only; the table is printed (``-s``) and each must be caught by the reproduced case or by the case that is
not flagged. The review's mutants (the step number off by one or counted from the run's start, the part's offset or
rotation dropped, the shortfall dropped from the text, the threshold shifted by the margin, the spacing doubled, a 1 % error
in the region's radius, the poses in half precision) each fail one test here on every run; the placement tests carry a
control that fails when the offset or rotation is dropped, so that they cannot pass for the wrong reason."""

import dataclasses
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, override

import numpy as np
import pytest
from hull_cases import DIAGONAL, TUBE_RADIUS, hull_points, rotation
from hypothesis import assume, given
from hypothesis import strategies as st
from script_a import MASS, VOLUME, WATER, reproduction_arm

import script_a
from mime import Body, Experiment, Medium, State, Structure
from mime.build._watch import ContainmentPlan, UnwatchedContainment
from mime.cli import _summary
from mime.core import CYLINDER
from mime.dynamics import RigidInertial
from mime.gradients import GradientStatus
from mime.interactions import Contact, Gravity
from mime.registry import Solid, register_geometry
from mime.results import _containment
from mime.results._containment import PAST, UNKNOWN, WITHIN, depths, read_containment, read_containments
from mime.scene import Cylinder, Scene, Sphere, Tube, X, Z
from mime.things import Part
from mime.units import Hz, Length, Quantity, m, m_per_s, m_per_s2, mm, ms, rad_per_s, s
from mime.vocabulary import register_namespace

DT = 5e-4
#: The first 0.5 s of script A's reproduction arm with FL-9's own diagonal inertia: the axis is across the tube by 0.35 s.
STEPS = 1000
#: The hull contact's arm, which compiles the hull contact's search (about 40 s) before its first step: a short run.
HULL_STEPS = 800
HALF_LENGTH = 7.47e-3 / 2


def plan_of(built) -> ContainmentPlan:
    (plan,) = [watch for watch in built.report.watches if isinstance(watch, ContainmentPlan)]
    return plan


def case(inertia: bool, contact=None):
    """Script A's reproduction arm, with FL-9's own diagonal inertia (``inertia``) or its isotropic effective inertia, held
    by the centre-held contact or by ``contact``."""
    held = (script_a.INERTIA, script_a.CentreContact)
    if inertia:
        script_a.INERTIA = DIAGONAL
    if contact is not None:
        script_a.CentreContact = contact
    try:
        return reproduction_arm(frequency=Hz(10.0), dt=ms(DT * 1e3)).build()
    finally:
        script_a.INERTIA, script_a.CentreContact = held


def poses_of(result) -> np.ndarray:
    return np.asarray(result.history["umr"]["pose"], dtype=np.float64)


def axis_end_excess(poses: np.ndarray) -> np.ndarray:
    """How far the screw's axis end (a point of its body, ``L / 2`` along its x) lies beyond the tube's wall at each pose
    (m, positive beyond it): a first-principles lower bound on the hull's depth, independent of the watch."""
    ends = poses[:, None, :3] + np.stack([rotation(q) @ np.array([1.0, 0.0, 0.0]) for q in poses[:, 3:]])[:, None, :] \
        * np.array([[HALF_LENGTH], [-HALF_LENGTH]])
    return np.max(np.hypot(ends[..., 1], ends[..., 2]), axis=1) - TUBE_RADIUS


@pytest.fixture(scope="module")
def reproduced() -> SimpleNamespace:
    built = case(inertia=True)
    result = built.run(s(STEPS * DT), history=True)
    return SimpleNamespace(built=built, plan=plan_of(built), result=result, poses=poses_of(result))


@pytest.fixture(scope="module")
def isotropic() -> SimpleNamespace:
    """Script A as it is, for its 5 s."""
    built = case(inertia=False)
    result = built.run(s(5.0), history=True)
    return SimpleNamespace(built=built, plan=plan_of(built), result=result, poses=poses_of(result))


def read_of(run: SimpleNamespace, poses: np.ndarray | None = None):
    held = run.poses if poses is None else poses
    return read_containment(run.plan, {"umr": {"pose": held[-1]}}, {"umr": {"pose": held[1:]}},
                            last_step=len(held) - 1, dt=DT)


# --- the reproduced case ---------------------------------------------------------------------------------------------


def test_the_axis_turns_across_the_tube_within_the_claimed_time_and_the_run_is_flagged_on_it(reproduced):
    (read,) = reproduced.result.containment
    assert read.status == PAST and read.raised, read.text()
    assert read.first_step is not None and read.first_step * DT < 0.35, read.text()
    assert read.depth is not None and read.depth > 1e-3, "the hull is through the wall by millimetres, not rounding"
    assert "Contact.Projection" in read.text() and f"step {read.worst_step:,}" in read.text()
    assert f"{read.depth * 1e3:.3g} mm" in read.text()
    assert f"{reproduced.plan.shortfall * 1e6:.3g} um short of the body" in read.text(), "the text names the shortfall"


def test_the_flag_agrees_with_the_axis_end_bound_at_every_step(reproduced):
    excess = axis_end_excess(reproduced.poses[1:])
    deep = excess > 10 * reproduced.plan.shortfall
    assert deep.any(), "the case turns the axis end well past the wall"
    per_step = depths(reproduced.plan, reproduced.poses[1:])
    assert (per_step[deep] > 0).all(), "wherever the axis end is past the wall, the hull is"
    assert (per_step[deep] >= excess[deep] - reproduced.plan.shortfall).all(), "the hull is at least as deep as its axis end"


def test_the_check_removed_the_same_run_passes_silently_while_the_body_is_through_the_wall(reproduced, monkeypatch):
    monkeypatch.setattr("mime.experiment._experiment.read_containments", lambda *a, **k: ())
    built = case(inertia=True)
    removed = built.run(s(STEPS * DT), history=True)
    assert removed.containment == ()
    raised = [read.path for read in removed.flags if read.was_raised]
    assert all(path.startswith("drag:") for path in raised), (
        f"no flag of the contact or the body rises; only the drag's, which read the pose's tilt against the surrogate's "
        f"validity (measured: tilt_fired at 0.235 s, bore_fired at 0.302 s), say nothing of the wall: {raised}")
    assert (axis_end_excess(poses_of(removed)[1:]).max()) > 1e-3, "and the body is through the wall"


@pytest.mark.parametrize("mode", ["plain", "record"])
def test_a_run_that_kept_less_is_flagged_all_the_same_and_says_what_it_read(mode):
    built = case(inertia=True)
    result = built.run(s(STEPS * DT)) if mode == "plain" else built.run(
        s(STEPS * DT), record=["swim_speed:umr.speed"])
    (read,) = result.containment
    assert read.raised, read.text()
    if mode == "plain":
        assert not read.peak and read.read == 1 and "last step only" in read.text()
    else:
        assert read.peak and read.read == STEPS, "record= keeps the pose for the watch"
        assert read.first_step is not None and read.first_step * DT < 0.35


def test_the_watch_survives_the_manifest_and_the_text_and_the_summary(reproduced, tmp_path):
    from mime.results import ContainmentRead

    (read,) = reproduced.result.containment
    again = ContainmentRead.from_manifest(read.manifest())
    assert again == read
    assert read.text() in reproduced.result.text()
    assert reproduced.result.manifest()["containment"][0]["raised"] is True
    names = pytest.importorskip("scipy.io").loadmat  # the .mat file holds the struct
    reproduced.result.to_mat(tmp_path / "run.mat")
    held = names(tmp_path / "run.mat", simplify_cells=True)
    assert held["containment"][next(iter(held["containment"]))]["raised"] == "true"


# --- not flagged ---------------------------------------------------------------------------------------------------------


def test_script_a_as_it_is_is_not_flagged_over_its_full_run(isotropic):
    (read,) = isotropic.result.containment
    assert read.status == WITHIN and not read.raised, read.text()
    assert read.read == 10_000 and read.peak
    assert read.depth is not None and read.depth < -5e-5, "the hull keeps well inside the wall, by the contact's margin"
    assert axis_end_excess(isotropic.poses[1:]).max() < 0


def test_the_record_arm_is_not_flagged():
    result = script_a.record_arm(frequency=Hz(10.0), dt=ms(DT * 1e3)).build().run(s(1.0), history=True)
    (read,) = result.containment
    assert read.status == WITHIN, read.text()


def test_a_body_held_by_its_hull_gets_no_containment_plan():
    """The hull contact on FL-9's own inertia holds the pose: nothing is watched (the hull is read by the contact). Only a
    build, so it is cheap; the run is the next test."""
    built = case(inertia=True, contact=Contact)
    assert not [watch for watch in built.report.watches if isinstance(watch, (ContainmentPlan, UnwatchedContainment))]


def test_a_body_held_by_its_hull_has_its_pose_inside_by_the_centre_arms_hull(reproduced):
    """The hull contact on FL-9's own inertia holds the pose, and the centre arm's hull, read at the hull arm's poses, is
    inside the wall: the contact keeps the pose the centre-held one lets through."""
    built = case(inertia=True, contact=Contact)
    result = built.run(s(HULL_STEPS * DT), history=True)
    assert result.containment == ()
    held = poses_of(result)
    tilt = np.degrees(np.arccos(np.abs(1 - 2 * (held[:, 5] ** 2 + held[:, 6] ** 2))))
    assert tilt.max() < 1.0, "the axis stays along the tube's, as script A's isotropic inertia holds it (0.106 degrees at most)"
    assert depths(reproduced.plan, held).max() < 0.0
    assert axis_end_excess(held).max() < 0


# --- the reading against an independent computation ---------------------------------------------------------------------

FINE_SPACING = 5e-5
FINE = hull_points(spacing=FINE_SPACING)
#: The least a bound on a depth can give away to rounding and to the estimate of the shortfall: the shortfall is an
#: estimate over finitely many directions, which "may miss the largest by a little" (``samples_shortfall``); measured at
#: 23.2 um against the plan's 23.05 um over 20,000 random directions, 1 %: the 5 % here is that, with room.
SHORTFALL_SLACK = 1.05


def oracle_depth(position, quaternion) -> float:
    """The hull's samples at a spacing a fourth of the watch's, radial distance from the tube's axis less its radius."""
    placed = np.asarray(position) + FINE @ rotation(quaternion).T
    return float(np.max(np.hypot(placed[:, 1], placed[:, 2])) - TUBE_RADIUS)


def vertex_depths(plan: ContainmentPlan, poses: np.ndarray) -> np.ndarray:
    """The largest radial distance from the tube's axis less its radius, over the plan's own points, at each pose: the
    watch's second computation, in numpy, by the tube's closed form rather than the region's shape (a pose is
    ``position, quaternion``)."""
    found = np.empty(len(poses))
    for start in range(0, len(poses), 50):
        chunk = poses[start:start + 50]
        turn = np.stack([rotation(q) for q in chunk[:, 3:]])
        placed = chunk[:, None, :3] + np.einsum("nij,kj->nki", turn, plan.vertices)
        found[start:start + 50] = np.hypot(placed[..., 1], placed[..., 2]).max(axis=1) - TUBE_RADIUS
    return found


@st.composite
def poses(draw) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """A pose of FL-9 in the tube: its centre within a millimetre of the axis (the extent keeps it there), its axis turned
    across by up to a right angle about z and rolled about itself."""
    across = draw(st.floats(0.0, math.pi / 2))
    roll = draw(st.floats(0.0, math.pi))
    offset = draw(st.floats(0.0, 1e-3))
    angle = draw(st.floats(0.0, 2 * math.pi))
    half_across, half_roll = across / 2, roll / 2
    turn = np.array([math.cos(half_across), 0.0, 0.0, math.sin(half_across)])
    spin = np.array([math.cos(half_roll), math.sin(half_roll), 0.0, 0.0])
    w1, x1, y1, z1 = turn
    w2, x2, y2, z2 = spin
    q = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    return (0.0, offset * math.cos(angle), offset * math.sin(angle)), q


@pytest.fixture(scope="module")
def plan(reproduced) -> ContainmentPlan:
    """The reproduced case's plan, its evaluation already traced (the first reading of a block of one pose compiles for
    seconds; the properties below are timed on their own work)."""
    depths(reproduced.plan, reproduced.poses[:1])
    return reproduced.plan


@given(poses())
def test_the_depth_agrees_with_the_independent_computation_to_the_samples_shortfall(plan, drawn):
    """The samples lie in the body, so the depth read is the body's less up to the shortfall; the oracle's samples (a
    fourth of the spacing, so a sixteenth of the shortfall) are the body's less up to their own. A radius wrong by 1 %
    (32 um, more than this bound) fails every draw."""
    position, quaternion = drawn
    found = depths(plan, np.array([[*position, *quaternion]]))[0]
    oracle_shortfall = plan.shortfall * (FINE_SPACING / plan.spacing) ** 2
    difference = found - oracle_depth(position, quaternion)
    assert -SHORTFALL_SLACK * plan.shortfall - 1e-9 <= difference <= oracle_shortfall + 1e-9, difference


@given(poses())
def test_the_depth_is_the_distance_of_the_plans_points_to_rounding(plan, drawn):
    """A second computation of the same distance over the same points, in numpy, by the tube's closed form: it must agree
    to rounding, so a radius wrong by 1 % (32 um) or poses held in half precision (some um) fail every draw, which the
    bound on the body above, wide as the shortfall is, cannot show."""
    position, quaternion = drawn
    pose = np.array([[*position, *quaternion]])
    read = read_containment(plan, {"umr": {"pose": pose[0]}}, None, last_step=1, dt=DT)
    assert read.depth is not None and abs(read.depth - vertex_depths(plan, pose)[0]) < 1e-9


@given(poses())
def test_a_pose_planted_past_the_wall_is_flagged_on_every_draw_and_one_inside_it_on_none(plan, drawn):
    position, quaternion = drawn
    truth = oracle_depth(position, quaternion)
    assume(abs(truth) > 3.0 * plan.shortfall)  # the plant sits above the samples' resolution, either side
    pose = np.array([[*position, *quaternion]])
    read = read_containment(plan, {"umr": {"pose": pose[0]}}, None, last_step=1, dt=DT)
    assert read.raised == (truth > 0.0)


def test_a_surface_that_takes_one_point_at_a_time_is_not_computed_and_says_so(plan):
    class One:
        takes_many = False

    broken = SimpleNamespace(**{name: getattr(plan, name) for name in ("owner", "entry", "body", "surface", "node",
                                                                        "field", "margin")}, shape="nowhere",
                             vertices=plan.vertices, parameters=plan.parameters, shortfall=0.0)
    read = read_containment(broken, {"umr": {"pose": np.array([0.0, 0, 0, 1, 0, 0, 0])}}, None, last_step=1,  # type: ignore[arg-type]
                            dt=DT)
    assert read.status == UNKNOWN and "signed distance could not be read" in read.text()


def test_a_pose_that_is_not_finite_is_not_computed_never_within(plan):
    read = read_containment(plan, {"umr": {"pose": np.full(7, np.nan)}}, None, last_step=1, dt=DT)
    assert read.status == UNKNOWN and "not finite" in read.text()


def test_the_watch_adds_no_node_and_the_graph_is_what_it_was(reproduced):
    nodes = set(reproduced.built.graph.node_names)
    assert not [name for name in nodes if "contain" in name]


# --- the mutants ---------------------------------------------------------------------------------------------------------


def mutants() -> Iterator[tuple[str, object]]:
    yield "the depth's sign flipped", lambda real: lambda plan, held: -real(plan, held)
    yield "the centre read, not the hull", None
    yield "the check run at step 0 only", lambda real: lambda plan, held: np.where(
        np.arange(len(held)) == 0, real(plan, held), -np.inf)


def test_each_mutant_is_caught_by_the_reproduced_case_or_the_case_not_flagged(reproduced, isotropic, monkeypatch, capsys):
    real = _containment.depths

    def verdict() -> tuple[bool, bool]:
        return read_of(reproduced).raised, read_of(isotropic).raised

    flagged, quiet = verdict()
    assert flagged and not quiet, "the unmutated check: the reproduced case flagged, script A not"
    table = []
    for name, make in mutants():
        with monkeypatch.context() as patch:
            if make is None:
                patch.setattr(ContainmentPlan, "vertices", property(lambda self: np.zeros((1, 3))))
            else:
                patch.setattr(_containment, "depths", make(real))  # type: ignore[operator]
            flagged, raised = verdict()
        caught = not flagged or raised
        table.append((name, "caught" if caught else "MISSED", f"reproduced flagged: {flagged}, script A flagged: {raised}"))
    with capsys.disabled():
        print("\nmutants of the containment watch:")
        for row in table:
            print("  " + " | ".join(row))
        print(f"  caught {sum(r[1] == 'caught' for r in table)} of {len(table)}")
    assert all(row[1] == "caught" for row in table), table


def test_read_containments_reads_only_its_own_plans(reproduced):
    both = (*reproduced.built.report.watches,)
    assert len(read_containments(both, reproduced.result.final, None, last_step=STEPS, dt=DT)) == 1


def test_the_watch_is_read_when_first_asked_for_and_once(monkeypatch):
    """A run that never looks pays nothing for the watch: a script that runs many arms (the differential tests) is not
    slowed, and the read is the same when it comes."""
    calls = []
    real = _containment.depths
    monkeypatch.setattr(_containment, "depths", lambda plan, held: calls.append(len(held)) or real(plan, held))
    built = case(inertia=False)
    result = built.run(s(0.1), history=True)
    assert calls == [], "nothing was read by running"
    first = result.containment
    assert calls == [200] and result.containment is first, "read once, at the first ask"
    assert result.text() and calls == [200]


# --- the step the watch names --------------------------------------------------------------------------------------------


def test_the_first_step_past_the_wall_is_the_step_of_the_kept_pose(reproduced):
    """The step the text names, against a per-step series of the plan's own points computed in numpy (``poses[i]`` is the
    state after step ``i + 1``), and, as a bound on the body that does not use the sampling, the screw's axis end: it
    was not past the wall the step before by more than the shortfall."""
    (read,) = reproduced.result.containment
    series = vertex_depths(reproduced.plan, reproduced.poses)
    past = np.flatnonzero(series > read.tolerance)
    assert read.first_step == 1 + int(past[0]) and read.worst_step == 1 + int(np.argmax(series))
    assert f"first past it at step {read.first_step:,}" in read.text() and f"at step {read.worst_step:,}" in read.text()
    before = read.first_step - 2
    assert axis_end_excess(reproduced.poses)[before] <= read.tolerance + SHORTFALL_SLACK * reproduced.plan.shortfall


def test_a_run_that_continues_another_names_its_steps_from_the_graphs_start():
    """The second run's steps count on from the first's, in the text and in the read: the control against a count from the
    run's own start, which gives the same numbers for a first run."""
    built = case(inertia=True)
    first = built.run(s(250 * DT), history=True)
    second = built.run(s(750 * DT), history=True)
    assert first.containment[0].status == WITHIN and second.start_step == 250
    (read,) = second.containment
    series = vertex_depths(plan_of(built), poses_of(second))
    expected = 250 + 1 + int(np.flatnonzero(series > read.tolerance)[0])
    assert read.first_step == expected and expected > 250, (read.first_step, expected)
    assert read.worst_step == 250 + 1 + int(np.argmax(series))
    assert f"first past it at step {expected:,}" in read.text()


# --- a run that kept nothing says what it did not see --------------------------------------------------------------------


def test_a_hull_that_crossed_the_wall_and_came_back_is_not_said_to_have_stayed_inside(plan, reproduced):
    """Out past the wall to step 700 and back to the start before the last step: read at every step it is past the wall;
    read at the last step alone it is inside and says that earlier steps were not kept, never that it stayed."""
    held = reproduced.poses[:699]
    excursion = np.concatenate([held, held[::-1]])
    last = len(excursion)
    every = read_containment(plan, {"umr": {"pose": excursion[-1]}}, {"umr": {"pose": excursion}}, last_step=last, dt=DT)
    only = read_containment(plan, {"umr": {"pose": excursion[-1]}}, None, last_step=last, dt=DT)
    assert every.status == PAST and every.depth is not None and every.depth > 1e-3
    assert only.status == WITHIN and not only.peak and only.read == 1
    assert "stayed" not in only.text() and "stayed" in read_containment(
        plan, {"umr": {"pose": held[0]}}, {"umr": {"pose": held[:300]}}, last_step=300, dt=DT).text()
    assert "is inside tube.inner_surface at the last step" in only.text()
    assert "earlier steps were not kept, so nothing is known of them" in only.text()


def test_a_run_that_kept_nothing_says_so_in_its_text_manifest_and_mat_struct_and_never_that_it_stayed(tmp_path):
    result = case(inertia=False).run(s(0.1))
    (read,) = result.containment
    words = "earlier steps were not kept, so nothing is known of them"
    assert read.status == WITHIN and not read.peak and "stayed" not in read.text() and words in read.text()
    assert words in result.text() and "stayed" not in result.text().split("post-run")[-1]
    assert words in result.manifest()["containment"][0]["text"]
    result.to_mat(tmp_path / "run.mat")
    held = pytest.importorskip("scipy.io").loadmat(tmp_path / "run.mat", simplify_cells=True)
    struct = held["containment"][next(iter(held["containment"]))]
    assert words in struct["text"] and struct["from"] == "the last step only" and "stayed" not in struct["text"]


def test_the_text_says_containment_once(reproduced):
    line = next(line for line in reproduced.result.text().splitlines() if "post-run" in line and "contain" in line)
    assert line.startswith("post-run containment:contact:umr") and line.count("containment") == 1, line


# --- the threshold, the spacing, the placement ---------------------------------------------------------------------------


def centre_for_depth(target: float) -> float:
    """The offset from the tube's axis (along y, the body unturned) at which the hull's samples at the fine spacing are
    ``target`` past the wall (negative inside it), by bisection."""
    low, high = 0.0, 2 * TUBE_RADIUS
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if oracle_depth((0.0, middle, 0.0), (1.0, 0.0, 0.0, 0.0)) < target else (low, middle)
    return low


@pytest.mark.parametrize("target", [-80e-6, -70e-6, -60e-6, 60e-6, 70e-6, 80e-6])
def test_a_hull_past_the_wall_by_less_than_the_margin_is_flagged_and_one_inside_it_by_as_much_is_not(plan, target):
    """The flag rises past the wall, not past the contact's margin (0.1 mm): a pose planted 60 to 80 um past it, more than
    the shortfall's 23 um and less than the margin, is flagged in float32 as the run holds it, and one planted that far
    inside it, within the margin's band, is not."""
    assert SHORTFALL_SLACK * plan.shortfall < abs(target) < plan.margin
    offset = centre_for_depth(target)
    pose = np.array([0.0, offset, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    read = read_containment(plan, {"umr": {"pose": pose}}, None, last_step=1, dt=DT)
    assert read.raised == (target > 0), read.text()


def test_the_spacing_is_a_twentieth_of_the_hulls_enclosing_radius_and_the_shortfall_is_what_that_spacing_leaves(plan):
    """The declared share (``WATCH_SPACING``, 0.05) of the radius that encloses the fine samples, and the shortfall as the
    largest, over 20,000 random directions, of the fine samples' reach less the plan's own samples': the watch's own
    function computed again here. The spacing doubled (a shortfall of about 82 um) fails both."""
    enclosing = float(np.linalg.norm(FINE, axis=1).max())
    assert plan.spacing == pytest.approx(0.05 * enclosing, rel=1e-3)
    directions = np.random.default_rng(0).normal(size=(20_000, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    coarse = hull_points(spacing=plan.spacing)
    estimate = float(((FINE @ directions.T).max(axis=0) - (coarse @ directions.T).max(axis=0)).max())
    assert plan.shortfall == pytest.approx(estimate, rel=0.15), (plan.shortfall, estimate)


ROD_RADIUS, ROD_HALF = 1e-3, 3e-3


def rod_in_the_tube(part, *, extent=None) -> Experiment:
    """A body with one part in script A's tube, held by the centre-held contact: a build, which is all a plan needs."""
    experiment = Experiment("rod", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=m(TUBE_RADIUS), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": part}))
    for path, value in (("umr.inertia", DIAGONAL), ("umr.mass", MASS), ("umr.buoyant_volume", VOLUME),
                        ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))), ("water.density", WATER)):
        experiment.declare(path, value, reason="script A's values")
    if extent is not None:
        experiment.declare("umr.extent", mm(extent), reason="the rod's radial reach")
    experiment.attach(script_a.CentreContact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    return experiment


def rod_depth(position, quaternion, centre, turn) -> float:
    """The farthest radial point of the rod (a cylinder along its own x, in a part placed at ``centre`` and turned by
    ``turn`` in the body) from the tube's axis, less the tube's radius, at the body's pose: a closed form (the farthest
    point of a convex body is on an end's rim), the rims sampled round at 7,200 angles, independent of the watch's
    samples."""
    angle = np.linspace(0.0, 2 * np.pi, 7200, endpoint=False)
    rim = np.stack([np.zeros_like(angle), ROD_RADIUS * np.cos(angle), ROD_RADIUS * np.sin(angle)], axis=1)
    ends = np.concatenate([rim + [ROD_HALF, 0.0, 0.0], rim - [ROD_HALF, 0.0, 0.0]])
    world = np.asarray(position) + (ends @ np.asarray(turn).T + np.asarray(centre)) @ rotation(quaternion).T
    return float(np.hypot(world[:, 1], world[:, 2]).max() - TUBE_RADIUS)


ROD_POSE = ((0.0, 0.3e-3, -0.2e-3), (1.0, 0.0, 0.0, 0.0))
TURNED_POSE = ((0.0, -0.4e-3, 0.3e-3), (math.cos(0.3), math.sin(0.3), 0.0, 0.0))
IDENTITY = np.eye(3)
ABOUT_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def depth_of(plan: ContainmentPlan, pose) -> float:
    position, quaternion = pose
    return float(depths(plan, np.array([[*position, *quaternion]]))[0])


def rod_plan(at) -> ContainmentPlan:
    built = rod_in_the_tube(Part(Cylinder(radius=m(ROD_RADIUS), length=m(2 * ROD_HALF), axis=(1.0, 0.0, 0.0)), at=at),
                            extent=1.0).build()
    (plan,) = [w for w in built.report.watches if isinstance(w, ContainmentPlan)]
    return plan


@pytest.mark.parametrize("pose", [ROD_POSE, TURNED_POSE])
def test_a_part_placed_off_the_bodys_origin_is_read_where_it_is(pose):
    """The rod 1.5 mm above the body's origin, against the closed form at the same pose; the control is the same closed
    form with the offset dropped, which differs by far more than the bound, so the test cannot pass without the placement."""
    offset = (0.0, 0.0, 1.5e-3)
    plan = rod_plan(mm((0.0, 0.0, 1.5)))
    assert plan.centre == pytest.approx(offset)
    bound = SHORTFALL_SLACK * plan.shortfall + 1e-9
    assert abs(depth_of(plan, pose) - rod_depth(*pose, offset, IDENTITY)) <= bound
    assert abs(rod_depth(*pose, offset, IDENTITY) - rod_depth(*pose, (0.0, 0.0, 0.0), IDENTITY)) > 10 * bound


@pytest.mark.parametrize("pose", [ROD_POSE, TURNED_POSE])
def test_a_part_turned_in_the_body_is_read_turned(pose):
    """A part's turn comes from an asset's transforms, which no Python-first part has: the plan with the rod turned a
    quarter about z (its axis along the body's y) and placed off the origin, against the closed form; the control drops
    the turn, and, separately, the offset, each by far more than the bound."""
    centre = (0.0, 0.2e-3, 0.5e-3)
    base = rod_plan(mm((0.0, 0.0, 0.0)))
    plan = dataclasses.replace(base, centre=centre, rotation=tuple(tuple(float(c) for c in row) for row in ABOUT_Z))
    bound = SHORTFALL_SLACK * plan.shortfall + 1e-9
    truth = rod_depth(*pose, centre, ABOUT_Z)
    assert abs(depth_of(plan, pose) - truth) <= bound
    assert abs(truth - rod_depth(*pose, centre, IDENTITY)) > 10 * bound, "the control: dropping the turn is seen"
    assert abs(truth - rod_depth(*pose, (0.0, 0.0, 0.0), ABOUT_Z)) > 10 * bound, "the control: dropping the offset is seen"


# --- a body the watch cannot check is said to be unwatched ----------------------------------------------------------------

register_namespace("uncontained", "0.0.1")


@register_geometry
@dataclasses.dataclass(frozen=True, kw_only=True)
class _Unsampled(Solid):
    """A ball that gives its volume, inertia and reach and not the protocol's samples: what a lab's kind that was written
    before the sampling is."""

    name = "uncontained.ball"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    builds = {"": ()}
    radius: Annotated[Quantity, Length]

    @override
    def volume(self, settings):
        return 4.0 / 3.0 * math.pi * settings["radius"] ** 3

    @override
    def inertia_per_mass(self, settings):
        moment = 2.0 * settings["radius"] ** 2 / 5.0
        return ((moment, 0.0, 0.0), (0.0, moment, 0.0), (0.0, 0.0, moment))

    @override
    def enclosing_radius(self, settings):
        return settings["radius"]

    @override
    def radial_reach(self, settings):
        return settings["radius"]


def unwatched_through_the_run(built, expected: str) -> None:
    """The three places a user reads it: the build report, the result's text, and the end-of-run summary."""
    (plan,) = [w for w in built.report.watches if isinstance(w, UnwatchedContainment)]
    assert expected in plan.reason
    assert any("NOT watched" in line and expected in line for line in built.report.watch_lines()), built.report.watch_lines()
    result = built.run(s(0.05))
    (read,) = result.containment
    assert read.status == UNKNOWN and expected in read.text() and "not computed" in read.text()
    assert read.text() in result.text() and not read.raised
    lines = _summary.attention_lines(built, SimpleNamespace(result=result, stopped=None), None, Path("exp"))
    assert any(line.startswith("post-run containment could not be checked") and expected in line for line in lines), lines


def withdraw_takes_many(monkeypatch):
    """The cylinder's ``takes_many`` withdrawn, on its class: the declaration is the class's, and a patch on the
    registered instance puts the old value back as an instance attribute that then shadows every later patch of the
    class (the hull contact's refusal test, in the same worker, saw its withdrawal ignored)."""
    monkeypatch.setattr(type(CYLINDER), "takes_many", False)


def test_a_surface_whose_shape_takes_one_point_at_a_time_is_unwatched_and_says_so(monkeypatch):
    """Through the planner: the tube's cylinder withdrawn from ``takes_many`` (a lab's shape need not take many). The
    control is the same case with it, which is watched."""
    assert [w for w in case(inertia=True).report.watches if isinstance(w, ContainmentPlan)]
    withdraw_takes_many(monkeypatch)
    built = case(inertia=True)
    assert not [w for w in built.report.watches if isinstance(w, ContainmentPlan)]
    unwatched_through_the_run(built, "takes one point at a time")


def test_a_withdrawal_of_takes_many_leaves_the_cylinder_as_it_was_for_a_later_withdrawal():
    """The withdrawal above, undone, and then a later test's withdrawal of the same declaration on the class (the hull
    contact's, which expects ``confinement.surface-shape``): it takes effect. The control is the cylinder between them,
    which takes many and carries nothing of its own."""
    with pytest.MonkeyPatch.context() as earlier:
        withdraw_takes_many(earlier)
    assert CYLINDER.takes_many and "takes_many" not in vars(CYLINDER)
    with pytest.MonkeyPatch.context() as later:
        later.setattr(type(CYLINDER), "takes_many", False)
        built = case(inertia=True)
        assert not [w for w in built.report.watches if isinstance(w, ContainmentPlan)]


def test_a_hull_kind_that_gives_no_samples_is_unwatched_and_says_so():
    """Through the planner: a lab's kind with a reach and no samples; the control is a sphere in the same rig, watched."""
    (watched,) = [w for w in rod_in_the_tube(Part(Sphere(radius=mm(1.0)))).build().report.watches
                  if isinstance(w, ContainmentPlan)]
    assert watched.body == "umr"
    unwatched_through_the_run(rod_in_the_tube(Part(_Unsampled(radius=mm(1.0)))).build(), "no sample points")
