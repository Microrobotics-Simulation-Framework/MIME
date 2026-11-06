"""The containment watch (lane U's U9): a body the centre-held contact (0.1.0) holds, whose hull leaves the region it is
held in, is flagged after the run.

**What these bound.** That the watch reads the surface's signed distance at the hull's points, from the body's pose, as
stated; that it agrees with an independent computation of the same distance (``hull_cases``: the samples' radial
distance from the tube's axis, at a spacing an order finer) to the samples' shortfall; that the case F0's probe found
(script A with FL-9's own diagonal inertia, whose axis turns across the tube within 0.35 s under that contact) is
flagged on every run, with the check removed it passes silently, and script A as it is (its isotropic effective inertia)
and the record arm are never flagged. They do not bound the physics of the body going through the wall (nothing here
says what the real body would do), and the watch reads the hull's samples, which fall short of the body by up to the
plan's ``shortfall``.

**Controls.** The reproduced case is deterministic, so it is flagged on every run by construction; the check removed
(``read_containments`` replaced by nothing) leaves the same run with no flag of the contact's (the surrogate drag's own
validity flags, ``tilt_fired`` and ``bore_fired``, do rise, later, and say nothing of the wall) while an independent first-principles bound (the
screw's axis end, a point of its body, past the tube's wall) shows the body through it. The mutants are run against the
same recorded poses: the depth's sign flipped, the hull replaced by the centre, and the check run at step 0 only; the
table is printed (``-s``) and each must be caught by the reproduced case or by the case that is not flagged."""

import math
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
from hull_cases import DIAGONAL, TUBE_RADIUS, hull_points, rotation
from hypothesis import assume, given
from hypothesis import strategies as st
from script_a import reproduction_arm

import script_a
from mime.build._watch import ContainmentPlan
from mime.interactions import Contact
from mime.results import _containment
from mime.results._containment import PAST, UNKNOWN, WITHIN, depths, read_containment, read_containments
from mime.units import Hz, ms, s

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


def test_a_body_held_by_its_hull_has_no_watch_and_its_pose_is_inside_by_the_centre_arms_hull(reproduced):
    """The hull contact on FL-9's own inertia holds the pose: nothing is watched (the hull is read by the contact), and the
    centre arm's hull, read at the hull arm's poses, is inside the wall."""
    built = case(inertia=True, contact=Contact)
    assert not [watch for watch in built.report.watches if isinstance(watch, ContainmentPlan)]
    result = built.run(s(HULL_STEPS * DT), history=True)
    assert result.containment == ()
    held = poses_of(result)[1:]
    tilt = np.degrees(np.arccos(np.abs(1 - 2 * (held[:, 5] ** 2 + held[:, 6] ** 2))))
    assert tilt.max() < 1.0, "the axis stays along the tube's, as script A's isotropic inertia holds it (0.05 degrees)"
    assert depths(reproduced.plan, held).max() < 0.0
    assert axis_end_excess(held).max() < 0


# --- the reading against an independent computation ---------------------------------------------------------------------

FINE = hull_points(spacing=5e-5)


def oracle_depth(position, quaternion) -> float:
    """The hull's samples at a spacing a twentieth of the watch's, radial distance from the tube's axis less its radius."""
    placed = np.asarray(position) + FINE @ rotation(quaternion).T
    return float(np.max(np.hypot(placed[:, 1], placed[:, 2])) - TUBE_RADIUS)


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
    return reproduced.plan


@given(poses())
def test_the_depth_agrees_with_the_independent_computation_to_the_samples_shortfall(plan, drawn):
    position, quaternion = drawn
    found = depths(plan, np.array([[*position, *quaternion]]))[0]
    assert abs(found - oracle_depth(position, quaternion)) <= 2.0 * plan.shortfall + 1e-12


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
