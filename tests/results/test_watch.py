"""The post-run Reynolds-number watch (dev1 phase 7l-c, design #191 item 7): the number a body moved at, against the
number its rung is declared up to, and the size of the inertia the rung neglects where it is known.

**What these bound.** That the watch computes ``rho |v| D / mu`` (and the turning's) as stated, compares it with the
bound the entry declares, raises a flag on the result and never a refusal, and changes no graph. They do not bound the
bound: 0.1 is a declaration (Oseen's correction to a sphere), and no body but the sphere has a measured size of what is
neglected.

**Controls.** A run planted above the range is flagged on every draw and one below it on none: the planted runs sit a
factor of at least 1.05 (log-uniform to a thousand) either side of the bound, and the CI stress workflow runs this
file twenty times with fresh Hypothesis seeds (the ``stress`` label), so the rate is 20 of 20 each; the draws are the
physical scales of a body in a liquid, so the plant is above the rounding by many orders. The no-graph-change test is
planted too: an experiment that does differ hashes differently, and a watch that did add a node would show as one.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from rigs import WATER, falling_bead

from mime.build._plan import _Planner
from mime.build._watch import WatchPlan
from mime.interactions import Drag
from mime.registry._registry import _reynolds_problems
from mime.registry import ReynoldsRange
from mime.results import RunResult
from mime.results._watch import ABOVE, NO_RANGE, OSEEN, UNKNOWN, WITHIN, WatchRead, read_watch
from mime.units import s

LOG = dict(allow_nan=False, allow_infinity=False)


def plan_of(*, diameter: float, density: float, viscosity: float, upper: float | None, kind: str = "sphere",
            missing: tuple[str, ...] = ()) -> WatchPlan:
    return WatchPlan("drag:b", "drag 0.3.0", "mrs", "b", "m", "b", "twist", kind, diameter, density, viscosity, upper,
                     "the planted bound's basis" if upper is not None else None, missing)


def state(speed: float, rate: float = 0.0) -> dict[str, dict[str, np.ndarray]]:
    return {"b": {"twist": np.array([speed, 0.0, 0.0, rate, 0.0, 0.0], dtype=np.float64)}}


bodies = st.fixed_dictionaries({
    "diameter": st.floats(1e-5, 1e-2, **LOG), "density": st.floats(500.0, 2000.0, **LOG),
    "viscosity": st.floats(1e-3, 1.5, **LOG), "speed": st.floats(1e-6, 1.0, **LOG),
    "rate": st.floats(0.0, 1e3, **LOG)})


@given(bodies, st.floats(1.05, 1e3, **LOG), st.booleans())
def test_a_run_planted_above_the_range_is_flagged_on_every_draw_by_whichever_number_is_above(body, factor, by_turning):
    plan_args = {k: body[k] for k in ("diameter", "density", "viscosity")}
    scale = body["density"] * body["diameter"] / body["viscosity"]
    translation, turning = scale * body["speed"], scale * body["rate"] * body["diameter"] / 2.0
    held = max(turning, translation) if by_turning else translation
    read = read_watch(plan_of(upper=held / factor, **plan_args), state(body["speed"], body["rate"]), None)
    assert read.status == ABOVE and read.raised and read.reynolds == pytest.approx(max(translation, turning))


@given(bodies, st.floats(1.05, 1e3, **LOG))
def test_a_run_planted_below_the_range_is_never_flagged(body, factor):
    plan_args = {k: body[k] for k in ("diameter", "density", "viscosity")}
    scale = body["density"] * body["diameter"] / body["viscosity"]
    largest = max(scale * body["speed"], scale * body["rate"] * body["diameter"] / 2.0)
    read = read_watch(plan_of(upper=largest * factor, **plan_args), state(body["speed"], body["rate"]), None)
    assert read.status == WITHIN and not read.raised


@given(bodies)
def test_the_two_numbers_are_the_stated_arithmetic(body):
    read = read_watch(plan_of(diameter=body["diameter"], density=body["density"], viscosity=body["viscosity"],
                              upper=0.1), state(body["speed"], body["rate"]), None)
    assert read.translation == pytest.approx(body["density"] * body["speed"] * body["diameter"] / body["viscosity"])
    assert read.turning == pytest.approx(
        body["density"] * body["rate"] * (body["diameter"] / 2.0) * body["diameter"] / body["viscosity"])


def test_the_history_gives_the_peak_over_the_run_and_the_last_step_alone_gives_the_last():
    plan = plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=0.1)
    speeds = np.array([0.0, 2e-3, 5e-3, 1e-3])
    history = {"b": {"twist": np.stack([speeds, *[np.zeros(4)] * 5], axis=1)}}
    peak, last = read_watch(plan, state(1e-3), history), read_watch(plan, state(1e-3), None)
    assert peak.peak and peak.speed == 5e-3 and peak.translation == pytest.approx(0.5)
    assert not last.peak and last.speed == 1e-3 and "last step only" in last.text()


def test_a_sphere_s_neglected_inertia_is_oseens_first_correction_and_another_hull_has_none():
    sphere = read_watch(plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=0.1), state(0.037), None)
    assert sphere.translation == pytest.approx(3.7) and sphere.inertia == pytest.approx(OSEEN * 3.7)
    assert "3 Re / 16 = 69.4%" in sphere.text()
    screw = read_watch(plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=0.1, kind="screw"), state(0.037),
                       None)
    assert screw.inertia is None and "not estimated: no correction is known for a screw hull" in screw.text()


def test_a_turning_sphere_s_correction_is_said_to_be_of_second_order_and_not_estimated():
    read = read_watch(plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=0.1), state(0.0, 1e3), None)
    assert "second order" in read.text() and read.turning == pytest.approx(5.0)


def test_a_missing_input_is_said_and_never_passes_silently():
    read = read_watch(plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=0.1,
                              missing=("water.density is declared or read: exp.declare('water.density', ...)",)),
                      state(1.0), None)
    assert read.status == UNKNOWN and not read.raised and "not computed" in read.text() and "water.density" in read.text()
    assert read_watch(plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=0.1), {"other": {}}, None).status \
        == UNKNOWN


def test_an_entry_with_no_declared_range_gets_the_number_and_says_it_has_nothing_to_compare_with():
    read = read_watch(plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=None), state(1.0), None)
    assert read.status == NO_RANGE and not read.raised and "declares no range" in read.text()


def test_the_manifest_form_round_trips_a_watch():
    read = read_watch(plan_of(diameter=1e-4, density=1000.0, viscosity=1e-3, upper=0.1), state(0.037, 3.0), None)
    assert WatchRead.from_manifest(json.loads(json.dumps(read.manifest()))) == read


@pytest.mark.parametrize("held, problem", [
    ((ReynoldsRange(0.0, "a basis"),), "not a number above zero"),
    ((ReynoldsRange(0.1, " "),), "declares no basis"),
    ((ReynoldsRange(0.1, "a basis", rung="fast"),), "not one of its rungs"),
    ((ReynoldsRange(0.1, "a basis"), ReynoldsRange(0.2, "another")), "two bounds for every rung"),
    (("0.1",), "a tuple of mime.registry.ReynoldsRange"),
])
def test_a_malformed_reynolds_declaration_is_refused_at_registration_naming_what_is_wrong(held, problem):
    found = _reynolds_problems(SimpleNamespace(reynolds=held, rungs=("surrogate", "mrs")))  # type: ignore[arg-type]
    assert any(problem in line for line in found), found


def test_the_drag_declares_a_bound_with_its_basis_for_each_rung_and_the_declaration_is_well_formed():
    assert _reynolds_problems(Drag) == []
    assert sorted(r.rung for r in Drag.reynolds if r.rung) == ["mrs", "surrogate"]
    assert all(r.upper > 0.0 and "Oseen" in r.basis for r in Drag.reynolds)


# -- end to end, on the rig of a bead falling through water under the MRS drag -----------------------

DENSITY, VOLUME, VISCOSITY, DT = 2000.0, 1e-12, 1e-2, 1e-4


def bead_run(speed: float, *, history: bool = True) -> RunResult:
    """A bead of 62 um radius thrown at ``speed`` m/s downwards in a liquid of 10 mPa s, 10 steps (its relaxation time is
    0.17 ms, so a thrown bead slows within the run and an unthrown one settles at 0.8 mm/s): its Reynolds number
    at the speed is 12 times it in m/s."""
    experiment, _ = falling_bead(DT, 0.0, DENSITY, VOLUME, speed, viscosity=VISCOSITY)
    return experiment.build().run(s(10 * DT), history=history)


def test_a_bead_thrown_past_the_range_is_flagged_in_the_result_and_one_slower_is_not():
    fast, slow = bead_run(3.0), bead_run(1e-4)
    (high,), (low,) = fast.watches, slow.watches
    assert high.status == ABOVE and 5.0 < high.reynolds < 40.0 and high.peak
    assert low.status == WITHIN and low.reynolds < 0.05
    assert "ABOVE" in fast.text() and "ABOVE" not in slow.text()
    assert "3 Re / 16" in fast.text() and f"{100 * OSEEN * high.translation:.3g}%" in fast.text()


def test_a_run_that_kept_no_history_reads_the_last_step_and_says_so():
    (read,) = bead_run(3.0, history=False).watches
    assert read.status == ABOVE and not read.peak and "last step only" in read.text()


def test_the_watch_reaches_the_report_the_text_the_manifest_and_the_mat_file(tmp_path):
    from scipy.io import loadmat

    experiment, _ = falling_bead(DT, 0.0, DENSITY, VOLUME, 3.0, viscosity=VISCOSITY)
    built = experiment.build()
    section = built.report.text()
    assert "post-run watches (computed after each run" in section and "drag:bead" in section
    assert "post-run watches" in built.report.text(details=True)
    written = built.run_to(tmp_path / "run", s(10 * DT), history=True)
    manifest = json.loads((tmp_path / "run" / "outputs.json").read_text(encoding="utf-8"))
    (held,) = manifest["watches"]
    assert held["status"] == ABOVE and held["raised"] is True and held["neglected_inertia"] > 1.0
    (read,) = RunResult.read(tmp_path / "run").watches
    assert read == written.result.watches[0] and "ABOVE" in RunResult.read(tmp_path / "run").text()
    mat = loadmat(str(written.result.to_mat(tmp_path / "run.mat")), squeeze_me=True, struct_as_record=False)
    watch = mat["watches"].reynolds_drag_bead
    assert watch.status == ABOVE and watch.raised == "true" and watch.reynolds == pytest.approx(read.reynolds)


def test_a_watch_changes_no_graph_the_derived_hash_is_the_same_with_and_without_it(monkeypatch):
    experiment, _ = falling_bead(DT, 0.0, DENSITY, VOLUME, 3.0, viscosity=VISCOSITY)
    with_watch = experiment.build()
    assert with_watch.plans.watches, "the control is meaningful only where a watch is planned"
    monkeypatch.setattr(_Planner, "watch_plans", lambda self: ())
    without = experiment.build()
    assert not without.plans.watches
    assert with_watch.record.hashes() == without.record.hashes()
    assert with_watch.report.derived_hash == without.report.derived_hash
    assert with_watch.snapshot() == without.snapshot() and "reynolds" not in with_watch.snapshot()
    other, _ = falling_bead(DT, 0.0, DENSITY, 2e-12, 3.0, viscosity=VISCOSITY)
    assert other.build().report.derived_hash != with_watch.report.derived_hash, "the comparison can tell a difference"


def test_where_the_medium_s_density_is_declared_nowhere_the_build_goes_on_and_the_watch_says_why():
    from mime import Body, Experiment, Medium, State
    from mime.dynamics import RigidInertial
    from mime.interactions import Drag as MrsDrag
    from mime.scene import Scene, Sphere, Z
    from mime.things import Part
    from mime.units import kg_per_m3, m, m_per_s, m_per_s2, mPa_s, rad_per_s, um

    exp = Experiment("no_density", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(DT))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, -1.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(50.0)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.attach(MrsDrag.MRS(spacing=um(25.0), blob=0.5), body=ball, medium=water)
    built = exp.build()
    assert "no number can be computed until water.density is declared" in built.report.text()
    (read,) = built.run(s(2 * DT), history=True).watches
    assert read.status == UNKNOWN and "water.density" in read.text() and not read.raised


def test_the_density_an_entry_read_counts_as_declared_without_a_second_declaration():
    experiment, _ = falling_bead(DT, 0.0, DENSITY, VOLUME, 3.0, viscosity=VISCOSITY)
    (plan,) = experiment.build().plans.watches
    assert plan.density == WATER and plan.missing == () and plan.viscosity == pytest.approx(VISCOSITY)


def test_an_entry_without_a_declared_bound_is_watched_all_the_same(monkeypatch):
    monkeypatch.setattr(Drag, "reynolds", ())
    experiment, _ = falling_bead(DT, 0.0, DENSITY, VOLUME, 3.0, viscosity=VISCOSITY)
    built = experiment.build()
    assert "no Reynolds number declared for it" in built.report.text()
    (read,) = built.run(s(10 * DT), history=True).watches
    assert read.status == NO_RANGE and math.isfinite(read.reynolds or math.nan)
