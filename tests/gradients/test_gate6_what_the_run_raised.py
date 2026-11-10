"""Gate 6: a study's derivative carries what its run raised, member by member: a flag (the tube's bypass and Stokes
lines; the MRS drag's guard), a node's stop (the MRS drag's indefinite resistance, read from the differentiated run's
final state) and a table read extrapolated by the experiment's declaration (the water's density read past its table's
range, from the member's own plan); and the guard flag form: where the MRS drag stops a forward run by writing NaN, the
differentiated run writes the resistance the body took at the step before, raises the same flag at the same step and
records the same stop, so its derivative is finite and names it.

**Controls** (``gradient_plants.py``): the result's assembly dropping one item (``MIME_PLANT_GRADIENT_DROP``: ``flag``,
``refusal``, ``extrapolation``) fails this gate on every run; the guard writing NaN in a differentiated run too
(``MIME_PLANT_GUARD=nan``) gives a NaN derivative."""

from __future__ import annotations

import pytest
from gradient_rigs import RADIUS, STEPS, TUBE_STEPS, flux, gentle, pose_loss, tube
from gradient_plants import DROP, GUARD, drop_plant, from_environment, guard_plant, stopping_every_step
from records import density_table

from mime.build import RunFailed
from mime.materials import Extrapolation, Record, record_file
from mime.study import Study, along
from mime.units import K, Condition, m_per_s2, mm, ms

ACKNOWLEDGED = {"water.flow": "the gate reads what the run raised, through the tube's viscosity"}


def stopped_sphere(mode: str, patch: pytest.MonkeyPatch):
    """The gently pulled sphere with its guard made to stop at every step, differentiated in its radius."""
    drop_plant(patch, from_environment(DROP))
    guard_plant(patch, from_environment(GUARD))
    with stopping_every_step():
        return Study(gentle(1.05)).sweep(along({RADIUS: [mm(1.05), mm(1.1)]}), duration=ms(STEPS)).gradient(
            pose_loss, mode=mode)


@pytest.fixture(scope="module", params=["reverse", "forward"])
def sphere(request):
    with pytest.MonkeyPatch.context() as patch:
        return stopped_sphere(request.param, patch)


def test_the_planted_stop_stops_the_forward_run_with_its_code():
    """The case raises what the gate looks for: the forward run of the same experiment is refused on the drag's own
    code, with NaN in the resistance it never gave the body."""
    with stopping_every_step(), pytest.raises(RunFailed) as failed:
        gentle(1.05, name="gentle_forward").build().run(ms(STEPS))
    assert "drag.mrs-indefinite" in failed.value.codes()


def sphere_problems(result) -> list[str]:
    found = []
    for index in (0, 1):
        member = result.member(index)
        flags = {flag.path: flag for flag in member.flags}
        held = flags.get("drag:ball.indefinite")
        if held is None or (held.count, held.first, held.last) != (STEPS, 1, STEPS):
            found.append(f"member {index}: the guard's flag is {held}, not raised at every step from the first")
        if [refusal.code for refusal in member.refusals] != ["drag.mrs-indefinite"]:
            found.append(f"member {index}: the stop is {[r.code for r in member.refusals]}")
        if not member.finite:
            found.append(f"member {index}: the derivative is not finite: {member.value}, {dict(member.gradient)}")
    return found


def test_a_stopped_members_flag_and_stop_reach_its_gradient_result_and_its_derivative_is_finite(sphere):
    assert sphere_problems(sphere) == []


@pytest.mark.parametrize("plant", ["flag", "refusal"])
def test_control_a_result_that_drops_the_flag_or_the_stop_fails_the_gate(plant):
    with pytest.MonkeyPatch.context() as patch:
        drop_plant(patch, plant)
        result = stopped_sphere("reverse", patch)
    assert sphere_problems(result)


def test_control_the_guard_writing_nan_when_differentiated_gives_a_nan_derivative():
    with pytest.MonkeyPatch.context() as patch:
        guard_plant(patch, "nan")
        result = stopped_sphere("reverse", patch)
    assert not result.member(0).finite


def hot_tube(folder):
    """The tube with its water's density read from a table past its range (350 K against a range to 340 K), extrapolated
    by the experiment's declaration (its end value)."""
    exp = tube(name="hot_tube", material=Record("table", record_file(density_table(folder)), extrapolation={
        "bulk.density": Extrapolation("constant", basis="the end value", reason="the gate's extrapolated read")}),
        conditions={"temperature": Condition(K(350.0), K(0.5))})
    return exp


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    folder = tmp_path_factory.mktemp("table")
    with pytest.MonkeyPatch.context() as patch:
        drop_plant(patch, from_environment(DROP))
        base = Study(hot_tube(folder)).sweep(along({"body_force:water.steady": [m_per_s2(0.01), m_per_s2(1.0)]}),
                                             duration=ms(20.0 * TUBE_STEPS))
        return base.gradient(flux, wrt=["water.viscosity"], mode="reverse", acknowledged=ACKNOWLEDGED)


def flow_problems(result) -> list[str]:
    found = []
    calm, strong = result.member(0), result.member(1)
    paths = [{flag.path for flag in member.flags} for member in (calm, strong)]
    if paths[0] != {"water.flow.bypass_flagged"}:
        found.append(f"the calm member's flags are {sorted(paths[0])}, not the bypass alone")
    if paths[1] != {"water.flow.bypass_flagged", "water.flow.reynolds_flagged"}:
        found.append(f"the strongly driven member's flags are {sorted(paths[1])}, not the bypass and the Stokes line")
    for member in (calm, strong):
        if len(member.extrapolations) != 1 or "density" not in member.extrapolations[0] or \
                "basis: the end value" not in member.extrapolations[0]:
            found.append(f"member {member.index}'s extrapolations are {member.extrapolations}")
    return found


def test_a_members_flags_and_its_extrapolated_read_reach_its_gradient_result(flow):
    assert flow_problems(flow) == []


@pytest.mark.parametrize("plant", ["flag", "extrapolation"])
def test_control_a_result_that_drops_a_flag_or_the_extrapolation_fails_the_gate(plant, tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        drop_plant(patch, plant)
        result = Study(hot_tube(tmp_path)).sweep(
            along({"body_force:water.steady": [m_per_s2(0.01), m_per_s2(1.0)]}), duration=ms(20.0 * TUBE_STEPS)) \
            .gradient(flux, wrt=["water.viscosity"], mode="reverse", acknowledged=ACKNOWLEDGED)
    assert flow_problems(result)


def test_the_flags_are_the_forward_runs_own(flow, tmp_path):
    """Each flag's count, first and last step in the gradient result are the forward run's of the same member (its
    run kept no step's state, so its tally is the scan's own reduction of the flags)."""
    forward = Study(hot_tube(tmp_path)).sweep(along({"body_force:water.steady": [m_per_s2(0.01), m_per_s2(1.0)]}),
                                              duration=ms(20.0 * TUBE_STEPS)).run()
    dt = 0.02
    for index in (0, 1):
        raised = {flag.path: (flag.count, flag.first, flag.last) for flag in flow.member(index).flags}
        reads = {read.path: (read.raised, round(read.first / dt), round(read.last / dt))
                 for read in forward.member(index).flags if read.raised}
        assert raised and raised == reads
