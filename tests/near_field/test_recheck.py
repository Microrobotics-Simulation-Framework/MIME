"""The near field's GMRES under a gradient study's recheck (design #337): the joint node declares ``rechecks``, and a
member whose solve stopped short is re-solved with GMRES's cap lifted and its stall restarts raised, as the FVM's CG
is, the recheck naming the stop's cause from its own solves.

The case: two spheres a radius apart, one pulled towards the other, on the declared GMRES route, the derivative of their
relative velocity after three steps in the water's viscosity, reverse mode (the velocity, not the separation, whose
millimetres would hide a solve's error in its last figures). The plants, each through what the production solve
reads:

- **A one-iteration cap** (the node's ``Stopping`` built with ``cap=1``; a clean solve takes 7 to 20 iterations): every
  primal solve stops at its cap and is warned of; the recheck lifts the cap, converges, reaches the clean run's answer
  and reports the change, the cause ``cap-limited``. The recheck's stopping left as it was (mutant R1) leaves it at the
  cap, unconverged.
- **A floor** (float32 at a tolerance of 1e-9, under float32's floor, which a pair's residual reaches at about 1e-6):
  every solve stagnates; the recheck, with the cap lifted, the stall restarts raised and a stagnation stopping only at
  the rounding floor, stagnates again there and is reported ``floor-limited`` with float32's route out (float64 or the
  dense route), never converged. The stoppings the node's GMRES is given there are the run's and, for the recheck, the
  run's with the cap lifted, the stall restarts raised *and* the floor stop (mutant R2, the cap alone, fails it).
- **A stall, not a floor** (float64, three spheres 0.2 mm apart, GMRES's Krylov length planted to 1 through the node's
  ``Stopping``, the review's case): every solve stagnates far above its floor; the recheck runs past the stall,
  converges to the clean run's answer and names it ``stagnated``, never ``floor-limited`` (the old rule, a stagnation
  named a floor, fails it).
- **A cap on the transpose solves alone that survives the recheck** (the seam the transposed preconditioner marks): the
  primal solves converge, the transpose solves stop at their cap in the run and in the recheck, which therefore does
  not report itself converged (mutant R4, converged judged on the primal solves alone, fails it).

Float64 but the floor's arm (``tests/near_field`` runs in the process's precision; the arms set it)."""

from __future__ import annotations

import functools

import numpy as np
import pytest
from near_field_cases import NODE, bodies, mrs, precision, spheres
from rigid_oracles import ConstantWrench

from mime.interactions import NearField
from mime.nodes.flow.fvm._solve import DEFAULT_CAP, RECHECK_CAP_FACTOR, STALL_RESTARTS
from mime.nodes.hydro.near_field import JointDragNode, _contract, _gmres, _node
from mime.nodes.hydro.near_field._gmres import CAP, RESTART, Stopping
from mime.study import Study, along
from mime.study._gradient import CAP_LIMITED, FLOOR_LIMITED, STAGNATED
from mime.units import mPa_s, ms

STEPS = 3
PATH = "water.viscosity"
#: A recheck that reached the clean run's answer is within the solve's tolerance noise of it (1e-10 in float64); a
#: capped member's is far from it.
SAME, CHANGED = 1e-6, 1e-4


def pulled(name: str, gap: float = 1e-3, count: int = 2, **near: object):
    """``count`` spheres ``gap`` apart (a radius by default) on the GMRES route (``NearField(solve="gmres", **near)``),
    one pulled."""
    experiment = bodies(name, spheres(gap, count), mrs(0.4e-3, near_field=NearField(solve="gmres", **near)),
                        velocity=(1e-4, 0.0, 0.0))
    experiment.attach(ConstantWrench(value=(-2e-7, 0.0, 0.0, 0.0, 0.0, 0.0)), body="s1")
    return experiment


def closing(final):
    """The bodies' relative velocity along their line, micrometres a second, after the run."""
    return 1e6 * (final["s1"]["twist"][0] - final["s0"]["twist"][0])


def derivative(experiment):
    return Study(experiment).sweep(along({PATH: [mPa_s(1.0)]}), duration=ms(float(STEPS))).gradient(
        closing, mode="reverse").member(0)


def capped(patch: pytest.MonkeyPatch) -> None:
    """Every GMRES of the joint node built with a cap of one iteration (the recheck lifts it from there)."""
    patch.setattr(_node, "Stopping", lambda tolerance, restart, cap: Stopping(tolerance, restart=restart, cap=1))


def transpose_capped(patch: pytest.MonkeyPatch) -> None:
    """The transpose solves alone at a cap of one iteration, whatever stopping they are given (the lifted one too)."""
    solve, real = _contract.reported_solve, _gmres.gmres

    def marked(matrix, b, tap, precondition, transposed, stopping):
        def planted(v):
            return transposed(v)

        planted.planted = True
        return solve(matrix, b, tap, precondition, planted, stopping)

    def gmres(operator, b, precondition, stopping, magnitude=None):
        if getattr(precondition, "planted", False):
            stopping = Stopping(stopping.tolerance, restart=stopping.restart, cap=1)
        return real(operator, b, precondition, stopping, magnitude)

    patch.setattr(_contract, "reported_solve", marked)
    patch.setattr(_gmres, "gmres", gmres)


def short_krylov(patch: pytest.MonkeyPatch) -> None:
    """Every GMRES of the joint node built with a Krylov length of one (the production length is ``RESTART``)."""
    patch.setattr(_node, "Stopping", lambda tolerance, restart, cap: Stopping(tolerance, restart=1, cap=cap))


#: The stall's case: three spheres 0.2 mm apart, where a Krylov length of one stagnates far above the floor.
STALL = {"gap": 0.2e-3, "count": 3}


@functools.cache
def run(plant: str | None, case: tuple[tuple[str, float | int], ...] = ()):
    with pytest.MonkeyPatch.context() as patch, precision("float64"):
        if plant == "cap":
            capped(patch)
        elif plant == "transpose":
            transpose_capped(patch)
        elif plant == "krylov":
            short_krylov(patch)
        return derivative(pulled(f"recheck_{plant or 'clean'}{len(case)}", **dict(case)))


def change(found: float, held: float) -> float:
    return abs(found - held) / (abs(held) or 1.0)


def test_the_clean_runs_solves_converge_and_nothing_is_rechecked():
    member = run(None)
    (report,) = member.solves
    assert report.node == NODE and report.forward_short == report.backward_short == 0
    assert member.recheck is None and not member.warnings
    assert float(np.asarray(report.forward)[0]) > STEPS, "more than one iteration a step: the cap of one can bite"


def test_a_capped_solve_is_rechecked_with_its_cap_lifted_converges_to_the_clean_answer_and_is_cap_limited():
    member, clean = run("cap"), run(None)
    (report,) = member.solves
    assert report.forward_short >= STEPS and any("primal solves stopped short" in text for text in member.warnings)
    recheck = member.recheck
    assert recheck is not None and recheck.made and recheck.lifted == (NODE,) and recheck.not_rechecked == {}
    assert recheck.converged and recheck.causes == {NODE: CAP_LIMITED} and recheck.routes == {}
    assert change(recheck.value, clean.value) < SAME and change(recheck.gradient[PATH], clean.gradient[PATH]) < SAME
    assert change(member.value, clean.value) > CHANGED and recheck.value_change > CHANGED
    assert recheck.gradient_change[PATH] > CHANGED
    text = member.text()
    print("\n" + text)
    assert f"recheck of {NODE}: cap-limited: it converged" in text


def test_a_solve_at_float32s_floor_is_floor_limited_with_its_route_and_its_recheck_lifts_the_cap_and_the_restarts():
    given: list[tuple[int, int, bool]] = []
    solving = _node.grand_resistance_by_gmres

    def recording(joint, blocks, transposed, stopping, tap):
        given.append((stopping.cap, stopping.stall_restarts, stopping.floor_stop))
        return solving(joint, blocks, transposed, stopping, tap)

    with pytest.MonkeyPatch.context() as patch, precision("float32"):
        patch.setattr(_node, "grand_resistance_by_gmres", recording)
        member = derivative(pulled("recheck_floor", tolerance=1e-9))
    (report,) = member.solves
    assert report.forward_short >= STEPS and report.stagnated >= report.forward_short
    recheck = member.recheck
    assert recheck is not None and recheck.made and not recheck.converged
    assert recheck.causes == {NODE: FLOOR_LIMITED} and "run in float64" in recheck.routes[NODE]
    assert "solve='dense'" in recheck.routes[NODE]
    text = member.text()
    print("\n" + text)
    assert f"recheck of {NODE}: floor-limited" in text and "stopped short as well" in text
    assert set(given) == {(CAP, 0, False), (max(CAP * RECHECK_CAP_FACTOR, DEFAULT_CAP), STALL_RESTARTS, True)}, given


def test_a_stall_above_the_floor_is_named_stagnated_and_its_recheck_runs_past_it_to_the_clean_answer():
    """M1's control: GMRES's Krylov length planted to one (``RESTART`` in production) stagnates far above float64's
    floor. The recheck is never ``floor-limited``: it runs past the stall, converges to the clean run's answer, and
    names the stop ``stagnated`` with the dense route."""
    case = tuple(STALL.items())
    member, clean = run("krylov", case), run(None, case)
    assert RESTART > 1 and not clean.warnings
    (report,) = member.solves
    assert report.forward_short >= STEPS and report.stagnated >= report.forward_short
    recheck = member.recheck
    assert recheck is not None and recheck.made and recheck.causes == {NODE: STAGNATED}, recheck.causes
    assert recheck.converged and "solve='dense'" in recheck.routes[NODE]
    assert change(recheck.value, clean.value) < SAME and change(recheck.gradient[PATH], clean.gradient[PATH]) < SAME
    assert change(member.value, clean.value) > CHANGED
    text = member.text()
    print("\n" + text)
    assert f"recheck of {NODE}: stagnated: a stall above its rounding floor" in text


def test_the_routes_out_follow_the_runs_dtype():
    """A floor in float32 is left by float64; in float64 by a tolerance at or above it; a stall by the dense route. The
    FVM's CG does not test its floor in a recheck, so its stagnation is ``stagnated``, by its own dtype's route."""
    from mime.core._node import CAP_LIMITED as CAP_NAME
    from mime.nodes.flow.fvm._node import ImplicitFlowNode

    node = JointDragNode.__new__(JointDragNode)
    assert JointDragNode.floor_stops and not ImplicitFlowNode.floor_stops
    assert "run in float64" in node.recheck_route(FLOOR_LIMITED, "float32")
    held = node.recheck_route(FLOOR_LIMITED, "float64")
    assert "float64" not in held and "tolerance=..." in held and "solve='dense'" in held
    assert "solve='dense'" in node.recheck_route(STAGNATED, "float64") and node.recheck_route(CAP_NAME, "float64") == ""
    flow = ImplicitFlowNode.__new__(ImplicitFlowNode)
    assert "float64" in flow.recheck_route(STAGNATED, "float32") and "float64" not in flow.recheck_route(STAGNATED,
                                                                                                        "float64")
    assert flow.recheck_route(FLOOR_LIMITED, "float32") == ""


def test_a_recheck_whose_transpose_solves_still_stop_short_does_not_report_converged():
    member = run("transpose")
    (report,) = member.solves
    assert report.forward_short == 0 and report.backward_short >= 1, "only the transpose solves stopped short"
    recheck = member.recheck
    assert recheck is not None and recheck.made and recheck.lifted == (NODE,)
    (own,) = recheck.solves
    assert own.forward_short == 0 and own.backward_short >= 1, "the recheck's transpose solves are capped as well"
    assert not recheck.converged and recheck.causes == {NODE: CAP_LIMITED}
    assert f"recheck of {NODE}: cap-limited: it stopped at its cap again" in member.text()
