"""The near field's GMRES under a gradient study's recheck (design #337): the joint node declares ``rechecks``, and a
member whose solve stopped short is re-solved with GMRES's cap lifted and its stall restarts raised, as the FVM's CG
is, the recheck naming the stop's cause from its own solves.

The case: two spheres a radius apart, one pulled towards the other, on the declared GMRES route, the derivative of their
separation after three steps in the water's viscosity, reverse mode. The plants, each through what the production solve
reads:

- **A one-iteration cap** (the node's ``Stopping`` built with ``cap=1``; a clean solve takes 7 to 20 iterations): every
  primal solve stops at its cap and is warned of; the recheck lifts the cap, converges, reaches the clean run's answer
  and reports the change, the cause ``cap-limited``. The recheck's stopping left as it was (mutant R1) leaves it at the
  cap, unconverged.
- **A floor** (float32 at a tolerance of 1e-9, under float32's floor, which a pair's residual reaches at about 1e-6):
  every solve stagnates; the recheck, with the cap lifted and the stall restarts raised, stagnates again and is
  reported ``floor-limited`` with the route out (float64 or the dense route), never converged. The stoppings the node's
  GMRES is given there are the run's and, for the recheck, the run's with the cap lifted *and* the stall restarts
  raised (mutant R2, the cap alone, fails it).
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
from mime.nodes.hydro.near_field import _contract, _gmres, _node
from mime.nodes.hydro.near_field._gmres import CAP, Stopping
from mime.study import Study, along
from mime.study._gradient import CAP_LIMITED, FLOOR_LIMITED
from mime.units import mPa_s, ms

STEPS = 3
PATH = "water.viscosity"
#: A recheck that reached the clean run's answer is within the solve's tolerance noise of it (1e-10 in float64); a
#: capped member's is far from it.
SAME, CHANGED = 1e-6, 1e-4


def pulled(name: str, **near: object):
    """The two spheres a radius apart on the GMRES route (``NearField(solve="gmres", **near)``), one pulled."""
    experiment = bodies(name, spheres(1e-3, 2), mrs(0.4e-3, near_field=NearField(solve="gmres", **near)),
                        velocity=(1e-4, 0.0, 0.0))
    experiment.attach(ConstantWrench(value=(-2e-7, 0.0, 0.0, 0.0, 0.0, 0.0)), body="s1")
    return experiment


def separation(final):
    """The bodies' separation along their line in micrometres after the run."""
    return 1e6 * (final["s1"]["pose"][0] - final["s0"]["pose"][0])


def derivative(experiment):
    return Study(experiment).sweep(along({PATH: [mPa_s(1.0)]}), duration=ms(float(STEPS))).gradient(
        separation, mode="reverse").member(0)


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

    def gmres(operator, b, precondition, stopping):
        if getattr(precondition, "planted", False):
            stopping = Stopping(stopping.tolerance, restart=stopping.restart, cap=1)
        return real(operator, b, precondition, stopping)

    patch.setattr(_contract, "reported_solve", marked)
    patch.setattr(_gmres, "gmres", gmres)


@functools.cache
def run(plant: str | None):
    with pytest.MonkeyPatch.context() as patch, precision("float64"):
        if plant == "cap":
            capped(patch)
        elif plant == "transpose":
            transpose_capped(patch)
        return derivative(pulled(f"recheck_{plant or 'clean'}"))


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
    assert recheck.converged and recheck.causes == {NODE: CAP_LIMITED} and recheck.floor_routes == {}
    assert change(recheck.value, clean.value) < SAME and change(recheck.gradient[PATH], clean.gradient[PATH]) < SAME
    assert change(member.value, clean.value) > CHANGED and recheck.value_change > CHANGED
    assert recheck.gradient_change[PATH] > CHANGED
    text = member.text()
    print("\n" + text)
    assert f"recheck of {NODE}: cap-limited: it converged" in text


def test_a_solve_at_float32s_floor_is_floor_limited_with_its_route_and_its_recheck_lifts_the_cap_and_the_restarts():
    given: list[tuple[int, int]] = []
    solving = _node.grand_resistance_by_gmres

    def recording(joint, blocks, transposed, stopping, tap):
        given.append((stopping.cap, stopping.stall_restarts))
        return solving(joint, blocks, transposed, stopping, tap)

    with pytest.MonkeyPatch.context() as patch, precision("float32"):
        patch.setattr(_node, "grand_resistance_by_gmres", recording)
        member = derivative(pulled("recheck_floor", tolerance=1e-9))
    (report,) = member.solves
    assert report.forward_short >= STEPS and report.stagnated >= report.forward_short
    recheck = member.recheck
    assert recheck is not None and recheck.made and not recheck.converged
    assert recheck.causes == {NODE: FLOOR_LIMITED} and "float64" in recheck.floor_routes[NODE]
    assert "solve='dense'" in recheck.floor_routes[NODE]
    text = member.text()
    print("\n" + text)
    assert f"recheck of {NODE}: floor-limited" in text and "stopped short as well" in text
    assert set(given) == {(CAP, 0), (max(CAP * RECHECK_CAP_FACTOR, DEFAULT_CAP), STALL_RESTARTS)}, given


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
