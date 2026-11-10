"""Gate 7: the iterative solve's report in a study's derivative (the FVM's conjugate gradient, the design every
iterative solve in MIME writes): a solve stopped at its cap is warned of with its count, shows in the per-step report,
and is rechecked with the cause removed (its cap lifted, its stall restarts raised, at the member's own tolerance), the
change in the objective and the derivative reported with whether the recheck's own solves converged. In reverse mode the
solves are the primal's and the transpose solves'; in forward mode the primal's and the tangent solves'.

The case: the periodic tube's derivative in its viscosity, 20 steps. The plants (``gradient_plants.cap_plant``,
``MIME_PLANT_SOLVE_CAP`` for the gate kit's rate), both through what the production solve reads: the node's own ``cap``
setting at one iteration (the primal solves, and so the tangent and transpose solves that share their stopping), and
the seam by which the solve takes its transpose solves' stopping apart, a cap of one on them alone. A clean run takes 4
to 6 iterations a step at the default tolerance, so one iteration cannot reach it; the clean run's own counts are
asserted to show that.

**What the tests discriminate** (``MIME_PLANT_SOLVE_REPORT``, ``gradient_plants.report_plant``): forward mode's tangent
report zeroed, reverse mode's transpose report zeroed, the warnings dropped, and the recheck re-solving nothing each
fail a test of this file on every run; the recheck is judged against the clean run, whose answer it must reach once
the cap that stopped it is lifted."""

from __future__ import annotations

import functools

import numpy as np
import pytest
from gradient_plants import CAP, REPORT, cap_plant, from_environment, report_plant
from gradient_rigs import TUBE_STEPS, flux, tube

from mime.nodes.flow.fvm import REPORT_FIELDS
from mime.study import Study, along
from mime.units import mPa_s, ms

AT_CAP, ITERATIONS = REPORT_FIELDS.index("at_cap"), REPORT_FIELDS.index("iterations")
ACKNOWLEDGED = {"water.flow": "gate 7 reads the solve's reports through the tube's viscosity"}
PATH = "water.viscosity"
#: A recheck that reached the clean run's answer is within the solve's tolerance noise of it (the tolerance is 1e-8, the
#: forward and reverse derivatives agree to 4e-11 and the objectives to rounding); a capped member's is far from it.
SAME, CHANGED = 1e-6, 1e-4


def run(plant: str | None, mode: str):
    with pytest.MonkeyPatch.context() as patch:
        cap_plant(patch, plant)
        report_plant(patch, from_environment(REPORT))
        return Study(tube(report_rows=TUBE_STEPS, name=f"tube_{plant or 'clean'}_{mode}")).sweep(
            along({PATH: [mPa_s(1.0)]}), duration=ms(20.0 * TUBE_STEPS)).gradient(
            flux, mode=mode, acknowledged=ACKNOWLEDGED)


@pytest.fixture(scope="module", params=["reverse", "forward"])
def clean(request):
    return request.param, run(from_environment(CAP), request.param)


@functools.cache
def reference(mode: str):
    """The clean run in ``mode``, untouched by any plant (made by the first test that needs it)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(CAP, raising=False)
        patch.delenv(REPORT, raising=False)
        return Study(tube(report_rows=TUBE_STEPS, name=f"tube_reference_{mode}")).sweep(
            along({PATH: [mPa_s(1.0)]}), duration=ms(20.0 * TUBE_STEPS)).gradient(
            flux, mode=mode, acknowledged=ACKNOWLEDGED).member(0)


@pytest.fixture(scope="module", params=[("primal", "reverse"), ("primal", "forward"), ("transpose", "reverse"),
                                        ("transpose", "forward")], ids="-".join)
def capped(request):
    plant, mode = request.param
    return plant, mode, run(plant, mode)


def clean_problems(result, mode: str) -> list[str]:
    member = result.member(0)
    (report,) = member.solves
    found = []
    if member.warnings or member.recheck is not None:
        found.append(f"warned: {member.warnings}")
    if report.forward_short or report.backward_short:
        found.append(f"short: forward {report.forward_short}, backward {report.backward_short}")
    steps = np.asarray(report.steps)
    if steps.shape != (TUBE_STEPS, len(REPORT_FIELDS)) or not np.all(steps[:, ITERATIONS] > 1):
        found.append(f"the per-step primal report is {steps.shape}, its iterations {steps[:, ITERATIONS]}")
    backward = np.asarray(report.backward).reshape(-1, len(REPORT_FIELDS))
    if mode == "reverse" and (backward.shape != (TUBE_STEPS, len(REPORT_FIELDS)) or not np.all(backward[:, ITERATIONS] > 1)):
        found.append(f"the per-step backward report is {backward.shape}, its iterations {backward[:, ITERATIONS]}")
    if mode == "forward" and not backward[:, ITERATIONS].sum() > TUBE_STEPS:
        found.append(f"the tangent solves took {backward[:, ITERATIONS].sum()} iterations over {TUBE_STEPS} steps")
    return found


def change(found: float, held: float) -> float:
    return abs(found - held) / (abs(held) or 1.0)


def test_the_clean_runs_solves_converge_each_step_in_more_than_one_iteration(clean):
    mode, result = clean
    assert clean_problems(result, mode) == []


def test_a_capped_solve_is_warned_of_with_its_count_and_shown_in_the_report(capped):
    plant, mode, result = capped
    member = result.member(0)
    (report,) = member.solves
    steps, backward = np.asarray(report.steps), np.asarray(report.backward).reshape(-1, len(REPORT_FIELDS))
    if plant == "primal":
        assert report.forward_short == TUBE_STEPS and np.all(steps[:, AT_CAP] == 1)
        assert report.backward_short == TUBE_STEPS, "the transpose or tangent solves share the primal's stopping"
        assert any("primal solves stopped short" in text for text in member.warnings)
        assert any(("transpose" if mode == "reverse" else "tangent") + " solves stopped short" in text
                   for text in member.warnings), member.warnings
        assert any(flag.path == "water.flow.solve_flagged" and flag.count == TUBE_STEPS for flag in member.flags)
    elif mode == "reverse":
        assert report.forward_short == 0 and np.all(steps[:, AT_CAP] == 0)
        assert report.backward_short == TUBE_STEPS and np.all(backward[:, AT_CAP] == 1)
        assert [text for text in member.warnings if "primal" in text] == []
        assert any("transpose solves stopped short" in text for text in member.warnings)
    else:
        assert (report.forward_short, report.backward_short, member.warnings) == (0, 0, ()), (
            "forward mode takes no transpose solve, so the transpose cap caps nothing there")
        assert member.recheck is None


def test_a_capped_solve_is_rechecked_with_its_cap_lifted_and_reaches_the_clean_runs_answer(capped):
    """Where something stopped short, the recheck re-solves with the cap lifted: it converges, and its objective and
    derivative are the clean run's, where the capped member's are far from it (the primal cap; the transpose cap moves
    the derivative alone, the primal solves being converged)."""
    plant, mode, result = capped
    member, held = result.member(0), reference(mode)
    if plant == "transpose" and mode == "forward":
        pytest.skip("nothing capped: asserted in the report test")
    recheck = member.recheck
    assert recheck is not None and recheck.made and recheck.converged
    assert recheck.lifted == ("water.flow",) and recheck.not_rechecked == {}
    assert change(recheck.gradient[PATH], held.gradient[PATH]) < SAME
    assert change(recheck.value, held.value) < SAME
    assert change(member.gradient[PATH], held.gradient[PATH]) > CHANGED, "the capped derivative is far off"
    assert recheck.gradient_change[PATH] > CHANGED
    if plant == "primal":
        assert change(member.value, held.value) > CHANGED and recheck.value_change > CHANGED
    else:
        assert recheck.value_change < SAME, "the primal solves were converged: the objective does not move"


def test_the_rechecks_own_solves_are_run_and_converge_in_more_than_the_one_iteration_they_were_capped_at(capped):
    """The recheck's own solves are re-run and reported (a stub reporting a change of zero has none)."""
    plant, mode, result = capped
    if plant == "transpose" and mode == "forward":
        pytest.skip("nothing capped: asserted in the report test")
    recheck = result.member(0).recheck
    assert recheck is not None and len(recheck.solves) == 1
    (own,) = recheck.solves
    assert own.forward_short == 0 and own.backward_short == 0
    assert np.all(np.asarray(own.steps)[:, ITERATIONS] > 1), "converged in more than the one iteration it was capped at"


def test_a_flagged_node_with_a_non_finite_solve_or_no_recheck_is_reported_not_rechecked_and_nothing_is_re_solved():
    """The recheck's structure with the reports made by hand: a non-finite residual is not what a lifted cap or a
    restart removes, and a node whose solve reads no recheck has none; each member is told which, and no program is
    traced (``again`` fails if called)."""
    from mime.results import SolveReport
    from mime.study._gradient import NO_RECHECK_REASON, NOT_FINITE_REASON, _Member, _recheck

    def row(**counts: float) -> np.ndarray:
        return np.asarray([counts.get(name, 0.0) for name in REPORT_FIELDS])

    class Checking:
        rechecks = True

    class Plain:
        rechecks = False

    class Graph:
        def get_node(self, name: str):
            return Checking() if name == "checks" else Plain()

    def again():
        raise AssertionError("a program was traced for a recheck nothing could change")

    held = _Member(1.0, {PATH: np.asarray(2.0)}, (np.zeros(0), np.zeros(0), np.zeros(0)), {}, {}, {})
    reports = [(SolveReport("checks", row(not_finite=1.0), None, row(), "reverse"),),
               (SolveReport("plain", row(at_cap=2.0), None, row(), "reverse"),)]
    found = _recheck(Graph(), again, reports, [held, held], {}, {}, {}, {}, {}, "reverse")
    assert [recheck.made for recheck in found] == [False, False]
    assert [recheck.converged for recheck in found] == [False, False]
    assert found[0].not_rechecked == {"checks": NOT_FINITE_REASON}
    assert found[1].not_rechecked == {"plain": NO_RECHECK_REASON}
    assert all(np.isnan(recheck.value_change) for recheck in found)
