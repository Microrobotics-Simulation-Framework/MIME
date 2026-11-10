"""Gate 7: the iterative solve's report in a study's derivative (the FVM's conjugate gradient, the design every
iterative solve in MIME writes): a solve stopped at its cap, on the primal (and so its tangent) solve or on its
transpose solve alone, is warned of with its count, shows in the per-step report (forward: each step's primal report;
backward: the transpose solves' report a step, the tube keeping a row a step), and is rechecked at a tighter tolerance,
the change in the objective and the derivative reported with whether the recheck's own solves converged.

The case: the periodic tube's derivative in its viscosity, reverse mode, 20 steps; the cap planted at one iteration
(``gradient_plants.cap_plant``, ``MIME_PLANT_SOLVE_CAP`` for the gate kit's rate). Its clean run takes 4 to 6 iterations a step
at the default tolerance, so one iteration cannot reach it; the clean run's own counts are asserted to show that."""

from __future__ import annotations

import numpy as np
import pytest
from gradient_rigs import TUBE_STEPS, flux, tube
from gradient_plants import CAP, cap_plant, from_environment

from mime.nodes.flow.fvm import REPORT_FIELDS
from mime.study import Study, along
from mime.units import mPa_s, ms

AT_CAP, ITERATIONS = REPORT_FIELDS.index("at_cap"), REPORT_FIELDS.index("iterations")
ACKNOWLEDGED = {"water.flow": "gate 7 reads the solve's reports through the tube's viscosity"}


def run(plant: str | None):
    with pytest.MonkeyPatch.context() as patch:
        cap_plant(patch, plant)
        return Study(tube(report_rows=TUBE_STEPS, name=f"tube_{plant or 'clean'}")).sweep(
            along({"water.viscosity": [mPa_s(1.0)]}), duration=ms(20.0 * TUBE_STEPS)).gradient(
            flux, mode="reverse", acknowledged=ACKNOWLEDGED)


@pytest.fixture(scope="module")
def clean():
    return run(from_environment(CAP))


@pytest.fixture(scope="module", params=["primal", "transpose"])
def capped(request):
    return request.param, run(request.param)


def clean_problems(result) -> list[str]:
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
    backward = np.asarray(report.backward)
    if backward.shape != (TUBE_STEPS, len(REPORT_FIELDS)) or not np.all(backward[:, ITERATIONS] > 1):
        found.append(f"the per-step backward report is {backward.shape}, its iterations {backward[:, ITERATIONS]}")
    return found


def test_the_clean_runs_solves_converge_each_step_in_more_than_one_iteration(clean):
    assert clean_problems(clean) == []


def test_a_capped_solve_is_warned_of_with_its_count_shown_each_step_and_rechecked(capped):
    which, result = capped
    member = result.member(0)
    (report,) = member.solves
    steps, backward = np.asarray(report.steps), np.asarray(report.backward)
    if which == "primal":
        assert report.forward_short == TUBE_STEPS and np.all(steps[:, AT_CAP] == 1)
        assert any("primal solves stopped short" in text for text in member.warnings)
        assert any(flag.path == "water.flow.solve_flagged" and flag.count == TUBE_STEPS for flag in member.flags)
    else:
        assert report.forward_short == 0 and np.all(steps[:, AT_CAP] == 0)
        assert report.backward_short == TUBE_STEPS and np.all(backward[:, AT_CAP] == 1)
        assert any("transpose solves stopped short" in text for text in member.warnings)
    recheck = member.recheck
    assert recheck is not None and recheck.tolerance is not None
    assert recheck.tolerance == pytest.approx(1e-10, rel=1e-12)
    assert not recheck.converged, "the cap holds in the recheck too, and the recheck says so"
    assert set(recheck.gradient_change) == {"water.viscosity"} and np.isfinite(recheck.value_change)


def test_control_the_clean_gate_fails_on_a_capped_run(capped):
    """The clean run's assertions catch either cap."""
    assert clean_problems(capped[1])
