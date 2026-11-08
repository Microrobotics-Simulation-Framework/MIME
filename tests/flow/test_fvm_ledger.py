"""Gate 5: the FVM's step identity to P1a-1's floor (16 eps of the step's scale) with convection on, and the gift
control; the stall detector's report (gate 8c) and its line; the per-step CFL check and its stop (gate 11)."""

from __future__ import annotations

import numpy as np
import pytest
from fvm_cases import eps, in_dtype, run, tube_node

from mime.core import ledger as books
from mime.nodes.flow.fvm import IMEXEuler, ImplicitFlowNode

#: The hosted laws of a metered flow node, as the build will name them.
HOSTED = (
    books.LedgerDelivery("body_force", "drive:water.body_force", books.UNMETERED, books.HOSTED).encoded(),
    books.LedgerDelivery("viscous", "viscosity:water", books.DISSIPATIVE, books.HOSTED).encoded(),
    books.LedgerDelivery("penalty", "wall:vessel", books.DISSIPATIVE, books.HOSTED).encoded(),
)


def _stirred(node: ImplicitFlowNode, amplitude: float, seed: int):
    """A divergence-free start with a random swirl over the lumen, so that convection does work at mixed time levels."""
    rng = np.random.default_rng(seed)
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    swirl = grid.project(tuple(amplitude * rng.standard_normal(grid.cells) for _ in range(3)))
    return {f"initial_velocity_{c}": np.asarray(swirl[k]) for k, c in enumerate("xyz")}


def _metered(dtype: str, seed: int, **settings):
    with in_dtype(dtype):
        probe = tube_node(8, length_cells=4, dt=0.002, convection=True)
        start = _stirred(probe, 0.5, seed)
        node = tube_node(8, length_cells=4, dt=0.002, convection=True, ledger=HOSTED, **start, **settings)
        return node, run(node, 20)


@pytest.mark.parametrize("seed", range(3))
def test_the_flow_nodes_identity_closes_to_the_ledgers_floor_with_convection_on(dtype, seed):
    node, state = _metered(dtype, seed)
    worst = float(state[f"{books.RECORDS}identity_worst"])
    assert worst <= books.IDENTITY_ULPS * eps(dtype)
    assert float(state[f"{books.RECORDS}unattributed_worst"]) == 0.0
    assert float(np.sum(state[f"{books.RECORDS}gift_steps"])) == 0.0
    terms = np.asarray(state[f"{books.RECORDS}terms"])
    assert abs(terms[1]) > 0.0  # convection did book something at mixed time levels


@pytest.mark.parametrize("seed", range(3))
def test_control_an_energy_gift_is_caught_by_the_identity(dtype, seed, monkeypatch):
    """The solution scaled by ``1 + gamma`` after the solve, unbooked: the identity's residual is about ``2 gamma KE``,
    planted above each dtype's floor (16 eps of the step's scale) with a margin."""
    gamma = 1e-12 if dtype == "float64" else 2e-5
    original = IMEXEuler.step

    def gifted(self, context, velocity, time, dt):
        solved, parts = original(self, context, velocity, time, dt)
        return tuple((1.0 + gamma) * c for c in solved), parts

    monkeypatch.setattr(IMEXEuler, "step", gifted)
    _, state = _metered(dtype, seed)
    assert float(state[f"{books.RECORDS}identity_worst"]) > books.IDENTITY_ULPS * eps(dtype)


def test_an_unnamed_hosted_law_is_unattributed(dtype):
    with in_dtype(dtype):
        node = tube_node(8, length_cells=2, dt=0.002, convection=True, ledger=HOSTED[:2])
        state = run(node, 3)
    assert float(state[f"{books.RECORDS}unattributed_worst"]) > 0.0


def test_the_stall_detector_reads_a_transient_far_under_its_line(dtype):
    with in_dtype(dtype):
        node = tube_node(16, dt=0.002, convection=False, ledger=HOSTED)
        state = run(node, 20)
    assert 0.0 <= float(state["stall_ratio"]) < 1e-3
    assert float(state["stall_flagged"]) == 0.0


def test_the_stall_detector_is_zero_at_rest_with_no_drive(dtype):
    with in_dtype(dtype):
        node = tube_node(8, length_cells=2, dt=0.002, g=0.0, convection=False, ledger=HOSTED)
        state = run(node, 2)
    assert float(state["stall_ratio"]) == 0.0


def test_a_ledger_off_run_has_no_stall_report(dtype):
    with in_dtype(dtype):
        state = run(tube_node(8, length_cells=2, dt=0.002, convection=False), 2)
    assert "stall_ratio" not in state


# -- the per-step CFL check ----------------------------------------------------------------------------------------

def _poiseuille_start(node: ImplicitFlowNode, peak: float):
    c = node.constants()
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    points = np.asarray(grid.face_points(2, "float64"))
    r2 = (points[..., 0] - c["tube_centre"][0]) ** 2 + (points[..., 1] - c["tube_centre"][1]) ** 2
    w = np.clip(peak * (1.0 - r2 / c["tube_radius"] ** 2), 0.0, None)
    zero = np.zeros(grid.cells)
    return {"initial_velocity_x": zero, "initial_velocity_y": zero, "initial_velocity_z": w}


@pytest.mark.parametrize("peak", [9.0, 25.0])
def test_a_run_past_the_cfl_line_stops_at_its_first_step_past_it(dtype, peak):
    """Peak speed ``peak`` at dt 0.005 and h 1/8: a CFL of ``peak dt / h`` (0.36 and 1.0), held at or past the line from
    the start; the check names step 1 when it is past (the first step past, kept however the run goes on) and never
    when it is not."""
    with in_dtype(dtype):
        probe = tube_node(8, length_cells=2, dt=0.005, g=0.0)
        node = tube_node(8, length_cells=2, dt=0.005, g=0.0, **_poiseuille_start(probe, peak))
        state = run(node, 5)
        cfl = peak * 0.005 * 8
        if cfl > 0.4 + 1e-6:
            assert int(state["stopped_at"]) == 1
            problem = node.stop_problem(state)
            assert problem is not None and problem.code == "flow.cfl-exceeded"
            assert "step 1" in problem.message and "dt <=" in problem.action
        else:
            assert int(state["stopped_at"]) == -1 and node.stop_problem(state) is None


def test_a_stokes_run_at_the_same_step_is_not_stopped(dtype):
    with in_dtype(dtype):
        probe = tube_node(8, length_cells=2, dt=0.005, g=0.0)
        node = tube_node(8, length_cells=2, dt=0.005, g=0.0, convection=False, **_poiseuille_start(probe, 25.0))
        state = run(node, 5)
    assert int(state["stopped_at"]) == -1
    assert node.stop_problem(state) is None


def test_a_cfl_that_is_not_finite_is_past_the_line(dtype):
    """A NaN in the velocity the convection uses: the check reads it as past the line (``cfl <= line`` is false for a
    NaN, where ``cfl > line`` would be false too) and stops the run at that step, naming it."""
    with in_dtype(dtype):
        probe = tube_node(8, length_cells=2, dt=0.005, g=0.0)
        start = _poiseuille_start(probe, 2.0)
        start["initial_velocity_x"] = start["initial_velocity_x"].copy()
        start["initial_velocity_x"][3, 4, 0] = np.nan
        node = tube_node(8, length_cells=2, dt=0.005, g=0.0, **start)
        state = run(node, 3)
    assert int(state["stopped_at"]) == 1
    problem = node.stop_problem(state)
    assert problem is not None and problem.code == "flow.cfl-exceeded" and "not finite" in problem.message


def test_control_a_cfl_check_written_as_greater_than_misses_the_nan(dtype):
    """The control: ``cfl > line`` never fires on a NaN."""
    nan = np.float64("nan") if dtype == "float64" else np.float32("nan")
    assert not (nan > 0.4) and not (nan <= 0.4)


def test_the_cfl_line_along_a_tilted_axis_carries_its_factor():
    """The line along a vessel's axis ``n`` is ``0.4 max|n_d| / sum|n_d|``: 0.4 aligned with z, 0.3 along (1, 0, 3)."""
    scheme = IMEXEuler()
    assert float(scheme.cfl_limit((0.0, 0.0, 1.0))) == pytest.approx(0.4)
    assert float(scheme.cfl_limit((1.0, 0.0, 3.0))) == pytest.approx(0.3)
    assert float(scheme.cfl_limit((1.0, 1.0, 1.0))) == pytest.approx(0.4 / 3.0)
