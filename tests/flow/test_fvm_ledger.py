"""Gate 5: the FVM's step identity to P1a-1's floor (16 eps of the step's scale) with convection on, and the gift
control, each judged by the ledger harness's one check (``mime.testing.ledger_closes``); the stall detector's report (gate 8c) and its line; the per-step CFL check and its stop (gate 11). F2b's
wiring: the ``fluid`` port kind's pairing (each law's drive its force against the fluid's momentum at the solution),
its step-identity case and its controls, and the projection's term shown zero to rounding at two tolerances."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import METERED, eps, in_dtype, run, tube_node

from mime.core import ledger as books
from mime.nodes.flow.fvm import IMEXEuler, ImplicitFlowNode
from mime.testing import ledger_closes


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
        node = tube_node(8, length_cells=4, dt=0.002, convection=True, ledger=METERED, **start, **settings)
        return node, run(node, 20)


@pytest.mark.parametrize("seed", range(3))
def test_the_flow_nodes_identity_closes_to_the_ledgers_floor_with_convection_on(dtype, seed):
    node, state = _metered(dtype, seed)
    judged = ledger_closes(state)
    assert judged.passed, judged.detail
    terms = np.asarray(state[f"{books.RECORDS}terms"])
    assert abs(terms[1]) > 0.0  # convection did book something at mixed time levels


@pytest.mark.parametrize("seed", range(3))
def test_control_an_energy_gift_is_caught_by_the_identity(dtype, seed, monkeypatch):
    """The solution scaled by ``1 + gamma`` after the solve, unbooked: the identity's residual is about ``2 gamma KE``,
    planted above each dtype's floor (16 eps of the step's scale) with a margin."""
    gamma = 1e-12 if dtype == "float64" else 2e-5
    original = IMEXEuler.step

    def gifted(self, context, state, time, dt):
        solved, parts = original(self, context, state, time, dt)
        return {"velocity": tuple((1.0 + gamma) * c for c in solved["velocity"])}, parts

    monkeypatch.setattr(IMEXEuler, "step", gifted)
    _, state = _metered(dtype, seed)
    judged = ledger_closes(state)
    assert judged.failed and "of its floor" in judged.detail


def test_an_unnamed_hosted_law_is_unattributed(dtype):
    with in_dtype(dtype):
        node = tube_node(8, length_cells=2, dt=0.002, convection=True, ledger=METERED[:3])
        state = run(node, 3)
    judged = ledger_closes(state)
    assert judged.failed and "unattributed" in judged.detail


def test_the_stall_detector_reads_a_transient_far_under_its_line(dtype):
    with in_dtype(dtype):
        node = tube_node(16, dt=0.002, convection=False, ledger=METERED)
        state = run(node, 20)
    assert 0.0 <= float(state["stall_ratio"]) < 1e-3
    assert float(state["stall_flagged"]) == 0.0
    judged = ledger_closes(state)
    assert judged.passed, judged.detail


def test_the_stall_detector_is_zero_at_rest_with_no_drive(dtype):
    with in_dtype(dtype):
        node = tube_node(8, length_cells=2, dt=0.002, g=0.0, convection=False, ledger=METERED)
        state = run(node, 2)
    assert float(state["stall_ratio"]) == 0.0
    judged = ledger_closes(state)
    assert judged.passed, judged.detail


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
    when it is not. Past it the state is held from that step: five steps leave it as it started. The control is the
    run inside the line, whose state moves."""
    with in_dtype(dtype):
        probe = tube_node(8, length_cells=2, dt=0.005, g=0.0)
        node = tube_node(8, length_cells=2, dt=0.005, g=0.0, **_poiseuille_start(probe, peak))
        start = node.initial_state()
        state = run(node, 5)
        cfl = peak * 0.005 * 8
        unchanged = all(np.array_equal(np.asarray(state[f"velocity_{c}"]), np.asarray(start[f"velocity_{c}"]))
                        for c in "xyz")
        if cfl > 0.4 + 1e-6:
            assert int(state["stopped_at"]) == 1 and unchanged and int(state["steps"]) == 0
            problem = node.stop_problem(state)
            assert problem is not None and problem.code == "flow.cfl-exceeded"
            assert "step 1" in problem.message and "dt <=" in problem.action
        else:
            assert int(state["stopped_at"]) == -1 and node.stop_problem(state) is None
            assert not unchanged and int(state["steps"]) == 5


def test_a_stokes_run_at_the_same_step_is_not_stopped(dtype):
    with in_dtype(dtype):
        probe = tube_node(8, length_cells=2, dt=0.005, g=0.0)
        node = tube_node(8, length_cells=2, dt=0.005, g=0.0, convection=False, **_poiseuille_start(probe, 25.0))
        state = run(node, 5)
    assert int(state["stopped_at"]) == -1
    assert node.stop_problem(state) is None


def test_a_cfl_that_is_not_finite_is_past_the_line(dtype):
    """A NaN in the velocity the convection uses: the check reads it as past the line (``cfl <= line`` is false for a
    NaN, where ``cfl > line`` would be false too) and stops the run at that step. The state is not finite, so the
    flow names no CFL problem: the run's own refusal names the first node and field that went so (a NaN from
    elsewhere is not read as a timestep's)."""
    with in_dtype(dtype):
        probe = tube_node(8, length_cells=2, dt=0.005, g=0.0)
        start = _poiseuille_start(probe, 2.0)
        start["initial_velocity_x"] = start["initial_velocity_x"].copy()
        start["initial_velocity_x"][3, 4, 0] = np.nan
        node = tube_node(8, length_cells=2, dt=0.005, g=0.0, **start)
        state = run(node, 3)
    assert int(state["stopped_at"]) == 1
    assert node.stop_problem(state) is None


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


# -- the fluid port kind: each law's drive paired with the fluid's momentum ---------------------------------------------

#: Two laws driving the flow's body force, each into its own input (a load of the ``fluid`` port kind), and the hosted
#: laws.
TWO_DRIVES = (
    books.LedgerDelivery("body_force", "body_force:a", books.UNMETERED, books.LOAD, "", "body_force:a").encoded(),
    books.LedgerDelivery("body_force", "body_force:b", books.UNMETERED, books.LOAD, "", "body_force:b").encoded(),
    *METERED[2:],
)


def _paired(node: ImplicitFlowNode, inputs: dict, seed: int) -> dict:
    """One metered step from a stirred start, compiled: the laws' rows, the fluid's momentum at the solution and its
    magnitude formed apart (``rho = 1``), the hosted drive's ``dt rho <g (1 - chi), u'>``, and the step's identity."""
    start = _stirred(node, 0.5, seed)
    stirred = {**node.initial_state(), **{f"velocity_{c}": start[f"initial_velocity_{c}"] for c in "xyz"}}
    p = node._step_constants(None)  # noqa: SLF001
    dt = node.delta_t

    def step(state, given):
        ended = node.advance(p, state, given, dt)
        _, parts = node.stepped(p, {k: v for k, v in state.items() if not k.startswith(books.RECORDS)},
                                {"body_force": sum(given.values())}, dt)
        grid = node.grid(p)
        fluid = tuple(1.0 - parts.fields.chi[c] for c in range(3))
        solved = parts.stage.solved
        volume = grid.cell_volume
        g = sum(given.values())
        return {"rows": ended[f"{books.RECORDS}step_work"], "identity": ended[f"{books.RECORDS}step_identity"],
                "unattributed": ended[f"{books.RECORDS}unattributed_worst"],
                "momentum": jnp.stack([volume * jnp.sum(fluid[c] * solved[c]) for c in range(3)]),
                "magnitude": jnp.stack([volume * jnp.sum(jnp.abs(fluid[c] * solved[c])) for c in range(3)]),
                "total": dt * grid.inner(tuple(g[c] * fluid[c] for c in range(3)), solved)}

    return {k: np.asarray(v, dtype=np.float64) for k, v in jax.jit(step)(stirred, inputs).items()}


@pytest.mark.parametrize("seed", range(3))
def test_each_law_s_drive_is_its_force_paired_with_the_fluid_s_momentum_at_the_solution(dtype, seed):
    """Two laws driving the flow, of different sizes and directions: each law's row is ``dt g . P'``, ``P' = rho <1 -
    chi, u'>`` formed apart from the node's records, to its rounding; the two rows sum to the hosted drive's ``dt rho
    <g (1 - chi), u'>``; and the step identity closes with the drive split by law."""
    with in_dtype(dtype):
        node = tube_node(8, length_cells=4, dt=0.002, convection=True, ledger=TWO_DRIVES)
        g = {books.law_input("body_force", 0): np.asarray([0.3, -0.2, 8.0], dtype=dtype),
             books.law_input("body_force", 1): np.asarray([-1.0, 0.5, -2.5], dtype=dtype)}
        found = _paired(node, g, seed)
    expected = [node.delta_t * float(np.dot(np.asarray(v, dtype=np.float64), found["momentum"])) for v in g.values()]
    size = node.delta_t * sum(float(np.dot(np.abs(np.asarray(v, dtype=np.float64)), found["magnitude"]))
                              for v in g.values())
    rows = found["rows"]
    assert np.allclose(rows[:2], expected, rtol=0.0, atol=16 * eps(dtype) * size)
    assert abs(rows[0] + rows[1] - float(found["total"])) <= 16 * eps(dtype) * size
    assert float(found["identity"]) <= books.IDENTITY_ULPS * eps(dtype)
    assert float(found["unattributed"]) == 0.0


def test_the_flow_refuses_a_drive_held_in_its_step():
    """The drive is a delivered load of the ``fluid`` port kind, never held: F1a's hosted ``body_force`` is refused by
    name, as any delivery the flow does not take."""
    hosted = books.LedgerDelivery("body_force", "drive:water", books.UNMETERED, books.HOSTED).encoded()
    with pytest.raises(ValueError, match="it takes no delivery"):
        run(tube_node(8, length_cells=2, dt=0.002, ledger=(hosted, *METERED[2:])), 1)


def test_the_fluid_port_case_passes():
    """The ``fluid`` port kind's step identity on drawn states (``mime.testing.step_identity``, the registry walk's case
    for the FVM), on a short run of draws; the walk runs the full 200."""
    from mime.dynamics import FVM
    from mime.testing import step_identity

    with in_dtype("float64"):
        assert step_identity(FVM.__new__(FVM), max_examples=20).passed


def test_control_the_fluid_case_fails_where_the_plain_input_s_work_is_booked_under_the_law(monkeypatch):
    """The plain input's work booked under the law: the identity still closes, and attribution fails (the case's own
    control; its rate over 200 draws is in the PR's report)."""
    from mime.dynamics import FVM
    from mime.testing import step_identity

    original = ImplicitFlowNode.drive_work

    def misbooked(self, *args):
        work, unattributed, scale = original(self, *args)
        return work.at[0].add(unattributed), 0.0 * unattributed, scale

    monkeypatch.setattr(ImplicitFlowNode, "drive_work", misbooked)
    with in_dtype("float64"):
        found = step_identity(FVM.__new__(FVM), max_examples=20)
    assert found.failed and "unattributed" in found.detail


def test_control_the_fluid_case_fails_where_the_step_gives_energy(monkeypatch):
    """The solution scaled by ``1 + 1e-9`` unbooked: a gift far above the floor in float64, which the identity sees."""
    from mime.dynamics import FVM
    from mime.testing import step_identity

    step = IMEXEuler.step

    def gifted(self, context, state, time, dt):
        solved, parts = step(self, context, state, time, dt)
        return {"velocity": tuple((1.0 + 1e-9) * c for c in solved["velocity"])}, parts

    monkeypatch.setattr(IMEXEuler, "step", gifted)
    with in_dtype("float64"):
        found = step_identity(FVM.__new__(FVM), max_examples=20)
    assert found.failed and "floor" in found.detail


# -- N_proj: the projection's term, zero to rounding on the periodic box (R5's C13, its second option) ----------------

#: Each arm, ``(scheme, dtype, tolerance, plant)``: the control's plant is a gradient field of ``plant`` times the
#: solution's size, added to it unprojected, each measured above the floor on ten draws of ten at both tolerances:
#: IMEX Euler in float64 at 1e-9 (14 to 224 floors), the steady limit in float64 at 1e-6 (2.0e4 to 2.2e4; at 1e-9 it
#: reads 0.02) and in float32 at 1e-3 (38 to 40; at 1e-4 it reads 0.4). **No float32 arm for IMEX Euler**: there a
#: plant of 1e-3 of the solution's size reads 0.07 to 0.52 floors, so float32's floor bounds ``N_proj`` only to about a
#: hundredth of the solution there, and no plant at or under that discriminates.
PROJECTION_ARMS = [("imex_euler", "float64", 1e-6, 1e-9), ("imex_euler", "float64", 1e-10, 1e-9),
                   ("steady", "float64", 1e-6, 1e-6), ("steady", "float64", 1e-10, 1e-6),
                   ("steady", "float32", 1e-4, 1e-3), ("steady", "float32", None, 1e-3)]


def _projection_terms(node: ImplicitFlowNode, seeds: range, plant_share: float) -> list[tuple[float, float, float]]:
    """For each seed, ``dt rho <p', D u'>`` of one step from a stirred start, ``p'`` the pressure recovered from the
    step's remainder (B2's ``pressure_of``); the same with the plant on the solution; and the step's scale. Compiled
    once."""
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)

    def terms(own, bump):
        _, parts = node.stepped(p, own, {"body_force": jnp.asarray([0.0, 0.0, 8.0])}, node.delta_t)
        stage = parts.stage
        size = jnp.sqrt(grid.inner(stage.solved, stage.solved) / grid.inner(bump, bump))
        sigma = node.resistivity(p, parts.fields)
        found = []
        for plant in (0.0, plant_share):
            solved = tuple(stage.solved[c] + plant * size * bump[c] for c in range(3))
            applied = node.operator_of(p, grid, sigma)(stage.shift)(solved)
            pressure = grid.pressure_of(tuple(stage.rhs[c] - applied[c] for c in range(3)))
            found.append(node.delta_t * jnp.sum(pressure * grid.divergence(solved)) * grid.cell_volume)
        return found[0], found[1], parts.energy["scale"]

    compiled = jax.jit(terms)
    held = []
    for seed in seeds:
        start = _stirred(node, 0.5, seed)
        state = {**node.initial_state(), **{f"velocity_{c}": start[f"initial_velocity_{c}"] for c in "xyz"}}
        own = {k: v for k, v in state.items() if not k.startswith(books.RECORDS)}
        bump = grid.gradient(jnp.asarray(np.random.default_rng(seed).standard_normal(grid.cells)))
        held.append(tuple(float(v) for v in compiled(own, bump)))
    return held


@pytest.mark.parametrize(("scheme", "dtype", "tolerance", "plant"), PROJECTION_ARMS,
                         ids=[f"{a}-{d}-{t}" for a, d, t, _ in PROJECTION_ARMS])
def test_the_projection_s_term_is_zero_to_rounding_at_two_tolerances(scheme, dtype, tolerance, plant):
    """``N_proj`` is not booked on the periodic box: the projected CG's answer passes through the grid's exact
    projector, so ``D u'`` is its rounding, at any tolerance; ``N_Krylov``'s unprojected remainder holds whatever is
    left. Its control, the solution moved off the divergence-free fields by a gradient of the arm's share of its size
    (:data:`PROJECTION_ARMS`), puts the term above the floor on every draw."""
    with in_dtype(dtype):
        settings = {} if tolerance is None else {"tolerance": tolerance}
        node = tube_node(8, length_cells=4, dt=0.002, convection=scheme != "steady", scheme=scheme, ledger=METERED,
                         **settings)
        found = _projection_terms(node, range(3), plant)
    for term, planted, scale in found:
        assert abs(term) <= books.IDENTITY_ULPS * eps(dtype) * scale
        assert abs(planted) > books.IDENTITY_ULPS * eps(dtype) * scale
