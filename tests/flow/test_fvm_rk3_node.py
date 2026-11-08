"""RK3's node (``ExplicitFlowNode``): route 1 per step with its one-step gift control (F1b's gate 7), the per-step CFL
check at RK3's line along the vessel's axis (gate 3 (c)), a state given from outside restarted from its consistent
pressure (gate 4's transplant half), and a batch over the viscosity equal to its separate runs (open item 8)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import Waveform, eps, in_dtype, step_of

from mime.core import ledger as books
from mime.nodes.flow.fvm import RK3, ExplicitFlowNode, _rk3, cfl_check, direction_factor, periodic_tube_layout

HOSTED = (
    books.LedgerDelivery("body_force", "drive:water.body_force", books.UNMETERED, books.HOSTED).encoded(),
    books.LedgerDelivery("viscous", "viscosity:water", books.DISSIPATIVE, books.HOSTED).encoded(),
    books.LedgerDelivery("penalty", "wall:vessel", books.DISSIPATIVE, books.HOSTED).encoded(),
)


def rk3_node(dh: int = 8, *, nu: float = 1.0, dt: float | None = None, length_cells: int = 4,
             lattice: tuple[int, int] = (0, 1), **settings) -> ExplicitFlowNode:
    layout = periodic_tube_layout(0.5, length_cells / dh, cells_across=dh, lattice=lattice)
    step = dt if dt is not None else 0.1 * layout.spacing ** 2 / nu
    return ExplicitFlowNode("flow", step, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                            spacing=layout.spacing, viscosity=nu, density=1.0, tube_centre=layout.centre,
                            tube_axis=layout.axis, tube_radius=layout.radius,
                            periodic_images="own" if lattice == (0, 1) else "transverse", **settings)


def run(node, steps, waveform=Waveform(steady=8.0), state=None, p=None):
    step = step_of(node, node._step_constants(None) if p is None else p, waveform)  # noqa: SLF001
    start = dict(node.initial_state()) if state is None else state
    return dict(jax.jit(lambda s: jax.lax.scan(lambda c, _: (step(c), None), s, None, length=steps)[0])(start))


def _stirred(node, amplitude, seed):
    rng = np.random.default_rng(seed)
    grid = node.grid(node._step_constants(None))  # noqa: SLF001
    swirl = grid.project(tuple(jnp.asarray(amplitude * rng.standard_normal(grid.cells)) for _ in range(3)))
    return {f"initial_velocity_{c}": np.asarray(swirl[k]) for k, c in enumerate("xyz")}


def _metered(dtype, seed):
    with in_dtype(dtype):
        start = _stirred(rk3_node(), 0.5, seed)
        node = rk3_node(ledger=HOSTED, **start)
        return run(node, 20)


# -- route 1 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_every_rk3_step_dissipates_and_no_identity_is_claimed(dtype, seed):
    state = _metered(dtype, seed)
    assert float(state[f"{books.RECORDS}route_one_negative_steps"]) == 0.0
    assert float(state[f"{books.RECORDS}route_one_worst"]) > 0.0
    assert float(state[f"{books.RECORDS}identity_worst"]) == 0.0
    assert float(state[f"{books.RECORDS}unattributed_worst"]) == 0.0


def _gifted(share):
    """RK3's step with an unbooked gift each step: the velocity scaled after the stages so the kinetic energy rises by
    ``share`` times the step's own dissipation (its viscous and penalty terms, the scheme's own pairing)."""
    original = _rk3.step

    def step(grid, velocity, pressure, viscosity, resistivity, force, convection, time, dt):
        u, p, parts = original(grid, velocity, pressure, viscosity, resistivity, force, convection, time, dt)
        kinetic = 0.5 * grid.inner(u, u)
        lost = parts.viscous + parts.penalty
        factor = jnp.sqrt(1.0 + share * lost / kinetic)
        return (factor * u[0], factor * u[1], factor * u[2]), p, parts

    return step


@pytest.mark.parametrize("seed", range(3))
def test_control_a_gift_twice_a_steps_dissipation_fails_route_one_on_every_step(dtype, seed, monkeypatch):
    """By the identity, ``W - dH = D - G``: a gift of twice the step's dissipation makes every step give energy. The
    control's size comes from the step's own paired terms, so it fails on every draw (F1b's first step: a gift sized
    from the kinetic energy alone was off by 1.1 to 1.7 times, and the period's gift diverged before its check)."""
    monkeypatch.setattr(_rk3, "step", _gifted(2.0))
    state = _metered(dtype, seed)
    assert float(state[f"{books.RECORDS}route_one_negative_steps"]) == 20.0


@pytest.mark.parametrize("seed", range(3))
def test_a_gift_under_half_a_steps_dissipation_is_not_counted(dtype, seed, monkeypatch):
    """The check's specificity: a gift that leaves the step dissipating is not a negative step."""
    monkeypatch.setattr(_rk3, "step", _gifted(0.5))
    state = _metered(dtype, seed)
    assert float(state[f"{books.RECORDS}route_one_negative_steps"]) == 0.0


# -- the per-step CFL check ----------------------------------------------------------------------------------------

def _plug(node, speed):
    """The lumen at a uniform axial ``speed`` (m/s) along the tube's axis, zero in the solid."""
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    fields = node.fields(p, grid)
    axis = np.asarray(p["tube_axis"])
    return {f"initial_velocity_{c}": np.asarray(speed * axis[k] * (1.0 - np.clip(np.asarray(fields.chi[k]), 0, 1)))
            for k, c in enumerate("xyz")}


@pytest.mark.parametrize(("lattice", "factor"), [((0, 1), 1.0), ((1, 1), 0.5)])
@pytest.mark.parametrize("ratio", [0.98, 1.02])
def test_the_check_stops_at_the_first_step_past_rk3s_line_along_the_axis(dtype, lattice, factor, ratio):
    """A plug at a known CFL: past ``1.7 g(n)`` the first step is named, under it none (the tilted tube's line is half
    the aligned one's)."""
    with in_dtype(dtype):
        probe = rk3_node(dt=1.0, lattice=lattice)
        h = probe.constants()["spacing"]
        dt = 0.05 * h * h
        line = RK3.cfl_line * factor
        start = _plug(rk3_node(dt=dt, lattice=lattice), 1.0)
        top = max(float(np.max(np.abs(v))) for v in start.values())
        speed = ratio * line * h / dt / top
        node = rk3_node(dt=dt, lattice=lattice, convection=True, **_plug(rk3_node(dt=dt, lattice=lattice), speed))
        state = run(node, 1, Waveform())
    assert float(direction_factor(node.constants()["tube_axis"])) == pytest.approx(factor, rel=1e-12)
    assert int(state["stopped_at"]) == (1 if ratio > 1.0 else -1)
    if ratio > 1.0:
        problem = node.stop_problem(state)
        assert problem is not None and problem.code == "flow.cfl-exceeded" and "dt='auto'" in problem.action


def test_a_stokes_run_is_never_stopped(dtype):
    with in_dtype(dtype):
        h = rk3_node(dt=1.0).constants()["spacing"]
        dt = 0.05 * h * h
        node = rk3_node(dt=dt, convection=False, **_plug(rk3_node(dt=dt), 10.0 * h / dt))
        state = run(node, 1, Waveform())
    assert int(state["stopped_at"]) == -1


def test_a_non_finite_velocity_counts_as_past_the_line(dtype):
    with in_dtype(dtype):
        node = rk3_node()
        start = dict(node.initial_state())
        start["velocity_z"] = start["velocity_z"].at[0, 0, 0].set(jnp.nan)
        state = run(node, 1, Waveform(), state=start)
    assert int(state["stopped_at"]) == 1
    assert node.stop_problem(state) is not None


def test_control_a_check_on_the_wrong_spacing_misses_the_first_step(dtype):
    """The CFL read on twice the spacing halves it: a plug at 1.2 times the line then passes, so the check's own
    measure is what the stop test sees."""
    with in_dtype(dtype):
        h = rk3_node(dt=1.0).constants()["spacing"]
        dt = 0.05 * h * h
        start = _plug(rk3_node(dt=dt), 1.0)
        top = max(float(np.max(np.abs(v))) for v in start.values())
        node = rk3_node(dt=dt, **_plug(rk3_node(dt=dt), 1.2 * RK3.cfl_line * h / dt / top))
        p = node._step_constants(None)  # noqa: SLF001
        velocity = (jnp.asarray(node.constants()[f"initial_velocity_{c}"]) for c in "xyz")
        _, right, _ = cfl_check(tuple(velocity), dt, p["spacing"], node.line(p), True, jnp.asarray(-1),
                                jnp.asarray(0.0), jnp.asarray(0))
        velocity = (jnp.asarray(node.constants()[f"initial_velocity_{c}"]) for c in "xyz")
        _, wrong, _ = cfl_check(tuple(velocity), dt, 2.0 * p["spacing"], node.line(p), True, jnp.asarray(-1),
                                jnp.asarray(0.0), jnp.asarray(0))
    assert int(right) == 1
    assert int(wrong) == -1


# -- a state given from outside -------------------------------------------------------------------------------------

def test_a_state_given_from_outside_restarts_from_its_consistent_pressure(float64):
    """A transplant writes the velocity and ``pressure_ready = 0``: its next step takes the consistent pressure, so
    whatever pressure the state carried does not matter, and the run equals one started from that velocity."""
    node = rk3_node()
    driven = run(node, 10)
    fresh = rk3_node(**{f"initial_velocity_{c}": np.asarray(driven[f"velocity_{c}"]) for c in "xyz"})
    given = dict(driven)
    given["pressure"] = jnp.asarray(np.random.default_rng(0).standard_normal(given["pressure"].shape))
    given["pressure_ready"] = jnp.zeros(())
    given["steps"] = jnp.asarray(0, dtype=jnp.int32)
    after_given = run(node, 3, state=given)
    after_fresh = run(fresh, 3)
    for c in "xyz":
        assert np.max(np.abs(after_given[f"velocity_{c}"] - after_fresh[f"velocity_{c}"])) <= 1e-12


def test_control_a_given_state_that_keeps_its_pressure_differs(float64):
    node = rk3_node()
    driven = run(node, 10)
    fresh = rk3_node(**{f"initial_velocity_{c}": np.asarray(driven[f"velocity_{c}"]) for c in "xyz"})
    given = dict(driven)
    given["pressure"] = jnp.asarray(np.random.default_rng(0).standard_normal(given["pressure"].shape))
    given["pressure_ready"] = jnp.ones(())
    given["steps"] = jnp.asarray(0, dtype=jnp.int32)
    after_given = run(node, 3, state=given)
    after_fresh = run(fresh, 3)
    assert max(np.max(np.abs(after_given[f"velocity_{c}"] - after_fresh[f"velocity_{c}"])) for c in "xyz") > 1e-6


# -- a batch over the viscosity -------------------------------------------------------------------------------------

def test_a_batch_over_the_viscosity_equals_its_separate_runs(dtype):
    """RK3 under ``vmap`` over the viscosity leaf, the box and its fields shared: each member its own run (R4-B-01's
    gap), to rounding on the run's scale."""
    with in_dtype(dtype):
        node = rk3_node(dt=0.05 * (1.0 / 8) ** 2)
        p = node._step_constants(None)  # noqa: SLF001
        viscosities = jnp.asarray([0.6, 0.8, 1.0])

        def one(mu):
            return run(node, 15, p={**p, "viscosity": mu})["velocity_z"]

        batched = jax.vmap(one)(viscosities)
        separate = [one(mu) for mu in viscosities]
    scale = float(np.max(np.abs(np.asarray(separate))))
    for k in range(3):
        assert np.max(np.abs(np.asarray(batched[k]) - np.asarray(separate[k]))) <= 64 * eps(dtype) * scale
