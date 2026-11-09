"""Gate 1: the manufactured solution with convection and the pressure gradient, IMEX Euler on the projected monolithic
CG, judged against the exact solution (order 2 in space with dt tied to h^2, order 1 in time); its planted controls
(a convection sign flip, one component scaled by 1.05, a one-cell index shift; the recovered pressure's convective term
dropped and its sign flipped); and the stored golden step for the time level, which no order test can see.

S1's fields on a (2 pi)^3 box, nu = 0.1: ``u = cos(t) U1 + (1 + sin(2t)/2)/2 U2``, ``p = cos(t) sin x sin y cos z``, the
forcing made from the analytic fields by automatic differentiation, never from the code's own operators. dt = 0.15 h^2
keeps the CFL ``max|u| dt / h`` under 0.33 at N 8 (max|u| about 2.75), inside IMEX Euler's line of 0.4."""

from __future__ import annotations

import functools
import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import eps

from mime.nodes.flow.fvm import IMEXEuler, Jacobi, MACGrid, Stopping, projected_solve
from mime.nodes.flow.fvm._schemes import StepContext

NU = 0.1
T_END = 0.5
GOLDEN = Path(__file__).parent / "data" / "golden_imex_step.npz"
GOLDEN_SHA256 = "3cc2fb83866106c4fce9add3a43b278727d6a03b8ddcf0cb66cbf3f42e5697ea"


def _u1(x):
    return jnp.stack([jnp.sin(x[0]) * jnp.cos(x[1]) * jnp.cos(x[2]), jnp.cos(x[0]) * jnp.sin(x[1]) * jnp.cos(x[2]),
                      -2 * jnp.cos(x[0]) * jnp.cos(x[1]) * jnp.sin(x[2])])


def _u2(x):
    return jnp.stack([jnp.cos(x[1] - x[2]), jnp.sin(x[2] + x[0]), jnp.cos(x[0] + 2 * x[1])])


def _pressure(x):
    return jnp.sin(x[0]) * jnp.sin(x[1]) * jnp.cos(x[2])


def _a(t):
    return jnp.cos(t)


def _b(t):
    return 0.5 * (1.0 + 0.5 * jnp.sin(2 * t))


def _velocity(x, t):
    return _a(t) * _u1(x) + _b(t) * _u2(x)


def _forcing(x, t):
    """``du/dt + (u . grad) u - nu lap u + grad p``, every derivative by automatic differentiation."""
    u = _velocity(x, t)
    dudt = jax.jacfwd(_velocity, argnums=1)(x, t)
    gradient = jax.jacfwd(_velocity, argnums=0)(x, t)
    laplacian = jnp.trace(jax.hessian(_velocity, argnums=0)(x, t), axis1=1, axis2=2)
    return dudt + gradient @ u - NU * laplacian + jnp.cos(t) * jax.grad(_pressure)(x)


class Planted(MACGrid):
    """The grid kind with its convective term planted: ``sign`` (-1 flips it), ``scale_x`` (the x component's
    factor), ``shift`` (the x component rolled one cell along y)."""

    plant = "none"

    def convection(self, velocity):
        c = super().convection(velocity)
        if self.plant == "sign":
            return (-c[0], -c[1], -c[2])
        if self.plant == "scale":
            return (1.05 * c[0], c[1], c[2])
        if self.plant == "shift":
            return (jnp.roll(c[0], 1, axis=1), c[1], c[2])
        return c


def _grid(n: int, plant: str) -> MACGrid:
    cls = type(f"Planted_{plant}", (Planted,), {"plant": plant})
    return cls((n, n, n), 2.0 * math.pi / n)


@functools.cache
def _run(n: int, dt_steps: int | None, plant: str, pressure_plant: str, dtype: str):
    """The run to T_END at dt = 0.15 h^2 (or T_END / dt_steps): its velocity and recovered pressure errors, relative
    in L2, the pressure mean-free (judged up to its constant)."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        grid = _grid(n, plant)
        h = grid.spacing
        steps = dt_steps if dt_steps is not None else math.ceil(T_END / (0.15 * h * h))
        dt = T_END / steps
        points = [grid.face_points(c, dtype).reshape(-1, 3) for c in range(3)]
        sample_velocity = jax.jit(lambda t: tuple(
            jax.vmap(lambda x: _velocity(x, t)[c])(points[c]).reshape(grid.cells) for c in range(3)))
        sample_force = jax.jit(lambda t: tuple(
            jax.vmap(lambda x: _forcing(x, t)[c])(points[c]).reshape(grid.cells) for c in range(3)))
        tolerance = 1e-12 if dtype == "float64" else 1e-5
        stopping = Stopping(tolerance=jnp.asarray(tolerance, dtype=dtype))
        tap = jnp.zeros(6, dtype=dtype)

        def operator(shift):
            def apply(u):
                lap = grid.laplacian(u)
                return tuple(shift * a - NU * b for a, b in zip(u, lap, strict=True))
            return apply

        def solve(shift, rhs, guess):
            diagonal = tuple(jnp.full(grid.cells, shift - NU * grid.laplacian_diagonal(), dtype=dtype)
                             for _ in range(3))
            return projected_solve(grid, operator(shift), diagonal, rhs, guess, stopping, tap, Jacobi())

        context = StepContext(grid, operator, lambda stage, t: sample_force(t), True, solve)
        scheme = IMEXEuler()

        def body(u, k):
            ended, parts = scheme.step(context, {"velocity": u}, k * dt, dt)
            return ended["velocity"], None

        start = grid.project(sample_velocity(jnp.asarray(0.0, dtype=dtype)))
        run = jax.jit(lambda u: jax.lax.scan(body, u, jnp.arange(steps - 1, dtype=dtype))[0])
        before = run(start)
        ended, parts = jax.jit(lambda u: scheme.step(context, {"velocity": u}, (steps - 1) * dt, dt))(before)
        last = ended["velocity"]
        remainder = tuple(r - a for r, a in zip(parts.rhs, operator(parts.shift)(last), strict=True))
        if pressure_plant == "no_convection":
            remainder = tuple(r - c for r, c in zip(remainder, parts.convection, strict=True))
        pressure = grid.pressure_of(remainder)
        if pressure_plant == "sign":
            pressure = -pressure
        exact = sample_velocity(jnp.asarray(T_END, dtype=dtype))
        error = math.sqrt(sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(last, exact, strict=True))
                          / sum(float(jnp.sum(b ** 2)) for b in exact))
        centres = grid.cell_points(dtype).reshape(-1, 3)
        exact_p = (jnp.cos(T_END) * jax.vmap(_pressure)(centres)).reshape(grid.cells)
        exact_p = exact_p - jnp.mean(exact_p)
        pressure = pressure - jnp.mean(pressure)
        p_error = math.sqrt(float(jnp.sum((pressure - exact_p) ** 2)) / float(jnp.sum(exact_p ** 2)))
        speed = max(float(jnp.max(jnp.abs(c))) for c in exact)
        return error, p_error, speed * dt / h
    finally:
        jax.config.update("jax_enable_x64", held)


def _orders(errors, ns):
    return [math.log(errors[i] / errors[i + 1]) / math.log(ns[i + 1] / ns[i]) for i in range(len(ns) - 1)]


#: The space sweep, and the gate's lines on its finest pair: the velocity's observed order and the recovered pressure's.
SWEEP = (8, 16, 32)
VELOCITY_ORDER, PRESSURE_ORDER = 1.8, 1.5


def _sweep(dtype: str, plant: str = "none", pressure_plant: str = "none") -> tuple[float, float]:
    runs = [_run(n, None, plant, pressure_plant, dtype) for n in SWEEP]
    for _, _, cfl in runs:
        assert cfl <= 0.4
    return _orders([r[0] for r in runs], SWEEP)[-1], _orders([r[1] for r in runs], SWEEP)[-1]


def test_the_manufactured_solution_converges_at_second_order_in_space(dtype):
    velocity, pressure = _sweep(dtype)
    assert velocity >= VELOCITY_ORDER
    assert pressure >= PRESSURE_ORDER


@pytest.mark.parametrize("plant", ["sign", "scale", "shift"])
def test_control_a_planted_convective_term_fails_the_space_sweep(dtype, plant):
    """Each plant fails the gate's own line. One component scaled by 1.05 is the thinnest: its error sits under the
    clean one's at N 16 and is seen only as convergence stalls between N 16 and 32 (order 1.44 against 1.96 in
    float64)."""
    velocity, _ = _sweep(dtype, plant)
    assert velocity < VELOCITY_ORDER


@pytest.mark.parametrize("plant", ["no_convection", "sign"])
def test_control_a_planted_pressure_recovery_fails_the_space_sweep(dtype, plant):
    _, pressure = _sweep(dtype, "none", plant)
    assert pressure < PRESSURE_ORDER


def test_the_manufactured_solution_is_first_order_in_time(float64):
    """On one grid (N 16) against its own run at a sixteenth of the step, so the space error cancels: order 1."""
    errors = [_time_error(steps) for steps in (10, 20, 40)]
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(2.0) for i in range(2)]
    assert all(0.85 <= order <= 1.15 for order in orders)


@functools.cache
def _final(n: int, steps: int):
    """The velocity at T_END on an N grid with ``steps`` steps (float64), for the time order."""
    grid = _grid(n, "none")
    dt = T_END / steps
    points = [grid.face_points(c, "float64").reshape(-1, 3) for c in range(3)]
    sample_velocity = jax.jit(lambda t: tuple(
        jax.vmap(lambda x: _velocity(x, t)[c])(points[c]).reshape(grid.cells) for c in range(3)))
    sample_force = jax.jit(lambda t: tuple(
        jax.vmap(lambda x: _forcing(x, t)[c])(points[c]).reshape(grid.cells) for c in range(3)))
    stopping = Stopping(tolerance=jnp.asarray(1e-12))
    tap = jnp.zeros(6)

    def operator(shift):
        return lambda u: tuple(shift * a - NU * b for a, b in zip(u, grid.laplacian(u), strict=True))

    def solve(shift, rhs, guess):
        diagonal = tuple(jnp.full(grid.cells, shift - NU * grid.laplacian_diagonal()) for _ in range(3))
        return projected_solve(grid, operator(shift), diagonal, rhs, guess, stopping, tap, Jacobi())

    context = StepContext(grid, operator, lambda stage, t: sample_force(t), True, solve)
    scheme = IMEXEuler()

    def body(u, k):
        return scheme.step(context, {"velocity": u}, k * dt, dt)[0]["velocity"], None

    start = grid.project(sample_velocity(jnp.asarray(0.0)))
    return jax.jit(lambda u: jax.lax.scan(body, u, jnp.arange(steps, dtype=jnp.float64))[0])(start)


def _time_error(steps: int) -> float:
    reference = _final(16, 320)
    found = _final(16, steps)
    return math.sqrt(sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(found, reference, strict=True))
                     / sum(float(jnp.sum(b ** 2)) for b in reference))


# -- the golden step -----------------------------------------------------------------------------------------------

def _golden_step(plant: str):
    data = np.load(GOLDEN)
    n, nu, dt, t0 = int(data["n"]), float(data["nu"]), float(data["dt"]), float(data["t0"])
    grid = MACGrid((n, n, n), 2.0 * math.pi / n)
    force = tuple(jnp.asarray(f) for f in data["force"])
    stopping = Stopping(tolerance=jnp.asarray(1e-13))
    tap = jnp.zeros(6)

    def operator(shift):
        return lambda u: tuple(shift * a - nu * b for a, b in zip(u, grid.laplacian(u), strict=True))

    def solve(shift, rhs, guess):
        diagonal = tuple(jnp.full(grid.cells, shift - nu * grid.laplacian_diagonal()) for _ in range(3))
        return projected_solve(grid, operator(shift), diagonal, rhs, guess, stopping, tap, Jacobi())

    context = StepContext(grid, operator, lambda stage, t: tuple(jnp.cos(t) * f for f in force), True, solve)
    start = tuple(jnp.asarray(v) for v in data["velocity_in"])
    ended = IMEXEuler().step(context, {"velocity": start}, jnp.asarray(t0), dt)[0]["velocity"]
    if plant == "picard":
        # The time-level plant: convection evaluated at the step's own solution, by one more pass.
        shift = 1.0 / dt
        convection = grid.convection(ended)
        rhs = tuple(shift * a + jnp.cos(t0 + dt) * f + c for a, f, c in zip(start, force, convection, strict=True))
        ended, _ = solve(shift, rhs, start)
    expected = tuple(jnp.asarray(v) for v in data["velocity_out"])
    difference = math.sqrt(sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(ended, expected, strict=True)))
    return difference / math.sqrt(sum(float(jnp.sum(b ** 2)) for b in expected))


def test_the_golden_steps_file_is_the_one_recorded():
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == GOLDEN_SHA256


def test_one_step_equals_the_independent_implementations_golden_step(float64):
    assert _golden_step("none") <= 1e-11


def test_control_convection_at_the_wrong_time_level_misses_the_golden_step(float64):
    assert _golden_step("picard") > 1e-6


# -- the registry's interface, written for RK3 (M4) ----------------------------------------------------------------

def test_a_scheme_whose_state_carries_a_pressure_and_whose_stages_read_their_own_instants_steps(float64):
    """Round 4, item 7: a scheme's step and start carry a scheme state keyed by its ``state_fields``, and its stages
    read the drive at their own instants through the context, which also gives an explicit stage the viscosity and the
    resistivity. A toy two-stage scheme carrying a pressure steps through the interface unchanged; the flow node gives
    a scheme whose stages read off the step's end the delivered waveform at each stage's time. The control: the
    registered monolithic schemes read at the step's end alone, where the held delivery is made."""
    from mime.nodes.flow._waveform import waveform, waveform_at
    from mime.nodes.flow.fvm._schemes import StageKind, TimeScheme
    from fvm_cases import tube_node

    asked: list[tuple[int, float]] = []

    class TwoStage(TimeScheme):
        name, version = "two_stage", "0.0.0"
        stage_kind = StageKind.EXPLICIT_PROJECTED
        state_fields = ("velocity", "pressure")
        stage_times = drive_instants = (0.0, 0.5)
        ledger_route, ledger_terms, cfl_line = 1, ("convection",), 1.0
        steady, stores_inertia, takes_convection = False, True, True

        def start(self, grid, velocity):
            return {"velocity": grid.project(velocity), "pressure": jnp.zeros(grid.cells)}

        def step(self, context, state, time, dt):
            u = state["velocity"]
            for k, instant in enumerate(self.drive_instants):
                asked.append((k, float(time + instant * dt)))
                f = context.force(k, time + instant * dt)
                u = tuple(a + 0.5 * dt * (b - context.resistivity[i] * a) for i, (a, b) in enumerate(zip(u, f)))
            return {"velocity": context.grid.project(u), "pressure": state["pressure"] + context.viscosity}, None

    node = tube_node(8, length_cells=2)
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    fields = node.fields(p, grid)
    coefficients = waveform(jnp.asarray([0.0, 0.0, 1.0]), jnp.asarray(2.0), 1.0, 3.0, 0.25)
    scheme = TwoStage()
    force = node.forcing(fields, {"body_force": jnp.zeros(3), "waveform": coefficients}, scheme, jnp.float64)
    context = StepContext(grid, node.operator_of(p, grid, node.resistivity(p, fields)), force, False,
                          lambda *a: None, node.kinematic(p), node.resistivity(p, fields))
    state, _ = scheme.step(context, scheme.start(grid, grid.zeros(jnp.float64)), 0.2, 0.1)
    assert set(state) == {"velocity", "pressure"} and asked == [(0, 0.2), (1, pytest.approx(0.25))]
    assert float(state["pressure"][0, 0, 0]) == pytest.approx(float(node.kinematic(p)))
    for t in (0.2, 0.25):
        expected = node.drive(fields, waveform_at(coefficients, t))
        assert all(bool(jnp.array_equal(a, b)) for a, b in zip(force(1, t), expected))
    held = node.forcing(fields, {"body_force": jnp.asarray([0.0, 0.0, 7.0])}, IMEXEuler(), jnp.float64)
    assert IMEXEuler().drive_instants == (1.0,)
    assert float(jnp.max(held(0, 123.0)[2])) == pytest.approx(7.0)
