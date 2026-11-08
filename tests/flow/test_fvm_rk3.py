"""RK3's gates (F1b's gates 1, 4 and 9, and its step's own checks): S1's manufactured solution with convection and the
pressure gradient, judged against the exact solution in space (order 2, dt tied to h^2) and by self-convergence in
time (order 3 on the velocity, the pressure at first order), each with its planted controls; the consistent-pressure
start; the stability line along a plug flow, by the step's own Jacobian; the step equal to S1's measured stage.

**Detection rates** (F1b's first step, five drawn manufactured fields, S1's code): the space sweep catches a convection
sign flip, one component's flux x 1.05 and a one-cell index shift on 5 of 5 draws (orders -0.02, 1.26 to 1.38, 1.09
to 1.11 against the clean 1.97 to 2.00) and is blind to the tableau's plants; the time sweep catches a Wray
coefficient with its row sum broken, one with alpha kept and the forcing read at the step's start on 5 of 5 (orders
1.10 to 1.12 against 3.00). Float64 only: the time ladder ends near float32's resolution (S1: 1.1e-7)."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.flow.fvm import RK3, MACGrid, _rk3

NU_SPACE, NU_TIME, T_END = 0.1, 0.01, 1.0


def _u1(x):
    return jnp.stack([jnp.sin(x[0]) * jnp.cos(x[1]) * jnp.cos(x[2]), jnp.cos(x[0]) * jnp.sin(x[1]) * jnp.cos(x[2]),
                      -2 * jnp.cos(x[0]) * jnp.cos(x[1]) * jnp.sin(x[2])])


def _u2(x):
    return jnp.stack([jnp.cos(x[1] - x[2]), jnp.sin(x[2] + x[0]), jnp.cos(x[0] + 2 * x[1])])


def _pressure(x):
    return jnp.sin(x[0]) * jnp.sin(x[1]) * jnp.cos(x[2])


def _velocity(x, t):
    return jnp.cos(t) * _u1(x) + 0.5 * (1.0 + 0.5 * jnp.sin(2 * t)) * _u2(x)


def _forcing(nu):
    def forcing(x, t):
        u = _velocity(x, t)
        dudt = jax.jacfwd(_velocity, argnums=1)(x, t)
        gradient = jax.jacfwd(_velocity, argnums=0)(x, t)
        laplacian = jnp.trace(jax.hessian(_velocity, argnums=0)(x, t), axis1=1, axis2=2)
        return dudt + gradient @ u - nu * laplacian + jnp.cos(t) * jax.grad(_pressure)(x)
    return forcing


class Planted(MACGrid):
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


#: The tableau plants: a Wray coefficient with the row sum broken, one with alpha kept (consistent, order broken), and
#: the forcing read at the step's start in every stage.
TABLEAU_PLANTS = {
    "row_sum": dict(GAMMA=(_rk3.GAMMA[0], 1.05 * _rk3.GAMMA[1], _rk3.GAMMA[2]),
                    ALPHA=(_rk3.ALPHA[0], 1.05 * _rk3.GAMMA[1] + _rk3.ZETA[1], _rk3.ALPHA[2])),
    "alpha_kept": dict(GAMMA=(_rk3.GAMMA[0], 1.05 * _rk3.GAMMA[1], _rk3.GAMMA[2]),
                       ZETA=(_rk3.ZETA[0], _rk3.ZETA[1] - 0.05 * _rk3.GAMMA[1], _rk3.ZETA[2])),
    "stage_time": dict(STAGE_TIMES=(0.0, 0.0, 0.0)),
}


def _sampler(grid, function):
    points = [grid.face_points(c, "float64").reshape(-1, 3) for c in range(3)]
    return jax.jit(lambda t: tuple(jax.vmap(lambda x: function(x, t)[c])(points[c]).reshape(grid.cells)
                                   for c in range(3)))


def _run(n: int, nu: float, steps: int, plant: str = "none", start: str = "consistent"):
    """The velocity and pressure at T_END on an N grid in ``steps`` steps (float64), the drive the manufactured
    forcing at each stage's time."""
    grid = type(f"Planted_{plant}", (Planted,), {"plant": plant})((n, n, n), 2.0 * math.pi / n)
    dt = T_END / steps
    velocity_at = _sampler(grid, _velocity)
    force_at = _sampler(grid, _forcing(nu))
    zero = grid.zeros(jnp.float64)

    def force(stage, time):
        del stage
        return force_at(time)

    u0 = grid.project(velocity_at(jnp.asarray(0.0)))
    if start == "consistent":
        p0 = _rk3.consistent_pressure(grid, u0, nu, zero, force(0, jnp.asarray(0.0)), True)
    else:
        p0 = jnp.zeros(grid.cells)

    def body(carry, k):
        u, p = carry
        u, p, _ = _rk3.step(grid, u, p, nu, zero, force, True, k * dt, dt)
        return (u, p), None

    (u, p), _ = jax.jit(lambda c: jax.lax.scan(body, c, jnp.arange(steps, dtype=jnp.float64)))((u0, p0))
    return grid, u, p, velocity_at


def _relative(a, b):
    return math.sqrt(sum(float(jnp.sum((x - y) ** 2)) for x, y in zip(a, b, strict=True))
                     / sum(float(jnp.sum(y ** 2)) for y in b))


@functools.cache
def _space_order(plant: str = "none") -> float:
    errors = []
    for n in (16, 32):
        h = 2.0 * math.pi / n
        steps = math.ceil(T_END / (0.05 * h * h / NU_SPACE))
        _, u, _, exact = _run(n, NU_SPACE, steps, plant)
        errors.append(_relative(u, exact(jnp.asarray(T_END))))
    return math.log2(errors[0] / errors[1])


def _mean_free(p):
    return p - jnp.mean(p)


@functools.cache
def _time_orders(plant: str = "none") -> tuple[float, float]:
    """The velocity's and the pressure's observed orders between dt 0.02 and 0.01 against dt / 8, N 16."""
    held = {name: getattr(_rk3, name) for name in TABLEAU_PLANTS.get(plant, {})}
    try:
        for name, value in TABLEAU_PLANTS.get(plant, {}).items():
            setattr(_rk3, name, value)
        _, ur, pr, _ = _run(16, NU_TIME, 800)
        found = [_run(16, NU_TIME, steps)[1:3] for steps in (50, 100)]
    finally:
        for name, value in held.items():
            setattr(_rk3, name, value)
    ev = [_relative(u, ur) for u, _ in found]
    ep = [_relative((_mean_free(p),), (_mean_free(pr),)) for _, p in found]
    return math.log2(ev[0] / ev[1]), math.log2(ep[0] / ep[1])


#: The gate's lines on the observed orders: space on N 16 to 32, time on dt 0.02 to 0.01 (clean: 1.97 to 2.00 and
#: 3.00; the pressure 1.09 to 1.12, judged at first order).
SPACE_ORDER, TIME_ORDER, PRESSURE_TIME_ORDER = 1.8, 2.5, 0.9


def test_the_manufactured_solution_converges_at_second_order_in_space(float64):
    assert _space_order() >= SPACE_ORDER


@pytest.mark.parametrize("plant", ["sign", "scale", "shift"])
def test_control_a_planted_convective_term_fails_the_space_sweep(float64, plant):
    assert _space_order(plant) < SPACE_ORDER


def test_the_manufactured_solution_is_third_order_in_time_and_its_pressure_first(float64):
    velocity, pressure = _time_orders()
    assert velocity >= TIME_ORDER
    assert pressure >= PRESSURE_TIME_ORDER


@pytest.mark.parametrize("plant", sorted(TABLEAU_PLANTS))
def test_control_a_planted_tableau_fails_the_time_sweep(float64, plant):
    assert _time_orders(plant)[0] < TIME_ORDER


# -- the start ------------------------------------------------------------------------------------------------------

def _start_error(n: int, plant: str = "none") -> float:
    """The start's pressure against the manufactured ``p(0)``, both mean-free, relative in L2."""
    grid = MACGrid((n, n, n), 2.0 * math.pi / n)
    u0 = grid.project(_sampler(grid, _velocity)(jnp.asarray(0.0)))
    force = _sampler(grid, _forcing(NU_SPACE))(jnp.asarray(0.0))
    if plant == "zero":
        p0 = jnp.zeros(grid.cells)
    else:
        p0 = _rk3.consistent_pressure(grid, u0, NU_SPACE, grid.zeros(jnp.float64), force, True)
    exact = jax.vmap(_pressure)(grid.cell_points("float64").reshape(-1, 3)).reshape(grid.cells)
    return _relative((_mean_free(p0),), (_mean_free(exact),))


def test_the_start_takes_the_consistent_pressure_at_second_order_in_space(float64):
    errors = [_start_error(n) for n in (16, 32)]
    assert math.log2(errors[0] / errors[1]) >= 1.8
    assert errors[1] < 0.05


def test_control_a_start_from_zero_pressure_is_far_from_the_manufactured_one(float64):
    assert _start_error(32, "zero") > 0.5


# -- the line, by the step's Jacobian about a plug flow --------------------------------------------------------------

def _spectral_radius(cfl: float, viscous_number: float, direction=(1.0, 0.0, 0.0), n: int = 4) -> float:
    """The largest amplification of one RK3 step linearised about a uniform flow of face CFL ``cfl`` along
    ``direction`` (its largest component that CFL), at ``nu dt / h^2 = viscous_number``, on an n^3 periodic box (n 4
    holds the modes at a quarter and a half of the grid's wavenumber, where the plug's limit is set)."""
    grid = MACGrid((n, n, n), 1.0)
    dt = viscous_number
    d = np.asarray(direction) / np.max(np.abs(direction))
    flow = [float(cfl * c / dt) for c in d]
    zero = grid.zeros(jnp.float64)
    count = n ** 3

    def stepped(x):
        u = tuple(x[i * count:(i + 1) * count].reshape(grid.cells) for i in range(3))
        p = x[3 * count:].reshape(grid.cells)
        u, p, _ = _rk3.step(grid, u, p, 1.0, zero, lambda k, t: zero, True, 0.0, dt)
        return jnp.concatenate([c.ravel() for c in u] + [p.ravel()])

    base = jnp.concatenate([jnp.full(count, f) for f in flow] + [jnp.zeros(count)])
    return float(np.abs(np.linalg.eigvals(np.asarray(jax.jacfwd(stepped)(base)))).max())


@pytest.mark.parametrize("viscous_number", [0.01, 0.15])
def test_rk3s_line_holds_for_a_plug_flow_along_an_axis_and_a_lattice_direction(float64, viscous_number):
    """At the line, ``1.7 g(n)``, every mode of a plug flow along n is bounded: along an axis and along (1, 0, 1), at
    a small viscous number and at the limit's 0.15."""
    for direction in ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)):
        line = RK3.cfl_line * float(np.max(np.abs(direction)) / np.sum(np.abs(direction)))
        assert _spectral_radius(line, viscous_number, direction) <= 1.0 + 1e-12


def test_control_the_decided_two_is_unstable_for_a_plug_at_the_viscous_limit(float64):
    """The measurement 1.7 rests on: at face CFL 2.0 and ``nu dt / h^2 = 0.15`` a plug along an axis grows 1.066 a
    step (the derived limit there is 1.89)."""
    assert _spectral_radius(2.0, 0.15) == pytest.approx(1.0657629635357, rel=1e-9)
