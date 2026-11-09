"""S1's explicit RK3 on the Cartesian MAC grid: the stage of the scheme where convection limits the step (F0's B1, (c1);
a port of the spike's measured stage, not a design).

**The stage** (Wray's low-storage three-stage Runge-Kutta, order 3). Convection and viscosity are explicit; the wall's
penalty is implicit and pointwise in each stage, ``u* = r / (1 + alpha_k dt Sigma)``; each stage ends with the
incremental projection: the previous pressure's gradient in the predictor, the increment ``phi`` from the fast solve,
``u = u* - alpha_k dt G phi`` and ``p <- p + phi``. The pressure is the scheme's state; its fixed point is the discrete
Brinkman problem whatever dt is (B2). Every number comes from the step's arguments (the kinematic viscosity, the
resistivity ``nu x sum(chi / K)``, the drive at a time), so nothing derived from a parameter is made at build.

**The start** (B2; B6 and B7, decision 14; round 4, item 8). A run from rest or from a given velocity starts from a
consistent pressure, ``p = (D G)^-1 D h(u)`` with ``h`` the momentum's right-hand side without the pressure (one fast
solve), computed in the program at the run's first step (:func:`consistent_pressure`), from a state the scheme's start
marks unmade (:func:`ready_pressure`).

**The ledger** (route 1, B1 "Energy", B9 "Other tableaux"). The step records the terms S1's step records, each by
RK3's own quadrature (``b = (1/4, 0, 3/4)`` at the stage starts) or, for the penalty, at its implicit stage solutions:
the drive's work, the viscous dissipation (by summation by parts, ``nu <-L u, u>``), the penalty's, and the convective
power (which vanishes for a divergence-free field on a periodic box, up to rounding). The split penalty has no formula
term, so the step identity is not claimed: route 1 judges that each step dissipated (``W - dH >= 0``) and reports the
residual ``N = W - D_visc - D_pen - C - dH``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._grid import MACGrid, Velocity

#: Wray's coefficients: the stage weights ``gamma`` and ``zeta``, their sums ``alpha`` (each stage's projection step as a
#: fraction of dt) and the stage start times as fractions of dt.
GAMMA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
ZETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)
ALPHA = tuple(g + z for g, z in zip(GAMMA, ZETA, strict=True))
STAGE_TIMES = (0.0, 8.0 / 15.0, 2.0 / 3.0)
#: The scheme's quadrature weights at its stage starts (its Butcher tableau's ``b``), by which its works are paired.
QUADRATURE = (0.25, 0.0, 0.75)

Force = Callable[[int, Any], Velocity]
#: ``jax.lax``, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_LAX: Any = jax.lax


class ExplicitStage(NamedTuple):
    """What the ledger reads of an RK3 step, each in m^2/s^2 per unit volume times the cell volume (multiply by the
    density for joules): the drive's work, the viscous and penalty dissipations, the convective power, and the
    magnitude of the arithmetic forming them (the step's scale)."""

    drive: Array
    viscous: Array
    penalty: Array
    convection: Array
    scale: Array


def _add(a: Velocity, b: Velocity, s: Any = 1.0) -> Velocity:
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def _explicit(grid: MACGrid, velocity: Velocity, viscosity: Any, force: Velocity,
              convection: bool) -> tuple[Velocity, Velocity, Velocity]:
    """The explicit right-hand side ``h = C(u) + nu L u + f``, with its convective and viscous parts."""
    dtype = velocity[0].dtype
    conv = grid.convection(velocity) if convection else grid.zeros(dtype)
    lap = grid.laplacian(velocity)
    viscous = (viscosity * lap[0], viscosity * lap[1], viscosity * lap[2])
    return _add(_add(conv, viscous), force), conv, viscous


def consistent_pressure(grid: MACGrid, velocity: Velocity, viscosity: Any, resistivity: Velocity, force: Velocity,
                        convection: bool) -> Array:
    """The pressure (kinematic, mean-free) consistent with ``velocity``: ``(D G)^-1 D (h(u) - Sigma u)``, so that the
    velocity's rate of change is divergence-free. One fast solve."""
    rhs, _, _ = _explicit(grid, velocity, viscosity, force, convection)
    penalised = (resistivity[0] * velocity[0], resistivity[1] * velocity[1], resistivity[2] * velocity[2])
    return grid.poisson(grid.divergence(_add(rhs, penalised, -1.0)))


def ready_pressure(grid: MACGrid, velocity: Velocity, pressure: Array, ready: Array, viscosity: Any,
                   resistivity: Velocity, force: Velocity, convection: bool) -> Array:
    """The pressure a step starts from: ``pressure`` where ``ready`` marks it made, otherwise (a run's start, a state
    given from outside) the pressure consistent with ``velocity`` at the stage-0 drive ``force``, one fast solve."""
    return cast(Array, _LAX.cond(jnp.asarray(ready) > 0.5, lambda: pressure,
                                 lambda: consistent_pressure(grid, velocity, viscosity, resistivity, force, convection)))


def step(grid: MACGrid, velocity: Velocity, pressure: Array, viscosity: Any, resistivity: Velocity, force: Force,
         convection: bool, time: Any, dt: Any) -> tuple[Velocity, Array, ExplicitStage]:
    """One RK3 step: the new velocity and pressure, and what the ledger reads of the step. ``force(k, t)`` is the drive
    at stage ``k``'s time ``t``."""
    inner, size = grid.inner, grid.inner_magnitude
    dtype = velocity[0].dtype
    zero = jnp.zeros((), dtype=dtype)
    drive = viscous_d = penalty_d = conv_p = scale = zero
    previous: Velocity | None = None
    for k in range(3):
        g, z, a = GAMMA[k], ZETA[k], ALPHA[k]
        stage_dt = a * dt
        f = force(k, time + STAGE_TIMES[k] * dt)
        h, conv, visc = _explicit(grid, velocity, viscosity, f, convection)
        gp = grid.gradient(pressure)
        r = _add(_add(velocity, h, dt * g), gp, -stage_dt)
        if previous is not None:
            r = _add(r, previous, dt * z)
        held = (r[0] / (1.0 + stage_dt * resistivity[0]), r[1] / (1.0 + stage_dt * resistivity[1]),
                r[2] / (1.0 + stage_dt * resistivity[2]))
        if QUADRATURE[k] != 0.0:
            w = dt * QUADRATURE[k]
            drive = drive + w * inner(f, velocity)
            viscous_d = viscous_d - w * inner(visc, velocity)
            conv_p = conv_p + w * inner(conv, velocity)
            scale = scale + w * (size(f, velocity) + size(visc, velocity) + size(conv, velocity))
        penalised = (resistivity[0] * held[0], resistivity[1] * held[1], resistivity[2] * held[2])
        penalty_d = penalty_d + stage_dt * inner(penalised, held)
        scale = scale + stage_dt * size(penalised, held)
        phi = grid.poisson(grid.divergence(held) / stage_dt)
        correction = grid.gradient(phi)
        velocity = _add(held, correction, -stage_dt)
        pressure = pressure + phi
        previous = h
    scale = scale + size(velocity, velocity)
    return velocity, pressure, ExplicitStage(drive, viscous_d, penalty_d, conv_p, scale)


__all__ = ["ALPHA", "GAMMA", "QUADRATURE", "STAGE_TIMES", "ZETA", "ExplicitStage", "consistent_pressure",
           "ready_pressure", "step"]
