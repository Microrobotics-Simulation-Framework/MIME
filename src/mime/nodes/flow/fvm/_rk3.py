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
#: An explicit force read from the stage's start velocity, ``explicit(k, u_k)``.
StageForce = Callable[[int, Velocity], Velocity]
#: A force a stage takes with its pressure's weight ``alpha_k dt``, ``held_force(k)``.
HeldForce = Callable[[int], Velocity]
#: A stage's projection, ``project(k, held, alpha_k dt) -> (u_k+1, phi, held)``: the stage's solution, its pressure's
#: increment and the field its penalty was solved at (an open box's, its openings' unit fields superposed).
Project = Callable[[int, Velocity, Any], tuple[Velocity, Array, Velocity]]
#: An open box's start, ``ready(u, p, ready, f_0) -> p``: the pressure a step starts from, made consistent with the
#: velocity and the openings' pressures where ``ready`` marks it unmade.
Ready = Callable[[Velocity, Array, Array, Velocity], Array]


class StagePorts(NamedTuple):
    """An open box's openings in an RK3 step (``mime.nodes.flow.fvm._open_explicit``): its start (:data:`Ready`), the
    explicit force from each stage's start velocity, the openings' forces each stage takes with its pressure's weight,
    and each stage's projection with the openings' rows solved."""

    ready: Ready
    explicit: StageForce
    held_force: HeldForce
    project: Project


#: The instant each stage's solution is at, as a fraction of the step (its end ``c_k + alpha_k``): where a law held at
#: an opening is evaluated, backward Euler over the stage.
STAGE_ENDS = (8.0 / 15.0, 2.0 / 3.0, 1.0)
#: ``jax.lax``, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_LAX: Any = jax.lax


class ExplicitStage(NamedTuple):
    """What the ledger reads of an RK3 step, each in m^2/s^2 per unit volume times the cell volume (multiply by the
    density for joules): the drive's work, the viscous and penalty dissipations, the convective power, and the
    magnitude of the arithmetic forming them (the step's scale). Where the step is given the fluid's faces, also the
    fluid's momentum per unit density at each stage's start, ``<1 - chi, u_k>`` a component each, a row per stage (zero
    where the quadrature's weight is), and its magnitude ``<1 - chi, |u_k|>``: what a law's drive, evaluated at the
    stage's instant, is paired with."""

    drive: Array
    viscous: Array
    penalty: Array
    convection: Array
    scale: Array
    momentum: Array | None = None
    momentum_size: Array | None = None
    #: Where the step is given an explicit force of its own beside the drive (``step``'s ``explicit``: an open box's
    #: backflow term), its work by the same quadrature.
    explicit: Array | None = None


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


def explicit_rhs(grid: MACGrid, velocity: Velocity, viscosity: Any, force: Velocity, convection: bool) -> Velocity:
    """The explicit right-hand side ``h = C(u) + nu L u + f`` (a stage's)."""
    return _explicit(grid, velocity, viscosity, force, convection)[0]


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
         convection: bool, time: Any, dt: Any, *, fluid: Velocity | None = None, explicit: StageForce | None = None,
         held_force: HeldForce | None = None, project: Project | None = None) -> tuple[Velocity, Array, ExplicitStage]:
    """One RK3 step: the new velocity and pressure, and what the ledger reads of the step. ``force(k, t)`` is the drive
    at stage ``k``'s time ``t``; ``fluid``, the faces' ``1 - chi`` (weighted as the grid's inner product weighs them),
    where the step records the fluid's momentum at its stages. An open box's stages (``mime.nodes.flow.fvm.
    _open_explicit``) give ``explicit``, a force from each stage's start velocity beside the drive (the backflow term),
    ``held_force``, one each stage takes with its pressure's weight (the openings' at their predicted pressures), and
    ``project``, the stage's projection with its openings' rows solved; none given, the step is the periodic box's."""
    inner, size = grid.inner, grid.inner_magnitude
    dtype = velocity[0].dtype
    zero = jnp.zeros((), dtype=dtype)
    drive = viscous_d = penalty_d = conv_p = scale = explicit_w = zero
    momenta: list[Array] = []
    momentum_sizes: list[Array] = []
    previous: Velocity | None = None
    for k in range(3):
        g, z, a = GAMMA[k], ZETA[k], ALPHA[k]
        stage_dt = a * dt
        f = force(k, time + STAGE_TIMES[k] * dt)
        extra = explicit(k, velocity) if explicit is not None else None
        h, conv, visc = _explicit(grid, velocity, viscosity, f if extra is None else _add(f, extra), convection)
        gp = grid.gradient(pressure)
        r = _add(_add(velocity, h, dt * g), gp, -stage_dt)
        if previous is not None:
            r = _add(r, previous, dt * z)
        if held_force is not None:
            r = _add(r, held_force(k), stage_dt)
        held = (r[0] / (1.0 + stage_dt * resistivity[0]), r[1] / (1.0 + stage_dt * resistivity[1]),
                r[2] / (1.0 + stage_dt * resistivity[2]))
        if fluid is not None:
            weighted = [fluid[c] * velocity[c] for c in range(3)] if QUADRATURE[k] != 0.0 else [zero, zero, zero]
            momenta.append(jnp.stack([grid.cell_volume * jnp.sum(x) for x in weighted]))
            momentum_sizes.append(jnp.stack([grid.cell_volume * jnp.sum(jnp.abs(x)) for x in weighted]))
        if QUADRATURE[k] != 0.0:
            w = dt * QUADRATURE[k]
            drive = drive + w * inner(f, velocity)
            viscous_d = viscous_d - w * inner(visc, velocity)
            conv_p = conv_p + w * inner(conv, velocity)
            scale = scale + w * (size(f, velocity) + size(visc, velocity) + size(conv, velocity))
            if extra is not None:
                explicit_w = explicit_w + w * inner(extra, velocity)
                scale = scale + w * size(extra, velocity)
        projected = project(k, held, stage_dt) if project is not None else None
        if projected is not None:
            held = projected[2]
        penalised = (resistivity[0] * held[0], resistivity[1] * held[1], resistivity[2] * held[2])
        penalty_d = penalty_d + stage_dt * inner(penalised, held)
        scale = scale + stage_dt * size(penalised, held)
        if projected is None:
            phi = grid.poisson(grid.divergence(held) / stage_dt)
            correction = grid.gradient(phi)
            velocity = _add(held, correction, -stage_dt)
        else:
            velocity, phi = projected[0], projected[1]
        pressure = pressure + phi
        previous = h
    scale = scale + size(velocity, velocity)
    if fluid is None:
        return velocity, pressure, ExplicitStage(drive, viscous_d, penalty_d, conv_p, scale,
                                                 explicit=None if explicit is None else explicit_w)
    return velocity, pressure, ExplicitStage(drive, viscous_d, penalty_d, conv_p, scale, jnp.stack(momenta),
                                             jnp.stack(momentum_sizes), None if explicit is None else explicit_w)


__all__ = ["ALPHA", "GAMMA", "QUADRATURE", "STAGE_ENDS", "STAGE_TIMES", "ZETA", "ExplicitStage", "HeldForce", "Project",
           "Ready", "StageForce", "StagePorts", "consistent_pressure", "explicit_rhs", "ready_pressure", "step"]
