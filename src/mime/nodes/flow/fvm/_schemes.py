"""The time-scheme registry: each scheme a registered tableau with its stage kind, an accuracy-bearing numerical
setting; F1a registers IMEX Euler and its steady limit.

**The interface is written against two consumers**: IMEX Euler, built here, and S1's explicit RK3, F1b's, on paper.
A scheme declares its stage kind (the monolithic projected CG, or the explicit stage with its split penalty and fast
projection), the fields its state carries (the velocity alone, or the velocity and the pressure), how a run starts
(project the initial velocity once; RK3 also takes a consistent pressure, one fast solve, in the program), the stage
times at which the medium's drives are evaluated and its port laws called, its ledger route and its numerical terms,
its CFL line (which bounds convection only) and its viscous limit, and whether it has a steady limit. A scheme's step
takes a :class:`StepContext` (what the field kind gives every scheme: its grid, its operators, its drive, its solve) and
returns the new velocity and what the ledger reads of the step. Nothing here names the CG, the projector or a fluid:
those are the context's.

**IMEX Euler**: backward Euler for viscosity, the penalty and incompressibility in one solve on the projected
operator, forward Euler for convection when it is on; the drive at the step's end. **The steady solve** is the same
operator with the time term dropped (``c = 0``), a setting of the same tableau, not a second solver; it needs the
Stokes approximation, and a stepped graph solves it at each step's drive (exact for a constant drive from the first
step).

A script names them ``Scheme.IMEXEuler()`` and ``Scheme.Steady()`` (``mime.dynamics.flow``, the one place those public
names are written)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple, override

from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._grid import MACGrid, Velocity


class StageKind(StrEnum):
    MONOLITHIC_PROJECTED_CG = "monolithic_projected_cg"
    EXPLICIT_PROJECTED = "explicit_projected"


class StageParts(NamedTuple):
    """What the ledger reads of an implicit stage: its right-hand side's parts (the drive's force, the convective
    term, the time term's coefficient ``c``), its solution and the primal solve's report."""

    force: Velocity
    convection: Velocity
    rhs: Velocity
    solved: Velocity
    shift: Any
    report: Array


@dataclass(frozen=True)
class StepContext:
    """What a field kind gives a scheme for one step: the grid, the operator ``A_c u = c u - nu L u + Sigma u`` and its
    diagonal for a time coefficient ``c``, the drive's body force at a time, whether convection is on, and the solve
    (``solve(c, rhs, guess) -> (u, report)``)."""

    grid: MACGrid
    operator: Callable[[Any], Callable[[Velocity], Velocity]]
    force: Callable[[Any], Velocity]
    convection: bool
    solve: Callable[[Any, Velocity, Velocity | None], tuple[Velocity, Array]]


class TimeScheme:
    """A registered time scheme (the module's interface)."""

    name: ClassVar[str]
    version: ClassVar[str]
    stage_kind: ClassVar[StageKind]
    state_fields: ClassVar[tuple[str, ...]]
    stage_times: ClassVar[tuple[float, ...]]
    ledger_route: ClassVar[int]
    #: The scheme's numerical terms in its step identity, the order its records keep them.
    ledger_terms: ClassVar[tuple[str, ...]]
    #: ``max|u| dt / h`` at or below which convection is stable for the scheme; ``None`` for one with no line.
    cfl_line: ClassVar[float | None]
    steady: ClassVar[bool]
    #: Whether the fluid's inertia is stored (its kinetic energy is the scheme's storage).
    stores_inertia: ClassVar[bool]
    #: Whether convection may be on (the steady limit rests on linearity: Stokes only).
    takes_convection: ClassVar[bool]

    def viscous_limit(self, viscosity: float, spacing: float) -> float | None:
        """The largest dt the scheme's explicit viscosity allows; ``None`` for a scheme whose viscosity is implicit."""
        return None

    def start(self, grid: MACGrid, velocity: Velocity) -> Velocity:
        """The velocity a run starts from, made in the program from the initial one."""
        return grid.project(velocity)

    def step(self, context: StepContext, velocity: Velocity, time: Any, dt: Any) -> tuple[Velocity, StageParts]:
        raise NotImplementedError


class IMEXEuler(TimeScheme):
    name = "imex_euler"
    version = "0.1.0"
    stage_kind = StageKind.MONOLITHIC_PROJECTED_CG
    state_fields = ("velocity",)
    stage_times = (1.0,)
    ledger_route = 2
    ledger_terms = ("backward_euler", "convection", "krylov")
    cfl_line = 0.4
    steady = False
    stores_inertia = True
    takes_convection = True

    @override
    def step(self, context: StepContext, velocity: Velocity, time: Any, dt: Any) -> tuple[Velocity, StageParts]:
        grid = context.grid
        shift = 1.0 / dt
        force = context.force(time + self.stage_times[0] * dt)
        convection = grid.convection(velocity) if context.convection else grid.zeros(velocity[0].dtype)
        rhs = (shift * velocity[0] + force[0] + convection[0], shift * velocity[1] + force[1] + convection[1],
               shift * velocity[2] + force[2] + convection[2])
        solved, report = context.solve(shift, rhs, velocity)
        return solved, StageParts(force, convection, rhs, solved, shift, report)


class Steady(TimeScheme):
    name = "steady"
    version = "0.1.0"
    stage_kind = StageKind.MONOLITHIC_PROJECTED_CG
    state_fields = ("velocity",)
    stage_times = (1.0,)
    ledger_route = 2
    ledger_terms = ("backward_euler", "convection", "krylov")
    cfl_line = None
    steady = True
    stores_inertia = False
    takes_convection = False

    @override
    def step(self, context: StepContext, velocity: Velocity, time: Any, dt: Any) -> tuple[Velocity, StageParts]:
        grid = context.grid
        force = context.force(time + self.stage_times[0] * dt)
        zero = jnp.zeros((), dtype=velocity[0].dtype)
        solved, report = context.solve(zero, force, velocity)
        return solved, StageParts(force, grid.zeros(velocity[0].dtype), force, solved, zero, report)


SCHEMES: MappingProxyType[str, TimeScheme] = MappingProxyType({IMEXEuler.name: IMEXEuler(), Steady.name: Steady()})


def scheme_named(name: str) -> TimeScheme:
    if name not in SCHEMES:
        raise ValueError(f"no time scheme {name!r} is registered: {sorted(SCHEMES)}")
    return SCHEMES[name]


__all__ = ["SCHEMES", "IMEXEuler", "StageKind", "StageParts", "Steady", "StepContext",
           "TimeScheme", "scheme_named"]
