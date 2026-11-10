"""The time-scheme registry: each scheme a registered tableau with its stage kind, an accuracy-bearing numerical
setting: IMEX Euler and its steady limit (F1a), and S1's explicit RK3 (F1b).

**The interface is written against two consumers**: IMEX Euler, built here, and S1's explicit RK3, F1b's.
A scheme declares its stage kind (the monolithic projected CG, or the explicit stage with its split penalty and fast
projection), the fields its state carries (``state_fields``: the velocity alone, or the velocity and the pressure), how
a run starts, the instants at which each stage reads the medium's drive, its ledger route and its numerical terms, its
CFL line (which bounds convection only) and its viscous limit, and whether it has a steady limit. **Its step and start
carry a scheme state**, a mapping keyed by ``state_fields`` (``"velocity"`` a :data:`Velocity`, any other field an
array), never the velocity alone: ``start(grid, velocity)`` gives the state a run starts from, and ``step(context,
state, time, dt)`` the next state and what the ledger reads of the step, **its stage kind's record** (``TimeScheme`` is
generic in it): :class:`StageParts` for the monolithic projected CG, one implicit stage's parts, from which the node
forms route 2's terms; for the explicit stage kind, the scheme's own record of the step's terms as its stages form them
(route 1: a quadrature over the stages, which only the stages can make; RK3's is F1b's). A start the scheme makes in
the program (RK3's consistent pressure, round 4, item 8) is its step's, from a state its start marks unmade. A scheme's
step takes a :class:`StepContext` (what the field kind gives every scheme: its grid, its operators, the viscosity and
the regions' resistivity, its drive at each stage's instant, its solve). Nothing here names the CG, the projector or a
fluid: those are the context's.

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
from mime.core.ledger import TermOrder
from mime.nodes.flow.fvm import _rk3
from mime.nodes.flow.fvm._grid import MACGrid, Velocity


#: The monolithic projected CG's residual term: its stop bounds it by the tolerance, a floor on how fast it falls.
KRYLOV_ORDER = TermOrder(tolerance=1, at_least=True)


class StageKind(StrEnum):
    """How a scheme's stages are solved, and so which record its step returns (:class:`TimeScheme`)."""

    MONOLITHIC_PROJECTED_CG = "monolithic_projected_cg"
    EXPLICIT_PROJECTED = "explicit_projected"


class StageParts(NamedTuple):
    """The monolithic projected CG's record, what the ledger reads of its one implicit stage: its right-hand side's
    parts (the drive's force, the convective term, the time term's coefficient ``c``), its solution and the primal
    solve's report. Route 2's terms are formed from it by the node; the explicit stage kind records its own."""

    force: Velocity
    convection: Velocity
    rhs: Velocity
    solved: Velocity
    shift: Any
    report: Array


#: A scheme's state, keyed by its ``state_fields``: ``"velocity"`` a :data:`Velocity`, any other field an array.
type SchemeState = dict[str, Any]


@dataclass(frozen=True)
class StepContext:
    """What a field kind gives a scheme for one step: the grid; the operator ``A_c u = c u - nu L u + Sigma u`` for a
    time coefficient ``c``; the drive, ``force(stage, time)``, the body force per unit mass on each face set that the
    scheme's stage ``stage`` reads at ``time`` (its instant, ``t_n + drive_instants[stage] dt``); whether convection is
    on; the solve (``solve(c, rhs, guess) -> (u, report)``); for a stage kind that applies them explicitly, the
    kinematic viscosity and the regions' resistivity ``nu x sum(1/K)`` on each face set; and, where the field is
    metered, the faces' fluid fraction ``1 - chi``, which such a stage kind pairs the drive with at its stages."""

    grid: MACGrid
    operator: Callable[[Any], Callable[[Velocity], Velocity]]
    force: Callable[[int, Any], Velocity]
    convection: bool
    solve: Callable[[Any, Velocity, Velocity | None], tuple[Velocity, Array]]
    viscosity: Any = None
    resistivity: Velocity | None = None
    fluid: Velocity | None = None


class TimeScheme[Record]:
    """A registered time scheme (the module's interface), generic in ``Record``, what the ledger reads of its step: its
    stage kind's record (the module says which)."""

    name: ClassVar[str]
    version: ClassVar[str]
    stage_kind: ClassVar[StageKind]
    state_fields: ClassVar[tuple[str, ...]]
    stage_times: ClassVar[tuple[float, ...]]
    ledger_route: ClassVar[int]
    #: The scheme's numerical terms in its step identity, the order its records keep them.
    ledger_terms: ClassVar[tuple[str, ...]]
    #: Each numerical term's expected order in dt and in the solve's tolerance (``mime.testing.term_orders`` checks
    #: it), or why it has none.
    ledger_orders: ClassVar[MappingProxyType[str, TermOrder]]
    #: The instant, as a fraction of the step, at which each stage reads its drive (the medium's body force): what the
    #: law that drives the medium delivers the drive at.
    drive_instants: ClassVar[tuple[float, ...]]
    #: ``max|u| dt / h`` at or below which convection is stable for the scheme along a grid axis, a declared value
    #: (the maintainer's to settle); along a vessel's axis ``n`` the line is ``cfl_line g(n)``, ``g(n) = max|n_d| /
    #: sum|n_d|`` (:meth:`cfl_limit`, the one place the line is read). ``None`` for a scheme with no line.
    cfl_line: ClassVar[float | None]
    steady: ClassVar[bool]
    #: Whether the fluid's inertia is stored (its kinetic energy is the scheme's storage).
    stores_inertia: ClassVar[bool]
    #: Whether convection may be on (the steady limit rests on linearity: Stokes only).
    takes_convection: ClassVar[bool]

    def cfl_limit(self, axis: Any) -> Any:
        """The scheme's CFL line along a vessel's axis ``axis`` (any length): ``cfl_line max|n_d| / sum|n_d|``."""
        assert self.cfl_line is not None
        magnitude = jnp.abs(jnp.asarray(axis))
        return self.cfl_line * jnp.max(magnitude) / jnp.sum(magnitude)

    def viscous_limit(self, viscosity: float, spacing: float) -> float | None:
        """The largest dt the scheme's explicit viscosity allows; ``None`` for a scheme whose viscosity is implicit."""
        return None

    def start(self, grid: MACGrid, velocity: Velocity) -> SchemeState:
        """The state a run starts from, given its initial velocity: the velocity projected once (B2)."""
        return {"velocity": grid.project(velocity)}

    def step(self, context: StepContext, state: SchemeState, time: Any, dt: Any) -> tuple[SchemeState, Record]:
        """The state one step on from ``state`` at ``time``, and what the ledger reads of the step (its record)."""
        raise NotImplementedError


class IMEXEuler(TimeScheme[StageParts]):
    name = "imex_euler"
    version = "0.1.0"
    stage_kind = StageKind.MONOLITHIC_PROJECTED_CG
    state_fields = ("velocity",)
    stage_times = (1.0,)
    #: Backward Euler's own instant, the step's end, where every implicit term is taken (measured on the low-Womersley
    #: gate against the middle: nu_fit - 1 of -5.0e-4 and -6.7e-5 at alpha 0.137 and 0.29, against -3.7e-4 and -1.2e-3).
    drive_instants = (1.0,)
    ledger_route = 2
    ledger_terms = ("backward_euler", "convection", "krylov")
    #: ``N_BE`` and ``N_conv`` are a step's change squared and convection at mixed time levels, each ``O(dt^2)`` a step;
    #: ``N_Krylov`` the solve's residual against its solution, which its stop bounds by its tolerance: it falls at least
    #: as the tolerance does, and the CG lands under it by a varying share (2.1 to 3.9 times smaller for a halving,
    #: measured on the order tests' rig in both dtypes).
    ledger_orders = MappingProxyType({"backward_euler": TermOrder(dt=2), "convection": TermOrder(dt=2),
                                      "krylov": KRYLOV_ORDER})
    cfl_line = 0.4
    steady = False
    stores_inertia = True
    takes_convection = True

    @override
    def step(self, context: StepContext, state: SchemeState, time: Any, dt: Any) -> tuple[SchemeState, StageParts]:
        grid = context.grid
        velocity: Velocity = state["velocity"]
        shift = 1.0 / dt
        force = context.force(0, time + self.drive_instants[0] * dt)
        convection = grid.convection(velocity) if context.convection else grid.zeros(velocity[0].dtype)
        rhs = (shift * velocity[0] + force[0] + convection[0], shift * velocity[1] + force[1] + convection[1],
               shift * velocity[2] + force[2] + convection[2])
        solved, report = context.solve(shift, rhs, velocity)
        return {"velocity": solved}, StageParts(force, convection, rhs, solved, shift, report)


class Steady(TimeScheme[StageParts]):
    name = "steady"
    version = "0.1.0"
    stage_kind = StageKind.MONOLITHIC_PROJECTED_CG
    state_fields = ("velocity",)
    stage_times = (1.0,)
    drive_instants = (1.0,)
    ledger_route = 2
    ledger_terms = ("backward_euler", "convection", "krylov")
    ledger_orders = MappingProxyType({
        "backward_euler": TermOrder(reason="zero: the steady limit stores no inertia"),
        "convection": TermOrder(reason="zero: the steady limit is Stokes only"),
        "krylov": KRYLOV_ORDER})
    cfl_line = None
    steady = True
    stores_inertia = False
    takes_convection = False

    @override
    def step(self, context: StepContext, state: SchemeState, time: Any, dt: Any) -> tuple[SchemeState, StageParts]:
        grid = context.grid
        velocity: Velocity = state["velocity"]
        force = context.force(0, time + self.drive_instants[0] * dt)
        zero = jnp.zeros((), dtype=velocity[0].dtype)
        solved, report = context.solve(zero, force, velocity)
        return {"velocity": solved}, StageParts(force, grid.zeros(velocity[0].dtype), force, solved, zero, report)


class RK3(TimeScheme[_rk3.ExplicitStage]):
    """S1's explicit RK3 (``mime.nodes.flow.fvm._rk3``), the scheme where convection limits the step: Wray's low-storage
    three stages, convection and viscosity explicit, the wall's penalty implicit and pointwise in each stage, the
    incremental fast projection each stage, the pressure carried as state.

    **Its lines** (one value each, the maintainer's decision): ``max|u| dt / h <= 1.7 g(n)`` and ``nu dt / h^2 <=
    0.15``, ``g(n) = max_d |n_d| / sum_d |n_d|`` along the vessel's axis ``n``. Below the linear limit of a plug flow
    along ``n`` at every viscous number up to 0.15 (the limit on the component sum is at least 1.73 there, the axis the
    worst direction; derived, and the step's Jacobian about a uniform flow matches it to 1e-14); a plug at the
    decided 2.0 and ``r = 0.15`` grows 1.06 a step, a sphere in the tube at D/h 32 was stable to 2.34 there (F1b's
    first step, measured)."""

    name = "rk3"
    version = "0.1.0"
    stage_kind = StageKind.EXPLICIT_PROJECTED
    #: The velocity, the kinematic pressure, and whether the pressure is made (the start marks it unmade).
    state_fields = ("velocity", "pressure", "pressure_ready")
    stage_times = _rk3.STAGE_TIMES
    #: Each stage reads its drive at its own start time: third order in a time-varying drive (held at the step's end
    #: instead, the alpha-6 gate's fit moved +5.4 %).
    drive_instants = _rk3.STAGE_TIMES
    ledger_route = 1
    ledger_terms = ("convection",)
    ledger_orders = MappingProxyType({"convection": TermOrder(
        reason="zero to rounding on a periodic box: the divergence form's convective power vanishes for a "
               "divergence-free field at each stage, so no order in dt or tolerance is there to measure")})
    cfl_line = 1.7
    steady = False
    stores_inertia = True
    takes_convection = True
    #: The largest viscous number ``nu dt / h^2`` its explicit viscosity takes (the pure-diffusion edge is 0.209).
    viscous_number = 0.15

    @override
    def viscous_limit(self, viscosity: float, spacing: float) -> float | None:
        return self.viscous_number * spacing ** 2 / viscosity

    @override
    def start(self, grid: MACGrid, velocity: Velocity) -> SchemeState:
        """The velocity projected once (B2), its pressure marked unmade: the first step makes the consistent pressure
        in the program, one fast solve (round 4, item 8), as it does for a state given from outside so marked."""
        dtype = velocity[0].dtype
        return {"velocity": grid.project(velocity), "pressure": jnp.zeros(grid.cells, dtype=dtype),
                "pressure_ready": jnp.zeros((), dtype=dtype)}

    @override
    def step(self, context: StepContext, state: SchemeState, time: Any,
             dt: Any) -> tuple[SchemeState, _rk3.ExplicitStage]:
        """One RK3 step on the scheme's state, and its record: the stages' quadrature of each term (route 1's)."""
        grid, viscosity, resistivity = context.grid, context.viscosity, context.resistivity
        assert viscosity is not None and resistivity is not None, "the explicit stage applies them itself"
        velocity: Velocity = state["velocity"]
        pressure = _rk3.ready_pressure(grid, velocity, state["pressure"], state["pressure_ready"], viscosity,
                                       resistivity, context.force(0, time + self.drive_instants[0] * dt),
                                       context.convection)
        ended, ended_pressure, stage = _rk3.step(grid, velocity, pressure, viscosity, resistivity, context.force,
                                                 context.convection, time, dt, fluid=context.fluid)
        return ({"velocity": ended, "pressure": ended_pressure,
                 "pressure_ready": jnp.ones((), dtype=velocity[0].dtype)}, stage)


SCHEMES: MappingProxyType[str, TimeScheme[Any]] = MappingProxyType({IMEXEuler.name: IMEXEuler(),
                                                                  Steady.name: Steady(), RK3.name: RK3()})


def scheme_named(name: str) -> TimeScheme[Any]:
    if name not in SCHEMES:
        raise ValueError(f"no time scheme {name!r} is registered: {sorted(SCHEMES)}")
    return SCHEMES[name]


__all__ = ["RK3", "SCHEMES", "IMEXEuler", "SchemeState", "StageKind", "StageParts", "Steady", "StepContext",
           "TimeScheme", "scheme_named"]
