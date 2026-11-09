"""The explicit flow node: the FVM's field on a periodic box stepped by a scheme of the explicit stage kind (S1's RK3,
``mime.nodes.flow.fvm._rk3``) through the time-scheme registry's interface (``TimeScheme.start`` and ``step`` on the
scheme's state, the drive at each stage's instant from ``StepContext.force``), sharing the implicit node's grid, wall,
drive, fluxes, reports and CFL stop.

**What a step computes, in the program from the step's parameters**: the tube's signed distance, the band's indicator
and the regions' inverse permeability; the resistivity ``nu x sum(1/K)`` from the viscosity; the drive on the fluid,
the delivered waveform at each stage's time (the implicit node's ``forcing``); then the scheme's step on its state: the
velocity, the pressure, and whether the pressure is made (the scheme's start marks it unmade, as does a state given
from outside with ``pressure_ready`` zero, and the next step makes the consistent pressure, one fast solve).

**The CFL check and its stop.** As the implicit node's, at the scheme's line along the vessel's axis, ``line x g(n)``
(``TimeScheme.cfl_limit``): a NaN reads as past the line; from the first step past it the state is held, the run ends at
its chunk and is refused (``ImplicitFlowNode.held_past_the_line`` and ``stop_problem``).

**The flags the entry raises.** ``solve_flagged`` is zero on every step (the stages' projections are the grid's direct
fast solve, which cannot stop short); ``bypass_flagged`` and, under the Stokes approximation, ``reynolds_flagged`` as
the implicit node's.

**The ledger: route 1.** The step books the drive's work under the law that delivered the waveform it read (one law
delivers a waveform), the viscosity's and the wall's dissipations under their laws' names (the implicit node's
:meth:`~ImplicitFlowNode.hosted_laws`) and the scheme's terms (``TimeScheme.ledger_terms``: the convective power), each
by RK3's own quadrature; the split penalty has no formula term, so the identity is not claimed and each step is judged
to have dissipated (``mime.core.ledger.route_one_recorded``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast, override

import numpy as np
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, run_dtype
from mime.core import ledger as books
from mime.nodes.flow.fvm import _rk3
from mime.nodes.flow.fvm._band import WallFields, band_named, interior_depth, negative_resistivity
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._node import BYPASS_LINE, STOKES_REYNOLDS, ImplicitFlowNode, check_deliveries
from mime.nodes.flow.fvm._schemes import SchemeState, StageKind, StepContext, TimeScheme, scheme_named
from mime.registry import Problem

EQUATIONS = (r"\mathbf{u}^{(k)} = \operatorname{proj}\left[\frac{\mathbf{u}^{(k-1)} + \gamma_{k} h\, "
             r"\operatorname{F}^{(k-1)} + \zeta_{k} h\, \operatorname{F}^{(k-2)} - \alpha_{k} h\, \nabla p^{(k-1)}}{1 + "
             r"\alpha_{k} h\, \frac{\mu}{\rho} \operatorname{K}^{-1}}\right], \quad \operatorname{F} = -\nabla \cdot "
             r"\left( \mathbf{u} \mathbf{u} \right) + \frac{\mu}{\rho} \nabla^{2} \mathbf{u} + \left( 1 - "
             r"\operatorname{band} \right) \mathbf{g}")

_STEP = "mime.nodes.flow.fvm._explicit.ExplicitFlowNode.explicit_stepped"
_START = "mime.nodes.flow.fvm._rk3.consistent_pressure"


class _ExplicitFields(TypedDict):
    velocity_x: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    velocity_y: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    velocity_z: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    #: The kinematic pressure (p / rho), the scheme's state, and whether it is consistent with the velocity yet (zero
    #: at the run's start and in a state given from outside: the next step makes it, one fast solve).
    pressure: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("m^2/s^2")]]  # noqa: F722 - a jaxtyping shape
    pressure_ready: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    solve_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cfl: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_at: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_cfl: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    lumen_flux: ReadOnly[Annotated[Float[Array, ""], Units("m^3/s")]]  # noqa: F722 - a jaxtyping shape
    bypass_ratio: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    bypass_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    reynolds: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    reynolds_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    negative_resistivity: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class ExplicitFlowState(_ExplicitFields, closed=True):
    """The explicit flow's state without the energy ledger."""


class ExplicitParts(NamedTuple):
    """What the route-1 account reads of a step: the stage's recorded terms and the fields."""

    stage: _rk3.ExplicitStage
    fields: WallFields


def _velocity(state: Mapping[str, Array]) -> Velocity:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


def _no_monolithic_solve(shift: Any, rhs: Velocity, guess: Velocity | None) -> tuple[Velocity, Array]:
    """The context's monolithic solve, which a scheme of the explicit stage kind never calls (its projections are the
    grid's fast solve)."""
    raise TypeError("a scheme of the explicit stage kind called the monolithic projected CG's solve")


def cfl_check(velocity: Velocity, dt: Any, spacing: Any, line: Any, checked: bool,
              stopped_at: Array, stopped_cfl: Array, steps: Array) -> tuple[Array, Array, Array]:
    """``max|u| dt / h`` of the velocity the step's convection reads, and the stop's record: the first step past
    ``line`` (a NaN reads as past it) and the CFL it read, kept once set."""
    speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in velocity]))
    cfl = speed * dt / spacing
    past = ~(cfl <= line) if checked else jnp.asarray(False)
    before = stopped_at >= 0
    at = jnp.where(before, stopped_at, jnp.where(past, steps + 1, -1)).astype(jnp.int32)
    held = jnp.where(before, stopped_cfl, jnp.where(past, cfl, 0.0)).astype(velocity[0].dtype)
    return cfl.astype(velocity[0].dtype), at, held


class ExplicitFlowNode(ImplicitFlowNode):
    """The flow field of a medium on a periodic box holding a straight tube, stepped by S1's RK3 (the module states
    the step)."""

    LEDGER_TERMS = ("convection",)

    meta = NodeMeta(
        algorithm_id="MIME-NODE-124",
        description="A medium's flow on a periodic box holding a straight tube: the Cartesian MAC grid, the wall a smooth "
                    "Brinkman band, stepped by S1's explicit RK3 with the incremental fast projection.",
        governing_equations=EQUATIONS,
        assumptions=(
            "Incompressible Newtonian fluid, its viscosity and density uniform",
            "The wall is a first-order immersed wall: a smooth band of registered width and depth, its effective "
            "position shifted from the signed distance's zero by the band's planar calibration, constant in cells",
            "Convection and viscosity explicit (Wray's three stages), the wall's penalty implicit and pointwise in "
            "each stage, the incremental projection each stage, the pressure carried as state",
            "The run starts from a pressure consistent with its velocity, computed in the program",
            "The drive is read at each stage's time from the waveform its law delivers",
            "The time of a step is its count times the timestep, never a sum of timesteps"),
        limitations=(
            "A periodic box only: no openings, the flow driven by a body force on the fluid",
            "Third order in time without a wall; with a wall its flux is first order in time (the split penalty) and "
            "its pressure first order; second order in space in the fluid, first order at the wall",
            "Stable only within its CFL line along the vessel's axis and its viscous number, which the build checks; "
            "the node checks the CFL number each step, its state held from the first step past and the run refused",
            "No step identity is claimed: each step is judged to have dissipated (the ledger's first route)",
        ),
        references=(Reference("HarlowWelch1965"), Reference("Angot1999"), Reference("SpalartMoserRogers1991")),
    )
    symbols = (
        Symbol(r"\mathbf{u}", SymbolKind.STATE, "velocity_x", "m/s", _STEP),
        Symbol("p", SymbolKind.STATE, "pressure", "m^2/s^2", _START),
        Symbol(r"\mu", SymbolKind.PARAMETER, "viscosity", "Pa s", _STEP),
        Symbol(r"\rho", SymbolKind.PARAMETER, "density", "kg/m^3", _STEP),
        Symbol(r"\mathbf{g}", SymbolKind.INPUT, "body_force", "m/s^2", _STEP),
        Symbol(r"\gamma_{k}", SymbolKind.CONSTANT, "mime.nodes.flow.fvm._rk3.GAMMA", "1", _STEP),
        Symbol(r"\zeta_{k}", SymbolKind.CONSTANT, "mime.nodes.flow.fvm._rk3.ZETA", "1", _STEP),
        Symbol(r"\alpha_{k}", SymbolKind.CONSTANT, "mime.nodes.flow.fvm._rk3.ALPHA", "1", _STEP),
    )

    def __init__(self, name: str, timestep: float, *, cells_x: int, cells_y: int, cells_z: int, spacing: float,
                 viscosity: float, density: float, tube_centre: Sequence[float], tube_axis: Sequence[float],
                 tube_radius: float, scheme: str = "rk3", convection: bool = True, band: str = "smooth_2h_0.3h",
                 interior: str = "0.1h", periodic_images: str = "own", wall_name: str = "",
                 initial_velocity_x: Any = None, initial_velocity_y: Any = None, initial_velocity_z: Any = None,
                 ledger: Sequence[str] = ()) -> None:
        chosen = scheme_named(scheme)
        if chosen.stage_kind is not StageKind.EXPLICIT_PROJECTED or chosen.ledger_terms != self.LEDGER_TERMS:
            raise ValueError(f"the {scheme} scheme is of the {chosen.stage_kind} stage kind and books "
                             f"{chosen.ledger_terms}: this node steps the explicit stage kind's schemes, whose terms "
                             f"are {self.LEDGER_TERMS}")
        band_named(band)
        interior_depth(interior)
        if periodic_images not in ("own", "transverse"):
            raise ValueError(f"periodic_images is 'own' (an aligned tube) or 'transverse', not {periodic_images!r}")
        initial: dict[str, Any] = {}
        for key, value in (("initial_velocity_x", initial_velocity_x), ("initial_velocity_y", initial_velocity_y),
                           ("initial_velocity_z", initial_velocity_z)):
            if value is not None:
                initial[key] = np.asarray(value)
        MimeNode.__init__(self, name, timestep, cells_x=int(cells_x), cells_y=int(cells_y), cells_z=int(cells_z),
                          spacing=float(spacing), viscosity=float(viscosity), density=float(density),
                          tube_centre=tuple(float(c) for c in tube_centre),
                          tube_axis=tuple(float(c) for c in tube_axis), tube_radius=float(tube_radius),
                          scheme=str(scheme), convection=bool(convection), band=str(band), interior=str(interior),
                          periodic_images=str(periodic_images), **initial,
                          **({"wall_name": str(wall_name)} if wall_name else {}),
                          **({"ledger": tuple(ledger)} if ledger else {}))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        """The implicit node's, without the monolithic solve's tolerance and report (this stage kind has neither)."""
        specs = super().param_specs()
        del specs["tolerance"], specs["solve_report"]
        return specs

    def line(self, p: Mapping[str, Any]) -> Array:
        """The scheme's CFL line along the vessel's axis, ``line x g(n)``, made in the step from the axis."""
        return self.scheme.cfl_limit(as_run_array(p["tube_axis"]))

    @staticmethod
    def scheme_state(state: Mapping[str, Array]) -> SchemeState:
        """The scheme's state (keyed by its ``state_fields``) from the node's."""
        return {"velocity": _velocity(state), "pressure": state["pressure"], "pressure_ready": state["pressure_ready"]}

    @staticmethod
    def held_of(scheme_state: SchemeState) -> dict[str, Array]:
        """The node's state fields of a scheme's state: the velocity as a column per MAC component."""
        velocity: Velocity = scheme_state["velocity"]
        return {"velocity_x": velocity[0], "velocity_y": velocity[1], "velocity_z": velocity[2],
                "pressure": scheme_state["pressure"], "pressure_ready": scheme_state["pressure_ready"]}

    @override
    def initial_state(self) -> NodeFields:
        c = self.constants()
        grid = MACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), c["spacing"])
        dtype = run_dtype()
        zero = jnp.zeros(grid.cells, dtype=dtype)
        given = tuple(jnp.asarray(c[key], dtype=dtype) if key in c else zero
                      for key in ("initial_velocity_x", "initial_velocity_y", "initial_velocity_z"))
        start = self.held_of(self.scheme.start(grid, (given[0], given[1], given[2])))
        z = as_run_array(0.0)
        fields = node_fields(ExplicitFlowState(
            velocity_x=start["velocity_x"], velocity_y=start["velocity_y"], velocity_z=start["velocity_z"],
            pressure=start["pressure"], pressure_ready=start["pressure_ready"], steps=jnp.asarray(0, dtype=jnp.int32),
            solve_flagged=z, cfl=z, stopped_at=jnp.asarray(-1, dtype=jnp.int32), stopped_cfl=z, lumen_flux=z,
            bypass_ratio=z, bypass_flagged=z, reynolds=z, reynolds_flagged=z, negative_resistivity=z))
        if not self.metered:
            return fields
        return books.route_one_records(books.with_records(self, fields))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        if self.metered:
            ended = books.metered_route_one_step(self, self.own_inputs(), p, state, boundary_inputs, dt,
                                                 self.explicit_stepped)
        else:
            ended = self.explicit_stepped(p, state, boundary_inputs, dt)[0]
        return self.held_past_the_line(state, ended)

    def explicit_stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                         dt: float) -> tuple[dict[str, Array], ExplicitParts]:
        c = self.constants()
        scheme: TimeScheme[_rk3.ExplicitStage] = self.scheme
        grid = self.grid(p)
        dtype = run_dtype()
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        convection = bool(c["convection"])
        context = StepContext(grid, self.operator_of(p, grid, sigma), self.forcing(fields, boundary_inputs, scheme, dtype),
                              convection, _no_monolithic_solve, self.kinematic(p), sigma)
        steps = jnp.asarray(state["steps"])
        time = steps.astype(dtype) * dt
        cfl, stopped_at, stopped_cfl = cfl_check(_velocity(state), dt, grid.spacing, self.line(p), convection,
                                                 jnp.asarray(state["stopped_at"]), jnp.asarray(state["stopped_cfl"]),
                                                 steps)
        ended_state, stage = scheme.step(context, self.scheme_state(state), time, dt)
        axis = as_run_array(p["tube_axis"])
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, ended_state["velocity"])
        ratio = self.ratio(solid, lumen, lumen_scale)
        reynolds = 2.0 * jnp.abs(lumen) / (jnp.pi * as_run_array(p["tube_radius"]) * self.kinematic(p))
        ended: dict[str, Array] = dict(
            **self.held_of(ended_state), steps=steps + 1, solve_flagged=jnp.zeros((), dtype=dtype), cfl=cfl,
            stopped_at=stopped_at, stopped_cfl=stopped_cfl, lumen_flux=lumen, bypass_ratio=ratio,
            bypass_flagged=(jnp.abs(ratio) > BYPASS_LINE).astype(dtype), reynolds=reynolds.astype(dtype),
            reynolds_flagged=((reynolds > STOKES_REYNOLDS) & (not convection)).astype(dtype),
            negative_resistivity=negative_resistivity(fields).astype(dtype))
        return node_fields(ExplicitFlowState(**ended)), ExplicitParts(stage, fields)

    @override
    def identity_terms(self, p: Mapping[str, Any], held: Sequence[books.LedgerDelivery], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: Any,
                       dt: float) -> books.StepAccount:
        """The step's account on route 1: the drive's ``+W`` under the law that delivered what the scheme read, the
        hosted laws' work under their names (the viscosity's and the wall's ``-D``), each by RK3's quadrature, the
        scheme's terms (``ledger_terms``: the convective power, ``N_conv = -C``), the storage at both ends. The
        residual, the split penalty's numerical dissipation, is not claimed to vanish. The quadrature records the
        drive's work summed, so one law delivers what the scheme reads (a waveform is not summed); fed into the plain
        input as well, the work is unattributed."""
        stage = parts.stage
        rho = as_run_array(p["density"])
        dtype = jnp.asarray(stage.drive).dtype
        check_deliveries(held)
        rows = books.laws(held)
        read = self.drive_read()
        driving = [d for d in held if d.kind == books.LOAD and d.field == read]
        if len(driving) > 1:
            raise ValueError(f"{len(driving)} laws deliver the flow's {read}; the explicit stage records the drive's work "
                             f"summed, so one law delivers it")
        work_of = {"viscous": -rho * stage.viscous, "penalty": -rho * stage.penalty}
        numerical = {"convection": -rho * stage.convection}
        work = jnp.zeros(len(rows), dtype=dtype)
        hosted = jnp.zeros((len(rows), 2), dtype=dtype)
        drive = rho * stage.drive
        unattributed = jnp.zeros((), dtype=dtype)
        if driving and read not in boundary_inputs:
            work = work.at[rows.index(driving[0].law)].add(drive)
        else:
            unattributed = unattributed + drive
        named: set[str] = set()
        for delivery in held:
            if delivery.kind == books.HOSTED:
                work = work.at[rows.index(delivery.law)].add(work_of[delivery.field])
                named.add(delivery.field)
        unattributed = cast(Array, unattributed + sum((value for field, value in work_of.items() if field not in named),
                                                      start=jnp.zeros((), dtype=dtype)))
        terms = jnp.stack([numerical[name] for name in self.scheme.ledger_terms])
        return books.StepAccount(work=work, hosted=hosted, unattributed=unattributed, terms=terms,
                                 storage_start=self.storage(p, state), storage_end=self.storage(p, ended),
                                 scale=rho * stage.scale, operators={})

    @override
    def run_problem(self, end: int) -> Problem | None:
        """None: the explicit stage keeps no per-step rows of derivative-solve reports."""
        del end
        return None


__all__ = ["ExplicitFlowNode", "ExplicitFlowState", "ExplicitParts", "cfl_check"]
