"""The explicit flow node: the FVM's field on a periodic box stepped by a scheme of the explicit stage kind (S1's RK3,
``mime.nodes.flow.fvm._rk3``), sharing the implicit node's grid, wall, drive, fluxes and reports.

**What a step computes, in the program from the step's parameters**: the tube's signed distance, the band's indicator
and the regions' inverse permeability; the resistivity ``nu x sum(1/K)`` from the viscosity; the drive on the fluid; on
the run's first step (or the first after a state given from outside, whose ``pressure_ready`` is zero) the consistent
pressure, one fast solve; then the three stages.

**The CFL check.** As the implicit node's, at the scheme's line along the vessel's axis, ``line x g(n)``
(``TimeScheme.cfl_limit``): a NaN reads as past the line. The check selects nothing in the
step; the run is refused after it from the first step past.

**The ledger: route 1.** The step books the drive's work, the viscosity's and the wall's dissipations under their laws'
names and the convective power as the scheme's one term, each by RK3's own quadrature; the split penalty has no formula
term, so the identity is not claimed and each step is judged to have dissipated (``mime.core.ledger.
route_one_recorded``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast, override

import numpy as np
import jax
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, \
    run_dtype
from mime.core import ledger as books
from mime.nodes.flow.fvm import _rk3
from mime.nodes.flow.fvm._band import WallFields, band_named, interior_depth, negative_resistivity
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._node import ImplicitFlowNode
from mime.nodes.flow._waveform import WAVEFORM_SIZE, waveform_at
from mime.nodes.flow.fvm._schemes import StageKind, scheme_named
from mime.registry import Problem

EQUATIONS = (r"\mathbf{u}^{(k)} = \operatorname{proj}\left[\frac{\mathbf{u}^{(k-1)} + \gamma_{k} h\, "
             r"\operatorname{F}^{(k-1)} + \zeta_{k} h\, \operatorname{F}^{(k-2)} - \alpha_{k} h\, \nabla p^{(k-1)}}{1 + "
             r"\alpha_{k} h\, \frac{\mu}{\rho} \operatorname{K}^{-1}}\right], \quad \operatorname{F} = -\nabla \cdot "
             r"\left( \mathbf{u} \mathbf{u} \right) + \frac{\mu}{\rho} \nabla^{2} \mathbf{u} + \left( 1 - "
             r"\operatorname{band} \right) \mathbf{g}")

_STEP = "mime.nodes.flow.fvm._explicit.ExplicitFlowNode.explicit_stepped"
#: ``jax.lax``, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_LAX: Any = jax.lax
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
    cfl: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_at: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_cfl: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    lumen_flux: ReadOnly[Annotated[Float[Array, ""], Units("m^3/s")]]  # noqa: F722 - a jaxtyping shape
    bypass_ratio: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    negative_resistivity: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class ExplicitFlowState(_ExplicitFields, closed=True):
    """The explicit flow's state without the energy ledger."""


class ExplicitFlowInputs(TypedDict, closed=True):
    body_force: NotRequired[ReadOnly[Annotated[Float[Array, "3"], Units("m/s^2")]]]  # noqa: F722 - a jaxtyping shape
    waveform: NotRequired[ReadOnly[Annotated[Float[Array, "7"], Units("1")]]]  # noqa: F722 - a jaxtyping shape


class ExplicitParts(NamedTuple):
    """What the route-1 account reads of a step: the stage's recorded terms and the fields."""

    stage: _rk3.ExplicitStage
    fields: WallFields


def _velocity(state: Mapping[str, Array]) -> Velocity:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


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
            "Stable only within its CFL line along the vessel's axis and its viscous number, which the build checks "
            "and the node checks each step, the run refused past",
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
                 interior: str = "0.1h", periodic_images: str = "own", drive: str = "waveform",
                 initial_velocity_x: Any = None, initial_velocity_y: Any = None, initial_velocity_z: Any = None,
                 ledger: Sequence[str] = ()) -> None:
        chosen = scheme_named(scheme)
        if drive not in ("waveform", "held"):
            raise ValueError(f"drive is 'waveform' (read at each stage's time) or 'held' (a value a step), not {drive!r}")
        if chosen.stage_kind is not StageKind.EXPLICIT_PROJECTED:
            raise ValueError(f"the {scheme} scheme is of the {chosen.stage_kind} stage kind: the implicit node steps it")
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
                          periodic_images=str(periodic_images), drive=str(drive), **initial,
                          **({"ledger": tuple(ledger)} if ledger else {}))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        fixed = "a structural setting: the box's cells were chosen with it"
        return {**MimeNode.param_specs(self),
                "spacing": ParamSpec(trainable=False, units="m", description=fixed),
                "viscosity": ParamSpec(units="Pa s", bounds=(0.0, None),
                                       description="the fluid's dynamic viscosity, mu; the step uses mu / rho"),
                "density": ParamSpec(units="kg/m^3", bounds=(0.0, None)),
                "tube_centre": ParamSpec(units="m"), "tube_axis": ParamSpec(units="1"),
                "tube_radius": ParamSpec(units="m", bounds=(0.0, None))}

    @override
    def boundary_input_spec(self) -> dict[str, Any]:
        """The drive: with ``drive="waveform"`` (the build's), the waveform the law bound to the medium delivers, read
        at each stage's time; with ``drive="held"``, a value a step (``body_force``) held over the step (first order in
        a time-varying drive at the step's end: the alpha-6 gate's fit moved +5.4 %)."""
        return {**super().boundary_input_spec(),
                "waveform": node_input("1", (WAVEFORM_SIZE,),
                                       description="the drive's waveform: along (3), steady and amplitude (m/s^2), "
                                                   "frequency (Hz), phase (rad)")}

    def line(self, p: Mapping[str, Any]) -> Array:
        """The scheme's CFL line along the vessel's axis, ``line x g(n)``, made in the step from the axis."""
        return self.scheme.cfl_limit(as_run_array(p["tube_axis"]))

    @override
    def initial_state(self) -> NodeFields:
        c = self.constants()
        grid = MACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), c["spacing"])
        dtype = run_dtype()
        zero = jnp.zeros(grid.cells, dtype=dtype)
        given = tuple(jnp.asarray(c[key], dtype=dtype) if key in c else zero
                      for key in ("initial_velocity_x", "initial_velocity_y", "initial_velocity_z"))
        start = self.scheme.start(grid, (given[0], given[1], given[2]))
        z = as_run_array(0.0)
        fields = node_fields(ExplicitFlowState(
            velocity_x=start[0], velocity_y=start[1], velocity_z=start[2], pressure=zero, pressure_ready=z,
            steps=jnp.asarray(0, dtype=jnp.int32), cfl=z, stopped_at=jnp.asarray(-1, dtype=jnp.int32),
            stopped_cfl=z, lumen_flux=z, bypass_ratio=z, negative_resistivity=z))
        if not self.metered:
            return fields
        return books.route_one_records(books.with_records(self, fields))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        if self.metered:
            return books.metered_route_one_step(self, self.boundary_input_spec(), p, state, boundary_inputs, dt,
                                                self.explicit_stepped)
        return self.explicit_stepped(p, state, boundary_inputs, dt)[0]

    def explicit_stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                         dt: float) -> tuple[dict[str, Array], ExplicitParts]:
        c = self.constants()
        grid = self.grid(p)
        dtype = run_dtype()
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        nu = self.kinematic(p)
        convection = bool(c["convection"])
        if c["drive"] == "waveform":
            coefficients = jnp.asarray(boundary_inputs.get("waveform", jnp.zeros(WAVEFORM_SIZE, dtype=dtype)),
                                       dtype=dtype)

            def force(stage: int, time: Any) -> Velocity:
                del stage
                return self.drive(fields, waveform_at(coefficients, time))
        else:
            delivered = jnp.asarray(boundary_inputs.get("body_force", jnp.zeros(3, dtype=dtype)), dtype=dtype)

            def force(stage: int, time: Any) -> Velocity:
                del stage, time
                return self.drive(fields, delivered)

        steps = jnp.asarray(state["steps"])
        time = steps.astype(dtype) * dt
        velocity = _velocity(state)
        held_pressure = jnp.asarray(state["pressure"])
        pressure = cast(Array, _LAX.cond(
            jnp.asarray(state["pressure_ready"]) > 0.5, lambda: held_pressure,
            lambda: _rk3.consistent_pressure(grid, velocity, nu, sigma, force(0, time), convection)))
        cfl, stopped_at, stopped_cfl = cfl_check(velocity, dt, grid.spacing, self.line(p), convection,
                                                 jnp.asarray(state["stopped_at"]), jnp.asarray(state["stopped_cfl"]),
                                                 steps)
        ended_velocity, ended_pressure, stage = _rk3.step(grid, velocity, pressure, nu, sigma, force, convection,
                                                          time, dt)
        axis = as_run_array(p["tube_axis"])
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, ended_velocity)
        ratio = self.ratio(solid, lumen, lumen_scale)
        ended: dict[str, Array] = dict(
            velocity_x=ended_velocity[0], velocity_y=ended_velocity[1], velocity_z=ended_velocity[2],
            pressure=ended_pressure, pressure_ready=jnp.ones((), dtype=dtype), steps=steps + 1, cfl=cfl,
            stopped_at=stopped_at, stopped_cfl=stopped_cfl, lumen_flux=lumen, bypass_ratio=ratio,
            negative_resistivity=negative_resistivity(fields).astype(dtype))
        return node_fields(ExplicitFlowState(**ended)), ExplicitParts(stage, fields)

    @override
    def identity_terms(self, p: Mapping[str, Any], held: Sequence[books.LedgerDelivery], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: Any,
                       dt: float) -> books.StepAccount:
        """The step's account on route 1: the hosted laws' work under their names (the drive's ``+W``, the viscosity's
        and the wall's ``-D``, each by RK3's quadrature), the convective power as the scheme's one term (``N_conv =
        -C``), the storage at both ends. The residual, the split penalty's numerical dissipation, is not claimed to
        vanish."""
        stage = parts.stage
        rho = as_run_array(p["density"])
        dtype = jnp.asarray(stage.drive).dtype
        work_of = {"body_force": rho * stage.drive, "viscous": -rho * stage.viscous, "penalty": -rho * stage.penalty}
        rows = books.laws(held)
        work = jnp.zeros(len(rows), dtype=dtype)
        hosted = jnp.zeros((len(rows), 2), dtype=dtype)
        named: set[str] = set()
        for delivery in held:
            if delivery.kind != books.HOSTED or delivery.field not in work_of:
                raise ValueError(f"the flow node hosts {list(work_of)}: it takes no delivery {delivery}")
            work = work.at[rows.index(delivery.law)].add(work_of[delivery.field])
            named.add(delivery.field)
        unattributed = cast(Array, sum((value for field, value in work_of.items() if field not in named),
                                       start=jnp.zeros((), dtype=dtype)))
        terms = jnp.stack([-rho * stage.convection])
        return books.StepAccount(work=work, hosted=hosted, unattributed=unattributed, terms=terms,
                                 storage_start=self.storage(p, state), storage_end=self.storage(p, ended),
                                 scale=rho * stage.scale, operators={})

    @override
    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """The step the CFL check stopped the run at, with the CFL it read and the timestep that keeps the line."""
        step = int(np.asarray(state["stopped_at"])) if "stopped_at" in state else -1
        if step < 0:
            return None
        cfl = float(np.asarray(state["stopped_cfl"]))
        line = float(self.line(self.constants()))
        safe = self.delta_t * line / cfl if np.isfinite(cfl) and cfl > 0.0 else 0.0
        return Problem(
            "flow.cfl-exceeded",
            f"at its step {step} (t = {step * self.delta_t:.6g} s) the flow's CFL number max|u| dt / h read {cfl:.4g}, "
            f"past the {self.scheme.name} scheme's line of {line:.4g} along the vessel's axis: past it the scheme's "
            f"explicit convection is not stable, so the run is refused from that step on",
            f"use a timestep dt <= {safe:.4g} s (the line at the speed it reached), or dt='auto'" if safe > 0.0 else
            "the velocity went non-finite: use a smaller timestep, or dt='auto'")


__all__ = ["ExplicitFlowInputs", "ExplicitFlowNode", "ExplicitFlowState", "ExplicitParts", "cfl_check"]
