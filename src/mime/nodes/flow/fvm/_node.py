"""The implicit flow node: the FVM's field on a periodic box, stepped by a scheme of the monolithic projected CG's
stage kind (IMEX Euler, or its steady limit), with its wall, its drive, its solve report, its per-step CFL check and
its step identity.

**What a step computes, all in the program from the step's parameters** (no value derived from a parameter is made at
build: round 3, item 2): the tube's signed distance from its centre, axis and radius (data, item 3), the band's
indicator and the regions' inverse permeability on each face set; the resistivity ``nu x sum(1/K)`` and Jacobi's
diagonal from the viscosity; the body force on the fluid, ``g (1 - chi)``, ``g`` what the laws bound to the medium
deliver for the step (the waveform is theirs); then the scheme's step.

**The CFL check** (item 1, F1a's half). With convection on, each step reads ``max|u| dt / h`` of the velocity the
explicit convection uses; past the scheme's line the node records the first such step in ``stopped_at`` with the CFL it
read, and :meth:`stop_problem` refuses the run after it, naming the step and the route. The check selects nothing in
the step (it is a report, as the build's other flags are), so it writes no NaN and the field's derivative keeps its
status; a run past the line is refused, so no result or derivative across it is returned. ``stokes=True`` runs are
exempt (the lines bound convection only).

**The report.** Each step's primal solve report (``mime.nodes.flow.fvm._solve``'s block) is a state field, with its
running sum; the derivative solves' reports come out through the ``solve_report`` parameter's gradient (summed over
the run by default; a row a step with ``report_rows``) and the tangent of the running sum.

**The step identity** (route 2, every term by formula, paired at the step's solution ``u'``): ``dKE = W_drive -
D_visc - D_wall - N_BE - N_conv - N_Krylov``, ``KE`` by the grid's inner product. The drive, the viscosity and the wall's
penalty are laws hosted in the step, each booked under its own name; the scheme's terms are ``N_BE = (1/2) rho |u' -
u|^2 dV``, ``N_conv = -dt rho <C(u), u'>`` (convection at mixed time levels) and ``N_Krylov = dt rho <rhs - A u', u'>``
(the solve's true residual against the solution). With the ledger attached the node also reports ``|N_Krylov| /
D_visc`` per step (the stall detector, flagged above :data:`STALL_LINE`)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, override

import numpy as np
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, run_dtype
from mime.core import ledger as books
from mime.nodes.flow.fvm._band import TubeSolid, WallFields, band_named, interior_depth, negative_resistivity, \
    wall_fields
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._schemes import StageParts, StepContext, TimeScheme, scheme_named
from mime.nodes.flow.fvm._solve import REPORT_SIZE, Stopping, preconditioner_named, projected_solve
from mime.registry import Problem

EQUATIONS = (r"\frac{\mathbf{u}_{n+1} - \mathbf{u}_{n}}{h} = -\nabla \cdot \left( \mathbf{u}_{n} \mathbf{u}_{n} \right) + "
             r"\frac{\mu}{\rho} \left( \nabla^{2} \mathbf{u}_{n+1} - \operatorname{K}^{-1} \mathbf{u}_{n+1} \right) + "
             r"\left( 1 - \operatorname{band} \right) \mathbf{g} - \nabla \operatorname{p}_{n+1}, \quad \nabla \cdot "
             r"\mathbf{u}_{n+1} = 0")

#: Where the symbol rows' functions are.
_STEP = "mime.nodes.flow.fvm._node.ImplicitFlowNode.stepped"
_OPERATOR = "mime.nodes.flow.fvm._node.ImplicitFlowNode.operator_of"
_DRIVE = "mime.nodes.flow.fvm._node.ImplicitFlowNode.drive"

#: The stall detector's line: ``|N_Krylov| / D_visc`` above it is a stall (the stall spike's transient values sat about
#: three orders below a stall's reading) [inference from the measured separation].
STALL_LINE = 1e-3
#: The hosted laws a metered node books, by the field their delivery names.
HOSTED_FIELDS = ("body_force", "viscous", "penalty")


class _FlowFields(TypedDict):
    velocity_x: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    velocity_y: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    velocity_z: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: The step's primal solve report (``mime.nodes.flow.fvm.REPORT_FIELDS``), and its running sum.
    solve_report: ReadOnly[Annotated[Float[Array, "6"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    solve_report_sum: ReadOnly[Annotated[Float[Array, "6"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    solve_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cfl: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_at: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_cfl: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    lumen_flux: ReadOnly[Annotated[Float[Array, ""], Units("m^3/s")]]  # noqa: F722 - a jaxtyping shape
    bypass_ratio: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    negative_resistivity: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class FlowState(_FlowFields, closed=True):
    """The flow's state without the energy ledger."""


class _FlowRecords(books.LedgerRecords):
    """With the energy ledger attached: the stall detector, ``|N_Krylov| / D_visc``, and its flag, beside the ledger's
    records (``mime.core.ledger.LedgerRecords``)."""

    stall_ratio: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stall_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class MeteredFlowState(_FlowFields, _FlowRecords, closed=True):
    """A metered flow's state: the flow's fields, the stall detector and the ledger's records."""


class FlowInputs(TypedDict, closed=True):
    body_force: NotRequired[ReadOnly[Annotated[Float[Array, "3"], Units("m/s^2")]]]  # noqa: F722 - a jaxtyping shape


class StepParts(NamedTuple):
    """What the step identity reads of a step: the stage's parts, the fields the operator was made from, and its
    energy terms."""

    stage: StageParts
    fields: WallFields
    energy: Mapping[str, Array]


def _velocity(state: Mapping[str, Array]) -> Velocity:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


class ImplicitFlowNode(MimeNode):
    """The flow field of a medium on a periodic box holding a straight tube, stepped by a scheme of the monolithic
    projected CG's stage kind (the module states the step)."""

    LEDGER_TERMS = ("backward_euler", "convection", "krylov")

    meta = NodeMeta(
        algorithm_id="MIME-NODE-123",
        description="A medium's flow on a periodic box holding a straight tube: the Cartesian MAC grid, the wall a smooth "
                    "Brinkman band, stepped by IMEX Euler (or its steady limit) on the projected monolithic CG.",
        governing_equations=EQUATIONS,
        assumptions=(
            "Incompressible Newtonian fluid, its viscosity and density uniform",
            "The operators are the MAC grid's: the cells' seven-point Laplacian on each face set, the faces' divergence "
            "and gradient, convection in divergence form; band is the wall's indicator and K its permeability field, "
            "made in the step from the tube's centre, axis and radius; the pressure p is recovered on request, not held",
            "The wall is a first-order immersed wall: a smooth band of registered width and depth, its effective "
            "position shifted from the signed distance's zero by the band's planar calibration, constant in cells",
            "The solid outside the lumen is porous with a resistivity, a Darcy flow through it the bypass the node "
            "reports each step",
            "Convection is explicit at the step's start (forward Euler); the viscosity, the penalty and "
            "incompressibility implicit (backward Euler); the drive is evaluated at the step's end",
            "The time of a step is its count times the timestep, never a sum of timesteps"),
        limitations=(
            "A periodic box only: no openings, the flow driven by a body force on the fluid",
            "First order in time; second order in space in the fluid, first order at the wall",
            "Convection's stability bounds the step by the scheme's CFL line, which the node checks each step and the "
            "run is refused past",
        ),
        references=(Reference("HarlowWelch1965"), Reference("Angot1999")),
    )
    symbols = (
        Symbol(r"\mathbf{u}", SymbolKind.STATE, "velocity_x", "m/s", _STEP),
        Symbol(r"\mu", SymbolKind.PARAMETER, "viscosity", "Pa s", _OPERATOR),
        Symbol(r"\rho", SymbolKind.PARAMETER, "density", "kg/m^3", _OPERATOR),
        Symbol(r"\mathbf{g}", SymbolKind.INPUT, "body_force", "m/s^2", _DRIVE),
    )

    def __init__(self, name: str, timestep: float, *, cells_x: int, cells_y: int, cells_z: int, spacing: float,
                 viscosity: float, density: float, tube_centre: Sequence[float], tube_axis: Sequence[float],
                 tube_radius: float, scheme: str = "imex_euler", convection: bool = True, tolerance: float | None = None,
                 cap: int = 5000, check_every: int = 20, band: str = "smooth_2h_0.3h", interior: str = "0.1h",
                 preconditioner: str = "jacobi", scaling_start: bool = True, periodic_images: str = "own",
                 report_rows: int = 0,
                 initial_velocity_x: Any = None, initial_velocity_y: Any = None, initial_velocity_z: Any = None,
                 ledger: Sequence[str] = ()) -> None:
        chosen = scheme_named(scheme)
        if convection and not chosen.takes_convection:
            raise ValueError(f"the {scheme} scheme needs the Stokes approximation (convection off)")
        band_named(band)
        interior_depth(interior)
        preconditioner_named(preconditioner)
        if periodic_images not in ("own", "transverse"):
            raise ValueError(f"periodic_images is 'own' (an aligned tube) or 'transverse', not {periodic_images!r}")
        if tolerance is None:
            tolerance = 1e-8 if run_dtype() == jnp.float64 else 1e-5
        initial: dict[str, Any] = {}
        for key, value in (("initial_velocity_x", initial_velocity_x), ("initial_velocity_y", initial_velocity_y),
                           ("initial_velocity_z", initial_velocity_z)):
            if value is not None:
                initial[key] = np.asarray(value)
        super().__init__(name, timestep, cells_x=int(cells_x), cells_y=int(cells_y), cells_z=int(cells_z),
                         spacing=float(spacing), viscosity=float(viscosity), density=float(density),
                         tube_centre=tuple(float(c) for c in tube_centre),
                         tube_axis=tuple(float(c) for c in tube_axis), tube_radius=float(tube_radius),
                         scheme=str(scheme), convection=bool(convection), tolerance=float(tolerance), cap=int(cap),
                         check_every=int(check_every), band=str(band), interior=str(interior),
                         preconditioner=str(preconditioner), scaling_start=bool(scaling_start),
                         periodic_images=str(periodic_images), report_rows=int(report_rows),
                         solve_report=np.zeros((max(int(report_rows), 1), REPORT_SIZE)), **initial,
                         **({"ledger": tuple(ledger)} if ledger else {}))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        fixed = "a structural setting: the box's cells were chosen with it"
        return {**super().param_specs(),
                "spacing": ParamSpec(trainable=False, units="m", description=fixed),
                "viscosity": ParamSpec(units="Pa s", bounds=(0.0, None),
                                       description="the fluid's dynamic viscosity, mu; the step uses mu / rho"),
                "density": ParamSpec(units="kg/m^3", bounds=(0.0, None)),
                "tube_centre": ParamSpec(units="m"), "tube_axis": ParamSpec(units="1"),
                "tube_radius": ParamSpec(units="m", bounds=(0.0, None)),
                "tolerance": ParamSpec(trainable=False, units="1",
                                       description="the solve's relative tolerance on its true residual"),
                "solve_report": ParamSpec(trainable=False, units="1",
                                          description="zeros, never moved: the derivative solves' reports come out "
                                                      "as its gradient")}

    @override
    def boundary_input_spec(self) -> dict[str, Any]:
        """The body force per unit mass the laws bound to the medium drive its fluid with (summed), this step's."""
        return {"body_force": node_input("m/s^2", (3,), additive=True,
                                         description="a uniform body force per unit mass on the fluid")}

    # -- the pieces of a step ------------------------------------------------------------------------------------
    @property
    def scheme(self) -> TimeScheme:
        return scheme_named(self.constants()["scheme"])

    def grid(self, p: Mapping[str, Any]) -> MACGrid:
        c = self.constants()
        return MACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), p["spacing"])

    def solids(self, p: Mapping[str, Any], grid: MACGrid) -> list[TubeSolid]:
        extents = [count * grid.spacing for count in grid.cells]
        if self.constants()["periodic_images"] == "own":
            shifts: tuple[tuple[Any, Any, Any], ...] = ((0.0, 0.0, 0.0),)
        else:
            shifts = tuple((m * extents[0], n * extents[1], 0.0) for m in (-1, 0, 1) for n in (-1, 0, 1))
        return [TubeSolid(as_run_array(p["tube_centre"]), as_run_array(p["tube_axis"]), as_run_array(p["tube_radius"]),
                          shifts)]

    def fields(self, p: Mapping[str, Any], grid: MACGrid) -> WallFields:
        c = self.constants()
        return wall_fields(grid, self.solids(p, grid), band_named(c["band"]), interior_depth(c["interior"]),
                           run_dtype())

    @staticmethod
    def kinematic(p: Mapping[str, Any]) -> Array:
        """``nu = mu / rho`` (m^2/s), made in the step from the two leaves (never at build)."""
        return as_run_array(p["viscosity"]) / as_run_array(p["density"])

    def resistivity(self, p: Mapping[str, Any], fields: WallFields) -> Velocity:
        k = fields.inverse_permeability
        nu = self.kinematic(p)
        return (nu * k[0], nu * k[1], nu * k[2])

    def drive(self, fields: WallFields, delivered: Array) -> Velocity:
        """The body force per unit mass (m/s^2) on each face set: what the laws bound to the medium delivered for the
        step (at its end, IMEX Euler's stage time), on the fluid only (``1 - chi``). The waveform is the drive's, never
        the field's."""
        chi = fields.chi
        return (delivered[0] * (1.0 - chi[0]), delivered[1] * (1.0 - chi[1]), delivered[2] * (1.0 - chi[2]))

    def operator_of(self, p: Mapping[str, Any], grid: MACGrid, sigma: Velocity) -> Any:
        nu = self.kinematic(p)

        def operator(shift: Any) -> Any:
            def apply(u: Velocity) -> Velocity:
                viscous = grid.laplacian(u)
                return (shift * u[0] - nu * viscous[0] + sigma[0] * u[0],
                        shift * u[1] - nu * viscous[1] + sigma[1] * u[1],
                        shift * u[2] - nu * viscous[2] + sigma[2] * u[2])
            return apply

        return operator

    def diagonal_of(self, p: Mapping[str, Any], grid: MACGrid, sigma: Velocity, shift: Any) -> Velocity:
        base = shift - self.kinematic(p) * grid.laplacian_diagonal()
        return (base + sigma[0], base + sigma[1], base + sigma[2])

    def report_row(self, p: Mapping[str, Any], steps: Array) -> Array:
        """The ``solve_report`` row this step's solve writes its derivative solves' reports through."""
        rows = int(self.constants()["report_rows"])
        if rows == 0:
            return jnp.asarray(p["solve_report"])[0]
        return jnp.asarray(p["solve_report"])[jnp.clip(steps, 0, rows - 1)]

    def fluxes(self, grid: MACGrid, fields: WallFields, axis: Array, velocity: Velocity) -> tuple[Array, Array, Array]:
        """The lumen's and the solid's axial flux (m^3/s): each the volume integral of the axial velocity over its
        region (the lumen and band, ``phi < W/2``; the solid beyond) divided by the tube's axial length per box, exact
        for a periodic divergence-free field; and the lumen's flux in magnitude, the scale its rounding is judged on."""
        width = band_named(self.constants()["band"]).width * grid.spacing
        lumen = jnp.zeros((), dtype=velocity[0].dtype)
        solid = jnp.zeros((), dtype=velocity[0].dtype)
        scale = jnp.zeros((), dtype=velocity[0].dtype)
        for component in range(3):
            inside = fields.signed_distance[component] < 0.5 * width
            lumen = lumen + axis[component] * jnp.sum(jnp.where(inside, velocity[component], 0.0))
            solid = solid + axis[component] * jnp.sum(jnp.where(inside, 0.0, velocity[component]))
            scale = scale + jnp.abs(axis[component]) * jnp.sum(jnp.abs(jnp.where(inside, velocity[component], 0.0)))
        length = grid.cells[2] * grid.spacing / axis[2]
        weight = grid.cell_volume / length
        return lumen * weight, solid * weight, scale * weight

    # -- the node's interface -------------------------------------------------------------------------------------
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
        common: dict[str, Array] = dict(
            velocity_x=start[0], velocity_y=start[1], velocity_z=start[2], steps=jnp.asarray(0, dtype=jnp.int32),
            solve_report=jnp.zeros(REPORT_SIZE, dtype=dtype), solve_report_sum=jnp.zeros(REPORT_SIZE, dtype=dtype),
            solve_flagged=z, cfl=z, stopped_at=jnp.asarray(-1, dtype=jnp.int32), stopped_cfl=z, lumen_flux=z,
            bypass_ratio=z, negative_resistivity=z)
        if not self.metered:
            return node_fields(FlowState(**common))
        return books.with_records(self, {**node_fields(FlowState(**common)), "stall_ratio": z, "stall_flagged": z})

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        if self.metered:
            return books.metered_step(self, self.boundary_input_spec(), p, state, boundary_inputs, dt, self.stepped)
        return self.stepped(p, state, boundary_inputs, dt)[0]

    def stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> tuple[dict[str, Array], StepParts]:
        c = self.constants()
        scheme = scheme_named(c["scheme"])
        grid = self.grid(p)
        dtype = run_dtype()
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        operator = self.operator_of(p, grid, sigma)
        stopping = Stopping(tolerance=as_run_array(p["tolerance"]), cap=int(c["cap"]),
                            check_every=int(c["check_every"]))
        preconditioner = preconditioner_named(c["preconditioner"])
        steps = jnp.asarray(state["steps"])
        tap = self.report_row(p, steps)

        def solve(shift: Any, rhs: Velocity, guess: Velocity | None) -> tuple[Velocity, Array]:
            return projected_solve(grid, operator(shift), self.diagonal_of(p, grid, sigma, shift), rhs,
                                   guess if c["scaling_start"] else None, stopping, tap, preconditioner)

        delivered = jnp.asarray(boundary_inputs.get("body_force", jnp.zeros(3, dtype=dtype)), dtype=dtype)
        context = StepContext(grid, operator, lambda time: self.drive(fields, delivered), bool(c["convection"]),
                              solve)
        velocity = _velocity(state)
        time = steps.astype(dtype) * dt
        speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in velocity]))
        cfl = speed * dt / grid.spacing
        checked = bool(c["convection"]) and scheme.cfl_line is not None
        stopped_before = jnp.asarray(state["stopped_at"]) >= 0
        # Past the line unless provably at or under it: a NaN or an infinite CFL is past it too.
        past = ~(cfl <= scheme.cfl_limit(p["tube_axis"])) if checked else jnp.asarray(False)
        ended_velocity, stage = scheme.step(context, velocity, time, dt)
        report = stage.report
        flagged = (report[2] + report[3] + report[4] > 0).astype(dtype)
        axis = as_run_array(p["tube_axis"])
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis,
                                                (ended_velocity[0], ended_velocity[1], ended_velocity[2]))
        ratio = self.ratio(solid, lumen, lumen_scale)
        energy = self.energy_terms(p, grid, velocity, stage, sigma, dt) if self.metered else {}
        ended: dict[str, Array] = dict(
            velocity_x=ended_velocity[0], velocity_y=ended_velocity[1], velocity_z=ended_velocity[2],
            steps=steps + 1, solve_report=report, solve_report_sum=jnp.asarray(state["solve_report_sum"]) + report,
            solve_flagged=flagged, cfl=cfl.astype(dtype),
            stopped_at=jnp.where(stopped_before, jnp.asarray(state["stopped_at"]),
                                 jnp.where(past, steps + 1, -1)).astype(jnp.int32),
            stopped_cfl=jnp.where(stopped_before, jnp.asarray(state["stopped_cfl"]),
                                  jnp.where(past, cfl, 0.0)).astype(dtype),
            lumen_flux=lumen, bypass_ratio=ratio,
            negative_resistivity=negative_resistivity(fields).astype(dtype))
        held = node_fields(FlowState(**ended))
        if self.metered:
            stall = self.stall_ratio(energy)
            held = {**held, "stall_ratio": stall, "stall_flagged": (stall > STALL_LINE).astype(dtype)}
        return held, StepParts(stage, fields, energy)

    # -- the step identity ------------------------------------------------------------------------------------------
    @override
    def storage(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array:
        """``(1/2) rho <u, u>`` (J) where the scheme stores the fluid's inertia; zero for the steady limit."""
        velocity = _velocity(state)
        if not self.scheme.stores_inertia:
            return jnp.zeros((), dtype=velocity[0].dtype)
        return 0.5 * as_run_array(p["density"]) * self.grid(p).inner(velocity, velocity)

    def energy_terms(self, p: Mapping[str, Any], grid: MACGrid, before: Velocity, stage: StageParts, sigma: Velocity,
                     dt: float) -> dict[str, Array]:
        """Each term of the step's identity (J), and the magnitude of the arithmetic forming them (``scale``)."""
        rho, nu = as_run_array(p["density"]), self.kinematic(p)
        solved = stage.solved
        inner, size = grid.inner, grid.inner_magnitude
        laplacian = grid.laplacian(solved)
        minus_laplacian = (-laplacian[0], -laplacian[1], -laplacian[2])
        penalised = (sigma[0] * solved[0], sigma[1] * solved[1], sigma[2] * solved[2])
        applied = self.operator_of(p, grid, sigma)(stage.shift)(solved)
        remainder = (stage.rhs[0] - applied[0], stage.rhs[1] - applied[1], stage.rhs[2] - applied[2])
        change = (solved[0] - before[0], solved[1] - before[1], solved[2] - before[2])
        stores = self.scheme.stores_inertia
        zero = jnp.zeros((), dtype=solved[0].dtype)
        terms = {
            "drive": dt * rho * inner(stage.force, solved),
            "viscous": dt * rho * nu * inner(minus_laplacian, solved),
            "penalty": dt * rho * inner(penalised, solved),
            "backward_euler": 0.5 * rho * inner(change, change) if stores else zero,
            "convection": -dt * rho * inner(stage.convection, solved),
            "krylov": dt * rho * inner(remainder, solved),
        }
        scale = rho * (dt * (size(stage.rhs, solved) + size(applied, solved) + size(stage.force, solved)
                             + size(stage.convection, solved) + nu * size(minus_laplacian, solved)
                             + size(penalised, solved))
                       + (size(before, before) + size(solved, solved) + size(before, solved) if stores else 0.0))
        return {**terms, "scale": scale}

    @staticmethod
    def ratio(numerator: Array, denominator: Array, scale: Array) -> Array:
        """A report's ratio ``a / b``, read as zero where ``b`` is within its own rounding on ``scale`` (the magnitude
        of the arithmetic forming it: a flux that cancels to rounding has no ratio, and two compilations round it
        differently) or too small for the derivative, ``-a / b^2``, to be a finite number of the dtype, so that the
        report and its derivative are finite and reproducible at every state."""
        finfo = jnp.finfo(denominator.dtype)
        held = (jnp.abs(denominator) > 64.0 * finfo.eps * scale) & \
            (denominator * denominator > 4.0 * finfo.tiny * jnp.maximum(jnp.abs(numerator), 1.0))
        return jnp.where(held, numerator / jnp.where(held, denominator, 1.0), 0.0)

    @classmethod
    def stall_ratio(cls, energy: Mapping[str, Array]) -> Array:
        """``|N_Krylov| / D_visc``: zero where both are zero (rest, no drive), infinite where only the dissipation is."""
        krylov, viscous = jnp.abs(energy["krylov"]), energy["viscous"]
        return jnp.where((viscous == 0.0) & (krylov > 0.0), jnp.inf, cls.ratio(krylov, viscous, viscous))

    @override
    def identity_terms(self, p: Mapping[str, Any], held: Sequence[books.LedgerDelivery], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: StepParts,
                       dt: float) -> books.StepAccount:
        """The step's identity: the hosted laws' work under their names (the drive's ``+W``, the viscosity's and the
        wall's ``-D``), the scheme's three terms, the storage at both ends. A hosted law the ledger does not name is
        booked as unattributed, which the ledger counts as a failure of attribution."""
        energy = parts.energy
        dtype = jnp.asarray(energy["krylov"]).dtype
        work_of = {"body_force": energy["drive"], "viscous": -energy["viscous"], "penalty": -energy["penalty"]}
        rows = books.laws(held)
        work = jnp.zeros(len(rows), dtype=dtype)
        hosted = jnp.zeros((len(rows), 2), dtype=dtype)
        named: set[str] = set()
        for delivery in held:
            if delivery.kind != books.HOSTED or delivery.field not in work_of:
                raise ValueError(f"the flow node hosts {list(work_of)}: it takes no delivery {delivery}")
            work = work.at[rows.index(delivery.law)].add(work_of[delivery.field])
            named.add(delivery.field)
        unattributed = sum((value for field, value in work_of.items() if field not in named),
                           start=jnp.zeros((), dtype=dtype))
        terms = jnp.stack([energy["backward_euler"], energy["convection"], energy["krylov"]])
        return books.StepAccount(work=work, hosted=hosted, unattributed=unattributed, terms=terms,
                                 storage_start=self.storage(p, state), storage_end=self.storage(p, ended),
                                 scale=energy["scale"], operators={})

    # -- after the run ------------------------------------------------------------------------------------------------
    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """The step the CFL check stopped the run at, with the CFL it read and the timestep that keeps the line."""
        step = int(np.asarray(state["stopped_at"])) if "stopped_at" in state else -1
        if step < 0:
            return None
        cfl = float(np.asarray(state["stopped_cfl"]))
        line = float(self.scheme.cfl_limit(self.constants()["tube_axis"]))
        finite = math.isfinite(cfl) and cfl > 0.0
        safe = self.delta_t * line / cfl if finite else float("nan")
        return Problem(
            "flow.cfl-exceeded",
            f"at its step {step} (t = {step * self.delta_t:.6g} s) the flow's CFL number max|u| dt / h read {cfl:.4g}, "
            f"past the {self.scheme.name} scheme's line of {line:.4g} along the vessel's axis: past it the scheme's "
            f"explicit convection is not stable (a CFL that is not finite is past it too), so the run is refused from "
            f"that step on",
            (f"use a timestep dt <= {safe:.4g} s (the line at the speed it reached), or the Stokes approximation "
             f"FVM(stokes=True) where convection is negligible") if finite else
            "the flow went non-finite before the step: use a smaller timestep, dt=..., or the Stokes approximation "
            "FVM(stokes=True)")


__all__ = ["HOSTED_FIELDS", "STALL_LINE", "FlowInputs", "FlowState", "ImplicitFlowNode", "MeteredFlowState",
           "StepParts"]
