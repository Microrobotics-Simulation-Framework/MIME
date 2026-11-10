"""The implicit flow node: the FVM's field on a periodic box, stepped by a scheme of the monolithic projected CG's
stage kind (IMEX Euler, or its steady limit), with its wall, its drive, its solve report, its per-step CFL check and
its step identity.

**What a step computes, all in the program from the step's parameters** (no value derived from a parameter is made at
build: round 3, item 2): the tube's signed distance from its centre, axis and radius (data, item 3), the band's
indicator and the regions' inverse permeability on each face set; the resistivity ``nu x sum(1/K)`` and Jacobi's
diagonal from the viscosity; the body force on the fluid, ``g (1 - chi)``, ``g`` what the laws bound to the medium
deliver for the step (the waveform is theirs); then the scheme's step.

**The CFL check** (item 1, F1a's half). With convection on, each step reads ``max|u| dt / h`` of the velocity the
explicit convection uses. At the first step past the scheme's line the node records the step in ``stopped_at`` with the
CFL it read, and **holds its state**: that step and every later one leave it unchanged (a held state writes no NaN).
The run ends at the end of the chunk the step lies in (``STOP_FIELD``, read between chunks) and :meth:`stop_problem`
refuses it after, naming the step and the route. A finished run's every step is on the other branch, so no derivative
of a run that is returned crosses the hold. ``stokes=True`` runs are exempt (the lines bound convection only).

**The validity lines it flags each step** (the entry lists every line): a solve that stopped other than converging
(``solve_flagged``); with ``stokes=True``, the lumen's Reynolds number past the declared Stokes line
(``reynolds_flagged``, :data:`STOKES_REYNOLDS`); the solid's bypass flux past its declared line (``bypass_flagged``,
:data:`BYPASS_LINE`). The resolution in cells and float32's rounding floor are fixed when the node is built, so the
build reports them (:meth:`built_lines`).

**The report.** Each step's primal solve report (``mime.nodes.flow.fvm._solve``'s block) is a state field, with its
running sum; the derivative solves' reports come out through the ``solve_report`` parameter's gradient (summed over
the run by default; a row a step with ``solve_report_rows``) and the tangent of the running sum.

**The step identity** (route 2, every term by formula, paired at the step's solution ``u'``): ``dKE = W_drive -
D_visc - D_wall - N_BE - N_conv - N_Krylov``, ``KE`` by the grid's inner product. The node is a receiver of the ``fluid``
port kind: each law driving it delivers its body force per unit mass ``g`` into its own input, and the step pairs it
with the fluid's momentum at the instant it is applied, ``W = dt g . P'``, ``P' = rho <1 - chi, u'>`` (the fluid's
alone, as the force acts on it alone): the drive's work regrouped by law. The viscosity and the wall's penalty are laws
held in the step (:meth:`hosted_laws`), each booked under its own name; the scheme's terms are ``N_BE = (1/2) rho |u' -
u|^2 dV``, ``N_conv = -dt rho <C(u), u'>`` (convection at mixed time levels) and ``N_Krylov = dt rho <rhs - A u', u'>``
(the solve's true residual against the solution: the remainder unprojected, so it holds the projection's own term,
``dt rho <p', D u'>``, which the exact projector leaves at rounding). With the ledger attached the node also reports
``|N_Krylov| / D_visc`` per step (the stall detector, flagged above :data:`STALL_LINE`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, NamedTuple, override

import numpy as np
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, run_dtype
from mime.core import ledger as books
from mime.core._node import rechecked
from mime.nodes.flow.fvm._band import TubeSolid, WallFields, band_named, interior_depth, negative_resistivity, \
    wall_fields
from mime.nodes.flow.fvm._flags import FLAGS
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow._body_force import STEP_END
from mime.nodes.flow._waveform import WAVEFORM_SIZE, waveform_at
from mime.nodes.flow.fvm._schemes import SchemeState, StageKind, StageParts, StepContext, TimeScheme, scheme_named
from mime.nodes.flow.fvm._solve import AT_FLOOR_BY_DEFAULT, DEFAULT_CAP, REPORT_SIZE, Stopping, default_tolerance, \
    preconditioner_named, projected_solve, recheck_stopping, stagnation_window
from mime.registry import Problem

EQUATIONS = (r"\frac{\mathbf{u}_{n+1} - \mathbf{u}_{n}}{h} = -\nabla \cdot \left( \mathbf{u}_{n} \mathbf{u}_{n} \right) + "
             r"\frac{\mu}{\rho} \left( \nabla^{2} \mathbf{u}_{n+1} - \operatorname{K}^{-1} \mathbf{u}_{n+1} \right) + "
             r"\left( 1 - \operatorname{band} \right) \mathbf{g} - \nabla \operatorname{p}_{n+1}, \quad \nabla \cdot "
             r"\mathbf{u}_{n+1} = 0")

#: Where the symbol rows' functions are.
_STEP = "mime.nodes.flow.fvm._node.ImplicitFlowNode.stepped"
_OPERATOR = "mime.nodes.flow.fvm._node.ImplicitFlowNode.operator_of"
_DRIVE = "mime.nodes.flow.fvm._node.ImplicitFlowNode.drive"

#: The Reynolds number, by the lumen's diameter and mean speed, up to which the Stokes approximation (``stokes=True``)
#: is declared: a declaration, not a fit (B8.3 asks for the line, and no measurement sets it). In the straight
#: periodic tube F1a builds the axial flow's convection vanishes identically, so the line bounds nothing measured here:
#: it is for the geometries and bodies that follow.
STOKES_REYNOLDS = 1.0
#: The solid's bypass flux, as a share of the lumen's, above which a step is flagged: a declaration (B3 makes the
#: bypass a validity line and offers the 0.01h interior "when it fires"; no measurement sets the line). Measured at the
#: defaults (0.1h): BYPASS_MEASURED.
BYPASS_LINE = 1e-3
BYPASS_MEASURED = ("measured on the straight tube, steady Stokes: 1.4e-4 at D/h 32, 4.8e-4 at 16 and 2.6e-3 at 8 with "
                   "the 0.1h interior; 6.4e-5, 1.4e-4 and 1.2e-3 with 0.01h")
#: The state field a node that stops a run holds the step it stopped at in (-1 while it has not): the run reads it
#: between chunks and ends at the chunk the stop lies in.
STOP_FIELD = "stopped_at"
#: The lumen's diameter in cells at and above which the band's shift is calibrated (B3; round 1, item 2).
CALIBRATED_CELLS = 32.0
#: The slowest decay of a section's viscous mode, ``tau = R_max^2 / (STALL_SECTION nu)``: pi^2/4, conservative for
#: every convex section (round 4, item 3; round 5, item 10).
STALL_SECTION = math.pi ** 2 / 4.0
#: The stall detector's line: ``|N_Krylov| / D_visc`` above it is a stall (the stall spike's transient values sat about
#: three orders below a stall's reading) [inference from the measured separation].
STALL_LINE = 1e-3
#: Half of each dtype's digits, sqrt(eps): the share of its scale a report's denominator must exceed (:meth:`ratio`).
HALF_DIGITS = {np.dtype(np.float32): math.sqrt(float(np.finfo(np.float32).eps)),
               np.dtype(np.float64): math.sqrt(float(np.finfo(np.float64).eps))}
#: What a geometry leaf's derivative is worth, where a user reads it (its parameter's description, the entry's gradient
#: record and validity line): no gate has earned it.
GEOMETRY = ("its derivative is unverified: the tube's radius, centre and axis are not in the gradient gate yet (its "
            "loss is to be the box's whole axial flux, since the lumen's mask moves with the radius); through the "
            "band a geometry derivative carries a ripple within a cell of about +-20 % at D/h 8 to 16 and +-10 % at "
            "32 (measured on a static tube)")
#: The fields a held state still writes: the stop's own record.
_STOP_RECORD = frozenset({STOP_FIELD, "stopped_cfl"})
#: The hosted laws a metered node books, by the field their delivery names: the fluid's viscosity and the wall's band.
HOSTED_FIELDS = ("viscous", "penalty")
#: The inputs a law driving the fluid delivers into: its body force, read where the scheme reads its drive at the
#: step's end, and its waveform, read where the scheme evaluates its drive at its own stage instants.
DRIVE_FIELDS = ("body_force", "waveform")


class FlowFields(TypedDict):
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
    bypass_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: The lumen's Reynolds number by its diameter and mean speed, ``2 |Q| / (pi R nu)``, and its Stokes flag.
    reynolds: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    reynolds_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    negative_resistivity: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class FlowState(FlowFields, closed=True):
    """The flow's state without the energy ledger."""


class FlowRecords(books.LedgerRecords):
    """With the energy ledger attached: the stall detector, ``|N_Krylov| / D_visc``, and its flag, beside the ledger's
    records (``mime.core.ledger.LedgerRecords``)."""

    stall_ratio: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stall_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class MeteredFlowState(FlowFields, FlowRecords, closed=True):
    """A metered flow's state: the flow's fields, the stall detector and the ledger's records."""


class FlowInputs(TypedDict, closed=True):
    body_force: NotRequired[ReadOnly[Annotated[Float[Array, "3"], Units("m/s^2")]]]  # noqa: F722 - a jaxtyping shape
    waveform: NotRequired[ReadOnly[Annotated[Float[Array, "7"], Units("1")]]]  # noqa: F722 - a jaxtyping shape


class StepParts(NamedTuple):
    """What the step identity reads of a step: the stage's parts, the fields the operator was made from, and its
    energy terms."""

    stage: StageParts
    fields: WallFields
    energy: Mapping[str, Array]


def _velocity(state: Mapping[str, Array]) -> Velocity:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


def _scheme_state(state: Mapping[str, Array]) -> SchemeState:
    """The scheme's state (keyed by its ``state_fields``) from the node's: the velocity's three face sets as one."""
    return {"velocity": _velocity(state)}


def _held_of(scheme_state: SchemeState) -> dict[str, Array]:
    """The node's state fields of a scheme's state: the velocity as a column per MAC component (``mime.mappings``'
    first workaround), any other field under its own name."""
    held = {name: value for name, value in scheme_state.items() if name != "velocity"}
    velocity: Velocity = scheme_state["velocity"]
    return {"velocity_x": velocity[0], "velocity_y": velocity[1], "velocity_z": velocity[2], **held}


class ImplicitFlowNode(MimeNode):
    """The flow field of a medium on a periodic box holding a straight tube, stepped by a scheme of the monolithic
    projected CG's stage kind (the module states the step)."""

    LEDGER_TERMS = ("backward_euler", "convection", "krylov")
    #: The grid kind's own terms its identity books after the scheme's (none on a periodic box).
    GRID_TERMS: ClassVar[tuple[str, ...]] = ()
    #: The stall, a convective term not formed from its velocity, the drive in a solid; and the openings' two, pending
    #: (``mime.nodes.flow.fvm._flags``).
    LEDGER_FLAGS = FLAGS
    #: The state field the run reads between chunks for the step the CFL check stopped it at.
    STOP_FIELD: ClassVar[str] = STOP_FIELD
    rechecks: ClassVar[bool] = True

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
                 at_floor: bool | None = None, cap: int = DEFAULT_CAP, check_every: int = 20, band: str = "smooth_2h_0.3h",
                 interior: str = "0.1h",
                 preconditioner: str = "jacobi", scaling_start: bool = True, periodic_images: str = "own",
                 solve_report_rows: int = 0, wall_name: str = "", solve_report: Any = None,
                 initial_velocity_x: Any = None, initial_velocity_y: Any = None, initial_velocity_z: Any = None,
                 ledger: Sequence[str] = (), **extra: Any) -> None:
        chosen = scheme_named(scheme)
        if chosen.stage_kind is not StageKind.MONOLITHIC_PROJECTED_CG or chosen.state_fields != ("velocity",) or \
                chosen.ledger_terms + self.GRID_TERMS != self.LEDGER_TERMS:
            raise ValueError(f"the {scheme} scheme is of the {chosen.stage_kind} stage kind, carries "
                             f"{chosen.state_fields} and books {chosen.ledger_terms}: this node steps the monolithic "
                             f"projected CG's schemes, whose state is the velocity and whose terms are "
                             f"{self.LEDGER_TERMS}")
        if convection and not chosen.takes_convection:
            raise ValueError(f"the {scheme} scheme needs the Stokes approximation (convection off)")
        band_named(band)
        interior_depth(interior)
        preconditioner_named(preconditioner)
        if periodic_images not in ("own", "transverse"):
            raise ValueError(f"periodic_images is 'own' (an aligned tube) or 'transverse', not {periodic_images!r}")
        # A node made again from its own arguments (the ledger's wiring, a test's rebuild) carries the stop it was made
        # with; otherwise the stop at the floor is float32's default, where no tolerance is given. Its solve_report
        # parameter is zeros sized by solve_report_rows, made here again, so the one it is given back is not read.
        del solve_report
        if at_floor is None:
            at_floor = tolerance is None and str(jnp.dtype(run_dtype())) in AT_FLOOR_BY_DEFAULT
        if tolerance is None:
            tolerance = default_tolerance(run_dtype())
        initial: dict[str, Any] = {}
        for key, value in (("initial_velocity_x", initial_velocity_x), ("initial_velocity_y", initial_velocity_y),
                           ("initial_velocity_z", initial_velocity_z)):
            if value is not None:
                initial[key] = np.asarray(value)
        super().__init__(name, timestep, cells_x=int(cells_x), cells_y=int(cells_y), cells_z=int(cells_z),
                         spacing=float(spacing), viscosity=float(viscosity), density=float(density),
                         tube_centre=tuple(float(c) for c in tube_centre),
                         tube_axis=tuple(float(c) for c in tube_axis), tube_radius=float(tube_radius),
                         scheme=str(scheme), convection=bool(convection), tolerance=float(tolerance),
                         at_floor=bool(at_floor), cap=int(cap),
                         check_every=int(check_every), band=str(band), interior=str(interior),
                         preconditioner=str(preconditioner), scaling_start=bool(scaling_start),
                         periodic_images=str(periodic_images), solve_report_rows=int(solve_report_rows),
                         solve_report=np.zeros((max(int(solve_report_rows), 1), REPORT_SIZE)), **initial,
                         **({"wall_name": str(wall_name)} if wall_name else {}), **extra,
                         **({"ledger": tuple(ledger)} if ledger else {}))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        fixed = "a structural setting: the box's cells were chosen with it"
        return {**super().param_specs(),
                "spacing": ParamSpec(trainable=False, units="m", description=fixed),
                "viscosity": ParamSpec(units="Pa s", bounds=(0.0, None),
                                       description="the fluid's dynamic viscosity, mu; the step uses mu / rho"),
                "density": ParamSpec(units="kg/m^3", bounds=(0.0, None)),
                "tube_centre": ParamSpec(units="m", description=f"the tube's axis passes through it; {GEOMETRY}"),
                "tube_axis": ParamSpec(units="1", description=f"the tube's direction, the lattice direction the box "
                                                              f"was built for; {GEOMETRY}"),
                "tube_radius": ParamSpec(units="m", bounds=(0.0, None),
                                         description=f"the lumen's radius; {GEOMETRY}"),
                "tolerance": ParamSpec(trainable=False, units="1",
                                       description="the solve's relative tolerance on its true residual"),
                "solve_report": ParamSpec(trainable=False, units="1",
                                          description="zeros, never moved: the derivative solves' reports come out "
                                                      "as its gradient")}

    @override
    def boundary_input_spec(self) -> dict[str, Any]:
        """:meth:`own_inputs`; with the ledger attached, each law's own input beside the one it would sum into
        (``mime.core.ledger.with_law_inputs``)."""
        return books.with_law_inputs(self.own_inputs(), books.deliveries(self.constants()))

    def own_inputs(self) -> dict[str, Any]:
        """The body force per unit mass the laws bound to the medium drive its fluid with (summed), delivered once a step
        at its end; and the drive's waveform, which a scheme whose stages read their drive at other instants evaluates
        at each (``StepContext.force``). The monolithic schemes registered read at the step's end, so they take the
        delivered value."""
        return {"body_force": node_input("m/s^2", (3,), additive=True,
                                         description="a uniform body force per unit mass on the fluid"),
                "waveform": node_input("1", (WAVEFORM_SIZE,),
                                       description="the drive's waveform, read at a stage's instant where it is not "
                                                   "the step's end")}

    # -- the pieces of a step ------------------------------------------------------------------------------------
    @property
    def scheme(self) -> TimeScheme[Any]:
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

    def forcing(self, fields: WallFields, boundary_inputs: Mapping[str, Array], scheme: TimeScheme[Any],
                dtype: Any) -> Callable[[int, Any], Velocity]:
        """``force(stage, time)``, the drive each stage reads at its instant (:class:`StepContext`): the body force the
        laws bound to the medium delivered, summed, where the scheme's stages read at the step's end, the instant it is
        delivered for; otherwise the delivered waveform evaluated at the stage's time."""
        if all(instant == STEP_END for instant in scheme.drive_instants):
            delivered = jnp.asarray(boundary_inputs.get("body_force", jnp.zeros(3, dtype=dtype)), dtype=dtype)
            return lambda stage, time: self.drive(fields, delivered)
        coefficients = jnp.asarray(boundary_inputs.get("waveform", jnp.zeros(WAVEFORM_SIZE, dtype=dtype)), dtype=dtype)
        return lambda stage, time: self.drive(fields, waveform_at(coefficients, time).astype(dtype))

    def drive(self, fields: WallFields, delivered: Array) -> Velocity:
        """The body force per unit mass (m/s^2) on each face set, on the fluid only (``1 - chi``): what the laws bound
        to the medium deliver. The waveform is the drive's, never the field's."""
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

    def magnitude_of(self, p: Mapping[str, Any], grid: MACGrid, sigma: Velocity) -> Any:
        """:meth:`operator_of` with every coefficient in magnitude, applied to a field of magnitudes: the scale the
        solve's rounding floor is judged on. The shift (``1/dt``, or none), the viscosity (bounded below by zero) and
        the resistivity (its positivity reported at sampling) are each its own magnitude; the Laplacian's is the grid
        kind's."""
        nu = self.kinematic(p)

        def magnitude(shift: Any) -> Any:
            def apply(u: Velocity) -> Velocity:
                viscous = grid.laplacian_magnitude(u)
                return (shift * u[0] + nu * viscous[0] + sigma[0] * u[0],
                        shift * u[1] + nu * viscous[1] + sigma[1] * u[1],
                        shift * u[2] + nu * viscous[2] + sigma[2] * u[2])
            return apply

        return magnitude

    def diagonal_of(self, p: Mapping[str, Any], grid: MACGrid, sigma: Velocity, shift: Any) -> Velocity:
        base = shift - self.kinematic(p) * grid.laplacian_diagonal()
        return (base + sigma[0], base + sigma[1], base + sigma[2])

    def report_row(self, p: Mapping[str, Any], steps: Array) -> Array:
        """The ``solve_report`` row this step's solve writes its derivative solves' reports through."""
        rows = int(self.constants()["solve_report_rows"])
        if rows == 0:
            return jnp.asarray(p["solve_report"])[0]
        return jnp.asarray(p["solve_report"])[jnp.clip(steps, 0, rows - 1)]

    def fluxes(self, grid: MACGrid, fields: WallFields, axis: Array, velocity: Velocity) -> tuple[Array, Array, Array]:
        """The lumen's and the solid's axial flux (m^3/s): each the volume integral of the axial velocity over its
        region (the lumen and band, ``phi < W/2``; the solid beyond) divided by the tube's axial length per box, exact
        for a periodic divergence-free field; and the scale their rounding is judged on, the whole velocity's magnitude
        over the same integral: the projection couples the components, so an axial flux among much larger transverse
        velocities is their rounding."""
        width = band_named(self.constants()["band"]).width * grid.spacing
        lumen = jnp.zeros((), dtype=velocity[0].dtype)
        solid = jnp.zeros((), dtype=velocity[0].dtype)
        scale = jnp.zeros((), dtype=velocity[0].dtype)
        for component in range(3):
            inside = fields.signed_distance[component] < 0.5 * width
            lumen = lumen + axis[component] * jnp.sum(jnp.where(inside, velocity[component], 0.0))
            solid = solid + axis[component] * jnp.sum(jnp.where(inside, 0.0, velocity[component]))
            scale = scale + jnp.sum(jnp.abs(velocity[component]))
        length = grid.cells[2] * grid.spacing / axis[2]
        weight = grid.cell_volume / length
        return lumen * weight, solid * weight, scale * weight

    # -- the node's interface -------------------------------------------------------------------------------------
    @override
    def initial_state(self) -> NodeFields:
        common = self.initial_fields()
        if not self.metered:
            return node_fields(FlowState(**common))
        z = as_run_array(0.0)
        return books.with_records(self, {**node_fields(FlowState(**common)), "stall_ratio": z, "stall_flagged": z})

    def initial_fields(self) -> dict[str, Array]:
        """The flow's state fields at the run's start, without the ledger's."""
        c = self.constants()
        grid = self.grid(c)
        dtype = run_dtype()
        zero = grid.zeros(dtype)
        given = tuple(jnp.asarray(c[key], dtype=dtype) if key in c else zero[k]
                      for k, key in enumerate(("initial_velocity_x", "initial_velocity_y", "initial_velocity_z")))
        start = _held_of(self.scheme.start(grid, (given[0], given[1], given[2])))
        z = as_run_array(0.0)
        common: dict[str, Array] = dict(
            **start, steps=jnp.asarray(0, dtype=jnp.int32),
            solve_report=jnp.zeros(REPORT_SIZE, dtype=dtype), solve_report_sum=jnp.zeros(REPORT_SIZE, dtype=dtype),
            solve_flagged=z, cfl=z, stopped_at=jnp.asarray(-1, dtype=jnp.int32), stopped_cfl=z, lumen_flux=z,
            bypass_ratio=z, bypass_flagged=z, reynolds=z, reynolds_flagged=z, negative_resistivity=z)
        return common

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        if self.metered:
            ended = books.metered_step(self, self.own_inputs(), p, state, boundary_inputs, dt, self.stepped)
        else:
            ended = self.stepped(p, state, boundary_inputs, dt)[0]
        return self.held_past_the_line(state, ended)

    @staticmethod
    def held_past_the_line(state: Mapping[str, Array], ended: NodeFields) -> NodeFields:
        """``ended``, or ``state`` unchanged from the first step past the CFL line on (the stop's record excepted):
        the run past the line does not advance, and ends at the chunk's end (:data:`STOP_FIELD`)."""
        stopped = jnp.asarray(ended[STOP_FIELD]) >= 0
        return {name: value if name in _STOP_RECORD else jnp.where(stopped, jnp.asarray(state[name]), value)
                for name, value in ended.items()}

    def stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> tuple[dict[str, Array], StepParts]:
        c = self.constants()
        scheme: TimeScheme[StageParts] = self.scheme
        grid = self.grid(p)
        dtype = run_dtype()
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        operator = self.operator_of(p, grid, sigma)
        stopping = Stopping(tolerance=as_run_array(p["tolerance"]), at_floor=bool(c["at_floor"]), cap=int(c["cap"]),
                            check_every=int(c["check_every"]),
                            window=stagnation_window(grid.cells, int(c["check_every"])))
        if rechecked(p):
            stopping = recheck_stopping(stopping)
        preconditioner = preconditioner_named(c["preconditioner"])
        steps = jnp.asarray(state["steps"])
        tap = self.report_row(p, steps)

        magnitude = self.magnitude_of(p, grid, sigma)

        def solve(shift: Any, rhs: Velocity, guess: Velocity | None) -> tuple[Velocity, Array]:
            return projected_solve(grid, operator(shift), self.diagonal_of(p, grid, sigma, shift), rhs,
                                   guess if c["scaling_start"] else None, stopping, tap, preconditioner,
                                   magnitude(shift))

        context = StepContext(grid, operator, self.forcing(fields, boundary_inputs, scheme, dtype),
                              bool(c["convection"]), solve, self.kinematic(p), sigma)
        velocity = _velocity(state)
        time = steps.astype(dtype) * dt
        speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in velocity]))
        cfl = speed * dt / grid.spacing
        checked = bool(c["convection"]) and scheme.cfl_line is not None
        stopped_before = jnp.asarray(state["stopped_at"]) >= 0
        # Past the line unless provably at or under it: a NaN or an infinite CFL is past it too.
        past = ~(cfl <= scheme.cfl_limit(p["tube_axis"])) if checked else jnp.asarray(False)
        ended_state, stage = scheme.step(context, _scheme_state(state), time, dt)
        ended_velocity: Velocity = ended_state["velocity"]
        report = stage.report
        flagged = (report[2] + report[3] + report[4] > 0).astype(dtype)
        axis = as_run_array(p["tube_axis"])
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, ended_velocity)
        ratio = self.ratio(solid, lumen, lumen_scale)
        reynolds = 2.0 * jnp.abs(lumen) / (jnp.pi * as_run_array(p["tube_radius"]) * self.kinematic(p))
        stokes = not bool(c["convection"])
        energy = self.energy_terms(p, grid, velocity, stage, sigma, fields, dt) if self.metered else {}
        ended: dict[str, Array] = dict(
            **_held_of(ended_state), steps=steps + 1, solve_report=report,
            solve_report_sum=jnp.asarray(state["solve_report_sum"]) + report,
            solve_flagged=flagged, cfl=cfl.astype(dtype),
            stopped_at=jnp.where(stopped_before, jnp.asarray(state["stopped_at"]),
                                 jnp.where(past, steps + 1, -1)).astype(jnp.int32),
            stopped_cfl=jnp.where(stopped_before, jnp.asarray(state["stopped_cfl"]),
                                  jnp.where(past, cfl, 0.0)).astype(dtype),
            lumen_flux=lumen, bypass_ratio=ratio, bypass_flagged=(jnp.abs(ratio) > BYPASS_LINE).astype(dtype),
            reynolds=reynolds.astype(dtype),
            reynolds_flagged=((reynolds > STOKES_REYNOLDS) & stokes).astype(dtype),
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
                     fields: WallFields, dt: float) -> dict[str, Array]:
        """Each term of the step's identity (J), the fluid's momentum at the solution the drive is paired with
        (:meth:`fluid_momentum`), and the magnitude of the arithmetic forming them (``scale``)."""
        fluid = (1.0 - fields.chi[0], 1.0 - fields.chi[1], 1.0 - fields.chi[2])
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
            **self.fluid_momentum(grid, fluid, solved, rho),
            "viscous": dt * rho * nu * inner(minus_laplacian, solved),
            "penalty": dt * rho * inner(penalised, solved),
            "backward_euler": 0.5 * rho * inner(change, change) if stores else zero,
            "convection": -dt * rho * inner(stage.convection, solved),
            "krylov": dt * rho * inner(remainder, solved),
        }
        # The drive's products are judged on the laws' own magnitude (drive_work's), counted once.
        scale = rho * (dt * (size(stage.rhs, solved) + size(applied, solved)
                             + size(stage.convection, solved) + nu * size(minus_laplacian, solved)
                             + size(penalised, solved))
                       + (size(before, before) + size(solved, solved) + size(before, solved) if stores else 0.0))
        return {**terms, "scale": scale}

    @staticmethod
    def fluid_momentum(grid: MACGrid, fluid: Velocity, velocity: Velocity, rho: Array) -> dict[str, Array]:
        """The fluid's momentum ``rho <1 - chi, u>`` (kg m/s, a component each, ``fluid`` the faces' ``1 - chi``), the
        ``fluid`` port kind's flow, and its magnitude ``rho <1 - chi, |u|>``: what a body force per unit mass on the fluid
        is paired with, and its rounding judged on."""
        volume = grid.cell_volume
        momentum = jnp.stack([rho * volume * jnp.sum(fluid[c] * velocity[c]) for c in range(3)])
        size = jnp.stack([rho * volume * jnp.sum(jnp.abs(fluid[c] * velocity[c])) for c in range(3)])
        return {"momentum": momentum, "momentum_size": size}

    def drive_read(self) -> str:
        """The input the scheme reads its drive from (:data:`DRIVE_FIELDS`): the body force where every stage reads it
        at the step's end, the waveform where they evaluate it at their own instants."""
        return "body_force" if all(instant == STEP_END for instant in self.scheme.drive_instants) else "waveform"

    @override
    def hosted_laws(self) -> tuple[tuple[str, str, str], ...]:
        """The laws the flow holds in its own step, each ``(role, law, class)``: the fluid's viscosity under the node's
        name (``water.flow.viscosity``) and the wall's band under the wall's own (``wall_name``, as the build gives it:
        ``tube.wall``; the node's ``.wall`` otherwise), both dissipative. The drive is delivered, not held."""
        wall = str(self.constants().get("wall_name", "")) or f"{self.name}.wall"
        return (("viscous", f"{self.name}.viscosity", books.DISSIPATIVE), ("penalty", wall, books.DISSIPATIVE))

    def drive_work(self, held: Sequence[books.LedgerDelivery], boundary_inputs: Mapping[str, Array],
                   rows: Sequence[str], dt: float, momentum: Array, size: Array) -> tuple[Array, Array, Array]:
        """Each law's drive paired by the one rule as the ``fluid`` port kind says, ``dt g . P'``, ``g`` its own input
        and ``P'`` the fluid's momentum at the instant the scheme applied it (a row per law); the unattributed input's
        likewise; and the magnitude of the products (law by law, so that drives that cancel are judged on their own
        size). The input the scheme does not read pairs no work: what it delivered was not applied."""
        dtype = momentum.dtype
        work = jnp.zeros(len(rows), dtype=dtype)
        scale = jnp.zeros((), dtype=dtype)
        read = self.drive_read()
        for index, delivery in enumerate(held):
            if delivery.kind == books.LOAD and delivery.field == "body_force" == read:
                g = jnp.asarray(boundary_inputs[books.law_input(delivery.field, index)], dtype=dtype)
                work = work.at[rows.index(delivery.law)].add(dt * jnp.dot(g, momentum))
                scale = scale + dt * jnp.dot(jnp.abs(g), size)
        plain = jnp.asarray(boundary_inputs.get("body_force", jnp.zeros(3, dtype=dtype)), dtype=dtype)
        unattributed = dt * jnp.dot(plain, momentum) if read == "body_force" else jnp.zeros((), dtype=dtype)
        return work, unattributed, scale + dt * jnp.dot(jnp.abs(plain), size)

    @staticmethod
    def ratio(numerator: Array, denominator: Array, scale: Array) -> Array:
        """A report's ratio ``a / b``, read as zero where ``b`` holds fewer than half the dtype's digits of ``scale``
        (the magnitude of the arithmetic forming it, whose rounding is some eps of it: under ``sqrt(eps) scale`` the
        ratio carries more than ``sqrt(eps)`` of rounding, and two compilations round it differently) or is too small
        for the derivative, ``-a / b^2``, to be a finite number of the dtype, so that the report and its derivative are
        finite and reproducible at every state."""
        finfo = jnp.finfo(denominator.dtype)
        held = (jnp.abs(denominator) > HALF_DIGITS[np.dtype(denominator.dtype)] * scale) & \
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
        """The step's identity: each driving law's work, paired with the fluid's momentum (:meth:`drive_work`), the
        hosted laws' under their names (the viscosity's and the wall's ``-D``), the scheme's three terms, the storage at
        both ends. A hosted law the ledger does not name is booked as unattributed, which the ledger counts as a failure
        of attribution, as is work fed into the plain body-force input."""
        energy = parts.energy
        dtype = jnp.asarray(energy["krylov"]).dtype
        rows = books.laws(held)
        check_deliveries(held)
        work, unattributed, drive_scale = self.drive_work(held, boundary_inputs, rows, dt, energy["momentum"],
                                                          energy["momentum_size"])
        hosted_work = {"viscous": -energy["viscous"], "penalty": -energy["penalty"]}
        named: set[str] = set()
        for delivery in held:
            if delivery.kind == books.HOSTED:
                work = work.at[rows.index(delivery.law)].add(hosted_work[delivery.field])
                named.add(delivery.field)
        unattributed = unattributed + sum((value for field, value in hosted_work.items() if field not in named),
                                          start=jnp.zeros((), dtype=dtype))
        terms = jnp.stack([energy[name] for name in self.scheme.ledger_terms])
        return books.StepAccount(work=work, hosted=jnp.zeros((len(rows), 2), dtype=dtype), unattributed=unattributed,
                                 terms=terms, storage_start=self.storage(p, state), storage_end=self.storage(p, ended),
                                 scale=energy["scale"] + drive_scale, operators={})

    # -- at the build and after the run ---------------------------------------------------------------------------
    def built_lines(self, start: Mapping[str, Any] | None, closest_approach: float | None) -> list[str]:
        """The validity lines fixed when the node is built, each reported where it is crossed (never a refusal): the
        lumen's resolution in cells under the band's calibration, and float32's rounding floor near a steady state
        above the monolithic solve's tolerance (a node with no such solve has no such line)."""
        del start, closest_approach
        c = self.constants()
        cells = 2.0 * float(c["tube_radius"]) / float(c["spacing"])
        lines: list[str] = []
        if cells < CALIBRATED_CELLS:
            lines.append(f"the lumen is {cells:.3g} cells across, under the {CALIBRATED_CELLS:g} at and above which "
                         f"the band's wall shift is calibrated on straight tubes (0.3 to 0.4 % of the flux at 32); "
                         f"provisional until a stenosis case measures it, the band added +2.1 % of the flux at 16 "
                         f"and +10.8 % at 8, and under about 5 cells the build refuses")
        nu = float(c["viscosity"]) / float(c["density"])
        if run_dtype() == jnp.float32 and self.scheme.stores_inertia and "tolerance" in c:
            tau = float(c["tube_radius"]) ** 2 / (STALL_SECTION * nu)
            floor = float(np.finfo(np.float32).eps) * tau / self.delta_t
            if floor > float(c["tolerance"]):
                lines.append(f"float32's rounding floor near a steady state, eps x tau/dt = {floor:.3g} (tau = R^2 / "
                             f"((pi^2/4) nu) = {tau:.3g} s), is above the solve's tolerance {c['tolerance']:.3g}: a "
                             f"run near its steady state stops short of it by about that share; run in float64 "
                             f"(JAX_ENABLE_X64=1), or with a larger dt")
        return lines

    def run_problem(self, end: int) -> Problem | None:
        """A run that would end past the last row of the derivative solves' per-step reports (``solve_report_rows``)
        is refused before it runs: a row is a step's, and a step past the last would be summed into it."""
        rows = int(self.constants()["solve_report_rows"])
        if rows == 0 or end <= rows:
            return None
        return Problem("flow.solve-report-rows",
                       f"the derivative solves' reports are kept a row a step for {rows} steps, and this run would end "
                       f"at step {end}",
                       f"build with FVM(solve_report_rows={end}) or more, or solve_report_rows=0 to sum them over the "
                       f"run")

    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """The step the CFL check stopped the run at, with the CFL it read and the timestep that keeps the line. A state
        that is not finite is the run's own refusal's to name (the first node and field that went so), so none here."""
        step = int(np.asarray(state[STOP_FIELD])) if STOP_FIELD in state else -1
        if step < 0 or not all(np.isfinite(np.asarray(state[f"velocity_{c}"])).all() for c in "xyz"):
            return None
        cfl = float(np.asarray(state["stopped_cfl"]))
        line = float(self.scheme.cfl_limit(self.constants()["tube_axis"]))
        finite = math.isfinite(cfl) and cfl > 0.0
        safe = self.delta_t * line / cfl if finite else float("nan")
        return Problem(
            "flow.cfl-exceeded",
            f"at its step {step} (t = {step * self.delta_t:.6g} s) the flow's CFL number max|u| dt / h read {cfl:.4g}, "
            f"past the {self.scheme.name} scheme's line of {line:.4g} along the vessel's axis: past it the scheme's "
            f"explicit convection is not stable, so the flow's state was held from that step and the run ended at the "
            f"end of its chunk",
            (f"use a timestep under dt <= {safe:.4g} s, the line at the speed the flow had reached by then: a flow "
             f"still speeding up needs a smaller one (for a steady drive g, the Poiseuille centreline speed g R^2 / "
             f"(4 nu) bounds it); or dt='auto', the step from the drive's predicted peak; or the Stokes approximation "
             f"FVM(stokes=True) where convection is negligible")
            if finite else "use a smaller timestep, dt=..., or dt='auto', or the Stokes approximation FVM(stokes=True)")


def check_deliveries(held: Sequence[books.LedgerDelivery]) -> None:
    """Refuse a delivery the flow does not take: a hosted law other than its viscosity and its wall's band, or a
    delivered one into anything but its drive's inputs (``fluid``'s loads)."""
    for delivery in held:
        hosted = delivery.kind == books.HOSTED and delivery.field in HOSTED_FIELDS
        driven = delivery.kind == books.LOAD and delivery.field in DRIVE_FIELDS
        if not (hosted or driven):
            raise ValueError(f"the flow node hosts {list(HOSTED_FIELDS)} and is driven through {list(DRIVE_FIELDS)}: "
                             f"it takes no delivery {delivery}")


__all__ = ["BYPASS_LINE", "CALIBRATED_CELLS", "DRIVE_FIELDS", "GEOMETRY", "HOSTED_FIELDS", "STALL_LINE", "STALL_SECTION", "STOKES_REYNOLDS",
           "STOP_FIELD", "FlowInputs", "FlowState", "ImplicitFlowNode", "MeteredFlowState", "StepParts",
           "check_deliveries"]
