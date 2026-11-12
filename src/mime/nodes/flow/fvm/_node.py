"""The implicit flow node: the FVM's field on a periodic box, stepped by a scheme of the monolithic projected CG's
stage kind (IMEX Euler, or its steady limit), with its wall, its drive, its solve report, its per-step CFL check and
its step identity.

**What a step computes, all in the program from the step's parameters** (no value derived from a parameter is made at
build: round 3, item 2): the wall's signed distance, a straight tube's from its centre, axis and radius (data, item 3),
or a tube tree's from its points, radii and fillet (``mime.nodes.flow.fvm.TreeSolid``: the union's exact distance where
a field is not flat, its candidates run where its check fails, chosen in the program; only how many may run is held at
build, a rung, as the cell counts are held; constant over a run, so a chunk makes it once, the node's run-constant),
the band's indicator and the regions' inverse permeability on each face set; the resistivity ``nu x sum(1/K)`` and Jacobi's
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
:data:`BYPASS_LINE`); with a tube tree's wall, its distance not exact somewhere (``wall_inexact``: the union's counts,
``wall_unrefined``, ``wall_unconverged`` and ``wall_unplanned``, are each step's fields). The resolution in cells and
float32's rounding floor are fixed when the node is built, so the build reports them (:meth:`built_lines`), with a tube
tree's wall's resolution (``mime.scene.wall_resolution``: its least radius of curvature and its narrowest lumen in
cells against their lines).

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
``|N_Krylov| / D_visc`` per step (the stall detector, flagged above :data:`STALL_PER_TOLERANCE` times the solve's
tolerance)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import Any, ClassVar, NamedTuple, cast, override

import jax
import numpy as np
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, \
    run_constants_given, run_dtype
from mime.core import ledger as books
from mime.core._node import STAGNATED, rechecked
from mime.nodes.flow.fvm._band import TreeSolid, TubeSolid, WallFields, band_named, interior_depth, \
    negative_resistivity, plan_rung, wall_fields
from mime.nodes.flow.fvm._flags import FLAGS, STALL_PER_TOLERANCE
from mime.nodes.flow.fvm._exchange import box_side, faxen, held_spectrum, steady_response
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._partners import DELIVERIES, SOLUTION_COLUMNS, BodyPartner, Partnered, combined, \
    partnered_step
from mime.nodes.flow.fvm._schwarz import BOOKS as SCHWARZ_BOOKS, INNER_SHARE, RESIDUAL_LINE, SCHWARZ_TOLERANCE, Composed, \
    Prepared, composed, image_terms, law_books, own_decayed, prepared, stencils
from mime.nodes.flow.fvm._schwarz_host import PREFIX as SCHWARZ, REPORT_BLOCKS, WIDTH as SCHWARZ_WIDTH, SchwarzHost, \
    inputs_of as schwarz_inputs, starts as schwarz_starts
from mime.nodes.flow.fvm._exchange import OwnBox, own_box
from mime.nodes.hydro.near_field._gmres import Stopping as GmresStopping
from mime.nodes.hydro.mrs._quadrature import scaled_least_eigenvalue
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
#: Half of each dtype's digits, sqrt(eps): the share of its scale a report's denominator must exceed (:meth:`ratio`).
HALF_DIGITS = {np.dtype(np.float32): math.sqrt(float(np.finfo(np.float32).eps)),
               np.dtype(np.float64): math.sqrt(float(np.finfo(np.float64).eps))}
#: What a geometry leaf's derivative is worth, where a user reads it (its parameter's description, the entry's gradient
#: record and validity line): no gate has earned it.
GEOMETRY = ("its derivative is unverified: the tube's radius, centre and axis are not in the gradient gate yet (its "
            "loss is to be the box's whole axial flux, since the lumen's mask moves with the radius); through the "
            "band a geometry derivative carries a ripple within a cell of about +-20 % at D/h 8 to 16 and +-10 % at "
            "32 (measured on a static tube)")
#: The leaves a tube tree's wall is made from in the step, by the names its lumen reads them under.
TREE_LEAVES = {"tree_points": "points", "tree_radii": "radii", "tree_fillet": "fillet"}
#: A tube tree's report fields: the union's counts over the faces (``mime.nodes.flow.fvm.TreeSolid``).
WALL_COUNTS = ("wall_unrefined", "wall_unconverged", "wall_unplanned")
#: A tube tree's run-constants (``MimeNode.run_constants``): its wall's distance on the three face sets' faces in order,
#: and its counts, made from its leaves once a chunk.
TREE_WALL = "tree_wall"
TREE_WALL_COUNTS = "tree_wall_counts"
#: The fields a held state still writes: the stop's own record (with a Schwarz partner, the indefinite response that
#: stopped it and its margin).
_STOP_RECORD = frozenset({STOP_FIELD, "stopped_cfl", "schwarz_indefinite", "schwarz_0_margin"})
#: The hosted laws a metered node books, by the field their delivery names: the fluid's viscosity and the wall's band.
HOSTED_FIELDS = ("viscous", "penalty")
#: The inputs a law driving the fluid delivers into: its body force, read where the scheme reads its drive at the
#: step's end, and its waveform, read where the scheme evaluates its drive at its own stage instants.
DRIVE_FIELDS = ("body_force", "waveform")
#: The inputs an exchange partner's law delivers its scattered force into, a force density (N/m^3) on each face set
#: (``mime.nodes.flow.fvm._exchange``): a node built with ``exchange=True`` takes them.
EXCHANGE_FIELDS = ("exchange_x", "exchange_y", "exchange_z")
#: A point body's inputs a flow holding it in its one elimination takes (``partner_<b>_<name>``, each with its shape and
#: units): its pose and its own step's row at the step's start (``body.step_row``: its stepping mass beside its free
#: load, as its dynamics publishes it), and the loads and resistances of its other laws, summed (the build delivers
#: each to the flow as to the body: ``mime.build._construct``).
PARTNER_INPUTS = {"pose": ((7,), "m; 1"), "step_row": ((6, 7), "kg; N"), "loads": ((6,), "N; N m"),
                  "resistances": ((6, 6), "N s/m; N s; N m s")}
#: The inputs among them the build sums each of the body's other laws' deliveries into, by the body's input.
PARTNER_SUMS = {"applied_wrench": "loads", "background_load": "loads", "resistance": "resistances"}
#: The prefix of a held point body's inputs, fields and hosted row.
PARTNER = "partner_"
#: The fields a flow holding point bodies returns to each (``partner_<b>_<name>``): the load and response it is handed.
PARTNER_HANDED = ("resistance", "load")
#: The count of held point bodies from which the build report names the lagged exchange as the cheaper route: six unit
#: solves a body a step against the lagged exchange's none. Measured through the node (F3's second PR, float64, 40 um
#: spheres in the 1 mm periodic tube at D/h 16, a 4 vCPU runner), one pass's step took 4.9, 6.8 and 9.5 times the
#: lagged one's at 1, 2 and 5 bodies, and its compile 3.5, 5.4 and 9.7 times: the line is where both reach about ten
#: times (a declared line on those ratios; ten bodies unmeasured).
HELD_SUGGESTS_LAGGED = 5


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


class _WallCounts(TypedDict):
    """A tube tree's wall's report: where its distance was not exact, counted over the faces (failing faces in the band's
    reach past the plan's size; chosen faces whose candidates did not stand; pairs of tubes within reach the union does
    not hold as meeting), and its flag."""

    wall_unrefined: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    wall_unconverged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    wall_unplanned: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    wall_inexact: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class TreeFlowState(FlowFields, _WallCounts, closed=True):
    """A flow walled by a tube tree, without the energy ledger: the flow's fields and the wall's report."""


class FlowRecords(books.LedgerRecords):
    """With the energy ledger attached: the stall detector, ``|N_Krylov| / D_visc``, and its flag, beside the ledger's
    records (``mime.core.ledger.LedgerRecords``)."""

    stall_ratio: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stall_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class MeteredFlowState(FlowFields, FlowRecords, closed=True):
    """A metered flow's state: the flow's fields, the stall detector and the ledger's records."""


class MeteredTreeFlowState(FlowFields, _WallCounts, FlowRecords, closed=True):
    """A metered flow walled by a tube tree: the flow's fields, the wall's report, the stall detector and the ledger's
    records."""


class PartnerFields(TypedDict):
    """A flow holding point bodies in its one elimination (:meth:`ImplicitFlowNode.partner_fields`): beside these, the
    load and response each body is handed, ``partner_<b>_load`` and ``partner_<b>_resistance``, a state's extra items,
    their units the edges' that deliver them."""

    #: Each body's six unit fields a face set, the velocity a unit force (the first three of each six, m/(s N)) or a
    #: unit couple (the last three, 1/(s N)) drives over a step: one block, as verify judges it.
    partner_units_x: ReadOnly[Annotated[Float[Array, "units nx ny nz"], Units("m/(s N), 1/(s N)")]]  # noqa: F722
    partner_units_y: ReadOnly[Annotated[Float[Array, "units nx ny nz"], Units("m/(s N), 1/(s N)")]]  # noqa: F722
    partner_units_z: ReadOnly[Annotated[Float[Array, "units nx ny nz"], Units("m/(s N), 1/(s N)")]]  # noqa: F722
    #: What the step solved for each body (``mime.vocabulary.FLOW_PARTNER_SOLUTION``, its units).
    partner_solution: ReadOnly[Annotated[Float[Array, "bodies 21"],  # noqa: F722 - a jaxtyping shape
                                         Units("N; N m; m/s; rad/s; m/s; 1/(m s); rad/s")]]
    partner_reach: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class PartneredFlowState(FlowFields, PartnerFields, extra_items=ReadOnly[Float[Array, "..."]]):
    """A flow holding point bodies: the flow's fields, the bodies' and their handed loads and responses (whatever else
    it holds, the ledger's records or an open vessel's fields, another of its states names)."""


class MeteredPartneredFlowState(FlowFields, FlowRecords, PartnerFields, extra_items=ReadOnly[Float[Array, "..."]]):
    """A metered flow holding point bodies: the flow's fields, the stall detector, the ledger's records and the bodies'
    fields."""


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
    """The flow field of a medium on a periodic box holding a straight tube (``tube_centre``, ``tube_axis``,
    ``tube_radius``) or a tube tree periodic along z (``tree_points``, ``tree_radii``, ``tree_fillet``, its
    ``tree_segments`` structural), stepped by a scheme of the monolithic projected CG's stage kind (the module states
    the step)."""

    LEDGER_TERMS = ("backward_euler", "convection", "krylov")
    #: The grid kind's own terms its identity books after the scheme's (none on a periodic box).
    GRID_TERMS: ClassVar[tuple[str, ...]] = ()
    #: The Schwarz partner's host (``mime.nodes.flow.fvm._schwarz_host``), where the node holds one; none on a node of
    #: a scheme that holds none (the explicit stage kind's node, which builds no host).
    schwarz: SchwarzHost | None = None
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
                 viscosity: float, density: float, tube_centre: Sequence[float] | None = None,
                 tube_axis: Sequence[float] | None = None, tube_radius: float | None = None, tree_points: Any = None,
                 tree_radii: Any = None, tree_fillet: float | None = None,
                 tree_segments: Sequence[Sequence[int]] | None = None, scheme: str = "imex_euler",
                 convection: bool = True, tolerance: float | None = None,
                 at_floor: bool | None = None, cap: int = DEFAULT_CAP, check_every: int = 20, band: str = "smooth_2h_0.3h",
                 interior: str = "0.1h",
                 preconditioner: str = "jacobi", scaling_start: bool = True, periodic_images: str = "own",
                 solve_report_rows: int = 0, wall_name: str = "", solve_report: Any = None,
                 initial_velocity_x: Any = None, initial_velocity_y: Any = None, initial_velocity_z: Any = None,
                 exchange: bool = False, partner_radius: Any = None, partner_widths: Sequence[float] = (),
                 partner_ids: Sequence[str] = (), partner_origin: Sequence[float] = (0.0, 0.0, 0.0),
                 partner_bodies: Sequence[str] = (), ledger: Sequence[str] = (), **extra: Any) -> None:
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
        tube = [value is not None for value in (tube_centre, tube_axis, tube_radius)]
        tree = [value is not None for value in (tree_points, tree_radii, tree_fillet, tree_segments)]
        if not ((all(tube) and not any(tree)) or (all(tree) and not any(tube))):
            raise ValueError("a flow's wall is a straight tube (tube_centre, tube_axis and tube_radius) or a tube tree "
                             "(tree_points, tree_radii, tree_fillet and tree_segments), one of them whole")
        if tree_points is not None and periodic_images != "own":
            raise ValueError("a tube tree's box holds the tree with no transverse image (periodic_images='own')")
        walled: dict[str, Any]
        if tube_centre is not None:
            assert tube_axis is not None and tube_radius is not None
            walled = dict(tube_centre=tuple(float(c) for c in tube_centre),
                          tube_axis=tuple(float(c) for c in tube_axis), tube_radius=float(tube_radius))
        else:
            assert tree_segments is not None and tree_fillet is not None
            walled = dict(tree_points=np.asarray(tree_points, dtype=np.float64),
                          tree_radii=np.asarray(tree_radii, dtype=np.float64), tree_fillet=float(tree_fillet),
                          tree_segments=tuple((int(a), int(b)) for a, b in tree_segments))
        # A node made again from its own arguments (the ledger's wiring, a test's rebuild) carries the stop it was made
        # with; otherwise the stop at the floor is float32's default, where no tolerance is given. Its solve_report
        # parameter is zeros sized by solve_report_rows, made here again, so the one it is given back is not read.
        del solve_report
        extra.pop("schwarz_report", None)
        if "schwarz_members" in extra and extra.get("schwarz_tolerance") is None:
            extra["schwarz_tolerance"] = SCHWARZ_TOLERANCE[str(jnp.dtype(run_dtype()))]
        if at_floor is None:
            at_floor = tolerance is None and str(jnp.dtype(run_dtype())) in AT_FLOOR_BY_DEFAULT
        if tolerance is None:
            tolerance = default_tolerance(run_dtype())
        widths = tuple(float(w) for w in partner_widths)
        radii = np.atleast_1d(np.asarray(partner_radius if partner_radius is not None else (), dtype=np.float64))
        if len(radii) != len(widths) or len(tuple(partner_ids)) != len(widths) or \
                (partner_bodies and len(tuple(partner_bodies)) != len(widths)):
            raise ValueError(f"each point body the flow holds gives its radius, its kernel's width and its law's id: "
                             f"{len(radii)} radii, {len(widths)} widths, {len(tuple(partner_ids))} ids")
        partners: dict[str, Any] = {} if not widths else dict(
            partner_radius=radii, partner_widths=widths, partner_ids=tuple(str(i) for i in partner_ids),
            partner_origin=tuple(float(c) for c in partner_origin),
            **({"partner_bodies": tuple(str(b) for b in partner_bodies)} if partner_bodies else {}))
        initial: dict[str, Any] = {}
        for key, value in (("initial_velocity_x", initial_velocity_x), ("initial_velocity_y", initial_velocity_y),
                           ("initial_velocity_z", initial_velocity_z)):
            if value is not None:
                initial[key] = np.asarray(value)
        super().__init__(name, timestep, cells_x=int(cells_x), cells_y=int(cells_y), cells_z=int(cells_z),
                         spacing=float(spacing), viscosity=float(viscosity), density=float(density), **walled,
                         scheme=str(scheme), convection=bool(convection), tolerance=float(tolerance),
                         at_floor=bool(at_floor), cap=int(cap),
                         check_every=int(check_every), band=str(band), interior=str(interior),
                         preconditioner=str(preconditioner), scaling_start=bool(scaling_start),
                         periodic_images=str(periodic_images), solve_report_rows=int(solve_report_rows),
                         solve_report=np.zeros((max(int(solve_report_rows), 1), REPORT_SIZE)), **initial,
                         **({"schwarz_report": np.zeros(REPORT_BLOCKS)} if "schwarz_members" in extra else {}),
                         **({"wall_name": str(wall_name)} if wall_name else {}),
                         **({"exchange": True} if exchange else {}), **partners, **extra,
                         **({"ledger": tuple(ledger)} if ledger else {}))
        self.partner_spectra = {width: (held_spectrum("force", width, box_side(width), str(run_dtype())),
                                        held_spectrum("couple", width, box_side(width), str(run_dtype())))
                                for width in sorted(set(widths))}
        self.schwarz: SchwarzHost | None = None
        if "schwarz_members" in extra:
            if widths or exchange:
                raise ValueError("a flow holds its Schwarz partner beside no point body and no lagged exchange "
                                 "(flow.schwarz-bodies until the body-set host)")
            self.schwarz = SchwarzHost(name, timestep, self.constants(), float(spacing))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        fixed = "a structural setting: the box's cells were chosen with it"
        held = ("the wall is made from it in the program, once a chunk; only how many faces its candidates may run at "
                "was held at build (a wall needing more is counted, wall_unrefined); its derivative is FA8's to judge")
        made = ("made from the tree's leaves once a chunk (a run-constant) and read only as the chunk made it; its value "
                "as given is never read")
        wall = {"tube_centre": ParamSpec(units="m", description=f"the tube's axis passes through it; {GEOMETRY}"),
                "tube_axis": ParamSpec(units="1", description=f"the tube's direction, the lattice direction the box "
                                                              f"was built for; {GEOMETRY}"),
                "tube_radius": ParamSpec(units="m", bounds=(0.0, None),
                                         description=f"the lumen's radius; {GEOMETRY}"),
                "tree_points": ParamSpec(units="m", description=f"the tube tree's points, in the box's frame; {held}"),
                "tree_radii": ParamSpec(units="m", bounds=(0.0, None),
                                        description=f"each segment's radius at its start and its end; {held}"),
                "tree_fillet": ParamSpec(units="1", bounds=(0.0, None),
                                         description=f"the fillet's radius as a share of the smallest radius at a "
                                                     f"junction; {held}"),
                TREE_WALL: ParamSpec(trainable=False, units="m",
                                     description=f"the tube tree's wall's signed distance on every face; {made}"),
                TREE_WALL_COUNTS: ParamSpec(trainable=False, units="1",
                                            description=f"the tube tree's wall's counts (wall_unrefined, "
                                                        f"wall_unconverged, wall_unplanned); {made}")}
        if self.body_count:
            wall["partner_radius"] = ParamSpec(units="m", bounds=(0.0, None),
                                               description="each point body's radius, whose drag law the flow holds in "
                                                           "its step (one pass): its Stokes drag and couple and its "
                                                           "Faxén coefficients")
            wall["partner_widths"] = ParamSpec(trainable=False, units="1",
                                               description="each point body's exchange kernel's width in cells "
                                                           "(structural: its stencil's reach was held with it)")
            wall["partner_origin"] = ParamSpec(trainable=False, units="m",
                                               description="the world point of the grid's origin, which a body's pose "
                                                           "is read from (structural)")
        if self.schwarz is not None:
            wall.update(self.schwarz.hosted_specs())
            wall["schwarz_report"] = ParamSpec(trainable=False, units="1",
                                               description="zeros, never moved: the Schwarz preparation's derivative "
                                                           "solves' reports come out as its gradient")
            for key, text in (("schwarz_origin", "the world point of the grid's origin, which a body's pose is read "
                                                 "from (structural)"),
                              ("schwarz_starts", "each Schwarz body's starting pose (m; the orientation's quaternion "
                                                 "unitless), the initial preparation's (an initial condition)"),
                              ("schwarz_drive", "the drive at the first step's end, the initial preparation's "
                                                "(read from the laws bound to the medium at the build)"),
                              ("schwarz_extent", "the Schwarz body's extent, which sizes its free box (structural)"),
                              ("schwarz_box", "the free box's side in cells, where written (structural)")):
                wall[key] = ParamSpec(trainable=False, units={"schwarz_drive": "m/s^2", "schwarz_box": "1"}.get(key, "m"),
                                      description=text)
            wall["schwarz_tolerance"] = ParamSpec(trainable=False, units="1",
                                                  description="the Schwarz columns' GMRES: its relative tolerance on "
                                                              "each column's true residual; the walled solves inside "
                                                              "its products at a hundredth of it")
        held_keys = {*self.constants(), *((TREE_WALL, TREE_WALL_COUNTS) if self.walled_by_tree else ()),
                     *(self.schwarz.hosted_params() if self.schwarz is not None else ())}
        return {**super().param_specs(), **{key: spec for key, spec in wall.items() if key in held_keys},
                "spacing": ParamSpec(trainable=False, units="m", description=fixed),
                "viscosity": ParamSpec(units="Pa s", bounds=(0.0, None),
                                       description="the fluid's dynamic viscosity, mu; the step uses mu / rho"),
                "density": ParamSpec(units="kg/m^3", bounds=(0.0, None)),
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
        delivered value. Built with ``exchange=True``, the force density its exchange partners scatter onto each face
        set (:data:`EXCHANGE_FIELDS`), held over the step."""
        own = {"body_force": node_input("m/s^2", (3,), additive=True,
                                        description="a uniform body force per unit mass on the fluid"),
               "waveform": node_input("1", (WAVEFORM_SIZE,),
                                      description="the drive's waveform, read at a stage's instant where it is not the "
                                                  "step's end")}
        if self.exchanges:
            c = self.constants()
            shape = (int(c["cells_x"]), int(c["cells_y"]), int(c["cells_z"]))
            for field in EXCHANGE_FIELDS:
                own[field] = node_input("N/m^3", shape, additive=True,
                                        description="the force density the exchange partners' laws scatter onto the "
                                                    "faces of one set, on the fluid and the solid alike")
        return {**own, **self.partner_inputs(), **{
            name: node_input(units, shape, description=f"the Schwarz body's {name.rpartition('_')[2]} after its own "
                                                       f"step, which closes the step the flow prepared for it")
            for name, (shape, units) in schwarz_inputs(self.schwarz.count if self.schwarz else 0).items()}}

    @property
    def exchanges(self) -> bool:
        """Whether the node takes its exchange partners' scattered force (built with ``exchange=True``)."""
        return bool(self.constants().get("exchange", False))

    @property
    def body_count(self) -> int:
        """How many point bodies the flow holds in its one elimination (their drag laws held in its step)."""
        return len(self.constants().get("partner_widths", ()))

    def partner_inputs(self) -> dict[str, Any]:
        """Each held point body's inputs (:data:`PARTNER_INPUTS`), ``partner_<b>_<name>``."""
        return {f"partner_{b}_{name}": node_input(units, shape, additive=name in PARTNER_SUMS.values(),
                                                  description=f"held point body {b}'s {name.replace('_', ' ')}")
                for b in range(self.body_count) for name, (shape, units) in PARTNER_INPUTS.items()}

    def row_inputs(self) -> tuple[str, ...]:
        """The inputs each held body's row is read into (its pose and step's row, its other laws' loads and
        resistances): state the flow reads, no delivery to it, so the energy ledger books none of them here."""
        return (*self.partner_inputs(), *schwarz_inputs(self.schwarz.count if self.schwarz else 0))

    def body_partners(self, p: Mapping[str, Any], boundary_inputs: Mapping[str, Array], dt: Any,
                      dtype: Any) -> list[BodyPartner]:
        """Each held point body's law and own row over the step (``mime.nodes.flow.fvm._partners``): its point from
        its pose (less the grid's origin), its Stokes drag and couple less the kernel's own steady self-responses in a
        free box by the grid's operators (``M0 - M_inf``, ``K0 - K_inf``: the constant correction), Faxén's
        coefficient beyond the kernel's own second moment, and its own step's matrix and right-hand side from what it
        publishes and its other laws deliver."""
        c = self.constants()
        h = float(c["spacing"])
        mu = as_run_array(p["viscosity"])
        radii = as_run_array(p["partner_radius"])
        origin = as_run_array(p["partner_origin"])
        found: list[BodyPartner] = []
        for b, width in enumerate(c["partner_widths"]):
            a = radii[b]
            spectrum, dipole = self.partner_spectra[width]
            coefficient = faxen(a, width * h, 1.0 / 6.0)
            steady = steady_response(spectrum, coefficient / h ** 2, coefficient / h ** 2) / (mu * h)
            spin = steady_response(dipole, images=False) / (mu * h ** 3)
            translation = 1.0 / (6.0 * jnp.pi * mu * a) - steady
            rotation = 1.0 / (8.0 * jnp.pi * mu * a ** 3) - spin
            law = jnp.concatenate([jnp.full(3, translation, dtype=dtype), jnp.full(3, rotation, dtype=dtype)])

            def given(name: str, b: int = b) -> Array:
                shape = PARTNER_INPUTS[name][0]
                return jnp.asarray(boundary_inputs.get(f"partner_{b}_{name}", jnp.zeros(shape, dtype=dtype)),
                                   dtype=dtype)

            row = given("step_row")
            found.append(BodyPartner(given("pose")[:3] - origin, float(width), coefficient, coefficient, law,
                                     row[:, :DELIVERIES] / dt + given("resistances"), row[:, DELIVERIES] + given("loads")))
        return found

    def exchanged(self, boundary_inputs: Mapping[str, Array], dtype: Any) -> Velocity | None:
        """The exchange partners' force density on each face set (N/m^3), as the step reads it (every law's own input
        summed into it where the ledger is attached); zero where nothing feeds it; ``None`` for a node with none."""
        if not self.exchanges:
            return None
        c = self.constants()
        zero = jnp.zeros((int(c["cells_x"]), int(c["cells_y"]), int(c["cells_z"])), dtype=dtype)
        held = [jnp.asarray(boundary_inputs[field], dtype=dtype) if field in boundary_inputs else zero
                for field in EXCHANGE_FIELDS]
        return (held[0], held[1], held[2])

    # -- the pieces of a step ------------------------------------------------------------------------------------
    @property
    def scheme(self) -> TimeScheme[Any]:
        return scheme_named(self.constants()["scheme"])

    def grid(self, p: Mapping[str, Any]) -> MACGrid:
        c = self.constants()
        return MACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), p["spacing"])

    @property
    def walled_by_tree(self) -> bool:
        """Whether the wall is a tube tree's (its leaves ``tree_*``) rather than a straight tube's."""
        return "tree_points" in self.constants()

    def tree(self) -> Any:
        """The tube tree as built (``mime.scene.TubeTree``): its structure, which tubes meet, and the values the held
        faces and the plan were made at."""
        from mime.scene import TubeTree

        c = self.constants()
        return TubeTree.si(np.asarray(c["tree_points"], dtype=np.float64), tuple(c["tree_segments"]),
                           np.asarray(c["tree_radii"], dtype=np.float64), float(c["tree_fillet"]))

    @cached_property
    def _plan_sizes(self) -> dict[str, int]:
        return {}

    def plan_size(self) -> int:
        """How many faces a tube tree's wall may run the union's candidates at (``mime.nodes.flow.fvm.plan_rung``): the
        rung above the count the tree's values as built choose (:meth:`chosen_count`)."""
        return plan_rung(self.chosen_count())

    def chosen_count(self) -> int:
        """How many faces the tree's values as built choose for the union's candidates (``ExactUnion.chosen``), made
        once per dtype."""
        dtype = str(jnp.dtype(run_dtype()))
        cache = self._plan_sizes
        if dtype not in cache:
            tree = self.tree()
            c = self.constants()
            grid = MACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), float(c["spacing"]))
            # Made from the values built, outside whatever traces the step that first asks for it.
            with cast(Any, jax).ensure_compile_time_eval():
                points = jnp.concatenate([grid.face_points(k, run_dtype()).reshape(-1, 3) for k in range(3)])
                chosen = tree.lumen().chosen(tree.parameters(), points, self.wall_floor(grid.spacing))
                cache[dtype] = int(np.sum(np.asarray(chosen)))
        return cache[dtype]

    def wall_floor(self, spacing: Any) -> Any:
        """The depth in the fluid below which every field the step reads of the wall is flat: the band's fluid half."""
        return -0.5 * band_named(self.constants()["band"]).width * spacing

    def tree_solid(self, p: Mapping[str, Any], grid: MACGrid) -> TreeSolid:
        """The tube tree's solid at the step's leaves: its wall as the chunk made it (the run-constants), where it did;
        else made in the step."""
        given = (p[TREE_WALL], p[TREE_WALL_COUNTS]) if run_constants_given(p) and TREE_WALL in p else None
        return TreeSolid(self.tree().lumen(), self.tree_leaves(p), self.wall_floor(grid.spacing), self.plan_size(),
                         given)

    @override
    def params_pytree(self) -> dict[str, Any]:
        """The leaves; a tube tree's with its run-constants' (:data:`TREE_WALL`, :data:`TREE_WALL_COUNTS`): zeros,
        which a chunk makes from the tree's leaves before its steps read them."""
        inherited: Any = super()
        leaves = cast(dict[str, Any], inherited.params_pytree())
        if self.schwarz is not None:
            leaves = {**leaves, **self.schwarz.hosted_params()}
        if not self.walled_by_tree:
            return leaves
        c = self.constants()
        faces = 3 * int(c["cells_x"]) * int(c["cells_y"]) * int(c["cells_z"])
        return {**leaves, TREE_WALL: jnp.zeros(faces, dtype=run_dtype()),
                TREE_WALL_COUNTS: jnp.zeros(len(WALL_COUNTS), dtype=run_dtype())}

    @override
    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Array]:
        """A tube tree's wall, made once a chunk from its leaves (:class:`TreeSolid`): its distance on every face and
        its counts. None for a straight tube's, whose wall the step makes (a few operations a face)."""
        held = self.schwarz.run_constants(params) if self.schwarz is not None else {}
        if not self.walled_by_tree or TREE_WALL not in params:
            return held
        grid = self.grid(params)
        points = (grid.face_points(0, run_dtype()), grid.face_points(1, run_dtype()), grid.face_points(2, run_dtype()))
        distance, counts = self.tree_solid(params, grid).made(points)
        return {**held, TREE_WALL: distance.astype(jnp.asarray(params[TREE_WALL]).dtype),
                TREE_WALL_COUNTS: counts.astype(jnp.asarray(params[TREE_WALL_COUNTS]).dtype)}

    @override
    def made_again(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The Schwarz partner's hosted near field's held free parts made again (``JointDragNode.made_again``)."""
        return self.schwarz.made_again(params) if self.schwarz is not None else {}

    def tree_leaves(self, p: Mapping[str, Any]) -> dict[str, Array]:
        """The tree's leaves as its lumen reads them, from the step's parameters."""
        return {name: as_run_array(p[leaf]) for leaf, name in TREE_LEAVES.items()}

    def vessel_axis(self, p: Mapping[str, Any]) -> Array:
        """The direction the reported fluxes are taken along: the tube's axis, or a tree's period, z."""
        if self.walled_by_tree:
            return as_run_array((0.0, 0.0, 1.0))
        return as_run_array(p["tube_axis"])

    def vessel_radius(self, p: Mapping[str, Any]) -> Array:
        """The radius the reported Reynolds number is taken by: the tube's, or a tree's narrowest (the box's whole
        flux through its narrowest lumen, an upper bound)."""
        if self.walled_by_tree:
            return jnp.min(as_run_array(p["tree_radii"]))
        return as_run_array(p["tube_radius"])

    def cfl_axis(self, p: Mapping[str, Any]) -> Any:
        """The direction the scheme's CFL line is read along: the tube's axis; a tree's flow along any direction, so
        the diagonal, where the line is least (``g(n)`` a third)."""
        if self.walled_by_tree:
            return as_run_array((1.0, 1.0, 1.0))
        return p["tube_axis"]

    def solids(self, p: Mapping[str, Any], grid: MACGrid) -> Sequence[TubeSolid | TreeSolid]:
        if self.walled_by_tree:
            return [self.tree_solid(p, grid)]
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

    def with_exchange(self, p: Mapping[str, Any], force: Callable[[int, Any], Velocity],
                      boundary_inputs: Mapping[str, Array], dtype: Any) -> Callable[[int, Any], Velocity]:
        """``force`` with the exchange partners' force density over the fluid's density added at every stage (held
        over the step: what a lagged partner delivered for it); ``force`` itself for a node with no partners."""
        density = self.exchanged(boundary_inputs, dtype)
        if density is None:
            return force
        rho = as_run_array(p["density"])
        per_mass = (density[0] / rho, density[1] / rho, density[2] / rho)

        def forced(stage: int, time: Any) -> Velocity:
            base = force(stage, time)
            return (base[0] + per_mass[0], base[1] + per_mass[1], base[2] + per_mass[2])

        return forced

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
        held = node_fields(FlowState(**common)) if not self.walled_by_tree else node_fields(
            TreeFlowState(**common, **self.wall_report(jnp.zeros(len(WALL_COUNTS), dtype=run_dtype()))))
        held = {**held, **self.partner_fields(), **self.schwarz_initial()}
        if not self.metered:
            return held
        z = as_run_array(0.0)
        return books.with_records(self, {**held, "stall_ratio": z, "stall_flagged": z, **self.schwarz_book_fields()})

    def partner_fields(self) -> dict[str, Array]:
        """The held point bodies' fields at the run's start (none without them): each body's six unit fields a face
        set (``partner_units_<x|y|z>``, the last step's, which warm-start this one's), the solution a row a body
        (``partner_solution``: its force and couple, its twist as the field closed it, its reading, ``SOLUTION``), the
        load and response each is handed (``partner_<b>_load``, ``partner_<b>_resistance``) and whether a body's kernel
        reached an open plane in the step (``partner_reach``)."""
        count = self.body_count
        if not count:
            return {}
        grid = self.grid(self.constants())
        dtype = run_dtype()
        held: dict[str, Array] = {f"partner_units_{x}": jnp.zeros((DELIVERIES * count, *grid.shape(k)), dtype=dtype)
                                  for k, x in enumerate("xyz")}
        held["partner_solution"] = jnp.zeros((count, SOLUTION_COLUMNS), dtype=dtype)
        for b in range(count):
            held[f"partner_{b}_resistance"] = jnp.zeros((DELIVERIES, DELIVERIES), dtype=dtype)
            held[f"partner_{b}_load"] = jnp.zeros(DELIVERIES, dtype=dtype)
        held["partner_reach"] = jnp.zeros((), dtype=dtype)
        return held

    def held_partner_units(self, state: Mapping[str, Array]) -> list[Velocity]:
        """The held bodies' unit fields the last step solved, in order (each body's six)."""
        return [(state["partner_units_x"][k], state["partner_units_y"][k], state["partner_units_z"][k])
                for k in range(DELIVERIES * self.body_count)]

    def partner_ended(self, grid: MACGrid, bodies: Sequence[BodyPartner], partnered: Partnered,
                      held_units: int) -> dict[str, Array]:
        """The held bodies' fields at the step's end from its one elimination: the last ``DELIVERIES`` times the
        bodies' unit fields (the openings' come first), the solution, what each body is handed and the reach flag."""
        dtype = partnered.deliveries.dtype
        units = partnered.units[held_units:]
        count = len(bodies)
        first = int(partnered.deliveries.shape[0]) - DELIVERIES * count
        forces = jnp.reshape(partnered.deliveries[first:], (count, DELIVERIES))
        ended: dict[str, Array] = {f"partner_units_{x}": jnp.stack([unit[k] for unit in units])
                                   for k, x in enumerate("xyz")}
        ended["partner_solution"] = jnp.concatenate([forces, partnered.twists, partnered.kernels], axis=1)
        for b in range(count):
            ended[f"partner_{b}_resistance"] = partnered.handed_resistance[b]
            ended[f"partner_{b}_load"] = partnered.handed_load[b]
        reach = getattr(grid, "stencil_reaches_plane", None)
        reached = jnp.zeros((), dtype=bool) if reach is None else jnp.any(jnp.stack(
            [jnp.any(reach(body.point[None, :], body.width)) for body in bodies]))
        ended["partner_reach"] = reached.astype(dtype)
        return ended

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
        if self.schwarz is not None:
            return self.schwarz_stepped(p, state, boundary_inputs, dt)
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
        bodies = self.body_partners(p, boundary_inputs, dt, dtype) if self.body_count else []
        held_units = self.held_partner_units(state) if bodies else []
        partnered: dict[str, Partnered] = {}

        def solve(shift: Any, rhs: Velocity, guess: Velocity | None) -> tuple[Velocity, Array]:
            def one(b: Velocity, start: Velocity | None) -> tuple[Velocity, Array]:
                return projected_solve(grid, operator(shift), self.diagonal_of(p, grid, sigma, shift), b,
                                       start if c["scaling_start"] else None, stopping, tap, preconditioner,
                                       magnitude(shift))

            if not bodies:
                return one(rhs, guess)
            result = partnered_step(one, grid, rhs, guess, bodies, held_units, as_run_array(p["density"]))
            partnered["step"] = result
            return result.solved, result.report

        context = StepContext(grid, operator, self.with_exchange(p, self.forcing(fields, boundary_inputs, scheme, dtype),
                                                                 boundary_inputs, dtype),
                              bool(c["convection"]), solve, self.kinematic(p), sigma)
        velocity = _velocity(state)
        time = steps.astype(dtype) * dt
        speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in velocity]))
        cfl = speed * dt / grid.spacing
        checked = bool(c["convection"]) and scheme.cfl_line is not None
        stopped_before = jnp.asarray(state["stopped_at"]) >= 0
        # Past the line unless provably at or under it: a NaN or an infinite CFL is past it too.
        past = ~(cfl <= scheme.cfl_limit(self.cfl_axis(p))) if checked else jnp.asarray(False)
        ended_state, stage = scheme.step(context, _scheme_state(state), time, dt)
        ended_velocity: Velocity = ended_state["velocity"]
        report = stage.report
        flagged = (report[2] + report[3] + report[4] > 0).astype(dtype)
        axis = self.vessel_axis(p)
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, ended_velocity)
        ratio = self.ratio(solid, lumen, lumen_scale)
        reynolds = 2.0 * jnp.abs(lumen) / (jnp.pi * self.vessel_radius(p) * self.kinematic(p))
        stokes = not bool(c["convection"])
        solved_partners = partnered.get("step")
        energy = self.energy_terms(p, grid, velocity, stage, sigma, fields, dt, solved_partners) if self.metered else {}
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
        held = node_fields(FlowState(**ended)) if fields.counts is None else node_fields(
            TreeFlowState(**ended, **self.wall_report(fields.counts)))
        if solved_partners is not None:
            held = {**held, **self.partner_ended(grid, bodies, solved_partners, 0)}
        if self.metered:
            stall = self.stall_ratio(energy)
            line = STALL_PER_TOLERANCE[jnp.dtype(dtype).name] * as_run_array(p["tolerance"])
            held = {**held, "stall_ratio": stall, "stall_flagged": (stall > line).astype(dtype)}
        return held, StepParts(stage, fields, energy)

    # -- the Schwarz partner (F5: ``mime.nodes.flow.fvm._schwarz_host`` states the order) ----------------------------
    def stopping_of(self, p: Mapping[str, Any], grid: MACGrid, tolerance: Any) -> Stopping:
        """The CG's stop at ``tolerance`` with the node's own cap, checks and window (a recheck's where the step is
        one)."""
        c = self.constants()
        stopping = Stopping(tolerance=tolerance, at_floor=bool(c["at_floor"]), cap=int(c["cap"]),
                            check_every=int(c["check_every"]),
                            window=stagnation_window(grid.cells, int(c["check_every"])))
        return recheck_stopping(stopping) if rechecked(p) else stopping

    def walled_solve(self, p: Mapping[str, Any], grid: MACGrid, sigma: Velocity, shift: Any, stopping: Stopping,
                     tap: Array) -> Callable[..., tuple[Velocity, Array]]:
        """One solve of the step's operator, ``walled(rhs, guess=None) -> (u, report)``."""
        operator = self.operator_of(p, grid, sigma)(shift)
        diagonal = self.diagonal_of(p, grid, sigma, shift)
        magnitude = self.magnitude_of(p, grid, sigma)(shift)
        preconditioner = preconditioner_named(self.constants()["preconditioner"])

        def walled(rhs: Velocity, guess: Velocity | None = None) -> tuple[Velocity, Array]:
            return projected_solve(grid, operator, diagonal, rhs, guess, stopping, tap, preconditioner, magnitude)

        return walled

    def schwarz_box(self) -> OwnBox:
        host = self.schwarz
        assert host is not None
        return own_box(host.side, host.spacing, run_dtype())

    def schwarz_forcing(self, fields: WallFields, boundary_inputs: Mapping[str, Array],
                        drive: Array | None, dtype: Any) -> Callable[[int, Any], Velocity]:
        """The drive the prepared step reads at its instant: the delivered waveform evaluated there (a declared
        schedule's coefficients are the run's), or ``drive`` itself (the initial preparation's, as the build gives it)."""
        if drive is not None:
            held = drive.astype(dtype)
            return lambda stage, time: self.drive(fields, held)
        coefficients = jnp.asarray(boundary_inputs.get("waveform", jnp.zeros(WAVEFORM_SIZE, dtype=dtype)), dtype=dtype)
        return lambda stage, time: self.drive(fields, waveform_at(coefficients, time).astype(dtype))

    def schwarz_prepare(self, p: Mapping[str, Any], velocity: Velocity, steps: Array, dt: float,
                        force: Callable[[int, Any], Velocity], pose: Array, own: Velocity,
                        image: Array) -> tuple[Velocity, StageParts, Prepared, Array]:
        """The next step's preparation from the field ``velocity`` after ``steps`` steps: the scheme's step with no
        body's force (its base), and the seven columns at ``pose`` with the box's field ``own`` and image term ``image``
        carried a step; with the scheme's stage and the handed response's margin (its least scaled eigenvalue)."""
        host = self.schwarz
        assert host is not None
        dtype = run_dtype()
        grid = self.grid(p)
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        scheme: TimeScheme[StageParts] = self.scheme
        shift = 1.0 / dt if scheme.stores_inertia else 0.0
        tap = self.report_row(p, steps)
        flow = self.walled_solve(p, grid, sigma, shift, self.stopping_of(p, grid, as_run_array(p["tolerance"])), tap)
        context = StepContext(grid, self.operator_of(p, grid, sigma), force, bool(self.constants()["convection"]),
                              lambda shift, rhs, guess: flow(rhs, guess if self.constants()["scaling_start"] else None),
                              self.kinematic(p), sigma)
        ended, stage = scheme.step(context, {"velocity": velocity}, steps.astype(dtype) * dt, dt)
        base: Velocity = ended["velocity"]
        box = self.schwarz_box()
        nu, mu, rho = self.kinematic(p), as_run_array(p["viscosity"]), as_run_array(p["density"])
        now, decay = image_terms(host.side, host.spacing, shift, nu, mu)
        body = host.body(p, pose)
        held = stencils(grid, box, body.placed.force - as_run_array(p["schwarz_origin"]), SCHWARZ_WIDTH)
        tolerance = as_run_array(p["schwarz_tolerance"])
        inner = self.walled_solve(p, grid, sigma, shift, self.stopping_of(p, grid, INNER_SHARE * tolerance),
                                  jnp.zeros(REPORT_SIZE, dtype=dtype))
        constants = host.near.constants()
        stopping = GmresStopping(tolerance, restart=int(constants["restart"]), cap=int(constants["cap"]))
        made = prepared(grid, box, held, body, lambda rhs: inner(rhs), base, own_decayed(box, own, shift, nu),
                        decay * image, shift, nu, rho, now, stopping, jnp.asarray(p["schwarz_report"], dtype=dtype))
        return base, stage, made, scaled_least_eigenvalue(made.resistance).astype(dtype)

    def schwarz_initial(self) -> dict[str, Array]:
        """The Schwarz partner's fields at the run's start, its first step prepared (:meth:`started`) from the build's
        leaves (none without a partner)."""
        host = self.schwarz
        if host is None:
            return {}
        return dict(self._schwarz_initial)

    @cached_property
    def _schwarz_initial(self) -> dict[str, Array]:
        host = self.schwarz
        assert host is not None
        c = self.constants()
        p = {**c, **self.params_pytree()}
        dtype = run_dtype()
        held = host.fields(host.count, host.near.member_nodes[0].sample_count, self.grid(c))
        held[f"{SCHWARZ}0_at"] = schwarz_starts(c, host.count)[0]
        start = {**self.initial_fields(), **held}

        # Made once, from the build's leaves, outside whatever traces the step that first asks for it.
        with cast(Any, jax).ensure_compile_time_eval():
            made = cast(Any, jax).jit(lambda: self.started(p, start))()
        return {key: jnp.asarray(value, dtype=dtype) for key, value in {**held, **made}.items()}

    @override
    def started_fields(self) -> tuple[str, ...]:
        """A Schwarz flow's first preparation (:meth:`started`); none without a partner."""
        if self.schwarz is None:
            return ()
        return (*(f"{SCHWARZ}base_{x}" for x in "xyz"), f"{SCHWARZ}0_columns", f"{SCHWARZ}0_resistance",
                f"{SCHWARZ}0_load", f"{SCHWARZ}0_margin", f"{SCHWARZ}report")

    @override
    def started(self, params: Mapping[str, Any], state: Mapping[str, Array]) -> dict[str, Array]:
        """The Schwarz partner's first step prepared from ``params``: at the body's starting pose (``state``'s
        ``schwarz_0_at``) from the flow's initial field, the drive at the first step's end as the build gives it. The
        initial state holds it made from the build's leaves; a differentiated run makes it again in its program, so
        that a derivative in the viscosity or the body's leaves passes through the run's first step."""
        if self.schwarz is None:
            return {}
        dtype = run_dtype()
        grid = self.grid(params)
        velocity = (jnp.asarray(state["velocity_x"]), jnp.asarray(state["velocity_y"]), jnp.asarray(state["velocity_z"]))
        drive = jnp.asarray(params.get("schwarz_drive", (0.0, 0.0, 0.0)), dtype=dtype)
        base, _, made, margin = self.schwarz_prepare(
            params, velocity, jnp.asarray(0, dtype=jnp.int32), self.delta_t,
            self.schwarz_forcing(self.fields(params, grid), {}, drive, dtype),
            jnp.asarray(state[f"{SCHWARZ}0_at"], dtype=dtype), grid_zero_box(self.schwarz_box()),
            jnp.zeros(3, dtype=dtype))
        made_fields = {f"{SCHWARZ}base_{x}": base[k] for k, x in enumerate("xyz")}
        made_fields.update({f"{SCHWARZ}0_columns": made.columns, f"{SCHWARZ}0_resistance": made.resistance,
                            f"{SCHWARZ}0_load": made.load, f"{SCHWARZ}0_margin": margin,
                            f"{SCHWARZ}report": made.report})
        return {key: jnp.asarray(value, dtype=dtype) for key, value in made_fields.items()}

    def schwarz_stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                        dt: float) -> tuple[dict[str, Array], StepParts]:
        """A Schwarz flow's step (``mime.nodes.flow.fvm._schwarz_host`` states it): the body's step composed, the
        scheme's step for the next base, and the next step prepared at the body's pose."""
        host = self.schwarz
        assert host is not None
        dtype = run_dtype()
        grid = self.grid(p)
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        scheme: TimeScheme[StageParts] = self.scheme
        shift = 1.0 / dt if scheme.stores_inertia else 0.0
        steps = jnp.asarray(state["steps"])
        tap = self.report_row(p, steps)
        flow = self.walled_solve(p, grid, sigma, shift, self.stopping_of(p, grid, as_run_array(p["tolerance"])), tap)
        box = self.schwarz_box()
        nu, mu, rho = self.kinematic(p), as_run_array(p["viscosity"]), as_run_array(p["density"])
        now, decay = image_terms(host.side, host.spacing, shift, nu, mu)
        origin = as_run_array(p["schwarz_origin"])
        at = jnp.asarray(state[f"{SCHWARZ}0_at"], dtype=dtype)
        body = host.body(p, at)
        held = stencils(grid, box, body.placed.force - origin, SCHWARZ_WIDTH)
        step = Prepared(state[f"{SCHWARZ}0_columns"], state[f"{SCHWARZ}0_resistance"], state[f"{SCHWARZ}0_load"],
                        state[f"{SCHWARZ}report"])
        base = (state[f"{SCHWARZ}base_x"], state[f"{SCHWARZ}base_y"], state[f"{SCHWARZ}base_z"])
        own = (state[f"{SCHWARZ}own_x"], state[f"{SCHWARZ}own_y"], state[f"{SCHWARZ}own_z"])
        twist = jnp.asarray(boundary_inputs.get(f"{SCHWARZ}0_twist", jnp.zeros(6, dtype=dtype)), dtype=dtype)
        done: Composed = composed(grid, box, held, body, step, twist, lambda rhs: flow(rhs), base,
                                  own_decayed(box, own, shift, nu), decay * state[f"{SCHWARZ}image"], shift, nu, rho,
                                  now)
        velocity = done.field
        pose = jnp.asarray(boundary_inputs.get(f"{SCHWARZ}0_pose", at), dtype=dtype)
        speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in velocity]))
        cfl = speed * dt / grid.spacing
        checked = bool(self.constants()["convection"]) and scheme.cfl_line is not None
        past = ~(cfl <= scheme.cfl_limit(self.cfl_axis(p))) if checked else jnp.asarray(False)
        base_next, stage, made, margin = self.schwarz_prepare(
            p, velocity, steps + 1, dt, self.schwarz_forcing(fields, boundary_inputs, None, dtype), pose, done.own,
            done.image)
        report = combined([done.report, stage.report])
        indefinite = ~(margin > 0.0)
        stopped_before = jnp.asarray(state["stopped_at"]) >= 0
        stops = past | indefinite
        axis = self.vessel_axis(p)
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, velocity)
        ratio = self.ratio(solid, lumen, lumen_scale)
        reynolds = 2.0 * jnp.abs(lumen) / (jnp.pi * self.vessel_radius(p) * self.kinematic(p))
        tolerance = as_run_array(p["schwarz_tolerance"])
        inner_short = made.report[REPORT_SIZE + 2] + made.report[REPORT_SIZE + 3] + made.report[REPORT_SIZE + 4]
        flagged = (done.residual > RESIDUAL_LINE * tolerance) | (made.report[2] + made.report[3] + made.report[4] > 0) \
            | (inner_short > 0)
        ended: dict[str, Array] = dict(
            velocity_x=velocity[0], velocity_y=velocity[1], velocity_z=velocity[2], steps=steps + 1,
            solve_report=report, solve_report_sum=jnp.asarray(state["solve_report_sum"]) + report,
            solve_flagged=(report[2] + report[3] + report[4] > 0).astype(dtype), cfl=cfl.astype(dtype),
            stopped_at=jnp.where(stopped_before, jnp.asarray(state["stopped_at"]),
                                 jnp.where(stops, steps + 1, -1)).astype(jnp.int32),
            stopped_cfl=jnp.where(stopped_before, jnp.asarray(state["stopped_cfl"]),
                                  jnp.where(past, cfl, 0.0)).astype(dtype),
            lumen_flux=lumen, bypass_ratio=ratio, bypass_flagged=(jnp.abs(ratio) > BYPASS_LINE).astype(dtype),
            reynolds=reynolds.astype(dtype),
            reynolds_flagged=((reynolds > STOKES_REYNOLDS) & (not bool(self.constants()["convection"]))).astype(dtype),
            negative_resistivity=negative_resistivity(fields).astype(dtype))
        held_fields = node_fields(FlowState(**ended)) if fields.counts is None else node_fields(
            TreeFlowState(**ended, **self.wall_report(fields.counts)))
        mine: dict[str, Array] = {f"{SCHWARZ}base_{x}": base_next[k] for k, x in enumerate("xyz")}
        mine.update({f"{SCHWARZ}own_{x}": done.own[k] for k, x in enumerate("xyz")})
        mine.update({f"{SCHWARZ}0_columns": made.columns, f"{SCHWARZ}0_at": pose,
                     f"{SCHWARZ}0_resistance": made.resistance, f"{SCHWARZ}0_load": made.load,
                     f"{SCHWARZ}0_margin": margin, f"{SCHWARZ}image": done.image, f"{SCHWARZ}report": made.report,
                     f"{SCHWARZ}residual": done.residual.astype(dtype), f"{SCHWARZ}flagged": flagged.astype(dtype),
                     f"{SCHWARZ}indefinite": jnp.where(stopped_before, jnp.asarray(state[f"{SCHWARZ}indefinite"]),
                                                       indefinite.astype(dtype))})
        if not self.metered:
            return {**held_fields, **mine}, StepParts(stage, fields, {})
        before = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
        took = self.schwarz_stage(p, grid, sigma, before, base, steps, dt,
                                  self.schwarz_forcing(fields, boundary_inputs, None, dtype))._replace(solved=velocity)
        density = tuple(d / rho for d in done.density)
        energy = self.energy_terms(p, grid, before, took, sigma, fields, dt, None, (density[0], density[1], density[2]))
        account, size = law_books(body.placed, done, twist, dt)
        # The law's two rows, the flow's and the body's (the exchange with its sign turned), judged in their sum.
        floor = books.IDENTITY_ULPS * jnp.finfo(dtype).eps * (rho * dt * grid.inner_magnitude(density, velocity) + size)
        gift = (energy["schwarz"] - account[0] > floor).astype(dtype)
        stall = self.stall_ratio(energy)
        line = STALL_PER_TOLERANCE[jnp.dtype(dtype).name] * as_run_array(p["tolerance"])
        mine.update({"stall_ratio": stall, "stall_flagged": (stall > line).astype(dtype),
                     f"{SCHWARZ}0_books": account, f"{SCHWARZ}0_books_sum": state[f"{SCHWARZ}0_books_sum"] + account,
                     f"{SCHWARZ}0_gift_steps": state[f"{SCHWARZ}0_gift_steps"] + gift})
        return {**held_fields, **mine}, StepParts(took, fields, energy)

    def schwarz_stage(self, p: Mapping[str, Any], grid: MACGrid, sigma: Velocity, velocity: Velocity, base: Velocity,
                      steps: Array, dt: float, force: Callable[[int, Any], Velocity]) -> StageParts:
        """The scheme's stage as the step before took it to prepare this step's base from ``velocity`` (the field at
        this step's start): its parts formed again by the same arithmetic, its solve the held ``base``."""
        dtype = run_dtype()
        scheme: TimeScheme[StageParts] = self.scheme
        context = StepContext(grid, self.operator_of(p, grid, sigma), force, bool(self.constants()["convection"]),
                              lambda shift, rhs, guess: (base, jnp.zeros(REPORT_SIZE, dtype=dtype)),
                              self.kinematic(p), sigma)
        return scheme.step(context, {"velocity": velocity}, steps.astype(dtype) * dt, dt)[1]

    def schwarz_book_fields(self) -> dict[str, Array]:
        """A metered Schwarz flow's books of its law at the run's start, zeros: the step's (``schwarz_0_books``, in
        ``mime.nodes.flow.fvm._schwarz.BOOKS``' order), their sums over the run, and the steps the law's two rows
        summed above their rounding floor (``schwarz_0_gift_steps``: a dissipative law's gift). None without one."""
        if self.schwarz is None:
            return {}
        zeros = jnp.zeros(len(SCHWARZ_BOOKS), dtype=run_dtype())
        return {f"{SCHWARZ}0_books": zeros, f"{SCHWARZ}0_books_sum": zeros, f"{SCHWARZ}0_gift_steps": as_run_array(0.0)}

    @staticmethod
    def wall_report(counts: Array) -> dict[str, Array]:
        """A tube tree's report fields from its counts (``WallFields.counts``), and its flag: any of them above zero."""
        report = {name: counts[index] for index, name in enumerate(WALL_COUNTS)}
        return {**report, "wall_inexact": (jnp.sum(counts) > 0).astype(counts.dtype)}

    # -- the step identity ------------------------------------------------------------------------------------------
    @override
    def storage(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array:
        """``(1/2) rho <u, u>`` (J) where the scheme stores the fluid's inertia; zero for the steady limit."""
        velocity = _velocity(state)
        if not self.scheme.stores_inertia:
            return jnp.zeros((), dtype=velocity[0].dtype)
        return 0.5 * as_run_array(p["density"]) * self.grid(p).inner(velocity, velocity)

    def energy_terms(self, p: Mapping[str, Any], grid: MACGrid, before: Velocity, stage: StageParts, sigma: Velocity,
                     fields: WallFields, dt: float, partnered: Partnered | None = None,
                     schwarz: Velocity | None = None) -> dict[str, Array]:
        """Each term of the step's identity (J), the fluid's momentum at the solution the drive is paired with
        (:meth:`fluid_momentum`), each held point body's force's work on the fluid at the solution (``partners``, under
        its law: ``dt <S_F f, u'>``), a Schwarz body's tractions' (``schwarz``: ``schwarz`` their force density over the
        fluid's density, ``dt rho <schwarz, u'>``), and the magnitude of the arithmetic forming them (``scale``)."""
        fluid = (1.0 - fields.chi[0], 1.0 - fields.chi[1], 1.0 - fields.chi[2])
        rho, nu = as_run_array(p["density"]), self.kinematic(p)
        solved = stage.solved
        inner, size = grid.inner, grid.inner_magnitude
        laplacian = grid.laplacian(solved)
        minus_laplacian = (-laplacian[0], -laplacian[1], -laplacian[2])
        penalised = (sigma[0] * solved[0], sigma[1] * solved[1], sigma[2] * solved[2])
        applied = self.operator_of(p, grid, sigma)(stage.shift)(solved)
        forced = stage.rhs
        partner_forces = partnered.forces if partnered is not None else ()
        for force in (*partner_forces, *((schwarz,) if schwarz is not None else ())):
            forced = (forced[0] + force[0], forced[1] + force[1], forced[2] + force[2])
        remainder = (forced[0] - applied[0], forced[1] - applied[1], forced[2] - applied[2])
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
        if partner_forces:
            terms["partners"] = jnp.stack([dt * rho * inner(force, solved) for force in partner_forces])
            scale = scale + rho * dt * sum((size(force, solved) for force in partner_forces), start=zero)
        if schwarz is not None:
            terms["schwarz"] = dt * rho * inner(schwarz, solved)
            scale = scale + rho * dt * size(schwarz, solved)
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
        return (("viscous", f"{self.name}.viscosity", books.DISSIPATIVE), ("penalty", wall, books.DISSIPATIVE),
                *self.partner_laws(), *self.schwarz_laws())

    def partner_laws(self) -> tuple[tuple[str, str, str], ...]:
        """Each held point body's drag law, held in the step under its own id (``partner_<b>``): dissipative, its
        sign judged in the sum of its two rows by the drag's own node (``cluster_gift_steps``)."""
        return tuple((f"partner_{b}", str(law), books.DISSIPATIVE)
                     for b, law in enumerate(self.constants().get("partner_ids", ())))

    def schwarz_laws(self) -> tuple[tuple[str, str, str], ...]:
        """The Schwarz body's drag law, held in the step under its own id (``schwarz_<b>``): dissipative, its sign judged
        in the sum of its two rows by the flow itself (``schwarz_<b>_gift_steps``): the flow carries the body."""
        return tuple((f"{SCHWARZ}{b}", str(law), books.DISSIPATIVE)
                     for b, law in enumerate(self.constants().get("schwarz_ids", ())))

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

    def exchange_work(self, held: Sequence[books.LedgerDelivery], boundary_inputs: Mapping[str, Array],
                      rows: Sequence[str], dt: float, grid: MACGrid, solved: Velocity) -> tuple[Array, Array, Array]:
        """Each exchange partner's law's scattered force paired with the step's solution, ``dt <S f, u'>`` (the force
        density on each face set against the velocity there, by the grid's inner product), a row per law; the plain
        inputs' likewise; and the magnitude of the products. The force is held over the step, so the solution is the
        velocity it was applied at."""
        dtype = solved[0].dtype
        work = jnp.zeros(len(rows), dtype=dtype)
        scale = jnp.zeros((), dtype=dtype)
        plain = jnp.zeros((), dtype=dtype)
        component = {field: index for index, field in enumerate(EXCHANGE_FIELDS)}
        for index, delivery in enumerate(held):
            if delivery.kind == books.LOAD and delivery.field in component:
                density = jnp.asarray(boundary_inputs[books.law_input(delivery.field, index)], dtype=dtype)
                velocity = solved[component[delivery.field]]
                work = work.at[rows.index(delivery.law)].add(dt * grid.cell_volume * jnp.sum(density * velocity))
                scale = scale + dt * grid.cell_volume * jnp.sum(jnp.abs(density * velocity))
        for field, index in component.items():
            if field in boundary_inputs:
                density = jnp.asarray(boundary_inputs[field], dtype=dtype)
                plain = plain + dt * grid.cell_volume * jnp.sum(density * solved[index])
                scale = scale + dt * grid.cell_volume * jnp.sum(jnp.abs(density * solved[index]))
        return work, plain, scale

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
        if self.exchanges:
            exchanged, plain, exchange_scale = self.exchange_work(held, boundary_inputs, rows, dt, self.grid(p),
                                                                  parts.stage.solved)
            work, unattributed, drive_scale = work + exchanged, unattributed + plain, drive_scale + exchange_scale
        hosted_work = {"viscous": -energy["viscous"], "penalty": -energy["penalty"],
                       **{f"partner_{b}": energy["partners"][b] for b in range(self.body_count)},
                       **({f"{SCHWARZ}0": energy["schwarz"]} if "schwarz" in energy else {})}
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
        """The validity lines fixed when the node is built, each reported where it is crossed (never a refusal): a tube
        tree's wall's resolution entries below their lines (``mime.scene.wall_resolution``: its least radius of
        curvature, its narrowest lumen), the lumen's resolution in cells under the band's calibration, and float32's
        rounding floor near a steady state above the monolithic solve's tolerance (a node with no such solve has no
        such line)."""
        del start, closest_approach
        c = self.constants()
        radius = float(np.min(c["tree_radii"])) if self.walled_by_tree else float(c["tube_radius"])
        cells = 2.0 * radius / float(c["spacing"])
        lines: list[str] = []
        if self.walled_by_tree:
            from mime.scene import wall_resolution

            report = wall_resolution(self.tree().wall_features(), float(c["spacing"]))
            lines += [f"the tube tree's wall's {entry.text()}" for entry in report.entries if entry.below]
        if cells < CALIBRATED_CELLS:
            lines.append(f"the lumen is {cells:.3g} cells across, under the {CALIBRATED_CELLS:g} at and above which "
                         f"the band's wall shift is calibrated on straight tubes (0.3 to 0.4 % of the flux at 32); "
                         f"provisional until a stenosis case measures it, the band added +2.1 % of the flux at 16 "
                         f"and +10.8 % at 8, and under about 5 cells the build refuses")
        nu = float(c["viscosity"]) / float(c["density"])
        if run_dtype() == jnp.float32 and self.scheme.stores_inertia and "tolerance" in c:
            widest = float(np.max(c["tree_radii"])) if self.walled_by_tree else float(c["tube_radius"])
            tau = widest ** 2 / (STALL_SECTION * nu)
            floor = float(np.finfo(np.float32).eps) * tau / self.delta_t
            if floor > float(c["tolerance"]):
                lines.append(f"float32's rounding floor near a steady state, eps x tau/dt = {floor:.3g} (tau = R^2 / "
                             f"((pi^2/4) nu) = {tau:.3g} s), is above the solve's tolerance {c['tolerance']:.3g}: a "
                             f"run near its steady state stops short of it by about that share; run in float64 "
                             f"(JAX_ENABLE_X64=1), or with a larger dt")
        lines += self.held_lines()
        lines += self.schwarz_lines()
        return lines

    def schwarz_lines(self) -> list[str]:
        """The Schwarz partner's lines at the build: the coupling and its order, the free box, the tolerances, and the
        body's gap to the wall at its start in coarse cells against the window spike's lines (C4: the walls' share of
        the drag to 1 % wall-normal from about 6.6 to 8.5 cells, 2 % from 5 to 7, wall-parallel 1 % from 3 to 4)."""
        host = self.schwarz
        if host is None or self.walled_by_tree:
            return []
        c = self.constants()
        h = float(c["spacing"])
        start: np.ndarray = np.asarray(c["schwarz_starts"], dtype=np.float64)[:3] - np.asarray(
            c["schwarz_origin"], dtype=np.float64)
        axis: np.ndarray = np.asarray(c["tube_axis"], dtype=np.float64)
        axis = axis / float(np.linalg.norm(axis))
        off: np.ndarray = start - np.asarray(c["tube_centre"], dtype=np.float64)
        radial = float(np.linalg.norm(off - float(np.dot(off, axis)) * axis))
        gap = (float(c["tube_radius"]) - radial - 0.5 * float(c["schwarz_extent"])) / h
        body = ", ".join(host.near.members)
        line = (f"the flow couples {body}'s MRS drag by Schwarz, one pass and exact: the near field in free space, the "
                f"walls by the grid, the body's own field removed by a free box of {host.side}^3 cells; the body steps "
                f"before the flow, which composes its step and prepares the next (the staggered order); the columns' "
                f"GMRES to {float(c['schwarz_tolerance']):.3g} on each column's true residual, the walled solves in "
                f"its products at {INNER_SHARE:g} of it (provisional); at its start the body is {gap:.3g} coarse "
                f"cells from the wall")
        if gap < 6.6:
            line += (", under the window spike's line for the walls' share of its drag to 1 % wall-normal (about 6.6 "
                     "to 8.5 cells) and, under 5, to 2 %: measured to that line, unknown beyond it")
        return [line]

    def held_lines(self) -> list[str]:
        """The point bodies the flow holds in its one elimination, and what holding them costs a step: a unit field a
        delivery, six a body, each a projected solve warm-started from the last step's, and the bodies' unit fields
        held between steps; where the bodies are many (:data:`HELD_SUGGESTS_LAGGED` or more), coupling='lagged' named
        as the cheaper route."""
        count = self.body_count
        if not count:
            return []
        c = self.constants()
        faces = sum(int(np.prod(self.grid(c).shape(k))) for k in range(3))
        held = DELIVERIES * count * faces * int(jnp.dtype(run_dtype()).itemsize)
        line = (f"the flow holds {count} point bod{'y' if count == 1 else 'ies'} in its one elimination (one pass: "
                f"{', '.join(str(i) for i in c['partner_ids'])}): {DELIVERIES * count} unit fields solved a step "
                f"beside its own solve, each warm-started from the last step's, and {held / 2 ** 20:.3g} MiB of them "
                f"held between steps")
        if count >= HELD_SUGGESTS_LAGGED:
            line += (f"; with {count} bodies the lagged exchange, Drag.Stokes(coupling='lagged') on each, takes one "
                     f"solve a step in all and is the cheaper route (its error first order in the step)")
        return [line]

    @override
    def recheck_route(self, cause: str, dtype: str) -> str:
        """A stagnation (the CG does not test its floor in a recheck, so a floor and a stall are not told apart): the
        tolerance at or above the solve's rounding floor, and in a float32 run float64."""
        if cause != STAGNATED:
            return ""
        if dtype == "float32":
            return ("leave the flow's tolerance unwritten (float32's stop takes the solve's own floor where its "
                    "tolerance is out of reach) or write it at or above the floor, FVM(tolerance=...), or run in "
                    "float64")
        return "write the flow's tolerance at or above the solve's rounding floor, FVM(tolerance=...)"

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
        if float(np.asarray(state.get(f"{SCHWARZ}indefinite", 0.0))) > 0.0:
            margin = float(np.asarray(state[f"{SCHWARZ}0_margin"]))
            return Problem(
                "flow.schwarz-indefinite",
                f"at its step {step} (t = {step * self.delta_t:.6g} s) the response the flow prepared for its Schwarz "
                f"body turned indefinite: its least eigenvalue, each row and column scaled by its diagonal, was "
                f"{margin:.3g}, so the body's step through it would not be dissipative; the flow's state was held from "
                f"that step and the run ended at the end of its chunk",
                "move the body off the wall by a coarse cell or more, or refine the flow's grid (FVM(spacing=...)) "
                "so that the gap to the wall is resolved")
        cfl = float(np.asarray(state["stopped_cfl"]))
        c = self.constants()
        line = float(self.scheme.cfl_limit((1.0, 1.0, 1.0) if self.walled_by_tree else c["tube_axis"]))
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


def grid_zero_box(box: OwnBox) -> Velocity:
    """A free box's three face sets, zero (its field at the run's start)."""
    zero = jnp.zeros((box.side, box.side, box.side), dtype=box.eigenvalue.dtype)
    return (zero, zero, zero)


def check_deliveries(held: Sequence[books.LedgerDelivery]) -> None:
    """Refuse a delivery the flow does not take: a hosted law other than its viscosity and its wall's band, or a
    delivered one into anything but its drive's inputs (``fluid``'s loads)."""
    for delivery in held:
        hosted = delivery.kind == books.HOSTED and (delivery.field in HOSTED_FIELDS or
                                                    delivery.field.startswith((PARTNER, SCHWARZ)))
        driven = delivery.kind == books.LOAD and delivery.field in (*DRIVE_FIELDS, *EXCHANGE_FIELDS)
        if not (hosted or driven):
            raise ValueError(f"the flow node hosts {list(HOSTED_FIELDS)} and is driven through {list(DRIVE_FIELDS)}: "
                             f"it takes no delivery {delivery}")


__all__ = ["BYPASS_LINE", "CALIBRATED_CELLS", "DRIVE_FIELDS", "EXCHANGE_FIELDS", "GEOMETRY", "HOSTED_FIELDS", "STALL_SECTION", "STOKES_REYNOLDS",
           "STOP_FIELD", "TREE_LEAVES", "TREE_WALL", "TREE_WALL_COUNTS", "WALL_COUNTS", "FlowInputs", "FlowState", "ImplicitFlowNode", "MeteredFlowState", "MeteredPartneredFlowState", "PartnerFields", "PartneredFlowState",
           "MeteredTreeFlowState", "StepParts", "TreeFlowState", "check_deliveries"]
