"""The finite-volume flow field of a medium (F0's Part B): ``Medium("water", fills=tube.lumen, material=...,
fields={"flow": FVM(domain=PeriodicBox(length=mm(1)))})``, or open at the vessel's ends,
``FVM(domain=VesselRegion())`` with a law attached at each (``Inflow``, ``Windkessel``).

A field kind's value is a frozen settings object naming its kind and version, its settings grouped by the owner each
belongs to (B8.1): the grid kind's ``spacing`` (or ``cells_across``, the lumen's diameter in cells), the time scheme's
``scheme``, the solve's ``tolerance`` and ``preconditioner``, the wall's lift onto the grid (``wall=Band(...)``, the
stiff interior ``delta_in`` the wall's), and the domain it lives on. ``stokes`` is physical: the Stokes approximation, convection off. The fluid's properties
are never its settings: they come from the medium's material record or ``exp.declare`` (``viscosity``, ``density``).
The flow is driven by laws bound to the medium (``BodyForce``) and by the laws at its openings, which its step holds
(the open box: IMEX Euler or its steady limit; RK3 with openings is a later PR's).

F1a builds the Cartesian MAC grid kind, IMEX Euler and its steady limit, Jacobi, the default smooth band, on one
domain kind, the periodic box around a straight circular tube (aligned with z, or tilted in the x-z plane along a
lattice direction of the box). F2a adds a tube tree's lumen (``mime.scene.TubeTree``) in the periodic box, the tree
periodic along z with the box's period and reaching past its two ends: its wall is the union's exact distance where
the band reads it (``mime.nodes.flow.fvm.TreeSolid``), stepped by IMEX Euler or its steady limit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated, Any, cast, override

import numpy as np

from mime.core import jnp, run_dtype, shape_named
from mime.gradients import GradientStatus
from mime.nodes.flow.fvm import BANDS, DEFAULT_BAND, DEFAULT_CELLS_ACROSS, INTERIORS, STALL_PER_TOLERANCE, \
    ExplicitFlowNode, ImplicitFlowNode, OpenFlowNode, periodic_tube_layout
from mime.nodes.flow.fvm._box import TreeLayout, TubeLayout, period_mismatch, periodic_tree_layout
from mime.nodes.flow.fvm._band import FLUID_FLOOR
from mime.nodes.flow.fvm._open_node import FLOAT32_PORTS, OPEN_STEP
from mime.nodes.flow.fvm._porous import GAP_MEASURED_DOWN_TO, HANDOVER_LINE, HANDOVER_MEASURED, POROUS_MEASURED, \
    RESOLVED_CELLS
from mime.nodes.flow.fvm._resolved_node import ResolvedFlowNode
from mime.nodes.flow.fvm._ports import Inflow as InflowLaw
from mime.nodes.flow.fvm._ports import SharedDistal as SharedDistalLaw
from mime.nodes.flow.fvm._ports import Valve as ValveLaw
from mime.nodes.flow.fvm._ports import Windkessel as WindkesselLaw
from mime.nodes.flow.fvm._ports import RowKind, port_law_named, well_posed
from mime.nodes.flow.fvm._node import BYPASS_LINE, BYPASS_MEASURED, CALIBRATED_CELLS, EXCHANGE_FIELDS, GEOMETRY, \
    STOKES_REYNOLDS, porous_leaves
from mime.nodes.flow.fvm._schemes import RK3, SCHEMES, IMEXEuler, StageKind, Steady, scheme_named
from mime.nodes.flow.fvm._solve import DEFAULT_PRECONDITIONER, DEFAULT_TOLERANCE, FLOOR_MEASURED, \
    FLOOR_STOP_MEASURED, FLOOR_STOP_TOLERANCES, PRECONDITIONERS, STAGNATION_FACTOR, STAGNATION_LEAST_WINDOW, \
    STAGNATION_MEASURED, STAGNATION_SPAN, STALL_RESTART_MEASURED, STALL_RESTARTS
from mime.registry import NUMERICAL, PHYSICAL, BuildContext, Dynamics, Flag, GradientRecord, Port, Problem, Region, \
    SettingGroup, StepQuery, StepRule, convention, register_dynamics, register_setting_group
from mime.dynamics.flow._fields import fvm_field
from mime.dynamics.flow._steps import WALL_REACH_CELLS, axial_drive, predicted_peak, report_lines, womersley_number
from mime.fields._solved import ExchangeFrame, SolvedField
from mime.units import Magnitude
from mime.scene import PeriodicBox, VesselRegion
from mime.things import ThingKind
from mime.units import Length, Quantity
from mime.vocabulary import EXCHANGE_PARTNER, FLOW_BODY_FORCE, FLOW_EXCHANGE_X, FLOW_EXCHANGE_Y, FLOW_EXCHANGE_Z, \
    FLOW_FACES, FLOW_OPENING, FLOW_PARTNERS, FLOW_POROUS, FLOW_RESOLVED, FLOW_WAVEFORM, FLUID, OPENING, \
    RESOLVED_PARTNER

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration
    from mime.build._plan import GeometryPlan


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class IMEXEulerScheme(SettingGroup):
    """IMEX Euler on the monolithic projected CG: backward Euler for viscosity, the wall's penalty and
    incompressibility in one solve, forward Euler for convection, the drive at the step's end."""

    group = IMEXEuler.name


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class SteadyScheme(SettingGroup):
    """IMEX Euler's steady limit: the same operator with the time term dropped, solved at each step's drive. It needs
    the Stokes approximation."""

    group = Steady.name


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class RK3Scheme(SettingGroup):
    """S1's explicit RK3: Wray's three stages, convection and viscosity explicit, the wall's penalty implicit and
    pointwise in each stage, the incremental fast projection each stage, the drive read at each stage's time. Its
    explicit viscosity bounds the step (``nu dt / h^2 <= 0.15``) and its convection the CFL number along the vessel's
    axis (``1.7 g(n)``)."""

    group = RK3.name


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Band(SettingGroup):
    """The wall's lift onto the grid (B3, B8.1): the one registered band, smooth, ``W`` 2h wide with its penalty depth
    ``delta`` 0.3h (the sweet spot whose wall shift is smooth in the wall's sub-cell position), and beyond it the
    stiff interior ``delta_in``, in cells: 0.1 by default; 0.01 where the solid's bypass must be smaller (a clot's
    pressure drop), at about ten times the iterations."""

    group = "band"
    setting_roles = {"delta_in": convention(NUMERICAL, "the default interior lets about a ninth of the band's Darcy "
                                                       "velocity through the solid, at no extra cost (B3)")}
    delta_in: float = 0.1


#: The bytes a cell a run of the step holds, by the stage kind (the monolithic projected CG's, the explicit one's) and
#: the run's dtype: the slope of the process's peak resident memory over the box's cells, a build and two steps of a
#: real run on a CPU (the arrays the step takes, its temporaries and the host's copies the build holds; the process's
#: fixed part, its imports and compile, cancels), so the estimate is the cells times it (#258's form).
BYTES_PER_CELL = {
    (StageKind.MONOLITHIC_PROJECTED_CG, "float64"): 832.0,
    (StageKind.MONOLITHIC_PROJECTED_CG, "float32"): 539.0,
    (StageKind.EXPLICIT_PROJECTED, "float64"): 868.0,
    (StageKind.EXPLICIT_PROJECTED, "float32"): 462.0,
}
MEMORY_MEASURED = ("measured on a CPU with 24 threads as the slope of the process's peak resident memory, a build and "
                   "two steps, over boxes of 0.72 to 2.88 million cells, the largest of 3 to 8 runs: 832 bytes a cell "
                   "for IMEX Euler and its steady limit in float64 (689 to 832 over the runs), 868 for RK3 (771 to "
                   "868), and in float32 539 (447 to 539) and 462 (433 to 462); fewer threads hold less (the steady "
                   "limit in float64 about 560 with 8 and 430 with 4) and more may hold more; the process's fixed "
                   "part, its imports and compile, about 0.8 to 0.9 GiB, is not in it; a GPU's is not measured; the "
                   "figures are without the energy ledger, whose terms add to a step: XLA's analysis of one step's "
                   "program at D/h 32 gives 1.26 and 1.20 times its size without it for IMEX Euler (float64 and "
                   "float32) and 1.15 for RK3 in both, which the estimate does not include")


#: The least margin of solid, in cells, between a tube's band and its image: their fields flatten 3 cells into the
#: solid (the band's outer half, 1, and the interior's ramp, 2), and the images' tie lies halfway across the gap of
#: the box's reach (3) and its margin, so the margin keeps it flat from 3 on.
MARGIN_FLOOR = 3


#: The stiff interior's depths in cells, by the node's name for each.
_INTERIOR_NAMES = {depth: name for name, depth in INTERIORS.items()}
#: How far a tube tree may fail to repeat over its period where the band reads it, as a share of the spacing: past it
#: the tree is not periodic with the box (flow.tree-period).
PERIOD_LINE = 1e-6
#: The region shape a tube tree's lumen is.
TREE = "tube_tree"
#: The registration a porous structure's law makes with the flow (``mime.interactions.Darcy``), held in its step.
POROUS_LAW = "darcy"


def porous_held(registrations: Sequence[ResolvedRegistration]) -> list[ResolvedRegistration]:
    """The porous regions' registrations among a flow's, in their ids' order."""
    return sorted((r for r in registrations if r.kind == POROUS_LAW), key=lambda r: r.id)


#: The geometry kinds whose structure a flow holds as a porous region, each a solid circular cylinder its settings
#: give (``centre``, ``axis``, ``radius``, ``length``): the one place the flow names a kind (``porous_cylinder``).
POROUS_KINDS = ("plug",)


def porous_cylinder(geometry: GeometryPlan | None) -> dict[str, Any] | None:
    """A structure's porous region as the node takes it, a solid circular cylinder (its centre, axis, radius and length
    in SI, the world's frame), from its geometry kind's settings; ``None`` for a kind the flow does not hold."""
    if geometry is None or geometry.geometry.name not in POROUS_KINDS:
        return None
    settings = geometry.settings
    return dict(centre=tuple(float(c) for c in cast(tuple[float, ...], settings["centre"])),
                axis=tuple(float(c) for c in cast(tuple[float, ...], settings["axis"])),
                radius=float(cast(float, settings["radius"])), length=float(cast(float, settings["length"])))


def porous_arguments(registrations: Sequence[ResolvedRegistration], origin: Sequence[float]) -> dict[str, Any]:
    """The node's porous regions (``ImplicitFlowNode(porous=..., porous_laws=...)``) from their laws' registrations:
    each structure's ``Plug`` settings in SI, its centre moved into the box's frame (``origin``, the world point at
    the box's corner), and its permeability as phase 1 resolved it; none where no law registers one."""
    held = porous_held(registrations)
    regions: list[dict[str, Any]] = []
    for registration in held:
        cylinder = porous_cylinder(registration.geometry)
        assert cylinder is not None
        centre = tuple(float(c) - float(o) for c, o in zip(cylinder["centre"], origin, strict=True))
        regions.append({**cylinder, "centre": centre,
                        "permeability": float(cast(float, registration.values["permeability"].value))})
    return {**porous_leaves(regions), "porous_laws": [r.id for r in held]} if held else {}


def porous_scheme(given: IMEXEulerScheme | SteadyScheme | RK3Scheme | None) -> str:
    """The scheme a flow holding a porous region steps with: the one named (never RK3: refused when planned), else
    IMEX Euler, whose one solve holds the region's resistivity and the divergence together (RK3's stages split them,
    which leaves a Darcy flux to the splitting)."""
    return given.group if given is not None else IMEXEuler.name


def tree_of(region: Region) -> tuple[Any, Any, tuple[tuple[int, int], ...], float]:
    """A tube tree's lumen's parameters (``TubeTree.region_parameters``) as its points ``(k, 3)``, radii ``(n, 2)``
    (m), segments and fillet share."""
    parameters = region.parameters
    points = np.asarray(parameters["points"], dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(parameters["radii"], dtype=np.float64).reshape(-1, 2)
    flat = [int(round(float(i))) for i in cast(tuple[float, ...], parameters["segments"])]
    segments = tuple((flat[k], flat[k + 1]) for k in range(0, len(flat), 2))
    return points, radii, segments, float(cast(float, parameters["fillet"]))


class Scheme:
    """The time schemes a script names: ``FVM(scheme=Scheme.RK3())``, ``FVM(scheme=Scheme.IMEXEuler())``,
    ``FVM(stokes=True, scheme=Scheme.Steady())``; none named, the build picks RK3 where its viscous number holds at the
    run's dt and IMEX Euler otherwise (:func:`chosen_scheme`). The one place their public names are written."""

    @staticmethod
    def RK3() -> RK3Scheme:  # noqa: N802 - a kind's name, as the script writes it
        return RK3Scheme()

    @staticmethod
    def IMEXEuler() -> IMEXEulerScheme:  # noqa: N802 - a kind's name, as the script writes it
        return IMEXEulerScheme()

    @staticmethod
    def Steady() -> SteadyScheme:  # noqa: N802 - a kind's name, as the script writes it
        return SteadyScheme()


def chosen_scheme(given: IMEXEulerScheme | SteadyScheme | RK3Scheme | None, viscosity: float, spacing: float,
                  dt: float) -> str:
    """The scheme a run steps with: the one named, or, none named, **RK3 where its explicit viscosity holds at dt**
    (``nu dt / h^2 <= 0.15``) and IMEX Euler otherwise (F0's B1 and item 1: the scheme from the viscous limit, RK3
    first). The one function this value-keyed choice is made in, so a declared range reaches it unchanged."""
    if given is not None:
        return given.group
    rk3 = scheme_named(RK3.name)
    limit = rk3.viscous_limit(viscosity, spacing)
    return RK3.name if limit is not None and dt <= limit * (1.0 + 1e-12) else IMEXEuler.name


def lattice_of(axis: Sequence[float], largest: int = 12) -> tuple[int, int] | None:
    """The lattice direction ``(a, 0, b)`` of the box along ``axis`` (a unit vector), or ``None`` where the axis is not
    along z or in the x-z plane at a rational slope with small integers."""
    x, y, z = (float(c) for c in axis)
    if z < 0.0:
        x, y, z = -x, -y, -z
    if abs(y) > 1e-9 or z <= 0.0:
        return None
    if abs(x) <= 1e-12:
        return (0, 1)
    if x < 0.0:
        return None
    slope = Fraction(x / z).limit_denominator(largest)
    if abs(float(slope) - x / z) > 1e-9 * max(1.0, x / z):
        return None
    return (slope.numerator, slope.denominator)


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class FVM(Dynamics):
    """A medium's flow on the Cartesian MAC grid with its wall lifted onto the grid by a smooth band: the
    incompressible Navier-Stokes equations (or Stokes, with ``stokes=True``) for the fluid filling the medium's region,
    the solid outside it a Brinkman penalty, stepped by S1's explicit RK3 or by an implicit scheme on the projected
    monolithic CG."""

    name = "fvm"
    version = "0.1.0"
    differentiability = GradientStatus.UNVERIFIED
    gradient = GradientRecord(
        where="in the tube's radius, centre and axis (the geometry leaves); in the viscosity, the density and the drive "
              "the gradient gate checks it exact, reverse and forward, against a central difference that rebuilds",
        reason=f"{GEOMETRY}; one status is declared per entry, so the entry's is the geometry's",
        route="the dQ/dR gate: the radius in the gradient gate, its loss the box's whole axial flux, its central "
              "difference taken inside one discretisation (the layout's dead band; or from a declared range)")
    references = ("HarlowWelch1965", "Angot1999", "Womersley1955")
    builds = {"": (ImplicitFlowNode, ExplicitFlowNode, OpenFlowNode, ResolvedFlowNode)}
    applies_to = frozenset({ThingKind.MEDIUM})
    port_kinds = (FLUID.name, OPENING.name)
    accepts = {FLOW_BODY_FORCE.name: Port("body_force"), FLOW_WAVEFORM.name: Port("waveform"),
               FLOW_OPENING.name: Port("openings"), FLOW_POROUS.name: Port("porous"),
               FLOW_EXCHANGE_X.name: Port("exchange_x"),
               FLOW_EXCHANGE_Y.name: Port("exchange_y"), FLOW_EXCHANGE_Z.name: Port("exchange_z"),
               FLOW_PARTNERS.name: Port("partners"), FLOW_RESOLVED.name: Port("resolved")}
    publishes = {FLOW_FACES.name: Port("velocity_x")}
    #: The laws a user attaches at an opening or several, by their port laws' names (``mime.interactions.Inflow``,
    #: ``Windkessel``, ``Valve``, ``SharedDistal``), the exchange partners a point body's law registers
    #: (``Drag.Stokes``) and the bodies a resolved body's law registers (``Drag.Penalty``).
    registration_kinds = frozenset({InflowLaw.name, WindkesselLaw.name, ValveLaw.name, SharedDistalLaw.name,
                                    EXCHANGE_PARTNER, POROUS_LAW, RESOLVED_PARTNER})
    raises = (
        Flag("solve_flagged", "the step's solve stopped other than converging: at its iteration cap, stagnated (its "
                              "true residual no longer falling), so the step's values hold only to the residual it "
                              "reached; or at a residual that is not finite, whose values are then NaN, so the run is "
                              "refused as not finite. Its report, solve_report (iterations, "
                              "the true relative residual against the tolerance, and one count for each way it "
                              "stopped), says which; a run recording it per step (history=True) keeps every step's",
             warns=True),
        Flag("reynolds_flagged", f"with the Stokes approximation (stokes=True), the lumen's Reynolds number by its "
                                 f"diameter and mean speed was above the declared {STOKES_REYNOLDS:g} (reynolds)"),
        Flag("bypass_flagged", f"the flux through the solid beyond the band's middle was more than {BYPASS_LINE:g} "
                               f"of the lumen's (bypass_ratio): a finer spacing lowers it, and the stiffer interior "
                               f"wall=Band(delta_in=0.01) about halves it (the validity line's measurements). Past a "
                               f"porous region closing the tube of a box driven by a body force it is a loop, not a "
                               f"leak (bypass_ratio near minus one): the solid carries the open part's flux back, lumen_flux "
                               f"is the box's whole flux, which the loop does not cross, and the velocity next to the "
                               f"region carries it (the porous validity line gives its size)"),
    )
    validity = (
        "the CFL number max|u| dt / h, with convection on, at most the scheme's line along the vessel's axis n, "
        "1.7 g(n) for RK3 and 0.4 g(n) for IMEX Euler, g(n) = max|n_d| / sum|n_d| (declared): predicted from the "
        "drive before the run and refused past it (flow.cfl-predicted), and checked each step; at the first step past "
        "it the flow's state is held, the run ends at its chunk's end and is refused (flow.cfl-exceeded). With "
        "stokes=True no line applies (they bound convection only)",
        f"RK3's explicit viscosity: nu dt / h^2 at most {RK3.viscous_number:g} (declared; the pure-diffusion edge is "
        f"0.209), refused at build past it (flow.viscous-limit); with no scheme named the build takes IMEX Euler there",
        "the stop: each step's solve stops at a true relative residual of its tolerance (by default "
        f"{DEFAULT_TOLERANCE['float64']:g} in float64; in float32 {DEFAULT_TOLERANCE['float32']:g}, or the solve's "
        "rounding floor eps || |A| |x| + |b| || / ||b|| where that is higher, made at each check from the iterate; a "
        f"tolerance given is held as given; {FLOOR_MEASURED}), the floor counting only while it is under "
        f"{FLOOR_STOP_TOLERANCES} times the tolerance, past which the tolerance holds ({FLOOR_STOP_MEASURED}), the "
        "step's values and gradients holding to about it; "
        f"where the stop would find a stagnation the CG restarts from the true residual instead, up to "
        f"{STALL_RESTARTS} times, its window begun again ({STALL_RESTART_MEASURED}); a stagnation is no "
        "check of the true residual (one every 20 iterations) over a window of "
        f"{STAGNATION_SPAN} iterations per cell along the box's longest side (at least "
        f"{STAGNATION_LEAST_WINDOW} checks) below {STAGNATION_FACTOR:g} of the best before them ({STAGNATION_MEASURED}); "
        "a step stopped otherwise raises solve_flagged, with a warning after the run. RK3's projections are direct "
        "solves (the fast transform), so an RK3 step never raises it",
        f"the wall: {BANDS[DEFAULT_BAND].validity}. Its resolution: the lumen's diameter in cells, the band calibrated "
        f"at {CALIBRATED_CELLS:g} and finer (0.3 to 0.4 % of the flux at 32); provisional until a stenosis case "
        f"measures it, +2.1 % at 16 and +10.8 % at 8 [measured on straight tubes]; under about 5 cells refused "
        f"(flow.resolution). A build under {CALIBRATED_CELLS:g} cells reports it",
        f"the solid's bypass: the flux through the solid beyond the band's middle, reported each step as a share of "
        f"the lumen's (bypass_ratio), flagged above {BYPASS_LINE:g} (declared; {BYPASS_MEASURED})",
        f"the Stokes approximation (stokes=True): the lumen's Reynolds number by its diameter and mean speed at most "
        f"{STOKES_REYNOLDS:g} (declared, not fitted; in the straight periodic tube the axial flow's convection "
        f"vanishes, so the line bounds nothing measured here), flagged above it each step (reynolds_flagged)",
        "float32 near a steady state: its rounding floor eps x tau/dt, tau = R^2 / ((pi^2/4) nu), the share by which "
        "a near-steady run stops short of its fixed point; a float32 build whose floor is above its tolerance reports "
        "it",
        f"the memory: the box's cells times the bytes a cell a run holds ({MEMORY_MEASURED}), reported at build and "
        f"refused past the memory a run can have (flow.memory): a GPU's limit, or on a CPU the smaller of the host's "
        f"available memory and the limit of the control group the process runs in, where one is set",
        f"geometry: {GEOMETRY}",
        "openings (domain=VesselRegion()): a straight vessel along z cut at the two ends its geometry kind declares "
        "(a Tube's start and end), each an opening whose law the step holds, coupled implicitly through the field's "
        "affine response (one unit field a step for the opening that is not the gauge's reference, solved again from "
        "the last), its rows a pressure (Windkessel) or a flux (Inflow, Valve); an Inflow pins the opening's whole net "
        "flux, the lumen's, the band's and the solid's faces on its plane together, never the lumen's alone, so the "
        "lumen carries Q less what the solid carries, which runs either way (measured beyond the band's half width, "
        "OpenFlowNode.opening_solid_fluxes): in steady flow 0.4 to 3.7 % of Q, with the inflow, at D/h 16 to 8 at the "
        "default stiff interior 0.1h, 0.01 to 0.4 % at 0.01h; in pulsed flow it swings with the drive where the drive's "
        "frequency nears the stiff interior's damping rate nu / (delta_in h)^2, the solid then no longer stiff against "
        "it: on the open box's drawn runs (omega (delta_in h)^2 / nu 1.3e-3 to 0.95) from 32 % of Q against the "
        "inflow to 19 % with it, the lumen then carrying up to 1.32 Q, the steady shares holding below "
        "a ratio of about 4e-3; a heartbeat in a millimetre vessel sits at 7e-5 (blood, D/h 16) to 1e-3 (water, D/h 8); "
        "the inflow's face flux held to it to its summation floor, (N + 2) eps of its superposition's "
        "scale over the plane's N faces, at any pressure level (a relative pressure's move is never formed from "
        "absolute pressures; gated to a downstream resistance of 1e4 times the vessel's); the Windkessel stepped by "
        "backward Euler, its N_WK first order in dt; a Valve holds its opening's net flux at zero, not each face's: "
        "the lumen behind it carries out what the solid's faces on the opening bring back in, the solid's Darcy flow, "
        "falling as delta_in squared (on two parallel tubes, one outlet closed, D/h 8 to 12: 1.5 to 5.0 % of "
        "the inflow at the default stiff interior 0.1h, 1.8e-4 to 7.0e-4 at 0.01h); gated against the walled-off "
        "vessel (that outlet capped by solid, whose cap and the solid around it leak as much: 1.4 to 4.8 % at "
        "0.1h), the inlet pressures within the two runs' solid flows carried through the open tube's resistance, so "
        "the gate sees a valve error from about 8 % of the inflow at 0.1h and from 1e-3 (float64) and 2e-3 (float32) at "
        "0.01h; "
        "a law over several openings (SharedDistal) solved whole, its block's off-diagonals included; several openings "
        "on one plane (a vessel with two outlets, gated at the node level until the tube tree's wall is in the open "
        "box) split the plane between their centres, the solid's faces included, so the solid between their patches "
        "carries Darcy flow from one to the other: on two parallel tubes at the default stiff interior the outlets' "
        "cross conductance is 2.5 to 9 % of one's own (D/h 8 to 12), and a split differs from two independent "
        "tubes' by up to 1.8 %; "
        "IMEX Euler or its steady limit only (RK3 with openings is not built); no step rule yet: dt is given, "
        "dt='auto' refused, the CFL checked each step; tilted open vessels not built; float32 behind its guards: "
        f"{FLOAT32_PORTS}",
        f"porous regions (Darcy, a Plug's solid of declared permeability K): a resistivity mu / K added to the wall's, "
        f"on IMEX Euler or its steady limit in a box along z; in the wall's band its share of the fluid over the face's "
        f"fluid fraction (at least {FLUID_FLOOR:g}, declared) and carried into the wall's solid behind it. {POROUS_MEASURED}. "
        f"No OpenMaterica record gives a permeability yet: it is declared. The build reports the run's sqrt(K) in cells "
        f"where it is under {RESOLVED_CELLS:g}, and the route its throughflow takes: resolved two-way at or above the "
        f"hand-over's line (sqrt(K)/h {HANDOVER_LINE['0.1h']:g} with the 0.1h interior, {HANDOVER_LINE['0.01h']:g} "
        f"with 0.01h), below it held as a wall at the line's permeability and read one way from its face pressures "
        f"(porous_flux, porous_seepage). {HANDOVER_MEASURED}. A region leaving a gap to the wall under sqrt(K)/h "
        f"{GAP_MEASURED_DOWN_TO:g}, or below the line in a box holding a resolved body, is unmeasured, and the run raises "
        f"porous_flagged at each such step, with a warning; where a law fixes the flux on every path through the plug "
        f"below the line the run raises forced_flagged, with a warning; a plug whose extent along the axis is not under a periodic "
        f"box's period is refused (flow.porous-period), as is one outside an open vessel's span (flow.porous-outside), "
        f"and a periodic box reads a plug anywhere along the tube at its image in the period",
        f"the stall detector |N_Krylov| / D_visc, flagged above {STALL_PER_TOLERANCE['float64']:g} solve tolerances in "
        f"float64 and {STALL_PER_TOLERANCE['float32']:g} in float32 (stall_flagged), is written with the energy ledger "
        f"attached (exp.observe(EnergyLedger())); a run without it has no stall report. The reading scales with the "
        f"solve's tolerance, so the line does: a run built with a looser tolerance is judged against that solve. "
        f"Correct runs read up to 510 tolerances in float64 and 200 in float32 at the default tolerance (a tenfold "
        f"tighter or looser one: 391 and 329 in float64, 479 and 245 in float32, float32's tighter one below its "
        f"rounding floor at the worst draws) [measured, 200 drawn runs a dtype and tolerance, tau/dt up to 4.1e3, D/h 8 "
        f"to 16, Re 1 to 200; beyond that envelope the line is unvalidated]. A solve "
        f"stopped far short of its own tolerance is flagged; a tolerance set loose on purpose is not, so the flag "
        f"cannot see a tolerance that is wrong for the question. The ledger harness's stall flag judges the same "
        f"number at the default tolerance",
        "the energy ledger: IMEX Euler and its steady limit close the step identity to its rounding floor at every "
        "step; RK3 judges each step to have dissipated, and its drive's attribution (the work its stages applied "
        "against the law's own delivery at the stage instants) to the same floor. A gift a numerical term's own "
        "formula books is booked under that term, where the identity cannot see it (convection's, at mixed time "
        "levels): the order tests see a term of the wrong order (first or third in dt, or a loss written into the "
        "solve's residual), each term summed over three drawn swirls [measured, every ensemble], and the ledger "
        "harness's convection flag, on drawn runs in CI, sees convection doing work beyond its time level's (a stale "
        "level, the wrong field, the advective form), not an error in the convective operator that keeps it skew "
        "(scaled, negated), which the manufactured-solution gate sees; no check of it runs in a user's run",
    )
    values = {"viscosity": "viscosity", "density": "density"}
    setting_roles = {
        "domain": PHYSICAL,
        "spacing": convention(NUMERICAL, "none given: the lumen's diameter over cells_across, the band's sweet-spot "
                                         "resolution on straight tubes"),
        "cells_across": convention(NUMERICAL, "the band's calibrated resolution across the narrowest lumen"),
        "scheme": convention(NUMERICAL, "none named: RK3 where its viscous number holds at the run's dt, IMEX Euler "
                                        "otherwise (the scheme from the viscous limit, RK3 first)"),
        "stokes": convention(PHYSICAL, "convection is on unless the Stokes approximation is declared"),
        "tolerance": convention(NUMERICAL, "none given: each dtype's own, float32's raised to the solve's rounding "
                                           "floor at each check where that is higher"),
        "preconditioner": convention(NUMERICAL, "Jacobi, the one preconditioner measured in three dimensions"),
        "wall": convention(NUMERICAL, "the one registered smooth band, with its default stiff interior (B3)"),
        "solve_report_rows": convention(NUMERICAL, "the derivative solves' reports summed over the run"),
    }

    domain: PeriodicBox | VesselRegion
    spacing: Annotated[Quantity | None, Length] = None
    cells_across: int = DEFAULT_CELLS_ACROSS
    scheme: IMEXEulerScheme | SteadyScheme | RK3Scheme | None = None
    stokes: bool = False
    tolerance: float | None = None
    preconditioner: str = DEFAULT_PRECONDITIONER
    wall: Band = Band()
    solve_report_rows: int = 0

    @override
    def value_problems(self, values: Mapping[str, Magnitude]) -> list[Problem]:
        """What the settings rule out, refused when planned."""
        found: list[Problem] = []
        if self.open and isinstance(self.scheme, RK3Scheme):
            found.append(Problem("flow.open-scheme", "RK3 with openings is not built yet: the open vessel steps by "
                                                     "IMEX Euler or its steady limit",
                                 "write FVM(domain=VesselRegion(), scheme=Scheme.IMEXEuler())"))
        if isinstance(self.scheme, SteadyScheme) and not self.stokes:
            found.append(Problem("flow.steady-convection",
                                 "the steady solve rests on linearity, so it needs the Stokes approximation",
                                 "write FVM(stokes=True, scheme=Scheme.Steady()), or step in time with "
                                 "Scheme.IMEXEuler()"))
        if self.wall.delta_in not in _INTERIOR_NAMES:
            found.append(Problem("flow.interior", f"the stiff interior's depth delta_in is one of "
                                                  f"{sorted(_INTERIOR_NAMES)} cells, not {self.wall.delta_in!r}",
                                 "write FVM(wall=Band(delta_in=0.1)), or 0.01"))
        if self.preconditioner not in PRECONDITIONERS:
            found.append(Problem("flow.preconditioner", f"no preconditioner {self.preconditioner!r} is registered",
                                 f"write FVM(preconditioner={DEFAULT_PRECONDITIONER!r})"))
        if self.domain.margin < MARGIN_FLOOR:
            found.append(Problem("flow.box-margin", f"a periodic box's margin of {self.domain.margin} cells leaves the "
                                                    f"tube's images closer than the band and the stiff interior's "
                                                    f"ramp reach: the image nearest a face is then not the tube's own "
                                                    f"where the fields are not flat",
                                 f"write PeriodicBox(..., margin={MARGIN_FLOOR}) or more (8 by default)"))
        if self.cells_across < 5:
            found.append(Problem("flow.resolution", f"{self.cells_across} cells across the lumen is under the band's "
                                                    f"floor of about five",
                                 "write FVM(cells_across=32), or a finer spacing"))
        return found

    @property
    def open(self) -> bool:
        """Whether the flow's domain is open at the vessel's ends (a ``VesselRegion``)."""
        return isinstance(self.domain, VesselRegion)

    def tree_layout(self, region: Region, spacing: float | None = None) -> TreeLayout:
        """The periodic box a tube tree's lumen is held in: its period the domain's length, its spacing ``spacing`` or
        the narrowest lumen's diameter over ``cells_across`` (a tree on an open domain is refused when planned)."""
        points, radii, segments, _ = tree_of(region)
        box = cast(PeriodicBox, self.domain)
        return periodic_tree_layout(points, segments, radii, float(cast(float, box.length.value)),
                                    cells_across=self.cells_across, spacing=spacing, margin_cells=box.margin)

    def tree_problems(self, region: Region) -> list[Problem]:
        """A tube tree is stepped by IMEX Euler or its steady limit (RK3 on a tree is not built) and must repeat over
        the box's period where the band reads it."""
        if isinstance(self.scheme, RK3Scheme):
            return [Problem("flow.tree-scheme", "RK3 on a tube tree's wall is not built: a tree's flow steps by IMEX "
                                                "Euler or its steady limit",
                            "write FVM(scheme=Scheme.IMEXEuler()), or FVM(stokes=True, scheme=Scheme.Steady())")]
        from mime.scene import TubeTree

        _, radii, segments, fillet = tree_of(region)
        given = self.spacing.value if self.spacing is not None else None
        layout = self.tree_layout(region, float(cast(float, given)) if given is not None else None)
        tree = TubeTree.si(np.asarray(layout.points), segments, radii, fillet)
        lumen, parameters = tree.lumen(), tree.parameters()
        reach = (BANDS[DEFAULT_BAND].width / 2.0 + 3.0) * layout.spacing + tree.fillet_radius_value()
        def minimum(at: Any) -> Any:
            return lumen.minimum(parameters, at)

        mismatch = period_mismatch(minimum, layout, reach)
        if mismatch > PERIOD_LINE * layout.spacing:
            return [Problem("flow.tree-period",
                            f"the tube tree's lumen does not repeat over the periodic box's length "
                            f"{layout.cells[2] * layout.spacing:.6g} m: its distance one period on differs by "
                            f"{mismatch / layout.spacing:.3g} cells where the wall's band reads it",
                            "make the tree periodic along z with the box's period, its two ends reaching past the "
                            "period's (PeriodicBox(length=...) the tree's period)")]
        return []

    @override
    def region_problems(self, region: Region | None) -> list[Problem]:
        """The region must be a straight circular tube's (a cylinder), its axis along z or along a lattice direction
        ``(a, 0, b)`` of the periodic box; or a tube tree's lumen, periodic along z with the box."""
        if region is not None and region.shape == TREE:
            if self.open:
                return [Problem("flow.region", f"the flow's medium fills {region.path}, a tube tree's lumen; an open "
                                               f"vessel is a straight vessel cut at its two ends (open trees come later)",
                                "run the tree on a periodic box, FVM(domain=PeriodicBox(length=...))")]
            return self.tree_problems(region)
        if region is None or region.shape != "cylinder":
            fills = f"fills {region.path}, a {region.shape} region" if region is not None else "fills no region"
            return [Problem("flow.region", f"the flow's medium {fills}; the FVM's periodic box holds a straight "
                                           f"circular tube's lumen or a tube tree's",
                            "fill a Tube's lumen, Medium(fills=tube.lumen), or a TubeTree's")]
        if lattice_of(_unit(region.parameters["axis"])) is None:
            return [Problem("flow.box-axis", f"the tube's axis {_unit(region.parameters['axis'])} is not along z or a "
                                             f"lattice direction (a, 0, b) of the periodic box",
                            "align the tube with z, Tube(axis=+Z, ...), or tilt it in the x-z plane at a slope a/b of "
                            "small integers, Tube(axis=(a, 0, b), ...)")]
        if self.open and len(region.openings) != 2:
            return [Problem("flow.region", f"the flow's medium fills {region.path}, whose geometry kind declares "
                                           f"{len(region.openings)} openings there; an open vessel is a straight "
                                           f"vessel cut at the two ends its kind declares (a Tube's start and end)",
                            "fill a Tube's lumen, Medium(fills=tube.lumen), or run the flow on a periodic box, "
                            "FVM(domain=PeriodicBox(length=...))")]
        if self.open and lattice_of(_unit(region.parameters["axis"])) != (0, 1):
            return [Problem("flow.open-axis", f"the tube's axis {_unit(region.parameters['axis'])} is tilted, and an "
                                              f"open vessel's box is built along z only (tilted open vessels come "
                                              f"later)", "align the tube with z, Tube(axis=+Z, ...), or run it on a "
                                                         "periodic box, FVM(domain=PeriodicBox(length=...))")]
        return []

    @staticmethod
    def openings_of(region: Region) -> tuple[str, ...]:
        """The openings of the vessel the medium fills, cut at its own ends: the surfaces its geometry kind declares
        for the region, in order along its axis (a Tube's ``start`` and ``end``)."""
        return region.openings

    @override
    def registration_problems(self, registrations: Sequence[ResolvedRegistration],
                              region: Region | None) -> list[Problem]:
        """A periodic box takes no law at an opening; an open vessel takes one at each of its two ends and at nothing
        else, and their rows are well posed (``mime.nodes.flow.fvm._ports.well_posed``). A point body's exchange
        registration (``Drag.Stokes``) and a resolved body's (``Drag.Penalty``) are no laws at an opening, and an open
        vessel takes neither yet, nor a tube tree's flow a resolved body."""
        found_porous = self.porous_problems(
            porous_held(registrations), region,
            [r for r in registrations if r.kind not in (POROUS_LAW, EXCHANGE_PARTNER, RESOLVED_PARTNER)])
        if found_porous:
            return found_porous
        registrations = [r for r in registrations if r.kind != POROUS_LAW]
        partners = [r for r in registrations if r.kind == EXCHANGE_PARTNER]
        if self.open and partners:
            return [Problem("flow.open-exchange", f"{', '.join(r.id for r in partners)} exchanges with the flow, and an "
                                                  f"open vessel's flow takes no exchange partners yet (a point body's "
                                                  f"exchange with its openings comes with the one elimination)",
                            "run the flow on a periodic box, FVM(domain=PeriodicBox(length=...))")]
        resolved = [r for r in registrations if r.kind == RESOLVED_PARTNER]
        if self.open and resolved:
            return [Problem("flow.open-resolved", f"{', '.join(r.id for r in resolved)} resolves its body on the flow's "
                                                  f"grid, and an open vessel's flow takes no resolved bodies yet (they "
                                                  f"come with the one elimination of bodies and openings)",
                            "run the flow on a periodic box, FVM(domain=PeriodicBox(length=...))")]
        if resolved and region is not None and region.shape == TREE:
            return [Problem("flow.tree-resolved", f"{', '.join(r.id for r in resolved)} resolves its body on the flow's "
                                                  f"grid, and a tube tree's flow takes no resolved bodies yet (its node "
                                                  f"holds the tree's solid alone)",
                            "run the body in a straight tube's flow, Medium(fills=tube.lumen), or use Drag.MRS(...) "
                            "for it")]
        registrations = [r for r in registrations if r.kind not in (EXCHANGE_PARTNER, RESOLVED_PARTNER)]
        if not self.open:
            return [Problem("flow.no-openings", f"{', '.join(r.id for r in registrations)} is attached at an opening "
                                                f"and the flow's periodic box has none",
                            "open the vessel's ends, FVM(domain=VesselRegion()), or drive the periodic box with "
                            "BodyForce")] if registrations else []
        if region is None or len(region.openings) != 2:
            return []  # the region's own refusal names it
        openings = self.openings_of(region)
        found: list[Problem] = []
        for registration in registrations:
            unknown = [surface for surface in registration.surfaces if surface not in openings]
            if unknown:
                found.append(Problem("flow.opening-unknown", f"{registration.id} is attached at "
                                                             f"{', '.join(repr(u) for u in unknown)}, which is not an "
                                                             f"opening of the flow's region (its openings: "
                                                             f"{', '.join(openings)})",
                                     f"attach it at one of them, exp.attach(..., medium=..., "
                                     f"surface={openings[0]!r}) or surface={openings[1]!r}"))
        held = {o: [r for r in registrations if o in r.surfaces] for o in openings}
        for opening, laws in held.items():
            if len(laws) != 1:
                attached = f"{', '.join(r.id for r in laws)} are attached" if laws else "no law is attached"
                found.append(Problem("flow.opening-law", f"the flow's opening {opening!r} takes one law, and {attached}",
                                     f"attach one law there, exp.attach(Inflow(...), Windkessel(...) or Valve(), "
                                     f"medium=..., surface={opening}), or one over several openings, "
                                     f"SharedDistal(...) with surface=(<opening>, <opening>)"))
        if found:
            return found
        laws = [held[o][0] for o in openings]
        one = [port_law_named(r.kind).openings == 1 for r in laws]
        reason = well_posed([port_law_named(r.kind).row for r in laws],
                            [port_law_named(r.kind).row_of(r.settings) if alone else (0.0,) * 4
                             for r, alone in zip(laws, one, strict=True)])
        given = ", ".join(r.id for r in laws if port_law_named(r.kind).row is RowKind.FLUX)
        return [Problem("flow.openings-ill-posed", reason,
                        f"give one of the openings a pressure law in place of its flux law ({given}): "
                        f"exp.attach(Windkessel(proximal=..., compliance=..., distal=...), medium=..., "
                        f"surface=<that opening>), whose pressure the flow's rows solve for")] if reason else []

    def porous_problems(self, held: Sequence[ResolvedRegistration], region: Region | None,
                        openings: Sequence[ResolvedRegistration] = ()) -> list[Problem]:
        """A porous region is a ``Plug``'s volume, held by IMEX Euler or its steady limit, in a box whose axis is z
        (a straight tube along z, a tube tree, an open vessel along +z); in a periodic box its extent along z under the
        period (a longer one would overlap its own image), and in an open vessel within the span between its openings
        (``openings``, the laws at them), outside which the box holds no face of it."""
        if not held:
            return []
        found: list[Problem] = []
        span = open_span(openings, region) if self.open and region is not None else None
        for registration in held:
            kind = registration.geometry.geometry if registration.geometry is not None else None
            cylinder = porous_cylinder(registration.geometry)
            if cylinder is not None:
                found += self.placement_problems(registration, cylinder, span)
            if cylinder is None:
                named = f"a {type(kind).name} geometry" if kind is not None else "no geometry"
                found.append(Problem("flow.porous-kind", f"{registration.id} binds {registration.subject!r}, which has "
                                                         f"{named}: the flow holds a porous region of a Plug's solid",
                                     f"give {registration.subject!r} a Plug geometry, Structure("
                                     f"{registration.subject!r}, geometry=Plug(axis=..., radius=..., length=..., "
                                     f"centre=...))"))
        if isinstance(self.scheme, RK3Scheme):
            found.append(Problem("flow.porous-scheme", "RK3's stages split the penalty from the projection, which for a "
                                                       "porous region's resistivity leaves its Darcy flux to the "
                                                       "splitting: a flow holding one steps by IMEX Euler or its "
                                                       "steady limit",
                                 "write FVM(scheme=Scheme.IMEXEuler()), or FVM(stokes=True, scheme=Scheme.Steady()), "
                                 "or name no scheme"))
        axis = _unit(region.parameters["axis"]) if region is not None and region.shape == "cylinder" else None
        if axis is not None and (lattice_of(axis) != (0, 1) or (self.open and axis[2] < 0.0)):
            found.append(Problem("flow.porous-axis", f"the tube's axis {axis} is not along "
                                                     f"{'+z' if self.open else 'z'}: a porous region is held in a box "
                                                     f"whose axis is z (tilted boxes come later)",
                                 "align the tube with +z, Tube(axis=+Z, ...)"))
        return found

    def placement_problems(self, registration: ResolvedRegistration, cylinder: Mapping[str, Any],
                           span: tuple[float, float] | None) -> list[Problem]:
        """Where a periodic box or an open vessel cannot place a porous region (:meth:`porous_problems`): its extent
        along z, ``|a_z| L + 2 R sqrt(1 - a_z^2)``, against the period or the open span (world z)."""
        axis = _unit(cylinder["axis"])
        along = abs(float(axis[2]))
        extent = along * cylinder["length"] + 2.0 * cylinder["radius"] * math.sqrt(max(1.0 - along * along, 0.0))
        if not self.open:
            period = float(cast(float, cast(PeriodicBox, self.domain).length.value))
            if extent < period:
                return []
            return [Problem("flow.porous-period", f"{registration.id}'s region reaches {extent:.4g} m along the box's "
                                                  f"axis, not under its period of {period:.4g} m: a periodic box repeats "
                                                  f"it every period, so it would overlap its own image",
                            f"give the box a period longer than the region, FVM(domain=PeriodicBox(length=...)) above "
                            f"{extent:.4g} m, or shorten {registration.subject!r}'s Plug")]
        if span is None:
            return []
        centre = float(cylinder["centre"][2])
        if centre + 0.5 * extent > span[0] and centre - 0.5 * extent < span[1]:
            return []
        return [Problem("flow.porous-outside", f"{registration.id}'s region, along z from {centre - 0.5 * extent:.4g} "
                                               f"to {centre + 0.5 * extent:.4g} m, lies outside the open vessel's span "
                                               f"between its openings, {span[0]:.4g} to {span[1]:.4g} m: the box holds "
                                               f"no face of it",
                        f"place {registration.subject!r}'s Plug within the vessel, Plug(centre=...) between its "
                        f"openings")]

    def field_for(self, scheme: str) -> SolvedField:
        """The solved-field contract's declarations of this FVM stepped by ``scheme`` (its ledger route and its steady
        solve are the scheme's; its ports and affine response its domain's), which SF-12's suite holds it to."""
        return fvm_field(scheme, self.open)

    @classmethod
    @override
    def field_declarations(cls) -> tuple[SolvedField, ...]:
        """One declaration a scheme on a periodic box, and one a scheme that steps an open vessel (:meth:`field_for`)."""
        return (*(fvm_field(name) for name in SCHEMES),
                *(fvm_field(name, opened=True) for name in SCHEMES
                  if scheme_named(name).stage_kind is StageKind.MONOLITHIC_PROJECTED_CG))

    @override
    def time_step(self, query: StepQuery) -> StepRule | None:
        """The flow's step rule (``mime.dynamics.flow._steps``): the scheme from the viscous limit, the CFL predicted
        from the drive's peak in the empty tube, the refusal past the line, and ``dt="auto"``'s largest stable step,
        each judged at the step the flow's node takes (its ``rate=`` included)."""
        region = query.region
        if self.open:  # an open vessel's step rule (the CFL from the inflow's peak) comes with the region builder
            return StepRule(None, lambda step, every: [], lambda step, every: (), unbounded=OPEN_STEP)
        if region is None or "viscosity" not in query.values or "density" not in query.values:
            return None
        given = query.settings.get("spacing")
        spacing = float(given) if isinstance(given, (int, float)) else None
        if region.shape == TREE:
            # A tree's flow along any direction: the line along the diagonal, its least (g(n) a third); its peak from
            # its widest lumen; its cells the tree's box's.
            tree = self.tree_layout(region, spacing)
            axis = (1.0, 1.0, 1.0)
            radius = float(np.max(tree_of(region)[1]))
            h, cells = tree.spacing, tree.cells
        elif region.shape == "cylinder" and lattice_of(_unit(region.parameters["axis"])) is not None:
            axis = _unit(region.parameters["axis"])
            radius = float(region.parameters["radius"])  # type: ignore[arg-type]
            asked = self.resolved_spacing(region, query.registrations) if spacing is None else None
            layout = periodic_tube_layout(radius, float(query.settings["domain.length"]),  # type: ignore[arg-type]
                                          cells_across=self.cells_across, spacing=spacing if asked is None else asked,
                                          margin_cells=self.domain.margin, lattice=lattice_of(axis) or (0, 1))
            h, cells = layout.spacing, layout.cells
        else:
            return None
        nu = query.values["viscosity"] / query.values["density"]
        drive = axial_drive(query.drives, axis)
        wide = radius + WALL_REACH_CELLS * h
        peak = predicted_peak(drive, wide, nu)
        alpha = womersley_number(drive, wide, nu) if drive.amplitude else None
        convection = not self.stokes
        if convection and not math.isfinite(peak):
            return StepRule(None, lambda step, every: [], lambda step, every: (), refusals=(Problem(
                "flow.cfl-predicted",
                f"the drive's predicted peak speed is {peak!r} m/s (from its steady part {drive.steady!r}, its "
                f"harmonic's amplitude {drive.amplitude!r} and frequency {drive.frequency!r} Hz, nu {nu!r} m^2/s): no "
                f"CFL number can be predicted, so no step is known to be stable",
                "write the fluid's viscosity and density, exp.declare(...), and the drive's BodyForce(steady=..., "
                "amplitude=..., frequency=...) as finite numbers"),))

        def line_of(name: str) -> float | None:
            return host_line(name, axis) if convection else None

        def safe(name: str) -> float | None:
            scheme = scheme_named(name)
            bounds = [b for b in (scheme.viscous_limit(nu, h),) if b is not None]
            line = line_of(name)
            if line is not None and peak > 0.0:
                bounds.append(line * h / peak)
            return min(bounds) if bounds else None

        porous = any(drive.law == POROUS_LAW for drive in query.drives)
        held = region.shape == TREE or porous
        candidates = [self.scheme.group] if self.scheme is not None else \
            [IMEXEuler.name] if held else [RK3.name, IMEXEuler.name]
        steps = {name: safe(name) for name in candidates}
        bounded = {name: step for name, step in steps.items() if step is not None}
        if self.scheme is None and IMEXEuler.name in steps and steps[IMEXEuler.name] is None:
            largest = None  # IMEX Euler is stable at any step here (Stokes, or no flow predicted): nothing bounds it
        else:
            largest = max(bounded.values()) if bounded else None

        count = cells[0] * cells[1] * cells[2]
        where, available = device_memory()
        coarser = (f"a coarser spacing (FVM(spacing=...) over {h:.4g} m: doubling it divides the cells by about "
                   f"eight)" if given is not None else
                   f"a coarser grid (FVM(cells_across=...) under {self.cells_across}: halving it divides the cells by "
                   f"about eight)")

        def problems(step: float, every: int) -> list[Problem]:
            name = tree_scheme(self.scheme) if held else chosen_scheme(self.scheme, nu, h, step)
            found: list[Problem] = []
            needed = memory_estimate(name, count)
            if available is not None and needed > available:
                found.append(Problem(
                    "flow.memory",
                    f"the flow's box is {cells[0]} x {cells[1]} x {cells[2]} = {count} cells at "
                    f"{2.0 * radius / h:.4g} cells across the lumen, whose {name} run needs about "
                    f"{needed / 2 ** 30:.3g} GiB (the cells times the bytes a cell a CPU run was measured to hold, "
                    f"as the validity line says), more than {where}, {available / 2 ** 30:.3g} GiB",
                    f"{coarser}, a shorter period (PeriodicBox(length=...)), or a machine with more memory"))
            limit = scheme_named(name).viscous_limit(nu, h)
            if limit is not None and step > limit * (1.0 + 1e-12):
                found.append(Problem(
                    "flow.viscous-limit",
                    f"the flow's step {step:.4g} s gives the {name} scheme's explicit viscosity nu dt / h^2 = "
                    f"{nu * step / h ** 2:.3g}, past its viscous number {nu * limit / h ** 2:.3g}: it would not be "
                    f"stable",
                    f"use dt <= {limit:.4g} s, or dt='auto', or Scheme.IMEXEuler() (implicit viscosity)" if every == 1
                    else f"use dt <= {limit / every:.4g} s (the flow steps every {every} base steps, its rate=, so a "
                         f"step of {limit:.4g} s), a shorter rate=, or Scheme.IMEXEuler() (implicit viscosity)"))
            line = line_of(name)
            if line is not None and not peak * step / h <= line * (1.0 + 1e-9):
                bound = line * h / peak
                found.append(Problem(
                    "flow.cfl-predicted",
                    f"the drive gives the empty tube a peak speed of {peak:.4g} m/s from rest (Poiseuille's centreline "
                    f"and twice the harmonic's Womersley peak), so max|u| dt / h = {peak * step / h:.3g} at the flow's "
                    f"step {step:.4g} s, past the {name} scheme's CFL line of {line:.3g} along the vessel's axis: its "
                    f"convection would not be stable",
                    f"use dt <= {bound:.4g} s, or dt='auto'" if every == 1
                    else f"use dt <= {bound / every:.4g} s (the flow steps every {every} base steps, its rate=, so a "
                         f"step of {bound:.4g} s), or a shorter rate="))
            return found

        def report(step: float, every: int) -> tuple[str, ...]:
            name = tree_scheme(self.scheme) if held else chosen_scheme(self.scheme, nu, h, step)
            line = line_of(name)
            room = f"{available / 2 ** 30:.3g} GiB" if available is not None else "which the system does not report"
            box = (f"the box: {cells[0]} x {cells[1]} x {cells[2]} = {count} cells of "
                   f"{h:.4g} m, {2.0 * radius / h:.4g} across the {'widest ' if region.shape == TREE else ''}lumen; "
                   f"the {name} run's memory about {memory_estimate(name, count) / 2 ** 20:.4g} MiB (the cells times the bytes a cell a CPU run was "
                   f"measured to hold), against {where}, {room}")
            return (box, *report_lines(name, step, every, nu * step / h ** 2,
                                       RK3.viscous_number if name == RK3.name else None,
                                       peak * step / h if line is not None else None, line, drive.period, bounded,
                                       alpha))

        return StepRule(largest, problems, report, period=drive.period)

    def layout_of(self, region: Region, settings: Mapping[str, Magnitude],
                  registrations: Sequence[ResolvedRegistration] = ()) -> TubeLayout:
        """The periodic box this FVM lays out around the tube ``region`` from its settings in SI: what its node is
        built on and what its exchange partners read their grid from (:meth:`exchange_frame`). With no spacing written,
        the finer of the lumen's ``cells_across`` and N cells across each resolved body's smallest feature (B8.2)."""
        axis = _unit(region.parameters["axis"])
        lattice = lattice_of(axis)
        assert lattice is not None
        given = settings.get("spacing") if self.spacing is not None else None
        spacing = float(given) if isinstance(given, (int, float)) else None
        asked = self.resolved_spacing(region, registrations) if spacing is None else None
        return periodic_tube_layout(float(region.parameters["radius"]),  # type: ignore[arg-type]
                                    float(settings["domain.length"]),  # type: ignore[arg-type]
                                    cells_across=self.cells_across, spacing=spacing if asked is None else asked,
                                    margin_cells=self.domain.margin, lattice=lattice)

    def registration_flags(self, registrations: Sequence[ResolvedRegistration]) -> tuple[tuple[str, str, str], ...]:
        """The flags the flow's node raises for each body resolved on its grid, owned by the drag that registered it
        (its id, the node's field, what it means): the body nearer the tube's wall than the free body's measured line
        (``body_<i>_near_wall``)."""
        from mime.nodes.flow.fvm._resolved_node import LAGGED_RATIOS, NEAR_WALL_CELLS

        resolved = [r for r in registrations if r.kind == RESOLVED_PARTNER]
        return tuple(flag for index, r in enumerate(resolved) for flag in (
            (r.id, f"body_{index}_near_wall",
             f"{r.subject}'s solid came within {NEAR_WALL_CELLS:g} cells of the tube's wall (its part's enclosing "
             f"sphere's gap, body_{index}_gap, in cells): a free resolved body's accuracy was measured to fall there "
             f"(spun force-free beside a tube's wall, 8.3 % off the MRS's rate at 3 cells and 23 % at 2), so within "
             f"it the body is the near-wall resolved-body phase's"),
            (r.id, f"body_{index}_ratio_unmeasured",
             f"{r.subject}'s density ratio (body_{index}_ratio, its mass over the fluid's density times its hull's "
             f"volume) lay outside the {LAGGED_RATIOS[0]:g} to {LAGGED_RATIOS[1]:g} over which the lagged coupling's "
             f"edge was measured: the build judged its step against the nearest measured ratio's edge")))

    def resolved_spacing(self, region: Region, registrations: Sequence[object]) -> float | None:
        """The finer of the lumen's spacing (its diameter over ``cells_across``) and each resolved body's (N cells
        across its smallest feature's radius), where a resolved body asks a finer one; ``None`` where none does."""
        from mime.dynamics.flow._resolved import finest_spacing

        asked = [spacing for r in registrations if getattr(r, "kind", None) == RESOLVED_PARTNER
                 for spacing in (finest_spacing(getattr(r, "hull", None), getattr(r, "settings", {})),)
                 if spacing is not None]
        lumen = 2.0 * float(region.parameters["radius"]) / self.cells_across  # type: ignore[arg-type]
        return min(asked) if asked and min(asked) < lumen else None

    def exchange_frame(self, node: str, region: Region | None, settings: Mapping[str, Magnitude],
                       registrations: Sequence[ResolvedRegistration] = ()) -> ExchangeFrame | None:
        """What a partner exchanging with this field reads of it (SF-4): its grid, and the world point of the grid's
        origin, the tube's axis point less where the layout puts the axis in the box. ``None`` where the region is no
        straight tube the layout takes, or where the vessel is open (its node takes no exchange yet; the partner's plan
        refuses it then)."""
        if self.open or region is None or region.shape != "cylinder" or \
                lattice_of(_unit(region.parameters["axis"])) is None:
            return None
        layout = self.layout_of(region, settings, registrations)
        point = shape_named(region.shape).axis_point(region.parameters)
        assert point is not None
        origin = tuple(float(point[i]) - layout.centre[i] for i in range(3))
        return ExchangeFrame(node, "cartesian_mac", layout.cells, layout.spacing,
                             (origin[0], origin[1], origin[2]), ("velocity_x", "velocity_y", "velocity_z"),
                             EXCHANGE_FIELDS, None if self.scheme is None else self.scheme.group)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        region = ctx.region("medium")
        assert region is not None
        if self.open:
            _open_build(self, ctx, b, [r for r in registrations if r.kind != EXCHANGE_PARTNER], region)
            return
        if region.shape == TREE:
            self.build_tree(ctx, b, region, registrations)
            return
        layout = self.layout_of(region, ctx.settings, registrations)
        lattice = lattice_of(_unit(region.parameters["axis"]))
        viscosity, density = ctx.scalar_value("viscosity"), ctx.scalar_value("density")
        partners = any(registration.kind == EXCHANGE_PARTNER for registration in registrations)
        point = shape_named(region.shape).axis_point(region.parameters)
        assert point is not None
        porous = porous_arguments(registrations, [float(point[i]) - layout.centre[i] for i in range(3)])
        resolved = [registration for registration in registrations if registration.kind == RESOLVED_PARTNER]
        scheme = porous_scheme(self.scheme) if porous else \
            chosen_scheme(self.scheme, viscosity / density, layout.spacing, ctx.timestep)
        band, interior = DEFAULT_BAND, _INTERIOR_NAMES[self.wall.delta_in]
        images = "own" if lattice == (0, 1) else "transverse"
        # The wall's band is booked under the wall's own region: the medium fills a tube's lumen, its solid that tube's
        # wall (B9: a region's loss under its own name).
        wall = f"{region.path.rpartition('.')[0]}.wall"
        if scheme_named(scheme).stage_kind is StageKind.EXPLICIT_PROJECTED:
            b.add_node(ExplicitFlowNode(
                ctx.node_name(), ctx.timestep, cells_x=layout.cells[0], cells_y=layout.cells[1],
                cells_z=layout.cells[2], spacing=layout.spacing, viscosity=viscosity, density=density,
                tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius, scheme=scheme,
                convection=not self.stokes, band=band, interior=interior, periodic_images=images, wall_name=wall))
            return
        settings: dict[str, Any] = dict(
            cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
            spacing=layout.spacing, viscosity=viscosity, density=density,
            tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius, scheme=scheme,
            convection=not self.stokes, tolerance=self.tolerance, preconditioner=self.preconditioner, band=band,
            interior=interior, periodic_images=images, solve_report_rows=self.solve_report_rows,
            wall_name=wall, exchange=partners, **porous)
        if not resolved:
            b.add_node(ImplicitFlowNode(ctx.node_name(), ctx.timestep, **settings))
            return
        _resolved_build(self, ctx, b, resolved, layout, region, settings)

    def build_tree(self, ctx: BuildContext, b: DynamicsBuilder, region: Region,
                   registrations: Sequence[ResolvedRegistration] = ()) -> None:
        """A tube tree's flow: its box from the tree (:meth:`tree_layout`), the tree's points moved into it, and a
        porous region's with them."""
        _, radii, segments, fillet = tree_of(region)
        layout = self.tree_layout(region, ctx.scalar("spacing") if self.spacing is not None else None)
        b.add_node(ImplicitFlowNode(
            ctx.node_name(), ctx.timestep, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
            spacing=layout.spacing, viscosity=ctx.scalar_value("viscosity"), density=ctx.scalar_value("density"),
            tree_points=np.asarray(layout.points), tree_radii=radii, tree_fillet=fillet, tree_segments=segments,
            scheme=tree_scheme(self.scheme), convection=not self.stokes, tolerance=self.tolerance,
            preconditioner=self.preconditioner, band=DEFAULT_BAND, interior=_INTERIOR_NAMES[self.wall.delta_in],
            solve_report_rows=self.solve_report_rows, wall_name=f"{region.path.rpartition('.')[0]}.wall",
            **porous_arguments(registrations, layout.origin)))


def _resolved_build(fvm: FVM, ctx: BuildContext, b: DynamicsBuilder, resolved: Sequence[ResolvedRegistration],
                    layout: TubeLayout, region: Region, settings: Mapping[str, Any]) -> None:
    """The flow's node with the resolved bodies on its grid (``mime.nodes.flow.fvm._resolved_node``): each body's
    pose and twist read at the step's start, its load delivered back to it under its drag's id."""
    from mime.dynamics.flow._resolved import resolved_spec
    from mime.nodes.flow.fvm._resolved_node import ResolvedSpec

    del fvm
    band = BANDS[DEFAULT_BAND]
    specs: list[ResolvedSpec] = []
    for registration in resolved:
        assert registration.hull is not None, "the drag's plan refuses a body with no hull"
        specs.append(resolved_spec(registration.hull, registration.id, layout.spacing, band))
    point = shape_named(region.shape).axis_point(region.parameters)
    assert point is not None
    origin = tuple(layout.centre[i] - float(point[i]) for i in range(3))
    node = b.add_node(ResolvedFlowNode(ctx.node_name(), ctx.timestep, bodies=specs, origin=origin, **settings))
    for index, registration in enumerate(resolved):
        b.connect(b.read(registration, "pose"), into=node[f"body_{index}_pose"])
        b.connect(b.read(registration, "twist"), into=node[f"body_{index}_twist"])
        b.connect(b.read(registration, "mass"), into=node[f"body_{index}_mass"])
        b.connect(node[f"body_{index}_load"], into=b.target(registration, "load"))
        b.connect(node[f"body_{index}_share"], into=b.target(registration, "displaced"), part="displaced")


def takes_exchange(frame: ExchangeFrame, viscosity: float, density: float, dt: float) -> bool:
    """Whether the FVM a frame describes steps with a scheme whose node takes exchange deliveries at these values: the
    monolithic projected CG's (IMEX Euler, its steady limit); RK3's explicit stages take none yet."""
    scheme = frame.scheme if frame.scheme is not None else chosen_scheme(None, viscosity / density, frame.spacing, dt)
    return scheme_named(scheme).stage_kind is StageKind.MONOLITHIC_PROJECTED_CG


def tree_scheme(given: IMEXEulerScheme | SteadyScheme | RK3Scheme | None) -> str:
    """The scheme a tube tree's flow steps with: the one named (never RK3: refused when planned), else IMEX Euler."""
    return given.group if given is not None else IMEXEuler.name


def open_planes(registrations: Sequence[ResolvedRegistration],
                region: Region) -> tuple[Mapping[str, Magnitude], Mapping[str, Magnitude]] | None:
    """An open vessel's low and high opening's parameters (its normal and offset), from the laws at its openings: the
    low one's normal into the vessel along its axis. ``None`` where a plane has no law at it (refused elsewhere)."""
    openings = FVM.openings_of(region)
    at: dict[str, Mapping[str, Magnitude]] = {}
    for registration in registrations:
        given = registration.subjects_parameters or (registration.subject_parameters,)
        at.update(zip(registration.surfaces, given, strict=True))
    if any(o not in at for o in openings):
        return None
    axis = _unit(region.parameters["axis"])
    planes = tuple("low" if sum(float(n) * a for n, a in zip(cast(tuple[float, ...], at[o]["normal"]), axis,
                                                             strict=True)) > 0.0 else "high" for o in openings)
    if sorted(planes) != ["high", "low"]:
        return None
    low, high = (next(at[o] for o, plane in zip(openings, planes, strict=True) if plane == side)
                 for side in ("low", "high"))
    return low, high


def open_span(registrations: Sequence[ResolvedRegistration], region: Region) -> tuple[float, float] | None:
    """An open vessel's span along z (world, m) between its low and high openings' planes (:func:`open_planes`)."""
    planes = open_planes(registrations, region)
    if planes is None:
        return None
    low, high = planes
    return float(cast(float, low["offset"])), -float(cast(float, high["offset"]))


def _open_build(fvm: FVM, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration],
                region: Region) -> None:
    """The open vessel's node: the box along z from its low plane to its high one, each opening on the plane its
    surface faces (its normal into the vessel along the axis: the low plane; against it: the high one), each law from
    its registration (its kind, its settings as a row of parameters, its id the ledger's name, the openings it binds
    in the order written), and each porous region from its law's."""
    porous = porous_held(registrations)
    registrations = [r for r in registrations if r.kind != POROUS_LAW]
    openings = fvm.openings_of(region)
    at: dict[str, Mapping[str, Magnitude]] = {}
    for registration in registrations:
        given = registration.subjects_parameters or (registration.subject_parameters,)
        at.update(zip(registration.surfaces, given, strict=True))
    axis = _unit(region.parameters["axis"])
    planes = tuple("low" if sum(float(n) * a for n, a in zip(cast(tuple[float, ...], at[o]["normal"]), axis,
                                                             strict=True)) > 0.0 else "high" for o in openings)
    low, high = (next(at[o] for o, plane in zip(openings, planes, strict=True) if plane == side)
                 for side in ("low", "high"))
    length = -(float(cast(float, low["offset"])) + float(cast(float, high["offset"])))
    laws = sorted(registrations, key=lambda r: min(openings.index(s) for s in r.surfaces))
    placed: dict[str, Any] = {} if planes == ("low", "high") and all(len(r.surfaces) == 1 for r in laws) else \
        {"opening_planes": planes, "law_openings": tuple(tuple(openings.index(s) for s in r.surfaces) for r in laws)}
    radius = float(region.parameters["radius"])  # type: ignore[arg-type]
    spacing = ctx.scalar("spacing") if fvm.spacing is not None else None
    layout = periodic_tube_layout(radius, length, cells_across=fvm.cells_across, spacing=spacing,
                                  margin_cells=fvm.domain.margin)
    viscosity, density = ctx.scalar_value("viscosity"), ctx.scalar_value("density")
    scheme = fvm.scheme.group if fvm.scheme is not None else IMEXEuler.name
    wall = f"{region.path.rpartition('.')[0]}.wall"
    point = shape_named(region.shape).axis_point(region.parameters)
    assert point is not None
    # The box runs along +z from the low opening's plane (its offset along its normal, +z), the tube's axis at the
    # layout's centre across it.
    origin = (float(point[0]) - layout.centre[0], float(point[1]) - layout.centre[1], float(cast(float, low["offset"])))
    b.add_node(OpenFlowNode(
        ctx.node_name(), ctx.timestep, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
        spacing=layout.spacing, viscosity=viscosity, density=density, tube_centre=layout.centre,
        tube_axis=(0.0, 0.0, 1.0), tube_radius=layout.radius, scheme=scheme, convection=not fvm.stokes,
        tolerance=fvm.tolerance, preconditioner=fvm.preconditioner, band=DEFAULT_BAND,
        interior=_INTERIOR_NAMES[fvm.wall.delta_in], solve_report_rows=fvm.solve_report_rows, wall_name=wall,
        openings=openings, opening_laws=tuple(r.kind for r in laws),
        opening_parameters=[port_law_named(r.kind).row_of(r.settings) for r in laws],
        opening_law_ids=tuple(r.id for r in laws),
        **placed, **porous_arguments(porous, origin)))


def memory_estimate(scheme: str, cells: int) -> float:
    """The bytes a run of ``scheme`` on ``cells`` cells holds, in the run's dtype (:data:`BYTES_PER_CELL`)."""
    return cells * BYTES_PER_CELL[(scheme_named(scheme).stage_kind, str(jnp.dtype(run_dtype())))]


def device_memory() -> tuple[str, int | None]:
    """The memory a run's arrays can have and its size (``mime.build._maddening.device_memory``)."""
    from mime.build._maddening import device_memory as held

    return held()


def host_line(scheme: str, axis: Sequence[float]) -> float | None:
    """The scheme's CFL line along ``axis``, ``TimeScheme.cfl_limit``'s ``line g(n)``, on the host in float64 whatever
    the run's dtype (the node's per-step check computes it in the run's); ``None`` for a scheme with no line."""
    line = scheme_named(scheme).cfl_line
    if line is None:
        return None
    magnitude = [abs(float(c)) for c in axis]
    return line * max(magnitude) / sum(magnitude)


def _unit(vector: object) -> tuple[float, float, float]:
    held = [float(c) for c in vector]  # type: ignore[attr-defined]
    length = (held[0] ** 2 + held[1] ** 2 + held[2] ** 2) ** 0.5
    return (held[0] / length, held[1] / length, held[2] / length)


__all__ = ["FVM", "Band", "IMEXEulerScheme", "RK3Scheme", "Scheme", "SteadyScheme", "chosen_scheme", "host_line",
           "lattice_of", "takes_exchange"]
