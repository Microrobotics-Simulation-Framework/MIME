"""The finite-volume flow field of a medium (F0's Part B): ``Medium("water", fills=tube.lumen, material=...,
fields={"flow": FVM(domain=PeriodicBox(length=mm(1)))})``.

A field kind's value is a frozen settings object naming its kind and version, its settings grouped by the owner each
belongs to (B8.1): the grid kind's ``spacing`` (or ``cells_across``, the lumen's diameter in cells), the time scheme's
``scheme``, the solve's ``tolerance`` and ``preconditioner``, the wall's lift onto the grid (``wall=Band(...)``, the
stiff interior ``delta_in`` the wall's), and the domain it lives on. ``stokes`` is physical: the Stokes approximation, convection off. The fluid's properties
are never its settings: they come from the medium's material record or ``exp.declare`` (``viscosity``, ``density``).
The flow is driven by laws bound to the medium (``BodyForce`` on a periodic box).

F1a builds the Cartesian MAC grid kind, IMEX Euler and its steady limit, Jacobi, the default smooth band, on one
domain kind, the periodic box around a straight circular tube (aligned with z, or tilted in the x-z plane along a
lattice direction of the box)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated, override

from mime.gradients import GradientStatus
from mime.nodes.flow.fvm import BANDS, DEFAULT_BAND, DEFAULT_CELLS_ACROSS, INTERIORS, ExplicitFlowNode, \
    ImplicitFlowNode, periodic_tube_layout
from mime.nodes.flow.fvm._node import BYPASS_LINE, BYPASS_MEASURED, CALIBRATED_CELLS, GEOMETRY, STALL_LINE, \
    STOKES_REYNOLDS
from mime.nodes.flow.fvm._schemes import RK3, IMEXEuler, StageKind, Steady, scheme_named
from mime.nodes.flow.fvm._solve import DEFAULT_PRECONDITIONER, DEFAULT_TOLERANCE, PRECONDITIONERS, \
    STAGNATION_FACTOR, STAGNATION_LEAST_WINDOW, STAGNATION_MEASURED, STAGNATION_SPAN
from mime.registry import NUMERICAL, PHYSICAL, BuildContext, Dynamics, Flag, GradientRecord, Port, Problem, Region, \
    SettingGroup, StepQuery, StepRule, convention, register_dynamics, register_setting_group
from mime.dynamics.flow._fields import fvm_field
from mime.dynamics.flow._steps import WALL_REACH_CELLS, axial_drive, predicted_peak, report_lines
from mime.fields._solved import SolvedField
from mime.units import Magnitude
from mime.scene import PeriodicBox
from mime.things import ThingKind
from mime.units import Length, Quantity
from mime.vocabulary import FLOW_BODY_FORCE, FLOW_WAVEFORM

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration


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


#: The least margin of solid, in cells, between a tube's band and its image: their fields flatten 3 cells into the
#: solid (the band's outer half, 1, and the interior's ramp, 2), and the images' tie lies halfway across the gap of
#: the box's reach (3) and its margin, so the margin keeps it flat from 3 on.
MARGIN_FLOOR = 3


#: The stiff interior's depths in cells, by the node's name for each.
_INTERIOR_NAMES = {depth: name for name, depth in INTERIORS.items()}


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
    builds = {"": (ImplicitFlowNode, ExplicitFlowNode)}
    applies_to = frozenset({ThingKind.MEDIUM})
    accepts = {FLOW_BODY_FORCE.name: Port("body_force"), FLOW_WAVEFORM.name: Port("waveform")}
    raises = (
        Flag("solve_flagged", "the step's solve stopped other than converging: at its iteration cap, stagnated (its "
                              "true residual no longer falling), or at a residual that is not finite, so the step's "
                              "values hold only to the residual it reached. Its report, solve_report (iterations, "
                              "the true relative residual against the tolerance, and one count for each way it "
                              "stopped), says which; a run recording it per step (history=True) keeps every step's",
             warns=True),
        Flag("reynolds_flagged", f"with the Stokes approximation (stokes=True), the lumen's Reynolds number by its "
                                 f"diameter and mean speed was above the declared {STOKES_REYNOLDS:g} (reynolds)"),
        Flag("bypass_flagged", f"the flux through the solid beyond the band's middle was more than {BYPASS_LINE:g} "
                               f"of the lumen's (bypass_ratio): a finer spacing lowers it, and the stiffer interior "
                               f"wall=Band(delta_in=0.01) about halves it (the validity line's measurements)"),
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
        f"{DEFAULT_TOLERANCE['float64']:g} in float64, {DEFAULT_TOLERANCE['float32']:g} in float32), the step's "
        "values and gradients holding to about it; a stagnation is no check of the true residual (one every 20 "
        f"iterations) over a window of {STAGNATION_SPAN} iterations per cell along the box's longest side (at least "
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
        f"geometry: {GEOMETRY}",
        f"the stall detector |N_Krylov| / D_visc, flagged above {STALL_LINE:g} (stall_flagged), is written with the "
        f"energy ledger attached; the build does not yet meter the flow, so a run through the build has no stall "
        f"report",
    )
    values = {"viscosity": "viscosity", "density": "density"}
    unmetered = ("its step identity is written and gated on its node (every term by formula, the solve's residual "
                 "included), but the build does not yet name its hosted laws (the viscosity, the wall's penalty, the "
                 "drive) for the ledger, so it leaves the flow unmetered")
    setting_roles = {
        "domain": PHYSICAL,
        "spacing": convention(NUMERICAL, "none given: the lumen's diameter over cells_across, the band's sweet-spot "
                                         "resolution on straight tubes"),
        "cells_across": convention(NUMERICAL, "the band's calibrated resolution across the narrowest lumen"),
        "scheme": convention(NUMERICAL, "none named: RK3 where its viscous number holds at the run's dt, IMEX Euler "
                                        "otherwise (the scheme from the viscous limit, RK3 first)"),
        "stokes": convention(PHYSICAL, "convection is on unless the Stokes approximation is declared"),
        "tolerance": convention(NUMERICAL, "none given: each dtype's own default, at its reachable floor for the true "
                                           "residual"),
        "preconditioner": convention(NUMERICAL, "Jacobi, the one preconditioner measured in three dimensions"),
        "wall": convention(NUMERICAL, "the one registered smooth band, with its default stiff interior (B3)"),
        "report_rows": convention(NUMERICAL, "the derivative solves' reports summed over the run"),
    }

    domain: PeriodicBox
    spacing: Annotated[Quantity | None, Length] = None
    cells_across: int = DEFAULT_CELLS_ACROSS
    scheme: IMEXEulerScheme | SteadyScheme | RK3Scheme | None = None
    stokes: bool = False
    tolerance: float | None = None
    preconditioner: str = DEFAULT_PRECONDITIONER
    wall: Band = Band()
    report_rows: int = 0

    @override
    def value_problems(self, values: Mapping[str, Magnitude]) -> list[Problem]:
        """What the settings rule out, refused when planned."""
        found: list[Problem] = []
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

    @override
    def region_problems(self, region: Region | None) -> list[Problem]:
        """The region must be a straight circular tube's (a cylinder), its axis along z or along a lattice direction
        ``(a, 0, b)`` of the periodic box."""
        if region is None or region.shape != "cylinder":
            fills = f"fills {region.path}, a {region.shape} region" if region is not None else "fills no region"
            return [Problem("flow.region", f"the flow's medium {fills}; the FVM's periodic box holds a straight "
                                           f"circular tube's lumen", "fill a Tube's lumen: Medium(fills=tube.lumen)")]
        if lattice_of(_unit(region.parameters["axis"])) is None:
            return [Problem("flow.box-axis", f"the tube's axis {_unit(region.parameters['axis'])} is not along z or a "
                                             f"lattice direction (a, 0, b) of the periodic box",
                            "align the tube with z, Tube(axis=+Z, ...), or tilt it in the x-z plane at a slope a/b of "
                            "small integers, Tube(axis=(a, 0, b), ...)")]
        return []

    def field_for(self, scheme: str) -> SolvedField:
        """The solved-field contract's declarations of this FVM stepped by ``scheme`` (its ledger route and its steady
        solve are the scheme's), which SF-12's suite holds it to."""
        return fvm_field(scheme)

    @override
    def time_step(self, query: StepQuery) -> StepRule | None:
        """The flow's step rule (``mime.dynamics.flow._steps``): the scheme from the viscous limit, the CFL predicted
        from the drive's peak in the empty tube, the refusal past the line, and ``dt="auto"``'s largest stable step."""
        region = query.region
        if region is None or region.shape != "cylinder" or "viscosity" not in query.values \
                or "density" not in query.values or lattice_of(_unit(region.parameters["axis"])) is None:
            return None
        axis = _unit(region.parameters["axis"])
        radius = float(region.parameters["radius"])  # type: ignore[arg-type]
        nu = query.values["viscosity"] / query.values["density"]
        given = query.settings.get("spacing")
        layout = periodic_tube_layout(radius, float(query.settings["domain.length"]),  # type: ignore[arg-type]
                                      cells_across=self.cells_across,
                                      spacing=float(given) if isinstance(given, (int, float)) else None,
                                      margin_cells=self.domain.margin, lattice=lattice_of(axis) or (0, 1))
        h = layout.spacing
        drive = axial_drive(query.drives, axis)
        peak = predicted_peak(drive, radius + WALL_REACH_CELLS * h, nu)
        convection = not self.stokes

        def line_of(name: str) -> float | None:
            scheme = scheme_named(name)
            return float(scheme.cfl_limit(axis)) if convection and scheme.cfl_line is not None else None

        def safe(name: str) -> float | None:
            scheme = scheme_named(name)
            bounds = [b for b in (scheme.viscous_limit(nu, h),) if b is not None]
            line = line_of(name)
            if line is not None and peak > 0.0:
                bounds.append(line * h / peak)
            return min(bounds) if bounds else None

        candidates = [self.scheme.group] if self.scheme is not None else [RK3.name, IMEXEuler.name]
        steps = {name: safe(name) for name in candidates}
        bounded = {name: step for name, step in steps.items() if step is not None}
        if self.scheme is None and IMEXEuler.name in steps and steps[IMEXEuler.name] is None:
            largest = None  # IMEX Euler is stable at any step here (Stokes, or no flow predicted): nothing bounds it
        else:
            largest = max(bounded.values()) if bounded else None

        def problems(dt: float) -> list[Problem]:
            name = chosen_scheme(self.scheme, nu, h, dt)
            found: list[Problem] = []
            limit = scheme_named(name).viscous_limit(nu, h)
            if limit is not None and dt > limit * (1.0 + 1e-12):
                found.append(Problem(
                    "flow.viscous-limit",
                    f"dt = {dt:.4g} s gives the {name} scheme's explicit viscosity nu dt / h^2 = {nu * dt / h ** 2:.3g}, "
                    f"past its viscous number {nu * limit / h ** 2:.3g}: it would not be stable",
                    f"use dt <= {limit:.4g} s, or dt='auto', or Scheme.IMEXEuler() (implicit viscosity)"))
            line = line_of(name)
            if line is not None and peak * dt / h > line * (1.0 + 1e-9):
                found.append(Problem(
                    "flow.cfl-predicted",
                    f"the drive gives the empty tube a peak speed of {peak:.4g} m/s from rest (Poiseuille's centreline "
                    f"and twice the harmonic's Womersley peak), so max|u| dt / h = {peak * dt / h:.3g} at dt = {dt:.4g} s, past the "
                    f"{name} scheme's CFL line of {line:.3g} along the vessel's axis: its convection would not be stable",
                    f"use dt <= {line * h / peak:.4g} s, or dt='auto'"))
            return found

        def report(dt: float) -> tuple[str, ...]:
            name = chosen_scheme(self.scheme, nu, h, dt)
            line = line_of(name)
            return report_lines(name, nu * dt / h ** 2, RK3.viscous_number if name == RK3.name else None,
                                peak * dt / h if line is not None else None, line, drive.period, dt, bounded)

        return StepRule(largest, problems, report, period=drive.period)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        region = ctx.region("medium")
        assert region is not None
        axis = _unit(region.parameters["axis"])
        lattice = lattice_of(axis)
        assert lattice is not None
        radius = float(region.parameters["radius"])  # type: ignore[arg-type]
        spacing = ctx.scalar("spacing") if self.spacing is not None else None
        layout = periodic_tube_layout(radius, ctx.scalar("domain.length"), cells_across=self.cells_across,
                                      spacing=spacing, margin_cells=self.domain.margin, lattice=lattice)
        viscosity, density = ctx.scalar_value("viscosity"), ctx.scalar_value("density")
        scheme = chosen_scheme(self.scheme, viscosity / density, layout.spacing, ctx.timestep)
        band, interior = DEFAULT_BAND, _INTERIOR_NAMES[self.wall.delta_in]
        images = "own" if lattice == (0, 1) else "transverse"
        if scheme_named(scheme).stage_kind is StageKind.EXPLICIT_PROJECTED:
            b.add_node(ExplicitFlowNode(
                ctx.node_name(), ctx.timestep, cells_x=layout.cells[0], cells_y=layout.cells[1],
                cells_z=layout.cells[2], spacing=layout.spacing, viscosity=viscosity, density=density,
                tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius, scheme=scheme,
                convection=not self.stokes, band=band, interior=interior, periodic_images=images))
            return
        b.add_node(ImplicitFlowNode(
            ctx.node_name(), ctx.timestep, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
            spacing=layout.spacing, viscosity=viscosity, density=density,
            tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius, scheme=scheme,
            convection=not self.stokes, tolerance=self.tolerance, preconditioner=self.preconditioner, band=band,
            interior=interior, periodic_images=images, report_rows=self.report_rows))


def _unit(vector: object) -> tuple[float, float, float]:
    held = [float(c) for c in vector]  # type: ignore[attr-defined]
    length = (held[0] ** 2 + held[1] ** 2 + held[2] ** 2) ** 0.5
    return (held[0] / length, held[1] / length, held[2] / length)


__all__ = ["FVM", "Band", "IMEXEulerScheme", "RK3Scheme", "Scheme", "SteadyScheme", "chosen_scheme", "lattice_of"]
