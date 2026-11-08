"""The finite-volume flow field of a medium (F0's Part B): ``Medium("water", fills=tube.lumen, material=...,
fields={"flow": FVM(domain=PeriodicBox(length=mm(1)))})``.

A field kind's value is a frozen settings object naming its kind and version, its settings grouped by the owner each
belongs to: the grid kind's ``spacing`` (or ``cells_across``, the lumen's diameter in cells), the time scheme's
``scheme``, the solve's ``tolerance`` and ``preconditioner``, the wall's lift onto the grid (``band``, ``interior``),
and the domain it lives on. ``stokes`` is physical: the Stokes approximation, convection off. The fluid's properties
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
from mime.nodes.flow.fvm import BANDS, DEFAULT_BAND, DEFAULT_CELLS_ACROSS, DEFAULT_INTERIOR, INTERIORS, \
    ImplicitFlowNode, periodic_tube_layout
from mime.nodes.flow.fvm._schemes import IMEXEuler, Steady
from mime.nodes.flow.fvm._solve import DEFAULT_PRECONDITIONER, PRECONDITIONERS
from mime.registry import NUMERICAL, PHYSICAL, BuildContext, Dynamics, Port, Problem, Region, SettingGroup, \
    convention, register_dynamics, register_setting_group
from mime.units import Magnitude
from mime.scene import PeriodicBox
from mime.things import ThingKind
from mime.units import Length, Quantity
from mime.vocabulary import FLOW_BODY_FORCE

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


class Scheme:
    """The time schemes a script names: ``FVM(scheme=Scheme.IMEXEuler())`` (the default), ``FVM(stokes=True,
    scheme=Scheme.Steady())``. The one place their public names are written."""

    @staticmethod
    def IMEXEuler() -> IMEXEulerScheme:  # noqa: N802 - a kind's name, as the script writes it
        return IMEXEulerScheme()

    @staticmethod
    def Steady() -> SteadyScheme:  # noqa: N802 - a kind's name, as the script writes it
        return SteadyScheme()


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
    the solid outside it a Brinkman penalty, stepped by an implicit scheme on the projected monolithic CG."""

    name = "fvm"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ("HarlowWelch1965", "Angot1999", "Womersley1955")
    builds = {"": (ImplicitFlowNode,)}
    applies_to = frozenset({ThingKind.MEDIUM})
    accepts = {FLOW_BODY_FORCE.name: Port("body_force")}
    values = {"viscosity": "viscosity", "density": "density"}
    unmetered = ("its step identity is written and gated on its node (every term by formula, the solve's residual "
                 "included), but the build does not yet name its hosted laws (the viscosity, the wall's penalty, the "
                 "drive) for the ledger, so it leaves the flow unmetered")
    setting_roles = {
        "domain": PHYSICAL,
        "spacing": convention(NUMERICAL, "none given: the lumen's diameter over cells_across, the band's sweet-spot "
                                         "resolution on straight tubes"),
        "cells_across": convention(NUMERICAL, "the band's calibrated resolution across the narrowest lumen"),
        "scheme": convention(NUMERICAL, "IMEX Euler, the one scheme of the low-Reynolds regimes"),
        "stokes": convention(PHYSICAL, "convection is on unless the Stokes approximation is declared"),
        "tolerance": convention(NUMERICAL, "none given: each dtype's own default, at its reachable floor for the true "
                                           "residual"),
        "preconditioner": convention(NUMERICAL, "Jacobi, the one preconditioner measured in three dimensions"),
        "band": convention(NUMERICAL, "the one registered smooth band"),
        "interior": convention(NUMERICAL, "the default stiff interior beyond the band"),
        "report_rows": convention(NUMERICAL, "the derivative solves' reports summed over the run"),
    }

    domain: PeriodicBox
    spacing: Annotated[Quantity | None, Length] = None
    cells_across: int = DEFAULT_CELLS_ACROSS
    scheme: IMEXEulerScheme | SteadyScheme = IMEXEulerScheme()
    stokes: bool = False
    tolerance: float | None = None
    preconditioner: str = DEFAULT_PRECONDITIONER
    band: str = DEFAULT_BAND
    interior: str = DEFAULT_INTERIOR
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
        if self.band not in BANDS:
            found.append(Problem("flow.band", f"no band {self.band!r} is registered: only smooth bands are offered",
                                 f"write FVM(band={DEFAULT_BAND!r})"))
        if self.interior not in INTERIORS:
            found.append(Problem("flow.interior", f"the stiff interior is one of {sorted(INTERIORS)}, not "
                                                  f"{self.interior!r}", "write FVM(interior='0.1h') or '0.01h'"))
        if self.preconditioner not in PRECONDITIONERS:
            found.append(Problem("flow.preconditioner", f"no preconditioner {self.preconditioner!r} is registered",
                                 f"write FVM(preconditioner={DEFAULT_PRECONDITIONER!r})"))
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
        scheme = SteadyScheme.group if isinstance(self.scheme, SteadyScheme) else IMEXEulerScheme.group
        b.add_node(ImplicitFlowNode(
            ctx.node_name(), ctx.timestep, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
            spacing=layout.spacing, viscosity=ctx.scalar_value("viscosity"), density=ctx.scalar_value("density"),
            tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius, scheme=scheme,
            convection=not self.stokes, tolerance=self.tolerance, preconditioner=self.preconditioner, band=self.band,
            interior=self.interior, periodic_images="own" if lattice == (0, 1) else "transverse",
            report_rows=self.report_rows))


def _unit(vector: object) -> tuple[float, float, float]:
    held = [float(c) for c in vector]  # type: ignore[attr-defined]
    length = (held[0] ** 2 + held[1] ** 2 + held[2] ** 2) ** 0.5
    return (held[0] / length, held[1] / length, held[2] / length)


__all__ = ["FVM", "IMEXEulerScheme", "Scheme", "SteadyScheme", "lattice_of"]
