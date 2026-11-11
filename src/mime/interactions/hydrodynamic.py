"""The hydrodynamic family: the resistance of a medium to a body's motion, and the load its flow puts on a body held
still. Here now: ``Drag`` with its surrogate rung, the learned resistance of a de Jongh screw body in a straight tube,
with the near-wall lubrication correction as a setting of that rung; and its MRS rung, the resistance of
any rigid body whose hull kind gives surface samples, in a medium whose region a wall kind matches (``mime.walls``):
an unbounded one, a straight tube's lumen or the half-space a plane bounds."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, cast, override

import numpy as np

from mime.core import run_dtype, shape_named, switch
from mime.data import Asset, asset_problem
from mime.gradients import GradientStatus
from mime.nodes.hydro import RFTDragNode, SurrogateDragNode
from mime.nodes.hydro._point_drag import AVERAGES, COUPLE, FAXEN_FORCES, FLOOR, FORCES, HISTORY_LINE, LAPLACIANS, \
    REYNOLDS_LINE, RINGING_STEPS, ROTATIONS, ROUTES, WALL_LINE, PointDragNode, free_space_margins
from mime.nodes.hydro._rft_drag import (
    COEFFICIENT_SETS,
    RFT_POINTS_PER_RADIUS,
    coefficient_problem,
    default_coefficients,
)
from mime.nodes.hydro.mrs import SHAPE, WALL, MRSDragNode, wall_correction_named
from mime.nodes.hydro.mrs._handover import HANDOVER_SPACINGS, WINDOW, least_radius, window_text
from mime.nodes.hydro.mrs._node import CONFINED_GAP, INTEGRATED_GAP
from mime.nodes.hydro.mrs._quadrature import empty_cells
from mime.nodes.hydro.near_field import DERIVATIVE_MEASURED, MEMORY_HEADROOM, MEMORY_MARGIN, MEMORY_MEASURED, ROUTES, \
    JointDragNode, member_prefix
from mime.scene import KEPT, quadrature, quadrature_counts, quadrature_owners, sampled_points
from mime.scene._sampling import FINEST, quadrature_arrays
from mime.scene._grading import (
    CONTACT_SPACINGS,
    Contact,
    GradedDensity,
    chosen_at,
    figures,
    finest_share,
    graded_density,
    graded_patch,
)
from mime.registry import (
    DESIGN,
    Centreline,
    NUMERICAL,
    PHYSICAL,
    BuildContext,
    EntryKind,
    Delivery,
    Flag,
    Geometry,
    GradientRecord,
    Hull,
    Interaction,
    ChoiceRule,
    Plan,
    PlanContext,
    Problem,
    Read,
    Region,
    Registration,
    ReynoldsRange,
    RoleKind,
    RungGradient,
    Sampling,
    SettingGroup,
    SupportRow,
    Value,
    WallKind,
    choose_over,
    convention,
    determined,
    implements,
    lookup,
    numerical_default,
    register_interaction,
    register_setting_group,
    registered,
    requires,
    centreline_counts,
    versions_of,
    wall_kinds_matching,
)
from mime.surrogates import SURROGATES, ScrewSurrogate
from mime.units import Length, Magnitude, Quantity, m
from mime.vocabulary import BACKGROUND_LOAD, EXCHANGE_PARTNER, FLOW_EXCHANGE_X, FLOW_EXCHANGE_Y, FLOW_EXCHANGE_Z, \
    FLOW_FACES, FLOW_PARTNERS, POSE, RESISTANCE, TWIST, Capability, Definiteness, EnergyClass
from mime.walls import RESERVED

if TYPE_CHECKING:
    from mime.build import InteractionBuilder
    from mime.fields._solved import ExchangeFrame


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Lubrication(SettingGroup):
    """The near-wall lubrication correction, blended into a drag rung's resistance as one model: the leading terms of
    the sphere-wall asymptotes for the body's nearest point (Cox & Brenner 1967; Goldman, Cox & Brenner 1967, part
    I), without the terms after them or the translation-rotation coupling (``SurrogateDragNode`` names them),
    weighted by ``exp(-gap / epsilon)`` so that they count only near the wall (the 0.2 line's blend: a modelling
    choice, with ``epsilon`` its setting, not a value a reference fixes). A drag that blends it in includes near-wall
    lubrication and declares so (``Interaction.includes_lubrication``): a contact never adds it, and the plan refuses a
    second law on the body that includes it (``lubrication.twice``)."""

    group = "lubrication"
    setting_roles = {"epsilon": NUMERICAL}
    epsilon: Annotated[Quantity, Length]


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class ForceSampling(SettingGroup):
    """How the mrs rung samples a body's force points where it grades them (drag 0.5.0, opt-in):
    ``grading='features'`` towards the hull kind's declared features (its edges and crests), and ``contact``, a
    direction in the body's frame, towards the side of the body held facing the wall, graded for ``contact_gap``. Left
    out, the force points are quasi-uniform, as every released program's. What grading buys is measured, never bounded;
    a claim of it stands against a uniform sampling of the graded count, the count-matched uniform arm, as well as
    against the uniform sampling itself, and one without that arm (the cylinder's rim's) says the gain may be the
    count's."""

    group = "force_sampling"
    setting_roles = {
        "grading": convention(
            NUMERICAL, "none: the force points quasi-uniform, as every released program's; written 'features', graded "
                       "towards the hull kind's declared features (its edges and crests), each force point at its "
                       "cell's area centre"),
        "contact": convention(
            NUMERICAL, "none: no contact graded; written, a direction in the body's frame, the side of the body held "
                       "facing the wall, whose patch is graded for contact_gap (opt-in, for a body whose orientation to "
                       "the wall is held: the sampling is fixed on the body, and a step whose nearest point leaves the "
                       "patch near the wall raises off_patch)"),
        "contact_gap": determined(
            NUMERICAL, "with contact written, the experiment's closest approach to the wall (the margin a contact "
                       "holds the body off it by), where one is declared",
            by="default_contact_gap"),
    }
    grading: str | None = None
    contact: tuple[float, float, float] | None = None
    contact_gap: Annotated[Quantity | None, Length] = None


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class NearField(SettingGroup):
    """How the mrs rung solves a medium's near field, the joint system of its bodies (drag 0.5.0): ``solve``, the route,
    ``"dense"`` (one dense LU) or ``"gmres"`` (GMRES, each body's held free part's inverse its preconditioner, each solve
    to ``tolerance`` on its true residual); and ``share``, the most of a source's quadrature points the cross-kernel
    integrates. The dense route is taken at every size unless GMRES is declared; on either route the build refuses a
    cluster whose run would not fit in the memory a run can have (``drag.near-field-memory``), the only bound on its
    size."""

    group = "near_field"
    setting_roles = {
        "solve": convention(
            NUMERICAL, "'dense', one dense LU at every size, exact to rounding, the memory refusal its only bound: "
                       "through the node GMRES holds more memory than the dense LU, not less (1.11 "
                       "to 1.35 times its peak at 3,042 to 21,168 unknowns, either dtype), and neither route's time "
                       "places a size line (a radius apart GMRES's step was the shorter at every size measured, near "
                       "contact the dense route's mostly was, on a shared laptop); 'gmres' a declared choice, its "
                       "answer the dense route's to its tolerance, a solve that stops short flagged and warned of"),
        "tolerance": convention(
            NUMERICAL, "with solve='gmres', GMRES's relative tolerance on each solve's true residual, "
                       "DEFAULT_TOLERANCE by the run's dtype: float64's holds the grand resistance to the dense "
                       "route's to within its gates' lines near contact; float32's sits above the floor its own "
                       "arithmetic leaves a pair's residual at down to the near-contact line, by a small margin there "
                       "(DEFAULT_TOLERANCE's measurement), so a denser cluster near contact may stop short, flagged "
                       "and warned of"),
        "share": convention(
            NUMERICAL, "every quadrature point of a source, the whole share: the share that holds each registered "
                       "hull kind's conformance case within the cross-kernel's reach (measured: at a quarter of a "
                       "source's points, five kinds of six at their cases' spacings fell past the case's line near "
                       "contact, the box furthest, while MIME's sphere at its default count was unmoved to within a few "
                       "hundredths of a per cent)"),
    }
    solve: str = "dense"
    tolerance: float | None = None
    share: float | None = None


@dataclass(frozen=True)
class UniformTube:
    """A region uniform along its axis, as the surrogate reads it: a point on the axis, the axis, the radius."""

    centre: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float


def _triple(value: Any) -> tuple[float, float, float]:
    held = [float(v) for v in np.asarray(value, dtype=np.float64).ravel()]
    return (held[0], held[1], held[2])


#: The parameters of the shape the drag's node evaluates the tube with, each its node's ``vessel_<name>``.
VESSEL_PARAMETERS = ("centre", "axis", "radius")


def at_rest_problems(ctx: PlanContext, rung: str | None) -> list[Problem]:
    """Every drag rung here assumes its medium at rest (its background load is zero), so it is refused where the region
    its medium fills carries a field, the medium's own or another's filling the same region. No quantity a field kind
    provides shows its medium at rest, so a field counts whatever it declares it provides: the refusal does not rest
    on a kind declaring ``field.flow.velocity``. A rung that reads a provided flow is a later release's."""
    carried = ctx.provided.get("medium", {})
    if not carried:
        return []
    medium = ctx.bindings["medium"]
    fields = ", ".join(f"{name} (providing {', '.join(provides) or 'nothing it declares'})"
                       for name, provides in sorted(carried.items()))
    return [Problem("drag.medium-flows",
                    f"the {rung} rung assumes medium {medium!r} at rest (it delivers no load from a flow), and the "
                    f"region it fills carries {fields}: a drag that reads a medium's flow is a later release's",
                    "leave the flow out of the region the drag's medium fills (a medium with no fields=), or leave "
                    "the drag out")]


def uniform_tube(region: Region | None, medium: str) -> UniformTube | Problem:
    """The tube the medium fills, through the shape protocol: its axis, a point on it, and its radius (the shape's
    signed distance on its axis, negated); or why there is none."""
    if region is None:
        return Problem("drag.no-region", f"medium {medium!r} fills no region with a shape, so it has no axis and no "
                                         f"radius: the surrogate reads both from the tube the medium fills",
                       f"write Medium({medium!r}, fills=tube.lumen) for a structure with a Tube geometry")
    shape = shape_named(region.shape)
    axis, point = shape.axis(region.parameters), shape.axis_point(region.parameters)
    if axis is None or point is None:
        return Problem("drag.no-axis", f"medium {medium!r} fills {region.path}, a {region.shape} region with no axis: "
                                       f"the surrogate needs a tube's axis and radius",
                       "fill a tube's lumen, Medium(..., fills=tube.lumen), or choose a rung for this geometry when "
                       "one is built")
    if shape.parameter_names != VESSEL_PARAMETERS:
        return Problem("drag.no-axis", f"medium {medium!r} fills {region.path}, a {region.shape} region, whose "
                                       f"parameters are not a circular tube's {VESSEL_PARAMETERS}: the surrogate "
                                       "needs a tube's axis and "
                                       "radius", "fill a tube's lumen: Medium(..., fills=tube.lumen)")
    radius = -float(shape.signed_distance(region.parameters, point))
    if not (math.isfinite(radius) and radius > 0.0):
        return Problem("drag.no-axis", f"medium {medium!r} fills {region.path}, whose axis lies outside it (signed "
                                       f"distance {-radius!r} m on the "
                                       "axis)", "fill a tube's lumen: Medium(..., fills=tube.lumen)")
    return UniformTube(_triple(point), _triple(axis), radius)


#: The surrogate rung's switches: those its node evaluates, from the tube's axis to the lubrication's floors. The mrs
#: rung evaluates none.
SURROGATE_SWITCHES = (
    switch.Declaration(
        owner="drag.axis",
        indicator="the body's distance from the tube's axis, m; on the axis within AXIS_ULPS units in the last "
                  "place of the tube's radius (a band), where the frame takes the body's own axes",
        orders={"resistance": -1},
        route="a surrogate whose canonical resistance at zero offset is invariant under rotation about the tube's "
              "axis, which makes the resistance continuous across the axis, and which changes smoothly in the "
              "offset across it, for example through the offset's square, which makes it differentiable there "
              "(invariance alone leaves a cone, in the offset's length); the trained surrogate has neither, and no "
              "frame taken on the axis can give it them"),
    switch.Declaration(
        owner="drag.across",
        indicator="the squared length of the body's x axis across the tube, less a quarter (on the axis, it "
                  "picks the body's x or y axis for the frame)",
        orders={"resistance": -1},
        route="as drag.axis's: a surrogate symmetric about the tube's axis needs no frame on it"),
    switch.Declaration(
        owner="drag.end",
        indicator="the cosine between the body's x axis and the tube's, plus END_TIE: the side the resistance is "
                  "moved from",
        orders={"resistance": -1},
        route="none: the switch lies where the body's axis lies across the tube, outside the training data, "
              "where a body would be refused"),
    switch.Declaration(
        owner="drag.room",
        indicator="0.95 of the room between the body's outer radius and the wall, in the body's radius",
        orders={"resistance": 0},
        route="none needed: a body that does not fit its tube is refused when planned, so it is never live"),
    switch.Declaration(
        owner="drag.clamp",
        indicator="the offset less the largest the surrogate is read at, in the body's radius",
        orders={"resistance": 0},
        route="a smooth saturation of the offset, or refusing a pose beyond the clamp"),
    switch.Declaration(
        owner="drag.clamp_divisor",
        indicator="the offset the clamp divides by, less its floor of 1e-10, in the body's radius",
        orders={"resistance": 0},
        route="none needed: a clamped body is never that close to the axis, so it is never live"),
    switch.Declaration(
        owner="drag.feature_gap_floor",
        indicator="the gap from the body's outer radius to the wall, less the surrogate's floor, in the body's "
                  "radius",
        orders={"resistance": 0},
        route="a smooth floor, a softplus"),
    switch.Declaration(
        owner="drag.gap_floor",
        indicator="the lubrication's gap less its floor of 1 nm, m",
        orders={"resistance": 0},
        route="a smooth floor, a softplus"),
    switch.Declaration(
        owner="drag.log_floor",
        indicator="the lubrication's logarithm of the outer radius over the gap",
        orders={"resistance": 0},
        route="a smooth floor, a softplus"),
)


#: The mrs rung's default count of samples on the hull kind's grid (the sum over its patches of the products of
#: ``sample_counts``), by the kind (a numerical default: a first look, not a converged one), chosen by measurement
#: (``scripts/default_spacing_table.py``): for a compact body the most whose run of a 100 um steel sphere settling for
#: 1000 steps stays within about 30 s on a laptop's CPU (a sphere's 338 samples: 512 take 50 s); for a screw, where no
#: count does (6000 steps in a tube take minutes at the coarsest), the coarsest that keeps its shape (FL-9's 222
#: samples, its two caps sampled alike, at 0.829 mm). A kind not measured takes the compact bodies' count.
DEFAULT_SAMPLES: Mapping[str, int] = MappingProxyType({"sphere": 338, "screw": 191})
COMPACT_SAMPLES = 338
#: The Reynolds number, by the hull's diameter and the body's speed, up to which both rungs are declared, and why. A
#: convention, not a fit: the number is taken from a sphere sedimenting in a still fluid, whose drag has a known first
#: correction in the Reynolds number, and declared for every body the rung serves although a force-free swimmer has no
#: such correction (the post-run watch states what it implies for a sphere, and says it does not for another body).
STOKES_REYNOLDS = 0.1
STOKES_BASIS = ("Stokes flow is the limit of vanishing Reynolds number, and its error grows with it. The bound is a "
                "convention taken from a sphere sedimenting in a still fluid, whose drag the first correction of "
                "Oseen's expansion raises by three sixteenths of the Reynolds number, about a fiftieth at the bound. "
                "That correction is a sedimenting sphere's, and a force-free swimmer, which has no net force, does not "
                "have it; no correction of the neglected inertia has been measured for any body of MIME's. The bound "
                "is declared for every body the rung serves from the sphere alone, and says how far a run is from "
                "Stokes flow, not how much its answer is off")
#: The default blob, as a fraction of the samples' mean spacing.
DEFAULT_BLOB = 0.5
#: What was measured at the default spacing, in numbers and on what (``scripts/default_spacing_table.py``, float64 for
#: the misses, the default float32 for the run times, on a 24-thread laptop CPU). The misses are the quasi-uniform
#: sampling's (phase 7m-a: the sphere's drag 1.3% and 5.2% above on its earlier grid, the screw's speed 63% and 61%
#: below); the run times were taken on the earlier grid, at the same counts, and the sampling adds well under a second
#: to a build at them.
SPACING_MEASURED = ("on a 100 um steel sphere settling in still water (338 samples), its drag 1.8% and its rotational "
                    "drag 5.9% above Stokes's law, 1000 steps of it in 22 s; on FL-9 (222 samples), its speed per turn "
                    "23% below the finest spacing measured on the axis of a 1/4 in tube (5984 samples) and 16% below "
                    "it in free space (6784), 6000 steps in the tube 11 minutes: write spacing= for a number to use")


#: The mrs rung's choices over a declared range of its hull's numbers (``mime.registry.choose_over``). The force points
#: at one spacing never fall as these numbers grow, measured on dense scans in CI over a window of each kind's own
#: (``tests/ranges/test_mrs_monotone.py``): there the range's corners hold the most, and each is judged at them. Any
#: other number of the kind, or a kind not measured, is judged at :data:`SAMPLED_COUNT` points along it as well: the
#: screw's radius is not monotone (106 force points at 2.153 mm, 91 at 2.184 mm, at 1.5 mm).
MONOTONE_NUMBERS: Mapping[str, frozenset[str]] = MappingProxyType({
    "sphere": frozenset({"radius"}), "capsule": frozenset({"radius", "height"}),
    "screw": frozenset({"length", "wavenumber", "modulation"})})
SAMPLED_COUNT = 9
FORCE_POINTS_MONOTONE = ChoiceRule.Monotone(
    "at one spacing a larger body has at least as many force points, measured in each of the kind's ranged numbers "
    "over a window of them (tests/ranges/test_mrs_monotone.py)")
FORCE_POINTS_SAMPLED = ChoiceRule.Sampled(
    SAMPLED_COUNT, "the force points are not measured monotone in this number of the kind (the screw's radius is "
                   "not: tests/ranges/test_mrs_monotone.py)")
#: The default spacing over a range is sampled in every number: it is the largest spacing to three figures at which
#: the kind gives the family's sample count, and it falls as the body grows in two ways, measured. Where the bisection's
#: boundary lies in the last half unit under a power of ten, the rounded spacing (1.00 mm) gives too few samples and
#: the rule steps a unit of its own third figure down, ten times the unit below it (0.99 mm, not 0.999: the ball at a
#: radius of 3.9785 mm, against 0.999 mm at 3.976); and over FL-9's modulation the boundary itself falls (0.817 to
#: 0.809 mm as the modulation grows from 0.231 to 0.238; inside [0.231, 0.2504] 0.98 % finer than at its ends). A
#: body inside a range whose own spacing lies in a dip the samples miss is refused as the rung's defect (its choice
#: falls short), never run coarser.
SPACING_SAMPLED = ChoiceRule.Sampled(
    SAMPLED_COUNT, "the family's default spacing falls as the body grows where its rounding steps under a power of ten "
                   "and over the screw's modulation (tests/ranges/test_mrs_monotone.py)")


def _owner(entry: Interaction, bindings: Mapping[str, str]) -> str:
    """The id a choice over a range names its entry by: the build's (``mime.build.entry_id``)."""
    from mime.build import entry_id

    return entry_id(type(entry), bindings)


def force_points_rule(hull: Hull) -> ChoiceRule.Monotone | ChoiceRule.Sampled:
    """The rule the hull's force points are chosen over its ranged numbers by: monotone where each ranged number is one
    measured monotone for its kind (:data:`MONOTONE_NUMBERS`), sampled otherwise."""
    numbers = MONOTONE_NUMBERS.get(hull.kind.name, frozenset())
    return FORCE_POINTS_MONOTONE if set(hull.ranges) <= numbers else FORCE_POINTS_SAMPLED


def spacing_rule(hull: Hull) -> ChoiceRule.Monotone | ChoiceRule.Sampled:
    """The rule the family's default spacing is chosen over the hull's ranged numbers by: sampled, in every number
    (:data:`SPACING_SAMPLED`)."""
    return SPACING_SAMPLED


def _step_down(spacing: float, boundary: float, from_boundary: bool) -> float:
    """``spacing``, rounded to three figures past ``boundary``, one unit of the third figure lower. The unit is the
    boundary's (``from_boundary``: exact rounding-down) or the rounded value's (the released step, which within half a
    unit below a power of ten takes the power's, ten times the true next value down)."""
    unit = 10.0 ** (math.floor(math.log10(boundary if from_boundary else spacing)) - 2)
    return float(f"{spacing - unit:.3g}")


@functools.lru_cache(maxsize=256)
def _force_points(kind: Any, settings: tuple[tuple[str, Any], ...], spacing: float, patches: tuple[str, ...]) -> float:
    sampling = kind.sampling(dict(settings), spacing, patches)
    if sampling.chosen is not None:
        return float(len(sampling.chosen))
    return float(sum(math.prod(count) for count in sampling.counts))


def force_points(hull: Hull, spacing: float, patches: tuple[str, ...]) -> float:
    """How many force points the hull's kind gives at ``spacing`` on ``patches``: its sampling's choice, or its
    quadrature where the kind's sampling is the quadrature itself."""
    return _force_points(hull.kind, tuple(sorted(hull.settings.items())), spacing, patches)


#: A graded sampling's choice over a declared range: sampled in every number. Its count is the uniform one's times the
#: density's mean over the surface, and between the uniform count's steps a larger body's graded patch is a smaller
#: share of it, so its count falls as the body grows: the sphere's contact graded for half of a 0.3 mm spacing, 106
#: force points at a radius of 0.5 mm and 104 at 0.525 mm (measured: a scan of 61 radii from 0.5 to 2 mm fell 29 times at
#: 0.3 mm and 40 at 0.6 mm).
GRADED_FORCE_POINTS = ChoiceRule.Sampled(
    SAMPLED_COUNT, "a graded sampling's force points fall as the body grows between the uniform count's steps, its "
                   "patch a smaller share of the surface (the sphere's contact graded for half of 0.3 mm: 106 at a "
                   "radius of 0.5 mm, 104 at 0.525 mm; tests/ranges/test_mrs_graded_range.py)")


def graded_force_points(hull: Hull, spacing: float, patches: tuple[str, ...], density: GradedDensity | None) -> float:
    """How many force points the hull's kind gives at ``spacing`` on ``patches`` at the grading's ``density``: its
    graded sampling's choice (the uniform one's, :func:`force_points`, where ``density`` is ``None``)."""
    if density is None:
        return force_points(hull, spacing, patches)
    chosen = hull.kind.sampling(hull.settings, spacing, patches, density).chosen
    return float(len(chosen)) if chosen is not None else force_points(hull, spacing, patches)


def _solve_cost(chosen: float, own: float) -> str:
    return (f"its dense solve of {3 * chosen:g} unknowns against {3 * own:g}, about {(chosen / own) ** 3:.2g} times "
            f"the factorisation") if own > 0 else ""


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PointForceDrag(Interaction):
    """The drag 0.3.0, kept registered so that a folder saved with it loads and reproduces (the frozen
    register's experiments name it): its ``mrs`` rung puts a point force at each sample, the samples its own quadrature, with the
    blob half their mean spacing and its count of samples by family (``DEFAULT_SAMPLES``), as before the separated
    quadrature of 0.4.0 (``Drag``); its ``surrogate`` rung is 0.4.0's. A script that writes ``Drag.MRS(...)`` gets
    0.4.0; a document naming 0.3.0 gets this. Version 0.3.0 read the body's shape from its hull's kind, not from
    declared values: its values lost ``helix_wavenumber``, ``length`` and ``radius``, its node gained the body's outer
    radius and the tube's shape, and a body with no helix for its hull was refused; version 0.2.0 moved the resistance
    to the centre; 0.1.0 delivered it about the end unshifted (MIME-ANO-112). A document naming either is refused at
    load (``load.version``). Its default spacing, within half a unit below a power of ten, steps to 0.99, not 0.999,
    kept so released runs reproduce (``step_from_boundary``; 0.5.0's does not)."""

    name = "drag"
    version = "0.3.0"
    setting_roles = {
        "rung": DESIGN, "weights": DESIGN,
        "lubrication": convention(DESIGN, "no lubrication correction is blended in unless one is declared"),
        "spacing": numerical_default(
            "on the mrs rung, the largest spacing, to three significant figures, at which the hull's kind gives its "
            "family's count of samples on its grid (DEFAULT_SAMPLES: a compact body's, a screw's)",
            "a first look, chosen by measurement for a first run's cost, never a converged value: the most samples "
            "whose run of a first experiment's size stays within a first run's budget on a laptop's CPU, and for a "
            "screw, where none does, the coarsest that keeps its shape (scripts/default_spacing_table.py measures the "
            "misses and the run times on a ladder of spacings, and the stamp states them)", by="default_spacing",
            measured=SPACING_MEASURED),
        "blob": numerical_default(
            "on the mrs rung, half the samples' mean spacing",
            "the blob the earlier line's MRS drag ran with; it bounds nothing: the spacing's measured "
            "misses are at this blob, and they fall as the spacing falls (scripts/default_spacing_table.py)",
            by="default_blob", measured="measured with the spacing's default, at this blob: see the spacing's"),
        "patches": convention(NUMERICAL, "the hull's closed surface: every patch its kind has"),
        "wall_kind": determined(PHYSICAL, "on the mrs rung, the one wall kind that matches the region the medium "
                                          "fills (free_space where it fills none)", by="default_wall_kind"),
    }
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=SURROGATE_SWITCHES)
    rung_gradients = {
        "surrogate": RungGradient(differentiability=GradientStatus.PIECEWISE_JUMP,
                                  gradient=GradientRecord(switches=SURROGATE_SWITCHES)),
        "mrs": RungGradient(differentiability=GradientStatus.EXACT),
    }
    references = ("deJongh2025NonlinDyn", "CoxBrenner1967", "GoldmanCoxBrenner1967", "BradyBossis1988", "Cortez2005")
    family = "hydrodynamic"
    builds = {"surrogate": (SurrogateDragNode,), "mrs": (MRSDragNode,)}
    raises = (Flag("clamp_fired", "the body's offset from the tube's axis lay beyond the clamp near the wall, so the "
                                  "surrogate read the resistance at the clamp, not at the offset", rung="surrogate"),
              Flag("tilt_fired", "the body's axis lay further from the tube's than the tilt the surrogate is declared "
                                 "valid to (see the rung's validity line): it was trained for a helix along the "
                                 "tube's axis and cannot see a tilt, so its resistance is that of the body along "
                                 "the axis", rung="surrogate"),
              Flag("bore_fired", "the body's axis lay so far from the tube's that the body, as the cylinder of its "
                                 "length and outer radius, no longer fits the tube's bore: the geometric limit, "
                                 "beyond the declared tilt's, past which a body longer than the bore has gone "
                                 "through the wall", rung="surrogate"))
    reynolds = (
        ReynoldsRange(STOKES_REYNOLDS, STOKES_BASIS, rung="surrogate"),
        ReynoldsRange(STOKES_REYNOLDS, STOKES_BASIS, rung="mrs"),
    )
    validity = (
        "each body's drag is computed alone, from its own pose and the medium's wall: a second body in the same "
        "medium does not change it, on either rung (no hydrodynamic interaction between bodies in MIME 0.3.0; the "
        "build reports the bodies that share a medium)",
        "surrogate: quasi-steady Stokes flow (the limit of vanishing Reynolds number), as its training resistances were "
        "computed; the Reynolds number, by the body's translation and by its turning, is computed after each run "
        "(the post-run watch) and flagged past the bound the entry declares",
        "surrogate: de Jongh screw bodies (cross-section modulation 0.33, two starts) in a straight tube; nu, kappa "
        "and L/R checked against the training data at plan time; the offset declared up to 0.3 of the tube's radius "
        "and clamped beyond 0.95 of the room to the wall; its accuracy against the MRS it was checked against is "
        "MIME-ANO-114's, up to 16% in swim speed between 0.2 and 0.3 of the radius",
        "surrogate: the resistance is moved from the end of the body's axis, about which it was trained, to the "
        "centre of mass, half the body's length along its x axis taken as the helix's, which the surrogate takes to lie along the "
        "tube's; it cannot see a tilt (away from the axis its resistance is a function of the body's centre and "
        "of which way the body's axis points along the tube, and of nothing else about the orientation)",
        "surrogate: valid for a body whose axis lies within 5 degrees of the tube's, a limit declared "
        "(mime.nodes.hydro.TILT_VALIDITY), not fitted, from one MRS measurement of FL-9 on the axis "
        "(docs/data/tilt_validity.json, scripts/tilt_validity.py): at that tilt the force-free swim speed differs "
        "from the aligned body's by less than the 16% this entry states for the surrogate against the boundary "
        "element, and at the next tilt measured, in one of two directions, it does not; one spacing, one offset, "
        "one body, and the MRS itself not converged at that spacing; tilt_fired is raised at the step "
        "the tilt passes the limit, and bore_fired at the step the body, as the cylinder of its length and outer "
        "radius with its centre on the axis, no longer fits the bore",
        "mrs: quasi-steady Stokes flow, the limit of vanishing Reynolds number; the Reynolds number, by the body's "
        "translation and by its turning, is computed after each run (the post-run watch) and flagged past the bound "
        "the entry declares",
        "mrs: a medium at rest that its wall kind matches: unbounded (free_space), a straight tube's lumen (tube) or "
        "the half-space a plane bounds (plane); "
        "its error is the regularised method's, "
        "of the order of the blob over the body's size, falling as the spacing falls, and is measured against the "
        "sphere's and the prolate spheroid's closed forms by its oracle tests, not bounded here",
        "mrs: the body's hull alone is sampled, on the patches written; a patch left out (the de Jongh helix's caps, "
        "which the line before this one left out) is a model difference, not a rounding",
        "mrs: on the de Jongh helix, the kind's exact-sample quadrature and the end caps change the resistance against "
        "the v0.2 line's MRS on its own mesh (flat triangles, a point at each centroid), and neither is converged at "
        "the surrogate's training count: see MIME-ANO-116",
    )
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    rungs = ("surrogate", "mrs")
    energy_class = EnergyClass.DISSIPATIVE
    support = (SupportRow("surrogate", {
        "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name], publishes=[POSE.name]),
    }, note="a rigid body whose dynamics takes a resistance, in a medium at rest filling a straight tube"),
        SupportRow("mrs", {
            "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name],
                             publishes=[POSE.name]),
        }, note="a rigid body whose dynamics takes a resistance and whose hull kind gives surface samples, in a "
                "medium at rest that a wall kind matches: an unbounded one, a straight tube or a plane"))
    reads = {"pose": Read("body", POSE.name)}
    delivers = {"resistance": Delivery("body", RESISTANCE.name), "background_load": Delivery("body", BACKGROUND_LOAD.name)}
    values = {"viscosity": Value("medium", "viscosity")}
    #: The surrogates the rung can evaluate, by the name of their weights' asset, each with its training domain.
    surrogates: ClassVar[Mapping[str, ScrewSurrogate]] = SURROGATES
    #: The mrs rung's default count of samples by the hull kind's name, and a kind's the table does not name.
    samples_by_family: ClassVar[Mapping[str, int]] = DEFAULT_SAMPLES
    compact_samples: ClassVar[int] = COMPACT_SAMPLES
    #: The settings the mrs rung reads and the surrogate rung refuses.
    mrs_settings: ClassVar[tuple[str, ...]] = ("spacing", "blob", "patches", "wall_kind")
    #: Whether a spacing neither written nor given by its rule is left to the plan, which refuses it with its cause
    #: (no hull that samples, a body held on the wall), rather than refused as a setting with no value.
    unset_spacing_planned: ClassVar[bool] = False
    #: Whether the default spacing's step below a rounding past its bound is sized from the bound (0.5.0: exact
    #: rounding-down to three figures) or from the rounded value (0.3.0 and 0.4.0, kept so released runs reproduce: within
    #: half a unit below a power of ten the rounded value is that power, and the step is ten times the true next value
    #: down, 0.99 and not 0.999).
    step_from_boundary: ClassVar[bool] = False
    rung: str | None = None
    weights: Asset | None = None
    lubrication: Lubrication | None = None
    spacing: Annotated[Quantity | None, Length] = None
    blob: float | None = None
    patches: tuple[str, ...] | None = None
    wall_kind: str | None = None

    @classmethod
    def Surrogate(cls, **settings: Any) -> Self:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="surrogate", **settings)

    def default_spacing(self, ctx: PlanContext) -> Quantity | None:
        """The mrs rung's spacing when none is written: the largest, to three significant figures, at which the hull's
        kind gives at least its family's sample count (``DEFAULT_SAMPLES``) over the patches sampled. Over a declared
        range of the hull's numbers, the finest of those across the range (:func:`spacing_rule`), so that every body
        inside it has at least its family's count."""
        hull = ctx.geometry(of="body")
        if self.rung != "mrs" or hull is None or not implements(hull.kind, "sampling"):
            return None
        chosen = choose_over(_owner(self, ctx.bindings), hull, "default spacing", spacing_rule(hull),
                             self._family_spacing, largest=False, unit="mm")
        return m(self._family_spacing(chosen))

    def _family_spacing(self, hull: Hull) -> float:
        """The largest spacing, to three significant figures, at which ``hull``'s kind gives at least its family's
        sample count over the patches sampled."""
        coarse = float(hull.kind.radial_reach(hull.settings) if implements(hull.kind, "radial_reach") else
                       hull.kind.enclosing_radius(hull.settings))
        target = self.samples_by_family.get(hull.kind.name, self.compact_samples)
        patches = self.patches if self.patches is not None else tuple(hull.kind.patches)

        def count(spacing: float) -> int:
            return sum(math.prod(hull.kind.sample_counts(hull.settings, spacing, patch)) for patch in patches)

        fine = coarse
        while count(fine) < target:
            fine /= 2.0
        for _ in range(60):  # the boundary between the spacings that give the count and those that do not
            middle = math.sqrt(coarse * fine)
            coarse, fine = (coarse, middle) if count(middle) >= target else (middle, fine)
        spacing = float(f"{fine:.3g}")
        if count(spacing) < target:  # rounded past the boundary: one unit of the third figure below it
            spacing = _step_down(spacing, fine, self.step_from_boundary)
        return spacing

    def default_blob(self, ctx: PlanContext) -> float | None:
        """The mrs rung's blob when none is written: half the samples' mean spacing."""
        return DEFAULT_BLOB if self.rung == "mrs" else None

    def default_wall_kind(self, ctx: PlanContext) -> str | None:
        """The mrs rung's wall kind when none is written: the one that matches the region the medium fills, by name
        (and version, where the name has several); none when no wall kind or several match, which the plan refuses with
        those that do."""
        if self.rung != "mrs":
            return None
        region = ctx.region("medium")
        matching = wall_kinds_matching(region.shape if region is not None else None)
        if len(matching) != 1:
            return None
        held = matching[0]
        return held.name if len(versions_of(EntryKind.WALL_KIND, held.name)) == 1 else f"{held.name} {held.version}"

    @classmethod
    def MRS(cls, **settings: Any) -> Self:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="mrs", **settings)

    @override
    def definiteness(self) -> Definiteness | None:
        """The surrogate's resistance is its network's ``L L^T`` carried through congruences, with the lubrication's
        positive semidefinite part added: definite by that construction, and measured so in the family's suite over
        script A's tube (the centre anywhere in its room, rolled, tilted up to 5 degrees, either end first), which is
        the domain the measurement covers. The MRS's is solved at the body's state, and near a wall its definiteness
        depends on it: checked at run time."""
        return Definiteness.CONSTRUCTION if self.rung == "surrogate" else Definiteness.STATE

    @override
    def includes_lubrication(self) -> bool:
        """The surrogate rung with ``lubrication=`` includes near-wall lubrication, the blend of the sphere-wall
        asymptotes; the mrs rung resolves the wall's flow at its spacing and blends in none."""
        return self.rung == "surrogate" and self.lubrication is not None

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        if self.rung == "mrs":
            return self._mrs_setting_problems(settings)
        found = [f"{name}= is a setting of the mrs rung, and the surrogate rung reads none"
                 for name in self.mrs_settings if getattr(self, name) is not None]
        weights = cast(object, self.weights)  # a script is not type-checked
        if weights is None:
            found.append("the surrogate rung reads its weights: write weights=asset('dejongh_mlp_v2'), with asset from "
                         "mime.data")
        elif not isinstance(weights, Asset):
            found.append(f"weights={weights!r} is not a packaged asset: the surrogate's weights are named by "
                         f"mime.data.asset('dejongh_mlp_v2'), not by a file's path")
        elif weights.name not in self.surrogates:
            found.append(f"weights={weights.name!r} is no surrogate's; those declared are {sorted(self.surrogates)}")
        else:
            problem = asset_problem(weights)
            if problem is not None:
                found.append(f"weights: {problem}")
        epsilon = settings.get("lubrication.epsilon")
        if epsilon is not None and not (isinstance(epsilon, float) and math.isfinite(epsilon) and epsilon > 0.0):
            found.append(f"lubrication epsilon is {epsilon!r} m, not a length above zero")
        return found

    #: The settings only the surrogate rung reads, which the mrs rung refuses written.
    surrogate_settings: ClassVar[tuple[str, ...]] = ("weights", "lubrication")

    def _mrs_setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = [f"{name}= is a setting of the surrogate rung, and the mrs rung reads none"
                 for name in self.surrogate_settings if getattr(self, name) is not None]
        spacing = settings.get("spacing")
        if spacing is None and self.spacing is None and self.unset_spacing_planned:
            pass
        elif not (isinstance(spacing, float) and math.isfinite(spacing) and spacing > 0.0):
            found.append(f"spacing is {spacing!r} m, not a length above zero: the mrs rung samples the body's surface "
                         f"with neighbouring samples about spacing apart, so write spacing=mm(...)")
        blob = cast(object, self.blob)
        if isinstance(blob, bool) or not (isinstance(blob, (int, float)) and math.isfinite(blob) and blob > 0.0):
            found.append(f"blob is {blob!r}, not a number above zero: the regularised Stokeslet's blob is blob times the "
                         f"samples' mean spacing, so write blob=0.5 (the 0.2 line's) or another")
        patches = cast(object, self.patches)
        if patches is not None and not (isinstance(patches, tuple) and patches
                                        and all(isinstance(p, str) for p in cast(tuple[object, ...], patches))):
            found.append(f"patches={patches!r} is not a tuple of the names of the body's kind's patches")
        if self.wall_kind is not None:  # none written and none determined is refused with the candidates, at plan
            wall = _wall_kind(cast(object, self.wall_kind))
            if isinstance(wall, str):
                found.append(wall)
        return found

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        if self.rung == "mrs":
            problems = at_rest_problems(ctx, self.rung) + self._mrs_problems(ctx)
            hull = ctx.geometry(of="body")
            if not problems and hull is not None and "spacing" in ctx.settings:
                self._sampled_hull(_owner(self, ctx.bindings), hull, ctx.scalar("spacing"), ctx.settings)
            return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values),
                        problems=tuple(problems), assets=self._wall_assets())
        problems: list[Problem] = at_rest_problems(ctx, self.rung)
        tube = uniform_tube(ctx.region("medium"), ctx.bindings["medium"])
        surrogate = self.surrogates.get(self.weights.name) if self.weights is not None else None
        if isinstance(tube, Problem):
            problems.append(tube)
        if surrogate is not None:
            hull = _trained_kind(surrogate, ctx.geometry(of="body"), ctx.bindings["body"])
            if isinstance(hull, Problem):
                problems.append(hull)
            elif not isinstance(tube, Problem):
                problems += _outside_training(surrogate, ctx, tube, hull)
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values),
                    problems=tuple(problems))

    def _wall_assets(self) -> tuple[str, ...]:
        """The shipped asset references the wall kind's correction reads (the tube's table), for the lock."""
        wall = _wall_kind(cast(object, self.wall_kind))
        if isinstance(wall, str):
            return ()
        try:
            return wall_correction_named(wall.name, wall.version).assets
        except KeyError:
            return ()

    def _mrs_problems(self, ctx: PlanContext) -> list[Problem]:
        """The mrs rung's refusals: a body with no hull whose kind gives samples, a patch the kind does not have, and a
        medium whose region the wall kind does not match."""
        body, medium = ctx.bindings["body"], ctx.bindings["medium"]
        found: list[Problem] = []
        hull = ctx.geometry(of="body")
        if hull is None or not implements(hull.kind, "sampling"):
            held = f"{body} has no hull" if hull is None else (
                f"{hull.label()}, a kind that does not implement the geometry protocol's sampling "
                f"(Geometry.sampling, which mime.scene.quasi_uniform gives any kind with samples)")
            found.append(Problem("drag.no-samples", f"the mrs rung samples the body's hull, and {held}",
                                 f"give {body} one part outside every other whose kind samples its surface (a sphere, "
                                 "a capsule, a screw, a cylinder, a box, an ellipsoid), "
                                 f"parts={{\"shell\": Part(Sphere(...))}}, or choose another rung"))
        elif self.patches is not None and not set(self.patches) <= set(hull.kind.patches):
            found.append(Problem("drag.patches", f"{hull.label()} has the patches {sorted(hull.kind.patches)}, and "
                                                 f"patches={self.patches!r} names others",
                                 "name patches= of the body's kind, or leave patches out for its closed surface"))
        if self.wall_kind is None:
            region = ctx.region("medium")
            shape = region.shape if region is not None else None
            matching = [f"{record.name} {record.version}" for record in wall_kinds_matching(shape)]
            fills = f"fills {region.path}, a {shape} region" if region is not None else "fills no region"
            found.append(Problem("drag.wall-kind", f"wall_kind is not written, and it is determined by the region the "
                                                   f"medium fills only where one wall kind matches it: medium {medium!r} "
                                                   f"{fills}, which {len(matching) or 'no'} wall kinds match"
                                                   + (f" ({', '.join(matching)})" if matching else ""),
                                 f"write wall_kind= the one this experiment has; the wall kinds are "
                                 f"{_wall_kinds_listed()}"))
            return found
        found += _wall_problems(self.wall_kind, ctx.region("medium"), medium, hull)
        return found

    def _sampled_hull(self, owner: str, hull: Hull, spacing: float, settings: Mapping[str, Magnitude]) -> Hull:
        """The hull the rung samples at ``spacing``: ``hull`` itself, or over a declared range of its numbers the
        point of the range with the most force points (:func:`force_points_rule`), whose sampling every body inside
        the range takes, moved with its own shape. ``settings`` are the entry's, in SI."""
        patches = self.patches if self.patches is not None else tuple(hull.kind.patches)
        return choose_over(owner, hull, "force points", force_points_rule(hull),
                           lambda held: force_points(held, spacing, patches), cost=_solve_cost)

    def _mrs_arguments(self, ctx: BuildContext) -> dict[str, Any]:
        """The arguments of the body's ``MRSDragNode`` but its name and timestep: its own (``body``) and the wall's
        constants (``wall``, ``wall_<name>``, the medium's)."""
        hull, wall = ctx.geometry(of="body"), _wall_kind(cast(object, self.wall_kind))
        if hull is None or isinstance(wall, str) or self.blob is None:
            raise ValueError("phase 1 refuses an mrs drag with no hull, no wall kind or no blob")
        correction = wall_correction_named(wall.name, wall.version)
        region = ctx.region("medium")
        constants = correction.prepare(dict(region.parameters) if region is not None else {}, run_dtype())
        spacing = ctx.scalar("spacing")
        patches = self.patches if self.patches is not None else tuple(hull.kind.patches)
        settings = {setting.name: hull.settings.get(setting.name, getattr(hull.kind, setting.name))
                    for setting in dataclasses.fields(hull.kind)}
        # The discretisation is the sampled hull's (over a declared range, the range's), and the node's settings its
        # own body's, so that its samples move with its shape.
        sampled = self._sampled_hull(ctx.id, hull, spacing, ctx.settings)
        sampling, wall_spacing, density = self._mrs_sampling(ctx, sampled, spacing, patches)
        counts, owners = self._quadrature_arguments(sampled, sampling, spacing)
        reference_counts, reference_owners = self._reference_quadrature(sampled, sampling, spacing)
        wall_counts, wall_owners, refinement = self._wall_quadrature(sampled, sampling, wall_spacing)
        return dict(**self._graded_arguments(sampled, sampling, density, counts, spacing),
            viscosity=ctx.scalar_value("viscosity"), blob=float(self.blob),
            kind=hull.kind.name, kind_version=hull.kind.version, patches=patches, counts=sampling.counts,
            part_centre=hull.centre, part_rotation=[c for row in hull.rotation for c in row],
            wall_kind=wall.name, wall_kind_version=wall.version, sampling_chosen=sampling.chosen,
            sampling_members=sampling.members, quadrature_counts=counts, quadrature_owners=owners,
            force_spacing=None if correction.adds_nothing else self._judged_spacing(spacing),
            reference_quadrature_counts=reference_counts,
            reference_quadrature_owners=reference_owners, wall_quadrature_counts=wall_counts,
            wall_quadrature_owners=wall_owners, wall_cells=None,
            wall_refinement=refinement,
            wall_refinement_written=None if refinement is None else cast(int, getattr(self, "wall_refinement")),
            **self._handover_arguments(ctx, wall_spacing),
            **{f"{WALL}{name}": value for name, value in constants.items()},
            **{f"{SHAPE}{name}": value for name, value in settings.items()})

    def _handover_arguments(self, ctx: BuildContext, spacing: float) -> dict[str, Any]:
        """The node's arguments for lubrication's hand-over at the local force ``spacing`` (m): none here, where the
        mrs rung takes none."""
        return {}

    def _mrs_sampling(self, ctx: BuildContext, hull: Hull, spacing: float, patches: tuple[str, ...]) -> tuple[
            Sampling, float, GradedDensity | None]:
        """The force points' sampling, the spacing the wall's quadrature is refined from, and the density it was
        graded at: the kind's sampling at the spacing, the spacing itself and none, here."""
        return hull.kind.sampling(hull.settings, spacing, patches), spacing, None

    def _graded_arguments(self, hull: Hull, sampling: Sampling, density: GradedDensity | None,
                          counts: tuple[tuple[int, ...], ...] | None, spacing: float) -> dict[str, Any]:
        """The node's arguments for a graded sampling beside the sampling's own: none here, where nothing is graded."""
        return {}

    def _build_mrs(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(MRSDragNode(ctx.node_name(), ctx.timestep, **self._mrs_arguments(ctx)))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))

    def _quadrature_arguments(self, hull: Hull, sampling: Sampling, spacing: float) -> tuple[
            tuple[tuple[int, ...], ...] | None, np.ndarray | None]:
        """The node's separated quadrature, its counts and its owner map: none here, where the samples are the
        quadrature."""
        return None, None

    def _judged_spacing(self, spacing: float) -> float | None:
        """The force spacing the node judges the body's distance from a wall against: none here, where nothing
        judges it."""
        return None

    def _reference_quadrature(self, hull: Hull, sampling: Sampling, spacing: float) -> tuple[
            tuple[tuple[int, ...], ...] | None, np.ndarray | None]:
        """The quadrature the node judges its own against at build, its counts and owner map: none here."""
        return None, None

    def _wall_quadrature(self, hull: Hull, sampling: Sampling, spacing: float) -> tuple[
            tuple[tuple[int, ...], ...] | None, np.ndarray | None, int | None]:
        """The quadrature the wall's image is integrated over each cell on, its counts, owner map and how many times
        finer than the force spacing it is: none here, where the image is taken at the force points."""
        return None, None, None

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        if self.rung == "mrs":
            self._build_mrs(ctx, b)
            return
        region, hull, weights = ctx.region("medium"), ctx.geometry(of="body"), self.weights
        tube = uniform_tube(region, ctx.bindings["medium"])
        if isinstance(tube, Problem) or region is None or hull is None or weights is None:
            raise ValueError("phase 1 refuses a drag with no tube, no hull or no weights")
        family = hull.kind.family_parameters(hull.settings)
        node = b.add_response_node(SurrogateDragNode(
            ctx.node_name(), ctx.timestep, weights=weights.name, weights_sha256=weights.sha256,
            viscosity=ctx.scalar_value("viscosity"), helix_wavenumber=float(family["wavenumber"]),
            length=float(family["length"]), radius=float(family["radius"]),
            outer_radius=hull.kind.radial_reach(hull.settings), vessel_shape=region.shape, vessel_centre=tube.centre,
            vessel_axis=tube.axis, vessel_radius=tube.radius,
            lubrication_epsilon=ctx.scalar("lubrication.epsilon") if self.lubrication is not None else None))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))


#: The mrs rung's defaults in drag 0.4.0, by the hull kind's name (a kind the tables do not name takes the compact
#: bodies'), chosen by measurement (``scripts/default_spacing_table.py``, labelled decisions of phase 7m-b): the count
#: of force points, as 0.3.0's ``DEFAULT_SAMPLES`` is read; the blob, a fraction of the force points' mean spacing.
SEPARATED_SAMPLES: Mapping[str, int] = MappingProxyType({"sphere": 338, "screw": 191})
SEPARATED_COMPACT_SAMPLES = 338
SEPARATED_BLOB: Mapping[str, float] = MappingProxyType({"sphere": 0.1, "screw": 0.25})
SEPARATED_COMPACT_BLOB = 0.1
#: A compact body's blob near a wall with lubrication's hand-over (drag 0.5.0): a labelled default, measured on a sphere
#: above a plane through the drag's node (W2a's first step): the coarse MRS's worst entry at the hand-over gap, 3.5 to
#: 4.2 % at this blob against 5.6 to 6.1 % at 0.1, at 1.3 to 2.4 times the build and the step (the quadrature's points
#: grow as one over the blob squared, and the step rebuilds them).
LUBRICATED_BLOB = 0.05
#: The default quadrature's refinement is this over the blob, rounded up: the quadrature's spacing at most half the
#: blob, at which the sphere's drag had converged to within 0.01% and FL-9's speed to within 0.16% (refinement 8 to 16
#: at blob 0.25; measured, phase 7m-b and its review).
QUADRATURE_PER_BLOB = 2.0
#: What was measured at the defaults (``scripts/default_spacing_table.py``, float64 for the misses, on a 24-thread laptop
#: CPU; phase 7m-b), as ``mime pending`` states it.
SEPARATED_SPACING_MEASURED = (
    "on a 100 um steel sphere settling in still water (338 force points), its drag 0.50% and its rotational drag "
    "1.45% above Stokes's law, within 2e-4 of the exact regularised drag at its blob, 1000 steps of it in 4.4 s; on "
    "FL-9 (222 force points), "
    "estimates: its speed per turn in free space 2.6% below the shipped free-space reference, itself known to 1.9%, a "
    "point on a ladder that is not monotone (6.1%, 2.6%, 4.0%, 3.5% and 1.3% below from 206 to 567 force points), and "
    "on the axis of a 1/4 in tube only 11% below the finest spacing measured (5984 force points), which is still "
    "moving (about 12% below a fit of the ladder); near a wall no error is stated (MIME-ANO-130); 6000 steps in the "
    "tube 12 minutes: write spacing= for a number to use")
SEPARATED_BLOB_MEASURED = (
    "with the spacing's default, at this blob: the sphere's drag error, the regularisation's, nearly in proportion to "
    "the blob (0.25%, 0.50% and 0.75% at blobs 0.05, 0.1 and 0.15 and 338 force points); FL-9's speed moved by under "
    "1% over blobs 0.1 to 0.25 at 416 to 957 force points")
SEPARATED_REFINEMENT_MEASURED = (
    "at two over the blob, refining further moved the sphere's drag by under 0.01% and FL-9's speed by 0.16% (8 to 16 "
    "at blob 0.25); at blob 0.1 a refinement of 8, under one over the blob, moved the sphere's drag by 0.08% and FL-9's "
    "speed by 0.9%; a refinement written below the rule is refused where the rule's quadrature moves the resistance by "
    "more than 5%")


#: Where a lab is told how to add a centreline to a kind (a custom geometry, an STL or CAD mesh): the guide's page,
#: ``docs/guide/centrelines.md``.
CENTRELINE_GUIDE = "centrelines"


def centreline_guide() -> str:
    """The address of the guide's page on adding a centreline, in this MIME version's documentation."""
    from mime.build._refusals import guide_page_of

    return guide_page_of(CENTRELINE_GUIDE)


def hull_centreline(hull: Hull | None) -> Centreline | str:
    """The body's hull kind's centreline, or why it has none: no hull, a kind that declares none, or one that refuses
    its settings (an ellipsoid that is no prolate spheroid)."""
    if hull is None:
        return "it has no hull"
    if not implements(hull.kind, "centreline"):
        screw = (" (a screw's only straight axis has no handedness, so a slender-body law along it would give no "
                 "coupling between turning and advancing: use Drag.MRS(...) or the surrogate rung)"
                 if hull.kind.name == "screw" else "")
        return f"its hull is {hull.label()}, a kind that declares no centreline{screw}"
    try:
        return hull.kind.centreline(hull.settings)
    except (ValueError, KeyError) as error:
        return f"its hull is {hull.label()}, whose centreline is refused: {error}"


@register_interaction
@dataclass(frozen=True, kw_only=True)
class UncoupledDrag(PointForceDrag):
    """The drag 0.4.0, each body's alone: the medium's resistance to the body's motion, by the written rung, with no
    hydrodynamic interaction between bodies in one medium (0.5.0, ``Drag``, couples a medium's mrs bodies). Kept
    registered so that a folder saved with it, and an experiment whose comparisons were declared against it, reproduce.

    ``surrogate``: the Cholesky MLP's resistance for a de Jongh screw body in a straight tube, from ``weights`` (a
    packaged asset, ``mime.data.asset("dejongh_mlp_v2")``), with ``lubrication=Lubrication(epsilon=...)`` blended in when given.
    It reads the body's pose at the start of the step, and the tube from the region the medium fills (``Medium(...,
    fills=tube.lumen)``), whose shape its node evaluates; it delivers the resistance (an operator the body's dynamics
    solves with implicitly) and the background load, zero in a medium at rest. It reads the medium's ``viscosity``, and
    the body's shape from its hull's kind (``ctx.geometry(of="body")``), which must be the kind the surrogate was
    trained on, the de Jongh helix, along the body's x axis: its ``nu``, ``R`` and ``L`` are the features, and its
    radial reach the outer radius the gap is measured from. A body or tube outside the surrogate's training data in
    ``nu``, ``kappa = R / R_tube`` or ``L / R``, or of another modulation or number of starts, is refused when the
    experiment is planned, in every profile (a value outside them is a model change, a declared clamp, which no profile
    relaxes);
    the offset from the axis moves during the run and is clamped, as the 0.2 line did, with ``clamp_fired`` in the
    node's state, ``tilt_fired`` where the body's axis has left the tube's by more than the tilt the surrogate is
    declared valid to, and ``bore_fired`` where it has left it by more than the bore allows (the surrogate cannot see a
    tilt; a screw whose axial moment is its smallest turns broadside under a rotating field, and in a bore narrower
    than it is long goes through the wall, unless a contact holds its hull: ``Contact.Projection`` 0.2.0 holds it at
    the bore's limit and raises its ``touching`` flag on the same result, beside these). Its derivative is piecewise,
    as its switches say: at the offset's clamp,
    across the tube's axis
    (where the surrogate is not symmetric), where the body's axis turns across the tube, and at the lubrication's
    floors. The resistance is delivered about the body's centre of mass:
    the surrogate's, trained about the end of the body's axis (``zeta = 0``), is moved there, ``L / 2`` along the
    body's x axis taken as the helix's (``SurrogateDragNode``).

    ``mrs``: the regularised-Stokeslet resistance (``MRSDragNode``) of the body's hull on a separated quadrature
    (Smith 2018; Gallagher & Smith 2021): the force points are its kind's sampling (``Geometry.sampling``: for MIME's
    kinds quasi-uniform, as many as their quadrature has at the spacing, none crowding another) on ``patches`` (all of
    them, the closed surface, when none are written), neighbouring force points about ``spacing`` apart, one traction
    unknown at each, constant over its cell; the integral is the kind's own quadrature ``quadrature_refinement`` times
    finer, each of its points in the cell of the force point nearest it (a map made once at build and held; with
    ``quadrature_refinement=1`` the quadrature is the force points themselves, one to a cell, the point-force method of
    0.3.0); the blob is ``blob`` times the force points' mean spacing, set on its own, since the quadrature resolves a
    blob smaller than the force spacing. The resistance is about the body's centre of mass at the pose it reads each
    step. The free part's matrix is the same at every pose of a rigid body, so it is assembled once per build in the
    body's frame, streamed over the quadrature in chunks (never every force point against every quadrature point at
    once), and held: in free space the resistance itself is held and turned to the pose, with no solve per step; with a
    wall, the matrix is turned to the pose and the wall's correction is added and solved at each step. Any kind that
    implements the sampling serves: a sphere, a capsule, the de Jongh helix, a cylinder, a box, an ellipsoid, a lab's;
    a kind without it is refused by name when the experiment is planned. ``wall_kind`` names the wall kind whose
    correction the solve adds (``mime.walls``; ``"<name>"`` for its one registered version, or ``"<name> <version>"``),
    and it must match the region the medium fills exactly: ``free_space`` a medium that fills none, ``tube`` a tube's
    lumen (it reads the asset ``unit_tube_R0.997``, refused by name when planned if it is not in the cache), ``plane``
    the half-space a plane bounds (its regularised image, of the drag's own blob, in closed form); a wall kind that
    matches another region is refused with the wall kinds that match, and a reserved one with its reason. **A wall's
    correction is evaluated at the force points**, each cell's force placed at its force point and
    weighted by the cell's area: a wall's image system is smooth inside the fluid, varying over the distance to the
    wall, which the force points resolve where the body is further from the wall than about a force spacing, while the
    free part's near field varies over the blob, which needs the fine quadrature; and the image is then read at
    ``N^2`` pairs per step, not ``N`` times the quadrature's count. Every wall kind's correction is evaluated there, the
    tube's included, so the rung is the same for any wall kind. ``wall_refinement``, written, integrates it instead over
    each cell near the wall (MIME-ANO-130), on the kind's quadrature that many times finer than the force spacing, each
    cell's weights scaled to its area and its points capped (``WALL_CELL_CAP``). It reads the medium's ``viscosity`` and delivers the
    resistance and a zero background load, as the surrogate does. Its derivative is exact: no switch is evaluated on
    its path, the force points and the quadrature are smooth functions of the kind's numbers while the map is held, the
    held quantity's derivative is the one made from the leaves, and the solve is differentiated by the linear-solve
    rule. **Its symmetry and definiteness:** ``W^T A^-1 M`` is symmetric only to the discretisation's error (the
    traction's columns carry the cells' areas), so the node holds its symmetric part. The build checks the free part:
    what it holds is positive definite, the part it drops is under a hundredth (a gross-defect check: the point forces
    are symmetric to rounding whatever they miss), and a quadrature_refinement written below its rule's does not move
    the free resistance's significant entries by more than 5% from the rule's quadrature (``drag.mrs-quadrature``: that
    move is the quadrature's own error, estimated); with a wall, the resistance at the body's starting pose is positive
    definite (``drag.mrs-resistance``), and the build reports the body's least distance from the wall against the force
    spacing and the spacing the experiment's closest approach needs. **Near a wall each step judges its resistance
    before the body takes it** (MIME-ANO-130): one that is not positive definite (its least eigenvalue, each row and
    column scaled by its diagonal, not above zero) is replaced by NaN, so that the run stops at that step
    (``drag.mrs-indefinite``, naming the step, the pose and the gap) rather than taking energy from the drag; a body
    within one force spacing of the wall raises ``near_wall``, and a resistance symmetric only to more than a
    twentieth of itself raises ``asymmetric``. Every indefinite resistance measured lay within half a force spacing of
    the wall, on the point forces too.

    Each rung's status is its own (``rung_gradients``), and the build records the attached rung's: the surrogate's
    ``piecewise_jump``, with its switches, and the mrs's ``exact``. The entry's is the least of them.

    Version 0.4.0 separates the mrs rung's quadrature from its force points and chooses its defaults
    again; its surrogate rung is 0.3.0's. Version 0.3.0 (``PointForceDrag``, kept registered so that a folder saved
    with it reproduces) put a point force at each sample. Its default spacing, like 0.3.0's, within half a unit below
    a power of ten steps to 0.99, not 0.999, kept so released runs reproduce (``step_from_boundary``; 0.5.0's does
    not)."""

    version = "0.4.0"
    setting_roles = {
        **PointForceDrag.setting_roles,
        "wall_kind": determined(PHYSICAL, "on the mrs and rft rungs, the one wall kind that matches the region the "
                                          "medium fills (free_space where it fills none)", by="default_wall_kind"),
        "coefficients": determined(NUMERICAL, "on the rft rung, Lighthill's where every segment of the body's "
                                              "centreline declares a wavelength (a helix), else the slender "
                                              "spheroid's (a straight centreline)", by="default_coefficients"),
        "spacing": numerical_default(
            "on the mrs rung, the largest spacing, to three significant figures, at which the hull's kind gives its "
            "family's count of force points on its grid (SEPARATED_SAMPLES: a compact body's, a screw's); for a body a "
            "law holds off the wall of its medium's region (a contact), no coarser than that closest approach over "
            "CONFINED_GAP force spacings, or INTEGRATED_GAP where wall_refinement is written; on the rft rung, its "
            "points' spacing along the centreline, RFT_POINTS_PER_RADIUS a filament radius",
            "a first look at a laptop's cost, chosen by measurement, never a converged value: for each family the "
            "count whose miss against its reference (Stokes's law, the shipped helix reference) is within the stated "
            "error at a run's cost a laptop's CPU takes (scripts/default_spacing_table.py measures the misses and "
            "the run times on a ladder of spacings, and the stamp states them)", by="default_spacing",
            measured=SEPARATED_SPACING_MEASURED),
        "blob": numerical_default(
            "on the mrs rung, by the hull kind's family (SEPARATED_BLOB): a tenth of the force points' mean spacing "
            "for a compact body, a quarter for a screw",
            "chosen by measurement with the spacing's default: the regularisation's error grows with the blob, so a "
            "smooth body's drag wants it small; a screw's swim speed barely depends on it and its quadrature costs "
            "the square of the blob's inverse (scripts/default_spacing_table.py)", by="default_blob",
            measured=SEPARATED_BLOB_MEASURED),
        "wall_refinement": convention(
            NUMERICAL, "none, the image at the force points: integrating it over each cell is written where it is wanted "
                       "(MIME-ANO-130)"),
        "quadrature_refinement": numerical_default(
            "on the mrs rung, two over the blob, rounded up: the quadrature's spacing at most half the blob",
            "chosen by measurement: finer, neither family's number moved by what the stamp states (the sphere's drag "
            "and rotational drag, a screw's speed, over a ladder of blobs); coarser, the quadrature no longer resolves "
            "the blob (scripts/default_spacing_table.py)",
            by="default_quadrature_refinement", measured=SEPARATED_REFINEMENT_MEASURED),
    }
    references = ("deJongh2025NonlinDyn", "CoxBrenner1967", "GoldmanCoxBrenner1967", "BradyBossis1988", "Cortez2005",
                  "Smith2018", "GallagherSmith2021")
    samples_by_family = SEPARATED_SAMPLES
    compact_samples = SEPARATED_COMPACT_SAMPLES
    unset_spacing_planned = True
    mrs_settings = (*PointForceDrag.mrs_settings, "quadrature_refinement", "wall_refinement")
    rungs = (*PointForceDrag.rungs, "rft")
    builds = {**PointForceDrag.builds, "rft": (RFTDragNode,)}
    rung_gradients = {**PointForceDrag.rung_gradients, "rft": RungGradient(differentiability=GradientStatus.EXACT)}
    reynolds = (*PointForceDrag.reynolds, ReynoldsRange(STOKES_REYNOLDS, STOKES_BASIS, rung="rft"))
    support = (*PointForceDrag.support, SupportRow("rft", {
        "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name], publishes=[POSE.name]),
    }, note="a rigid body whose dynamics takes a resistance and whose hull kind declares a centreline, in a medium at "
            "rest that a wall kind matches"))
    #: The settings the rft rung reads, beside the wall kind and the spacing.
    rft_settings: ClassVar[tuple[str, ...]] = ("coefficients",)
    raises = (
        *PointForceDrag.raises,
        Flag("indefinite", "the step's resistance, the wall's correction in it, was not positive definite (its least "
                           "eigenvalue, each row and column scaled by its diagonal, not above zero): the node replaced "
                           "it by NaN before the body took it, so the run stopped there (drag.mrs-indefinite, "
                           "MIME-ANO-130)", rung="mrs"),
        Flag("near_wall", "the body lay within one force spacing of the wall (NEAR_WALL), where the resistance has been "
                          "measured indefinite closer in (MIME-ANO-130) and a definite one is not shown accurate; above the plane wall the "
                          "error there is estimated (a sphere, blob 0.1, against exact answers): the normal entry "
                          "+3.6% at one spacing, +10.3% at 0.55 and +14% at 0.5 with the image at the force points, "
                          "the worst entry 2.6% to 14.8% at 0.5 with it integrated (the plane wall kind's page); the "
                          "tube's measurements are MIME-ANO-130's", rung="mrs"),
        Flag("asymmetric", "the step's resistance, before its symmetric part was taken, was symmetric only to more than "
                           "a twentieth of itself (CONFINED_ASYMMETRY_LIMIT), so its error is at least half that share "
                           "(the share bounds it from below; a small share bounds nothing)", rung="mrs"),
        Flag("unmeasured", "the body lay outside the points the rft rung's validity lines were measured over (another "
                           "family, slenderness, coefficient set or a coarser spacing; with a wall, a gap other than "
                           "the one measured): its error against the MRS is not measured there", rung="rft"),
        Flag("indefinite", "the step's resistance, the wall's image in it, was indefinite past the semidefinite floor "
                           "(its least eigenvalue, each row and column scaled by its diagonal, below minus "
                           "SEMIDEFINITE_FLOOR_ULPS units in the last place): the node replaced it by NaN before the "
                           "body took it, so the run stopped there (drag.rft-indefinite)", rung="rft"),
    )
    validity = (
        *(line for line in PointForceDrag.validity if not line.startswith("mrs: a medium at rest")
          and not line.startswith("mrs: on the de Jongh helix")),
        "mrs: a medium at rest that its wall kind matches; on the separated quadrature its error is the "
        "regularisation's, of the order of the blob over the body's size, and the force points' constant tractions', "
        "of the order of the force spacing, the quadrature's falling as quadrature_refinement rises; measured against "
        "the sphere's exact regularised drag and Stokes's law, the prolate spheroid's closed form, Haberman and "
        "Sayre's confined sphere and a de Jongh screw's shipped free-space reference by its oracle tests, not bounded "
        "here",
        "rft: quasi-steady Stokes flow, the limit of vanishing Reynolds number (declared, STOKES_REYNOLDS, not "
        "measured); a medium at rest; a slender body whose segments do not interact through the free fluid; its "
        "validity lines measured against the mrs rung on the same bodies (a prolate spheroid's against its closed "
        "form), each the largest miss over the points it names, an estimate against the MRS at the MRS's own "
        "resolution, never a bound, held only over those points (the families, their slenderness and the misses are "
        "the record's, docs/data/slender_validity, and MEASURED in mime.nodes.hydro._rft_drag); a body outside them "
        "raises unmeasured at every step and the build report says so. In the helix families measured it is a model "
        "of the swimming speed, not of the forces: its translational resistance is several times the MRS's, since its "
        "segments do not interact; a straight body's spin about its own axis meets no drag",
        "rft: with the tube's table, the helix family was measured at one gap to the wall only, the "
        "largest of four tried, where the wall's feedback of its local drag made its resistance many times the MRS's; at "
        "the three narrower gaps its resistance was indefinite, which the build refuses (drag.rft-indefinite) and a step "
        "stops; at any other gap, wider included, a step raises unmeasured, since the line is that one point's",
        "mrs: a wall's correction is taken at the force points unless wall_refinement is written, and they do not "
        "resolve the image near the wall: within half a force spacing of it the resistance was measured indefinite, and "
        "within a quarter of one with the image integrated over each cell (a step whose resistance is indefinite "
        "stops the run, drag.mrs-indefinite), within one a step raises near_wall, and there a definite resistance is "
        "not shown accurate; in the tube the confinement's effect converges more slowly than the free part "
        "(MIME-ANO-130)",
    )
    quadrature_refinement: int | None = None
    wall_refinement: int | None = None
    coefficients: str | None = None

    @override
    def unrecorded_settings(self) -> frozenset[str]:
        """Drag 0.4.0 was released without ``coefficients``; the other rungs read none, so their resolved record, and
        the input hash of every program on them, are the released version's."""
        return frozenset() if self.rung == "rft" else frozenset(self.rft_settings)

    @classmethod
    def RFT(cls, **settings: Any) -> Self:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="rft", **settings)

    @classmethod
    def BEM(cls, **settings: Any) -> Self:  # noqa: N802 - the old rung's alias, refused by its name
        """Refused, naming the new name: formerly the bem rung, renamed ``Drag.MRS`` in 0.3.0."""
        from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass

        written = ", ".join(f"{name}=..." for name in settings)
        raise ArgumentRefused([Refusal(
            "call.renamed", RefusalClass.USER,
            "Drag.BEM is Drag.MRS since 0.3.0: the rung is the method of regularised Stokeslets, not a boundary-element "
            "method, and the old name is kept for a true boundary-element rung; its settings are the same, and drag "
            "0.4.0 moved its numbers (the changelog says by how much and how to keep the old ones)",
            f"write Drag.MRS({written}) where Drag.BEM({written}) was written")])

    @override
    def default_spacing(self, ctx: PlanContext) -> Quantity | None:
        """The mrs rung's spacing when none is written: its family's (as 0.3.0's rule reads ``SEPARATED_SAMPLES``), and
        for a body a law holds off the wall of the region its medium fills (a contact's margin, its closest approach:
        ``PlanContext.approach``), no coarser than that approach over ``CONFINED_GAP``, or ``INTEGRATED_GAP`` where
        ``wall_refinement`` integrates the wall's image over the cells, so that its closest approach lies outside the
        band where the resistance was measured indefinite (MIME-ANO-130). A body nothing holds off the wall takes its
        family's, and the build reports that its closest approach is the wall itself. A body a law holds on the wall
        (a margin of 0) takes none: no force spacing clears it, and its plan is refused (``drag.mrs-on-wall``)."""
        if self.rung == "rft":
            return self._rft_spacing(ctx)
        family = super().default_spacing(ctx)
        approach = ctx.approach(of="body")
        if family is None or self.rung != "mrs" or ctx.region("medium") is None or approach is None:
            return family
        if approach <= 0.0:
            return None
        held = float(cast(float, family.value))
        written = cast(object, self.wall_refinement)
        confined = approach / (INTEGRATED_GAP if isinstance(written, int) and written > 1 else CONFINED_GAP)
        if confined >= held:
            return family
        spacing = float(f"{confined:.3g}")
        if spacing > confined:  # rounded up past the bound: one unit of the third figure below it
            spacing = _step_down(spacing, confined, self.step_from_boundary)
        return m(spacing)

    def _rft_spacing(self, ctx: PlanContext) -> Quantity | None:
        """The rft rung's spacing when none is written: ``RFT_POINTS_PER_RADIUS`` points a filament radius, the
        thinnest of the centreline's segments', to three significant figures."""
        centreline = hull_centreline(ctx.geometry(of="body"))
        hull = ctx.geometry(of="body")
        if isinstance(centreline, str) or hull is None:
            return None
        radii: list[float] = []
        for segment in centreline.segments:
            if segment.filament_radius is None:
                return None
            radii.append(float(segment.filament_radius(hull.settings)))
        return m(float(f"{min(radii) / RFT_POINTS_PER_RADIUS:.3g}"))

    def default_coefficients(self, ctx: PlanContext) -> str | None:
        """The rft rung's coefficients when none are written: Lighthill's where every segment of the body's centreline
        declares a wavelength (a helix), else the slender spheroid's (a straight centreline)."""
        if self.rung != "rft":
            return None
        centreline = hull_centreline(ctx.geometry(of="body"))
        return None if isinstance(centreline, str) else default_coefficients(centreline)

    @override
    def default_wall_kind(self, ctx: PlanContext) -> str | None:
        """The mrs and rft rungs' wall kind when none is written: the one that matches the region the medium fills."""
        if self.rung != "rft":
            return super().default_wall_kind(ctx)
        return dataclasses.replace(self, rung="mrs").default_wall_kind(ctx)

    @override
    def definiteness(self) -> Definiteness | None:
        """The rft rung's resistance is solved at the body's state, and with a wall its definiteness depends on it:
        checked at run time, as the MRS's is."""
        return Definiteness.STATE if self.rung == "rft" else super().definiteness()

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        if self.rung == "rft":
            return self._rft_setting_problems(settings)
        found = super().setting_problems(settings)
        if self.coefficients is not None:
            found.append(f"coefficients= is a setting of the rft rung, and the {self.rung} rung reads none")
        return found

    def _rft_setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        refused = ("weights", "lubrication", "blob", "patches", "quadrature_refinement", "wall_refinement")
        found = [f"{name}= is a setting of another rung, and the rft rung reads none" for name in refused
                 if getattr(self, name) is not None]
        spacing = settings.get("spacing")
        if spacing is not None and not (isinstance(spacing, float) and math.isfinite(spacing) and spacing > 0.0):
            found.append(f"spacing is {spacing!r} m, not a length above zero: the rft rung reads the body's centreline "
                         f"at points about spacing apart, so write spacing=um(...)")
        written = cast(object, self.coefficients)
        if written is not None and written not in COEFFICIENT_SETS:
            found.append(f"coefficients={written!r} is not one of {COEFFICIENT_SETS}")
        if self.wall_kind is not None:
            wall = _wall_kind(cast(object, self.wall_kind))
            if isinstance(wall, str):
                found.append(wall)
        return found

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        if self.rung != "rft":
            return super().plan(ctx)
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values),
                    problems=tuple(at_rest_problems(ctx, self.rung) + self._rft_problems(ctx)),
                    assets=self._wall_assets())

    def _rft_problems(self, ctx: PlanContext) -> list[Problem]:
        """The rft rung's refusals: a body whose hull declares no centreline (``drag.no-centreline``, naming the guide's
        page on adding one), a coefficient set its centreline cannot take, and the medium's wall as the mrs rung's."""
        body, medium = ctx.bindings["body"], ctx.bindings["medium"]
        hull = ctx.geometry(of="body")
        centreline = hull_centreline(hull)
        found: list[Problem] = []
        if isinstance(centreline, str):
            found.append(Problem(
                "drag.no-centreline",
                f"the rft rung reads the body's centreline, and {body} has none: {centreline}. A slender body's drag "
                f"along its centreline needs the curve its kind declares; adding one to a kind (a custom geometry, an "
                f"STL or CAD mesh) is set out at {centreline_guide()}",
                f"give {body} a hull whose kind declares a centreline (Helix, Capsule, a prolate Ellipsoid), add one "
                f"to its kind as {centreline_guide()} shows, or choose another rung: Drag.MRS(...)"))
        else:
            name = self.coefficients if self.coefficients is not None else default_coefficients(centreline)
            problem = coefficient_problem(name, centreline)
            if problem is not None:
                found.append(Problem("drag.rft-coefficients", f"{body}'s centreline cannot take this coefficient set: "
                                                              f"{problem}",
                                     "write coefficients= one its centreline takes"))
        if self.wall_kind is None:
            region = ctx.region("medium")
            shape = region.shape if region is not None else None
            matching = [f"{record.name} {record.version}" for record in wall_kinds_matching(shape)]
            fills = f"fills {region.path}, a {shape} region" if region is not None else "fills no region"
            found.append(Problem("drag.wall-kind", f"wall_kind is not written, and it is determined by the region the "
                                                   f"medium fills only where one wall kind matches it: medium {medium!r} "
                                                   f"{fills}, which {len(matching) or 'no'} wall kinds match"
                                                   + (f" ({', '.join(matching)})" if matching else ""),
                                 f"write wall_kind= the one this experiment has; the wall kinds are "
                                 f"{_wall_kinds_listed()}"))
            return found
        found += _wall_problems(self.wall_kind, ctx.region("medium"), medium, hull)
        return found

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        if self.rung != "rft":
            super().build(ctx, b)
            return
        hull, wall = ctx.geometry(of="body"), _wall_kind(cast(object, self.wall_kind))
        centreline = hull_centreline(hull)
        if hull is None or isinstance(wall, str) or isinstance(centreline, str) or self.coefficients is None:
            raise ValueError("phase 1 refuses an rft drag with no centreline, no wall kind or no coefficients")
        correction = wall_correction_named(wall.name, wall.version)
        region = ctx.region("medium")
        constants = correction.prepare(dict(region.parameters) if region is not None else {}, run_dtype())
        settings = {setting.name: hull.settings.get(setting.name, getattr(hull.kind, setting.name))
                    for setting in dataclasses.fields(hull.kind)}
        node = b.add_response_node(RFTDragNode(
            ctx.node_name(), ctx.timestep, viscosity=ctx.scalar_value("viscosity"), kind=hull.kind.name,
            kind_version=hull.kind.version, counts=centreline_counts(centreline, hull.settings, ctx.scalar("spacing")),
            part_centre=hull.centre, part_rotation=[c for row in hull.rotation for c in row],
            coefficients=self.coefficients, wall_kind=wall.name, wall_kind_version=wall.version,
            spacing=ctx.scalar("spacing"),
            **cast(dict[str, Any], {f"{WALL}{name}": value for name, value in constants.items()}),
            **cast(dict[str, Any], {f"{SHAPE}{name}": value for name, value in settings.items()})))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))

    @override
    def _mrs_problems(self, ctx: PlanContext) -> list[Problem]:
        """0.3.0's refusals, and a body a law holds on the wall of the region its medium fills (its closest approach
        0, a contact's margin of 0: ``drag.mrs-on-wall``). Every indefinite resistance mapped lay within half a force
        spacing of the wall, and a body on the wall is within every share of every force spacing, so no spacing clears
        it, written or default (MIME-ANO-130)."""
        found = super()._mrs_problems(ctx)
        approach = ctx.approach(of="body")
        if ctx.region("medium") is None or approach is None or approach > 0.0:
            return found
        body, medium = ctx.bindings["body"], ctx.bindings["medium"]
        found.append(Problem(
            "drag.mrs-on-wall",
            f"a law holds {body} on the wall of the region medium {medium!r} fills, with a margin of 0, so its closest "
            f"approach is the wall itself: the resistance was measured indefinite within half a force spacing of the "
            f"wall, and no force spacing keeps a body on the wall clear of that band (MIME-ANO-130)",
            f"hold {body} off the wall by a margin above 0, Contact.Projection(margin=mm(0.1)): the default force "
            f"spacing then keeps that margin {CONFINED_GAP:g} force spacings or more from the wall"))
        return found

    @override
    def default_blob(self, ctx: PlanContext) -> float | None:
        """The mrs rung's blob when none is written, by the hull kind's family."""
        if self.rung != "mrs":
            return None
        hull = ctx.geometry(of="body")
        return SEPARATED_BLOB.get(hull.kind.name, SEPARATED_COMPACT_BLOB) if hull is not None else \
            SEPARATED_COMPACT_BLOB

    def default_quadrature_refinement(self, ctx: PlanContext) -> int | None:
        """The mrs rung's quadrature refinement when none is written: ``QUADRATURE_PER_BLOB`` over the blob (the one
        written, or its default), rounded up."""
        if self.rung != "mrs":
            return None
        blob = self.blob if self.blob is not None else self.default_blob(ctx)
        if not (isinstance(blob, (int, float)) and not isinstance(blob, bool) and blob > 0.0):
            return None
        return max(1, math.ceil(QUADRATURE_PER_BLOB / float(blob) - 1e-9))

    @override
    def _mrs_setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = super()._mrs_setting_problems(settings)
        refinement = cast(object, self.quadrature_refinement)
        if isinstance(refinement, bool) or not (isinstance(refinement, int) and refinement >= 1):
            found.append(f"quadrature_refinement is {refinement!r}, not a whole number of 1 or more: the quadrature's "
                         f"spacing is the spacing over it, so write quadrature_refinement=8 (or 1 for the force points "
                         f"themselves)")
        wall = cast(object, self.wall_refinement)
        if wall is not None and (isinstance(wall, bool) or not (isinstance(wall, int) and wall >= 1)):
            found.append(f"wall_refinement is {wall!r}, not a whole number of 1 or more: the wall's image is integrated "
                         f"over each cell on a quadrature that many times finer than the force spacing, so write "
                         f"wall_refinement=4 (or 1 for the image at the force points)")
        return found

    @override
    def _judged_spacing(self, spacing: float) -> float | None:
        return spacing

    @override
    def _reference_quadrature(self, hull: Hull, sampling: Sampling, spacing: float) -> tuple[
            tuple[tuple[int, ...], ...] | None, np.ndarray | None]:
        """Where the written quadrature_refinement is below its rule's (two over the blob, rounded up), the rule's
        quadrature, or the next finer that leaves no cell empty, which the node assembles once more at build to
        estimate the written one's error (``drag.mrs-quadrature``); none at or above the rule."""
        written = int(cast(int, self.quadrature_refinement))
        rule = max(1, math.ceil(QUADRATURE_PER_BLOB / float(cast(float, self.blob)) - 1e-9))
        for refinement in range(rule, 2 * rule + 2):
            if refinement <= written:
                return None, None
            counts = quadrature_counts(hull.kind, hull.settings, spacing / refinement, sampling.patches)
            owners = quadrature_owners(hull.kind, hull.settings, sampling, counts)
            if not empty_cells(owners, len(sampled_points(hull.kind, hull.settings, sampling))):
                return counts, owners
        return None, None

    @override
    def _wall_quadrature(self, hull: Hull, sampling: Sampling, spacing: float) -> tuple[
            tuple[tuple[int, ...], ...] | None, np.ndarray | None, int | None]:
        """With a wall_refinement above 1, the kind's quadrature that many times finer than the force spacing, or the
        next finer that leaves no cell empty, its owner map and the refinement taken, which the build report states:
        the wall's image is integrated over each cell on it (``mime.nodes.hydro.mrs.cell_image``). None at 1, where
        the image is taken at the force points, and in free space."""
        written = cast(object, self.wall_refinement)
        if not isinstance(written, int) or written <= 1:
            return None, None, None
        cells = len(sampled_points(hull.kind, hull.settings, sampling))
        for refinement in range(written, 2 * written + 2):
            counts = quadrature_counts(hull.kind, hull.settings, spacing / refinement, sampling.patches)
            owners = quadrature_owners(hull.kind, hull.settings, sampling, counts)
            if not empty_cells(owners, cells):
                return counts, owners, refinement
        return None, None, None

    @override
    def _quadrature_arguments(self, hull: Hull, sampling: Sampling, spacing: float) -> tuple[
            tuple[tuple[int, ...], ...] | None, np.ndarray | None]:
        """The separated quadrature: the kind's quadrature ``quadrature_refinement`` times finer and its owner map, or
        with a refinement of 1 the force points themselves, one to a cell (no counts)."""
        refinement = int(cast(int, self.quadrature_refinement))
        if refinement == 1:
            return None, np.arange(len(sampled_points(hull.kind, hull.settings, sampling)), dtype=np.int32)
        counts = quadrature_counts(hull.kind, hull.settings, spacing / refinement, sampling.patches)
        return counts, quadrature_owners(hull.kind, hull.settings, sampling, counts)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Drag(UncoupledDrag):
    """The medium's resistance to the body's motion, by the written rung; ``Drag.Surrogate(...)`` is
    ``Drag(rung="surrogate", ...)`` and saves identically. Every rung as 0.4.0's (``UncoupledDrag``), but for bodies
    sharing a medium on the mrs rung.

    **A medium's near field.** Where two bodies or more in one medium take the mrs rung, the build groups them into one
    joint node, ``near_field:<medium>`` (``JointDragNode``), and no body has a node of its own: each step it solves
    their joint system (each body's free part as 0.4.0 holds it, each pair coupled by the source's regularised
    Stokeslet, integrated over every cell where the pair is within the reach of each other and taken at its force
    points beyond, and the wall's image between them), and delivers each body its own coupled resistance and, as its background load, the
    others' lagged load on it, ``-sum_b R_ab V_b`` at their twists at the step's start. The bodies' steps are lagged: a
    body takes its own resistance implicitly and the others' loads from the step before. **The lagged step's gain**,
    the spectral radius of ``(M/dt + R_d)^-1 R_off`` from each body's mass matrix (its dynamics publishes it where the
    near field reads it), is checked every step: at one or above the lagged error would grow (three overdamped bodies
    in a triangle near contact), and the run stops there (``drag.near-field-gain``); the gain and the relaxation in
    steps, ``1/(1-g)``, are reported. Each body's coupled resistance and the cluster's grand resistance are judged as
    0.4.0's (their scaled least eigenvalue above zero, else the run stops, ``drag.near-field-indefinite``, the margin
    reported). A body alone in its medium keeps its own node, 0.4.0's, step for step. The joint system is solved by
    one dense LU at every size, or where its drags declare it (``near_field=NearField(solve="gmres")``) by GMRES, each
    body's own block's inverse its preconditioner, each solve to the group's ``tolerance`` on its true residual and
    never silently short of it (``solve_flagged``, warned of with its count; a gradient study's recheck re-solves a
    member whose solve stopped short); on either route the build refuses a cluster whose run would not fit in the
    memory a run can have (``drag.near-field-memory``), the only bound on its size.

    **Its settings by concern**: the core flat (``spacing``, ``blob``, ``patches``, ``wall_kind``, and 0.4.0's others);
    the force points' grading in ``sampling=ForceSampling(...)``; the near field's joint solve in
    ``near_field=NearField(...)``; near contact (``lubrication``) where it was, until its seam's form is decided.

    Version 0.5.0 sizes the default spacing's step below a rounding past its bound from the bound
    (``step_from_boundary``): exact rounding-down to three figures. Versions 0.3.0 and 0.4.0 size it from the rounded
    value, so that within half a unit below a power of ten it steps to 0.99, not 0.999, kept so released runs
    reproduce.

    Version 0.5.0 couples a medium's mrs bodies; version 0.4.0 (``UncoupledDrag``, kept registered) computed each
    body's drag alone, and its surrogate rung, like 0.5.0's, is 0.3.0's; its rft rung, added before 0.5.0, is 0.5.0's.
    Version 0.3.0 (``PointForceDrag``) put a point force at each sample.

    **Its record** holds every setting on every rung, ``coefficients`` included (``unrecorded_settings`` is empty): the
    hook keeps a setting added to a version after its release out of that version's record, so that its released
    programs keep their input hash (0.4.0's ``coefficients``), and 0.5.0 has no release before its settings."""

    version = "0.5.0"
    step_from_boundary = True
    builds = {**UncoupledDrag.builds, "mrs": (MRSDragNode, JointDragNode), "stokes": (PointDragNode,)}
    rungs = (*UncoupledDrag.rungs, "stokes")
    rung_gradients = {**UncoupledDrag.rung_gradients, "stokes": RungGradient(differentiability=GradientStatus.EXACT)}
    reynolds = (*UncoupledDrag.reynolds, ReynoldsRange(STOKES_REYNOLDS, STOKES_BASIS, rung="stokes"))
    support = (*UncoupledDrag.support, SupportRow("stokes", {
        "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name],
                         publishes=[POSE.name, TWIST.name]),
        "medium": requires(accepts=[FLOW_EXCHANGE_X.name, FLOW_EXCHANGE_Y.name, FLOW_EXCHANGE_Z.name,
                                    FLOW_PARTNERS.name], publishes=[FLOW_FACES.name]),
    }, note="a rigid sphere whose dynamics takes a resistance, in a medium whose flow is solved on a grid that takes "
            "exchange deliveries (the FVM)"))
    #: The kernel widths, in cells, the stokes rung's build chooses among: the narrowest at or above max(1, a/(1.7 h))
    #: whose coupled one-step margin clears the floor at the run's values.
    WIDTHS: ClassVar[tuple[float, ...]] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
    setting_roles = {
        **UncoupledDrag.setting_roles,
        "one_way": convention(PHYSICAL, "two-way: on the stokes rung the drag's and the couple's reactions are "
                                        "scattered back onto the medium's grid; one_way=True is the declared "
                                        "approximation, its reactions not delivered"),
        "kernel_width": convention(NUMERICAL, "on the stokes rung, none: the exchange kernel's width the narrowest of "
                                              "the ladder (WIDTHS) at or above the rule's least width whose coupled "
                                              "one-step margins clear the floor at the run's values (the validity line "
                                              "states the rule; the build reports the width)"),
        "self_history": convention(NUMERICAL, "on the stokes rung, 'exact': the body's own free-box flow advanced by "
                                              "the grid's operator, which the rate terms' reading removes; "
                                              "self_history='modes' the declared opt-in, fitted decay modes, cheaper "
                                              "for many bodies, its gate the swap test against the exact route (exact "
                                              "by default, the cheaper approximation declared)"),
        "sampling": convention(
            NUMERICAL, "on the mrs rung, none: the force points quasi-uniform, as every released program's (each of "
                       "ForceSampling's settings states its own)"),
        "quadrature_refinement": numerical_default(
            "on the mrs rung, two over the blob, rounded up: the quadrature's spacing at most half the blob; with "
            "grading or contact written, at least two over FINEST, so that the finest graded cells hold points",
            "chosen by measurement: finer, neither family's number moved by what the stamp states (the sphere's drag "
            "and rotational drag, a screw's speed, over a ladder of blobs); coarser, the quadrature no longer resolves "
            "the blob (scripts/default_spacing_table.py); a graded cell a quarter of the force spacing across is left "
            "empty by a quadrature as coarse as itself (drag.empty-cell)",
            by="default_quadrature_refinement", measured=SEPARATED_REFINEMENT_MEASURED),
        "near_field": convention(
            NUMERICAL, "on the mrs rung, none: a medium's near field solved by one dense LU at every size, the "
                       "cross-kernel integrating every quadrature point of a source (each of NearField's settings states "
                       "its own)"),
        "lubrication": determined(
            NUMERICAL, "on the mrs rung, in a medium whose wall opens (a plane, a tube) about a body whose hull kind "
                       "has a support and whose least radius of curvature holds the hand-over gap within the film's "
                       "window (HANDOVER_SPACINGS of the force spacing, the written one or the family's, at most "
                       "WINDOW of that radius): on, lubrication's hand-over, the MRS's own completion in the gap (the "
                       "coarse MRS evaluated with the body's near contact opened to the hand-over gap, plus the thin "
                       "film's increment); a body past the window (a crest or an edge sharper than the spacing "
                       "resolves), False, the coarse MRS with its sampling, the build reporting the gap against the "
                       "radius; written False, the coarse MRS alone, a declared numerical choice whose validity line "
                       "is its near-wall error; on the surrogate rung, none unless Lubrication(epsilon=...) is "
                       "written (its blend)",
            by="default_lubrication"),
    }
    mrs_settings = (*UncoupledDrag.mrs_settings, "sampling", "near_field")
    #: The values ``grading`` takes beside none.
    gradings: ClassVar[tuple[str, ...]] = ("features",)
    raises = (
        *UncoupledDrag.raises,
        Flag("gain_exceeded", "the lagged coupling of the medium's near field reached a gain of one or more (the "
                              "per-step fixed point's contraction, a check sufficient for the lagged step's stability "
                              "and not necessary): the node replaced every delivery by NaN before the bodies took it, "
                              "so the run stopped there (drag.near-field-gain)",
             rung="mrs", group=True),
        Flag("cluster_gift", "the near field's own work on its bodies over the step before, as the energy ledger "
                             "books it from what the node delivered and the twists the bodies reached, was positive "
                             "past the ledger's floor: the fluid gave the cluster energy (an energy check sees an "
                             "error that makes the fluid a source, not one that leaves it dissipating)",
             rung="mrs", group=True),
        Flag("solve_flagged", "the near field's GMRES stopped other than converging (at its cap, stagnated, or at a "
                              "residual not finite): that step's resistances and loads are not the joint system's to "
                              "the solve's tolerance (the dense route, an LU, never raises it)",
             rung="mrs", group=True, warns=True),
        Flag("cross_cells_short", "a source cell within the cross-kernel's reach of the other body did not fit the "
                                  "cross-kernel's budget (NearField(share=...) set below one), which kept that cell's "
                                  "kernel at its force point, where the reach's conformance cases do not hold it",
             rung="mrs", group=True),
        Flag("overlap", "two bodies of the medium's near field overlapped: a point of one lay behind the other's "
                        "surface by more than a quarter of a cell, as their discretisations estimate it; the node "
                        "replaced every delivery by NaN before the bodies took it, so the run stopped there "
                        "(drag.near-field-overlap)", rung="mrs", group=True),
        Flag("near_contact", "two bodies of the medium's near field lay closer than a twentieth of the smaller one's "
                             "equivalent radius (CONTACT_GAP), the validity line's lower edge, where the MRS's own "
                             "error is not monotone in the spacing", rung="mrs", group=True),
        Flag("off_patch", "on a graded sampling (sampling written), the body lay within one force spacing "
                          "of the wall (NEAR_WALL) with a point of its quadrature outside the graded patch (where the "
                          "grading made the spacing finer than the force spacing) within OFF_PATCH_BAND force "
                          "spacings of its least distance: the sampling is fixed on the body, so a body that turned "
                          "carried the patch away from the wall, and there the resistance's error near the wall is not "
                          "the graded one's (on the gates' sphere it climbs towards the uniform sampling's as the "
                          "patch's edge nears the wall)",
             rung="mrs", group=False, written=("sampling",)),
        Flag("below_floor", f"the point body's coupled one-step margin (M0 + M_c - M_inf)/M0, the force's or the "
                            f"couple's, at the step's values was under the floor of {FLOOR:g}, below which its one-step "
                            f"resistance grows as one over the margin away from the exact sphere's (the build refuses a "
                            f"start below it: a study's values can move it)", rung="stokes"),
        Flag("history_flagged", f"with self_history='modes' (route 1), the fitted self-history (the body's own field, "
                                f"which the rate terms' reading removes) missed the free box's response by more than "
                                f"{HISTORY_LINE:g} of the one-step response over its fitted range, or the run went past "
                                f"the range's last step (its span, in the build report), where the fit is unjudged",
             rung="stokes"),
        Flag("rate_feedback", f"the Maxey-Riley rate terms' load reversed its sign and grew in {RINGING_STEPS} steps in "
                              f"a row: the lagged loop through what remains of the body's own field in the reading "
                              f"(a wall's reflection of it) has a gain above one, as for a light body at a small step; "
                              f"a larger step, a denser body or one_way=True are the routes", rung="stokes"),
        Flag("near_wall", f"the point body lay within {WALL_LINE:g} of its radii of the vessel's wall, where the point "
                          f"model reads -5.0 % at a = h and -8.9 % at a = 2h (the window spike, re-read): the MRS near "
                          f"field or a resolved body is the route", rung="stokes"),
        Flag("reynolds_flagged", f"the particle Reynolds number by the body's diameter and its speed relative to the "
                                 f"flow it reads was above {REYNOLDS_LINE:g}, the line up to which Stokes drag is "
                                 f"declared", rung="stokes"),
        Flag("handover_window", "with lubrication's hand-over, the hand-over gap lay past WINDOW of the contact's smaller "
                                "radius of curvature: the film's form was measured within about 7 % up to it, 15 to 19 "
                                "% at force spacings of 0.45 to 0.6 of the radius and about 30 % at 0.9 (a sphere and "
                                "a spheroid), so the near-wall resistance there is outside the measured reach",
             rung="mrs", group=False, written=("lubrication",)),
        Flag("film_annulus", "with lubrication's hand-over and the contact opened, the film's patch reached round the "
                             "lumen (a body nearly filling it): the contact is a ring, which the film's point form "
                             "does not cover, so the film there is the point contact's and the axial resistance is "
                             "not the ring's", rung="mrs", group=False, written=("lubrication",)),
        Flag("handover_far_side", "with lubrication's hand-over in a tube and the contact opened, the body's width "
                                  "across the lumen filled more than FAR_SIDE_FILL of its diameter: opening the near "
                                  "contact by dilating the tube opens the far side as well, which moves the coarse "
                                  "MRS's axial entry past 7 % (measured on a sphere at the deepest opening the window "
                                  "allows: -6.3 % at a/R 0.5, -8.9 % at 0.55, -18 % at 0.67, -37 % at 0.8)",
             rung="mrs", group=False, written=("lubrication",)),
        Flag("film_unsolved", "with lubrication's hand-over, the film's solve was not finite or its residual lay past "
                              "FILM_RESIDUAL_ULPS of the dtype's resolution (1e4 units in the last place: 2.2e-12 "
                              "relative in float64, 1.2e-3 in float32, so a float32 solve wrong by less than about a "
                              "tenth of a per cent passes unflagged): that step's film is not the thin film's to "
                              "rounding", rung="mrs", group=False, warns=True, written=("lubrication",)),
    )
    validity = (
        *(line for line in UncoupledDrag.validity if not line.startswith("each body's drag is computed alone")),
        "surrogate and rft: each body's drag is computed alone, from its own pose and the medium's wall: a second body "
        "in the same medium does not change it (the build reports the bodies that share a medium); on the mrs rung "
        "bodies in one medium interact through their joint near field, on the stokes rung through the solved flow "
        "(each rung's own lines)",
        f"stokes: a point sphere in a flow solved on a grid, coupled through the exchange law (a Gaussian kernel of "
        f"width sigma, at least max(h, a/1.7), widened on {list(WIDTHS)} cells until the coupled one-step margins, the "
        f"force's and the couple's, clear the floor {FLOOR:g} at the run's values: refused below it). The floor is the "
        f"one definiteness rule's for the affine (one-pass) scheme, whose one-step resistance was within -0.42..+0.61 "
        f"of the exact unsteady Stokes sphere's (drag, Basset and added mass) above it over r = nu dt/h^2 of 0.03 to 30 "
        f"[measured, free box, sigma = h and 1.5h; that scheme is a later PR's]. The lagged scheme this rung steps by "
        f"reads its own field a step late, so its first step resists R0 (1 + lambda^2/9) against the exact R0 (1 + "
        f"lambda + lambda^2/9), lambda = a/sqrt(nu dt): down to -0.60 at lambda = 3 [derived]; from rest under a step "
        f"force at bodies just above the floor its largest error was 0.06 to 0.28 of the terminal speed, 0.01 to 0.02 "
        f"at the end [measured by the rung's refuting review, a of h to 4h, r 0.03 to 3, near-massless and neutral]: for this "
        f"scheme the floor is forward-compatible, not an accuracy bound. The missing lambda is the Basset force, which "
        f"no memory law holds: the build reports a sqrt(omega/nu) of a drive's frequency against the line 0.3 (against "
        f"Stokes drag the history force scales as it), and sees no transient and no body driven by a field in still "
        f"fluid. The drag and the couple read the flow with the kernel's steady self-field removed (the grid's "
        f"undeveloped field acting as a history term); the fluid-acceleration force and the added mass read the flow "
        f"without the body, the body's own free-box flow removed exactly (route 2: advanced by the grid's operator, "
        f"Hasimoto's image term relaxed at the box's slowest rate), or by fitted modes with self_history='modes' "
        f"(route 1, its residual and span reported and flagged; it reads the body's own field as if the body had not "
        f"moved since each force, exact for a body at rest on the grid); under the field's steady limit the own field is the "
        f"steady response to last step's force, exactly, and the margins are one",
        "stokes: the couple: the fluid's rotation read by the kernel's antisymmetric dipole (its derivative, about the "
        "sampled weights' own mean, so a couple carries no net force and a unit couple is one) and the couple "
        "scattered back by its transpose, two-way, the body's rotation resisted by 8 pi mu a^3 (Faxén's second law); "
        "no torque from the fluid's rotational acceleration",
        f"stokes: near a wall the point model holds to {WALL_LINE:g} radius (flagged within it); the particle "
        f"Reynolds number by the slip is flagged above {REYNOLDS_LINE:g}; point bodies in one medium interact through "
        f"its solved flow only, each one's scattered reaction read by the others a step later (lagged: no joint step "
        f"for several bodies), resolved to the kernel's width, and where several share a medium the build reports each "
        f"one's nearest starting separation in radii and their volume fraction in the grid's box (reported, not "
        f"judged). Two bodies' interaction against Rotne and Prager's far field in a free box (a = h = sigma, 4 and 8 "
        f"radii apart, along and across their line of centres) was within 0.2 to 2.1 % of the cross term at 48 cells "
        f"and 0.2 to 1.2 % at 64 [measured by the rung's refuting review]; the exact two-sphere solution's nearer terms (about "
        f"15/(4 rho^4) along the line, 1.5 % of one body's speed at 4 radii) no point body carries",
        "mrs: two bodies or more in one medium are solved as one joint system (their near field), each body's steps "
        "lagged against the others' (a first-order error in the step, and a step's delay in the others' loads after a "
        "sudden change, both relaxing at the reported gain); the cross-kernel's reach is in each source cell's own "
        "size, its default from the sphere's data, and each registered hull kind's conformance case is its check; "
        "below a gap of about a twentieth of a radius the MRS's own error is not monotone in the spacing, lubrication's "
        "territory, not this rung's (a step there raises near_contact; two bodies that overlap stop the run, "
        "drag.near-field-overlap, and the build refuses a start where they do)",
        "mrs: the near field's lagged step is checked by its gain, the per-step fixed point's contraction, which is "
        "sufficient for the step's stability, not necessary: with inertia the step's amplification lies below the "
        "gain, so the check refuses some stable clusters (measured: three spheres a tenth of a radius apart, each "
        "body's relaxation time between about a twentieth and a tenth of the step), and the amplification and its "
        "relaxation are reported",
        "mrs: with sampling=ForceSampling(...) written, the force points are graded (opt-in): towards the hull kind's "
        "edges (EDGE_SHARE of the force spacing there) and crests (CREST_SHARE of their radius across) and towards a "
        "held contact (its gap to the wall over CONTACT_SPACINGS), the spacing growing by its distance from them and "
        "never finer than FINEST of the force spacing, each force point at its cell's area centre, chosen at the "
        "settings rounded to FIGURES. What it buys is measured per kind and bounded nowhere: a sphere's contact at a "
        "plane against Brenner's series (tests/oracles/test_graded_sampling.py), where it helps beyond its count (a "
        "uniform sampling of the graded count measured beside it); a cylinder's rim at a plane against a finer "
        "reference in the build's gates, nearer it than the uniform sampling, with no uniform sampling of the graded "
        "count measured, so whether the gain is the grading's or the count's is not known; a screw's crest at a plane "
        "against one, where a uniform sampling of the graded count did as well, so the gain was the count; a box's "
        "edges, an ellipsoid's and a capsule's surfaces not measured; at the bodies, blobs and gaps those name. The "
        "sampling is fixed on the body: a graded contact serves a body whose orientation to the wall is held (a crest "
        "or a rim is the contact whichever way the body rolls about its axis), and a step with a point within "
        "OFF_PATCH_BAND of its least distance outside the patch near the wall raises off_patch. With wall_refinement "
        "written, the wall's self-image measures a graded cell's distance in the force spacing, as a uniform run's "
        "(GRADED_WALL_MEASURE): on the gates' sphere graded for half a force spacing from a plane, "
        "farther from Brenner's series than the uniform sampling at a blob of a tenth, the compact default, and nearer "
        "at a fortieth (GRADED_WALL_MEASURE's comment gives the numbers); a graded body's wall budget follows its "
        "density, every cell of its graded patch and WALL_CELL_SHARE of the rest, so the gates' sphere and the graded "
        "round-trip cylinder raise wall_cells_short on no step (it is kept for a real shortfall), at the cost "
        "WALL_BUDGET_MEASURED gives",
        "mrs: the near field's cross loads are judged by the energy ledger's first route on the cluster's books, which "
        "sees an error that makes the fluid a source and none that leaves it dissipating (an error in the grand "
        "resistance that leaves it definite, a cross load at half its value or with its sign flipped); those are "
        "the force oracles' (the two-sphere series, Rotne and Prager's mobility, the dense assembly)",
        f"mrs: the near field's memory, on either route: the run's peak past the process's fixed part, so many of its "
        f"joint matrices and of its members' held blocks ({MEMORY_MEASURED}), and a gradient study's derivative's, "
        f"forward or reverse mode ({DERIVATIVE_MEASURED}), reported at build and refused (drag.near-field-memory), a "
        f"run at build and a derivative before the study compiles it, where {MEMORY_MARGIN:g} times the estimate "
        f"(MEMORY_MARGIN, the measurements' largest miss) passes the memory a run can have less {MEMORY_HEADROOM:.0%} "
        f"of it (MEMORY_HEADROOM): a GPU's limit, or on a CPU the smaller of the host's available memory and the limit "
        f"of the control group the process runs in, where one is set. Measured on CPUs, not a bound; a GPU's is not "
        f"measured",
        "mrs: with lubrication (on by default near a wall that opens), the near-wall resistance is the coarse MRS "
        "evaluated once with the body's near contact opened to the hand-over gap (HANDOVER_SPACINGS of the force "
        "spacing, the finest graded spacing where graded), never closer, plus the thin film's increment between the "
        "contact's gap and the opened one (Reynolds' equation on the gap's second-order form): the MRS's own "
        "completion in the gap, not physics beyond Stokes flow. It is taken only where the hand-over gap lies within "
        "WINDOW of the body's least radius of curvature (the build's rule: a body past it, a crest or an edge sharper "
        "than the spacing resolves, is the coarse MRS's, the build reporting the gap against the radius, and "
        "lubrication=True written there is refused, drag.lubrication-window). Within the window, a sphere above a "
        "plane against exact answers: every entry within 7 % on its own diagonal scale at force spacings of 0.15 and "
        "0.2 radii, but the normal at 0.06 radii, -6.8 to -7.3 % over orientations at 0.2; the coupling of rotation "
        "and translation judged against its own size is over-counted below the hand-over gap (MIME-ANO-154: a sphere "
        "spun on the wall translates up to 43 % too fast per rotation at 0.2 radii, the excess falling as one over "
        "the logarithm of the gap, 25 % at a thousandth of the radius). The coarse MRS's share at the hand-over gap, "
        "a sphere above a plane with the image at the force points: 5.6 to 6.1 % at blob 0.1, 3.5 to 4.2 % at 0.05 "
        "(the default near a wall with lubrication), 2.6 to 3.4 % at 0.025. In a tube the contact is opened by "
        "dilating the lumen, which opens its far side as well: past FAR_SIDE_FILL of the diameter the step raises "
        "handover_far_side",
        "mrs: with lubrication, below the composite roughness of the body's and the wall's surfaces (each a declared "
        "length, DEFAULT_ROUGHNESS where none is declared, never fitted) the film's gap is floored: its resistance "
        "stops growing there and a contact carries the gap; the near-wall resistance owns every hydrodynamic coupling "
        "of rotation and translation down to that cutoff (film_coupling reports the film's own share), a contact "
        "none. One point contact a body (its deepest point against the wall); a lumen the body nearly fills is a ring "
        "contact the point form does not cover (film_annulus); several contacts of one body, a ring's film, and a "
        "body between two walls (no two-wall kind) are later items",
    )

    lubrication: Lubrication | bool | None = None  # pyright: ignore[reportIncompatibleVariableOverride] - the mrs rung's hand-over is a switch, the surrogate's blend a group
    sampling: ForceSampling | None = None
    near_field: NearField | None = None
    one_way: bool | None = None
    kernel_width: float | None = None
    self_history: str | None = None

    @classmethod
    def Stokes(cls, **settings: Any) -> Self:  # noqa: N802 - a rung's alias reads as a class
        """A point sphere in a flow solved on a grid, two-way (``one_way=True`` the declared approximation):
        ``Drag(rung="stokes", ...)``."""
        return cls(rung="stokes", **settings)

    @override
    def unrecorded_settings(self) -> frozenset[str]:
        """None: 0.5.0 records every setting (the class's docstring says why)."""
        return frozenset()

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = super().setting_problems(settings) if self.rung != "stokes" else []
        written = cast(object, self.lubrication)
        if self.rung == "mrs" and written is not None and not isinstance(written, bool):
            found.append(f"lubrication={written!r} on the mrs rung: its lubrication is the hand-over to the thin film, "
                         f"on by default near a wall that opens, so write lubrication=False to leave it out (a "
                         f"Lubrication(epsilon=...) blend is the surrogate rung's)")
        if self.rung != "mrs" and isinstance(written, bool):
            found.append(f"lubrication={written!r} on the {self.rung} rung: True or False is the mrs rung's hand-over; "
                         f"the surrogate's lubrication is its blend, lubrication=Lubrication(epsilon=mm(0.15))")
        if self.rung == "rft" and self.near_field is not None:
            found.append("near_field= is a setting of the mrs rung's near field, and the rft rung reads none")
        if self.rung != "mrs" and self.sampling is not None:
            found.append(f"sampling= is a setting of the mrs rung's force points, and the {self.rung} rung reads none")
        if self.rung != "stokes":
            found += [f"{name}= is a setting of the stokes rung's exchange with a grid, and the {self.rung} rung "
                      f"reads none" for name in ("one_way", "kernel_width", "self_history")
                      if getattr(self, name) is not None]
        else:
            found += [f"{name}= is a setting of another rung, and the stokes rung reads none"
                      for name in ("weights", "lubrication", "spacing", "blob", "patches", "wall_kind",
                                   "coefficients", "quadrature_refinement", "wall_refinement", "near_field")
                      if getattr(self, name) is not None]
            written_way = cast(object, self.one_way)
            if written_way is not None and not isinstance(written_way, bool):
                found.append(f"one_way={written_way!r} is a switch: write one_way=True for the declared one-way "
                             f"approximation, or leave it out for the two-way exchange")
            width = cast(object, self.kernel_width)
            if width is not None and (isinstance(width, bool) or not isinstance(width, int | float)
                                      or not 1.0 <= float(width) <= self.WIDTHS[-1]):
                found.append(f"kernel_width={width!r} is the exchange kernel's width in the grid's cells, a number from "
                             f"1 to {self.WIDTHS[-1]:g}: write kernel_width=1.5, or leave it out for the narrowest that "
                             f"clears the floor")
            route = cast(object, self.self_history)
            if route is not None and route not in ROUTES:
                found.append(f"self_history={route!r} is one of {list(ROUTES)}: leave it out for the exact route, or "
                             f"write self_history='modes' for the fitted one")
        return found

    def _sampled(self) -> ForceSampling:
        """The force points' sampling as written, or the uniform one's (nothing graded)."""
        held = cast(object, self.sampling)
        return held if isinstance(held, ForceSampling) else ForceSampling()

    def _near(self) -> NearField:
        """The near field's settings as written, or the defaults (the dense route, every point a source)."""
        held = cast(object, self.near_field)
        return held if isinstance(held, NearField) else NearField()

    def _stokes_plan(self, ctx: PlanContext) -> Plan:
        """The stokes rung's plan: the body's pose and twist and the medium's face sets read, the drag, the couple and
        their Maxey-Riley part delivered to the body, the scatters to the medium's exchange (unless one-way), the body
        registered with the medium's field as an exchange partner."""
        problems = self._stokes_problems(ctx)
        delivers = {"resistance": Delivery("body", RESISTANCE.name),
                    "background_load": Delivery("body", BACKGROUND_LOAD.name),
                    "added_resistance": Delivery("body", RESISTANCE.name),
                    "added_load": Delivery("body", BACKGROUND_LOAD.name)}
        if not self.one_way:
            delivers.update({f"exchange_{axis}": Delivery("medium", interface.name) for axis, interface in
                             zip("xyz", (FLOW_EXCHANGE_X, FLOW_EXCHANGE_Y, FLOW_EXCHANGE_Z), strict=True)})
        return Plan(reads={"pose": Read("body", POSE.name), "twist": Read("body", TWIST.name),
                           "faces": Read("medium", FLOW_FACES.name)},
                    delivers=delivers, values={"viscosity": Value("medium", "viscosity"),
                                               "density": Value("medium", "density")},
                    registrations=(Registration("medium", FLOW_PARTNERS.name, EXCHANGE_PARTNER, "body"),),
                    problems=tuple(problems))

    def _stokes_problems(self, ctx: PlanContext) -> list[Problem]:
        """A body whose hull is not one sphere at its centre (a medium whose flow takes no exchange is refused by the
        rung's support row, an open vessel by its flow)."""
        body = ctx.bindings["body"]
        found: list[Problem] = []
        hull = ctx.geometry(of="body")
        centred = hull is not None and math.dist(hull.centre, (0.0, 0.0, 0.0)) <= 1e-12 * max(
            hull.kind.enclosing_radius(hull.settings), 1e-300)
        if hull is None or hull.kind.name != "sphere" or not centred:
            held = f"{body} has no hull" if hull is None else f"its hull is {hull.label()}"
            found.append(Problem("drag.stokes-sphere", f"the stokes rung is a point sphere's law, and {held}",
                                 f"give {body} one Sphere part at its origin, or use Drag.MRS(...) for its shape"))
        return found

    @staticmethod
    def stokes_scheme(frame: ExchangeFrame, values: Mapping[str, Magnitude], dt: float) -> str:
        """The scheme the medium's flow steps by at these values (its written one, else the FVM's choice)."""
        from mime.dynamics.flow._fvm import chosen_scheme

        if frame.scheme is not None:
            return frame.scheme
        viscosity, density = float(cast(float, values["viscosity"])), float(cast(float, values["density"]))
        return chosen_scheme(None, viscosity / density, frame.spacing, dt)

    @override
    def value_problems(self, values: Mapping[str, Magnitude], ctx: PlanContext) -> list[Problem]:
        """The stokes rung at the run's values: a flow stepped by RK3; a body the width rule puts past the widest kernel
        (``drag.stokes-size``); a body whose coupled one-step margins stay under the floor at every allowed width
        (``drag.stokes-floor``), with the remedies that raise them, measured at these values."""
        if self.rung != "stokes" or "medium" not in ctx.exchanges or ctx.geometry(of="body") is None:
            return []
        frame = ctx.exchanges["medium"]
        scheme = self.stokes_scheme(frame, values, ctx.dt)
        if scheme not in ("imex_euler", "steady"):
            return [Problem("drag.stokes-scheme",
                            f"medium {ctx.bindings['medium']!r}'s flow steps with RK3 at this step, whose explicit "
                            f"stages take no exchange deliveries yet",
                            "name the implicit scheme, FVM(scheme=Scheme.IMEXEuler()), or its steady limit with "
                            "stokes=True")]
        chosen, margin, tried = self.stokes_width(ctx, values)
        if chosen is not None:
            return []
        body = ctx.bindings["body"]
        hull = ctx.geometry(of="body")
        assert hull is not None
        radius = float(hull.kind.enclosing_radius(hull.settings))
        h, widest = frame.spacing, self.WIDTHS[-1]
        across = self.lumen_across(ctx)
        cells = self.clearing_cells(ctx, values, radius, across) if across is not None else None
        named = str(cells) if cells is not None else "..."
        note = " plans it at this step" if cells is not None else " (none down to five cells across clears it here)"
        if self.kernel_width is None and radius / (1.7 * h) > widest + 1e-12:
            return [Problem("drag.stokes-size",
                            f"{body}'s radius is {radius / h:.3g} cells, past the {1.7 * widest:g} cells the widest "
                            f"kernel ({widest:g} cells) admits under the rule sigma >= a/1.7: no kernel on the ladder "
                            f"is wide enough for it (a finer grid makes it larger in cells, and a larger step changes "
                            f"nothing of it)",
                            f"a coarser grid, FVM(cells_across={named}){note}; or a resolved body (Drag.Penalty) or "
                            f"the MRS near field (Drag.MRS) for a body this large against the grid")]
        larger = self.clearing_step(ctx, values, radius, tried[-1])
        step = f"{larger:.3g} s clears it at {tried[-1]:g} cells" if larger is not None else \
            f"... (no step up to {self.STEP_FACTORS[-1]:g} times this one clears it)"
        return [Problem("drag.stokes-floor",
                        f"{body}'s coupled one-step margin (M0 + M_c - M_inf)/M0 stays under the floor {FLOOR:g} at "
                        f"every kernel width tried ({', '.join(f'{w:g}' for w in tried)} cells: {margin:.3g} at the "
                        f"widest): below it the point body's one-step resistance grows as one over the margin away "
                        f"from the exact sphere's",
                        f"a larger step, dt={step} (a larger step raises the margin, a finer grid lowers it); a coarser "
                        f"grid, FVM(cells_across={named}){note}; or a resolved body (Drag.Penalty) or the MRS near "
                        f"field (Drag.MRS)")]

    #: The multiples of the run's step the ``drag.stokes-floor`` refusal tries for the step it names.
    STEP_FACTORS: ClassVar[tuple[float, ...]] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)

    @staticmethod
    def lumen_across(ctx: PlanContext) -> float | None:
        """The lumen's diameter of the medium's tube (m), where it fills one."""
        region = ctx.region("medium")
        held = region.parameters.get("radius") if region is not None else None
        return 2.0 * float(held) if isinstance(held, int | float) else None

    def clearing_cells(self, ctx: PlanContext, values: Mapping[str, Magnitude], radius: float,
                       across: float) -> int | None:
        """The most cells across the lumen, fewer than the grid's, at which a width on the ladder clears the floor at
        the run's step (the coarser grid a refusal names), or ``None`` down to five."""
        viscosity, density = float(cast(float, values["viscosity"])), float(cast(float, values["density"]))
        steady = self.stokes_scheme(ctx.exchanges["medium"], values, ctx.dt) == "steady"
        cells = math.floor(across / ctx.exchanges["medium"].spacing + 1e-9) - 1
        while cells >= 5:
            h = across / cells
            least = max(1.0, radius / (1.7 * h))
            for width in (w for w in self.WIDTHS if w >= least - 1e-12):
                if min(free_space_margins(radius, viscosity, density, h, width, ctx.dt, steady)) >= FLOOR:
                    return cells
            cells = min(cells - 1, math.floor(cells * 0.85))
        return None

    def clearing_step(self, ctx: PlanContext, values: Mapping[str, Magnitude], radius: float,
                      width: float) -> float | None:
        """The smallest multiple of the run's step (:data:`STEP_FACTORS`) at which the kernel of ``width`` cells clears
        the floor, or ``None``."""
        frame = ctx.exchanges["medium"]
        viscosity, density = float(cast(float, values["viscosity"])), float(cast(float, values["density"]))
        for factor in self.STEP_FACTORS:
            if min(free_space_margins(radius, viscosity, density, frame.spacing, width, ctx.dt * factor)) >= FLOOR:
                return ctx.dt * factor
        return None

    @override
    def judged_in_sum(self) -> str:
        """The stokes rung's drag: the flow carries the body (the drag's work on the body is positive there) and the
        fluid's side pays for it (two-way) or the field supplies it (one way), so no one receiver's row is judged a
        gift; its node judges the law's power at one velocity each step instead (``cluster_gift_steps``)."""
        if self.rung != "stokes":
            return ""
        return ("its sign judged by its node each step, not on this row (the flow carries the body): the law's power "
                "at one velocity, its load on the body at the body's solved twist and its reaction against the reading "
                "it was solved with, never above its rounding floor (cluster_gift_steps)")

    def stokes_width(self, ctx: PlanContext | BuildContext, values: Mapping[str, Magnitude]) -> tuple[
            float | None, float, tuple[float, ...]]:
        """The kernel's width in cells the stokes rung takes: the written one, or the narrowest on :data:`WIDTHS` at
        or above max(1, a/(1.7 h)) whose free-space coupled one-step margins (the force's and the couple's) clear the
        floor at ``values``; with the least margin read there and the widths tried. ``None`` where none clears it."""
        frame = ctx.exchanges["medium"]
        hull = ctx.geometry(of="body")
        assert hull is not None
        radius = float(hull.kind.enclosing_radius(hull.settings))
        viscosity, density = float(cast(float, values["viscosity"])), float(cast(float, values["density"]))
        dt = ctx.dt if isinstance(ctx, PlanContext) else ctx.timestep
        steady = self.stokes_scheme(frame, values, dt) == "steady"
        least = max(1.0, radius / (1.7 * frame.spacing))
        ladder = (float(self.kernel_width),) if self.kernel_width is not None else tuple(
            w for w in self.WIDTHS if w >= least - 1e-12) or (self.WIDTHS[-1],)
        margin = float("nan")
        tried: list[float] = []
        for width in ladder:
            tried.append(width)
            margin = min(free_space_margins(radius, viscosity, density, frame.spacing, width, dt, steady))
            if margin >= FLOOR and width >= least - 1e-12:
                return width, margin, tuple(tried)
        return None, margin, tuple(tried)

    def default_contact_gap(self, ctx: PlanContext) -> Quantity | None:
        """The gap a written contact is graded for when none is written: the experiment's closest approach to the
        wall (``PlanContext.approach``), where a law holds the body off it by a margin above zero."""
        approach = ctx.approach(of="body")
        if self.rung != "mrs" or self._sampled().contact is None or approach is None or approach <= 0.0:
            return None
        return m(approach)

    def graded(self) -> bool:
        """Whether the force points are graded: the sampling's ``grading`` or ``contact`` written."""
        sampled = self._sampled()
        return self.rung == "mrs" and (sampled.grading is not None or sampled.contact is not None)

    @override
    def default_quadrature_refinement(self, ctx: PlanContext) -> int | None:
        """0.4.0's rule, and where graded at least ``QUADRATURE_PER_BLOB`` over ``FINEST``: the quadrature's spacing at
        most half the finest graded cell's."""
        found = super().default_quadrature_refinement(ctx)
        if found is None or not self.graded():
            return found
        return max(found, math.ceil(QUADRATURE_PER_BLOB / FINEST - 1e-9))

    surrogate_settings = ("weights",)

    def default_lubrication(self, ctx: PlanContext) -> bool | None:
        """On the mrs rung, on where the medium's wall opens (``WallCorrection.opened``: a plane, a tube) about a body
        whose hull kind has a support (its curvature at the contact), and whose least radius of curvature holds the
        hand-over gap within the film's window (:meth:`handover_window`); False past the window, the build's rule (a
        contact the film does not cover is the coarse MRS's); none elsewhere (a hull whose window cannot be judged
        among them), and on the surrogate rung."""
        if self.rung != "mrs" or ctx.region("medium") is None:
            return None
        hull = ctx.geometry(of="body")
        written = cast(object, self.wall_kind) if self.wall_kind is not None else self.default_wall_kind(ctx)
        wall = _wall_kind(written)
        if hull is None or isinstance(wall, str) or not implements(hull.kind, "support"):
            return None
        correction = wall_correction_named(wall.name, wall.version)
        if correction.opened is None or correction.region is None:
            return None
        window = self.handover_window(ctx)
        return None if window is None else window[0] <= WINDOW * window[1]

    def handover_window(self, ctx: PlanContext) -> tuple[float, float] | None:
        """The hand-over gap and the body's least radius of curvature (m) the film's window is judged on at the plan:
        ``HANDOVER_SPACINGS`` of the force spacing the hand-over takes (the written one, or the family's; where
        graded, ``FINEST`` of it, the finest a graded sampling holds), against the least radius over the body's
        samples (``least_radius``): any contact the body can make, whatever its pose. ``None`` where the hull gives
        no sampling."""
        hull = ctx.geometry(of="body")
        if hull is None or not implements(hull.kind, "sampling") or not implements(hull.kind, "support"):
            return None
        written = ctx.settings.get("spacing")
        family = PointForceDrag.default_spacing(self, ctx)
        spacing = float(written) if isinstance(written, float) else \
            None if family is None else float(cast(float, family.value))
        if spacing is None:
            return None
        return self._window_at(hull, spacing)

    def _window_at(self, hull: Hull, spacing: float) -> tuple[float, float]:
        patches = self.patches if self.patches is not None else tuple(hull.kind.patches)
        share = FINEST if self.graded() else 1.0
        return HANDOVER_SPACINGS * spacing * share, least_radius(hull.kind, hull.settings, spacing, patches)

    def lubricated(self, ctx: PlanContext) -> bool:
        """Whether the mrs rung hands its near contact over to the film: ``lubrication`` written True, or not
        written and its default on."""
        if self.rung != "mrs":
            return False
        written = cast(object, self.lubrication)
        return written is True if written is not None else self.default_lubrication(ctx) is True

    @override
    def includes_lubrication(self) -> bool:
        """The mrs rung with its hand-over includes the near-wall film, as the surrogate with its blend does."""
        if self.rung == "mrs":
            return self.lubrication is True
        return super().includes_lubrication()

    @override
    def default_blob(self, ctx: PlanContext) -> float | None:
        """0.4.0's, and with lubrication a compact body's :data:`LUBRICATED_BLOB` (the hand-over gap's measured
        share of the coarse MRS's error falls with the blob)."""
        found = super().default_blob(ctx)
        hull = ctx.geometry(of="body")
        if found is None or hull is None or hull.kind.name in SEPARATED_BLOB and hull.kind.name != "sphere" \
                or not self.lubricated(ctx):
            return found
        return min(found, LUBRICATED_BLOB)

    @override
    def default_spacing(self, ctx: PlanContext) -> Quantity | None:
        """0.4.0's, but with lubrication the family's whatever the closest approach: the hand-over gap, not a
        contact's margin, keeps the coarse MRS off the wall, so the margin no longer sets the force spacing."""
        if self.lubricated(ctx):
            return PointForceDrag.default_spacing(self, ctx)
        return super().default_spacing(ctx)

    @override
    def _handover_arguments(self, ctx: BuildContext, spacing: float) -> dict[str, Any]:
        """With lubrication on: the hand-over at the local force ``spacing`` and each surface's declared roughness
        (``DEFAULT_ROUGHNESS`` where none is declared). Off where the film's window leaves it out, the build report's
        reason: the hand-over gap at the force spacing the hand-over would take (the written one or the family's, the
        coarser of this build's and the family's) against the body's least radius of curvature."""
        if self.rung != "mrs":
            return {}
        if self.lubrication is not True:
            hull = ctx.geometry(of="body")
            if self.lubrication is not False or hull is None or not implements(hull.kind, "support") \
                    or not implements(hull.kind, "sampling") or ctx.region("medium") is None:
                return {}
            handover, radius = self._window_at(hull, max(ctx.scalar("spacing"), self._family_spacing(hull)))
            return {"handover_declined": window_text(handover, radius)} if handover > WINDOW * radius else {}
        return {"lubrication": True, "handover_spacing": float(spacing),
                "roughness_body": ctx.values.get("roughness_body"), "roughness_wall": ctx.values.get("roughness_wall")}

    def joins_near_field(self) -> bool:
        """Whether this drag, with others of its version in its medium, is built as their joint near field: the mrs
        rung's."""
        return self.rung == "mrs"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        if ctx.members:
            self._build_near_field(ctx, b)
            return
        if self.rung == "stokes":
            self._build_stokes(ctx, b)
            return
        super().build(ctx, b)

    def _build_stokes(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        """The point body's node, its reads of the body, the exchange law's gathers (three a face set: the kernel's
        average, its Laplacian and its dipole's rotation) and scatters (the drag's reaction, its Faxén part and the
        couple's), all at the point it holds."""
        from mime.mappings import gaussian_mac

        frame = ctx.exchanges["medium"]
        hull = ctx.geometry(of="body")
        assert hull is not None
        values = {"viscosity": ctx.scalar_value("viscosity"), "density": ctx.scalar_value("density")}
        width, _, _ = self.stokes_width(ctx, values)
        assert width is not None, "phase 1 refuses a body under the floor at every width"
        region = ctx.region("medium")
        tube = uniform_tube(region, ctx.bindings["medium"]) if region is not None else None
        vessel = {} if not isinstance(tube, UniformTube) else {
            "vessel_centre": tube.centre, "vessel_axis": tube.axis, "vessel_radius": tube.radius}
        start = ctx.starts.get("body", {}).get("position", (0.0, 0.0, 0.0))
        node = b.add_response_node(PointDragNode(
            ctx.node_name(), ctx.timestep, radius=float(hull.kind.enclosing_radius(hull.settings)),
            viscosity=values["viscosity"], density=values["density"], spacing=frame.spacing, width_cells=width,
            origin=tuple(-c for c in frame.origin), initial_points=cast(tuple[float, ...], start),
            one_way=bool(self.one_way), scheme=self.stokes_scheme(frame, values, ctx.timestep),
            self_history=self.self_history or ROUTES[0], flow_cells=math.prod(frame.cells),
            drive_frequency=ctx.frequencies.get("medium", 0.0), **cast(dict[str, Any], vessel)))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(b.read("twist"), into=node["twist"])

        def kernel(component: int, mode: str, direction: str) -> object:
            return gaussian_mac(nx=frame.cells[0], ny=frame.cells[1], nz=frame.cells[2], spacing=frame.spacing,
                                width_cells=width, component=component, mode=mode, direction=direction, count=1)

        for component, axis in enumerate("xyz"):
            b.gather("faces", frame.faces[component], into=node[AVERAGES[component]],
                     mapping=kernel(component, "average", "gather"), points="points")
            b.gather("faces", frame.faces[component], into=node[LAPLACIANS[component]],
                     mapping=kernel(component, "laplacian", "gather"), points="points", units="1/(m s)")
            b.gather("faces", frame.faces[component], into=node[ROTATIONS[component]],
                     mapping=kernel(component, "curl", "gather"), points="points", units="rad/s")
            if not self.one_way:
                b.scatter(node[FORCES[component]], into=f"exchange_{axis}",
                          mapping=kernel(component, "average", "scatter"), points="points")
                b.scatter(node[FAXEN_FORCES[component]], into=f"exchange_{axis}",
                          mapping=kernel(component, "laplacian", "scatter"), points="points")
                b.scatter(node[COUPLE], into=f"exchange_{axis}", mapping=kernel(component, "curl", "scatter"),
                          points="points")
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))
        b.connect(node["added_resistance"], into=b.target("added_resistance"), part="maxey_riley")
        b.connect(node["added_load"], into=b.target("added_load"), part="maxey_riley")

    @override
    def _mrs_setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = super()._mrs_setting_problems(settings)
        for name, group in (("sampling", ForceSampling), ("near_field", NearField)):
            held = cast(object, getattr(self, name))
            if held is not None and not isinstance(held, group):
                found.append(f"{name}={held!r} is not a {group.__name__}: write {name}={group.__name__}(...)")
        near = self._near()
        share = cast(object, near.share)
        if share is not None and (isinstance(share, bool) or not isinstance(share, int | float)
                                  or not 0.0 < float(share) <= 1.0):
            found.append(f"the near field's share is {share!r}, not a share above 0 and at most 1: the most of a "
                         f"source's quadrature points its cross-kernel integrates, so write NearField(share=0.25) (or "
                         f"leave it out for every point, 1)")
        solve = cast(object, near.solve)
        if solve not in ROUTES:
            found.append(f"the near field's solve is {solve!r}, not one of {ROUTES}: 'dense', one dense LU at every "
                         f"size (the default), or 'gmres', NearField(solve='gmres')")
        tolerance = cast(object, near.tolerance)
        if tolerance is not None and (isinstance(tolerance, bool) or not isinstance(tolerance, int | float)
                                      or not 0.0 < float(tolerance) < 1.0):
            found.append(f"the near field's tolerance is {tolerance!r}, not a number between 0 and 1: GMRES's "
                         f"relative tolerance on each solve's true residual, so write NearField(solve='gmres', "
                         f"tolerance=1e-10) (float64's default; float32's is 1e-5)")
        elif tolerance is not None and solve != "gmres":
            found.append(f"the near field's tolerance is GMRES's, and its solve is {solve!r}, which reads none: write "
                         f"NearField(solve='gmres', tolerance={tolerance!r}), or leave the tolerance out")
        sampled = self._sampled()
        grading = cast(object, sampled.grading)
        if grading is not None and grading not in self.gradings:
            found.append(f"the sampling's grading is {grading!r}, not one of {self.gradings}: "
                         f"ForceSampling(grading='features') grades the force points towards the hull kind's edges and "
                         f"crests (leave it out for the uniform sampling)")
        contact = cast(object, sampled.contact)
        if contact is not None and not _direction(contact):
            found.append(f"the sampling's contact is {contact!r}, not a direction: three numbers, not all zero, in the "
                         f"body's frame, the side of the body held facing the wall, so write "
                         f"ForceSampling(contact=(0.0, 0.0, -1.0))")
        gap = settings.get("sampling.contact_gap")
        if gap is not None and not (isinstance(gap, float) and math.isfinite(gap) and gap > 0.0):
            found.append(f"the sampling's contact_gap is {gap!r} m, not a length above zero: the gap to the wall a "
                         f"contact is graded for, so write ForceSampling(contact=..., contact_gap=mm(0.1))")
        if gap is not None and contact is None:
            found.append("the sampling's contact_gap is the gap a written contact is graded for, and no contact is "
                         "written")
        return found

    @override
    def _mrs_problems(self, ctx: PlanContext) -> list[Problem]:
        """0.4.0's refusals, and a grading the body or the experiment cannot take: features asked of a kind that
        declares none (``drag.grading-features``), or of a kept version, whose sampling is its quadrature
        (``drag.grading-kind``); a contact with no wall (``drag.grading-contact``), no closest approach to grade it
        for (``drag.grading-approach``), or a gap so wide that it grades nothing (``drag.grading-gap``)."""
        found = super()._mrs_problems(ctx)
        sampled = self._sampled()
        if self.lubricated(ctx):
            found = [problem for problem in found if problem.code != "drag.mrs-on-wall"]
            window = self.handover_window(ctx) if cast(object, self.lubrication) is True else None
            if window is not None and window[0] > WINDOW * window[1]:
                body = ctx.bindings["body"]
                found.append(Problem(
                    "drag.lubrication-window",
                    f"lubrication=True asks the thin film to carry {body}'s gaps below the hand-over gap, and "
                    f"{window_text(*window)}, past the film's window ({WINDOW:g} of it): the film's form is measured "
                    f"within about 7 % inside the window and about 30 % at 4.5 times it, and a contact the film does not "
                    f"cover is the coarse MRS's, with its graded sampling at a sharp edge",
                    f"leave lubrication unwritten, which gives {body}'s gaps to the coarse MRS and reports why, or "
                    f"write a force spacing at most {WINDOW / HANDOVER_SPACINGS:.3g} of that radius, "
                    f"Drag.MRS(spacing=mm({window[1] * WINDOW / HANDOVER_SPACINGS * 1e3:.3g}))"))
        if not self.graded():
            return found
        body = ctx.bindings["body"]
        hull = ctx.geometry(of="body")
        if hull is None:
            return found
        if (hull.kind.name, hull.kind.version) in KEPT:
            found.append(Problem("drag.grading-kind", f"{hull.label()} is a kept version, whose sampling is its "
                                                      f"quadrature, so it cannot be graded",
                                 f"give {body} the kind's current version (a kind's default), Part(Screw(...)), or "
                                 f"leave the sampling out, Drag.MRS(spacing=...)"))
        elif sampled.grading == "features" and not implements(hull.kind, "features"):
            declaring = sorted({record.name for record in registered(EntryKind.GEOMETRY)
                                if issubclass(record.cls, Geometry) and implements(record.cls, "features")})
            found.append(Problem("drag.grading-features", f"grading='features' grades the force points towards the "
                                                          f"hull kind's features, and {hull.label()} declares none",
                                 f"leave grading out for this kind (the kinds that declare features: "
                                 f"{', '.join(declaring)}), or grade a held contact, "
                                 f"ForceSampling(contact=(0.0, 0.0, -1.0))"))
        if sampled.contact is None:
            return found
        if ctx.region("medium") is None:
            found.append(Problem("drag.grading-contact", f"the sampling's contact grades {body}'s side held facing the "
                                                         f"wall, and its medium fills no region, so it has no wall",
                                 "leave contact out in free space, Drag.MRS(sampling=ForceSampling(grading="
                                 "'features')), or fill the medium's region with a wall, Medium('water', "
                                 "fills=floor.half_space)"))
        elif ctx.settings.get("sampling.contact_gap") is None:
            found.append(Problem("drag.grading-approach",
                                 f"the sampling's contact grades {body}'s side held facing the wall for a gap to it, "
                                 f"and none is written (contact_gap=) and nothing holds {body} off the wall by a "
                                 f"margin above 0, whose closest approach it would take",
                                 f"write the gap to grade for, Drag.MRS(sampling=ForceSampling(contact=..., "
                                 f"contact_gap=mm(0.1))), or hold {body} off the wall, "
                                 f"Contact.Projection(margin=mm(0.1))"))
        else:
            gap, spacing = ctx.settings.get("sampling.contact_gap"), ctx.settings.get("spacing")
            if isinstance(gap, float) and isinstance(spacing, float) and gap >= CONTACT_SPACINGS * spacing:
                found.append(Problem(
                    "drag.grading-gap", f"the sampling's contact grades {body}'s side held facing the wall for a gap "
                                        f"of {gap * 1e3:.4g} mm, where {CONTACT_SPACINGS:g} local spacings from the "
                                        f"wall ask a spacing of {gap / CONTACT_SPACINGS * 1e3:.4g} mm, no finer than "
                                        f"the force spacing ({spacing * 1e3:.4g} mm): the contact grades nothing",
                    f"leave contact out at this gap (the uniform sampling is as fine there), or grade for a gap below "
                    f"{CONTACT_SPACINGS:g} force spacings, ForceSampling(contact=..., "
                    f"contact_gap=mm({CONTACT_SPACINGS * spacing * 1e3 / 2:.3g}))"))
        return found

    @override
    def _sampled_hull(self, owner: str, hull: Hull, spacing: float, settings: Mapping[str, Magnitude]) -> Hull:
        """As 0.4.0's, and where graded the range's point with the most force points at the grading's own density
        (:func:`graded_force_points`), sampled in every ranged number (:data:`GRADED_FORCE_POINTS`), its numbers
        rounded to the figures a graded sampling is chosen at (``mime.scene._grading.FIGURES``)."""
        if not self.graded():
            return super()._sampled_hull(owner, hull, spacing, settings)
        patches = self.patches if self.patches is not None else tuple(hull.kind.patches)
        at = figures(spacing)

        def rounded(held: Hull) -> Hull:
            return held.at(chosen_at(held.settings))

        return rounded(choose_over(owner, hull, "force points", GRADED_FORCE_POINTS,
                                   lambda held: graded_force_points(rounded(held), at, patches,
                                                                    self._density(settings, rounded(held), at)),
                                   cost=_solve_cost))

    @override
    def _mrs_sampling(self, ctx: BuildContext, hull: Hull, spacing: float, patches: tuple[str, ...]) -> tuple[
            Sampling, float, GradedDensity | None]:
        """Where graded (``graded()``), the kind's sampling at the policy's density (``mime.scene._grading``), chosen
        at the spacing rounded as the hull's numbers are (``FIGURES``), its wall quadrature refined from its finest
        spacing, and the density."""
        at = figures(spacing)
        density = self._density(ctx.settings, hull, at)
        if density is None:
            return super()._mrs_sampling(ctx, hull, spacing, patches)
        sampling = hull.kind.sampling(hull.settings, at, patches, density)
        if sampling.generators is None or sampling.chosen is None:
            raise ValueError(f"{hull.kind.label()}'s sampling took no graded density: its method is "
                             f"{sampling.method!r}")
        return sampling, spacing * finest_share(density, sampled_points(hull.kind, hull.settings, sampling)), density

    @override
    def _graded_arguments(self, hull: Hull, sampling: Sampling, density: GradedDensity | None,
                          counts: tuple[tuple[int, ...], ...] | None, spacing: float) -> dict[str, Any]:
        """The node's graded patch, judged at each point of its separated quadrature (``counts``), the points its
        step finds the body's nearest among, and the build report's line."""
        if density is None or counts is None or sampling.chosen is None:
            return {}
        quadrature_points = quadrature_arrays(hull.kind, hull.settings, sampling.patches, counts)[0]
        finest = finest_share(density, sampled_points(hull.kind, hull.settings, sampling))
        count = len(sampled_points(hull.kind, hull.settings, quadrature(hull.kind, hull.settings, spacing,
                                                                       sampling.patches)))
        towards = ["the kind's features (its edges and crests)"] if density.features else []
        if density.contact is not None:
            towards.append(f"the contact facing the wall, graded for a gap of {density.contact.gap * 1e3:.4g} mm "
                           f"at {CONTACT_SPACINGS:g} local spacings")
        text = (f"towards {' and '.join(towards)}, {len(sampling.chosen)} force points against the uniform "
                f"sampling's {count}, the finest {finest:.3g} of the force spacing")
        return {"graded_patch": graded_patch(density, quadrature_points), "grading": text}

    def _density(self, settings: Mapping[str, Magnitude], hull: Hull, spacing: float) -> GradedDensity | None:
        """The grading's density over the hull's kind, or ``None`` where nothing is graded: its features where
        ``grading='features'``, and its contact, the direction written turned into the kind's frame, for the sampling's
        ``contact_gap`` (of the entry's ``settings``, SI), each rounded as the hull's numbers are (``FIGURES``)."""
        if not self.graded():
            return None
        sampled = self._sampled()
        contact = None
        if sampled.contact is not None:
            turn = np.asarray(hull.rotation, dtype=np.float64)
            facing = [figures(float(c)) for c in turn.T @ np.asarray(sampled.contact, dtype=np.float64)]
            contact = Contact((facing[0], facing[1], facing[2]),
                              figures(float(cast(float, settings["sampling.contact_gap"]))))
        return graded_density(hull.kind, hull.settings, spacing, features=sampled.grading == "features",
                              contact=contact)

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        """0.4.0's, and with lubrication the roughness of the body's surface and of the wall's, each read where the
        experiment declares it (``<body>.roughness``, ``<structure>.roughness``; ``DEFAULT_ROUGHNESS`` otherwise); the
        stokes rung's its own (:meth:`_stokes_plan`)."""
        if self.rung == "stokes":
            return self._stokes_plan(ctx)
        found = super().plan(ctx)
        if not self.lubricated(ctx):
            return found
        values = dict(found.values)
        if ctx.value("body", "roughness") is not None:
            values["roughness_body"] = Value("body", "roughness")
        if ctx.value("medium", "wall.roughness") is not None:
            values["roughness_wall"] = Value("medium", "roughness", wall=True)
        return dataclasses.replace(found, values=values)

    def _build_near_field(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        """The medium's joint node for every member's body, each member's own node's arguments under ``m<i>_``, the
        wall's constants once; each body's pose, twist and mass matrix read, its resistance and background load
        delivered. The cross-kernel's budget is the largest ``share`` its members' ``NearField`` writes (every point
        where none writes one); the route GMRES only where every member's declares it, and its tolerance the smallest
        they write (the more accurate route and solve any member asks for)."""
        ids = sorted(ctx.members)
        written = [cast(Drag, ctx.members[owner].entry)._near() for owner in ids]  # noqa: SLF001 - its own class's
        shares = [float(near.share) for near in written if near.share is not None]
        bodies: list[str] = []
        held: dict[str, Any] = {}
        walls: dict[str, Any] = {}
        common: dict[str, Any] | None = None
        for i, owner in enumerate(ids):
            member = ctx.members[owner]
            entry = cast(UncoupledDrag, member.entry)
            arguments = entry._mrs_arguments(member)
            for key in ("lubrication", "handover_spacing", "roughness_body", "roughness_wall"):
                arguments.pop(key, None)  # no hand-over in a medium's near field yet (validity: W2a's PR 2)
            correction = wall_correction_named(arguments["wall_kind"], arguments["wall_kind_version"])
            names = {f"{WALL}{name}" for name in correction.parameters}
            shared = {key: arguments.pop(key) for key in ("viscosity", "wall_kind", "wall_kind_version")}
            if common is None:
                common = shared
                walls = {key: arguments[key] for key in names}
            elif shared != common:
                raise ValueError(f"the near field's members read one medium and one wall: {owner} reads {shared}, "
                                 f"{ids[0]} {common}")
            held.update({f"{member_prefix(i)}{key}": value for key, value in arguments.items() if key not in names})
            bodies.append(member.bindings["body"])
        assert common is not None
        tolerances = [float(near.tolerance) for near in written if near.tolerance is not None]
        solve = {"solve": "gmres", **({"tolerance": min(tolerances)} if tolerances else {})} \
            if all(near.solve == "gmres" for near in written) else {}
        node = b.add_response_node(JointDragNode(ctx.node_name(), ctx.timestep, members=tuple(bodies), **common,
                                                 **walls, **({"cross_share": max(shares)} if shares else {}),
                                                 **solve, **held))
        for i, owner in enumerate(ids):
            b.connect(b.read(f"{owner}/pose"), into=node[f"pose_{i}"])
            b.connect(b.read(f"{owner}/twist"), into=node[f"twist_{i}"])
            b.connect(b.read(f"{owner}/mass_matrix"), into=node[f"mass_matrix_{i}"])
            b.connect(node[f"resistance_{i}"], into=b.target(f"{owner}/resistance"))
            b.connect(node[f"background_load_{i}"], into=b.target(f"{owner}/background_load"))


def _direction(value: object) -> bool:
    """Whether ``value`` is three finite numbers, not all zero."""
    if not isinstance(value, tuple) or len(cast(tuple[object, ...], value)) != 3:
        return False
    held = cast(tuple[object, ...], value)
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in held):
        return False
    return any(float(cast(float, c)) != 0.0 for c in held)


def _wall_kinds_listed() -> str:
    """Every registered wall kind with the region it matches, and every reserved one, for a refusal to list."""
    built = [f"{record.name} {record.version} (matches "
             f"{_matched(cast(type[WallKind], record.cls).matches)})" for record in registered(EntryKind.WALL_KIND)]
    return ", ".join(built + [f"{name} (reserved: {reason})" for name, reason in RESERVED.items()])


def _matched(shape: str | None) -> str:
    return "a medium that fills no region" if shape is None else f"a {shape} region"


def _wall_kind(written: object) -> type[WallKind] | str:
    """The wall kind ``wall_kind`` names, ``"<name>"`` (its one registered version) or ``"<name> <version>"``; or why
    it names none."""
    if not isinstance(written, str) or not written.split():
        return f"wall_kind={written!r} is not the name of a wall kind; the wall kinds are {_wall_kinds_listed()}"
    name, *version = written.split()
    if name in RESERVED and not versions_of(EntryKind.WALL_KIND, name):
        return f"wall_kind={written!r}: {RESERVED[name]}"
    versions = versions_of(EntryKind.WALL_KIND, name) if len(version) <= 1 else []
    if version:
        versions = [v for v in versions if v == version[0]]
    if len(versions) != 1:
        held = versions_of(EntryKind.WALL_KIND, name)
        why = f"several versions of {name!r} are registered, {held}: write wall_kind='{name} <version>'" \
            if len(held) > 1 and not version else f"it is no registered wall kind"
        return f"wall_kind={written!r}: {why}; the wall kinds are {_wall_kinds_listed()}"
    record = lookup(EntryKind.WALL_KIND, name, versions[0])
    if record is None or not issubclass(record.cls, WallKind):
        return f"wall_kind={written!r} is no registered wall kind; the wall kinds are {_wall_kinds_listed()}"
    return record.cls


def _wall_problems(written: object, region: Region | None, medium: str, hull: Hull | None) -> list[Problem]:
    """The mrs rung's refusals of its wall: a wall kind that does not match the region the medium fills exactly
    (``drag.wall-kind``, naming those that do), one the method of regularised Stokeslets has no correction for, and a body its
    correction's data cannot reach at any pose (``drag.wall-range``)."""
    wall = _wall_kind(written)
    if isinstance(wall, str):
        return []
    shape = region.shape if region is not None else None
    if wall.matches != shape:
        fills = f"fills {region.path}, a {shape} region" if region is not None else "fills no region"
        matching = [f"{record.name} {record.version}" for record in wall_kinds_matching(shape)]
        offered = f"the wall kinds that match it: {matching}" if matching else \
            (f"no wall kind matches {_matched(shape)} yet" if shape is not None else "none")
        return [Problem("drag.wall-kind",
                        f"medium {medium!r} {fills}, and wall_kind {wall.label()} matches {_matched(wall.matches)}: "
                        f"a wall left out, or one the region does not have, changes the resistance; {offered}",
                        "write wall_kind= the wall kind the medium's region matches; the wall kinds are "
                        f"{_wall_kinds_listed()}")]
    try:
        correction = wall_correction_named(wall.name, wall.version)
    except KeyError:
        return [Problem("drag.wall-kind", f"wall kind {wall.label()} has no correction for the regularised-Stokeslet "
                                          f"method registered", "import the package that registers it, or choose "
                                                                "another wall kind: wall_kind=\"<name>\"")]
    if region is None or hull is None:
        return []
    reach = hull.kind.enclosing_radius(hull.settings) + math.dist(hull.centre, (0.0, 0.0, 0.0))
    return [Problem("drag.wall-range", message, f"{action}: Drag.MRS(wall_kind=...) for a wall kind that reaches the "
                                                f"body, or a body and tube its table covers")
            for message, action in correction.beyond(dict(region.parameters), reach)]


_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _trained_kind(surrogate: ScrewSurrogate, hull: Hull | None, body: str) -> Hull | Problem:
    """The body's hull, when it is the kind the surrogate was trained on, centred on the body's origin with its axis
    along the body's x, as the surrogate takes it; or why it is not."""
    action = (f"give {body} one part outside every other, a {surrogate.family} geometry kind at the body's origin and "
              f"unturned, with the shape the surrogate covers, parts={{\"shell\": Part({surrogate.family}(...))}}; or "
              f"choose another rung for this body, Drag.MRS(...)")
    if hull is None:
        return Problem("drag.body-kind", f"the surrogate {surrogate.asset!r} was trained on {surrogate.family} bodies, "
                                         f"and {body} has no hull whose kind gives its shape", action)
    if hull.kind.name != surrogate.family:
        return Problem("drag.body-kind", f"the surrogate {surrogate.asset!r} was trained on {surrogate.family} bodies, "
                                         f"and {hull.label()}", action)
    turned = any(abs(hull.rotation[i][j] - _IDENTITY[i][j]) > 1e-12 for i in range(3) for j in range(3))
    if turned or any(c != 0.0 for c in hull.centre):
        return Problem("drag.body-kind", f"the surrogate takes the helix's axis to be the body's x and its centroid "
                                         f"the body's origin, and {hull.label()} is placed at {hull.centre} m or "
                                         f"turned in the body's frame", action)
    return hull


def _outside_training(surrogate: ScrewSurrogate, ctx: PlanContext, tube: UniformTube, hull: Hull) -> list[Problem]:
    """Each of the body's and tube's shape values outside the surrogate's training data, as a refusal naming the
    value, the range and where the range comes from; and a modulation or number of starts other than its own."""
    found: list[Problem] = []
    family = hull.kind.family_parameters(hull.settings)
    nu, radius, length = float(family["wavenumber"]), float(family["radius"]), float(family["length"])
    for setting, trained in (("modulation", surrogate.modulation), ("starts", surrogate.starts)):
        if not math.isclose(float(family[setting]), trained, rel_tol=1e-12):
            found.append(Problem(
                "drag.outside-training",
                f"the surrogate {surrogate.asset!r} was trained at {setting} {trained!r} alone, and {hull.label()} "
                f"has {setting} {family[setting]!r}",
                "declare a body of the surrogate's family, Part(Screw(...)), or choose another rung, Drag.MRS(...)"))
    of = f"{hull.label()}:"
    checked: list[tuple[str, str, float]] = [
        ("helix_wavenumber", f"{of} its wavenumber", nu),
        ("confinement", f"{of} its radius / the tube's radius ({tube.radius!r} m)", radius / tube.radius),
        ("aspect", f"{of} its length / its radius", length / radius),
    ]
    for quantity, written, value in checked:
        declared = surrogate.shape_range(quantity)
        if not declared.holds(value):
            found.append(Problem(
                "drag.outside-training",
                f"the surrogate {surrogate.asset!r} covers {declared.describe()} ({declared.basis}); {written} is "
                f"{value!r}",
                "declare a body and tube inside the surrogate's training data, or choose another rung, Drag.MRS(...) "
                "(the MRS rung covers any screw); no profile relaxes the training domain: carrying past "
                "it is a declared model change, a1's"))
    return found


__all__ = ["Drag", "ForceSampling", "Lubrication", "NearField", "PointForceDrag", "UncoupledDrag", "UniformTube", "uniform_tube"]
