"""The hydrodynamic family: the resistance of a medium to a body's motion, and the load its flow puts on a body held
still. Here now: ``Drag`` with its surrogate rung, the learned resistance of a de Jongh screw body in a straight tube,
with the near-wall lubrication correction as a setting of that rung; and its MRS rung, the resistance of
any rigid body whose hull kind gives surface samples, in a medium whose region a wall kind matches (``mime.walls``):
in MIME 0.3, an unbounded one."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, cast, override

import numpy as np

from mime.core import run_dtype, shape_named, switch
from mime.data import Asset, asset_problem
from mime.gradients import GradientStatus
from mime.nodes.hydro import SurrogateDragNode
from mime.nodes.hydro.mrs import SHAPE, WALL, MRSDragNode, wall_correction_named
from mime.nodes.hydro.mrs._node import CONFINED_GAP
from mime.nodes.hydro.mrs._quadrature import empty_cells
from mime.scene import quadrature_counts, quadrature_owners, sampled_points
from mime.registry import (
    DESIGN,
    NUMERICAL,
    PHYSICAL,
    BuildContext,
    EntryKind,
    Delivery,
    Flag,
    GradientRecord,
    Hull,
    Interaction,
    Plan,
    PlanContext,
    Problem,
    Read,
    Region,
    ReynoldsRange,
    RoleKind,
    RungGradient,
    Sampling,
    SettingGroup,
    SupportRow,
    Value,
    WallKind,
    convention,
    determined,
    implements,
    lookup,
    numerical_default,
    register_interaction,
    register_setting_group,
    registered,
    requires,
    versions_of,
    wall_kinds_matching,
)
from mime.surrogates import SURROGATES, ScrewSurrogate
from mime.units import Length, Magnitude, Quantity, m
from mime.vocabulary import BACKGROUND_LOAD, POSE, RESISTANCE, Capability
from mime.walls import RESERVED

if TYPE_CHECKING:
    from mime.build import InteractionBuilder


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Lubrication(SettingGroup):
    """The near-wall lubrication correction, blended into a drag rung's resistance as one model: the leading terms of
    the sphere-wall asymptotes for the body's nearest point (Cox & Brenner 1967; Goldman, Cox & Brenner 1967, part
    I), without the terms after them or the translation-rotation coupling (``SurrogateDragNode`` names them),
    weighted by ``exp(-gap / epsilon)`` so that they count only near the wall (the 0.2 line's blend: a modelling
    choice, with ``epsilon`` its setting, not a value a reference fixes)."""

    group = "lubrication"
    setting_roles = {"epsilon": NUMERICAL}
    epsilon: Annotated[Quantity, Length]


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
#: (``scripts/default_spacing_table.py``, labelled decision D9): for a compact body the most whose run of the walk's T1
#: stays within about 30 s on a laptop's CPU (a sphere's 338 samples: 512 take 50 s); for a screw, where no count does
#: (a run of the walk's T2 size in a tube takes minutes at the coarsest), the coarsest that keeps its shape (FL-9's 222
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
SPACING_MEASURED = ("on the walk's T1 sphere (338 samples), its drag 1.8% and its rotational drag 5.9% above Stokes's "
                    "law, T1's 1000 steps in 22 s; on FL-9 (222 samples), its speed per turn 23% below the finest "
                    "spacing measured on the walk's tube's axis (5984 samples) and 16% below it in free space (6784), "
                    "a run of T2's size in the tube (6000 steps) 11 minutes: write spacing= for a number to use")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PointForceDrag(Interaction):
    """The drag 0.3.0, kept registered so that a folder saved with it loads and reproduces (7j's frozen register's
    experiments name it): its ``mrs`` rung puts a point force at each sample, the samples its own quadrature, with the
    blob half their mean spacing and its count of samples by family (``DEFAULT_SAMPLES``), as before the separated
    quadrature of 0.4.0 (``Drag``); its ``surrogate`` rung is 0.4.0's. A script that writes ``Drag.MRS(...)`` gets
    0.4.0; a document naming 0.3.0 gets this. Version 0.3.0 read the body's shape from its hull's kind, not from
    declared values: its values lost ``helix_wavenumber``, ``length`` and ``radius``, its node gained the body's outer
    radius and the tube's shape, and a body with no helix for its hull was refused; version 0.2.0 moved the resistance
    to the centre; 0.1.0 delivered it about the end unshifted (MIME-ANO-112). A document naming either is refused at
    load (``load.version``)."""

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
        "mrs: a medium at rest that its wall kind matches, unbounded (free_space, the one wall kind built in MIME 0.3); "
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
    support = (SupportRow("surrogate", {
        "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name], publishes=[POSE.name]),
    }, note="a rigid body whose dynamics takes a resistance, in a medium at rest filling a straight tube"),
        SupportRow("mrs", {
            "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name],
                             publishes=[POSE.name]),
        }, note="a rigid body whose dynamics takes a resistance and whose hull kind gives surface samples, in a "
                "medium at rest that a wall kind matches: in MIME 0.3, an unbounded one"))
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
        kind gives at least its family's sample count (``DEFAULT_SAMPLES``) over the patches sampled."""
        hull = ctx.geometry(of="body")
        if self.rung != "mrs" or hull is None or not implements(hull.kind, "sampling"):
            return None
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
            spacing = float(f"{spacing - 10.0 ** (math.floor(math.log10(spacing)) - 2):.3g}")
        return m(spacing)

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

    def _mrs_setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = [f"{name}= is a setting of the surrogate rung, and the mrs rung reads none"
                 for name in ("weights", "lubrication") if getattr(self, name) is not None]
        spacing = settings.get("spacing")
        if not (isinstance(spacing, float) and math.isfinite(spacing) and spacing > 0.0):
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
            return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values),
                        problems=tuple(self._mrs_problems(ctx)), assets=self._wall_assets())
        problems: list[Problem] = []
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

    def _build_mrs(self, ctx: BuildContext, b: InteractionBuilder) -> None:
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
        sampling = hull.kind.sampling(hull.settings, spacing, patches)
        counts, owners = self._quadrature_arguments(hull, sampling, spacing)
        reference_counts, reference_owners = self._reference_quadrature(hull, sampling, spacing)
        wall_counts, wall_owners = self._wall_quadrature(hull, sampling, spacing)
        node = b.add_response_node(MRSDragNode(
            ctx.node_name(), ctx.timestep, viscosity=ctx.scalar_value("viscosity"), blob=float(self.blob),
            kind=hull.kind.name, kind_version=hull.kind.version, patches=patches, counts=sampling.counts,
            part_centre=hull.centre, part_rotation=[c for row in hull.rotation for c in row],
            wall_kind=wall.name, wall_kind_version=wall.version, sampling_chosen=sampling.chosen,
            sampling_members=sampling.members, quadrature_counts=counts, quadrature_owners=owners,
            force_spacing=self._judged_spacing(spacing), reference_quadrature_counts=reference_counts,
            reference_quadrature_owners=reference_owners, wall_quadrature_counts=wall_counts,
            wall_quadrature_owners=wall_owners, **{f"{WALL}{name}": value for name, value in constants.items()},
            **{f"{SHAPE}{name}": value for name, value in settings.items()}))
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
            tuple[tuple[int, ...], ...] | None, np.ndarray | None]:
        """The quadrature the wall's image is integrated over each cell on, its counts and owner map: none here, where
        the image is taken at the force points."""
        return None, None

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
#: The default quadrature's refinement is this over the blob, rounded up: the quadrature's spacing at most half the
#: blob, at which the sphere's drag had converged to within 0.01% and FL-9's speed to within 0.16% (refinement 8 to 16
#: at blob 0.25; measured, phase 7m-b and its review).
QUADRATURE_PER_BLOB = 2.0
#: The wall image's refinement when none is written, for a body held off the wall: its quadrature over each cell this
#: many times finer than the force spacing (MIME-ANO-130). A labelled decision of phase 7m-b's fix round (design #204),
#: measured on FL-9 in the 1/4 in tube at the screw's defaults (float64): at 2 (on FL-9's grid 3, since 2 leaves cells
#: empty) the speed per turn on the axis came within 0.5% (0.829 mm) and 0.2% (0.4 mm) of the image integrated on the
#: whole quadrature, where the image at the force points was 4.6% and 1.0% below it, and on the floor a quarter of a
#: spacing from the wall (0.4 mm) every roll was definite, as on the whole quadrature; 4, measured at 0.829 mm only,
#: cost twice as much for 0.3% on the axis. At an eighth of a spacing (0.829 mm) no refinement measured made the floor
#: definite (three of eight rolls indefinite at 2 and on the whole quadrature, seven at 4): the spacing's rule keeps a
#: held body out of that band (CONFINED_GAP).
WALL_REFINEMENT = 2
WALL_REFINEMENT_MEASURED = (
    "on FL-9 in the 1/4 in tube at the screw's defaults: on the axis the speed per turn 299.2 um at 0.829 mm and "
    "311.5 um at 0.4 mm, against 297.6 and 312.1 with the image integrated on the whole quadrature and 283.8 and 308.9 "
    "with it at the force points; a step 1.4 s and 27 s on a 24-thread laptop CPU, against 0.13 s and 1.8 s at the "
    "force points; on the floor a quarter of a force spacing from the wall definite at every roll measured")

#: What was measured at the defaults (``scripts/default_spacing_table.py``, float64 for the misses, on a 24-thread laptop
#: CPU; phase 7m-b), as the stamp and ``mime pending`` state it.
SEPARATED_SPACING_MEASURED = (
    "on the walk's T1 sphere (338 force points), its drag 0.50% and its rotational drag 1.45% above Stokes's law, "
    "within 2e-4 of the exact regularised drag at its blob, T1's 1000 steps in 4.4 s; on FL-9 (222 force points), "
    "estimates: its speed per turn in free space 2.6% below the shipped free-space reference, itself known to 1.9%, a "
    "point on a ladder that is not monotone (6.1%, 2.6%, 4.0%, 3.5% and 1.3% below from 206 to 567 force points), and "
    "on the walk's tube's axis only 11% below the finest spacing measured (5984 force points), which is still moving "
    "(about 12% below a fit of the ladder); near a wall no error is stated (MIME-ANO-130); a run of T2's size in the "
    "tube (6000 steps) 12 minutes: write spacing= for a number to use")
SEPARATED_BLOB_MEASURED = (
    "with the spacing's default, at this blob: the sphere's drag error, the regularisation's, nearly in proportion to "
    "the blob (0.25%, 0.50% and 0.75% at blobs 0.05, 0.1 and 0.15 and 338 force points); FL-9's speed moved by under "
    "1% over blobs 0.1 to 0.25 at 416 to 957 force points")
SEPARATED_REFINEMENT_MEASURED = (
    "at two over the blob, refining further moved the sphere's drag by under 0.01% and FL-9's speed by 0.16% (8 to 16 "
    "at blob 0.25); at blob 0.1 a refinement of 8, under one over the blob, moved the sphere's drag by 0.08% and FL-9's "
    "speed by 0.9%; a refinement written below the rule is refused where the rule's quadrature moves the resistance by "
    "more than 5%")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Drag(PointForceDrag):
    """The medium's resistance to the body's motion, by the written rung; ``Drag.Surrogate(...)`` is
    ``Drag(rung="surrogate", ...)`` and saves identically.

    ``surrogate``: the Cholesky MLP's resistance for a de Jongh screw body in a straight tube, from ``weights`` (a
    packaged asset, ``mime.data.asset("dejongh_mlp_v2")``), with ``lubrication=Lubrication(epsilon=...)`` blended in when given.
    It reads the body's pose at the start of the step, and the tube from the region the medium fills (``Medium(...,
    fills=tube.lumen)``), whose shape its node evaluates; it delivers the resistance (an operator the body's dynamics
    solves with implicitly) and the background load, zero in a medium at rest. It reads the medium's ``viscosity``, and
    the body's shape from its hull's kind (``ctx.geometry(of="body")``), which must be the kind the surrogate was
    trained on, the de Jongh helix, along the body's x axis: its ``nu``, ``R`` and ``L`` are the features, and its
    radial reach the outer radius the gap is measured from. A body or tube outside the surrogate's training data in
    ``nu``, ``kappa = R / R_tube`` or ``L / R``, or of another modulation or number of starts, is refused when the
    experiment is planned, in every profile (a value outside them is a model change, a1's declared clamp, design #184
    item 2, which no profile relaxes);
    the offset from the axis moves during the run and is clamped, as the 0.2 line did, with ``clamp_fired`` in the
    node's state, ``tilt_fired`` where the body's axis has left the tube's by more than the tilt the surrogate is
    declared valid to, and ``bore_fired`` where it has left it by more than the bore allows (the surrogate cannot see a
    tilt; a screw whose axial moment is its smallest turns broadside under a rotating field, and in a bore narrower
    than it is long goes through the wall). Its derivative is piecewise, as its switches say: at the offset's clamp,
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
    lumen (it reads the asset ``unit_tube_R0.997``, refused by name when planned if it is not in the cache); a wall kind
    that matches another region is refused with the wall kinds that match, and a reserved one (the plane's) with its
    reason. **A wall's correction is evaluated at the force points**, each cell's force placed at its force point and
    weighted by the cell's area: a wall's image system is smooth inside the fluid, varying over the distance to the
    wall, which the force points resolve where the body is further from the wall than about a force spacing, while the
    free part's near field varies over the blob, which needs the fine quadrature; and the image is then read at
    ``N^2`` pairs per step, not ``N`` times the quadrature's count. Every wall kind's correction is evaluated there, the
    tube's included, so the rung is the same for any wall kind. It reads the medium's ``viscosity`` and delivers the
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

    Version 0.4.0 (phase 7m-b) separates the mrs rung's quadrature from its force points and chooses its defaults
    again; its surrogate rung is 0.3.0's. Version 0.3.0 (``PointForceDrag``, kept registered so that a folder saved
    with it reproduces) put a point force at each sample."""

    version = "0.4.0"
    setting_roles = {
        **PointForceDrag.setting_roles,
        "spacing": numerical_default(
            "on the mrs rung, the largest spacing, to three significant figures, at which the hull's kind gives its "
            "family's count of force points on its grid (SEPARATED_SAMPLES: a compact body's, a screw's); for a body a "
            "law holds off the wall of its medium's region (a contact), no coarser than that closest approach over "
            "CONFINED_GAP force spacings",
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
        "wall_refinement": numerical_default(
            "on the mrs rung, for a body a law holds off the wall of its medium's region (a contact), WALL_REFINEMENT: "
            "the wall's image integrated over each force point's cell on the kind's quadrature that many times finer "
            "than the force spacing; otherwise none, the image at the force points",
            "chosen by measurement near the wall (MIME-ANO-130): the image taken at the force points left the "
            "resistance indefinite within half a force spacing of the wall, and integrated over each cell it is "
            "definite further in, at a step's cost that grows as the square of the refinement",
            by="default_wall_refinement", measured=WALL_REFINEMENT_MEASURED),
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
    mrs_settings = (*PointForceDrag.mrs_settings, "quadrature_refinement", "wall_refinement")
    raises = (
        *PointForceDrag.raises,
        Flag("indefinite", "the step's resistance, the wall's correction in it, was not positive definite (its least "
                           "eigenvalue, each row and column scaled by its diagonal, not above zero): the node replaced "
                           "it by NaN before the body took it, so the run stopped there (drag.mrs-indefinite, "
                           "MIME-ANO-130)", rung="mrs"),
        Flag("near_wall", "the body lay within one force spacing of the wall (NEAR_WALL), where the resistance has been "
                          "measured indefinite closer in (MIME-ANO-130) and a definite one is not shown accurate", rung="mrs"),
        Flag("asymmetric", "the step's resistance, before its symmetric part was taken, was symmetric only to more than "
                           "a twentieth of itself (CONFINED_ASYMMETRY_LIMIT), so its error is at least half that share "
                           "(the share bounds it from below; a small share bounds nothing)", rung="mrs"),
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
        "mrs: a wall's correction is taken at the force points, which do not resolve the image near the wall: within "
        "half a force spacing of it the resistance was measured indefinite (a step whose resistance is indefinite "
        "stops the run, drag.mrs-indefinite), within one a step raises near_wall, and there a definite resistance is "
        "not shown accurate; in the tube the confinement's effect converges more slowly than the free part "
        "(MIME-ANO-130)",
    )
    quadrature_refinement: int | None = None
    wall_refinement: int | None = None

    @override
    def default_spacing(self, ctx: PlanContext) -> Quantity | None:
        """The mrs rung's spacing when none is written: its family's (as 0.3.0's rule reads ``SEPARATED_SAMPLES``), and
        for a body a law holds off the wall of the region its medium fills (a contact's margin, its closest approach:
        ``PlanContext.approach``), no coarser than that approach over ``CONFINED_GAP``, so that its closest approach
        lies outside the band where the resistance was measured indefinite with the wall's image in place
        (MIME-ANO-130). A body nothing holds off the wall takes its family's, and the build reports that its closest
        approach is the wall itself."""
        family = super().default_spacing(ctx)
        approach = ctx.approach(of="body")
        if family is None or self.rung != "mrs" or ctx.region("medium") is None or approach is None or approach <= 0.0:
            return family
        held = float(cast(float, family.value))
        confined = approach / CONFINED_GAP
        if confined >= held:
            return family
        spacing = float(f"{confined:.3g}")
        if spacing > confined:  # rounded up past the bound: one unit of the third figure below it
            spacing = float(f"{spacing - 10.0 ** (math.floor(math.log10(spacing)) - 2):.3g}")
        return m(spacing)

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

    def default_wall_refinement(self, ctx: PlanContext) -> int | None:
        """The mrs rung's wall refinement when none is written: ``WALL_REFINEMENT`` for a body a law holds off the
        wall of the region its medium fills (a contact's margin, ``PlanContext.approach``), which the experiment keeps
        near that wall; none otherwise, where the image is taken at the force points: in free space, which has no
        image, and for a body nothing holds near the wall, for which the image's reads cost a step most and a step
        near the wall raises ``near_wall``."""
        if self.rung != "mrs" or ctx.region("medium") is None or ctx.approach(of="body") is None:
            return None
        return WALL_REFINEMENT

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
            tuple[tuple[int, ...], ...] | None, np.ndarray | None]:
        """With a wall_refinement above 1, the kind's quadrature that many times finer than the force spacing, or the
        next finer that leaves no cell empty, and its owner map: the wall's image is integrated over each cell on it
        (``mime.nodes.hydro.mrs.cell_image``). None at 1, where the image is taken at the force points, and in free
        space."""
        written = cast(object, self.wall_refinement)
        if not isinstance(written, int) or written <= 1:
            return None, None
        cells = len(sampled_points(hull.kind, hull.settings, sampling))
        for refinement in range(written, 2 * written + 2):
            counts = quadrature_counts(hull.kind, hull.settings, spacing / refinement, sampling.patches)
            owners = quadrature_owners(hull.kind, hull.settings, sampling, counts)
            if not empty_cells(owners, cells):
                return counts, owners
        return None, None

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
            (f"no wall kind matches {_matched(shape)} yet, and a plane wall has none in 0.3"
             if shape is not None else "none")
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


__all__ = ["Drag", "Lubrication", "PointForceDrag", "UniformTube", "uniform_tube"]
