"""The hydrodynamic family: the resistance of a medium to a body's motion, and the load its flow puts on a body held
still. Here now: ``Drag`` with its surrogate rung, the learned resistance of a de Jongh screw body in a straight tube,
with the near-wall lubrication correction as a setting of that rung; and its boundary-element rung, the resistance of
any rigid body whose hull kind gives surface samples, in an unbounded medium."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast, override

import numpy as np

from mime.core import shape_named, switch
from mime.data import Asset, asset_problem
from mime.gradients import GradientStatus
from mime.nodes.hydro import SurrogateDragNode
from mime.nodes.hydro.bem import FREE_SPACE, SHAPE, BEMDragNode, wall_corrections
from mime.registry import (
    BuildContext,
    Delivery,
    GradientRecord,
    Hull,
    Interaction,
    Plan,
    PlanContext,
    Problem,
    Read,
    Region,
    RoleKind,
    RungGradient,
    SettingGroup,
    SupportRow,
    Value,
    implements,
    register_interaction,
    register_setting_group,
    requires,
)
from mime.surrogates import SURROGATES, ScrewSurrogate
from mime.units import Length, Magnitude, Quantity
from mime.vocabulary import BACKGROUND_LOAD, POSE, RESISTANCE, Capability

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
                       "fill a tube's lumen, or choose a rung for this geometry when one is built")
    if shape.parameter_names != VESSEL_PARAMETERS:
        return Problem("drag.no-axis", f"medium {medium!r} fills {region.path}, a {region.shape} region, whose "
                                       f"parameters are not a circular tube's {VESSEL_PARAMETERS}: the surrogate "
                                       f"needs a tube's axis and radius", "fill a tube's lumen")
    radius = -float(shape.signed_distance(region.parameters, point))
    if not (math.isfinite(radius) and radius > 0.0):
        return Problem("drag.no-axis", f"medium {medium!r} fills {region.path}, whose axis lies outside it (signed "
                                       f"distance {-radius!r} m on the axis)", "fill a tube's lumen")
    return UniformTube(_triple(point), _triple(axis), radius)


#: The surrogate rung's switches: those its node evaluates, from the tube's axis to the lubrication's floors. The bem
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


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Drag(Interaction):
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
    experiment is planned (a recorded acceptance is for the profiles to come);
    the offset from the axis moves during the run and is clamped, as the 0.2 line did, with ``clamp_fired`` in the
    node's state. Its derivative is piecewise, as its switches say: at the offset's clamp, across the tube's axis
    (where the surrogate is not symmetric), where the body's axis turns across the tube, and at the lubrication's
    floors. The resistance is delivered about the body's centre of mass:
    the surrogate's, trained about the end of the body's axis (``zeta = 0``), is moved there, ``L / 2`` along the
    body's x axis taken as the helix's (``SurrogateDragNode``).

    ``bem``: the regularised-Stokeslet boundary-element resistance (``BEMDragNode``) of the body's hull, sampled by its
    kind on ``patches`` (all of them, the closed surface, when none are written) with neighbouring samples about
    ``spacing`` apart, the blob ``blob`` times their mean spacing, about the body's centre of mass at the pose it reads
    each step. Any kind that gives samples serves: a sphere, a capsule, the de Jongh helix; a kind without them is
    refused when the experiment is planned. ``wall_kind`` names the wall whose correction the solve adds, and
    ``free_space``, an unbounded medium, is the only one built: a medium that fills a region (a tube's lumen) is
    refused with it, since a wall it would leave out changes the resistance. It reads the medium's ``viscosity`` and
    delivers the resistance and a zero background load, as the surrogate does. Its derivative is exact: no switch is
    evaluated on its path, the samples are a smooth function of the kind's numbers, and the solve is differentiated
    by the linear-solve rule.

    Each rung's status is its own (``rung_gradients``), and the build records the attached rung's: the surrogate's
    ``piecewise_jump``, with its switches, and the bem's ``exact``. The entry's is the least of them.

    Version 0.3.0 reads the body's shape from its hull's kind, not from declared values: its values lost
    ``helix_wavenumber``, ``length`` and ``radius``, its node gained the body's outer radius and the tube's shape, and
    a body with no helix for its hull is refused; what it computes is unchanged. Version 0.2.0 moved the resistance to
    the centre; 0.1.0 delivered it about the end unshifted (MIME-ANO-112). A document naming either is refused at load
    (``load.version``). The ``bem`` rung was added to 0.3.0 with no change to what the surrogate computes."""

    name = "drag"
    version = "0.3.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=SURROGATE_SWITCHES)
    rung_gradients = {
        "surrogate": RungGradient(differentiability=GradientStatus.PIECEWISE_JUMP,
                                  gradient=GradientRecord(switches=SURROGATE_SWITCHES)),
        "bem": RungGradient(differentiability=GradientStatus.EXACT),
    }
    references = ("deJongh2025NonlinDyn", "CoxBrenner1967", "GoldmanCoxBrenner1967", "BradyBossis1988", "Cortez2005")
    family = "hydrodynamic"
    builds = {"surrogate": (SurrogateDragNode,), "bem": (BEMDragNode,)}
    validity = (
        "surrogate: quasi-steady Stokes flow (the limit of vanishing Reynolds number), as its training resistances were computed; Re and the "
        "unsteady parameter are declared here, not yet computed from the operating point",
        "surrogate: de Jongh screw bodies (cross-section modulation 0.33, two starts) in a straight tube; nu, kappa "
        "and L/R checked against the training data at plan time; the offset declared up to 0.3 of the tube's radius "
        "and clamped beyond 0.95 of the room to the wall; its accuracy against the BEM it was checked against is "
        "MIME-ANO-114's, up to 16% in swim speed between 0.2 and 0.3 of the radius",
        "surrogate: the resistance is moved from the end of the body's axis, about which it was trained, to the "
        "centre of mass, half the body's length along its x axis taken as the helix's, which the surrogate takes to lie along the "
        "tube's",
        "bem: quasi-steady Stokes flow, the limit of vanishing Reynolds number; Re and the unsteady parameter are "
        "declared here, not yet computed from the operating point",
        "bem: an unbounded medium at rest (free_space, the one wall kind built); its error is the regularised method's, "
        "of the order of the blob over the body's size, falling as the spacing falls, and is measured against the "
        "sphere's and the prolate spheroid's closed forms by its oracle tests, not bounded here",
        "bem: the body's hull alone is sampled, on the patches written; a patch left out (the de Jongh helix's caps, "
        "which the line before this one left out) is a model difference, not a rounding",
    )
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    rungs = ("surrogate", "bem")
    support = (SupportRow("surrogate", {
        "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name], publishes=[POSE.name]),
    }, note="a rigid body whose dynamics takes a resistance, in a medium at rest filling a straight tube"),
        SupportRow("bem", {
            "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name],
                             publishes=[POSE.name]),
        }, note="a rigid body whose dynamics takes a resistance and whose hull kind gives surface samples, in an "
                "unbounded medium at rest"))
    reads = {"pose": Read("body", POSE.name)}
    delivers = {"resistance": Delivery("body", RESISTANCE.name), "background_load": Delivery("body", BACKGROUND_LOAD.name)}
    values = {"viscosity": Value("medium", "viscosity")}
    #: The surrogates the rung can evaluate, by the name of their weights' asset, each with its training domain.
    surrogates: ClassVar[Mapping[str, ScrewSurrogate]] = SURROGATES
    rung: str | None = None
    weights: Asset | None = None
    lubrication: Lubrication | None = None
    spacing: Annotated[Quantity | None, Length] = None
    blob: float | None = None
    patches: tuple[str, ...] | None = None
    wall_kind: str | None = None

    @classmethod
    def Surrogate(cls, **settings: Any) -> Drag:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="surrogate", **settings)

    @classmethod
    def BEM(cls, **settings: Any) -> Drag:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="bem", **settings)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        if self.rung == "bem":
            return self._bem_setting_problems(settings)
        found = [f"{name}= is a setting of the bem rung, and the surrogate rung reads none"
                 for name in ("spacing", "blob", "patches", "wall_kind") if getattr(self, name) is not None]
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

    def _bem_setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = [f"{name}= is a setting of the surrogate rung, and the bem rung reads none"
                 for name in ("weights", "lubrication") if getattr(self, name) is not None]
        spacing = settings.get("spacing")
        if not (isinstance(spacing, float) and math.isfinite(spacing) and spacing > 0.0):
            found.append(f"spacing is {spacing!r} m, not a length above zero: the bem rung samples the body's surface "
                         f"with neighbouring samples about spacing apart, so write spacing=mm(...)")
        blob = cast(object, self.blob)
        if isinstance(blob, bool) or not (isinstance(blob, (int, float)) and math.isfinite(blob) and blob > 0.0):
            found.append(f"blob is {blob!r}, not a number above zero: the regularised Stokeslet's blob is blob times the "
                         f"samples' mean spacing (the 0.2 line's was 0.5)")
        patches = cast(object, self.patches)
        if patches is not None and not (isinstance(patches, tuple) and patches
                                        and all(isinstance(p, str) for p in cast(tuple[object, ...], patches))):
            found.append(f"patches={patches!r} is not a tuple of the names of the body's kind's patches")
        if self.wall_kind not in wall_corrections():
            found.append(f"wall_kind={self.wall_kind!r} is not a wall kind; those built are {wall_corrections()}")
        return found

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        if self.rung == "bem":
            return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values),
                        problems=tuple(self._bem_problems(ctx)))
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

    def _bem_problems(self, ctx: PlanContext) -> list[Problem]:
        """The bem rung's refusals: a body with no hull whose kind gives samples, a patch the kind does not have, and a
        medium whose region the wall kind does not match."""
        body, medium = ctx.bindings["body"], ctx.bindings["medium"]
        found: list[Problem] = []
        hull = ctx.geometry(of="body")
        if hull is None or not implements(hull.kind, "samples"):
            held = f"{body} has no hull" if hull is None else f"{hull.label()}, a kind that gives no surface samples"
            found.append(Problem("drag.no-samples", f"the bem rung samples the body's hull, and {held}",
                                 f"give {body} one part outside every other whose kind gives samples (a sphere, a "
                                 f"capsule, the de Jongh helix), or choose another rung"))
        elif self.patches is not None and not set(self.patches) <= set(hull.kind.patches):
            found.append(Problem("drag.patches", f"{hull.label()} has the patches {sorted(hull.kind.patches)}, and "
                                                 f"patches={self.patches!r} names others",
                                 "name patches of the body's kind, or leave patches out for its closed surface"))
        region = ctx.region("medium")
        if self.wall_kind == FREE_SPACE and region is not None:
            found.append(Problem(
                "drag.wall-kind",
                f"medium {medium!r} fills {region.path}, a {region.shape} region, and wall_kind={FREE_SPACE!r} matches "
                f"only a medium that fills none: leaving its wall out changes the resistance; no wall kind built "
                f"matches a {region.shape} region",
                f"write Medium({medium!r}) with no region for an unbounded medium, or choose the surrogate rung for a "
                f"body in a tube"))
        return found

    def _build_bem(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        hull = ctx.geometry(of="body")
        if hull is None or self.wall_kind is None or self.blob is None:
            raise ValueError("phase 1 refuses a bem drag with no hull, no wall kind or no blob")
        spacing = ctx.scalar("spacing")
        patches = self.patches if self.patches is not None else tuple(hull.kind.patches)
        settings = {setting.name: hull.settings.get(setting.name, getattr(hull.kind, setting.name))
                    for setting in dataclasses.fields(hull.kind)}
        node = b.add_response_node(BEMDragNode(
            ctx.node_name(), ctx.timestep, viscosity=ctx.scalar_value("viscosity"), blob=float(self.blob),
            kind=hull.kind.name, kind_version=hull.kind.version, patches=patches,
            counts=[hull.kind.sample_counts(hull.settings, spacing, patch) for patch in patches],
            part_centre=hull.centre, part_rotation=[c for row in hull.rotation for c in row],
            wall_kind=self.wall_kind, **{f"{SHAPE}{name}": value for name, value in settings.items()}))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        if self.rung == "bem":
            self._build_bem(ctx, b)
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


_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _trained_kind(surrogate: ScrewSurrogate, hull: Hull | None, body: str) -> Hull | Problem:
    """The body's hull, when it is the kind the surrogate was trained on, centred on the body's origin with its axis
    along the body's x, as the surrogate takes it; or why it is not."""
    action = (f"give {body} one part outside every other, a {surrogate.family} geometry kind at the body's origin and "
              f"unturned, with the shape the surrogate covers; or choose another rung for this body")
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
                "declare a body of the surrogate's family, or choose another rung"))
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
                "declare a body and tube inside the surrogate's training data, or choose another rung; accepting a "
                "value outside them, with its reason, is for the profiles to come"))
    return found


__all__ = ["Drag", "Lubrication", "UniformTube", "uniform_tube"]
