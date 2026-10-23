"""The hydrodynamic family: the resistance of a medium to a body's motion, and the load its flow puts on a body held
still. Here now: ``Drag`` with its surrogate rung, the learned resistance of a de Jongh screw body in a straight tube,
with the near-wall lubrication correction as a setting of that rung."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast, override

import numpy as np

from mime.core import shape_named
from mime.data import Asset, asset_problem
from mime.gradients import GradientStatus
from mime.nodes.hydro import SurrogateDragNode
from mime.registry import (
    BuildContext,
    Delivery,
    Interaction,
    Plan,
    PlanContext,
    Problem,
    Read,
    Region,
    RoleKind,
    SettingGroup,
    SupportRow,
    Value,
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
    """The near-wall lubrication correction, blended into a drag rung's resistance as one model: the sphere-wall
    asymptotes for the body's nearest point (Cox & Brenner 1967; Goldman, Cox & Brenner 1967), weighted by
    ``exp(-gap / epsilon)`` so that they count only near the wall (the 0.2 line's blend: a modelling choice, with
    ``epsilon`` its setting, not a value a reference fixes)."""

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
    radius = -float(shape.signed_distance(region.parameters, point))
    if not (math.isfinite(radius) and radius > 0.0):
        return Problem("drag.no-axis", f"medium {medium!r} fills {region.path}, whose axis lies outside it (signed "
                                       f"distance {-radius!r} m on the axis)", "fill a tube's lumen")
    return UniformTube(_triple(point), _triple(axis), radius)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Drag(Interaction):
    """The medium's resistance to the body's motion, by the written rung; ``Drag.Surrogate(...)`` is
    ``Drag(rung="surrogate", ...)`` and saves identically.

    ``surrogate``: the Cholesky MLP's resistance for a de Jongh screw body in a straight tube, from ``weights`` (a
    packaged asset, ``mime.data.asset("dejongh_mlp_v2")``), with ``lubrication=Lubrication(epsilon=...)`` blended in when given.
    It reads the body's pose at the start of the step, and the tube's axis and radius from the region the medium fills
    (``Medium(..., fills=tube.lumen)``); it delivers the resistance (an operator the body's dynamics solves with
    implicitly) and the background load, zero in a medium at rest. It reads the medium's ``viscosity`` and the body's
    ``helix_wavenumber``, ``length`` and ``radius``. A body or tube outside the surrogate's training data in ``nu``,
    ``kappa = R / R_tube`` or ``L / R`` is refused when the experiment is planned (a recorded acceptance is for the profiles to come);
    the offset from the axis moves during the run and is clamped, as the 0.2 line did, with ``clamp_fired`` in the
    node's state. Its derivative is piecewise: zero along the offset beyond the clamp, and discontinuous across the
    tube's axis, where the surrogate is not symmetric. The resistance is delivered about the body's centre of mass:
    the surrogate's, trained about the end of the body's axis (``zeta = 0``), is moved there, ``L / 2`` along the
    body's x axis taken as the helix's (``SurrogateDragNode``).

    Version 0.2.0 moves the resistance to the centre; 0.1.0 delivered it about the end unshifted (MIME-ANO-112). A
    document naming 0.1.0 is refused at load (``load.version``)."""

    name = "drag"
    version = "0.2.0"
    differentiability = GradientStatus.PIECEWISE
    validity = (
        "surrogate: quasi-steady Stokes flow (Re -> 0), as its training resistances were computed; Re and the "
        "unsteady parameter are declared here, not yet computed from the operating point",
        "surrogate: de Jongh screw bodies (cross-section modulation 0.33, two starts) in a straight tube; nu, kappa "
        "and L/R checked against the training data at plan time; the offset declared up to 0.3 of the tube's radius "
        "and clamped beyond 0.95 of the room to the wall; its accuracy against the BEM it was checked against is "
        "MIME-ANO-114's, up to 16% in swim speed between 0.2 and 0.3 of the radius",
        "surrogate: the resistance is moved from the end of the body's axis, about which it was trained, to the "
        "centre of mass, L/2 along the body's x axis taken as the helix's, which the surrogate takes to lie along the "
        "tube's",
    )
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    rungs = ("surrogate",)
    support = (SupportRow("surrogate", {
        "body": requires(Capability.RIGID, accepts=[RESISTANCE.name, BACKGROUND_LOAD.name], publishes=[POSE.name]),
    }, note="a rigid body whose dynamics takes a resistance, in a medium at rest filling a straight tube"),)
    reads = {"pose": Read("body", POSE.name)}
    delivers = {"resistance": Delivery("body", RESISTANCE.name), "background_load": Delivery("body", BACKGROUND_LOAD.name)}
    values = {"viscosity": Value("medium", "viscosity"), "helix_wavenumber": Value("body", "helix_wavenumber"),
              "length": Value("body", "length"), "radius": Value("body", "radius")}
    #: The surrogates the rung can evaluate, by the name of their weights' asset, each with its training domain.
    surrogates: ClassVar[Mapping[str, ScrewSurrogate]] = SURROGATES
    rung: str | None = None
    weights: Asset | None = None
    lubrication: Lubrication | None = None

    @classmethod
    def Surrogate(cls, **settings: Any) -> Drag:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="surrogate", **settings)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
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

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        problems: list[Problem] = []
        tube = uniform_tube(ctx.region("medium"), ctx.bindings["medium"])
        surrogate = self.surrogates.get(self.weights.name) if self.weights is not None else None
        if isinstance(tube, Problem):
            problems.append(tube)
        elif surrogate is not None:
            problems += _outside_training(surrogate, ctx, tube)
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values),
                    problems=tuple(problems))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        tube = uniform_tube(ctx.region("medium"), ctx.bindings["medium"])
        weights = self.weights
        if isinstance(tube, Problem) or weights is None:
            raise ValueError("phase 1 refuses a drag with no tube or no weights")
        node = b.add_response_node(SurrogateDragNode(
            ctx.node_name(), ctx.timestep, weights=weights.name, weights_sha256=weights.sha256,
            viscosity=ctx.scalar_value("viscosity"), helix_wavenumber=ctx.scalar_value("helix_wavenumber"),
            length=ctx.scalar_value("length"), radius=ctx.scalar_value("radius"), vessel_centre=tube.centre,
            vessel_axis=tube.axis, vessel_radius=tube.radius,
            lubrication_epsilon=ctx.scalar("lubrication.epsilon") if self.lubrication is not None else None))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))


def _outside_training(surrogate: ScrewSurrogate, ctx: PlanContext, tube: UniformTube) -> list[Problem]:
    """Each of the body's and tube's shape values outside the surrogate's training data, as a refusal naming the
    value, the range and where the range comes from."""
    found: list[Problem] = []
    body = ctx.bindings["body"]
    nu, radius = ctx.value("body", "helix_wavenumber"), ctx.value("body", "radius")
    length = ctx.value("body", "length")
    checked: list[tuple[str, str, Magnitude | None]] = [
        ("helix_wavenumber", f"{body}.helix_wavenumber", nu),
        ("confinement", f"{body}.radius / the tube's radius ({tube.radius!r} m)",
         radius / tube.radius if isinstance(radius, float) else None),
        ("aspect", f"{body}.length / {body}.radius",
         length / radius if isinstance(length, float) and isinstance(radius, float) else None),
    ]
    for quantity, written, value in checked:
        if not isinstance(value, float):
            continue  # an undeclared value is refused as such
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
