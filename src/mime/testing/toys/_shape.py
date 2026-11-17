"""A toy plugin shape with only a distance, and the geometry kind whose region it is: the extensibility test's subject.

``ToyCapsuleShape`` (registered ``toy.capsule``) gives its exact signed distance and nothing else, written as a lab
would write it, with no MIME helper: a capsule's (the distance from its segment less its radius), its length's square
floored at one unit in the last place of the radius squared so that its derivative is finite on the segment, and its
clamp to the segment through ``mime.core.switch`` (``toy.capsule.end``). Its outward normal and its camera sign
function are the derived ones, and its normal's switch, ``toy.capsule.kink``, is declared when it is registered. The
kind ``ToyCapsule`` (``toy.capsule``, registered by :func:`register_toy_capsule`, which its tests call) gives no drawing, no sign function and no closed form: its display mesh is
contoured and the camera images it by the sampled route on its derived sign function.

Each derived form has an analytic reference, MIME's capsule: ``mime.core.CAPSULE``'s closed-form distance and normal,
route 1's closed-form image of the ``capsule`` kind, and the capsule's own volume and surface
(``tests/shapes/test_toy_shape.py``)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

from mime.core import jnp, switch
from mime.core.shapes import Parameters, Point, Shape, register_shape
from mime.gradients import GradientStatus
from mime.registry import (
    PHYSICAL,
    Direction,
    DirectionLike,
    EntryKind,
    GradientRecord,
    Solid,
    Tensor,
    lookup,
    register_geometry,
)
from mime.units import Length, Magnitude, Quantity

#: The toy's own switch: the clamp of the point's axial coordinate to the segment's ends.
TOY_CAPSULE_END = switch.Declaration(
    owner="toy.capsule.end",
    indicator="the point's axial coordinate less the nearer end of the segment, m",
    orders={"distance": 1, "normal": 0},
    route="none needed: the distance from a segment is once differentiable across the planes through its ends")


class ToyCapsuleShape(Shape):
    """A capsule given by its signed distance alone: ``centre``, ``axis`` (normalised here), ``radius`` and ``height``
    (between the centres of its caps). It takes an array of points as well as one."""

    parameter_names = ("centre", "axis", "radius", "height")
    takes_many = True

    @override
    def signed_distance(self, parameters: Parameters, point: Point) -> Any:
        point = jnp.asarray(point)
        axis = jnp.asarray(parameters["axis"], dtype=point.dtype)
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        offset = point - jnp.asarray(parameters["centre"], dtype=point.dtype)
        half = parameters["height"] / 2.0
        along = jnp.sum(offset * axis, axis=-1)
        nearest = switch.clip(along, -half, half, owner="toy.capsule.end", resolution=lambda: switch.rounding(half),
                              indicator=along)
        radial = offset - nearest[..., None] * axis
        floor = jnp.finfo(point.dtype).eps * parameters["radius"]
        return jnp.sqrt(jnp.sum(radial * radial, axis=-1) + floor * floor) - parameters["radius"]


TOY_CAPSULE_SHAPE = register_shape("toy.capsule", ToyCapsuleShape(), switches=(TOY_CAPSULE_END,))


def _length(settings: Mapping[str, Magnitude], name: str) -> float:
    return cast(float, settings[name])


@dataclass(frozen=True, kw_only=True)
class ToyCapsule(Solid):
    """A capsule whose one region, ``solid``, is the toy plugin shape ``toy.capsule``: a kind that gives its shape's
    parameters and its mass properties, and takes its drawing and its camera sign function from its shape's distance."""

    name = "toy.capsule"
    version = "0.1.0"
    setting_roles = dict.fromkeys(("radius", "height", "axis"), PHYSICAL)
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    references = ()
    builds = {"": ()}
    regions = ("solid",)
    region_shapes = {"solid": "toy.capsule"}
    radius: Annotated[Quantity, Length]
    height: Annotated[Quantity, Length]
    axis: Annotated[DirectionLike, Direction]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        return [f"{name} is {settings[name]!r} m, not a length above zero" for name in ("radius", "height")
                if not (isinstance(settings[name], float) and math.isfinite(cast(float, settings[name]))
                        and _length(settings, name) > 0.0)]

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        radius, height = _length(settings, "radius"), _length(settings, "height")
        return math.pi * radius * radius * height + 4.0 / 3.0 * math.pi * radius ** 3

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        """A ball's of its radius on every axis: a toy's, not a capsule's (it is never stepped)."""
        moment = 2.0 * _length(settings, "radius") ** 2 / 5.0
        return (moment, 0.0, 0.0), (0.0, moment, 0.0), (0.0, 0.0, moment)

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius") + _length(settings, "height") / 2.0

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius")

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "solid":
            raise KeyError(region)
        return {"centre": (0.0, 0.0, 0.0), "axis": settings["axis"], "radius": settings["radius"],
                "height": settings["height"]}


def register_toy_capsule() -> type[ToyCapsule]:
    """Register the toy kind, once: a test registers it on its own, so the registry walk (which lists the entries a
    fresh process registers) asks no round-trip case of it."""
    if lookup(EntryKind.GEOMETRY, ToyCapsule.name, ToyCapsule.version) is None:
        register_geometry(ToyCapsule)
    return ToyCapsule


__all__ = ["TOY_CAPSULE_END", "TOY_CAPSULE_SHAPE", "ToyCapsule", "ToyCapsuleShape", "register_toy_capsule"]
