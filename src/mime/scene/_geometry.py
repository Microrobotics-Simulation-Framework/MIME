"""Analytic geometry kinds a script declares: :class:`Tube` so far. Each is a registered geometry kind (a
``mime.registry.Geometry``), saved by name and version with its settings; each of its surfaces is evaluated by a
:class:`~mime.core.Shape` from ``mime.core``, which a reading node holds by name with its parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

import numpy as np

from mime.core import CYLINDER, jnp, switch
from mime.gradients import GradientStatus
from mime.registry import PHYSICAL, convention
from mime.registry import (
    Centreline,
    Direction,
    DirectionLike,
    Drawing,
    Geometry,
    GradientRecord,
    NoPrimitive,
    RegionPrimitive,
    register_geometry,
)
from mime.scene._centreline import straight
from mime.scene._sampling import length_of, magnitude_of
from mime.units import Length, Magnitude, Quantity, m


class Axis(tuple[float, float, float]):
    """A direction written as three numbers, with a sign: ``+X``, ``-Z``."""

    __slots__ = ()

    def __new__(cls, x: float, y: float, z: float) -> Axis:
        return super().__new__(cls, (float(x), float(y), float(z)))

    def __neg__(self) -> Axis:
        return Axis(-self[0], -self[1], -self[2])

    def __pos__(self) -> Axis:
        return self


X = Axis(1.0, 0.0, 0.0)
Y = Axis(0.0, 1.0, 0.0)
Z = Axis(0.0, 0.0, 1.0)


def _vector(settings: Mapping[str, Magnitude], name: str) -> tuple[float, float, float]:
    value = settings[name]
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"setting {name!r} is three components, not {value!r}")
    return (value[0], value[1], value[2])


def _end_plane(settings: Mapping[str, Magnitude], start: bool) -> dict[str, Magnitude]:
    """A tube's end as the half-space its lumen lies in: the plane normal to its axis at ``centre -+ (L/2) axis``, its
    normal into the tube (``+axis`` at its ``start``, ``-axis`` at its ``end``)."""
    axis = np.asarray(_vector(settings, "axis"), dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    sign = 1.0 if start else -1.0
    end = np.asarray(_vector(settings, "centre"), dtype=np.float64) - sign * float(cast(float, settings["length"])) \
        / 2.0 * axis
    normal = sign * axis
    return {"normal": (float(normal[0]), float(normal[1]), float(normal[2])), "offset": float(np.dot(normal, end))}


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Tube(Geometry):
    """A straight tube: a wall of thickness ``wall`` around a circular lumen of radius ``inner_radius``, ``length``
    long, along ``axis`` through ``centre``. Its regions are the ``lumen`` and the ``wall``; its ``inner_surface``
    faces the lumen and is evaluated as an unbounded cylinder, so a contact with it bounds a body radially and its
    ends bound nothing (a body past an end is outside the tube, which nothing checks yet); the lumen is that
    cylinder's inside, so a law reading the region a medium fills sees a uniform tube along its axis, with no ends;
    ``axis`` is its direction. Its ``start`` and ``end`` are its two ends along its declared axis, the planes normal
    to it at ``centre -+ (L/2) axis``, each facing the lumen (evaluated as the half-space the lumen lies in): the
    lumen's declared openings (``openings``), named by geometry, never by the role a law gives them, where a flow's
    region is cut at the vessel's own ends. Its reach along ``d`` is the finite tube's,
    ``(r + wall) |d_across| + L |d . a| / 2`` from its centre.

    Its derivative is the cylinder's (its switches, ``cylinder.axis`` and ``cylinder.axis_norm``): a kink in the
    distance and a jump in the normal on the axis. Within the axis band (``AXIS_ULPS`` units in the last place of the
    inner radius: 3e-15 m in float64 and 1.6 um in float32, for a 3.175 mm radius) the normal is zero with the
    derivative of the offset from the axis, the projection across it; a standing test holds that Jacobian
    (``tests/saved/test_shape_suite.py``), and within the band, the switch's at-switch set, the status promises
    nothing."""

    name = "tube"
    version = "0.1.0"
    setting_roles = {
        **dict.fromkeys(("axis", "inner_radius", "wall", "length"), PHYSICAL),
        "centre": convention(PHYSICAL, "a tube is centred on the world's origin unless placed"),
    }
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    references = ()
    builds = {"": ()}
    regions = ("lumen", "wall")
    region_shapes = {"lumen": "cylinder"}
    surfaces = {"inner_surface": "cylinder", "start": "half_space", "end": "half_space"}
    openings = {"lumen": ("start", "end")}
    directions = ("axis",)
    placement = {"centre": "origin", "axis": "z_axis"}
    axis: Annotated[DirectionLike, Direction]
    inner_radius: Annotated[Quantity, Length]
    wall: Annotated[Quantity, Length]
    length: Annotated[Quantity, Length]
    centre: Annotated[Quantity, Length] = m((0.0, 0.0, 0.0))

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        for name in ("inner_radius", "wall", "length"):
            value = settings[name]
            if not (isinstance(value, float) and math.isfinite(value) and value > 0):
                found.append(f"{name} is {value!r} m, not a length above zero")
        centre = settings["centre"]
        if not (isinstance(centre, tuple) and len(centre) == 3):
            found.append(f"centre is {centre!r}, not a point (three lengths)")
        return found

    @override
    def surface_parameters(self, surface: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if surface in ("start", "end"):
            return _end_plane(settings, surface == "start")
        if surface != "inner_surface":
            raise KeyError(surface)
        return {"centre": _vector(settings, "centre"), "axis": _vector(settings, "axis"),
                "radius": settings["inner_radius"]}

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "lumen":
            raise KeyError(region)
        return self.surface_parameters("inner_surface", settings)

    @override
    def direction(self, name: str, settings: Mapping[str, Magnitude]) -> tuple[float, float, float]:
        if name != "axis":
            raise KeyError(name)
        return _vector(settings, "axis")

    @override
    def centreline(self, settings: Mapping[str, Magnitude]) -> Centreline:
        """Its axis from end to end, one segment of its length through its centre; its radius along it the lumen's,
        derived from the lumen's signed distance at the axis's points, so that it needs no stored data."""
        axis = np.asarray(_vector(settings, "axis"), dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        def start(parameters: Mapping[str, Any]) -> Any:
            return jnp.asarray(parameters["centre"]) - parameters["length"] / 2.0 * jnp.asarray(axis)

        def radius(parameters: Mapping[str, Any], s: Any) -> Any:
            at = jnp.asarray(s)
            points = jnp.asarray(start(parameters), dtype=at.dtype)[None, :] + at[:, None] * jnp.asarray(axis,
                                                                                                     dtype=at.dtype)
            lumen = {"centre": jnp.asarray(parameters["centre"], dtype=at.dtype), "axis": jnp.asarray(axis, dtype=at.dtype),
                     "radius": parameters["inner_radius"]}
            return -CYLINDER.signed_distance(lumen, points)

        return Centreline((straight(start, axis, lambda parameters: parameters["length"], radius=radius),))

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        d = jnp.asarray(direction)
        axis = jnp.asarray(parameters["axis"], dtype=d.dtype)
        axis = axis / switch.nonzero_norm(axis)
        along = jnp.dot(d, axis)
        outer = parameters["inner_radius"] + parameters["wall"]
        return (jnp.dot(d, jnp.asarray(parameters["centre"], dtype=d.dtype)) + outer * length_of(d - along * axis)
                + parameters["length"] / 2.0 * magnitude_of(along))

    @override
    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        """The wall: a closed annular mesh (outer side, inner side facing the lumen, and the two end rings), in the
        tube's own frame, its axis along +Z and its centre at the origin."""
        inner, wall, length = (float(cast(float, settings[n])) for n in ("inner_radius", "wall", "length"))
        outer, half, n = inner + wall, length / 2.0, segments
        ring = [(math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n)) for i in range(n)]
        points = tuple((radius * c, radius * s, z) for radius, z in ((outer, -half), (outer, half), (inner, -half),
                                                                     (inner, half)) for c, s in ring)
        outer_low, outer_high, inner_low, inner_high = 0, n, 2 * n, 3 * n
        faces: list[int] = []
        for i in range(n):
            j = (i + 1) % n
            faces += [outer_low + i, outer_low + j, outer_high + j, outer_high + i]
            faces += [inner_low + i, inner_high + i, inner_high + j, inner_low + j]
            faces += [outer_low + i, inner_low + i, inner_low + j, outer_low + j]
            faces += [outer_high + i, outer_high + j, inner_high + j, inner_high + i]
        return Drawing(points, (4,) * (4 * n), tuple(faces),
                       f"the wall as a closed annular mesh of {n} segments, its axis along the prim's +Z")

    @override
    def region_primitive(self, region: str, settings: Mapping[str, Magnitude]) -> RegionPrimitive | NoPrimitive:
        """The lumen region clipped at the tube's length: the finite cylinder of the inner radius and the tube's length,
        centred. The region itself (``region_shapes``, what a law reads) is the unbounded cylinder, so the primitive
        is the region within the tube's ends and not the region. The wall has none."""
        if region == "lumen":
            return RegionPrimitive("cylinder", {"centre": (0.0, 0.0, 0.0), "direction": (0.0, 0.0, 1.0),
                                                "radius": float(cast(float, settings["inner_radius"])),
                                                "length": float(cast(float, settings["length"]))})
        if region == "wall":
            return NoPrimitive("the wall is the annulus between two cylinders, which no one primitive is. MIME hands "
                               "over the lumen, the inner cylinder, and not the outer one: the wall is the outer "
                               "cylinder less the lumen, which a manifest writes as two regions with the lumen's "
                               "priority above the wall's, and priorities are the helper's to choose, so a region "
                               "of the outer cylinder alone would be handed over as a wall it is not")
        raise KeyError(region)


#: The half-width of the cone about a plane's normal in which :meth:`Plane.reach` reads a direction as the normal, in
#: machine epsilons of the direction's length: a few rounding errors of the direction's own unit-normal arithmetic.
REACH_CONE = 64.0


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Plane(Geometry):
    """An unbounded plane wall: the plane ``x . n = offset``, ``normal`` its direction (towards the medium) and
    ``offset`` its signed distance from the world's origin along that direction. Its regions are the ``half_space`` its
    normal points into, which a medium fills (``Medium(..., fills=floor.half_space)``), evaluated as the half-space
    (``mime.core.HalfSpace``), and the ``wall``, the other side. It has no surface yet (a contact against it comes
    later), no drawing and no region primitive: its ``offset`` is a length that reaches a law as the node leaf
    ``wall_offset`` (a study's axis and a run's override, ``floor.offset``), and its normal a direction, held as built.
    Its derivative is the half-space's: its distance is linear in the point and the offset, so it has no switch."""

    name = "plane"
    version = "0.1.0"
    setting_roles = {
        "normal": PHYSICAL,
        "offset": convention(PHYSICAL, "a plane passes through the world's origin unless placed"),
    }
    differentiability = GradientStatus.EXACT
    gradient = GradientRecord()
    references = ()
    builds = {"": ()}
    regions = ("half_space", "wall")
    region_shapes = {"half_space": "half_space"}
    directions = ("normal",)
    placement = {"normal": "z_axis"}
    normal: Annotated[DirectionLike, Direction]
    offset: Annotated[Quantity, Length] = m(0.0)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        offset = settings["offset"]
        if not (isinstance(offset, float) and math.isfinite(offset)):
            return [f"offset is {offset!r} m, not a length (one number)"]
        return []

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "half_space":
            raise KeyError(region)
        return {"normal": _vector(settings, "normal"), "offset": settings["offset"]}

    @override
    def direction(self, name: str, settings: Mapping[str, Magnitude]) -> tuple[float, float, float]:
        if name != "normal":
            raise KeyError(name)
        return _vector(settings, "normal")

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        """The wall's body is the half-space behind the plane, unbounded: along the normal (towards the medium) it
        reaches the plane, ``|d| offset``, and along any other direction without bound (infinity). That reach is
        discontinuous at the normal by nature (a half-space's support function is), so "along the normal" is a cone
        whose half-width is ``REACH_CONE`` machine epsilons of ``|d|`` across: a direction that is the normal to
        rounding (a unit normal tilted off the axes, say) is on it, and no switch in the answer is judged finer than
        the rounding of the direction itself."""
        d = jnp.asarray(direction)
        normal = jnp.asarray(parameters["normal"], dtype=d.dtype)
        normal = normal / switch.nonzero_norm(normal)
        along = jnp.dot(d, normal)
        across = d - along * normal
        cone = REACH_CONE * jnp.finfo(d.dtype).eps
        bounded = (jnp.sum(across * across) <= cone * cone * jnp.sum(d * d)) & (along > 0)
        return jnp.where(bounded, along * parameters["offset"], jnp.inf)
