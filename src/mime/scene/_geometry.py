"""Analytic geometry kinds a script declares: :class:`Tube` so far. Each is a registered geometry kind (a
``mime.registry.Geometry``), saved by name and version with its settings; each of its surfaces is evaluated by a
:class:`~mime.core.Shape` from ``mime.core``, which a reading node holds by name with its parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, cast, override

from mime.gradients import GradientStatus
from mime.registry import (
    Direction,
    DirectionLike,
    Drawing,
    Geometry,
    GradientRecord,
    NoPrimitive,
    RegionPrimitive,
    register_geometry,
)
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


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Tube(Geometry):
    """A straight tube: a wall of thickness ``wall`` around a circular lumen of radius ``inner_radius``, ``length``
    long, along ``axis`` through ``centre``. Its regions are the ``lumen`` and the ``wall``; its ``inner_surface``
    faces the lumen and is evaluated as an unbounded cylinder, so a contact with it bounds a body radially and its
    ends bound nothing (a body past an end is outside the tube, which nothing checks yet); the lumen is that
    cylinder's inside, so a law reading the region a medium fills sees a uniform tube along its axis, with no ends;
    ``axis`` is its direction.

    Its derivative is the cylinder's (its switches, ``cylinder.axis`` and ``cylinder.axis_norm``): a kink in the
    distance and a jump in the normal on the axis. Within the axis band (``AXIS_ULPS`` units in the last place of the
    inner radius: 3e-15 m in float64 and 1.6 um in float32, for a 3.175 mm radius) the normal is zero with the
    derivative of the offset from the axis, the projection across it; a standing test holds that Jacobian
    (``tests/saved/test_shape_suite.py``), and within the band, the switch's at-switch set, the status promises
    nothing."""

    name = "tube"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    references = ()
    builds = {"": ()}
    regions = ("lumen", "wall")
    region_shapes = {"lumen": "cylinder"}
    surfaces = {"inner_surface": "cylinder"}
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
