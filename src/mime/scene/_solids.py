"""Solids, the geometry kinds a body's part is: each a registered ``mime.registry.Solid`` with a closed-form volume
and inertia, in the part's own frame, centred on its centroid. The simplest shapes that have both: a cylinder, an
ellipsoid and a box. A part whose true shape is none of them (a screw) is represented by one of them, and the
experiment says which approximation it makes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, override

from mime.gradients import GradientStatus
from mime.registry import Direction, DirectionLike, Solid, Tensor, register_geometry
from mime.units import Length, Magnitude, Quantity


def _positive(settings: Mapping[str, Magnitude], name: str, components: int = 0) -> list[str]:
    value = settings[name]
    held = value if isinstance(value, tuple) else (value,)
    if (len(held) != components if components else isinstance(value, tuple)) or not all(
            math.isfinite(v) and v > 0 for v in held):
        shape = f"{components} lengths" if components else "a length"
        return [f"{name} is {value!r} m, not {shape} above zero"]
    return []


def _length(settings: Mapping[str, Magnitude], name: str) -> float:
    value = settings[name]
    if isinstance(value, tuple):
        raise TypeError(f"setting {name!r} is one length, not {value!r}")
    return value


def _lengths(settings: Mapping[str, Magnitude], name: str) -> tuple[float, float, float]:
    value = settings[name]
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"setting {name!r} is three lengths, not {value!r}")
    return value[0], value[1], value[2]


def _diagonal(x: float, y: float, z: float) -> Tensor:
    return ((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, z))


@register_geometry
@dataclass(frozen=True, kw_only=True)
class SolidCylinder(Solid):
    """A solid circular cylinder of ``radius`` and ``length`` along ``axis`` (three numbers in the part's frame),
    centred on its centroid."""

    name = "solid_cylinder"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    radius: Annotated[Quantity, Length]
    length: Annotated[Quantity, Length]
    axis: Annotated[DirectionLike, Direction]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        return _positive(settings, "radius") + _positive(settings, "length")

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        radius, length = _length(settings, "radius"), _length(settings, "length")
        return math.pi * radius * radius * length

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        """``r^2/2`` about the axis and ``r^2/4 + L^2/12`` about every direction across it."""
        radius, length = _length(settings, "radius"), _length(settings, "length")
        a = _lengths(settings, "axis")
        along, across = radius * radius / 2.0, radius * radius / 4.0 + length * length / 12.0

        def entry(i: int, j: int) -> float:
            return across * ((1.0 if i == j else 0.0) - a[i] * a[j]) + along * a[i] * a[j]

        return (entry(0, 0), entry(0, 1), entry(0, 2)), (entry(1, 0), entry(1, 1), entry(1, 2)), \
            (entry(2, 0), entry(2, 1), entry(2, 2))


@register_geometry
@dataclass(frozen=True, kw_only=True)
class SolidEllipsoid(Solid):
    """A solid ellipsoid with ``semi_axes`` (three lengths) along the part frame's x, y and z."""

    name = "solid_ellipsoid"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    semi_axes: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        return _positive(settings, "semi_axes", 3)

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        a, b, c = _lengths(settings, "semi_axes")
        return 4.0 / 3.0 * math.pi * a * b * c

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        """``(b^2 + c^2) / 5`` about x, and so on."""
        a, b, c = _lengths(settings, "semi_axes")
        return _diagonal((b * b + c * c) / 5.0, (a * a + c * c) / 5.0, (a * a + b * b) / 5.0)


@register_geometry
@dataclass(frozen=True, kw_only=True)
class SolidBox(Solid):
    """A solid rectangular box with edges of ``size`` (three lengths) along the part frame's x, y and z."""

    name = "solid_box"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    size: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        return _positive(settings, "size", 3)

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        x, y, z = _lengths(settings, "size")
        return x * y * z

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        """``(y^2 + z^2) / 12`` about x, and so on, with ``x``, ``y`` and ``z`` the edges."""
        x, y, z = _lengths(settings, "size")
        return _diagonal((y * y + z * z) / 12.0, (x * x + z * z) / 12.0, (x * x + y * y) / 12.0)


__all__ = ["SolidBox", "SolidCylinder", "SolidEllipsoid"]
