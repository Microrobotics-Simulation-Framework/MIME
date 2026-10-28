"""Solids, the geometry kinds a body's part is: each a registered ``mime.registry.Solid`` with a closed-form volume
and inertia, in the part's own frame, centred on its centroid. A cylinder, an ellipsoid and a box, with their volume
and inertia only; and a sphere and a capsule (and the de Jongh helix, ``_helix.py``), which also give their samples,
area, family parameters and radial reach, and are evaluated at run time by their shapes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

import numpy as np

from mime.core import jnp
from mime.gradients import GradientStatus
from mime.registry import (
    Direction,
    DirectionLike,
    GradientRecord,
    Samples,
    Solid,
    Tensor,
    register_geometry,
)
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
    references = ("WeissteinCylinder",)
    builds = {"": ()}
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

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        """To a rim: ``sqrt(r^2 + (L/2)^2)``."""
        radius, length = _length(settings, "radius"), _length(settings, "length")
        return math.hypot(radius, length / 2.0)


@register_geometry
@dataclass(frozen=True, kw_only=True)
class SolidEllipsoid(Solid):
    """A solid ellipsoid with ``semi_axes`` (three lengths) along the part frame's x, y and z."""

    name = "solid_ellipsoid"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ("WeissteinEllipsoid",)
    builds = {"": ()}
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

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        """Its longest semi-axis."""
        return max(_lengths(settings, "semi_axes"))


@register_geometry
@dataclass(frozen=True, kw_only=True)
class SolidBox(Solid):
    """A solid rectangular box with edges of ``size`` (three lengths) along the part frame's x, y and z."""

    name = "solid_box"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ("WeissteinCuboid", "WeissteinMomentOfInertia")
    builds = {"": ()}
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

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        """Half its diagonal."""
        x, y, z = _lengths(settings, "size")
        return math.sqrt(x * x + y * y + z * z) / 2.0


def _gauss_legendre(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1]: exact for polynomials below degree ``2 count``."""
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return nodes, weights


def _around(count: int) -> np.ndarray:
    """``count`` angles evenly round a turn, each in the middle of its interval: the trapezoid rule on a periodic
    integrand, exact for trigonometric polynomials below degree ``count``."""
    return 2.0 * math.pi * (np.arange(count) + 0.5) / count


def _counted(length: float, spacing: float, least: int) -> int:
    return max(least, math.ceil(length / spacing))


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Sphere(Solid):
    """A solid ball of ``radius``, centred on its centroid. Its one patch, ``surface``, is sampled by Gauss-Legendre
    nodes in the cosine of the polar angle and evenly in azimuth, so its weights sum to its area and its divergence
    integrals are exact to rounding; its run-time shape is ``sphere``."""

    name = "sphere"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    references = ("WeissteinEllipsoid",)
    builds = {"": ()}
    regions = ("solid",)
    region_shapes = {"solid": "sphere"}
    patches = {"surface": "solid"}
    radius: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        return _positive(settings, "radius")

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        """An ellipsoid's ``4/3 pi a b c`` with its three semi-axes the radius."""
        radius = _length(settings, "radius")
        return 4.0 / 3.0 * math.pi * radius ** 3

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        """An ellipsoid's ``(b^2 + c^2) / 5`` about each axis, its semi-axes the radius: ``2 r^2 / 5``."""
        radius = _length(settings, "radius")
        return _diagonal(*(2.0 * radius * radius / 5.0,) * 3)

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "solid":
            raise KeyError(region)
        return {"centre": (0.0, 0.0, 0.0), "radius": _length(settings, "radius")}

    @override
    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        if patch != "surface":
            raise KeyError(patch)
        radius = _length(settings, "radius")
        return _counted(math.pi * radius, spacing, 2), _counted(2.0 * math.pi * radius, spacing, 3)

    @override
    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        if patch != "surface":
            raise KeyError(patch)
        polar, around = counts
        cosine, weight = _gauss_legendre(polar)
        angle = _around(around)
        sine = np.sqrt(1.0 - cosine ** 2)
        unit = np.stack([np.outer(sine, np.cos(angle)), np.outer(sine, np.sin(angle)),
                         np.outer(cosine, np.ones(around))], axis=-1).reshape(-1, 3)
        radius = parameters["radius"]
        areas = np.outer(weight, np.full(around, 2.0 * math.pi / around)).ravel()
        return Samples(radius * jnp.asarray(unit), jnp.asarray(unit), radius ** 2 * jnp.asarray(areas))

    @override
    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        if patch != "surface":
            raise KeyError(patch)
        return 4.0 * math.pi * _length(settings, "radius") ** 2

    @override
    def family_parameters(self, settings: Mapping[str, Magnitude]) -> dict[str, float | int]:
        return {"radius": _length(settings, "radius")}

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius")

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius")


def _across(axis: Any) -> tuple[Any, Any]:
    """Two unit vectors across ``axis`` (a unit vector), completing a right-handed frame with it: from x, or from y
    where the axis lies near x."""
    helper = jnp.where(jnp.abs(axis[0]) < 0.9, jnp.asarray((1.0, 0.0, 0.0)), jnp.asarray((0.0, 1.0, 0.0)))
    first = jnp.cross(axis, helper)
    first = first / jnp.sqrt(jnp.sum(first * first))
    return first, jnp.cross(axis, first)


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Capsule(Solid):
    """A solid capsule: a cylinder of ``radius`` and ``height`` (between the centres of its caps, as USD's
    ``Capsule`` counts it) along ``axis`` (three numbers in the part's frame), capped by two hemispheres, centred on
    its centroid. Its patches are the ``lateral`` cylinder and the two hemispherical ``caps``; its run-time shape is
    ``capsule``.

    Its inertia is the cylinder's and the two hemispheres': a hemisphere has half a ball's moment about any axis
    through the ball's centre in its flat face, and its centroid lies ``3 r / 8`` from that face (``int z dV / V``), so
    about the capsule's centroid, across the axis, the two have ``m_h (2 r^2 / 5 + H^2 / 4 + 3 H r / 8)`` by the
    parallel-axis theorem (``m_h`` their mass together), and ``2 m_h r^2 / 5`` about it."""

    name = "capsule"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    references = ("WeissteinCylinder", "WeissteinEllipsoid")
    builds = {"": ()}
    regions = ("solid",)
    region_shapes = {"solid": "capsule"}
    patches = {"lateral": "solid", "caps": "solid"}
    radius: Annotated[Quantity, Length]
    height: Annotated[Quantity, Length]
    axis: Annotated[DirectionLike, Direction]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = _positive(settings, "radius")
        height = settings["height"]
        if not (isinstance(height, float) and math.isfinite(height) and height >= 0.0):
            found.append(f"height is {height!r} m, not a length of zero or more")
        return found

    def _masses(self, settings: Mapping[str, Magnitude]) -> tuple[float, float]:
        """The cylinder's volume and the two hemispheres' together."""
        radius, height = _length(settings, "radius"), _length(settings, "height")
        return math.pi * radius * radius * height, 4.0 / 3.0 * math.pi * radius ** 3

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        return math.fsum(self._masses(settings))

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        radius, height = _length(settings, "radius"), _length(settings, "height")
        cylinder, caps = self._masses(settings)
        total = cylinder + caps
        along = (cylinder * radius * radius / 2.0 + caps * 2.0 * radius * radius / 5.0) / total
        across = (cylinder * (height * height / 12.0 + radius * radius / 4.0)
                  + caps * (2.0 * radius * radius / 5.0 + height * height / 4.0 + 3.0 * height * radius / 8.0)) / total
        a = _lengths(settings, "axis")

        def entry(i: int, j: int) -> float:
            return across * ((1.0 if i == j else 0.0) - a[i] * a[j]) + along * a[i] * a[j]

        return (entry(0, 0), entry(0, 1), entry(0, 2)), (entry(1, 0), entry(1, 1), entry(1, 2)), \
            (entry(2, 0), entry(2, 1), entry(2, 2))

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "solid":
            raise KeyError(region)
        return {"centre": (0.0, 0.0, 0.0), "axis": _lengths(settings, "axis"), "radius": _length(settings, "radius"),
                "height": _length(settings, "height")}

    @override
    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        radius, height = _length(settings, "radius"), _length(settings, "height")
        if patch == "lateral":
            return _counted(2.0 * math.pi * radius, spacing, 3), _counted(height, spacing, 1)
        if patch == "caps":
            return _counted(math.pi * radius / 2.0, spacing, 2), _counted(2.0 * math.pi * radius, spacing, 3)
        raise KeyError(patch)

    @override
    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        axis = jnp.asarray(parameters["axis"])
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        first, second = _across(axis)
        radius, half = parameters["radius"], parameters["height"] / 2.0
        if patch == "lateral":
            around, along = counts
            angle = _around(around)
            node, weight = _gauss_legendre(along)
            unit = jnp.asarray(np.cos(angle))[:, None] * first + jnp.asarray(np.sin(angle))[:, None] * second
            normals = jnp.repeat(unit, along, axis=0)
            points = radius * normals + jnp.tile(half * jnp.asarray(node)[:, None] * axis, (around, 1))
            areas = jnp.asarray(np.outer(np.full(around, 2.0 * math.pi / around), weight).ravel())
            return Samples(points, normals, radius * half * areas)
        if patch == "caps":
            polar, around = counts
            node, weight = _gauss_legendre(polar)
            cosine, weight = (node + 1.0) / 2.0, weight / 2.0
            sine = np.sqrt(1.0 - cosine ** 2)
            angle = _around(around)
            found: list[Samples] = []
            for end in (-1.0, 1.0):
                normals = (end * jnp.asarray(np.repeat(cosine, around))[:, None] * axis
                           + jnp.asarray(np.outer(sine, np.cos(angle)).ravel())[:, None] * first
                           + jnp.asarray(np.outer(sine, np.sin(angle)).ravel())[:, None] * second)
                areas = jnp.asarray(np.outer(weight, np.full(around, 2.0 * math.pi / around)).ravel())
                found.append(Samples(end * half * axis + radius * normals, normals, radius ** 2 * areas))
            return Samples(*(jnp.concatenate([getattr(s, name) for s in found]) for name in
                             ("points", "normals", "weights")))
        raise KeyError(patch)

    @override
    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        radius, height = _length(settings, "radius"), _length(settings, "height")
        if patch == "lateral":
            return 2.0 * math.pi * radius * height
        if patch == "caps":
            return 4.0 * math.pi * radius * radius
        raise KeyError(patch)

    @override
    def family_parameters(self, settings: Mapping[str, Magnitude]) -> dict[str, float | int]:
        return {"radius": _length(settings, "radius"), "height": _length(settings, "height")}

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius")

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        """To a cap's far point: ``r + h/2``."""
        return _length(settings, "radius") + _length(settings, "height") / 2.0


__all__ = ["Capsule", "SolidBox", "SolidCylinder", "SolidEllipsoid", "Sphere"]
