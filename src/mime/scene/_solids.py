"""Solids, the geometry kinds a body's part is: each a registered ``mime.registry.Solid`` with a closed-form volume
and inertia, in the part's own frame, centred on its centroid. A cylinder, an ellipsoid and a box, with their volume
and inertia only; and a sphere and a capsule (and the screw, ``_screw.py``), which also give their samples,
area, family parameters and radial reach, and are evaluated at run time by their shapes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import numpy as np

from mime.core import AXIS_ULPS, jnp, switch
from mime.gradients import GradientStatus
from mime.registry import PHYSICAL
from mime.registry import (
    Density,
    Direction,
    DirectionLike,
    Drawing,
    GradientRecord,
    Native,
    NoPrimitive,
    RegionPrimitive,
    Samples,
    Sampling,
    Solid,
    Support,
    Tensor,
    register_geometry,
    uniform,
)
from mime.scene._caps import cap_rings, ring_floor
from mime.scene._kinds import NO_PRIMITIVE_ADVICE, banded_faces, closed_drawing
from mime.scene._sampling import length_of, magnitude_of, quasi_uniform
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


def _round(d: Any, scale: Any) -> Support:
    """The support of a ball of radius ``scale`` about the origin along ``d`` (``(..., 3)``): ``r d / |d|``, and its
    derivative in ``d``, ``r (I - u u^T) / |d|``."""
    length = switch.nonzero_norm(d, axis=-1)[..., None]
    unit = d / length
    eye = jnp.eye(3, dtype=unit.dtype)
    return Support(scale * unit, scale * (eye - unit[..., :, None] * unit[..., None, :]) / length[..., None])


def _end(near: Any, axis: Any) -> Any:
    """The end of a cylinder's or a capsule's axis that the body's point ``near`` lies towards: ``+1``, ``-1``, or ``0``
    for a point on the mid-plane (the generator's middle, where the two ends tie). ``near`` is one of the kind's
    samples, whose place along the axis is the length times a number fixed when the samples are made, so the end a
    sample lies towards never changes along a study's derivative."""
    return jnp.sign(jnp.sum(jnp.asarray(near) * axis, axis=-1))


def _lengths(settings: Mapping[str, Magnitude], name: str) -> tuple[float, float, float]:
    value = settings[name]
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"setting {name!r} is three lengths, not {value!r}")
    return value[0], value[1], value[2]


def _diagonal(x: float, y: float, z: float) -> Tensor:
    return ((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, z))


#: A box's regions, the slab across each of its axes, and the pairs of faces that bound them.
_SPANS = ("span_x", "span_y", "span_z")
_FACES = ("faces_x", "faces_y", "faces_z")
#: The nodes from pole to pole an ellipsoid's area is taken at (and twice as many round), and the ratio of its longest
#: semi-axis to its shortest up to which that is converged to rounding: measured, the area at 256 nodes against 384
#: within 4e-16 of it up to a ratio of 20.
AREA_NODES = 256
ELLIPSOID_ASPECT = 20.0


def _unit_axes() -> tuple[tuple[float, float, float], ...]:
    return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _face_edges(patch: str) -> tuple[int, int]:
    """The axes along a box's face pair's two edges, in order: y then z for ``faces_x``, x then z for ``faces_y``, x
    then y for ``faces_z``."""
    if patch not in _FACES:
        raise KeyError(patch)
    across = _FACES.index(patch)
    first, second = (i for i in range(3) if i != across)
    return first, second


#: The cylinder's support: a direction along its axis, where its whole cap is farthest and the support is the cap's
#: centre, beside directions off it, whose support is a point of the rim.
CYLINDER_SUPPORT = switch.Declaration(
    owner="cylinder.support",
    indicator="the length of the direction's part across the cylinder's axis, in the direction's units; along the "
              "axis within AXIS_ULPS units in the last place of the direction's length (a band)",
    orders={"support": -1},
    route="none: along the axis the farthest set is the whole cap, so the farthest point jumps from the cap's centre "
          "to its rim as the direction leaves the axis, as the body's own does")


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Cylinder(Solid):
    """A solid circular cylinder of ``radius`` and ``length`` along ``axis`` (three numbers in the part's frame),
    centred on its centroid. Its regions are the unbounded cylinder ``side`` and the ``span`` between its end planes
    (the body is their intersection), the boundaries of its patches, the ``lateral`` side and the two flat ``caps``.
    Its quadrature is the trapezoid rule round the axis with Gauss-Legendre nodes along it, and each cap in rings
    about a sample at its centre (``mime.scene._caps``, the screw's caps' layout, which no ring crowds), so that its
    area and its divergence integrals are exact; its sampling at a spacing is quasi-uniform, chosen from that quadrature
    (``mime.scene.quasi_uniform``), and its reach along ``d`` is ``r |d_across| + L |d . a| / 2``. Its derivative is its
    regions' shapes', as their switches declare (the axis, the mid-plane): its volume and inertia are exact."""

    name = "cylinder"
    version = "0.1.0"
    setting_roles = dict.fromkeys(("radius", "length", "axis"), PHYSICAL)
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(CYLINDER_SUPPORT,))
    support_basis = "closed form"
    references = ("WeissteinCylinder",)
    builds = {"": ()}
    regions = ("side", "span")
    region_shapes = {"side": "cylinder", "span": "slab"}
    patches = {"lateral": "side", "caps": "span"}
    placement = {"axis": "z_axis"}
    native = Native("Cylinder", {"radius": "radius", "height": "length"}, {"axis": "Z"})
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

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        axis = _lengths(settings, "axis")
        if region == "side":
            return {"centre": (0.0, 0.0, 0.0), "axis": axis, "radius": _length(settings, "radius")}
        if region == "span":
            return {"centre": (0.0, 0.0, 0.0), "axis": axis, "half_thickness": _length(settings, "length") / 2.0}
        raise KeyError(region)

    @override
    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        radius, length = _length(settings, "radius"), _length(settings, "length")
        around = _counted(2.0 * math.pi * radius, spacing, 3)
        if patch == "lateral":
            return around, _counted(length, spacing, 1)
        if patch == "caps":
            # a cap's samples as dense as the side's, its area over the spacing squared, in rings (``_caps``)
            return (max(1 + ring_floor(1), math.ceil(math.pi * radius * radius / (spacing * spacing))),)
        raise KeyError(patch)

    @override
    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        """``radius`` and ``length`` may be traced; ``axis`` is a number fixed when the samples are made, as the
        capsule's is."""
        axis = np.asarray(parameters["axis"], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        first, second = _across(axis)
        first, second, axis = jnp.asarray(first), jnp.asarray(second), jnp.asarray(axis)
        radius, half = parameters["radius"], parameters["length"] / 2.0
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
            rings = cap_rings(math.prod(counts), 1)
            reach = jnp.asarray(rings.fraction)[:, None]
            unit = (jnp.asarray(np.cos(rings.angle))[:, None] * first
                    + jnp.asarray(np.sin(rings.angle))[:, None] * second)
            found = [Samples(end * half * axis + radius * reach * unit, jnp.broadcast_to(end * axis, unit.shape),
                             radius * radius * jnp.asarray(rings.share)) for end in (-1.0, 1.0)]
            return Samples(*(jnp.concatenate([getattr(s, name) for s in found]) for name in
                             ("points", "normals", "weights")))
        raise KeyError(patch)

    @override
    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        radius, length = _length(settings, "radius"), _length(settings, "length")
        if patch == "lateral":
            return 2.0 * math.pi * radius * length
        if patch == "caps":
            return 2.0 * math.pi * radius * radius
        raise KeyError(patch)

    @override
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quasi_uniform(self, settings, spacing, patches, density)

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        d = jnp.asarray(direction)
        axis = jnp.asarray(parameters["axis"], dtype=d.dtype)
        axis = axis / switch.nonzero_norm(axis)
        along = jnp.dot(d, axis)
        return parameters["radius"] * length_of(d - along * axis) + parameters["length"] / 2.0 * magnitude_of(along)

    @override
    def support(self, parameters: Mapping[str, Any], direction: Any, near: Any) -> Support:
        """The farthest point along ``d`` of the rim at the end ``near`` lies towards: ``r u + e L a / 2``, ``u`` the
        unit direction of ``d``'s part across the axis and ``e`` the end (``_end``), with the derivative ``r (I - a
        a^T - u u^T) / |d_across|``. A cylinder's farthest points along any direction lie on its rims, so each end's
        is the farthest point of its half; where ``d`` lies along the axis (within ``AXIS_ULPS`` units in the last
        place of its length, so that rounding never turns the point about the rim) the whole cap ties, and the point is
        the cap's centre, the switch ``cylinder.support``."""
        d = jnp.asarray(direction)
        axis = jnp.asarray(parameters["axis"], dtype=d.dtype)
        axis = axis / switch.nonzero_norm(axis)
        across = d - jnp.sum(d * axis, axis=-1)[..., None] * axis
        squared = jnp.sum(across * across, axis=-1)
        share = AXIS_ULPS * float(jnp.finfo(d.dtype).eps)
        off = squared > share * share * jnp.sum(d * d, axis=-1)
        length = jnp.sqrt(switch.domain_guard(off, squared, 1.0))
        unit = switch.where(off[..., None], across / length[..., None], jnp.zeros_like(across),
                            owner="cylinder.support", resolution=lambda: share * jnp.sqrt(jnp.sum(d * d, axis=-1)),
                            indicator=lambda: jnp.sqrt(squared))
        eye = jnp.eye(3, dtype=d.dtype)
        bend = (eye - axis[:, None] * axis[None, :] - unit[..., :, None] * unit[..., None, :]) / length[..., None, None]
        jacobian = switch.where(off[..., None, None], parameters["radius"] * bend, jnp.zeros_like(bend),
                                owner="cylinder.support", resolution=lambda: share * jnp.sqrt(jnp.sum(d * d, axis=-1)),
                                indicator=lambda: jnp.sqrt(squared))
        end = _end(near, axis)
        return Support(parameters["radius"] * unit + (end * parameters["length"] / 2.0)[..., None] * axis, jacobian)


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Ellipsoid(Solid):
    """A solid ellipsoid with ``semi_axes`` (three lengths) along the part frame's x, y and z. Its one patch,
    ``surface``, bounds its region ``solid``, which no registered shape evaluates (its distance has no closed form);
    its quadrature is Gauss-Legendre nodes in the cosine of the polar angle about z and the trapezoid rule round it,
    so that its divergence integrals are exact and its area converges on the surface integral's; its sampling at a
    spacing is quasi-uniform, chosen from that quadrature (``mime.scene.quasi_uniform``), and its reach along ``d`` is
    ``|(a d_x, b d_y, c d_z)|``."""

    name = "ellipsoid"
    version = "0.1.0"
    setting_roles = {"semi_axes": PHYSICAL}
    differentiability = GradientStatus.EXACT
    support_basis = "closed form"
    references = ("WeissteinEllipsoid",)
    builds = {"": ()}
    regions = ("solid",)
    patches = {"surface": "solid"}
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

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "solid":
            raise KeyError(region)
        return {"semi_axes": _lengths(settings, "semi_axes")}

    @override
    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        """By the root mean square of a meridian's semi-axes and of the equator's, which a half meridian and the
        equator are about pi and two pi times long."""
        if patch != "surface":
            raise KeyError(patch)
        a, b, c = _lengths(settings, "semi_axes")
        wide = max(a, b)
        return (_counted(math.pi * math.sqrt((wide * wide + c * c) / 2.0), spacing, 2),
                _counted(2.0 * math.pi * math.sqrt((a * a + b * b) / 2.0), spacing, 3))

    @override
    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        """At ``(a sin t cos p, b sin t sin p, c cos t)``, the area element over ``d(cos t) dp`` the length of
        ``(b c sin t cos p, a c sin t sin p, a b cos t)``, which is the outward normal's direction."""
        if patch != "surface":
            raise KeyError(patch)
        polar, around = counts
        cosine, weight = _gauss_legendre(polar)
        angle = _around(around)
        sine = np.sqrt(1.0 - cosine ** 2)
        axes = jnp.asarray(parameters["semi_axes"])
        a, b, c = axes[0], axes[1], axes[2]
        x, y = jnp.asarray(np.outer(sine, np.cos(angle)).ravel()), jnp.asarray(np.outer(sine, np.sin(angle)).ravel())
        z = jnp.asarray(np.repeat(cosine, around))
        element = jnp.stack([b * c * x, a * c * y, a * b * z], axis=-1)
        size = switch.nonzero_norm(element, axis=-1)
        areas = jnp.asarray(np.outer(weight, np.full(around, 2.0 * math.pi / around)).ravel())
        return Samples(jnp.stack([a * x, b * y, c * z], axis=-1), element / size[:, None], size * areas)

    @override
    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        """The surface integral, which has no closed form, by its own quadrature at ``AREA_NODES`` nodes from pole to
        pole and twice as many round, converged to rounding on every ellipsoid whose longest semi-axis is within
        ``ELLIPSOID_ASPECT`` of its shortest."""
        if patch != "surface":
            raise KeyError(patch)
        a, b, c = _lengths(settings, "semi_axes")
        cosine, weight = _gauss_legendre(AREA_NODES)
        angle = _around(2 * AREA_NODES)
        sine = np.sqrt(1.0 - cosine ** 2)[:, None]
        element = np.sqrt((b * c * sine * np.cos(angle)) ** 2 + (a * c * sine * np.sin(angle)) ** 2
                          + (a * b * cosine[:, None]) ** 2)
        return float(np.sum(weight[:, None] * element)) * math.pi / AREA_NODES

    @override
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quasi_uniform(self, settings, spacing, patches, density)

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        """Its farthest point along ``d`` is ``A^2 d / |A d|``, ``A`` the diagonal of its semi-axes."""
        return switch.nonzero_norm(jnp.asarray(parameters["semi_axes"]) * jnp.asarray(direction))

    @override
    def support(self, parameters: Mapping[str, Any], direction: Any, near: Any) -> Support:
        """``A^2 d / |A d|``, with the derivative ``(A^2 - s s^T) / |A d|``, ``s`` the point: one point along every
        direction, so ``near`` is not read."""
        d = jnp.asarray(direction)
        axes = jnp.asarray(parameters["semi_axes"], dtype=d.dtype)
        scale = switch.nonzero_norm(axes * d, axis=-1)[..., None]
        point = axes * axes * d / scale
        return Support(point, (jnp.diag(axes * axes) - point[..., :, None] * point[..., None, :]) / scale[..., None])

    @override
    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        """Rings of constant polar angle, ``segments`` points round each and half as many bands from pole to pole,
        closed by a fan at each pole."""
        a, b, c = _lengths(settings, "semi_axes")
        bands = max(2, segments // 2)
        polar = np.pi * np.arange(1, bands) / bands
        around = 2.0 * np.pi * np.arange(segments) / segments
        rings = np.stack([a * np.outer(np.sin(polar), np.cos(around)), b * np.outer(np.sin(polar), np.sin(around)),
                          c * np.outer(np.cos(polar), np.ones(segments))], axis=-1).reshape(-1, 3)
        points = np.concatenate([[(0.0, 0.0, c)], rings, [(0.0, 0.0, -c)]])
        return closed_drawing(points, banded_faces(bands - 1, segments, 0, len(points) - 1),
                              f"the ellipsoid as rings of constant polar angle, {segments} points round each and "
                              f"{bands} bands from pole to pole")


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Box(Solid):
    """A solid rectangular box with edges of ``size`` (three lengths) along the part frame's x, y and z. Its regions are
    the three slabs between its opposite faces (``span_x``, ``span_y``, ``span_z``; the body is their intersection),
    the boundaries of its patches, each pair of opposite faces (``faces_x`` across x, and so on). Its quadrature is
    Gauss-Legendre nodes along both edges of each face, so that its divergence integrals are exact; its sampling at a
    spacing is quasi-uniform, chosen from that quadrature (``mime.scene.quasi_uniform``), and its reach along ``d`` is
    ``(x |d_x| + y |d_y| + z |d_z|) / 2``. Its derivative is its slabs', as their switch declares (the mid-plane): its
    volume and inertia are exact."""

    name = "box"
    version = "0.1.0"
    setting_roles = {"size": PHYSICAL}
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    references = ("WeissteinCuboid", "WeissteinMomentOfInertia")
    builds = {"": ()}
    regions = ("span_x", "span_y", "span_z")
    region_shapes = {"span_x": "slab", "span_y": "slab", "span_z": "slab"}
    patches = {"faces_x": "span_x", "faces_y": "span_y", "faces_z": "span_z"}
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

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region not in _SPANS:
            raise KeyError(region)
        i = _SPANS.index(region)
        return {"centre": (0.0, 0.0, 0.0), "axis": _unit_axes()[i],
                "half_thickness": _lengths(settings, "size")[i] / 2.0}

    @override
    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        size = _lengths(settings, "size")
        first, second = _face_edges(patch)
        return _counted(size[first], spacing, 1), _counted(size[second], spacing, 1)

    @override
    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        """Each face of the pair at ``-x/2`` then ``+x/2`` (for ``faces_x``), Gauss-Legendre nodes along its first and
        then its second edge; ``size`` may be traced."""
        first, second = _face_edges(patch)
        across = 3 - first - second
        size = jnp.asarray(parameters["size"])
        one, one_weight = _gauss_legendre(counts[0])
        two, two_weight = _gauss_legendre(counts[1])
        grid = [jnp.asarray(np.repeat(one, counts[1])), jnp.asarray(np.tile(two, counts[0]))]
        areas = jnp.asarray(np.outer(one_weight, two_weight).ravel()) * size[first] * size[second] / 4.0
        found: list[Samples] = []
        for end in (-1.0, 1.0):
            columns = [jnp.zeros_like(grid[0])] * 3
            columns[first], columns[second] = grid[0] * size[first] / 2.0, grid[1] * size[second] / 2.0
            columns[across] = jnp.full_like(grid[0], end) * size[across] / 2.0
            normal = jnp.zeros(3).at[across].set(end)
            found.append(Samples(jnp.stack(columns, axis=-1), jnp.broadcast_to(normal, (len(areas), 3)), areas))
        return Samples(*(jnp.concatenate([getattr(s, name) for s in found]) for name in
                         ("points", "normals", "weights")))

    @override
    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        size = _lengths(settings, "size")
        first, second = _face_edges(patch)
        return 2.0 * size[first] * size[second]

    @override
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quasi_uniform(self, settings, spacing, patches, density)

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        """Its farthest point along ``d`` is the corner on ``d``'s side, and where a component of ``d`` is zero the
        middle of the edge or face that faces it (the derivative of a zero component's ``|d_i|`` is zero)."""
        return jnp.sum(jnp.asarray(parameters["size"]) / 2.0 * magnitude_of(jnp.asarray(direction)))

    @override
    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        """Its six faces, exactly at any resolution: corner ``4 i + 2 j + k`` is at ``(+-x, +-y, +-z) / 2``, the sign
        of each by its bit."""
        x, y, z = _lengths(settings, "size")
        points = np.asarray([((i - 0.5) * x, (j - 0.5) * y, (k - 0.5) * z) for i in (0, 1) for j in (0, 1)
                             for k in (0, 1)])
        faces = [(4, 6, 7, 5), (0, 1, 3, 2), (2, 3, 7, 6), (0, 4, 5, 1), (1, 5, 7, 3), (0, 2, 6, 4)]
        return closed_drawing(points, faces, "the box's six faces")


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
    """A solid ball of ``radius``, centred on its centroid. Its one patch, ``surface``, has as its quadrature
    Gauss-Legendre nodes in the cosine of the polar angle and evenly in azimuth, so its weights sum to its area and its
    divergence integrals are exact to rounding; its sampling at a spacing is quasi-uniform, chosen from that quadrature
    (``mime.scene.quasi_uniform``), and its reach in a direction its radius; its run-time shape is ``sphere``.

    Version 0.2.0 samples quasi-uniformly: 0.1.0's sampling was its quadrature, which crowds at the poles (a quarter of
    its samples closer than half the mean spacing, MIME-ANO-123), and stays registered so that a folder saved with it
    reproduces."""

    name = "sphere"
    version = "0.2.0"
    setting_roles = {"radius": PHYSICAL}
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    support_basis = "closed form"
    references = ("WeissteinEllipsoid",)
    builds = {"": ()}
    regions = ("solid",)
    region_shapes = {"solid": "sphere"}
    patches = {"surface": "solid"}
    native = Native("Sphere", {"radius": "radius"})
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
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quasi_uniform(self, settings, spacing, patches, density)

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        """``r |d|``: its farthest point along ``d`` is ``r d / |d|``."""
        return parameters["radius"] * switch.nonzero_norm(jnp.asarray(direction))

    @override
    def support(self, parameters: Mapping[str, Any], direction: Any, near: Any) -> Support:
        """``r d / |d|``, with the derivative ``r (I - u u^T) / |d|``: one point along every direction, so ``near`` is
        not read."""
        return _round(jnp.asarray(direction), parameters["radius"])

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius")

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius")

    @override
    def region_primitive(self, region: str, settings: Mapping[str, Magnitude]) -> RegionPrimitive | NoPrimitive:
        if region != "solid":
            raise KeyError(region)
        return RegionPrimitive("sphere", {"centre": (0.0, 0.0, 0.0), "radius": _length(settings, "radius")})


def _across(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors across ``axis`` (a unit vector), completing a right-handed frame with it: from x, or from y
    where the axis lies near x. Taken in NumPy, of an axis fixed when the samples are made: no frame across every
    axis is continuous, so a traced axis would carry a jump in where the samples sit."""
    helper = np.array((1.0, 0.0, 0.0)) if abs(float(axis[0])) < 0.9 else np.array((0.0, 1.0, 0.0))
    first = np.cross(axis, helper)
    first = first / np.linalg.norm(first)
    return first, np.cross(axis, first)


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
    parallel-axis theorem (``m_h`` their mass together), and ``2 m_h r^2 / 5`` about it.

    Its sampling at a spacing is quasi-uniform, chosen from its quadrature (``mime.scene.quasi_uniform``), from version
    0.2.0: 0.1.0's was the quadrature, which crowds at the caps' poles (MIME-ANO-123), and stays registered so that a
    folder saved with it reproduces. Its reach along ``d`` is ``r |d| + h |d . a| / 2``, the sphere's about the end of
    its segment that lies along ``d``."""

    name = "capsule"
    version = "0.2.0"
    setting_roles = dict.fromkeys(("radius", "height", "axis"), PHYSICAL)
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    support_basis = "closed form"
    references = ("WeissteinCylinder", "WeissteinEllipsoid")
    builds = {"": ()}
    regions = ("solid",)
    region_shapes = {"solid": "capsule"}
    patches = {"lateral": "solid", "caps": "solid"}
    placement = {"axis": "z_axis"}
    native = Native("Capsule", {"radius": "radius", "height": "height"}, {"axis": "Z"})
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
        """``radius`` and ``height`` may be traced; ``axis`` is a number fixed when the samples are made, the part's
        frame (a body turned is its pose), since the frame across it cannot turn continuously with it."""
        axis = np.asarray(parameters["axis"], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        first, second = _across(axis)
        first, second, axis = jnp.asarray(first), jnp.asarray(second), jnp.asarray(axis)
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
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quasi_uniform(self, settings, spacing, patches, density)

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        d = jnp.asarray(direction)
        axis = jnp.asarray(parameters["axis"], dtype=d.dtype)
        axis = axis / switch.nonzero_norm(axis)
        return (parameters["radius"] * switch.nonzero_norm(d)
                + parameters["height"] / 2.0 * magnitude_of(jnp.dot(d, axis)))

    @override
    def support(self, parameters: Mapping[str, Any], direction: Any, near: Any) -> Support:
        """The farthest point along ``d`` of the hemisphere at the end ``near`` lies towards: ``e h a / 2 + r d / |d|``,
        ``e`` the end (the cylinder's ``_end``; ``0`` on the mid-plane, the segment's middle), with the derivative
        ``r (I - u u^T) / |d|``. A capsule's farthest points along any direction lie on its two hemispheres, so each
        end's is the farthest point of its half, and a capsule lying along a wall is held at both."""
        d = jnp.asarray(direction)
        axis = jnp.asarray(parameters["axis"], dtype=d.dtype)
        axis = axis / switch.nonzero_norm(axis)
        held = _round(d, parameters["radius"])
        end = _end(near, axis)
        return Support(held.point + (end * parameters["height"] / 2.0)[..., None] * axis, held.jacobian)

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "radius")

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        """To a cap's far point: ``r + h/2``."""
        return _length(settings, "radius") + _length(settings, "height") / 2.0

    @override
    def region_primitive(self, region: str, settings: Mapping[str, Magnitude]) -> RegionPrimitive | NoPrimitive:
        if region != "solid":
            raise KeyError(region)
        return NoPrimitive(f"a capsule is a cylinder capped by two hemispheres, which no one primitive is. "
                           f"{NO_PRIMITIVE_ADVICE}")


__all__ = ["Box", "Capsule", "Cylinder", "Ellipsoid", "Sphere"]
