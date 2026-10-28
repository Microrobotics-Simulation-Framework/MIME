"""The kind-conformance harness: does a geometry kind's sampled surface agree with its run-time shapes and its closed
forms?

For each of the kind's patches, sampled at a spacing (or at counts the caller chooses): the samples lie on the boundary
of the region the patch names (its shape's signed distance zero, to rounding), and on or inside every other region of
a solid, whose body is its regions' intersection (a cap within the screw's cross-section, the screw's surface within
the span of its caps); the sample normals equal the shape's outward normal there, to the rounding of the sample's own
position carried through the normal (``KIND_ROUNDING_ULPS`` units in the last place of ``1 + |p| |grad n|``, the
normal's Jacobian there being the section's curvature and twist: a sample far along a twisted, deeply grooved screw
turns its normal by more than one near the origin); and the weights sum to the patch's area. Over every patch together, a closed kind's volume by the divergence theorem, ``1/3``
of the integral of ``x . n``, equals its closed form; and, for a solid, its second moments by the same theorem, ``1/5``
of the integral of ``x_i x_j (x . n)``, give its inertia per unit mass. Each is judged on rounding of the inputs' scale:
``KIND_ROUNDING_ULPS`` units in the last place of the sum's terms, growing as the square root of their number.

The drawing check (:func:`drawing_problems`), for a kind that draws itself, at the writer's resolution: the mesh is
closed and consistently oriented (each edge in exactly two faces, once each way), outward (its divergence-theorem
volume above zero), and within tolerance of the surface, for every kind. Every vertex lies on the body's boundary, to
the rounding above (a drawing's vertices are on the surface at any resolution, by its contract; the resolution is
what the volume and area test), where the surface is the largest of a solid's regions' signed distances, carried by the
kind's own frame into the frame its shapes are in, or, for a kind with no shapes of its own (the tube, the box, the
ellipsoid), a surface written here from its definition, independently of its drawing (``_SURFACES``). The mesh's volume
and area by the divergence theorem and by summed triangles converge on the closed forms (the kind's, or the surface's
here) at second order, as an inscribed mesh's do: drawn at twice the segments the error is at most a third of what it
was (a quarter, in theory), and the error extrapolated from the drawings at half, the same and twice the segments
(``(4 e(2n) - e(n)) / 3``, which cancels the second-order term) is at most what the extrapolation changes between the
two pairs, which cancels again a term only the fourth order leaves: so a mesh scaled, permuted or placed wrong stays
as far off at any resolution and is caught, while the legitimate error shrinks sixteen-fold at the next, whatever the
kind's curvature, where a fixed share would not. What it checks of each kind is returned beside its problems
(:func:`drawing_checks`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.core import jnp, shape_named
from mime.registry import Geometry, Solid, implements
from mime.scene import kind_frame
from mime.scene._kinds import rotate
from mime.units import Magnitude

#: The tolerance's units in the last place, for a value computed in a few operations from inputs of its scale.
KIND_ROUNDING_ULPS = 64.0
#: How many times smaller a drawing's volume error must be at twice its segments: an inscribed mesh's falls as the
#: square of the facets' size, four times.
VOLUME_CONVERGENCE = 3.0
_EPSILON = float(np.finfo(np.float64).eps)
_vmap: Any = cast(Any, jax).vmap
_jit: Any = cast(Any, jax).jit
_jacfwd: Any = cast(Any, jax).jacfwd


def _at_samples(shape: Any, parameters: Mapping[str, Any], points: Any) -> tuple[np.ndarray, np.ndarray]:
    """The shape's signed distance and outward normal at every sample, in float64."""
    held = dict(parameters)

    def one(point: Any) -> tuple[Any, Any]:
        return shape.signed_distance(held, point), shape.outward_normal(held, point)

    distance, normal = _jit(_vmap(one))(jnp.asarray(points, dtype=jnp.float64))
    return np.asarray(distance, dtype=np.float64), np.asarray(normal, dtype=np.float64)


def _sensitivity(shape: Any, parameters: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    """The largest singular value of the shape's outward normal's Jacobian at each point: how far the normal turns
    per metre the point moves."""
    held = dict(parameters)

    def normal(point: Any) -> Any:
        return shape.outward_normal(held, point)

    def jacobian(point: Any) -> Any:
        return _jacfwd(normal)(point)

    found = _jit(_vmap(jacobian))(jnp.asarray(points, dtype=jnp.float64))
    return np.linalg.norm(np.asarray(found, dtype=np.float64), ord=2, axis=(1, 2))


def kind_problems(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float,
                  counts: Mapping[str, tuple[int, ...]] | None = None) -> list[str]:
    """Every way ``kind``, with its ``settings`` in SI, sampled about ``spacing`` apart (or at ``counts`` for each
    patch it names), fails the harness; none for a kind that samples no patch (it has nothing to check), which the
    registry walk reports."""
    if not kind.patches:
        return []
    label = kind.label()
    found: list[str] = []
    parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                  for name, value in settings.items()}
    flux, moments, terms, scale = 0.0, np.zeros((3, 3)), 0, 0.0
    for patch, region in kind.patches.items():
        chosen = (counts or {}).get(patch) or kind.sample_counts(settings, spacing, patch)
        samples = kind.samples(parameters, patch, chosen)
        points, normals, weights = (np.asarray(held, dtype=np.float64) for held in
                                    (samples.points, samples.normals, samples.weights))
        size = float(np.max(np.linalg.norm(points, axis=1)))
        scale = max(scale, size)
        shape = shape_named(kind.region_shapes[region])
        distance, normal = _at_samples(shape, kind.region_parameters(region, settings), points)
        terms += len(weights)
        allowed = KIND_ROUNDING_ULPS * _EPSILON * math.sqrt(len(weights))
        off = float(np.max(np.abs(distance)))
        if not off <= KIND_ROUNDING_ULPS * _EPSILON * size:
            found.append(f"{label}: patch {patch!r}'s samples lie up to {off:.3g} m off the boundary of its region "
                         f"{region!r} ({kind.region_shapes[region]})")
        for other in kind.regions if isinstance(kind, Solid) else ():
            if other == region or other not in kind.region_shapes:
                continue
            beyond, _ = _at_samples(shape_named(kind.region_shapes[other]), kind.region_parameters(other, settings),
                                    points)
            outside = float(np.max(beyond))
            if not outside <= KIND_ROUNDING_ULPS * _EPSILON * size:
                found.append(f"{label}: patch {patch!r}'s samples lie up to {outside:.3g} m outside its region "
                             f"{other!r} ({kind.region_shapes[other]}), which the body lies within")
        apart = np.max(np.abs(normal - normals), axis=1)
        beyond = apart > KIND_ROUNDING_ULPS * _EPSILON  # the sensitivity is read only where it is needed
        allowed_turn = np.full(len(apart), KIND_ROUNDING_ULPS * _EPSILON)
        if np.any(beyond):
            reach = np.linalg.norm(points[beyond], axis=1)
            allowed_turn[beyond] *= 1.0 + reach * _sensitivity(shape, kind.region_parameters(region, settings),
                                                                points[beyond])
        if not np.all(apart <= allowed_turn):
            worst = int(np.argmax(apart / allowed_turn))
            found.append(f"{label}: patch {patch!r}'s sample normals differ from {kind.region_shapes[region]}'s "
                         f"outward normal by up to {float(apart[worst]):.3g}, beyond {float(allowed_turn[worst]):.3g} "
                         f"there")
        area = kind.area(settings, patch)
        if not abs(float(np.sum(weights)) - area) <= allowed * area:
            found.append(f"{label}: patch {patch!r}'s weights sum to {float(np.sum(weights))!r} m^2, and its area is "
                         f"{area!r} m^2")
        reach = np.sum(points * normals, axis=1) * weights
        flux += float(np.sum(reach)) / 3.0
        moments += np.einsum("ni,nj,n->ij", points, points, reach) / 5.0
    if isinstance(kind, Solid):
        volume = kind.volume(settings)
        allowed = KIND_ROUNDING_ULPS * _EPSILON * math.sqrt(terms)
        if not abs(flux - volume) <= allowed * volume:
            found.append(f"{label}: its volume by the divergence theorem is {flux!r} m^3, and its closed form "
                         f"{volume!r} m^3")
        second = moments / volume
        inertia = np.trace(second) * np.eye(3) - second
        stated = np.asarray(kind.inertia_per_mass(settings), dtype=np.float64)
        apart = float(np.max(np.abs(inertia - stated)))
        if not apart <= allowed * scale * scale:
            found.append(f"{label}: its inertia per unit mass by the divergence theorem is {inertia.tolist()} m^2, "
                         f"and its closed form {stated.tolist()} m^2 (apart by {apart:.3g})")
    return found


def _edges(counts: np.ndarray, indices: np.ndarray) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    at = 0
    for count in counts.tolist():
        face = indices[at:at + count].tolist()
        edges += [(face[i], face[(i + 1) % count]) for i in range(count)]
        at += count
    return edges


def _triangles(points: np.ndarray, counts: np.ndarray,
               indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each face fanned into triangles from its first vertex: the three corners of every triangle."""
    starts = np.cumsum(counts) - counts
    per = counts - 2
    face = np.repeat(np.arange(len(counts)), per)
    step = np.arange(int(per.sum())) - np.repeat(np.cumsum(per) - per, per) + 1
    first = starts[face]
    return points[indices[first]], points[indices[first + step]], points[indices[first + step + 1]]


def _mesh_volume(points: np.ndarray, counts: np.ndarray, indices: np.ndarray) -> float:
    """The divergence theorem over the mesh."""
    a, b, c = _triangles(points, counts, indices)
    return float(np.sum(np.asarray(np.einsum("ij,ij->i", a, np.cross(b, c)), dtype=np.float64))) / 6.0


def _mesh_area(points: np.ndarray, counts: np.ndarray, indices: np.ndarray) -> float:
    a, b, c = _triangles(points, counts, indices)
    return float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1))) / 2.0


@dataclass(frozen=True)
class _Surface:
    """A kind's surface from its definition, for a kind with no shape of its own to hold its drawing to: how far each
    point of the kind's own frame is from the surface (zero on it, in metres to the kind's own scale), and the
    body's volume and surface area."""

    off: Callable[[np.ndarray], np.ndarray]
    volume: float
    area: float


def _tube_surface(settings: Mapping[str, Magnitude]) -> _Surface:
    inner, wall, length = (float(cast(float, settings[name])) for name in ("inner_radius", "wall", "length"))
    outer, half = inner + wall, length / 2.0

    def off(points: np.ndarray) -> np.ndarray:
        radius = np.hypot(points[:, 0], points[:, 1])
        return np.maximum.reduce([radius - outer, inner - radius, np.abs(points[:, 2]) - half])

    return _Surface(off, math.pi * (outer * outer - inner * inner) * length,
                    2.0 * math.pi * (outer + inner) * length + 2.0 * math.pi * (outer * outer - inner * inner))


def _box_surface(settings: Mapping[str, Magnitude]) -> _Surface:
    size = np.asarray(settings["size"], dtype=np.float64)

    def off(points: np.ndarray) -> np.ndarray:
        return np.max(np.abs(points) - size / 2.0, axis=1)

    return _Surface(off, float(np.prod(size)),
                    2.0 * float(size[0] * size[1] + size[1] * size[2] + size[0] * size[2]))


def _ellipsoid_surface(settings: Mapping[str, Magnitude]) -> _Surface:
    axes = np.asarray(settings["semi_axes"], dtype=np.float64)
    a, b, c = axes

    def off(points: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(points / axes, axis=1) - 1.0) * float(np.min(axes))

    # the area, a surface integral with no closed form: Gauss-Legendre in the polar angle and the trapezoid rule round
    # a turn, both spectrally accurate, converged to rounding
    nodes, weights = np.polynomial.legendre.leggauss(200)
    polar, around = np.arccos(nodes), 2.0 * np.pi * np.arange(400) / 400
    sin_p, cos_p = np.sin(polar)[:, None], np.cos(polar)[:, None]
    element = a * b * c * np.sqrt((sin_p * np.cos(around) / a) ** 2 + (sin_p * np.sin(around) / b) ** 2
                                  + (cos_p / c) ** 2)
    area = float(np.sum(np.asarray(weights[:, None] * element, dtype=np.float64))) * (2.0 * np.pi / 400)
    return _Surface(off, 4.0 / 3.0 * math.pi * a * b * c, area)


#: The surface of each kind that draws itself with no shape to hold its drawing to, by name: written from the kind's
#: definition and not from its drawing, which it holds. A drawing kind with neither shapes nor an entry here is not
#: checked on its surface, and the registry walk fails on it.
_SURFACES: Mapping[str, Callable[[Mapping[str, Magnitude]], _Surface]] = {
    "tube": _tube_surface, "solid_box": _box_surface, "solid_ellipsoid": _ellipsoid_surface}


def _has_shapes(kind: Geometry) -> bool:
    return isinstance(kind, Solid) and bool(kind.regions) and all(region in kind.region_shapes
                                                                  for region in kind.regions)


def drawing_checks(kind: Geometry) -> tuple[str, ...]:
    """What :func:`drawing_problems` checks of ``kind``'s drawing: ``closed`` always; ``on the surface``, ``volume`` and
    ``area`` where the kind has shapes, closed forms or a surface here to hold them to."""
    checks = ["closed"]
    if _has_shapes(kind) or kind.name in _SURFACES:
        checks.append("on the surface")
    if isinstance(kind, Solid) or kind.name in _SURFACES:
        checks.append("volume")
    if (implements(kind, "area") and bool(kind.patches)) or kind.name in _SURFACES:
        checks.append("area")
    return tuple(checks)


def _converges(label: str, name: str, closed: float, values: tuple[float, float, float], segments: int,
               rounding: float) -> str | None:
    """Whether ``values``, the mesh's volume or area at half, the same and twice ``segments``, converge on ``closed`` at
    second order, and to it: the error at twice the segments is at most a third of the error at the same (unless within
    ``rounding`` already), and the error extrapolated by cancelling the second-order term, from each pair of
    neighbouring resolutions, falls at least to its own change between the pairs."""
    coarse, same, fine = (abs(value - closed) / closed for value in values)
    errors = [(value - closed) / closed for value in values]
    from_coarser = (4.0 * errors[1] - errors[0]) / 3.0
    from_finer = (4.0 * errors[2] - errors[1]) / 3.0
    if not (same <= rounding or fine <= same / VOLUME_CONVERGENCE):
        return (f"{label}: its drawing's {name} does not converge on its closed form {closed!r} at second order: "
                f"{coarse:.3g} of it apart at {segments // 2} segments, {same:.3g} at {segments} and {fine:.3g} at "
                f"{2 * segments}")
    if not abs(from_finer) <= abs(from_coarser - from_finer) + rounding:
        return (f"{label}: its drawing's {name} converges on {closed * (1.0 + from_finer)!r} and not on its closed "
                f"form {closed!r}: the error left by cancelling the second-order term is {abs(from_finer):.3g} of it "
                f"at {segments} and {2 * segments} segments, and {abs(from_coarser):.3g} at {segments // 2} and "
                f"{segments}")
    return None


def drawing_problems(kind: Geometry, settings: Mapping[str, Magnitude], segments: int) -> list[str]:
    """Every way ``kind``'s drawing at ``segments`` (even), with its ``settings`` in SI, fails the drawing check; none
    for a kind that does not draw itself (a kind with a native form is drawn by USD)."""
    if not implements(kind, "drawing"):
        return []
    label = kind.label()
    drawing = kind.drawing(settings, segments)
    points = np.asarray(drawing.points, dtype=np.float64).reshape(-1, 3)
    counts = np.asarray(drawing.face_vertex_counts, dtype=np.int64)
    indices = np.asarray(drawing.face_vertex_indices, dtype=np.int64)
    found: list[str] = []
    if int(np.sum(counts)) != len(indices) or np.any(counts < 3) or len(indices) == 0 or \
            int(indices.min()) < 0 or int(indices.max()) >= len(points):
        return [f"{label}: its drawing's faces do not index its {len(points)} points"]
    edges = _edges(counts, indices)
    directed = set(edges)
    if len(directed) != len(edges) or any((b, a) not in directed for a, b in edges):
        repeated = len(edges) - len(directed)
        open_edges = sum((b, a) not in directed for a, b in directed)
        found.append(f"{label}: its drawing is not closed and consistently oriented: {repeated} edges run twice one "
                     f"way and {open_edges} have no face the other way")
    unused = len(points) - len(set(indices.tolist()))
    if unused:
        found.append(f"{label}: {unused} of its drawing's points are in no face")
    volume = _mesh_volume(points, counts, indices)
    if not volume > 0.0:
        found.append(f"{label}: its drawing encloses {volume!r} m^3 by the divergence theorem: it faces inward")
    checks = drawing_checks(kind)
    surface = _SURFACES[kind.name](settings) if kind.name in _SURFACES else None
    if "on the surface" in checks:
        if surface is not None:
            placed = points
            boundary = surface.off(points)
        else:
            translation, orientation = kind_frame(kind, settings)
            placed = np.asarray([np.add(rotate(orientation, (float(p[0]), float(p[1]), float(p[2]))), translation)
                                 for p in points])
            boundary = np.full(len(placed), -np.inf)
            for region in kind.regions:
                distance, _ = _at_samples(shape_named(kind.region_shapes[region]),
                                          kind.region_parameters(region, settings), placed)
                boundary = np.maximum(boundary, distance)
        size = float(np.max(np.linalg.norm(placed, axis=1)))
        off = float(np.max(np.abs(boundary)))
        if not off <= KIND_ROUNDING_ULPS * _EPSILON * size:
            worst = int(np.argmax(np.abs(boundary)))
            found.append(f"{label}: its drawing's vertices lie up to {off:.3g} m off its surface (vertex {worst}, at "
                         f"{placed[worst].tolist()})")
    if "volume" in checks or "area" in checks:
        drawn = {at: kind.drawing(settings, at) for at in (segments // 2, 2 * segments)}
        drawn[segments] = drawing
        meshes = [(np.asarray(held.points, dtype=np.float64).reshape(-1, 3),
                   np.asarray(held.face_vertex_counts, dtype=np.int64),
                   np.asarray(held.face_vertex_indices, dtype=np.int64))
                  for _, held in sorted(drawn.items())]
        rounding = KIND_ROUNDING_ULPS * _EPSILON * math.sqrt(len(meshes[-1][1]))
        if "volume" in checks:
            closed = surface.volume if surface is not None else cast(Solid, kind).volume(settings)
            problem = _converges(label, "volume", closed, cast(Any, tuple(_mesh_volume(*mesh) for mesh in meshes)),
                                 segments, rounding)
            found += [problem] if problem else []
        if "area" in checks:
            closed = surface.area if surface is not None else sum(kind.area(settings, patch) for patch in kind.patches)
            problem = _converges(label, "area", closed, cast(Any, tuple(_mesh_area(*mesh) for mesh in meshes)),
                                 segments, rounding)
            found += [problem] if problem else []
    return found


def sampling_kinds(kinds: list[type[Geometry]]) -> list[type[Geometry]]:
    """The kinds among ``kinds`` that sample their surface: those the harness checks."""
    return [kind for kind in kinds if implements(kind, "samples")]


def drawing_kinds(kinds: list[type[Geometry]]) -> list[type[Geometry]]:
    """The kinds among ``kinds`` that draw themselves: those the drawing check checks."""
    return [kind for kind in kinds if implements(kind, "drawing")]


__all__ = ["KIND_ROUNDING_ULPS", "VOLUME_CONVERGENCE", "drawing_checks", "drawing_kinds", "drawing_problems",
           "kind_problems", "sampling_kinds"]
