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
volume above zero), and within tolerance of the surface: for a solid whose regions all have shapes, every vertex lies
on the body's boundary (the largest of the regions' signed distances zero, to the rounding above), carried by the
kind's own frame into the frame its shapes are in; and a solid's volume by the divergence theorem over the mesh
converges on its closed form at second order, as an inscribed mesh's does: drawn at twice the segments, the difference
is at most a third of what it was (a quarter, in theory), unless it is already within rounding. That holds for any
kind's curvature, where a fixed share would not, and a mesh scaled or placed wrong stays as far off at any resolution.
What it checks of each kind is returned beside its problems (:func:`drawing_checks`)."""

from __future__ import annotations

import math
from collections.abc import Mapping
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


def _mesh_volume(points: np.ndarray, counts: np.ndarray, indices: np.ndarray) -> float:
    """The divergence theorem over the mesh, each face fanned into triangles from its first vertex."""
    total, at = 0.0, 0
    for count in counts.tolist():
        face = points[indices[at:at + count]]
        for i in range(1, count - 1):
            total += float(np.dot(face[0], np.cross(face[i], face[i + 1]))) / 6.0
        at += count
    return total


def drawing_checks(kind: Geometry) -> tuple[str, ...]:
    """What :func:`drawing_problems` checks of ``kind``'s drawing: ``closed`` always; ``on the surface`` for a solid
    whose regions all have shapes; ``volume`` for a solid."""
    checks = ["closed"]
    if isinstance(kind, Solid) and kind.regions and all(region in kind.region_shapes for region in kind.regions):
        checks.append("on the surface")
    if isinstance(kind, Solid):
        checks.append("volume")
    return tuple(checks)


def drawing_problems(kind: Geometry, settings: Mapping[str, Magnitude], segments: int) -> list[str]:
    """Every way ``kind``'s drawing at ``segments``, with its ``settings`` in SI, fails the drawing check; none for a
    kind that does not draw itself (a kind with a native form is drawn by USD)."""
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
    if "on the surface" in drawing_checks(kind):
        translation, orientation = kind_frame(kind, settings)
        placed = np.asarray([np.add(rotate(orientation, (float(p[0]), float(p[1]), float(p[2]))), translation)
                             for p in points])
        size = float(np.max(np.linalg.norm(placed, axis=1)))
        boundary = np.full(len(placed), -np.inf)
        for region in kind.regions:
            distance, _ = _at_samples(shape_named(kind.region_shapes[region]),
                                      kind.region_parameters(region, settings), placed)
            boundary = np.maximum(boundary, distance)
        off = float(np.max(np.abs(boundary)))
        if not off <= KIND_ROUNDING_ULPS * _EPSILON * size:
            worst = int(np.argmax(np.abs(boundary)))
            found.append(f"{label}: its drawing's vertices lie up to {off:.3g} m off its surface (vertex {worst}, at "
                         f"{placed[worst].tolist()})")
    if "volume" in drawing_checks(kind):
        closed = cast(Solid, kind).volume(settings)
        finer = kind.drawing(settings, 2 * segments)
        finer_counts = np.asarray(finer.face_vertex_counts, dtype=np.int64)
        apart = abs(volume - closed) / closed
        finer_apart = abs(_mesh_volume(np.asarray(finer.points, dtype=np.float64).reshape(-1, 3), finer_counts,
                                       np.asarray(finer.face_vertex_indices, dtype=np.int64)) - closed) / closed
        rounding = KIND_ROUNDING_ULPS * _EPSILON * math.sqrt(len(finer_counts))
        if not (apart <= rounding or finer_apart <= apart / VOLUME_CONVERGENCE):
            found.append(f"{label}: its drawing's volume does not converge on its closed form {closed!r} m^3 at "
                         f"second order: {apart:.3g} of it apart at {segments} segments and {finer_apart:.3g} at "
                         f"{2 * segments}")
    return found


def sampling_kinds(kinds: list[type[Geometry]]) -> list[type[Geometry]]:
    """The kinds among ``kinds`` that sample their surface: those the harness checks."""
    return [kind for kind in kinds if implements(kind, "samples")]


def drawing_kinds(kinds: list[type[Geometry]]) -> list[type[Geometry]]:
    """The kinds among ``kinds`` that draw themselves: those the drawing check checks."""
    return [kind for kind in kinds if implements(kind, "drawing")]


__all__ = ["KIND_ROUNDING_ULPS", "VOLUME_CONVERGENCE", "drawing_checks", "drawing_kinds", "drawing_problems", "kind_problems",
           "sampling_kinds"]
