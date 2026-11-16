"""A surface as the stage draws it: whole within :data:`SURFACE_TRIANGLES`, decimated over it.

Decimation is vertex clustering at rest: the rest's bounding box is cut into cubes, every vertex in one cube joins one
cluster, a face whose corners fall in fewer than three clusters is dropped, and a face that repeats another's three
clusters is dropped after its first. The cube is the smallest whose clustering keeps the surface within the budget. The
clusters are fixed when the experiment is built, so drawing a sample is a fixed map of the node's values at its vertices
(:func:`reduce_clusters`): a cluster's point is its members' mean, and a field's value their mean or, where the field
declares it (``MeshField.reduction``), their maximum, so that a damage front is not smeared. The stage's rest is the
same map applied to the declared rest, so a surface standing at rest is drawn at its stage's points exactly.

No remeshing: the triangles are the declaration's, merged. A feature thinner than a cube collapses, as in any vertex
clustering; a cluster's point is its members' mean in every sample, so a smooth deformation over many cubes moves the
drawn surface as it moves the declared one (``tests/build/test_surface_meshes.py`` holds that no drawn triangle folds
against the declared surface under a crater)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The most triangles a surface is drawn with in the stage: a surface over it is drawn decimated. A presentation budget,
#: with its basis: a page of about 15 MB holds about 3.5e5 vertex-frames of a surface with one moving field (28.5 bytes
#: a vertex-frame, measured), so a surface at the budget (about 8,000 vertices, two triangles each) fits a page at about
#: 41 drawn samples; the stage, which every run folder carries, holds it in about 0.6 MB. The puncture section of the
#: spike that measured it, 14,044 triangles, is drawn whole.
SURFACE_TRIANGLES = 16_000
#: The finest cubes a clustering is tried at, along the rest box's longest side: past it the declared faces repeat (the
#: only way a clustering finer than every vertex's own cube stays within the budget).
_FINEST = 1 << 21


@dataclass(frozen=True, eq=False)
class Clustered:
    """A surface as the stage draws it: its rest points (``(C, 3)``, metres) and triangles over them, and the cluster of
    each declared vertex (``(V,)``), or ``None`` where it is drawn whole (the rest and the faces as declared)."""

    rest: np.ndarray
    faces: np.ndarray
    clusters: np.ndarray | None


def _merged(faces: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """The faces over the clusters: each face's corners by their clusters, those with fewer than three dropped, and each
    repeat of a set of three clusters dropped after its first, in the declaration's order."""
    held = clusters[faces]
    kept = held[(held[:, 0] != held[:, 1]) & (held[:, 1] != held[:, 2]) & (held[:, 0] != held[:, 2])]
    if not len(kept):
        return kept
    _, first = np.unique(np.sort(kept, axis=1), axis=0, return_index=True)
    return kept[np.sort(first)]


def _clusters_at(rest: np.ndarray, cells: int) -> np.ndarray:
    """Each vertex's cluster with ``cells`` cubes along the rest box's longest side, the clusters numbered in the order
    of their first vertex."""
    low = rest.min(axis=0)
    side = float(np.max(rest.max(axis=0) - low)) / cells
    index = np.minimum(np.floor((rest - low) / side).astype(np.int64), cells - 1)
    key = (index[:, 0] * cells + index[:, 1]) * cells + index[:, 2]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse.reshape(-1)]


def reduce_clusters(values: np.ndarray, clusters: np.ndarray, count: int, reduction: str = "mean", *,
                    points: bool = False) -> np.ndarray:
    """``values`` on ``count`` clusters, in double precision: a field's ``(..., V)``, or with ``points`` a surface's
    points ``(..., V, 3)``, each cluster's members' mean, or their maximum for ``reduction="max"``. The members are
    taken in the order of their vertices, so the same values give the same bits."""
    held = np.asarray(values, dtype=np.float64)
    axis = held.ndim - 2 if points else held.ndim - 1
    order = np.argsort(clusters, kind="stable")
    starts = np.searchsorted(clusters[order], np.arange(count))
    taken = np.take(held, order, axis=axis)
    if reduction == "max":
        return np.maximum.reduceat(taken, starts, axis=axis)
    sums = np.add.reduceat(taken, starts, axis=axis)
    shape = [1] * held.ndim
    shape[axis] = count
    return sums / np.bincount(clusters, minlength=count).astype(np.float64).reshape(shape)


def clustered(rest: np.ndarray, faces: np.ndarray, budget: int = SURFACE_TRIANGLES) -> Clustered:
    """The surface as the stage draws it: as declared within ``budget`` triangles, else clustered at the finest cubes
    that keep it within (found by bisection over the cubes along the rest box's longest side)."""
    points = np.asarray(rest, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if len(triangles) <= budget:
        return Clustered(points, triangles, None)
    within, over = 1, 2
    while over < _FINEST and len(_merged(triangles, _clusters_at(points, over))) <= budget:
        within, over = over, over * 2
    while over - within > 1:
        middle = (within + over) // 2
        if len(_merged(triangles, _clusters_at(points, middle))) <= budget:
            within = middle
        else:
            over = middle
    clusters = _clusters_at(points, within)
    count = int(clusters.max()) + 1
    return Clustered(reduce_clusters(points, clusters, count, points=True), _merged(triangles, clusters), clusters)


__all__ = ["SURFACE_TRIANGLES", "Clustered", "clustered", "reduce_clusters"]
