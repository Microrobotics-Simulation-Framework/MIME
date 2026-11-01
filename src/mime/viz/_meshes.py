"""Meshes for the stage: an STL file read (binary or ASCII), welded and decimated to a triangle budget, and written as
a USD layer. Decimation is vertex clustering on a cubic grid: each cell's vertices become their mean, and a triangle
whose corners fall in fewer than three cells is dropped. Every vertex kept moves at most a cell's diagonal, so the
decimated surface's vertices lie within it of the original; the original's need not lie within it of the decimated
surface, where a feature thinner than a cell collapses and its triangles are dropped, which is why both distances are
measured. The cell is the smallest the bisection finds that meets the budget."""

from __future__ import annotations

import importlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from mime.scene._pxr import Sdf, Usd, UsdGeom, Vt


@dataclass(frozen=True)
class TriangleBudget:
    """How many triangles a stage holds of a description's meshes: at most ``per_mesh`` in one, and ``per_stage`` over
    all of a visual set's (:meth:`shares`). A presentation setting with its basis: the
    0.2 line's AR4 meshes at CAD density (about a million triangles) shimmered in the viewport, with triangles smaller
    than its pixels, and drew slowly on a software renderer; at these counts they did neither."""

    per_mesh: int = 20_000
    per_stage: int = 100_000

    def shares(self, counts: list[int]) -> list[int]:
        """Each mesh's budget: all its triangles up to one cap shared by the set's meshes, the largest cap (never more
        than ``per_mesh``) under which their total is at most ``per_stage``; so a small mesh keeps every triangle and
        only the large ones are decimated."""
        def total(cap: int) -> int:
            return sum(min(count, cap) for count in counts)

        if total(self.per_mesh) <= self.per_stage:
            return [min(count, self.per_mesh) for count in counts]
        low, high = 0, self.per_mesh
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if total(middle) <= self.per_stage else (low, middle - 1)
        return [min(count, low) for count in counts]


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh: its points (``(n, 3)``, metres) and its triangles (``(m, 3)`` indices, counter-clockwise seen
    from outside, as the file wrote them)."""

    points: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class Decimated:
    """A mesh decimated: the mesh, the clustering's cell edge (m; 0 where it was within its budget already) and the
    most a vertex moved (the cell's diagonal)."""

    mesh: Mesh
    cell: float
    moved: float


def read_stl(data: bytes) -> Mesh:
    """An STL file's triangles as a welded mesh: its exactly repeated corners made one point, its triangles with a
    repeated corner dropped. A file is binary when its length is the header, the count and fifty bytes a triangle,
    whatever its first bytes say (a CAD tool's binary file may begin with ``solid``)."""
    corners: np.ndarray
    if len(data) >= 84 and 84 + 50 * struct.unpack_from("<I", data, 80)[0] == len(data):
        count = struct.unpack_from("<I", data, 80)[0]
        record = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
        corners = np.frombuffer(data, dtype=record, count=count, offset=84)["corners"].astype(np.float64)
    else:
        corners = np.asarray(_ascii_corners(data.decode("ascii", errors="replace").split()),
                             dtype=np.float64).reshape(-1, 3, 3)
    if not len(corners):
        raise ValueError("it holds no triangles")
    flat = corners.reshape(-1, 3)
    points, inverse = np.unique(flat, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3).astype(np.int64)
    return Mesh(points, _proper(triangles))


def _ascii_corners(words: list[str]) -> list[float]:
    found: list[float] = []
    for index, word in enumerate(words):
        if word == "vertex" and index + 3 < len(words):
            found += [float(words[index + 1]), float(words[index + 2]), float(words[index + 3])]
    if len(found) % 9:
        raise ValueError("an ASCII STL file whose vertices do not make whole triangles")
    return found


def _proper(triangles: np.ndarray) -> np.ndarray:
    """The triangles with three distinct corners, each set of corners once (the first one's orientation kept)."""
    if not len(triangles):
        return triangles.reshape(0, 3)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    kept = triangles[(a != b) & (b != c) & (a != c)]
    if not len(kept):
        return kept.reshape(0, 3)
    _, first = np.unique(np.sort(kept, axis=1), axis=0, return_index=True)
    return kept[np.sort(first)]


def cluster(mesh: Mesh, cell: float) -> Mesh:
    """``mesh`` with every vertex in one cell of edge ``cell`` made their mean."""
    origin = mesh.points.min(axis=0)
    keys = np.floor((mesh.points - origin) / cell).astype(np.int64)
    sizes = keys.max(axis=0) + 1
    linear = (keys[:, 0] * sizes[1] + keys[:, 1]) * sizes[2] + keys[:, 2]
    _, owner = np.unique(linear, return_inverse=True)
    owner = owner.reshape(-1)
    count = int(owner.max()) + 1
    sums = np.zeros((count, 3))
    np.add.at(sums, owner, mesh.points)
    points = sums / np.bincount(owner, minlength=count)[:, None]
    triangles = _proper(owner[mesh.triangles])
    used = np.unique(triangles)
    renumber = np.full(count, -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    return Mesh(points[used], renumber[triangles])


def decimate(mesh: Mesh, budget: int) -> Decimated:
    """``mesh`` with at most ``budget`` triangles: itself where it has no more, else clustered at the smallest cell the
    bisection finds that meets the budget."""
    if len(mesh.triangles) <= budget:
        return Decimated(mesh, 0.0, 0.0)
    extent = float(np.linalg.norm(np.ptp(mesh.points, axis=0)))
    low, high = extent * 1e-6, extent
    best = cluster(mesh, high)
    best_cell = high
    for _ in range(40):
        middle = math.sqrt(low * high)
        held = cluster(mesh, middle)
        if len(held.triangles) <= budget:
            best, best_cell, high = held, middle, middle
        else:
            low = middle
        if high / low < 1.02:
            break
    return Decimated(best, best_cell, best_cell * math.sqrt(3.0))


def _closest_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The distance from each point ``p`` to the triangle ``(a, b, c)`` broadcast against it (``(..., 3)`` each): the
    closest point by Ericson's regions (Real-Time Collision Detection, 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, -1), np.sum(ac * ap, -1)
    bp = p - b
    d3, d4 = np.sum(ab * bp, -1), np.sum(ac * bp, -1)
    cp = p - c
    d5, d6 = np.sum(ab * cp, -1), np.sum(ac * cp, -1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = va + vb + vc
        v, w = vb / denominator, vc / denominator
        closest = a + v[..., None] * ab + w[..., None] * ac
        on_ab = a + (d1 / (d1 - d3))[..., None] * ab
        on_ac = a + (d2 / (d2 - d6))[..., None] * ac
        on_bc = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None] * (c - b)
    closest = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], on_bc, closest)
    closest = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], on_ac, closest)
    closest = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], on_ab, closest)
    closest = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, closest)
    closest = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, closest)
    closest = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, closest)
    return np.linalg.norm(p - closest, axis=-1)


def _point_triangle(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The distance from each point (``(p, 3)``) to each triangle (``(t, 3)`` corners each), ``(p, t)``."""
    return _closest_distance(points[:, None, :], a, b, c)


#: The most point and triangle pairs one block of a distance computation holds, which bounds its memory.
PAIRS = 2_000_000


def brute_surface_distance(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Each point's distance to ``mesh``'s surface (its nearest triangle), by brute force in blocks of points: every
    point against every triangle."""
    corners = mesh.points[mesh.triangles]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    found = np.empty(len(points))
    chunk = max(1, PAIRS // max(1, len(mesh.triangles)))
    for start in range(0, len(points), chunk):
        found[start:start + chunk] = _point_triangle(points[start:start + chunk], a, b, c).min(axis=1)
    return found


def _tree(points: np.ndarray) -> Any:
    """SciPy's k-d tree of ``points``, which MIME already depends on (its stubs read as unknown to a strict checker)."""
    return cast(Any, importlib.import_module("scipy.spatial")).cKDTree(points)


def surface_distance(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Each point's distance to ``mesh``'s surface (its nearest triangle), exactly as :func:`brute_surface_distance`
    finds it, with the triangles that cannot be nearest left out: a point's nearest vertex of the mesh is on its surface,
    so its nearest triangle lies within that vertex's distance of it, and a triangle whose bounding sphere is farther
    is never measured. The triangles are taken in classes of bounding radius, each a k-d tree of their centres."""
    found = np.full(len(points), np.inf)
    if not len(points) or not len(mesh.triangles):
        return found
    corners = mesh.points[mesh.triangles]
    centres = corners.mean(axis=1)
    radii = np.linalg.norm(corners - centres[:, None, :], axis=-1).max(axis=1)
    used = np.unique(mesh.triangles)
    # Each point's nearest vertex of the mesh: its nearest triangle is no farther.
    reach = cast(tuple[np.ndarray, np.ndarray], _tree(mesh.points[used]).query(points))[0]
    smallest = max(float(radii.min()), 1e-300)
    classes = np.floor(np.log2(np.maximum(radii, smallest) / smallest)).astype(np.int64)
    for held in np.unique(classes):
        members = np.flatnonzero(classes == held)
        widest = float(radii[members].max())
        near = cast(Sequence[Sequence[int]], _tree(centres[members]).query_ball_point(
            points, r=(reach + widest) * (1.0 + 1e-9) + 1e-300))
        counts = np.fromiter((len(each) for each in near), dtype=np.int64, count=len(points))
        if not counts.sum():
            continue
        which = np.repeat(np.arange(len(points)), counts)
        triangles = members[np.fromiter((index for each in near for index in each), dtype=np.int64,
                                        count=int(counts.sum()))]
        block = PAIRS // 4  # each pair's arrays are a few times a cross block's
        for start in range(0, len(which), block):
            point, triangle = which[start:start + block], triangles[start:start + block]
            held_corners = corners[triangle]
            distance = _closest_distance(points[point], held_corners[:, 0], held_corners[:, 1], held_corners[:, 2])
            np.minimum.at(found, point, distance)
    return found


def one_sided_distances(original: Mesh, decimated: Mesh, *, samples: int | None = None,
                        seed: int = 0) -> tuple[float, float]:
    """The two one-sided distances between the meshes, each measured from vertices to the other's surface: the
    original's vertices to the decimated surface, then the decimated vertices to the original's; from every vertex (the
    default, so that no denser sample of vertices finds more), or ``samples`` of each drawn with ``seed``. A decimated
    vertex, a cell's mean, lies within the cell's diagonal of the original; an original vertex need not, where a
    feature thinner than a cell collapsed and its triangles were dropped."""
    generator = np.random.default_rng(seed)

    def chosen(points: np.ndarray) -> np.ndarray:
        if samples is None or len(points) <= samples:
            return points
        return points[generator.choice(len(points), samples, replace=False)]

    return (float(surface_distance(chosen(original.points), decimated).max(initial=0.0)),
            float(surface_distance(chosen(decimated.points), original).max(initial=0.0)))


def mesh_distance(original: Mesh, decimated: Mesh, *, samples: int | None = None, seed: int = 0) -> float:
    """The larger of the two one-sided distances (:func:`one_sided_distances`)."""
    return max(one_sided_distances(original, decimated, samples=samples, seed=seed))


def write_mesh(path: Path, mesh: Mesh, name: str) -> Path:
    """``mesh`` as a USD layer of one ``Mesh`` prim, its default prim, in metres (``metersPerUnit`` 1), with no
    subdivision and no colour of its own (it takes the colour of the prim it is referenced into), exported fresh."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    held = UsdGeom.Mesh.Define(stage, f"/{name}")
    points = mesh.points.astype(np.float32)
    held.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    held.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.triangles)))
    held.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.triangles.reshape(-1).astype(np.int32)))
    held.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if len(points):
        held.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(axis=0), points.max(axis=0)])))
    stage.SetDefaultPrim(held.GetPrim())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    if not layer.Export(str(path)):
        raise OSError(f"usd-core could not write {path}")
    return path


__all__ = ["Decimated", "Mesh", "TriangleBudget", "brute_surface_distance", "cluster", "decimate", "mesh_distance",
           "one_sided_distances", "read_stl", "surface_distance", "write_mesh"]
