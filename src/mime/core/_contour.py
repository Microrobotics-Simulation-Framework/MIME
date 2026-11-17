"""A signed distance's zero set as a triangle mesh: the display mesh a shape with only a distance is drawn by.

**Marching tetrahedra.** The box is cut into cubes, each cube into six tetrahedra about its main diagonal (Kuhn's
split, whose faces two neighbouring cubes share, so the mesh has no cracks), and each tetrahedron's zero set is one
triangle (one corner apart from the other three) or two (two and two), its vertices on the tetrahedron's edges where
the distance changes sign, placed by linear interpolation of the corners' values. A vertex is one point for every
tetrahedron that has its edge, so the mesh is closed wherever the zero set lies inside the box; each triangle is
turned to face the corners outside (counter-clockwise seen from outside) by the tetrahedron's own orientation, exact
from its corners' places on the grid, so a sliver of a triangle is turned as its neighbours are. A corner where the
distance is zero counts as outside.

**Accuracy.** Along an edge of length ``h`` crossing the surface at angle ``theta`` to it, an exact distance is
``s sin(theta) + O(s^2 / (2 r))`` (``r`` the surface's smallest radius of curvature there), so the interpolated vertex
lies off the surface by about ``h^2 / (8 r)`` at most where the edge crosses it squarely, and no farther than the
cell's diagonal however it crosses. The mesh is a drawing: its vertices are not on the surface."""

from __future__ import annotations

from collections.abc import Callable
from itertools import permutations

import numpy as np

#: The six tetrahedra of a cube about its diagonal from corner 0 to corner 7, each a path along the cube's edges:
#: corner ``i``'s bits are its offsets along x, y and z.
_TETRAHEDRA = np.asarray([[0, 1 << a, (1 << a) | (1 << b), 7] for a, b, _ in permutations(range(3))],
                         dtype=np.int64)


def contour(distance: Callable[[np.ndarray], np.ndarray], low: float, high: float,
            cells: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The zero set of ``distance`` (a function of an ``(n, 3)`` array of points, giving ``(n,)``) in the cube from
    ``low`` to ``high`` on each axis, widened by a cell each side, cut into ``cells`` cubes across the unwidened span:
    its points ``(m, 3)``, its triangles ``(t, 3)`` (indices, counter-clockwise seen from where the distance is
    positive) and the cell's side."""
    cell = (high - low) / cells
    axis = low - cell + cell * np.arange(cells + 3)
    count = len(axis)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.asarray(distance(grid), dtype=np.float64).reshape(-1)
    base = np.stack(np.meshgrid(*(np.arange(count - 1),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    bits = np.asarray([[(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)])
    corners = base[:, None, :] + bits[None, :, :]
    index = np.asarray((corners[..., 0] * count + corners[..., 1]) * count + corners[..., 2], dtype=np.int64)
    tetrahedra = np.take(index, _TETRAHEDRA, axis=1).reshape(-1, 4)
    inside = values[tetrahedra] < 0.0
    crossed = (inside.sum(axis=1) > 0) & (inside.sum(axis=1) < 4)
    tetrahedra, inside = tetrahedra[crossed], inside[crossed]
    keys: dict[tuple[int, int], int] = {}
    points: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []

    def vertex(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        found = keys.get(key)
        if found is None:
            fa, fb = values[a], values[b]
            found = keys[key] = len(points)
            points.append(grid[a] + fa / (fa - fb) * (grid[b] - grid[a]))
        return found

    def at(corner: int) -> np.ndarray:
        return np.asarray([corner // (count * count), (corner // count) % count, corner % count], dtype=np.int64)

    def turned(a: int, b: int, c: int, d: int) -> bool:
        """Whether the tetrahedron ``a b c d`` is positively oriented, from its corners' whole-number places
        (exact): then the triangle across the edges from ``a`` towards ``b``, ``c`` and ``d`` faces away from
        ``a``."""
        return int(np.linalg.det(np.stack([at(b) - at(a), at(c) - at(a), at(d) - at(a)]).astype(np.float64))) > 0

    for row in range(len(tetrahedra)):
        corners_of = [int(c) for c in tetrahedra[row]]
        ins = [c for c, m in zip(corners_of, inside[row], strict=True) if m]
        outs = [c for c, m in zip(corners_of, inside[row], strict=True) if not m]
        if len(ins) in (1, 3):
            alone, rest = (ins[0], outs) if len(ins) == 1 else (outs[0], ins)
            made = (vertex(alone, rest[0]), vertex(alone, rest[1]), vertex(alone, rest[2]))
            away = turned(alone, rest[0], rest[1], rest[2])
            triangles.append(made if away == (len(ins) == 1) else (made[0], made[2], made[1]))
        else:
            quad = (vertex(ins[0], outs[0]), vertex(ins[0], outs[1]), vertex(ins[1], outs[1]), vertex(ins[1], outs[0]))
            pair = [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
            if not turned(ins[0], outs[0], outs[1], ins[1]):
                pair = [(a, c, b) for a, b, c in pair]
            triangles.extend(pair)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3), np.asarray(triangles, dtype=np.int64).reshape(-1, 3), \
        float(cell)


__all__ = ["contour"]
