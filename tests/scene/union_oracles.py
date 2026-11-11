"""Oracles for the seam-aware union (``mime.scene.ExactUnion``) and the tube tree, written independently of them:
a brute force over a sampled surface, and two closed forms of a filleted union.

**The brute force.** Each capsule's surface sampled at a spacing (rings along its cylinder, rings over its caps), kept
where it lies outside every other capsule (NumPy's distance to a segment, not the union's), so what is kept is the
union's boundary; a point's distance is its nearest kept sample's, its sign the plain minimum's (which is exact in
sign). Its floor is the sampling's: about ``spacing / sqrt(2)`` next to the wall, falling as ``spacing^2 / d`` a
distance ``d`` away.

**The closed forms**, each in the plane through the seam (its cross-section), filleted by a ball of radius ``r``:

- *a wedge*: the union of two half-spaces ``n_i . x <= 0``, the solid the wedge between them. The ball's centres that
  fit in the solid fill ``n_i . x >= r``, a wedge whose corner ``P`` lies where both are ``r``; the filleted fluid is
  the two half-spaces and, in the cone at ``P`` spanned by ``-n_1`` and ``-n_2``, everything farther than ``r`` from
  ``P``. In the fluid the depth is the depth in the widened half-spaces (the distance to the faces ``n_i . x = r``,
  each where its foot lies on the face, else to ``P``) less ``r``; in the solid the distance is the least of the
  distances to the two half-spaces and, in the cone, ``r - |x - P|``;
- *two balls*: the union of two overlapping balls, axisymmetric, so the full plane through their centres holds it: the
  same with the widened circles' crossing ``P`` (radii ``R_i + r``), the cone at ``P`` towards the two centres, and the
  depth in the widened union from the arcs of each widened circle that lie outside the other (a point's distance to an
  arc is its distance to the circle where its direction from the centre falls on the arc, else to ``P``)."""

from __future__ import annotations

import math
import os

import numpy as np
from scipy.spatial import cKDTree

from mime.scene import TubeTree

#: The parent's diameter of the gates' vessels, m: a millimetre vessel.
D = 1e-3
#: The daughters' angle from the parent's axis.
ANGLE = math.radians(30.0)
#: How far from the junction the gates' points lie, in diameters.
REACH = 1.3


def seed_list(fixed: tuple[int, ...]) -> tuple[int, ...]:
    """The draws' seeds: ``DETECTION_SEED`` alone where the detection-rate harness sets it, else ``fixed``."""
    held = os.environ.get("DETECTION_SEED")
    return (int(held),) if held else fixed


def rotation(seed: int) -> np.ndarray:
    """A uniform random rotation (a unit quaternion of four normal draws), never the identity in practice."""
    q = np.random.default_rng(seed).normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                       [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                       [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def y_tree(kind: str, fillet: float = 0.0) -> TubeTree:
    """A Y meeting at the origin, its parent along -z: ``spike`` (radii D/2, segments 3 D, as F1a's spike) or
    ``asymmetric`` (radii 0.5, 0.4 and 0.3 D, segments 2 D, the first step's loop's daughters)."""
    radii, length = {"spike": ((0.5, 0.5, 0.5), 3.0), "asymmetric": ((0.5, 0.4, 0.3), 2.0)}[kind]
    s, c = math.sin(ANGLE) * length * D, math.cos(ANGLE) * length * D
    return TubeTree.si(points=((0.0, 0.0, -length * D), (0.0, 0.0, 0.0), (s, 0.0, c), (-s, 0.0, c)),
                    segments=((0, 1), (1, 2), (1, 3)),
                    radii=tuple((r * D, r * D) for r in radii), fillet=fillet)


def loop_tree(fillet: float = 0.0) -> TubeTree:
    """The first step's split-and-merge loop: a parent of D splits at J1 (the origin) into daughters of 0.4 and 0.3 D at
    +-30 degrees, straight for 0.8 D, which merge again at J2."""
    s, c, middle = math.sin(ANGLE) * D, math.cos(ANGLE) * D, 0.8 * D
    top = 2.0 * c + middle
    points = ((0.0, 0.0, -2.0 * D), (0.0, 0.0, 0.0), (s, 0.0, c), (-s, 0.0, c), (s, 0.0, c + middle),
              (-s, 0.0, c + middle), (0.0, 0.0, top), (0.0, 0.0, top + 2.0 * D))
    segments = ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (6, 7))
    radii = (0.5, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.5)
    return TubeTree.si(points, segments, tuple((r * D, r * D) for r in radii), fillet)


def hairpin_tree(wall: float, fillet: float = 0.0, length: float = 3.0) -> TubeTree:
    """Two parallel legs of radius D/2 and ``length`` D along z, their walls ``wall`` apart (below zero: overlapping),
    joined at the top by a third tube: a hairpin in the xz-plane, its legs sharing no point."""
    s = D + wall
    return TubeTree.si(((0.0, 0.0, 0.0), (0.0, 0.0, length * D), (s, 0.0, length * D), (s, 0.0, 0.0)),
                    ((0, 1), (1, 2), (2, 3)), ((0.5 * D, 0.5 * D),) * 3, fillet)


def aligned_rotation(seed: int) -> np.ndarray:
    """One of the 24 rotations that take the world's axes to its axes (a signed permutation of determinant one)."""
    rng = np.random.default_rng(seed)
    while True:
        turn = np.eye(3)[rng.permutation(3)] * rng.choice((-1.0, 1.0), 3)[:, None]
        if np.linalg.det(turn) > 0.0:
            return turn


def lattice(centre: np.ndarray, reach: float, spacing: float) -> np.ndarray:
    """The points of the world-aligned lattice of ``spacing`` through the world's origin within ``reach`` of
    ``centre``: a vessel moved by a sub-cell offset sits at that offset on it."""
    count = int(math.ceil(reach / spacing)) + 1
    axis = np.arange(-count, count + 1) * spacing
    base = np.round(np.asarray(centre) / spacing) * spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3) + base
    return grid[np.linalg.norm(grid - centre, axis=1) <= reach]


def capsule_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """A capsule's signed distance, in NumPy: the distance from its segment less its radius."""
    d = b - a
    t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1) - radius


def capsule_samples(a: np.ndarray, b: np.ndarray, radius: float, spacing: float) -> np.ndarray:
    """A capsule's surface sampled about ``spacing`` apart: rings along its cylinder, rings over its caps."""
    length = float(np.linalg.norm(b - a))
    u = (b - a) / length
    helper = np.eye(3)[int(np.argmin(np.abs(u)))]
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    around = int(math.ceil(2.0 * math.pi * radius / spacing))
    theta = 2.0 * math.pi * np.arange(around) / around
    ring = radius * (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2)
    along = np.linspace(0.0, length, int(math.ceil(length / spacing)) + 1)
    found = [(a[None, None] + along[:, None, None] * u + ring[None]).reshape(-1, 3)]
    polar = int(math.ceil(0.5 * math.pi * radius / spacing))
    for centre, sign in ((a, -1.0), (b, 1.0)):
        for k in range(1, polar + 1):
            angle = 0.5 * math.pi * k / polar
            across = radius * math.cos(angle)
            count = max(1, int(math.ceil(2.0 * math.pi * across / spacing)))
            phi = 2.0 * math.pi * np.arange(count) / count
            found.append(centre + sign * radius * math.sin(angle) * u
                         + across * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
    return np.concatenate(found)


def capsules(tree: TubeTree) -> list[tuple[np.ndarray, np.ndarray, float]]:
    points = tree.parameters()["points"]
    radii = tree.parameters()["radii"]
    assert np.all(radii[:, 0] == radii[:, 1]), "the brute force samples capsules: one radius a segment"
    return [(points[a], points[b], float(radii[i, 0])) for i, (a, b) in enumerate(tree.segments)]


def plain_minimum(tree: TubeTree, points: np.ndarray) -> np.ndarray:
    return np.min([capsule_distance(points, a, b, r) for a, b, r in capsules(tree)], axis=0)


def brute_distance(tree: TubeTree, points: np.ndarray, spacing: float, near: np.ndarray, reach: float) -> np.ndarray:
    """The union's signed distance at ``points`` from its boundary sampled ``spacing`` apart within ``reach`` of
    ``near``."""
    held = capsules(tree)
    kept = []
    for index, (a, b, r) in enumerate(held):
        samples = capsule_samples(a, b, r, spacing)
        samples = samples[np.linalg.norm(samples - near, axis=1) <= reach]
        outside = np.ones(len(samples), dtype=bool)
        for other, (c, d, s) in enumerate(held):
            if other != index:
                outside &= capsule_distance(samples, c, d, s) >= 0.0
        kept.append(samples[outside])
    distance, _ = cKDTree(np.concatenate(kept)).query(points)
    return np.where(plain_minimum(tree, points) < 0.0, -distance, distance)


def filleted_distance(tree: TubeTree, points: np.ndarray, spacing: float, near: np.ndarray,
                      reach: float) -> np.ndarray:
    """A filleted tree's signed distance where the brute force reaches it, NaN elsewhere: in the closing's fluid (depth
    in the union widened by the fillet's radius ``r`` at least ``r``) that depth less ``r``, from the widened tree's
    brute force; outside the widened union the union's own distance (the plain minimum, exact there); the solid within
    ``r`` of the widened union has no oracle here."""
    r = tree.fillet_radius_value()
    widened = TubeTree.si(tree.parameters()["points"], tree.segments, tree.parameters()["radii"] + r, 0.0)
    depth = brute_distance(widened, points, spacing, near, reach)
    return np.where(depth <= -r, depth + r, np.where(depth >= 0.0, plain_minimum(tree, points), np.nan))


def _in_cone(offset: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Whether each 2-D ``offset`` is a combination of ``first`` and ``second`` with both weights from zero on."""
    basis = np.stack([first, second], axis=1)
    weights = np.linalg.solve(basis, offset.T).T
    return np.all(weights >= 0.0, axis=1)


def wedge_closed_form(normals: tuple[np.ndarray, np.ndarray], origin: np.ndarray, fillet: float,
                      points: np.ndarray) -> np.ndarray:
    """The filleted union of the half-spaces ``n_i . (x - origin) <= 0`` (the module's docstring), at ``points``."""
    n1, n2 = normals
    edge = np.cross(n1, n2)
    edge /= np.linalg.norm(edge)
    e1 = n1 - np.dot(n1, edge) * edge
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(edge, e1)
    flat = np.stack([(points - origin) @ e1, (points - origin) @ e2], axis=1)
    m1, m2 = np.asarray([n1 @ e1, n1 @ e2]), np.asarray([n2 @ e1, n2 @ e2])
    corner = np.linalg.solve(np.stack([m1, m2]), np.asarray([fillet, fillet]))
    a1, a2 = flat @ m1, flat @ m2
    from_corner = np.linalg.norm(flat - corner, axis=1)
    cone = _in_cone(flat - corner, -m1, -m2)
    fluid = (a1 < 0.0) | (a2 < 0.0) | (cone & (from_corner > fillet))
    foot1 = flat + (fillet - a1)[:, None] * m1
    foot2 = flat + (fillet - a2)[:, None] * m2
    face1 = np.where(foot1 @ m2 >= fillet, np.abs(fillet - a1), np.inf)
    face2 = np.where(foot2 @ m1 >= fillet, np.abs(fillet - a2), np.inf)
    depth = np.minimum(np.minimum(face1, face2), from_corner)
    solid = np.minimum(np.minimum(a1, a2), np.where(cone, fillet - from_corner, np.inf))
    return np.where(fluid, fillet - depth, solid)


def two_balls_closed_form(centres: tuple[np.ndarray, np.ndarray], radii: tuple[float, float], fillet: float,
                          points: np.ndarray) -> np.ndarray:
    """The filleted union of two overlapping balls (the module's docstring), at ``points``."""
    c1, c2 = centres
    r1, r2 = radii
    apart = float(np.linalg.norm(c2 - c1))
    axis = (c2 - c1) / apart
    z = (points - c1) @ axis
    rho = np.linalg.norm(points - c1 - z[:, None] * axis, axis=1)
    flat = np.stack([z, rho], axis=1)
    o1, o2 = np.zeros(2), np.asarray([apart, 0.0])
    w1, w2 = r1 + fillet, r2 + fillet
    zp = (w1 * w1 - w2 * w2 + apart * apart) / (2.0 * apart)
    corner = np.asarray([zp, math.sqrt(w1 * w1 - zp * zp)])
    d1, d2 = np.linalg.norm(flat - o1, axis=1), np.linalg.norm(flat - o2, axis=1)
    from_corner = np.linalg.norm(flat - corner, axis=1)
    cone = _in_cone(flat - corner, o1 - corner, o2 - corner)
    fluid = (d1 < r1) | (d2 < r2) | (cone & (from_corner > fillet))
    on_arc1 = np.arctan2(flat[:, 1], flat[:, 0]) >= math.atan2(corner[1], corner[0])
    on_arc2 = np.arctan2(flat[:, 1], apart - flat[:, 0]) >= math.atan2(corner[1], apart - corner[0])
    arc1 = np.where(on_arc1, np.abs(w1 - d1), np.inf)
    arc2 = np.where(on_arc2, np.abs(w2 - d2), np.inf)
    depth = np.minimum(np.minimum(arc1, arc2), from_corner)
    solid = np.minimum(np.minimum(d1 - r1, d2 - r2), np.where(cone, fillet - from_corner, np.inf))
    return np.where(fluid, fillet - depth, solid)
