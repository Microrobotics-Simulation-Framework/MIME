"""An exact oracle for convex bodies' images through a pinhole without distortion, written apart from ``mime.sensing``:
the image of a convex body is the convex hull of the images of any set whose hull is the body, so

- **a capped cylinder's** image is the hull of its two rim circles' images;
- **a capsule's** is the hull of its two end balls' silhouettes (each ball's tangent circle from the pinhole);

each circle drawn with ``POINTS`` points (a chord of the circle then lies within ``1 - cos(pi / POINTS)`` of its radius,
3e-7 at 4 000: under 1e-5 of a pixel here), the hull clipped to each pixel's square (Sutherland-Hodgman, then the
shoelace): each pixel's share inside the image.

The camera is the corner gate's: ``distance`` from the world's origin on -y, looking along +y, the image's x the world's
x and its y the world's -z (the world's ``(x, y, z)`` at depth ``y + distance``)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

POINTS = 4000


def project(points: np.ndarray, focal: float, distance: float, width: int, height: int) -> np.ndarray:
    depth = points[..., 1] + distance
    return np.stack([focal * points[..., 0] / depth + (width - 1) / 2,
                     -focal * points[..., 2] / depth + (height - 1) / 2], axis=-1)


def _across(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors across ``axis``."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _circle(centre: np.ndarray, normal: np.ndarray, radius: float) -> np.ndarray:
    first, second = _across(normal / np.linalg.norm(normal))
    phi = np.linspace(0.0, 2.0 * np.pi, POINTS, endpoint=False)[:, None]
    return centre + radius * (np.cos(phi) * first + np.sin(phi) * second)


def cylinder_outline(centre: np.ndarray, axis: np.ndarray, radius: float, length: float) -> list[np.ndarray]:
    """The rim circles of a capped cylinder (world points)."""
    return [_circle(centre + end * axis, axis, radius) for end in (-length / 2, length / 2)]


def capsule_outline(centre: np.ndarray, axis: np.ndarray, radius: float, height: float,
                    pinhole: np.ndarray) -> list[np.ndarray]:
    """The tangent circles, seen from ``pinhole``, of a capsule's two end balls (world points)."""
    circles = []
    for end in (-height / 2, height / 2):
        ball = centre + end * axis
        sight = ball - pinhole
        span = np.linalg.norm(sight)
        sine = radius / span
        circles.append(_circle(pinhole + sight * (1.0 - sine * sine), sight, radius * np.sqrt(1.0 - sine * sine)))
    return circles


def shares(outline: list[np.ndarray], focal: float, distance: float, width: int, height: int) -> np.ndarray:
    """Each pixel's share inside the hull of the outline's images."""
    hull = ConvexHull(np.concatenate([project(p, focal, distance, width, height) for p in outline]))
    polygon = hull.points[hull.vertices]
    v, u = np.mgrid[0:height + 1, 0:width + 1] - 0.5
    corners = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=-1) @ hull.equations.T
    inside = np.all(corners <= 0.0, axis=-1).reshape(u.shape)
    full = inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, :-1] & inside[1:, 1:]
    out = full.astype(np.float64)
    low, high = np.floor(polygon.min(axis=0)).astype(int) - 1, np.ceil(polygon.max(axis=0)).astype(int) + 1
    for j in range(max(0, low[1]), min(height, high[1] + 1)):
        # The row's strip first (the hull's few vertices in it), then each pixel of the row from the strip.
        strip = _clip(_clip(polygon, 1, j + 0.5, True), 1, j - 0.5, False)
        if len(strip) < 3:
            continue
        for i in range(max(0, low[0]), min(width, high[0] + 1)):
            if not full[j, i]:
                out[j, i] = clipped_area(strip, i - 0.5, i + 0.5, j - 0.5, j + 0.5)
    return out


def _clip(polygon: np.ndarray, axis: int, bound: float, keep_below: bool) -> np.ndarray:
    """A convex polygon clipped to one side of ``x[axis] = bound`` (Sutherland-Hodgman's one step)."""
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        a_in = a[axis] <= bound if keep_below else a[axis] >= bound
        b_in = b[axis] <= bound if keep_below else b[axis] >= bound
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (bound - a[axis]) / (b[axis] - a[axis])
            out.append(a + t * (b - a))
    return np.asarray(out)


def clipped_area(polygon: np.ndarray, u0: float, u1: float, v0: float, v1: float) -> float:
    """The area of a convex polygon inside an axis-aligned square."""
    held = polygon
    for axis, bound, keep_below in ((0, u1, True), (0, u0, False), (1, v1, True), (1, v0, False)):
        if len(held) == 0:
            return 0.0
        held = _clip(held, axis, bound, keep_below)
    if len(held) < 3:
        return 0.0
    x, y = held[:, 0], held[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def minor_axis(rim_centre: np.ndarray, axis: np.ndarray, radius: float, focal: float, distance: float) -> float:
    """A rim's image's width across its ellipse, in pixels (twice ``f R |cos|`` over its depth and distance, the
    cosine between the line of sight and the rim's normal)."""
    sight = rim_centre - np.array([0.0, -distance, 0.0])
    return float(2 * focal * radius * abs(sight @ axis) / (np.linalg.norm(sight) * (rim_centre[1] + distance)))
