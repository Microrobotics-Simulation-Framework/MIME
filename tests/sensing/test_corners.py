"""Corner sub-samples: a cylinder seen side-on, its image the convex hull of its two caps' ellipses, whose corners
meet its sides nearly at right angles; each pixel's coverage against the hull's exact area in that pixel (the caps'
ellipses drawn with 4 000 points each, the hull clipped to each pixel's square), in both dtypes, through a pinhole
without distortion. With one straight edge a right-angled corner's pixel is off by up to 24 % (the design's fourth
review); with 4 x 4 sub-samples the gate's line is 15 %. The prediction was 3 % and missed: 13.8 % is measured, at the
caps' outlines, which this view sees within a pixel of edge-on (their ellipses about half a pixel across, where a cap's
edge function changes faster across the ellipse than a sub-pixel's straight line follows), so the gate also checks that
the view raises ``thin_cap``; the sides are within the line.

**The control** (``CAMERA_PLANT=no-corners``) renders without sub-samples: the corner pixels then miss the line, every
run."""

from __future__ import annotations

import numpy as np
from camera_cases import planted
from scipy.spatial import ConvexHull

import jax.numpy as jnp
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

WIDTH, HEIGHT, FOCAL, DISTANCE = 96, 64, 900.0, 30e-3
RADIUS, LENGTH = 0.31e-3, 2.07e-3
CENTRE = (0.113e-3, 0.0, 0.071e-3)


def oracle() -> np.ndarray:
    """The exact share of each pixel inside the cylinder's image."""
    phi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    points = []
    for end in (-LENGTH / 2, LENGTH / 2):
        x = CENTRE[0] + end + 0 * phi
        y = CENTRE[1] + RADIUS * np.cos(phi)
        z = CENTRE[2] + RADIUS * np.sin(phi)
        depth = y + DISTANCE
        points.append(np.stack([FOCAL * x / depth + (WIDTH - 1) / 2, FOCAL * -z / depth + (HEIGHT - 1) / 2], axis=-1))
    hull = np.concatenate(points)
    polygon = hull[ConvexHull(hull).vertices]
    shares = np.zeros((HEIGHT, WIDTH))
    for j in range(HEIGHT):
        for i in range(WIDTH):
            shares[j, i] = _clipped_area(polygon, i - 0.5, i + 0.5, j - 0.5, j + 0.5)
    return shares


def _clipped_area(polygon: np.ndarray, u0: float, u1: float, v0: float, v1: float) -> float:
    """The area of a convex polygon inside an axis-aligned square (Sutherland-Hodgman, then the shoelace)."""
    held = polygon
    for axis, bound, keep_below in ((0, u1, True), (0, u0, False), (1, v1, True), (1, v0, False)):
        if len(held) == 0:
            return 0.0
        out = []
        for a, b in zip(held, np.roll(held, -1, axis=0), strict=True):
            a_in = a[axis] <= bound if keep_below else a[axis] >= bound
            b_in = b[axis] <= bound if keep_below else b[axis] >= bound
            if a_in:
                out.append(a)
            if a_in != b_in:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append(a + t * (b - a))
        held = np.asarray(out)
    if len(held) < 3:
        return 0.0
    x, y = held[:, 0], held[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def test_corner_pixels_are_within_the_line(dtype):
    camera = Pinhole(WIDTH, HEIGHT, FOCAL, FOCAL, (WIDTH - 1) / 2, (HEIGHT - 1) / 2)
    s = np.sqrt(0.5)
    pose = CameraPose(jnp.asarray([0.0, -DISTANCE, 0.0]), jnp.asarray([s, -s, 0.0, 0.0]))
    body = Subject("rod", "cylinder", {"radius": RADIUS, "length": LENGTH, "axis": (1.0, 0.0, 0.0)},
                   appearance=Appearance.opaque(), moving=True)
    samples = 1 if planted("no-corners") else 4
    frame = render([body], camera, pose, {"rod": (jnp.asarray(CENTRE), jnp.asarray((1.0, 0.0, 0.0, 0.0)))},
                   corner_samples=samples)
    covered = 1.0 - np.asarray(frame.transmission, dtype=np.float64)
    exact = oracle()
    edge = (exact > 0) & (exact < 1)
    ends = np.zeros_like(edge)
    for end in (-LENGTH / 2, LENGTH / 2):
        u = FOCAL * (CENTRE[0] + end) / DISTANCE + (WIDTH - 1) / 2
        ends[:, max(0, int(u) - 2):int(u) + 3] = True
    worst_corner = float(np.max(np.abs(covered - exact)[edge & ends]))
    worst_side = float(np.max(np.abs(covered - exact)[edge & ~ends]))
    assert worst_corner <= 0.15, worst_corner
    assert worst_side <= 0.03, worst_side
    assert bool(frame.flags["thin_cap:rod"])
