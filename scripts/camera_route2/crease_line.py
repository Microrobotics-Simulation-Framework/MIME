"""The crease gate's lines, derived before any of the gate's draws from the scheme's own arithmetic on exact geometry:
numpy only, never the renderer, its code or the gate's draws. ``python scripts/camera_route2/crease_line.py [OUT.json]``
prints each term and the lines; ``tests/sensing/test_crease.py`` holds them.

**The scheme the lines bound** (camera PR 5b): a creased body on the sampled route, here a capped cylinder (its side
marched, its span read at the side's least, its caps and their balls exact), every pixel of a frame with a corner in view
split into 4 x 4 sub-cells, each cell's share the exact area of the composition of its pieces' straight lines (the caps,
each the greater of it and its ball, and the side and the span's intersection, in one union). Each sub-ray reads the
side at its own least, found from its pixel's held point as the march finds it (32 samples across the enclosing ball
widened by two footprints, the least sample's parabola, the descent at the march's spacing and the quartic about where
it stops), its line through ``jvp`` with that place held; the span at the same place, its derivative through the
place's implicit step with the profile's second difference a spacing wide; the caps and balls in closed form.

**What is simulated here** is that scheme, re-implemented: the march and its refinement, each piece's value and its
derivative along the ray table (central differences at ``DELTA`` of a pixel, the direction moved and not renormalised, as
``jvp`` takes it), each line, and each cell's exact area of the composition (Sutherland-Hodgman clipping of the cell by
each term's half-planes, then inclusion and exclusion). **The reference** is the exact image (the hull of the rims'
images, ``tests/sensing/hull_oracle.py``, written apart from ``mime.sensing``). The simulated error is then every term the
scheme has on this geometry at once: each line's departure from the outline in a cell (curvature, a thin cap's turn
within a sub-cell), the place the march reads the side and the span at, and the composition.

**The scan** is the gate's domain (``test_crease.py``'s draws): the rod turned 0 to 30 degrees towards the line of sight
(its caps opening from edge-on), rolled 0 to 90 degrees in the image, moved by a fraction of a pixel; caps narrower than two
sub-pixels raise ``thin_cap`` and the pixels about them are judged by the flag, as route 1's gate judges them
(``test_corners.py``). **The pixel line** is ``SAFETY`` times the scan's largest error, plus float32's rounding of a
sub-ray's point (camera PR 5a's term) in float32. **The motion line** is the same for the silhouette's motion: the secant
over one pixel of the body's motion along x and z of the darkness's equivalent radius (``sqrt(area / pi)``, camera spike
1's metric) and its centroid, the relative norm of the scheme's against the exact one's, ``SAFETY`` times the scan's
largest.

**What the scan bounds and what the factor is.** The scan's largest is measured: the scheme's error at its 360 poses, by
this exact numpy re-implementation (equal to the renderer's to 1e-11 where both were run). ``SAFETY`` (1.25) is
**declared, not derived**: a margin for the poses between the scan's points, which the gate's drawn views test. The scan
at half its density is reported beside it, but its largest falls at the same points, so it does not test the factor.
The factor is fixed before the drawn views and never moved after them."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "sensing"))
import hull_oracle  # noqa: E402

WIDTH, HEIGHT, FOCAL, DISTANCE = 96, 64, 900.0, 30e-3
RADIUS, LENGTH = 0.31e-3, 2.07e-3
CENTRE = np.array((0.113e-3, 0.0, 0.071e-3))
#: The scheme's numbers, as ``mime.sensing`` declares them (camera PR 5a, read here as numbers, not imported).
CORNER_SAMPLES, MARGIN_PX, SAMPLES, LEAST_REACH, LEAST_STEPS = 4, 2.0, 32, 6, 2
#: The central difference's step, in pixels, for each piece's derivative along the ray table.
DELTA = 1e-4
#: The margin on the scan's largest error for the poses between its points: declared, not derived (the drawn views
#: test it); never moved after them.
SAFETY = 1.25
TURNS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0)
ROLLS = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)
OFFSETS = ((0.0, 0.0), (0.5, 0.5), (0.3, 0.8))
PIXEL = DISTANCE / FOCAL
#: Processes the scan runs on.
WORKERS = 4


def axis_of(turn: float, roll: float) -> np.ndarray:
    a, r = math.radians(turn), math.radians(roll)
    return np.array([math.cos(a) * math.cos(r), math.sin(a), math.cos(a) * math.sin(r)])


def rays(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit directions through image points, and their derivatives along u and v (the pinhole at -y looking +y, the
    image's u the world's x and its v the world's -z), as the ray table gives them."""
    x, z = (u - (WIDTH - 1) / 2) / FOCAL, -(v - (HEIGHT - 1) / 2) / FOCAL
    raw = np.stack([x, np.ones_like(x), z], -1)
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    d = raw / norm
    e_u, e_v = np.array([1.0 / FOCAL, 0.0, 0.0]), np.array([0.0, 0.0, -1.0 / FOCAL])
    du = e_u / norm - d * (d @ e_u)[..., None] / norm
    dv = e_v / norm - d * (d @ e_v)[..., None] / norm
    return d, du, dv


class Rod:
    """The cylinder's pieces in its own frame (centred, ``axis`` unit), on rays ``o + t d`` (``o`` one point)."""

    def __init__(self, axis: np.ndarray, centre: np.ndarray):
        self.axis, self.o = axis, np.array([0.0, -DISTANCE, 0.0]) - centre
        self.enclosing = math.hypot(RADIUS, LENGTH / 2)

    def side(self, point: np.ndarray) -> np.ndarray:
        along = point @ self.axis
        return np.linalg.norm(point - along[..., None] * self.axis, axis=-1) - RADIUS

    def span(self, point: np.ndarray) -> np.ndarray:
        return np.abs(point @ self.axis) - LENGTH / 2

    def cap(self, d: np.ndarray, z: float) -> np.ndarray:
        o, a = self.o, self.axis
        o_a, d_a = o @ a, d @ a
        height = z - o_a
        v = d_a[..., None] * o + height * d - (z * d_a)[..., None] * a
        facing = np.where(height < 0.0, -d_a, d_a)
        return np.linalg.norm(v, axis=-1) - RADIUS * facing

    def ball(self, d: np.ndarray, z: float) -> np.ndarray:
        w = self.o - z * self.axis
        along = d @ w
        return np.linalg.norm(w - along[..., None] * d, axis=-1) - RADIUS


def march(rod: Rod, d: np.ndarray, du: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pixel rays' held least (the least sample's parabola), the samples' spacing and whether the least is interior,
    as ``_sampled.march`` holds them."""
    o = rod.o
    b = d @ o
    footprint = np.sqrt((o @ o) * np.sum(du * du, -1))
    reach = rod.enclosing + MARGIN_PX * footprint
    half = np.sqrt(np.maximum(b * b - (o @ o - reach * reach), 0.0))
    t0 = np.maximum(-b - half, 0.0)
    t1 = np.maximum(-b + half, t0)
    dt = (t1 - t0) / (SAMPLES - 1)
    t = t0[:, None] + dt[:, None] * np.arange(SAMPLES)
    v = rod.side(o + t[..., None] * d[:, None, :])
    first = np.argmin(v, axis=1)
    near = [np.take_along_axis(v, np.clip(first + k, 0, SAMPLES - 1)[:, None], 1)[:, 0] for k in (-1, 0, 1)]
    curvature = near[0] - 2 * near[1] + near[2]
    inner = (first > 0) & (first < SAMPLES - 1) & (curvature > 0)
    shift = np.where(inner, 0.5 * (near[0] - near[2]) / np.where(inner, curvature, 1.0), 0.0)
    return t0 + (first + shift) * dt, dt, (first > 0) & (first < SAMPLES - 1)


def quartic_least(fl2, fl1, f0, fh1, fh2):
    """``_sampled._quartic_least``, re-implemented."""
    curvature = fl2 - 2 * f0 + fh2
    rising = curvature > 0
    at = np.where(rising, np.clip((fl2 - fh2) / np.where(rising, curvature, 1.0), -2, 2), 0.0)
    c1 = (fl2 - 8 * fl1 + 8 * fh1 - fh2) / 12
    c2 = (-fl2 + 16 * fl1 - 30 * f0 + 16 * fh1 - fh2) / 24
    c3 = (-fl2 + 2 * fl1 - 2 * fh1 + fh2) / 12
    c4 = (fl2 - 4 * fl1 + 6 * f0 - 4 * fh1 + fh2) / 24
    for _ in range(LEAST_STEPS):
        slope = c1 + at * (2 * c2 + at * (3 * c3 + at * 4 * c4))
        turn = 2 * c2 + at * (6 * c3 + at * 12 * c4)
        convex = turn > 0
        at = np.clip(at + np.where(convex, np.clip(-slope / np.where(convex, turn, 1.0), -1, 1), 0.0), -2, 2)
    fitted = f0 + at * (c1 + at * (c2 + at * (c3 + at * c4)))
    inner = np.where(fl1 < fh1, -1.0, 1.0)
    return np.where(fitted <= np.minimum(f0, np.minimum(fl1, fh1)), at, np.where(f0 <= np.minimum(fl1, fh1), 0.0, inner))


def least(rod: Rod, d: np.ndarray, start: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A sub-ray's least from its pixel's held point: ``_sampled._descended`` then the quartic."""
    def at(t):
        return rod.side(rod.o + t[..., None] * d)

    before, middle, after = at(start - h), at(start), at(start + h)
    toward = np.where(before < after, -1.0, 1.0)
    path = [np.where(toward < 0, after, before), middle, np.where(toward < 0, before, after)]
    path += [at(start + toward * k * h) for k in range(2, LEAST_REACH + 2)]
    centre, value, behind, ahead = start.copy(), middle, path[0], path[2]
    going = np.ones_like(middle, dtype=bool)
    for k in range(1, LEAST_REACH + 1):
        going = going & (path[k + 1] < path[k])
        centre = np.where(going, start + toward * k * h, centre)
        value, behind, ahead = (np.where(going, path[k + 1], value), np.where(going, path[k], behind),
                                np.where(going, path[k + 2], ahead))
    below, above = np.where(toward < 0, ahead, behind), np.where(toward < 0, behind, ahead)
    return centre + 0.5 * quartic_least(below, at(centre - 0.5 * h), value, at(centre + 0.5 * h), above) * h


def lines(rod: Rod, d: np.ndarray, du: np.ndarray, dv: np.ndarray, held: np.ndarray, h: np.ndarray,
          slope0: np.ndarray, curvature: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Each piece's line (distance in pixels from the sub-ray, positive outside; unit normal along u and v)."""
    o = rod.o

    def values(direction):
        e = 1e-7 * h
        slope = (rod.side(o + (held + e)[..., None] * direction)
                 - rod.side(o + (held - e)[..., None] * direction)) / (2 * e)
        place = held - (slope - slope0) / curvature
        place = np.where(curvature > 0, place, held)
        point = o + held[..., None] * direction
        return [rod.side(point), rod.span(o + place[..., None] * direction),
                rod.cap(direction, -LENGTH / 2), rod.cap(direction, LENGTH / 2),
                rod.ball(direction, -LENGTH / 2), rod.ball(direction, LENGTH / 2)]

    centre = values(d)
    plus_u, minus_u = values(d + DELTA * du), values(d - DELTA * du)
    plus_v, minus_v = values(d + DELTA * dv), values(d - DELTA * dv)
    found = []
    for g, pu, mu, pv, mv in zip(centre, plus_u, minus_u, plus_v, minus_v, strict=True):
        gu, gv = (pu - mu) / (2 * DELTA), (pv - mv) / (2 * DELTA)
        length = np.sqrt(np.maximum(gu * gu + gv * gv, 1e-30))
        found.append((g / length, gu / length, gv / length))
    return found


def clipped_area(nu: np.ndarray, nv: np.ndarray, c: np.ndarray, size: float) -> np.ndarray:
    """The area of a square ``size`` a side, centred at the origin, where ``nu x + nv y <= c`` for every half-plane
    (arrays ``(cells, planes)``), by Sutherland-Hodgman clipping vectorised over cells."""
    half = size / 2
    poly = np.tile(np.array([[-half, -half], [half, -half], [half, half], [-half, half]]), (nu.shape[0], 1, 1))
    for k in range(nu.shape[1]):
        a, b, cc = nu[:, k:k + 1], nv[:, k:k + 1], c[:, k:k + 1]
        p, q = poly, np.roll(poly, -1, axis=1)
        fp, fq = a * p[..., 0] + b * p[..., 1] - cc, a * q[..., 0] + b * q[..., 1] - cc
        keep_p, cross = fp <= 0, (fp <= 0) != (fq <= 0)
        w = fp / np.where(cross, fp - fq, 1.0)
        meet = p + w[..., None] * (q - p)
        slots = np.stack([p, meet], axis=2).reshape(poly.shape[0], -1, 2)
        valid = np.stack([keep_p, cross], axis=2).reshape(poly.shape[0], -1)
        order = np.argsort(~valid, axis=1, kind="stable")
        slots = np.take_along_axis(slots, order[..., None], 1)
        counts = valid.sum(1)
        index = np.minimum(np.arange(slots.shape[1])[None, :], np.maximum(counts - 1, 0)[:, None])
        slots = np.take_along_axis(slots, index[..., None], 1)
        poly = np.where((counts > 0)[:, None, None], slots, 0.0)
    x, y = poly[..., 0], poly[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - y * np.roll(x, -1, 1), axis=1))


def composed(lines_in: list, terms: tuple[tuple[int, ...], ...], size: float) -> np.ndarray:
    """A cell's exact share of the union of the terms, each an intersection of the lines' bodies."""
    total = 0.0
    for count in range(1, len(terms) + 1):
        for subset in itertools.combinations(terms, count):
            held = sorted({i for term in subset for i in term})
            nu = np.stack([lines_in[i][1] for i in held], 1)
            nv = np.stack([lines_in[i][2] for i in held], 1)
            c = -np.stack([lines_in[i][0] for i in held], 1)
            total = total + (1 if count % 2 else -1) * clipped_area(nu, nv, c, size)
    return total / size ** 2


#: The cylinder's composition, the pieces' order as ``lines`` gives them: the side and span, each cap and its ball.
TERMS = ((0, 1), (2, 4), (3, 5))


def image(turn: float, roll: float, offset: tuple[float, float], shift: np.ndarray = np.zeros(3)
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The scheme's image, the exact image and the judged mask (not about a thin cap)."""
    axis = axis_of(turn, roll)
    centre = CENTRE + np.array([offset[0], 0.0, -offset[1]]) * PIXEL + shift
    rims = hull_oracle.cylinder_outline(centre, axis, RADIUS, LENGTH)
    exact = hull_oracle.shares(rims, FOCAL, DISTANCE, WIDTH, HEIGHT)
    rod = Rod(axis, centre)
    v_index, u_index = np.mgrid[0:HEIGHT, 0:WIDTH]
    band = np.zeros_like(exact, dtype=bool)
    for dj in range(-2, 3):
        for di in range(-2, 3):
            shifted = np.roll(np.roll((exact > 0) & (exact < 1), dj, 0), di, 1)
            band |= shifted
    band |= (exact > 0) & (exact < 1)
    pick = np.nonzero(band.ravel())[0]
    u, v = u_index.ravel()[pick].astype(float), v_index.ravel()[pick].astype(float)
    d, du, dv = rays(u, v)
    held, h, interior = march(rod, d, du)
    share = np.zeros(len(pick))
    size = 1.0 / CORNER_SAMPLES
    for i in range(CORNER_SAMPLES):
        for j in range(CORNER_SAMPLES):
            su, sv = (i + 0.5) / CORNER_SAMPLES - 0.5, (j + 0.5) / CORNER_SAMPLES - 0.5
            moved = d + su * du + sv * dv
            sub = moved / np.linalg.norm(moved, axis=-1, keepdims=True)
            place = least(rod, sub, held, h)
            e = 1e-7 * h

            def side_slope(t, direction=sub):
                return (rod.side(rod.o + (t + e)[..., None] * direction)
                        - rod.side(rod.o + (t - e)[..., None] * direction)) / (2 * e)

            curvature = (rod.side(rod.o + (place + h)[..., None] * sub) - 2 * rod.side(rod.o + place[..., None] * sub)
                         + rod.side(rod.o + (place - h)[..., None] * sub)) / (h * h)
            found = lines(rod, sub, du, dv, place, h, side_slope(place), curvature)
            # A least at the samples' ends: the side is not there (``_sampled._creased_pieces``).
            g, nu, nv = found[0]
            found[0] = (np.where(interior, g, 1e9), nu, nv)
            # The renderer's lines in a cell are clamped at UNION_REACH cells: the same area.
            share += composed(found, TERMS, size) / CORNER_SAMPLES ** 2
    scheme = exact.copy()
    scheme.ravel()[pick] = share
    judged = np.ones_like(exact, dtype=bool)
    for end, rim in zip((-LENGTH / 2, LENGTH / 2), rims, strict=True):
        if hull_oracle.minor_axis(centre + end * axis, axis, RADIUS, FOCAL, DISTANCE) < 2.0 / CORNER_SAMPLES:
            projected = hull_oracle.project(rim, FOCAL, DISTANCE, WIDTH, HEIGHT)
            low = np.floor(projected.min(axis=0)).astype(int) - 2
            high = np.ceil(projected.max(axis=0)).astype(int) + 2
            judged[max(0, low[1]):high[1] + 1, max(0, low[0]):high[0] + 1] = False
    return scheme, exact, judged


def aggregates(covered: np.ndarray) -> np.ndarray:
    v, u = np.indices(covered.shape)
    total = covered.sum()
    return np.array([math.sqrt(total / math.pi), (covered * u).sum() / total, (covered * v).sum() / total])


def motion_error(turn: float, roll: float, offset: tuple[float, float], base: tuple) -> float:
    """The relative norm of the scheme's secant against the exact one's over one pixel of motion along x and z."""
    worst = 0.0
    scheme0, exact0, _ = base
    for step in (np.array([PIXEL, 0.0, 0.0]), np.array([0.0, 0.0, PIXEL])):
        scheme1, exact1, _ = image(turn, roll, offset, step)
        moved_scheme = aggregates(scheme1) - aggregates(scheme0)
        moved_exact = aggregates(exact1) - aggregates(exact0)
        worst = max(worst, float(np.linalg.norm(moved_scheme - moved_exact) / np.linalg.norm(moved_exact)))
    return worst


def rounding_term(dtype: str) -> float:
    """Float32's rounding of a sub-ray's point (camera PR 5a's term), in a pixel's share."""
    unit = 2.0 ** -24 if dtype == "float32" else 0.0
    return 2.0 * math.sqrt(2.0) * 5.0 * unit * DISTANCE / PIXEL


def one_pose(pose: tuple) -> dict:
    turn, roll, offset = pose
    base = image(turn, roll, offset)
    scheme, exact, judged = base
    row = {"turn": turn, "roll": roll, "offset": list(offset), "pixel": float(np.max(np.abs(scheme - exact)[judged])),
           "motion": motion_error(turn, roll, offset, base)}
    print(json.dumps(row), flush=True)
    return row


def scan(turns, rolls, offsets) -> tuple[float, float, list]:
    """Every pose of the scan, over ``WORKERS`` processes."""
    from multiprocessing import Pool

    with Pool(WORKERS) as pool:
        rows = pool.map(one_pose, list(itertools.product(turns, rolls, offsets)), chunksize=1)
    return max(r["pixel"] for r in rows), max(r["motion"] for r in rows), rows


def main() -> None:
    pixel, motion, rows = scan(TURNS, ROLLS, OFFSETS)
    coarse = [r for r in rows if r["turn"] in TURNS[::2] and r["roll"] in ROLLS[::2]]
    report = {
        "scan": {"turns": TURNS, "rolls": ROLLS, "offsets": OFFSETS, "poses": len(rows)},
        "largest_pixel_error": pixel, "largest_motion_error": motion,
        "half_density": {"largest_pixel_error": max(r["pixel"] for r in coarse),
                         "largest_motion_error": max(r["motion"] for r in coarse)},
        "safety": SAFETY,
        "pixel_line": {dtype: SAFETY * pixel + rounding_term(dtype) for dtype in ("float64", "float32")},
        "motion_line": SAFETY * motion,
        "worst_rows": sorted(rows, key=lambda r: -r["pixel"])[:5],
        "worst_motion_rows": sorted(rows, key=lambda r: -r["motion"])[:5],
    }
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
