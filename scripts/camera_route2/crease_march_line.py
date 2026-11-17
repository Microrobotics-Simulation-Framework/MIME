"""The crease gate's derivative line's march term (G17), derived before any renderer measurement of it from the
scheme's own arithmetic on exact geometry: numpy only, never the renderer, its code or the gate's views.
``python scripts/camera_route2/crease_march_line.py [OUT.json]`` prints each pose's term and the scan's band;
``--poses '[[turn, roll, du, dv], ...]'`` prints the term at the poses given.

**What the term is.** The forward-mode derivative of a creased body's image on the sampled route holds the march: each
sub-ray's least (the march's parabola, the descent and the quartic about where it stops) is a held point, read with no
derivative, and the span is read at that place moved by one implicit step whose curvature is the profile's second
difference a march spacing wide, held. The rendered image moves those held points with the body, so its central
difference sees what the jvp leaves out: the image's sensitivity to each held place times the place's motion. Where the
held place is the side's exact least the side's value does not move with it (its slope along the ray is zero there)
and the implicit step is the place's exact motion; the term is what the quartic's least lies off the profile's and the
second difference off its curvature, carried through each line into the composed share. It is first order in those
errors, which are the march's own (its spacing, the descent's grid, the quartic's fit), so it is a property of the
scheme at each pose, computed here, not a number of the gate's.

**How it is computed.** ``crease_line.py``'s re-implementation of the scheme (equal to the renderer's image to 1e-11
where both ran), extended with the two derivatives of its image in the body's place along x and z, each a central
difference over ``EPSILON`` of a pixel:

- **held**, the jvp's: each pixel's march (its held point, spacing and interior) and each sub-ray's least, its slope
  along the ray and its second difference are the view's own; only the body moves under them, the span's place moving
  by the implicit step with the view's curvature;
- **live**, the rendered image's: everything recomputed at the moved place but the scheme's discrete choices (the
  march's least sample and its parabola's guard, the descent's direction and moves, the quartic's branch), held at the
  view's, so that the difference is the image's derivative on its branch and never a select's jump.

The side's slope along the ray is in closed form here (its first-order distance's derivative, which is what the jvp
takes), so that the held arm's implicit step carries no rounding of a difference of slopes. The term is the held
derivative less the live one, of the test's aggregates (the darkness's equivalent radius and centroid) on the pixels
within two of the caps' rims' images and on the rest apart: its norm, in pixels a pixel of the body's motion (a share
of the body's motion, not of a set's aggregates': a set the motion leaves still, the rest at an edge-on rod moved along
its axis, has no motion of its own to be a share of).

**The band** the script reports is ``SAFETY`` (1.25, declared, not derived, as the crease line's factor is) times the
scan's largest norm, for each set of pixels, over the crease line's 360 poses and both directions. It was written to
replace camera PR 5a's measured 3e-4 (``ENVELOPE_BAND``), which a second form of the gate had borrowed after its first
form, the central difference's own error alone, failed: the jvp lay 2.56 and 3.44 times that error from the difference
at turns 4 and 15 (roll 20), the held march's error that line left out.

**What the scan found** (``tests/sensing/crease_march_line_pr5b.json``, 2026-11-16, before any renderer measurement of
the derivative after it): on 357 of the 360 poses the term is at most 9.8e-5 px a px on the crease pixels (median
2.3e-6) and 3.3e-5 on the rest; at three it is 0.024, 0.030 and 0.50 on the crease pixels, all of it at one pixel, where
a sub-ray passes within a few thousandths of a pixel of the side's axis at the rod's end, the cap's outline seen nearly
edge-on. There the span's line is ill-conditioned (its distance moves some 80 px a px of the body's motion, in both
arms, the sub-cell's share some 240 a px), and the jvp's implicit step (the profile's second difference a spacing wide,
against its curvature at the least some 400 times greater) moves it at another rate than the rendered image's held point
does. The term there is narrow in the pose (0.50 at an offset of 0.3 px, 7e-6 at 0.302) and not bounded by the scan: a
fresh view nearer that alignment can read more. The scan's band (0.62 px a px on the crease pixels) is therefore no
bound of the term, and as a line it would pass any missing term of the jvp under about a tenth of the crease pixels'
motion; the gate's form is the maintainer's to choose (``predictions_camera_pr5b.json``,
``march_derivation_finding``).

**Since the fix** (option (c) with design (i), ``predictions_camera_pr5b.json``, ``held_point_check``): the ``held`` arm
here is the jvp as it was. Each read point now carries the derivative of the image's own held point
(``_sampled._place``), and the implicit step's slope and curvature are read at it as it moves (``_sampled._moving``):
the jvp's arm is this script's ``live`` arm, so the term is zero by construction, every pose, and the derivative's line
is the central difference's own error (``test_crease.derivative_line``). The scan above is kept as the record of what
the term was."""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "sensing"))
import crease_line as C  # noqa: E402
import hull_oracle  # noqa: E402

#: The body's move each side of the view in the central differences of the two arms, in pixels.
EPSILON = 1e-4
#: The margin on the scan's largest norm for the poses between its points: declared, not derived (as the crease
#: line's); never moved after the views that test it.
SAFETY = 1.25
MASKS = ("crease", "rest")


def side_slope(rod: C.Rod, t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The side's first-order distance's derivative along the rays ``o + t d``, in closed form."""
    point = rod.o + t[..., None] * d
    a = rod.axis
    across = point - (point @ a)[..., None] * a
    d_across = d - (d @ a)[..., None] * a
    return np.sum(across * d_across, -1) / np.linalg.norm(across, axis=-1)


def march(rod: C.Rod, d: np.ndarray, du: np.ndarray, choice: tuple | None = None) -> tuple:
    """``crease_line.march`` with its least sample and its parabola's guard returned, or held (``choice``)."""
    o = rod.o
    b = d @ o
    footprint = np.sqrt((o @ o) * np.sum(du * du, -1))
    reach = rod.enclosing + C.MARGIN_PX * footprint
    half = np.sqrt(np.maximum(b * b - (o @ o - reach * reach), 0.0))
    t0 = np.maximum(-b - half, 0.0)
    t1 = np.maximum(-b + half, t0)
    dt = (t1 - t0) / (C.SAMPLES - 1)
    t = t0[:, None] + dt[:, None] * np.arange(C.SAMPLES)
    v = rod.side(o + t[..., None] * d[:, None, :])
    first = np.argmin(v, axis=1) if choice is None else choice[0]
    near = [np.take_along_axis(v, np.clip(first + k, 0, C.SAMPLES - 1)[:, None], 1)[:, 0] for k in (-1, 0, 1)]
    curvature = near[0] - 2 * near[1] + near[2]
    inner = (first > 0) & (first < C.SAMPLES - 1) & (curvature > 0) if choice is None else choice[1]
    shift = np.where(inner, 0.5 * (near[0] - near[2]) / np.where(inner, curvature, 1.0), 0.0)
    return t0 + (first + shift) * dt, dt, (first > 0) & (first < C.SAMPLES - 1), (first, inner)


def quartic_least(fl2, fl1, f0, fh1, fh2, branch=None):
    """``crease_line.quartic_least`` with its final branch returned, or held (0 the quartic's least, 1 the middle,
    2 and 3 the inner sample below and above)."""
    curvature = fl2 - 2 * f0 + fh2
    rising = curvature > 0
    at = np.where(rising, np.clip((fl2 - fh2) / np.where(rising, curvature, 1.0), -2, 2), 0.0)
    c1 = (fl2 - 8 * fl1 + 8 * fh1 - fh2) / 12
    c2 = (-fl2 + 16 * fl1 - 30 * f0 + 16 * fh1 - fh2) / 24
    c3 = (-fl2 + 2 * fl1 - 2 * fh1 + fh2) / 12
    c4 = (fl2 - 4 * fl1 + 6 * f0 - 4 * fh1 + fh2) / 24
    for _ in range(C.LEAST_STEPS):
        slope = c1 + at * (2 * c2 + at * (3 * c3 + at * 4 * c4))
        turn = 2 * c2 + at * (6 * c3 + at * 12 * c4)
        convex = turn > 0
        at = np.clip(at + np.where(convex, np.clip(-slope / np.where(convex, turn, 1.0), -1, 1), 0.0), -2, 2)
    if branch is None:
        fitted = f0 + at * (c1 + at * (c2 + at * (c3 + at * c4)))
        branch = np.where(fitted <= np.minimum(f0, np.minimum(fl1, fh1)), 0,
                          np.where(f0 <= np.minimum(fl1, fh1), 1, np.where(fl1 < fh1, 2, 3)))
    return np.select([branch == 0, branch == 1, branch == 2], [at, 0.0, -1.0], 1.0), branch


def least(rod: C.Rod, d: np.ndarray, start: np.ndarray, h: np.ndarray, choice: tuple | None = None) -> tuple:
    """``crease_line.least`` (the descent, then the quartic) with its direction, moves and the quartic's branch
    returned, or held (``choice``)."""
    def at(t):
        return rod.side(rod.o + t[..., None] * d)

    if choice is None:
        before, middle, after = at(start - h), at(start), at(start + h)
        toward = np.where(before < after, -1.0, 1.0)
        path = [np.where(toward < 0, after, before), middle, np.where(toward < 0, before, after)]
        path += [at(start + toward * k * h) for k in range(2, C.LEAST_REACH + 2)]
        moves = np.zeros(start.shape)
        going = np.ones_like(middle, dtype=bool)
        for k in range(1, C.LEAST_REACH + 1):
            going = going & (path[k + 1] < path[k])
            moves = np.where(going, k, moves)
        branch = None
    else:
        toward, moves, branch = choice
    centre = start + toward * moves * h
    step, branch = quartic_least(at(centre - h), at(centre - 0.5 * h), at(centre), at(centre + 0.5 * h),
                                 at(centre + h), branch)
    return centre + 0.5 * step * h, (toward, moves, branch)


def crease_mask(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """``test_crease.crease_mask``: the pixels within two of the caps' rims' images."""
    mask = np.zeros((C.HEIGHT, C.WIDTH), dtype=bool)
    for rim in hull_oracle.cylinder_outline(centre, axis, C.RADIUS, C.LENGTH):
        projected = hull_oracle.project(rim, C.FOCAL, C.DISTANCE, C.WIDTH, C.HEIGHT)
        for u, v in np.round(projected).astype(int)[::20]:
            mask[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = True
    return mask


class View:
    """One pose's scheme: its band of pixels (within two of a partial one), and each pixel's and sub-ray's held
    state, from which either arm images the body moved by ``shift``."""

    def __init__(self, turn: float, roll: float, offset: tuple[float, float]):
        self.axis = C.axis_of(turn, roll)
        self.centre = C.CENTRE + np.array([offset[0], 0.0, -offset[1]]) * C.PIXEL
        exact = hull_oracle.shares(hull_oracle.cylinder_outline(self.centre, self.axis, C.RADIUS, C.LENGTH), C.FOCAL,
                                   C.DISTANCE, C.WIDTH, C.HEIGHT)
        self.outside = (exact >= 0.5).astype(float)
        partial = (exact > 0) & (exact < 1)
        band = partial.copy()
        for dj in range(-2, 3):
            for di in range(-2, 3):
                band |= np.roll(np.roll(partial, dj, 0), di, 1)
        self.pick = np.nonzero(band.ravel())[0]
        v_index, u_index = np.mgrid[0:C.HEIGHT, 0:C.WIDTH]
        u, v = u_index.ravel()[self.pick].astype(float), v_index.ravel()[self.pick].astype(float)
        self.d, self.du, self.dv = C.rays(u, v)
        self.subs = []
        for i in range(C.CORNER_SAMPLES):
            for j in range(C.CORNER_SAMPLES):
                su, sv = (i + 0.5) / C.CORNER_SAMPLES - 0.5, (j + 0.5) / C.CORNER_SAMPLES - 0.5
                moved = self.d + su * self.du + sv * self.dv
                self.subs.append(moved / np.linalg.norm(moved, axis=-1, keepdims=True))
        rod = C.Rod(self.axis, self.centre)
        held, h, interior, march_choice = march(rod, self.d, self.du)
        self.march = (held, h, interior, march_choice)
        self.held = []
        for sub in self.subs:
            place, choice = least(rod, sub, held, h)
            curvature = (rod.side(rod.o + (place + h)[..., None] * sub) - 2 * rod.side(rod.o + place[..., None] * sub)
                         + rod.side(rod.o + (place - h)[..., None] * sub)) / (h * h)
            self.held.append((place, choice, side_slope(rod, place, sub), curvature))

    def image(self, shift: np.ndarray, arm: str) -> np.ndarray:
        """The scheme's image with the body moved by ``shift`` (m): ``held`` (the jvp's) or ``live`` (the rendered
        image's, its discrete choices the view's)."""
        rod = C.Rod(self.axis, self.centre + shift)
        held, h, interior, march_choice = self.march
        if arm == "live":
            held, h, interior, _ = march(rod, self.d, self.du, march_choice)
        share = np.zeros(len(self.pick))
        size = 1.0 / C.CORNER_SAMPLES
        for sub, (place, choice, slope0, curvature) in zip(self.subs, self.held, strict=True):
            if arm == "live":
                place, _ = least(rod, sub, held, h, choice)
                slope0 = side_slope(rod, place, sub)
                curvature = (rod.side(rod.o + (place + h)[..., None] * sub)
                             - 2 * rod.side(rod.o + place[..., None] * sub)
                             + rod.side(rod.o + (place - h)[..., None] * sub)) / (h * h)
            found = lines(rod, sub, self.du, self.dv, place, slope0, curvature)
            g, nu, nv = found[0]
            found[0] = (np.where(interior, g, 1e9), nu, nv)
            share += C.composed(found, C.TERMS, size) / C.CORNER_SAMPLES ** 2
        covered = self.outside.copy()
        covered.ravel()[self.pick] = share
        return covered


def lines(rod: C.Rod, d: np.ndarray, du: np.ndarray, dv: np.ndarray, held: np.ndarray, slope0: np.ndarray,
          curvature: np.ndarray) -> list:
    """``crease_line.lines`` with the side's slope in closed form."""
    o = rod.o

    def values(direction):
        place = held - (side_slope(rod, held, direction) - slope0) / np.where(curvature > 0, curvature, 1.0)
        place = np.where(curvature > 0, place, held)
        return [rod.side(o + held[..., None] * direction), rod.span(o + place[..., None] * direction),
                rod.cap(direction, -C.LENGTH / 2), rod.cap(direction, C.LENGTH / 2),
                rod.ball(direction, -C.LENGTH / 2), rod.ball(direction, C.LENGTH / 2)]

    centre = values(d)
    plus_u, minus_u = values(d + C.DELTA * du), values(d - C.DELTA * du)
    plus_v, minus_v = values(d + C.DELTA * dv), values(d - C.DELTA * dv)
    found = []
    for g, pu, mu, pv, mv in zip(centre, plus_u, minus_u, plus_v, minus_v, strict=True):
        gu, gv = (pu - mu) / (2 * C.DELTA), (pv - mv) / (2 * C.DELTA)
        length = np.sqrt(np.maximum(gu * gu + gv * gv, 1e-30))
        found.append((g / length, gu / length, gv / length))
    return found


def aggregates(covered: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """The test's aggregates on each mask: the darkness's equivalent radius and centroid (u, v), in pixels."""
    v, u = np.indices(covered.shape)
    dark = covered[None] * masks
    total = dark.sum(axis=(1, 2))
    return np.stack([np.sqrt(total / math.pi), (dark * u).sum(axis=(1, 2)) / total,
                     (dark * v).sum(axis=(1, 2)) / total], axis=1)


def one_pose(pose: tuple, epsilon: float = EPSILON) -> dict:
    """The march term at one pose, along x and z, on each mask: the held derivative less the live one (pixels a pixel
    of the body's motion), and its norm."""
    turn, roll, offset = pose
    view = View(turn, roll, offset)
    crease = crease_mask(view.centre, view.axis)
    masks = np.stack([crease, ~crease]).astype(float)
    row: dict = {"turn": turn, "roll": roll, "offset": list(offset)}
    for name, index in (("x", 0), ("z", 2)):
        e = np.zeros(3)
        e[index] = epsilon * C.PIXEL
        derivative = {}
        for arm in ("held", "live"):
            derivative[arm] = (aggregates(view.image(e, arm), masks) - aggregates(view.image(-e, arm), masks)) \
                / (2 * epsilon)
        term = derivative["held"] - derivative["live"]
        for k, mask in enumerate(MASKS):
            row[f"{name}_{mask}"] = {"live": derivative["live"][k].tolist(), "term": term[k].tolist(),
                                     "norm": float(np.linalg.norm(term[k]))}
    row["norm"] = {mask: max(row[f"{name}_{mask}"]["norm"] for name in ("x", "z")) for mask in MASKS}
    print(json.dumps({"turn": turn, "roll": roll, "offset": list(offset), "norm": row["norm"]}), flush=True)
    return row


def convergence(pose: tuple) -> dict:
    """The term at one pose at a quarter and four times ``EPSILON``: the central differences' own check."""
    return {str(epsilon): one_pose(pose, epsilon)["norm"] for epsilon in (EPSILON / 4, EPSILON, 4 * EPSILON)}


def scan(poses: list) -> list:
    from multiprocessing import Pool

    with Pool(C.WORKERS) as pool:
        return pool.map(one_pose, poses, chunksize=1)


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--poses":
        poses = [(float(t), float(r), (float(a), float(b))) for t, r, a, b in json.loads(sys.argv[2])]
        rows = [one_pose(pose) for pose in poses]
        print(json.dumps(rows, indent=1))
        return
    started = time.time()
    rows = scan(list(itertools.product(C.TURNS, C.ROLLS, C.OFFSETS)))
    largest = {mask: max(r["norm"][mask] for r in rows) for mask in MASKS}
    worst = {mask: sorted(rows, key=lambda r: -r["norm"][mask])[0] for mask in MASKS}
    report = {
        "scan": {"turns": C.TURNS, "rolls": C.ROLLS, "offsets": C.OFFSETS, "poses": len(rows),
                 "epsilon_px": EPSILON, "seconds": round(time.time() - started)},
        "largest_norm_px_a_px": largest,
        "safety": SAFETY,
        "march_band": {mask: SAFETY * largest[mask] for mask in MASKS},
        "worst": {mask: {key: worst[mask][key] for key in ("turn", "roll", "offset", "norm")} for mask in MASKS},
        "convergence_at_worst": {mask: convergence((worst[mask]["turn"], worst[mask]["roll"],
                                                    tuple(worst[mask]["offset"]))) for mask in MASKS},
        "rows": [{key: r[key] for key in ("turn", "roll", "offset", "norm")} for r in rows],
    }
    print(json.dumps({key: report[key] for key in report if key != "rows"}, indent=1))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
