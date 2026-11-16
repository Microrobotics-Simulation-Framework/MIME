"""The creased image's own derivative error (G17's image check), derived before any fresh view from the scheme's own
arithmetic on exact geometry: numpy only, never the renderer. ``python scripts/camera_route2/crease_image_line.py
[OUT.json]`` scans the crease line's 360 poses; ``--sweep`` prints the singular pixels' error as the body moves their
nearest sub-ray off the side's axis.

**What it measures.** At each pose and along the body's x and z, each band pixel's (within two of a partial pixel of
the exact image) **image error**: the rendered image's derivative on its branch (``crease_march_line``'s ``live`` arm,
which the renderer's jvp now is: each read point carries the derivative of the image's own held point) less the exact
image's (``hull_oracle``'s central difference at ``EXACT_STEP``), in pixels of share a pixel of the body's motion. That
is the scheme's departure from the outline's motion, which no derivative fix changes: only the image can.

**The declared classes**, set apart before the scan ran and never moved:

- **singular**: a pixel one of whose sixteen sub-rays passes within ``NEAR`` (a hundredth of a pixel, at the body) of
  the side's axis line. At the rod's end, with the cap's outline seen nearly edge-on, the span is read at the side's
  least along such a sub-ray, where the side's profile is nearly a V (the ray's distance from the axis a small share of
  the march's spacing); the span's line is ill-conditioned there and the sub-cell's share turns within about a
  thousandth of a pixel of motion, a spurious sliver the exact image does not have (``crease_singular_check.py``: 15.1
  px a px against an exact 0). ``NEAR`` is the near-axis plant's reach, declared with that plant before any data, four
  times the farthest the march scan found its term (0.0025 px). Inside the rod the class's pixels are wholly covered
  (the radius is about 9.3 px), so it reaches the band only at the rod's ends.
- **thin**: the pixels about a cap narrower than two sub-pixels (G18's ``judged_pixels``), which the gates judge by the
  ``thin_cap`` flag: the image does not carry such a cap's outline.

**The line** is ``SAFETY`` (1.25, declared, not derived, as the crease line's) times the scan's largest image error at
the other band pixels, over both directions; the singular and thin classes' largest are reported beside it, never
judged. G17's image check fails a view when any band pixel outside the two classes departs from the exact image's
derivative by more than the line: the image error is carried by the check, confined to its declared class, and its
size there stated, never hidden in a band."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "sensing"))
import crease_line as C  # noqa: E402
import hull_oracle  # noqa: E402

#: The singular class's reach from the side's axis, in pixels at the body (``test_crease.NEAR_AXIS``).
NEAR = 0.01
#: The exact image's step in its central difference, in pixels (settled there in the singular check).
EXACT_STEP = 1e-4
#: The margin on the scan's largest: declared, not derived (as the crease line's); never moved after the views.
SAFETY = 1.25
#: The singular poses and pixels (``crease_singular_check_pr5b.json``), and the offset each sweep moves.
SINGULAR = (((6.0, 75.0, (0.3, 0.8)), (43, 59), 0), ((3.0, 0.0, (0.0, 0.0)), (83, 29), 1),
            ((2.0, 0.0, (0.0, 0.0)), (81, 29), 1))
SWEEP = (0.0, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2)


def sub_rays(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Each pixel's sixteen sub-rays' unit directions (the renderer's corner sub-samples), shape (16, pixels, 3)."""
    d, du, dv = C.rays(u, v)
    subs = []
    for i in range(C.CORNER_SAMPLES):
        for j in range(C.CORNER_SAMPLES):
            su, sv = (i + 0.5) / C.CORNER_SAMPLES - 0.5, (j + 0.5) / C.CORNER_SAMPLES - 0.5
            moved = d + su * du + sv * dv
            subs.append(moved / np.linalg.norm(moved, axis=-1, keepdims=True))
    return np.stack(subs)


def axis_distance(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Each pixel's least distance of its sub-rays from the side's axis line, in pixels at the body."""
    v, u = np.mgrid[0:C.HEIGHT, 0:C.WIDTH]
    subs = sub_rays(u.ravel().astype(float), v.ravel().astype(float))
    o = np.array([0.0, -C.DISTANCE, 0.0]) - centre
    cross = np.cross(subs, axis)
    apart = np.abs(cross @ o) / np.linalg.norm(cross, axis=-1)
    return (apart.min(axis=0) / C.PIXEL).reshape(C.HEIGHT, C.WIDTH)


def flagged(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The pixels about a cap narrower than two sub-pixels (``test_crease.judged_pixels``' complement)."""
    found = np.zeros((C.HEIGHT, C.WIDTH), dtype=bool)
    rims = hull_oracle.cylinder_outline(centre, axis, C.RADIUS, C.LENGTH)
    for end, rim in zip((-C.LENGTH / 2, C.LENGTH / 2), rims, strict=True):
        if hull_oracle.minor_axis(centre + end * axis, axis, C.RADIUS, C.FOCAL, C.DISTANCE) < 2.0 / C.CORNER_SAMPLES:
            projected = hull_oracle.project(rim, C.FOCAL, C.DISTANCE, C.WIDTH, C.HEIGHT)
            low, high = np.floor(projected.min(axis=0)).astype(int) - 2, np.ceil(projected.max(axis=0)).astype(int) + 2
            found[max(0, low[1]):high[1] + 1, max(0, low[0]):high[0] + 1] = True
    return found


def classes(band: np.ndarray, near: np.ndarray, thin: np.ndarray) -> dict[str, np.ndarray]:
    """The band's pixels by class: ``singular`` (a sub-ray within ``NEAR`` of the axis), ``thin`` (about a thin cap,
    not singular) and ``rest`` (the pixels the line judges)."""
    singular = band & (near < NEAR)
    thin = band & thin & ~singular
    return {"rest": band & ~singular & ~thin, "singular": singular, "thin": thin}


def exact_derivative(centre: np.ndarray, axis: np.ndarray, index: int) -> np.ndarray:
    e = np.zeros(3)
    e[index] = EXACT_STEP * C.PIXEL
    moved = [hull_oracle.shares(hull_oracle.cylinder_outline(centre + sign * e, axis, C.RADIUS, C.LENGTH), C.FOCAL,
                                C.DISTANCE, C.WIDTH, C.HEIGHT) for sign in (1.0, -1.0)]
    return (moved[0] - moved[1]) / (2 * EXACT_STEP)


def errors(pose: tuple) -> tuple:
    """A pose's image error along x and z at every pixel (zero off the band), its band and each pixel's distance."""
    import crease_march_line as M

    turn, roll, offset = pose
    view = M.View(turn, roll, offset)
    band = np.zeros(C.HEIGHT * C.WIDTH, dtype=bool)
    band[view.pick] = True
    band = band.reshape(C.HEIGHT, C.WIDTH)
    found = {}
    for name, index in (("x", 0), ("z", 2)):
        e = np.zeros(3)
        e[index] = M.EPSILON * C.PIXEL
        live = (view.image(e, "live") - view.image(-e, "live")) / (2 * M.EPSILON)
        found[name] = np.where(band, np.abs(live - exact_derivative(view.centre, view.axis, index)), 0.0)
    return found, band, axis_distance(view.centre, view.axis), flagged(view.centre, view.axis)


def one_pose(pose: tuple) -> dict:
    """The largest image error in each class at one pose, over x and z, with its pixel and its distance."""
    found, band, near, thin = errors(pose)
    turn, roll, offset = pose
    row: dict = {"turn": turn, "roll": roll, "offset": list(offset)}
    for name, mask in classes(band, near, thin).items():
        best = None
        for direction, error in found.items():
            if mask.any():
                at = np.unravel_index(int(np.argmax(np.where(mask, error, -1.0))), error.shape)
                if best is None or error[at] > best["error"]:
                    best = {"error": float(error[at]), "direction": direction, "at_v_u": [int(at[0]), int(at[1])],
                            "near_px": float(near[at])}
        row[name] = best
    print(json.dumps(row), flush=True)
    return row


def sweep_point(item: tuple) -> dict:
    (turn, roll, offset), (u, v), moved, delta = item
    shifted = list(offset)
    shifted[moved] += delta
    found, band, near, thin = errors((turn, roll, tuple(shifted)))
    rest = classes(band, near, thin)["rest"]
    return {"pose": [turn, roll, shifted], "pixel_u_v": [u, v], "delta_px": delta, "near_px": float(near[v, u]),
            "singular": bool(near[v, u] < NEAR),
            "error": {name: float(error[v, u]) for name, error in found.items()},
            "rest_largest": max(float(np.max(np.where(rest, error, 0.0))) for error in found.values())}


def main() -> None:
    from multiprocessing import Pool

    if "--sweep" in sys.argv:
        items = [(pose, pixel, moved, delta) for pose, pixel, moved in SINGULAR for delta in SWEEP]
        with Pool(C.WORKERS) as pool:
            rows = pool.map(sweep_point, items, chunksize=1)
        for row in rows:
            print(json.dumps(row))
        out = [a for a in sys.argv[1:] if not a.startswith("--")]
        if out:
            Path(out[0]).write_text(json.dumps({"near_px": NEAR, "sweep": rows}, indent=1) + "\n")
        return
    started = time.time()
    poses = list(itertools.product(C.TURNS, C.ROLLS, C.OFFSETS))
    with Pool(C.WORKERS) as pool:
        rows = pool.map(one_pose, poses, chunksize=1)
    largest = {name: max((r[name] for r in rows if r[name] is not None), key=lambda b: b["error"])
               for name in ("rest", "singular", "thin")}
    worst = max(rows, key=lambda r: r["rest"]["error"] if r["rest"] else -1.0)
    report = {
        "scan": {"turns": C.TURNS, "rolls": C.ROLLS, "offsets": C.OFFSETS, "poses": len(rows), "near_px": NEAR,
                 "exact_step_px": EXACT_STEP, "seconds": round(time.time() - started)},
        "largest": largest,
        "worst_rest_pose": {key: worst[key] for key in ("turn", "roll", "offset")},
        "safety": SAFETY,
        "image_line": SAFETY * largest["rest"]["error"],
        "rows": rows,
    }
    print(json.dumps({key: report[key] for key in report if key != "rows"}, indent=1))
    out = [a for a in sys.argv[1:] if not a.startswith("--")]
    if out:
        Path(out[0]).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
