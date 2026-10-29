"""The slice of the corrected tube table (MIME-ANO-117): for one readable radius ``r`` (a source there; the hardest
pairs at and near it), what each approach needs. ``image``: the table holds the image. ``remainder``: it holds the
image minus the nearest wall's plane image (Blake 1971), which the evaluator would add back.

Two measures, each with its control:

* **truncation**, by self-convergence against a fine reference (each of modes, wavenumber nodes and the wavenumber
  reach swept with the others at the reference's) and by the no-slip residual on the wall; the 2026-05-21 tables'
  truncation is the control that the measures can fail;
* **knots**, the cubic read (``mime.core.tables``' scheme, the tube table's mirrored ends) between knots along each
  axis through the hardest pair, over graded families; the remainder's control is a wrong plane image (its sign
  flipped, or the far wall's plane), whose remainder must fail the same read.

Every error is pointwise, over the block's largest entry of the image at that point."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import build

MODE_LEVELS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
REACH_LEVELS = (3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0)
NODE_LEVELS = (8.0, 16.0, 32.0, 64.0, 128.0)
TOLERANCES = (1e-2, 1e-3)
READ_TOLERANCE = 1e-2
STEPS = (0.04, 0.06, 0.09, 0.13, 0.19, 0.28, 0.4)
#: The table's axial extent over the tube's radius (the 2026-05-21 tables').
AXIAL_REACH = 5.0
APPROACHES: dict[str, tuple[float, float] | None] = {"image": None, "remainder": build.NEAREST}
CONTROLS = {"flipped_sign": (0.0, -1.0), "far_wall_plane": (math.pi, 1.0)}
AXES = ("angle", "axial", "radius")


def relative(values: np.ndarray, reference: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Pointwise: the largest entry miss over the largest entry of ``scale``'s block."""
    return np.max(np.abs(values - reference), axis=(-2, -1)) / np.max(np.abs(scale), axis=(-2, -1))


def sample_grid(r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = 1.0 - r
    phi = sorted({0.0, g, 2 * g, 4 * g, 0.25, 0.5, 1.0, 2.0, math.pi})
    z = sorted({0.0, g, 2 * g, 4 * g, 0.25, 0.5, 1.0, 2.0, 3.0, AXIAL_REACH})
    rho = sorted({x for x in (0.0, 0.3, 0.6, 0.85, r - 2 * g, r - g, r) if 0.0 <= x <= r})
    return np.array([*rho, 1.0]), np.array(phi), np.array(z)


def truncation(r: float, modes: float, wavenumbers: float, reach: float) -> build.Truncation:
    return build.scaled_truncation(r, r, modes=modes, wavenumbers=wavenumbers, reach=reach)


def _level(levels: tuple[float, ...], rows: list[dict], key: str, bound: float) -> float:
    for row in rows:
        if row[key] <= bound:
            return row["level"]
    return levels[-1]


def _next(levels: tuple[float, ...], level: float) -> float:
    i = levels.index(level)
    return levels[min(i + 1, len(levels) - 1)]


def sweep(r: float, plane: tuple[float, float] | None, log: Callable[[str], None]) -> dict[str, Any]:
    rho, phi, z = sample_grid(r)
    added = build.blake_cylindrical(r, rho, phi, z, plane) if plane is not None else 0.0
    free_wall = build.free_cylindrical(r, rho[-1:], phi, z)[0]

    def run(tr: build.Truncation, checkpoints: list[int] | None = None) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        values = build.image_grid(r, rho, phi, z, tr, plane=plane, mode_checkpoints=checkpoints)
        return values, time.perf_counter() - start

    def judged(values: np.ndarray) -> dict[str, float]:
        image = values + added
        interior = relative(image[:-1], reference_image[:-1], reference_image[:-1])
        wall = np.max(np.abs(image[-1] + free_wall), axis=(-2, -1)) / np.max(np.abs(free_wall), axis=(-2, -1))
        return {"interior": float(np.max(interior)), "no_slip": float(np.max(wall))}

    h = 2.0 * (1.0 - r)
    ref = truncation(r, MODE_LEVELS[-1], NODE_LEVELS[-1], REACH_LEVELS[-1])
    counts = [int(math.ceil(m / h)) for m in MODE_LEVELS]
    parts, seconds = run(ref, counts)
    reference_image = parts[-1] + added
    log(f"r={r} plane={plane} reference {ref} in {seconds:.1f} s")
    out: dict[str, Any] = {"reference": ref.__dict__, "reference_seconds": seconds}
    out["modes"] = [{"level": m, "count": c, **judged(p)} for m, c, p in zip(MODE_LEVELS, counts, parts, strict=True)]
    out["reach"], out["nodes"] = [], []
    for level in REACH_LEVELS[:-1]:
        values, s = run(truncation(r, MODE_LEVELS[-1], NODE_LEVELS[-1], level))
        out["reach"].append({"level": level, "k_max": level / h, "seconds": s, **judged(values)})
    out["reach"].append({"level": REACH_LEVELS[-1], "k_max": REACH_LEVELS[-1] / h, "seconds": seconds,
                         **judged(parts[-1])})
    for level in NODE_LEVELS[:-1]:
        values, s = run(truncation(r, MODE_LEVELS[-1], level, REACH_LEVELS[-1]))
        out["nodes"].append({"level": level, "count": int(math.ceil(level / h)), "seconds": s, **judged(values)})
    out["nodes"].append({"level": NODE_LEVELS[-1], "count": ref.wavenumbers, "seconds": seconds,
                         **judged(parts[-1])})
    # the reference's own bound: each sweep's change between its two finest levels
    out["reference_bound"] = max(out[key][-2]["interior"] for key in ("modes", "reach", "nodes"))
    if plane is None:
        values, _ = run(build.series_truncation(15, 80, 64, r))
        out["control_2026_05_21_truncation"] = judged(values)
    chosen = {}
    for tol in TOLERANCES:
        m = _level(MODE_LEVELS, out["modes"], "interior", tol / 3)
        w = _level(NODE_LEVELS, out["nodes"], "interior", tol / 3)
        k = _level(REACH_LEVELS, out["reach"], "interior", tol / 3)
        for _ in range(4):
            tr = truncation(r, m, w, k)
            values, s = run(tr)
            found = judged(values)
            if found["interior"] <= tol or (m, w, k) == (MODE_LEVELS[-1], NODE_LEVELS[-1], REACH_LEVELS[-1]):
                break
            m, w, k = _next(MODE_LEVELS, m), _next(NODE_LEVELS, w), _next(REACH_LEVELS, k)
        chosen[str(tol)] = {"levels": [m, w, k], "truncation": tr.__dict__, "seconds": s, **found}
        log(f"  tol {tol}: {tr} -> {found}")
    out["chosen"] = chosen
    return out


def line_points(axis: str, r: float) -> tuple[float, list[tuple[float, float]], np.ndarray]:
    """The axis's length, the families ``(near, step)`` and the union of every family's knots and quarter points."""
    g = 1.0 - r
    length = {"angle": math.pi, "axial": AXIAL_REACH, "radius": r}[axis]
    families = [(near, step) for near in sorted({2 * g, 8 * g, 0.05, 0.15, 0.4}) for step in STEPS]
    points = []
    for near, step in families:
        knots = build.graded_knots(length, near, step)
        width = np.diff(knots)
        points.append(knots)
        for t in (0.25, 0.5, 0.75):
            points.append(knots[:-1] + t * width)
    union = np.unique(np.round(np.concatenate(points), 14))
    return length, families, union


def line_values(axis: str, r: float, union: np.ndarray, tr: build.Truncation,
                plane: tuple[float, float] | None) -> np.ndarray:
    """``(P, 3, 3)`` along the axis through the hardest pair (``rho_t = rho_s = r``, ``dphi = dz = 0``); the radius
    axis's coordinate is the distance in from ``r``."""
    if axis == "angle":
        return build.image_grid(r, [r], union, [0.0], tr, plane=plane)[0, :, 0]
    if axis == "axial":
        return build.image_grid(r, [r], [0.0], union, tr, plane=plane)[0, 0, :]
    return build.image_grid(r, r - union, [0.0], [0.0], tr, plane=plane)[:, 0, 0]


def line_plane(axis: str, r: float, union: np.ndarray, plane: tuple[float, float]) -> np.ndarray:
    if axis == "angle":
        return build.blake_cylindrical(r, np.array([r]), union, np.array([0.0]), plane)[0, :, 0]
    if axis == "axial":
        return build.blake_cylindrical(r, np.array([r]), np.array([0.0]), union, plane)[0, 0, :]
    return build.blake_cylindrical(r, r - union, np.array([0.0]), np.array([0.0]), plane)[:, 0, 0]


def read_error(axis: str, length: float, near: float, step: float, union: np.ndarray, values: np.ndarray,
               image: np.ndarray) -> tuple[int, float]:
    """The family's knot count and its worst read error between knots (over the image's block there)."""
    knots = build.graded_knots(length, near, step)
    at = np.searchsorted(union, np.round(knots, 14))
    between = np.setdiff1d(np.arange(len(union)), at)
    queries = union[between]
    read = np.empty((len(queries), 3, 3))
    for i in range(3):
        for j in range(3):
            f = values[at, i, j]
            if axis == "radius":
                read[:, i, j] = build.hermite_read(knots, f, queries)
            else:
                odd = build.SIN_IN_ANGLE[i, j] if axis == "angle" else build.SIN_ALONG[i, j]
                x, fx = build.mirrored(knots, f, -1.0 if odd else 1.0, wraps=axis == "angle")
                read[:, i, j] = build.hermite_read(x, fx, queries)
    return len(knots), float(np.max(relative(read, values[between], image[between])))


def knots(r: float, chosen: dict[str, dict], log: Callable[[str], None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis in AXES:
        length, families, union = line_points(axis, r)
        by_approach: dict[str, dict[str, np.ndarray]] = {}
        for name, plane in APPROACHES.items():
            tr = build.Truncation(**chosen[name]["0.001"]["truncation"])
            note: dict[str, Any] = {"truncation": tr.__dict__}
            if axis == "radius":
                # no oscillation along this line: fewer nodes, checked against the chosen truncation on a subset
                subset = union[:: max(1, len(union) // 40)]
                full = line_values(axis, r, subset, tr, plane)
                for divisor in (4, 2, 1):
                    lighter = build.Truncation(tr.modes, max(8, tr.wavenumbers // divisor), tr.k_max, tr.angles)
                    image_sub = full + (line_plane(axis, r, subset, plane) if plane else 0.0)
                    found = float(np.max(relative(line_values(axis, r, subset, lighter, plane), full, image_sub)))
                    if found <= 1e-4 or divisor == 1:
                        tr = lighter
                        note.update(line_truncation=tr.__dict__, line_truncation_vs_chosen=found)
                        break
            start = time.perf_counter()
            values = line_values(axis, r, union, tr, plane)
            note["seconds"] = time.perf_counter() - start
            image = values + (line_plane(axis, r, union, plane) if plane else 0.0)
            by_approach[name] = {"values": values, "image": image}
            rows = []
            for near, step in families:
                count, err = read_error(axis, length, near, step, union, values, image)
                rows.append({"near": near, "step": step, "knots": count, "error": err})
            passing = [row for row in rows if row["error"] <= READ_TOLERANCE]
            note["best"] = min(passing, key=lambda row: row["knots"]) if passing else None
            note["closest"] = min(rows, key=lambda row: row["error"])
            note["families"] = rows
            out.setdefault(axis, {})[name] = note
            log(f"  knots {axis} {name}: {note['best']}")
        # the remainder's controls: a wrong plane image, read with the remainder's chosen family
        best = out[axis]["remainder"]["best"] or out[axis]["remainder"]["closest"]
        image = by_approach["image"]["values"]
        controls = {}
        for label, plane in CONTROLS.items():
            wrong = image - line_plane(axis, r, union, plane)
            _, err = read_error(axis, length, best["near"], best["step"], union, wrong, image)
            controls[label] = {"error": err, "fails": err > READ_TOLERANCE}
        # and the two approaches agree where both are converged
        agreement = float(np.max(relative(by_approach["remainder"]["image"], image, image)))
        out[axis]["remainder_controls"] = controls
        out[axis]["image_vs_remainder_plus_plane"] = agreement
    return out


def two_axis_read(r: float, approach: str, plane: tuple[float, float] | None, tr: build.Truncation,
                  angle_family: dict, axial_family: dict) -> float:
    """The read across a (angle, axial) cell, at its centre, with the chosen families: the two axes' errors
    together."""
    a = build.graded_knots(math.pi, angle_family["near"], angle_family["step"])
    zk = build.graded_knots(AXIAL_REACH, axial_family["near"], axial_family["step"])
    am, zm = 0.5 * (a[1:] + a[:-1]), 0.5 * (zk[1:] + zk[:-1])
    phi, z = np.concatenate([a, am]), np.concatenate([zk, zm])
    values = build.image_grid(r, [r], phi, z, tr, plane=plane)[0]
    image = values + (build.blake_cylindrical(r, np.array([r]), phi, z, plane)[0] if plane else 0.0)
    na, nz = len(a), len(zk)
    worst = 0.0
    read = np.empty((len(am), len(zm), 3, 3))
    for i in range(3):
        for j in range(3):
            along = np.empty((na, len(zm)))
            for p in range(na):
                x, f = build.mirrored(zk, values[p, :nz, i, j], -1.0 if build.SIN_ALONG[i, j] else 1.0, wraps=False)
                along[p] = build.hermite_read(x, f, zm)
            for q in range(len(zm)):
                x, f = build.mirrored(a, along[:, q], -1.0 if build.SIN_IN_ANGLE[i, j] else 1.0, wraps=True)
                read[:, q, i, j] = build.hermite_read(x, f, am)
    worst = float(np.max(relative(read, values[na:, nz:], image[na:, nz:])))
    return worst


def table_job(r: float, plane: tuple[float, float] | None, tr: build.Truncation, families: dict[str, dict],
              device: Callable[..., Any] | None = None, budget: float = 4e6) -> tuple[np.ndarray, dict[str, Any]]:
    """One source radius of the full table at its knots: the job whose time, times the radial knots, is the
    table's."""
    radial = r - build.graded_knots(r, families["radius"]["near"], families["radius"]["step"])[::-1]
    angle = build.graded_knots(math.pi, families["angle"]["near"], families["angle"]["step"])
    axial = build.graded_knots(AXIAL_REACH, families["axial"]["near"], families["axial"]["step"])
    start = time.perf_counter()
    values = build.image_grid(r, radial, angle, axial, tr, plane=plane, device=device, budget=budget)
    seconds = time.perf_counter() - start
    size = len(radial) ** 2 * len(angle) * len(axial) * 9 * 8
    return values, {"radial": len(radial), "angle": len(angle), "axial": len(axial), "seconds": seconds,
                    "table_bytes_float64": size}


def measure(r: float, log: Callable[[str], None] = print, save: Callable[[str, dict], None] | None = None) -> dict:
    result: dict[str, Any] = {"radius": r, "started": time.time()}
    chosen = {}
    for name, plane in APPROACHES.items():
        result[f"sweep_{name}"] = sweep(r, plane, log)
        chosen[name] = result[f"sweep_{name}"]["chosen"]
    result["knots"] = knots(r, chosen, log)
    result["jobs"] = {}
    for name, plane in APPROACHES.items():
        families = {axis: (result["knots"][axis][name]["best"] or result["knots"][axis][name]["closest"])
                    for axis in AXES}
        tr01, tr001 = (build.Truncation(**chosen[name][t]["truncation"]) for t in ("0.01", "0.001"))
        result["knots"][f"two_axis_{name}"] = two_axis_read(r, name, plane, tr001, families["angle"],
                                                            families["axial"])
        values, info = table_job(r, plane, tr01, families)
        info.update(truncation=tr01.__dict__, families=families)
        result["jobs"][name] = info
        log(f"  job {name}: {info['seconds']:.1f} s, {info['table_bytes_float64'] / 1e9:.3f} GB")
        if save is not None:
            save(f"job_r{r}_{name}", {"values": values, "info": info})
    result["seconds"] = time.time() - result["started"]
    return result
