"""The blob-helix score's stages in the brief's order (the controls first, then ``surf5`` at its operating fraction coarse
to fine, then the rest), each solve saved through ``save(name, payload)`` as it is found and a rerun skipping what the
records hold. On Modal by ``blob_helix_modal.py``; here on the CPU at coarse sizes (``--dry``).

1. ``sphere``: the controls C1 (MIME's sphere sampling, 512 samples, the spike's numbers) and C2 (the quasi-uniform
   sampler on the sphere at the same counts).
2. ``samplers``: FL-9's own sampling and the quasi-uniform one at every level, their quality.
3. ``control``: Cortez 0.5 on FL-9's own sampling at the frozen spacing (the node's 2.7979 mm/s, C3).
4. ``main``: ``surf5`` at fraction 2.0 on the quasi-uniform sampling, coarse to fine.
5. ``rest``: every other arm, coarse to fine; the condition at the frozen level and the spectrum at the coarsest.
6. ``edges``: the spheroid and the capped cylinder, the same pipeline.
7. ``float32``: the surface arms and Cortez in float32 at the frozen level.
8. ``timing``: warm repeats at the frozen level (matched wall time).
9. ``fine``: one finer level for the best arms, if the time allows.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python blob_helix_run.py --dry --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import blob_helix as bh  # noqa: E402

Saver = Callable[[str, dict[str, Any]], None]
Holder = Callable[[str], "dict[str, Any] | None"]
STAGES = ("sphere", "samplers", "control", "main", "rest", "edges", "float32", "timing", "fine")
FROZEN_MM = 0.16
#: The level whose matrices' full spectra are taken (the coarsest).
SPECTRUM_MM = 0.3
MAIN_ARM = ("surf5", 2.0)
QU_ARMS = (("cortez", 0.5), ("surf3", 1.0), ("surf3", 1.5), ("surf3", 2.0), ("surf5", 1.5), ("surf5", 2.5),
           ("surf5", 3.0))
OWN_ARMS = (("cortez", 0.5),)
#: The surface blobs on FL-9's own sampling: only where the condition and spectrum are taken.
OWN_SURFACE_ARMS = (("surf3", 1.0), ("surf3", 1.5), ("surf3", 2.0), ("surf5", 1.5), ("surf5", 2.0), ("surf5", 2.5),
                    ("surf5", 3.0))
EDGE_ARMS = (("surf5", 2.0), ("surf5", 1.5), ("surf3", 1.5), ("cortez", 0.5))
FLOAT32_ARMS = (("qu", "surf5", 2.0), ("qu", "surf5", 1.5), ("qu", "surf3", 1.5), ("qu", "cortez", 0.5),
                ("own", "cortez", 0.5))
#: FL-9's size for the edge test (the screw's R and L): a spheroid of semi-axes L/2, R, R, and a cylinder R, L.
FL9_RADIUS_M, FL9_LENGTH_M = 1.56e-3, 7.47e-3
WATER_VISCOSITY = 1.0e-3

FULL = {"ladder": (0.3, 0.2, 0.16, 0.128, 0.1), "fine": (0.092,), "sphere": ((16, 32), (24, 48)),
        "sphere_dense_factor": 3, "edge_ladder": (0.3, 0.2, 0.16, 0.128, 0.1)}
DRY = {"ladder": (0.8, 0.6), "fine": (), "sphere": ((8, 16),), "sphere_dense_factor": 3, "edge_ladder": (0.8, 0.6)}

_CACHE: dict[float, dict[str, Any]] = {}
_EDGE_POINTS: dict[tuple[str, float], np.ndarray] = {}


def _key(sampling: str, kernel: str, fraction: float) -> str:
    return f"{sampling}:{kernel}:{fraction}"


SAMPLER = "v2"


def _labels(helix: bh.Helix, local: np.ndarray, end: np.ndarray, rim: np.ndarray, mean_h: float
            ) -> dict[str, np.ndarray]:
    from scipy.spatial import cKDTree

    lateral = end == 0
    s = (np.hypot(local[:, 1], local[:, 2]) / helix.radius - 1.0) / helix.modulation
    return {"caps": ~lateral, "lateral": lateral, "crest_band": lateral & (s > 0.6),
            "trough_band": lateral & (s < -0.6), "rim_strip": cKDTree(rim).query(local, workers=-1)[0] < mean_h}


def level(spacing_mm: float) -> dict[str, Any]:
    """FL-9 at ``spacing_mm``: its own sampling and the quasi-uniform one (sampler v2; cached in this process)."""
    if spacing_mm in _CACHE:
        return _CACHE[spacing_mm]
    helix = bh.Helix(spacing_mm)
    body = helix.body
    n, mean_h = len(body.weights), body.mean_h
    own = helix.local([tuple(c) for c in body.node.constants()["counts"]])
    dense = helix.local(body.counts_at(spacing_mm / bh.DENSE_FACTOR_V2))
    picked = bh.quasi_uniform_v2(dense["points"], dense["weights"], n, helix.projector(dense))
    qu_local = picked["points"]
    qu_pts = helix.place(qu_local)
    half = helix.length / 2.0
    rim = dense["points"][(dense["end"] == 0) & (np.abs(dense["zeta"]) > half - mean_h / bh.DENSE_FACTOR_V2)]
    dense_lab = _labels(helix, dense["points"], dense["end"], rim, mean_h)
    own_lab = _labels(helix, own["points"], own["end"], rim, mean_h)
    qu_lab = _labels(helix, qu_local, dense["end"][picked["near"]], rim, mean_h)
    dense_w = dense["weights"]
    record = {
        "sampler": SAMPLER, "samples": n, "mean_spacing_m": mean_h, "area_m2": float(body.weights.sum()),
        "dense_points": len(dense_w), "dense_area_m2": float(dense_w.sum()),
        "own_matches_node": float(np.max(np.abs(helix.place(own["points"]) - body.points))),
        "own": {**bh.nearest(body.points, mean_h),
                "regions": bh.regions_report(own_lab, dense_lab, dense_w, body.points, mean_h)},
        "qu": {**bh.nearest(qu_pts, mean_h), "regions": bh.regions_report(qu_lab, dense_lab, dense_w, qu_pts, mean_h),
               "lloyd_last": picked["lloyd"][-1], "lloyd": picked["lloyd"], "fps_s": picked["fps_s"],
               "empty_cells": picked["empty_cells"], "off_dense_over_h": picked["off_dense_over_h"],
               "cell_area_over_mean": {"min": float(picked["area"].min() / (dense_w.sum() / n)),
                                       "max": float(picked["area"].max() / (dense_w.sum() / n)),
                                       "std": float(np.std(picked["area"]) / (dense_w.sum() / n))}},
        "rim_points": len(rim),
    }
    found = {"helix": helix, "own": body.points, "qu": qu_pts, "record": record, "centre": body.centre,
             "viscosity": body.viscosity, "mean_h": mean_h}
    _CACHE[spacing_mm] = found
    return found


def _sphere_errors(r: np.ndarray) -> dict[str, float]:
    sym = (r + r.T) / 2.0
    errors = {"F_z": sym[2, 2] / (6 * math.pi) - 1, "T_z": sym[5, 5] / (8 * math.pi) - 1,
              "F_x": sym[0, 0] / (6 * math.pi) - 1, "T_x": sym[3, 3] / (8 * math.pi) - 1}
    return {**{k: float(v) for k, v in errors.items()}, "max_abs": float(max(abs(v) for v in errors.values()))}


def stage_sphere(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    out = held("sphere") or {}
    for polar, around in sizes["sphere"]:
        n = polar * around
        pts, w = bh.sphere_samples(polar, around)
        f = int(bh.DENSE_FACTOR_V2)
        dense, dense_w = bh.sphere_samples(f * polar, f * around)
        h = math.sqrt(w.sum() / n)
        picked = bh.quasi_uniform_v2(dense, dense_w, n, bh.project_sphere)
        qu = picked["points"]
        row = out.setdefault(f"{n}", {"sampler": SAMPLER, "mean_h": h, "own_quality": bh.nearest(pts, h),
                                      "qu_quality": {**bh.nearest(qu, h), "lloyd_last": picked["lloyd"][-1],
                                                     "dense": len(dense)}, "solves": {}})
        for sampling, points in (("own", pts), ("qu", qu)):
            for kernel, fraction in (("surf5", 2.0), ("surf3", 1.5), ("cortez", 0.5), ("surf5", 1.5)):
                key = _key(sampling, kernel, fraction)
                if key in row["solves"]:
                    continue
                if deadline is not None and time.time() > deadline:
                    return False
                found = bh.solve(kernel, points, fraction * h, 1.0, np.zeros(3), condition=True)
                r = np.asarray(found.pop("resistance"))
                row["solves"][key] = {**found, **_sphere_errors(r)}
                out[f"{n}"] = row
                save("sphere", out)
    return True


def _save_sampling(spacing: float, save: Saver, held: Holder) -> None:
    name = f"sampling_{spacing:.4f}"
    if held(name) is None:
        found = level(spacing)
        save(name, found["record"])
        save(f"points_{spacing:.4f}", {"qu": found["qu"].tolist()})


def stage_samplers(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    for spacing in sizes["ladder"]:
        if deadline is not None and time.time() > deadline:
            return False
        _save_sampling(spacing, save, held)
    return True


def _helix_arm(spacing: float, sampling: str, kernel: str, fraction: float, save: Saver, held: Holder,
               dtype: str = "float64", tag: str = "") -> dict[str, Any]:
    name = f"helix_{spacing:.4f}"
    out = held(name) or {"spacing_mm": spacing, "solves": {}}
    key = _key(sampling, kernel, fraction) + (f":{dtype}" if dtype != "float64" else "") + tag
    if key in out["solves"]:
        return out["solves"][key]
    lv = level(spacing)
    points = lv[sampling]
    found = bh.solve(kernel, points, fraction * lv["mean_h"], lv["viscosity"], lv["centre"], dtype=dtype,
                     condition=abs(spacing - FROZEN_MM) < 1e-12 and dtype == "float64" and not tag,
                     spectrum=abs(spacing - SPECTRUM_MM) < 1e-12 and dtype == "float64" and not tag)
    r = np.asarray(found.pop("resistance"))
    out["solves"][key] = {**found, "fraction": fraction, "sampling": sampling, **bh.helix_reading(r),
                          "resistance_si": r.tolist()}
    out["mean_spacing_m"] = lv["mean_h"]
    save(name, out)
    return out["solves"][key]


def _arms(save: Saver, held: Holder, deadline: float | None, ladder: tuple[float, ...],
          arms: list[tuple[str, str, float]]) -> bool:
    for sampling, kernel, fraction in arms:
        for spacing in ladder:
            if deadline is not None and time.time() > deadline:
                return False
            _helix_arm(spacing, sampling, kernel, fraction, save, held)
    return True


def stage_control(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    return _arms(save, held, deadline, (FROZEN_MM,), [("own", "cortez", 0.5)])


def stage_main(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    return _arms(save, held, deadline, sizes["ladder"], [("qu", *MAIN_ARM)])


def stage_rest(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    arms = [("own", k, f) for k, f in OWN_ARMS] + [("qu", k, f) for k, f in QU_ARMS]
    if not _arms(save, held, deadline, sizes["ladder"], arms):
        return False
    return _arms(save, held, deadline, tuple(sorted({SPECTRUM_MM, FROZEN_MM}, reverse=True)),
                 [("own", k, f) for k, f in OWN_SURFACE_ARMS])


def stage_edges(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    out = held("edges") or {}
    c, a = FL9_LENGTH_M / 2.0, FL9_RADIUS_M
    exact = bh.spheroid_exact(c, a, WATER_VISCOSITY)
    out["spheroid_exact"] = exact
    for body_name in ("spheroid", "cylinder"):
        rows = out.setdefault(body_name, {})
        for spacing in sizes["edge_ladder"]:
            h_nom = spacing * 1e-3
            row = rows.get(f"{spacing}")
            if row is None:
                if body_name == "spheroid":
                    dense, dense_w = bh.spheroid_dense(c, a, h_nom / bh.DENSE_FACTOR_V2)
                    labels = None
                    project = bh.spheroid_projector(c, a)
                else:
                    dense, dense_w, labels = bh.cylinder_dense(a, FL9_LENGTH_M, h_nom / bh.DENSE_FACTOR_V2)
                    project = bh.cylinder_projector(a, FL9_LENGTH_M, labels["caps"])
                area = float(dense_w.sum())
                n = max(16, round(area / h_nom ** 2))
                picked = bh.quasi_uniform_v2(dense, dense_w, n, project)
                pts = picked["points"]
                h = math.sqrt(area / n)
                row = {"sampler": SAMPLER, "samples": n, "mean_h": h, "area_m2": area, "quality": bh.nearest(pts, h),
                       "solves": {}, "lloyd_last": picked["lloyd"][-1]}
                if labels is not None:
                    row["regions"] = bh.regions_report({k: v[picked["near"]] for k, v in labels.items()}, labels,
                                                       dense_w, pts, h)
                _EDGE_POINTS[(body_name, spacing)] = pts
                rows[f"{spacing}"] = row
            if (body_name, spacing) not in _EDGE_POINTS:  # a resumed record: the sampling again (deterministic)
                rows.pop(f"{spacing}")
                return stage_edges(save, held, deadline, sizes)
            pts = _EDGE_POINTS[(body_name, spacing)]
            for kernel, fraction in EDGE_ARMS:
                key = _key("qu", kernel, fraction)
                if key in row["solves"]:
                    continue
                if deadline is not None and time.time() > deadline:
                    save("edges", out)
                    return False
                found = bh.solve(kernel, pts, fraction * row["mean_h"], WATER_VISCOSITY, np.zeros(3))
                r = np.asarray(found.pop("resistance"))
                reading = bh.axis_reading(r)
                if body_name == "spheroid" and "F_x" in reading:
                    reading["drag_error"] = reading["F_x"] / exact["drag"] - 1.0
                    reading["torque_error"] = reading["T_x"] / exact["torque"] - 1.0
                row["solves"][key] = {**found, **reading}
                save("edges", out)
    return True


def stage_float32(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    for sampling, kernel, fraction in FLOAT32_ARMS:
        if deadline is not None and time.time() > deadline:
            return False
        _helix_arm(FROZEN_MM, sampling, kernel, fraction, save, held, dtype="float32")
    return True


def stage_timing(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    frozen = FROZEN_MM
    for sampling, kernel, fraction in (("qu", "surf5", 2.0), ("own", "cortez", 0.5), ("qu", "surf3", 1.5)):
        for k in (1, 2):
            if deadline is not None and time.time() > deadline:
                return False
            _helix_arm(frozen, sampling, kernel, fraction, save, held, tag=f":warm{k}")
    return True


def stage_fine(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    arms = [("qu", "surf5", 2.0), ("qu", "surf3", 1.5), ("qu", "surf5", 1.5), ("qu", "cortez", 0.5),
            ("own", "cortez", 0.5)]
    for spacing in sizes["fine"]:
        _save_sampling(spacing, save, held)
    return _arms(save, held, deadline, sizes["fine"], arms)


RUNNERS = {"sphere": stage_sphere, "samplers": stage_samplers, "control": stage_control, "main": stage_main,
           "rest": stage_rest, "edges": stage_edges, "float32": stage_float32, "timing": stage_timing,
           "fine": stage_fine}


def run_stages(stages: tuple[str, ...], save: Saver, held: Holder, deadline: float | None,
               sizes: dict[str, Any]) -> dict[str, Any]:
    global FROZEN_MM, SPECTRUM_MM  # noqa: PLW0603 - the dry run's levels stand in for the frozen and coarsest
    done: dict[str, Any] = {}
    if sizes is DRY:
        FROZEN_MM, SPECTRUM_MM = sizes["ladder"][-1], sizes["ladder"][0]
    for stage in stages:
        began = time.time()
        finished = RUNNERS[stage](save, held, deadline, sizes)
        done[stage] = {"finished": finished, "seconds": time.time() - began}
        save("stages", {"done": done, "unix": time.time()})
        if not finished:
            break
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import operating_point as op

    op.require_float64()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)

    def save(name: str, payload: dict[str, Any]) -> None:
        (folder / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")

    def held(name: str) -> dict[str, Any] | None:
        path = folder / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    print(run_stages(tuple(args.stages.split(",")), save, held, None, DRY if args.dry else FULL), flush=True)


if __name__ == "__main__":
    main()
