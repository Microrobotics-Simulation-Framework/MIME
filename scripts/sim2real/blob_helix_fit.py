"""The blob-helix score's analysis: the records (``blob_helix_modal.py::pull``) into tables and ``analysis.json``.

- Errors against the helix reference (its preregistered and post hoc values, read from its analysis); what they bound
  is the reference's band: an error inside it is not resolved.
- Observed orders by self-convergence: at three successive levels ``U(h) = U0 + C h^p`` solved for ``p`` (and ``U0``,
  each arm's own limit), ``h`` the mean spacing; not monotone where the two differences change sign.
- The edge test: the spheroid's errors against Chwang & Wu (orders between successive levels), the cylinder's and the
  helix's self-convergence orders.

    python blob_helix_fit.py --records DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

#: The helix reference (its analysis): preregistered rule and post hoc, (value, sigma).
REFERENCE = {"axial_mm_s": {"pre": (3.059649, 0.147163), "post": (3.057296, 0.056487)},
             "R_xx": {"pre": (5.195490e-5, 1.5358e-7), "post": (5.195764e-5, 7.756e-8)},
             "R_xOx": {"pre": (-2.529923e-9, 1.2636e-10), "post": (-2.527819e-9, 5.083e-11)}}
#: Route (b)'s point values at the frozen eps, finest force sets (estimates, not a reference): R_OxOx at 0.1024 and
#: 0.092 mm.
ROUTE_B_OXOX = (4.26960e-10, 4.27061e-10)
QUANTITIES = ("axial_mm_s", "R_xx", "R_xOx", "R_OxOx")


def three_point(h: list[float], u: list[float]) -> dict[str, Any]:
    """``U = U0 + C h^p`` through three points (``h`` decreasing)."""
    d1, d2 = u[0] - u[1], u[1] - u[2]
    if d1 == 0 or d2 == 0 or (d1 > 0) != (d2 > 0):
        return {"order": None, "note": "not monotone"}
    ratio = d1 / d2

    def f(p: float) -> float:
        return (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p) - ratio

    lo, hi = 0.02, 16.0
    if f(lo) * f(hi) > 0:
        return {"order": None, "note": f"ratio {ratio:.3g} outside orders [{lo}, {hi}]"}
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    p = (lo + hi) / 2
    c = d2 / (h[1] ** p - h[2] ** p)
    return {"order": p, "limit": u[2] - c * h[2] ** p}


def ladder_orders(hs: list[float], us: list[float]) -> list[dict[str, Any]]:
    return [{"levels": [hs[i], hs[i + 1], hs[i + 2]], **three_point(hs[i:i + 3], us[i:i + 3])}
            for i in range(len(hs) - 2)]


def error_orders(hs: list[float], errors: list[float]) -> list[float | None]:
    out = []
    for i in range(len(hs) - 1):
        a, b = errors[i], errors[i + 1]
        out.append(math.log(abs(a / b)) / math.log(hs[i] / hs[i + 1]) if a and b else None)
    return out


def load(folder: Path) -> dict[str, Any]:
    return {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json") if not p.stem.startswith("points_")}


def helix_table(records: dict[str, Any]) -> dict[str, Any]:
    levels = sorted((k for k in records if k.startswith("helix_")), key=lambda k: -float(k.split("_")[1]))
    arms: dict[str, Any] = {}
    for name in levels:
        rec = records[name]
        for key, solve in rec["solves"].items():
            if ":warm" in key:
                continue
            arm = arms.setdefault(key, {"spacing_mm": [], "h_m": [], **{q: [] for q in QUANTITIES}, "methods": [],
                                        "seconds": []})
            arm["spacing_mm"].append(rec["spacing_mm"])
            arm["h_m"].append(rec["mean_spacing_m"])
            for q in QUANTITIES:
                arm[q].append(solve.get(q))
            arm["methods"].append(solve.get("method"))
            arm["seconds"].append(solve.get("seconds"))
    for key, arm in arms.items():
        arm["errors_pre"] = {q: [None if v is None else v / REFERENCE[q]["pre"][0] - 1.0 for v in arm[q]]
                             for q in REFERENCE}
        arm["inside_band_pre"] = {q: [None if v is None else abs(v - REFERENCE[q]["pre"][0]) <= REFERENCE[q]["pre"][1]
                                      for v in arm[q]] for q in REFERENCE}
        arm["errors_post"] = {q: [None if v is None else v / REFERENCE[q]["post"][0] - 1.0 for v in arm[q]]
                              for q in REFERENCE}
        if len(arm["h_m"]) >= 3 and all(v is not None for q in QUANTITIES for v in arm[q]):
            arm["orders"] = {q: ladder_orders(arm["h_m"], arm[q]) for q in QUANTITIES}
    return arms


def edges_table(records: dict[str, Any]) -> dict[str, Any]:
    edges = records.get("edges")
    if not edges:
        return {}
    out: dict[str, Any] = {"spheroid_exact": edges.get("spheroid_exact")}
    for body in ("spheroid", "cylinder"):
        rows = edges.get(body, {})
        spacings = sorted(rows, key=lambda s: -float(s))
        keys = sorted({k for s in spacings for k in rows[s]["solves"]})
        table = {}
        for key in keys:
            have = [s for s in spacings if key in rows[s]["solves"]]
            hs = [rows[s]["mean_h"] for s in have]
            got = [rows[s]["solves"][key] for s in have]
            entry: dict[str, Any] = {"spacing_mm": [float(s) for s in have], "h_m": hs,
                                     "F_x": [g["F_x"] for g in got], "T_x": [g["T_x"] for g in got],
                                     "F_y": [g["F_y"] for g in got], "T_y": [g["T_y"] for g in got],
                                     "methods": [g["method"] for g in got]}
            if body == "spheroid":
                entry["drag_error"] = [g["drag_error"] for g in got]
                entry["torque_error"] = [g["torque_error"] for g in got]
                entry["drag_error_orders"] = error_orders(hs, entry["drag_error"])
                entry["torque_error_orders"] = error_orders(hs, entry["torque_error"])
            if len(hs) >= 3:
                entry["orders"] = {q: ladder_orders(hs, entry[q]) for q in ("F_x", "T_x", "F_y", "T_y")}
                if body == "spheroid":
                    entry["orders_self"] = entry["orders"]
            table[key] = entry
        out[body] = {"quality": {s: rows[s]["quality"] for s in spacings}, "samples": {s: rows[s]["samples"]
                                                                                     for s in spacings},
                     "regions": {s: rows[s].get("regions") for s in spacings}, "arms": table}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = load(Path(args.records))
    analysis = {"reference": REFERENCE, "route_b_R_OxOx": ROUTE_B_OXOX, "helix": helix_table(records),
                "edges": edges_table(records), "sphere": records.get("sphere"),
                "samplings": {k: v for k, v in records.items() if k.startswith("sampling_")},
                "stages": records.get("stages"), "done": records.get("done")}
    Path(args.out).write_text(json.dumps(analysis, indent=1, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
