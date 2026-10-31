"""M1's convergence ladder at A, read from ``a0f_solves.level``'s records (register sections 1 B5, 2 b and 6 M1). No
measured speed enters. Predictions: ``predictions_a0f_stage1.json``, committed before any run.

For each sequence (the floored speed at blob 0.5, the same at blob 0.35, and the free-space speed at each blob):

- the speed at each level is the eight phases' mean axial speed (free space: phase 0's pose, which free space does not
  prefer); the spacing ``h`` of a level is the BEM's own mean spacing, the square root of the samples' area over their
  count, to which the blob is proportional (the nominal spacings' ratio is reported beside it);
- **B5's rules** (with review 1 #13, review 2 N16 and the light check's F4): at least four spacings; the values
  monotone; each consecutive triplet's observed order (from its three spacings, which need not share one ratio) in
  [0.5, 3]; the orders' pairwise spread at most 0.5. If any fails, the band is **unbounded**;
- otherwise the band is the Richardson estimate at the finest triplet's observed order: ``u_num = |f_finest - L|``,
  not divided by sqrt 3 (B5 item 4), and **the model value is the finest computed value** (B5 item 3);
- the gate is the register's at A, ``u_num <= 0.068 mm/s`` (section 1 B5, decided 2026-10-31).

**The common-limit check** (section 2 b): the blob-0.35 sequence must extrapolate to the blob-0.5 limit within the
bands, ``|L_0.5 - L_0.35| <= 2 sqrt(u_0.5^2 + u_0.35^2)`` (B2's k = 2 on the two numerical bands), or the numerical
band is unbounded; it cannot be run where either sequence's band is unbounded, and then the band is unbounded too.

**The margin scan** (section 6 M1): at each level and phase the first scale of the wall's correction (x1.000 to x1.040
by 0.0025) at which the resistance's smallest non-dimensional eigenvalue is not positive, and its least over the
phases. It **widens** if that least is non-decreasing from the coarsest level to the finest and larger at the finest
(a level with no onset by x1.040 counts as beyond the scan).

**Controls** on the analysis itself (each must give its planted verdict): an order-0.3 sequence (B8 control 5) is
unbounded; an exact order-1 sequence recovers its limit and order; a non-monotone sequence is unbounded; orders 1.0
and 1.7 (a spread of 0.7) are unbounded.

    python scripts/sim2real/m1_ladder.py --levels DIR --stage1 FLOORED.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

OMEGA = 2.0 * math.pi * 10.0
GATE_MM_S = 0.068
ORDER_RANGE = (0.5, 3.0)
ORDER_SPREAD = 0.5
MIN_SPACINGS = 4
SCAN_TOP = 1.04


def order_of(h: tuple[float, float, float], f: tuple[float, float, float]) -> float | None:
    """The observed order of a triplet ``h1 > h2 > h3`` with values ``f``: the ``p`` with
    ``(f1 - f2) / (f2 - f3) = (h1^p - h2^p) / (h2^p - h3^p)``; None where the increments change sign or vanish."""
    d1, d2 = f[0] - f[1], f[1] - f[2]
    if d1 == 0.0 or d2 == 0.0 or (d1 > 0) != (d2 > 0):
        return None
    q = d1 / d2

    def g(p: float) -> float:
        if abs(p) < 1e-12:
            return math.log(h[0] / h[1]) / math.log(h[1] / h[2]) - q
        return (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p) - q

    lo, hi = -10.0, 20.0
    if g(lo) * g(hi) > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def limit_at(h: tuple[float, float], f: tuple[float, float], p: float) -> float:
    """Richardson's limit from the two finest values at order ``p``."""
    return f[1] + (f[1] - f[0]) * h[1] ** p / (h[0] ** p - h[1] ** p)


def b5(h: list[float], f: list[float]) -> dict[str, Any]:
    """B5's verdict on a sequence, coarse to fine."""
    out: dict[str, Any] = {"h": h, "values": f, "spacings": len(h)}
    reasons = []
    if len(h) < MIN_SPACINGS:
        reasons.append(f"{len(h)} spacings, fewer than {MIN_SPACINGS}")
    increments = np.diff(f)
    monotone = bool(np.all(increments > 0) or np.all(increments < 0))
    out["monotone"] = monotone
    if not monotone:
        reasons.append("the values are not monotone")
    orders = [order_of((h[i], h[i + 1], h[i + 2]), (f[i], f[i + 1], f[i + 2])) for i in range(len(h) - 2)]
    out["orders"] = orders
    if any(p is None for p in orders):
        reasons.append("a triplet has no order (its increments change sign)")
    else:
        if any(not (ORDER_RANGE[0] <= p <= ORDER_RANGE[1]) for p in orders):
            reasons.append(f"an order outside {list(ORDER_RANGE)}")
        spread = max(orders) - min(orders) if orders else 0.0
        out["order_spread"] = spread
        if spread > ORDER_SPREAD:
            reasons.append(f"the orders' pairwise spread {spread:.3f} exceeds {ORDER_SPREAD}")
    if orders and orders[-1] is not None:
        p = orders[-1]
        out["limit_finest_triplet"] = limit_at((h[-2], h[-1]), (f[-2], f[-1]), p)
        out["richardson_estimate"] = abs(f[-1] - out["limit_finest_triplet"])
        if all(o is not None for o in orders):
            mean = float(np.mean(orders))
            out["limit_at_mean_order"] = limit_at((h[-2], h[-1]), (f[-2], f[-1]), mean)
    out["bounded"] = not reasons
    out["reasons"] = reasons
    out["model_value"] = f[-1]
    out["u_num"] = out.get("richardson_estimate") if out["bounded"] else None
    return out


def common_limit(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if not (a["bounded"] and b["bounded"]):
        return {"runnable": False, "holds": False,
                "note": "a sequence's band is unbounded, so the check cannot run and the numerical band is unbounded"}
    difference = abs(a["limit_finest_triplet"] - b["limit_finest_triplet"])
    allowed = 2.0 * math.hypot(a["u_num"], b["u_num"])
    return {"runnable": True, "difference": difference, "allowed": allowed, "holds": difference <= allowed,
            "ratio": difference / allowed if allowed > 0 else math.inf}


def controls(h: list[float]) -> dict[str, Any]:
    hh = np.asarray(h)
    found = {}
    fake = list(5.0 + 2.0 * hh ** 0.3)
    found["order_0.3_unbounded"] = not b5(h, fake)["bounded"]
    exact = list(5.0 + 3.0 * hh)
    verdict = b5(h, exact)
    found["order_1_recovered"] = bool(verdict["bounded"] and abs(verdict["limit_finest_triplet"] - 5.0) < 1e-9
                                      and all(abs(p - 1.0) < 1e-6 for p in verdict["orders"]))
    wobble = list(5.0 + 3.0 * hh)
    wobble[len(h) // 2] += 0.3
    found["non_monotone_unbounded"] = not b5(h, wobble)["bounded"]
    if len(h) >= 4:
        # orders about 1.0 and 1.7: a value per level from piecewise powers
        mixed = [5.0 + 3.0 * hh[0], 5.0 + 3.0 * hh[1], 5.0 + 3.0 * hh[2]]
        p2 = 1.7
        target = (mixed[1] - mixed[2]) / ((hh[1] ** p2 - hh[2] ** p2) / (hh[2] ** p2 - hh[3] ** p2))
        mixed.append(mixed[2] - target)
        found["spread_0.7_unbounded"] = not b5(h, mixed)["bounded"]
    found["all_caught"] = all(v for v in found.values() if v is not None)
    return found


def margin(levels: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for level in levels:
        onsets = []
        for k in range(8):
            scan = level["phases"][str(k)]["scan"]
            ordered = sorted(scan.values(), key=lambda r: r["scale"])
            first = next((r["scale"] for r in ordered if r["min_eigenvalue_nd"] <= 0.0), None)
            onsets.append(first)
        known = [o for o in onsets if o is not None]
        rows.append({"spacing_mm": level["rung"]["spacing_mm"], "onset_per_phase": onsets,
                     "least_onset": min(known) if known else None,
                     "beyond_scan": not known})
    values = [r["least_onset"] if r["least_onset"] is not None else math.inf for r in rows]
    widens = all(b >= a for a, b in zip(values, values[1:], strict=False)) and values[-1] > values[0]
    return {"levels": rows, "widens": widens}


def speed(r: list[list[float]]) -> float:
    m = np.asarray(r)
    return float(-m[0, 3] * OMEGA / m[0, 0])


def analyse(levels: list[dict[str, Any]], stage1: dict[str, Any] | None) -> dict[str, Any]:
    levels = sorted(levels, key=lambda lv: -lv["rung"]["spacing_mm"])
    h = [lv["rung"]["mean_spacing_m"] * 1e3 for lv in levels]
    nominal = [lv["rung"]["spacing_mm"] for lv in levels]
    sequences: dict[str, list[float]] = {"floored_blob_0.5": [], "floored_blob_0.35": [], "free_blob_0.5": [],
                                         "free_blob_0.35": []}
    every: dict[str, Any] = {"floored_min_eigenvalue_nd": [], "wall_accepted": True, "finite": True}
    for lv in levels:
        phases = [lv["phases"][str(k)] for k in range(8)]
        nominal_rows = [p["scan"]["1.0000"] for p in phases]
        sequences["floored_blob_0.5"].append(1e3 * float(np.mean([speed(r["resistance_si"]) for r in nominal_rows])))
        sequences["floored_blob_0.35"].append(1e3 * float(np.mean([speed(p["blob_0.35"]["resistance_si"])
                                                                   for p in phases])))
        sequences["free_blob_0.5"].append(1e3 * speed(lv["free_space"]["0.5"]["resistance_si"]))
        sequences["free_blob_0.35"].append(1e3 * speed(lv["free_space"]["0.35"]["resistance_si"]))
        every["floored_min_eigenvalue_nd"].append(min(min(r["min_eigenvalue_nd"] for r in nominal_rows),
                                                      min(p["blob_0.35"]["min_eigenvalue_nd"] for p in phases)))
        every["wall_accepted"] = every["wall_accepted"] and all(p["wall_accepted"] for p in phases)
        every["finite"] = every["finite"] and all(r["finite"] for r in nominal_rows)
    verdicts = {name: b5(h, values) for name, values in sequences.items()}
    out: dict[str, Any] = {"nominal_spacing_mm": nominal, "mean_spacing_mm": h,
                           "mean_spacing_ratios": [h[i + 1] / h[i] for i in range(len(h) - 1)],
                           "nominal_ratios": [nominal[i + 1] / nominal[i] for i in range(len(nominal) - 1)],
                           "sequences": verdicts, "every_reading": every,
                           "common_limit_floored": common_limit(verdicts["floored_blob_0.5"],
                                                                verdicts["floored_blob_0.35"]),
                           "common_limit_free": common_limit(verdicts["free_blob_0.5"], verdicts["free_blob_0.35"]),
                           "margin": margin(levels), "controls": controls(h),
                           "control_own_against_node": [lv["control_own_against_node"] for lv in levels]}
    floored = verdicts["floored_blob_0.5"]
    numerical_bounded = floored["bounded"] and out["common_limit_floored"]["holds"]
    out["numerical_band_at_A"] = {"bounded": numerical_bounded,
                                  "u_num_mm_s": floored["u_num"] if numerical_bounded else None,
                                  "gate_mm_s": GATE_MM_S,
                                  "gate_holds": bool(numerical_bounded and floored["u_num"] <= GATE_MM_S),
                                  "model_value_mm_s": floored["model_value"]}
    rises = {name: (v[-1] - v[0]) / v[0] for name, v in sequences.items()}
    out["relative_rise_coarsest_to_finest"] = rises
    if stage1 is not None:
        match = [lv for lv in levels if abs(lv["rung"]["spacing_mm"] - stage1["rung"]["spacing_mm"]) < 1e-9]
        if match:
            lv = match[0]
            diffs = []
            for k in range(8):
                a = np.asarray(lv["phases"][str(k)]["scan"]["1.0000"]["resistance_si"])
                b = np.asarray(stage1["phases"][str(k)]["resistance_si"])
                diffs.append(float(np.max(np.abs(a - b)) / np.max(np.abs(b))))
            out["control_stage1_repeat_largest_relative"] = max(diffs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--levels", required=True, help="a folder of level_<spacing>.json")
    parser.add_argument("--stage1", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    levels = [json.loads(p.read_text()) for p in sorted(Path(args.levels).glob("level_*.json"))]
    levels = [lv for lv in levels if lv.get("complete")]
    stage1 = json.loads(Path(args.stage1).read_text()) if args.stage1 else None
    payload = analyse(levels, stage1)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
