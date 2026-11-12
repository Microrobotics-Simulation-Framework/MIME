"""Camera spike R1: the tolerances, each from its mechanism, applied to ``r1_grad.py``'s outputs in both dtypes.

* **T1, reverse against forward mode** (same program, same dtype): the two differ only in the order the chain's local
  derivatives are multiplied and the terms summed. A pixel's chain is about 50 local Jacobians deep (pose, rotation, the
  disc's or the box edge's area, the reduction, the response, the centroid) and the sums are pairwise reductions over
  steps x pixels (about 2^18 terms: depth 18), so each term carries at most about (50 + 18) eps of relative error; the bound
  is ``128 eps S``, ``S`` the sum of the terms' absolute values (every pixel's and every projected centre's contribution to
  dJ/dtheta, from the forward tangents): the scale the result came from, never the result's own.
* **T2, against central differences**: on a ladder of steps sqrt(10) apart, the error of D(h) is bounded by the finer
  rung's distance, ``|D(h) - D(h / r)| / (1 - 1 / r)`` (exact for an error falling as h; the box edge is C1 and piecewise in
  its second derivative, so where a pixel's kink lies inside the step the error falls as h, and as h^2 elsewhere), plus the
  rounding of J at both rungs, ``2 r_J / h`` (``r_J`` the largest float32 - float64 difference of J over the ladder's
  points, measured; float64's taken as float32's times eps64 / eps32), plus the coarser rung's Richardson term against a
  chance agreement of the finer pair. The step judged is the one whose bound is least, chosen without the AD value.
* **T3, the render checkpointed against not**: the same derivative, recomputed: ``128 eps S`` as T1.
* **H1, held frames**: a held step's centroid derivative (reverse mode) equals its frame step's to ``128 eps`` of its
  parameter's largest per-step derivative (the same cotangent through the same render; what differs is the reverse
  pass's rounding, float32's checkpointed chunks recomputing the forward in another compiled order, which H2 bounds the
  same way), and the ball's speed's reaches it: the held step's derivative against its own central difference as T2.
* **H2, per step, reverse against forward**: ``128 eps`` of the parameter's largest per-step derivative.
* **H4, nothing of a moving body reaches the static cache**: its tangent in the ball's parameters is exactly zero.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

RATIO = math.sqrt(10.0)
EPS = {"float64": 2.220446049250313e-16, "float32": 1.1920928955078125e-07}


def _rounding_of_j(run: dict[str, Any], f64: dict[str, Any], f32: dict[str, Any] | None) -> float:
    """r_J for this run's dtype: measured in float32 against float64 over the ladder's points, scaled to float64."""
    if f32 is None:
        return 0.0
    gap = abs(f32["J"] - f64["J"])
    for name in f32["ladder"]:
        for a, b in zip(f32["ladder"][name], f64["ladder"][name], strict=True):
            gap = max(gap, abs(a["J_up"] - b["J_up"]), abs(a["J_down"] - b["J_down"]))
    return gap if run["dtype"] == "float32" else gap * EPS["float64"] / EPS["float32"]


def cd_bound(rows: list[dict[str, Any]], k: int, rounding: float, key: str = "D", index: Any = None) -> float:
    """The bound on |D(h_k) - true| from the ladder's neighbours (the finer one, k + 1, must exist)."""
    def value(row: dict[str, Any]) -> float:
        held = row[key]
        if index is not None:
            for i in index:
                held = held[i]
        return float(held)

    h = abs(rows[k]["h"])
    finer = abs(value(rows[k]) - value(rows[k + 1])) / (1.0 - 1.0 / RATIO)
    coarser = abs(value(rows[k - 1]) - value(rows[k])) / (RATIO ** 2 - 1.0) if k > 0 else 0.0
    return finer + coarser + 2.0 * rounding / h + 2.0 * rounding / abs(rows[k + 1]["h"]) / RATIO


def best_step(rows: list[dict[str, Any]], rounding: float, key: str = "D", index: Any = None) -> tuple[int, float]:
    bounds = [(cd_bound(rows, k, rounding, key, index), k) for k in range(len(rows) - 1)]
    bound, k = min(bounds)
    return k, bound


def judge(run: dict[str, Any], f64: dict[str, Any], f32: dict[str, Any] | None,
          checkpointed: dict[str, Any] | None = None) -> dict[str, Any]:
    eps = EPS[run["dtype"]]
    names = run["names"]
    rounding = _rounding_of_j(run, f64, f32)
    out: dict[str, Any] = {"dtype": run["dtype"], "r_J": rounding, "checks": {}}
    checks = out["checks"]
    for i, name in enumerate(names):
        g, s = run["g_vjp"][i], run["scale"][i]
        line = 128.0 * eps * s
        checks[f"T1 {name}"] = {"pass": abs(g - run["g_jvp"][i]) <= line, "diff": g - run["g_jvp"][i], "bound": line,
                                "in_eps_S": (g - run["g_jvp"][i]) / (eps * s) if s else None}
        if "ladder" in run:
            rows = run["ladder"][name]
            k, bound = best_step(rows, rounding)
            checks[f"T2 {name}"] = {"pass": abs(g - rows[k]["D"]) <= bound, "ad": g, "cd": rows[k]["D"],
                                    "h": rows[k]["h"], "bound": bound, "relative_bound": bound / abs(g) if g else None,
                                    "diff_over_bound": abs(g - rows[k]["D"]) / bound if bound else None}
        if checkpointed is not None:
            gc = checkpointed["g_vjp"][i]
            checks[f"T3 {name}"] = {"pass": abs(g - gc) <= line, "diff": g - gc, "bound": line, "bitwise": g == gc}
        if name in ("v0x", "z0") and "cache_tangent_maxabs" in run:
            checks[f"H4 {name}"] = {"pass": run["cache_tangent_maxabs"][i] == 0.0,
                                    "cache_tangent": run["cache_tangent_maxabs"][i]}
    if "dc_rev" in run:
        rev, fwd, frame_of = run["dc_rev"], run["dc_fwd"], run["frame_step_of"]
        start = run["continued_from"]
        held = [t for t in range(run["steps"]) if frame_of[t] - start != t and frame_of[t] - start >= 0]
        for i, name in enumerate(names):
            largest = max(abs(x) for step in fwd[i] for x in step) or 1.0
            worst = max((abs(rev[i][t][c] - rev[i][frame_of[t] - start][c]) for t in held for c in (0, 1)),
                        default=0.0)
            checks[f"H1 {name} held equals its frame"] = {"pass": worst <= 128.0 * eps * largest, "worst": worst,
                                                          "bound": 128.0 * eps * largest, "held_steps": len(held)}
            gap = max(abs(rev[i][t][c] - fwd[i][t][c]) for t in range(run["steps"]) for c in (0, 1))
            checks[f"H2 {name} per step rev vs fwd"] = {"pass": gap <= 128.0 * eps * largest, "worst": gap,
                                                        "bound": 128.0 * eps * largest}
        if "ladder" in run and held:
            # The ball's speed's derivative reaches every held step: each held step's u-derivative against its own
            # central difference, its rounding the float32 - float64 gap of the centroid (per step, at the start point).
            rows = run["ladder"]["v0x"]
            c_round = 0.0
            if f32 is not None:
                c_round = max(abs(a - b) for sa, sb in zip(f32["c"], f64["c"], strict=True) for a, b in zip(sa, sb,
                                                                                                           strict=True))
                if run["dtype"] == "float64":
                    c_round *= EPS["float64"] / EPS["float32"]
            failed = []
            for t in held:
                k, bound = best_step(rows, c_round, "dc", (t, 0))
                ad = rev[0][t][0]
                if not (abs(ad - rows[k]["dc"][t][0]) <= bound and ad != 0.0):
                    failed.append({"step": t, "ad": ad, "cd": rows[k]["dc"][t][0], "bound": bound})
            checks["H1 v0x reaches every held step (against CD)"] = {"pass": not failed, "failed": failed,
                                                                     "held_steps": len(held)}
    out["all_pass"] = all(c["pass"] for c in checks.values())
    return out


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    loaded = [json.loads(p.read_text()) for p in paths]
    f64 = next(d for d in loaded if d["dtype"] == "float64" and not d["render_checkpoint"])
    f32 = next((d for d in loaded if d["dtype"] == "float32" and not d["render_checkpoint"]), None)
    results = []
    for run in (f64, f32):
        if run is None:
            continue
        ck = next((d for d in loaded if d["dtype"] == run["dtype"] and d["render_checkpoint"]), None)
        results.append(judge(run, f64, f32, ck))
    print(json.dumps(results, indent=1))
    return 0 if all(r["all_pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
