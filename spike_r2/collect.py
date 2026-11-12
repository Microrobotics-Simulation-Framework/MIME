"""Camera spike R2's results, gathered from the probe's artifacts: ``python collect.py <download dir>`` prints one JSON
document of what the report quotes."""

from __future__ import annotations

import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: str) -> Any:
    return json.loads(Path(path).read_text())


def arm_of(path: str) -> str:
    for part in Path(path).parts:
        for prefix in ("probe-camera-r2-gates-", "probe-camera-r2-round2-"):
            if part.startswith(prefix):
                return part.removeprefix(prefix)
    return "?"


def main() -> None:
    root = sys.argv[1]
    out: dict[str, Any] = {}
    for path in sorted(glob.glob(f"{root}/**/detection_*.json", recursive=True)):
        d = load(path)
        planted, clean = d.get("planted", {}), d.get("clean", {})
        out.setdefault("detection", {})[arm_of(path)] = {
            "caught": planted.get("caught_of_runs") or f"{planted.get('caught')}/{planted.get('runs')}",
            "lower_bound_95": planted.get("lower_bound_95"), "missed": planted.get("missed_seeds"),
            "inconclusive": planted.get("inconclusive_seeds"), "clean_flagged": clean.get("flagged"),
            "clean_flagged_seeds": clean.get("flagged_seeds") or clean.get("seeds"),
            "clean_inconclusive": clean.get("inconclusive_seeds"), "every_run": d.get("every_run")}
    for path in sorted(glob.glob(f"{root}/**/swap_gate.json", recursive=True)):
        out.setdefault("swap", {})[arm_of(path)] = load(path)["summary"]
    for path in sorted(glob.glob(f"{root}/**/resolution_*.json", recursive=True)):
        d = load(path)
        out.setdefault("resolution", {})[arm_of(path)] = {k: d.get(k) for k in (
            "resolution", "resolution_refined", "plant_at", "false_alarms", "non_monotone")}
    runs = []
    for path in sorted(glob.glob(f"{root}/**/runs/*.json", recursive=True)):
        d = load(path)
        d["_arm"] = arm_of(path)
        runs.append(d)
    clean = [r for r in runs if r["variant"] == "double_where" and r["plant"] == "" and r["steps"] == 200]
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in clean:
        seen[(r["arm"], r["dtype"], r["seed"])] = r
    gate4: dict[str, Any] = {}
    for (arm, dtype) in sorted({(a, d) for a, d, _ in seen}):
        group = [r for (a, d, _), r in seen.items() if a == arm and d == dtype]
        rows = [(r, name, row) for r in group for name, row in r["rows"].items()]
        gate4[f"{arm}_{dtype}"] = {
            "runs": len(group), "verdicts": dict(sorted(_count(r["verdict"] for r in group).items())),
            "max_ratio_cd_discriminating": max((row["ratio_cd"] or 0) for _, _, row in rows if row["discriminates"]),
            "max_ratio_ad": max((row["ratio_ad"] or 0) for _, _, row in rows),
            "kinked_rows": sum(row["kinked"] for _, _, row in rows),
            "non_discriminating": dict(_count(name for _, name, row in rows if not row["discriminates"])),
            "line_over_dJdlimit": max(row["line_cd"] / abs(row["cd_h2"]) for r, name, row in rows
                                      if name == "limit" and row["cd_h2"]),
            "median_line_over_dJdlimit": statistics.median(row["line_cd"] / abs(row["cd_h2"]) for r, name, row in rows
                                                           if name == "limit" and row["cd_h2"]),
            "ticks_at_limit": [r["ticks_at_limit"] for r in sorted(group, key=lambda r: r["seed"])],
            "J": [round(r["J"], 4) for r in sorted(group, key=lambda r: r["seed"])],
            "build_s": statistics.median(r["clock_s"]["build"] for r in group),
            "total_s": statistics.median(r["clock_s"]["total"] for r in group)}
    out["gate4"] = gate4
    pairs = defaultdict(dict)
    for (arm, dtype, seed), r in seen.items():
        pairs[(dtype, seed)][arm] = r
    excess = {}
    for (dtype, seed), both in sorted(pairs.items()):
        if "camera" in both and "state" in both:
            excess.setdefault(dtype, []).append(both["camera"]["J"] / both["state"]["J"] - 1)
    out["camera_vs_state_J_excess"] = {d: {"min": min(v), "median": statistics.median(v), "max": max(v), "n": len(v)}
                                       for d, v in excess.items()}
    grad_excess: dict[str, list[float]] = defaultdict(list)
    for (dtype, seed), both in pairs.items():
        if "camera" in both and "state" in both:
            for name in ("kp", "kd", "limit"):
                c, s = both["camera"]["rows"][name]["reverse"], both["state"]["rows"][name]["reverse"]
                grad_excess[f"{dtype}_{name}"].append(c / s - 1 if s else math.nan)
    out["camera_vs_state_grad_ratio_minus_1"] = {k: {"min": min(v), "median": statistics.median(v), "max": max(v)}
                                                 for k, v in sorted(grad_excess.items())}
    f32 = defaultdict(list)
    for (arm, dtype, seed), r in seen.items():
        if dtype == "float32" and (arm, "float64", seed) in seen:
            wide = seen[(arm, "float64", seed)]
            f32[f"{arm}_J"].append(abs(r["J"] / wide["J"] - 1))
            for name in ("kp", "kd", "ki", "limit"):
                g64 = wide["rows"][name]["reverse"]
                if g64:
                    f32[f"{arm}_{name}"].append(abs(r["rows"][name]["reverse"] / g64 - 1))
    out["float32_vs_float64_relative"] = {k: {"median": statistics.median(v), "max": max(v)} for k, v in sorted(f32.items())}
    horizon = []
    for r in runs:
        if r["variant"] == "double_where" and r["plant"] == "" and r["steps"] != 200 or "kd0.3" in str(r.get("_arm")):
            pass
    for path in sorted(glob.glob(f"{root}/**/runs/*.json", recursive=True)):
        r = load(path)
        if r["steps"] == 200 or r["variant"] != "double_where" or r["plant"]:
            continue
        horizon.append({"arm": r["arm"], "dtype": r["dtype"], "steps": r["steps"], "seed": r["seed"],
                        "kd_scale_tag": Path(path).name.split("_kd")[1].split("_")[0], "verdict": r["verdict"], "J": r["J"],
                        "grad": {k: v["reverse"] for k, v in r["rows"].items()},
                        "ratio_cd": {k: v["ratio_cd"] for k, v in r["rows"].items()},
                        "ratio_ad": {k: v["ratio_ad"] for k, v in r["rows"].items()},
                        "kinked": [k for k, v in r["rows"].items() if v["kinked"]],
                        "non_discriminating": [k for k, v in r["rows"].items() if not v["discriminates"]],
                        "total_s": r["clock_s"]["total"], "reverse_s": r["clock_s"]["reverse"],
                        "ticks_at_limit": r["ticks_at_limit"]})
    out["horizon"] = sorted(horizon, key=lambda h: (h["kd_scale_tag"], h["arm"], h["dtype"], h["steps"], h["seed"]))
    sweeps = {}
    for path in sorted(glob.glob(f"{root}/**/sweep_*.json", recursive=True)):
        d = load(path)
        rows = d["rows"]
        kinks, agree = [], []
        for i in range(1, len(rows)):
            if rows[i]["pattern"] != rows[i - 1]["pattern"] or rows[i]["ticks_at_limit"] != rows[i - 1]["ticks_at_limit"]:
                before, after = rows[i - 1]["dJ"], rows[i]["dJ"]
                kinks.append({"between": [rows[i - 1]["value"], rows[i]["value"]], "dJ": [before, after],
                              "jump_rel": (after - before) / abs(before) if before else None,
                              "ticks": [rows[i - 1]["ticks_at_limit"], rows[i]["ticks_at_limit"]]})
        for i in range(1, len(rows) - 1):
            if rows[i - 1]["pattern"] == rows[i]["pattern"] == rows[i + 1]["pattern"] and rows[i]["dJ"]:
                agree.append(abs(rows[i]["neighbour_cd"] / rows[i]["dJ"] - 1))
        sweeps[Path(path).stem] = {"points": len(rows), "kinks": len(kinks), "kink_list": kinks[:30],
                                   "all_finite": all(r["finite"] for r in rows),
                                   "J_range": [min(r["J"] for r in rows), max(r["J"] for r in rows)],
                                   "smooth_neighbour_cd_rel_median": statistics.median(agree) if agree else None,
                                   "smooth_neighbour_cd_rel_max": max(agree) if agree else None,
                                   "largest_abs_jump_rel": max((abs(k["jump_rel"]) for k in kinks if k["jump_rel"] is not None),
                                                               default=None)}
    out["sweeps"] = sweeps
    print(json.dumps(out, indent=1, default=str))


def _count(items: Any) -> dict[Any, int]:
    found: dict[Any, int] = defaultdict(int)
    for item in items:
        found[item] += 1
    return dict(found)


if __name__ == "__main__":
    main()
