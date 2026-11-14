"""AP2 gate 5: allocate()'s currents, re-based (the head), against the base's (SciPy's BVLS on unweighted rows), on the
requests ``allocate_dump.py`` wrote from both trees: ``python probe/allocate_moves.py base.npz head.npz out.json``.

The control: both trees were asked the same (rows, targets, limits and kinds equal bit for bit). Per array and request
kind: where the minimiser is unique (``allocation_draws.unique`` on the unweighted rows), the largest movement in
amperes and as a share of the swap line (``allocation_draws.line`` about BVLS on rows normalised to order one, the
oracle), and every draw past the line, each read as a stop-short where the base's residual exceeds the oracle's past
the residual's line while the head's does not; where it is not unique, the residuals against each other and the
effort, ``|i|^2``, the head's against the base's."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "control"))
from allocation_draws import Problem, bvls, line, residual_line, unique  # noqa: E402

ROWS = {"field": 3, "gradient": 5, "both": 8, "small": 3}


def main(base_path: str, head_path: str, out_path: str) -> int:
    base, head = np.load(base_path), np.load(head_path)
    report: dict[str, object] = {}
    same = all(np.array_equal(base[k], head[k]) for k in base.files if not k.endswith("/currents"))
    report["same_requests"] = bool(same and set(base.files) == set(head.files))
    for system in sorted({k.split("/")[0] for k in base.files}):
        old, new = base[f"{system}/currents"], head[f"{system}/currents"]
        kinds, limits = head[f"{system}/kinds"], head[f"{system}/limits"]
        matrices, targets = head[f"{system}/matrices"], head[f"{system}/targets"]
        rows: dict[str, dict[str, object]] = {}
        for kind in ROWS:
            held: dict[str, object] = {"unique": 0, "not_unique": 0, "largest_movement_unique_A": 0.0,
                                       "largest_share_of_line_unique": 0.0, "past_line": [],
                                       "largest_movement_not_unique_A": 0.0,
                                       "effort_head_over_base_max": None, "effort_head_below_base": 0}
            ratios_residual, ratios_effort = [], []
            for i in np.flatnonzero(kinds == kind):
                r = ROWS[kind]
                problem = Problem(system, kind, matrices[i, :r], targets[i, :r], np.ones(r), float(limits[i]))
                moved = float(np.max(np.abs(new[i] - old[i])))
                a, b = problem.scaled("float64")
                if unique(problem):
                    held["unique"] += 1
                    oracle = bvls(problem)
                    allowed = line(problem, oracle, "float64")
                    held["largest_movement_unique_A"] = max(held["largest_movement_unique_A"], moved)
                    held["largest_share_of_line_unique"] = max(held["largest_share_of_line_unique"], moved / allowed)
                    if moved > allowed:
                        floor = residual_line(problem, "float64")
                        best = float(np.linalg.norm(a @ oracle - b))
                        was, now = float(np.linalg.norm(a @ old[i] - b)), float(np.linalg.norm(a @ new[i] - b))
                        held["past_line"].append({
                            "index": int(i), "moved_A": moved, "line_A": allowed,
                            "base_residual": was, "head_residual": now, "oracle_residual": best,
                            "stop_short": bool(was > best + floor and now <= best + floor),
                            "base_at_limit": int(np.sum(np.abs(old[i]) >= limits[i] * (1 - 1e-9))),
                            "head_at_limit": int(np.sum(np.abs(new[i]) >= limits[i] * (1 - 1e-9)))})
                else:
                    held["not_unique"] += 1
                    held["largest_movement_not_unique_A"] = max(held["largest_movement_not_unique_A"], moved)
                    was, now = float(np.linalg.norm(a @ old[i] - b)), float(np.linalg.norm(a @ new[i] - b))
                    ratios_residual.append((now - was) / max(residual_line(problem, "float64"), 1e-300))
                    ratios_effort.append(float(np.sum(new[i] ** 2)) / max(float(np.sum(old[i] ** 2)), 1e-300))
            if ratios_residual:
                held["residual_head_less_base_over_its_line_max"] = float(np.max(ratios_residual))
                held["effort_head_over_base_max"] = float(np.max(ratios_effort))
                held["effort_head_over_base_min"] = float(np.min(ratios_effort))
                held["effort_head_below_base"] = int(np.sum(np.asarray(ratios_effort) < 1 - 1e-9))
            past = held["past_line"]
            held["past_line_count"] = len(past)
            held["past_line_all_stop_short"] = all(p["stop_short"] for p in past)
            held["past_line"] = past[:10]
            rows[kind] = held
        report[system] = rows
        print(system, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "past_line"} for k, v in rows.items()}))
    Path(out_path).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
