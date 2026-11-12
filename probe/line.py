"""The hand-over's line in open boxes: the held solve with the line moved to LINE (sqrt(K)/h) and INTERIOR, on 50 draws
of one of gate 2's open cases (inflow, pressure, gap, twin): whether it converged at every step, finite.

    python probe/line.py CASE LINE INTERIOR OUT
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "tests/flow")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import porous_handover_cases as hc  # noqa: E402

case, line, interior, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
draws = int(sys.argv[5]) if len(sys.argv) > 5 else 50
settings = {"inflow": {}, "pressure": {"laws": ("pressure", "windkessel")}, "gap": {"radius": 0.25}}
rows = []
for seed in range(draws):
    with hc.line_at(line):
        if case == "twin":
            point, ratio = hc.quiet_draw(seed)
            final = hc.occluded_twin(point, ratio, interior=interior)
            cells, extra = point.cells, {"length": point.length, "ratio": ratio}
        else:
            drawn = hc.forced_draw(seed)
            final = hc.vessel(drawn, interior=interior, **settings[case])
            cells, extra = 16, {"length": drawn.length, "at": drawn.at, "ratio": drawn.ratio}
    total = np.asarray(final["solve_report_sum"], np.float64)
    finite = bool(all(np.all(np.isfinite(np.asarray(final[f"velocity_{c}"]))) for c in "xyz"))
    row = {"seed": seed, "cells": cells, **extra, "finite": finite,
           "converged": bool(finite and total[2] + total[3] + total[4] == 0.0),
           "report_sum": total.tolist(), "report": np.asarray(final["solve_report"], np.float64).tolist(),
           "solve_flagged": float(final["solve_flagged"]), "forced_flagged": float(final["forced_flagged"]),
           "porous_flux": float(np.asarray(final["porous_flux"])[0])}
    rows.append(row)
    print(json.dumps(row), flush=True)
    jax.clear_caches()
summary = {"case": case, "line": line, "interior": interior, "draws": len(rows),
           "converged": sum(row["converged"] for row in rows), "finite": sum(row["finite"] for row in rows),
           "stopped": [row["seed"] for row in rows if not row["converged"]], "rows": rows}
out.write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
