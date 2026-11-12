"""The hand-over's line at the CG's default cap (5,000): the held solve with the line moved to LINE (sqrt(K)/h), the
INTERIOR given, at D/h DH, on DRAWS drawn geometries of KIND: ``periodic`` (a plug closing a periodic tube, its period
2 to 4 diameters, its length 0.3 to 1, its centre 0.3 to 0.7 of the period; one cold steady step, as the line's first
step is) or one of the open vessel's cases (``inflow``, ``pressure``, ``gap``: gate 2's, two steady steps). Whether the
solve converged at every step, finite, and the reading against the closed form (periodic).

    python probe/line_cap.py KIND INTERIOR DH LINE DRAWS OUT
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests/flow")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import porous_handover_cases as hc  # noqa: E402
from porous_cases import R, plug_node, series_flux, steady  # noqa: E402

from mime.nodes.flow.fvm._solve import DEFAULT_CAP  # noqa: E402

kind, interior, dh, line, draws, out = (sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
                                        int(sys.argv[5]), Path(sys.argv[6]))
# An optional period range "low:high" (diameters) and its own seeds, for a supplement over the long periods.
periods, base = ((2.0, 4.0), 92_000) if len(sys.argv) < 8 else (tuple(map(float, sys.argv[7].split(":"))), 93_000)
settings = {"inflow": {}, "pressure": {"laws": ("pressure", "windkessel")}, "gap": {"radius": 0.25}}
rows = []
for seed in range(draws):
    started = time.perf_counter()
    with hc.line_at(line):
        if kind == "periodic":
            rng = np.random.default_rng(base + seed)
            ratio = float(math.exp(rng.uniform(math.log(1e-4), math.log(0.5))))
            cells_z = int(round(float(rng.uniform(*periods)) * dh))
            period, length, at = cells_z / dh, float(rng.uniform(0.3, 1.0)), float(rng.uniform(0.3, 0.7))
            K = ratio * (line / dh) ** 2
            node = plug_node(dh, [dict(radius=R, length=length, permeability=K, at=at)], length_cells=cells_z,
                             interior=interior, cap=DEFAULT_CAP)
            final = steady(node)
            closed = series_flux(K, length, period)
            extra = {"period": period, "length": length, "at": at, "ratio": ratio,
                     "reading": float(np.asarray(final["porous_flux"])[0]) / closed,
                     "lumen": float(final["lumen_flux"]) / closed}
        else:
            drawn = hc.forced_draw(seed)
            final = hc.vessel(drawn, cells=dh, interior=interior, **settings[kind])
            extra = {"length": drawn.length, "at": drawn.at, "ratio": drawn.ratio,
                     "forced_flagged": float(final["forced_flagged"])}
    total = np.asarray(final["solve_report_sum"], np.float64)
    finite = bool(all(np.all(np.isfinite(np.asarray(final[f"velocity_{c}"]))) for c in "xyz"))
    row = {"seed": seed, **extra, "finite": finite,
           "converged": bool(finite and total[2] + total[3] + total[4] == 0.0),
           "report_sum": total.tolist(), "report": np.asarray(final["solve_report"], np.float64).tolist(),
           "solve_flagged": float(final["solve_flagged"]), "seconds": time.perf_counter() - started}
    rows.append(row)
    print(json.dumps(row), flush=True)
    jax.clear_caches()
summary = {"kind": kind, "interior": interior, "dh": dh, "line": line, "cap": DEFAULT_CAP, "draws": len(rows),
           "converged": sum(row["converged"] for row in rows), "finite": sum(row["finite"] for row in rows),
           "stopped": [row["seed"] for row in rows if not row["converged"]],
           "most_iterations": max(row["report_sum"][0] for row in rows), "rows": rows}
out.write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
