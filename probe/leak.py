"""The open vessel's leak at the 0.1h interior: the pressure-driven vessel at D/h CELLS, a plug half a diameter long
at mid-length closing it, resolved (the line moved to 1e-4), at sqrt(K)/h 0.01 to 0.12; the conductance G = Q / dp_faces
against K F_B(K), its intercept the flux the wall lets past the plug whatever its K (the leak), over G at each line.

    python probe/leak.py CELLS OUT
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "tests/flow")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import porous_handover_cases as hc  # noqa: E402

import mime.nodes.flow.fvm._porous as porous  # noqa: E402

cells, out = int(sys.argv[1]), Path(sys.argv[2])
roots = (0.01, 0.02, 0.03, 0.06, 0.12)
moved = 1e-4
rows = []
for root in roots:
    with hc.line_at(moved):
        final = hc.vessel(hc.Forced((root / moved) ** 2, 0.5, 1.0, 1.0), cells=cells,
                          laws=("pressure", "windkessel"), interior="0.1h")
    K = (root / cells) ** 2
    faces = np.asarray(final["porous_face_pressure"], np.float64)[0]
    total = np.asarray(final["solve_report_sum"], np.float64)
    row = {"root": root, "K": K, "KF": K * float(porous.wall_factor(K, 0.5)), "drop": float(faces[0] - faces[1]),
           "lumen_flux": float(final["lumen_flux"]), "porous_flux": float(np.asarray(final["porous_flux"])[0]),
           "opening_flux": np.asarray(final["opening_flux"], np.float64).tolist(), "report_sum": total.tolist(),
           "converged": bool(total[2] + total[3] + total[4] == 0.0)}
    rows.append(row)
    print(json.dumps(row), flush=True)
    jax.clear_caches()
fits = {}
x = np.asarray([row["KF"] for row in rows])
root = np.asarray(roots)
for name, flux in (("outlet", lambda r: abs(r["opening_flux"][1])),
                   ("lumen_flux", lambda r: r["lumen_flux"]), ("porous_flux", lambda r: r["porous_flux"])):
    g = np.asarray([flux(row) / row["drop"] for row in rows])
    held = {}
    # G = a K F_B (+ b K F_B sqrt(K)/h, the conductance's proportional error rising with sqrt(K), seen on the laptop's
    # D/h 8 smoke) + c: c the leak; each point weighted by 1 / G, so its relative error counts alike over two decades.
    for form, columns in (("linear", [x, np.ones_like(x)]), ("with_root", [x, x * root, np.ones_like(x)])):
        basis = np.stack(columns, axis=1)
        solved = np.linalg.lstsq(basis / g[:, None], np.ones_like(g), rcond=None)[0]
        residual = (g - basis @ solved) / g
        held[form] = {"coefficients": solved.tolist(), "intercept": float(solved[-1]),
                      "intercept_over_G": {str(r): float(solved[-1] / gi) for r, gi in zip(roots, g, strict=True)},
                      "largest_relative_residual": float(np.max(np.abs(residual))),
                      "relative_residuals": residual.tolist()}
    fits[name] = held
summary = {"cells": cells, "rows": rows, "fits": fits}
out.write_text(json.dumps(summary, indent=1))
print(json.dumps(fits, indent=1))
