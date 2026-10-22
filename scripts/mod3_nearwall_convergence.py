#!/usr/bin/env python3
"""MOD-3 / EP-2b prerequisite — near-wall Fourier-Bessel mode-convergence study.

Question (the gate): the image-series near-field fails as ρ/R_ves→1 (high tilt +
tight confinement) — the truncation/asymmetry noise floor (presym_err) reaches or
exceeds the rotation→lateral coupling signal. Is this FIXABLE by more modes, or
FUNDAMENTAL (the cylinder image series converges too slowly near the wall)?

Method: compute the tilted-body confined R via the DIRECT Fourier-Bessel solver
(`assemble_image_correction_matrix`, bypassing the table) at INCREASING truncation
(n_max, n_k, n_phi), at near-wall tilted configs. Watch presym_err, the drag
diagonal, and the rotation→lateral coupling converge (→ fixable) or plateau high
(→ fundamental, motivates the lubrication hybrid for ρ/R_ves>0.9).

Coarse body mesh (trend, not absolute) for tractability. Checkpointed/resumable
jsonl (slow — survives a kill). Run: python scripts/mod3_nearwall_convergence.py
"""
from __future__ import annotations
import os, sys, json, time, dataclasses
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from dejongh_benchmark import R_CYL_UMR, MU  # type: ignore
from t25_bem_cross_validation import assemble_system_matrix_numpy  # type: ignore
from mime.nodes.environment.stokeslet.cylinder_greens_function_v2 import (
    assemble_image_correction_matrix)
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_fl_mesh

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                      "schwarz_vessel_helix", "output", "mod3_nearwall_convergence.jsonl"))

# (n_max, n_k, n_phi) truncation levels — increasing
LEVELS = [(15, 80, 64), (25, 120, 96), (40, 160, 128), (60, 220, 160)]
# near-wall configs: (R_ves_nd, alpha_deg). a/R=0.40 → R_ves_nd=2.5 (tightest feasible).
CONFIGS = [(2.5, 15.0), (2.5, 30.0)]


def _Ry(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R_direct(pts_nd, wts_nd, eps, R_ves_nd, n_max, n_k, n_phi):
    N = len(pts_nd)
    A = assemble_system_matrix_numpy(pts_nd, wts_nd, eps, MU) + \
        np.asarray(assemble_image_correction_matrix(
            pts_nd, wts_nd, R_ves_nd, MU, n_max=n_max, n_k=n_k, n_phi=n_phi))
    e = np.eye(3); rhs = [np.tile(e[i], N) for i in range(3)] + \
        [np.cross(e[i], pts_nd).ravel() for i in range(3)]
    lu, piv = lu_factor(A); sol = lu_solve((lu, piv), np.column_stack(rhs))
    R_raw = np.zeros((6, 6))
    for c in range(6):
        wf = sol[:, c].reshape(N, 3) * wts_nd[:, None]
        R_raw[:3, c] = wf.sum(0); R_raw[3:, c] = np.cross(pts_nd, wf).sum(0)
    presym = float(np.max(np.abs(R_raw - R_raw.T)))
    return (R_raw + R_raw.T) / 2.0, presym


def _done(path):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l); d[r["key"]] = r
    return d


def main():
    mesh = dejongh_fl_mesh(9, n_theta=12, n_zeta=16)        # coarse → trend
    p0 = np.asarray(mesh.points); p0 = p0 - p0.mean(0)
    wts = np.asarray(mesh.weights) / R_CYL_UMR**2
    eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0
    done = _done(OUT); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"results -> {OUT} ({len(done)} done)  N={len(p0)}", flush=True)
    for R_ves_nd, al in CONFIGS:
        pts = (p0 @ _Ry(al).T) / R_CYL_UMR
        rho_max = float(np.max(np.hypot(pts[:, 0], pts[:, 1])) / R_ves_nd)
        for (nm, nk, nph) in LEVELS:
            key = f"Rves{R_ves_nd}:a{al:g}:m{nm}_{nk}_{nph}"
            if key in done:
                print(f"  [skip] {key}", flush=True); continue
            t0 = time.time()
            try:
                R, presym = R_direct(pts, wts, eps, R_ves_nd, nm, nk, nph)
            except Exception as e:
                print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True); continue
            cpl = float(np.linalg.norm(R[:2, 3:]))
            rec = {"key": key, "R_ves_nd": R_ves_nd, "alpha": al, "rho_max_over_Rves": rho_max,
                   "n_max": nm, "n_k": nk, "n_phi": nph,
                   "R_axial": float(abs(R[2, 2])), "R_lat_x": float(abs(R[0, 0])),
                   "coupling": cpl, "presym_err": presym, "wall_s": round(time.time() - t0, 1)}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  a={al:g}° ρ/Rv={rho_max:.2f} modes=({nm},{nk},{nph}): "
                  f"R_ax={rec['R_axial']:.1f} cpl={cpl:.2f} presym={presym:.2f}  "
                  f"({rec['wall_s']:.0f}s)", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
