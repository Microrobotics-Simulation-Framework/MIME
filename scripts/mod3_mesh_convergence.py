#!/usr/bin/env python3
"""MOD-3 — mesh convergence at the worst corner (a/R=0.40, α=40°, ρ/R_ves≈0.97).

Holds n_max=80 FIXED (so mode-truncation error is constant across the sweep →
the CHANGE with N isolates the mesh effect) and varies body-mesh density:
N≈260, 500, 1000. Reads off the mesh-convergence rate of G1 + the coupling +
presym_err:
  - if G1/coupling converge cheaply with N → sufficient N is readable;
  - if convergence is slow → motivates the lubrication hybrid directly (the
    near-wall singular kernel needs ever-finer mesh).

Caveat: at ρ/R_ves≈0.97, n_max=80 may not be fully mode-converged (per-pair study
suggested ~140–220 there); fixing it isolates the mesh dependence but the absolute
values carry a constant mode-truncation offset. Cross-check against the a40 n_max
sweep in `mod3_g1_validation.jsonl` (whether a40 G1 is flat m80→m220).

Checkpointed/resumable jsonl (N=1000 is ~hours). Run: python scripts/mod3_mesh_convergence.py
"""
from __future__ import annotations
import os, sys, json, time, dataclasses
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from mod3_nearwall_convergence import R_direct, _Ry   # type: ignore
from dejongh_benchmark import compute_freespace_R, R_CYL_UMR  # type: ignore
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_fl_mesh

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                      "schwarz_vessel_helix", "output", "mod3_mesh_convergence.jsonl"))

R_VES_ND = 2.5          # a/R = 0.40
ALPHA = 40.0            # worst feasible corner
N_MAX, N_K, N_PHI = 80, 260, 200      # fixed truncation
MESHES = [(10, 14), (14, 20), (20, 28)]   # → N ≈ 260 / 500 / 1000


def _done(path):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l); d[r["key"]] = r
    return d


def main():
    done = _done(OUT); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"results -> {OUT} ({len(done)} done)  a/R=0.40 α=40° n_max={N_MAX} (fixed)", flush=True)
    for (nth, nz) in MESHES:
        mesh = dejongh_fl_mesh(9, n_theta=nth, n_zeta=nz)
        p0 = np.asarray(mesh.points); p0 = p0 - p0.mean(0)
        N = len(p0)
        key = f"N{N}"
        if key in done:
            print(f"  [skip] {key}", flush=True); continue
        ptsT = p0 @ _Ry(ALPHA).T
        rho_max = float(np.max(np.hypot(ptsT[:, 0], ptsT[:, 1])) / R_CYL_UMR / R_VES_ND)
        pts_nd = ptsT / R_CYL_UMR
        wts = np.asarray(mesh.weights) / R_CYL_UMR**2
        eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0
        t0 = time.time()
        try:
            Rw, presym = R_direct(pts_nd, wts, eps, R_VES_ND, N_MAX, N_K, N_PHI)
            Rf, _ = compute_freespace_R(dataclasses.replace(mesh, points=ptsT))
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True); continue
        Rw, Rf = np.asarray(Rw), np.asarray(Rf)
        rec = {"key": key, "N": N, "n_theta": nth, "n_zeta": nz,
               "rho_max_over_Rves": rho_max, "n_max": N_MAX,
               "G1_axial": float(abs(Rw[2, 2]) / abs(Rf[2, 2])),
               "G1_lat_x": float(abs(Rw[0, 0]) / abs(Rf[0, 0])),
               "coupling": float(np.linalg.norm(Rw[:2, 3:])),
               "presym_err": float(presym), "wall_s": round(time.time() - t0, 1)}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  N={N:>4} ρ/Rv={rho_max:.2f}: G1_ax={rec['G1_axial']:5.2f} "
              f"G1_lat={rec['G1_lat_x']:5.2f} cpl={rec['coupling']:5.2f} "
              f"presym={rec['presym_err']:5.2f}  [{rec['wall_s']:.0f}s]", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
