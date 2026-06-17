#!/usr/bin/env python3
"""MOD-3 — disentangle mesh-vs-mode for the rising near-wall presym (EP-2b gate).

The mesh study showed presym_err RISES with N at the worst corner (13.46 @N=260
→ 21.75 @N=532, ρ/R_ves 0.85→0.925), falsifying "finer mesh alone fixes the
coupling." Two mechanistically different causes (different fixes):
  (a) MODE deficit — n_max=80 is insufficient at the now-closer ρ/R=0.925
      (per-pair study hinted ρ/R≈0.96 needs n_max~140–220). Fix: raise n_max.
  (b) Genuine near-wall SINGULARITY — the kernel asymmetry grows as ρ/R→1
      regardless of modes. Fix: a lubrication/analytic near-wall kernel (hybrid).

Discriminator: at FIXED N=532, sweep n_max = 80 / 140 / 220.
  - presym DROPS as n_max rises  → mode deficit (a)
  - presym STAYS high            → genuine singularity (b) → hybrid justified

(n_max=80 @N=532 = 21.75 already in mod3_mesh_convergence.jsonl, same code path;
this run adds 140 and 220 for the discriminating trend.) Checkpointed/resumable.
Run: python scripts/mod3_disentangle.py
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
                      "schwarz_vessel_helix", "output", "mod3_disentangle.jsonl"))

R_VES_ND, ALPHA = 2.5, 40.0          # worst corner, a/R=0.40
N_THETA, N_ZETA = 14, 20             # N≈532 (matches mesh-study 2nd point)
LEVELS = [(140, 400, 300), (220, 600, 400)]   # n_max 80 already in mesh-study jsonl


def _done(path):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l); d[r["key"]] = r
    return d


def main():
    mesh = dejongh_fl_mesh(9, n_theta=N_THETA, n_zeta=N_ZETA)
    p0 = np.asarray(mesh.points); p0 = p0 - p0.mean(0); N = len(p0)
    ptsT = p0 @ _Ry(ALPHA).T
    rho_max = float(np.max(np.hypot(ptsT[:, 0], ptsT[:, 1])) / R_CYL_UMR / R_VES_ND)
    pts_nd = ptsT / R_CYL_UMR
    wts = np.asarray(mesh.weights) / R_CYL_UMR**2
    eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0
    Rf, _ = compute_freespace_R(dataclasses.replace(mesh, points=ptsT)); Rf = np.asarray(Rf)
    done = _done(OUT); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"results -> {OUT} ({len(done)} done)  N={N} ρ/Rv={rho_max:.3f} α={ALPHA}°", flush=True)
    print(f"  (compare against mesh-study N={N} n_max=80: presym≈21.75)", flush=True)
    for (nm, nk, nph) in LEVELS:
        key = f"N{N}:m{nm}"
        if key in done:
            print(f"  [skip] {key}", flush=True); continue
        t0 = time.time()
        try:
            Rw, presym = R_direct(pts_nd, wts, eps, R_VES_ND, nm, nk, nph)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True); continue
        rec = {"key": key, "N": N, "rho_max_over_Rves": rho_max, "n_max": nm,
               "G1_axial": float(abs(Rw[2, 2]) / abs(Rf[2, 2])),
               "G1_lat_x": float(abs(Rw[0, 0]) / abs(Rf[0, 0])),
               "coupling": float(np.linalg.norm(Rw[:2, 3:])),
               "presym_err": float(presym), "wall_s": round(time.time() - t0, 1)}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  n_max={nm:>3}: G1_ax={rec['G1_axial']:5.2f} cpl={rec['coupling']:5.2f} "
              f"presym={rec['presym_err']:6.2f}  [{rec['wall_s']:.0f}s]", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
