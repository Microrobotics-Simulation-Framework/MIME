#!/usr/bin/env python3
"""MOD-3 — validate the 6–12× G1 wall-amplification at the tight corner, RESOLVED.

The original R(α) run (`mod3_tilt_resistance.py`) used the wall TABLE, whose
near-wall values inherit the table's (low) build n_max — under-resolved exactly
where the body tilts toward the wall. With the underflow guard (`07f49f2`) the
DIRECT Fourier–Bessel solver now converges near the wall at higher n_max, so we
re-compute G1 = R_wall/R_free via the direct solver at increasing n_max and watch
it converge — confirming (or correcting) the 6–12×.

Also **benchmarks wall time per n_max** (feeds the hybrid-lubrication cost study):
near-wall cells need n_max~80–220 vs ~15–40 in the bulk.

Moderate mesh (the wall/free RATIO is mesh-robust; absolute R differs from the
N=1680 table run). Checkpointed/resumable jsonl (slow — survives a kill).
Run: python scripts/mod3_g1_validation.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import dataclasses

from mod3_nearwall_convergence import R_direct, _Ry   # type: ignore
from dejongh_benchmark import compute_freespace_R, R_CYL_UMR  # type: ignore
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_fl_mesh

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                      "schwarz_vessel_helix", "output", "mod3_g1_validation.jsonl"))

# a/R=0.40 (R_ves_nd=2.5) tight corner. Per-config n_max levels (skip wasteful
# high modes where the config already converges).
CONFIGS = {
    0.0:  [(15, 80, 64), (40, 160, 128)],                                    # ρ/R≈0.51 bulk
    30.0: [(15, 80, 64), (40, 160, 128), (80, 260, 200), (140, 400, 300)],   # ρ/R≈0.88
    40.0: [(15, 80, 64), (40, 160, 128), (80, 260, 200), (140, 400, 300), (220, 600, 400)],  # ρ/R≈0.97
}
R_VES_ND = 2.5


def _done(path):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l); d[r["key"]] = r
    return d


def main():
    mesh = dejongh_fl_mesh(9, n_theta=10, n_zeta=14)
    p0 = np.asarray(mesh.points); p0 = p0 - p0.mean(0)
    wts = np.asarray(mesh.weights) / R_CYL_UMR**2
    eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0
    done = _done(OUT); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"results -> {OUT} ({len(done)} done)  N={len(p0)}  a/R=0.40", flush=True)

    free_cache = {}
    for al, levels in CONFIGS.items():
        ptsT = p0 @ _Ry(al).T
        rho_max = float(np.max(np.hypot(ptsT[:, 0], ptsT[:, 1])) / R_CYL_UMR / R_VES_ND)
        pts_nd = ptsT / R_CYL_UMR
        # free-space R (n_max-independent) once per config
        if al not in free_cache:
            Rf, _ = compute_freespace_R(dataclasses.replace(mesh, points=ptsT))
            free_cache[al] = np.asarray(Rf)
        Rf = free_cache[al]
        for (nm, nk, nph) in levels:
            key = f"a{al:g}:m{nm}"
            if key in done:
                print(f"  [skip] {key}", flush=True); continue
            t0 = time.time()
            try:
                Rw, presym = R_direct(pts_nd, wts, eps, R_VES_ND, nm, nk, nph)
            except Exception as e:
                print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True); continue
            wall_s = round(time.time() - t0, 1)
            g1ax = float(abs(Rw[2, 2]) / abs(Rf[2, 2]))
            g1lx = float(abs(Rw[0, 0]) / abs(Rf[0, 0]))
            cpl = float(np.linalg.norm(Rw[:2, 3:]))
            rec = {"key": key, "alpha": al, "rho_max_over_Rves": rho_max,
                   "n_max": nm, "n_k": nk, "n_phi": nph,
                   "G1_axial": g1ax, "G1_lat_x": g1lx, "coupling": cpl,
                   "presym_err": presym, "wall_s": wall_s, "N": len(p0)}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  α={al:g}° ρ/Rv={rho_max:.2f} n_max={nm:>3}: "
                  f"G1_ax={g1ax:5.2f} G1_lat={g1lx:5.2f} cpl={cpl:5.2f} presym={presym:5.2f}  "
                  f"[{wall_s:.0f}s]", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
