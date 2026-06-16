#!/usr/bin/env python3
"""MOD-3 step 2 — tilted-body (angle-of-attack) confined resistance R(α).

Builds the confined 6×6 resistance of the de Jongh FL-9 screw at a sweep of
angle-of-attack α (long axis tilted from the cylinder axis), by rotating the
body points about a transverse axis and reusing the validated table assembly
(`dejongh_benchmark.compute_R_matrix`). Table accuracy across the cylinder
domain — incl. near-wall, where tilted tips reach — was confirmed in MOD-3
step 1 (`mod3_table_validation.py`), so no per-α direct solve is needed.

Frame: screw long axis = body-z = cylinder axis at α=0. Tilt = rotate points
about body-y by α so the long axis makes angle α with the cylinder axis (the
angle of attack). Centred on the cylinder axis (offset added later for the 2-D
(α,d) grid).

Outputs per α (non-dim R, [F]=μaU, [T]=μa²ω):
  - lateral drag R[0,0] (in-tilt-plane) and R[1,1] (out-of-plane) vs the axial R[2,2]
  - the rotation→lateral off-diagonal coupling (≈0 at α=0/axis-aligned; grows with α)
  - max body-point ρ/R_ves (wall proximity → which α carry the ~1.5% near-wall table error)
Checkpointed/resumable jsonl. Run: python scripts/mod3_tilt_resistance.py [--table R4.071]
"""
from __future__ import annotations
import os, sys, json, time, argparse, dataclasses
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from dejongh_benchmark import compute_R_matrix, compute_freespace_R, R_CYL_UMR  # type: ignore
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_fl_mesh
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(_ROOT, "experiments", "schwarz_vessel_helix", "output",
                   "mod3_tilt_resistance.jsonl")
TABLES = {
    "R4.071": "data/dejongh_benchmark/wall_tables/wall_R4.071.npz",  # a/R 0.25 (full α feasible)
    "R6.667": "data/wall_tables/wall_R6.667.npz",                     # a/R 0.15
    "R2.500": "data/dejongh_benchmark/wall_tables/wall_R2.500.npz",   # a/R 0.40 (α≲30° feasible)
}


def _Ry(alpha_deg):
    a = np.radians(alpha_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _done(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); d[r["key"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="R4.071")
    ap.add_argument("--alphas", default="0,15,30,45,60,75")
    args = ap.parse_args()
    table = load_wall_table(os.path.join(_ROOT, TABLES[args.table]))
    R_ves_nd = float(table.R_cyl)
    alphas = [float(a) for a in args.alphas.split(",")]

    mesh = dejongh_fl_mesh(9, n_theta=24, n_zeta=36)
    pts0 = np.asarray(mesh.points)
    pts0 = pts0 - pts0.mean(0)                       # centre on origin
    done = _done(OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"results -> {OUT} ({len(done)} done)  table={args.table} R_ves_nd={R_ves_nd:.3f}", flush=True)
    for al in alphas:
        key = f"{args.table}:a{al:g}"
        if key in done:
            print(f"  [skip] {key}", flush=True); continue
        t0 = time.time()
        pts_rot = pts0 @ _Ry(al).T
        rho_max_nd = float(np.max(np.sqrt(pts_rot[:, 0]**2 + pts_rot[:, 1]**2)) / R_CYL_UMR)
        if rho_max_nd >= R_ves_nd:
            print(f"  {key}: INFEASIBLE (ρ_max/R_ves={rho_max_nd/R_ves_nd:.2f} ≥ 1)", flush=True)
            continue
        m = dataclasses.replace(mesh, points=pts_rot)
        try:
            Rw, presym_w = compute_R_matrix(m, table, R_ves_nd, offset_xy=(0.0, 0.0))
            Rf, presym_f = compute_freespace_R(m)         # free-space baseline (same numerics)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        Rw, Rf = np.asarray(Rw), np.asarray(Rf)
        # lateral-rotation coupling magnitude = ‖R[:2,3:]‖_F (lateral force x,y from any rotation)
        cpl_w = float(np.linalg.norm(Rw[:2, 3:]))
        cpl_f = float(np.linalg.norm(Rf[:2, 3:]))
        def g1(i, j):
            return float(abs(Rw[i, j]) / (abs(Rf[i, j]) + 1e-300))
        rec = {"key": key, "table": args.table, "alpha_deg": al,
               "Rw_lat_x": float(abs(Rw[0, 0])), "Rw_lat_y": float(abs(Rw[1, 1])),
               "Rw_axial": float(abs(Rw[2, 2])),
               "Rf_lat_x": float(abs(Rf[0, 0])), "Rf_lat_y": float(abs(Rf[1, 1])),
               "Rf_axial": float(abs(Rf[2, 2])),
               "G1_lat_x": g1(0, 0), "G1_lat_y": g1(1, 1), "G1_axial": g1(2, 2),
               "cpl_wall": cpl_w, "cpl_free": cpl_f, "cpl_wall_induced": cpl_w - cpl_f,
               "noise_floor": float(max(presym_w, presym_f)),   # truncation/asymmetry floor
               "rho_max_over_Rves": rho_max_nd / R_ves_nd,
               "presym_w": float(presym_w), "presym_f": float(presym_f),
               "Rw6x6": Rw.tolist(), "Rf6x6": Rf.tolist(),
               "wall_s": round(time.time() - t0, 1)}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  α={al:4.0f}°: G1[axial]={rec['G1_axial']:.2f} G1[lat_x]={rec['G1_lat_x']:.2f} "
              f"G1[lat_y]={rec['G1_lat_y']:.2f} | cpl wall={cpl_w:.2f} free={cpl_f:.2f} "
              f"induced={cpl_w-cpl_f:+.2f} (noise≲{rec['noise_floor']:.2f}) "
              f"ρ/Rv={rec['rho_max_over_Rves']:.2f}  ({rec['wall_s']:.0f}s)", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
