#!/usr/bin/env python3
"""MOD-3 step 1 — validate the (old, 2026-05-21) confined wall tables against the
Fourier-Bessel DIRECT solver, checkpointed/resumable (same pattern as the G0
FVM re-validation sweeps).

Why: the wall tables predate recent BEM work; the table-GENERATION code
(cylinder_greens_function_v2) is unchanged since, but we don't rely on archaeology.
The table was BUILT by sampling the direct solver, so:
  table-interp vs direct  →  interpolation error  IF the code is unchanged,
                             a real discrepancy   IF the table is stale.
This both confirms the batch and anchors the MOD-3 tilted-body assembly (which
will lean on these tables near the wall, where interpolation error is worst).

Method: for each table, generate representative body-point clouds spanning the
cylinder (incl. NEAR-WALL ρ≳0.8 R_cyl, the MOD-3-relevant + worst-interp regime),
assemble G_wall via table-interp and via the direct solver, compare (relative
Frobenius overall + near-wall subset + max element). Several seeds per table give
resume granularity.

Pass: rel_fro < ~5% overall and near-wall (interpolation-level) → tables valid.
Resumable jsonl. Run: python scripts/mod3_table_validation.py [--smoke]
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    load_wall_table, assemble_image_correction_matrix_from_table)
from mime.nodes.environment.stokeslet.cylinder_greens_function_v2 import (
    assemble_image_correction_matrix)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(_ROOT, "experiments", "schwarz_vessel_helix", "output",
                   "mod3_table_validation.jsonl")

# EP-2a-relevant ratios (1/(a/R)): 6.667→a/R 0.15, 4.071→0.25, 2.500→0.40.
TABLES = {
    "R4.071": "data/dejongh_benchmark/wall_tables/wall_R4.071.npz",  # a/R 0.25 (primary)
    "R6.667": "data/wall_tables/wall_R6.667.npz",                     # a/R 0.15
    "R2.500": "data/dejongh_benchmark/wall_tables/wall_R2.500.npz",   # a/R 0.40 (tightest)
}


def _done(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); d[r["key"]] = r
    return d


def _save(rec):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _cloud(R_cyl, N, seed):
    """N body points spanning the cylinder, incl. near-wall ρ∈[0.05,0.95]·R_cyl."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.05, 0.95, N) * R_cyl          # uniform in ρ → near-wall covered
    phi = rng.uniform(0, 2 * np.pi, N)
    z = rng.uniform(-1.5 * R_cyl, 1.5 * R_cyl, N)      # |Δz| ≤ 3R_cyl ⊂ table L_max=5R_cyl
    pts = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
    return pts, rho


def _rel(A, B):
    nb = np.linalg.norm(B)
    return float(np.linalg.norm(A - B) / nb) if nb > 0 else float("nan")


def validate(name, path, N, seed):
    table = load_wall_table(os.path.join(_ROOT, path))
    R_cyl, mu = float(table.R_cyl), float(table.mu)
    pts, rho = _cloud(R_cyl, N, seed)
    wts = np.ones(N)
    G_t = np.asarray(assemble_image_correction_matrix_from_table(pts, wts, R_cyl, mu, table))
    G_d = np.asarray(assemble_image_correction_matrix(pts, wts, R_cyl, mu))
    rel_fro = _rel(G_t, G_d)
    max_el = float(np.max(np.abs(G_t - G_d)) / (np.max(np.abs(G_d)) + 1e-300))
    # near-wall subset (points with ρ > 0.8 R_cyl), block rows+cols
    nw = np.where(rho > 0.8 * R_cyl)[0]
    if len(nw) >= 2:
        idx = np.concatenate([3 * nw, 3 * nw + 1, 3 * nw + 2])
        idx.sort()
        nw_rel = _rel(G_t[np.ix_(idx, idx)], G_d[np.ix_(idx, idx)])
    else:
        nw_rel = float("nan")
    return {"R_cyl": R_cyl, "mu": mu, "N": N, "n_nearwall": int(len(nw)),
            "rel_fro": rel_fro, "max_el_rel": max_el, "nearwall_rel_fro": nw_rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        names, Ns, seeds = ["R4.071"], [40], [0]
    else:
        names, Ns, seeds = list(TABLES), [120], [0, 1, 2]

    done = _done(OUT)
    print(f"results -> {OUT} ({len(done)} done)", flush=True)
    for name in names:
        for N in Ns:
            for s in seeds:
                key = f"{name}:N{N}:seed{s}"
                if key in done:
                    print(f"  [skip] {key}", flush=True); continue
                t0 = time.time()
                try:
                    r = validate(name, TABLES[name], N, s)
                except Exception as e:
                    print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                    continue
                passed = r["rel_fro"] < 0.05 and (np.isnan(r["nearwall_rel_fro"])
                                                  or r["nearwall_rel_fro"] < 0.08)
                rec = {"key": key, "table": name, **r, "passed": bool(passed),
                       "wall_s": round(time.time() - t0, 1)}
                _save(rec)
                print(f"  {key} (R_cyl={r['R_cyl']:.3f}): rel_fro={r['rel_fro']*100:5.2f}%  "
                      f"near-wall={r['nearwall_rel_fro']*100:5.2f}%  max_el={r['max_el_rel']*100:5.1f}%  "
                      f"-> {'PASS' if passed else 'CHECK'}  ({rec['wall_s']:.0f}s)", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
