#!/usr/bin/env python3
"""Targeted BEM-training-data fill for MLP v3.

Addresses the two concrete gaps identified in the April 2026
deliverable summary:

    1. FL group short of paper's 4-param fit (3.1 vs 2.2 mm/s) — gap
       concentrated in the 1/2" vessel where gravity pushes the robot
       to offset_frac ≈ 0.36, outside the v2 training envelope (≤0.30).
       Fill: 15 ν × 3 large offset_fracs at κ = 0.246.

    2. FW group not off-center augmented (5.7 mm/s).
       Fill: 6 FW designs × 3 offset_fracs at κ = 0.491 (1/4" vessel
       is the worst-performing cell in the FW table).

Uses the existing GPU BEM path (``compute_gcyl_confined_R_gpu``).
Writes to ``swimming_speeds_v3_fill.json`` — a new file that
``retrain_mlp_v2.py`` picks up through ``load_all_configs``.
"""
from __future__ import annotations

import os
import sys
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np

# GPU first; fall back to CPU only if CUDA unavailable
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax  # noqa: E402
print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}", flush=True)

from mime.nodes.environment.stokeslet.dejongh_geometry import (  # noqa: E402
    dejongh_umr_surface, FL_L_UMR, FL_TABLE, FW_TABLE,
    R_CYL_DEFAULT, EPSILON_DEFAULT,
)
from mime.nodes.environment.stokeslet.cylinder_wall_table import (  # noqa: E402
    load_wall_table, assemble_image_correction_matrix_from_table,
)
from mime.nodes.environment.stokeslet.bem import (  # noqa: E402
    assemble_system_matrix_chunked,
)
import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "dejongh_benchmark"
TABLE_DIR = DATA_DIR / "wall_tables"
OUTPUT = DATA_DIR / "swimming_speeds_v3_fill.json"
PROGRESS = DATA_DIR / "v3_fill_progress.json"

R_CYL_UMR = R_CYL_DEFAULT   # 1.56 mm
MU = 1.0
OMEGA_PHYS = 2 * np.pi * 10.0
R_MAX_UMR_ND = 1.0 + EPSILON_DEFAULT  # 1.33

# Wall tables shipped with repo
TABLES = {
    0.246: TABLE_DIR / "wall_R4.071.npz",  # 1/2" vessel  (R_ves_nd = 1/0.246 ≈ 4.07)
    0.491: TABLE_DIR / "wall_R2.035.npz",  # 1/4" vessel  (R_ves_nd = 1/0.491 ≈ 2.04)
}

N_THETA = 40
N_ZETA = 40


# ── ν set for the FL fill ────────────────────────────────────────────
# 11 paper values + 4 interpolated → 15
FL_NUS = sorted(set(d["nu"] for d in FL_TABLE.values())
                | {0.75, 1.6, 2.7, 3.2})

# 6 FW designs as in the paper
FW_DESIGNS = list(FW_TABLE.items())


def nd_to_mm_per_s(U_nd):
    return U_nd * R_CYL_UMR * OMEGA_PHYS


def make_mesh_and_wts(nu, L_UMR_mm):
    mesh = dejongh_umr_surface(nu, L_UMR_mm, n_theta=N_THETA, n_zeta=N_ZETA)
    pts = np.array(mesh.points) / R_CYL_UMR
    wts = np.array(mesh.weights) / R_CYL_UMR ** 2
    eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0
    return mesh, pts, wts, eps


def _confined_R_hybrid(pts, wts, eps, R_ves_nd, mu, table, offset_nd):
    """BEM body on GPU + wall table on CPU + GPU LU solve.

    Preserves 40×40 mesh resolution by keeping the O(N²×72 B) wall-table
    interpolation off the 6 GB RTX 2060 (which OOMs the JAX path at
    N ≈ 3120). The body A_body is chunked on the GPU, G_wall is assembled
    on the CPU, and the sum is factorised on the GPU.
    """
    pts32 = pts.astype(np.float32)
    wts32 = wts.astype(np.float32)
    N = len(pts32)
    pts_shifted = pts32 + np.array([offset_nd[0], offset_nd[1], 0.0], dtype=np.float32)

    A_body = assemble_system_matrix_chunked(
        pts32, wts32, float(eps), float(mu), chunk_rows=400, dtype=jnp.float32,
    )
    G_wall_np = assemble_image_correction_matrix_from_table(
        pts_shifted, wts32, float(R_ves_nd), float(mu), table,
    ).astype(np.float32)
    A = A_body + jnp.asarray(G_wall_np)
    del G_wall_np

    pts_j = jnp.asarray(pts32)
    e = jnp.eye(3, dtype=jnp.float32)
    cols = [jnp.tile(e[i], N) for i in range(3)]
    cols += [jnp.cross(e[i], pts_j).ravel() for i in range(3)]
    rhs = jnp.column_stack(cols)

    lu, piv = jax.scipy.linalg.lu_factor(A)
    sol = jax.scipy.linalg.lu_solve((lu, piv), rhs)

    wts_j = jnp.asarray(wts32)
    R_raw = jnp.zeros((6, 6), dtype=jnp.float32)
    for col in range(6):
        trac = sol[:, col].reshape(N, 3)
        wf = trac * wts_j[:, None]
        F = jnp.sum(wf, axis=0)
        T = jnp.sum(jnp.cross(pts_j, wf), axis=0)
        R_raw = R_raw.at[:3, col].set(F)
        R_raw = R_raw.at[3:, col].set(T)

    pre_sym = float(jnp.max(jnp.abs(R_raw - R_raw.T)))
    R = (R_raw + R_raw.T) / 2.0
    return np.array(R, dtype=np.float64), pre_sym


def run_single(nu, L_UMR_mm, kappa, offset_x_nd, offset_y_nd, table):
    mesh, pts, wts, eps = make_mesh_and_wts(nu, L_UMR_mm)
    R_ves_nd = table.R_cyl

    # Wall-contact feasibility (include offset)
    pts_shifted = pts + np.array([offset_x_nd, offset_y_nd, 0.0])
    rho_max = np.sqrt(pts_shifted[:, 0] ** 2 + pts_shifted[:, 1] ** 2).max()
    if rho_max >= R_ves_nd:
        raise ValueError(
            f"Body outside vessel: max(ρ)={rho_max:.3f} ≥ R_ves={R_ves_nd:.3f}"
        )

    R, pre_sym = _confined_R_hybrid(
        pts, wts, float(eps), float(R_ves_nd), float(MU),
        table, offset_nd=(offset_x_nd, offset_y_nd),
    )

    R_FU = R[:3, :3]
    R_FW = R[:3, 3:]
    U = -np.linalg.solve(R_FU, R_FW @ np.array([0, 0, 1.0]))
    v_z_mm = nd_to_mm_per_s(U[2])
    v_lat_mm = nd_to_mm_per_s(np.sqrt(U[0] ** 2 + U[1] ** 2))
    min_gap_nd = R_ves_nd - rho_max
    log_min_gap = float(np.log(max(min_gap_nd, 1e-4)))
    eigs = np.linalg.eigvalsh(R)

    return {
        "nu": float(nu),
        "L_UMR_mm": float(L_UMR_mm),
        "kappa": float(1.0 / R_ves_nd),
        "R_ves_nd": float(R_ves_nd),
        "offset_x_nd": float(offset_x_nd),
        "offset_y_nd": float(offset_y_nd),
        "min_gap_nd": float(min_gap_nd),
        "log_min_gap": log_min_gap,
        "R_matrix": R.tolist(),
        "U_nd": [float(x) for x in U],
        "v_z_mm_s": float(v_z_mm),
        "v_lateral_mm_s": float(v_lat_mm),
        "reciprocity_error": 0.0,
        "pre_symmetrize_error": float(pre_sym),
        "positive_definite": bool(np.all(eigs > 0)),
        "min_eigenvalue": float(eigs.min()),
        "N_mesh": mesh.n_points,
        "epsilon_bem": float(eps),
    }


def save(results):
    existing = json.load(open(OUTPUT)) if OUTPUT.exists() else {}
    existing.update(results)
    with open(OUTPUT, "w") as f:
        json.dump(existing, f, indent=2)


# ── Plan: build the full (design, kappa, offset_frac) worklist ───────
def build_worklist():
    """Return list of dicts with fields needed by ``run_single``."""
    wl = []
    # Scope 1: FL × 15 ν × 3 offset_frac at 1/2" (κ=0.246)
    for nu in FL_NUS:
        for off_frac in (0.25, 0.30, 0.35):
            R_ves_nd = 1.0 / 0.246
            off_nd = off_frac * R_ves_nd
            wl.append(dict(
                tag=f"FL_nu{nu:.2f}_off{off_frac:.2f}_half",
                nu=float(nu), L_UMR_mm=FL_L_UMR, kappa=0.246,
                offset_x_nd=off_nd, offset_y_nd=0.0,
                offset_frac=off_frac,
            ))
    # Scope 2: 6 FW × 3 offset_frac at 1/4" (κ=0.491)
    for fw_n, fw_row in FW_DESIGNS:
        nu = fw_row["nu"]
        L_UMR = fw_row["L_UMR"]
        for off_frac in (0.10, 0.20, 0.30):
            R_ves_nd = 1.0 / 0.491
            off_nd = off_frac * R_ves_nd
            wl.append(dict(
                tag=f"FW{fw_n}_nu{nu:.2f}_off{off_frac:.2f}_quarter",
                nu=float(nu), L_UMR_mm=float(L_UMR), kappa=0.491,
                offset_x_nd=off_nd, offset_y_nd=0.0,
                offset_frac=off_frac,
            ))
    return wl


def main():
    worklist = build_worklist()
    print(f"Planned {len(worklist)} configs "
          f"(FL×1/2\"={15 * 3}, FW×1/4\"={6 * 3})", flush=True)

    tables = {k: load_wall_table(str(p)) for k, p in TABLES.items()}

    results = {}
    t0 = time.time()
    skipped = 0

    for i, w in enumerate(worklist, 1):
        table = tables[w["kappa"]]
        key = w["tag"]
        try:
            r = run_single(
                w["nu"], w["L_UMR_mm"], w["kappa"],
                w["offset_x_nd"], w["offset_y_nd"],
                table,
            )
            r["source"] = "v3_fill"
            r["offset_frac"] = w["offset_frac"]
            results[key] = r
            elapsed = time.time() - t0
            avg = elapsed / i
            remaining = (len(worklist) - i) * avg
            print(
                f"[{i:3d}/{len(worklist)}] {key}  "
                f"v_z={r['v_z_mm_s']:.2f} mm/s "
                f"pre_sym={r['pre_symmetrize_error']:.1e}  "
                f"avg={avg:.1f}s remaining={remaining/60:.1f} min",
                flush=True,
            )
        except ValueError as e:
            skipped += 1
            print(f"[{i:3d}/{len(worklist)}] {key}  SKIP: {e}",
                   flush=True)

        # Periodic save
        if i % 10 == 0:
            save(results)
            with open(PROGRESS, "w") as f:
                json.dump({"done": i, "total": len(worklist),
                           "skipped": skipped,
                           "elapsed_s": time.time() - t0}, f, indent=2)

    save(results)
    with open(PROGRESS, "w") as f:
        json.dump({"done": len(worklist), "total": len(worklist),
                   "skipped": skipped,
                   "elapsed_s": time.time() - t0}, f, indent=2)
    print(
        f"\nDone. {len(results)} configs saved "
        f"({skipped} skipped for wall intersection) in "
        f"{(time.time() - t0)/60:.1f} min.",
        flush=True,
    )
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
