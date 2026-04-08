#!/usr/bin/env python3
"""Phase 2: Discrete wall MFS at high resolution.

Tests whether increasing wall resolution (1280-10000 points) resolves
the Phase 0 Fx-Fz source-ratio trade-off (12% balanced error at 1280).

Uses direct LU for the wall system (feasible up to ~10k pts on 32 GB
RAM). For N_wall > 10k, use GMRES + stokeslet_matvec instead.

Reference: Liron-Shahar analytical table (<4% for all 6 entries).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh, cylinder_surface_mesh,
)
from mime.nodes.environment.stokeslet.cylinder_greens_function_v2 import (
    _assemble_free_space,
)

A = 1.0
MU = 1.0
R_CYL = A / 0.3

REF = {"F_x": 31.76, "F_z": 44.39, "T_z": 25.17}


def run_wall_mfs_direct(N_circ, N_axial, source_ratio,
                        cyl_length_R=15.0, max_gs_iter=30):
    """Discrete wall MFS with direct LU for the wall system."""
    cyl_len = cyl_length_R * R_CYL

    # Wall: collocation on cylinder surface, sources outside
    wc_mesh = cylinder_surface_mesh(
        center=(0, 0, 0), radius=R_CYL, length=cyl_len,
        n_circ=N_circ, n_axial=N_axial, cluster_center=False)
    ws_mesh = cylinder_surface_mesh(
        center=(0, 0, 0), radius=source_ratio * R_CYL, length=cyl_len,
        n_circ=N_circ, n_axial=N_axial, cluster_center=False)
    wall_colloc = wc_mesh.points
    wall_source = ws_mesh.points
    N_wall = len(wall_colloc)

    # Verify collocation ≠ source
    rho_c = np.sqrt(wall_colloc[:, 0]**2 + wall_colloc[:, 1]**2)
    rho_s = np.sqrt(wall_source[:, 0]**2 + wall_source[:, 1]**2)
    assert abs(rho_c.mean() - R_CYL) < 0.01, "Collocation not on cylinder"
    assert abs(rho_s.mean() - source_ratio * R_CYL) < 0.01, "Sources not offset"

    # Body MFS (sphere, projected centroids)
    body_mesh = sphere_surface_mesh(center=(0, 0, 0), radius=A, n_refine=2)
    pts = body_mesh.points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    body_colloc = pts / norms * A
    src_mesh = sphere_surface_mesh(center=(0, 0, 0), radius=0.7 * A, n_refine=2)
    pts_s = src_mesh.points
    norms_s = np.linalg.norm(pts_s, axis=1, keepdims=True)
    body_source = pts_s / norms_s * 0.7 * A
    N_body = len(body_colloc)

    # Assemble and LU-factorise body and wall systems
    t0 = time.time()
    G_body = _assemble_free_space(body_colloc, body_source, MU)
    body_lu, body_piv = lu_factor(G_body)
    print(f"    Body LU ({N_body} pts): {time.time()-t0:.1f}s")

    t0 = time.time()
    G_wall = _assemble_free_space(wall_colloc, wall_source, MU)
    wall_lu, wall_piv = lu_factor(G_wall)
    mem_gb = G_wall.nbytes / 1e9
    print(f"    Wall LU ({N_wall} pts, {mem_gb:.1f} GB): {time.time()-t0:.1f}s")

    # Cross-interaction matrices
    t0 = time.time()
    G_b2w = _assemble_free_space(wall_colloc, body_source, MU)
    G_w2b = _assemble_free_space(body_colloc, wall_source, MU)
    print(f"    Cross matrices: {time.time()-t0:.1f}s")

    # Run for F_x, F_z, T_z
    results = {}
    e = np.eye(3)
    center = np.zeros(3)

    for label, U, omega in [
        ("F_x", e[0], np.zeros(3)),
        ("F_z", e[2], np.zeros(3)),
        ("T_z", np.zeros(3), e[2]),
    ]:
        r = body_colloc - center
        u_body_bc = U + np.cross(omega, r)

        # Initial body solve
        lam_body = lu_solve(
            (body_lu, body_piv), u_body_bc.ravel()).reshape(-1, 3)

        # Gauss-Seidel
        for gs_it in range(max_gs_iter):
            # Wall solve
            u_at_wall = (G_b2w @ lam_body.ravel()).reshape(-1, 3)
            lam_wall = lu_solve(
                (wall_lu, wall_piv), (-u_at_wall).ravel()).reshape(-1, 3)

            # Body re-solve
            u_at_body = (G_w2b @ lam_wall.ravel()).reshape(-1, 3)
            lam_body_new = lu_solve(
                (body_lu, body_piv),
                (u_body_bc - u_at_body).ravel()).reshape(-1, 3)

            dF = np.linalg.norm(np.sum(lam_body_new - lam_body, axis=0))
            lam_body = lam_body_new
            if dF < 1e-8:
                break

        F = -np.sum(lam_body, axis=0)
        T = -np.sum(np.cross(body_source - center, lam_body), axis=0)

        if label == "F_x":
            val = abs(F[0])
        elif label == "F_z":
            val = abs(F[2])
        else:
            val = abs(T[2])

        ref_val = REF[label]
        err = abs(val - ref_val) / ref_val * 100
        results[label] = dict(value=val, err=err, gs_iters=gs_it + 1)
        print(f"    {label}: {val:.4f} (ref {ref_val}, err {err:.1f}%, "
              f"GS={gs_it+1} iters)")

    return results


if __name__ == "__main__":
    print("Phase 2: Discrete wall MFS convergence study")
    print(f"Sphere κ=0.3 (a={A}, R_cyl={R_CYL:.3f})")
    print(f"Reference: Liron-Shahar table (<4% on all entries)")
    print()

    configs = [
        # (n_circ, n_axial, sr)
        (32, 40, 1.3),    # 1,280 pts — matches Phase 0
        (50, 100, 1.3),   # 5,000 pts
        (71, 141, 1.3),   # ~10,000 pts
    ]

    all_results = {}
    for n_circ, n_axial, sr in configs:
        N_wall = n_circ * n_axial
        print(f"\n{'='*60}")
        print(f"  N_wall = {N_wall} ({n_circ}×{n_axial}), sr={sr}")
        print(f"{'='*60}")

        t0 = time.time()
        results = run_wall_mfs_direct(n_circ, n_axial, sr)
        dt = time.time() - t0
        print(f"  Total: {dt:.0f}s")
        all_results[N_wall] = results

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'N_wall':>8} | {'F_x':>8} {'err':>6} | {'F_z':>8} {'err':>6} | "
          f"{'T_z':>8} {'err':>6} | {'Fx-Fz':>6}")
    print("-" * 68)

    for N_wall, res in all_results.items():
        fx = res["F_x"]
        fz = res["F_z"]
        tz = res["T_z"]
        spread = abs(fx["err"] - fz["err"])
        print(f"{N_wall:8d} | {fx['value']:8.2f} {fx['err']:5.1f}% | "
              f"{fz['value']:8.2f} {fz['err']:5.1f}% | "
              f"{tz['value']:8.2f} {tz['err']:5.1f}% | {spread:5.1f}pp")
