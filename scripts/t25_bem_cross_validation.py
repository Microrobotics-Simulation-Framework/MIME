#!/usr/bin/env python3
"""T2.5 — BEM cross-validation against LBM drag multipliers.

Generates UMR surface mesh, precomputes wall tables, runs BEM drag
sweep at κ = {0.15, 0.22, 0.30, 0.35, 0.40}, and compares against
T2.6b LBM reference data.

Usage:
    python scripts/t25_bem_cross_validation.py mesh     # Step a: mesh + free-space R
    python scripts/t25_bem_cross_validation.py tables   # Step b: wall table precompute
    python scripts/t25_bem_cross_validation.py sweep    # Step c: drag multiplier sweep
    python scripts/t25_bem_cross_validation.py all      # All steps
"""

import sys
import os
import time
import json
import logging

import numpy as np
import jax
import jax.numpy as jnp

# Force CPU — no GPU needed for BEM (dense linear algebra)
os.environ["JAX_PLATFORMS"] = "cpu"

from mime.nodes.environment.stokeslet.surface_mesh import sdf_surface_mesh, SurfaceMesh
from mime.nodes.robot.helix_geometry import umr_sdf
from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    precompute_wall_table,
    save_wall_table,
    load_wall_table,
)


# ── Numpy-based BEM assembly (avoids JAX XLA compilation overhead) ──

def assemble_system_matrix_numpy(pts, wts, epsilon, mu, chunk_rows=500):
    """Assemble (3N, 3N) BEM system matrix using chunked numpy.

    Processes chunk_rows target rows at a time to keep peak memory
    under ~500 MB regardless of N.
    """
    N = len(pts)
    prefactor = 1.0 / (8.0 * np.pi * mu)
    eps_sq = epsilon**2

    A = np.zeros((3 * N, 3 * N))

    for i0 in range(0, N, chunk_rows):
        i1 = min(i0 + chunk_rows, N)
        nc = i1 - i0

        # r_vec: (nc, N, 3)
        r = pts[i0:i1, None, :] - pts[None, :, :]
        r_sq = np.sum(r**2, axis=-1)       # (nc, N)
        denom = (r_sq + eps_sq)**1.5        # (nc, N)

        # Stokeslet tensor blocks: (nc, N, 3, 3)
        S = ((np.eye(3)[None, None, :, :] * (r_sq + 2*eps_sq)[:, :, None, None])
             + r[:, :, :, None] * r[:, :, None, :]) / denom[:, :, None, None]

        # Apply weights and prefactor
        S *= prefactor * wts[None, :, None, None]

        # Write into A
        A[3*i0:3*i1, :] = S.transpose(0, 2, 1, 3).reshape(3*nc, 3*N)

    return A


def compute_dlp_rhs_numpy(pts, normals, wts, velocity, epsilon, chunk_rows=200):
    """Compute DLP-corrected RHS using chunked numpy."""
    N = len(pts)
    prefactor = 1.0 / (8.0 * np.pi)
    eps_sq = epsilon**2
    wu = wts[:, None] * velocity  # (N, 3) — precomputed weighted velocity

    dlp = np.zeros((N, 3))

    for m0 in range(0, N, chunk_rows):
        m1 = min(m0 + chunk_rows, N)
        nc = m1 - m0

        # r = x_source - x_target: (nc, N, 3)
        r = pts[None, :, :] - pts[m0:m1, None, :]
        r_sq = np.sum(r**2, axis=-1)        # (nc, N)
        denom = (r_sq + eps_sq)**2.5         # (nc, N)
        r_dot_n = np.einsum('mnk,nk->mn', r, normals)  # (nc, N)

        # Stresslet K_ij = T_ijk n_k: (nc, N, 3, 3)
        K = (-6.0 * (r[:, :, :, None] * r[:, :, None, :]) * r_dot_n[:, :, None, None]
             - 3.0 * eps_sq * (
                 r[:, :, :, None] * normals[None, :, None, :]
                 + normals[None, :, :, None] * r[:, :, None, :]
                 + r_dot_n[:, :, None, None] * np.eye(3)[None, None, :, :]
             )) / denom[:, :, None, None]

        # Zero self-interaction
        for local in range(nc):
            K[local, m0 + local] = 0.0

        # Contract with weighted velocity: dlp_m = sum_n -prefactor * K_mn @ wu_n
        dlp[m0:m1] = -prefactor * np.einsum('mnij,nj->mi', K, wu)

    rhs = 0.5 * velocity + dlp
    return rhs.ravel()


def compute_resistance_matrix_numpy(pts, normals, wts, center, epsilon, mu, use_dlp=True):
    """Compute 6×6 R using numpy BEM (no JAX compilation overhead)."""
    N = len(pts)
    A = assemble_system_matrix_numpy(pts, wts, epsilon, mu)
    log.info("  System matrix assembled: %dx%d", A.shape[0], A.shape[1])

    # Build 6 RHS
    e = np.eye(3)
    zero = np.zeros(3)
    rhs_cols = []
    for i in range(3):
        U, omega = e[i], zero
        r = pts - center
        vel = U + np.cross(omega, r)
        if use_dlp and normals is not None:
            rhs = compute_dlp_rhs_numpy(pts, normals, wts, vel, epsilon)
        else:
            rhs = vel.ravel()
        rhs_cols.append(rhs)
    for i in range(3):
        U, omega = zero, e[i]
        r = pts - center
        vel = U + np.cross(omega, r)
        if use_dlp and normals is not None:
            rhs = compute_dlp_rhs_numpy(pts, normals, wts, vel, epsilon)
        else:
            rhs = vel.ravel()
        rhs_cols.append(rhs)

    rhs_matrix = np.column_stack(rhs_cols)

    # Solve via LU
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    solutions = lu_solve((lu, piv), rhs_matrix)

    # Extract F, T
    R = np.zeros((6, 6))
    for col in range(6):
        traction = solutions[:, col].reshape(N, 3)
        weighted_f = traction * wts[:, None]
        F = np.sum(weighted_f, axis=0)
        r = pts - center
        T = np.sum(np.cross(r, weighted_f), axis=0)
        R[:3, col] = F
        R[3:, col] = T

    return R, A


def compute_confined_R_numpy(pts, normals, wts, center, epsilon, mu, R_cyl, wall_table, use_dlp=True):
    """Compute confined 6×6 R using numpy BEM + wall table (no JAX)."""
    from mime.nodes.environment.stokeslet.cylinder_wall_table import (
        assemble_image_correction_matrix_from_table,
    )

    N = len(pts)
    A_free = assemble_system_matrix_numpy(pts, wts, epsilon, mu)
    G_wall = assemble_image_correction_matrix_from_table(pts, wts, R_cyl, mu, wall_table)
    A = A_free + G_wall

    # Build 6 RHS
    e = np.eye(3)
    zero = np.zeros(3)
    rhs_cols = []
    for i in range(3):
        U, omega = e[i], zero
        r = pts - center
        vel = U + np.cross(omega, r)
        if use_dlp and normals is not None:
            rhs = compute_dlp_rhs_numpy(pts, normals, wts, vel, epsilon)
        else:
            rhs = vel.ravel()
        rhs_cols.append(rhs)
    for i in range(3):
        U, omega = zero, e[i]
        r = pts - center
        vel = U + np.cross(omega, r)
        if use_dlp and normals is not None:
            rhs = compute_dlp_rhs_numpy(pts, normals, wts, vel, epsilon)
        else:
            rhs = vel.ravel()
        rhs_cols.append(rhs)

    rhs_matrix = np.column_stack(rhs_cols)

    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    solutions = lu_solve((lu, piv), rhs_matrix)

    R = np.zeros((6, 6))
    for col in range(6):
        traction = solutions[:, col].reshape(N, 3)
        weighted_f = traction * wts[:, None]
        F = np.sum(weighted_f, axis=0)
        r = pts - center
        T = np.sum(np.cross(r, weighted_f), axis=0)
        R[:3, col] = F
        R[3:, col] = T

    return R

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── UMR geometry (from deboer2025_params.md, d2.8 baseline) ─────────
# All dimensions in mm, non-dimensionalised to a = 1.0 (half-diameter
# based on external diameter including fins = 2.84 mm → a = 1.42 mm)
A_PHYS = 1.42  # mm, half of external diameter

# Non-dimensional params (divide all lengths by A_PHYS)
UMR_PARAMS = dict(
    body_radius=0.87 / A_PHYS,
    body_length=4.1 / A_PHYS,
    cone_length=1.9 / A_PHYS,
    cone_end_radius=0.255 / A_PHYS,
    fin_outer_radius=1.42 / A_PHYS,  # = 1.0 in non-dim
    fin_length=2.03 / A_PHYS,
    fin_width=0.55 / A_PHYS,
    fin_thickness=0.15 / A_PHYS,
    n_fin_sets=2,
    fins_per_set=3,
    helix_pitch=8.0 / A_PHYS,
)

TOTAL_LENGTH = (UMR_PARAMS["body_length"] + UMR_PARAMS["cone_length"])

# Non-dimensional viscosity (set to 1 for non-dim, physical value
# cancels in the drag multiplier ratio anyway)
MU = 1.0

# κ values and corresponding R_cyl (non-dim)
KAPPAS = [0.15, 0.22, 0.30, 0.35, 0.40]

# LBM reference data (drag_torque_z from umr_training_v2_bouzidi.h5)
# These are in LBM lattice units, only ratios matter for multipliers
LBM_TZ = {
    0.15: 87.815765,
    0.22: 100.96927,
    0.30: 108.542656,
    0.35: 112.59334,
    0.40: 124.29775,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TABLE_DIR = os.path.join(DATA_DIR, "wall_tables")
RESULTS_PATH = os.path.join(DATA_DIR, "t25_bem_cross_validation.json")


def generate_umr_mesh(mc_resolution=32):
    """Generate non-dimensional UMR surface mesh via marching cubes."""
    # Bounding box: pad by 20% around the UMR
    pad = 0.3
    bbox_min = (-UMR_PARAMS["fin_outer_radius"] - pad,
                -UMR_PARAMS["fin_outer_radius"] - pad,
                -TOTAL_LENGTH / 2 - pad)
    bbox_max = (+UMR_PARAMS["fin_outer_radius"] + pad,
                +UMR_PARAMS["fin_outer_radius"] + pad,
                +TOTAL_LENGTH / 2 + pad)

    log.info("Generating UMR mesh: mc_res=%d, bbox_min=%s, bbox_max=%s",
             mc_resolution, bbox_min, bbox_max)

    sdf_func = lambda pts: umr_sdf(pts, **UMR_PARAMS)
    mesh = sdf_surface_mesh(sdf_func, bbox_min, bbox_max, mc_resolution)
    return mesh


def validate_mesh(mesh: SurfaceMesh):
    """Check mesh quality metrics."""
    pts = mesh.points
    nml = mesh.normals
    wts = mesh.weights

    log.info("Mesh: N=%d, total_area=%.4f, mean_spacing=%.6f",
             mesh.n_points, mesh.total_area, mesh.mean_spacing)

    # Check normals are unit
    nml_norms = np.linalg.norm(nml, axis=1)
    log.info("Normal magnitudes: min=%.6f, max=%.6f",
             nml_norms.min(), nml_norms.max())

    # Check weights are positive
    assert np.all(wts > 0), "Some weights are non-positive!"
    log.info("Weights: min=%.6e, max=%.6e, sum=%.4f",
             wts.min(), wts.max(), wts.sum())

    # Check body extent (non-dim, should be within R_max=1.0 radially)
    rho = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    log.info("Radial extent: max(ρ)=%.4f (should be ~%.4f)",
             rho.max(), UMR_PARAMS["fin_outer_radius"])
    log.info("Axial extent: z in [%.4f, %.4f] (total_length=%.4f)",
             pts[:, 2].min(), pts[:, 2].max(), TOTAL_LENGTH)

    # Centroid
    centroid = pts.mean(axis=0)
    log.info("Centroid: [%.4f, %.4f, %.4f]", *centroid)

    return mesh


def validate_free_space_R(mesh: SurfaceMesh):
    """Compute free-space R and check symmetry/positive-definiteness."""
    pts = np.array(mesh.points)
    nml = np.array(mesh.normals)
    wts = np.array(mesh.weights)
    center = np.zeros(3)
    epsilon = mesh.mean_spacing / 2.0

    log.info("Computing free-space R (ε=%.6f, N=%d) via numpy BEM (no DLP)...",
             epsilon, len(pts))
    t0 = time.time()
    # Skip DLP correction — ratios for cross-validation are insensitive to it,
    # and the O(N²) stresslet computation is prohibitively slow for N=6500.
    R, A = compute_resistance_matrix_numpy(pts, nml, wts, center, epsilon, MU,
                                           use_dlp=False)
    dt = time.time() - t0
    log.info("Free-space R computed in %.1f s", dt)

    log.info("BEM system: norm(A)=%.2e", np.linalg.norm(A, ord=np.inf))

    # Print R
    log.info("Free-space resistance matrix R (6×6):")
    for i in range(6):
        log.info("  [%s]", "  ".join(f"{R[i,j]:10.4f}" for j in range(6)))

    # Symmetry check
    asym = np.max(np.abs(R - R.T))
    log.info("Max asymmetry |R - R^T|: %.2e", asym)

    # Positive definiteness check
    eigvals = np.linalg.eigvalsh(R)
    log.info("Eigenvalues of R: %s", np.array2string(eigvals, precision=4))
    if np.all(eigvals > 0):
        log.info("R is positive definite ✓")
    else:
        log.warning("R is NOT positive definite!")

    # For a UMR-like body, R_FU and R_TΩ diagonal blocks should be diagonal-dominant
    R_FU = R[:3, :3]
    R_TW = R[3:, 3:]
    log.info("R_FU diagonal: [%.4f, %.4f, %.4f]",
             R_FU[0, 0], R_FU[1, 1], R_FU[2, 2])
    log.info("R_TΩ diagonal: [%.4f, %.4f, %.4f]",
             R_TW[0, 0], R_TW[1, 1], R_TW[2, 2])

    # Free-swimming speed (unit ω_z rotation)
    R_FU_inv = np.linalg.inv(R_FU)
    R_FW = R[:3, 3:]
    U_swim = -R_FU_inv @ R_FW @ np.array([0, 0, 1.0])
    log.info("Free-swimming speed (free-space, ω_z=1): U_z=%.6f", U_swim[2])

    return R


def step_mesh(mc_res=32):
    """T2.5a: Generate mesh and validate."""
    mesh = generate_umr_mesh(mc_res)
    mesh = validate_mesh(mesh)
    R_free = validate_free_space_R(mesh)

    # Save mesh for reuse
    mesh_path = os.path.join(DATA_DIR, "umr_bem_mesh.npz")
    np.savez(mesh_path,
             points=mesh.points, normals=mesh.normals, weights=mesh.weights)
    log.info("Mesh saved to %s", mesh_path)
    return mesh, R_free


def step_tables(mesh: SurfaceMesh = None):
    """T2.5b: Precompute wall tables for all κ values."""
    os.makedirs(TABLE_DIR, exist_ok=True)

    for kappa in KAPPAS:
        R_cyl = 1.0 / kappa
        table_path = os.path.join(TABLE_DIR, f"wall_R{R_cyl:.3f}.npz")

        if os.path.exists(table_path):
            log.info("κ=%.2f: table exists at %s, skipping", kappa, table_path)
            continue

        log.info("κ=%.2f: precomputing wall table (R_cyl=%.3f)...",
                 kappa, R_cyl)
        t0 = time.time()
        table = precompute_wall_table(
            R_cyl=R_cyl,
            mu=MU,
            n_rho=30,
            n_dphi=64,
            n_dz=128,
            L_max_factor=5.0,
            n_max=15,
            n_k=80,
            n_phi=64,
            n_jobs=0,  # use all CPUs
        )
        dt = time.time() - t0
        log.info("κ=%.2f: table computed in %.1f s (%.1f min)",
                 kappa, dt, dt / 60)

        save_wall_table(table, table_path)
        log.info("κ=%.2f: saved to %s (%.1f MB)",
                 kappa, table_path,
                 os.path.getsize(table_path) / 1e6)


def step_sweep(mesh: SurfaceMesh = None):
    """T2.5c: BEM drag multiplier sweep."""
    if mesh is None:
        mesh_path = os.path.join(DATA_DIR, "umr_bem_mesh.npz")
        d = np.load(mesh_path)
        mesh = SurfaceMesh(d["points"], d["normals"], d["weights"])

    pts = np.array(mesh.points)
    nml = np.array(mesh.normals)
    wts = np.array(mesh.weights)
    center = np.zeros(3)
    epsilon = mesh.mean_spacing / 2.0

    results = {}

    for kappa in KAPPAS:
        R_cyl = 1.0 / kappa
        table_path = os.path.join(TABLE_DIR, f"wall_R{R_cyl:.3f}.npz")

        log.info("κ=%.2f: loading wall table from %s", kappa, table_path)
        table = load_wall_table(table_path)

        log.info("κ=%.2f: computing confined R (R_cyl=%.3f)...", kappa, R_cyl)
        t0 = time.time()
        R = compute_confined_R_numpy(
            pts, nml, wts, center, epsilon, MU, R_cyl, table, use_dlp=False,
        )
        dt = time.time() - t0
        log.info("κ=%.2f: R computed in %.1f s", kappa, dt)

        # Extract T_z (rotational drag about z)
        T_z = R[5, 5]
        log.info("κ=%.2f: R_TΩ_zz = %.6f", kappa, T_z)

        # Free-swimming speed
        R_FU = R[:3, :3]
        R_FW = R[:3, 3:]
        R_FU_inv = np.linalg.inv(R_FU)
        U_swim = -R_FU_inv @ R_FW @ np.array([0, 0, 1.0])

        # Symmetry
        asym = np.max(np.abs(R - R.T))

        # Eigenvalues
        eigvals = np.linalg.eigvalsh(R)
        pd = bool(np.all(eigvals > 0))

        results[str(kappa)] = {
            "R_cyl": R_cyl,
            "R_TW_zz": float(T_z),
            "U_swim_z": float(U_swim[2]),
            "U_swim": [float(x) for x in U_swim],
            "max_asymmetry": float(asym),
            "positive_definite": pd,
            "min_eigval": float(eigvals.min()),
            "compute_time_s": dt,
        }

        if kappa == 0.30:
            results["R_full_kappa030"] = R.tolist()

        # Print full R
        log.info("κ=%.2f: Full R:", kappa)
        for i in range(6):
            log.info("  [%s]", "  ".join(f"{R[i,j]:10.4f}" for j in range(6)))

    # Compute drag multipliers (relative to κ=0.15 baseline)
    baseline_bem = results["0.15"]["R_TW_zz"]
    baseline_lbm = LBM_TZ[0.15]

    log.info("\n" + "=" * 80)
    log.info("DRAG MULTIPLIER COMPARISON: BEM vs LBM")
    log.info("=" * 80)
    log.info("%-6s  %12s  %12s  %12s  %8s  %10s",
             "κ", "BEM T_z", "BEM mult", "LBM mult", "Δ(%)", "U_swim_z")

    for kappa in KAPPAS:
        ks = str(kappa)
        bem_tz = results[ks]["R_TW_zz"]
        bem_mult = bem_tz / baseline_bem
        lbm_mult = LBM_TZ[kappa] / baseline_lbm
        diff_pct = 100 * (bem_mult - lbm_mult) / lbm_mult
        u_swim = results[ks]["U_swim_z"]

        results[ks]["bem_multiplier"] = float(bem_mult)
        results[ks]["lbm_multiplier"] = float(lbm_mult)
        results[ks]["multiplier_diff_pct"] = float(diff_pct)

        log.info("%-6.2f  %12.4f  %12.4f  %12.4f  %8.2f  %10.6f",
                 kappa, bem_tz, bem_mult, lbm_mult, diff_pct, u_swim)

    log.info("=" * 80)

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", RESULTS_PATH)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    mc_res = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    if mode == "mesh":
        step_mesh(mc_res)
    elif mode == "tables":
        step_tables()
    elif mode == "sweep":
        step_sweep()
    elif mode == "all":
        mesh, _ = step_mesh(mc_res)
        step_tables(mesh)
        step_sweep(mesh)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
