#!/usr/bin/env python3
"""Phase 1: Helix in cylinder — confined drag validation.

Tests:
  1. Free-space helix R: symmetry, positive-definite, coupling
  2. Confined helix R: direction-independence, confinement enhancement
  3. Convergence study across mesh resolutions
  4. Free-swimming speed: does confinement help or hurt?
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import time
from functools import partial

from mime.nodes.robot.helix_geometry import helix_sdf
from mime.nodes.environment.stokeslet.surface_mesh import sdf_surface_mesh
from mime.nodes.environment.stokeslet.resistance import compute_resistance_matrix
from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    load_wall_table,
    assemble_image_correction_matrix_from_table,
)
from mime.nodes.environment.stokeslet.bem import (
    assemble_system_matrix,
    compute_force_torque,
    compute_dlp_rhs_correction,
    solve_bem_multi_rhs,
)

# ── Helix parameters ──────────────────────────────────────────────────
HELIX_RADIUS = 0.5
HELIX_PITCH = 1.0
WIRE_RADIUS = 0.1
N_TURNS = 3.0
CENTER = (0.0, 0.0, -1.5)
MU = 1.0
R_CYL = 2.0


def make_helix_mesh(mc_res=32):
    sdf_fn = partial(
        helix_sdf,
        helix_radius=HELIX_RADIUS, helix_pitch=HELIX_PITCH,
        wire_radius=WIRE_RADIUS, n_turns=N_TURNS, center=CENTER,
    )
    R_env = HELIX_RADIUS + WIRE_RADIUS + 0.15
    z_lo = CENTER[2] - 0.15
    z_hi = CENTER[2] + N_TURNS * HELIX_PITCH + 0.15
    return sdf_surface_mesh(sdf_fn, (-R_env, -R_env, z_lo),
                            (R_env, R_env, z_hi), mc_resolution=mc_res)


def compute_R_free(mesh):
    pts = jnp.array(mesh.points)
    wts = jnp.array(mesh.weights)
    nml = jnp.array(mesh.normals)
    eps = mesh.mean_spacing / 2.0
    center = jnp.zeros(3)
    return np.array(compute_resistance_matrix(
        pts, wts, center, eps, MU, nml, use_dlp=True))


def compute_R_confined(mesh, table):
    pts_np = np.array(mesh.points)
    wts_np = np.array(mesh.weights)
    nml_np = np.array(mesh.normals)
    pts = jnp.array(pts_np)
    wts = jnp.array(wts_np)
    nml = jnp.array(nml_np)
    N = mesh.n_points
    eps = mesh.mean_spacing / 2.0
    center = jnp.zeros(3)

    t0 = time.time()
    A_bem = np.array(assemble_system_matrix(pts, wts, eps, MU))
    print(f"    BEM assembly: {time.time()-t0:.1f}s")

    t0 = time.time()
    G_wall = assemble_image_correction_matrix_from_table(
        pts_np, wts_np, R_CYL, MU, table)
    print(f"    Wall table interpolation: {time.time()-t0:.1f}s")

    A_conf = jnp.array(A_bem + G_wall)

    e = jnp.eye(3)
    zero = jnp.zeros(3)
    rhs_cols = []
    for i in range(3):
        r = pts - center
        vel = e[i] + jnp.cross(zero, r)
        rhs_cols.append(compute_dlp_rhs_correction(pts, nml, wts, vel, eps))
    for i in range(3):
        r = pts - center
        vel = zero + jnp.cross(e[i], r)
        rhs_cols.append(compute_dlp_rhs_correction(pts, nml, wts, vel, eps))

    rhs_matrix = jnp.stack(rhs_cols, axis=1)

    t0 = time.time()
    solutions = solve_bem_multi_rhs(A_conf, rhs_matrix)
    print(f"    LU solve: {time.time()-t0:.1f}s")

    R = jnp.zeros((6, 6))
    for col in range(6):
        trac = solutions[:, col].reshape(N, 3)
        F, T = compute_force_torque(pts, wts, trac, center)
        R = R.at[:3, col].set(F)
        R = R.at[3:, col].set(T)

    return np.array(R)


def compute_swimming_speed(R, omega=None):
    """Free-swimming speed for a force-free helix rotating at omega.

    For F=0: R_FU @ U + R_FΩ @ Ω = 0
    => U = -inv(R_FU) @ R_FΩ @ Ω
    """
    if omega is None:
        omega = np.array([0.0, 0.0, 1.0])
    R_FU = R[0:3, 0:3]
    R_FOmega = R[0:3, 3:6]
    return -np.linalg.inv(R_FU) @ R_FOmega @ omega


def analyze_R(R, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"\nR matrix (6×6):")
    for i in range(6):
        row = " ".join(f"{R[i,j]:9.4f}" for j in range(6))
        print(f"  {row}")

    sym_err = np.max(np.abs(R - R.T)) / np.max(np.abs(R))
    print(f"\nSymmetry: max|R-R^T|/max|R| = {sym_err:.4f}")

    eigvals = np.linalg.eigvalsh(R)
    print(f"Eigenvalues: {eigvals.round(4)}")
    print(f"Positive-definite: {np.all(eigvals > 0)}")

    print(f"\nDirection checks:")
    print(f"  F_x/F_y = {R[0,0]:.4f}/{R[1,1]:.4f} "
          f"(ratio: {R[0,0]/R[1,1]:.4f})")
    print(f"  T_x/T_y = {R[3,3]:.4f}/{R[4,4]:.4f} "
          f"(ratio: {R[3,3]/R[4,4]:.4f})")

    B_t = R[:3, 3:]
    max_coupling = np.max(np.abs(B_t))
    print(f"\nMax F-Ω coupling: {max_coupling:.4f}")

    U_swim = compute_swimming_speed(R)
    print(f"Swimming speed (Ω_z=1): U = [{U_swim[0]:.4f}, "
          f"{U_swim[1]:.4f}, {U_swim[2]:.4f}]")
    print(f"  |U_z| = {abs(U_swim[2]):.4f}")

    return dict(eigvals=eigvals, sym_err=sym_err, U_swim=U_swim)


if __name__ == "__main__":
    print("Phase 1: Helix in cylinder — convergence + swimming speed")
    print(f"Helix: R={HELIX_RADIUS}, pitch={HELIX_PITCH}, "
          f"wire_r={WIRE_RADIUS}, turns={N_TURNS}")
    print(f"Cylinder: R={R_CYL}")

    table = load_wall_table("/tmp/wall_table_helix_R2.npz")
    print(f"Wall table loaded: ρ grid [{table.rho_grid[0]:.3f}, "
          f"{table.rho_grid[-1]:.3f}]")

    # ── Convergence sweep ─────────────────────────────────────────
    MC_RESOLUTIONS = [32, 48]
    results = {}

    for mc_res in MC_RESOLUTIONS:
        print(f"\n{'#'*60}")
        print(f"  mc_res = {mc_res}")
        print(f"{'#'*60}")

        t_total = time.time()
        mesh = make_helix_mesh(mc_res=mc_res)
        N = mesh.n_points
        mem_gb = (3 * N)**2 * 8 / 1e9
        print(f"\nMesh: {N} pts, h={mesh.mean_spacing:.4f}, "
              f"area={mesh.total_area:.3f}")
        print(f"System matrix: {3*N}×{3*N}, ~{mem_gb:.1f} GB")

        rho = np.sqrt(mesh.points[:, 0]**2 + mesh.points[:, 1]**2)
        centroid = np.mean(mesh.points, axis=0)
        print(f"ρ range: [{rho.min():.3f}, {rho.max():.3f}], "
              f"centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, "
              f"{centroid[2]:.3f}]")

        # Free-space
        print("\nComputing R_free...")
        t0 = time.time()
        R_free = compute_R_free(mesh)
        dt_free = time.time() - t0
        print(f"  {dt_free:.0f}s")
        info_f = analyze_R(R_free, f"FREE-SPACE (mc={mc_res}, N={N})")

        # Confined
        print(f"\nComputing R_confined...")
        t0 = time.time()
        R_conf = compute_R_confined(mesh, table)
        dt_conf = time.time() - t0
        print(f"  Total: {dt_conf:.0f}s")
        info_c = analyze_R(R_conf, f"CONFINED (mc={mc_res}, N={N})")

        results[mc_res] = dict(
            N=N, R_free=R_free, R_conf=R_conf,
            info_free=info_f, info_conf=info_c,
            dt_free=dt_free, dt_conf=dt_conf,
        )

        print(f"\nTotal time for mc_res={mc_res}: "
              f"{time.time()-t_total:.0f}s")

    # ── Convergence summary ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CONVERGENCE SUMMARY")
    print(f"{'='*70}")

    labels = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]

    # Diagonal entries
    header = f"{'mc':>4} {'N':>6} |"
    for l in labels:
        header += f" {l:>7}"
    header += " | sym%   Fx/Fy"

    for tag in ["FREE", "CONFINED"]:
        print(f"\n  {tag} diagonal R entries:")
        print(f"  {header}")
        print(f"  {'-'*len(header)}")
        for mc_res in MC_RESOLUTIONS:
            r = results[mc_res]
            R = r['R_free'] if tag == "FREE" else r['R_conf']
            info = r['info_free'] if tag == "FREE" else r['info_conf']
            diag = np.diag(R)
            row = f"  {mc_res:4d} {r['N']:6d} |"
            for d in diag:
                row += f" {d:7.4f}"
            row += f" | {info['sym_err']*100:5.2f}% {R[0,0]/R[1,1]:6.4f}"
            print(row)

    # Swimming speed
    print(f"\n  SWIMMING SPEED (Ω_z = 1):")
    print(f"  {'mc':>4} {'N':>6} | {'U_z free':>10} {'U_z conf':>10} "
          f"{'ratio':>7} {'slower':>6}")
    print(f"  {'-'*55}")
    for mc_res in MC_RESOLUTIONS:
        r = results[mc_res]
        Uf = r['info_free']['U_swim'][2]
        Uc = r['info_conf']['U_swim'][2]
        ratio = Uc / Uf if abs(Uf) > 1e-10 else 0
        slower = "yes" if abs(Uc) < abs(Uf) else "no"
        print(f"  {mc_res:4d} {r['N']:6d} | {Uf:10.4f} {Uc:10.4f} "
              f"{ratio:7.3f} {slower:>6}")

    # Confinement enhancement
    print(f"\n  CONFINEMENT ENHANCEMENT (confined/free):")
    header2 = f"  {'mc':>4} |"
    for l in labels:
        header2 += f" {l:>6}"
    print(header2)
    print(f"  {'-'*len(header2)}")
    for mc_res in MC_RESOLUTIONS:
        r = results[mc_res]
        df = np.diag(r['R_free'])
        dc = np.diag(r['R_conf'])
        row = f"  {mc_res:4d} |"
        for i in range(6):
            ratio = dc[i] / df[i] if abs(df[i]) > 1e-10 else 0
            row += f" {ratio:6.3f}"
        print(row)

    print(f"\n{'='*70}")
    print("  Phase 1 complete.")
    print(f"{'='*70}")
