#!/usr/bin/env python3
"""Quick off-center BEM validation using existing wall table."""

import os, sys, time
import numpy as np
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    load_wall_table, assemble_image_correction_matrix_from_table,
)
from t25_bem_cross_validation import assemble_system_matrix_numpy
from scipy.linalg import lu_factor, lu_solve

# Use existing T2.5 table (R_cyl=3.333)
table = load_wall_table('data/wall_tables/wall_R3.333.npz')
R_cyl = table.R_cyl
MU = 1.0
print(f"Table: R_cyl={R_cyl:.3f}, rho_grid=[{table.rho_grid[0]:.4f}..{table.rho_grid[-1]:.4f}]")

# Sphere of radius 0.5
mesh = sphere_surface_mesh(radius=0.5, n_refine=2)  # 320 pts (fast)
N = mesh.n_points
eps = mesh.mean_spacing / 2.0
print(f"Sphere: N={N}, ε={eps:.4f}")

A_body = assemble_system_matrix_numpy(mesh.points, mesh.weights, eps, MU)
print(f"A_body: {A_body.shape}")

for offset_x in [0.0, 0.3, 0.7, 1.2, 2.0]:
    pts = mesh.points + np.array([offset_x, 0.0, 0.0])
    rho_max = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2).max()
    rho_min = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2).min()

    if rho_max >= R_cyl:
        print(f"\nOffset={offset_x}: SKIP (body outside cylinder)")
        continue

    t0 = time.time()
    G_wall = assemble_image_correction_matrix_from_table(
        pts, mesh.weights, R_cyl, MU, table)
    A = A_body + G_wall

    # Solve 6 RHS (body-frame velocities)
    e = np.eye(3)
    rhs_cols = []
    for i in range(3):
        vel = np.tile(e[i], N)
        rhs_cols.append(vel)
    for i in range(3):
        r = mesh.points  # body frame
        vel = np.cross(e[i], r).ravel()
        rhs_cols.append(vel)

    lu, piv = lu_factor(A)
    solutions = lu_solve((lu, piv), np.column_stack(rhs_cols))

    R = np.zeros((6, 6))
    for col in range(6):
        trac = solutions[:, col].reshape(N, 3)
        wf = trac * mesh.weights[:, None]
        R[:3, col] = np.sum(wf, axis=0)
        r = mesh.points
        R[3:, col] = np.sum(np.cross(r, wf), axis=0)

    recip = np.max(np.abs(R - R.T))
    eigvals = np.linalg.eigvalsh(R)
    pd = np.all(eigvals > 0)

    # Full 3D swimming velocity
    R_FU = R[:3, :3]
    R_FW = R[:3, 3:]
    omega = np.array([0.0, 0.0, 1.0])
    U = -np.linalg.inv(R_FU) @ R_FW @ omega
    U_lat = np.sqrt(U[0]**2 + U[1]**2)

    dt = time.time() - t0
    print(f"\nOffset={offset_x:.1f}: ρ∈[{rho_min:.3f},{rho_max:.3f}], {dt:.1f}s")
    print(f"  |R-R^T|={recip:.2e}, PD={pd}")
    print(f"  R_FU diag: [{R[0,0]:.3f}, {R[1,1]:.3f}, {R[2,2]:.3f}]")
    print(f"  R_TW diag: [{R[3,3]:.3f}, {R[4,4]:.3f}, {R[5,5]:.3f}]")
    print(f"  New couplings R_FW[:2,2] (Fx,Fy from ωz): [{R[0,5]:.4f}, {R[1,5]:.4f}]")
    print(f"  U_swim = [{U[0]:.6f}, {U[1]:.6f}, {U[2]:.6f}]")
    print(f"  U_lateral = {U_lat:.6f}, drift_angle = {np.degrees(np.arctan2(U[1], U[0])):.1f}°")
