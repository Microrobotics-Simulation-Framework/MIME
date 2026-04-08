#!/usr/bin/env python3
"""Quick validation of the cylindrical Green's function.

Tests:
1. Single Stokeslet no-slip residual on the cylinder wall
2. Convergence vs n_max and n_k
"""

import numpy as np
import time
from mime.nodes.environment.stokeslet.cylinder_greens_function import (
    assemble_cylinder_stokeslet_matrix,
    _wall_bc_k_transform,
    _image_system_matrix,
    _image_basis_at_rho,
    _azimuthal_mode_single,
    _assemble_free_space,
)

R_CYL = 1.0 / 0.3  # ≈ 3.333
MU = 1.0

def test_single_stokeslet_wall_residual():
    """Place a Stokeslet inside the cylinder and check u=0 at wall."""
    print("=" * 60)
    print("Test: Single Stokeslet wall no-slip residual")
    print("=" * 60)

    # Source point inside cylinder
    rho0 = 0.7  # off-axis
    source = np.array([[rho0, 0.0, 0.0]])

    # Wall test points (many points on the cylinder)
    N_wall = 200
    phi_wall = np.linspace(0, 2*np.pi, N_wall, endpoint=False)
    z_wall = np.linspace(-5.0, 5.0, N_wall)
    PHI, Z = np.meshgrid(phi_wall, z_wall[:20])  # 20 z-points for speed
    wall_pts = np.column_stack([
        R_CYL * np.cos(PHI.ravel()),
        R_CYL * np.sin(PHI.ravel()),
        Z.ravel(),
    ])
    N_test = len(wall_pts)
    print(f"Source at rho={rho0}, {N_test} wall test points")

    # Compute G_cyl at wall points
    for n_max in [5, 10, 15, 20]:
        t0 = time.time()
        G = assemble_cylinder_stokeslet_matrix(
            wall_pts, source, R_CYL, MU,
            n_max=n_max, n_k=60, n_phi=64,
        )
        dt = time.time() - t0

        # Apply unit force in each direction and check wall velocity
        for j, label in enumerate(["x", "y", "z"]):
            f = np.zeros(3)
            f[j] = 1.0
            u_wall = (G @ f).reshape(-1, 3)
            max_res = np.max(np.abs(u_wall))
            rms_res = np.sqrt(np.mean(u_wall**2))
            print(f"  n_max={n_max:2d}: f_{label} → "
                  f"max|u|={max_res:.6e}, rms|u|={rms_res:.6e}  "
                  f"({dt:.1f}s)")


def test_free_space_recovery():
    """With R_cyl → ∞, G_cyl should approach G_free."""
    print("\n" + "=" * 60)
    print("Test: Free-space recovery (large cylinder)")
    print("=" * 60)

    source = np.array([[0.5, 0.0, 0.0]])
    target = np.array([[0.0, 0.8, 0.3]])

    G_free = _assemble_free_space(target, source, MU)
    print(f"G_free:\n{G_free.reshape(3,3)}")

    # Large cylinder should give ~same result
    for R in [10.0, 50.0]:
        G_cyl = assemble_cylinder_stokeslet_matrix(
            target, source, R, MU, n_max=10, n_k=40, n_phi=32,
        )
        diff = np.max(np.abs(G_cyl - G_free))
        print(f"R={R:5.1f}: max|G_cyl - G_free| = {diff:.6e}")


if __name__ == "__main__":
    test_free_space_recovery()
    test_single_stokeslet_wall_residual()
