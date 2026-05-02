"""P1 verification — manufactured Poisson solution at 64³.

Discrete-mode test using FFT solver. The discrete Laplacian
eigenvector for periodic axes is exp(2πijk/N), and for cell-centred
Neumann is cos((2j+1)kπ/(2N)). With known eigenvalues we test that
the FFT solver returns the eigenvector itself when fed the
appropriate scaled rhs.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.pressure import (
    make_pressure_solver_fft, make_helmholtz_solver_fft,
)


def main():
    print("=" * 72)
    print("P1 verification — FFT Poisson + Helmholtz manufactured tests")
    print("=" * 72)

    # ---- Pressure: Neumann xy + periodic z, 64³ ----
    N = 64
    L = 1.0
    mesh = make_cartesian_mesh_3d(N, N, N, L, L, L,
                                  origin=(-L/2, -L/2, 0),
                                  periodic_z=True)
    dx, dy, dz = mesh.cartesian_spacing
    # Discrete eigenvector: cos((2i+1)π/(2N)) * cos((2j+1)π/(2N))
    #                       * cos(2π k_z / N)   (k_x = k_y = 1, k_z = 1)
    ix, jy, kz = jnp.meshgrid(jnp.arange(N, dtype=jnp.float32),
                              jnp.arange(N, dtype=jnp.float32),
                              jnp.arange(N, dtype=jnp.float32), indexing="ij")
    p_mode = (jnp.cos((2*ix+1)*jnp.pi/(2*N))
              * jnp.cos((2*jy+1)*jnp.pi/(2*N))
              * jnp.cos(2*jnp.pi*kz/N))
    lam_x = -(4/dx**2)*jnp.sin(jnp.pi/(2*N))**2
    lam_y = -(4/dy**2)*jnp.sin(jnp.pi/(2*N))**2
    lam_z = -(4/dz**2)*jnp.sin(jnp.pi/N)**2
    lam = lam_x + lam_y + lam_z
    rhs = (lam * p_mode).flatten() * mesh.V
    solver = jax.jit(make_pressure_solver_fft(
        mesh, bc=("neumann", "neumann", "periodic")
    ))
    p = solver(rhs)
    p_true = p_mode.flatten() - jnp.mean(p_mode)
    err = float(jnp.linalg.norm(p - p_true) / jnp.linalg.norm(p_true))
    print(f"  Pressure 64³ discrete-mode: rel err = {err:.4e}")

    # ---- Helmholtz: Dirichlet xy + periodic z, 64³ ----
    helm = jax.jit(make_helmholtz_solver_fft(
        mesh, bc=("dirichlet", "dirichlet", "periodic")
    ))
    u_mode = (jnp.sin((2*ix+1)*jnp.pi/(2*N))
              * jnp.sin((2*jy+1)*jnp.pi/(2*N))
              * jnp.cos(2*jnp.pi*kz/N))
    alpha = 0.1
    b = ((1 - alpha*lam) * u_mode).flatten()
    u = helm(b, alpha)
    u_true = u_mode.flatten()
    err = float(jnp.linalg.norm(u - u_true) / jnp.linalg.norm(u_true))
    print(f"  Helmholtz 64³ discrete-mode: rel err = {err:.4e}")

    # Vector field shape check
    b_vec = jnp.stack([b, b * 2.0, b * 0.5], axis=-1)
    u_vec = helm(b_vec, alpha)
    print(f"  Helmholtz vector shape: {u_vec.shape}, all finite: {bool(jnp.all(jnp.isfinite(u_vec)))}")


if __name__ == "__main__":
    main()
