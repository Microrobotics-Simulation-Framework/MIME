"""T0 — Ghia Re=100 lid-driven cavity + autodiff verification.

Reports:
  * RMS error vs Ghia, Ghia & Shin (1982) Table I across all 17
    reference y-positions on x=0.5 centreline.
  * Pointwise FVM-vs-Ghia comparison.
  * jax.grad(total_drag_on_lid)(U_lid) vs central finite difference.
"""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.fvm import make_cartesian_mesh_2d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.simple import (
    SimpleConfig, run_simple, continuity_residual_l2, momentum_residual_l2,
)
from mime.nodes.environment.fvm.operators import grad_green_gauss

GHIA_Y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
])
GHIA_U = np.array([
    1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
    -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
    -0.03717, 0.00000,
])


def _build_cavity(N: int, U_lid: float):
    L = 1.0
    mesh = make_cartesian_mesh_2d(N, N, L, L)
    zero_vel = jnp.zeros((N, 2))
    lid_vel = jnp.zeros((N, 2)).at[:, 0].set(U_lid)
    zero_F = jnp.zeros((N,))
    bcs = {
        "x_min": VelocityBC(u_wall=zero_vel, F_through=zero_F),
        "x_max": VelocityBC(u_wall=zero_vel, F_through=zero_F),
        "y_min": VelocityBC(u_wall=zero_vel, F_through=zero_F),
        "y_max": VelocityBC(u_wall=lid_vel,  F_through=zero_F),
    }
    return mesh, bcs


def solve_cavity(U_lid: jnp.ndarray, N: int = 128, Re: float = 100.0,
                 n_warm: int = 2000, n_acc: int = 8000):
    nu = U_lid * 1.0 / Re
    mesh, bcs = _build_cavity(N, U_lid)
    cfg_w = SimpleConfig(nu=nu, alpha_u=0.7, alpha_p=0.3, gamma_conv=0.0)
    state = run_simple(mesh, bcs, cfg_w, n_iter=n_warm)
    cfg_a = SimpleConfig(nu=nu, alpha_u=0.7, alpha_p=0.3, gamma_conv=0.7)
    state = run_simple(mesh, bcs, cfg_a, n_iter=n_acc, initial=state)
    return state, mesh, cfg_a, bcs


def total_drag_on_lid(state, mesh, cfg, bcs, U_lid):
    """Viscous drag exerted by the fluid on the moving lid (y_max).

    F_drag_x = ∫_lid μ ∂u/∂y dA evaluated at the wall.
    For each lid face, the wall-tangential viscous flux is
    μ * (u_wall - u_owner) * |Sf| / (dy/2).
    """
    mu = cfg.rho * cfg.nu
    patch = mesh.patch("y_max")
    u_wall = U_lid                # tangential lid velocity
    u_owner = state["u"][patch.owner, 0]   # x-component of owner cell
    d_b = jnp.linalg.norm(patch.d, axis=-1)  # half-cell distance
    f_face = mu * (u_wall - u_owner) * patch.area / d_b
    return jnp.sum(f_face)


def main():
    print("=" * 72)
    print("T0 — Ghia Re=100 lid-driven cavity")
    print("=" * 72)
    N = 128
    U_lid = jnp.float32(1.0)

    t0 = time.time()
    state, mesh, cfg, bcs = solve_cavity(U_lid, N=N)
    state["u"].block_until_ready()
    elapsed = time.time() - t0

    cont = float(continuity_residual_l2(state, mesh, bcs))
    mom = float(momentum_residual_l2(state, mesh, bcs, cfg))
    print(f"  solver wall time: {elapsed:.1f}s  | continuity={cont:.2e}  momentum={mom:.2e}")

    # u-velocity along x=0.5
    u = np.asarray(state["u"]).reshape(N, N, 2)
    u_centre = 0.5 * (u[N//2-1, :, 0] + u[N//2, :, 0])
    y_cells = (np.arange(N) + 0.5) / N
    y_aug = np.concatenate([[0.0], y_cells, [1.0]])
    u_aug = np.concatenate([[0.0], u_centre, [float(U_lid)]])
    u_pred = np.interp(GHIA_Y, y_aug, u_aug)

    rmse = float(np.sqrt(np.mean((u_pred - GHIA_U) ** 2)))
    max_abs = float(np.max(np.abs(u_pred - GHIA_U)))
    print(f"\n  RMSE vs Ghia: {rmse*100:.3f}%   max abs err: {max_abs*100:.3f}%")
    print(f"\n  pointwise (y, FVM, Ghia, err):")
    for y, up, ug in zip(GHIA_Y, u_pred, GHIA_U):
        print(f"    y={y:.4f}  FVM={up:+.5f}  Ghia={ug:+.5f}  err={up-ug:+.5f}")

    target_pass = rmse < 0.01
    print(f"\n  PASS criterion (RMS < 1.0%): {'PASS' if target_pass else 'FAIL'} (rmse={rmse*100:.3f}%)")

    # ---- Autodiff vs FD ----
    print("\n" + "=" * 72)
    print("T0 — Autodiff (drag on lid) vs finite difference")
    print("=" * 72)
    # Use a smaller grid + shorter horizon for FD reference cost
    Nad = 32

    @jax.jit
    def drag(U: jnp.ndarray):
        s, m, c, b = solve_cavity(U, N=Nad, n_warm=400, n_acc=2000)
        return total_drag_on_lid(s, m, c, b, U)

    # Compile + warm up
    drag(jnp.float32(1.0)).block_until_ready()

    t0 = time.time()
    grad_ad = float(jax.grad(drag)(jnp.float32(1.0)))
    print(f"  jax.grad evaluation: {time.time()-t0:.1f}s")

    eps = 1e-3
    t0 = time.time()
    f_plus  = float(drag(jnp.float32(1.0 + eps)))
    f_minus = float(drag(jnp.float32(1.0 - eps)))
    grad_fd = (f_plus - f_minus) / (2 * eps)
    print(f"  finite difference: {time.time()-t0:.1f}s")

    rel_err = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
    print(f"  AD={grad_ad:.6e}, FD={grad_fd:.6e}, rel_err={rel_err:.3e}")
    autodiff_pass = rel_err < 1e-3
    print(f"  PASS criterion (rel_err < 0.1%): {'PASS' if autodiff_pass else 'FAIL'}")

    print("\nSummary:")
    print(f"  T0 Ghia Re=100 RMS:  rmse={rmse*100:.3f}%  ({'PASS' if target_pass else 'FAIL'})")
    print(f"  T0 Autodiff:         rel_err={rel_err:.3e}  ({'PASS' if autodiff_pass else 'FAIL'})")


if __name__ == "__main__":
    main()
