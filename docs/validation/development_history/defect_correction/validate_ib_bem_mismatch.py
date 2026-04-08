#!/usr/bin/env python3
"""Measure IB-BEM body mismatch in free-space (no vessel wall).

Spreads BEM free-space traction into an UNWALLED LBM domain and
compares the resulting velocity at eval spheres with the BEM Stokeslet
velocity. The difference is the pure IB-BEM body mismatch — the
contamination that enters the defect correction wall measurement.

If this mismatch is significant and resolution-dependent, it explains
the anti-convergence observed in transverse translation (F_x getting
worse from 48³ to 128³).

Usage:
    python3 scripts/validate_ib_bem_mismatch.py [--resolutions 48,64]
"""

import argparse
import os
import time

os.environ["XLA_FLAGS"] = " ".join([
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_enable_triton_gemm=false",
])

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_mime")

import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.bem import assemble_system_matrix
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field
from mime.nodes.environment.lbm.d3q19 import init_equilibrium, Q
from mime.nodes.environment.lbm.immersed_boundary import (
    precompute_ib_stencil,
    spread_forces,
    interpolate_velocity,
)


def run_free_lbm_steps(f, force, tau, open_bc_axis, n_steps):
    """Run LBM steps with NO pipe wall (free-space)."""
    # Try Triton hybrid, fall back to JAX gather
    try:
        from mime.nodes.environment.lbm.triton_kernels import TRITON_AVAILABLE
        if TRITON_AVAILABLE:
            return _run_triton_hybrid(f, force, tau, open_bc_axis, n_steps)
    except ImportError:
        pass

    from mime.nodes.environment.lbm.pallas_lbm import lbm_full_step_pallas
    nx, ny, nz, _ = f.shape
    # No wall: pipe_wall = all False, pipe_missing = all zeros
    pipe_wall = jnp.zeros((nx, ny, nz), dtype=bool)
    pipe_missing = jnp.zeros((Q, nx, ny, nz), dtype=bool)
    for step in range(n_steps):
        f, u = lbm_full_step_pallas(f, force, tau, pipe_wall, pipe_missing, open_bc_axis)
    return f, u


def _run_triton_hybrid(f, force, tau, open_bc_axis, n_steps):
    """Triton collision + JAX streaming, NO wall."""
    import jax_triton as jt
    from mime.nodes.environment.lbm.triton_kernels import (
        _macroscopic_kernel, _collision_forcing_kernel,
    )
    from mime.nodes.environment.lbm.pallas_lbm import _build_stream_indices, _apply_open_bc
    from mime.nodes.environment.lbm.d3q19 import E, W, OPP

    E_NP = np.array(E, dtype=np.int32)
    W_NP = np.array(W, dtype=np.float32)
    OPP_NP = np.array(OPP, dtype=np.int32)

    nx, ny, nz, _ = f.shape
    N = nx * ny * nz
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)

    ex = jnp.array(E_NP[:, 0])
    ey = jnp.array(E_NP[:, 1])
    ez = jnp.array(E_NP[:, 2])
    w = jnp.array(W_NP)
    opp = jnp.array(OPP_NP)
    stream_idx = _build_stream_indices(nx, ny, nz)

    force_flat = force.reshape(N, 3)

    for step in range(n_steps):
        f_flat = f.reshape(N, Q)

        # Kernel 1: macroscopic
        rho_flat, ux, uy, uz = jt.triton_call(
            f_flat, force_flat, ex, ey, ez,
            kernel=_macroscopic_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((N,), jnp.float32),
                jax.ShapeDtypeStruct((N,), jnp.float32),
                jax.ShapeDtypeStruct((N,), jnp.float32),
                jax.ShapeDtypeStruct((N,), jnp.float32),
            ],
            grid=grid, N_FLAT=N, QQ=Q, BLOCK=BLOCK,
        )
        u = jnp.stack([ux, uy, uz], axis=-1).reshape(nx, ny, nz, 3)

        # Kernel 2: collision
        f_post_flat = jt.triton_call(
            f_flat, rho_flat, ux, uy, uz, force_flat, ex, ey, ez, w,
            kernel=_collision_forcing_kernel,
            out_shape=jax.ShapeDtypeStruct((N, Q), jnp.float32),
            grid=grid, N_FLAT=N, QQ=Q, BLOCK=BLOCK, TAU=tau,
        )

        # JAX streaming (NO bounce-back — free space)
        f_streamed = f_post_flat[stream_idx, jnp.arange(Q)].reshape(nx, ny, nz, Q)

        # Open BCs on axial faces only
        f = _apply_open_bc(f_streamed, open_bc_axis)

    return f, u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", default="48,64",
                        help="Comma-separated grid sizes")
    args = parser.parse_args()

    resolutions = [int(n) for n in args.resolutions.split(",")]

    print(f"Device: {jax.devices()[0]}")

    a = 1.0
    mu = 1.0
    rho = 1.0
    lam = 0.3
    R_cyl = a / lam
    epsilon = None  # auto from mesh
    tau = 0.8
    nu_lu = (tau - 0.5) / 3.0

    labels = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]
    e = jnp.eye(3)

    # NN-BEM reference wall correction magnitude (from validated results)
    # Free-space drag: F_x_free ≈ 18.91, confined F_x ≈ 31.76 → wall adds ~12.85
    print("\n" + "=" * 70)
    print("IB-BEM BODY MISMATCH TEST (free-space, no wall)")
    print("=" * 70)

    for N_lbm in resolutions:
        dx = 2 * R_cyl / (N_lbm * 0.8)
        dt_lbm = nu_lu * dx**2 / (mu / rho)
        force_conv = dt_lbm**2 / (rho * dx**4)

        body = sphere_surface_mesh(radius=a, n_refine=2)
        N_b = body.n_points
        body_pts = jnp.array(body.points)
        body_wts = jnp.array(body.weights)

        if epsilon is None:
            eps = body.mean_spacing / 2.0
        else:
            eps = epsilon

        # BEM system (free-space, body only)
        A = assemble_system_matrix(body_pts, body_wts, eps, mu)
        lu, piv = jax.scipy.linalg.lu_factor(A)

        # Grid setup
        domain_extent = 2.5 * R_cyl
        N_grid = int(np.ceil(domain_extent / dx))
        N_grid = ((N_grid + 7) // 8) * 8
        center_lu = np.array([N_grid / 2] * 3)

        # IB stencil
        body_pts_lu = np.array(body.points) / dx + center_lu
        ib_idx, ib_wts_ib = precompute_ib_stencil(body_pts_lu, (N_grid, N_grid, N_grid))
        ib_idx = jnp.array(ib_idx)
        ib_wts_ib = jnp.array(ib_wts_ib)

        # Eval sphere stencils at multiple radii
        eval_radii = [1.15, 1.2, 1.3, 1.5, 2.0, 3.0]
        eval_stencils = []
        for R_factor in eval_radii:
            R_ev = R_factor * a
            if R_ev >= R_cyl - dx:
                continue
            ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
            ev_lu = ev_mesh.points / dx + center_lu
            ei, ew = precompute_ib_stencil(ev_lu, (N_grid, N_grid, N_grid))
            eval_stencils.append({
                'pts_phys': jnp.array(ev_mesh.points),
                'idx': jnp.array(ei),
                'wts': jnp.array(ew),
                'R_factor': R_factor,
            })

        # Number of LBM steps: enough for flow to develop at eval spheres
        # but before mean drift dominates.
        # Flow develops at ~R²/ν steps; use 500 as a safe choice.
        n_steps = 500

        print(f"\n--- N={N_lbm}³ (grid={N_grid}³, dx={dx:.4f}, "
              f"dt={dt_lbm:.6f}, n_steps={n_steps}) ---")

        for col in [0, 2, 3]:  # transverse (x), axial (z), rotation (z)
            U = e[col] if col < 3 else jnp.zeros(3)
            omega = e[col - 3] if col >= 3 else jnp.zeros(3)

            r = body_pts - jnp.zeros(3)
            u_body = U + jnp.cross(omega, r)

            # BEM solve
            traction = jax.scipy.linalg.lu_solve((lu, piv), u_body.ravel()).reshape(N_b, 3)

            # Spread IB forces (same as defect correction)
            point_forces = traction * body_wts[:, None] * force_conv
            force_field = spread_forces(point_forces, ib_idx, ib_wts_ib,
                                        (N_grid, N_grid, N_grid))

            # Run FREE-SPACE LBM (no wall)
            t0 = time.time()
            f0 = init_equilibrium(N_grid, N_grid, N_grid)
            f, u_lbm = run_free_lbm_steps(f0, force_field, tau, 2, n_steps)
            elapsed = time.time() - t0

            # Sample at eval spheres and compare with BEM
            print(f"  col={col} ({labels[col]}): elapsed={elapsed:.1f}s")
            for es in eval_stencils:
                u_w = interpolate_velocity(u_lbm, es['idx'], es['wts']) * dx / dt_lbm
                u_fs = evaluate_velocity_field(
                    es['pts_phys'], body_pts, body_wts, traction, eps, mu,
                )
                du = u_w - u_fs  # (N_eval, 3) per-point mismatch
                du_mean = jnp.mean(du, axis=0)
                du_mag = float(jnp.linalg.norm(du_mean))

                # For context: what fraction of the wall correction is this?
                # Wall correction for F_x ≈ 12.85 in force → ~0.4 in velocity
                # (rough estimate, depends on eval radius)
                u_fs_mean = float(jnp.linalg.norm(jnp.mean(u_fs, axis=0)))
                rel = du_mag / (u_fs_mean + 1e-30) * 100

                print(f"    R={es['R_factor']:.2f}a: "
                      f"|Δu_IB|={du_mag:.6f}, "
                      f"|u_BEM|={u_fs_mean:.6f}, "
                      f"mismatch={rel:.2f}%, "
                      f"Δu=[{float(du_mean[0]):.6f}, "
                      f"{float(du_mean[1]):.6f}, "
                      f"{float(du_mean[2]):.6f}]")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  If |Δu_IB| is comparable to wall correction errors (~5-10% of u_BEM)")
    print("  and varies with resolution → theory confirmed: IB-BEM body mismatch")
    print("  contaminates the defect correction.")
    print("  If |Δu_IB| < 0.1% of u_BEM → theory rejected, look elsewhere.")
    print("=" * 70)


if __name__ == "__main__":
    main()
