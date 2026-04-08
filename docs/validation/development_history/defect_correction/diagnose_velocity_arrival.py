#!/usr/bin/env python3
"""Velocity development diagnostic at eval spheres.

Measures how quickly the IB-LBM velocity develops at the closest eval
sphere (R=1.15a) relative to the BEM free-space Stokeslet. Determines
whether there's a clean "acoustic window" — a plateau in the ratio
between signal arrival and boundary reflection.

Runs at 48³ and 64³. Reports ratio every 5 steps for steps 0-200.
"""

import os
import sys

os.environ["XLA_FLAGS"] = " ".join([
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_enable_triton_gemm=false",
])
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_mime")

import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.d3q19 import init_equilibrium, Q
from mime.nodes.environment.lbm.bounce_back import compute_missing_mask
from mime.nodes.environment.lbm.immersed_boundary import (
    precompute_ib_stencil,
    spread_forces,
    interpolate_velocity,
)
from mime.nodes.environment.stokeslet.bem import assemble_system_matrix
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field

import argparse


def run_diagnostic(N_lbm_target):
    a = 1.0; mu = 1.0; rho = 1.0; lam = 0.3
    R_cyl = a / lam

    # LBM grid
    domain_extent = 2.5 * R_cyl
    dx = domain_extent / N_lbm_target
    N_lbm = int(np.ceil(domain_extent / dx))
    N_lbm = ((N_lbm + 7) // 8) * 8
    N = N_lbm

    tau = 0.8
    nu_lu = (tau - 0.5) / 3.0
    nu_phys = mu / rho
    dt_lbm = nu_lu * dx**2 / nu_phys
    force_conv = dt_lbm**2 / (rho * dx**4)

    # Acoustic timescales
    c_s_lu = 1.0 / np.sqrt(3.0)
    R_eval_factor = 1.15
    R_eval = R_eval_factor * a
    r_eval_lu = R_eval / dx
    r_boundary_lu = N_lbm / 2.0  # distance from center to nearest face
    r_wall_lu = R_cyl / dx  # distance from center to pipe wall

    t_signal = r_eval_lu / c_s_lu
    t_reflect_face = 2 * r_boundary_lu / c_s_lu
    t_reflect_wall = 2 * r_wall_lu / c_s_lu

    # Viscous diffusion time to eval sphere
    t_diff_eval = r_eval_lu**2 / (2 * nu_lu)

    print(f"\n{'='*70}")
    print(f"VELOCITY ARRIVAL DIAGNOSTIC — N={N_lbm}³ (target {N_lbm_target}³)")
    print(f"{'='*70}")
    print(f"dx = {dx:.4f}, tau = {tau}, nu_lu = {nu_lu:.4f}")
    print(f"R_eval = {R_eval:.2f} ({R_eval_factor}a), r_eval_lu = {r_eval_lu:.1f}")
    print(f"r_wall_lu = {r_wall_lu:.1f}, r_boundary_lu = {r_boundary_lu:.1f}")
    print(f"Acoustic arrival at eval:    t_signal  = {t_signal:.0f} steps")
    print(f"Acoustic wall reflection:    t_wall    = {t_reflect_wall:.0f} steps")
    print(f"Acoustic face reflection:    t_face    = {t_reflect_face:.0f} steps")
    print(f"Viscous diffusion to eval:   t_diff    = {t_diff_eval:.0f} steps")
    print(f"{'='*70}\n", flush=True)

    # Body mesh + BEM
    body = sphere_surface_mesh(radius=a, n_refine=2)
    body_pts = jnp.array(body.points)
    body_wts = jnp.array(body.weights)
    eps = body.mean_spacing / 2.0

    A = assemble_system_matrix(body_pts, body_wts, eps, mu)
    lu, piv = jax.scipy.linalg.lu_factor(A)

    # Unit x-translation: u_body = (1, 0, 0) at all surface points
    N_b = body.n_points
    u_body_x = jnp.zeros((N_b, 3))
    u_body_x = u_body_x.at[:, 0].set(1.0)
    traction_x = jax.scipy.linalg.lu_solve((lu, piv), u_body_x.ravel()).reshape(N_b, 3)

    # IB stencil
    body_pts_lu = np.array(body.points) / dx + np.array([N / 2] * 3)
    ib_idx, ib_wts = precompute_ib_stencil(body_pts_lu, (N, N, N))
    ib_idx = jnp.array(ib_idx)
    ib_wts = jnp.array(ib_wts)

    # Force field from free-space traction
    point_forces = traction_x * body_wts[:, None] * force_conv
    force_field = spread_forces(point_forces, ib_idx, ib_wts, (N, N, N))

    # Eval sphere stencil at R=1.15a
    ev_mesh = sphere_surface_mesh(radius=R_eval, n_refine=2)
    ev_lu = ev_mesh.points / dx + np.array([N / 2] * 3)
    ev_idx, ev_wts = precompute_ib_stencil(ev_lu, (N, N, N))
    ev_idx = jnp.array(ev_idx)
    ev_wts = jnp.array(ev_wts)
    ev_pts_phys = jnp.array(ev_mesh.points)

    # BEM Stokeslet velocity at eval sphere (the "true" free-space answer)
    u_bem = evaluate_velocity_field(
        ev_pts_phys, body_pts, body_wts, traction_x, eps, mu,
    )
    u_bem_mean_x = float(jnp.mean(u_bem[:, 0]))
    print(f"BEM Stokeslet mean u_x at eval sphere: {u_bem_mean_x:.6f}")

    # Pipe wall mask (walled LBM — same as production)
    vessel_R_lu = R_cyl / dx
    cx, cy = N / 2.0, N / 2.0
    ix = jnp.arange(N, dtype=jnp.float32)
    iy = jnp.arange(N, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
    dist_2d = jnp.sqrt((gx - cx)**2 + (gy - cy)**2)
    pipe_wall = jnp.broadcast_to(
        (dist_2d >= vessel_R_lu)[..., None],
        (N, N, N),
    )
    pipe_missing = compute_missing_mask(pipe_wall)

    # LBM step function
    from mime.nodes.environment.lbm.pallas_lbm import _apply_open_bc
    open_bc_axis = 2

    try:
        from mime.nodes.environment.lbm.triton_kernels import (
            TRITON_AVAILABLE, lbm_full_step_triton, _get_d3q19_jax,
        )
        if TRITON_AVAILABLE:
            _get_d3q19_jax()  # populate cache before JIT
            pipe_missing_flat = pipe_missing.reshape(Q * N**3).astype(jnp.int32)
            def lbm_step(f, force):
                f, u = lbm_full_step_triton(f, force, tau, pipe_wall, pipe_missing_flat, open_bc_axis)
                for ax in range(3):
                    if ax != open_bc_axis:
                        f = _apply_open_bc(f, ax)
                return f, u
            lbm_step = jax.jit(lbm_step)
            print("Using Triton backend")
        else:
            raise ImportError
    except ImportError:
        from mime.nodes.environment.lbm.pallas_lbm import lbm_full_step_pallas
        def lbm_step(f, force):
            f, u = lbm_full_step_pallas(f, force, tau, pipe_wall, pipe_missing, open_bc_axis)
            for ax in range(3):
                if ax != open_bc_axis:
                    f = _apply_open_bc(f, ax)
            return f, u
        lbm_step = jax.jit(lbm_step)
        print("Using JAX gather backend")

    # Run from equilibrium, sample every 5 steps
    f_lbm = init_equilibrium(N, N, N)
    max_steps = 500

    print(f"\n{'step':>6s}  {'u_LBM_x':>12s}  {'u_BEM_x':>12s}  {'ratio':>10s}  {'u_LBM_z':>12s}  {'r_z':>10s}")
    print("-" * 78, flush=True)

    for step in range(max_steps):
        f_lbm, u_lbm = lbm_step(f_lbm, force_field)

        if step % 5 == 0 or step < 20:
            u_eval = interpolate_velocity(u_lbm, ev_idx, ev_wts) * dx / dt_lbm
            u_eval_mean_x = float(jnp.mean(u_eval[:, 0]))
            u_eval_mean_z = float(jnp.mean(u_eval[:, 2]))
            ratio_x = u_eval_mean_x / (u_bem_mean_x + 1e-30)

            u_bem_mean_z = float(jnp.mean(u_bem[:, 2]))
            ratio_z = u_eval_mean_z / (u_bem_mean_z + 1e-30) if abs(u_bem_mean_z) > 1e-10 else 0.0

            marker = ""
            if abs(step - t_signal) < 3:
                marker = " ← signal arrival"
            elif abs(step - t_reflect_wall) < 3:
                marker = " ← wall reflection"
            elif abs(step - t_reflect_face) < 3:
                marker = " ← face reflection"
            elif abs(step - t_diff_eval) < 3:
                marker = " ← diffusion time"

            print(f"{step:6d}  {u_eval_mean_x:12.6f}  {u_bem_mean_x:12.6f}  "
                  f"{ratio_x:10.4f}  {u_eval_mean_z:12.6f}  {ratio_z:10.4f}{marker}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", default="48,64")
    args = parser.parse_args()
    resolutions = [int(n) for n in args.resolutions.split(",")]

    print(f"Device: {jax.devices()[0]}")

    for N in resolutions:
        run_diagnostic(N)


if __name__ == "__main__":
    main()
