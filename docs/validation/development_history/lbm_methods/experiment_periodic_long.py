#!/usr/bin/env python3
"""Quick test: does the periodic ratio stabilise at long times?

The periodic diffusion time is L^2/(2*nu) ~ 15000 steps at 56^3.
Run both F_x and F_z to 20000 steps, sampling ratio at R=1.15a.
"""

import os
os.environ["XLA_FLAGS"] = " ".join([
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_enable_triton_gemm=false",
])
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_mime")

import jax.numpy as jnp
import numpy as np
import time

from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.bem import compute_force_torque
from mime.nodes.environment.defect_correction import DefectCorrectionFluidNode
from mime.nodes.environment.lbm.d3q19 import init_equilibrium
from mime.nodes.environment.lbm.immersed_boundary import interpolate_velocity

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM


def main():
    print(f"Device: {jax.devices()[0]}")

    dx = 2 * R_CYL / (48 * 0.8)
    body = sphere_surface_mesh(radius=A, n_refine=2)
    node = DefectCorrectionFluidNode(
        "dc", timestep=0.001, mu=MU, rho=RHO,
        body_mesh=body, body_radius=A,
        vessel_radius=R_CYL, dx=dx,
        open_bc_axis=2, max_defect_iter=25, alpha=0.3,
    )

    N = node._nx
    N_b = node._N_body
    center = jnp.zeros(3)
    e = jnp.eye(3)

    # Periodic LBM step (no wall, no open BCs)
    from mime.nodes.environment.lbm.triton_kernels import (
        lbm_full_step_triton, _get_d3q19_jax,
    )
    _get_d3q19_jax()
    _tau = node._tau
    _nw = node._no_wall
    _nm = node._no_missing_flat

    @jax.jit
    def periodic_step(f, force):
        f, u = lbm_full_step_triton(f, force, _tau, _nw, _nm, None)
        return f, u

    # Diffusion time estimate
    nu_lu = (_tau - 0.5) / 3.0
    L_lu = N
    t_diff = L_lu**2 / (2 * nu_lu)
    print(f"N={N}^3, tau={_tau}, nu_lu={nu_lu:.4f}")
    print(f"Periodic diffusion time: L^2/(2*nu) = {t_diff:.0f} steps")

    # Hasimoto
    L_phys = N * node._dx
    aL = A / L_phys
    hasimoto = 1.0 - 2.837 * aL + 4.19 * aL**3
    print(f"a/L = {aL:.4f}, Hasimoto = {hasimoto:.4f}\n")

    es = node._eval_stencils_all[0]  # R=1.15a

    max_steps = 20000
    sample_every = 500

    for col in [0, 2]:
        col_name = "F_x" if col == 0 else "F_z"

        U = e[col]
        r = node._body_pts - center
        u_body = U + jnp.cross(jnp.zeros(3), r)
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)
        F_free, _ = compute_force_torque(node._body_pts, node._body_wts, traction, center)
        F_free_val = float(F_free[col])

        force_field = node._spread_traction(traction)

        f_walled = init_equilibrium(N, N, N)
        f_periodic = init_equilibrium(N, N, N)

        print(f"--- {col_name} (F_free={F_free_val:.4f}) ---")
        print(f"{'steps':>6}  {'u_w':>10}  {'u_p':>10}  {'K_raw':>8}  {'dK/K%':>8}")
        print("-" * 50)

        prev_K = None
        t0 = time.time()

        for step in range(1, max_steps + 1):
            f_walled, u_walled = node._lbm_full_step(f_walled, force_field)
            f_periodic, u_periodic = periodic_step(f_periodic, force_field)

            if step % sample_every == 0:
                u_w = interpolate_velocity(u_walled, es['idx'], es['wts']) * node._dx / node._dt_lbm
                u_p = interpolate_velocity(u_periodic, es['idx'], es['wts']) * node._dx / node._dt_lbm

                u_w_comp = float(jnp.mean(u_w[:, col]))
                u_p_comp = float(jnp.mean(u_p[:, col]))

                K_raw = u_w_comp / (u_p_comp + 1e-30)

                if prev_K is not None:
                    dK = abs(K_raw - prev_K) / (abs(prev_K) + 1e-30) * 100
                else:
                    dK = float('nan')

                print(f"{step:6d}  {u_w_comp:>10.6f}  {u_p_comp:>10.6f}  {K_raw:>8.4f}  {dK:>7.2f}%",
                      flush=True)
                prev_K = K_raw

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
