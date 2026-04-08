#!/usr/bin/env python3
"""Periodic ratio with mean-velocity correction.

The periodic LBM with a net force has linearly growing mean velocity
(no wall to balance the force). The PERIODIC STOKESLET is the deviation
from this uniform drift. Subtract the domain-mean velocity to extract it.

u_periodic_stokeslet(R) = u_measured(R) - <u>_domain

This should stabilise once the Stokeslet profile develops (~1000 steps),
even though the raw velocity keeps growing.
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

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh, cylinder_surface_mesh,
)
from mime.nodes.environment.stokeslet.resistance import (
    compute_nn_confined_resistance_matrix,
)
from mime.nodes.environment.stokeslet.bem import compute_force_torque
from mime.nodes.environment.defect_correction import DefectCorrectionFluidNode
from mime.nodes.environment.lbm.d3q19 import init_equilibrium
from mime.nodes.environment.lbm.immersed_boundary import interpolate_velocity

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM
EPS_NN = min(0.05, 0.02 * (R_CYL - A))


def compute_nn_bem_reference():
    bc = sphere_surface_mesh(radius=A, n_refine=2)
    bf = sphere_surface_mesh(radius=A, n_refine=4)
    cl = 12.0 * R_CYL
    wc = cylinder_surface_mesh(radius=R_CYL, length=cl, n_circ=48, n_axial=16,
                                cluster_center=True)
    wf = cylinder_surface_mesh(radius=R_CYL, length=cl, n_circ=192, n_axial=64,
                                cluster_center=True)
    Rn = compute_nn_confined_resistance_matrix(
        jnp.array(bc.points), jnp.array(bc.weights),
        jnp.array(bf.points), jnp.array(bf.weights),
        jnp.array(wc.points), jnp.array(wc.weights),
        jnp.array(wf.points), jnp.array(wf.weights),
        jnp.zeros(3), EPS_NN, MU,
    )
    return [float(Rn[i, i]) for i in range(6)]


def main():
    print(f"Device: {jax.devices()[0]}")

    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}\n")

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

    # Periodic step
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

    # Hasimoto correction
    L_phys = N * node._dx
    aL = A / L_phys
    hasimoto = 1.0 - 2.837 * aL + 4.19 * aL**3
    print(f"N={N}^3, a/L={aL:.4f}, Hasimoto={hasimoto:.4f}")

    # Eval sphere stencils
    R_factors = [1.15, 1.5, 2.0]
    eval_stencils = []
    for r_idx, rf in enumerate(R_factors):
        if r_idx < len(node._eval_stencils_all):
            eval_stencils.append((rf, node._eval_stencils_all[r_idx]))

    max_steps = 15000
    sample_every = 250

    for col in [0, 2]:
        col_name = "F_x (transverse)" if col == 0 else "F_z (axial)"
        comp = col

        U = e[col]
        r = node._body_pts - center
        u_body = U + jnp.cross(jnp.zeros(3), r)
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)
        F_free, _ = compute_force_torque(node._body_pts, node._body_wts, traction, center)
        F_free_val = float(F_free[col])
        K_true = nn_diag[col] / F_free_val

        force_field = node._spread_traction(traction)

        f_walled = init_equilibrium(N, N, N)
        f_periodic = init_equilibrium(N, N, N)

        print(f"\n{'='*110}")
        print(f"{col_name}: F_free={F_free_val:.4f}, K_true={K_true:.4f}")
        print(f"{'='*110}")

        # Header
        header = f"{'steps':>6}  {'u_w':>8}  {'<u>_p':>8}"
        for rf, _ in eval_stencils:
            header += f"  |{'R=%.2fa' % rf:^24}|"
        print(header)

        subhdr = f"{'':>6}  {'':>8}  {'':>8}"
        for _ in eval_stencils:
            subhdr += f"  {'u_p_raw':>8} {'u_p_corr':>8} {'K_corr':>6}"
        print(subhdr)
        print("-" * (26 + 28 * len(eval_stencils)))

        prev_K = {rf: None for rf, _ in eval_stencils}
        t0 = time.time()

        for step in range(1, max_steps + 1):
            f_walled, u_walled = node._lbm_full_step(f_walled, force_field)
            f_periodic, u_periodic = periodic_step(f_periodic, force_field)

            if step % sample_every == 0:
                # Domain-mean velocity of periodic LBM
                u_p_mean = jnp.mean(u_periodic, axis=(0, 1, 2))  # (3,)
                u_p_mean_comp = float(u_p_mean[comp])

                # Walled velocity at closest eval sphere
                es_close = eval_stencils[0][1]
                u_w_eval = interpolate_velocity(u_walled, es_close['idx'], es_close['wts'])
                u_w_eval = u_w_eval * node._dx / node._dt_lbm
                u_w_comp = float(jnp.mean(u_w_eval[:, comp]))

                row = f"{step:6d}  {u_w_comp:>8.5f}  {u_p_mean_comp:>8.5f}"

                for rf, es in eval_stencils:
                    u_p_eval = interpolate_velocity(u_periodic, es['idx'], es['wts'])
                    u_p_eval = u_p_eval * node._dx / node._dt_lbm
                    u_p_raw = float(jnp.mean(u_p_eval[:, comp]))

                    # Corrected: subtract domain mean
                    u_p_corr = u_p_raw - u_p_mean_comp

                    if abs(u_p_corr) > 1e-10:
                        K_corr = u_w_comp / u_p_corr
                    else:
                        K_corr = float('nan')

                    row += f"  {u_p_raw:>8.5f} {u_p_corr:>8.5f} {K_corr:>6.3f}"

                print(row, flush=True)

        elapsed = time.time() - t0

        # Summary
        print(f"\n  Time: {elapsed:.0f}s")
        print(f"\n  Summary at step {max_steps}:")
        u_p_mean = jnp.mean(u_periodic, axis=(0, 1, 2))
        u_p_mean_comp = float(u_p_mean[comp])

        es_close = eval_stencils[0][1]
        u_w_eval = interpolate_velocity(u_walled, es_close['idx'], es_close['wts'])
        u_w_comp = float(jnp.mean(u_w_eval[:, comp] * node._dx / node._dt_lbm))

        for rf, es in eval_stencils:
            u_p_eval = interpolate_velocity(u_periodic, es['idx'], es['wts'])
            u_p_raw = float(jnp.mean(u_p_eval[:, comp]) * node._dx / node._dt_lbm)
            u_p_corr = u_p_raw - u_p_mean_comp
            K_corr = u_w_comp / u_p_corr if abs(u_p_corr) > 1e-10 else float('nan')
            K_wall = K_corr * hasimoto
            err = abs(K_wall - K_true) / K_true * 100 if not np.isnan(K_wall) else float('nan')
            print(f"  R={rf:.2f}a: K_corr={K_corr:.4f}, "
                  f"K_wall=K_corr*hasimoto={K_wall:.4f}, K_true={K_true:.4f}, err={err:.1f}%")


if __name__ == "__main__":
    main()
