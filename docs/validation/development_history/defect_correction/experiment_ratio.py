#!/usr/bin/env python3
"""Experiment 1: Drag ratio approach.

Tests whether K = u_LBM_walled / u_LBM_free stabilises at eval spheres,
giving the wall correction factor without any BEM comparison.

If the IB transfer function alpha cancels in the ratio, K should:
1. Stabilise quickly (within 100-200 steps)
2. Give the correct wall factor for both transverse and axial
3. Be direction-independent (same method for all columns)
4. Be eval-radius-independent (robust measurement)
"""

import os
import time

os.environ["XLA_FLAGS"] = " ".join([
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_enable_triton_gemm=false",
])
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_mime")

import jax.numpy as jnp
import numpy as np

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
logger = logging.getLogger(__name__)

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM
EPS_NN = min(0.05, 0.02 * (R_CYL - A))
LABELS = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]


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


def make_node(N_target):
    dx = 2 * R_CYL / (N_target * 0.8)
    body = sphere_surface_mesh(radius=A, n_refine=2)
    return DefectCorrectionFluidNode(
        "dc", timestep=0.001, mu=MU, rho=RHO,
        body_mesh=body, body_radius=A,
        vessel_radius=R_CYL, dx=dx,
        open_bc_axis=2,
        max_defect_iter=25,
        alpha=0.3,
    )


def run_ratio_diagnostic(node, nn_diag):
    """Run walled and free LBMs in lockstep, measure velocity ratio at eval spheres."""
    center = jnp.zeros(3)
    e = jnp.eye(3)
    N_b = node._N_body
    N = node._nx

    # BEM free-space drag for normalisation
    F_free_stokes = 6 * np.pi * MU * A  # Stokes drag = 6*pi*mu*a for unit velocity

    # Eval sphere stencils at multiple radii
    eval_radii_info = []
    R_factors = [1.15, 1.3, 1.5, 2.0, 2.5, 3.0]
    for r_idx, R_factor in enumerate(R_factors):
        if r_idx < len(node._eval_stencils_all):
            eval_radii_info.append((R_factor, node._eval_stencils_all[r_idx]))

    # Test columns: F_x (transverse) and F_z (axial)
    test_cols = [0, 2]
    step_samples = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]

    for col in test_cols:
        col_name = "F_x (transverse)" if col == 0 else "F_z (axial)"
        comp = col  # component to measure

        U = e[col]
        omega = jnp.zeros(3)
        r = node._body_pts - center
        u_body = U + jnp.cross(omega, r)

        # BEM free-space solve
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)
        F_free, _ = compute_force_torque(
            node._body_pts, node._body_wts, traction, center,
        )
        F_free_val = float(F_free[col])
        K_true = nn_diag[col] / F_free_val

        print(f"\n{'='*90}")
        print(f"RATIO DIAGNOSTIC: {col_name}")
        print(f"{'='*90}")
        print(f"  F_free (BEM) = {F_free_val:.4f}")
        print(f"  F_conf (NN-BEM) = {nn_diag[col]:.4f}")
        print(f"  K_true = {K_true:.4f}")
        print()

        # IB force field
        force_field = node._spread_traction(traction)

        # Start from equilibrium
        f_walled = init_equilibrium(N, N, N)
        f_free = init_equilibrium(N, N, N)

        prev_sample = 0

        # Print header for each eval radius
        header = f"{'steps':>6}"
        for R_factor, _ in eval_radii_info:
            header += f"  {'R=%.2fa' % R_factor:>20}"
        print(f"{header}")
        subheader = f"{'':>6}"
        for _ in eval_radii_info:
            subheader += f"  {'u_w':>6} {'u_f':>6} {'K':>6}"
        print(subheader)
        print("-" * (8 + 22 * len(eval_radii_info)))

        for target_steps in step_samples:
            # Run from prev_sample to target_steps
            steps_to_run = target_steps - prev_sample
            for s in range(steps_to_run):
                f_walled, u_walled = node._lbm_full_step(f_walled, force_field)
                # Free-space: use no_wall, no_missing
                f_free, u_free = DefectCorrectionFluidNode._lbm_step_core(
                    f_free, force_field, node._tau,
                    node._no_wall, node._no_missing, node._open_bc_axis,
                )
            prev_sample = target_steps

            row = f"{target_steps:6d}"
            for R_factor, es in eval_radii_info:
                u_w = interpolate_velocity(
                    u_walled, es['idx'], es['wts'],
                ) * node._dx / node._dt_lbm
                u_f = interpolate_velocity(
                    u_free, es['idx'], es['wts'],
                ) * node._dx / node._dt_lbm

                u_w_comp = float(jnp.mean(u_w[:, comp]))
                u_f_comp = float(jnp.mean(u_f[:, comp]))

                if abs(u_f_comp) > 1e-10:
                    K = u_w_comp / u_f_comp
                else:
                    K = float('nan')

                row += f"  {u_w_comp:>6.4f} {u_f_comp:>6.4f} {K:>6.3f}"

            print(row, flush=True)

        # Summary: predicted drag from K at each radius using last step count
        print(f"\nPredicted drag at step {step_samples[-1]}:")
        for R_factor, es in eval_radii_info:
            u_w = interpolate_velocity(
                u_walled, es['idx'], es['wts'],
            ) * node._dx / node._dt_lbm
            u_f = interpolate_velocity(
                u_free, es['idx'], es['wts'],
            ) * node._dx / node._dt_lbm

            u_w_comp = float(jnp.mean(u_w[:, comp]))
            u_f_comp = float(jnp.mean(u_f[:, comp]))

            if abs(u_f_comp) > 1e-10:
                K = u_w_comp / u_f_comp
                F_pred = F_free_val * K
                err = abs(F_pred - nn_diag[col]) / abs(nn_diag[col]) * 100
                print(f"  R={R_factor:.2f}a: K={K:.4f}, F_pred={F_pred:.2f}, "
                      f"target={nn_diag[col]:.2f}, err={err:.1f}%")


def run_ratio_full_matrix(node, nn_diag, n_steps, eval_radius_idx=0):
    """Run full 6x6 R matrix using the ratio approach at a given step count."""
    center = jnp.zeros(3)
    e = jnp.eye(3)
    N_b = node._N_body
    N = node._nx

    es = node._eval_stencils_all[eval_radius_idx]

    print(f"\n{'='*78}")
    print(f"RATIO-BASED R MATRIX (n_steps={n_steps}, eval_idx={eval_radius_idx})")
    print(f"{'='*78}")

    R = np.zeros((6, 6))

    for col in range(6):
        U = e[col] if col < 3 else jnp.zeros(3)
        omega = e[col - 3] if col >= 3 else jnp.zeros(3)

        r = node._body_pts - center
        u_body = U + jnp.cross(omega, r)

        # BEM free-space solve
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)
        F_free, T_free = compute_force_torque(
            node._body_pts, node._body_wts, traction, center,
        )

        # IB force field
        force_field = node._spread_traction(traction)

        # Run walled and free LBMs
        f_walled = init_equilibrium(N, N, N)
        f_free_lbm = init_equilibrium(N, N, N)

        for s in range(n_steps):
            f_walled, u_walled = node._lbm_full_step(f_walled, force_field)
            f_free_lbm, u_free_lbm = DefectCorrectionFluidNode._lbm_step_core(
                f_free_lbm, force_field, node._tau,
                node._no_wall, node._no_missing, node._open_bc_axis,
            )

        # Velocity at eval sphere
        u_w = interpolate_velocity(
            u_walled, es['idx'], es['wts'],
        ) * node._dx / node._dt_lbm
        u_f = interpolate_velocity(
            u_free_lbm, es['idx'], es['wts'],
        ) * node._dx / node._dt_lbm

        u_w_mean = jnp.mean(u_w, axis=0)  # (3,)
        u_f_mean = jnp.mean(u_f, axis=0)  # (3,)

        # Component-wise ratio
        K = u_w_mean / (u_f_mean + 1e-30)  # (3,)

        # Apply ratio to free-space drag
        F_conf = np.array(F_free) * np.array(K[:3])
        T_conf = np.array(T_free) * np.array(K[:3])  # simplified

        # Actually for torque columns, need to think about this differently
        # For now, use the full 3-component ratio for the force/torque
        if col < 3:
            R[:3, col] = F_conf
            # Cross-coupling: use same K for torque
            R[3:, col] = np.array(T_free) * np.array(K[:3])
        else:
            # For rotation columns, the eval sphere velocity is from rotation
            # The ratio still applies component-wise
            R[:3, col] = np.array(F_free) * np.array(K[:3])
            R[3:, col] = np.array(T_free) * np.array(K[:3])

        diag_val = float(R[col, col])
        logger.info("col %d: diag=%.4f, K=[%.3f,%.3f,%.3f]",
                    col, diag_val,
                    float(K[0]), float(K[1]), float(K[2]))

    # Results
    diag = [float(R[i, i]) for i in range(6)]
    errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]

    print(f"\n{'':>6}  {'Ratio':>10}  {'NN-BEM':>10}  {'error':>8}")
    print("-" * 44)
    for i in range(6):
        ok = "PASS" if errs[i] < 5 else "FAIL"
        print(f"{LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>6.1f}% [{ok}]")

    return R


def main():
    print(f"Device: {jax.devices()[0]}")

    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    node = make_node(48)
    print(f"Node: N={node._nx}^3, dx={node._dx:.4f}, spinup={node._spinup_steps}")

    # Part 1: Diagnostic — does the ratio stabilise?
    run_ratio_diagnostic(node, nn_diag)

    # Part 2: If ratio looks stable, run full R matrix
    # Try multiple step counts to find optimal
    for n_steps in [100, 200, 500]:
        for eval_idx in [0, 3]:  # R=1.15a and R=1.5a
            run_ratio_full_matrix(node, nn_diag, n_steps, eval_idx)


if __name__ == "__main__":
    main()
