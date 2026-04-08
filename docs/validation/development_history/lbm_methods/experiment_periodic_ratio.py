#!/usr/bin/env python3
"""Experiment: Periodic box twin-LBM ratio.

Tests whether K = u_walled / u_periodic stabilises and gives a
direction-independent wall correction factor.

The periodic LBM (no wall, periodic BCs on all faces) reaches steady
state via the periodic Stokeslet (Hasimoto 1959). Both walled and
periodic LBMs use the same IB body, so the IB transfer function cancels
in the ratio.
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


def make_periodic_step(node):
    """Build a periodic LBM step: collision + streaming, no walls, no open BCs.

    Pure periodic BCs: streaming wraps via modular indexing (default LBM).
    No pipe wall, no bounce-back, open_bc_axis=None → no open BCs.
    """
    try:
        from mime.nodes.environment.lbm.triton_kernels import (
            TRITON_AVAILABLE, lbm_full_step_triton, _get_d3q19_jax,
        )
        if TRITON_AVAILABLE:
            _get_d3q19_jax()  # populate cache before JIT
            _tau = node._tau
            _nw = node._no_wall
            _nm = node._no_missing_flat

            @jax.jit
            def periodic_step(f, force):
                # open_bc_axis=None → no open BCs inside Triton step
                f, u = lbm_full_step_triton(f, force, _tau, _nw, _nm, None)
                # No _apply_open_bc → fully periodic wrap
                return f, u

            return periodic_step
    except ImportError:
        pass

    from mime.nodes.environment.lbm.pallas_lbm import lbm_full_step_pallas

    @jax.jit
    def periodic_step(f, force):
        f, u = lbm_full_step_pallas(
            f, force, node._tau,
            node._no_wall, node._no_missing, None,
        )
        return f, u

    return periodic_step


def run_ratio_diagnostic(node, nn_diag):
    """Run walled and periodic LBMs, measure velocity ratio at eval spheres."""
    center = jnp.zeros(3)
    e = jnp.eye(3)
    N_b = node._N_body
    N = node._nx

    # Build periodic step function
    periodic_step = make_periodic_step(node)

    # Eval sphere stencils at multiple radii
    eval_radii_info = []
    R_factors = [1.15, 1.3, 1.5, 2.0, 2.5, 3.0]
    for r_idx, R_factor in enumerate(R_factors):
        if r_idx < len(node._eval_stencils_all):
            eval_radii_info.append((R_factor, node._eval_stencils_all[r_idx]))

    # Hasimoto correction for periodic box
    # F_periodic = 6*pi*mu*a*U * (1 - 2.837*(a/L) + 4.19*(a/L)^3 - ...)
    # where L is the box size
    L_phys = N * node._dx  # physical box size
    aL = A / L_phys
    hasimoto_factor = 1.0 - 2.837 * aL + 4.19 * aL**3
    print(f"Box size L = {L_phys:.3f}, a/L = {aL:.4f}")
    print(f"Hasimoto factor (1 - 2.837*a/L + 4.19*(a/L)^3) = {hasimoto_factor:.4f}")
    print(f"Periodic Stokes drag = {6*np.pi*MU*A*hasimoto_factor:.4f}")
    print()

    # Test both transverse (col 0) and axial (col 2)
    test_cols = [0, 2]
    step_samples = [50, 100, 150, 200, 300, 400, 500, 800, 1200]

    for col in test_cols:
        col_name = "F_x (transverse)" if col == 0 else "F_z (axial)"
        comp = col

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

        print(f"\n{'='*100}")
        print(f"PERIODIC RATIO DIAGNOSTIC: {col_name}")
        print(f"{'='*100}")
        print(f"  F_free (BEM) = {F_free_val:.4f}")
        print(f"  F_conf (NN-BEM) = {nn_diag[col]:.4f}")
        print(f"  K_true = F_conf/F_free = {K_true:.4f}")
        print(f"  Hasimoto correction: K_wall = K_raw * hasimoto = K_raw * {hasimoto_factor:.4f}")
        print()

        # IB force field
        force_field = node._spread_traction(traction)

        # Start from equilibrium
        f_walled = init_equilibrium(N, N, N)
        f_periodic = init_equilibrium(N, N, N)

        prev_sample = 0

        # Header
        header = f"{'steps':>6}"
        for R_factor, _ in eval_radii_info:
            header += f"  |{'R=%.2fa' % R_factor:^26}|"
        print(header)
        subhdr = f"{'':>6}"
        for _ in eval_radii_info:
            subhdr += f"  {'u_w':>8} {'u_p':>8} {'K_raw':>7} {'K_wall':>7}"
        print(subhdr)
        print("-" * (8 + 34 * len(eval_radii_info)))

        for target_steps in step_samples:
            steps_to_run = target_steps - prev_sample
            for s in range(steps_to_run):
                f_walled, u_walled = node._lbm_full_step(f_walled, force_field)
                f_periodic, u_periodic = periodic_step(f_periodic, force_field)
            prev_sample = target_steps

            row = f"{target_steps:6d}"
            for R_factor, es in eval_radii_info:
                u_w = interpolate_velocity(
                    u_walled, es['idx'], es['wts'],
                ) * node._dx / node._dt_lbm
                u_p = interpolate_velocity(
                    u_periodic, es['idx'], es['wts'],
                ) * node._dx / node._dt_lbm

                u_w_comp = float(jnp.mean(u_w[:, comp]))
                u_p_comp = float(jnp.mean(u_p[:, comp]))

                if abs(u_p_comp) > 1e-10:
                    K_raw = u_w_comp / u_p_comp
                    K_wall = K_raw * hasimoto_factor
                else:
                    K_raw = float('nan')
                    K_wall = float('nan')

                row += f"  {u_w_comp:>8.5f} {u_p_comp:>8.5f} {K_raw:>7.4f} {K_wall:>7.4f}"

            print(row, flush=True)

        # Summary at last step
        print(f"\nSummary at step {step_samples[-1]}:")
        print(f"  {'Radius':>10}  {'K_raw':>8}  {'K_wall':>8}  {'K_true':>8}  {'err%':>6}")
        print(f"  {'-'*48}")
        for R_factor, es in eval_radii_info:
            u_w = interpolate_velocity(
                u_walled, es['idx'], es['wts'],
            ) * node._dx / node._dt_lbm
            u_p = interpolate_velocity(
                u_periodic, es['idx'], es['wts'],
            ) * node._dx / node._dt_lbm

            u_w_comp = float(jnp.mean(u_w[:, comp]))
            u_p_comp = float(jnp.mean(u_p[:, comp]))

            if abs(u_p_comp) > 1e-10:
                K_raw = u_w_comp / u_p_comp
                K_wall = K_raw * hasimoto_factor
                err = abs(K_wall - K_true) / K_true * 100
                print(f"  R={R_factor:.2f}a    {K_raw:>8.4f}  {K_wall:>8.4f}  {K_true:>8.4f}  {err:>5.1f}%")
            else:
                print(f"  R={R_factor:.2f}a    {'nan':>8}  {'nan':>8}  {K_true:>8.4f}  {'---':>6}")

    # Check direction-independence
    print(f"\n{'='*100}")
    print("DIRECTION-INDEPENDENCE CHECK")
    print(f"{'='*100}")
    print("If K_raw is the same for F_x and F_z at the same radius,")
    print("the method is direction-independent and eliminates per-direction dispatch.")


def main():
    print(f"Device: {jax.devices()[0]}")

    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    node = make_node(48)
    print(f"Node: N={node._nx}^3, dx={node._dx:.4f}, spinup={node._spinup_steps}")

    run_ratio_diagnostic(node, nn_diag)


if __name__ == "__main__":
    main()
