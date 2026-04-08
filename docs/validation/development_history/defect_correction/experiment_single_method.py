#!/usr/bin/env python3
"""Experiment 3: Single-method test at multiple eps_eval values.

Tests whether ANY single wall correction method (no per-direction dispatch)
can achieve <5% error on all 6 R-matrix entries.

Tests:
  (a) Direct closest-radius at R=1.15a, iterated α=0.3
  (b) Richardson (2-radius), iterated α=0.3
  (c) Lamb 1-pass α=1.0 for all translation

At eps_eval values: eps_body (baseline), 0.05, 0.15
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
from mime.nodes.environment.defect_correction.wall_correction import (
    wall_correction_richardson,
    wall_correction_lamb,
)
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field
from mime.nodes.environment.lbm.immersed_boundary import interpolate_velocity
from mime.nodes.environment.lbm.d3q19 import init_equilibrium

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


def _richardson_fn(node, eps_eval):
    def fn(u_lbm, traction):
        return wall_correction_richardson(
            u_lbm, traction, node._body_pts, node._body_wts,
            node._eval_data['stencils_close'], node._eval_data['d_vals_close'],
            eps_eval, node._mu, node._dx, node._dt_lbm,
        )
    return fn


def _lamb_fn(node, motion_axis, eps_eval):
    def fn(u_lbm, traction):
        return wall_correction_lamb(
            u_lbm, traction, node._body_pts, node._body_wts,
            node._eval_data['stencils_all'], node._eval_data['R_phys_all'],
            node._body_radius,
            eps_eval, node._mu, node._dx, node._dt_lbm,
            motion_axis=motion_axis,
        )
    return fn


def _direct_fn(node, eps_eval):
    def fn(u_lbm, traction):
        es = node._eval_stencils_close[0]
        u_w = interpolate_velocity(u_lbm, es['idx'], es['wts']) * node._dx / node._dt_lbm
        u_fs = evaluate_velocity_field(
            es['pts_phys'], node._body_pts, node._body_wts,
            traction, eps_eval, node._mu,
        )
        return jnp.mean(u_w - u_fs, axis=0)
    return fn


def compute_R_matrix(node, get_col_config):
    center = jnp.zeros(3)
    e = jnp.eye(3)
    N_b = node._N_body
    N = node._nx
    warmstart = 200

    R = np.zeros((6, 6))

    for col in range(6):
        U = e[col] if col < 3 else jnp.zeros(3)
        omega = e[col - 3] if col >= 3 else jnp.zeros(3)

        r = node._body_pts - center
        u_body = U + jnp.cross(omega, r)
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)

        force_field = node._spread_traction(traction)
        f_lbm = init_equilibrium(N, N, N)
        f_lbm, u_lbm = node._run_lbm_fixed(f_lbm, force_field, node._spinup_steps)

        method_fn, n_iter, alpha_col = get_col_config(col)
        prev_drag = 0.0

        for iteration in range(n_iter):
            delta_u = method_fn(u_lbm, traction)
            delta_u_body = jnp.broadcast_to(delta_u, (N_b, 3))
            traction_new = node._bem_solve(
                (u_body - delta_u_body).ravel()
            ).reshape(N_b, 3)
            traction = (1 - alpha_col) * traction + alpha_col * traction_new
            force_field = node._spread_traction(traction)
            f_lbm, u_lbm = node._run_lbm_fixed(f_lbm, force_field, warmstart)

            F_iter, T_iter = compute_force_torque(
                node._body_pts, node._body_wts, traction, center,
            )
            drag_diag = float(F_iter[col]) if col < 3 else float(T_iter[col - 3])
            rel_change = abs(drag_diag - prev_drag) / (abs(drag_diag) + 1e-30)
            prev_drag = drag_diag
            if iteration > 0 and rel_change < node._tol:
                logger.info("  col %d converged iter %d: %.4f", col, iteration + 1, drag_diag)
                break

        F, T = compute_force_torque(node._body_pts, node._body_wts, traction, center)
        R[:3, col] = np.array(F)
        R[3:, col] = np.array(T)
        logger.info("col %d: F=[%.2f,%.2f,%.2f] T=[%.2f,%.2f,%.2f]",
                    col, *[float(x) for x in F], *[float(x) for x in T])

    return R


def main():
    print(f"Device: {jax.devices()[0]}")

    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    node = make_node(48)
    eps_body = node._epsilon
    print(f"Node: N={node._nx}^3, dx={node._dx:.4f}, eps_body={eps_body:.4f}")

    eps_values = [eps_body, 0.05, 0.15]

    methods = {
        "Direct": lambda node, eps: (
            lambda col: (_direct_fn(node, eps), node._max_defect_iter if col < 3 else 2, 0.3)
        ),
        "Richardson": lambda node, eps: (
            lambda col: (_richardson_fn(node, eps), node._max_defect_iter if col < 3 else 2, 0.3)
        ),
        "Lamb-1pass": lambda node, eps: (
            lambda col: (
                _lamb_fn(node, motion_axis=col, eps_eval=eps) if col < 3
                else _richardson_fn(node, eps),
                1 if col < 3 else 2,
                1.0 if col < 3 else 0.3,
            )
        ),
        "Per-dir": lambda node, eps: (
            lambda col: (
                _lamb_fn(node, motion_axis=col, eps_eval=eps) if col < 2
                else _richardson_fn(node, eps) if col == 2
                else _richardson_fn(node, eps),
                1 if col < 2 else (node._max_defect_iter if col == 2 else 2),
                1.0 if col < 2 else 0.3,
            )
        ),
    }

    all_results = {}

    for eps_eval in eps_values:
        print(f"\n{'='*78}")
        print(f"eps_eval = {eps_eval:.4f} (eps/dx = {eps_eval/node._dx:.3f})")
        print(f"{'='*78}")

        for method_name, make_config in methods.items():
            print(f"\n--- {method_name} ---", flush=True)
            get_col_config = make_config(node, eps_eval)

            t0 = time.time()
            R = compute_R_matrix(node, get_col_config)
            elapsed = time.time() - t0

            diag = [float(R[i, i]) for i in range(6)]
            errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]
            all_pass = all(e < 5 for e in errs)

            key = (eps_eval, method_name)
            all_results[key] = {"diag": diag, "errs": errs, "pass": all_pass}

            print(f"  Time: {elapsed:.0f}s | {'ALL PASS' if all_pass else 'FAIL'}")

    # ── Grand summary ──
    print(f"\n{'='*90}")
    print("GRAND SUMMARY: method × eps_eval → max(F_x_err, F_z_err)")
    print(f"{'='*90}")
    header = f"{'Method':>12}"
    for eps_eval in eps_values:
        header += f"  {'eps=%.4f' % eps_eval:>22}"
    print(header)
    print("-" * 90)

    for method_name in methods:
        row = f"{method_name:>12}"
        for eps_eval in eps_values:
            key = (eps_eval, method_name)
            r = all_results[key]
            fx_err = r['errs'][0]
            fz_err = r['errs'][2]
            status = "PASS" if r['pass'] else "FAIL"
            row += f"  Fx={fx_err:>4.1f}% Fz={fz_err:>4.1f}% [{status}]"
        print(row)

    # Detailed table for each eps
    for eps_eval in eps_values:
        print(f"\n--- eps_eval = {eps_eval:.4f} (eps/dx = {eps_eval/node._dx:.3f}) ---")
        header2 = f"{'':>6}"
        for method_name in methods:
            header2 += f"  {method_name:>14}"
        header2 += f"  {'NN-BEM':>10}"
        print(header2)
        print("-" * 90)
        for i in range(6):
            row = f"{LABELS[i]:>6}"
            for method_name in methods:
                key = (eps_eval, method_name)
                r = all_results[key]
                row += f"  {r['diag'][i]:>6.1f}({r['errs'][i]:>4.1f}%)"
            row += f"  {nn_diag[i]:>10.2f}"
            print(row)


if __name__ == "__main__":
    main()
