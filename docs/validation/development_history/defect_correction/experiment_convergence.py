#!/usr/bin/env python3
"""Experiment 4: Resolution convergence test.

Tests per-direction dispatch at 48^3 and 64^3 with different eps_eval
values to check monotonic convergence.

Also tests: does eps_eval = c * dx (scaling with resolution) improve
convergence compared to fixed eps_body?
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


def make_perdir_config(node, eps_eval):
    """Per-direction dispatch: Lamb 1-pass for transverse, Richardson iterated for axial."""
    rich_fn = _richardson_fn(node, eps_eval)
    def get_col_config(col):
        if col < 2:
            return _lamb_fn(node, motion_axis=col, eps_eval=eps_eval), 1, 1.0
        elif col == 2:
            return rich_fn, node._max_defect_iter, 0.3
        else:
            return rich_fn, 2, 0.3
    return get_col_config


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

    resolutions = [48, 64]

    # Test 3 eps strategies:
    # 1. Fixed eps_body (resolution-independent, current default)
    # 2. Scaled eps_eval = 1.0 * dx (scales with resolution)
    # 3. Small fixed eps = 0.05 (best F_x from sweep)
    eps_strategies = {
        "eps_body": lambda node: node._epsilon,
        "eps=1.0*dx": lambda node: 1.0 * node._dx,
        "eps=0.05": lambda node: 0.05,
    }

    all_results = {}

    for strategy_name, get_eps in eps_strategies.items():
        print(f"\n{'='*78}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'='*78}")

        for N_target in resolutions:
            node = make_node(N_target)
            eps_eval = get_eps(node)
            print(f"\n--- N={N_target}^3, dx={node._dx:.4f}, eps_eval={eps_eval:.4f} ---",
                  flush=True)

            get_col_config = make_perdir_config(node, eps_eval)

            t0 = time.time()
            R = compute_R_matrix(node, get_col_config)
            elapsed = time.time() - t0

            diag = [float(R[i, i]) for i in range(6)]
            errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]
            all_results[(strategy_name, N_target)] = {"diag": diag, "errs": errs}

            for i, l in enumerate(LABELS):
                ok = "PASS" if errs[i] < 5 else "FAIL"
                print(f"  {l}: {diag[i]:>10.2f} (err {errs[i]:>5.1f}%) [{ok}]")
            print(f"  Time: {elapsed:.0f}s", flush=True)

    # ── Grand summary ──
    print(f"\n{'='*90}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*90}")

    for strategy_name in eps_strategies:
        print(f"\n--- {strategy_name} ---")
        header = f"{'':>6}"
        for N in resolutions:
            header += f"  {f'{N}^3':>14}"
        header += f"  {'NN-BEM':>10}"
        print(header)
        print("-" * 52)

        for i in range(6):
            row = f"{LABELS[i]:>6}"
            for N in resolutions:
                r = all_results[(strategy_name, N)]
                row += f"  {r['diag'][i]:>6.1f}({r['errs'][i]:>4.1f}%)"
            row += f"  {nn_diag[i]:>10.2f}"
            print(row)

        # Convergence check
        for col, name in [(0, "F_x"), (2, "F_z"), (5, "T_z")]:
            e48 = all_results[(strategy_name, 48)]["errs"][col]
            e64 = all_results[(strategy_name, 64)]["errs"][col]
            status = "CONVERGING" if e64 < e48 else ("FLAT" if abs(e64 - e48) < 0.3 else "ANTI")
            print(f"  {name}: {e48:.1f}% -> {e64:.1f}% [{status}]")

    # Best strategy
    print(f"\n{'='*90}")
    print("STRATEGY COMPARISON AT 64^3:")
    print(f"{'='*90}")
    header = f"{'':>6}"
    for s in eps_strategies:
        header += f"  {s:>14}"
    print(header)
    print("-" * 60)
    for i in range(6):
        row = f"{LABELS[i]:>6}"
        for s in eps_strategies:
            r = all_results[(s, 64)]
            row += f"  {r['errs'][i]:>12.1f}%"
        print(row)


if __name__ == "__main__":
    main()
