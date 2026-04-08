#!/usr/bin/env python3
"""Matched regularisation experiments.

Tests whether using eps_eval = c * dx (matched to the IB Peskin kernel's
effective hydrodynamic radius) in the eval-sphere BEM subtraction can
eliminate the per-direction wall correction dispatch.

Experiments 1-4, run sequentially on a single GPU.
"""

import os
import time
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

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Physics constants ───────────────────────────────────────────────

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM
EPS_NN = min(0.05, 0.02 * (R_CYL - A))
LABELS = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]


def compute_nn_bem_reference():
    """Compute NN-BEM reference resistance matrix."""
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
    """Create a DefectCorrectionFluidNode for a given resolution."""
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


# ── Wall correction helpers with configurable eps_eval ──────────────

def _richardson_with_eps(node, eps_eval):
    """Richardson wall correction using eps_eval for BEM eval."""
    def fn(u_lbm, traction):
        return wall_correction_richardson(
            u_lbm, traction, node._body_pts, node._body_wts,
            node._eval_data['stencils_close'], node._eval_data['d_vals_close'],
            eps_eval, node._mu, node._dx, node._dt_lbm,
        )
    return fn


def _lamb_with_eps(node, motion_axis, eps_eval):
    """Lamb wall correction using eps_eval for BEM eval."""
    def fn(u_lbm, traction):
        return wall_correction_lamb(
            u_lbm, traction, node._body_pts, node._body_wts,
            node._eval_data['stencils_all'], node._eval_data['R_phys_all'],
            node._body_radius,
            eps_eval, node._mu, node._dx, node._dt_lbm,
            motion_axis=motion_axis,
        )
    return fn


def _direct_with_eps(node, eps_eval):
    """Direct closest-radius wall correction using eps_eval for BEM eval."""
    def fn(u_lbm, traction):
        es = node._eval_stencils_close[0]  # R = 1.15a
        u_w = interpolate_velocity(
            u_lbm, es['idx'], es['wts'],
        ) * node._dx / node._dt_lbm
        u_fs = evaluate_velocity_field(
            es['pts_phys'], node._body_pts, node._body_wts,
            traction, eps_eval, node._mu,
        )
        return jnp.mean(u_w - u_fs, axis=0)
    return fn


# ── Generic R-matrix computation ───────────────────────────────────

def compute_R_matrix(node, get_col_config, columns=None):
    """Compute R matrix columns with a given wall correction strategy.

    get_col_config(col) -> (method_fn, n_iter, alpha)

    If columns is not None, only compute those columns and return
    a dict {col: (F, T)} instead of a full 6x6 matrix.
    """
    center = jnp.zeros(3)
    e = jnp.eye(3)
    N_b = node._N_body
    N = node._nx
    warmstart = 200

    if columns is None:
        columns = list(range(6))

    results = {}

    for col in columns:
        U = e[col] if col < 3 else jnp.zeros(3)
        omega = e[col - 3] if col >= 3 else jnp.zeros(3)

        r = node._body_pts - center
        u_body = U + jnp.cross(omega, r)

        # BEM free-space solve
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)

        # IB spread + single walled LBM (fixed spinup)
        force_field = node._spread_traction(traction)
        f_lbm = init_equilibrium(N, N, N)
        f_lbm, u_lbm = node._run_lbm_fixed(
            f_lbm, force_field, node._spinup_steps,
        )

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

        F, T = compute_force_torque(
            node._body_pts, node._body_wts, traction, center,
        )
        results[col] = (np.array(F), np.array(T))
        logger.info("col %d: F=[%.2f,%.2f,%.2f] T=[%.2f,%.2f,%.2f]",
                    col, *[float(x) for x in F], *[float(x) for x in T])

    return results


def diag_and_errors(results, nn_diag):
    """Extract diagonal entries and compute % errors."""
    diag = {}
    errs = {}
    for col, (F, T) in results.items():
        val = float(F[col]) if col < 3 else float(T[col - 3])
        err = abs(val - nn_diag[col]) / abs(nn_diag[col]) * 100
        diag[col] = val
        errs[col] = err
    return diag, errs


# ────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Does eps_eval = 1.255 * dx work out of the box?
# ────────────────────────────────────────────────────────────────────

def experiment_1(node, nn_diag):
    print("\n" + "=" * 78)
    print("EXPERIMENT 1: eps_eval = 1.255 * dx with per-direction dispatch")
    print("=" * 78)

    eps_eval = 1.255 * node._dx
    eps_body = node._epsilon
    print(f"  eps_body = {eps_body:.4f} (body BEM, unchanged)")
    print(f"  eps_eval = {eps_eval:.4f} (eval-sphere BEM)")
    print(f"  dx = {node._dx:.4f}")
    print(f"  ratio eps_eval/dx = {eps_eval/node._dx:.3f}")
    print()

    rich_fn = _richardson_with_eps(node, eps_eval)

    def get_col_config(col):
        if col < 2:
            return _lamb_with_eps(node, motion_axis=col, eps_eval=eps_eval), 1, 1.0
        elif col == 2:
            return rich_fn, node._max_defect_iter, 0.3
        else:
            return rich_fn, 2, 0.3

    t0 = time.time()
    results = compute_R_matrix(node, get_col_config)
    elapsed = time.time() - t0

    diag, errs = diag_and_errors(results, nn_diag)

    print(f"\n{'':>6}  {'value':>10}  {'NN-BEM':>10}  {'error':>8}")
    print("-" * 44)
    all_pass = True
    for i in range(6):
        ok = "PASS" if errs[i] < 5 else ("WARN" if errs[i] < 10 else "FAIL")
        if errs[i] >= 5:
            all_pass = False
        print(f"{LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>6.1f}% [{ok}]")
    print(f"\nTime: {elapsed:.0f}s | {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print()
    return diag, errs


# ────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Sweep eps_eval
# ────────────────────────────────────────────────────────────────────

def experiment_2(node, nn_diag):
    print("\n" + "=" * 78)
    print("EXPERIMENT 2: Sweep eps_eval (F_x, F_z, T_z columns only)")
    print("=" * 78)

    eps_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40,
                  0.50, 0.60, 0.80, 1.0 * node._dx, 1.255 * node._dx, 1.5 * node._dx]
    eps_values = sorted(set(eps_values))  # deduplicate
    test_cols = [0, 2, 5]  # F_x, F_z, T_z

    print(f"  dx = {node._dx:.4f}")
    print(f"  eps_body = {node._epsilon:.4f}")
    print(f"  Testing {len(eps_values)} eps_eval values for cols {test_cols}")
    print()

    # Use per-direction dispatch for consistent comparison
    results_table = []

    for eps_eval in eps_values:
        rich_fn = _richardson_with_eps(node, eps_eval)

        def get_col_config(col, _eps=eps_eval):
            if col < 2:
                return _lamb_with_eps(node, motion_axis=col, eps_eval=_eps), 1, 1.0
            elif col == 2:
                return rich_fn, node._max_defect_iter, 0.3
            else:
                return rich_fn, 2, 0.3

        t0 = time.time()
        results = compute_R_matrix(node, get_col_config, columns=test_cols)
        elapsed = time.time() - t0

        diag, errs = diag_and_errors(results, nn_diag)
        results_table.append((eps_eval, diag, errs, elapsed))
        print(f"  eps={eps_eval:.4f}: F_x={diag[0]:.2f}({errs[0]:.1f}%) "
              f"F_z={diag[2]:.2f}({errs[2]:.1f}%) T_z={diag[5]:.2f}({errs[5]:.1f}%) "
              f"[{elapsed:.0f}s]", flush=True)

    # Summary table
    print(f"\n{'eps_eval':>10}  {'eps/dx':>8}  {'F_x':>8}  {'err%':>6}  {'F_z':>8}  {'err%':>6}  {'T_z':>8}  {'err%':>6}")
    print("-" * 78)
    best_combined = None
    best_combined_err = 999
    for eps_eval, diag, errs, _ in results_table:
        combined = errs[0] + errs[2]  # minimize F_x + F_z error
        if combined < best_combined_err:
            best_combined_err = combined
            best_combined = eps_eval
        print(f"{eps_eval:>10.4f}  {eps_eval/node._dx:>8.3f}  "
              f"{diag[0]:>8.2f}  {errs[0]:>5.1f}%  "
              f"{diag[2]:>8.2f}  {errs[2]:>5.1f}%  "
              f"{diag[5]:>8.2f}  {errs[5]:>5.1f}%")
    print(f"\nBest combined (F_x+F_z): eps_eval = {best_combined:.4f} "
          f"(eps/dx = {best_combined/node._dx:.3f})")
    print(f"NN-BEM reference: F_x={nn_diag[0]:.2f}, F_z={nn_diag[2]:.2f}, T_z={nn_diag[5]:.2f}")

    return best_combined, results_table


# ────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Single-method test at optimal eps_eval
# ────────────────────────────────────────────────────────────────────

def experiment_3(node, nn_diag, eps_eval_opt):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 3: Single-method test at eps_eval = {eps_eval_opt:.4f}")
    print("=" * 78)

    # (a) Direct closest-radius, iterated alpha=0.3, 25 iter
    print("\n--- (a) Direct closest-radius, iterated alpha=0.3 ---")
    direct_fn = _direct_with_eps(node, eps_eval_opt)

    def get_direct(col):
        if col < 3:
            return direct_fn, node._max_defect_iter, 0.3
        else:
            return direct_fn, 2, 0.3

    t0 = time.time()
    res_a = compute_R_matrix(node, get_direct)
    elapsed_a = time.time() - t0
    diag_a, errs_a = diag_and_errors(res_a, nn_diag)

    # (b) Richardson (2-radius), iterated alpha=0.3, 25 iter
    print("\n--- (b) Richardson (2-radius), iterated alpha=0.3 ---")
    rich_fn = _richardson_with_eps(node, eps_eval_opt)

    def get_rich(col):
        if col < 3:
            return rich_fn, node._max_defect_iter, 0.3
        else:
            return rich_fn, 2, 0.3

    t0 = time.time()
    res_b = compute_R_matrix(node, get_rich)
    elapsed_b = time.time() - t0
    diag_b, errs_b = diag_and_errors(res_b, nn_diag)

    # (c) Lamb (8-radius) 1-pass alpha=1.0 for all translation
    print("\n--- (c) Lamb 1-pass alpha=1.0, all translation ---")

    def get_lamb(col):
        if col < 3:
            return _lamb_with_eps(node, motion_axis=col, eps_eval=eps_eval_opt), 1, 1.0
        else:
            return _richardson_with_eps(node, eps_eval_opt), 2, 0.3

    t0 = time.time()
    res_c = compute_R_matrix(node, get_lamb)
    elapsed_c = time.time() - t0
    diag_c, errs_c = diag_and_errors(res_c, nn_diag)

    # Summary
    print(f"\n{'':>6}  {'Direct':>14}  {'Richardson':>14}  {'Lamb-1pass':>14}  {'NN-BEM':>10}")
    print("-" * 70)
    for i in range(6):
        row = f"{LABELS[i]:>6}"
        row += f"  {diag_a[i]:>6.1f}({errs_a[i]:>4.1f}%)"
        row += f"  {diag_b[i]:>6.1f}({errs_b[i]:>4.1f}%)"
        row += f"  {diag_c[i]:>6.1f}({errs_c[i]:>4.1f}%)"
        row += f"  {nn_diag[i]:>10.2f}"
        print(row)
    print(f"\nDirect: {elapsed_a:.0f}s | Richardson: {elapsed_b:.0f}s | Lamb: {elapsed_c:.0f}s")

    # Determine best single method
    for name, errs in [("Direct", errs_a), ("Richardson", errs_b), ("Lamb", errs_c)]:
        all_ok = all(errs[i] < 5 for i in range(6))
        print(f"  {name}: {'ALL PASS' if all_ok else 'FAIL'}")

    return {
        "direct": (diag_a, errs_a),
        "richardson": (diag_b, errs_b),
        "lamb": (diag_c, errs_c),
    }


# ────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Resolution convergence with matched eps
# ────────────────────────────────────────────────────────────────────

def experiment_4(nn_diag, eps_factor, method_name, make_config_fn):
    """Test convergence at 48 and 64 with eps_eval = eps_factor * dx.

    make_config_fn(node, eps_eval) -> get_col_config function
    """
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 4: Resolution convergence with eps_eval = {eps_factor:.3f} * dx")
    print(f"              Method: {method_name}")
    print("=" * 78)

    resolutions = [48, 64]
    all_results = {}

    for N_target in resolutions:
        node = make_node(N_target)
        eps_eval = eps_factor * node._dx
        print(f"\n--- N={N_target}^3, dx={node._dx:.4f}, eps_eval={eps_eval:.4f} ---")

        get_col_config = make_config_fn(node, eps_eval)

        t0 = time.time()
        results = compute_R_matrix(node, get_col_config)
        elapsed = time.time() - t0

        diag, errs = diag_and_errors(results, nn_diag)
        all_results[N_target] = (diag, errs)
        print(f"  Time: {elapsed:.0f}s")

    # Summary
    print(f"\n{'':>6}", end="")
    for N in resolutions:
        print(f"  {f'{N}^3':>14}", end="")
    print(f"  {'NN-BEM':>10}")
    print("-" * 52)
    for i in range(6):
        row = f"{LABELS[i]:>6}"
        for N in resolutions:
            d, e = all_results[N]
            row += f"  {d[i]:>6.1f}({e[i]:>4.1f}%)"
        row += f"  {nn_diag[i]:>10.2f}"
        print(row)

    # Convergence check
    print()
    for col, name in [(0, "F_x"), (2, "F_z"), (5, "T_z")]:
        e48 = all_results[48][1][col]
        e64 = all_results[64][1][col]
        status = "CONVERGING" if e64 <= e48 else "ANTI-CONVERGING"
        print(f"  {name}: {e48:.1f}% -> {e64:.1f}% [{status}]")

    return all_results


# ────────────────────────────────────────────────────────────────────
# EXPERIMENT 5: Free-space mismatch diagnostic per eval radius
# ────────────────────────────────────────────────────────────────────

def experiment_5(node, nn_diag):
    """Measure IB-BEM mismatch at each eval radius for different eps_eval.

    Runs a FREE-SPACE LBM (no walls) and compares u_IB to u_BEM(eps_eval)
    at each eval sphere. Finds the eps_eval that minimizes mismatch per radius.
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 5: Free-space IB-BEM mismatch per eval radius")
    print("=" * 78)

    center = jnp.zeros(3)
    N_b = node._N_body
    N = node._nx

    # Unit x-translation: BEM solve
    e = jnp.eye(3)
    u_body = jnp.broadcast_to(e[0], (N_b, 3))
    traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)

    # IB spread + NO-WALL LBM (free space)
    force_field = node._spread_traction(traction)
    f_lbm = init_equilibrium(N, N, N)
    # Run for spinup steps with no walls
    # We need to use the node's LBM step but WITHOUT walls
    # Use the _lbm_step_core with no_wall, no_missing
    for step in range(node._spinup_steps):
        f_lbm, u_lbm = DefectCorrectionFluidNode._lbm_step_core(
            f_lbm, force_field, node._tau,
            node._no_wall, node._no_missing, node._open_bc_axis,
        )

    print(f"  Ran {node._spinup_steps} free-space LBM steps")

    # Now check mismatch at each eval radius for different eps_eval
    eps_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                  0.60, 0.80, 1.0 * node._dx, 1.255 * node._dx, 1.5 * node._dx,
                  2.0 * node._dx]
    eps_values = sorted(set(eps_values))

    R_factors = [1.15, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    eval_stencils = node._eval_stencils_all

    print(f"\n  {'eps_eval':>10}", end="")
    for rf in R_factors[:len(eval_stencils)]:
        print(f"  {'R=%.2fa' % rf:>10}", end="")
    print()
    print("-" * (12 + 12 * min(len(R_factors), len(eval_stencils))))

    best_per_radius = {}

    for eps_eval in eps_values:
        row = f"  {eps_eval:>10.4f}"
        for r_idx, es in enumerate(eval_stencils):
            if r_idx >= len(R_factors):
                break
            # IB-LBM velocity at this eval sphere
            u_ib = interpolate_velocity(
                u_lbm, es['idx'], es['wts'],
            ) * node._dx / node._dt_lbm

            # BEM velocity with this eps_eval
            u_bem = evaluate_velocity_field(
                es['pts_phys'], node._body_pts, node._body_wts,
                traction, eps_eval, node._mu,
            )

            # Mismatch: mean relative difference in x-component
            u_ib_mean_x = float(jnp.mean(u_ib[:, 0]))
            u_bem_mean_x = float(jnp.mean(u_bem[:, 0]))
            ratio = u_ib_mean_x / (u_bem_mean_x + 1e-30)
            mismatch_pct = abs(ratio - 1.0) * 100

            row += f"  {mismatch_pct:>8.1f}%%"

            rf = R_factors[r_idx]
            if rf not in best_per_radius or mismatch_pct < best_per_radius[rf][1]:
                best_per_radius[rf] = (eps_eval, mismatch_pct)

        print(row, flush=True)

    print(f"\nBest eps_eval per radius:")
    for rf in R_factors[:len(eval_stencils)]:
        eps_best, mm = best_per_radius[rf]
        print(f"  R={rf:.2f}a: eps_eval={eps_best:.4f} (eps/dx={eps_best/node._dx:.3f}), mismatch={mm:.1f}%")


# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="1,2,3,4",
                        help="Comma-separated experiment numbers to run")
    parser.add_argument("--resolution", type=int, default=48,
                        help="LBM resolution for experiments 1-3")
    args = parser.parse_args()
    experiments = [int(x) for x in args.experiments.split(",")]

    print(f"Device: {jax.devices()[0]}")

    # NN-BEM reference
    print("Computing NN-BEM reference...", flush=True)
    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    # Node for experiments 1-3
    node = make_node(args.resolution)
    print(f"Node: N={node._nx}^3, dx={node._dx:.4f}, eps_body={node._epsilon:.4f}")
    print(f"      spinup={node._spinup_steps}, warmstart={node._warmstart_steps}")

    # Baseline: current eps_body
    print(f"\nBaseline: eps_body = {node._epsilon:.4f} (= mean_spacing/2)")
    print(f"Predicted: eps_eval = 1.255 * dx = {1.255 * node._dx:.4f}")

    eps_eval_opt = None

    if 1 in experiments:
        experiment_1(node, nn_diag)

    if 2 in experiments:
        eps_eval_opt, sweep_results = experiment_2(node, nn_diag)

    if 3 in experiments:
        if eps_eval_opt is None:
            eps_eval_opt = 1.255 * node._dx
            print(f"\n(No sweep result, using default eps_eval = {eps_eval_opt:.4f})")
        experiment_3(node, nn_diag, eps_eval_opt)

    if 4 in experiments:
        if eps_eval_opt is None:
            eps_factor = 1.255
        else:
            eps_factor = eps_eval_opt / node._dx
            print(f"\n(Using eps_factor = {eps_factor:.3f} from sweep)")

        # Use per-direction dispatch (proven) with matched eps for convergence test
        def make_perdir(node, eps_eval):
            rich_fn = _richardson_with_eps(node, eps_eval)
            def get_col_config(col):
                if col < 2:
                    return _lamb_with_eps(node, motion_axis=col, eps_eval=eps_eval), 1, 1.0
                elif col == 2:
                    return rich_fn, node._max_defect_iter, 0.3
                else:
                    return rich_fn, 2, 0.3
            return get_col_config

        experiment_4(nn_diag, eps_factor, "per-direction (Lamb+Richardson)", make_perdir)

    if 5 in experiments:
        experiment_5(node, nn_diag)


if __name__ == "__main__":
    main()
