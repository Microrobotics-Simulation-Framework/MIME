#!/usr/bin/env python3
"""Validate 3 wall correction configurations at 48³ and 64³.

Config 1 (per-direction): Lamb for transverse (0,1), Richardson for axial (2)
Config 2 (all-Richardson): Richardson for all translation columns (0,1,2)
Config 3 (all-Lamb): Lamb for all translation columns (0,1,2)

All configs use:
  - Fixed spinup steps (physics-based: wall round-trip time)
  - Fixed warmstart = 100 steps
  - Rotation columns (3,4,5): single pass, no extrapolation needed
  - Single walled LBM (no twin)
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

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_R_matrix(node, config_name, get_col_config):
    """Compute 6x6 R matrix with a given wall correction strategy.

    get_col_config(col) -> (method_fn, n_iter, alpha)
      method_fn(u_lbm, traction) -> (3,) delta_u
      n_iter: max defect correction iterations for this column
      alpha: under-relaxation (1.0 for one-pass methods like Lamb)
    """
    center = jnp.zeros(3)
    e = jnp.eye(3)
    N_b = node._N_body
    N = node._nx
    warmstart = 200  # match original recipe

    R = np.zeros((6, 6))

    for col in range(6):
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
                logger.info("  [%s] col %d converged iter %d: %.4f",
                            config_name, col, iteration + 1, drag_diag)
                break

        F, T = compute_force_torque(
            node._body_pts, node._body_wts, traction, center,
        )
        R[:3, col] = np.array(F)
        R[3:, col] = np.array(T)
        logger.info("[%s] col %d: F=[%.2f,%.2f,%.2f] T=[%.2f,%.2f,%.2f]",
                    config_name, col, *[float(x) for x in F], *[float(x) for x in T])

    return R


def _make_richardson_fn(node):
    """Create a Richardson wall correction function."""
    def fn(u_lbm, traction):
        return wall_correction_richardson(
            u_lbm, traction, node._body_pts, node._body_wts,
            node._eval_data['stencils_close'], node._eval_data['d_vals_close'],
            node._epsilon, node._mu, node._dx, node._dt_lbm,
        )
    return fn


def _make_lamb_fn(node, motion_axis):
    """Create a Lamb wall correction function for a given motion axis."""
    def fn(u_lbm, traction):
        return wall_correction_lamb(
            u_lbm, traction, node._body_pts, node._body_wts,
            node._eval_data['stencils_all'], node._eval_data['R_phys_all'],
            node._body_radius,
            node._epsilon, node._mu, node._dx, node._dt_lbm,
            motion_axis=motion_axis,
        )
    return fn


def _make_direct_fn(node):
    """Create a direct closest-radius wall correction function."""
    def fn(u_lbm, traction):
        return node._compute_wall_correction_single(u_lbm, traction)
    return fn


def make_per_direction(node):
    """Per-direction: 1-pass Lamb(α=1) for transverse, Richardson(α=0.3) for axial.
    Matches the original e0f3d0a recipe exactly."""
    rich_fn = _make_richardson_fn(node)
    direct_fn = _make_direct_fn(node)
    def get_col_config(col):
        if col < 2:
            # Transverse: 1-pass Lamb, α=1.0 (no relaxation)
            return _make_lamb_fn(node, motion_axis=col), 1, 1.0
        elif col == 2:
            # Axial: inline Richardson, iterated with α=0.3
            return rich_fn, node._max_defect_iter, 0.3
        else:
            # Rotation: 2 passes Richardson
            return rich_fn, 2, 0.3
    return get_col_config


def make_all_richardson(node):
    """All-Richardson: inline Richardson(α=0.3) for all translation columns."""
    rich_fn = _make_richardson_fn(node)
    def get_col_config(col):
        if col < 3:
            return rich_fn, node._max_defect_iter, 0.3
        else:
            return rich_fn, 2, 0.3
    return get_col_config


def make_all_lamb(node):
    """All-Lamb: 1-pass Lamb(α=1.0) for all translation columns."""
    rich_fn = _make_richardson_fn(node)
    def get_col_config(col):
        if col < 3:
            # 1-pass Lamb, α=1.0
            return _make_lamb_fn(node, motion_axis=col), 1, 1.0
        else:
            return rich_fn, 2, 0.3
    return get_col_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", default="48")
    args = parser.parse_args()
    resolutions = [int(n) for n in args.resolutions.split(",")]

    print(f"Device: {jax.devices()[0]}")

    a = 1.0; mu = 1.0; rho = 1.0; lam = 0.3
    R_cyl = a / lam
    eps_nn = min(0.05, 0.02 * (R_cyl - a))

    labels = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]

    # NN-BEM reference
    print("Computing NN-BEM reference...", flush=True)
    bc = sphere_surface_mesh(radius=a, n_refine=2)
    bf = sphere_surface_mesh(radius=a, n_refine=4)
    cl = 12.0 * R_cyl
    wc = cylinder_surface_mesh(radius=R_cyl, length=cl, n_circ=48, n_axial=16,
                                cluster_center=True)
    wf = cylinder_surface_mesh(radius=R_cyl, length=cl, n_circ=192, n_axial=64,
                                cluster_center=True)
    Rn = compute_nn_confined_resistance_matrix(
        jnp.array(bc.points), jnp.array(bc.weights),
        jnp.array(bf.points), jnp.array(bf.weights),
        jnp.array(wc.points), jnp.array(wc.weights),
        jnp.array(wf.points), jnp.array(wf.weights),
        jnp.zeros(3), eps_nn, mu,
    )
    nn_diag = [float(Rn[i, i]) for i in range(6)]
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n",
          flush=True)

    configs = [
        ("per-dir", make_per_direction),
        ("all-Rich", make_all_richardson),
        ("all-Lamb", make_all_lamb),
    ]

    for N_target in resolutions:
        dx = 2 * R_cyl / (N_target * 0.8)
        body = sphere_surface_mesh(radius=a, n_refine=2)

        node = DefectCorrectionFluidNode(
            "dc", timestep=0.001, mu=mu, rho=rho,
            body_mesh=body, body_radius=a,
            vessel_radius=R_cyl, dx=dx,
            open_bc_axis=2,
            max_defect_iter=25,
            alpha=0.3,
        )
        print(f"\n=== N={N_target}³ (spinup={node._spinup_steps}, warm={node._warmstart_steps}) ===\n",
              flush=True)

        results = {}
        for config_name, make_wc in configs:
            print(f"--- {config_name} ---", flush=True)
            t0 = time.time()
            R = compute_R_matrix(node, config_name, make_wc(node))
            elapsed = time.time() - t0

            diag = [float(R[i, i]) for i in range(6)]
            errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100
                    for i in range(6)]
            results[config_name] = {"diag": diag, "errors": errs}
            print(f"  Time: {elapsed:.0f}s\n", flush=True)

        # Summary table
        print(f"\n{'='*78}", flush=True)
        print(f"N={N_target}³  WALL CORRECTION COMPARISON", flush=True)
        print(f"{'='*78}", flush=True)
        header = f"{'':>6}"
        for cname, _ in configs:
            header += f"  {cname:>14}"
        header += f"  {'NN-BEM':>10}"
        print(header, flush=True)
        print("-" * 78, flush=True)

        all_pass = {cname: True for cname, _ in configs}
        for i, l in enumerate(labels):
            row = f"{l:>6}"
            for cname, _ in configs:
                val = results[cname]["diag"][i]
                err = results[cname]["errors"][i]
                if err >= 5:
                    all_pass[cname] = False
                row += f"  {val:>6.1f}({err:>4.1f}%)"
            row += f"  {nn_diag[i]:>10.2f}"
            print(row, flush=True)

        print("-" * 78, flush=True)
        for cname, _ in configs:
            status = "ALL PASS" if all_pass[cname] else "FAIL"
            print(f"  {cname}: {status}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
