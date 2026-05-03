"""Step 3 — held-out evaluation of the trained GNN flux corrector.

Loads the val_A coarse state, runs (a) without correction and
(b) with the trained correction for the same number of PISO steps,
and compares both against the fine-mesh reference.

Reports: drag K_FVM (uncorrected, corrected, fine reference) and
velocity-field MSE relative to the downsampled fine reference.
"""
from __future__ import annotations
import json
import pickle
import time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_pipe_mesh, make_poiseuille_lift, make_poiseuille_p_lift,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, make_piso_step
from mime.nodes.environment.fvm.ibm import IBMBody, surface_integral_force
from mime.nodes.environment.fvm.sdf import sphere_sdf

from step2_train import (
    build_mesh_and_step, load_initial_state, load_fine_ref,
    correction_body_force, DATA_DIR,
)


def happel_brenner(lam):
    return 1.0/(1.0-2.10443*lam+2.08877*lam**3-0.94813*lam**5
                -1.372*lam**6+3.87*lam**8-4.19*lam**10)


def run_uncorrected(mesh, bcs, piso_cfg, bodies, L_lift, dt, init_state,
                     n_steps):
    """Plain PISO from the coarse-init state, no GNN."""
    step = make_piso_step(mesh, bcs, piso_cfg,
                           body_force_fn=None,
                           ibm_bodies=bodies, lifting=L_lift)
    @jax.jit
    def go(s):
        return jax.lax.fori_loop(0, n_steps, lambda _, x: step(x, dt), s)
    return go(init_state)


def run_corrected(mesh, bcs, piso_cfg, bodies, L_lift, dt, init_state,
                   n_steps, corrector):
    """PISO with GNN correction injected as a body force at each step."""
    rho = piso_cfg.rho

    @jax.jit
    def go(s):
        def step_fn(_, state):
            f_gnn = correction_body_force(
                corrector, state["u"], state["p"], mesh, rho,
            )
            body_force_fn = lambda t: f_gnn
            step = make_piso_step(
                mesh, bcs, piso_cfg,
                body_force_fn=body_force_fn,
                ibm_bodies=bodies, lifting=L_lift,
            )
            return step(state, dt)
        return jax.lax.fori_loop(0, n_steps, step_fn, s)
    return go(init_state)


def drag_K(state, mesh, L_lift, val_cfg):
    """surface_integral_force at the trained shell + matched ref."""
    r_b = val_cfg["r_b"]; R_pipe = val_cfg["R_pipe"]
    mu  = val_cfg["mu"]
    U_dc = val_cfg["U_dc"]
    L_pipe_actual = mesh.cartesian_shape[2] * mesh.cartesian_spacing[2]
    sphere_centre = jnp.array([0.0, 0.0, L_pipe_actual / 2],
                               dtype=mesh.V.dtype)
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_b)
    u_phys = state["u"] + L_lift.u_lift_static
    p_lift_fn = make_poiseuille_p_lift(mu=mu, U_mean=U_dc, pipe_radius=R_pipe)
    F_vec, _ = surface_integral_force(
        u_phys, state["p"], mesh, sphere_sdf_fn,
        mu=mu, dx=mesh.cartesian_spacing[0],
        shell_inner=0.5, shell_outer=2.5,
        ref_point=sphere_centre, p_lift_fn=p_lift_fn, pipe_axis=2,
    )
    F_z = float(F_vec[2])
    F_uncon = 6.0 * np.pi * mu * r_b * (2 * U_dc)
    return F_z, F_z / F_uncon


def main():
    print("=" * 78)
    print("Step 3 — held-out validation (val_A: λ=0.30, Re=150)")
    print("=" * 78)

    with open(DATA_DIR / "manifest.json") as f:
        manifest = json.load(f)
    val_cfg = next(m for m in manifest if m["label"] == "val_A")
    K_h = happel_brenner(val_cfg["lambda_"])
    print(f"  λ={val_cfg['lambda_']}  Re={val_cfg['Re']}  "
          f"K_Happel={K_h:.3f}")
    print(f"  fine drag (reference): {val_cfg['F_z_fine']:.4e} N, "
          f"K_fine = {val_cfg['K_FVM_fine']:.3f}")
    print(f"  coarse drag (no GNN, from data gen): "
          f"{val_cfg['F_z_coarse']:.4e} N, "
          f"K_coarse = {val_cfg['K_FVM_coarse']:.3f}, "
          f"err = {val_cfg['coarse_vs_fine_err_pct']:.1f}%")

    # Build the val coarse mesh + cfg
    mesh, bcs, piso_cfg, bodies, L_lift, dt, sphere_centre = \
        build_mesh_and_step(val_cfg, coarse=True)
    print(f"  val coarse mesh {mesh.cartesian_shape} = {mesh.N_cells} cells")

    # Load initial state (the coarse PISO snapshot from Step 1) and
    # fine reference (downsampled to coarse resolution).
    init_state = load_initial_state("val_A", mesh, L_lift)
    fine_u, fine_p = load_fine_ref("val_A", mesh)
    u_phys_init = init_state["u"] + L_lift.u_lift_static

    # Load trained corrector
    with open(DATA_DIR / "gnn_params_local.pkl", "rb") as f:
        corrector = pickle.load(f)
    print(f"  loaded corrector params: {corrector.param_count()}")

    N_EVAL = 50   # short rollout — already-converged init
    # ---- Uncorrected baseline ----
    print(f"\n  Running {N_EVAL} PISO steps WITHOUT GNN correction...",
          flush=True)
    t0 = time.time()
    state_uncorr = run_uncorrected(mesh, bcs, piso_cfg, bodies,
                                    L_lift, dt, init_state, N_EVAL)
    state_uncorr["u"].block_until_ready()
    t_uncorr = time.time() - t0
    F_uncorr, K_uncorr = drag_K(state_uncorr, mesh, L_lift, val_cfg)
    print(f"    done in {t_uncorr:.0f}s")
    print(f"    drag = {F_uncorr:.4e} N, K_FVM = {K_uncorr:.3f}")

    # ---- Corrected ----
    print(f"\n  Running {N_EVAL} PISO steps WITH trained GNN correction...",
          flush=True)
    t0 = time.time()
    state_corr = run_corrected(mesh, bcs, piso_cfg, bodies,
                                L_lift, dt, init_state, N_EVAL, corrector)
    state_corr["u"].block_until_ready()
    t_corr = time.time() - t0
    F_corr, K_corr = drag_K(state_corr, mesh, L_lift, val_cfg)
    print(f"    done in {t_corr:.0f}s")
    print(f"    drag = {F_corr:.4e} N, K_FVM = {K_corr:.3f}")

    # ---- Reporting ----
    F_fine = val_cfg["F_z_fine"]
    K_fine = val_cfg["K_FVM_fine"]
    err_uncorr = abs(K_uncorr - K_fine) / abs(K_fine) * 100
    err_corr   = abs(K_corr   - K_fine) / abs(K_fine) * 100

    u_phys_uncorr = state_uncorr["u"] + L_lift.u_lift_static
    u_phys_corr   = state_corr["u"]   + L_lift.u_lift_static
    mse_init   = float(jnp.mean((u_phys_init - fine_u) ** 2))
    mse_uncorr = float(jnp.mean((u_phys_uncorr - fine_u) ** 2))
    mse_corr   = float(jnp.mean((u_phys_corr - fine_u) ** 2))

    print("\n" + "=" * 78)
    print("Val_A drag comparison (vs fine reference)")
    print("=" * 78)
    print(f"  Fine mesh K_FVM (reference)      : {K_fine:.3f}")
    print(f"  Coarse, no correction (50 step)  : {K_uncorr:.3f}  "
          f"err = {err_uncorr:.2f}%")
    print(f"  Coarse, GNN correction (50 step) : {K_corr:.3f}  "
          f"err = {err_corr:.2f}%")
    if err_corr < err_uncorr:
        print(f"  Drag error REDUCED by "
              f"{(err_uncorr - err_corr) / err_uncorr * 100:.1f}% relative")
        verdict_drag = "PASS (correction improves drag)"
    else:
        print(f"  Drag error WORSE by "
              f"{(err_corr - err_uncorr) / err_uncorr * 100:.1f}% relative")
        verdict_drag = "FAIL (correction hurts drag — possible overfitting)"
    print(f"\n  Verdict: {verdict_drag}")

    print(f"\n  Velocity MSE (vs fine_downsampled):")
    print(f"    init (Step1 coarse PISO)    : {mse_init:.4e}")
    print(f"    uncorrected (50 PISO step)  : {mse_uncorr:.4e}")
    print(f"    corrected (50 PISO step)    : {mse_corr:.4e}")
    if mse_corr < mse_uncorr:
        print(f"    MSE reduction              : {mse_uncorr/mse_corr:.2f}×")
        verdict_mse = "PASS"
    else:
        print(f"    MSE WORSE by {mse_corr/mse_uncorr:.2f}×")
        verdict_mse = "FAIL"
    print(f"  Verdict: {verdict_mse}")

    out = dict(
        K_fine=K_fine, K_uncorrected=K_uncorr, K_corrected=K_corr,
        err_uncorrected_pct=err_uncorr, err_corrected_pct=err_corr,
        mse_init=mse_init, mse_uncorrected=mse_uncorr, mse_corrected=mse_corr,
        verdict_drag=verdict_drag, verdict_mse=verdict_mse,
        wall_uncorr_s=t_uncorr, wall_corr_s=t_corr,
    )
    with open(DATA_DIR / "step3_evaluation.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
