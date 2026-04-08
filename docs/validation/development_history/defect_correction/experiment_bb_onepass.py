#!/usr/bin/env python3
"""One-pass BB defect correction — no iteration, no per-direction dispatch.

The BB body gives 7% direction-independent mismatch at R=1.15a.
One BEM re-solve with the eval-sphere wall correction should suffice.

Method A: Direct closest-radius (R=1.15a), one pass
Method B: Richardson (R=1.15a + R=1.3a), one pass
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
from mime.nodes.environment.stokeslet.bem import (
    assemble_system_matrix, compute_force_torque,
)
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field
from mime.nodes.environment.lbm.d3q19 import (
    init_equilibrium, E, W, Q, OPP, CS2,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    apply_bounce_back,
    apply_bouzidi_bounce_back,
    compute_q_values_sdf_sparse,
)
from mime.nodes.environment.lbm.immersed_boundary import (
    precompute_ib_stencil,
    interpolate_velocity,
)
from mime.nodes.environment.lbm.pallas_lbm import (
    _apply_open_bc, _build_stream_indices,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM
EPS_NN = min(0.05, 0.02 * (R_CYL - A))
LABELS = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]
CS4 = CS2 * CS2


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


def setup(N_target):
    dx = 2.5 * R_CYL / N_target
    N = int(np.ceil(2.5 * R_CYL / dx))
    N = ((N + 7) // 8) * 8

    tau = 0.8
    nu_lu = (tau - 0.5) / 3.0
    dt = nu_lu * dx**2 / (MU / RHO)

    body_mesh = sphere_surface_mesh(radius=A, n_refine=2)
    body_pts = jnp.array(body_mesh.points)
    body_wts = jnp.array(body_mesh.weights)
    eps = body_mesh.mean_spacing / 2.0
    N_b = body_mesh.n_points
    A_bem = assemble_system_matrix(body_pts, body_wts, eps, MU)
    lu, piv = jax.scipy.linalg.lu_factor(A_bem)

    vessel_R_lu = R_CYL / dx
    cx, cy = N / 2.0, N / 2.0
    ix = jnp.arange(N, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(ix, ix, indexing='ij')
    dist_2d = jnp.sqrt((gx - cx)**2 + (gy - cy)**2)
    pipe_wall = jnp.broadcast_to(
        (dist_2d >= vessel_R_lu)[..., None], (N, N, N),
    )
    pipe_missing = compute_missing_mask(pipe_wall)

    body_R_lu = A / dx
    iz = jnp.arange(N, dtype=jnp.float32)
    gx3, gy3, gz3 = jnp.meshgrid(ix, ix, iz, indexing='ij')
    body_dist = jnp.sqrt((gx3 - N/2)**2 + (gy3 - N/2)**2 + (gz3 - N/2)**2)
    body_wall = body_dist <= body_R_lu
    body_missing = compute_missing_mask(body_wall)

    def sphere_sdf(pts):
        d = pts - jnp.array([N/2.0, N/2.0, N/2.0])
        return jnp.sqrt(jnp.sum(d**2, axis=-1)) - body_R_lu

    body_q = compute_q_values_sdf_sparse(body_missing, sphere_sdf)

    # Eval spheres
    eval_data = {}
    for rf in [1.15, 1.2, 1.3]:
        R_ev = rf * A
        ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
        ev_lu = ev_mesh.points / dx + np.array([N / 2] * 3)
        ei, ew = precompute_ib_stencil(ev_lu, (N, N, N))
        eval_data[rf] = {
            'pts_phys': jnp.array(ev_mesh.points),
            'idx': jnp.array(ei),
            'wts': jnp.array(ew),
        }

    stream_idx = _build_stream_indices(N, N, N)
    c_s_lu = 1.0 / np.sqrt(3.0)
    spinup = max(500, int(3.0 * vessel_R_lu / c_s_lu))

    print(f"  N={N}^3, dx={dx:.4f}, dt={dt:.6f}, spinup={spinup}")
    print(f"  Body: R={body_R_lu:.2f} lu, Pipe: R={vessel_R_lu:.1f} lu")

    return {
        'N': N, 'dx': dx, 'dt': dt, 'tau': tau,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b, 'lu': lu, 'piv': piv,
        'pipe_wall': pipe_wall, 'pipe_missing': pipe_missing,
        'body_wall': body_wall, 'body_missing': body_missing,
        'body_q': body_q,
        'eval_data': eval_data,
        'stream_idx': stream_idx,
        'spinup': spinup,
    }


def make_lbm_step(cfg, body_wall_vel):
    """JIT'd LBM step: BB body (moving) + BB pipe (stationary)."""
    N = cfg['N']; tau = cfg['tau']

    def step(f):
        e = jnp.array(E, dtype=jnp.float32)
        w = jnp.array(W, dtype=jnp.float32)
        rho = jnp.sum(f, axis=-1)
        u = (f @ e) / jnp.maximum(rho[..., None], 1e-10)
        edu = u @ e.T
        usq = jnp.sum(u**2, axis=-1, keepdims=True)
        feq = w * rho[..., None] * (1+edu/CS2+edu**2/(2*CS4)-usq/(2*CS2))
        fp = f - (f - feq) / tau
        fs = fp.reshape(N**3, Q)[cfg['stream_idx'], jnp.arange(Q)].reshape(N, N, N, Q)
        # Pass 1: pipe wall (stationary)
        fs = apply_bounce_back(fs, fp, cfg['pipe_missing'], cfg['pipe_wall'],
                               wall_velocity=None)
        # Pass 2: body Bouzidi (moving)
        fs = apply_bouzidi_bounce_back(fs, fp, cfg['body_missing'], cfg['body_wall'],
                                        cfg['body_q'], wall_velocity=body_wall_vel)
        for ax in range(3):
            fs = _apply_open_bc(fs, ax)
        return fs, u

    return jax.jit(step)


def compute_body_vel(cfg, col):
    """Body BB wall velocity for column col."""
    N = cfg['N']; dx = cfg['dx']; dt = cfg['dt']
    vc = dt / dx
    e_eye = jnp.eye(3)
    if col < 3:
        return jnp.broadcast_to(e_eye[col] * vc, (N, N, N, 3))
    else:
        omega = e_eye[col - 3]
        ix = jnp.arange(N, dtype=jnp.float32)
        gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
        rx = (gx - N/2) * dx; ry = (gy - N/2) * dx; rz = (gz - N/2) * dx
        return jnp.stack([
            omega[1]*rz - omega[2]*ry,
            omega[2]*rx - omega[0]*rz,
            omega[0]*ry - omega[1]*rx,
        ], axis=-1) * vc


def run_onepass(cfg, nn_diag):
    """Run all 6 columns with one-pass BB defect correction."""
    N = cfg['N']; N_b = cfg['N_b']
    dx = cfg['dx']; dt = cfg['dt']
    center = jnp.zeros(3)
    e_eye = jnp.eye(3)

    R_direct = np.zeros((6, 6))
    R_rich = np.zeros((6, 6))

    for col in range(6):
        U = e_eye[col] if col < 3 else jnp.zeros(3)
        omega = e_eye[col - 3] if col >= 3 else jnp.zeros(3)
        r = cfg['body_pts'] - center
        u_body = U + jnp.cross(omega, r)

        # Step 1: BEM free-space
        traction_0 = jax.scipy.linalg.lu_solve(
            (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
            u_body.ravel(),
        ).reshape(N_b, 3)

        # Step 2: BB LBM
        body_wv = compute_body_vel(cfg, col)
        lbm_step = make_lbm_step(cfg, body_wv)

        f = init_equilibrium(N, N, N)
        for s in range(cfg['spinup']):
            f, u_lbm = lbm_step(f)

        # Step 3: Eval sphere Δu at multiple radii
        du_evals = {}
        for rf, es in cfg['eval_data'].items():
            u_lbm_ev = interpolate_velocity(u_lbm, es['idx'], es['wts']) * dx / dt
            u_bem_ev = evaluate_velocity_field(
                es['pts_phys'], cfg['body_pts'], cfg['body_wts'],
                traction_0, cfg['eps'], MU,
            )
            du_evals[rf] = jnp.mean(u_lbm_ev - u_bem_ev, axis=0)  # (3,)

        # Print Δu for diagnostics
        du_115 = du_evals[1.15]
        print(f"  col {col} ({LABELS[col]}): "
              f"Δu_115=[{float(du_115[0]):.5f},{float(du_115[1]):.5f},{float(du_115[2]):.5f}]", end="")

        # ── Method A: Direct closest-radius (R=1.15a) ──
        delta_u_direct = du_evals[1.15]
        delta_u_body_A = jnp.broadcast_to(delta_u_direct, (N_b, 3))
        traction_A = jax.scipy.linalg.lu_solve(
            (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
            (u_body - delta_u_body_A).ravel(),
        ).reshape(N_b, 3)
        F_A, T_A = compute_force_torque(
            cfg['body_pts'], cfg['body_wts'], traction_A, center,
        )
        R_direct[:3, col] = np.array(F_A)
        R_direct[3:, col] = np.array(T_A)

        # ── Method B: Richardson from R=1.15a and R=1.3a ──
        du_inner = du_evals[1.15]
        du_outer = du_evals[1.3]
        d_inner = 1.15 - 1.0  # = 0.15
        d_outer = 1.3 - 1.0   # = 0.30
        delta_u_rich = (du_inner * d_outer - du_outer * d_inner) / (d_outer - d_inner)
        delta_u_body_B = jnp.broadcast_to(delta_u_rich, (N_b, 3))
        traction_B = jax.scipy.linalg.lu_solve(
            (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
            (u_body - delta_u_body_B).ravel(),
        ).reshape(N_b, 3)
        F_B, T_B = compute_force_torque(
            cfg['body_pts'], cfg['body_wts'], traction_B, center,
        )
        R_rich[:3, col] = np.array(F_B)
        R_rich[3:, col] = np.array(T_B)

        dA = float(R_direct[col, col])
        dB = float(R_rich[col, col])
        eA = abs(dA - nn_diag[col]) / abs(nn_diag[col]) * 100
        eB = abs(dB - nn_diag[col]) / abs(nn_diag[col]) * 100
        print(f" → Direct={dA:.2f}({eA:.1f}%), Rich={dB:.2f}({eB:.1f}%)")

    return R_direct, R_rich


def main():
    print(f"Device: {jax.devices()[0]}")
    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    for N_target in [48, 64]:
        print(f"\n{'='*80}")
        print(f"ONE-PASS BB DEFECT CORRECTION: N={N_target}^3")
        print(f"{'='*80}")
        cfg = setup(N_target)

        t0 = time.time()
        R_direct, R_rich = run_onepass(cfg, nn_diag)
        elapsed = time.time() - t0

        for label, R_mat in [("Direct R=1.15a", R_direct), ("Richardson", R_rich)]:
            diag = [float(R_mat[i, i]) for i in range(6)]
            errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]

            print(f"\n  {label}:")
            print(f"  {'':>6}  {'value':>10}  {'NN-BEM':>10}  {'err':>6}")
            print(f"  {'-'*38}")
            all_pass = True
            for i in range(6):
                ok = "PASS" if errs[i] < 5 else ("WARN" if errs[i] < 10 else "FAIL")
                if errs[i] >= 5: all_pass = False
                print(f"  {LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>5.1f}% [{ok}]")
            print(f"  {'ALL PASS' if all_pass else 'SOME FAILURES'}")

        print(f"\n  Time: {elapsed:.0f}s")

    print("\nAll cloud instances should be torn down.")


if __name__ == "__main__":
    main()
