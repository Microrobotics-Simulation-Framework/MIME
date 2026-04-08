#!/usr/bin/env python3
"""Defect correction with Bouzidi BB body — single method for all directions.

Replaces IB Peskin delta with Bouzidi BB body surface. BEM handles force
extraction. The 7% direction-independent BB-BEM mismatch at R=1.15a
should allow a single Richardson method for ALL translation columns.

Body: Bouzidi BB (moving at unit velocity for each column)
Pipe: Simple BB (stationary)
Wall correction: Richardson iterated, alpha=0.3, for ALL translation columns
Force: BEM traction integral (exact)
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
from mime.nodes.environment.defect_correction.wall_correction import (
    wall_correction_richardson,
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

    # Pipe wall
    vessel_R_lu = R_CYL / dx
    cx, cy = N / 2.0, N / 2.0
    ix = jnp.arange(N, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(ix, ix, indexing='ij')
    dist_2d = jnp.sqrt((gx - cx)**2 + (gy - cy)**2)
    pipe_wall = jnp.broadcast_to(
        (dist_2d >= vessel_R_lu)[..., None], (N, N, N),
    )
    pipe_missing = compute_missing_mask(pipe_wall)

    # Body BB
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

    # Eval sphere stencils (for Richardson wall correction)
    eval_stencils_close = []
    d_vals_close = []
    for rf in [1.15, 1.2, 1.3]:
        R_ev = rf * A
        ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
        ev_lu = ev_mesh.points / dx + np.array([N / 2] * 3)
        ei, ew = precompute_ib_stencil(ev_lu, (N, N, N))
        eval_stencils_close.append({
            'pts_phys': jnp.array(ev_mesh.points),
            'idx': jnp.array(ei),
            'wts': jnp.array(ew),
        })
        d_vals_close.append(rf - 1.0)

    stream_idx = _build_stream_indices(N, N, N)
    c_s_lu = 1.0 / np.sqrt(3.0)
    spinup = max(500, int(3.0 * vessel_R_lu / c_s_lu))

    print(f"  N={N}^3, dx={dx:.4f}, dt={dt:.6f}")
    print(f"  Body: R={body_R_lu:.2f} lu, {int(jnp.sum(body_missing))} BB links")
    print(f"  Pipe: R={vessel_R_lu:.1f} lu, spinup={spinup}")

    return {
        'N': N, 'dx': dx, 'dt': dt, 'tau': tau,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b, 'lu': lu, 'piv': piv,
        'pipe_wall': pipe_wall, 'pipe_missing': pipe_missing,
        'body_wall': body_wall, 'body_missing': body_missing,
        'body_q': body_q,
        'eval_stencils_close': eval_stencils_close,
        'd_vals_close': jnp.array(d_vals_close),
        'stream_idx': stream_idx,
        'spinup': spinup,
    }


def make_bb_lbm_step(cfg, body_wall_vel):
    """Create a JIT'd LBM step with BB body + BB pipe wall.

    body_wall_vel: (N, N, N, 3) velocity at body BB links (in lattice units).
    """
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

        # Pass 1: Pipe wall BB (stationary, u=0)
        fs = apply_bounce_back(
            fs, fp, cfg['pipe_missing'], cfg['pipe_wall'],
            wall_velocity=None,
        )

        # Pass 2: Body Bouzidi BB (moving at body_wall_vel)
        fs = apply_bouzidi_bounce_back(
            fs, fp, cfg['body_missing'], cfg['body_wall'],
            cfg['body_q'], wall_velocity=body_wall_vel,
        )

        # Open BCs
        for ax in range(3):
            fs = _apply_open_bc(fs, ax)

        return fs, u

    return jax.jit(step)


def compute_body_wall_velocity(cfg, col):
    """Compute the body BB wall velocity field for a given R-matrix column.

    For translation: uniform velocity at all body nodes.
    For rotation: spatially varying (omega x r) at each body node.
    """
    N = cfg['N']; dx = cfg['dx']; dt = cfg['dt']
    vel_conv = dt / dx
    e_eye = jnp.eye(3)

    if col < 3:
        # Translation: uniform body velocity
        U = e_eye[col]
        return jnp.broadcast_to(U * vel_conv, (N, N, N, 3))
    else:
        # Rotation: omega x r at each grid node
        omega = e_eye[col - 3]
        ix = jnp.arange(N, dtype=jnp.float32)
        gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
        rx = (gx - N/2) * dx
        ry = (gy - N/2) * dx
        rz = (gz - N/2) * dx
        vx = omega[1]*rz - omega[2]*ry
        vy = omega[2]*rx - omega[0]*rz
        vz = omega[0]*ry - omega[1]*rx
        return jnp.stack([vx, vy, vz], axis=-1) * vel_conv


def run_defect_correction(cfg, nn_diag):
    """Run full 6x6 R matrix with BB body + Richardson for all directions."""
    N = cfg['N']; N_b = cfg['N_b']
    dx = cfg['dx']; dt = cfg['dt']
    center = jnp.zeros(3)
    e_eye = jnp.eye(3)
    warmstart = 200
    max_iter = 25
    alpha = 0.3
    tol = 0.005

    R = np.zeros((6, 6))

    for col in range(6):
        U = e_eye[col] if col < 3 else jnp.zeros(3)
        omega = e_eye[col - 3] if col >= 3 else jnp.zeros(3)
        r = cfg['body_pts'] - center
        u_body = U + jnp.cross(omega, r)

        print(f"\n--- Column {col} ({LABELS[col]}) ---")

        # Body wall velocity for this column
        body_wv = compute_body_wall_velocity(cfg, col)

        # JIT the LBM step with this body velocity
        lbm_step = make_bb_lbm_step(cfg, body_wv)

        # BEM free-space solve
        traction = jax.scipy.linalg.lu_solve(
            (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
            u_body.ravel(),
        ).reshape(N_b, 3)

        # Initial LBM run (spinup from equilibrium)
        f = init_equilibrium(N, N, N)
        for s in range(cfg['spinup']):
            f, u_lbm = lbm_step(f)

        # Richardson wall correction function
        def richardson_fn(u_lbm_field, trac):
            return wall_correction_richardson(
                u_lbm_field, trac,
                cfg['body_pts'], cfg['body_wts'],
                cfg['eval_stencils_close'], cfg['d_vals_close'],
                cfg['eps'], MU, dx, dt,
            )

        # Defect correction iterations
        n_iter = max_iter if col < 3 else 2
        prev_drag = 0.0

        for iteration in range(n_iter):
            delta_u = richardson_fn(u_lbm, traction)
            delta_u_body = jnp.broadcast_to(delta_u, (N_b, 3))

            traction_new = jax.scipy.linalg.lu_solve(
                (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
                (u_body - delta_u_body).ravel(),
            ).reshape(N_b, 3)

            traction = (1 - alpha) * traction + alpha * traction_new

            # Warmstart LBM (body velocity unchanged, traction updated but
            # the LBM doesn't use traction — it uses BB body.
            # The traction only enters via the eval-sphere BEM comparison.)
            for s in range(warmstart):
                f, u_lbm = lbm_step(f)

            F_iter, T_iter = compute_force_torque(
                cfg['body_pts'], cfg['body_wts'], traction, center,
            )
            drag_diag = float(F_iter[col]) if col < 3 else float(T_iter[col - 3])
            rel_change = abs(drag_diag - prev_drag) / (abs(drag_diag) + 1e-30)
            prev_drag = drag_diag

            if iteration % 5 == 0 or iteration < 3 or rel_change < tol:
                logger.info("  iter %d: diag=%.4f, rel_change=%.4f",
                           iteration + 1, drag_diag, rel_change)

            if iteration > 0 and rel_change < tol:
                logger.info("  Converged at iteration %d", iteration + 1)
                break

        F, T = compute_force_torque(
            cfg['body_pts'], cfg['body_wts'], traction, center,
        )
        R[:3, col] = np.array(F)
        R[3:, col] = np.array(T)

    return R


def main():
    print(f"Device: {jax.devices()[0]}")
    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    for N_target in [48, 64]:
        print(f"\n{'='*80}")
        print(f"BB DEFECT CORRECTION (Richardson ALL): N={N_target}^3")
        print(f"{'='*80}")
        cfg = setup(N_target)

        t0 = time.time()
        R = run_defect_correction(cfg, nn_diag)
        elapsed = time.time() - t0

        diag = [float(R[i, i]) for i in range(6)]
        errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]

        print(f"\n{'='*60}")
        print(f"N={N_target}^3 BB + RICHARDSON (UNIFORM) RESULTS")
        print(f"{'='*60}")
        print(f"{'':>6}  {'BB+Rich':>10}  {'NN-BEM':>10}  {'error':>8}")
        print("-" * 44)
        all_pass = True
        for i in range(6):
            ok = "PASS" if errs[i] < 5 else ("WARN" if errs[i] < 10 else "FAIL")
            if errs[i] >= 5: all_pass = False
            print(f"{LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>6.1f}% [{ok}]")
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
        print(f"Total time: {elapsed:.0f}s")

    # Convergence check
    print(f"\nDone. Tear down cloud instances.")


if __name__ == "__main__":
    main()
