#!/usr/bin/env python3
"""Schwarz decomposition with corrected open BCs.

Fixes the spurious pressure gradient by imposing u = -u_free at the
open faces (not u = 0). Both inlet and outlet set to equilibrium
at the correct wall-correction velocity.

Body: Bouzidi IBB (u_body=0, exact sphere SDF)
Pipe: Simple BB with Ladd correction (u_wall=-u_free)
Faces: Equilibrium at u = -u_free (not u = 0)
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
    compute_momentum_exchange_force,
    compute_momentum_exchange_torque,
)
from mime.nodes.environment.lbm.pallas_lbm import _build_stream_indices

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


def setup_schwarz(N_target):
    """Set up Schwarz decomposition."""
    dx = 2.5 * R_CYL / N_target
    N = int(np.ceil(2.5 * R_CYL / dx))
    N = ((N + 7) // 8) * 8

    tau = 0.8
    nu_lu = (tau - 0.5) / 3.0
    dt_lbm = nu_lu * dx**2 / (MU / RHO)

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

    # Body wall
    body_R_lu = A / dx
    iz = jnp.arange(N, dtype=jnp.float32)
    gx3, gy3, gz3 = jnp.meshgrid(ix, ix, iz, indexing='ij')
    body_dist = jnp.sqrt((gx3 - N/2)**2 + (gy3 - N/2)**2 + (gz3 - N/2)**2)
    body_wall = body_dist <= body_R_lu
    body_missing = compute_missing_mask(body_wall)

    # Bouzidi q-values for body
    def sphere_sdf(pts):
        d = pts - jnp.array([N/2.0, N/2.0, N/2.0])
        return jnp.sqrt(jnp.sum(d**2, axis=-1)) - body_R_lu

    print(f"  Computing Bouzidi q-values...")
    body_q_values = compute_q_values_sdf_sparse(body_missing, sphere_sdf)
    q_flat = body_q_values[body_missing]
    mean_q = float(jnp.mean(q_flat))
    print(f"  Body: R={body_R_lu:.2f} lu, {int(jnp.sum(body_missing))} BB links, mean q={mean_q:.4f}")

    # Pipe fluid mask
    pipe_fluid_mask = jnp.any(pipe_missing, axis=0)

    # Streaming indices
    stream_idx = _build_stream_indices(N, N, N)

    # Acoustic spinup
    c_s_lu = 1.0 / np.sqrt(3.0)
    spinup_steps = max(500, int(3.0 * vessel_R_lu / c_s_lu))

    print(f"  Grid: N={N}^3, dx={dx:.4f}, dt={dt_lbm:.6f}")
    print(f"  Pipe: R={vessel_R_lu:.1f} lu, spinup={spinup_steps}")

    return {
        'N': N, 'dx': dx, 'dt': dt_lbm, 'tau': tau,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b, 'lu': lu, 'piv': piv,
        'pipe_wall': pipe_wall, 'pipe_missing': pipe_missing,
        'body_wall': body_wall, 'body_missing': body_missing,
        'body_q_values': body_q_values,
        'pipe_fluid_mask': pipe_fluid_mask,
        'stream_idx': stream_idx,
        'spinup_steps': spinup_steps,
    }


def compute_face_and_wall_velocity(cfg, traction):
    """Evaluate u_free at pipe wall nodes AND at inlet/outlet face nodes.

    Returns:
        pipe_wall_vel: (N, N, N, 3) with -u_free_lu at pipe wall nodes
        face_feq_z0: (N, N, Q) equilibrium at z=0 face with u=-u_free_lu
        face_feq_zN: (N, N, Q) equilibrium at z=N-1 face with u=-u_free_lu
    """
    N = cfg['N']; dx = cfg['dx']; dt = cfg['dt']

    # ── Pipe wall nodes ──
    mask_np = np.array(cfg['pipe_fluid_mask'])
    wall_indices = np.argwhere(mask_np)
    wall_phys = (wall_indices.astype(float) - N / 2) * dx

    # ── Face nodes (z=0 and z=N-1) ──
    # All (x, y) positions at z=0 and z=N-1
    ix_np = np.arange(N, dtype=float)
    face_gx, face_gy = np.meshgrid(ix_np, ix_np, indexing='ij')
    face_gx = face_gx.ravel()
    face_gy = face_gy.ravel()

    # z=0 face physical positions
    face_z0_phys = np.stack([
        (face_gx - N/2) * dx,
        (face_gy - N/2) * dx,
        np.full(N*N, (0 - N/2) * dx),
    ], axis=-1)

    # z=N-1 face physical positions
    face_zN_phys = np.stack([
        (face_gx - N/2) * dx,
        (face_gy - N/2) * dx,
        np.full(N*N, (N - 1 - N/2) * dx),
    ], axis=-1)

    # ── Batch BEM evaluation ──
    all_pts = np.concatenate([wall_phys, face_z0_phys, face_zN_phys], axis=0)
    n_wall = len(wall_indices)
    n_face = N * N

    print(f"  BEM eval at {n_wall} wall + {2*n_face} face nodes...", end="", flush=True)
    t0 = time.time()
    u_free_all = evaluate_velocity_field(
        jnp.array(all_pts),
        cfg['body_pts'], cfg['body_wts'],
        traction, cfg['eps'], MU,
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")

    u_free_all_np = np.array(u_free_all)
    u_wall_phys = u_free_all_np[:n_wall]
    u_face_z0_phys = u_free_all_np[n_wall:n_wall + n_face]
    u_face_zN_phys = u_free_all_np[n_wall + n_face:]

    # Convert to lattice units
    vel_conv = dt / dx
    u_wall_lu = u_wall_phys * vel_conv
    u_z0_lu = u_face_z0_phys * vel_conv
    u_zN_lu = u_face_zN_phys * vel_conv

    print(f"  |u_wall_lu|_max = {np.max(np.abs(u_wall_lu)):.6f} "
          f"(Ma = {np.max(np.abs(u_wall_lu)) * np.sqrt(3):.4f})")
    print(f"  |u_face_z0_lu|_max = {np.max(np.abs(u_z0_lu)):.6f}")
    print(f"  |u_face_zN_lu|_max = {np.max(np.abs(u_zN_lu)):.6f}")

    # ── Pipe wall velocity field ──
    pipe_wall_vel = np.zeros((N, N, N, 3), dtype=np.float32)
    pipe_wall_vel[wall_indices[:, 0], wall_indices[:, 1], wall_indices[:, 2]] = \
        (-u_wall_lu).astype(np.float32)

    # ── Face equilibrium distributions ──
    # f_eq = w * rho * (1 + e·u/cs² + (e·u)²/(2*cs⁴) - u²/(2*cs²))
    e_arr = np.array(E, dtype=np.float32)  # (Q, 3)
    w_arr = np.array(W, dtype=np.float32)  # (Q,)
    rho_0 = 1.0

    def compute_face_feq(u_face_lu):
        """Compute equilibrium at face nodes. u_face: (N*N, 3)"""
        u = -u_face_lu  # negate: wall correction velocity = -u_free
        u = u.reshape(N, N, 3)
        e_dot_u = np.einsum('qa,xya->xyq', e_arr, u)  # (N, N, Q)
        u_sq = np.sum(u**2, axis=-1, keepdims=True)  # (N, N, 1)
        feq = w_arr[None, None, :] * rho_0 * (
            1.0 + e_dot_u / CS2 + e_dot_u**2 / (2.0 * CS4) - u_sq / (2.0 * CS2)
        )
        return feq.astype(np.float32)  # (N, N, Q)

    face_feq_z0 = jnp.array(compute_face_feq(u_z0_lu))
    face_feq_zN = jnp.array(compute_face_feq(u_zN_lu))

    return jnp.array(pipe_wall_vel), face_feq_z0, face_feq_zN


def schwarz_lbm_step(f, cfg, pipe_wall_velocity, face_feq_z0, face_feq_zN):
    """One LBM step with corrected BCs.

    Two-pass BB: pipe wall (simple+Ladd) + body (Bouzidi).
    Open faces: equilibrium at u = -u_free (both inlet and outlet).
    """
    N = cfg['N']
    tau = cfg['tau']
    e = jnp.array(E, dtype=jnp.float32)
    w = jnp.array(W, dtype=jnp.float32)
    opp = jnp.array(OPP, dtype=jnp.int32)

    # 1. Macroscopic
    rho = jnp.sum(f, axis=-1)
    momentum = f @ e
    u = momentum / jnp.maximum(rho[..., None], 1e-10)

    # 2. Collision (BGK, no body forces)
    e_dot_u = u @ e.T
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    f_eq = w * rho[..., None] * (
        1.0 + e_dot_u / CS2 + e_dot_u**2 / (2.0 * CS4) - u_sq / (2.0 * CS2)
    )
    f_post = f - (f - f_eq) / tau

    # 3. Streaming
    N_flat = N**3
    f_flat = f_post.reshape(N_flat, Q)
    f_streamed = f_flat[cfg['stream_idx'], jnp.arange(Q)].reshape(N, N, N, Q)

    # 4. Pass 1: Pipe wall BB with Ladd wall velocity
    f_bb = apply_bounce_back(
        f_streamed, f_post,
        cfg['pipe_missing'], cfg['pipe_wall'],
        wall_velocity=pipe_wall_velocity,
    )

    # 5. Pass 2: Body Bouzidi IBB (u_body = 0)
    f_bb = apply_bouzidi_bounce_back(
        f_bb, f_post,
        cfg['body_missing'], cfg['body_wall'],
        cfg['body_q_values'],
        wall_velocity=None,
    )

    # 6. Corrected open BCs on z-faces: equilibrium at u = -u_free
    f_bb = f_bb.at[:, :, 0, :].set(face_feq_z0)
    f_bb = f_bb.at[:, :, -1, :].set(face_feq_zN)

    # 7. Open BCs on x and y faces (pipe wall covers these, but apply for safety)
    # These faces are inside the pipe wall solid region, so the BC has no effect
    # on the flow. Keep the standard u=0 equilibrium.
    from mime.nodes.environment.lbm.pallas_lbm import _apply_open_bc
    f_bb = _apply_open_bc(f_bb, 0)
    f_bb = _apply_open_bc(f_bb, 1)

    return f_bb, u, f


def main():
    print(f"Device: {jax.devices()[0]}")
    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    resolutions = [48, 64]

    for N_target in resolutions:
        print(f"\n{'='*80}")
        print(f"SCHWARZ + BOUZIDI + CORRECTED BC: N={N_target}^3")
        print(f"{'='*80}")

        cfg = setup_schwarz(N_target)
        N = cfg['N']; N_b = cfg['N_b']
        center = jnp.zeros(3)
        e_eye = jnp.eye(3)

        R = np.zeros((6, 6))
        force_factor = RHO * cfg['dx']**4 / cfg['dt']**2
        torque_factor = RHO * cfg['dx']**5 / cfg['dt']**2

        for col in range(6):
            U = e_eye[col] if col < 3 else jnp.zeros(3)
            omega = e_eye[col - 3] if col >= 3 else jnp.zeros(3)

            r = cfg['body_pts'] - center
            u_body = U + jnp.cross(omega, r)

            print(f"\n--- Column {col} ({LABELS[col]}) ---")

            # Step 1: BEM
            traction = jax.scipy.linalg.lu_solve(
                (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
                u_body.ravel(),
            ).reshape(N_b, 3)
            F_free, T_free = compute_force_torque(
                cfg['body_pts'], cfg['body_wts'], traction, center,
            )
            print(f"  F_free = [{float(F_free[0]):.4f}, {float(F_free[1]):.4f}, {float(F_free[2]):.4f}]")

            # Step 2: Wall + face velocities
            pipe_wv, feq_z0, feq_zN = compute_face_and_wall_velocity(cfg, traction)

            # Step 3: JIT and run LBM
            step_fn = jax.jit(
                lambda f, pwv, fz0, fzN: schwarz_lbm_step(f, cfg, pwv, fz0, fzN)
            )

            print(f"  Running LBM ({cfg['spinup_steps']} steps)...", end="", flush=True)
            t0 = time.time()

            f = init_equilibrium(N, N, N)
            for s in range(cfg['spinup_steps']):
                f, u_lbm, f_pre = step_fn(f, pipe_wv, feq_z0, feq_zN)

            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s")

            # Step 4: Momentum exchange
            F_wall_lu = compute_momentum_exchange_force(f_pre, f, cfg['body_missing'])
            body_ctr = jnp.array([N/2.0, N/2.0, N/2.0])
            T_wall_lu = compute_momentum_exchange_torque(
                f_pre, f, cfg['body_missing'], body_ctr,
            )

            F_wall = np.array(F_wall_lu) * force_factor
            T_wall = np.array(T_wall_lu) * torque_factor

            print(f"  F_wall = [{F_wall[0]:.4f}, {F_wall[1]:.4f}, {F_wall[2]:.4f}]")
            print(f"  T_wall = [{T_wall[0]:.4f}, {T_wall[1]:.4f}, {T_wall[2]:.4f}]")

            F_conf = np.array(F_free) + F_wall
            T_conf = np.array(T_free) + T_wall

            R[:3, col] = F_conf
            R[3:, col] = T_conf

        diag = [float(R[i, i]) for i in range(6)]
        errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]

        print(f"\n{'='*60}")
        print(f"N={N_target}^3 SCHWARZ + BOUZIDI + CORRECTED BC")
        print(f"{'='*60}")
        print(f"{'':>6}  {'Schwarz':>10}  {'NN-BEM':>10}  {'error':>8}")
        print("-" * 44)
        all_pass = True
        for i in range(6):
            ok = "PASS" if errs[i] < 5 else ("WARN" if errs[i] < 10 else "FAIL")
            if errs[i] >= 5: all_pass = False
            print(f"{LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>6.1f}% [{ok}]")
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")

        print(f"\nDecomposition (col 0):")
        print(f"  F_free = {float(R[0,0]) - F_wall[0]:.4f}")
        print(f"  F_wall = {F_wall[0]:.4f}")
        print(f"  Total  = {float(R[0,0]):.4f}  (target {nn_diag[0]:.4f})")

    # Convergence check
    if len(resolutions) >= 2:
        print(f"\n{'='*60}")
        print("CONVERGENCE CHECK")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
