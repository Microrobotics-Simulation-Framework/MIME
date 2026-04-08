#!/usr/bin/env python3
"""Two diagnostics for the Schwarz BB approach.

Diagnostic 1: Does F_wall converge with more LBM steps? (F_x column, 48³)
Diagnostic 2: Does momentum exchange give correct drag for a sphere in
              Poiseuille flow? (known-answer validation)
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

from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
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
from mime.nodes.environment.lbm.pallas_lbm import (
    _apply_open_bc, _build_stream_indices,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM
CS4 = CS2 * CS2


def setup(N_target):
    """Common setup for both diagnostics."""
    dx = 2.5 * R_CYL / N_target
    N = int(np.ceil(2.5 * R_CYL / dx))
    N = ((N + 7) // 8) * 8

    tau = 0.8
    nu_lu = (tau - 0.5) / 3.0
    dt_lbm = nu_lu * dx**2 / (MU / RHO)

    # BEM
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

    # Bouzidi q-values
    def sphere_sdf(pts):
        d = pts - jnp.array([N/2.0, N/2.0, N/2.0])
        return jnp.sqrt(jnp.sum(d**2, axis=-1)) - body_R_lu

    body_q_values = compute_q_values_sdf_sparse(body_missing, sphere_sdf)

    # Pipe fluid mask
    pipe_fluid_mask = jnp.any(pipe_missing, axis=0)

    # Streaming indices
    stream_idx = _build_stream_indices(N, N, N)

    # Combined wall
    combined_wall = body_wall | pipe_wall
    combined_missing = compute_missing_mask(combined_wall)

    print(f"  N={N}^3, dx={dx:.4f}, dt={dt_lbm:.6f}, tau={tau}")
    print(f"  Body: R={body_R_lu:.2f} lu, {int(jnp.sum(body_missing))} BB links")
    print(f"  Pipe: R={vessel_R_lu:.1f} lu")
    print(f"  nu_lu={nu_lu:.4f}")

    return {
        'N': N, 'dx': dx, 'dt': dt_lbm, 'tau': tau, 'nu_lu': nu_lu,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b, 'lu': lu, 'piv': piv,
        'pipe_wall': pipe_wall, 'pipe_missing': pipe_missing,
        'body_wall': body_wall, 'body_missing': body_missing,
        'body_q_values': body_q_values,
        'combined_wall': combined_wall, 'combined_missing': combined_missing,
        'pipe_fluid_mask': pipe_fluid_mask,
        'stream_idx': stream_idx,
        'body_R_lu': A / dx,
        'vessel_R_lu': R_CYL / dx,
    }


def lbm_step_schwarz(f, cfg, pipe_wall_vel, face_feq_z0, face_feq_zN):
    """One LBM step: collision + stream + 2-pass BB + corrected face BCs."""
    N = cfg['N']; tau = cfg['tau']
    e = jnp.array(E, dtype=jnp.float32)
    w = jnp.array(W, dtype=jnp.float32)

    rho = jnp.sum(f, axis=-1)
    momentum = f @ e
    u = momentum / jnp.maximum(rho[..., None], 1e-10)

    e_dot_u = u @ e.T
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    f_eq = w * rho[..., None] * (
        1.0 + e_dot_u / CS2 + e_dot_u**2 / (2.0 * CS4) - u_sq / (2.0 * CS2)
    )
    f_post = f - (f - f_eq) / tau

    N_flat = N**3
    f_flat = f_post.reshape(N_flat, Q)
    f_streamed = f_flat[cfg['stream_idx'], jnp.arange(Q)].reshape(N, N, N, Q)

    f_bb = apply_bounce_back(
        f_streamed, f_post,
        cfg['pipe_missing'], cfg['pipe_wall'],
        wall_velocity=pipe_wall_vel,
    )
    f_bb = apply_bouzidi_bounce_back(
        f_bb, f_post,
        cfg['body_missing'], cfg['body_wall'],
        cfg['body_q_values'],
        wall_velocity=None,
    )

    f_bb = f_bb.at[:, :, 0, :].set(face_feq_z0)
    f_bb = f_bb.at[:, :, -1, :].set(face_feq_zN)
    f_bb = _apply_open_bc(f_bb, 0)
    f_bb = _apply_open_bc(f_bb, 1)

    return f_bb, u, f


def lbm_step_poiseuille(f, cfg, body_force_z):
    """One LBM step with uniform axial body force (Poiseuille flow driver).

    Uses combined wall (body + pipe), no wall velocity (all walls at rest).
    """
    N = cfg['N']; tau = cfg['tau']
    e = jnp.array(E, dtype=jnp.float32)
    w = jnp.array(W, dtype=jnp.float32)

    force = jnp.zeros((N, N, N, 3), dtype=jnp.float32)
    force = force.at[:, :, :, 2].set(body_force_z)

    rho = jnp.sum(f, axis=-1)
    momentum = f @ e + 0.5 * force
    u = momentum / jnp.maximum(rho[..., None], 1e-10)

    e_dot_u = u @ e.T
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    f_eq = w * rho[..., None] * (
        1.0 + e_dot_u / CS2 + e_dot_u**2 / (2.0 * CS4) - u_sq / (2.0 * CS2)
    )
    f_post = f - (f - f_eq) / tau

    # Guo forcing
    pref = 1.0 - 0.5 / tau
    F_dot_e = force @ e.T
    F_dot_u = jnp.sum(force * u, axis=-1, keepdims=True)
    S = pref * w * (F_dot_e / CS2 - F_dot_u / CS2 + e_dot_u * F_dot_e / CS4)
    f_post = f_post + S

    N_flat = N**3
    f_flat = f_post.reshape(N_flat, Q)
    f_streamed = f_flat[cfg['stream_idx'], jnp.arange(Q)].reshape(N, N, N, Q)

    # Single-pass BB: combined wall, all at rest
    f_bb = apply_bounce_back(
        f_streamed, f_post,
        cfg['combined_missing'], cfg['combined_wall'],
        wall_velocity=None,
    )

    # Periodic in z (no open BCs — the body force drives periodic Poiseuille)
    # Actually use open BCs to let flow develop naturally
    for ax in range(3):
        f_bb = _apply_open_bc(f_bb, ax)

    return f_bb, u, f


def diagnostic_1(cfg):
    """Track F_wall convergence over many LBM steps for F_x column."""
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 1: F_wall convergence with step count (F_x, 48^3)")
    print(f"{'='*70}")

    N = cfg['N']; N_b = cfg['N_b']
    dx = cfg['dx']; dt = cfg['dt']
    center = jnp.zeros(3)
    e_eye = jnp.eye(3)

    # BEM for F_x column
    u_body = jnp.broadcast_to(e_eye[0], (N_b, 3))
    traction = jax.scipy.linalg.lu_solve(
        (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
        u_body.ravel(),
    ).reshape(N_b, 3)
    F_free, _ = compute_force_torque(
        cfg['body_pts'], cfg['body_wts'], traction, center,
    )
    print(f"  F_free_x = {float(F_free[0]):.4f}")

    # Wall + face velocities
    mask_np = np.array(cfg['pipe_fluid_mask'])
    wall_indices = np.argwhere(mask_np)
    wall_phys = (wall_indices.astype(float) - N/2) * dx

    ix_np = np.arange(N, dtype=float)
    fgx, fgy = np.meshgrid(ix_np, ix_np, indexing='ij')
    face_z0 = np.stack([(fgx.ravel() - N/2)*dx, (fgy.ravel() - N/2)*dx,
                         np.full(N*N, (0-N/2)*dx)], axis=-1)
    face_zN = np.stack([(fgx.ravel() - N/2)*dx, (fgy.ravel() - N/2)*dx,
                         np.full(N*N, (N-1-N/2)*dx)], axis=-1)

    all_pts = np.concatenate([wall_phys, face_z0, face_zN], axis=0)
    n_wall = len(wall_indices); n_face = N*N

    u_free_all = evaluate_velocity_field(
        jnp.array(all_pts), cfg['body_pts'], cfg['body_wts'],
        traction, cfg['eps'], MU,
    )
    u_all_np = np.array(u_free_all)
    vel_conv = dt / dx

    # Pipe wall velocity
    pipe_wv = np.zeros((N, N, N, 3), dtype=np.float32)
    pipe_wv[wall_indices[:,0], wall_indices[:,1], wall_indices[:,2]] = \
        (-u_all_np[:n_wall] * vel_conv).astype(np.float32)
    pipe_wv = jnp.array(pipe_wv)

    # Face equilibrium
    e_arr = np.array(E, dtype=np.float32)
    w_arr = np.array(W, dtype=np.float32)

    def make_feq(u_phys):
        u_lu = (-u_phys * vel_conv).reshape(N, N, 3)
        edu = np.einsum('qa,xya->xyq', e_arr, u_lu)
        usq = np.sum(u_lu**2, axis=-1, keepdims=True)
        return jnp.array((w_arr * (1 + edu/CS2 + edu**2/(2*CS4) - usq/(2*CS2))).astype(np.float32))

    feq_z0 = make_feq(u_all_np[n_wall:n_wall+n_face])
    feq_zN = make_feq(u_all_np[n_wall+n_face:])

    # JIT
    step_fn = jax.jit(lambda f: lbm_step_schwarz(f, cfg, pipe_wv, feq_z0, feq_zN))

    force_factor = RHO * dx**4 / dt**2
    expected_F_wall = 31.76 - 18.89  # ≈ 12.87

    # Diffusion time from wall to body
    t_diff = (cfg['vessel_R_lu'] - cfg['body_R_lu'])**2 / (2 * cfg['nu_lu'])
    print(f"  Diffusion time (wall→body): {t_diff:.0f} steps")
    print(f"  Expected F_wall_x = {expected_F_wall:.2f}")
    print()

    print(f"{'step':>6}  {'F_wall_x':>10}  {'F_wall_z':>10}  {'err_x%':>8}")
    print("-" * 42)

    f = init_equilibrium(N, N, N)
    max_steps = 10000
    sample_every = 250

    t0 = time.time()
    for step in range(1, max_steps + 1):
        f, u_lbm, f_pre = step_fn(f)

        if step % sample_every == 0:
            F_lu = compute_momentum_exchange_force(f_pre, f, cfg['body_missing'])
            F_phys = np.array(F_lu) * force_factor
            err_x = abs(F_phys[0] - expected_F_wall) / expected_F_wall * 100
            print(f"{step:6d}  {F_phys[0]:>10.4f}  {F_phys[2]:>10.4f}  {err_x:>7.1f}%",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


def diagnostic_2(cfg):
    """Validate momentum exchange for sphere in Poiseuille flow."""
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 2: Momentum exchange validation (Poiseuille flow)")
    print(f"{'='*70}")

    N = cfg['N']
    dx = cfg['dx']; dt = cfg['dt']
    nu_lu = cfg['nu_lu']

    # Body force to drive Poiseuille flow
    # Target: U_max ~ 0.01 lu (low Ma)
    # For Poiseuille in pipe: U_max = g*R²/(4*nu)
    # g = 4*nu*U_max/R²
    R_lu = cfg['vessel_R_lu']
    U_max_target = 0.01
    g_z = 4 * nu_lu * U_max_target / R_lu**2
    print(f"  Body force g_z = {g_z:.8f} (target U_max = {U_max_target})")

    force_factor = RHO * dx**4 / dt**2

    step_fn = jax.jit(lambda f: lbm_step_poiseuille(f, cfg, g_z))

    f = init_equilibrium(N, N, N)
    n_steps = 10000

    print(f"\n{'step':>6}  {'U_max_z':>10}  {'F_drag_z':>10}  {'F_stokes':>10}  {'ratio':>8}")
    print("-" * 54)

    t0 = time.time()
    for step in range(1, n_steps + 1):
        f, u_lbm, f_pre = step_fn(f)

        if step % 1000 == 0:
            # Max velocity
            u_max = float(jnp.max(jnp.abs(u_lbm[:, :, :, 2])))

            # Mean velocity at mid-plane (away from open BC faces)
            mid_z = N // 2
            fluid_mask = ~np.array(cfg['combined_wall'][:, :, mid_z])
            u_z_mid = np.array(u_lbm[:, :, mid_z, 2])
            u_mean = float(np.mean(u_z_mid[fluid_mask]))

            # Momentum exchange force on body
            F_lu = compute_momentum_exchange_force(f_pre, f, cfg['body_missing'])
            F_phys = np.array(F_lu) * force_factor

            # Stokes drag estimate: F = 6πμaU
            # U = undisturbed velocity at body center ≈ U_max_poiseuille
            # For a centered sphere in Poiseuille: U_undisturbed = U_max
            u_phys = u_mean * dx / dt
            F_stokes = 6 * np.pi * MU * A * u_phys

            ratio = F_phys[2] / (F_stokes + 1e-30) if abs(F_stokes) > 1e-10 else float('nan')

            print(f"{step:6d}  {u_max:>10.6f}  {F_phys[2]:>10.4f}  {F_stokes:>10.4f}  {ratio:>8.3f}",
                  flush=True)

    elapsed = time.time() - t0

    # Final detailed analysis
    fluid_mask = ~np.array(cfg['combined_wall'][:, :, N//2])
    u_z_mid = np.array(u_lbm[:, :, N//2, 2])
    u_mean_final = float(np.mean(u_z_mid[fluid_mask]))
    u_max_final = float(np.max(u_z_mid[fluid_mask]))
    u_phys_mean = u_mean_final * dx / dt
    u_phys_max = u_max_final * dx / dt

    F_lu = compute_momentum_exchange_force(f_pre, f, cfg['body_missing'])
    F_phys = np.array(F_lu) * force_factor

    T_lu = compute_momentum_exchange_torque(
        f_pre, f, cfg['body_missing'],
        jnp.array([N/2.0, N/2.0, N/2.0]),
    )
    T_phys = np.array(T_lu) * RHO * dx**5 / dt**2

    # Reference: Stokes drag with mean undisturbed velocity
    F_stokes_mean = 6 * np.pi * MU * A * u_phys_mean
    # Reference: with max velocity (Poiseuille center)
    F_stokes_max = 6 * np.pi * MU * A * u_phys_max

    # Faxén correction for parabolic profile:
    # F = 6πμa[U(0) + a²/6 ∇²U] for sphere in non-uniform flow
    # For Poiseuille: ∇²U = -4U_max/R², so correction = a²/6 * (-4U_max/R²)
    # F_faxen = 6πμa * U_max * (1 - 2a²/(3R²))
    lam = A / R_CYL
    faxen_factor = 1.0 - 2.0/3.0 * lam**2
    F_faxen = 6 * np.pi * MU * A * u_phys_max * faxen_factor

    print(f"\n  Final state (step {n_steps}):")
    print(f"  u_mean_z (mid) = {u_mean_final:.6f} lu = {u_phys_mean:.6f} phys")
    print(f"  u_max_z (mid)  = {u_max_final:.6f} lu = {u_phys_max:.6f} phys")
    print(f"  F_drag_z (ME)  = {F_phys[2]:.4f}")
    print(f"  F_drag_x (ME)  = {F_phys[0]:.4f} (should be ~0)")
    print(f"  T_drag (ME)    = [{T_phys[0]:.4f}, {T_phys[1]:.4f}, {T_phys[2]:.4f}]")
    print(f"  F_stokes(mean) = {F_stokes_mean:.4f}")
    print(f"  F_stokes(max)  = {F_stokes_max:.4f}")
    print(f"  F_faxen(max)   = {F_faxen:.4f} (with ∇²U correction)")
    print(f"  Ratio ME/Stokes(mean) = {F_phys[2]/F_stokes_mean:.3f}")
    print(f"  Ratio ME/Stokes(max)  = {F_phys[2]/F_stokes_max:.3f}")
    print(f"  Ratio ME/Faxen        = {F_phys[2]/F_faxen:.3f}")
    print(f"  Time: {elapsed:.0f}s")


def main():
    print(f"Device: {jax.devices()[0]}")

    cfg = setup(48)

    diagnostic_1(cfg)
    diagnostic_2(cfg)


if __name__ == "__main__":
    main()
