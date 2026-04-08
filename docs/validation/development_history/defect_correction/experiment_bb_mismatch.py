#!/usr/bin/env python3
"""Free-space BB-BEM mismatch diagnostic.

Compare u_LBM(eval sphere) vs u_BEM(eval sphere) when the body is
represented as a Bouzidi BB surface (translating at U) instead of
IB Peskin delta spread forces.

The BB body produces a sharp velocity field. If it matches the BEM
Stokeslet better than the IB, the per-direction dispatch may be
eliminated.
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
)
from mime.nodes.environment.lbm.immersed_boundary import (
    precompute_ib_stencil,
    spread_forces,
    interpolate_velocity,
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
    dx = 2.5 * R_CYL / N_target
    N = int(np.ceil(2.5 * R_CYL / dx))
    N = ((N + 7) // 8) * 8

    tau = 0.8
    nu_lu = (tau - 0.5) / 3.0
    dt = nu_lu * dx**2 / (MU / RHO)

    # BEM
    body_mesh = sphere_surface_mesh(radius=A, n_refine=2)
    body_pts = jnp.array(body_mesh.points)
    body_wts = jnp.array(body_mesh.weights)
    eps = body_mesh.mean_spacing / 2.0
    N_b = body_mesh.n_points
    A_bem = assemble_system_matrix(body_pts, body_wts, eps, MU)
    lu, piv = jax.scipy.linalg.lu_factor(A_bem)

    # Body Bouzidi BB
    body_R_lu = A / dx
    ix = jnp.arange(N, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
    body_dist = jnp.sqrt((gx - N/2)**2 + (gy - N/2)**2 + (gz - N/2)**2)
    body_wall = body_dist <= body_R_lu
    body_missing = compute_missing_mask(body_wall)

    def sphere_sdf(pts):
        d = pts - jnp.array([N/2.0, N/2.0, N/2.0])
        return jnp.sqrt(jnp.sum(d**2, axis=-1)) - body_R_lu

    body_q = compute_q_values_sdf_sparse(body_missing, sphere_sdf)

    # IB stencils (for comparison and eval sphere interpolation)
    body_pts_lu = np.array(body_mesh.points) / dx + np.array([N / 2] * 3)
    ib_idx, ib_wts = precompute_ib_stencil(body_pts_lu, (N, N, N))

    # Eval sphere stencils
    eval_stencils = {}
    R_factors = [1.15, 1.3, 1.5, 2.0, 2.5, 3.0]
    for rf in R_factors:
        R_ev = rf * A
        ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
        ev_lu = ev_mesh.points / dx + np.array([N / 2] * 3)
        ei, ew = precompute_ib_stencil(ev_lu, (N, N, N))
        eval_stencils[rf] = {
            'pts_phys': jnp.array(ev_mesh.points),
            'idx': jnp.array(ei),
            'wts': jnp.array(ew),
        }

    stream_idx = _build_stream_indices(N, N, N)
    force_conv = dt**2 / (RHO * dx**4)

    print(f"  N={N}^3, dx={dx:.4f}, dt={dt:.6f}")
    print(f"  Body: R={body_R_lu:.2f} lu, {int(jnp.sum(body_missing))} BB links")

    return {
        'N': N, 'dx': dx, 'dt': dt, 'tau': tau,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b, 'lu': lu, 'piv': piv,
        'body_wall': body_wall, 'body_missing': body_missing,
        'body_q': body_q,
        'ib_idx': jnp.array(ib_idx), 'ib_wts': jnp.array(ib_wts),
        'eval_stencils': eval_stencils,
        'stream_idx': stream_idx,
        'force_conv': force_conv,
        'R_factors': R_factors,
    }


def lbm_step_bb_body(f, cfg, body_velocity_lu):
    """LBM step with Bouzidi BB body (translating at body_velocity_lu).
    No pipe wall — body only in open domain. Open BCs on all faces."""
    N = cfg['N']; tau = cfg['tau']
    e = jnp.array(E, dtype=jnp.float32)
    w = jnp.array(W, dtype=jnp.float32)

    rho = jnp.sum(f, axis=-1)
    u = (f @ e) / jnp.maximum(rho[..., None], 1e-10)
    edu = u @ e.T
    usq = jnp.sum(u**2, axis=-1, keepdims=True)
    feq = w * rho[..., None] * (1+edu/CS2+edu**2/(2*CS4)-usq/(2*CS2))
    fp = f - (f - feq) / tau

    fs = fp.reshape(N**3, Q)[cfg['stream_idx'], jnp.arange(Q)].reshape(N, N, N, Q)

    # Bouzidi BB with moving wall velocity = body_velocity_lu
    wall_vel = jnp.broadcast_to(body_velocity_lu, (N, N, N, 3))
    fs = apply_bouzidi_bounce_back(
        fs, fp,
        cfg['body_missing'], cfg['body_wall'],
        cfg['body_q'],
        wall_velocity=wall_vel,
    )

    # Open BCs on all faces (no pipe wall)
    for ax in range(3):
        fs = _apply_open_bc(fs, ax)

    return fs, u


def lbm_step_ib_body(f, cfg, force_field):
    """LBM step with IB body (Peskin spread forces). No pipe wall."""
    N = cfg['N']; tau = cfg['tau']
    e = jnp.array(E, dtype=jnp.float32)
    w = jnp.array(W, dtype=jnp.float32)

    rho = jnp.sum(f, axis=-1)
    momentum = f @ e + 0.5 * force_field
    u = momentum / jnp.maximum(rho[..., None], 1e-10)

    edu = u @ e.T
    usq = jnp.sum(u**2, axis=-1, keepdims=True)
    feq = w * rho[..., None] * (1+edu/CS2+edu**2/(2*CS4)-usq/(2*CS2))
    fp = f - (f - feq) / tau

    # Guo forcing
    pref = 1.0 - 0.5 / tau
    Fde = force_field @ e.T
    Fdu = jnp.sum(force_field * u, axis=-1, keepdims=True)
    S = pref * w * (Fde/CS2 - Fdu/CS2 + edu*Fde/CS4)
    fp = fp + S

    fs = fp.reshape(N**3, Q)[cfg['stream_idx'], jnp.arange(Q)].reshape(N, N, N, Q)

    # Open BCs on all faces
    for ax in range(3):
        fs = _apply_open_bc(fs, ax)

    return fs, u


def main():
    print(f"Device: {jax.devices()[0]}")

    N_target = 48
    cfg = setup(N_target)
    N = cfg['N']; N_b = cfg['N_b']
    dx = cfg['dx']; dt = cfg['dt']
    center = jnp.zeros(3)
    e_eye = jnp.eye(3)

    n_steps = 500
    R_factors = cfg['R_factors']

    # Columns to test: F_x (transverse), F_z (axial), T_x (rotation)
    test_cols = [(0, "F_x"), (2, "F_z"), (3, "T_x")]

    results = {}

    for col, label in test_cols:
        U = e_eye[col] if col < 3 else jnp.zeros(3)
        omega = e_eye[col - 3] if col >= 3 else jnp.zeros(3)
        r = cfg['body_pts'] - center
        u_body = U + jnp.cross(omega, r)

        # BEM traction
        traction = jax.scipy.linalg.lu_solve(
            (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
            u_body.ravel(),
        ).reshape(N_b, 3)

        print(f"\n--- {label} (col {col}) ---")

        # ── BB body: Bouzidi with moving wall velocity ──
        if col < 3:
            body_vel_lu = jnp.array(U) * dt / dx
        else:
            # For rotation: each body surface point has different velocity
            # The BB moving wall velocity should be the local surface velocity
            # For simplicity, use the angular velocity to compute per-node velocity
            # But apply_bouzidi_bounce_back takes (nx,ny,nz,3), not per-link
            # Use the angular velocity at each grid node as an approximation
            omega_vec = e_eye[col - 3]
            body_vel_lu = jnp.zeros(3)  # will use wall_velocity field below

        # For rotation, compute per-node velocity field
        if col >= 3:
            omega_vec = e_eye[col - 3]
            # r_from_centre for each grid node
            ix = jnp.arange(N, dtype=jnp.float32)
            gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
            rx = (gx - N/2) * dx
            ry = (gy - N/2) * dx
            rz = (gz - N/2) * dx
            # v = omega × r
            wall_vel_field = jnp.stack([
                omega_vec[1] * rz - omega_vec[2] * ry,
                omega_vec[2] * rx - omega_vec[0] * rz,
                omega_vec[0] * ry - omega_vec[1] * rx,
            ], axis=-1) * dt / dx  # convert to lattice units
        else:
            wall_vel_field = jnp.broadcast_to(
                jnp.array(U) * dt / dx, (N, N, N, 3)
            )

        # BB LBM step
        def bb_step(f):
            N_ = cfg['N']; tau_ = cfg['tau']
            e_ = jnp.array(E, dtype=jnp.float32)
            w_ = jnp.array(W, dtype=jnp.float32)
            rho = jnp.sum(f, axis=-1)
            u = (f @ e_) / jnp.maximum(rho[..., None], 1e-10)
            edu = u @ e_.T
            usq = jnp.sum(u**2, axis=-1, keepdims=True)
            feq = w_ * rho[..., None] * (1+edu/CS2+edu**2/(2*CS4)-usq/(2*CS2))
            fp = f - (f - feq) / tau_
            fs = fp.reshape(N_**3, Q)[cfg['stream_idx'], jnp.arange(Q)].reshape(N_, N_, N_, Q)
            fs = apply_bouzidi_bounce_back(
                fs, fp, cfg['body_missing'], cfg['body_wall'],
                cfg['body_q'], wall_velocity=wall_vel_field,
            )
            for ax in range(3):
                fs = _apply_open_bc(fs, ax)
            return fs, u

        bb_step_jit = jax.jit(bb_step)

        print(f"  Running BB body LBM ({n_steps} steps)...", end="", flush=True)
        t0 = time.time()
        f_bb = init_equilibrium(N, N, N)
        for s in range(n_steps):
            f_bb, u_bb = bb_step_jit(f_bb)
        print(f" {time.time()-t0:.1f}s")

        # ── IB body: Peskin force spreading ──
        point_forces = traction * cfg['body_wts'][:, None] * cfg['force_conv']
        force_field = spread_forces(point_forces, cfg['ib_idx'], cfg['ib_wts'], (N, N, N))

        ib_step_jit = jax.jit(lambda f: lbm_step_ib_body(f, cfg, force_field))

        print(f"  Running IB body LBM ({n_steps} steps)...", end="", flush=True)
        t0 = time.time()
        f_ib = init_equilibrium(N, N, N)
        for s in range(n_steps):
            f_ib, u_ib = ib_step_jit(f_ib)
        print(f" {time.time()-t0:.1f}s")

        # ── Compare at eval spheres ──
        comp = col if col < 3 else 0  # component to measure

        print(f"\n  {'R_factor':>8}  {'IB_mismatch':>12}  {'BB_mismatch':>12}  {'BB/IB':>8}")
        print(f"  {'-'*46}")

        for rf in R_factors:
            es = cfg['eval_stencils'][rf]

            # BEM reference
            u_bem = evaluate_velocity_field(
                es['pts_phys'], cfg['body_pts'], cfg['body_wts'],
                traction, cfg['eps'], MU,
            )
            u_bem_mean = float(jnp.mean(u_bem[:, comp]))

            # IB-LBM
            u_ib_eval = interpolate_velocity(u_ib, es['idx'], es['wts']) * dx / dt
            u_ib_mean = float(jnp.mean(u_ib_eval[:, comp]))
            ib_mm = abs(u_ib_mean / u_bem_mean - 1.0) * 100 if abs(u_bem_mean) > 1e-10 else 0

            # BB-LBM
            u_bb_eval = interpolate_velocity(u_bb, es['idx'], es['wts']) * dx / dt
            u_bb_mean = float(jnp.mean(u_bb_eval[:, comp]))
            bb_mm = abs(u_bb_mean / u_bem_mean - 1.0) * 100 if abs(u_bem_mean) > 1e-10 else 0

            ratio = bb_mm / (ib_mm + 1e-10)
            results[(label, "IB", rf)] = ib_mm
            results[(label, "BB", rf)] = bb_mm

            print(f"  R={rf:.2f}a   {ib_mm:>10.1f}%  {bb_mm:>10.1f}%  {ratio:>7.2f}x")

    # Grand summary
    print(f"\n{'='*70}")
    print("BB vs IB FREE-SPACE MISMATCH SUMMARY")
    print(f"{'='*70}")
    header = f"{'':>12}"
    for rf in R_factors:
        header += f"  R={rf:.2f}a"
    print(header)
    print("-" * (14 + 10 * len(R_factors)))

    for label in ["F_x", "F_z", "T_x"]:
        for kernel in ["IB", "BB"]:
            row = f"  {label:>4} {kernel:>4}:"
            for rf in R_factors:
                mm = results.get((label, kernel, rf), float('nan'))
                row += f"  {mm:>7.1f}%"
            print(row)
        print()

    # Direction independence check
    print("Direction independence (BB):")
    for rf in [1.15, 1.5, 2.0]:
        fx = results.get(("F_x", "BB", rf), 0)
        fz = results.get(("F_z", "BB", rf), 0)
        diff = abs(fx - fz)
        print(f"  R={rf:.2f}a: F_x={fx:.1f}%, F_z={fz:.1f}%, |diff|={diff:.1f}%")


if __name__ == "__main__":
    main()
