#!/usr/bin/env python3
"""Schwarz + Faxén force extraction.

Instead of momentum exchange (broken at 6 lu), extract F_wall from the
wall-correction velocity field using Faxén's law:

  F_wall = 6πμa × u_∞(body_centre)

where u_∞ is the undisturbed wall-correction flow.

Experiment 1: LBM WITH body BB. Evaluate u_wall at eval spheres outside body.
Experiment 2: LBM WITHOUT body BB. Evaluate u_wall at body centre directly.
              This gives the UNDISTURBED wall flow — exact Faxén input.
Both at 48³ and 64³.
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
    """Setup for both experiments."""
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
    pipe_fluid_mask = jnp.any(pipe_missing, axis=0)

    # Body wall + Bouzidi
    body_R_lu = A / dx
    iz = jnp.arange(N, dtype=jnp.float32)
    gx3, gy3, gz3 = jnp.meshgrid(ix, ix, iz, indexing='ij')
    body_dist = jnp.sqrt((gx3 - N/2)**2 + (gy3 - N/2)**2 + (gz3 - N/2)**2)
    body_wall = body_dist <= body_R_lu
    body_missing = compute_missing_mask(body_wall)

    def sphere_sdf(pts):
        d = pts - jnp.array([N/2.0, N/2.0, N/2.0])
        return jnp.sqrt(jnp.sum(d**2, axis=-1)) - body_R_lu

    body_q_values = compute_q_values_sdf_sparse(body_missing, sphere_sdf)

    # Eval sphere stencils (for sampling u_wall)
    eval_stencils = {}
    for rf in [1.15, 1.3, 1.5, 2.0]:
        R_ev = rf * A
        ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
        ev_lu = ev_mesh.points / dx + np.array([N / 2] * 3)
        ei, ew = precompute_ib_stencil(ev_lu, (N, N, N))
        eval_stencils[rf] = {
            'idx': jnp.array(ei),
            'wts': jnp.array(ew),
        }

    # Streaming
    stream_idx = _build_stream_indices(N, N, N)

    c_s_lu = 1.0 / np.sqrt(3.0)
    spinup = max(500, int(3.0 * vessel_R_lu / c_s_lu))

    print(f"  N={N}^3, dx={dx:.4f}, dt={dt:.6f}")
    print(f"  Body R={body_R_lu:.2f} lu, Pipe R={vessel_R_lu:.1f} lu")
    print(f"  Spinup={spinup} steps")

    return {
        'N': N, 'dx': dx, 'dt': dt, 'tau': tau,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b, 'lu': lu, 'piv': piv,
        'pipe_wall': pipe_wall, 'pipe_missing': pipe_missing,
        'body_wall': body_wall, 'body_missing': body_missing,
        'body_q_values': body_q_values,
        'pipe_fluid_mask': pipe_fluid_mask,
        'eval_stencils': eval_stencils,
        'stream_idx': stream_idx,
        'spinup': spinup,
    }


def compute_bcs(cfg, traction):
    """Compute pipe wall velocity and face equilibria from BEM traction."""
    N = cfg['N']; dx = cfg['dx']; dt = cfg['dt']
    vel_conv = dt / dx

    mask_np = np.array(cfg['pipe_fluid_mask'])
    wall_idx = np.argwhere(mask_np)
    wall_phys = (wall_idx.astype(float) - N/2) * dx

    ix_np = np.arange(N, dtype=float)
    fgx, fgy = np.meshgrid(ix_np, ix_np, indexing='ij')
    face_z0 = np.stack([(fgx.ravel()-N/2)*dx, (fgy.ravel()-N/2)*dx,
                         np.full(N*N, (0-N/2)*dx)], axis=-1)
    face_zN = np.stack([(fgx.ravel()-N/2)*dx, (fgy.ravel()-N/2)*dx,
                         np.full(N*N, (N-1-N/2)*dx)], axis=-1)

    all_pts = np.concatenate([wall_phys, face_z0, face_zN], axis=0)
    n_wall = len(wall_idx); n_face = N*N

    u_free_all = evaluate_velocity_field(
        jnp.array(all_pts), cfg['body_pts'], cfg['body_wts'],
        traction, cfg['eps'], MU,
    )
    u_np = np.array(u_free_all)

    # Pipe wall velocity (negated, lattice units)
    pwv = np.zeros((N, N, N, 3), dtype=np.float32)
    pwv[wall_idx[:,0], wall_idx[:,1], wall_idx[:,2]] = \
        (-u_np[:n_wall] * vel_conv).astype(np.float32)

    # Face equilibria
    e_arr = np.array(E, dtype=np.float32)
    w_arr = np.array(W, dtype=np.float32)

    def feq(u_phys_flat):
        u_lu = (-u_phys_flat * vel_conv).reshape(N, N, 3)
        edu = np.einsum('qa,xya->xyq', e_arr, u_lu)
        usq = np.sum(u_lu**2, axis=-1, keepdims=True)
        return jnp.array((w_arr * (1+edu/CS2+edu**2/(2*CS4)-usq/(2*CS2))).astype(np.float32))

    return jnp.array(pwv), feq(u_np[n_wall:n_wall+n_face]), feq(u_np[n_wall+n_face:])


def lbm_step_with_body(f, cfg, pwv, fz0, fzN):
    """LBM step WITH body Bouzidi BB (body at rest)."""
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

    fs = apply_bounce_back(fs, fp, cfg['pipe_missing'], cfg['pipe_wall'],
                           wall_velocity=pwv)
    fs = apply_bouzidi_bounce_back(fs, fp, cfg['body_missing'], cfg['body_wall'],
                                    cfg['body_q_values'], wall_velocity=None)

    fs = fs.at[:,:,0,:].set(fz0)
    fs = fs.at[:,:,-1,:].set(fzN)
    fs = _apply_open_bc(fs, 0)
    fs = _apply_open_bc(fs, 1)
    return fs, u


def lbm_step_no_body(f, cfg, pwv, fz0, fzN):
    """LBM step WITHOUT body — only pipe wall BCs.
    Gives the undisturbed wall-correction flow."""
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

    # Only pipe wall BB (with moving wall velocity)
    fs = apply_bounce_back(fs, fp, cfg['pipe_missing'], cfg['pipe_wall'],
                           wall_velocity=pwv)

    fs = fs.at[:,:,0,:].set(fz0)
    fs = fs.at[:,:,-1,:].set(fzN)
    fs = _apply_open_bc(fs, 0)
    fs = _apply_open_bc(fs, 1)
    return fs, u


def run_and_extract(cfg, nn_diag, mode):
    """Run all 6 columns and extract F_wall via Faxén.

    mode: "with_body" or "no_body"
    """
    N = cfg['N']; N_b = cfg['N_b']
    dx = cfg['dx']; dt = cfg['dt']
    center = jnp.zeros(3)
    e_eye = jnp.eye(3)

    if mode == "with_body":
        step_raw = lambda f, p, z0, zN: lbm_step_with_body(f, cfg, p, z0, zN)
    else:
        step_raw = lambda f, p, z0, zN: lbm_step_no_body(f, cfg, p, z0, zN)

    step_fn = jax.jit(step_raw)

    # Faxén: F = 6πμa × u_∞
    faxen_prefactor = 6.0 * np.pi * MU * A

    R = np.zeros((6, 6))

    for col in range(6):
        U = e_eye[col] if col < 3 else jnp.zeros(3)
        omega = e_eye[col - 3] if col >= 3 else jnp.zeros(3)
        r = cfg['body_pts'] - center
        u_body = U + jnp.cross(omega, r)

        # BEM
        traction = jax.scipy.linalg.lu_solve(
            (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
            u_body.ravel(),
        ).reshape(N_b, 3)
        F_free, T_free = compute_force_torque(
            cfg['body_pts'], cfg['body_wts'], traction, center,
        )

        # BCs
        pwv, fz0, fzN = compute_bcs(cfg, traction)

        # Run LBM
        f = init_equilibrium(N, N, N)
        for s in range(cfg['spinup']):
            f, u_lbm = step_fn(f, pwv, fz0, fzN)

        # Faxén force extraction at multiple eval radii
        print(f"  col {col} ({LABELS[col]}): F_free=[{float(F_free[0]):.2f},{float(F_free[1]):.2f},{float(F_free[2]):.2f}]", end="")

        # Sample u_wall at eval spheres
        u_evals = {}
        for rf, es in cfg['eval_stencils'].items():
            u_ev = interpolate_velocity(u_lbm, es['idx'], es['wts']) * dx / dt
            u_evals[rf] = jnp.mean(u_ev, axis=0)  # (3,) mean velocity

        # Faxén force from each eval radius
        for rf in [1.15, 1.5]:
            u_inf = u_evals[rf]
            F_wall_fax = faxen_prefactor * np.array(u_inf)
            F_conf = np.array(F_free) + F_wall_fax
            diag_val = float(F_conf[col]) if col < 3 else float(
                np.array(T_free)[col-3] + faxen_prefactor * float(u_inf[col-3]) if col < 6 else 0)

        # Use R=1.15a as primary
        u_inf = u_evals[1.15]
        F_wall = faxen_prefactor * np.array(u_inf)
        # For torque: T_wall ≈ 8πμa³ × ω_wall (Faxén for rotation)
        # ω_wall = 0.5 * curl(u_wall) at centre. For now approximate from eval sphere.
        # Actually for torque columns: T_free dominates, T_wall is small.
        # Use the same Faxén-like approach: T_wall ≈ 8πμa³ × (curl u_wall / 2)
        # Simpler: just use eval sphere mean velocity for the force part

        F_conf = np.array(F_free) + F_wall
        # For torque: use free-space torque + small correction
        # The rotational Faxén: T = 8πμa³ω_∞ where ω = curl(u)/2
        # Approximate: T_wall ≈ 0 for translation columns, small for rotation
        T_conf = np.array(T_free)  # first approximation

        # For rotation columns: estimate torque from vorticity
        if col >= 3:
            # Use the rotation axis velocity gradient as an approximation
            # T_wall = 8πμa³ × ω_∞ where ω is from the eval sphere
            # For a sphere, the torque Faxén is: T = 8πμa³ × ω_∞
            # ω from eval sphere: approximate as u_eval × (some factor)
            # For now: use the force Faxén for the force rows,
            # and keep T_free for torque rows (small correction)
            pass

        R[:3, col] = F_conf
        R[3:, col] = T_conf

        diag_v = float(R[col, col])
        err = abs(diag_v - nn_diag[col]) / abs(nn_diag[col]) * 100
        print(f" → diag={diag_v:.2f} (err {err:.1f}%)")

        # Print u_wall at each eval radius
        for rf in sorted(u_evals.keys()):
            u_ev = u_evals[rf]
            comp = col if col < 3 else col - 3
            print(f"    R={rf:.2f}a: u_wall=[{float(u_ev[0]):.5f},{float(u_ev[1]):.5f},{float(u_ev[2]):.5f}]")

    return R


def main():
    print(f"Device: {jax.devices()[0]}")
    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    for N_target in [48, 64]:
        print(f"\n{'='*80}")
        print(f"N={N_target}^3")
        print(f"{'='*80}")
        cfg = setup(N_target)

        # Experiment 1: WITH body BB
        print(f"\n--- Experiment 1: WITH body BB + Faxén ---")
        t0 = time.time()
        R1 = run_and_extract(cfg, nn_diag, "with_body")
        print(f"  Time: {time.time()-t0:.0f}s")

        diag1 = [float(R1[i,i]) for i in range(6)]
        errs1 = [abs(diag1[i]-nn_diag[i])/abs(nn_diag[i])*100 for i in range(6)]
        print(f"\n  {'':>6}  {'Faxén+BB':>10}  {'NN-BEM':>10}  {'err':>6}")
        for i in range(6):
            ok = "PASS" if errs1[i]<5 else ("WARN" if errs1[i]<10 else "FAIL")
            print(f"  {LABELS[i]:>6}  {diag1[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs1[i]:>5.1f}% [{ok}]")

        # Experiment 2: WITHOUT body BB (undisturbed flow)
        print(f"\n--- Experiment 2: NO body BB + Faxén (undisturbed flow) ---")
        t0 = time.time()
        R2 = run_and_extract(cfg, nn_diag, "no_body")
        print(f"  Time: {time.time()-t0:.0f}s")

        diag2 = [float(R2[i,i]) for i in range(6)]
        errs2 = [abs(diag2[i]-nn_diag[i])/abs(nn_diag[i])*100 for i in range(6)]
        print(f"\n  {'':>6}  {'Faxén-NB':>10}  {'NN-BEM':>10}  {'err':>6}")
        for i in range(6):
            ok = "PASS" if errs2[i]<5 else ("WARN" if errs2[i]<10 else "FAIL")
            print(f"  {LABELS[i]:>6}  {diag2[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs2[i]:>5.1f}% [{ok}]")

        # Comparison
        print(f"\n  COMPARISON at N={N_target}^3:")
        print(f"  {'':>6}  {'w/ body':>10}  {'no body':>10}  {'NN-BEM':>10}")
        for i in range(6):
            print(f"  {LABELS[i]:>6}  {diag1[i]:>10.2f}  {diag2[i]:>10.2f}  {nn_diag[i]:>10.2f}")


if __name__ == "__main__":
    main()
