#!/usr/bin/env python3
"""One-pass no-body Schwarz with BEM body-surface re-solve.

The cleanest formulation:
1. BEM free-space solve → traction_0, u_free at pipe wall
2. LBM: pipe wall at u=-u_free, NO body → u_wall everywhere
3. Interpolate u_wall at 320 BEM body surface points
4. BEM re-solve: traction_conf = lu_solve((U_body - u_wall_at_body).ravel())
5. F_confined = integrate(traction_conf)

No IB, no body BB, no momentum exchange, no eval spheres, no extrapolation.
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
    compute_missing_mask, apply_bounce_back,
)
from mime.nodes.environment.lbm.immersed_boundary import (
    precompute_ib_stencil, interpolate_velocity,
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
    pipe_fluid_mask = jnp.any(pipe_missing, axis=0)

    mask_np = np.array(pipe_fluid_mask)
    wall_indices = np.argwhere(mask_np)
    wall_phys = (wall_indices.astype(float) - N / 2) * dx

    ix_np = np.arange(N, dtype=float)
    fgx, fgy = np.meshgrid(ix_np, ix_np, indexing='ij')
    face_z0 = np.stack([(fgx.ravel()-N/2)*dx, (fgy.ravel()-N/2)*dx,
                         np.full(N*N, (0-N/2)*dx)], axis=-1)
    face_zN = np.stack([(fgx.ravel()-N/2)*dx, (fgy.ravel()-N/2)*dx,
                         np.full(N*N, (N-1-N/2)*dx)], axis=-1)

    # IB stencil for body surface points (for interpolating LBM velocity)
    body_pts_lu = np.array(body_mesh.points) / dx + np.array([N / 2] * 3)
    body_ib_idx, body_ib_wts = precompute_ib_stencil(body_pts_lu, (N, N, N))

    stream_idx = _build_stream_indices(N, N, N)
    c_s_lu = 1.0 / np.sqrt(3.0)
    spinup = max(2000, int(6.0 * vessel_R_lu / c_s_lu))

    print(f"  N={N}^3, dx={dx:.4f}, dt={dt:.6f}, spinup={spinup}")
    print(f"  Body: {N_b} BEM pts, Pipe: R={vessel_R_lu:.1f} lu")

    return {
        'N': N, 'dx': dx, 'dt': dt, 'tau': tau,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b, 'lu': lu, 'piv': piv,
        'pipe_wall': pipe_wall, 'pipe_missing': pipe_missing,
        'wall_indices': wall_indices,
        'wall_phys': jnp.array(wall_phys),
        'face_z0': jnp.array(face_z0),
        'face_zN': jnp.array(face_zN),
        'body_ib_idx': jnp.array(body_ib_idx),
        'body_ib_wts': jnp.array(body_ib_wts),
        'stream_idx': stream_idx,
        'spinup': spinup,
    }


def bem_eval_wall_faces(cfg, traction):
    """Evaluate BEM velocity at wall + face nodes → pipe wall vel + face feq."""
    N = cfg['N']; dx = cfg['dx']; dt = cfg['dt']
    vc = dt / dx
    n_wall = len(cfg['wall_indices'])
    n_face = N * N

    all_pts = jnp.concatenate([cfg['wall_phys'], cfg['face_z0'], cfg['face_zN']])
    u_all = evaluate_velocity_field(
        all_pts, cfg['body_pts'], cfg['body_wts'], traction, cfg['eps'], MU,
    )
    u_np = np.array(u_all)

    pwv = np.zeros((N, N, N, 3), dtype=np.float32)
    wi = cfg['wall_indices']
    pwv[wi[:,0], wi[:,1], wi[:,2]] = (-u_np[:n_wall] * vc).astype(np.float32)

    e_arr = np.array(E, dtype=np.float32)
    w_arr = np.array(W, dtype=np.float32)

    def feq(u_phys):
        u_lu = (-u_phys * vc).reshape(N, N, 3)
        edu = np.einsum('qa,xya->xyq', e_arr, u_lu)
        usq = np.sum(u_lu**2, axis=-1, keepdims=True)
        return jnp.array((w_arr*(1+edu/CS2+edu**2/(2*CS4)-usq/(2*CS2))).astype(np.float32))

    return jnp.array(pwv), feq(u_np[n_wall:n_wall+n_face]), feq(u_np[n_wall+n_face:])


def run_lbm_no_body(cfg, pwv, fz0, fzN):
    """LBM: pipe wall only, no body. Returns velocity field."""
    N = cfg['N']; tau = cfg['tau']
    e = jnp.array(E, dtype=jnp.float32)
    w = jnp.array(W, dtype=jnp.float32)

    def step(f):
        rho = jnp.sum(f, axis=-1)
        u = (f @ e) / jnp.maximum(rho[..., None], 1e-10)
        edu = u @ e.T
        usq = jnp.sum(u**2, axis=-1, keepdims=True)
        feq = w * rho[..., None] * (1+edu/CS2+edu**2/(2*CS4)-usq/(2*CS2))
        fp = f - (f - feq) / tau
        fs = fp.reshape(N**3, Q)[cfg['stream_idx'], jnp.arange(Q)].reshape(N, N, N, Q)
        fs = apply_bounce_back(fs, fp, cfg['pipe_missing'], cfg['pipe_wall'],
                               wall_velocity=pwv)
        fs = fs.at[:,:,0,:].set(fz0)
        fs = fs.at[:,:,-1,:].set(fzN)
        fs = _apply_open_bc(fs, 0)
        fs = _apply_open_bc(fs, 1)
        return fs, u

    step_jit = jax.jit(step)
    f = init_equilibrium(N, N, N)
    for s in range(cfg['spinup']):
        f, u = step_jit(f)
    return u


def main():
    print(f"Device: {jax.devices()[0]}")
    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    for N_target in [48, 64]:
        print(f"\n{'='*80}")
        print(f"ONE-PASS BODY-SURFACE SCHWARZ: N={N_target}^3")
        print(f"{'='*80}")
        cfg = setup(N_target)
        N = cfg['N']; N_b = cfg['N_b']
        dx = cfg['dx']; dt = cfg['dt']
        center = jnp.zeros(3)
        e_eye = jnp.eye(3)

        R = np.zeros((6, 6))

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
            F_free, T_free = compute_force_torque(
                cfg['body_pts'], cfg['body_wts'], traction_0, center,
            )

            # Step 2: Evaluate u_free at pipe wall → LBM BCs
            pwv, fz0, fzN = bem_eval_wall_faces(cfg, traction_0)

            # Step 3: LBM no body
            t0 = time.time()
            u_lbm = run_lbm_no_body(cfg, pwv, fz0, fzN)
            lbm_time = time.time() - t0

            # Step 4: Interpolate u_wall at 320 body surface points
            u_wall_at_body = interpolate_velocity(
                u_lbm, cfg['body_ib_idx'], cfg['body_ib_wts'],
            ) * dx / dt  # (N_b, 3) physical

            # Diagnostics
            u_wall_mean = np.mean(np.array(u_wall_at_body), axis=0)
            u_wall_std = np.std(np.array(u_wall_at_body), axis=0)

            # Step 5: BEM re-solve with corrected velocity
            u_effective = u_body - u_wall_at_body  # (N_b, 3)
            traction_conf = jax.scipy.linalg.lu_solve(
                (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
                u_effective.ravel(),
            ).reshape(N_b, 3)

            # Step 6: Confined drag
            F_conf, T_conf = compute_force_torque(
                cfg['body_pts'], cfg['body_wts'], traction_conf, center,
            )
            R[:3, col] = np.array(F_conf)
            R[3:, col] = np.array(T_conf)

            diag_val = float(R[col, col])
            err = abs(diag_val - nn_diag[col]) / abs(nn_diag[col]) * 100

            comp = col if col < 3 else col - 3
            print(f"  col {col} ({LABELS[col]}): "
                  f"F_free={float(F_free[comp]) if col<3 else float(T_free[comp]):.4f}, "
                  f"F_conf={diag_val:.4f}, err={err:.1f}%, LBM={lbm_time:.1f}s")
            print(f"    u_wall mean=[{u_wall_mean[0]:.5f},{u_wall_mean[1]:.5f},{u_wall_mean[2]:.5f}]")
            print(f"    u_wall std =[{u_wall_std[0]:.5f},{u_wall_std[1]:.5f},{u_wall_std[2]:.5f}]")

        diag = [float(R[i, i]) for i in range(6)]
        errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]

        print(f"\n{'='*60}")
        print(f"N={N_target}^3 BODY-SURFACE SCHWARZ RESULTS")
        print(f"{'='*60}")
        print(f"{'':>6}  {'Schwarz':>10}  {'NN-BEM':>10}  {'error':>8}")
        print("-" * 44)
        all_pass = True
        for i in range(6):
            ok = "PASS" if errs[i] < 5 else ("WARN" if errs[i] < 10 else "FAIL")
            if errs[i] >= 5: all_pass = False
            print(f"{LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>6.1f}% [{ok}]")
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")

    print("\nDone. Tear down all cloud instances.")


if __name__ == "__main__":
    main()
