#!/usr/bin/env python3
"""Schwarz decomposition: F_confined = F_free(BEM) + F_wall(LBM-BB).

The body is a Bouzidi BB surface at rest (u=0) in the LBM.
The pipe wall has velocity u=-u_free to cancel the BEM free-space flow.
No IB spreading, no Peskin delta, no eval-sphere comparison.

F_wall extracted from momentum exchange at body surface.
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
    init_equilibrium, E, W, Q, OPP,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    compute_momentum_exchange_force,
    compute_momentum_exchange_torque,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM
EPS_NN = min(0.05, 0.02 * (R_CYL - A))
LABELS = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]


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
    """Set up the Schwarz decomposition LBM for a given resolution."""
    a = A; mu = MU; rho = RHO

    # Grid
    domain_extent = 2.5 * R_CYL
    dx = domain_extent / N_target
    N = int(np.ceil(domain_extent / dx))
    N = ((N + 7) // 8) * 8

    tau = 0.8
    nu_lu = (tau - 0.5) / 3.0
    nu_phys = mu / rho
    dt_lbm = nu_lu * dx**2 / nu_phys
    force_conv = dt_lbm**2 / (rho * dx**4)

    # BEM setup
    body_mesh = sphere_surface_mesh(radius=a, n_refine=2)
    body_pts = jnp.array(body_mesh.points)
    body_wts = jnp.array(body_mesh.weights)
    eps = body_mesh.mean_spacing / 2.0
    N_b = body_mesh.n_points

    A_bem = assemble_system_matrix(body_pts, body_wts, eps, mu)
    lu, piv = jax.scipy.linalg.lu_factor(A_bem)

    # Pipe wall mask (same as DefectCorrectionFluidNode)
    vessel_R_lu = R_CYL / dx
    cx, cy = N / 2.0, N / 2.0
    ix = jnp.arange(N, dtype=jnp.float32)
    iy = jnp.arange(N, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
    dist_2d = jnp.sqrt((gx - cx)**2 + (gy - cy)**2)
    pipe_wall = jnp.broadcast_to(
        (dist_2d >= vessel_R_lu)[..., None], (N, N, N),
    )
    pipe_missing = compute_missing_mask(pipe_wall)

    # Body wall mask (sphere at grid centre)
    body_R_lu = a / dx
    iz = jnp.arange(N, dtype=jnp.float32)
    gx3, gy3, gz3 = jnp.meshgrid(ix, iy, iz, indexing='ij')
    body_dist = jnp.sqrt((gx3 - N/2)**2 + (gy3 - N/2)**2 + (gz3 - N/2)**2)
    body_wall = body_dist <= body_R_lu
    body_missing = compute_missing_mask(body_wall)

    # Combined wall
    combined_wall = body_wall | pipe_wall
    combined_missing = compute_missing_mask(combined_wall)

    # Identify pipe wall fluid nodes (for BEM evaluation)
    pipe_fluid_mask = jnp.any(pipe_missing, axis=0)  # (N, N, N)
    n_pipe_fluid = int(jnp.sum(pipe_fluid_mask))

    # Body surface fluid nodes
    body_fluid_mask = jnp.any(body_missing, axis=0)  # (N, N, N)
    n_body_fluid = int(jnp.sum(body_fluid_mask))

    # Acoustic timescale for spinup
    c_s_lu = 1.0 / np.sqrt(3.0)
    spinup_steps = max(500, int(3.0 * vessel_R_lu / c_s_lu))

    print(f"  Grid: N={N}^3, dx={dx:.4f}, dt={dt_lbm:.6f}, tau={tau}")
    print(f"  Body: R={body_R_lu:.1f} lu ({N_b} BEM pts, eps={eps:.4f})")
    print(f"  Pipe: R={vessel_R_lu:.1f} lu")
    print(f"  Combined wall: {int(jnp.sum(combined_wall))} solid nodes")
    print(f"  Pipe wall fluid nodes: {n_pipe_fluid}")
    print(f"  Body surface fluid nodes: {n_body_fluid}")
    print(f"  Spinup: {spinup_steps} steps")

    return {
        'N': N, 'dx': dx, 'dt': dt_lbm, 'tau': tau,
        'body_pts': body_pts, 'body_wts': body_wts,
        'eps': eps, 'N_b': N_b,
        'lu': lu, 'piv': piv,
        'pipe_wall': pipe_wall, 'pipe_missing': pipe_missing,
        'body_wall': body_wall, 'body_missing': body_missing,
        'combined_wall': combined_wall, 'combined_missing': combined_missing,
        'pipe_fluid_mask': pipe_fluid_mask,
        'body_fluid_mask': body_fluid_mask,
        'spinup_steps': spinup_steps,
        'force_conv': force_conv,
        'body_R_lu': body_R_lu,
        'vessel_R_lu': vessel_R_lu,
    }


def compute_ladd_correction(cfg, u_wall_lu):
    """Precompute the Ladd wall velocity correction field.

    For each pipe-wall BB link, compute:
      correction[x, q] = 2 * w_q * (e_q · u_wall_lu[x]) / cs²

    Applied additively to f after Triton static BB.
    """
    N = cfg['N']
    pipe_missing = cfg['pipe_missing']  # (Q, N, N, N)

    e_arr = jnp.array(E, dtype=jnp.float32)  # (Q, 3)
    w_arr = jnp.array(W, dtype=jnp.float32)  # (Q,)
    opp_arr = jnp.array(OPP, dtype=jnp.int32)  # (Q,)
    cs2 = 1.0 / 3.0

    # e_q · u_wall at each node for each direction q
    e_dot_u = jnp.einsum('qa,xyza->xyzq', e_arr, u_wall_lu)  # (N, N, N, Q)

    # Raw correction
    raw = 2.0 * w_arr[None, None, None, :] * e_dot_u / cs2  # (N, N, N, Q)

    # Mask: only at pipe wall BB links (incoming convention)
    # pipe_missing[q, x, y, z] = True if neighbor in direction q is pipe wall
    # For pull-stream BB on direction q at node x: source is x - e_q
    # This is solid if pipe_missing[opp_q, x] = True (neighbor in direction opp_q is solid)
    # So the BB correction for incoming q is masked by pipe_missing[opp_q, ...]
    pipe_miss_incoming = pipe_missing[opp_arr]  # (Q, N, N, N) reindexed
    mask = pipe_miss_incoming.transpose(1, 2, 3, 0)  # (N, N, N, Q)

    correction = raw * mask
    return correction


def compute_wall_velocity(cfg, traction):
    """Evaluate BEM free-space velocity at pipe wall fluid nodes.

    Returns u_wall_lu: (N, N, N, 3) with -u_free in lattice units at pipe wall nodes.
    """
    N = cfg['N']
    dx = cfg['dx']
    dt = cfg['dt']
    pipe_fluid_mask = cfg['pipe_fluid_mask']

    # Extract pipe wall fluid node positions
    mask_np = np.array(pipe_fluid_mask)
    indices = np.argwhere(mask_np)  # (M, 3) integer indices
    positions_phys = (indices.astype(float) - N / 2) * dx  # (M, 3) physical coords

    print(f"  Evaluating BEM velocity at {len(indices)} pipe wall nodes...", end="", flush=True)
    t0 = time.time()

    # BEM evaluation (vectorised)
    u_free = evaluate_velocity_field(
        jnp.array(positions_phys),
        cfg['body_pts'], cfg['body_wts'],
        traction, cfg['eps'], MU,
    )  # (M, 3) physical velocity

    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")

    # Convert to lattice units and negate
    u_free_lu = np.array(u_free) * dt / dx
    print(f"  |u_free_lu|_max = {np.max(np.abs(u_free_lu)):.6f} "
          f"(Ma = {np.max(np.abs(u_free_lu)) * np.sqrt(3):.4f})")

    # Scatter into grid
    u_wall_lu = np.zeros((N, N, N, 3), dtype=np.float32)
    u_wall_lu[indices[:, 0], indices[:, 1], indices[:, 2]] = -u_free_lu.astype(np.float32)

    return jnp.array(u_wall_lu)


def run_schwarz_lbm(cfg, ladd_correction, n_steps):
    """Run the wall-correction LBM with body BB and pipe wall velocity.

    Returns (f_final, u_final, f_pre_last) for momentum exchange.
    """
    N = cfg['N']
    tau = cfg['tau']
    combined_wall = cfg['combined_wall']
    combined_missing = cfg['combined_missing']
    open_axis = 2

    from mime.nodes.environment.lbm.pallas_lbm import _apply_open_bc

    # Try Triton path
    try:
        from mime.nodes.environment.lbm.triton_kernels import (
            TRITON_AVAILABLE, lbm_full_step_triton, _get_d3q19_jax,
        )
        if TRITON_AVAILABLE:
            _get_d3q19_jax()
            N_flat = N ** 3
            combined_missing_flat = combined_missing.reshape(Q * N_flat).astype(jnp.int32)
            force_zero = jnp.zeros((N, N, N, 3), dtype=jnp.float32)

            @jax.jit
            def step(f, _ladd):
                f_out, u = lbm_full_step_triton(
                    f, force_zero, tau, combined_wall, combined_missing_flat, None,
                )
                f_out = f_out + _ladd
                for ax in range(3):
                    f_out = _apply_open_bc(f_out, ax)
                return f_out, u

            f = init_equilibrium(N, N, N)
            for s in range(n_steps):
                f_pre = f
                f, u = step(f, ladd_correction)

            return f, u, f_pre

    except ImportError:
        pass

    # Fallback: JAX path
    from mime.nodes.environment.lbm.pallas_lbm import lbm_full_step_pallas
    force_zero = jnp.zeros((N, N, N, 3), dtype=jnp.float32)

    @jax.jit
    def step_jax(f, _ladd):
        f_out, u = lbm_full_step_pallas(
            f, force_zero, tau, combined_wall, combined_missing, None,
        )
        f_out = f_out + _ladd
        for ax in range(3):
            f_out = _apply_open_bc(f_out, ax)
        return f_out, u

    f = init_equilibrium(N, N, N)
    for s in range(n_steps):
        f_pre = f
        f, u = step_jax(f, ladd_correction)

    return f, u, f_pre


def extract_wall_force_torque(cfg, f_pre, f_post):
    """Extract F_wall and T_wall from momentum exchange at body surface."""
    N = cfg['N']
    dx = cfg['dx']
    dt = cfg['dt']

    # Force in lattice units
    F_lu = compute_momentum_exchange_force(
        f_pre, f_post, cfg['body_missing'],
    )

    # Torque in lattice units (about grid centre)
    body_centre_lu = jnp.array([N / 2.0, N / 2.0, N / 2.0])
    T_lu = compute_momentum_exchange_torque(
        f_pre, f_post, cfg['body_missing'], body_centre_lu,
    )

    # Convert to physical units
    # F_phys = F_lu * rho * dx^4 / dt^2 (momentum exchange convention)
    F_phys = F_lu * RHO * dx**4 / dt**2
    T_phys = T_lu * RHO * dx**5 / dt**2

    return F_phys, T_phys


def main():
    print(f"Device: {jax.devices()[0]}")

    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    resolutions = [48]

    for N_target in resolutions:
        print(f"\n{'='*80}")
        print(f"SCHWARZ DECOMPOSITION: N={N_target}^3")
        print(f"{'='*80}")

        cfg = setup_schwarz(N_target)
        N = cfg['N']
        N_b = cfg['N_b']
        center = jnp.zeros(3)
        e = jnp.eye(3)

        R = np.zeros((6, 6))

        for col in range(6):
            U = e[col] if col < 3 else jnp.zeros(3)
            omega = e[col - 3] if col >= 3 else jnp.zeros(3)

            r = cfg['body_pts'] - center
            u_body = U + jnp.cross(omega, r)

            print(f"\n--- Column {col} ({LABELS[col]}) ---")

            # Step 1: BEM free-space solve
            traction = jax.scipy.linalg.lu_solve(
                (jnp.array(cfg['lu']), jnp.array(cfg['piv'])),
                u_body.ravel(),
            ).reshape(N_b, 3)
            F_free, T_free = compute_force_torque(
                cfg['body_pts'], cfg['body_wts'], traction, center,
            )
            print(f"  F_free = [{float(F_free[0]):.4f}, {float(F_free[1]):.4f}, {float(F_free[2]):.4f}]")
            print(f"  T_free = [{float(T_free[0]):.4f}, {float(T_free[1]):.4f}, {float(T_free[2]):.4f}]")

            # Step 2: Evaluate u_free at pipe wall → wall velocity
            u_wall_lu = compute_wall_velocity(cfg, traction)

            # Step 3: Precompute Ladd correction
            ladd_correction = compute_ladd_correction(cfg, u_wall_lu)
            print(f"  |Ladd correction|_max = {float(jnp.max(jnp.abs(ladd_correction))):.6f}")

            # Step 4: Run wall-correction LBM
            print(f"  Running LBM ({cfg['spinup_steps']} steps)...", end="", flush=True)
            t0 = time.time()
            f_post, u_lbm, f_pre = run_schwarz_lbm(
                cfg, ladd_correction, cfg['spinup_steps'],
            )
            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s")

            # Step 5: Momentum exchange → F_wall, T_wall
            F_wall, T_wall = extract_wall_force_torque(cfg, f_pre, f_post)
            print(f"  F_wall = [{float(F_wall[0]):.4f}, {float(F_wall[1]):.4f}, {float(F_wall[2]):.4f}]")
            print(f"  T_wall = [{float(T_wall[0]):.4f}, {float(T_wall[1]):.4f}, {float(T_wall[2]):.4f}]")

            # Step 6: Confined drag = free + wall
            F_conf = np.array(F_free) + np.array(F_wall)
            T_conf = np.array(T_free) + np.array(T_wall)
            print(f"  F_conf = [{F_conf[0]:.4f}, {F_conf[1]:.4f}, {F_conf[2]:.4f}]")
            print(f"  T_conf = [{T_conf[0]:.4f}, {T_conf[1]:.4f}, {T_conf[2]:.4f}]")

            R[:3, col] = F_conf
            R[3:, col] = T_conf

        # Results
        diag = [float(R[i, i]) for i in range(6)]
        errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]

        print(f"\n{'='*60}")
        print(f"N={N_target}^3 SCHWARZ DECOMPOSITION RESULTS")
        print(f"{'='*60}")
        print(f"{'':>6}  {'Schwarz':>10}  {'NN-BEM':>10}  {'error':>8}")
        print("-" * 44)
        all_pass = True
        for i in range(6):
            ok = "PASS" if errs[i] < 5 else "FAIL"
            if errs[i] >= 5:
                all_pass = False
            print(f"{LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>6.1f}% [{ok}]")
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")

        # Decomposition check
        print(f"\nDecomposition check (F_x column):")
        print(f"  F_free[0] = {float(R[0,0]) - float(np.array(F_wall)[0]):.4f}")
        print(f"  F_wall[0] = {float(np.array(F_wall)[0]):.4f}")
        print(f"  F_total   = {float(R[0,0]):.4f}")
        print(f"  NN-BEM    = {nn_diag[0]:.4f}")


if __name__ == "__main__":
    main()
