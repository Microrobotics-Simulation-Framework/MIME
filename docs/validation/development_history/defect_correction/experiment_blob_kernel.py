#!/usr/bin/env python3
"""Experiment 1: Free-space IB-BEM mismatch with matched blob kernel.

Replace the Peskin 4-point delta (tensor product, grid-aligned) with the
Cortez regularised Stokeslet blob (isotropic). The blob function is
EXACTLY the one used in the BEM solver, so the IB-spread velocity field
should match the BEM evaluation at eval spheres in ALL directions.

φ_ε(r) = 15ε⁴ / (8π(r² + ε²)^(7/2))

Tests:
1. Free-space mismatch diagnostic (blob vs Peskin) for F_x, F_z, T_x
2. LBM stability check (2000 steps)
3. Full R matrix with single method (Richardson, no per-direction dispatch)
4. Resolution convergence (48³ and 64³)
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
from mime.nodes.environment.defect_correction import DefectCorrectionFluidNode
from mime.nodes.environment.defect_correction.wall_correction import (
    wall_correction_richardson,
)
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field
from mime.nodes.environment.lbm.d3q19 import init_equilibrium
from mime.nodes.environment.lbm.immersed_boundary import (
    precompute_ib_stencil,
    spread_forces,
    interpolate_velocity,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

A = 1.0; MU = 1.0; RHO = 1.0; LAM = 0.3
R_CYL = A / LAM
EPS_NN = min(0.05, 0.02 * (R_CYL - A))
LABELS = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]


# ── Blob kernel ─────────────────────────────────────────────────────

def cortez_blob(r, epsilon):
    """Cortez regularised Stokeslet blob function.

    φ_ε(r) = 15ε⁴ / (8π(r² + ε²)^(7/2))

    Normalised: ∫ φ_ε(r) d³r = 1.
    Units: 1/length³ (r and ε in physical units).
    """
    eps4 = epsilon**4
    denom = (r**2 + epsilon**2) ** 3.5
    return 15.0 * eps4 / (8.0 * np.pi * denom)


def precompute_ib_stencil_blob(
    lagrangian_points_lu, grid_shape, dx, epsilon,
    support_half=3, weight_threshold=1e-14,
):
    """Precompute IB stencil using the Cortez blob kernel.

    For each Lagrangian point, finds grid nodes within ±support_half
    and computes blob weights. Weights are normalised to sum to 1.

    Parameters
    ----------
    lagrangian_points_lu : (N_lag, 3) positions in lattice units
    grid_shape : (nx, ny, nz)
    dx : physical grid spacing
    epsilon : BEM regularisation parameter (physical units)
    support_half : half-width of stencil in lattice units

    Returns
    -------
    stencil_indices : (N_lag, n_stencil) int32
    stencil_weights : (N_lag, n_stencil) float64
    """
    N_lag = len(lagrangian_points_lu)
    nx, ny, nz = grid_shape
    width = 2 * support_half + 1
    n_stencil = width ** 3

    # Stencil offsets
    offsets_1d = np.arange(-support_half, support_half + 1, dtype=np.int32)
    di, dj, dk = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
    di = di.ravel()  # (n_stencil,)
    dj = dj.ravel()
    dk = dk.ravel()

    # Reference node (nearest grid point)
    ref = np.round(lagrangian_points_lu).astype(np.int32)  # (N_lag, 3)

    # Grid indices with periodic wrapping
    si = (ref[:, 0:1] + di[None, :]) % nx  # (N_lag, n_stencil)
    sj = (ref[:, 1:2] + dj[None, :]) % ny
    sk = (ref[:, 2:3] + dk[None, :]) % nz

    stencil_indices = (si * ny * nz + sj * nz + sk).astype(np.int32)

    # Physical distance from each Lagrangian point to each stencil node
    # Distance in lattice units
    rx = lagrangian_points_lu[:, 0:1] - (ref[:, 0:1] + di[None, :])  # (N_lag, n_stencil)
    ry = lagrangian_points_lu[:, 1:2] - (ref[:, 1:2] + dj[None, :])
    rz = lagrangian_points_lu[:, 2:3] - (ref[:, 2:3] + dk[None, :])

    # Convert to physical distance
    r_phys = np.sqrt((rx * dx)**2 + (ry * dx)**2 + (rz * dx)**2)

    # Blob weights: φ_ε(r_phys) × dx³
    raw_weights = cortez_blob(r_phys, epsilon) * dx**3

    # Zero out negligible weights
    raw_weights[raw_weights < weight_threshold] = 0.0

    # Normalise each Lagrangian point's weights to sum to 1
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-30] = 1.0  # avoid division by zero
    stencil_weights = raw_weights / row_sums

    # Report statistics
    total_captured = raw_weights.sum(axis=1).mean()
    nonzero_per_point = (stencil_weights > 0).sum(axis=1).mean()
    print(f"  Blob stencil: support=±{support_half} ({n_stencil} nodes), "
          f"avg {nonzero_per_point:.0f} nonzero weights, "
          f"pre-normalisation sum={total_captured:.6f}")

    return stencil_indices.astype(np.int32), stencil_weights.astype(np.float64)


# ── Helpers ──────────────────────────────────────────────────────────

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


def make_node(N_target):
    dx = 2 * R_CYL / (N_target * 0.8)
    body = sphere_surface_mesh(radius=A, n_refine=2)
    return DefectCorrectionFluidNode(
        "dc", timestep=0.001, mu=MU, rho=RHO,
        body_mesh=body, body_radius=A,
        vessel_radius=R_CYL, dx=dx,
        open_bc_axis=2, max_defect_iter=25, alpha=0.3,
    )


# ── Experiment 1: Free-space mismatch diagnostic ────────────────────

def experiment_1_mismatch(node):
    """Compare IB-BEM mismatch for Peskin vs blob kernel."""
    print(f"\n{'='*90}")
    print("EXPERIMENT 1: Free-space IB-BEM mismatch — Peskin vs Blob")
    print(f"{'='*90}")

    N = node._nx
    N_b = node._N_body
    center = jnp.zeros(3)
    e = jnp.eye(3)
    dx = node._dx
    eps = node._epsilon

    # Precompute blob stencil for body
    body_pts_lu = np.array(node._body_pts_lu)
    print(f"\nBlob kernel (ε={eps:.4f}, dx={dx:.4f}):")
    blob_idx, blob_wts = precompute_ib_stencil_blob(
        body_pts_lu, (N, N, N), dx, eps,
    )
    blob_idx_j = jnp.array(blob_idx)
    blob_wts_j = jnp.array(blob_wts)

    # Precompute blob stencils for eval spheres
    R_factors = [1.15, 1.5, 2.0, 3.0]
    eval_blob_stencils = []
    eval_peskin_stencils = []
    for rf in R_factors:
        R_ev = rf * A
        ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
        ev_lu = ev_mesh.points / dx + np.array([N / 2] * 3)

        # Blob stencil for eval sphere
        print(f"  Eval R={rf:.2f}a:")
        eb_idx, eb_wts = precompute_ib_stencil_blob(ev_lu, (N, N, N), dx, eps)

        # Peskin stencil for eval sphere
        ep_idx, ep_wts = precompute_ib_stencil(ev_lu, (N, N, N))

        eval_blob_stencils.append({
            'pts_phys': jnp.array(ev_mesh.points),
            'idx': jnp.array(eb_idx),
            'wts': jnp.array(eb_wts),
        })
        eval_peskin_stencils.append({
            'pts_phys': jnp.array(ev_mesh.points),
            'idx': jnp.array(ep_idx),
            'wts': jnp.array(ep_wts),
        })

    print(f"\nPeskin 4-pt stencil: 4³ = 64 nodes per point")

    # Test columns: F_x (transverse), F_z (axial), T_x (rotation)
    test_cols = [(0, "F_x"), (2, "F_z"), (3, "T_x")]

    # Run LBM 500 steps with each kernel and compare
    n_steps = 500

    results = {}

    for col, label in test_cols:
        U = e[col] if col < 3 else jnp.zeros(3)
        omega = e[col - 3] if col >= 3 else jnp.zeros(3)

        r = node._body_pts - center
        u_body = U + jnp.cross(omega, r)
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)

        # Force conversion
        point_forces = traction * node._body_wts[:, None] * node._force_conv

        print(f"\n--- {label} ---")

        for kernel_name, body_idx, body_wts_arr, eval_stencils in [
            ("Peskin", node._ib_idx, node._ib_wts, eval_peskin_stencils),
            ("Blob", blob_idx_j, blob_wts_j, eval_blob_stencils),
        ]:
            # Spread forces
            force_field = spread_forces(point_forces, body_idx, body_wts_arr, (N, N, N))

            # Run walled LBM for n_steps
            f_lbm = init_equilibrium(N, N, N)
            for s in range(n_steps):
                f_lbm, u_lbm = node._lbm_full_step(f_lbm, force_field)

            # Measure mismatch at each eval radius
            row = f"  {kernel_name:>8}:"
            for r_idx, rf in enumerate(R_factors):
                es = eval_stencils[r_idx]

                # IB-LBM velocity at eval sphere (using MATCHED interpolation kernel)
                u_ib = interpolate_velocity(u_lbm, es['idx'], es['wts'])
                u_ib = u_ib * dx / node._dt_lbm

                # BEM velocity at eval sphere
                u_bem = evaluate_velocity_field(
                    es['pts_phys'], node._body_pts, node._body_wts,
                    traction, eps, MU,
                )

                # Component-wise mismatch for the motion direction
                comp = col if col < 3 else 0  # for rotation, check x-component
                u_ib_mean = float(jnp.mean(u_ib[:, comp]))
                u_bem_mean = float(jnp.mean(u_bem[:, comp]))

                if abs(u_bem_mean) > 1e-10:
                    mismatch = abs(u_ib_mean / u_bem_mean - 1.0) * 100
                else:
                    mismatch = 0.0

                row += f"  R={rf:.2f}a: {mismatch:>5.1f}%"

                key = (label, kernel_name, rf)
                results[key] = mismatch

            print(row, flush=True)

    # Summary table
    print(f"\n{'='*90}")
    print("MISMATCH SUMMARY (% difference in motion-direction component)")
    print(f"{'='*90}")
    header = f"{'':>12}"
    for rf in R_factors:
        header += f"  {'R=%.2fa' % rf:>12}"
    print(header)
    print("-" * (14 + 14 * len(R_factors)))

    for label in ["F_x", "F_z", "T_x"]:
        for kernel in ["Peskin", "Blob"]:
            row = f"  {label:>4} {kernel:>6}:"
            for rf in R_factors:
                mm = results.get((label, kernel, rf), float('nan'))
                row += f"  {mm:>10.1f}%"
            print(row)
        print()

    return results


# ── Experiment 2: LBM stability ─────────────────────────────────────

def experiment_2_stability(node):
    """Check LBM stability with blob-spread forces."""
    print(f"\n{'='*90}")
    print("EXPERIMENT 2: LBM stability with blob kernel")
    print(f"{'='*90}")

    N = node._nx
    N_b = node._N_body
    dx = node._dx
    eps = node._epsilon

    body_pts_lu = np.array(node._body_pts_lu)
    blob_idx, blob_wts = precompute_ib_stencil_blob(
        body_pts_lu, (N, N, N), dx, eps,
    )
    blob_idx_j = jnp.array(blob_idx)
    blob_wts_j = jnp.array(blob_wts)

    # Unit x-translation traction
    center = jnp.zeros(3)
    e = jnp.eye(3)
    u_body = jnp.broadcast_to(e[0], (N_b, 3))
    traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)
    point_forces = traction * node._body_wts[:, None] * node._force_conv

    force_field = spread_forces(point_forces, blob_idx_j, blob_wts_j, (N, N, N))

    f_lbm = init_equilibrium(N, N, N)
    n_steps = 2000

    print(f"{'step':>6}  {'rho_min':>10}  {'rho_max':>10}  {'|u|_max':>10}")
    print("-" * 44)

    for step in range(1, n_steps + 1):
        f_lbm, u_lbm = node._lbm_full_step(f_lbm, force_field)

        if step % 200 == 0 or step <= 10:
            rho = jnp.sum(f_lbm, axis=-1)
            rho_min = float(jnp.min(rho))
            rho_max = float(jnp.max(rho))
            u_max = float(jnp.max(jnp.linalg.norm(u_lbm, axis=-1)))
            print(f"{step:6d}  {rho_min:>10.6f}  {rho_max:>10.6f}  {u_max:>10.6f}",
                  flush=True)

            if rho_min < 0.9 or rho_max > 1.1 or u_max > 0.5:
                print("  UNSTABLE! Aborting.")
                return False

    print("  STABLE: rho bounded, no oscillations.")
    return True


# ── Experiment 3: Full R matrix with single method ──────────────────

def experiment_3_single_method(node, nn_diag):
    """Full R matrix with blob kernel + single Richardson method."""
    print(f"\n{'='*90}")
    print("EXPERIMENT 3: Full R matrix with blob kernel + single method")
    print(f"{'='*90}")

    N = node._nx
    N_b = node._N_body
    dx = node._dx
    eps = node._epsilon
    center = jnp.zeros(3)
    e = jnp.eye(3)
    warmstart = 200

    # Blob stencil for body
    body_pts_lu = np.array(node._body_pts_lu)
    blob_idx, blob_wts = precompute_ib_stencil_blob(
        body_pts_lu, (N, N, N), dx, eps,
    )
    blob_idx_j = jnp.array(blob_idx)
    blob_wts_j = jnp.array(blob_wts)

    # Blob stencils for eval spheres (close radii for Richardson)
    eval_stencils_close_blob = []
    d_vals_close = []
    for R_factor in [1.15, 1.2, 1.3]:
        R_ev = R_factor * A
        ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
        ev_lu = ev_mesh.points / dx + np.array([N / 2] * 3)
        eb_idx, eb_wts = precompute_ib_stencil_blob(ev_lu, (N, N, N), dx, eps)
        eval_stencils_close_blob.append({
            'pts_phys': jnp.array(ev_mesh.points),
            'idx': jnp.array(eb_idx),
            'wts': jnp.array(eb_wts),
        })
        d_vals_close.append(R_factor - 1.0)
    d_vals_close = jnp.array(d_vals_close)

    def richardson_blob(u_lbm, traction):
        """Richardson wall correction using blob interpolation."""
        return wall_correction_richardson(
            u_lbm, traction, node._body_pts, node._body_wts,
            eval_stencils_close_blob, d_vals_close,
            eps, MU, dx, node._dt_lbm,
        )

    # Full 6x6 R matrix
    R = np.zeros((6, 6))

    for col in range(6):
        U = e[col] if col < 3 else jnp.zeros(3)
        omega = e[col - 3] if col >= 3 else jnp.zeros(3)

        r = node._body_pts - center
        u_body = U + jnp.cross(omega, r)
        traction = node._bem_solve(u_body.ravel()).reshape(N_b, 3)

        # Spread via blob
        point_forces = traction * node._body_wts[:, None] * node._force_conv
        force_field = spread_forces(point_forces, blob_idx_j, blob_wts_j, (N, N, N))

        f_lbm = init_equilibrium(N, N, N)
        f_lbm, u_lbm = node._run_lbm_fixed(f_lbm, force_field, node._spinup_steps)

        n_iter = node._max_defect_iter if col < 3 else 2
        alpha_col = 0.3
        prev_drag = 0.0

        for iteration in range(n_iter):
            delta_u = richardson_blob(u_lbm, traction)
            delta_u_body = jnp.broadcast_to(delta_u, (N_b, 3))
            traction_new = node._bem_solve(
                (u_body - delta_u_body).ravel()
            ).reshape(N_b, 3)
            traction = (1 - alpha_col) * traction + alpha_col * traction_new

            # Re-spread with blob
            point_forces = traction * node._body_wts[:, None] * node._force_conv
            force_field = spread_forces(point_forces, blob_idx_j, blob_wts_j, (N, N, N))
            f_lbm, u_lbm = node._run_lbm_fixed(f_lbm, force_field, warmstart)

            F_iter, T_iter = compute_force_torque(
                node._body_pts, node._body_wts, traction, center,
            )
            drag_diag = float(F_iter[col]) if col < 3 else float(T_iter[col - 3])
            rel_change = abs(drag_diag - prev_drag) / (abs(drag_diag) + 1e-30)
            prev_drag = drag_diag

            if iteration > 0 and rel_change < node._tol:
                logger.info("  col %d converged iter %d: %.4f", col, iteration + 1, drag_diag)
                break

        F, T = compute_force_torque(node._body_pts, node._body_wts, traction, center)
        R[:3, col] = np.array(F)
        R[3:, col] = np.array(T)
        logger.info("col %d: F=[%.2f,%.2f,%.2f] T=[%.2f,%.2f,%.2f]",
                    col, *[float(x) for x in F], *[float(x) for x in T])

    diag = [float(R[i, i]) for i in range(6)]
    errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100 for i in range(6)]

    print(f"\n{'':>6}  {'Blob+Rich':>10}  {'NN-BEM':>10}  {'error':>8}")
    print("-" * 44)
    all_pass = True
    for i in range(6):
        ok = "PASS" if errs[i] < 5 else "FAIL"
        if errs[i] >= 5:
            all_pass = False
        print(f"{LABELS[i]:>6}  {diag[i]:>10.2f}  {nn_diag[i]:>10.2f}  {errs[i]:>6.1f}% [{ok}]")
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")

    return R, diag, errs, all_pass


# ── Experiment 4: Resolution convergence ─────────────────────────────

def experiment_4_convergence(nn_diag):
    """Resolution convergence at 48 and 64 with blob + Richardson."""
    print(f"\n{'='*90}")
    print("EXPERIMENT 4: Resolution convergence with blob kernel")
    print(f"{'='*90}")

    resolutions = [48, 64]
    all_results = {}

    for N_target in resolutions:
        node = make_node(N_target)
        print(f"\n--- N={N_target}^3 ---")
        _, diag, errs, _ = experiment_3_single_method(node, nn_diag)
        all_results[N_target] = {"diag": diag, "errs": errs}

    print(f"\n{'':>6}", end="")
    for N in resolutions:
        print(f"  {f'{N}^3':>14}", end="")
    print(f"  {'NN-BEM':>10}")
    print("-" * 52)
    for i in range(6):
        row = f"{LABELS[i]:>6}"
        for N in resolutions:
            r = all_results[N]
            row += f"  {r['diag'][i]:>6.1f}({r['errs'][i]:>4.1f}%)"
        row += f"  {nn_diag[i]:>10.2f}"
        print(row)

    for col, name in [(0, "F_x"), (2, "F_z"), (5, "T_z")]:
        e48 = all_results[48]["errs"][col]
        e64 = all_results[64]["errs"][col]
        status = "CONVERGING" if e64 < e48 else "ANTI"
        print(f"  {name}: {e48:.1f}% -> {e64:.1f}% [{status}]")


def main():
    print(f"Device: {jax.devices()[0]}")

    nn_diag = compute_nn_bem_reference()
    print(f"NN-BEM: F_x={nn_diag[0]:.4f}, F_z={nn_diag[2]:.4f}, T_z={nn_diag[5]:.4f}\n")

    node = make_node(48)
    print(f"Node: N={node._nx}^3, dx={node._dx:.4f}, eps={node._epsilon:.4f}")

    # Experiment 1: mismatch diagnostic (gate for all subsequent experiments)
    mismatch_results = experiment_1_mismatch(node)

    # Check gate: did blob reduce mismatch to <10% for both F_x and F_z?
    fx_blob = mismatch_results.get(("F_x", "Blob", 1.15), 999)
    fz_blob = mismatch_results.get(("F_z", "Blob", 1.15), 999)
    print(f"\nGate check: F_x blob mismatch = {fx_blob:.1f}%, F_z blob mismatch = {fz_blob:.1f}%")

    if fx_blob > 10 or fz_blob > 10:
        print("GATE FAILED: Blob mismatch too large. Per-direction dispatch needed.")
        print("Skipping experiments 2-4.")
        return

    print("GATE PASSED: Proceeding to experiments 2-4.\n")

    # Experiment 2: stability
    stable = experiment_2_stability(node)
    if not stable:
        print("STABILITY FAILED. Aborting.")
        return

    # Experiment 3: single-method R matrix
    R, diag, errs, all_pass = experiment_3_single_method(node, nn_diag)

    if not all_pass:
        print("\nSingle-method FAILED with Richardson. Stopping here.")
        return

    # Experiment 4: resolution convergence
    experiment_4_convergence(nn_diag)


if __name__ == "__main__":
    main()
