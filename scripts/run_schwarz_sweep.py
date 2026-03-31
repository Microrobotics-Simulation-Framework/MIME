#!/usr/bin/env python3
"""VER-029: Schwarz BEM-LBM drag multiplier sweep across confinement ratios.

Runs the Schwarz-coupled BEM+LBM system for the d2.8 UMR at 5 confinement
ratios, extracts rotational drag multipliers, and compares against T2.6b
LBM reference values.

The dx is set from the interface sphere radius (fixed physical size) to
ensure consistent interface resolution across all κ values. The domain
size (N_lattice) adapts per κ to fit the vessel.

Usage:
    python3 scripts/run_schwarz_sweep.py [--output results.json]
"""

import argparse
import json
import os
import sys
import time
import traceback

# XLA compilation flags
os.environ.setdefault("XLA_FLAGS", " ".join([
    "--xla_gpu_autotune_level=1",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_autotune_max_solutions=4",
]))

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_mime")

import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from mime.nodes.environment.stokeslet.surface_mesh import (
    sdf_surface_mesh, sphere_surface_mesh,
)
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.stokeslet.bem import (
    assemble_system_matrix, compute_force_torque,
)
from mime.nodes.environment.stokeslet.nearest_neighbour import (
    compute_nearest_neighbour_map, assemble_nn_confined_system,
)
from mime.nodes.environment.lbm.far_field_node import LBMFarFieldNode
from mime.nodes.robot.helix_geometry import umr_sdf


# ── Constants ──────────────────────────────────────────────────────────

# T2.6b reference (Bouzidi+FSI, 192³, from umr_training_v2_bouzidi.h5)
T26B_TORQUE_LU = {
    0.15: 87.82,
    0.22: 100.97,
    0.30: 108.54,
    0.35: 112.59,
    0.40: 124.30,
}

UMR_GEOM_MM = dict(
    body_radius=0.87, body_length=4.1, cone_length=1.9,
    cone_end_radius=0.255, fin_outer_radius=1.42,
    fin_length=2.03, fin_width=0.55, fin_thickness=0.15,
    helix_pitch=8.0,
)

FIN_R_M = 1.42e-3       # m
MU = 0.69e-3             # Pa·s
RHO = 997.0              # kg/m³
IFACE_FACTOR = 2.0       # interface sphere = factor × fin_R
MIN_NODES_PER_IFACE_R = 12  # minimum lattice nodes across interface radius
TAU = 0.8


# ── Helper functions ───────────────────────────────────────────────────

def generate_umr_mesh(mc_resolution=24):
    """Generate UMR surface mesh via marching cubes on SDF."""
    def sdf_func(pts):
        return umr_sdf(pts, **UMR_GEOM_MM)
    mesh_mm = sdf_surface_mesh(sdf_func, (-2, -2, -4), (2, 2, 4),
                                mc_resolution=mc_resolution)
    return (
        jnp.array(mesh_mm.points * 1e-3),
        jnp.array(mesh_mm.weights * 1e-6),
        mesh_mm.n_points,
    )


def compute_domain_params(kappa, iface_R):
    """Compute LBM domain parameters for a given confinement ratio.

    dx is set from the interface sphere radius to ensure consistent
    resolution. Domain size adapts to fit the vessel.
    """
    vessel_R_m = FIN_R_M / kappa
    gap_m = vessel_R_m - iface_R

    # dx from interface sphere: iface_R = MIN_NODES lu
    dx = iface_R / MIN_NODES_PER_IFACE_R

    # Domain extent: 2.5× vessel radius (vessel + buffer)
    domain_extent = 2.5 * vessel_R_m
    N_lattice = int(np.ceil(domain_extent / dx))
    # Round up to next multiple of 8 for GPU efficiency
    N_lattice = ((N_lattice + 7) // 8) * 8

    vessel_R_lu = vessel_R_m / dx
    iface_R_lu = iface_R / dx
    gap_lu = vessel_R_lu - iface_R_lu

    nu_lu = (TAU - 0.5) / 3.0
    nu_phys = MU / RHO
    dt_lbm = nu_lu * dx**2 / nu_phys

    eps = min(0.05 * FIN_R_M, 0.02 * gap_m)

    # Diffusion time for spin-up scaling
    diff_time = gap_lu**2 / nu_lu
    n_spinup = max(500, int(diff_time * 3))

    return {
        "dx": dx, "N": N_lattice, "dt_lbm": dt_lbm,
        "vessel_R_lu": vessel_R_lu, "iface_R_lu": iface_R_lu,
        "gap_lu": gap_lu, "gap_m": gap_m, "eps": eps,
        "n_spinup": n_spinup,
    }


def compute_unconfined_resistance(body_pts, body_wts, eps):
    """Compute free-space 6×6 resistance matrix via 6 BEM solves."""
    center = jnp.zeros(3)
    e = jnp.eye(3)
    N_b = len(body_pts)

    A = assemble_system_matrix(body_pts, body_wts, eps, MU)
    lu_f, piv_f = jax.scipy.linalg.lu_factor(A)

    R = np.zeros((6, 6))
    for col in range(6):
        U = e[col] if col < 3 else jnp.zeros(3)
        omega = e[col - 3] if col >= 3 else jnp.zeros(3)
        r = body_pts - center
        u_body = U + jnp.cross(omega, r)
        rhs = u_body.ravel()

        sol = jax.scipy.linalg.lu_solve((lu_f, piv_f), rhs)
        trac = sol.reshape(N_b, 3)
        F, T = compute_force_torque(body_pts, body_wts, trac, center)
        R[:3, col] = np.array(F)
        R[3:, col] = np.array(T)

    return R


def run_schwarz_at_kappa(kappa, body_pts, body_wts, N_b, iface_R,
                          n_schwarz_iter=5, n_schwarz_lbm=100):
    """Run Schwarz-coupled BEM+LBM at a single confinement ratio.

    Returns result dict or None on failure.
    """
    params = compute_domain_params(kappa, iface_R)
    N = params["N"]
    dx = params["dx"]
    dt_lbm = params["dt_lbm"]
    eps = params["eps"]
    n_spinup = params["n_spinup"]

    print(f"  N={N}³, dx={dx*1e3:.4f}mm, vessel_R={params['vessel_R_lu']:.1f}lu, "
          f"iface_R={params['iface_R_lu']:.1f}lu, gap={params['gap_lu']:.1f}lu, "
          f"spinup={n_spinup}")

    # BEM: body + interface sphere
    iface = create_interface_mesh(center=(0, 0, 0), radius=iface_R, n_refine=2)
    iface_pts = jnp.array(iface.points)
    iface_wts = jnp.array(iface.weights)
    N_i = iface.n_points

    body_nn = compute_nearest_neighbour_map(
        np.array(body_pts), np.array(body_pts))
    iface_nn = compute_nearest_neighbour_map(
        np.array(iface.points), np.array(iface.points))
    A = assemble_nn_confined_system(
        body_pts, body_pts, body_wts, body_nn,
        iface_pts, iface_pts, iface_wts, iface_nn, eps, MU)
    lu, piv = jax.scipy.linalg.lu_factor(A)

    # LBM far-field
    center_lu = (N / 2, N / 2, N / 2)
    iface_lbm = sphere_surface_mesh(
        center=(N / 2 * dx,) * 3, radius=iface_R, n_refine=2)
    lbm = LBMFarFieldNode(
        "lbm", timestep=dt_lbm,
        nx=N, ny=N, nz=N, tau=TAU,
        vessel_radius_lu=params["vessel_R_lu"],
        interface_center_lu=center_lu,
        interface_radius_lu=params["iface_R_lu"],
        interface_points_physical=iface_lbm.points,
        dx_physical=dx, open_bc_axis=2,
    )

    center = jnp.zeros(3)
    e = jnp.eye(3)

    # LBM spin-up with a representative interface velocity (z-rotation)
    omega_init = jnp.array([0.0, 0.0, 1.0])
    u_iface_init = jnp.cross(omega_init, iface_pts) * (FIN_R_M / iface_R)**3
    u_iface_lu = u_iface_init * dt_lbm / dx

    lbm_s = lbm.initial_state()
    for step in range(n_spinup):
        lbm_s = lbm.update(lbm_s, {"interface_velocity": u_iface_lu}, dt_lbm)

    # Get the converged LBM background flow at interface
    # (run a few extra Schwarz iterations to settle)
    for si in range(n_schwarz_iter):
        bg = lbm_s["interface_background_velocity"] * dx / dt_lbm
        # Use z-rotation RHS for the settling iterations
        r = body_pts - center
        u_body_rot = jnp.cross(omega_init, r)
        rhs = jnp.concatenate([u_body_rot.ravel(), bg.ravel()])
        sol = jax.scipy.linalg.lu_solve((lu, piv), rhs)
        for step in range(n_schwarz_lbm):
            lbm_s = lbm.update(
                lbm_s, {"interface_velocity": u_iface_lu}, dt_lbm)

    # Now compute the full 6×6 resistance matrix using the converged
    # LBM background flow. Each column is a unit motion (3 translations
    # + 3 rotations). The LBM background is held fixed — it represents
    # the far-field wall effect which is independent of the unit motion
    # direction (linear Stokes superposition).
    bg_converged = lbm_s["interface_background_velocity"] * dx / dt_lbm

    R = np.zeros((6, 6))
    for col in range(6):
        U = e[col] if col < 3 else jnp.zeros(3)
        omega = e[col - 3] if col >= 3 else jnp.zeros(3)
        r = body_pts - center
        u_body = U + jnp.cross(omega, r)
        rhs = jnp.concatenate([u_body.ravel(), bg_converged.ravel()])

        sol = jax.scipy.linalg.lu_solve((lu, piv), rhs)
        body_trac = sol[:3 * N_b].reshape(N_b, 3)
        F, T = compute_force_torque(body_pts, body_wts, body_trac, center)
        R[:3, col] = np.array(F)
        R[3:, col] = np.array(T)

    return {
        "kappa": kappa,
        "R_matrix": R.tolist(),
        "T_z": float(R[5, 5]),       # rotational drag (z-axis)
        "F_z": float(R[2, 2]),       # translational drag (z-axis)
        "F_x": float(R[0, 0]),       # translational drag (transverse)
        "N_lattice": N,
        "dx_m": float(dx),
        "vessel_R_lu": float(params["vessel_R_lu"]),
        "iface_R_lu": float(params["iface_R_lu"]),
        "gap_lu": float(params["gap_lu"]),
        "eps": float(eps),
        "n_spinup": n_spinup,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VER-029 Schwarz sweep")
    parser.add_argument("--mc-resolution", type=int, default=24,
                        help="Marching cubes resolution for UMR mesh")
    parser.add_argument("--output", type=str, default="ver029_results.json",
                        help="Output JSON file")
    parser.add_argument("--kappas", type=str,
                        default="0.15,0.22,0.30,0.35,0.40",
                        help="Comma-separated confinement ratios")
    parser.add_argument("--iface-factor", type=float, default=IFACE_FACTOR,
                        help="Interface sphere radius = factor × fin_R")
    parser.add_argument("--schwarz-iter", type=int, default=5,
                        help="Schwarz iterations per configuration")
    args = parser.parse_args()

    kappas = [float(k) for k in args.kappas.split(",")]
    iface_R = args.iface_factor * FIN_R_M

    print("=" * 60)
    print("VER-029: Schwarz BEM-LBM Drag Multiplier Sweep")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"iface_R: {iface_R*1e3:.2f}mm ({args.iface_factor}× fin_R)")
    print(f"min_nodes_per_iface_R: {MIN_NODES_PER_IFACE_R}")
    print(f"dx: {iface_R/MIN_NODES_PER_IFACE_R*1e3:.4f}mm (fixed)")
    print(f"kappas: {kappas}")
    print()

    # Domain size preview
    print("Domain sizes:")
    for k in kappas:
        p = compute_domain_params(k, iface_R)
        print(f"  κ={k:.2f}: N={p['N']}³, vessel_R={p['vessel_R_lu']:.1f}lu, "
              f"gap={p['gap_lu']:.1f}lu, spinup={p['n_spinup']}")
    print()

    # Generate UMR mesh
    body_pts, body_wts, N_b = generate_umr_mesh(args.mc_resolution)
    print(f"UMR mesh: N={N_b}")

    # Unconfined reference (full 6×6 resistance matrix)
    eps_free = 0.05 * FIN_R_M
    R_free = compute_unconfined_resistance(body_pts, body_wts, eps_free)
    T_free = R_free[5, 5]
    F_free_z = R_free[2, 2]
    print(f"Unconfined R_FU_xx: {R_free[0,0]:.6e}")
    print(f"Unconfined R_FU_zz: {F_free_z:.6e}")
    print(f"Unconfined R_Tw_zz: {T_free:.6e}")
    print()

    # Sweep
    results = {
        "unconfined_R_matrix": R_free.tolist(),
        "unconfined_T_z": float(T_free),
        "unconfined_F_z": float(F_free_z),
        "iface_R_m": float(iface_R),
        "iface_factor": args.iface_factor,
        "min_nodes_per_iface_R": MIN_NODES_PER_IFACE_R,
        "runs": [],
    }
    t_total = time.time()

    for kappa in kappas:
        print(f"κ = {kappa:.2f}:")
        t0 = time.time()

        try:
            result = run_schwarz_at_kappa(
                kappa, body_pts, body_wts, N_b, iface_R,
                n_schwarz_iter=args.schwarz_iter,
            )
            if result is not None:
                result["elapsed_s"] = time.time() - t0
                result["mult_abs"] = result["T_z"] / T_free
                results["runs"].append(result)
                print(f"  T_z={result['T_z']:.4e}, mult={result['mult_abs']:.3f}x "
                      f"({result['elapsed_s']:.0f}s)")
            else:
                print(f"  SKIPPED (domain too large)")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results["runs"].append({
                "kappa": kappa, "error": str(e),
                "elapsed_s": time.time() - t0,
            })
        print(flush=True)

    # ── Comparison table ───────────────────────────────────────────────

    t26b_ref = T26B_TORQUE_LU[0.15]
    runs_by_k = {r["kappa"]: r for r in results["runs"] if "T_z" in r}

    if 0.15 in runs_by_k:
        ref_T = runs_by_k[0.15]["T_z"]
        print()
        print("=" * 60)
        print("Normalized drag multipliers (vs κ=0.15)")
        print("=" * 60)
        print(f"{'κ':>6} {'N³':>5} {'gap_lu':>7} {'Schwarz':>8} "
              f"{'T2.6b':>8} {'err%':>6} {'result':>6}")
        print("-" * 50)
        for k in sorted(runs_by_k.keys()):
            r = runs_by_k[k]
            s = r["T_z"] / ref_T
            t = T26B_TORQUE_LU[k] / t26b_ref
            err = abs(s - t) / t * 100
            passed = "PASS" if err < 10 else "FAIL"
            r["mult_norm"] = s
            r["t26b_mult"] = t
            r["error_pct"] = err
            r["passed"] = passed
            N = r.get("N_lattice", "?")
            gap = r.get("gap_lu", 0)
            print(f"{k:>6.2f} {N:>5} {gap:>7.1f} {s:>8.3f} "
                  f"{t:>8.3f} {err:>5.1f}% {passed:>6}")

    results["total_time_s"] = time.time() - t_total
    print(f"\nTotal time: {results['total_time_s']:.0f}s")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
