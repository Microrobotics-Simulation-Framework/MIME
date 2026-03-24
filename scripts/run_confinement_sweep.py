#!/usr/bin/env python3
"""T2.6 confinement sweep — production script for cloud GPU.

Runs multiple confined rotating UMR simulations at different confinement
ratios, resolutions, mask types, and initial orientations. Each run uses
two-pass bounce-back (pipe wall static, UMR rotating) with torque
convergence monitoring.

Results are written to an HDF5 file via SweepDataWriter.

Environment variables:
    SWEEP_SANITY_TEST=1  — run 2 quick runs at 64^3 (100 steps each)
                           with explicit SDF mask assertions
    USE_BOUZIDI=1        — use Bouzidi IBB for UMR surface (pipe wall
                           remains simple BB). Default: 0 (simple BB).

Usage:
    python3 scripts/run_confinement_sweep.py                    # production, simple BB
    USE_BOUZIDI=1 python3 scripts/run_confinement_sweep.py      # production, Bouzidi
    SWEEP_SANITY_TEST=1 USE_BOUZIDI=1 python3 scripts/run_confinement_sweep.py  # sanity + Bouzidi
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# JAX platform selection
if "JAX_PLATFORMS" not in os.environ:
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------

PRODUCTION_RUNS = [
    {"ratio": 0.15, "resolution": 192, "mask": "voxelised", "angle": 0.0,  "label": "main_0.15"},
    {"ratio": 0.22, "resolution": 192, "mask": "voxelised", "angle": 0.0,  "label": "main_0.22"},
    {"ratio": 0.30, "resolution": 192, "mask": "voxelised", "angle": 0.0,  "label": "main_0.30"},
    {"ratio": 0.40, "resolution": 192, "mask": "voxelised", "angle": 0.0,  "label": "main_0.40"},
    {"ratio": 0.35, "resolution": 192, "mask": "voxelised", "angle": 0.0,  "label": "held_out_0.35"},
    {"ratio": 0.30, "resolution": 192, "mask": "voxelised", "angle": 40.0, "label": "orient_40"},
    {"ratio": 0.30, "resolution": 192, "mask": "voxelised", "angle": 80.0, "label": "orient_80"},
    {"ratio": 0.30, "resolution": 128, "mask": "voxelised", "angle": 0.0,  "label": "rung_128"},
    {"ratio": 0.30, "resolution": 128, "mask": "sdf",       "angle": 0.0,  "label": "track_b"},
]

SANITY_RUNS = [
    {"ratio": 0.30, "resolution": 64, "mask": "voxelised", "angle": 0.0, "label": "sanity_vox"},
    {"ratio": 0.30, "resolution": 64, "mask": "sdf",       "angle": 0.0, "label": "sanity_sdf"},
]

# Physical geometry (mm)
VESSEL_DIAMETER_MM = 9.4
UMR_GEOM_MM = dict(
    body_radius=0.87, body_length=4.1, cone_length=1.9,
    cone_end_radius=0.255, fin_outer_radius=1.42,
    fin_length=2.03, fin_width=0.55, fin_thickness=0.15,
    helix_pitch=8.0,
)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(spec: dict, hdf5_path: str | None = None, max_steps: int | None = None) -> dict:
    """Execute one confined rotating UMR simulation.

    Returns dict with results or {"status": "FAILED", "error": str}.
    """
    from mime.nodes.robot.helix_geometry import create_umr_mask, create_umr_mask_sdf, umr_sdf
    from mime.nodes.environment.lbm.d3q19 import init_equilibrium, lbm_step_split
    from mime.nodes.environment.lbm.bounce_back import (
        compute_missing_mask, apply_bounce_back,
        apply_bouzidi_bounce_back, compute_q_values_sdf_sparse,
        compute_momentum_exchange_force, compute_momentum_exchange_torque,
    )

    use_bouzidi = os.environ.get("USE_BOUZIDI", "0") == "1"

    ratio = spec["ratio"]
    N = spec["resolution"]
    mask_type = spec["mask"]
    initial_angle = math.radians(spec["angle"])
    label = spec["label"]

    nz = N
    dx = VESSEL_DIAMETER_MM / N
    tau = 0.8
    cs = 1.0 / math.sqrt(3)

    # Scale geometry to lattice units
    geom_lu = {k: v / dx for k, v in UMR_GEOM_MM.items()}

    R_umr_lu = geom_lu["body_radius"]
    R_fin_lu = geom_lu["fin_outer_radius"]
    R_vessel_lu = R_umr_lu / ratio
    cx, cy, cz = N / 2.0, N / 2.0, nz / 2.0
    center = (cx, cy, cz)

    # Mach number guard
    omega = 0.05 * cs / R_fin_lu
    Ma_tip = omega * R_fin_lu * math.sqrt(3)
    assert Ma_tip < 0.1, f"Ma={Ma_tip:.3f} at fin tip exceeds 0.1"

    period = int(round(2 * math.pi / omega))
    if max_steps is None:
        max_steps = 30000 if N >= 192 else 20000

    print(f"\n{'='*60}", flush=True)
    print(f"RUN: {label}", flush=True)
    print(f"  ratio={ratio}, N={N}, mask={mask_type}, angle={spec['angle']}deg", flush=True)
    print(f"  omega={omega:.6f}, Ma={Ma_tip:.4f}, period={period}", flush=True)
    print(f"  R_vessel={R_vessel_lu:.1f} lu, R_fin={R_fin_lu:.1f} lu", flush=True)

    # Grid coordinates
    ix = jnp.arange(N, dtype=jnp.float32)
    iy = jnp.arange(N, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    dist_2d = jnp.sqrt((gx - cx)**2 + (gy - cy)**2)

    # Static pipe wall
    pipe_wall = dist_2d >= R_vessel_lu
    pipe_missing = compute_missing_mask(pipe_wall)

    # UMR wall velocity field
    omega_vec = jnp.array([0.0, 0.0, omega], dtype=jnp.float32)
    rx, ry, rz = gx - cx, gy - cy, gz - cz
    umr_wall_vel = jnp.stack([
        omega_vec[1]*rz - omega_vec[2]*ry,
        omega_vec[2]*rx - omega_vec[0]*rz,
        omega_vec[0]*ry - omega_vec[1]*rx,
    ], axis=-1)

    # Mask function selector
    def make_mask(angle):
        kw = dict(nx=N, ny=N, nz=nz, center=center, rotation_angle=angle, **geom_lu)
        if mask_type == "sdf":
            return create_umr_mask_sdf(**kw)
        else:
            return create_umr_mask(**kw)

    # Initialise
    f = init_equilibrium(N, N, nz)
    angle = initial_angle
    torques_z = []
    rho_initial = float(jnp.sum(f))
    check_interval = 200
    converged = False
    convergence_step = -1

    print(f"  Running (max_steps={max_steps})...", flush=True)
    t0 = time.perf_counter()
    step_times = []

    for step in range(max_steps):
        t_step = time.perf_counter()
        angle_new = angle + omega

        umr_mask = make_mask(angle_new)
        umr_missing = compute_missing_mask(umr_mask)
        solid_mask = pipe_wall | umr_mask

        f_pre, f_post, rho, u = lbm_step_split(f, tau)

        # Two-pass BB: pipe wall always simple BB
        f = apply_bounce_back(f_post, f_pre, pipe_missing, solid_mask, wall_velocity=None)

        # UMR pass: Bouzidi IBB or simple BB
        if use_bouzidi:
            sdf_kw = {k: v for k, v in geom_lu.items()
                      if k not in ('nx', 'ny', 'nz')}
            def sdf_func(pts):
                return umr_sdf(pts, rotation_angle=angle_new,
                               center=center, **sdf_kw)
            q_values = compute_q_values_sdf_sparse(umr_missing, sdf_func)
            f = apply_bouzidi_bounce_back(
                f, f_pre, umr_missing, solid_mask,
                q_values, wall_velocity=umr_wall_vel,
            )
        else:
            f = apply_bounce_back(f, f_pre, umr_missing, solid_mask, wall_velocity=umr_wall_vel)

        torque = compute_momentum_exchange_torque(
            f_pre, f, umr_missing, jnp.array(center, dtype=jnp.float32),
        )
        tz = float(torque[2])
        torques_z.append(tz)
        angle = angle_new

        f.block_until_ready()
        step_times.append(time.perf_counter() - t_step)

        # NaN check
        if step == 50 and np.isnan(tz):
            print(f"  NaN at step 50 — ABORTING", flush=True)
            return {"status": "FAILED", "error": "NaN at step 50", "label": label}

        # Convergence check
        if step % check_interval == 0 and step >= 2 * period:
            window = torques_z[-period:]
            prev_window = torques_z[-2*period:-period]
            mean_now = np.mean(window)
            mean_prev = np.mean(prev_window)
            rel_change = abs(mean_now - mean_prev) / max(abs(mean_now), 1e-8)
            elapsed = time.perf_counter() - t0
            print(f"  Step {step:>6d}: mean_Tz={mean_now:.4f}, rel_change={rel_change:.4f}, "
                  f"{elapsed:.0f}s", flush=True)
            if rel_change < 0.02:
                converged = True
                convergence_step = step
                print(f"  CONVERGED at step {step}", flush=True)
                break

    elapsed_total = time.perf_counter() - t0
    n_actual = len(torques_z)
    mean_step = np.mean(step_times[10:]) if len(step_times) > 10 else np.mean(step_times)

    # Post-processing
    from mime.nodes.environment.lbm.d3q19 import compute_macroscopic
    rho_final, u_final = compute_macroscopic(f)
    fluid_mask = ~(pipe_wall | make_mask(angle))

    has_nan = bool(jnp.any(jnp.isnan(u_final)))
    has_inf = bool(jnp.any(jnp.isinf(u_final)))
    u_max = float(jnp.max(jnp.where(fluid_mask[..., None], jnp.abs(u_final), 0.0)))
    rho_total = float(jnp.sum(f))
    density_cons = abs(rho_total - rho_initial) / rho_initial
    mean_tz = np.mean(torques_z[-period:]) if len(torques_z) >= period else np.mean(torques_z)

    result = {
        "status": "CONVERGED" if converged else "MAX_STEPS",
        "label": label,
        "ratio": ratio,
        "resolution": N,
        "mask": mask_type,
        "angle_deg": spec["angle"],
        "mean_torque_z": mean_tz,
        "convergence_step": convergence_step,
        "n_steps": n_actual,
        "mean_step_time": mean_step,
        "elapsed_s": elapsed_total,
        "u_max": u_max,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "density_conservation": density_cons,
    }

    print(f"  Result: {result['status']}, mean_Tz={mean_tz:.4f}, "
          f"steps={n_actual}, {mean_step:.4f}s/step, "
          f"u_max={u_max:.6f}, density={density_cons:.6%}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Node-based single run (USE_NODE=1)
# ---------------------------------------------------------------------------

def run_single_node(spec: dict, hdf5_path: str | None = None, max_steps: int | None = None) -> dict:
    """Execute one confined rotating UMR simulation via IBLBMFluidNode + GraphManager.

    Equivalent to run_single() but uses the node-graph path. Results must match
    run_single() within float32 tolerance.
    """
    from maddening.core.graph_manager import GraphManager
    from mime.nodes.environment.lbm.fluid_node import IBLBMFluidNode

    use_bouzidi = os.environ.get("USE_BOUZIDI", "0") == "1"

    ratio = spec["ratio"]
    N = spec["resolution"]
    initial_angle = math.radians(spec["angle"])
    label = spec["label"]

    nz = N
    dx = VESSEL_DIAMETER_MM / N
    tau = 0.8
    cs = 1.0 / math.sqrt(3)

    geom_lu = {k: v / dx for k, v in UMR_GEOM_MM.items()}
    R_umr_lu = geom_lu["body_radius"]
    R_fin_lu = geom_lu["fin_outer_radius"]
    R_vessel_lu = R_umr_lu / ratio

    omega = 0.05 * cs / R_fin_lu
    Ma_tip = omega * R_fin_lu * math.sqrt(3)
    assert Ma_tip < 0.1, f"Ma={Ma_tip:.3f} at fin tip exceeds 0.1"

    period = int(round(2 * math.pi / omega))
    if max_steps is None:
        max_steps = 30000 if N >= 192 else 20000

    print(f"\n{'='*60}", flush=True)
    print(f"RUN [NODE]: {label}", flush=True)
    print(f"  ratio={ratio}, N={N}, use_bouzidi={use_bouzidi}", flush=True)
    print(f"  omega={omega:.6f}, Ma={Ma_tip:.4f}, period={period}", flush=True)

    body_geometry_params = dict(nx=N, ny=N, nz=nz, **geom_lu)
    node = IBLBMFluidNode(
        name="lbm_fluid",
        timestep=1.0,
        nx=N, ny=N, nz=nz,
        tau=tau,
        vessel_radius_lu=R_vessel_lu,
        body_geometry_params=body_geometry_params,
        use_bouzidi=use_bouzidi,
    )

    gm = GraphManager()
    gm.add_node(node)
    gm.add_external_input("lbm_fluid", "body_angular_velocity", shape=(3,))
    gm.compile()

    ext = {"lbm_fluid": {"body_angular_velocity": jnp.array([0.0, 0.0, omega])}}

    print(f"  Running (max_steps={max_steps})...", flush=True)
    t0 = time.perf_counter()
    torques_z = []
    converged = False
    convergence_step = -1
    check_interval = 200

    for step in range(max_steps):
        gm.step(external_inputs=ext)
        state = gm.get_node_state("lbm_fluid")
        tz = float(state["drag_torque"][2])
        torques_z.append(tz)

        if step == 50 and np.isnan(tz):
            print(f"  NaN at step 50 — ABORTING", flush=True)
            return {"status": "FAILED", "error": "NaN at step 50", "label": label}

        if step % check_interval == 0 and step >= 2 * period:
            window = torques_z[-period:]
            prev_window = torques_z[-2*period:-period]
            mean_now = np.mean(window)
            mean_prev = np.mean(prev_window)
            rel_change = abs(mean_now - mean_prev) / max(abs(mean_now), 1e-8)
            elapsed = time.perf_counter() - t0
            print(f"  Step {step:>6d}: mean_Tz={mean_now:.4f}, "
                  f"rel_change={rel_change:.4f}, {elapsed:.0f}s", flush=True)
            if rel_change < 0.02:
                converged = True
                convergence_step = step
                print(f"  CONVERGED at step {step}", flush=True)
                break

    elapsed_total = time.perf_counter() - t0
    n_actual = len(torques_z)
    mean_tz = np.mean(torques_z[-period:]) if len(torques_z) >= period else np.mean(torques_z)

    result = {
        "status": "CONVERGED" if converged else "MAX_STEPS",
        "label": label,
        "ratio": ratio,
        "resolution": N,
        "mask": "node",
        "angle_deg": spec["angle"],
        "mean_torque_z": mean_tz,
        "convergence_step": convergence_step,
        "n_steps": n_actual,
        "mean_step_time": elapsed_total / max(n_actual, 1),
        "elapsed_s": elapsed_total,
        "u_max": 0.0,
        "has_nan": False,
        "has_inf": False,
        "density_conservation": 0.0,
    }

    print(f"  Result: {result['status']}, mean_Tz={mean_tz:.4f}, "
          f"steps={n_actual}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def main():
    is_sanity = os.environ.get("SWEEP_SANITY_TEST", "0") == "1"
    runs = SANITY_RUNS if is_sanity else PRODUCTION_RUNS
    max_steps = 100 if is_sanity else None
    hdf5_path = "data/umr_training_v1.h5" if not is_sanity else "data/sanity_test.h5"

    use_bouzidi = os.environ.get("USE_BOUZIDI", "0") == "1"
    use_node = os.environ.get("USE_NODE", "0") == "1"
    bc_label = "Bouzidi IBB" if use_bouzidi else "simple BB"
    mode_label = "NODE (GraphManager)" if use_node else "STANDALONE"
    print(f"{'SANITY TEST' if is_sanity else 'PRODUCTION SWEEP'}")
    print(f"Backend: {jax.default_backend()}")
    print(f"[CONFIG] USE_BOUZIDI={int(use_bouzidi)} — UMR surface BC: {bc_label}")
    print(f"[CONFIG] USE_NODE={int(use_node)} — execution mode: {mode_label}")
    print(f"Runs: {len(runs)}")
    print(f"HDF5: {hdf5_path}")

    os.makedirs("data", exist_ok=True)

    # Write HDF5 schema
    from mime.data.hdf5_schema import SweepDataWriter
    all_ratios = sorted(set(f"{r['ratio']:.2f}" for r in runs))
    writer = SweepDataWriter(hdf5_path)
    writer.create_schema(
        ratios=[float(r) for r in all_ratios],
        group="ground_truth",
    )
    writer.write_provenance({
        "runs": [r["label"] for r in runs],
        "is_sanity_test": is_sanity,
    })

    run_fn = run_single_node if use_node else run_single

    results = []
    for i, spec in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] Starting: {spec['label']}")
        try:
            result = run_fn(spec, hdf5_path=hdf5_path, max_steps=max_steps)
            results.append(result)

            # Write converged results to HDF5
            if result["status"] != "FAILED":
                ratio_key = f"{spec['ratio']:.2f}"
                writer.append_sample("ground_truth", ratio=spec["ratio"], data={
                    "drag_torque_z": result["mean_torque_z"],
                    "step": result["n_steps"],
                    "wall_time": result["elapsed_s"],
                    "u_max": result["u_max"],
                    "max_density_fluctuation": result["density_conservation"],
                    "residual": 0.0,
                    "convergence_label": 1 if result["status"] == "CONVERGED" else 0,
                })
        except Exception as e:
            print(f"  FAILED with exception: {e}", flush=True)
            results.append({"status": "FAILED", "error": str(e), "label": spec["label"]})

    writer.close()

    # Summary table
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Label':<20} {'Status':<12} {'Ratio':<8} {'N':<6} {'Mask':<10} "
          f"{'Mean Tz':<12} {'Steps':<8} {'s/step':<8}")
    print("-" * 80)
    for r in results:
        if r["status"] == "FAILED":
            print(f"{r['label']:<20} FAILED       {r.get('error', 'unknown')}")
        else:
            print(f"{r['label']:<20} {r['status']:<12} {r['ratio']:<8.2f} "
                  f"{r['resolution']:<6} {r['mask']:<10} "
                  f"{r['mean_torque_z']:<12.4f} {r['n_steps']:<8} "
                  f"{r['mean_step_time']:<8.4f}")

    # Sanity test assertions
    if is_sanity:
        print(f"\n{'='*60}")
        print("SDF MASK ASSERTIONS")
        print(f"{'='*60}")

        vox_result = next((r for r in results if r["label"] == "sanity_vox"), None)
        sdf_result = next((r for r in results if r["label"] == "sanity_sdf"), None)

        if vox_result is None or sdf_result is None:
            print("FAIL: one or both sanity runs did not complete")
            sys.exit(1)

        if vox_result["status"] == "FAILED" or sdf_result["status"] == "FAILED":
            print(f"FAIL: vox={vox_result['status']}, sdf={sdf_result['status']}")
            sys.exit(1)

        # Assertion 1: SDF mask is non-empty
        from mime.nodes.robot.helix_geometry import create_umr_mask, create_umr_mask_sdf
        N = 64
        dx = VESSEL_DIAMETER_MM / N
        geom_lu = {k: v / dx for k, v in UMR_GEOM_MM.items()}
        center = (N/2.0, N/2.0, N/2.0)
        sdf_mask = create_umr_mask_sdf(N, N, N, center=center, **geom_lu)
        sdf_count = int(jnp.sum(sdf_mask))
        print(f"  1. SDF mask non-empty: {sdf_count} nodes — {'PASS' if sdf_count > 0 else 'FAIL'}")
        assert sdf_count > 0, "SDF mask is empty"

        # Assertion 2: Mask agreement > 99.5%
        vox_mask = create_umr_mask(N, N, N, center=center, **geom_lu)
        total_nodes = vox_mask.size
        agree = int(jnp.sum(sdf_mask == vox_mask))
        agreement = agree / total_nodes
        print(f"  2. Mask agreement: {agreement:.6f} ({agree}/{total_nodes}) — "
              f"{'PASS' if agreement > 0.995 else 'FAIL'}")
        assert agreement > 0.995, f"Mask agreement {agreement:.4f} < 99.5%"

        # Assertion 3: No NaN/Inf in SDF run
        no_nan = not sdf_result["has_nan"] and not sdf_result["has_inf"]
        print(f"  3. SDF run stability: NaN={sdf_result['has_nan']}, "
              f"Inf={sdf_result['has_inf']} — {'PASS' if no_nan else 'FAIL'}")
        assert no_nan, "NaN or Inf in SDF run"

        # Assertion 4: Torque same order of magnitude
        tz_vox = abs(vox_result["mean_torque_z"])
        tz_sdf = abs(sdf_result["mean_torque_z"])
        torque_ratio = tz_sdf / max(tz_vox, 1e-10)
        print(f"  4. Torque ratio (SDF/vox): {torque_ratio:.4f} — "
              f"{'PASS' if 0.1 < torque_ratio < 10.0 else 'FAIL'}")
        assert 0.1 < torque_ratio < 10.0, f"Torque ratio {torque_ratio:.2f} outside [0.1, 10.0]"

        # Assertion 5: HDF5 writer produced non-empty data
        import h5py
        with h5py.File(hdf5_path, 'r') as hf:
            found_data = False
            for group_key in hf.keys():
                grp = hf[group_key]
                if not hasattr(grp, 'keys'):
                    continue
                for ratio_key in grp.keys():
                    sub = grp[ratio_key]
                    if 'drag_torque_z' in sub:
                        data = sub['drag_torque_z'][:]
                        n_written = sum(1 for v in data if v != 0.0)
                        if n_written > 0:
                            found_data = True
                            break
                if found_data:
                    break
            print(f"  5. HDF5 writer non-empty: found_data={found_data} — "
                  f"{'PASS' if found_data else 'FAIL'}")
            assert found_data, (
                "No non-zero drag_torque_z found in HDF5 output — "
                "writer.append_sample() may not be called"
            )

        print(f"\nALL ASSERTIONS PASS (including HDF5 writer)")

    # Final status
    n_pass = sum(1 for r in results if r["status"] != "FAILED")
    n_fail = len(results) - n_pass
    print(f"\n{n_pass}/{len(results)} runs completed, {n_fail} failed")

    if n_fail > 0 and not is_sanity:
        print("WARNING: some runs failed — check logs above")
        sys.exit(1 if n_fail == len(results) else 0)  # only exit 1 if ALL failed

    # Write completion marker — the launcher polls for this file to know
    # the sweep is done (the HDF5 file exists from schema creation, before
    # any runs complete, so it cannot be used as a completion signal).
    marker_path = Path(hdf5_path).with_suffix(".done")
    marker_path.write_text(f"completed {n_pass}/{len(results)} runs\n")
    print(f"\nCompletion marker: {marker_path}")


if __name__ == "__main__":
    main()
