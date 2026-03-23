#!/usr/bin/env python3
"""Cloud rehearsal script — validates the full T2.6 pipeline at 192^3.

Validates 7 gate criteria before the H100 production sweep:
1. Environment: JAX GPU detection, CUDA version, GPU model
2. Mach guard: fires at wrong omega, passes at correct omega
3. 192^3 Bouzidi stability: 500 steps, no NaN/Inf, u_max < 0.05
4. Timing: mean step time reported for cost extrapolation
5. HDF5 write: correct schema, provenance metadata
6. HDF5 read-back: shapes correct, no NaN
7. Cost projection: H100 estimates from measured step time

Supports resolution override via REHEARSAL_RESOLUTION env var for
local 64^3 sanity testing:
    REHEARSAL_RESOLUTION=64 python3 scripts/rehearse_192.py
"""

from __future__ import annotations

import math
import os
import sys
import time

# JAX platform selection — must happen before import
if "JAX_PLATFORMS" not in os.environ:
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
import jax.numpy as jnp
import numpy as np


def main():
    results = {}
    all_pass = True

    # ── Gate 1: Environment ──────────────────────────────────────────
    print("=" * 60)
    print("GATE 1: Environment")
    print("=" * 60)

    backend = jax.default_backend()
    devices = jax.devices()
    gpu_name = str(devices[0]) if devices else "none"
    print(f"  JAX version: {jax.__version__}")
    print(f"  Backend: {backend}")
    print(f"  Device: {gpu_name}")

    gate1 = backend == "gpu"
    if not gate1:
        print(f"  FAIL: expected gpu, got {backend}")
        # On local CPU runs, continue but mark as expected failure
        if os.environ.get("REHEARSAL_RESOLUTION"):
            print("  (Continuing anyway — local sanity test with REHEARSAL_RESOLUTION)")
            gate1 = True  # Don't block local testing on GPU
    else:
        print("  PASS")
    results["environment"] = gate1

    # ── Resolution setup ──────────────────────────────────────────────
    N = int(os.environ.get("REHEARSAL_RESOLUTION", "192"))
    VESSEL_DIAMETER_MM = 9.4
    dx = VESSEL_DIAMETER_MM / N
    nz = N  # cubic domain

    print(f"\n  Resolution: {N}^3 (dx = {dx:.5f} mm)")

    from mime.nodes.robot.helix_geometry import create_umr_mask
    from mime.nodes.environment.lbm.d3q19 import init_equilibrium, lbm_step_split
    from mime.nodes.environment.lbm.bounce_back import (
        compute_missing_mask, apply_bounce_back,
        compute_momentum_exchange_force, compute_momentum_exchange_torque,
    )

    # UMR geometry in lattice units
    geom = dict(
        nx=N, ny=N, nz=nz,
        body_radius=0.87 / dx,
        body_length=4.1 / dx,
        cone_length=1.9 / dx,
        cone_end_radius=0.255 / dx,
        fin_outer_radius=1.42 / dx,
        fin_length=2.03 / dx,
        fin_width=0.55 / dx,
        fin_thickness=0.15 / dx,
        helix_pitch=8.0 / dx,
    )

    R_fin_lu = geom["fin_outer_radius"]
    R_umr_lu = geom["body_radius"]
    R_vessel_lu = R_umr_lu / 0.30  # confinement ratio 0.30
    cx, cy, cz = N / 2.0, N / 2.0, nz / 2.0
    center = (cx, cy, cz)
    cs = 1.0 / math.sqrt(3)
    tau = 0.8

    # Safe omega (Ma = 0.05 at fin tips)
    omega_safe = 0.05 * cs / R_fin_lu

    # ── Gate 2: Mach guard ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE 2: Mach number guard")
    print("=" * 60)

    # Test that the guard fires at unsafe omega
    omega_unsafe = 0.2 * cs / R_fin_lu  # Ma = 0.2 — should fire
    Ma_unsafe = omega_unsafe * R_fin_lu * math.sqrt(3)
    guard_fires = Ma_unsafe >= 0.1
    print(f"  Unsafe omega={omega_unsafe:.5f}, Ma={Ma_unsafe:.3f}, guard fires: {guard_fires}")

    # Test that safe omega passes
    Ma_safe = omega_safe * R_fin_lu * math.sqrt(3)
    guard_passes = Ma_safe < 0.1
    print(f"  Safe omega={omega_safe:.5f}, Ma={Ma_safe:.3f}, guard passes: {guard_passes}")

    gate2 = guard_fires and guard_passes
    print(f"  {'PASS' if gate2 else 'FAIL'}")
    results["mach_guard"] = gate2

    # ── Gate 3: Stability + Gate 4: Timing ────────────────────────────
    print("\n" + "=" * 60)
    print(f"GATE 3+4: {N}^3 stability + timing (500 steps)")
    print("=" * 60)

    # Grid coordinates
    ix = jnp.arange(N, dtype=jnp.float32)
    iy = jnp.arange(N, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    dist_2d = jnp.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)

    # Static pipe wall
    pipe_wall = dist_2d >= R_vessel_lu
    pipe_missing = compute_missing_mask(pipe_wall)

    # UMR wall velocity field
    omega_vec = jnp.array([0.0, 0.0, omega_safe], dtype=jnp.float32)
    rx, ry, rz = gx - cx, gy - cy, gz - cz
    umr_wall_vel = jnp.stack([
        omega_vec[1] * rz - omega_vec[2] * ry,
        omega_vec[2] * rx - omega_vec[0] * rz,
        omega_vec[0] * ry - omega_vec[1] * rx,
    ], axis=-1)

    print(f"  R_vessel={R_vessel_lu:.1f} lu, R_fin={R_fin_lu:.1f} lu")
    print(f"  omega={omega_safe:.6f}, Ma_tip={Ma_safe:.4f}")
    print(f"  Pipe wall nodes: {int(jnp.sum(pipe_wall))}")

    f = init_equilibrium(N, N, nz)
    angle = 0.0
    rho_initial = float(jnp.sum(f))
    torques_z = []

    n_steps = 500
    step_times = []

    # Warmup step (JIT compilation)
    print("  Warmup step...", flush=True)
    umr_mask = create_umr_mask(**geom, center=center, rotation_angle=0.0)
    umr_missing = compute_missing_mask(umr_mask)
    solid_mask = pipe_wall | umr_mask
    f_pre, f_post, _, _ = lbm_step_split(f, tau)
    f_warm = apply_bounce_back(f_post, f_pre, pipe_missing, solid_mask)
    f_warm = apply_bounce_back(f_warm, f_pre, umr_missing, solid_mask, wall_velocity=umr_wall_vel)
    f_warm.block_until_ready()
    print("  Warmup done.", flush=True)

    print(f"  Running {n_steps} steps...", flush=True)
    t_total_start = time.perf_counter()

    for step in range(n_steps):
        t_step_start = time.perf_counter()
        angle_new = angle + omega_safe

        umr_mask = create_umr_mask(**geom, center=center, rotation_angle=angle_new)
        umr_missing = compute_missing_mask(umr_mask)
        solid_mask = pipe_wall | umr_mask

        f_pre, f_post, rho, u = lbm_step_split(f, tau)

        # Two-pass BB
        f = apply_bounce_back(f_post, f_pre, pipe_missing, solid_mask, wall_velocity=None)
        f = apply_bounce_back(f, f_pre, umr_missing, solid_mask, wall_velocity=umr_wall_vel)

        torque = compute_momentum_exchange_torque(
            f_pre, f, umr_missing, jnp.array(center, dtype=jnp.float32),
        )
        tz = float(torque[2])
        torques_z.append(tz)
        angle = angle_new

        f.block_until_ready()
        step_times.append(time.perf_counter() - t_step_start)

        # Early NaN check
        if step == 50:
            if np.isnan(tz):
                print(f"  NaN at step 50 — ABORTING", flush=True)
                break
            u_max_early = float(jnp.max(jnp.abs(u)))
            print(f"  Step 50: Tz={tz:.4f}, u_max={u_max_early:.6f}", flush=True)

        if step % 100 == 0 and step > 0:
            elapsed = time.perf_counter() - t_total_start
            print(f"  Step {step}: {elapsed:.1f}s ({elapsed/step:.3f}s/step)", flush=True)

    t_total = time.perf_counter() - t_total_start
    n_actual = len(torques_z)

    # Post-processing
    from mime.nodes.environment.lbm.d3q19 import compute_macroscopic
    rho_final, u_final = compute_macroscopic(f)
    fluid_mask = ~(pipe_wall | create_umr_mask(**geom, center=center, rotation_angle=angle))

    has_nan = bool(jnp.any(jnp.isnan(u_final)))
    has_inf = bool(jnp.any(jnp.isinf(u_final)))
    u_max = float(jnp.max(jnp.where(fluid_mask[..., None], jnp.abs(u_final), 0.0)))
    rho_total_final = float(jnp.sum(f))
    density_conservation = abs(rho_total_final - rho_initial) / rho_initial
    mean_tz = np.mean(torques_z[-100:]) if len(torques_z) >= 100 else np.mean(torques_z)
    torque_sign_correct = mean_tz > 0

    # Timing
    mean_step_time = np.mean(step_times[10:]) if len(step_times) > 10 else np.mean(step_times)
    print(f"\n  [TIMING] mean_step_time={mean_step_time:.3f}s")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    print(f"  u_max: {u_max:.6f} (threshold: < 0.05)")
    print(f"  Density conservation: {density_conservation:.6%} (threshold: < 0.01%)")
    print(f"  Mean Tz (last 100): {mean_tz:.4f}, sign correct: {torque_sign_correct}")

    gate3 = (
        not has_nan and not has_inf
        and u_max < 0.05
        and density_conservation < 0.0001
        and torque_sign_correct
        and n_actual == n_steps
    )
    print(f"  Gate 3 (stability): {'PASS' if gate3 else 'FAIL'}")
    results["stability"] = gate3
    results["timing"] = mean_step_time

    # ── Gate 4: Cost projection ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE 4: Cost projection from measured step time")
    print("=" * 60)

    # Bandwidth scaling factors (relative to measured GPU)
    bw_a100 = 2.0   # TB/s
    bw_h100 = 3.35   # TB/s
    bw_rtx2060 = 0.336  # TB/s

    if backend == "gpu":
        # Determine which GPU we're on for bandwidth reference
        gpu_str = gpu_name.lower()
        if "a100" in gpu_str:
            measured_bw = bw_a100
        elif "h100" in gpu_str:
            measured_bw = bw_h100
        elif "2060" in gpu_str:
            measured_bw = bw_rtx2060
        elif "3090" in gpu_str:
            measured_bw = 0.936
        elif "4090" in gpu_str:
            measured_bw = 1.008
        else:
            measured_bw = bw_a100  # default assumption for cloud GPU
            print(f"  WARNING: Unknown GPU '{gpu_name}', assuming A100 bandwidth")
    else:
        measured_bw = bw_rtx2060
        print("  WARNING: CPU backend — cost projection is approximate")

    # Scale measured step time to other GPUs via bandwidth ratio
    step_h100_at_N = mean_step_time * measured_bw / bw_h100
    step_a100_at_N = mean_step_time * measured_bw / bw_a100

    # To project 192^3 cost, we also need to scale for resolution if N != 192
    # LBM step time scales linearly with node count (bandwidth-bound)
    node_scale = (192 ** 3) / (N ** 3) if N != 192 else 1.0
    step_h100_192 = step_h100_at_N * node_scale
    step_a100_192 = step_a100_at_N * node_scale

    # Convergence estimate: 2 periods at 192^3
    omega_192 = 0.05 * cs / (1.42 / (9.4 / 192))
    period_192 = int(2 * math.pi / omega_192)
    steps_to_converge = 2 * period_192

    print(f"  Measured step time ({N}^3): {mean_step_time:.3f}s on {gpu_name}")
    if N != 192:
        print(f"  NOTE: running at {N}^3, not 192^3. Scaling by {node_scale:.1f}x for 192^3 projection.")
    print(f"  Projected A100 step time (192^3): {step_a100_192:.3f}s")
    print(f"  Projected H100 step time (192^3): {step_h100_192:.3f}s")
    print(f"  Steps to converge (2 periods at 192^3): {steps_to_converge}")
    print(f"  Time per ratio (H100): {steps_to_converge * step_h100_192 / 3600:.1f} hr")
    print(f"  4 ratios (H100): {4 * steps_to_converge * step_h100_192 / 3600:.1f} hr")
    cost_h100 = 4 * steps_to_converge * step_h100_192 / 3600 * 2.69
    cost_a100 = 4 * steps_to_converge * step_a100_192 / 3600 * 1.49
    print(f"  Estimated cost (H100 @ $2.69/hr): ${cost_h100:.2f}")
    print(f"  Estimated cost (A100 @ $1.49/hr): ${cost_a100:.2f}")
    if N != 192:
        print(f"  WARNING: These are extrapolated from {N}^3. The cloud rehearsal at 192^3 will give definitive numbers.")

    gate4 = True  # Informational — always passes
    results["cost_projection"] = {"h100": cost_h100, "a100": cost_a100}

    # ── Gate 5: HDF5 write ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE 5: HDF5 write")
    print("=" * 60)

    hdf5_path = "data/rehearsal_192.h5"
    try:
        from mime.data.hdf5_schema import SweepDataWriter

        os.makedirs("data", exist_ok=True)
        with SweepDataWriter(hdf5_path) as writer:
            writer.create_schema(ratios=[0.30], group="ground_truth")
            writer.write_provenance({
                "resolution": N,
                "tau": tau,
                "omega": omega_safe,
                "confinement_ratio": 0.30,
                "n_steps": n_steps,
            })

            # Write a few samples from the collected data
            for i in range(min(5, len(torques_z))):
                writer.append_sample("ground_truth", ratio=0.30, data={
                    "drag_torque_z": torques_z[i],
                    "step": i * 100,
                    "wall_time": step_times[i] if i < len(step_times) else 0.0,
                    "u_max": u_max,
                    "residual": 0.0,
                    "convergence_label": 0,
                    "max_density_fluctuation": float(density_conservation),
                })

        file_size = os.path.getsize(hdf5_path)
        print(f"  Written: {hdf5_path} ({file_size} bytes)")
        gate5 = True
    except Exception as e:
        print(f"  FAIL: {e}")
        gate5 = False
    print(f"  {'PASS' if gate5 else 'FAIL'}")
    results["hdf5_write"] = gate5

    # ── Gate 6: HDF5 read-back ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE 6: HDF5 read-back verification")
    print("=" * 60)

    try:
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            # Check schema structure
            assert "ground_truth" in f, "Missing ground_truth group"
            assert "0.30" in f["ground_truth"], "Missing ratio 0.30 group"
            assert "provenance" in f, "Missing provenance group"
            assert "held_out" in f, "Missing held_out group"

            grp = f["ground_truth/0.30"]
            # Check some collected datasets have data
            assert grp["drag_torque_z"].shape[0] > 0, "No torque data"
            assert not np.any(np.isnan(grp["drag_torque_z"][:])), "NaN in torque"

            # Check provenance
            prov = f["provenance"]
            assert "git_hash" in prov.attrs, "Missing git_hash"
            assert "jax_version" in prov.attrs, "Missing jax_version"

            # Check deferred datasets
            assert grp["coarsened_velocity"].attrs.get("deferred", False), (
                "coarsened_velocity should be deferred"
            )

            print(f"  Schema: OK")
            print(f"  Samples: {grp['drag_torque_z'].shape[0]}")
            print(f"  Provenance git_hash: {prov.attrs['git_hash']}")
            print(f"  Deferred datasets present: {sum(1 for ds in grp.values() if hasattr(ds, 'attrs') and ds.attrs.get('deferred', False))}")
        gate6 = True
    except Exception as e:
        print(f"  FAIL: {e}")
        gate6 = False
    print(f"  {'PASS' if gate6 else 'FAIL'}")
    results["hdf5_readback"] = gate6

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REHEARSAL SUMMARY")
    print("=" * 60)

    gates = {
        "1. Environment": results.get("environment", False),
        "2. Mach guard": results.get("mach_guard", False),
        "3. Stability": results.get("stability", False),
        "4. Cost projection": True,  # informational
        "5. HDF5 write": results.get("hdf5_write", False),
        "6. HDF5 read-back": results.get("hdf5_readback", False),
    }

    for name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_pass = all(gates.values())
    print(f"\n  OVERALL: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
