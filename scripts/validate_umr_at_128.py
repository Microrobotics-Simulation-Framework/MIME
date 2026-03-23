#!/usr/bin/env python3
"""Pre-T2.6 stability check: rotating UMR inside pipe at 128^3.

Uses the corrected two-pass bounce-back architecture:
  Pass 1: Pipe wall (stationary, no wall velocity)
  Pass 2: UMR (rotating, wall velocity = omega x r)

Also validates: Mach number guard, density conservation, torque sign.
This is the mandatory gate check before launching the A100/H100 sweep.
"""

import os

if "JAX_PLATFORMS" not in os.environ:
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import math
import time
import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX backend: {jax.default_backend()}", flush=True)

from mime.nodes.environment.lbm.d3q19 import (
    init_equilibrium,
    compute_macroscopic,
    lbm_step_split,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    apply_bounce_back,
    compute_momentum_exchange_force,
    compute_momentum_exchange_torque,
)
from mime.nodes.robot.helix_geometry import create_umr_mask


# ---------------------------------------------------------------------------
# Physical setup
# ---------------------------------------------------------------------------

N = 128
nz = N
VESSEL_DIAMETER_MM = 9.4
dx = VESSEL_DIAMETER_MM / N

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

# Confinement ratio 0.30
R_umr_lu = geom["body_radius"]
R_fin_lu = geom["fin_outer_radius"]
R_vessel_lu = R_umr_lu / 0.30

tau = 0.8
nu = (tau - 0.5) / 3.0
cs = 1.0 / math.sqrt(3)

# Mach number guard: omega must keep Ma < 0.1 at fin tips
Ma_target = 0.05
omega = Ma_target * cs / R_fin_lu
Ma_tip = omega * R_fin_lu / cs
period = int(round(2 * math.pi / omega))

print(f"Grid: {N}x{N}x{nz}, dx = {dx:.5f} mm", flush=True)
print(f"R_vessel = {R_vessel_lu:.1f} lu, R_umr = {R_umr_lu:.1f} lu, R_fin = {R_fin_lu:.1f} lu", flush=True)
print(f"tau = {tau}, nu = {nu:.4f}", flush=True)
print(f"omega = {omega:.6f} rad/step, Ma_tip = {Ma_tip:.4f}", flush=True)
print(f"Period = {period} steps", flush=True)

# MACH NUMBER GUARD
assert Ma_tip < 0.1, (
    f"Mach number at fin tip = {Ma_tip:.3f} exceeds 0.1. "
    f"Reduce omega to < {0.1 * cs / R_fin_lu:.5f} rad/step."
)
print("Mach number guard: PASS", flush=True)

# ---------------------------------------------------------------------------
# Geometry setup
# ---------------------------------------------------------------------------

print("\nCreating geometry...", flush=True)
t0 = time.perf_counter()

cx, cy, cz = N / 2.0, N / 2.0, nz / 2.0
center = (cx, cy, cz)

# Grid coordinates (reused for pipe mask and wall velocity)
ix = jnp.arange(N, dtype=jnp.float32)
iy = jnp.arange(N, dtype=jnp.float32)
iz = jnp.arange(nz, dtype=jnp.float32)
gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
dist_2d = jnp.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)

# Pipe wall mask (STATIC — computed once)
pipe_wall = dist_2d >= R_vessel_lu
pipe_missing = compute_missing_mask(pipe_wall)

# UMR wall velocity field (omega x r — constant since omega is constant)
omega_vec = jnp.array([0.0, 0.0, omega], dtype=jnp.float32)
rx, ry, rz = gx - cx, gy - cy, gz - cz
umr_wall_vel = jnp.stack([
    omega_vec[1] * rz - omega_vec[2] * ry,
    omega_vec[2] * rx - omega_vec[0] * rz,
    omega_vec[0] * ry - omega_vec[1] * rx,
], axis=-1)

t_geom = time.perf_counter() - t0
print(f"Geometry setup: {t_geom:.2f}s", flush=True)
print(f"Pipe wall nodes: {int(jnp.sum(pipe_wall))}", flush=True)

# ---------------------------------------------------------------------------
# Run rotating simulation with two-pass BB
# ---------------------------------------------------------------------------

n_steps = 1000
check_interval = 100
print(f"\nRunning {n_steps} steps with two-pass BB (pipe static, UMR rotating)...", flush=True)

f = init_equilibrium(N, N, nz)
angle = 0.0
torques_z = []
rho_initial = float(jnp.sum(f))

t0 = time.perf_counter()
for step in range(n_steps):
    angle_new = angle + omega

    # UMR mask at new angle (recomputed each step)
    umr_mask = create_umr_mask(**geom, center=center, rotation_angle=angle_new)
    umr_missing = compute_missing_mask(umr_mask)

    # Combined solid for collision skipping
    solid_mask = pipe_wall | umr_mask

    # LBM collision + streaming
    f_pre, f_post, rho, u = lbm_step_split(f, tau)

    # TWO-PASS BOUNCE-BACK
    # Pass 1: Pipe wall (stationary — no wall velocity)
    f = apply_bounce_back(f_post, f_pre, pipe_missing, solid_mask, wall_velocity=None)
    # Pass 2: UMR (rotating — wall velocity = omega x r)
    f = apply_bounce_back(f, f_pre, umr_missing, solid_mask, wall_velocity=umr_wall_vel)

    # Momentum exchange torque on UMR only
    torque = compute_momentum_exchange_torque(
        f_pre, f, umr_missing, jnp.array(center, dtype=jnp.float32),
    )
    tz = float(torque[2])
    torques_z.append(tz)
    angle = angle_new

    # Early NaN check
    if step == 100:
        if np.isnan(tz):
            print(f"NaN at step 100 — ABORTING", flush=True)
            break
        u_max_early = float(jnp.max(jnp.abs(u)))
        print(f"  Step 100: Tz={tz:.6f}, u_max={u_max_early:.6f}, Ma_max={u_max_early*math.sqrt(3):.4f}", flush=True)

    if step % check_interval == 0 and step > 0:
        elapsed = time.perf_counter() - t0
        print(f"  Step {step:>5d}: Tz={tz:.6f}, elapsed={elapsed:.1f}s ({elapsed/step:.3f}s/step)", flush=True)

elapsed_total = time.perf_counter() - t0
n_actual = len(torques_z)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

rho_final, u_final = compute_macroscopic(f)
fluid_mask = ~(pipe_wall | create_umr_mask(**geom, center=center, rotation_angle=angle))

has_nan = bool(jnp.any(jnp.isnan(u_final)))
has_inf = bool(jnp.any(jnp.isinf(u_final)))
u_max = float(jnp.max(jnp.where(fluid_mask[..., None], jnp.abs(u_final), 0.0)))
rho_total_final = float(jnp.sum(f))
density_conservation = abs(rho_total_final - rho_initial) / rho_initial

# Torque analysis
mean_tz = np.mean(torques_z[-200:]) if len(torques_z) >= 200 else np.mean(torques_z)
torque_sign_correct = mean_tz > 0  # body pumps angular momentum into fluid

print(f"\n{'=' * 60}")
print(f"PRE-T2.6 GATE CHECK RESULTS")
print(f"{'=' * 60}")
print(f"Steps completed: {n_actual}")
print(f"Wall time: {elapsed_total:.1f}s ({elapsed_total/max(n_actual,1):.3f}s/step)")
print(f"")
print(f"Stability:")
print(f"  NaN in velocity: {has_nan}")
print(f"  Inf in velocity: {has_inf}")
print(f"  u_max: {u_max:.6f} lu (threshold: < 0.05)")
print(f"  Ma_max: {u_max * math.sqrt(3):.4f} (threshold: < 0.1)")
print(f"  Density conservation: {density_conservation:.6%} (threshold: < 0.01%)")
print(f"")
print(f"Physics:")
print(f"  Mean Tz (last 200 steps): {mean_tz:.6f}")
print(f"  Torque sign correct (opposing rotation): {torque_sign_correct}")
print(f"  Mach number guard fired: False")
print(f"")

# Gate criteria
gate_pass = (
    not has_nan
    and not has_inf
    and u_max < 0.05
    and density_conservation < 0.0001
    and torque_sign_correct
)
print(f"GATE: {'PASS' if gate_pass else 'FAIL'}")
if not gate_pass:
    reasons = []
    if has_nan: reasons.append("NaN detected")
    if has_inf: reasons.append("Inf detected")
    if u_max >= 0.05: reasons.append(f"u_max={u_max:.4f} >= 0.05")
    if density_conservation >= 0.0001: reasons.append(f"density conservation {density_conservation:.4%} >= 0.01%")
    if not torque_sign_correct: reasons.append(f"torque sign wrong: mean_Tz={mean_tz:.6f}")
    print(f"  Failure reasons: {', '.join(reasons)}")
