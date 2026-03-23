#!/usr/bin/env python3
"""Pre-T2.6 stability check: combined pipe-wall + UMR body at 128^3.

Runs a single stationary UMR body (no fins, no rotation) inside a
cylindrical pipe with a body-force-driven flow. Verifies numerical
stability and reports drag force and convergence time.

This is a gate check — not an accuracy validation. It confirms that
the combined solid mask (pipe wall + UMR body) produces stable LBM
dynamics before adding fins and rotation in T2.6.
"""

import os

# Prefer GPU; limit memory preallocation to leave room for intermediates
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
    create_pipe_walls,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    compute_momentum_exchange_force,
)
from mime.nodes.environment.lbm.convergence import run_to_convergence
from mime.nodes.robot.helix_geometry import create_cylinder_body_mask


# ---------------------------------------------------------------------------
# Setup: 128^3 with pipe wall + UMR body
# ---------------------------------------------------------------------------

N = 128
VESSEL_DIAMETER_MM = 9.4
UMR_BODY_RADIUS_MM = 0.87
UMR_BODY_LENGTH_MM = 4.1
UMR_CONE_LENGTH_MM = 1.9
UMR_CONE_END_RADIUS_MM = 0.255

dx = VESSEL_DIAMETER_MM / N  # mm per lattice unit
# Use cubic domain: UMR body is ~82 lu long at this dx, so 128 gives ~23 lu clearance per side
nz = N

print(f"Grid: {N}x{N}x{nz}, dx = {dx:.5f} mm", flush=True)
print(f"Vessel radius: {(VESSEL_DIAMETER_MM / 2) / dx:.1f} lu", flush=True)
print(f"UMR body radius: {UMR_BODY_RADIUS_MM / dx:.2f} lu", flush=True)

# Lattice parameters
tau = 0.8  # nu_lattice = (0.8 - 0.5)/3 = 0.1
# Body force to drive Poiseuille-like flow along z-axis
# Target: Re ~ 0.1 (low Re for stability check)
# u_target ~ 0.01 (lattice units)
# For pipe flow: F = 8 * nu * u_mean / R^2
R_vessel_lu = (VESSEL_DIAMETER_MM / 2) / dx
nu_lattice = (tau - 0.5) / 3.0
# Stronger force for faster convergence (stability check only, not accuracy)
u_target = 0.02
F_body = 8.0 * nu_lattice * u_target / (R_vessel_lu ** 2)
print(f"Body force: {F_body:.2e} (lattice units)", flush=True)
print(f"Target u_mean: {u_target} lu", flush=True)

# ---------------------------------------------------------------------------
# Create geometry
# ---------------------------------------------------------------------------

print("\nCreating geometry...", flush=True)
t0 = time.perf_counter()

center = (N / 2.0, N / 2.0, nz / 2.0)

# Pipe wall (flow along z, periodic in z, walls in xy-plane)
# create_pipe_walls creates flow in x with walls in yz.
# We need flow in z with walls in xy. So use a custom pipe mask.
ix = jnp.arange(N, dtype=jnp.float32)
iy = jnp.arange(N, dtype=jnp.float32)
xx, yy = jnp.meshgrid(ix, iy, indexing='ij')
cx, cy = N / 2.0, N / 2.0
dist_2d = jnp.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
pipe_wall_2d = dist_2d >= R_vessel_lu
pipe_wall = jnp.broadcast_to(pipe_wall_2d[:, :, None], (N, N, nz))

# UMR body (cylinder + cone, no fins, along z-axis at center)
umr_body = create_cylinder_body_mask(
    N, N, nz,
    body_radius=UMR_BODY_RADIUS_MM / dx,
    body_length=UMR_BODY_LENGTH_MM / dx,
    cone_length=UMR_CONE_LENGTH_MM / dx,
    cone_end_radius=UMR_CONE_END_RADIUS_MM / dx,
    center=center,
    axis=2,
)

# Combined solid mask
solid_mask = pipe_wall | umr_body

pipe_count = int(jnp.sum(pipe_wall))
body_count = int(jnp.sum(umr_body))
total_solid = int(jnp.sum(solid_mask))
total_nodes = N * N * nz
fluid_count = total_nodes - total_solid

print(f"Pipe wall nodes: {pipe_count}", flush=True)
print(f"UMR body nodes: {body_count}", flush=True)
print(f"Total solid: {total_solid} ({total_solid/total_nodes:.1%})", flush=True)
print(f"Fluid nodes: {fluid_count}", flush=True)

# Missing mask
missing_mask = compute_missing_mask(solid_mask)
t_geom = time.perf_counter() - t0
print(f"Geometry setup: {t_geom:.2f}s", flush=True)

# ---------------------------------------------------------------------------
# Body force field (uniform along z, only at fluid nodes)
# ---------------------------------------------------------------------------

force_field = jnp.zeros((N, N, nz, 3))
force_field = force_field.at[..., 2].set(F_body)
# Zero force at solid nodes
force_field = jnp.where(solid_mask[..., None], 0.0, force_field)

# ---------------------------------------------------------------------------
# Run to convergence
# ---------------------------------------------------------------------------

print("\nRunning LBM to convergence (simple BB, rtol=1e-4)...", flush=True)
t0 = time.perf_counter()

f_init = init_equilibrium(N, N, nz)

f_final, n_steps, residual_history = run_to_convergence(
    f_init, tau, solid_mask, missing_mask,
    max_steps=2000,
    check_interval=100,
    rtol=1e-4,
    norm_type="L2",
    use_bouzidi=False,
    force=force_field,
)

t_sim = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

rho, u = compute_macroscopic(f_final, force=force_field)
fluid_mask = ~solid_mask

# Check for NaN/Inf
has_nan = bool(jnp.any(jnp.isnan(u)))
has_inf = bool(jnp.any(jnp.isinf(u)))
u_max = float(jnp.max(jnp.where(fluid_mask[..., None], jnp.abs(u), 0.0)))
u_mean_z = float(jnp.mean(jnp.where(fluid_mask, u[..., 2], 0.0)))

# Drag force on UMR body
# Need to recompute with just UMR body as the solid for momentum exchange
# (not pipe walls — we want the drag on the UMR specifically)
umr_missing = compute_missing_mask(umr_body)
from mime.nodes.environment.lbm.d3q19 import lbm_step_split
f_pre, f_post, _, _ = lbm_step_split(f_final, tau, force=force_field)
from mime.nodes.environment.lbm.bounce_back import apply_bounce_back
f_bb = apply_bounce_back(f_post, f_pre, missing_mask, solid_mask)
drag_force = compute_momentum_exchange_force(f_pre, f_bb, umr_missing)
drag_z = float(drag_force[2])

# Residual info
converged = n_steps < 50000
final_residual = residual_history[-1] if residual_history else float('inf')

print(f"\n{'=' * 60}")
print(f"RESULTS")
print(f"{'=' * 60}")
print(f"Steps to convergence: {n_steps} ({'CONVERGED' if converged else 'DID NOT CONVERGE'})")
print(f"Final residual: {final_residual:.2e}")
print(f"Wall time: {t_sim:.1f}s ({t_sim/n_steps:.3f}s per step)")
print(f"NaN in velocity: {has_nan}")
print(f"Inf in velocity: {has_inf}")
print(f"Max velocity magnitude: {u_max:.6f} lu")
print(f"Mean z-velocity (fluid): {u_mean_z:.6f} lu")
print(f"Drag force on UMR (z): {drag_z:.6e} lu")
print(f"Drag force on UMR (all): [{float(drag_force[0]):.6e}, {float(drag_force[1]):.6e}, {drag_z:.6e}]")

# Stability assessment
stable = not has_nan and not has_inf and converged and u_max < 0.3
print(f"\nSTABILITY CHECK: {'PASS' if stable else 'FAIL'}")
if not stable:
    reasons = []
    if has_nan:
        reasons.append("NaN detected")
    if has_inf:
        reasons.append("Inf detected")
    if not converged:
        reasons.append(f"Did not converge in {n_steps} steps")
    if u_max >= 0.3:
        reasons.append(f"Max velocity {u_max:.4f} too high (Ma > 0.5)")
    print(f"  Failure reasons: {', '.join(reasons)}")

# Residual convergence history
if residual_history:
    print(f"\nResidual history (every 200 steps):")
    for i, r in enumerate(residual_history):
        print(f"  Step {(i+1)*200:>6d}: {r:.2e}")
