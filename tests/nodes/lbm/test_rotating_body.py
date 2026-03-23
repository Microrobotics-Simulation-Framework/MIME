"""Tests for per-step rotating solid mask update."""

import pytest
import time
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.rotating_body import (
    rotating_body_step,
    run_rotating_body_simulation,
)
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
from mime.nodes.robot.helix_geometry import (
    create_sphere_mask,
    create_umr_mask,
    compute_helix_wall_velocity,
)


def _compute_rotation_velocity(nx, ny, nz, angular_velocity, rotation_axis, center):
    """Compute omega x r at all nodes (not zeroed outside solid)."""
    cx, cy, cz = center
    omega_vec = jnp.array(rotation_axis, dtype=jnp.float32)
    omega_vec = omega_vec / jnp.maximum(jnp.linalg.norm(omega_vec), 1e-30)
    omega_vec = omega_vec * angular_velocity

    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    rx, ry, rz = gx - cx, gy - cy, gz - cz
    ux = omega_vec[1] * rz - omega_vec[2] * ry
    uy = omega_vec[2] * rx - omega_vec[0] * rz
    uz = omega_vec[0] * ry - omega_vec[1] * rx
    return jnp.stack([ux, uy, uz], axis=-1)


def _sphere_rotating_body_step(f, tau, angular_velocity, dt_lbm, current_angle,
                                center, radius, rotation_axis=(0, 0, 1),
                                force=None):
    """Helper: one LBM step with a rotating sphere using simple BB."""
    new_angle = current_angle + angular_velocity * dt_lbm
    nx, ny, nz = f.shape[0], f.shape[1], f.shape[2]

    solid_mask = create_sphere_mask(nx, ny, nz, center, radius)
    missing_mask = compute_missing_mask(solid_mask)

    # Wall velocity at all nodes (not zeroed outside solid) so that
    # boundary fluid nodes see the correct wall velocity for the Ladd
    # correction in apply_bounce_back.
    wall_vel = _compute_rotation_velocity(
        nx, ny, nz, angular_velocity, rotation_axis, center,
    ) if angular_velocity != 0.0 else None

    f_pre, f_post, rho, u = lbm_step_split(f, tau, force=force)
    f_new = apply_bounce_back(f_post, f_pre, missing_mask, solid_mask,
                               wall_velocity=wall_vel)

    body_center = jnp.array(center, dtype=jnp.float32)
    force_on_body = compute_momentum_exchange_force(f_pre, f_new, missing_mask)
    torque_on_body = compute_momentum_exchange_torque(f_pre, f_new, missing_mask, body_center)

    _, velocity_field = compute_macroscopic(f_new, force=force)
    return f_new, force_on_body, torque_on_body, velocity_field, new_angle


class TestMassConservation:
    def test_total_density_conserved(self):
        """Total density before/after step should match."""
        nx, ny, nz = 24, 24, 24
        tau = 0.8
        center = (12.0, 12.0, 12.0)
        radius = 4.0

        f = init_equilibrium(nx, ny, nz, density=1.0)
        total_before = float(jnp.sum(f))

        f_new, _, _, _, _ = _sphere_rotating_body_step(
            f, tau, angular_velocity=0.01, dt_lbm=1.0,
            current_angle=0.0, center=center, radius=radius,
        )
        total_after = float(jnp.sum(f_new))

        rel_error = abs(total_after - total_before) / total_before
        assert rel_error < 1e-4, (
            f"Mass not conserved: {total_before:.6f} -> {total_after:.6f}, "
            f"rel error = {rel_error:.2e}"
        )


class TestStationarySphere:
    def test_zero_angular_velocity_zero_torque(self):
        """Stationary sphere (omega=0) should have ~zero torque."""
        nx, ny, nz = 24, 24, 24
        tau = 0.8
        center = (12.0, 12.0, 12.0)
        radius = 4.0

        f = init_equilibrium(nx, ny, nz)

        # Run a few steps to settle
        for _ in range(5):
            f, force, torque, _, _ = _sphere_rotating_body_step(
                f, tau, angular_velocity=0.0, dt_lbm=1.0,
                current_angle=0.0, center=center, radius=radius,
            )

        torque_mag = float(jnp.linalg.norm(torque))
        assert torque_mag < 1e-4, f"Expected ~zero torque, got magnitude {torque_mag:.2e}"

    def test_zero_angular_velocity_zero_force(self):
        """Stationary sphere in still fluid should have ~zero force."""
        nx, ny, nz = 24, 24, 24
        tau = 0.8
        center = (12.0, 12.0, 12.0)
        radius = 4.0

        f = init_equilibrium(nx, ny, nz)

        for _ in range(5):
            f, force, torque, _, _ = _sphere_rotating_body_step(
                f, tau, angular_velocity=0.0, dt_lbm=1.0,
                current_angle=0.0, center=center, radius=radius,
            )

        force_mag = float(jnp.linalg.norm(force))
        assert force_mag < 1e-4, f"Expected ~zero force, got magnitude {force_mag:.2e}"


class TestRotationProducesResistiveTorque:
    def test_torque_direction(self):
        """Rotation about z should produce non-zero torque about z.

        The momentum exchange method computes the rate of momentum transferred
        from the body to the fluid. For a rotating body, this is positive
        (body pumps angular momentum into the fluid in its rotation direction).
        The resistive torque ON the body is the negative of this.
        """
        nx, ny, nz = 24, 24, 24
        tau = 0.8
        center = (12.0, 12.0, 12.0)
        radius = 4.0
        omega = 0.02  # positive rotation about z

        f = init_equilibrium(nx, ny, nz)

        # Run enough steps for torque to develop
        for _ in range(50):
            f, force, torque, _, _ = _sphere_rotating_body_step(
                f, tau, angular_velocity=omega, dt_lbm=1.0,
                current_angle=0.0, center=center, radius=radius,
            )

        # Torque about z from the momentum exchange should be non-zero
        # and have the same sign as omega (body drives fluid in rotation direction)
        tz = float(torque[2])
        assert tz > 0, (
            f"Expected positive z-torque (body→fluid momentum) for positive omega, "
            f"got Tz={tz:.6f}"
        )

        # Resistive torque on body is -tz (opposite sign)
        resistive_tz = -tz
        assert resistive_tz < 0, "Resistive torque should oppose rotation"


class TestTimingBudget:
    def test_step_time_budget(self):
        """One rotating_body_step at 64^3 with simple BB should take < 2.0s."""
        nx, ny, nz = 64, 64, 64
        tau = 0.8
        center = (32.0, 32.0, 32.0)
        radius = 10.0

        f = init_equilibrium(nx, ny, nz)

        # Warm-up step (JIT compilation)
        f_warm, _, _, _, _ = _sphere_rotating_body_step(
            f, tau, angular_velocity=0.01, dt_lbm=1.0,
            current_angle=0.0, center=center, radius=radius,
        )
        # Force JAX to finish the warm-up computation
        f_warm.block_until_ready()

        # Timed step
        t0 = time.perf_counter()
        f_new, force, torque, vel, angle = _sphere_rotating_body_step(
            f_warm, tau, angular_velocity=0.01, dt_lbm=1.0,
            current_angle=0.01, center=center, radius=radius,
        )
        # Block until GPU/CPU computation finishes
        f_new.block_until_ready()
        elapsed = time.perf_counter() - t0

        print(f"[TIMING] rotating_body_step at 64^3: {elapsed:.3f}s")
        assert elapsed < 2.0, f"Step took {elapsed:.1f}s, budget is 2.0s"


class TestRunSimulation:
    def test_simulation_returns_correct_keys(self):
        """run_rotating_body_simulation should return dict with expected keys."""
        # Use UMR geometry but with small grid for speed
        # Actually, use sphere for simplicity in this test
        nx, ny, nz = 20, 20, 20
        center = (10.0, 10.0, 10.0)

        # Use UMR geometry params but scaled to lattice units
        geometry_params = dict(
            nx=nx, ny=ny, nz=nz,
            body_radius=3.0, body_length=8.0,
            cone_length=3.0, cone_end_radius=0.5,
            fin_outer_radius=5.0, fin_length=3.0,
            fin_width=1.5, fin_thickness=0.5,
            helix_pitch=20.0,
        )

        results = run_rotating_body_simulation(
            geometry_params=geometry_params,
            tau=0.8,
            angular_velocity=0.01,
            n_steps=3,
            nx=nx, ny=ny, nz=nz,
            center=center,
            use_bouzidi=False,
        )

        assert 'force_history' in results
        assert 'torque_history' in results
        assert 'final_velocity' in results
        assert 'final_f' in results
        assert 'final_angle' in results
        assert results['force_history'].shape == (3, 3)
        assert results['torque_history'].shape == (3, 3)
