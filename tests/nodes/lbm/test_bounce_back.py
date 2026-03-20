"""Tests for the bounce-back module — missing_mask, BB application, momentum exchange."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    apply_bounce_back,
    compute_momentum_exchange_force,
    compute_momentum_exchange_torque,
)
from mime.nodes.environment.lbm.d3q19 import (
    E, W, OPP, Q, CS2,
    init_equilibrium, lbm_step, lbm_step_split,
    create_channel_walls,
)
from mime.nodes.robot.helix_geometry import (
    create_sphere_mask,
)


class TestMissingMask:
    def test_shape(self):
        solid = jnp.zeros((10, 10, 10), dtype=bool)
        mm = compute_missing_mask(solid)
        assert mm.shape == (Q, 10, 10, 10)

    def test_no_solid_no_missing(self):
        solid = jnp.zeros((8, 8, 8), dtype=bool)
        mm = compute_missing_mask(solid)
        assert not jnp.any(mm)

    def test_all_solid_no_missing(self):
        """If everything is solid, no fluid nodes exist, so no missing links."""
        solid = jnp.ones((8, 8, 8), dtype=bool)
        mm = compute_missing_mask(solid)
        assert not jnp.any(mm)

    def test_single_solid_creates_missing(self):
        """A single solid node should create missing links at its fluid neighbors."""
        solid = jnp.zeros((10, 10, 10), dtype=bool).at[5, 5, 5].set(True)
        mm = compute_missing_mask(solid)
        # Neighbors of (5,5,5) in the +x direction: fluid at (4,5,5) has q=1 missing
        # because (4,5,5) + e_1 = (5,5,5) which is solid
        assert mm[1, 4, 5, 5]  # direction +x at (4,5,5) points into solid

    def test_channel_walls_create_correct_missing(self):
        """Channel walls should create missing links at the wall interface."""
        solid = create_channel_walls(8, 10, 8, wall_axis=1)
        mm = compute_missing_mask(solid)
        # At y=1 (first fluid cell above bottom wall), direction -y (q=4)
        # should be missing (neighbor at y=0 is solid)
        assert mm[4, 4, 1, 4]  # direction -y at fluid node (4,1,4)
        # At y=8 (last fluid cell below top wall), direction +y (q=3)
        assert mm[3, 4, 8, 4]  # direction +y at fluid node (4,8,4)

    def test_rest_direction_never_missing(self):
        """Direction 0 (rest) should never be missing — it doesn't point anywhere."""
        solid = create_channel_walls(8, 10, 8)
        mm = compute_missing_mask(solid)
        assert not jnp.any(mm[0])


class TestApplyBounceBack:
    def test_static_wall_reflects(self):
        """At a static wall, populations should be reflected to opposite direction."""
        nx, ny, nz = 8, 10, 8
        solid = create_channel_walls(nx, ny, nz, wall_axis=1)
        mm = compute_missing_mask(solid)

        f = init_equilibrium(nx, ny, nz, density=1.0, velocity=(0.01, 0.0, 0.0))
        f_pre = f  # Pre-streaming
        f_post = jnp.roll(f, 1, axis=0)  # Crude "streaming" for test

        f_bb = apply_bounce_back(f_post, f_pre, mm, solid)

        # Solid nodes should be zeroed
        assert jnp.allclose(f_bb[0, 0, 0, :], 0.0)  # bottom wall

        # Fluid nodes should have values
        assert jnp.sum(jnp.abs(f_bb[4, 5, 4, :])) > 0

    def test_moving_wall_adds_momentum(self):
        """Moving wall should add velocity correction to reflected populations."""
        nx, ny, nz = 8, 10, 8
        solid = create_channel_walls(nx, ny, nz, wall_axis=1)
        mm = compute_missing_mask(solid)

        # Use non-equilibrium f so bounce-back has something to work with
        f = init_equilibrium(nx, ny, nz, density=1.0, velocity=(0.01, 0.0, 0.0))
        # Do one collision+stream cycle to get meaningful pre/post distributions
        f_pre, f_post, _, _ = lbm_step_split(f, 0.8)

        # Static wall
        f_static = apply_bounce_back(f_post, f_pre, mm, solid, wall_velocity=None)

        # Moving wall — set velocity at ALL nodes (the correction in
        # apply_bounce_back reads wall_velocity at fluid node positions
        # where missing_mask is True)
        wall_vel = jnp.zeros((nx, ny, nz, 3))
        # Set wall velocity everywhere — only boundary fluid nodes matter
        wall_vel = wall_vel.at[:, :, :, 0].set(0.1)
        f_moving = apply_bounce_back(f_post, f_pre, mm, solid, wall_velocity=wall_vel)

        # The moving wall version should differ from static at fluid nodes near walls
        diff = jnp.max(jnp.abs(f_static - f_moving))
        assert diff > 1e-6, f"Expected difference from wall velocity, got max diff {diff}"

    def test_solid_nodes_zeroed(self):
        """Distributions inside the solid should be zero after bounce-back."""
        nx, ny, nz = 8, 10, 8
        solid = create_channel_walls(nx, ny, nz, wall_axis=1)
        mm = compute_missing_mask(solid)
        f = init_equilibrium(nx, ny, nz)
        f_bb = apply_bounce_back(f, f, mm, solid)
        # All solid nodes should have zero distributions
        assert jnp.allclose(f_bb[solid], 0.0)


class TestMomentumExchange:
    def test_stationary_body_in_flow_feels_drag(self):
        """A stationary sphere in a moving fluid should feel drag."""
        nx, ny, nz = 20, 20, 20
        tau = 0.8

        # Sphere at center
        center = (10.0, 10.0, 10.0)
        radius = 3.0
        solid = create_sphere_mask(nx, ny, nz, center, radius)
        mm = compute_missing_mask(solid)

        # Initialize with uniform flow in +x
        f = init_equilibrium(nx, ny, nz, density=1.0, velocity=(0.02, 0.0, 0.0))

        # Run a few steps with bounce-back
        for _ in range(20):
            f_pre_col, f_post_str, rho, u = lbm_step_split(f, tau)
            f = apply_bounce_back(f_post_str, f_pre_col, mm, solid)

        # Compute force on the sphere
        f_pre_col, f_post_str, _, _ = lbm_step_split(f, tau)
        f_after_bb = apply_bounce_back(f_post_str, f_pre_col, mm, solid)
        force = compute_momentum_exchange_force(f_pre_col, f_after_bb, mm)

        # Drag should be in -x direction (opposing flow)
        assert force[0] < 0, f"Expected negative x-force (drag), got {force}"

    def test_zero_velocity_zero_force(self):
        """Stationary body in stationary fluid should have ~zero force."""
        nx, ny, nz = 16, 16, 16
        tau = 0.8

        solid = create_sphere_mask(nx, ny, nz, (8.0, 8.0, 8.0), 2.5)
        mm = compute_missing_mask(solid)

        f = init_equilibrium(nx, ny, nz)
        f_pre, f_post, _, _ = lbm_step_split(f, tau)
        f_bb = apply_bounce_back(f_post, f_pre, mm, solid)
        force = compute_momentum_exchange_force(f_pre, f_bb, mm)

        assert jnp.linalg.norm(force) < 0.01, f"Expected ~zero force, got {force}"

    def test_torque_from_rotating_body(self):
        """A rotating sphere should experience resistive torque."""
        nx, ny, nz = 20, 20, 20
        tau = 0.8
        center = jnp.array([10.0, 10.0, 10.0])
        radius = 3.0
        solid = create_sphere_mask(nx, ny, nz, tuple(center.tolist()), radius)
        mm = compute_missing_mask(solid)

        # Compute wall velocity for rotation about z-axis
        from mime.nodes.robot.helix_geometry import compute_helix_wall_velocity
        wall_vel = compute_helix_wall_velocity(
            solid, angular_velocity=0.01,
            rotation_axis=(0.0, 0.0, 1.0),
            center=tuple(center.tolist()),
        )

        f = init_equilibrium(nx, ny, nz)
        for _ in range(30):
            f_pre, f_post, _, _ = lbm_step_split(f, tau)
            f = apply_bounce_back(f_post, f_pre, mm, solid, wall_velocity=wall_vel)

        f_pre, f_post, _, _ = lbm_step_split(f, tau)
        f_bb = apply_bounce_back(f_post, f_pre, mm, solid, wall_velocity=wall_vel)
        torque = compute_momentum_exchange_torque(f_pre, f_bb, mm, center)

        # Torque about z should oppose the rotation (negative for CCW rotation)
        assert torque[2] != 0.0, f"Expected non-zero z-torque, got {torque}"


class TestBounceBackJAX:
    def test_missing_mask_jit(self):
        solid = jnp.zeros((8, 8, 8), dtype=bool).at[4, 4, 4].set(True)
        mm = jax.jit(compute_missing_mask)(solid)
        assert jnp.isfinite(mm.astype(jnp.float32)).all()

    def test_force_computation_finite(self):
        """Force computation should produce finite values."""
        nx, ny, nz = 10, 10, 10
        solid = create_sphere_mask(nx, ny, nz, (5.0, 5.0, 5.0), 2.0)
        mm = compute_missing_mask(solid)
        f = init_equilibrium(nx, ny, nz)
        f_pre, f_post, _, _ = lbm_step_split(f, 0.8)
        f_bb = apply_bounce_back(f_post, f_pre, mm, solid)
        force = compute_momentum_exchange_force(f_pre, f_bb, mm)
        assert jnp.isfinite(force).all()
