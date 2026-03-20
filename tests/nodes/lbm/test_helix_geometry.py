"""Tests for helix and sphere geometry generation."""

import pytest
import jax
import jax.numpy as jnp
import math

from mime.nodes.robot.helix_geometry import (
    helix_centreline,
    create_helix_mask,
    create_sphere_mask,
    compute_helix_wall_velocity,
    _rotate_points,
)


class TestHelixCentreline:
    def test_shape(self):
        s = jnp.linspace(0, 1, 50)
        pts = helix_centreline(s, radius=5.0, pitch=10.0, n_turns=2.0)
        assert pts.shape == (50, 3)

    def test_start_at_center(self):
        s = jnp.array([0.0])
        pts = helix_centreline(s, radius=5.0, pitch=10.0, n_turns=2.0,
                               center=(10.0, 20.0, 30.0))
        # At s=0: x = cx + R*cos(0) = 10+5=15, y = cy + R*sin(0) = 20, z = cz = 30
        assert jnp.abs(pts[0, 0] - 15.0) < 1e-5
        assert jnp.abs(pts[0, 1] - 20.0) < 1e-5
        assert jnp.abs(pts[0, 2] - 30.0) < 1e-5

    def test_radius_correct(self):
        """Points should all be at distance R from the helix axis."""
        s = jnp.linspace(0, 1, 100)
        cx, cy = 10.0, 10.0
        pts = helix_centreline(s, radius=5.0, pitch=10.0, n_turns=2.0,
                               center=(cx, cy, 0.0))
        distances = jnp.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        assert jnp.allclose(distances, 5.0, atol=1e-5)

    def test_axial_advance(self):
        """End point should be at z = pitch * n_turns."""
        s = jnp.array([1.0])
        pitch, n_turns = 10.0, 3.0
        pts = helix_centreline(s, radius=5.0, pitch=pitch, n_turns=n_turns,
                               center=(0.0, 0.0, 0.0))
        expected_z = pitch * n_turns
        assert jnp.abs(pts[0, 2] - expected_z) < 1e-4


class TestSphereMask:
    def test_shape(self):
        mask = create_sphere_mask(20, 20, 20, (10.0, 10.0, 10.0), 5.0)
        assert mask.shape == (20, 20, 20)
        assert mask.dtype == jnp.bool_

    def test_center_is_solid(self):
        mask = create_sphere_mask(20, 20, 20, (10.0, 10.0, 10.0), 5.0)
        assert mask[10, 10, 10]

    def test_far_corner_is_fluid(self):
        mask = create_sphere_mask(20, 20, 20, (10.0, 10.0, 10.0), 5.0)
        assert not mask[0, 0, 0]

    def test_volume_reasonable(self):
        """Volume should be approximately (4/3)*pi*R^3."""
        R = 5.0
        mask = create_sphere_mask(30, 30, 30, (15.0, 15.0, 15.0), R)
        volume = float(jnp.sum(mask))
        expected = 4.0/3.0 * math.pi * R**3
        # Discretisation error ~10% at R=5 lattice units
        assert abs(volume - expected) / expected < 0.15


class TestHelixMask:
    def test_shape(self):
        mask = create_helix_mask(
            30, 30, 30,
            helix_radius=5.0, helix_pitch=10.0,
            wire_radius=2.0, n_turns=1.0,
        )
        assert mask.shape == (30, 30, 30)
        assert mask.dtype == jnp.bool_

    def test_has_solid_nodes(self):
        mask = create_helix_mask(
            30, 30, 30,
            helix_radius=5.0, helix_pitch=10.0,
            wire_radius=2.0, n_turns=1.0,
        )
        assert jnp.sum(mask) > 0

    def test_center_axis_is_fluid(self):
        """The helix axis (center) should be fluid (hollow center)."""
        mask = create_helix_mask(
            30, 30, 30,
            helix_radius=8.0, helix_pitch=10.0,
            wire_radius=1.5, n_turns=1.0,
        )
        # Center of the grid — should be hollow (helix wraps around the axis)
        assert not mask[15, 15, 15]

    def test_rotation_changes_mask(self):
        """Rotating the helix should produce a different mask."""
        mask1 = create_helix_mask(
            20, 20, 20,
            helix_radius=4.0, helix_pitch=8.0,
            wire_radius=1.5, n_turns=1.0,
            rotation_angle=0.0,
        )
        mask2 = create_helix_mask(
            20, 20, 20,
            helix_radius=4.0, helix_pitch=8.0,
            wire_radius=1.5, n_turns=1.0,
            rotation_angle=math.pi / 4,
        )
        # Masks should differ (helix rotated by 45 degrees)
        assert not jnp.array_equal(mask1, mask2)

    def test_more_turns_more_solid(self):
        """More turns should mean more solid nodes."""
        mask1 = create_helix_mask(
            40, 20, 20,
            helix_radius=4.0, helix_pitch=8.0,
            wire_radius=1.5, n_turns=1.0,
        )
        mask2 = create_helix_mask(
            40, 20, 20,
            helix_radius=4.0, helix_pitch=8.0,
            wire_radius=1.5, n_turns=2.0,
        )
        assert jnp.sum(mask2) > jnp.sum(mask1)


class TestWallVelocity:
    def test_shape(self):
        solid = create_sphere_mask(20, 20, 20, (10.0, 10.0, 10.0), 3.0)
        vel = compute_helix_wall_velocity(solid, 1.0)
        assert vel.shape == (20, 20, 20, 3)

    def test_zero_outside_solid(self):
        solid = create_sphere_mask(20, 20, 20, (10.0, 10.0, 10.0), 3.0)
        vel = compute_helix_wall_velocity(solid, 1.0)
        # Velocity should be zero at fluid nodes
        fluid = ~solid
        assert jnp.allclose(vel[fluid], 0.0)

    def test_center_is_zero(self):
        """At the rotation center, omega x r = 0."""
        solid = create_sphere_mask(20, 20, 20, (10.0, 10.0, 10.0), 3.0)
        vel = compute_helix_wall_velocity(
            solid, 1.0, center=(10.0, 10.0, 10.0),
        )
        assert jnp.linalg.norm(vel[10, 10, 10]) < 1e-5

    def test_velocity_magnitude_proportional_to_omega(self):
        solid = create_sphere_mask(20, 20, 20, (10.0, 10.0, 10.0), 3.0)
        vel1 = compute_helix_wall_velocity(solid, 1.0)
        vel2 = compute_helix_wall_velocity(solid, 2.0)
        # At a solid node away from center, velocity should double
        idx = (10, 13, 10)  # 3 units from center in y
        if solid[idx]:
            assert jnp.abs(jnp.linalg.norm(vel2[idx]) /
                          jnp.maximum(jnp.linalg.norm(vel1[idx]), 1e-30) - 2.0) < 0.1


class TestRotation:
    def test_identity_rotation(self):
        pts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = _rotate_points(pts, 0.0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        assert jnp.allclose(rotated, pts, atol=1e-6)

    def test_90_deg_rotation_z(self):
        """Rotate (1,0,0) by 90 deg around z -> (0,1,0)."""
        pts = jnp.array([[1.0, 0.0, 0.0]])
        rotated = _rotate_points(pts, math.pi/2, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        assert jnp.allclose(rotated[0], jnp.array([0.0, 1.0, 0.0]), atol=1e-5)

    def test_rotation_preserves_distance(self):
        """Rotation should preserve distance from center."""
        pts = jnp.array([[3.0, 4.0, 0.0]])
        center = (1.0, 1.0, 0.0)
        rotated = _rotate_points(pts, 1.23, (0.0, 0.0, 1.0), center)
        d_before = jnp.linalg.norm(pts[0] - jnp.array(center))
        d_after = jnp.linalg.norm(rotated[0] - jnp.array(center))
        assert jnp.abs(d_before - d_after) < 1e-5
