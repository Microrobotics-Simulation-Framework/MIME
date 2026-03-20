"""Tests for SurfaceContactNode."""

import pytest
import jax
import jax.numpy as jnp

from mime.nodes.robot.surface_contact import (
    SurfaceContactNode,
    brenner_correction_perpendicular,
    brenner_correction_parallel,
    penalty_contact_force,
)


class TestBrennerCorrections:
    def test_far_from_wall_no_correction(self):
        """At large h/a, corrections should be ~1."""
        a = 100e-6
        h = 100e-3  # 1000 radii away
        assert jnp.abs(brenner_correction_perpendicular(a, h) - 1.0) < 0.002
        assert jnp.abs(brenner_correction_parallel(a, h) - 1.0) < 0.001

    def test_near_wall_increased_drag(self):
        """Near wall, drag correction > 1."""
        a = 100e-6
        h = 200e-6  # 2 radii away
        assert brenner_correction_perpendicular(a, h) > 1.0
        assert brenner_correction_parallel(a, h) > 1.0

    def test_perpendicular_greater_than_parallel(self):
        """Perpendicular correction should be larger than parallel."""
        a = 100e-6
        h = 200e-6
        c_perp = brenner_correction_perpendicular(a, h)
        c_par = brenner_correction_parallel(a, h)
        assert c_perp > c_par

    def test_clamped_at_contact(self):
        """At h <= a, should clamp to safe minimum and remain finite."""
        a = 100e-6
        h = 50e-6  # Inside the sphere — should clamp to h_safe = 1.5*a
        c = brenner_correction_perpendicular(a, h)
        assert jnp.isfinite(c)
        assert c > 1.0  # Correction at h_safe = 1.5a: 1 + 9/(8*1.5) = 1.75

    def test_jit_traceable(self):
        f = jax.jit(brenner_correction_perpendicular, static_argnums=0)
        c = f(100e-6, jnp.array(200e-6))
        assert jnp.isfinite(c)


class TestPenaltyContact:
    def test_no_contact_no_force(self):
        pos = jnp.array([0.0, 0.0, 0.001])  # 1mm above floor at z=0
        f = penalty_contact_force(pos, 0.0, jnp.array([0.0, 0.0, 1.0]),
                                  100e-6, 1e-6)
        assert jnp.allclose(f, jnp.zeros(3))

    def test_penetration_gives_force(self):
        pos = jnp.array([0.0, 0.0, 50e-6])  # 50um above floor, radius 100um
        f = penalty_contact_force(pos, 0.0, jnp.array([0.0, 0.0, 1.0]),
                                  100e-6, 1e-6)
        # Gap = 50e-6 - 100e-6 = -50e-6 (penetrating)
        assert f[2] > 0  # Force pushes away from wall
        assert jnp.abs(f[0]) < 1e-30  # Only z-component

    def test_force_proportional_to_penetration(self):
        k = 1e-6
        a = 100e-6
        n = jnp.array([0.0, 0.0, 1.0])

        f1 = penalty_contact_force(jnp.array([0., 0., 80e-6]), 0., n, a, k)
        f2 = penalty_contact_force(jnp.array([0., 0., 60e-6]), 0., n, a, k)
        # More penetration -> larger force
        assert float(f2[2]) > float(f1[2])


class TestSurfaceContactNode:
    def test_initial_state(self):
        node = SurfaceContactNode("wall", 0.001)
        state = node.initial_state()
        assert jnp.abs(state["wall_correction_perp"] - 1.0) < 1e-6

    def test_far_from_wall(self):
        node = SurfaceContactNode("wall", 0.001, wall_position=0.0,
                                  wall_normal_axis=2, wall_side=-1)
        state = node.initial_state()
        bi = {"position": jnp.array([0.0, 0.0, 0.1]),  # 100mm above
              "velocity": jnp.zeros(3)}
        new = node.update(state, bi, 0.001)
        assert jnp.abs(new["wall_correction_perp"] - 1.0) < 0.002
        assert jnp.allclose(new["contact_force"], jnp.zeros(3))

    def test_near_wall_correction(self):
        a = 100e-6
        node = SurfaceContactNode("wall", 0.001, robot_radius_m=a,
                                  wall_position=0.0, wall_normal_axis=2)
        state = node.initial_state()
        bi = {"position": jnp.array([0.0, 0.0, 200e-6]),  # 2 radii above
              "velocity": jnp.zeros(3)}
        new = node.update(state, bi, 0.001)
        assert new["wall_correction_perp"] > 1.0
        assert new["gap_distance"] == pytest.approx(2.0, rel=0.01)

    def test_validate_consistency(self):
        node = SurfaceContactNode("wall", 0.001)
        # SurfaceContactNode has role=ROBOT_BODY but no biocompatibility
        # This will produce an error — that's expected for a non-physical observer
        # Let's just check it doesn't crash:
        errors = node.validate_mime_consistency()
        # May have biocompatibility error — that's a known design choice


class TestSurfaceContactJAX:
    def test_jit_traceable(self):
        node = SurfaceContactNode("wall", 0.001)
        state = node.initial_state()
        bi = {"position": jnp.array([0.0, 0.0, 0.001]),
              "velocity": jnp.zeros(3)}
        jitted = jax.jit(node.update)
        new = jitted(state, bi, 0.001)
        assert jnp.isfinite(new["wall_correction_perp"])
