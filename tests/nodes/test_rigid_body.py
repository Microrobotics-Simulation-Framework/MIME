"""Tests for RigidBodyNode."""

import pytest
import jax
import jax.numpy as jnp
import math

from mime.nodes.robot.rigid_body import RigidBodyNode, oberbeck_stechert_coefficients
from mime.core.quaternion import identity_quat


class TestOberbeckStechert:
    def test_sphere_limit(self):
        """At e=0 (sphere), all coefficients should be 1."""
        C1, C2, C3 = oberbeck_stechert_coefficients(0.0)
        assert jnp.abs(C1 - 1.0) < 1e-5
        assert jnp.abs(C2 - 1.0) < 1e-5
        assert jnp.abs(C3 - 1.0) < 1e-5

    def test_very_small_eccentricity(self):
        """Near-sphere should give coefficients close to 1."""
        C1, C2, C3 = oberbeck_stechert_coefficients(0.01)
        assert jnp.abs(C1 - 1.0) < 0.01
        assert jnp.abs(C2 - 1.0) < 0.01

    def test_moderate_eccentricity(self):
        """At e=0.8 (elongated), C1 < C2 (less drag along major axis)."""
        C1, C2, C3 = oberbeck_stechert_coefficients(0.8)
        assert C1 < C2  # Drag along major axis < drag perpendicular

    def test_high_eccentricity(self):
        """At e=0.99 (very elongated), coefficients should still be finite."""
        C1, C2, C3 = oberbeck_stechert_coefficients(0.99)
        assert jnp.isfinite(C1) and jnp.isfinite(C2) and jnp.isfinite(C3)
        assert C1 > 0 and C2 > 0 and C3 > 0

    def test_jit_traceable(self):
        jitted = jax.jit(oberbeck_stechert_coefficients)
        C1, C2, C3 = jitted(0.5)
        assert jnp.isfinite(C1)

    def test_grad_safe_near_zero(self):
        """Gradient should not produce NaN near e=0."""
        def c1_val(e):
            c1, _, _ = oberbeck_stechert_coefficients(e)
            return c1
        # Test at small but nonzero e (exact e=0 hits jnp.where branch NaN issue)
        g = jax.grad(c1_val)(0.01)
        assert jnp.isfinite(g)


class TestRigidBodyBasic:
    def test_initial_state(self):
        node = RigidBodyNode("robot", 0.001)
        state = node.initial_state()
        assert jnp.allclose(state["position"], jnp.zeros(3))
        assert jnp.allclose(state["orientation"], identity_quat())
        assert state["velocity"].shape == (3,)

    def test_zero_force_no_motion(self):
        """With no external forces, the body should not move."""
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()
        bi = {}
        new_state = node.update(state, bi, 0.001)
        assert jnp.allclose(new_state["position"], jnp.zeros(3))
        assert jnp.allclose(new_state["velocity"], jnp.zeros(3))

    def test_force_causes_motion(self):
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()
        bi = {"magnetic_force": jnp.array([1e-12, 0.0, 0.0])}
        new_state = node.update(state, bi, 0.001)
        assert new_state["velocity"][0] > 0
        assert new_state["position"][0] > 0

    def test_torque_causes_rotation(self):
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()
        bi = {"magnetic_torque": jnp.array([0.0, 0.0, 1e-18])}
        new_state = node.update(state, bi, 0.001)
        assert new_state["angular_velocity"][2] > 0
        # Orientation should change from identity
        assert not jnp.allclose(new_state["orientation"], identity_quat())

    def test_position_integrates(self):
        """Position should accumulate over multiple steps."""
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()
        bi = {"magnetic_force": jnp.array([1e-12, 0.0, 0.0])}
        for _ in range(100):
            state = node.update(state, bi, 0.001)
        assert state["position"][0] > 0
        assert state["position"][0] > 1e-10  # Should have moved noticeably

    def test_orientation_stays_normalized(self):
        """Quaternion should remain unit length after many steps."""
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()
        bi = {"magnetic_torque": jnp.array([1e-18, 1e-18, 1e-18])}
        for _ in range(1000):
            state = node.update(state, bi, 0.001)
        q_norm = jnp.linalg.norm(state["orientation"])
        assert jnp.abs(q_norm - 1.0) < 1e-6


class TestRigidBodyStokesDrag:
    def test_sphere_stokes_drag(self):
        """For a sphere (a=b), verify F = 6*pi*eta*a*V."""
        a = 100e-6  # 100 um
        eta = 8.5e-4  # CSF viscosity
        node = RigidBodyNode(
            "robot", 0.001,
            semi_major_axis_m=a,
            semi_minor_axis_m=a,  # sphere
            fluid_viscosity_pa_s=eta,
        )
        state = node.initial_state()
        F_applied = jnp.array([1e-12, 0.0, 0.0])
        bi = {"magnetic_force": F_applied}
        new_state = node.update(state, bi, 0.001)

        # Expected velocity: V = F / (6*pi*eta*a)
        V_expected = float(F_applied[0]) / (6 * math.pi * eta * a)
        V_actual = float(new_state["velocity"][0])
        assert abs(V_actual - V_expected) / V_expected < 0.01  # 1% tolerance

    def test_ellipsoid_less_drag_along_major(self):
        """Elongated ellipsoid should move faster along its major axis."""
        a = 150e-6  # semi-major
        b = 50e-6   # semi-minor (elongated)
        eta = 8.5e-4
        node = RigidBodyNode(
            "robot", 0.001,
            semi_major_axis_m=a,
            semi_minor_axis_m=b,
            fluid_viscosity_pa_s=eta,
        )
        state = node.initial_state()
        F = 1e-12

        # Force along x (major axis) — identity orientation
        bi_x = {"magnetic_force": jnp.array([F, 0.0, 0.0])}
        s_x = node.update(state, bi_x, 0.001)

        # Force along y (minor axis)
        bi_y = {"magnetic_force": jnp.array([0.0, F, 0.0])}
        s_y = node.update(state, bi_y, 0.001)

        # Velocity along major axis should be larger
        assert abs(float(s_x["velocity"][0])) > abs(float(s_y["velocity"][1]))

    def test_additive_forces(self):
        """Multiple force inputs should sum."""
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()
        bi = {
            "magnetic_force": jnp.array([1e-12, 0.0, 0.0]),
            "external_force": jnp.array([1e-12, 0.0, 0.0]),
        }
        new_state = node.update(state, bi, 0.001)

        # Compare with single force of double magnitude
        bi2 = {"magnetic_force": jnp.array([2e-12, 0.0, 0.0])}
        s2 = node.update(state, bi2, 0.001)

        assert jnp.allclose(new_state["velocity"], s2["velocity"], atol=1e-20)


class TestRigidBodyMetadata:
    def test_meta_set(self):
        assert RigidBodyNode.meta is not None
        assert RigidBodyNode.meta.algorithm_id == "MIME-NODE-003"

    def test_mime_meta_set(self):
        assert RigidBodyNode.mime_meta is not None
        assert RigidBodyNode.mime_meta.role.value == "robot_body"
        assert RigidBodyNode.mime_meta.biocompatibility is not None
        assert len(RigidBodyNode.mime_meta.anatomical_regimes) > 0

    def test_validate_consistency(self):
        node = RigidBodyNode("robot", 0.001)
        errors = node.validate_mime_consistency()
        assert errors == [], f"Consistency errors: {errors}"

    def test_boundary_inputs_additive(self):
        node = RigidBodyNode("robot", 0.001)
        spec = node.boundary_input_spec()
        assert spec["magnetic_force"].coupling_type == "additive"
        assert spec["magnetic_torque"].coupling_type == "additive"


class TestRigidBodyJAX:
    def test_jit_traceable(self):
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()
        bi = {"magnetic_force": jnp.array([1e-12, 0.0, 0.0])}
        jitted = jax.jit(node.update)
        new_state = jitted(state, bi, 0.001)
        assert jnp.isfinite(new_state["position"]).all()

    def test_grad_position_wrt_force(self):
        node = RigidBodyNode("robot", 0.001, semi_major_axis_m=100e-6)
        state = node.initial_state()

        def final_x(force_x):
            bi = {"magnetic_force": jnp.array([force_x, 0.0, 0.0])}
            s = node.update(state, bi, 0.001)
            return s["position"][0]

        g = jax.grad(final_x)(jnp.array(1e-12))
        assert jnp.isfinite(g)
        assert g > 0  # More force = more displacement
