"""Tests for MagneticResponseNode."""

import pytest
import jax
import jax.numpy as jnp

from mime.nodes.robot.magnetic_response import MagneticResponseNode, MU_0
from mime.core.quaternion import identity_quat, quat_from_angular_velocity


class TestMagneticResponseBasic:
    def test_initial_state_zeros(self):
        node = MagneticResponseNode("mag", 0.001)
        state = node.initial_state()
        assert jnp.allclose(state["magnetization"], jnp.zeros(3))
        assert jnp.allclose(state["magnetic_torque"], jnp.zeros(3))
        assert jnp.allclose(state["magnetic_force"], jnp.zeros(3))

    def test_zero_field_zero_torque(self):
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15)
        state = node.initial_state()
        bi = {
            "field_vector": jnp.zeros(3),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.allclose(new_state["magnetic_torque"], jnp.zeros(3), atol=1e-30)

    def test_aligned_field_zero_torque(self):
        """When B is aligned with the body e1 axis, torque should be ~zero."""
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()
        # B along x (body e1), identity orientation
        bi = {
            "field_vector": jnp.array([0.01, 0.0, 0.0]),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        # m is along x, B is along x => m x B = 0
        assert jnp.linalg.norm(new_state["magnetic_torque"]) < 1e-25

    def test_oblique_field_nonzero_torque(self):
        """When B is at an oblique angle to body axis, torque should be non-zero.

        For an anisotropic body (n_axi != n_rad), a field at 45 degrees
        induces m that is NOT parallel to B (because chi is anisotropic),
        so m x B != 0.
        """
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()
        # B at 45 degrees in xy-plane — oblique to both body axes
        B_45 = jnp.array([0.01, 0.01, 0.0]) / jnp.sqrt(2.0)
        bi = {
            "field_vector": B_45,
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.linalg.norm(new_state["magnetic_torque"]) > 0

    def test_torque_direction_is_along_z(self):
        """For B at 45 deg in xy-plane with anisotropic body, torque is along z.

        m has different magnitudes along body x (chi_axi) and y (chi_rad),
        so m x B has a z-component.
        """
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()
        B_45 = jnp.array([0.01, 0.01, 0.0]) / jnp.sqrt(2.0)
        bi = {
            "field_vector": B_45,
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        T = new_state["magnetic_torque"]
        # Torque should be primarily along z (from cross product in xy-plane)
        assert jnp.abs(T[2]) > 0
        # x and y components should be zero (2D cross product only has z)
        assert jnp.abs(T[0]) < 1e-25
        assert jnp.abs(T[1]) < 1e-25

    def test_uniform_field_zero_force(self):
        """In a uniform field (zero gradient), magnetic force should be zero."""
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15)
        state = node.initial_state()
        bi = {
            "field_vector": jnp.array([0.01, 0.0, 0.0]),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.allclose(new_state["magnetic_force"], jnp.zeros(3), atol=1e-30)

    def test_gradient_field_nonzero_force(self):
        """In a field gradient, magnetic force should be non-zero."""
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()
        grad_B = jnp.eye(3) * 1.0  # 1 T/m gradient in all directions
        bi = {
            "field_vector": jnp.array([0.01, 0.0, 0.0]),
            "field_gradient": grad_B,
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.linalg.norm(new_state["magnetic_force"]) > 0

    def test_saturation_clipping(self):
        """With very strong field, magnetization should be clipped at m_sat."""
        m_sat = 1e6  # A/m
        node = MagneticResponseNode(
            "mag", 0.001, volume_m3=1e-15, n_axi=0.2, m_sat=m_sat,
        )
        state = node.initial_state()
        # Very strong field — should saturate
        bi = {
            "field_vector": jnp.array([10.0, 0.0, 0.0]),  # 10 T!
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        m_mag = jnp.linalg.norm(new_state["magnetization"])
        assert m_mag <= m_sat * 1.01  # Allow small floating-point margin

    def test_no_saturation_by_default(self):
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()
        bi = {
            "field_vector": jnp.array([10.0, 0.0, 0.0]),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        # Without saturation, magnetization should be very large
        m_mag = jnp.linalg.norm(new_state["magnetization"])
        assert m_mag > 1e6


class TestMagneticResponseRotation:
    def test_rotated_body_changes_torque_direction(self):
        """Rotating the body should change the torque direction.

        Both B and the body's principal axis must be in a configuration
        where their cross product is non-trivially affected by the
        rotation. Aligning B with the body's symmetry axis (e1) and
        then rotating about that same axis keeps m parallel to B in
        both cases (torque = 0 identically), so we choose an off-axis
        rotation that tilts e1 out of the field.
        """
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()

        B = jnp.array([0.01, 0.0, 0.0])

        # Identity orientation: e1 is aligned with B → m ‖ B → τ = 0.
        bi1 = {"field_vector": B, "field_gradient": jnp.zeros((3, 3)),
               "orientation": identity_quat()}
        s1 = node.update(state, bi1, 0.001)

        # 45° rotation about world-y: tilts the body's e1 out of B,
        # so m picks up a perpendicular component and m × B ≠ 0.
        h = 0.5 * (jnp.pi / 4.0)
        q_rot = jnp.array([jnp.cos(h), 0.0, jnp.sin(h), 0.0])
        bi2 = {"field_vector": B, "field_gradient": jnp.zeros((3, 3)),
               "orientation": q_rot}
        s2 = node.update(state, bi2, 0.001)

        # s2's torque is small in absolute terms because v = 1e-15 m³
        # is a 0.1 µm body, but it's many orders of magnitude above
        # the s1 ≈ 0 case — the test's job is to confirm the rotation
        # *matters*, not to verify the magnitude.
        s1_n = float(jnp.linalg.norm(s1["magnetic_torque"]))
        s2_n = float(jnp.linalg.norm(s2["magnetic_torque"]))
        assert s2_n > 1e-15
        assert s2_n > 1e6 * max(s1_n, 1e-30)


class TestMagneticResponseMetadata:
    def test_meta_set(self):
        assert MagneticResponseNode.meta is not None
        assert MagneticResponseNode.meta.algorithm_id == "MIME-NODE-002"

    def test_mime_meta_set(self):
        assert MagneticResponseNode.mime_meta is not None
        assert MagneticResponseNode.mime_meta.role.value == "robot_body"
        assert MagneticResponseNode.mime_meta.biocompatibility is not None

    def test_validate_consistency(self):
        node = MagneticResponseNode("mag", 0.001)
        errors = node.validate_mime_consistency()
        assert errors == [], f"Consistency errors: {errors}"


class TestMagneticResponseJAX:
    def test_jit_traceable(self):
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()
        bi = {
            "field_vector": jnp.array([0.01, 0.0, 0.0]),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        jitted = jax.jit(node.update)
        new_state = jitted(state, bi, 0.001)
        assert new_state["magnetic_torque"].shape == (3,)

    def test_grad_wrt_field(self):
        node = MagneticResponseNode("mag", 0.001, volume_m3=1e-15, n_axi=0.2)
        state = node.initial_state()

        def torque_z(B_y):
            # Use oblique field so torque is non-zero
            bi = {
                "field_vector": jnp.array([0.01, B_y, 0.0]),
                "field_gradient": jnp.zeros((3, 3)),
                "orientation": identity_quat(),
            }
            s = node.update(state, bi, 0.001)
            # Use z-component directly (avoids norm-of-zero gradient issue)
            return s["magnetic_torque"][2]

        g = jax.grad(torque_z)(jnp.array(0.01))
        assert jnp.isfinite(g)
        assert g != 0.0
