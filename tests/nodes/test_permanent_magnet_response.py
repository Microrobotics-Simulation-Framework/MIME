"""Tests for PermanentMagnetResponseNode."""

import pytest
import jax
import jax.numpy as jnp

from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.core.quaternion import identity_quat


class TestPermanentMagnetBasic:
    def test_initial_state_zeros(self):
        node = PermanentMagnetResponseNode("pm", 0.001)
        state = node.initial_state()
        assert jnp.allclose(state["magnetization"], jnp.zeros(3))
        assert jnp.allclose(state["magnetic_torque"], jnp.zeros(3))
        assert jnp.allclose(state["magnetic_force"], jnp.zeros(3))

    def test_zero_field_zero_torque(self):
        node = PermanentMagnetResponseNode("pm", 0.001, n_magnets=1)
        state = node.initial_state()
        bi = {
            "field_vector": jnp.zeros(3),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.allclose(new_state["magnetic_torque"], jnp.zeros(3), atol=1e-30)

    def test_perpendicular_field_max_torque(self):
        """When B is perpendicular to moment axis, torque = n*m*B."""
        n_mag = 1
        m_single = 1.07e-3
        B_val = 3e-3  # 3 mT
        node = PermanentMagnetResponseNode(
            "pm", 0.001, n_magnets=n_mag, m_single=m_single,
            moment_axis=(0.0, 1.0, 0.0),
        )
        state = node.initial_state()
        # B along x, moment along y => cross product along z
        bi = {
            "field_vector": jnp.array([B_val, 0.0, 0.0]),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        T = new_state["magnetic_torque"]
        expected_magnitude = n_mag * m_single * B_val
        # Torque should be along -z (y cross x = -z)
        assert jnp.abs(T[2]) > 0
        assert jnp.allclose(jnp.linalg.norm(T), expected_magnitude, rtol=1e-6)

    def test_torque_scales_with_n_magnets(self):
        """Torque should scale linearly with n_magnets."""
        B_val = 3e-3
        bi = {
            "field_vector": jnp.array([B_val, 0.0, 0.0]),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }

        node1 = PermanentMagnetResponseNode("pm1", 0.001, n_magnets=1)
        s1 = node1.update(node1.initial_state(), bi, 0.001)
        T1 = jnp.linalg.norm(s1["magnetic_torque"])

        node2 = PermanentMagnetResponseNode("pm2", 0.001, n_magnets=3)
        s2 = node2.update(node2.initial_state(), bi, 0.001)
        T2 = jnp.linalg.norm(s2["magnetic_torque"])

        assert jnp.allclose(T2, 3.0 * T1, rtol=1e-6)

    def test_uniform_field_zero_force(self):
        """In a uniform field (zero gradient), magnetic force should be zero."""
        node = PermanentMagnetResponseNode("pm", 0.001, n_magnets=1)
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
        node = PermanentMagnetResponseNode("pm", 0.001, n_magnets=1)
        state = node.initial_state()
        grad_B = jnp.eye(3) * 1.0  # 1 T/m gradient
        bi = {
            "field_vector": jnp.array([0.01, 0.0, 0.0]),
            "field_gradient": grad_B,
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.linalg.norm(new_state["magnetic_force"]) > 0

    def test_parallel_field_zero_torque(self):
        """When B is parallel to moment axis, torque should be zero."""
        node = PermanentMagnetResponseNode(
            "pm", 0.001, n_magnets=1, moment_axis=(0.0, 1.0, 0.0),
        )
        state = node.initial_state()
        bi = {
            "field_vector": jnp.array([0.0, 0.01, 0.0]),
            "field_gradient": jnp.zeros((3, 3)),
            "orientation": identity_quat(),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.linalg.norm(new_state["magnetic_torque"]) < 1e-20


class TestPermanentMagnetMetadata:
    def test_meta_set(self):
        assert PermanentMagnetResponseNode.meta is not None
        assert PermanentMagnetResponseNode.meta.algorithm_id == "MIME-NODE-008"

    def test_mime_meta_set(self):
        assert PermanentMagnetResponseNode.mime_meta is not None
        assert PermanentMagnetResponseNode.mime_meta.role.value == "robot_body"
        assert PermanentMagnetResponseNode.mime_meta.biocompatibility is not None

    def test_validate_consistency(self):
        node = PermanentMagnetResponseNode("pm", 0.001)
        errors = node.validate_mime_consistency()
        assert errors == [], f"Consistency errors: {errors}"


class TestPermanentMagnetJAX:
    def test_jit_traceable(self):
        node = PermanentMagnetResponseNode("pm", 0.001, n_magnets=2)
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
        node = PermanentMagnetResponseNode("pm", 0.001, n_magnets=1)
        state = node.initial_state()

        def torque_z(B_x):
            bi = {
                "field_vector": jnp.array([B_x, 0.0, 0.0]),
                "field_gradient": jnp.zeros((3, 3)),
                "orientation": identity_quat(),
            }
            s = node.update(state, bi, 0.001)
            return s["magnetic_torque"][2]

        g = jax.grad(torque_z)(jnp.array(0.01))
        assert jnp.isfinite(g)
        assert g != 0.0
