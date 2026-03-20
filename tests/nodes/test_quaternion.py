"""Tests for quaternion utilities."""

import pytest
import jax
import jax.numpy as jnp

from mime.core.quaternion import (
    identity_quat, quat_multiply, quat_conjugate, quat_normalize,
    quat_to_rotation_matrix, rotate_vector, rotate_vector_inverse,
    quat_from_angular_velocity,
)


class TestQuaternionBasic:
    def test_identity(self):
        q = identity_quat()
        assert jnp.allclose(q, jnp.array([1.0, 0.0, 0.0, 0.0]))

    def test_identity_rotation(self):
        q = identity_quat()
        v = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(rotate_vector(q, v), v)

    def test_multiply_identity(self):
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        qi = identity_quat()
        assert jnp.allclose(quat_multiply(q, qi), q, atol=1e-6)
        assert jnp.allclose(quat_multiply(qi, q), q, atol=1e-6)

    def test_conjugate(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        qc = quat_conjugate(q)
        assert jnp.allclose(qc, jnp.array([1.0, -2.0, -3.0, -4.0]))

    def test_normalize(self):
        q = jnp.array([1.0, 1.0, 1.0, 1.0])
        qn = quat_normalize(q)
        assert jnp.abs(jnp.linalg.norm(qn) - 1.0) < 1e-6

    def test_rotation_matrix_identity(self):
        R = quat_to_rotation_matrix(identity_quat())
        assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

    def test_90_deg_rotation_around_z(self):
        """Rotating [1,0,0] by 90 deg around z should give [0,1,0]."""
        angle = jnp.pi / 2
        q = jnp.array([jnp.cos(angle/2), 0.0, 0.0, jnp.sin(angle/2)])
        v = jnp.array([1.0, 0.0, 0.0])
        result = rotate_vector(q, v)
        assert jnp.allclose(result, jnp.array([0.0, 1.0, 0.0]), atol=1e-5)

    def test_inverse_rotation(self):
        angle = jnp.pi / 3
        q = jnp.array([jnp.cos(angle/2), 0.0, 0.0, jnp.sin(angle/2)])
        v = jnp.array([1.0, 2.0, 3.0])
        rotated = rotate_vector(q, v)
        recovered = rotate_vector_inverse(q, rotated)
        assert jnp.allclose(recovered, v, atol=1e-5)

    def test_angular_velocity_integration(self):
        """Small rotation from angular velocity should approximately match."""
        omega = jnp.array([0.0, 0.0, 1.0])  # 1 rad/s around z
        dt = 0.01  # small timestep
        dq = quat_from_angular_velocity(omega, dt)
        # Should be approximately [cos(0.005), 0, 0, sin(0.005)]
        expected_w = jnp.cos(0.005)
        expected_z = jnp.sin(0.005)
        assert jnp.abs(dq[0] - expected_w) < 1e-6
        assert jnp.abs(dq[3] - expected_z) < 1e-6

    def test_zero_angular_velocity(self):
        dq = quat_from_angular_velocity(jnp.zeros(3), 0.01)
        assert jnp.allclose(dq, identity_quat(), atol=1e-6)


class TestQuaternionJAX:
    def test_jit_all(self):
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        v = jnp.array([1.0, 0.0, 0.0])
        jax.jit(quat_multiply)(q, q)
        jax.jit(quat_conjugate)(q)
        jax.jit(quat_normalize)(q)
        jax.jit(quat_to_rotation_matrix)(q)
        jax.jit(rotate_vector)(q, v)
        jax.jit(rotate_vector_inverse)(q, v)

    def test_grad_rotation(self):
        def rotated_x(angle):
            q = jnp.array([jnp.cos(angle/2), 0.0, 0.0, jnp.sin(angle/2)])
            v = jnp.array([1.0, 0.0, 0.0])
            return rotate_vector(q, v)[1]  # y-component
        g = jax.grad(rotated_x)(jnp.array(0.0))
        # d/d(angle) of sin(angle) at angle=0 = 1/2 (chain rule through quaternion)
        assert jnp.isfinite(g)
