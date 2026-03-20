"""Quaternion utilities for rigid body orientation — JAX-traceable.

Convention: q = [w, x, y, z] where w is the scalar part.
All functions are pure jnp operations, fully differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of two quaternions q1 * q2.

    Parameters
    ----------
    q1, q2 : jnp.ndarray, shape (4,)
        Quaternions in [w, x, y, z] convention.

    Returns
    -------
    jnp.ndarray, shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion conjugate (inverse for unit quaternions)."""
    return q.at[1:].multiply(-1)


def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    """Normalize quaternion to unit length."""
    return q / jnp.linalg.norm(q)


def quat_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Convert unit quaternion to 3x3 rotation matrix.

    Parameters
    ----------
    q : jnp.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    jnp.ndarray, shape (3, 3)
        Rotation matrix R such that v_lab = R @ v_body.
    """
    w, x, y, z = q
    return jnp.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def rotate_vector(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotate vector v by quaternion q: v' = R(q) @ v."""
    R = quat_to_rotation_matrix(q)
    return R @ v


def rotate_vector_inverse(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Inverse rotate: v_body = R(q)^T @ v_lab."""
    R = quat_to_rotation_matrix(q)
    return R.T @ v


def quat_from_angular_velocity(omega: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Quaternion increment from angular velocity over timestep dt.

    Returns the quaternion dq such that q_new = quat_multiply(dq, q_old).
    Uses first-order approximation: dq ≈ [1, omega*dt/2].

    Parameters
    ----------
    omega : jnp.ndarray, shape (3,)
        Angular velocity in rad/s (lab frame).
    dt : float
        Timestep in seconds.
    """
    half_angle = jnp.linalg.norm(omega) * dt / 2.0
    # Safe normalization for zero angular velocity
    axis = omega / jnp.maximum(jnp.linalg.norm(omega), 1e-30)
    w = jnp.cos(half_angle)
    xyz = axis * jnp.sin(half_angle)
    return jnp.array([w, xyz[0], xyz[1], xyz[2]])


def identity_quat() -> jnp.ndarray:
    """Identity quaternion [1, 0, 0, 0]."""
    return jnp.array([1.0, 0.0, 0.0, 0.0])
