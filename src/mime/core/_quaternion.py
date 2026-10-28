"""Unit quaternions ``(w, x, y, z)``, body to world, as pure JAX functions with finite values and finite derivatives
everywhere a body can be, at rest included."""

from __future__ import annotations

from typing import Any

from jax import Array
from jaxtyping import Float

from mime.core import switch
from mime.core._node import jnp

type Quaternion = Float[Array, "4"]
type Vector = Float[Array, "3"]
type Matrix = Float[Array, "3 3"]  # noqa: F722 - a jaxtyping shape

#: Below this squared rotation angle the rotation's exponential is taken from its Taylor series, to the fourth power of
#: the angle: the next terms are O(angle^6) there, and their first two derivatives O(angle^4), all under float64's
#: resolution, so the series has the value and derivatives of the closed form. Stopped at the second power, the half
#: angle's cosine had a first derivative off by angle^3/96, 1e-14 at the threshold. (Above it, the closed form's own
#: second derivative loses digits to cancellation, about 1e-12 just above the threshold.)
_SERIES_BELOW = 1e-8


def safe_norm(v: Any) -> Array:
    """``|v|``, with a derivative of zero at ``v = 0`` rather than NaN: ``switch.norm``'s formula, for the callers
    that evaluate it raw. A node's step evaluates ``switch.norm``, with its owner."""
    return switch._safe_norm(v)  # pyright: ignore[reportPrivateUsage] - the formula's one home is the switch module


def multiply(p: Quaternion, q: Quaternion) -> Quaternion:
    """The Hamilton product ``p q``: the rotation ``q``, then ``p``."""
    w1, x1, y1, z1 = p[0], p[1], p[2], p[3]
    w2, x2, y2, z2 = q[0], q[1], q[2], q[3]
    return jnp.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def conjugate(q: Quaternion) -> Quaternion:
    """The inverse rotation of a unit quaternion."""
    return q * jnp.asarray((1.0, -1.0, -1.0, -1.0), dtype=q.dtype)


def normalized(q: Quaternion, *, owner: str) -> Quaternion:
    """``q`` scaled to unit length. Only for a quaternion near unit length, as an integrator's is: its length is the
    caller's switch ``owner``, at zero, where the direction has no limit."""
    return q / switch.norm(q, owner=owner, resolution=0.0, indicator=lambda: jnp.sqrt(jnp.sum(q * q)))


def rotation_matrix(q: Quaternion) -> Matrix:
    """The matrix ``R`` with ``v_world = R v_body`` for a unit quaternion."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def rotate(q: Quaternion, v: Vector) -> Vector:
    """``v`` from the body frame into the world frame."""
    return rotation_matrix(q) @ v


def rotate_back(q: Quaternion, v: Vector) -> Vector:
    """``v`` from the world frame into the body frame."""
    return rotation_matrix(q).T @ v


def exponential(rotation: Vector) -> Quaternion:
    """The unit quaternion of a rotation by ``|rotation|`` about ``rotation``'s direction (the rotation vector, as a
    body turning at ``omega`` for ``dt`` turns by ``omega * dt``). Finite, with finite derivatives, at a zero rotation,
    where 0.2's update divided by ``|omega|`` and differentiated to NaN."""
    squared = jnp.sum(rotation * rotation)
    small = squared < _SERIES_BELOW
    angle = jnp.sqrt(jnp.where(small, 1.0, squared))
    half_sine_over_angle = jnp.where(small, 0.5 - squared / 48.0 + squared * squared / 3840.0,
                                     jnp.sin(angle / 2) / angle)
    half_cosine = jnp.where(small, 1.0 - squared / 8.0 + squared * squared / 384.0, jnp.cos(angle / 2))
    return jnp.concatenate([half_cosine[None], half_sine_over_angle * rotation])


def identity() -> Quaternion:
    return jnp.asarray((1.0, 0.0, 0.0, 0.0))
