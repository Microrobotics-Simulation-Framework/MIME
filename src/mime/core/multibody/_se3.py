"""Rigid transforms as MIME's poses, pure JAX: ``(x, y, z, qw, qx, qy, qz)``, a translation and then a unit quaternion
in wxyz order, mapping a point from the pose's own frame into its parent's as ``R(q) p + t``. Built on ``mime.core``'s
quaternion helpers, so every function is finite, with finite derivatives, at a zero rotation."""

from __future__ import annotations

from typing import Any

from jax import Array
from jaxtyping import Float

from mime.core import _quaternion as quaternion
from mime.core._node import jnp

type Pose = Float[Array, "7"]
type Vector = Float[Array, "3"]
type Quaternion = Float[Array, "4"]

#: Below this squared sine of half the angle, the rotation vector is taken from its series, whose next term is under
#: float64's resolution there: the closed form divides by the sine.
_SERIES_BELOW = 1e-8


def pose(translation: Any, rotation: Any) -> Pose:
    return jnp.concatenate([translation, rotation])


def compose(outer: Pose, inner: Pose) -> Pose:
    """``inner``, a pose in ``outer``'s frame, expressed in ``outer``'s parent."""
    return pose(outer[:3] + quaternion.rotate(outer[3:], inner[:3]), quaternion.multiply(outer[3:], inner[3:]))


def inverse(p: Pose) -> Pose:
    """The parent's frame placed in the pose's own."""
    turned_back = quaternion.conjugate(p[3:])
    return pose(-quaternion.rotate(turned_back, p[:3]), turned_back)


def transform_point(p: Pose, point: Vector) -> Vector:
    return p[:3] + quaternion.rotate(p[3:], point)


def turn(axis: Vector, angle: Any) -> Quaternion:
    """The rotation by ``angle`` about the unit ``axis``."""
    return quaternion.exponential(angle * axis)


def rotation_vector(q: Quaternion) -> Vector:
    """The rotation of the unit quaternion ``q`` as a rotation vector (angle times unit axis), the shorter way round:
    the inverse of ``quaternion.exponential``. Finite, with finite derivatives, at a zero rotation."""
    flip = jnp.where(q[0] < 0, -1.0, 1.0)
    w, v = flip * q[0], flip * q[1:]
    sine_squared = jnp.sum(v * v)
    small = sine_squared < _SERIES_BELOW
    sine = jnp.sqrt(jnp.where(small, 1.0, sine_squared))
    # 2 atan2(sin, cos) / sin, the angle over the half-angle's sine; its series about zero where the sine is small,
    # there the cosine is near one, and elsewhere is given one so that neither branch divides by zero (a half turn).
    cosine = jnp.where(small, w, 1.0)
    series = (2.0 - 2.0 * sine_squared / (3.0 * cosine * cosine)) / cosine
    angle_over_sine = jnp.where(small, series, 2.0 * jnp.arctan2(sine, w) / sine)
    return angle_over_sine * v


def pose_error(current: Pose, target: Pose) -> Float[Array, "6"]:
    """``[t_target - t_current; rotation vector of R_target R_current^T]``, in the parent's axes: the twist (linear
    velocity of the frame's origin, angular velocity) that would carry ``current`` onto ``target`` in unit time, to
    first order."""
    turned = quaternion.multiply(target[3:], quaternion.conjugate(current[3:]))
    return jnp.concatenate([target[:3] - current[:3], rotation_vector(turned)])
