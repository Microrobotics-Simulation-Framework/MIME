"""Rigid transforms as MIME's poses, pure JAX: ``(x, y, z, qw, qx, qy, qz)``, a translation and then a unit quaternion
in wxyz order, mapping a point from the pose's own frame into its parent's as ``R(q) p + t``. Built on ``mime.core``'s
quaternion helpers, so every function is finite, with finite derivatives, at a zero rotation."""

from __future__ import annotations

from typing import Any

from jax import Array
from jaxtyping import Float

from mime.core import _quaternion as quaternion
from mime.core import switch
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


def shorter_way(q: Quaternion, *, owner: str) -> Quaternion:
    """``q`` or ``-q``, one rotation, whichever turns the shorter way round (``w >= 0``). At the half turn it changes
    side, a switch, the caller's ``owner``: the rotation vector of the result jumps there from the angle pi along the
    axis to pi against it."""
    flip = switch.where(q[0] < 0, -1.0, 1.0, owner=owner, resolution=lambda: switch.rounding(jnp.ones_like(q[0])),
                        indicator=lambda: -q[0])
    # The scalar and the vector part turned apart, as before this switch was named: turned whole, XLA fuses the
    # compiled step otherwise and its single-precision rounding moves (the controller's golden output shows it).
    return jnp.concatenate([(flip * q[0])[None], flip * q[1:]])


def shorter_rotation_vector(q: Quaternion, *, owner: str) -> Vector:
    """The rotation vector (angle times unit axis) of the unit quaternion ``q``, which turns the shorter way round
    (``w >= 0``). Finite, with finite derivatives, at a zero rotation: below a small angle it is taken from its series,
    which meets the closed form to rounding, a switch, the caller's ``owner``."""
    w, v = q[0], q[1:]
    sine_squared = jnp.sum(v * v)
    small, closed = sine_squared < _SERIES_BELOW, sine_squared >= _SERIES_BELOW
    sine = jnp.sqrt(switch.domain_guard(closed, sine_squared, 1.0))
    # 2 atan2(sin, cos) / sin, the angle over the half-angle's sine; its series about zero where the sine is small,
    # there the cosine is near one, and elsewhere is given one so that neither branch divides by zero (a half turn).
    cosine = switch.domain_guard(small, w, 1.0)
    series = (2.0 - 2.0 * sine_squared / (3.0 * cosine * cosine)) / cosine
    angle_over_sine = switch.where(small, series, 2.0 * jnp.arctan2(sine, w) / sine, owner=owner,
                                   resolution=lambda: switch.rounding(jnp.asarray(_SERIES_BELOW, sine_squared.dtype)),
                                   indicator=lambda: _SERIES_BELOW - sine_squared)
    return angle_over_sine * v


def rotation_vector(q: Quaternion, *, owner: str) -> Vector:
    """The rotation of the unit quaternion ``q`` as a rotation vector (angle times unit axis), the shorter way round:
    the inverse of ``quaternion.exponential``. Finite, with finite derivatives, at a zero rotation. Its two switches,
    the half turn (:func:`shorter_way`) and the series (:func:`shorter_rotation_vector`), are both the caller's
    ``owner``: a step that evaluates them calls the two with an owner each."""
    return shorter_rotation_vector(shorter_way(q, owner=owner), owner=owner)


def relative_turn(current: Pose, target: Pose) -> Quaternion:
    """The rotation ``R_target R_current^T``, which turns ``current``'s orientation onto ``target``'s, in the
    parent's axes."""
    return quaternion.multiply(target[3:], quaternion.conjugate(current[3:]))


def pose_error(current: Pose, target: Pose, *, owner: str) -> Float[Array, "6"]:
    """``[t_target - t_current; rotation vector of R_target R_current^T]``, in the parent's axes: the twist (linear
    velocity of the frame's origin, angular velocity) that would carry ``current`` onto ``target`` in unit time, to
    first order. Its rotation vector's switches are the caller's ``owner``."""
    return jnp.concatenate([target[:3] - current[:3], rotation_vector(relative_turn(current, target), owner=owner)])
