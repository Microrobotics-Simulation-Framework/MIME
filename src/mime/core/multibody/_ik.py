"""Inverse kinematics by damped least squares, pure JAX: a fixed number of iterations, so it compiles once, runs in a
predictable time and is differentiable with respect to its target. It reports how far it got rather than clamping:
a target out of reach, or one reached only outside the joint limits, is said so in the result."""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
from jax import Array
from jaxtyping import Bool, Float

from mime.core._node import jnp
from mime.core._quaternion import safe_norm
from mime.core.multibody._kinematics import Model, articulation_of, body_poses, computing_dtype, constants, \
    jacobian_at, placed_frame
from mime.core.multibody._se3 import Pose, pose_error


#: ``jax.lax.fori_loop``, whose hints a strict checker reads as partly unknown.
_fori_loop: Any = cast(Any, jax).lax.fori_loop


class InverseKinematics(NamedTuple):
    """What a solve reached. ``q`` is where the iterations ended, never clamped; ``position_error`` (m) and
    ``rotation_error`` (rad) are the frame's distance and angle from the target there; ``reached`` is whether both are
    within the tolerances asked for, and ``within_limits`` whether ``q`` is inside every joint's limits."""

    q: Float[Array, " dofs"]
    position_error: Float[Array, ""]
    rotation_error: Float[Array, ""]
    reached: Bool[Array, ""]
    within_limits: Bool[Array, ""]


def damped_least_squares(jacobian: Array, error: Array, damping: Any) -> Array:
    """The step ``J^T (J J^T + damping^2 1)^-1 e``: the least-squares step for ``e``, damped where ``J`` loses rank."""
    rows = jacobian.shape[0]
    damped = jacobian @ jacobian.T + damping * damping * jnp.eye(rows, dtype=jacobian.dtype)
    return jacobian.T @ jnp.linalg.solve(damped, error)


def inverse_kinematics(model: Model, frame: str, target: Pose, q_initial: Float[Array, " dofs"], *,
                       position_tolerance: float, rotation_tolerance: float, owner: str, iterations: int = 50,
                       damping: float = 0.05) -> InverseKinematics:
    """Joint positions that put the frame named ``frame`` at the pose ``target`` (root frame), by ``iterations``
    damped-least-squares steps from ``q_initial`` on the full pose error. ``damping`` is in the Jacobian's units
    (metres for its linear rows). Converges from an initial guess near a solution; the result says whether it got
    there. The pose error's switches are the caller's ``owner``."""
    articulation = articulation_of(model, "inverse_kinematics")
    dtype = computing_dtype(q_initial)
    q_initial = jnp.asarray(q_initial, dtype=dtype)
    target = jnp.asarray(target, dtype=dtype)
    held = constants(articulation, dtype)
    k = articulation.frame_index(frame)
    body = articulation.frame_bodies[k]

    def error_and_jacobian(q: Array) -> tuple[Array, Array]:
        bodies = body_poses(articulation, q, held)
        reached = placed_frame(articulation, held, bodies, k)
        return pose_error(reached, target, owner=owner), jacobian_at(articulation, held, bodies, body, reached[:3])

    def step(_: Any, q: Array) -> Array:
        error, jacobian = error_and_jacobian(q)
        return (q + damped_least_squares(jacobian, error, damping)).astype(dtype)

    q = _fori_loop(0, iterations, step, q_initial)
    error, _ = error_and_jacobian(q)
    position_error, rotation_error = safe_norm(error[:3]), safe_norm(error[3:])
    lower, upper = held["lower"], held["upper"]
    return InverseKinematics(q, position_error, rotation_error,
                             (position_error <= position_tolerance) & (rotation_error <= rotation_tolerance),
                             jnp.all((q >= lower) & (q <= upper)))

