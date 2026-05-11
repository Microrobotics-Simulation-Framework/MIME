"""Inverse kinematics — damped least-squares Newton iteration.

The Jacobian-pseudoinverse approach (DLS) is used over a closed-form
solver because it is generic across URDFs and degenerate configurations,
and JAX-friendly: every operation here is jit/grad/vmap traceable so the
solver can run inside a graph step.

Conventions
-----------

- Poses are 4×4 homogeneous matrices.
- Geometric Jacobian J ∈ ℝ^(6×N) follows the [linear; angular] block
  ordering from :mod:`spatial` and :func:`fk.frame_jacobian`.
- ``pose_error_6d`` produces a 6-vector with the same ordering — so the
  task-space update is just ``q_dot = damped_least_squares(J, e, lam)``.

Single-step usage
-----------------
::

    e = pose_error_6d(T_current, T_target)
    q_dot = damped_least_squares(J, e, lam=0.05)
    q_new = q + q_dot

For a from-seed solve where the initial guess is far from the target,
use :func:`solve_ik_iterative` which Newton-iterates this update.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .fk import frame_jacobian, joint_to_world_transforms
from .transform import pose_to_matrix
from .urdf import KinematicTree


# ── Pose / Jacobian helpers ──────────────────────────────────────────


def _so3_log(R: Array) -> Array:
    """Axis-angle log of a rotation matrix.

    Returns a 3-vector ``θ·k̂`` where ``k̂`` is the unit rotation axis and
    ``θ ∈ [0, π]`` is the rotation angle. Stable near θ=0 and θ=π.
    """
    # cos θ from trace, clamped to [-1, 1] for stability.
    cos_theta = jnp.clip((jnp.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    # Skew-symmetric part recovers the axis · sin θ.
    skew = 0.5 * (R - R.T)
    axis_sin_theta = jnp.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    sin_theta = jnp.sin(theta)
    # Near θ=0: axis_sin_theta ≈ axis · θ, divide by sinc(θ/2)/2 expansion
    # is just ≈ axis_sin_theta directly (sin θ ≈ θ → axis · θ ≈ axis · sin θ).
    # Use a small-angle bypass to avoid the 0/0.
    safe_factor = jnp.where(
        sin_theta > 1e-6,
        theta / jnp.maximum(sin_theta, 1e-30),
        1.0,
    )
    return axis_sin_theta * safe_factor


def pose_error_6d(T_current: Array, T_target: Array) -> Array:
    """6-vector pose error in [linear; angular] order.

    Parameters
    ----------
    T_current, T_target : (4, 4) Array
        Homogeneous transforms.

    Returns
    -------
    e : (6,) Array
        ``[t_target - t_current; log(R_target @ R_current.T)]`` —
        i.e. the world-frame velocity that, applied for unit time,
        would carry ``T_current`` onto ``T_target``.
    """
    t_err = T_target[:3, 3] - T_current[:3, 3]
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    omega = _so3_log(R_err)
    return jnp.concatenate([t_err, omega])


def _skew(v: Array) -> Array:
    return jnp.array([
        [   0.0, -v[2],  v[1]],
        [ v[2],    0.0, -v[0]],
        [-v[1],  v[0],    0.0],
    ])


def ee_jacobian(
    tree: KinematicTree,
    q: Array,
    ee_link_idx: int,
    ee_offset_pose7: Array,
) -> Array:
    """Geometric Jacobian at the EE point (with tool offset applied).

    :func:`fk.frame_jacobian` returns the Jacobian at the link's
    inertial frame (COM-based). For an EE that's offset by a static
    rigid pose ``ee_offset_pose7`` (in the link's joint frame), the
    Jacobian is shifted: ``J_ee_lin = J_link_lin - skew(p_ee - p_link) @ J_link_ang``.

    Parameters
    ----------
    ee_link_idx : int
        Static index of the link the EE is rigidly attached to.
    ee_offset_pose7 : (7,) Array
        Tool-offset pose in link's joint frame, [x, y, z, qw, qx, qy, qz].
    """
    J_link = frame_jacobian(tree, q, ee_link_idx)
    # Compute the EE world position to determine the lever arm.
    joint_world = joint_to_world_transforms(tree, q)
    T_offset = pose_to_matrix(ee_offset_pose7)
    T_ee = joint_world[ee_link_idx] @ T_offset
    p_ee = T_ee[:3, 3]
    # The link's *inertial* origin is the COM, which is what
    # frame_jacobian uses. Recompute that point so the lever arm is
    # measured between the two consistently.
    com_local = jnp.asarray(tree.link_com_local)
    p_link_inertial = (
        joint_world[ee_link_idx, :3, :3] @ com_local[ee_link_idx]
        + joint_world[ee_link_idx, :3, 3]
    )
    lever = p_ee - p_link_inertial
    # Shift the linear block; angular block unchanged.
    J_ee_lin = J_link[:3, :] - _skew(lever) @ J_link[3:, :]
    J_ee_ang = J_link[3:, :]
    return jnp.concatenate([J_ee_lin, J_ee_ang], axis=0)


# ── Solvers ──────────────────────────────────────────────────────────


def damped_least_squares(J: Array, e: Array, lam: float = 0.05) -> Array:
    """Damped pseudo-inverse step: ``q_dot = J^T (J J^T + λ²I)⁻¹ e``.

    Equivalent to one Newton step on ``½‖J q_dot − e‖² + ½λ²‖q_dot‖²``.
    Stable near singularities (J rank-deficient) thanks to the λ²I
    regulariser; collapses to ``J^+ e`` as ``λ → 0``.
    """
    m = J.shape[0]  # task dimension (6 for full pose)
    A = J @ J.T + (lam * lam) * jnp.eye(m, dtype=J.dtype)
    return J.T @ jnp.linalg.solve(A, e)


def solve_ik_iterative(
    tree: KinematicTree,
    q_init: Array,
    T_target: Array,
    ee_link_idx: int,
    ee_offset_pose7: Array,
    n_iters: int = 20,
    lam: float = 0.05,
    step_size: float = 1.0,
) -> Array:
    """Iterative DLS Newton solve for joint angles that achieve T_target.

    A bounded ``n_iters`` Newton iteration. Each step computes the
    pose error at the current ``q``, the EE Jacobian, and a DLS step.
    No convergence test inside the loop — the iteration count is fixed
    so the function is JIT-friendly and has predictable runtime.

    Parameters
    ----------
    q_init : (N,) Array
        Initial joint configuration.
    T_target : (4, 4) Array
        Desired EE world pose.
    ee_link_idx : int
        Index of the link the EE is rigidly attached to.
    ee_offset_pose7 : (7,) Array
        Tool offset pose in the link's joint frame.
    n_iters : int
        Number of Newton iterations (default 20).
    lam : float
        DLS damping factor.
    step_size : float
        Update gain. Default 1.0 (full Newton step).

    Returns
    -------
    q_solution : (N,) Array
        Joint configuration approximating T_target.
    """
    # Cast the carry back to the input dtype every iteration. JAX's
    # fori_loop requires the body's output type to match the input
    # type exactly, and a few operations downstream of float32 inputs
    # (pose_to_matrix constants, jnp.linalg.solve under x64) silently
    # upcast to float64 — without this cast the loop trace fails on
    # the second iteration with a carry-type mismatch.
    in_dtype = q_init.dtype

    def body(_, q):
        joint_world = joint_to_world_transforms(tree, q)
        T_offset = pose_to_matrix(ee_offset_pose7)
        T_ee = joint_world[ee_link_idx] @ T_offset
        e = pose_error_6d(T_ee, T_target)
        J = ee_jacobian(tree, q, ee_link_idx, ee_offset_pose7)
        dq = step_size * damped_least_squares(J, e, lam)
        return (q + dq).astype(in_dtype)

    return jax.lax.fori_loop(0, n_iters, body, q_init)
