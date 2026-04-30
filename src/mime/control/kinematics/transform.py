"""SE(3) transforms for URDF kinematics — pure JAX, fully traceable.

Quaternion convention
---------------------
This module uses **WXYZ (scalar-first)** quaternions: ``q = [w, x, y, z]``.
This matches:

  * ``mime.core.quaternion`` (used by RigidBodyNode, FlexibleBodyNode, etc.)
  * ``frax.utils.rotation_utils`` (the original reference for these helpers)
  * MuJoCo's quaternion ordering

The Wave A.3 plan suggested ``[qx, qy, qz, qw]`` but explicitly instructed
to "verify by checking ``src/mime/nodes/robot/rigid_body.py`` first — match
what's there". MIME uses scalar-first, so we use scalar-first here too.
A future XYZW-flavored helper (if ever needed by ROS 2 / Pinocchio interop)
should be added separately, named ``*_xyzw``.

Pose convention
---------------
``pose7 = [x, y, z, qw, qx, qy, qz]`` — translation first, then WXYZ quaternion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


# ---------------------------------------------------------------------------
# Joint transforms (child frame --> parent frame)
# ---------------------------------------------------------------------------


def revolute_transform(q: ArrayLike, axis: ArrayLike) -> Array:
    """Homogeneous transform for a revolute joint about ``axis`` by angle ``q``.

    Implements Rodrigues' formula in homogeneous form. Matches
    ``frax.utils.transform_utils.revolute_transform``.

    Parameters
    ----------
    q : scalar
        Joint angle [rad].
    axis : (3,) array-like
        Joint rotation axis in the joint's local frame. Need not be a unit
        vector; this function normalizes it.

    Returns
    -------
    (4, 4) Array
        Transform from child frame to parent frame.
    """
    axis = jnp.asarray(axis)
    axis = axis / jnp.linalg.norm(axis)
    a1, a2, a3 = axis
    c = jnp.cos(q)
    s = jnp.sin(q)
    t = 1.0 - c
    return jnp.array(
        [
            [t * a1 * a1 + c, t * a1 * a2 - s * a3, t * a1 * a3 + s * a2, 0.0],
            [t * a1 * a2 + s * a3, t * a2 * a2 + c, t * a2 * a3 - s * a1, 0.0],
            [t * a1 * a3 - s * a2, t * a2 * a3 + s * a1, t * a3 * a3 + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def prismatic_transform(q: ArrayLike, axis: ArrayLike) -> Array:
    """Homogeneous transform for a prismatic joint along ``axis`` by ``q``.

    Parameters
    ----------
    q : scalar
        Joint position [m].
    axis : (3,) array-like
        Joint translation axis in the joint's local frame.

    Returns
    -------
    (4, 4) Array
    """
    axis = jnp.asarray(axis)
    translation = q * axis / jnp.linalg.norm(axis)
    return jnp.array(
        [
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def joint_transform(q: ArrayLike, axis: ArrayLike, joint_type: ArrayLike) -> Array:
    """Joint transform dispatched by joint type.

    Parameters
    ----------
    q : scalar
    axis : (3,) array-like
    joint_type : scalar
        0 → revolute, 1 → prismatic. Implemented as a smooth blend so the
        function is fully ``vmap``-able.
    """
    rev = revolute_transform(q, axis)
    pri = prismatic_transform(q, axis)
    return (1.0 - joint_type) * rev + joint_type * pri


def compose_transform(A: Array, B: Array) -> Array:
    """Compose two homogeneous transforms: ``A @ B``.

    Provided as an explicit helper so caller code can stay framework-agnostic
    when we eventually swap in lie-group log/exp accumulation.
    """
    return A @ B


# ---------------------------------------------------------------------------
# Pose ⇄ matrix conversions
# ---------------------------------------------------------------------------


def _quat_normalize(q: Array) -> Array:
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-30)


def _quat_to_rotation_matrix(q: Array) -> Array:
    """WXYZ quaternion → 3x3 rotation matrix. Matches ``mime.core.quaternion``."""
    q = _quat_normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_matrix_to_quat(R: Array) -> Array:
    """3x3 rotation matrix → WXYZ quaternion. Numerically stable case split."""
    R = jnp.asarray(R)
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    def case_tr_gt_0(_):
        s = jnp.sqrt(tr + 1.0) * 2
        return jnp.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )

    def case_x_max(_):
        s = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return jnp.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        )

    def case_y_max(_):
        s = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return jnp.array(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        )

    def case_z_max(_):
        s = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        return jnp.array(
            [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ]
        )

    cond1 = tr > 0
    cond2 = (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2])
    cond3 = R[1, 1] > R[2, 2]

    res = jnp.where(
        cond1,
        case_tr_gt_0(None),
        jnp.where(
            cond2,
            case_x_max(None),
            jnp.where(cond3, case_y_max(None), case_z_max(None)),
        ),
    )
    return _quat_normalize(res)


def pose_to_matrix(pose7: Array) -> Array:
    """Pose ``[x, y, z, qw, qx, qy, qz]`` → 4×4 homogeneous transform."""
    pose7 = jnp.asarray(pose7)
    t = pose7[:3]
    q = pose7[3:]
    R = _quat_to_rotation_matrix(q)
    M = jnp.eye(4)
    M = M.at[:3, :3].set(R)
    M = M.at[:3, 3].set(t)
    return M


def matrix_to_pose(M: Array) -> Array:
    """4×4 homogeneous transform → pose ``[x, y, z, qw, qx, qy, qz]``."""
    M = jnp.asarray(M)
    t = M[:3, 3]
    q = _rotation_matrix_to_quat(M[:3, :3])
    return jnp.concatenate([t, q])


def transform_points(transform: Array, points: Array) -> Array:
    """Apply a 4x4 transform to a (..., 3) point set."""
    points = jnp.asarray(points)
    homogeneous = jnp.concatenate([points, jnp.ones(points.shape[:-1] + (1,))], axis=-1)
    return (homogeneous @ transform.T)[..., :3]
