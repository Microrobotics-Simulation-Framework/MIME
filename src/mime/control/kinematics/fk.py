"""Forward kinematics — pure JAX, traceable under jit/grad/vmap.

Two FK strategies, mirroring frax:

  * **Pure serial chain** (``parent_idxs[i] == i - 1`` for all i, with
    ``parent_idxs[0] == -1``): use :func:`jax.lax.associative_scan` with
    ``jnp.matmul`` for ``O(log N)`` parallel depth.
  * **Branching tree**: unrolled loop in Python (which JAX traces into a
    sequence of XLA ops). ``O(N)`` depth but works for arbitrary trees.

The choice is made statically at trace time using ``tree.parent_idxs``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .transform import joint_transform
from .urdf import KinematicTree


def _is_pure_serial_chain(parent_idxs: np.ndarray) -> bool:
    n = len(parent_idxs)
    if n == 0:
        return True
    if parent_idxs[0] != -1:
        return False
    for i in range(1, n):
        if parent_idxs[i] != i - 1:
            return False
    return True


def _local_joint_transforms(tree: KinematicTree, q: Array) -> Array:
    """Per-joint transform in its parent's frame: ``parent_xform @ joint_q``."""
    joint_axes = jnp.asarray(tree.joint_axes_local)
    joint_types = jnp.asarray(tree.joint_types).astype(jnp.float32)
    parent_xforms = jnp.asarray(tree.joint_parent_frame_xforms)
    # vmap joint_transform over (q, axis, type)
    qts = jax.vmap(joint_transform)(q, joint_axes, joint_types)
    return parent_xforms @ qts


def joint_to_world_transforms(tree: KinematicTree, q: Array) -> Array:
    """Per-joint world transforms (joint frame → world).

    Returns
    -------
    (N, 4, 4) Array
    """
    local_tfs = _local_joint_transforms(tree, q)

    if _is_pure_serial_chain(tree.parent_idxs):
        # Pure serial chain: cumulative matmul.
        return jax.lax.associative_scan(jnp.matmul, local_tfs, axis=0)

    # Branching tree: unrolled loop. Topological sort guarantees parents
    # are processed before children.
    parent_idxs = tree.parent_idxs
    world_tfs = jnp.zeros_like(local_tfs)
    for i in range(tree.num_joints):
        p = int(parent_idxs[i])
        if p == -1:
            world_tfs = world_tfs.at[i].set(local_tfs[i])
        else:
            world_tfs = world_tfs.at[i].set(world_tfs[p] @ local_tfs[i])
    return world_tfs


def link_to_world_transforms(tree: KinematicTree, q: Array) -> Array:
    """Per-link world transforms (link inertial frame → world).

    The "link frame" here is the **inertial frame** (origin at the COM,
    rotation aligned with the joint frame — the inertial-rpy was already
    folded into the inertia tensor at parse time, see :func:`_parse_inertial`).
    """
    joint_world = joint_to_world_transforms(tree, q)
    com_local = jnp.asarray(tree.link_com_local)
    # Build per-link "joint → COM" transform: identity rotation + COM offset
    # (we already absorbed any inertial-rpy at parse time).
    n = tree.num_joints
    com_offset = jnp.eye(4)[None].repeat(n, axis=0).at[:, :3, 3].set(com_local)
    return joint_world @ com_offset


def link_world_poses(
    tree: KinematicTree, q: Array, base_pose_world: Array | None = None
) -> Array:
    """Convenience: per-link pose ``[x, y, z, qw, qx, qy, qz]`` in world frame.

    Parameters
    ----------
    tree : KinematicTree
    q : (N,) Array
    base_pose_world : (7,) Array, optional
        Pose of the kinematic-tree root in world. If None, root is at
        identity.

    Returns
    -------
    (N, 7) Array
    """
    link_xforms = link_to_world_transforms(tree, q)
    if base_pose_world is not None:
        from .transform import pose_to_matrix
        T_base = pose_to_matrix(base_pose_world)
        link_xforms = T_base[None] @ link_xforms

    # Extract translation + quaternion per link
    n = tree.num_joints
    t = link_xforms[:, :3, 3]
    R = link_xforms[:, :3, :3]
    # Vectorized rotation matrix → quaternion (WXYZ), trace-positive branch
    # is the dominant case for typical configurations; we use the safe
    # case-split via a per-link function.
    from .transform import _rotation_matrix_to_quat
    quats = jax.vmap(_rotation_matrix_to_quat)(R)
    return jnp.concatenate([t, quats], axis=1)


def frame_jacobian(
    tree: KinematicTree, q: Array, link_idx: int
) -> Array:
    """Geometric Jacobian for the inertial frame of link ``link_idx``.

    Returns a ``(6, N)`` Jacobian with the **[linear; angular]** ordering
    consistent with :mod:`spatial`. For each ancestor joint ``j`` of
    ``link_idx`` (and ``link_idx`` itself):

      * Revolute:   J[:3, j] = ω̂_j × (p_link − p_j),    J[3:, j] = ω̂_j
      * Prismatic:  J[:3, j] = v̂_j,                     J[3:, j] = 0

    Joints that are *not* ancestors of ``link_idx`` contribute zero columns.

    Parameters
    ----------
    link_idx : int
        Static (Python) index — ancestor chain is computed from
        ``tree.parent_idxs`` at trace time.
    """
    if not (0 <= link_idx < tree.num_joints):
        raise IndexError(
            f"link_idx {link_idx} out of range [0, {tree.num_joints})"
        )

    joint_world = joint_to_world_transforms(tree, q)
    # Build the world-frame link-inertial position
    com_local = jnp.asarray(tree.link_com_local)
    p_link = (
        joint_world[link_idx, :3, :3] @ com_local[link_idx]
        + joint_world[link_idx, :3, 3]
    )

    # Determine ancestor chain at static (Python) trace time.
    chain: list[int] = []
    j = link_idx
    while j != -1:
        chain.append(j)
        j = int(tree.parent_idxs[j])
    chain = list(reversed(chain))  # root → link

    joint_axes_local = jnp.asarray(tree.joint_axes_local)
    revolute = jnp.asarray(tree.revolute_mask)

    cols_lin = []
    cols_ang = []
    for j in chain:
        Rj = joint_world[j, :3, :3]
        pj = joint_world[j, :3, 3]
        axis_world = Rj @ joint_axes_local[j]
        is_rev = revolute[j]
        lever = jnp.cross(axis_world, p_link - pj)
        col_lin = jnp.where(is_rev, lever, axis_world)
        col_ang = jnp.where(is_rev, axis_world, jnp.zeros(3))
        cols_lin.append(col_lin)
        cols_ang.append(col_ang)

    # Place the contributing columns at the correct joint indices in a
    # full-width Jacobian.
    J = jnp.zeros((6, tree.num_joints))
    chain_idx = jnp.asarray(chain, dtype=jnp.int32)
    cols_lin_arr = jnp.stack(cols_lin, axis=0)  # (k, 3)
    cols_ang_arr = jnp.stack(cols_ang, axis=0)  # (k, 3)
    J = J.at[:3, chain_idx].set(cols_lin_arr.T)
    J = J.at[3:, chain_idx].set(cols_ang_arr.T)
    return J
