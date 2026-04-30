"""6D spatial algebra for rigid-body dynamics — pure JAX.

Convention
----------
**[linear, angular]** ordering for spatial motion vectors and forces. That is:

  * ``v_spatial = [v_lin (3,); w_ang (3,)]``        ∈ ℝ⁶
  * ``f_spatial = [f_lin (3,); tau_ang (3,)]``      ∈ ℝ⁶

This matches Pinocchio, MuJoCo, and ``frax.utils.spatial_utils``. (It differs
from Featherstone's textbook, which uses [angular, linear].)

Spatial inertia matrix layout (consistent with the ordering above)::

    I_O = [[ m·I_3,         -m·[c]_×                ],
           [ m·[c]_×,        I_c - m·[c]_×·[c]_×    ]]

where ``c`` is the COM in the inertia frame and ``I_c`` is the inertia tensor
about the COM (rotated into that frame).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _skew(v: Array) -> Array:
    """3-vector → 3x3 skew-symmetric cross-product matrix."""
    return jnp.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def spatial_motion_cross(velocity: Array, motion: Array) -> Array:
    """Spatial cross-product for motion vectors.

    For ``v = [v0; w]`` and ``m = [m0; m]`` (linear, angular)::

        v ×_M m = [w × m0 + v0 × m;  w × m]

    Featherstone eq. 2.33 adapted to [linear, angular] ordering.
    """
    v0 = velocity[..., :3]
    w = velocity[..., 3:]
    m0 = motion[..., :3]
    m = motion[..., 3:]
    return jnp.concatenate(
        [jnp.cross(w, m0) + jnp.cross(v0, m), jnp.cross(w, m)], axis=-1
    )


def spatial_force_cross(velocity: Array, force: Array) -> Array:
    """Spatial cross-product for force vectors.

    For ``v = [v0; w]`` and ``f = [f; tau]`` (linear, angular)::

        v ×_F f = [w × f;  w × tau + v0 × f]

    Featherstone eq. 2.34 adapted to [linear, angular] ordering.
    """
    v0 = velocity[..., :3]
    w = velocity[..., 3:]
    f = force[..., :3]
    tau = force[..., 3:]
    return jnp.concatenate(
        [jnp.cross(w, f), jnp.cross(w, tau) + jnp.cross(v0, f)], axis=-1
    )


def spatial_inertia_from_link(
    mass: Array, com: Array, inertia_tensor: Array
) -> Array:
    """Build a 6×6 spatial inertia from link properties.

    Parameters
    ----------
    mass : scalar
        Link mass.
    com : (3,) Array
        Center-of-mass position in the **frame in which** the spatial inertia
        is expressed. (E.g. if you want world-frame spatial inertia, pass
        the world-frame COM.)
    inertia_tensor : (3, 3) Array
        Inertia tensor about the COM, expressed in the **same frame** as
        ``com``. Caller is responsible for any frame rotation.

    Returns
    -------
    (6, 6) Array — spatial inertia ``I_O``.
    """
    cx = _skew(com)
    eye3 = jnp.eye(3)
    top = jnp.concatenate([mass * eye3, -mass * cx], axis=1)
    bottom = jnp.concatenate(
        [mass * cx, inertia_tensor - mass * (cx @ cx)], axis=1
    )
    return jnp.concatenate([top, bottom], axis=0)


def spatial_inertias_from_links(
    masses: Array, link_transforms: Array, local_inertias: Array
) -> Array:
    """Vectorized spatial inertias for a set of links, expressed in world frame.

    Parameters
    ----------
    masses : (N,) Array
    link_transforms : (N, 4, 4) Array
        Link inertial-frame → world transforms.
    local_inertias : (N, 3, 3) Array
        Inertia tensors about each link's COM, in that link's local frame.

    Returns
    -------
    (N, 6, 6) Array
    """
    R = link_transforms[:, :3, :3]
    com = link_transforms[:, :3, 3]
    # Rotate local inertia tensors into world frame: I_world = R · I_local · Rᵀ
    Ic = jnp.einsum("nij,njk,nlk->nil", R, local_inertias, R)
    # Build skew matrices for COMs
    z = jnp.zeros_like(com[:, 0])
    cx = jnp.stack(
        [
            jnp.stack([z, -com[:, 2], com[:, 1]], axis=1),
            jnp.stack([com[:, 2], z, -com[:, 0]], axis=1),
            jnp.stack([-com[:, 1], com[:, 0], z], axis=1),
        ],
        axis=1,
    )
    m_col = masses[:, None, None]
    eye3 = jnp.eye(3)[None, :, :]
    top = jnp.concatenate([m_col * eye3, -m_col * cx], axis=2)
    bottom = jnp.concatenate([m_col * cx, Ic - m_col * (cx @ cx)], axis=2)
    return jnp.concatenate([top, bottom], axis=1)


def spatial_axis_for_joint(
    joint_axis_local: Array,
    joint_world_transform: Array,
    is_revolute: Array,
) -> Array:
    """Spatial joint axis (motion subspace for a 1-DOF joint), in world frame.

    For a revolute joint::    s = [p × ω̂;  ω̂]
    For a prismatic joint::   s = [v̂;       0]

    where ``ω̂`` / ``v̂`` is the joint axis rotated into world frame and ``p``
    is the world-frame position of the joint origin.

    Parameters
    ----------
    joint_axis_local : (3,) Array
        Joint axis in the joint's local frame.
    joint_world_transform : (4, 4) Array
        Joint frame → world transform.
    is_revolute : scalar (0/1 or bool)
        1 (or True) if revolute, 0 (or False) if prismatic.

    Returns
    -------
    (6,) Array — spatial axis ``s`` in world frame.
    """
    R = joint_world_transform[:3, :3]
    p = joint_world_transform[:3, 3]
    axis_world = R @ joint_axis_local
    rev = jnp.concatenate([jnp.cross(p, axis_world), axis_world])
    pri = jnp.concatenate([axis_world, jnp.zeros(3)])
    return jnp.where(is_revolute, rev, pri)


def spatial_axes_for_joints(
    joint_axes_local: Array,
    joint_world_transforms: Array,
    revolute_mask: Array,
) -> Array:
    """Vectorized spatial axes for a set of joints, expressed in world frame.

    See :func:`spatial_axis_for_joint`.

    Parameters
    ----------
    joint_axes_local : (N, 3) Array
    joint_world_transforms : (N, 4, 4) Array
    revolute_mask : (N,) Array — 1 for revolute, 0 for prismatic.

    Returns
    -------
    (N, 6) Array
    """
    R = joint_world_transforms[:, :3, :3]
    p = joint_world_transforms[:, :3, 3]
    axes_world = jnp.einsum("nij,nj->ni", R, joint_axes_local)
    rev = jnp.concatenate([jnp.cross(p, axes_world), axes_world], axis=1)
    pri = jnp.concatenate([axes_world, jnp.zeros_like(axes_world)], axis=1)
    return jnp.where(revolute_mask[:, None], rev, pri)
