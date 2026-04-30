"""Recursive Newton-Euler Algorithm — pure JAX.

Computes inverse dynamics: given (q, q̇, q̈) and gravity, returns the joint
torques τ that produce that motion. Vectorized form (no explicit recursion),
matching ``frax.core.robot._rnea_from_spatial_data``.

Convention
----------
Spatial vectors use **[linear, angular]** ordering (see :mod:`spatial`).

Gravity handling: gravity is added to the base spatial acceleration as
``-g_world`` (i.e. an upward fictitious acceleration of the base). This is
the standard RNEA trick — see Featherstone §6.2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .crba import _spatial_axes_and_inertias
from .spatial import spatial_force_cross, spatial_motion_cross
from .urdf import KinematicTree, ancestor_mask


def _build_gravity_spatial_accel(gravity_world: Array | None) -> Array:
    """Convert a 3-vector world-frame gravity into a spatial fictitious accel.

    Returns a (6,) spatial acceleration whose linear part is ``-gravity``
    (so a body in free-fall sees zero net force on its inertial frame).
    """
    if gravity_world is None:
        return jnp.zeros(6)
    g = jnp.asarray(gravity_world)
    return jnp.concatenate([-g, jnp.zeros(3)])


def _rnea_from_spatial_data(
    tree: KinematicTree,
    spatial_axes: Array,
    spatial_inertias: Array,
    qd: Array | None,
    qdd: Array | None,
    gravity_spatial: Array,
    F_ext: Array | None,
) -> Array:
    """Internal RNEA driver. See :func:`rnea` for argument docs."""
    A = jnp.asarray(ancestor_mask(tree.parent_idxs)).astype(spatial_axes.dtype)

    # Forward pass: spatial velocity + acceleration of every link in world frame.
    spatial_accel = jnp.broadcast_to(gravity_spatial[None, :], spatial_axes.shape)
    if qd is not None:
        s_qd = spatial_axes * qd[:, None]
        spatial_vel = A @ s_qd
        # Coriolis-type term: per-link sum of v × s_qd contributions from ancestors
        spatial_accel = spatial_accel + A @ spatial_motion_cross(spatial_vel, s_qd)
    else:
        spatial_vel = jnp.zeros_like(spatial_axes)

    if qdd is not None:
        spatial_accel = spatial_accel + A @ (spatial_axes * qdd[:, None])

    # Newton-Euler: f = I·a + v ×_F (I·v)
    link_forces = jnp.einsum("ijk,ik->ij", spatial_inertias, spatial_accel)
    if qd is not None:
        Iv = jnp.einsum("ijk,ik->ij", spatial_inertias, spatial_vel)
        link_forces = link_forces + spatial_force_cross(spatial_vel, Iv)

    if F_ext is not None:
        link_forces = link_forces - F_ext

    # Backward pass: sum descendant link forces back through the tree, then
    # project onto each joint's motion subspace to get torques.
    net_forces = A.T @ link_forces
    return jnp.einsum("ij,ij->i", spatial_axes, net_forces)


def rnea(
    tree: KinematicTree,
    q: Array,
    qd: Array | None,
    qdd: Array | None,
    gravity_world: Array | None,
    F_ext: Array | None = None,
) -> Array:
    """Inverse dynamics via RNEA.

    Parameters
    ----------
    q : (N,) Array
    qd : (N,) Array or None
        Joint velocities. None ⇒ all zeros (skip Coriolis terms).
    qdd : (N,) Array or None
        Joint accelerations. None ⇒ all zeros (use to compute c(q,q̇) + g(q)).
    gravity_world : (3,) Array or None
        World-frame gravity vector (typical: ``[0, 0, -9.81]``). None ⇒
        gravity-free.
    F_ext : (N, 6) Array or None
        Per-link external wrenches in world frame ([linear; angular]).

    Returns
    -------
    tau : (N,) Array — joint torques.
    """
    spatial_axes, spatial_inertias = _spatial_axes_and_inertias(tree, q)
    g_spatial = _build_gravity_spatial_accel(gravity_world)
    return _rnea_from_spatial_data(
        tree, spatial_axes, spatial_inertias, qd, qdd, g_spatial, F_ext
    )


def gravity_vector(
    tree: KinematicTree, q: Array, gravity_world: Array | None
) -> Array:
    """Gravity torques ``g(q)`` (RNEA with q̇ = q̈ = 0)."""
    return rnea(tree, q, qd=None, qdd=None, gravity_world=gravity_world)


def nonlinear_bias(
    tree: KinematicTree, q: Array, qd: Array, gravity_world: Array | None
) -> Array:
    """Nonlinear bias ``c(q, q̇) + g(q)`` in a single RNEA pass (q̈ = 0)."""
    return rnea(tree, q, qd=qd, qdd=None, gravity_world=gravity_world)
