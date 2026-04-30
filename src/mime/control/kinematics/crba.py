"""Composite Rigid Body Algorithm — pure JAX.

Computes the joint-space mass matrix ``M(q)`` such that ``M·q̈ + c(q,q̇) +
g(q) = τ + Jᵀ·F_ext``.

Implementation
--------------
Vectorized form (Featherstone Ch. 6, adapted to [linear, angular] ordering)::

    1. Compute all joint world transforms (FK).
    2. Build per-link spatial inertias I_world[i] (6×6).
    3. Composite inertia at joint i = sum over descendants j of I_world[j]
       — equivalently: ``I_C = ancestor_maskᵀ · I_world``.
    4. ``M[i,j] = sᵢᵀ · I_C[max(i,j)] · sⱼ`` for ancestor pairs (i is ancestor
       of j or vice-versa); zero otherwise.

Matches ``frax.core.robot._crba_from_spatial_data``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .fk import joint_to_world_transforms, link_to_world_transforms
from .spatial import spatial_axes_for_joints, spatial_inertias_from_links
from .urdf import KinematicTree, ancestor_mask


def _spatial_axes_and_inertias(tree: KinematicTree, q: Array):
    """Compute spatial joint axes and link spatial inertias from FK.

    Returns
    -------
    spatial_axes : (N, 6)
    spatial_inertias : (N, 6, 6)
    """
    joint_world = joint_to_world_transforms(tree, q)
    link_world = link_to_world_transforms(tree, q)
    spatial_axes = spatial_axes_for_joints(
        jnp.asarray(tree.joint_axes_local),
        joint_world,
        jnp.asarray(tree.revolute_mask),
    )
    spatial_inertias = spatial_inertias_from_links(
        jnp.asarray(tree.link_masses),
        link_world,
        jnp.asarray(tree.link_inertia_local),
    )
    return spatial_axes, spatial_inertias


def mass_matrix(tree: KinematicTree, q: Array) -> Array:
    """Joint-space mass matrix ``M(q)``, shape ``(N, N)``.

    The result is symmetric positive definite (for a physically valid URDF
    with all link masses ≥ 0 and at least one mass > 0 below each joint).
    """
    spatial_axes, spatial_inertias = _spatial_axes_and_inertias(tree, q)

    A = jnp.asarray(ancestor_mask(tree.parent_idxs)).astype(spatial_axes.dtype)

    # Composite inertia at each joint = sum over descendants of link spatial
    # inertias (Aᵀ accumulates from descendants up to ancestors).
    composite_inertias = jnp.einsum("ij,jkl->ikl", A.T, spatial_inertias)

    # Pairwise quadratic form using each composite inertia
    M_all = jnp.einsum("ij,ijk,lk->il", spatial_axes, composite_inertias, spatial_axes)
    # Mask: M[i,j] is nonzero only when i is on the ancestor chain of j (or
    # vice-versa). We keep the lower triangular wrt the ancestor relation
    # and reflect.
    M_lower = A * M_all
    return M_lower + jnp.tril(M_lower, k=-1).T
