"""Rigid-body dynamics of an open tree, pure JAX: the joint-space mass matrix by the composite-rigid-body algorithm
(CRBA), inverse dynamics by the recursive Newton-Euler algorithm (RNEA), both after Featherstone, *Rigid Body Dynamics
Algorithms* (Springer, 2008; chapters 5 and 6), and forward dynamics ``q_ddot = M^-1 (tau - h)`` from the two.

Spatial vectors are in the root's frame, about its origin, ordered ``[linear; angular]``: a motion vector is
``[v_O; w]``, ``v_O`` the velocity of the body point at the origin, and a force vector ``[f; n_O]``, ``n_O`` the moment
about the origin. With every quantity in one frame the recursions need no frame changes, and each is written as sums
over a body's ancestors (the ancestry mask) rather than as a loop. Gravity enters as the root's acceleration, ``-g``.

Each joint's armature adds to its own diagonal entry of the mass matrix and to its own torque in inverse dynamics, as
MuJoCo applies it. Damping and friction are not applied here: they are joint forces a dynamic chain adds to ``tau``.

Loads on the bodies (``link_forces``, one spatial force per rigid body about the root's origin) enter the recursion as
external forces (Featherstone 2008, equation 5.9): each body's net force is what its motion needs less what acts on it
from outside. :func:`frame_forces` gathers wrenches given on the named frames onto the bodies they are fixed to."""

from __future__ import annotations

from typing import Any, cast

import jax.scipy.linalg
import numpy as np
from jax import Array
from jaxtyping import Float

from mime.core import _quaternion as quaternion
from mime.core import switch
from mime.core._node import jnp
from mime.core.multibody._articulation import ROOT, Articulation
from mime.core.multibody._kinematics import Model, articulation_of, body_poses, computing_dtype, constants, \
    joint_axes, placed_frame

type JointVector = Float[Array, " dofs"]

#: ``jax.scipy.linalg.solve_triangular``, whose hints a strict checker reads as partly unknown.
_solve_triangular: Any = cast(Any, jax.scipy.linalg).solve_triangular


def _skew(v: Array) -> Array:
    zero = jnp.zeros_like(v[0])
    return jnp.stack([jnp.stack([zero, -v[2], v[1]]), jnp.stack([v[2], zero, -v[0]]),
                      jnp.stack([-v[1], v[0], zero])])


def motion_cross(v: Array, m: Array) -> Array:
    """``v x m`` for spatial motion vectors, rows ``[linear; angular]``."""
    return jnp.concatenate([jnp.cross(v[..., 3:], m[..., :3]) + jnp.cross(v[..., :3], m[..., 3:]),
                            jnp.cross(v[..., 3:], m[..., 3:])], axis=-1)


def force_cross(v: Array, f: Array) -> Array:
    """``v x* f``, the motion vector ``v`` acting on the spatial force ``f``, rows ``[linear; angular]``."""
    return jnp.concatenate([jnp.cross(v[..., 3:], f[..., :3]),
                            jnp.cross(v[..., 3:], f[..., 3:]) + jnp.cross(v[..., :3], f[..., :3])], axis=-1)


def spatial_inertia(mass: Array, centre: Array, inertia: Array) -> Array:
    """The 6 x 6 spatial inertia about the origin of a body with ``mass``, centre of mass ``centre`` and inertia
    ``inertia`` about it, both in the origin's frame: ``[[m 1, -m [c]x], [m [c]x, I_c - m [c]x [c]x]]``."""
    cross = _skew(centre)
    upper = jnp.concatenate([mass * jnp.eye(3, dtype=centre.dtype), -mass * cross], axis=1)
    lower = jnp.concatenate([mass * cross, inertia - mass * (cross @ cross)], axis=1)
    return jnp.concatenate([upper, lower], axis=0)


def spatial_terms(articulation: Articulation, q: Array) -> tuple[Array, Array, Array]:
    """Each joint's motion subspace ``s_i`` (revolute ``[o x a; a]``, prismatic ``[a; 0]``), each rigid body's spatial
    inertia, and the ancestry mask, in the dtype of ``q``."""
    held = constants(articulation, q.dtype)
    bodies = body_poses(articulation, q, held)
    axes, origins = joint_axes(articulation, held, bodies)
    revolute = np.asarray(articulation.revolute, dtype=bool)[:, None]
    subspaces = switch.static_select(revolute, jnp.concatenate([jnp.cross(origins, axes), axes], axis=1),
                                     jnp.concatenate([axes, jnp.zeros_like(axes)], axis=1))
    inertias: list[Array] = []
    for i, body in enumerate(bodies):
        rotation = quaternion.rotation_matrix(body[3:])
        centre = body[:3] + rotation @ held["centres_of_mass"][i]
        inertias.append(spatial_inertia(held["masses"][i], centre, rotation @ held["inertias"][i] @ rotation.T))
    return subspaces, jnp.stack(inertias), jnp.asarray(articulation.ancestry(), dtype=q.dtype)


def _joint_vector(value: Any, dtype: Any, dofs: int) -> Array:
    return jnp.zeros(dofs, dtype=dtype) if value is None else jnp.asarray(value, dtype=dtype)


def mass_matrix(model: Model, q: JointVector) -> Float[Array, "dofs dofs"]:
    """The joint-space mass matrix ``M(q)``, by the composite-rigid-body algorithm (Featherstone 2008, chapter 6): each
    joint's composite inertia ``I^c_j`` is the sum of the spatial inertias of the bodies it moves, and
    ``M_ij = s_i . I^c_i s_j`` where joint ``j`` moves body ``i``, mirrored above the diagonal. Symmetric, and positive
    definite wherever every joint moves some mass or has armature."""
    articulation = articulation_of(model, "mass_matrix")
    q = jnp.asarray(q, dtype=computing_dtype(q))
    if articulation.dofs == 0:
        return jnp.zeros((0, 0), dtype=q.dtype)
    subspaces, inertias, moved_by = spatial_terms(articulation, q)
    composite = jnp.einsum("ij,ikl->jkl", moved_by, inertias)
    every_pair = jnp.einsum("ik,ikl,jl->ij", subspaces, composite, subspaces)
    lower = moved_by * every_pair
    armature = jnp.asarray(articulation.armatures, dtype=q.dtype)
    return lower + jnp.tril(lower, -1).T + jnp.diag(armature)


def frame_forces(model: Model, q: JointVector, wrenches: Float[Array, "frames 6"]) -> Float[Array, "dofs 6"]:
    """The spatial force on each rigid body, ``[f; n_O]`` in the root's frame about its origin, of ``wrenches``: one
    row per named frame, in the tree's ``frame_names`` order, ``[force; torque about the frame's origin]`` in the
    root's frame, acting on the body the frame is fixed to. A frame fixed to the root loads nothing that moves."""
    articulation = articulation_of(model, "frame_forces")
    q = jnp.asarray(q, dtype=computing_dtype(q))
    held = constants(articulation, q.dtype)
    bodies = body_poses(articulation, q, held)
    frames = len(articulation.frame_names)
    loads = jnp.asarray(wrenches, dtype=q.dtype)
    if articulation.dofs == 0 or frames == 0:
        return jnp.zeros((articulation.dofs, 6), dtype=q.dtype)
    origins = jnp.stack([placed_frame(articulation, held, bodies, k)[:3] for k in range(frames)])
    about_origin = jnp.concatenate([loads[:, :3], loads[:, 3:] + jnp.cross(origins, loads[:, :3])], axis=1)
    onto = np.zeros((articulation.dofs, frames))
    for k, body in enumerate(articulation.frame_bodies):
        if body != ROOT:
            onto[body, k] = 1.0
    return jnp.asarray(onto, dtype=q.dtype) @ about_origin


def inverse_dynamics(model: Model, q: JointVector, qd: JointVector | None, qdd: JointVector | None,
                     gravity: Float[Array, "3"], link_forces: Float[Array, "dofs 6"] | None = None) -> JointVector:
    """The joint forces ``tau`` that give the accelerations ``qdd`` at ``(q, qd)`` under ``gravity`` (the root frame's
    gravitational acceleration) and the loads ``link_forces`` on the bodies (one spatial force per rigid body, about
    the root's origin, as :func:`frame_forces` gives them), by the recursive Newton-Euler algorithm (Featherstone 2008,
    chapter 5): each body's velocity ``v_i`` and acceleration ``a_i`` summed over the joints that move it, from the
    root's ``-g``; its net force ``f_i = I_i a_i + v_i x* I_i v_i - f_i^ext``; each joint's force the projection onto
    ``s_j`` of the forces of the bodies it moves. ``None`` stands for zero velocities, accelerations or loads."""
    articulation = articulation_of(model, "inverse_dynamics")
    q = jnp.asarray(q, dtype=computing_dtype(q))
    dofs = articulation.dofs
    if dofs == 0:
        return jnp.zeros(0, dtype=q.dtype)
    qd, qdd = _joint_vector(qd, q.dtype, dofs), _joint_vector(qdd, q.dtype, dofs)
    gravity = jnp.asarray(gravity, dtype=q.dtype)
    subspaces, inertias, moved_by = spatial_terms(articulation, q)
    joint_velocities = subspaces * qd[:, None]
    velocities = moved_by @ joint_velocities
    root = jnp.concatenate([-gravity, jnp.zeros_like(gravity)])
    accelerations = root + moved_by @ (subspaces * qdd[:, None] + motion_cross(velocities, joint_velocities))
    momenta = jnp.einsum("ikl,il->ik", inertias, velocities)
    forces = jnp.einsum("ikl,il->ik", inertias, accelerations) + force_cross(velocities, momenta)
    if link_forces is not None:
        forces = forces - jnp.asarray(link_forces, dtype=q.dtype)
    transmitted = moved_by.T @ forces
    armature = jnp.asarray(articulation.armatures, dtype=q.dtype)
    return jnp.sum(subspaces * transmitted, axis=1) + armature * qdd


def bias_forces(model: Model, q: JointVector, qd: JointVector, gravity: Float[Array, "3"],
                link_forces: Float[Array, "dofs 6"] | None = None) -> JointVector:
    """``h(q, qd) = C(q, qd) qd + g(q)``: the velocity-product and gravity forces, less the joint forces of the loads
    ``link_forces`` on the bodies: inverse dynamics at zero acceleration."""
    articulation = articulation_of(model, "bias_forces")
    return inverse_dynamics(articulation, q, qd, None, gravity, link_forces)


def gravity_forces(model: Model, q: JointVector, gravity: Float[Array, "3"]) -> JointVector:
    """``g(q)``: the joint forces that hold the tree still under ``gravity``."""
    articulation = articulation_of(model, "gravity_forces")
    return inverse_dynamics(articulation, q, None, None, gravity)


def forward_dynamics(model: Model, q: JointVector, qd: JointVector, tau: JointVector,
                     gravity: Float[Array, "3"], link_forces: Float[Array, "dofs 6"] | None = None) -> JointVector:
    """The joint accelerations ``q_ddot = M(q)^-1 (tau - h(q, qd))`` under the joint forces ``tau`` and the loads
    ``link_forces`` on the bodies (as :func:`inverse_dynamics` takes them), solved by Cholesky factorisation of the
    mass matrix."""
    articulation = articulation_of(model, "forward_dynamics")
    q = jnp.asarray(q, dtype=computing_dtype(q))
    if articulation.dofs == 0:
        return jnp.zeros(0, dtype=q.dtype)
    rhs = jnp.asarray(tau, dtype=q.dtype) - bias_forces(articulation, q, qd, gravity, link_forces)
    factor = jnp.linalg.cholesky(mass_matrix(articulation, q))
    return cholesky_solve(factor, rhs)


def cholesky_solve(lower: Array, rhs: Array) -> Array:
    """``x`` with ``L L^T x = rhs``, by two triangular solves."""
    return _solve_triangular(lower.T, _solve_triangular(lower, rhs, lower=True), lower=False)
