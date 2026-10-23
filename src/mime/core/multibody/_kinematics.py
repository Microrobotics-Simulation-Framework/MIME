"""Forward kinematics and frame Jacobians, pure JAX: traceable under ``jit``, ``grad`` and ``vmap``, finite at zero
rotation, and in the dtype of the joint positions given. Everything is in the root's frame."""

from __future__ import annotations

from typing import Any

from jax import Array
from jaxtyping import Float

from mime.core import _quaternion as quaternion
from mime.core._node import jnp
from mime.core.multibody._articulation import ROOT, Articulation
from mime.core.multibody._se3 import Pose, compose, pose, turn
from mime.core.multibody._tree import MultibodyTree

type Model = MultibodyTree | Articulation


def articulation_of(model: Model, algorithm: str) -> Articulation:
    """The articulation the algorithm reads, refused if the tree holds a loop closure."""
    articulation = model.articulation if isinstance(model, MultibodyTree) else model
    articulation.refuse_closures(algorithm)
    return articulation


def computing_dtype(q: Any) -> Any:
    """The floating dtype an algorithm computes in: the joint positions', float32 at least."""
    return jnp.promote_types(jnp.asarray(q).dtype, jnp.float32)


def constants(articulation: Articulation, dtype: Any) -> dict[str, Array]:
    """The articulation's arrays in ``dtype``."""
    names = ("axes", "origin_rotations", "origin_translations", "masses", "centres_of_mass", "inertias", "armatures",
             "frame_rotations", "frame_translations", "lower", "upper")
    return {name: jnp.asarray(getattr(articulation, name), dtype=dtype) for name in names}


def body_poses(articulation: Articulation, q: Any, held: dict[str, Array]) -> list[Pose]:
    """Each rigid body's pose in the root's frame, in index order. A revolute joint turns its child's frame about the
    axis through its origin; a prismatic joint slides it along the axis."""
    poses: list[Pose] = []
    for i in range(articulation.dofs):
        axis = held["axes"][i]
        if articulation.revolute[i]:
            motion = pose(jnp.zeros_like(axis), turn(axis, q[i]))
        else:
            motion = pose(q[i] * axis, jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=axis.dtype))
        local = compose(pose(held["origin_translations"][i], held["origin_rotations"][i]), motion)
        parent = articulation.parents[i]
        poses.append(local if parent == ROOT else compose(poses[parent], local))
    return poses


def placed_frame(articulation: Articulation, held: dict[str, Array], bodies: list[Pose], k: int) -> Pose:
    """The pose of frame ``k`` in the root's frame, from the poses of the rigid bodies."""
    offset = pose(held["frame_translations"][k], held["frame_rotations"][k])
    body = articulation.frame_bodies[k]
    return offset if body == ROOT else compose(bodies[body], offset)


def forward_kinematics(model: Model, q: Float[Array, " dofs"]) -> Float[Array, "frames 7"]:
    """The pose of every named frame in the root's frame, in the tree's ``frame_names`` order (its bodies' frames, then
    its extra frames), as ``(x, y, z, qw, qx, qy, qz)``."""
    articulation = articulation_of(model, "forward_kinematics")
    q = jnp.asarray(q, dtype=computing_dtype(q))
    held = constants(articulation, q.dtype)
    bodies = body_poses(articulation, q, held)
    return jnp.stack([placed_frame(articulation, held, bodies, k) for k in range(len(articulation.frame_names))])


def frame_pose(model: Model, q: Float[Array, " dofs"], frame: str) -> Pose:
    """The pose of the frame named ``frame`` in the root's frame."""
    articulation = articulation_of(model, "frame_pose")
    q = jnp.asarray(q, dtype=computing_dtype(q))
    held = constants(articulation, q.dtype)
    return placed_frame(articulation, held, body_poses(articulation, q, held), articulation.frame_index(frame))


def joint_axes(articulation: Articulation, held: dict[str, Array], bodies: list[Pose]) -> tuple[Array, Array]:
    """Each joint's axis and origin in the root's frame: the axis turned by its child's rotation (a revolute joint's
    turn leaves its own axis alone, and a prismatic one turns nothing), the origin at its child's frame."""
    axes = jnp.stack([quaternion.rotate(b[3:], held["axes"][i]) for i, b in enumerate(bodies)])
    origins = jnp.stack([b[:3] for b in bodies])
    return axes, origins


def jacobian_at(articulation: Articulation, held: dict[str, Array], bodies: list[Pose], body: int,
                point: Array) -> Float[Array, "6 dofs"]:
    """The geometric Jacobian of a point fixed to rigid body ``body``: rows ``[linear velocity of the point; angular
    velocity]``, in the root's axes. A revolute joint's column is ``[a x (point - o); a]``, a prismatic one's
    ``[a; 0]``, and a joint that does not move the body gives zeros."""
    dofs = articulation.dofs
    if dofs == 0:
        return jnp.zeros((6, 0), dtype=point.dtype)
    axes, origins = joint_axes(articulation, held, bodies)
    moved = set(articulation.chain(body)) if body != ROOT else set[int]()
    columns: list[Array] = []
    for j in range(dofs):
        if j not in moved:
            columns.append(jnp.zeros(6, dtype=point.dtype))
        elif articulation.revolute[j]:
            columns.append(jnp.concatenate([jnp.cross(axes[j], point - origins[j]), axes[j]]))
        else:
            columns.append(jnp.concatenate([axes[j], jnp.zeros_like(axes[j])]))
    return jnp.stack(columns, axis=1)


def frame_jacobian(model: Model, q: Float[Array, " dofs"], frame: str) -> Float[Array, "6 dofs"]:
    """The geometric Jacobian of the frame named ``frame``: ``J q_dot`` is ``[linear velocity of the frame's origin;
    angular velocity of the frame]``, in the root's axes."""
    articulation = articulation_of(model, "frame_jacobian")
    q = jnp.asarray(q, dtype=computing_dtype(q))
    held = constants(articulation, q.dtype)
    bodies = body_poses(articulation, q, held)
    k = articulation.frame_index(frame)
    point = placed_frame(articulation, held, bodies, k)[:3]
    return jacobian_at(articulation, held, bodies, articulation.frame_bodies[k], point)
