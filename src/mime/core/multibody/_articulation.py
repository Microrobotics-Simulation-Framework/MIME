"""The articulation: a tree compiled into the arrays the algorithms read. Bodies welded by fixed joints are one rigid
body here, their inertials combined; every named frame keeps its own offset in the rigid body it moved with.

Its arrays are JAX pytree leaves (float64 NumPy arrays as compiled), so a study can differentiate with respect to a
mass or an axis; its structure (names, parents, joint kinds) is static."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.core.multibody._tree import IDENTITY, ClosedChainRefused, Inertial, Joint, JointKind, MultibodyTree, \
    RigidTransform, combined

type Arrayish = Any

#: The index of the root's rigid body, which is fixed to the world.
ROOT = -1


@dataclass(frozen=True, eq=False)
class Articulation:
    """A tree's degrees of freedom and frames, as arrays. Joint ``i`` moves rigid body ``i``; ``parents[i]`` is the
    rigid body it is mounted on (``ROOT`` for the root's), always of a lower index, so a pass in index order meets a
    parent before its children (depth first from the root, siblings in the order their joints are listed). Each rigid
    body's frame is its joint's child frame; ``origins`` place it in its parent's at zero joint position.

    ``frames`` are every named frame of the tree, in its ``frame_names`` order, each with the rigid body it is fixed
    to and its offset there. ``closures`` names the tree's loop closures; every algorithm refuses an articulation with
    any."""

    dof_names: tuple[str, ...]
    parents: tuple[int, ...]
    revolute: tuple[bool, ...]
    frame_names: tuple[str, ...]
    frame_bodies: tuple[int, ...]
    closures: tuple[str, ...]
    axes: Arrayish
    origin_rotations: Arrayish
    origin_translations: Arrayish
    masses: Arrayish
    centres_of_mass: Arrayish
    inertias: Arrayish
    armatures: Arrayish
    frame_rotations: Arrayish
    frame_translations: Arrayish
    lower: Arrayish
    upper: Arrayish

    @property
    def dofs(self) -> int:
        return len(self.dof_names)

    def frame_index(self, frame: str) -> int:
        if frame not in self.frame_names:
            raise KeyError(f"no frame {frame!r}; the frames are {list(self.frame_names)}")
        return self.frame_names.index(frame)

    def chain(self, body: int) -> tuple[int, ...]:
        """The joints that move ``body``: its own and its ancestors', root first."""
        found: list[int] = []
        while body != ROOT:
            found.append(body)
            body = self.parents[body]
        return tuple(reversed(found))

    def ancestry(self) -> np.ndarray:
        """``A[i, j]`` is 1 where joint ``j`` moves rigid body ``i`` (``j`` is ``i`` or an ancestor of it)."""
        mask = np.zeros((self.dofs, self.dofs))
        for i in range(self.dofs):
            mask[i, list(self.chain(i))] = 1.0
        return mask

    def refuse_closures(self, algorithm: str) -> None:
        if self.closures:
            raise ClosedChainRefused(algorithm, self.closures)


_STATIC = ("dof_names", "parents", "revolute", "frame_names", "frame_bodies", "closures")
_ARRAYS = ("axes", "origin_rotations", "origin_translations", "masses", "centres_of_mass", "inertias", "armatures",
           "frame_rotations", "frame_translations", "lower", "upper")
cast(Any, jax).tree_util.register_dataclass(Articulation, data_fields=list(_ARRAYS), meta_fields=list(_STATIC))


def articulate(tree: MultibodyTree, frames: Sequence[str] | None = None) -> Articulation:
    """Compile ``tree``: number its moving joints depth first, weld each body to the rigid body it moves with, and
    combine the inertials welded together, each about their common centre of mass. ``frames`` names the frames it
    lists, in that order (every frame of the tree by default): a tree with payloads grafted on keeps its description's
    frames this way, its payloads' bodies adding mass and no frame."""
    rigid_body: dict[str, int] = {}
    placement: dict[str, RigidTransform] = {}
    moving: list[Joint] = []
    for body, joint in tree.walk():
        if joint is None:
            rigid_body[body], placement[body] = ROOT, IDENTITY
        elif joint.moves:
            rigid_body[body], placement[body] = len(moving), IDENTITY
            moving.append(joint)
        else:
            rigid_body[body] = rigid_body[joint.parent]
            placement[body] = placement[joint.parent].then(joint.origin)

    parts: dict[int, list[Inertial]] = {}
    for body in tree.bodies:
        if body.inertial is not None:
            parts.setdefault(rigid_body[body.name], []).append(body.inertial.placed(placement[body.name]))
    inertials = [combined(parts.get(i, [])) for i in range(len(moving))]

    named = {b.name: (rigid_body[b.name], placement[b.name]) for b in tree.bodies}
    named.update({f.name: (rigid_body[f.body], placement[f.body].then(f.offset)) for f in tree.frames})
    listed = tuple(frames) if frames is not None else tree.frame_names
    unknown = [name for name in listed if name not in named]
    if unknown:
        raise KeyError(f"no frames {unknown} in the tree; its frames are {list(tree.frame_names)}")
    placed = [named[name] for name in listed]
    origins = [placement[j.parent].then(j.origin) for j in moving]

    def stacked(values: list[Any], shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(values, dtype=np.float64).reshape((len(values), *shape))

    return Articulation(
        dof_names=tuple(j.name for j in moving),
        parents=tuple(rigid_body[j.parent] for j in moving),
        revolute=tuple(j.kind is JointKind.REVOLUTE for j in moving),
        frame_names=listed,
        frame_bodies=tuple(body for body, _ in placed),
        closures=tuple(c.name for c in tree.closures),
        axes=stacked([j.axis for j in moving], (3,)),
        origin_rotations=stacked([o.rotation for o in origins], (4,)),
        origin_translations=stacked([o.translation for o in origins], (3,)),
        masses=stacked([p.mass for p in inertials], ()),
        centres_of_mass=stacked([p.centre_of_mass for p in inertials], (3,)),
        inertias=stacked([p.inertia for p in inertials], (3, 3)),
        armatures=stacked([j.armature for j in moving], ()),
        frame_rotations=stacked([offset.rotation for _, offset in placed], (4,)),
        frame_translations=stacked([offset.translation for _, offset in placed], (3,)),
        lower=stacked([j.limits.lower for j in moving], ()),
        upper=stacked([j.limits.upper for j in moving], ()),
    )
