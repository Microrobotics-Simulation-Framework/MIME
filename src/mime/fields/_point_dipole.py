from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, override

import jax
import numpy as np

from mime.core import as_run_array, quaternion
from mime.core.multibody import compose
from mime.fields._magnetic_model import INPUTS, MOMENT, QUANTITIES, ROWS, magnetic_outputs
from mime.gradients import GradientStatus
from mime.nodes.actuation import PointDipoleNode, point_dipole
from mime.registry import PHYSICAL
from mime.registry import BuildContext, FieldProvider, ForwardModel, Signal, register_field_provider
from mime.things import PermanentMagnet
from mime.vocabulary import FLUX_DENSITY, GRADIENT, Capability

if TYPE_CHECKING:
    from mime.build import ProviderBuilder, Slot

_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


@register_field_provider
@dataclass(frozen=True, kw_only=True)
class PointDipole(FieldProvider):
    """The field of a permanent magnet as a point dipole: ``B = mu_0/(4 pi) (3 (m . r) r / r^2 - m) / r^3`` and its
    gradient, at each reader's point, ``r`` the reader's point less the magnet's centroid. ``part`` names the magnet,
    a part of the apparatus with the ``PermanentMagnet`` role: its moment is the apparatus's ``magnetic_moment``,
    declared as a magnitude along the direction the role declares in the part's frame, and it sits where its mount
    puts it, so it follows the chain that carries it with no edge written by hand.

    The near-field limit: a point dipole is faithful only far from the magnet, against its radius, the radius of the
    smallest sphere about its centroid that holds its solid, which the provider's node holds as ``magnet_radius``;
    closer, the finite magnet's field differs by more (MIME-ANO-100). No finite-magnet model is built. At the magnet's
    centroid the field is singular, so a reader is evaluated outside the magnet only: the build refuses a body that
    starts inside the sphere that holds it.

    A reader's slot is evaluated at the position of the pose it reads, or, where the read names a point fixed in the
    body (``Capability.BODY_POINTS``: a magnetic law's ``magnet_centre``), at that point, the pose's position plus its
    orientation applied to it."""

    name = "point_dipole"
    version = "0.1.0"
    setting_roles = {"part": PHYSICAL}
    differentiability = GradientStatus.EXACT
    references = ("Jackson1998",)
    builds = {"": (PointDipoleNode,)}
    outside_parts = True
    validity = ("evaluated outside the magnet: the field is singular at its centroid, and a point dipole is faithful "
                "only far from the magnet, against its radius (MIME-ANO-100); a validity line that checks the "
                "distance is a1's",)
    flags = frozenset({Capability.PROVISION_EVALUATED, Capability.BODY_POINTS})
    publishes = {FLUX_DENSITY.name: "flux_density", GRADIENT.name: "gradient"}
    parts = {"part": PermanentMagnet.model}
    values = {"moment": "magnetic_moment"}
    part: str

    @override
    def build(self, ctx: BuildContext, b: ProviderBuilder, slots: Sequence[Slot]) -> None:
        at_body_points = [slot for slot in slots if slot.offset is not None]
        part = ctx.parts["part"]
        mount = part.mount
        moment = ctx.vector_value("moment")
        node = b.add_node(PointDipoleNode(
            ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots], moment=moment, placement=part.offset,
            magnet_radius=part.radius, mount_index=mount.index if mount else -1,
            mount_count=mount.count if mount else 0, initial_mount_pose=mount.initial_pose if mount else _IDENTITY,
            offset_slots=tuple(slot.name for slot in at_body_points),
            offsets=tuple(c for slot in at_body_points for c in slot.offset or ())))
        if mount is not None:
            b.part("part", into=node["mount_poses"])

    @override
    def forward_model(self, ctx: BuildContext) -> ForwardModel:
        """The field, its gradient, and the force and torque on a moment stated at the point, at a world point
        (``x.point``, and ``x.moment``), with the frame the magnet's part is mounted on at ``x.mount_pose`` where it is
        mounted: the node's closed form, its parameters the node's leaves, ``moment`` (the magnet's own, in its part's
        frame) and ``placement`` (its part's pose in that frame). It takes no command, so it is affine in none: the
        magnet moves with what carries it."""
        part = ctx.parts["part"]
        mounted = part.mount is not None
        inputs = {**INPUTS, **({"mount_pose": Signal("m; 1", (7,))} if mounted else {})}
        theta = {"moment": np.asarray(ctx.vector_value("moment"), dtype=np.float64),
                 "placement": np.asarray(part.offset, dtype=np.float64)}
        return ForwardModel(ctx.id, {}, inputs, QUANTITIES, theta, {"moment": "A m^2", "placement": "m; 1"},
                            partial(_dipole_model, mounted=mounted), optional=frozenset({MOMENT}), rows=ROWS)


def _dipole_model(u: Any, x: Any, theta: Any, *, mounted: bool) -> dict[str, Any]:
    """The point dipole's forward model: the node's arithmetic, in the run's dtype."""
    jnp: Any = jax.numpy
    dtype = as_run_array(0.0).dtype
    placement = jnp.asarray(theta["placement"], dtype=dtype)
    part = compose(jnp.asarray(x["mount_pose"], dtype=dtype), placement) if mounted else placement
    moment = quaternion.rotate(part[3:], jnp.asarray(theta["moment"], dtype=dtype))
    field, gradient = point_dipole(moment, part[:3], jnp.asarray(x["point"], dtype=dtype))
    return magnetic_outputs(field, gradient, x, dtype)


__all__ = ["PointDipole"]
