from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from mime.gradients import GradientStatus
from mime.nodes.actuation import PointDipoleNode
from mime.registry import BuildContext, FieldProvider, register_field_provider
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
    starts inside the sphere that holds it."""

    name = "point_dipole"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ("Jackson1998",)
    builds = {"": (PointDipoleNode,)}
    outside_parts = True
    validity = ("evaluated outside the magnet: the field is singular at its centroid, and a point dipole is faithful "
                "only far from the magnet, against its radius (MIME-ANO-100); a validity line that checks the "
                "distance is a1's",)
    flags = frozenset({Capability.PROVISION_EVALUATED})
    publishes = {FLUX_DENSITY.name: "flux_density", GRADIENT.name: "gradient"}
    parts = {"part": PermanentMagnet.model}
    values = {"moment": "magnetic_moment"}
    part: str

    @override
    def build(self, ctx: BuildContext, b: ProviderBuilder, slots: Sequence[Slot]) -> None:
        part = ctx.parts["part"]
        mount = part.mount
        moment = ctx.vector_value("moment")
        node = b.add_node(PointDipoleNode(
            ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots], moment=moment, placement=part.offset,
            magnet_radius=part.radius, mount_index=mount.index if mount else -1,
            mount_count=mount.count if mount else 0, initial_mount_pose=mount.initial_pose if mount else _IDENTITY))
        if mount is not None:
            b.part("part", into=node["mount_poses"])


__all__ = ["PointDipole"]
