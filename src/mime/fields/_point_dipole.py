from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from mime.core import switch
from mime.gradients import GradientStatus
from mime.nodes.actuation import PointDipoleFieldNode
from mime.registry import BuildContext, FieldProvider, GradientRecord, register_field_provider
from mime.things import PermanentMagnet
from mime.vocabulary import FLUX_DENSITY, GRADIENT, Capability

if TYPE_CHECKING:
    from mime.build import ProviderBuilder, Slot

_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


@register_field_provider
@dataclass(frozen=True, kw_only=True)
class PointDipoleField(FieldProvider):
    """The field of a permanent magnet as a point dipole: ``B = mu_0/(4 pi) (3 (m . r) r / r^2 - m) / r^3`` and its
    gradient, at each reader's point, ``r`` the reader's point less the magnet's centroid. ``part`` names the magnet,
    a part of the apparatus with the ``PermanentMagnet`` role: its moment is the apparatus's ``magnetic_moment``,
    declared as a magnitude along the direction the role declares in the part's frame, and it sits where its mount
    puts it, so it follows the chain that carries it with no edge written by hand.

    The near-field limit: a point dipole is faithful only far from the magnet, against its radius, the radius of the
    smallest sphere about its centroid that holds its solid, which the provider's node holds as ``magnet_radius``;
    closer, the finite magnet's field differs by more (MIME-ANO-100). No finite-magnet model is built. At the magnet's
    centroid the field is singular, and zero there by a declared switch that no reader outside the magnet meets."""

    name = "point_dipole_field"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="point_dipole_field.centre",
            indicator="the squared distance from the magnet's centroid to the reader's point, in square metres",
            orders={"flux_density": -1, "gradient": -1},
            route="none: the field is singular at the dipole's centre, where no derivative exists; a reader outside "
                  "the magnet never meets it, and a point that near is past the model's validity (MIME-ANO-100)"),
    ))
    references = ("Jackson1998",)
    builds = {"": (PointDipoleFieldNode,)}
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
        node = b.add_node(PointDipoleFieldNode(
            ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots], moment=moment, placement=part.offset,
            magnet_radius=part.radius, mount_index=mount.index if mount else -1,
            mount_count=mount.count if mount else 0, initial_mount_pose=mount.initial_pose if mount else _IDENTITY))
        if mount is not None:
            b.part("part", into=node["mount_poses"])


__all__ = ["PointDipoleField"]
