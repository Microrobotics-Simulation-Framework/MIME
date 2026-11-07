"""The magnetic family: laws of a field on a body's magnetic moment. Here now: the torque on a permanent moment, and
the force of a field's gradient on it. Each delivers its reaction on the field's source, equal and opposite, where the
source is mounted on a chain segment that takes loads. Both read a field that varies in space at the body's magnet's
centroid, so together they are the derivative of the energy ``-m . B`` of a point dipole there; their versions 0.1.0,
which read it at the centre of mass (MIME-ANO-137), are kept registered (``mime.interactions.KEPT``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from mime.gradients import GradientStatus
from mime.nodes.actuation import SourceReactionNode
from mime.nodes.magnetic import MagneticForceNode, MagneticTorqueNode
from mime.registry import (
    BuildContext,
    Delivery,
    Interaction,
    Plan,
    PlanContext,
    Read,
    RoleKind,
    SupportRow,
    Value,
    register_interaction,
    requires,
)
from mime.vocabulary import APPLIED_WRENCH, FLUX_DENSITY, GRADIENT, LINK_WRENCH, POSE, Capability

#: What a rigid body must offer a magnetic law: it publishes its pose and takes an applied wrench.
_RIGID_BODY = requires(Capability.RIGID, accepts=[APPLIED_WRENCH.name], publishes=[POSE.name])
#: Where a magnetic law reads a field that varies in space: the point fixed in the body at its ``magnet_centre``.
_MAGNET_CENTRE = "magnet_centre"

if TYPE_CHECKING:
    from mime.build import InteractionBuilder, NodeRef

#: The local name of a law's reaction node.
REACTION = "reaction"


def with_reaction(plan: Plan, ctx: PlanContext) -> Plan:
    """``plan`` with the reaction on the field's source, where a segment takes it (``ctx.reaction("field")``): one more
    node, which reads the body's pose, and a delivery into that segment's loads."""
    if ctx.reaction("field") is None:
        return plan
    return Plan(nodes=(*plan.nodes, REACTION), reads={**plan.reads, REACTION: Read("body", POSE.name)},
                delivers={**plan.delivers, REACTION: Delivery("field", LINK_WRENCH.name)}, values=dict(plan.values))


def build_reaction(ctx: BuildContext, b: InteractionBuilder, law: NodeRef) -> None:
    """Where its plan delivers one, the reaction of ``law``'s wrench on the body: equal and opposite on the source,
    moved to the frame the source is mounted on."""
    if not b.delivers(REACTION):
        return
    mount = b.target(REACTION).target.mount
    assert mount is not None
    node = b.add_response_node(SourceReactionNode(ctx.node_name(REACTION), ctx.timestep, mount_index=mount.index,
                                                  mount_count=mount.count), REACTION)
    b.connect(b.read(REACTION), into=node["pose"])
    b.connect(law["wrench"], into=node["wrench"])
    b.mount(REACTION, into=node["mount_poses"])
    b.connect(node["link_wrench"], into=b.target(REACTION))


#: Why the magnetic laws declare no class for the energy ledger yet: their work on the body is recorded and attributed
#: to them all the same.
_UNMETERED = ("its class waits on the magnetic family's energy, -m . B at the magnet's centroid, of which the pair is "
              "the derivative, stated with the family")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MagneticTorque(Interaction):
    """``tau = m x B``: the torque of the field bound to ``field`` on the body's permanent moment ``m`` (its
    ``magnetic_moment``, body frame, turned into the world frame by the body's orientation), about the centre of
    mass. It delivers no force: a field's gradient pulls on the moment by another law, ``Magnetic.Force``. A field that
    varies in space is read where the magnet is, at the body's ``magnet_centre`` (from its parts, or declared) turned
    by its orientation from its position, so that with ``Magnetic.Force`` it is the derivative of a point dipole's
    energy there and does no net work round a closed loop of the pose; that holds while the field varies little across
    the magnet, outside a point dipole's near field (MIME-ANO-100). A uniform field is read as before, with no points
    and no centroid. It reads the body's pose at the start of the step, as every law reads its body's state, so the
    torque of step ``n`` turns the body as it stood after step ``n - 1``: a lag of one step, ``omega dt`` in phase at a
    field rate ``omega``.

    Its reaction, ``-tau`` on the field's source, is delivered to the frame the source is mounted on where that
    segment takes loads (a dynamic one); with ``Magnetic.Force``'s, it is the whole of what the body's moment exerts
    on a point dipole source.

    Version 0.2.0 reads the field at the magnet's centroid; 0.1.0 read it at the centre of mass, so with a magnet off
    the centre of mass, in a field that varies in space, the pair missed ``m x (G l)`` and was not conservative
    (MIME-ANO-137). 0.1.0 is kept registered (``mime.interactions.KEPT``)."""

    name = "magnetic.torque"
    version = "0.2.0"
    unmetered = _UNMETERED
    differentiability = GradientStatus.EXACT
    references = ("Ling2016UniversityPhysics2",)
    family = "magnetic"
    builds = {"": (MagneticTorqueNode, SourceReactionNode)}
    validity = ("a permanent moment in a field that varies in space, read at its magnet's centroid: a point dipole there, "
                "valid while the field varies little across the magnet, and outside a point dipole's near field "
                "(MIME-ANO-100)",)
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    support = (
        SupportRow(None, {"body": _RIGID_BODY, "field": requires(Capability.UNIFORM, publishes=[FLUX_DENSITY.name])},
                   note="a permanent moment in a uniform field"),
        SupportRow(None, {"body": _RIGID_BODY,
                          "field": requires(Capability.PROVISION_EVALUATED, Capability.BODY_POINTS,
                                            publishes=[FLUX_DENSITY.name])},
                   note="a permanent moment in a field that varies in space, read at its magnet's centroid: valid "
                        "while the field varies little across the magnet, and outside a point dipole's near field "
                        "(MIME-ANO-100)"),
    )
    reads = {"pose": Read("body", POSE.name), "flux_density": Read("field", FLUX_DENSITY.name)}
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}
    values = {"moment": Value("body", "magnetic_moment")}

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        """A uniform field is read with no points, as it always was; any other at the body's magnet's centroid, which
        it reads as a value. The reaction is planned where a segment takes it."""
        field = ctx.capabilities.get("field")
        if field is not None and Capability.UNIFORM in field.flags:
            return super().plan(ctx)
        reads = {**self.reads, "flux_density": Read("field", FLUX_DENSITY.name, at=Read("body", POSE.name),
                                                    offset=_MAGNET_CENTRE)}
        values = {**self.values, _MAGNET_CENTRE: Value("body", "magnet_centre")}
        return with_reaction(Plan(reads=reads, delivers=dict(self.delivers), values=values), ctx)

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        moment = ctx.values["moment"]
        node = b.add_response_node(MagneticTorqueNode(ctx.node_name(), ctx.timestep,
                                                      moment=moment if isinstance(moment, Sequence) else (moment,)))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(b.read("flux_density"), into=node["flux_density"])
        b.connect(node["wrench"], into=b.target("wrench"))
        build_reaction(ctx, b, node)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MagneticForce(Interaction):
    """``F = G^T m``: the force of the field bound to ``field`` on the body's permanent moment ``m`` (its
    ``magnetic_moment``, body frame, turned into the world frame by the body's orientation), a point dipole in a
    current-free field, ``G_ij = dB_i/dx_j`` the field's gradient read where the magnet is: the body's
    ``magnet_centre`` (from its parts, or declared) turned by its orientation from its position. The force acts there,
    so the wrench it delivers about the centre of mass is ``[F; l x F]``, ``l`` that centroid turned into the world
    frame. A uniform field's gradient is zero, exactly, and is read with no points. With ``Magnetic.Torque`` it is the
    derivative of a point dipole's energy at the centroid. It reads the body's pose at the start of the step, as every
    law reads its body's state.

    Its reaction, ``-F`` on the field's source along the same line of action, so with the torque ``-(x - o) x F``
    about the origin ``o`` of the frame the source is mounted on (``x`` the magnet's centroid), is delivered to that
    frame where its segment takes loads (a dynamic one). A source fixed in the world, or on a segment that takes no
    load (a kinematic one, whose record says so), takes none.

    Version 0.2.0 reads the gradient at the magnet's centroid; 0.1.0 read it at the centre of mass while the force
    acted at the centroid (MIME-ANO-137). 0.1.0 is kept registered (``mime.interactions.KEPT``)."""

    name = "magnetic.force"
    version = "0.2.0"
    unmetered = _UNMETERED
    differentiability = GradientStatus.EXACT
    references = ("Jackson1998",)
    family = "magnetic"
    builds = {"": (MagneticForceNode, SourceReactionNode)}
    validity = ("a point dipole at its magnet's centroid, in a field's gradient read there: valid while the gradient "
                "varies little across the magnet, and outside a point dipole's near field (MIME-ANO-100); a validity "
                "line that checks the distance is a1's",)
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    support = (
        SupportRow(None, {"body": _RIGID_BODY, "field": requires(Capability.UNIFORM, publishes=[GRADIENT.name])},
                   note="a permanent moment in a uniform field, whose gradient is zero"),
        SupportRow(None, {"body": _RIGID_BODY,
                          "field": requires(Capability.BODY_POINTS, publishes=[GRADIENT.name])},
                   note="a permanent moment, a point dipole, in a field's gradient read at its magnet's centroid: "
                        "valid while the gradient varies little across the magnet, and outside a point dipole's near "
                        "field (MIME-ANO-100)"),
    )
    reads = {"pose": Read("body", POSE.name),
             "gradient": Read("field", GRADIENT.name, at=Read("body", POSE.name), offset=_MAGNET_CENTRE)}
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}
    values = {"moment": Value("body", "magnetic_moment"), _MAGNET_CENTRE: Value("body", "magnet_centre")}

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        """Its reads, delivery and values, and the reaction where a segment takes it."""
        return with_reaction(super().plan(ctx), ctx)

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        moment, centre = ctx.vector_value("moment"), ctx.vector_value(_MAGNET_CENTRE)
        node = b.add_response_node(MagneticForceNode(ctx.node_name(), ctx.timestep, moment=moment,
                                                     magnet_centre=centre))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(b.read("gradient"), into=node["gradient"])
        b.connect(node["wrench"], into=b.target("wrench"))
        build_reaction(ctx, b, node)


class Magnetic:
    """The magnetic family's laws, by name: ``Magnetic.Torque()`` and ``Magnetic.Force()``. The magnetic lock is
    reserved as a rung of the magnetic law, not built: step-out emerges from the torque on an inertial body."""

    Torque = MagneticTorque
    Force = MagneticForce


__all__ = ["Magnetic", "MagneticForce", "MagneticTorque"]
