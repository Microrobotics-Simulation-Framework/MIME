"""The magnetic family: laws of a field on a body's magnetic moment. Here now: the torque on a permanent moment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from mime.gradients import GradientStatus
from mime.nodes.magnetic import MagneticTorqueNode
from mime.registry import (
    NOT_ESTABLISHED,
    BuildContext,
    Delivery,
    Interaction,
    Read,
    RoleKind,
    SupportRow,
    Value,
    register_interaction,
    requires,
)
from mime.vocabulary import APPLIED_WRENCH, FLUX_DENSITY, POSE, Capability

if TYPE_CHECKING:
    from mime.build import InteractionBuilder


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MagneticTorque(Interaction):
    """``tau = m x B``: the torque of the field bound to ``field`` on the body's permanent moment ``m`` (its
    ``magnetic_moment``, body frame, turned into the world frame by the body's orientation), about the centre of
    mass. It delivers no force (a uniform field has no gradient), so it binds a uniform field only. It reads the
    body's pose at the start of the step, as every law reads its body's state, so the torque of step ``n`` turns the
    body as it stood after step ``n - 1``: a lag of one step, ``omega dt`` in phase at a field rate ``omega``."""

    name = "magnetic.torque"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = NOT_ESTABLISHED
    family = "magnetic"
    builds = {"": (MagneticTorqueNode,)}
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    support = (SupportRow(None, {
        "body": requires(Capability.RIGID, accepts=[APPLIED_WRENCH.name], publishes=[POSE.name]),
        "field": requires(Capability.UNIFORM, publishes=[FLUX_DENSITY.name]),
    }, note="a permanent moment in a uniform field"),)
    reads = {"pose": Read("body", POSE.name), "flux_density": Read("field", FLUX_DENSITY.name)}
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}
    values = {"moment": Value("body", "magnetic_moment")}

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        moment = ctx.values["moment"]
        node = b.add_response_node(MagneticTorqueNode(ctx.node_name(), ctx.timestep,
                                                      moment=moment if isinstance(moment, Sequence) else (moment,)))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(b.read("flux_density"), into=node["flux_density"])
        b.connect(node["wrench"], into=b.target("wrench"))


class Magnetic:
    """The magnetic family's laws, by name: ``Magnetic.Torque()``. The magnetic lock is reserved as a rung of the
    magnetic law, not built: step-out emerges from the torque on an inertial body."""

    Torque = MagneticTorque


__all__ = ["Magnetic", "MagneticTorque"]
