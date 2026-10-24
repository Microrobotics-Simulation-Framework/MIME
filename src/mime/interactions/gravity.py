"""The gravity family: weight and buoyancy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from mime.gradients import GradientStatus
from mime.nodes.body import GravityNode
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
from mime.units import Magnitude
from mime.vocabulary import APPLIED_WRENCH, POSE, Capability

if TYPE_CHECKING:
    from mime.build import InteractionBuilder

_VALUES = {"mass": Value("body", "mass"), "buoyant_volume": Value("body", "buoyant_volume"),
           "fluid_density": Value("medium", "density"), "centre_of_buoyancy": Value("body", "centre_of_buoyancy")}


def _off_centre(centre: Magnitude | None) -> bool:
    return isinstance(centre, tuple) and any(c != 0.0 for c in centre)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Gravity(Interaction):
    """Weight minus buoyancy, ``(m - rho_f V) g`` with ``m`` the body's mass, ``V`` its buoyant volume, ``rho_f`` the
    medium's density and ``g = -|g| up`` the experiment's gravity, so ``(rho_body - rho_fluid) V g`` for a body of
    one density. The weight acts at the centre of mass and the buoyancy at the centre of buoyancy, so when the body
    declares that point off its centre of mass the law also gives the couple, and reads the body's pose; only then.
    The medium is at rest: with a flow that carries its own hydrostatic load, buoyancy would be counted twice, and no
    such flow is built yet."""

    name = "gravity"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    family = "gravity"
    builds = {"": (GravityNode,)}
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    support = (SupportRow(None, {"body": requires(Capability.RIGID, accepts=[APPLIED_WRENCH.name])},
                          note="a rigid body in a medium at rest"),)
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        reads = {"pose": Read("body", POSE.name)} if _off_centre(ctx.value("body", "centre_of_buoyancy")) else {}
        return Plan(reads=reads, delivers=dict(self.delivers), values=dict(_VALUES))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        centre = ctx.values["centre_of_buoyancy"]
        couples = _off_centre(centre)
        node = b.add_response_node(GravityNode(
            ctx.node_name(), ctx.timestep, mass=float(ctx.scalar_value("mass")),
            buoyant_volume=float(ctx.scalar_value("buoyant_volume")),
            fluid_density=float(ctx.scalar_value("fluid_density")), gravity=ctx.gravity,
            centre_of_buoyancy=centre if couples and isinstance(centre, tuple) else None))
        if couples:
            b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["wrench"], into=b.target("wrench"))


__all__ = ["Gravity"]
