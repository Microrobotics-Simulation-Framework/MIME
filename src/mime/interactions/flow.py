"""Laws that drive a medium's flow: ``BodyForce``, the body force that stands in for a pressure drop on a periodic box
(F0's B6 and B7: drives are laws, a body force bound to the medium with no surface)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, override

from mime.gradients import GradientStatus
from mime.nodes.flow._body_force import BodyForceNode
from mime.registry import (
    DESIGN,
    PHYSICAL,
    BuildContext,
    Delivery,
    Interaction,
    Plan,
    PlanContext,
    RoleKind,
    SupportRow,
    convention,
    register_interaction,
    requires,
)
from mime.units import Acceleration, Frequency, Quantity
from mime.vocabulary import FLOW_BODY_FORCE

if TYPE_CHECKING:
    from mime.build import InteractionBuilder


@register_interaction
@dataclass(frozen=True, kw_only=True)
class BodyForce(Interaction):
    """A uniform body force per unit mass on a medium's fluid, along a direction: ``steady``, plus ``amplitude cos(2 pi
    frequency t + phase)`` for a pulsatile drive, ``exp.attach(BodyForce(along=(0, 0, 1), steady=m_per_s2(8)),
    medium=water)``. The flow field applies it to its fluid alone, never its solid, so the flux it drives does not
    depend on how much solid the field's domain holds. The waveform is the drive's; the field kind is what responds."""

    name = "body_force"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ()
    family = None
    builds = {"": (BodyForceNode,)}
    roles = {"medium": RoleKind.MEDIUM}
    support = (SupportRow(None, {"medium": requires(accepts=[FLOW_BODY_FORCE.name])},
                          note="a medium whose flow field takes a body force"),)
    delivers = {"force": Delivery("medium", FLOW_BODY_FORCE.name)}
    unmetered = ("a drive's supplied work, paired by the flow field it drives, which the ledger does not yet meter "
                 "through the build")
    setting_roles = {
        "along": DESIGN, "steady": PHYSICAL,
        "amplitude": convention(PHYSICAL, "a steady drive unless a pulsatile one is declared"),
        "frequency": convention(PHYSICAL, "a steady drive unless a pulsatile one is declared"),
        "phase": convention(PHYSICAL, "the pulsatile part starts at its peak (radians)"),
    }

    along: tuple[float, float, float]
    steady: Annotated[Quantity, Acceleration]
    amplitude: Annotated[Quantity, Acceleration] | None = None
    frequency: Annotated[Quantity, Frequency] | None = None
    phase: float = 0.0

    @override
    def setting_problems(self, settings) -> list[str]:  # type: ignore[no-untyped-def]
        found: list[str] = []
        length = sum(float(c) ** 2 for c in self.along) ** 0.5
        if abs(length - 1.0) > 1e-9:
            found.append(f"along is a unit direction, and {self.along} has length {length:.6g}")
        if (self.amplitude is None) != (self.frequency is None):
            found.append("a pulsatile drive takes both its amplitude and its frequency")
        return found

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(delivers=dict(self.delivers))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(BodyForceNode(
            ctx.node_name(), ctx.timestep, along=self.along, steady=ctx.scalar("steady"),
            amplitude=ctx.scalar("amplitude") if self.amplitude is not None else 0.0,
            frequency=ctx.scalar("frequency") if self.frequency is not None else 0.0,
            phase=self.phase))
        b.connect(node["force"], into=b.target("force"))


__all__ = ["BodyForce"]
