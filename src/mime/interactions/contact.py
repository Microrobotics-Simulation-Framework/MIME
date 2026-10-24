"""The contact family: a body against a surface. Here now: projection against a fixed structure's surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.gradients import GradientStatus
from mime.registry import (
    BuildContext,
    Confinement,
    GradientRecord,
    Interaction,
    Plan,
    PlanContext,
    RoleKind,
    SupportRow,
    Value,
    register_interaction,
    requires,
)
from mime.units import Length, Magnitude, Quantity
from mime.vocabulary import CONFINEMENT, Capability

if TYPE_CHECKING:
    from mime.build import InteractionBuilder


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Contact(Interaction):
    """A body in contact with a surface (``surface=tube.inner_surface``), by the written rung.

    ``projection``: the body stays on the surface's free side (the side it faces: ``tube.inner_surface`` faces the
    lumen) by its ``extent`` plus ``margin``. It builds no node: its record goes to the body's dynamics at
    construction, which holds ``rule`` (``remove_outward``: the outward part removed) inside its implicit solve while
    the body starts a step on the surface, and projects the position back after each step, changing the velocity
    there by ``rule``. One per body; only against a fixed structure's surface, since
    projection between two moving things is reserved. Its derivative is piecewise: it jumps where a step changes the
    contact's branch (held in the solve, projected after the step, or free), as its gradient record says. ``penalty``
    (loads into each side's applied wrench) is reserved. ``Contact.Projection(...)`` is
    ``Contact(rung="projection", ...)`` and saves identically."""

    name = "contact"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE
    gradient = GradientRecord(
        where="where a step changes the contact's branch: across the edge of the contact band (the surface shrunk by "
              "the body's extent and margin, give or take CONTACT_ULPS units in the last place), where holding the "
              "contact in the solve starts or stops; where a step ends on that surface, where the projection and the "
              "velocity rule start; and where the outward velocity crosses zero",
        reason="the branch (held in the solve, projected after the step, or free) is chosen per step from the state, "
               "and the derivative is that of the branch taken; the twist jumps across the band's edge, and where a "
               "step ends on the surface moving outward (its outward velocity kept on the surface, removed beyond it)",
        route="none for projection: a step that jumps has no derivative there to compute. The only route is "
              "another rung, penalty, which is declared and not built (reserved): loads from a continuously "
              "differentiable law of the gap would give the step a derivative everywhere, and nothing of it exists "
              "yet to check that against")
    references = ()
    family = "contact"
    builds = {"projection": ()}
    roles = {"body": RoleKind.BODY, "surface": RoleKind.SURFACE}
    rungs = ("projection", "penalty")
    reserved_rungs = {"penalty": "penalty contact, delivering loads into each side's applied wrench, comes later"}
    support = (SupportRow("projection", {"body": requires(Capability.PROJECTS_CONFINEMENT, accepts=[CONFINEMENT.name])},
                          note="the body's dynamics projects a confinement in its own update"),)
    rung: str | None = None
    margin: Annotated[Quantity, Length]
    rule: str = "remove_outward"

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        margin = settings["margin"]
        if isinstance(margin, float) and margin >= 0.0:
            return []
        return [f"margin is {margin!r} m, not a length from 0 on"]

    @classmethod
    def Projection(cls, **settings: Any) -> Contact:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="projection", **settings)

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(nodes=(), values={"extent": Value("body", "extent")},
                    confines=(Confinement("body", CONFINEMENT.name, "surface", ctx.scalar("margin"), self.rule),))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        return None


__all__ = ["Contact"]
