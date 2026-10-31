"""The contact family: a body against a surface. Here now: projection against a fixed structure's surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.core import switch
from mime.gradients import GradientStatus
from mime.registry import DESIGN, NUMERICAL, PHYSICAL, convention
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
    contact's branch (held in the solve, projected after the step, or free), as its switches say. ``penalty``
    (loads into each side's applied wrench) is reserved. ``Contact.Projection(...)`` is
    ``Contact(rung="projection", ...)`` and saves identically."""

    name = "contact"
    version = "0.1.0"
    setting_roles = {
        "rung": DESIGN, "margin": NUMERICAL,
        "rule": convention(PHYSICAL, "a frictionless contact that does not bounce: the projection removes the "
                                     "velocity into the surface, a physical model of the contact, not a solver's "
                                     "choice"),
    }
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="contact.band",
            indicator="the body's signed distance past its surface shrunk by its extent and margin, m; on the surface "
                      "within CONTACT_ULPS units in the last place of |x| plus that inset (a band)",
            orders={"twist": -1, "pose": -1},
            route="a landing located within the step and its time differentiated, as MADDENING's events will be "
                  "(planned for MADDENING 0.5 and MIME 0.4.0); until then a fit starts in contact and keeps it"),
        switch.Declaration(
            owner="contact.outward",
            indicator="the outward speed the solve leaves the body with, less the speed that closes its gap in the "
                      "step, m/s",
            orders={"twist": 0, "pose": 0},
            route="none: the held and free twists are equal where it is zero, and their derivatives are not"),
        switch.Declaration(
            owner="contact.hold",
            indicator="the outward speed the hold removes while the body is in the band, m/s",
            orders={"twist": -1, "pose": -1},
            route="its jump, at the band's edge, is carried by contact.band's declaration, whose row measures it: "
                  "along its own indicator, the outward speed, the held and free twists meet at zero, so its row "
                  "measures it continuous; the route is contact.band's, and contact.outward's across the outward "
                  "speed"),
        switch.Declaration(
            owner="contact.closing",
            indicator="the gap to the shrunk surface, m (positive inside)",
            orders={"twist": 0, "pose": 0},
            route="none: the speed that closes the gap is its positive part"),
        switch.Declaration(
            owner="contact.scale",
            indicator="the body's distance from the world's origin, m, which the band's width grows with",
            orders={"twist": 0, "pose": 0},
            route="a band width smooth in the position, such as one from a norm smoothed at the origin",
            unmeasured="live at the world's origin, and with no effect an output shows: the norm is continuous there "
                       "and reaches only the band's width, about 1e-16 m, which only contact.band's comparison reads"),
        switch.Declaration(
            owner="contact.project",
            indicator="the body's signed distance past the shrunk surface at the end of the step, m",
            orders={"pose": 0},
            route="none: the projection removes that distance's positive part"),
        switch.Declaration(
            owner="contact.rule",
            indicator="the body's signed distance past the shrunk surface at the end of the step, m",
            orders={"twist": -1},
            route="as contact.band's: a landing located within the step"),
        switch.Declaration(
            owner="contact.rule_outward",
            indicator="the velocity's component along the outward normal where the body was projected, m/s",
            orders={"twist": 0},
            route="none: the rule removes that component's positive part"),
    ))
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
