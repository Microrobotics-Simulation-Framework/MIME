"""The version of the contact kept registered after 0.2.0 held the body's hull points, so that a folder saved with it
still loads and reproduces: the contact 0.1.0, which holds the body's centre by its extent, its orientation unread.
Each kept version is its successor's entry in every respect but what it holds. A script that writes
``Contact.Projection(...)`` gets the current version; a load of a document that names 0.1.0 gets it from here, and so
does an experiment whose comparisons were declared against it (script A's and A0's reproductions of the 0.2 line,
whose contact held the centre), by ``KEPT``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from mime.gradients import GradientStatus
from mime.interactions.contact import CENTRE_SWITCHES, Contact
from mime.registry import (
    DESIGN,
    NUMERICAL,
    PHYSICAL,
    Confinement,
    GradientRecord,
    Plan,
    PlanContext,
    Problem,
    Value,
    convention,
    register_interaction,
)
from mime.vocabulary import CONFINEMENT


@register_interaction
@dataclass(frozen=True, kw_only=True)
class CentreContact(Contact):
    """The contact 0.1.0: the body stays on the surface's free side by its ``extent`` plus ``margin``, its centre held
    and its orientation never read: the body's dynamics holds ``rule`` (``remove_outward``) inside its implicit solve
    while the body starts a step on the surface shrunk by both, and projects the position back after each step,
    changing the velocity there by ``rule``. A body that turns across a tube narrower than it is long passes through
    the wall, unseen (0.2.0 holds its hull's points). It reads no ``spacing``."""

    version = "0.1.0"
    setting_roles = {
        "rung": DESIGN, "margin": NUMERICAL,
        "rule": convention(PHYSICAL, "a frictionless contact that does not bounce: the projection removes the "
                                     "velocity into the surface, a physical model of the contact, not a solver's "
                                     "choice"),
        "spacing": convention(NUMERICAL, "0.1.0 holds the body's centre and samples no hull, so it reads no spacing"),
    }
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=CENTRE_SWITCHES)
    references = ()
    validity = (
        "projection: the body's centre held by its extent about its own axis plus the margin, its orientation unread: "
        "exact for a sphere, and for a body whose axis stays along a tube's; a body turned across the tube is held "
        "as if it were not",
    )

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        unread = () if self.spacing is None else (Problem(
            "confinement.no-hull", "spacing= is the spacing of a hull's points, which the contact 0.1.0 never "
                                   "samples: it holds the body's centre by its extent",
            "remove spacing=, or write Contact.Projection(...) for the contact that holds the hull"),)
        return Plan(nodes=(), values={"extent": Value("body", "extent")}, problems=unread,
                    confines=(Confinement("body", CONFINEMENT.name, "surface", ctx.scalar("margin"), self.rule),))


__all__ = ["CentreContact"]
