"""The versions of interactions kept registered after a later version changed what they compute, so that a folder
saved with one still loads and reproduces: the contact 0.1.0, which holds the body's centre by its extent, its
orientation unread; and the magnetic torque and force 0.1.0, which read a field that varies in space at the body's
centre of mass (MIME-ANO-137). Each kept version is its successor's entry in every respect but that one. A script that
writes ``Contact.Projection(...)`` or ``Magnetic.Torque()`` gets the current version; a load of a document that names
0.1.0 gets it from here, and so does an experiment whose comparisons were declared against it (script A's and A0's
reproductions of the 0.2 line, whose contact held the centre; 7j's frozen register), by ``KEPT``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from mime.gradients import GradientStatus
from mime.interactions.contact import CENTRE_SWITCHES, Contact
from mime.interactions.magnetic import MagneticForce, MagneticTorque, with_reaction
from mime.registry import (
    DESIGN,
    NUMERICAL,
    PHYSICAL,
    Confinement,
    GradientRecord,
    Plan,
    PlanContext,
    Problem,
    Read,
    SupportRow,
    Value,
    convention,
    register_interaction,
    requires,
)
from mime.vocabulary import APPLIED_WRENCH, CONFINEMENT, FLUX_DENSITY, GRADIENT, POSE, Capability


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


#: What a rigid body must offer a magnetic law: it publishes its pose and takes an applied wrench.
_RIGID_BODY = requires(Capability.RIGID, accepts=[APPLIED_WRENCH.name], publishes=[POSE.name])
#: Where 0.1.0's validity stood: one point, the body's centre of mass, for the whole body.
_AT_THE_CENTRE = ("valid while the field varies little across the body, and outside a point dipole's near field "
                  "(MIME-ANO-100); with the magnet off the centre of mass, in a field that varies in space, the torque "
                  "misses m x (G l) and the pair with the force is not conservative (MIME-ANO-137)")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class CentreMagneticTorque(MagneticTorque):
    """``magnetic.torque`` 0.1.0: ``tau = m x B``, a field that varies in space read at the body's centre of mass, a
    uniform one with no points. With its magnet off the centre of mass, in a field that varies in space, it misses
    ``m x (G l)`` and with ``magnetic.force`` 0.1.0 does net work round a closed loop of the pose (MIME-ANO-137)."""

    version = "0.1.0"
    validity = (f"a permanent moment in a field that varies in space, read at the body's centre of mass: "
                f"{_AT_THE_CENTRE}",)
    support = (
        SupportRow(None, {"body": _RIGID_BODY, "field": requires(Capability.UNIFORM, publishes=[FLUX_DENSITY.name])},
                   note="a permanent moment in a uniform field"),
        SupportRow(None, {"body": _RIGID_BODY,
                          "field": requires(Capability.PROVISION_EVALUATED, publishes=[FLUX_DENSITY.name])},
                   note="a permanent moment in a field that varies in space, read at the body's centre of mass: valid "
                        "while the field varies little across the body, and outside a point dipole's near field "
                        "(MIME-ANO-100)"),
    )

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        """A uniform field as 0.2.0 reads it (with no points, as both versions always did); any other at the body's
        centre of mass."""
        field = ctx.capabilities.get("field")
        if field is not None and Capability.UNIFORM in field.flags:
            return super().plan(ctx)
        reads = {**self.reads, "flux_density": Read("field", FLUX_DENSITY.name, at=Read("body", POSE.name))}
        return with_reaction(Plan(reads=reads, delivers=dict(self.delivers), values=dict(self.values)), ctx)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class CentreMagneticForce(MagneticForce):
    """``magnetic.force`` 0.1.0: ``F = G^T m``, the gradient read at the body's centre of mass, the force acting at the
    magnet's centroid, ``[F; l x F]`` (MIME-ANO-137)."""

    version = "0.1.0"
    validity = (f"a point dipole in a field read at the body's centre of mass: {_AT_THE_CENTRE}; a validity line "
                f"that checks the distance is a1's",)
    support = (SupportRow(None, {"body": _RIGID_BODY, "field": requires(publishes=[GRADIENT.name])},
                          note="a permanent moment, a point dipole, in a field's gradient read at the body's centre of "
                               "mass: valid while the gradient varies little across the body, and outside a point "
                               "dipole's near field (MIME-ANO-100)"),)
    reads = {"pose": Read("body", POSE.name),
             "gradient": Read("field", GRADIENT.name, at=Read("body", POSE.name))}


__all__ = ["CentreContact", "CentreMagneticForce", "CentreMagneticTorque"]
