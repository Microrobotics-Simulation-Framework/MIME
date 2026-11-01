"""The contact family: a body against a surface. Here now: projection against a fixed structure's surface, of the
body's hull points (``Contact`` 0.2.0), or of its centre by its extent (0.1.0, kept registered:
``mime.interactions.KEPT``)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.core import switch
from mime.gradients import GradientStatus
from mime.registry import DESIGN, NUMERICAL, PHYSICAL, convention, numerical_default
from mime.registry import (
    BuildContext,
    Confinement,
    GradientRecord,
    Interaction,
    Plan,
    PlanContext,
    Problem,
    RoleKind,
    SupportRow,
    Value,
    register_interaction,
    requires,
)
from mime.units import Length, Magnitude, Quantity, m
from mime.vocabulary import CONFINEMENT, Capability

if TYPE_CHECKING:
    from mime.build import InteractionBuilder


#: What was measured of the spacing's default (the margin), its basis: the shortfall of FL-9's points against a dense
#: quadrature across its axis, and the contact's cost a step (phase 7o's pull request).
SPACING_MEASURED = ("de Jongh's FL-9 (radius 1.56 mm, length 7.47 mm) across its axis: its farthest point lies 4.3 um "
                    "from the nearest of its points at a spacing of 0.1 mm (15,327 points), 9.2 um at 0.2 mm (3,958) "
                    "and 28 um at 0.4 mm (957), against a quadrature at 1/120 of its size")

#: The switches of a body held by its centre: the hold in the solve, the band it starts in, and the projection after
#: the step with its velocity rule (0.1.0's, and 0.2.0's for a body with no hull).
CENTRE_SWITCHES = (
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
)

#: The switches of a body held by its hull's points (``RigidInertialNode``'s hull contact).
HULL_SWITCHES = (
    switch.Declaration(
        owner="contact.point",
        indicator="the gap of a group's second nearest candidate to the shrunk surface less its nearest's, m (zero "
                  "where two tie)",
        orders={"twist": -1, "pose": -1},
        route="none where the hull's kind has a support: the nearest candidate only seeds the body's own deepest point "
              "near it, which two seeds of one surface's neighbourhood reach alike, so the step jumps only where two "
              "of the body's own local deepest points tie (a screw's crest passes, a cylinder's ends), as the body's "
              "contact does; for a kind held by its samples the point moves from one sample to the next, a jump "
              "falling with the spacing"),
    switch.Declaration(
        owner="contact.gap",
        indicator="a held point's gap to the shrunk surface at the step's start, m (positive inside)",
        orders={"twist": 0, "pose": 0},
        route="none: the speed a point may close in the step is its gap's positive part"),
    switch.Declaration(
        owner="contact.impulse",
        indicator="the force a sweep's update leaves a held point, N (negative where it would pull the body to the "
                  "surface)",
        orders={"twist": 0, "pose": 0},
        route="none: a point takes a force where the step would move it into the surface faster than its gap allows, "
              "and the twists either side meet where that force is zero; the derivative is that of the active set"),
    switch.Declaration(
        owner="contact.push",
        indicator="the multiplier a sweep of the projection after the step leaves a point, kg m (negative where it "
                  "would pull the point outward)",
        orders={"pose": 0},
        route="none: the projection moves a point back to the shrunk surface where it ended past it, and the poses "
              "either side meet where it ends on it"),
)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Contact(Interaction):
    """A body in contact with a surface (``surface=tube.inner_surface``), by the written rung.

    ``projection``: the body stays on the surface's free side (the side it faces: ``tube.inner_surface`` faces the
    lumen), its every point at least ``margin`` from the surface, its orientation read. It builds no node: its record
    goes to the body's dynamics at construction, which each step holds the body inside its implicit solve and projects
    the pose back after it (``RigidInertialNode``). The hull's samples at ``spacing`` on every patch are split into
    groups; each group's sample nearest the surface seeds a contact, and where the hull's kind answers the support
    query (``Geometry.support``: a closed form for the sphere, the ellipsoid, the cylinder and the capsule, Newton's
    steps on its own surface for the screw) the contact holds the body's own deepest point near it, so the touching
    point moves continuously along the body as it turns and the hold takes no spin from a body of revolution; a kind
    with neither (a lab's mesh) is held by its samples, which act as a friction set by the spacing (the validity line
    and the flag ``held_by_samples``). At each point the outward velocity is taken away with the body's rotation
    coupled in (so a turned body touching at two points takes the couple that holds it), inelastic and frictionless,
    by projected Gauss-Seidel over the points, as a dynamic chain's joint stops are solved, the forces scaled to do no
    work at the held twist, and the solve's residual reported. Against a surface whose shape declares its region convex
    (the tube's lumen), each group's candidates are only its own convex hull's vertices, which choose as all its
    points do. Projection, not penalty; no code of its own for a body or a wall, and the solve is laid out over contact
    pairs (a hull against a wall; two bodies' hulls, by their supports, are the layout's next). A body with no hull (no
    parts, its extent declared) is held as a sphere of its extent about its centre, by 0.1.0's projection, which is
    exact for one. One per body; only against a fixed structure's surface. Its derivative is piecewise: it jumps where
    two of the body's own deepest points tie, or where a step changes the contact's branch, as its switches say.
    ``penalty`` (loads into each side's applied wrench) is reserved. ``Contact.Projection(...)`` is
    ``Contact(rung="projection", ...)`` and saves identically.

    **Admissibility only.** The contact says where the body may be, and the force that keeps it there: it never adds
    near-wall lubrication, which a drag that includes it declares (``Interaction.includes_lubrication``), and the plan
    refuses two laws on a body that include it. **Frictionless in dev1**: friction at the contact (a traction law,
    Coulomb's with static friction among it) is to be decided with the near-contact physics; its seam is the sweeps'
    rows, to which a traction law adds two along the surface at each point, bounded by its friction times the
    point's normal force.

    Version 0.2.0 holds the body's hull, where 0.1.0 held the body's centre by its extent about its own axis and never
    read its orientation, so a body that turned across a narrow tube passed through the wall. A body whose hull no
    point reaches the margin of steps as the free body does, to the last bit, in both. 0.1.0 is kept registered
    (``mime.interactions.KEPT``), so a folder saved with it loads and reproduces."""

    name = "contact"
    version = "0.2.0"
    setting_roles = {
        "rung": DESIGN, "margin": NUMERICAL,
        "rule": convention(PHYSICAL, "a frictionless contact that does not bounce: the projection removes the "
                                     "velocity into the surface, a physical model of the contact, not a solver's "
                                     "choice"),
        "spacing": numerical_default(
            "for a body with a hull, the contact's margin: the hull's points about a margin apart",
            "a stated rule, never a converged value: the body reaches past its points by no more than its farthest "
            "point lies from the nearest of them, so a spacing of the margin keeps the body within the margin's "
            "share of it", by="default_spacing", measured=SPACING_MEASURED),
    }
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(*CENTRE_SWITCHES, *HULL_SWITCHES))
    references = ("StewartTrinkle1996",)
    family = "contact"
    builds = {"projection": ()}
    roles = {"body": RoleKind.BODY, "surface": RoleKind.SURFACE}
    rungs = ("projection", "penalty")
    reserved_rungs = {"penalty": "penalty contact, delivering loads into each side's applied wrench, comes later"}
    support = (SupportRow("projection", {"body": requires(Capability.PROJECTS_CONFINEMENT, accepts=[CONFINEMENT.name])},
                          note="the body's dynamics holds a confinement in its own update"),)
    validity = (
        "projection: a rigid body against a fixed surface, held by its hull's points (its kind's own samples at the "
        "spacing), each at least the margin from the surface: the body itself reaches past its points by no more than "
        "its farthest point lies from the nearest of them (the spacing's measured basis states it for script A's "
        "screw), so the margin it keeps is the margin less that",
        "projection: the margin is kept to what a fixed count of sweeps leaves: two points whose rows are coupled at "
        "c converge by c squared a sweep, and a body pressed into both walls with short levers is held to that, the "
        "pose projected back after the step taking the rest",
        "projection: inelastic and frictionless (no traction along the surface, no static friction, no adhesion): "
        "friction at the contact is to be decided with the near-contact physics, and its seam is left in the sweeps",
        "projection: admissibility only: it adds no near-wall lubrication, which a drag that includes it declares, "
        "and the plan refuses a second law on the body that includes it",
        "projection: a rigid surface and a rigid body: no compliance of the wall or the body",
        "projection: each step it holds one point of each of the hull's groups, the one nearest the surface, by a "
        "fixed count of projected Gauss-Seidel sweeps, and projects the pose back after it; a point of a group that "
        "reaches the surface within the step but is not its group's nearest at the step's start is taken by the "
        "projection, not by the solve",
        "projection: a body with no hull is held as a sphere of its declared extent about its centre",
    )
    rung: str | None = None
    margin: Annotated[Quantity, Length]
    rule: str = "remove_outward"
    spacing: Annotated[Quantity | None, Length] = None

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        margin = settings["margin"]
        if not (isinstance(margin, float) and margin >= 0.0):
            found.append(f"margin is {margin!r} m, not a length from 0 on")
        spacing = settings.get("spacing")
        if spacing is not None and not (isinstance(spacing, float) and math.isfinite(spacing) and spacing > 0.0):
            found.append(f"spacing is {spacing!r} m, not a length above zero: the hull's points lie about spacing "
                         f"apart, so write spacing=mm(...)")
        return found

    @classmethod
    def Projection(cls, **settings: Any) -> Contact:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="projection", **settings)

    def default_spacing(self, ctx: PlanContext) -> Quantity | None:
        """The spacing of the hull's points when none is written, for a body with a hull: the margin, or a
        thousandth of the hull's size where the margin is zero."""
        hull = ctx.geometry(of="body")
        if hull is None:
            return None
        margin = ctx.scalar("margin")
        return m(margin if margin > 0.0 else 1e-3 * float(hull.kind.enclosing_radius(hull.settings)))

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        if ctx.geometry(of="body") is None:
            unread = () if self.spacing is None else (Problem(
                "confinement.no-hull", f"spacing= is the spacing of a hull's points, and {ctx.bindings['body']} has no "
                                       f"hull: it is held as a sphere of its extent", "remove spacing=, or give the "
                                       "body a part outside every other: parts={\"shell\": Part(Sphere(...))}"),)
            return Plan(nodes=(), values={"extent": Value("body", "extent")}, problems=unread,
                        confines=(Confinement("body", CONFINEMENT.name, "surface", ctx.scalar("margin"), self.rule),))
        return Plan(nodes=(), confines=(Confinement("body", CONFINEMENT.name, "surface", ctx.scalar("margin"),
                                                    self.rule, extent=None, spacing=ctx.scalar("spacing")),))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        return None


__all__ = ["CENTRE_SWITCHES", "HULL_SWITCHES", "Contact"]
