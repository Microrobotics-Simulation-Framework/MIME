"""Lubrication's hand-over on the MRS rung (drag 0.5.0's ``near_contact``): the MRS's side of the near-contact seam
(``mime.nodes.hydro.near_contact``), the near-field model's completion in the gap, keeping the MRS's near-wall accuracy
at a coarse spacing, which a fine enough MRS would reach alone at far greater cost; not physics beyond Stokes flow.

**The lift form.** The resistance is the coarse MRS evaluated once with the wall opened so that each of the body's near
contacts stands at or above the hand-over gap ``h_m`` (the gap the rung declares it resolves down to), plus the sum of
the near-contact law's increments:

    R = R_MRS(the wall opened by delta) + sum_c w_c [A_c(h_c) - A_c(h_c + delta)],

``h_c`` each contact's gap and ``w_c`` its share (one within the opening's band, none past twice it, shared among
searches that found one point), ``delta`` the seam's opening of the contacts' soft least gap at ``h_m``: zero from
``h_m`` plus its width up, so that a body farther off is the coarse MRS's alone, step for step, and costs nothing more.
The wall is moved, never the body (``WallCorrection.opened``: the plane translated, the tube dilated), once for all its
contacts, so a contact the opening reaches though its own gap is wider takes the increment over its own span. The MRS
is never evaluated at a gap below ``h_m``. The law's form is the plan's, by the body's kind: a sphere's exact increment,
or the elliptic point contact's.

**The hand-over gap** is :data:`HANDOVER_SPACINGS` of the force spacing (the finest graded spacing where the sampling
is graded), measured: at it the coarse MRS's worst entry on a sphere above a plane, the normal, read 5.6 to 6.1 % off
at blob 0.1, 3.5 to 4.2 % at 0.05 and 2.6 to 3.4 % at 0.025, with the image at the force points. The elliptic form was
measured within 7 to 22 % on the worst rolling rate of spheroids at ``h_m`` 0.15 of the smaller radius (MIME-ANO-154's
second probe), and is not measured past it: beyond :data:`WINDOW` of a contact's smaller radius of curvature a step
raises ``handover_window``. The sphere's form is exact at every gap, and has no window.

**The roughness cutoff** is the seam's (``roughness_cutoff``): the roughness of each surface is a declared length,
never fitted (:data:`DEFAULT_ROUGHNESS` where none is declared).

**Who owns the coupling.** The near-wall resistance (the coarse MRS plus the law's increments) owns every hydrodynamic
coupling of rotation and translation at every gap down to the cutoff, and below it the law's coupling stays at its
floored value; a contact owns only the normal reaction, traction and rolling resistance and adds no hydrodynamic term.
The increments' own coupling block is reported (``contact_coupling``), so that a double count can be tested for.

**This form covers** point contacts on a smooth surface, as many as the body's groups find (``_search``); a line or a
ring contact is classed and flagged at every step it is reached (``near_contact_line``, ``near_contact_ring``), and the
plan refuses a body that can make one (the law has no form for either yet)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.nodes.hydro._film import tangent_frame
from mime.nodes.hydro.near_contact import (
    CURVATURE_FLOOR,
    LIFT_WIDTH,
    RING_SHARE,
    ROUGHNESS_WIDTH,
    Increment,
    NearContacts,
    roughness_cutoff,
    shares,
    summed,
    surface_opening,
)
from mime.nodes.hydro.near_contact._search import (
    GROUPS,
    RADIUS_FLOOR,
    Search,
    near_contacts,
    relative_curvature,
    search_for,
)
from mime.registry import Geometry

_lax: Any = cast(Any, jax).lax

#: The hand-over gap in force spacings (the finest graded spacing where graded): a labelled default, measured on a sphere
#: above a plane through the drag's node (W2a's first step; the module's docstring).
HANDOVER_SPACINGS = 0.75
#: Each surface's roughness where none is declared (m): a declaration, never fitted (sources span about 0.03 to a few
#: micrometres).
DEFAULT_ROUGHNESS = 1e-6
#: The hand-over gap over the contact's smaller radius of curvature beyond which the form's measured reach (about 7 %)
#: is left (``handover_window``).
WINDOW = 0.15
#: The share of a lumen's diameter the body's width across it (along the contact's normal) may fill before the opening
#: by dilation moves the far side past the gates' line (``handover_far_side``): dilating the tube to open the near
#: contact opens the far side as well, and the coarse MRS's worst entry (the axial) at the opened geometry, against the
#: near contact opened with the far side kept, read -6.3 % at a/R 0.5 and -8.9 % at 0.55 for a sphere at the deepest
#: opening the window allows (s/a 0.2, h_m 0.15 a), -18 % at 0.67 and -37 % at 0.8 [measured: W2a's fix round].
FAR_SIDE_FILL = 0.5
#: The largest sphere-to-tube radius ratio the sphere's form in a tube was gated at (0.22 to 0.34, against an independent
#: tube MFS): above it a step within the opening's band raises ``handover_tube_ratio``. Along the axis the rolling
#: misses by -12 % at 0.34 and was measured -18 % at 0.40 and -25 % at 0.46 (the PR's review, its tube MFS resolved).
TESTED_TUBE_RATIO = 0.34
#: The share past TESTED_TUBE_RATIO a ratio may read and still be the gated one: the sphere's radius is read from its
#: curvature at the contact, regularised by RADIUS_FLOOR of its size, so a sphere at exactly 0.34 reads 1e-6 more.
RATIO_ROUNDING = 1e-4
#: What the sphere's form is measured to do, by where it runs (a 1 mm sphere, blob 0.05, float64, through the node;
#: tests/oracles/test_near_contact.py and the fix round's probes): at a uniform force spacing above a plane, in a tube
#: by the sphere-to-tube ratio, and on a sampling graded towards the contact (refused, drag.near-contact-graded).
UNIFORM_SPHERE_TEXT = ("on a sphere above a plane at uniform force spacings of 0.15 to 0.6 radii, the normal +2.1 to "
                       "+3.6 % off Brenner's series at and below the hand-over gap, every other entry within 1.8 % and "
                       "the rolling rate within 1.4 % of exact")
TUBE_RATIOS_TEXT = ("against an independent tube MFS at sphere-to-tube ratios of 0.22, 0.28 and 0.34, at gaps of 0.1 to "
                    "0.0075 radii: the rolling round the circumference within 3.3 % of the plane's rate, every diagonal "
                    "entry within 4.7 % and the couplings within 10 % of the plane's coupling (along the axis +16.5 % "
                    "at 0.34); the rolling along the axis misses by -3.6, -7.0 and -11.9 % of the plane's rate at the "
                    "family's spacing and by -2.7, -5.2 and -8.9 % at a spacing of 0.2 radii, a stated limit, which a "
                    "finer spacing lowers; not gated above 0.34, where the same method read -18 % at 0.40 and -25 % at "
                    "0.46")
#: What the elliptic form was measured to do (MIME-ANO-154's second probe: prolate spheroids of aspect 2 and 4 above a
#: plane, lying and tilted 30 degrees, against a graded MFS, at hand-over gaps of 0.15 and 0.075 of the minor
#: semi-axis), which a non-sphere's build line states: the published coefficients, which a film extraction at 257 nodes
#: matched within 1 %.
ELLIPTIC_TEXT = ("on prolate spheroids above a plane the rolling rate's worst miss 7 to 22 % at a hand-over gap of 0.15 "
                 "of the minor semi-axis (the thin film's form missed up to 238 % there) and 4 to 15 % at 0.075, the "
                 "torque about the wall's axis +12 to +33 % and the tilted bodies 1.4 to 1.7 times the film's miss; "
                 "in a tube a non-sphere's form is not measured against an independent reference")
GRADED_SPHERE_TEXT = ("graded towards its contact at a force spacing of 0.2 radii, a sphere's normal read +9.5 % and "
                      "+19.7 % off Brenner's series at hand-over gaps of 0.075 and 0.0375 radii, against +2.1 to +3.6 % "
                      "at uniform spacings of 0.15 to 0.6 radii")


def handover_gap(spacing: float) -> float:
    """The hand-over gap at the local force ``spacing`` (m)."""
    return HANDOVER_SPACINGS * spacing


def least_radius(kind: Geometry, settings: Mapping[str, Any], spacing: float, patches: Sequence[str]) -> float:
    """The body's least radius of curvature over its surface (m), as the step reads a contact's: the smaller of the two
    radii the support's derivative gives along each sample's outward normal (``kind.support``), regularised by
    :data:`RADIUS_FLOOR` of its size, the least over the kind's samples at ``spacing``; near zero at a sharp edge. At
    the build, in float64: the radius the hand-over's window is judged against (:data:`WINDOW`), for any contact the
    body can make whatever its pose."""
    key = (kind.name, kind.version, repr(sorted(settings.items())), float(spacing), tuple(patches))
    found = _least_radii.get(key)
    if found is None:
        found = _least_radius(kind, settings, spacing, patches)
        _least_radii[key] = found
    return found


_least_radii: dict[tuple[Any, ...], float] = {}


def _least_radius(kind: Geometry, settings: Mapping[str, Any], spacing: float, patches: Sequence[str]) -> float:
    with cast(Any, jax).enable_x64(True):
        sampling = kind.sampling(settings, spacing, tuple(patches))
        parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                      for name, value in settings.items()}
        samples = sampling.samples(kind, parameters)
        points = jnp.asarray(samples.points, dtype=jnp.float64)
        normals = jnp.asarray(samples.normals, dtype=jnp.float64)
        jacobian = jnp.asarray(kind.support(parameters, normals, points).jacobian, dtype=jnp.float64)
        across = jax.vmap(tangent_frame)(normals)
        radii = jnp.linalg.eigvalsh(across @ (0.5 * (jacobian + jnp.swapaxes(jacobian, -1, -2)))
                                    @ jnp.swapaxes(across, -1, -2))
        size = float(jnp.max(jnp.sqrt(jnp.sum(points * points, axis=-1))))
        return max(float(jnp.min(radii)), 0.0) + RADIUS_FLOOR * size


class Capable(NamedTuple):
    """What a body can make against a wall at a hand-over gap, judged at the build over its samples, whatever its pose:
    a line contact (``line``: the least, over its samples, of its smaller curvature times its half extent squared along
    that direction, m, against twice the band's edge) and a ring in a lumen (``ring``: its least half-width across any
    direction, m, against the lumen's radius), each with the text the build reports, or ``None``."""

    line: str | None
    ring: str | None


def capable(kind: Geometry, settings: Mapping[str, Any], spacing: float, patches: Sequence[str], handover: float,
            lumen: float | None) -> Capable:
    """Whether the body can make a line contact against any wall, or a ring in a lumen of radius ``lumen`` (m), at the
    hand-over gap ``handover`` (m), by the seam's classes (``classified``) at the band's edge ``H``: a line where some
    sample's surface stays within ``H`` of its tangent over the body's whole extent along its flatter direction (a
    cylinder's or a capsule's side); a ring where a sphere of the body's least half-width would class as one in the
    lumen, so that every pose of the body does. At the build, in float64."""
    key = (kind.name, kind.version, repr(sorted(settings.items())), float(spacing), tuple(patches), float(handover),
           lumen)
    found = _capabilities.get(key)
    if found is None:
        found = _capable(kind, settings, spacing, patches, handover, lumen)
        _capabilities[key] = found
    return found


_capabilities: dict[tuple[Any, ...], Capable] = {}


def _capable(kind: Geometry, settings: Mapping[str, Any], spacing: float, patches: Sequence[str], handover: float,
             lumen: float | None) -> Capable:
    band = (1.0 + LIFT_WIDTH) * handover
    with cast(Any, jax).enable_x64(True):
        sampling = kind.sampling(settings, spacing, tuple(patches))
        parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                      for name, value in settings.items()}
        samples = sampling.samples(kind, parameters)
        points = jnp.asarray(samples.points, dtype=jnp.float64)
        normals = jnp.asarray(samples.normals, dtype=jnp.float64)
        size = float(jnp.max(jnp.sqrt(jnp.sum(points * points, axis=-1))))
        jacobian = jnp.asarray(kind.support(parameters, normals, points).jacobian, dtype=jnp.float64)
        across = jax.vmap(tangent_frame)(normals)
        radii, turns = jnp.linalg.eigh(across @ (0.5 * (jacobian + jnp.swapaxes(jacobian, -1, -2)))
                                       @ jnp.swapaxes(across, -1, -2))
        # the flatter direction: the larger radius's, in the kind's frame
        flatter = jnp.einsum("ni,nij->nj", turns[:, :, 1], across)
        reach = jax.vmap(lambda d: kind.reach(parameters, d))
        extent = 0.5 * (reach(flatter) + reach(-flatter))
        curvature = 1.0 / (jnp.maximum(radii[:, 1], 0.0) + RADIUS_FLOOR * size)
        line = float(jnp.min(curvature * extent ** 2))
        half = float(jnp.min(0.5 * (reach(normals) + reach(-normals))))
    line_text = (f"its surface lies within the hand-over's band ({band * 1e3:.4g} mm) of its tangent over its whole "
                 f"extent along a flatter direction (the least curvature times half-extent squared, {line * 1e3:.4g} "
                 f"mm, below twice the band): it can touch a wall along a line, which the near-contact law has no form "
                 f"for yet") if 2.0 * band > line else None
    ring_text = None
    if lumen is not None and half < lumen and 2.0 * band / lumen ** 2 > \
            (1.0 / half - 1.0 / lumen) * (RING_SHARE * 0.5 * math.pi) ** 2:
        ring_text = (f"its least half-width, {half * 1e3:.4g} mm, nearly fills the lumen ({lumen * 1e3:.4g} mm): at "
                     f"the hand-over gap {handover * 1e3:.4g} mm its gap stays within the band round a quarter turn "
                     f"of the lumen in any pose, a ring contact, which the near-contact law has no form for yet")
    return Capable(line_text, ring_text)


def window_text(handover: float, radius: float) -> str:
    """Where the hand-over gap ``handover`` lies against the film's window over the least ``radius`` of curvature."""
    return (f"the hand-over gap {handover * 1e3:.4g} mm is {handover / radius:.4g} of the body's least radius of "
            f"curvature, {radius * 1e3:.4g} mm")


def slots_for(form: str) -> int:
    """The contacts a body's search can find by the law's ``form``: one for a sphere, whose deepest point against a
    plane or a lumen it does not fill is its only contact, :data:`GROUPS` for any other kind."""
    return 1 if form == "sphere" else GROUPS


class Handover(NamedTuple):
    """A step's hand-over: the wall's constants opened, the law's summed increment (zero where no contact is within the
    lift), the contacts, the least of their gaps, the hand-over gap, the opening and the cutoff, and the step's flags as
    0 or 1, each raised by a contact within reach of the opening: ``window`` (the elliptic form's hand-over gap past
    :data:`WINDOW` of a contact's smaller radius), ``line`` and ``ring`` (a contact so classed, for which the law has no
    form), ``far_side`` (a lumen opened by dilation about a body filling more than :data:`FAR_SIDE_FILL` of its
    diameter) and ``tube_ratio`` (the sphere's form in a tube past :data:`TESTED_TUBE_RATIO`)."""

    opened: dict[str, Any]
    increment: Array
    contacts: NearContacts
    gap: Array
    handover: Array
    delta: Array
    cutoff: Array
    window: Array
    line: Array
    ring: Array
    far_side: Array
    tube_ratio: Array


def handed_over(search: Search, kind: Geometry, parameters: Mapping[str, Any], points: Array, pose: Array,
                wall: Any, constants: Mapping[str, Any], leaves: Mapping[str, Any], viscosity: Array,
                form: str = "elliptic") -> Handover:
    """The step's hand-over: the body's near contacts against the wall at its constants, the wall opened for them to the
    hand-over gap, and the near-contact law's increments by ``form``, each by its share, summed, where the wall was
    opened (a ``cond``: where it was not, the increment is zero and its branch is not evaluated; the two agree there in
    value and derivative, the opening being zero)."""
    dtype = pose.dtype
    region = wall.region(constants)[1]
    reach = jnp.max(jnp.sqrt(jnp.sum(points * points, axis=-1)))
    handover = HANDOVER_SPACINGS * jnp.asarray(leaves["handover_spacing"], dtype=dtype)
    found = near_contacts(search, kind, parameters, points, pose, region, leaves, reach, handover)
    contacts = found.contacts
    gaps = contacts.contacts.gap
    delta = surface_opening(gaps, contacts.merged, handover)
    cutoff = roughness_cutoff(jnp.asarray(leaves["roughness_body"], dtype=dtype),
                              jnp.asarray(leaves["roughness_wall"], dtype=dtype))
    share = shares(contacts, handover)

    def opened() -> Increment:
        return summed(form, contacts, share, delta, cutoff, viscosity, pose[:3], reach)

    def none() -> Increment:
        return Increment(jnp.zeros((6, 6), dtype=dtype), jnp.zeros((), dtype=dtype))

    summed_up = _lax.cond(delta > 0.0, opened, none)
    reached = _lax.stop_gradient(share) > 0.0
    lumen = region.get("radius")
    far_side = jnp.asarray(False) if lumen is None else \
        jnp.any(reached & (found.width > FAR_SIDE_FILL * 2.0 * lumen))
    # the sphere's radius is its own curvature at the contact, as its form reads it
    tube_ratio = jnp.asarray(False) if lumen is None or form != "sphere" else jnp.any(
        reached & (2.0 / (contacts.contacts.b1 + contacts.contacts.b2)
                   > TESTED_TUBE_RATIO * (1.0 + RATIO_ROUNDING) * lumen))
    window = jnp.asarray(False) if form == "sphere" else jnp.any(
        reached & (handover * jnp.maximum(contacts.contacts.k1, CURVATURE_FLOOR / reach) > WINDOW))
    line, ring = jnp.any(reached & (contacts.kind == 1)), jnp.any(reached & (contacts.kind == 2))
    flags = _lax.stop_gradient(jnp.stack([window, line, ring, far_side, tube_ratio]) & (delta > 0.0)).astype(dtype)
    return Handover(wall.opened(constants, delta), summed_up.resistance, contacts, jnp.min(gaps), handover, delta,
                    cutoff, flags[0], flags[1], flags[2], flags[3], flags[4])


__all__ = ["CURVATURE_FLOOR", "DEFAULT_ROUGHNESS", "ELLIPTIC_TEXT", "FAR_SIDE_FILL", "GRADED_SPHERE_TEXT", "GROUPS",
           "HANDOVER_SPACINGS", "LIFT_WIDTH", "RADIUS_FLOOR", "RATIO_ROUNDING", "ROUGHNESS_WIDTH", "TESTED_TUBE_RATIO",
           "TUBE_RATIOS_TEXT", "UNIFORM_SPHERE_TEXT", "WINDOW", "Capable", "Handover", "Increment", "Search", "capable", "handed_over",
           "handover_gap", "least_radius", "relative_curvature", "search_for", "slots_for", "window_text"]
