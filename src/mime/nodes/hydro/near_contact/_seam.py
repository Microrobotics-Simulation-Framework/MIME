"""The near-contact seam: one law supplies a near contact's resistance from the gap a rung resolves
down to, to the true gap, for every rung that hands a contact over (the MRS now; a resolved body on a grid, a joint
node's pairs and a surrogate later). The rung declares its resolved gap and evaluates itself with the contact opened
to it; the law adds the increment between the opened gap and the true one from the contact's local geometry
(:class:`NearContact`) and the body's relative motion (the increment is a resistance, linear in the twist).

**Several contacts.** A body may touch one opposing surface at several points (a screw's crests on a tube's floor, a body
wedged across a lumen): each is a :class:`NearContact`, held together as :class:`NearContacts`, each with its share of
the sum (:func:`shares`: one inside the opening's band, falling smoothly to none past twice it, and shared among
contacts that stand at one point). One surface is opened once, by the opening of its contacts' soft least gap
(:func:`surface_opening`), so that each contact sits at or above the resolved gap; each contact's increment spans its
own true gap to its opened gap, and the increments are summed about the body's centre (:func:`summed`).

**The opening** is a smooth floor of the contact's gap at the resolved gap (:func:`opening`): the identity from the
resolved gap plus :data:`LIFT_WIDTH` of it up, so that a body farther off is the rung's alone.

**The roughness cutoff.** Below the two surfaces' composite roughness (:func:`roughness_cutoff`) each gap the law reads
is floored, smoothly over :data:`ROUGHNESS_WIDTH` of it (:func:`floored`): the law's resistance stops growing without
vanishing, and a contact carries the gap. Each surface's roughness is a declared length, never fitted.

**The ambient's zero:** a rung evaluated at the opened geometry that reads an ambient flow reads it with its zero
on the opened wall (:data:`AMBIENT_ZERO`, :func:`ambient_point`), the same for every rung: a plane's points moved by
the opening along its normal, a dilated lumen's scaled about its axis, so that the whole opened wall, the far side too,
reads the true wall's zero. Its size, measured on a sphere in a shear flow with the exact increment (the first
MIME-ANO-154 probe's arm 5): the force-free speed +7.8 % and spin +3.1 % at a resolved gap of 0.15 radii (the true
wall's convention: -9.3 % and -0.5 %), both shrinking as the resolved gap does. No rung reads an ambient flow yet:
every drag rung assumes its medium at rest (``drag.medium-flows``).

**The class of a contact** (:func:`classified`, :data:`CLASSES`), from its curvatures and its extent within the band:
a point contact, which the law's forms serve; a line contact, the body's own surface flat within the band along the
gap's slower direction over its whole extent there (a cylinder's or a capsule's side); a ring, the footprint reaching round a lumen (a
body nearly filling it). The law has no form for a line or a ring yet (W2a's next pull request): a rung that meets one
says so.

**The forms** (:data:`FORMS`), the plan's choice by the body's kind (``Geometry.near_contact``), named in the build
report: ``"sphere"``, a sphere's exact increment (``mime.nodes.hydro.near_contact._sphere``); ``"elliptic"``, the
elliptic point contact's singular terms from its curvatures alone (``mime.nodes.hydro.near_contact._elliptic``), every
other point contact's, with MIME-ANO-154 open for it. The thin film (``mime.nodes.hydro._film``) is no form: it is
retired as a per-step increment and kept as the elliptic form's cross-check."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple, cast

import jax
from jax import Array

from mime.core import jnp
from mime.nodes.hydro._film import smooth_floor

_tree_util: Any = cast(Any, jax).tree_util
_vmap: Any = cast(Any, jax).vmap

#: The opening's width, a share of the resolved gap: the opened gap is the contact's own from the resolved gap plus it
#: up, and the resolved gap from the resolved gap less it down.
LIFT_WIDTH = 0.5
#: The roughness floor's width, a share of the cutoff: a declared numerical setting.
ROUGHNESS_WIDTH = 0.5
#: The forms of the law, by the name a kind declares (``Geometry.near_contact``), the general one last.
FORMS = ("sphere", "elliptic")
#: The classes of a contact (:func:`classified`): the law's forms serve the first.
CLASSES = ("point", "line", "ring")
#: Where a rung evaluated at the opened geometry puts the ambient flow's zero: on the opened wall.
AMBIENT_ZERO = "opened wall"
#: The elliptic form's gap curvatures are held at least this over the body's size (its reach from its own origin), so
#: that a flat contact's terms stay finite: a floor no convex body's own larger curvature falls below where it is round
#: (a sphere's is 1 over its reach, four times this). A contact that flat is a line or a face, which the rung classes
#: and flags.
CURVATURE_FLOOR = 0.25
#: The soft least gap of several contacts on one surface is soft over this share of the resolved gap (within this
#: share times ``ln 2`` of the least where two contacts tie): the surface's opening then turns smoothly from following
#: one contact to the other where their gaps cross. A declared numerical setting.
SOFT_SHARE = 0.1
#: Two contacts closer than this share of the body's size share one place in the sum (smoothly: wholly at one point,
#: apart from this distance up): two searches reaching one point count it once. A declaration, far below the separation
#: of two of a screw's crest passes (2.1 mm on FL-9, about half its size).
MERGE_SHARE = 0.05
#: The share of a quarter turn of the lumen a contact's footprint round it may reach before it is a ring.
RING_SHARE = 1.0


class NearContact(NamedTuple):
    """A body's near contact with an opposing surface, in the world frame: the body's nearest point; the contact's frame
    (rows: the gap's faster and slower directions, then the normal from the opposing surface into the medium, towards
    the body; right-handed); the gap (m); the gap's principal curvatures along the frame's first two rows, ``k1 >= k2``
    (1/m, the body's less the opposing surface's); and the body's own curvature in the same frame (1/m): along the first
    row, along the second, and their cross term ``b12`` (zero where the body's principal directions are the gap's, as a
    sphere's always are)."""

    point: Array
    frame: Array
    gap: Array
    k1: Array
    k2: Array
    b1: Array
    b2: Array
    b12: Any = 0.0


class NearContacts(NamedTuple):
    """A body's near contacts with one opposing surface: each field of ``contacts`` with a leading axis of one slot a
    contact; each slot's ``merged`` share (one, or less where other slots found the same point: :func:`merged`); its
    ``kind`` (:data:`CLASSES`' index); and the opposing surface's curvature along each contact's slower direction (1/m,
    positive where it curves towards the body, as a lumen does) and the body's half ``extent`` along it (m), which the
    class reads."""

    contacts: NearContact
    merged: Array
    kind: Array
    wall_along: Array
    extent: Array


class Increment(NamedTuple):
    """The law's increment, a 6 x 6 resistance about the body's centre in the world frame, and the largest residual
    of the solves it took (zero for a closed form)."""

    resistance: Array
    residual: Array


def roughness_cutoff(*roughness: Any) -> Any:
    """Where the near-contact law hands the gap over to contact: the composite roughness of the surfaces, the root of
    the sum of their squares (m)."""
    return sum(r * r for r in roughness) ** 0.5


def opening(gap: Array, resolved: Array) -> Array:
    """How far the contact is opened for a rung that resolves gaps down to ``resolved``: the smooth floor of ``gap`` at
    ``resolved`` less ``gap`` (zero from ``resolved (1 + LIFT_WIDTH)`` up)."""
    return smooth_floor(gap, resolved, LIFT_WIDTH * resolved) - gap


def surface_opening(gaps: Array, merged: Array, resolved: Array) -> Array:
    """How far one opposing surface is opened for its contacts at ``gaps`` (each counted by its ``merged`` share): the
    :func:`opening` of their soft least gap, ``-soft ln sum merged exp(-gap / soft)`` with ``soft`` :data:`SOFT_SHARE` of
    ``resolved``, written about the least gap. A single contact's opening is its own, bit for bit; contacts that are one
    point count once."""
    if gaps.shape[0] == 1:
        return opening(gaps[0], resolved)
    low = jnp.min(gaps)
    soft = SOFT_SHARE * resolved
    least = low - soft * jnp.log(jnp.sum(merged * jnp.exp(-(gaps - low) / soft)))
    return opening(least, resolved)


def merged(points: Array, size: Array) -> Array:
    """Each slot's share where several searches may find one point (``points``, ``K x 3``): one over one plus the
    nearness of every other slot's point, its nearness one at the same point and none from :data:`MERGE_SHARE` of the
    body's ``size`` apart (a smoothstep between), so that ``k`` slots at one point share it equally and the sum is
    continuous as two points part."""
    if points.shape[0] == 1:
        return jnp.ones((1,), dtype=points.dtype)
    apart = jnp.sqrt(jnp.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1) + (1e-12 * size) ** 2)
    t = jnp.clip(apart / (MERGE_SHARE * size), 0.0, 1.0)
    return 1.0 / jnp.sum(1.0 - t * t * (3.0 - 2.0 * t), axis=-1)


def in_band(gap: Array, resolved: Array) -> Array:
    """A contact's share by its gap: one within the opening's band (``resolved (1 + LIFT_WIDTH)``), falling by a
    smoothstep to none at twice it, so that a contact entering the search's reach adds nothing at first."""
    t = jnp.clip(gap / ((1.0 + LIFT_WIDTH) * resolved) - 1.0, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def shares(found: NearContacts, resolved: Array) -> Array:
    """Each slot's share of the sum: its :func:`merged` share times its :func:`in_band` share (one for a single slot
    within the band, bit for bit)."""
    return found.merged * in_band(found.contacts.gap, resolved)


def classified(k2: Array, b2: Array, extent: Array, wall_along: Array, resolved: Array) -> Array:
    """A contact's class (:data:`CLASSES`' index) at the band's edge ``H = resolved (1 + LIFT_WIDTH)``, from the gap's
    slower curvature ``k2``, the body's own curvature ``b2`` and half ``extent`` along that direction (m) and the opposing
    surface's curvature along it (``wall_along``, 1/m): the gap's footprint, ``sqrt(2 H / k2)``, reaching round a quarter
    turn of a lumen (:data:`RING_SHARE`) is a ring; the body's own surface within ``H`` of its tangent over its whole
    extent along the direction (``b2 extent^2 < 2 H``: a cylinder's or a capsule's side) is a line; else a point. A
    report: no value the law gives reads it."""
    footprint_squared = 2.0 * (1.0 + LIFT_WIDTH) * resolved
    ring = (wall_along > 0.0) & (footprint_squared * wall_along ** 2 > jnp.maximum(k2, 0.0)
                                 * (RING_SHARE * 0.5 * math.pi) ** 2)
    line = footprint_squared > jnp.maximum(b2, 0.0) * extent ** 2
    return jnp.where(ring, 2, jnp.where(line, 1, 0))


def floored(gap: Array, cutoff: Array) -> Array:
    """The gap the law reads: ``gap`` floored smoothly at the roughness ``cutoff``."""
    return smooth_floor(gap, cutoff, ROUGHNESS_WIDTH * cutoff)


def ambient_point(point: Array, contact: NearContact, delta: Array, lumen: Mapping[str, Any] | None = None) -> Array:
    """Where a rung evaluated with the contact's opposing surface opened by ``delta`` reads the ambient flow at its
    ``point``, so that the ambient's zero, on the true wall, sits on the whole opened wall (:data:`AMBIENT_ZERO`): above a
    plane, moved by the opening along the contact's normal; in a lumen dilated by it (``lumen``: its ``centre``, unit
    ``axis`` and ``radius``, the true wall's), its distance from the axis scaled by ``radius / (radius + delta)``, which
    takes the dilated wall to the true one all round, the far side included."""
    if lumen is None:
        return point + delta * contact.frame[2]
    centre, axis = jnp.asarray(lumen["centre"]), jnp.asarray(lumen["axis"])
    offset = point - centre
    across = offset - jnp.dot(offset, axis) * axis
    return point - across * (delta / (lumen["radius"] + delta))


def increment(form: str, contact: NearContact, delta: Array, cutoff: Array, viscosity: Array, centre: Array,
              reach: Array) -> Increment:
    """The law's increment by ``form``: ``A(h) - A(h + delta)`` at the contact, each gap :func:`floored` at the
    roughness ``cutoff``, about ``centre`` in the world frame; ``reach`` the body's size (the elliptic form's curvature
    floor reads it)."""
    if form == "sphere":
        from mime.nodes.hydro.near_contact import _sphere

        return _sphere.increment(contact, delta, cutoff, viscosity, centre)
    if form == "elliptic":
        from mime.nodes.hydro.near_contact import _elliptic

        return _elliptic.increment(contact, delta, cutoff, viscosity, centre, reach)
    raise ValueError(f"no near-contact form {form!r}; the forms are {FORMS}")


def summed(form: str, found: NearContacts, share: Array, delta: Array, cutoff: Array, viscosity: Array,
           centre: Array, reach: Array) -> Increment:
    """The increments of a body's contacts with one opposing surface opened by ``delta``, each by its ``share``,
    summed about ``centre``: a single slot's increment itself, bit for bit where its share is one."""
    if share.shape[0] == 1:
        one = increment(form, cast(NearContact, _tree_util.tree_map(lambda v: v[0], found.contacts)), delta, cutoff,
                        viscosity, centre, reach)
        return Increment(share[0] * one.resistance, one.residual)
    each = _vmap(lambda contact: increment(form, contact, delta, cutoff, viscosity, centre, reach))(found.contacts)
    return Increment(jnp.einsum("k,kij->ij", share, each.resistance), jnp.max(each.residual))


__all__ = ["AMBIENT_ZERO", "CLASSES", "CURVATURE_FLOOR", "FORMS", "LIFT_WIDTH", "MERGE_SHARE", "RING_SHARE",
           "ROUGHNESS_WIDTH", "SOFT_SHARE", "Increment", "NearContact", "NearContacts", "ambient_point", "classified",
           "floored", "in_band", "increment", "merged", "opening", "roughness_cutoff", "shares", "summed",
           "surface_opening"]
