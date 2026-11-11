"""The near-contact seam: one law supplies a near contact's resistance from the gap a rung resolves
down to, to the true gap, for every rung that hands a contact over (the MRS now; a resolved body on a grid, a joint
node's pairs and a surrogate later). The rung declares its resolved gap and evaluates itself with the contact opened
to it; the law adds the increment between the opened gap and the true one from the contact's local geometry
(:class:`NearContact`) and the body's relative motion (the increment is a resistance, linear in the twist).

**The opening** is a smooth floor of the contact's gap at the resolved gap (:func:`opening`): the identity from the
resolved gap plus :data:`LIFT_WIDTH` of it up, so that a body farther off is the rung's alone.

**The roughness cutoff.** Below the two surfaces' composite roughness (:func:`roughness_cutoff`) each gap the law reads
is floored, smoothly over :data:`ROUGHNESS_WIDTH` of it (:func:`floored`): the law's resistance stops growing without
vanishing, and a contact carries the gap. Each surface's roughness is a declared length, never fitted.

**The ambient's zero:** a rung evaluated at the opened geometry that reads an ambient flow reads it with its zero
on the opened wall (:data:`AMBIENT_ZERO`, :func:`ambient_point`), the same for every rung. Its size, measured on a
sphere in a shear flow with the exact increment (the first MIME-ANO-154 probe's arm 5): the force-free speed +7.8 % and
spin +3.1 % at a resolved gap of 0.15 radii (the true wall's convention: -9.3 % and -0.5 %), both shrinking as the
resolved gap does. No rung reads an ambient flow yet: every drag rung assumes its medium at rest (``drag.medium-flows``).

**The forms** (:data:`FORMS`), the plan's choice by the body's kind (``Geometry.near_contact``), named in the build
report: ``"sphere"``, a sphere's exact increment (``mime.nodes.hydro.near_contact._sphere``); ``"film"``, the thin
film's (``mime.nodes.hydro._film``) for every other point contact, with MIME-ANO-154 open for it."""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
from jax import Array

from mime.core import jnp
from mime.nodes.hydro._film import FILM_NODES, about, film, smooth_floor

_lax: Any = cast(Any, jax).lax

#: The opening's width, a share of the resolved gap: the opened gap is the contact's own from the resolved gap plus it
#: up, and the resolved gap from the resolved gap less it down.
LIFT_WIDTH = 0.5
#: The roughness floor's width, a share of the cutoff: a declared numerical setting.
ROUGHNESS_WIDTH = 0.5
#: The forms of the law, by the name a kind declares (``Geometry.near_contact``), the general one last.
FORMS = ("sphere", "film")
#: Where a rung evaluated at the opened geometry puts the ambient flow's zero: on the opened wall.
AMBIENT_ZERO = "opened wall"


class NearContact(NamedTuple):
    """A body's near contact with an opposing surface, in the world frame: the body's nearest point; the contact's frame
    (rows: the gap's faster and slower directions, then the normal from the opposing surface into the medium, towards
    the body; right-handed); the gap (m); the gap's principal curvatures along the frame's first two rows, ``k1 >= k2``
    (1/m, the body's less the opposing surface's); and the body's own normal curvatures along the same rows (1/m)."""

    point: Array
    frame: Array
    gap: Array
    k1: Array
    k2: Array
    b1: Array
    b2: Array


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


def floored(gap: Array, cutoff: Array) -> Array:
    """The gap the law reads: ``gap`` floored smoothly at the roughness ``cutoff``."""
    return smooth_floor(gap, cutoff, ROUGHNESS_WIDTH * cutoff)


def ambient_point(point: Array, contact: NearContact, delta: Array) -> Array:
    """Where a rung evaluated with the contact opened by ``delta`` reads the ambient flow at its ``point``: moved by the
    opening along the contact's normal, so that the ambient's zero, on the true wall, sits on the opened wall
    (:data:`AMBIENT_ZERO`); a dilated lumen opens locally as a translation along the contact's normal."""
    return point + delta * contact.frame[2]


def increment(form: str, contact: NearContact, delta: Array, cutoff: Array, resolved: Array, viscosity: Array,
              centre: Array, reach: Array, nodes: int = FILM_NODES) -> Increment:
    """The law's increment by ``form``: ``A(h) - A(h + delta)`` at the contact, each gap :func:`floored` at the
    roughness ``cutoff``, about ``centre`` in the world frame; ``resolved`` the rung's resolved gap and ``reach`` the
    body's size (the film's patch reads both)."""
    if form == "sphere":
        from mime.nodes.hydro.near_contact import _sphere

        return _sphere.increment(contact, delta, cutoff, viscosity, centre)
    if form == "film":
        return film_increment(contact, delta, cutoff, resolved, reach, viscosity, centre, nodes)
    raise ValueError(f"no near-contact form {form!r}; the forms are {FORMS}")


#: The film's larger curvature is held at least this over the body's size (its reach from its own origin, traced from
#: its shape's settings), so that a flat contact's film stays a contact's width: a floor no convex body's own larger
#: curvature falls below where it is round (a sphere's is 1 over its reach, four times this), so that it binds only at
#: a contact flatter than any round body's and never sits on the operating point of one.
CURVATURE_FLOOR = 0.25


def film_increment(contact: NearContact, delta: Array, cutoff: Array, resolved: Array, reach: Array,
                   viscosity: Array, centre: Array, nodes: int = FILM_NODES) -> Increment:
    """The film's form: ``TF(h) - TF(h + delta)``, Reynolds' equation on the gap's second-order form at the contact
    (``mime.nodes.hydro._film``), its patch's cap held above the widest gap it evaluates (the resolved gap and its
    opening's width) so that both solves share it and the film's O(1) part, which is not the body's, cancels between
    them."""
    dtype = contact.gap.dtype
    k1 = jnp.maximum(contact.k1, CURVATURE_FLOOR / reach)
    k2 = jnp.clip(contact.k2, 0.0, k1)
    cap_floor = 1.5 * (1.0 + LIFT_WIDTH) * resolved
    near = film(floored(contact.gap, cutoff), k1, k2, reach, viscosity, cap_floor, nodes)
    far = film(floored(contact.gap + delta, cutoff), k1, k2, reach, viscosity, cap_floor, nodes)
    local = near.resistance - far.resistance
    return Increment(about(local, contact.frame, contact.point - centre).astype(dtype),
                     jnp.maximum(near.residual, far.residual))


__all__ = ["AMBIENT_ZERO", "CURVATURE_FLOOR", "FORMS", "LIFT_WIDTH", "ROUGHNESS_WIDTH", "Increment", "NearContact",
           "ambient_point", "film_increment", "floored", "increment", "opening", "roughness_cutoff"]
