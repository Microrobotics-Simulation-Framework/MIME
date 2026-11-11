"""The elliptic point contact's form of the near-contact law: the singular terms of a smooth body's resistance at a point
contact, from the contact's curvatures alone, the general point form (every kind without a closed form of its own).

**The source.** Cox (1974) and Claeys and Brady (1989), as compiled by Janoschek, Harting and Toschi (2013, arXiv
1308.6482, Appendix A): the elements of the resistance that diverge as the gap closes and that the local curvatures
alone determine, Eqs. A.1, A.2, A.4 to A.6, A.8 to A.10 and A.13 to A.16, for a body whose own principal directions
are turned by any angle (their chi) from the gap's. Eqs. A.3, A.7, A.11 and A.12 (the normal force a tangential motion
or a tangential rotation gives) need the surfaces' cubic terms as well, which a contact's curvatures do not hold, and
are left out: they vanish where the body is symmetric about the contact (a sphere, a spheroid's pole or equator) and
the compilation notes a sign error in their first derivation. The spin about the normal has no singular term there.

**Written covariantly.** With ``B`` the body's curvature and ``G = diag(k1, k2)`` the gap's, both in the gap's principal
frame (rows ``e1``, ``e2``), ``s = sqrt(k1 k2)``, ``u_i = (B - G / 2) e_i``, ``A_1 = k1 (3 k1 + 2 k2)``, ``A_2 = k2 (2 k1
+ 3 k2)`` and ``J`` the quarter turn about the normal (``J e1 = e2``), the terms between two gaps (``log`` the
logarithm of their ratio, ``inverse`` the difference of their inverses), about the contact's point, unit viscosity:

    translation along the wall    (2 pi / s) (I + 12 sum_i u_i u_i^T / A_i) log
    rotation about the wall's      (24 pi / s) sum_i (J e_i)(J e_i)^T / A_i log
    their coupling                 (24 pi / s) sum_i u_i (J e_i)^T / A_i log
    the normal                     12 pi / (s (k1 + k2)) inverse
    the normal with the spin       12 pi b12 (k1 - k2) / (s k1 k2 (k1 + k2)) log

which in the body's own principal frame are the compilation's equations term for term, and with the body's frame the
gap's (``b12 = 0``) the sphere's form's ``singular`` (``mime.nodes.hydro.near_contact._sphere``), whose sign convention
(the normal from the opposing surface into the medium) these follow: the compilation's couplings of a force with a
rotation have the other sign in it. No eigenvectors of ``B`` are taken, so a body curved alike every way (a sphere)
has derivatives in its curvature everywhere."""

from __future__ import annotations

import math

from jax import Array

from mime.core import jnp
from mime.nodes.hydro.near_contact._seam import CURVATURE_FLOOR, Increment, NearContact, about, floored

_PI = math.pi


def singular(k1: Array, k2: Array, b1: Array, b2: Array, b12: Array, inverse: Array, log: Array) -> Array:
    """The elliptic contact's singular terms about its point, in its frame (the gap's principal directions, then the
    normal), unit viscosity, between two gaps: ``inverse`` the difference of the gaps' inverses, ``log`` the logarithm
    of their ratio; ``k1 >= k2 > 0`` the gap's principal curvatures, ``b1``, ``b2`` and ``b12`` the body's curvature in
    the same frame (the module's docstring). A module attribute so that a control can plant other coefficients."""
    s = jnp.sqrt(k1 * k2)
    first, second = k1 * (3.0 * k1 + 2.0 * k2), k2 * (2.0 * k1 + 3.0 * k2)
    o = jnp.zeros_like(k1)
    u1x, u1y = b1 - 0.5 * k1, b12  # (B - k1 / 2) e1
    u2x, u2y = b12, b2 - 0.5 * k2  # (B - k2 / 2) e2
    along = 2.0 * _PI / s * log
    xx = along * (1.0 + 12.0 * (u1x * u1x / first + u2x * u2x / second))
    yy = along * (1.0 + 12.0 * (u1y * u1y / first + u2y * u2y / second))
    xy = along * 12.0 * (u1x * u1y / first + u2x * u2y / second)
    turning = 24.0 * _PI / s * log
    rxrx, ryry = turning / second, turning / first
    # J e1 = e2 and J e2 = -e1: force along the wall i, rotation about the wall's axis j
    x_rx, y_rx = -turning * u2x / second, -turning * u2y / second
    x_ry, y_ry = turning * u1x / first, turning * u1y / first
    normal = 12.0 * _PI / ((k1 + k2) * s) * inverse
    z_rz = 12.0 * _PI * b12 * (k1 - k2) / (s * k1 * k2 * (k1 + k2)) * log
    return jnp.stack([jnp.stack([xx, xy, o, x_rx, x_ry, o]), jnp.stack([xy, yy, o, y_rx, y_ry, o]),
                      jnp.stack([o, o, normal, o, o, z_rz]), jnp.stack([x_rx, y_rx, o, rxrx, o, o]),
                      jnp.stack([x_ry, y_ry, o, o, ryry, o]), jnp.stack([o, o, z_rz, o, o, o])])


def increment(contact: NearContact, delta: Array, cutoff: Array, viscosity: Array, centre: Array,
              reach: Array) -> Increment:
    """The elliptic form's increment ``A(h) - A(h + delta)``, each gap floored at the roughness ``cutoff``, about
    ``centre`` in the world frame: the singular terms about the contact's point. The gap's curvatures are held at least
    :data:`CURVATURE_FLOOR` over the body's ``reach`` (a flat contact's terms finite: such a contact is a line or a face,
    flagged and given no increment by the rung that classes it)."""
    dtype = contact.gap.dtype
    floor = CURVATURE_FLOOR / reach
    k1 = jnp.maximum(contact.k1, floor)
    k2 = jnp.clip(contact.k2, floor, k1)
    near, far = floored(contact.gap, cutoff), floored(contact.gap + delta, cutoff)
    local = viscosity * singular(k1, k2, contact.b1, contact.b2, jnp.asarray(contact.b12, dtype=dtype),
                                 1.0 / near - 1.0 / far, jnp.log(far / near))
    return Increment(about(local, contact.frame, contact.point - centre).astype(dtype), jnp.zeros((), dtype=dtype))


__all__ = ["increment", "singular"]
