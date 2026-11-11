"""The sphere's form of the near-contact law: the exact increment of a sphere's resistance near
a wall, the hybrid of the plane's exact remainder and the contact's own singular terms.

**The plane.** A sphere's resistance near a plane, exact, from the bispherical solutions (Brenner 1961 for the normal,
Jeffery 1915 for the spin about it, Dean and O'Neill 1963 and O'Neill 1964 for the motions along the wall), is MIME's
packaged table ``sphere_plane_resistance`` (``scripts/sphere_plane_table.py`` builds it; its sha256 pinned): five
functions of the gap over the radius, ``d``, from ``1e-3`` to ``10``. Each is read as its singular terms, exactly,
plus a remainder interpolated by a cubic spline in ``ln d`` (C2), and below the table's range the remainder continued
as ``R0 + c1 d ln d + c2 d`` matched to the spline's value and slope at its edge and its value at the next knot (C1
there); past 10 radii the remainder is held at its value there (a rung resolves such gaps itself).

**The hybrid.** The increment between the true gap ``h`` and the opened one ``h'`` is the plane's exact remainder
``E(h/a) - S(h/a)`` (the table less the sphere-plane's singular terms: the normal's ``1/d``, the three logarithms of
Goldman, Cox and Brenner) between the two gaps, plus the contact's own singular terms between them: Cox's (1974) and
Claeys and Brady's (1989) elliptic contact as compiled by Janoschek, Toschi and Harting (2013, arXiv 1308.6482, Eqs.
45-60), whose coefficients carry the gap's principal curvatures and the body's own along them (they give Jeffrey and
Onishi's equal-sphere coefficients 1/6, -1/4, 1/5; the gap's curvature alone, MIME's film, does not). Against a plane
the contact's terms are the sphere-plane's, so the increment is the table's exactly. In a straight tube they carry
the lumen's curvature round its circumference: measured against an independent tube MFS at sphere-to-tube ratios of
0.22, 0.28 and 0.34 (MIME-ANO-154's follow-up probe), where the plane's form over-counts the rolling round the
circumference by 44 to 57 % of the plane's rate."""

from __future__ import annotations

import functools
import json
import math
from typing import Any, NamedTuple

import numpy as np
from jax import Array

from mime.core import jnp
from mime.data import asset, asset_bytes
from mime.nodes.hydro.near_contact._seam import Increment, NearContact, about, floored

#: The packaged table's asset name.
TABLE = "sphere_plane_resistance"
#: The table's entries: the unit sphere's resistance about its centre, the wall's normal +z.
ENTRIES = ("zz", "xx", "ryry", "x_ry", "rzrz")
_PI = math.pi


def interpolated_singular(d: np.ndarray) -> np.ndarray:
    """The singular terms taken out of each entry before it is interpolated (unit sphere and viscosity), rows in
    :data:`ENTRIES`' order: the normal's ``1/d`` and its fifth of the logarithm (Cox and Brenner 1967), Goldman, Cox and
    Brenner's three logarithms (1967), none for the spin."""
    log = np.log(1.0 / d)
    return np.stack([6 * _PI * (1.0 / d + log / 5.0), 6 * _PI * (8.0 / 15.0) * log, 8 * _PI * (2.0 / 5.0) * log,
                     -6 * _PI * (2.0 / 15.0) * log, np.zeros_like(d)])


class Table(NamedTuple):
    """The table as the law reads it: the first knot's ``ln d`` and the step, each interval's cubic in ``ln d``
    (``(4, intervals, 5)``, highest power first), the continuation below the edge (``(3, 5)``: ``R0``, ``c1``, ``c2``)
    and the edge ``d``."""

    first: float
    step: float
    cubics: np.ndarray
    below: np.ndarray
    edge: float


@functools.cache
def table() -> Table:
    """The packaged table, read once, its sha256 checked (``mime.data.asset_bytes``)."""
    from scipy.interpolate import CubicSpline

    held = json.loads(asset_bytes(asset(TABLE)))
    d = np.asarray(held["d"], dtype=np.float64)
    t = np.log(d)
    remainder = np.stack([np.asarray(held[name], dtype=np.float64) for name in ENTRIES]) - interpolated_singular(d)
    spline = CubicSpline(t, remainder.T, bc_type="not-a-knot")
    cubics = np.asarray(spline.c)  # (4, intervals, 5)
    edge, next_ = d[0], d[1]
    slope = np.asarray(spline(t[0], 1)) / edge  # dR/dd at the edge
    rows = np.array([[1.0, edge * math.log(edge), edge], [0.0, math.log(edge) + 1.0, 1.0],
                     [1.0, next_ * math.log(next_), next_]])
    below = np.linalg.solve(rows, np.stack([remainder[:, 0], slope, remainder[:, 1]]))
    return Table(float(t[0]), float(t[1] - t[0]), cubics, below, float(edge))


def remainder(d: Array) -> Array:
    """The five entries' remainders after :func:`interpolated_singular`, at ``d`` (the gap over the radius)."""
    held = table()
    dtype = d.dtype
    t = jnp.log(d)
    intervals = held.cubics.shape[1]
    at = jnp.minimum(t, held.first + intervals * held.step)
    index = jnp.clip(jnp.floor((at - held.first) / held.step), 0, intervals - 1).astype(jnp.int32)
    s = at - (held.first + index.astype(dtype) * held.step)
    c = jnp.asarray(held.cubics, dtype=dtype)[:, index, :]
    inside = ((c[0] * s + c[1]) * s + c[2]) * s + c[3]
    below = jnp.asarray(held.below, dtype=dtype)
    continued = below[0] + below[1] * d * t + below[2] * d
    return jnp.where(d < held.edge, continued, inside)


def plane_remainder(d: Array) -> Array:
    """The plane's exact remainder the hybrid adds, entry by entry: the table less the sphere-plane's singular terms
    the hybrid replaces by the contact's own (the normal's ``1/d``, the three logarithms), the normal's fifth of the
    logarithm left in it."""
    return remainder(d) + jnp.stack([6 * _PI * jnp.log(1.0 / d) / 5.0, *(jnp.zeros_like(d) for _ in range(4))])


def plane_matrix(entries: Array) -> Array:
    """The 6 x 6 of a sphere above a plane, normal +z, about its centre, from its five entries."""
    zz, xx, rr, c, spin = entries
    o = jnp.zeros_like(zz)
    return jnp.stack([jnp.stack([xx, o, o, o, c, o]), jnp.stack([o, xx, o, -c, o, o]), jnp.stack([o, o, zz, o, o, o]),
                      jnp.stack([o, -c, o, rr, o, o]), jnp.stack([c, o, o, o, rr, o]),
                      jnp.stack([o, o, o, o, o, spin])])


def singular(k1: Array, k2: Array, b1: Array, b2: Array, inverse: Array, log: Array) -> Array:
    """The elliptic contact's singular terms about its point, in its frame, unit viscosity, between two gaps:
    ``inverse`` the difference of the gaps' inverses, ``log`` the logarithm of their ratio (Cox 1974; Claeys and Brady
    1989; compiled by Janoschek, Toschi and Harting 2013, Eqs. 45-60): the normal ``12 pi / ((k1 + k2) s h)``; the
    translation along a direction ``pi / s (2 + 6 (2 b_t - k_t)^2 / (k_t (3 k_t + 2 k_o))) ln(1/h)``; the force along it
    per rotation about ``n x t``, ``12 pi (2 b_t - k_t) / (k_t (3 k_t + 2 k_o) s) ln(1/h)``; the rotation about ``n x
    t``, ``24 pi / (k_t (3 k_t + 2 k_o) s) ln(1/h)``; ``s = sqrt(k1 k2)``, ``k`` the gap's curvatures, ``b`` the body's
    along the same directions. Written for the body's principal directions along the gap's (Janoschek et al.'s chi =
    0), which is exact for a sphere, whose curvature is the same in every direction; a body whose own principal
    directions are turned from the gap's has the further terms the compilation gives for chi other than zero (its K12,
    K14, K25 and K45), which these leave out. A module attribute so that a control can plant other coefficients."""
    s = jnp.sqrt(k1 * k2)
    first, second = k1 * (3.0 * k1 + 2.0 * k2), k2 * (2.0 * k1 + 3.0 * k2)
    o = jnp.zeros_like(k1)
    normal = 12.0 * _PI / ((k1 + k2) * s) * inverse
    xx = _PI / s * (2.0 + 6.0 * (2.0 * b1 - k1) ** 2 / first) * log
    yy = _PI / s * (2.0 + 6.0 * (2.0 * b2 - k2) ** 2 / second) * log
    x_ry = 12.0 * _PI * (2.0 * b1 - k1) / (first * s) * log
    y_rx = -12.0 * _PI * (2.0 * b2 - k2) / (second * s) * log
    ryry = 24.0 * _PI / (first * s) * log
    rxrx = 24.0 * _PI / (second * s) * log
    return jnp.stack([jnp.stack([xx, o, o, o, x_ry, o]), jnp.stack([o, yy, o, y_rx, o, o]),
                      jnp.stack([o, o, normal, o, o, o]), jnp.stack([o, y_rx, o, rxrx, o, o]),
                      jnp.stack([x_ry, o, o, o, ryry, o]), jnp.stack([o, o, o, o, o, o])])


def sphere_radius(b1: Array, b2: Array) -> Array:
    """The sphere's radius as its form reads it: the inverse of the body's mean curvature at the contact. A module
    attribute so that a control can stop its derivative."""
    return 2.0 / (b1 + b2)


def increment(contact: NearContact, delta: Array, cutoff: Array, viscosity: Array, centre: Array) -> Increment:
    """The sphere's increment ``A(h) - A(h + delta)``, each gap floored at the roughness ``cutoff``, about ``centre`` in
    the world frame: the plane's exact remainder about the sphere's centre plus the contact's singular terms about its
    point (the module's docstring). The sphere's radius is the body's own curvature at the contact."""
    dtype = contact.gap.dtype
    near, far = floored(contact.gap, cutoff), floored(contact.gap + delta, cutoff)
    radius = sphere_radius(contact.b1, contact.b2)
    one = jnp.ones_like(radius)
    arms = jnp.stack([one, one, one, radius, radius, radius])
    plane = plane_matrix(plane_remainder(near / radius) - plane_remainder(far / radius))
    plane = viscosity * radius * plane * jnp.outer(arms, arms)
    own = viscosity * singular(contact.k1, contact.k2, contact.b1, contact.b2, 1.0 / near - 1.0 / far,
                               jnp.log(far / near))
    sphere = contact.point + radius * contact.frame[2]
    found = about(plane, contact.frame, sphere - centre) + about(own, contact.frame, contact.point - centre)
    return Increment(found.astype(dtype), jnp.zeros((), dtype=dtype))


def entries_at(d: Any) -> np.ndarray:
    """The table's five entries at ``d`` as the law reads them (float64): its singular terms plus the remainder."""
    held = np.asarray(d, dtype=np.float64)
    return np.asarray(remainder(jnp.asarray(held))) + interpolated_singular(held)


__all__ = ["ENTRIES", "TABLE", "Table", "entries_at", "increment", "interpolated_singular", "plane_matrix",
           "plane_remainder", "remainder", "singular", "sphere_radius", "table"]
