"""Crease measures (``Geometry.crease_measure``): a capped body's caps as exact edge functions on rays, and the
composition of a capped body's edge, shared by the kinds whose body is a marched region between two end planes (the
cylinder; the screw).

**A cap**, the face of the body in the plane ``x . a = z`` (``a`` the unit axis), by its cone from the pinhole:
``|V| - rho(V) f``, ``V = d_a o + (z - o_a) d - z d_a a`` the in-plane offset from the cap's centre of the ray's
crossing times ``d_a``, ``rho`` the rim's radius in the direction of ``V`` and ``f`` the facing (``d_a`` or ``-d_a``, by
the side of the plane the pinhole is on). Where the ray crosses the plane ahead it is ``|d_a|`` times the crossing's
in-plane distance from the centre less the rim's radius there: below zero inside the rim, vanishing linearly on it;
where it crosses behind it is above zero. Homogeneous in ``d``, it has no pole at the plane's vanishing line. It is the
closed-form route's cap (``mime.sensing._forms._cap``) for any axis and any star-shaped rim.

**Its bound**, the ball about the cap's centre of the radius that holds the rim: the distance from the centre to the ray's
line less that radius (a body behind the camera is flagged and not imaged, so its line, not its ray).

**The composition** (``capped``): the body is its caps and, for each branch of the march, the intersection of its
regions at that branch's least; each cap is taken as the greater of it and its ball (the same outline, and the ball's
distance where a thin cap's first-order distance falls short beyond its ends), as the closed form does."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mime.core import switch
from mime.core import _sign_forms as signs

type Vector = tuple[Any, Any, Any]


def cap_side(kind: str) -> switch.Declaration:
    """The switch of a kind's caps' facing, ``<kind>.cap_side``."""
    return switch.Declaration(
        owner=f"{kind}.cap_side",
        indicator="the pinhole's height above a cap's plane along the axis, in metres (positive where the pinhole lies "
                  "on the plane's side towards the axis's positive end)",
        orders={"crease_measure": -1},
        route="none: a pinhole in a cap's plane sees the cap edge-on, and the cone's side flips there, so the cap's edge "
              "function changes by twice its facing term")


def cap(o: Vector, d: Vector, axis: Vector, z: Any, rim: Callable[[Any, Any, Any], Any], *, owner: str,
        scale: Any) -> Any:
    """The cap in the plane ``x . axis = z`` on the rays ``o + t d``: ``|V| - rho f`` (the module's notes), ``rim(vx, vy,
    vz)`` the rim's radius in the direction of the in-plane vector ``V`` (homogeneous of degree zero in it), ``scale``
    the body's size for the length's floor."""
    o_a, d_a = signs.along(o, axis), signs.along(d, axis)
    height = z - o_a
    v = tuple(d_a * o[i] + height * d[i] - z * d_a * axis[i] for i in range(3))
    facing = switch.where(height < 0.0, -d_a, d_a, owner=owner, resolution=lambda: switch.rounding(scale),
                          indicator=lambda: -height)
    return signs.length(*v, scale=scale) - rim(*v) * facing


def ball(o: Vector, d: Vector, centre: Vector, radius: Any, scale: Any) -> Any:
    """The distance from ``centre`` to the lines ``o + t d`` (``|d| = 1``) less ``radius``."""
    w = tuple(o[i] - centre[i] for i in range(3))
    along = signs.along(w, d)
    return signs.length(*(w[i] - along * d[i] for i in range(3)), scale=scale) - radius


def capped(marched: Sequence[Sequence[Any]], creases: Sequence[Any], lesser: Callable[[Any, Any], Any],
           greater: Callable[[Any, Any], Any]) -> Any:
    """A capped body's edge: ``creases`` its two caps and then their two balls, ``marched`` each branch's regions at its
    least. The caps (each the greater of it and its ball) and, for each branch, the greatest of its regions, all in
    one union."""
    edge = lesser(greater(creases[0], creases[2]), greater(creases[1], creases[3]))
    for regions in marched:
        held = regions[0]
        for region in regions[1:]:
            held = greater(held, region)
        edge = lesser(edge, held)
    return edge


__all__ = ["ball", "cap", "cap_side", "capped"]
