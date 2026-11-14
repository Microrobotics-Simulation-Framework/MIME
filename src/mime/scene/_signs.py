"""The geometry kinds' cheap sign functions (``Geometry.sign_function``) on arrays of points' coordinates, three arrays
of any one shape (a camera's rays by their samples), never an ``(n, 3)`` array of points.

Each is its region's signed distance where that has a closed form (the ball, the capsule, the cylinder's side, the
slab), so its gradient's length is one. On the medial set (a centre, an axis, a segment) the distance has a kink and a
point there no direction; a length is taken with its square floored at ``FLOOR`` of the kind's scale squared (by the
dtype), so that it and its derivative are defined there (with the right sign: the medial set lies inside), and it
moves the surface and its gradient's length by a share of ``FLOOR^2 / 2``, far below either precision's rounding. The clamp to a capsule's segment and the slab's
mid-plane are the shapes' own switches, ``capsule.end`` and ``slab.mid``."""

from __future__ import annotations

from typing import Any

from mime.core import AXIS_ULPS, jnp, switch

type Point = tuple[Any, Any, Any]


#: The floor of a length, as a share of the kind's scale (its radius), by dtype: its square stays a normal number in
#: float32 at a micrometre's scale, and its share squared, ``FLOOR^2 / 2``, is under each dtype's rounding.
FLOOR = {"float32": 1e-7, "float64": 1e-100}


def length(*components: Any, scale: Any) -> Any:
    """The length of the vectors whose components are ``components`` (arrays of one shape), its square floored at
    ``FLOOR`` of ``scale`` squared."""
    floor = FLOOR[jnp.result_type(*components).name] * scale
    return jnp.sqrt(sum((c * c for c in components), start=floor * floor))


def unit(axis: Any) -> tuple[Any, Any, Any]:
    """A direction setting (three numbers, never zero) scaled to unit length."""
    held = jnp.asarray(axis)
    held = held / switch.nonzero_norm(held)
    return held[0], held[1], held[2]


def along(point: Point, axis: tuple[Any, Any, Any]) -> Any:
    return point[0] * axis[0] + point[1] * axis[1] + point[2] * axis[2]


def across(point: Point, axis: tuple[Any, Any, Any], at: Any, scale: Any) -> Any:
    """The distance of each point from the axis's point ``at`` along it."""
    return length(*(point[i] - at * axis[i] for i in range(3)), scale=scale)


def segment(point: Point, axis: tuple[Any, Any, Any], half: Any, scale: Any) -> Any:
    """The distance of each point from the segment of half-length ``half`` along ``axis`` through the origin: from
    its axial coordinate clamped to the segment (the capsule's switch ``capsule.end``)."""
    axial = along(point, axis)
    nearest = switch.clip(axial, -half, half, owner="capsule.end", resolution=lambda: switch.rounding(half),
                          indicator=axial)
    return across(point, axis, nearest, scale)


def slab(point: Point, axis: tuple[Any, Any, Any], half: Any) -> Any:
    """The signed distance from the slab of half-thickness ``half`` about the origin, across ``axis``: its axial
    coordinate's magnitude (the switch ``slab.mid``, the slab's mid-plane) less the half-thickness."""
    axial = along(point, axis)
    return switch.absolute(axial, owner="slab.mid",
                           resolution=lambda: AXIS_ULPS * float(jnp.finfo(jnp.result_type(axial)).eps) * half,
                           indicator=axial) - half


__all__ = ["FLOOR", "across", "along", "length", "segment", "slab", "unit"]
