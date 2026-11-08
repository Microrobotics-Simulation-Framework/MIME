"""Centrelines as kinds give them (``mime.registry.Centreline``): a straight segment, and a polyline through declared
points (straight segments joined at junctions, the route for a lab's curve and for a mesh's extracted skeleton)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mime.core import jnp
from mime.registry import Centreline, CentrelineFrame, CentrelineSegment, Junction
from mime.registry._centreline import Parameters

def across(direction: Sequence[float] | np.ndarray) -> np.ndarray:
    """A unit vector across ``direction`` (three numbers, any length above zero), fixed in NumPy: the cross product
    with the coordinate axis it is least along, so that the choice does not jump for a direction drawn near one."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    helper = np.eye(3)[int(np.argmin(np.abs(d)))]
    first = np.cross(d, helper)
    return first / np.linalg.norm(first)


def straight(start: Callable[[Parameters], Any], direction: Sequence[float] | np.ndarray,
             length: Callable[[Parameters], Any], *, normal: Sequence[float] | np.ndarray | None = None, radius: Callable[[Parameters, Any], Any] | None = None,
             filament_radius: Callable[[Parameters], Any] | None = None) -> CentrelineSegment:
    """A straight segment from ``start(parameters)`` along ``direction`` (fixed: a part's direction is its frame) for
    ``length(parameters)``; its frame's normal is constant, ``normal`` or :func:`across` the direction."""
    axis = np.asarray(direction, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    given: np.ndarray = across(axis) if normal is None else np.asarray(normal, dtype=np.float64)
    given = given - float(np.dot(given, axis)) * axis
    held: np.ndarray = given / np.linalg.norm(given)

    def frame(parameters: Parameters, s: Any) -> CentrelineFrame:
        at = jnp.asarray(s)
        tangent = jnp.asarray(axis, dtype=at.dtype)
        point = jnp.asarray(start(parameters), dtype=at.dtype)[None, :] + at[:, None] * tangent[None, :]
        count = at.shape[0]
        return CentrelineFrame(point, jnp.broadcast_to(tangent, (count, 3)),
                               jnp.broadcast_to(jnp.asarray(held, dtype=at.dtype), (count, 3)))

    return CentrelineSegment(length=length, frame=frame, radius=radius, filament_radius=filament_radius)


def transported(normal: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """``normal`` (across ``before``) carried across a corner from the unit tangent ``before`` to ``after`` by the
    rotation that turns one into the other about their cross product: the parallel transport of a frame round a
    polyline's corner. A corner that turns the tangent back on itself has no such rotation and is refused."""
    axis = np.cross(before, after)
    sine, cosine = float(np.linalg.norm(axis)), float(np.dot(before, after))
    if cosine <= -1.0 + 1e-12:
        raise ValueError("a polyline's corner turns its direction back on itself, where no frame is carried across")
    if sine == 0.0:
        return normal
    k = axis / sine
    turned: np.ndarray = normal * cosine + np.cross(k, normal) * sine + k * float(np.dot(k, normal)) * (1.0 - cosine)
    return turned / np.linalg.norm(turned)


def _interpolated(low: float, high: float, span: float) -> Callable[[Parameters, Any], Any]:
    """A segment's local radius, linear from ``low`` at its start to ``high`` at its end, ``span`` long."""

    def radius(parameters: Parameters, s: Any) -> Any:
        return low + (high - low) * jnp.asarray(s) / span

    return radius


def polyline(points: Sequence[Sequence[float]], radii: Sequence[float] | None = None, *,
             filament_radius: float | None = None) -> Centreline:
    """A centreline through declared ``points`` (``(k, 3)``, m, in the kind's frame, two or more): straight segments
    joined at junctions ``j1`` to ``j(k-2)`` at the inner points, the frame's normal carried round each corner
    (:func:`transported`), each segment's local radius interpolated linearly between ``radii`` at its ends where they
    are given; ``filament_radius`` the one radius a slender law's coefficients take (the largest of ``radii`` where
    they are given and it is not). The points are fixed into the program, as undeclared geometry is: the route for a
    curve a lab declares and for a mesh's extracted skeleton."""
    held = np.asarray(points, dtype=np.float64)
    if held.ndim != 2 or held.shape[1] != 3 or len(held) < 2:
        raise ValueError(f"a polyline takes two points or more, each three numbers; it was given shape {held.shape}")
    steps = np.diff(held, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("a polyline's consecutive points are distinct and finite")
    tangents = steps / lengths[:, None]
    widths = None if radii is None else np.asarray(radii, dtype=np.float64)
    if widths is not None and (widths.shape != (len(held),) or np.any(widths <= 0.0)):
        raise ValueError(f"a polyline's radii are one length above zero a point, {len(held)}")
    thickest = filament_radius if filament_radius is not None else (
        None if widths is None else float(np.max(widths)))
    segments: list[CentrelineSegment] = []
    normal = across(tangents[0])
    for i, tangent in enumerate(tangents):
        if i:
            normal = transported(normal, tangents[i - 1], tangent)
        local: Callable[[Parameters, Any], Any] | None = None
        if widths is not None:
            local = _interpolated(float(widths[i]), float(widths[i + 1]), float(lengths[i]))
        segments.append(straight((lambda parameters, start=held[i]: start), tangent,
                                 (lambda parameters, span=float(lengths[i]): span), normal=normal, radius=local,
                                 filament_radius=None if thickest is None else (lambda parameters, r=thickest: r)))
    junctions = tuple(Junction(f"j{i}", ((i - 1, 1), (i, 0))) for i in range(1, len(segments)))
    return Centreline(tuple(segments), junctions)


__all__ = ["across", "polyline", "straight", "transported"]
