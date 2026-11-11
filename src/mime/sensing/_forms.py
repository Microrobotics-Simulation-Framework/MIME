"""Route 1, the closed form: each shape's chord along a ray and its edge function, in the shape's own frame, on
per-ray arrays (a component each: no ``(N, 3)`` intermediates).

A form gives, for rays ``o + t d`` (``|d| = 1``):

- ``interval``: where the ray is inside, ``(t_in, t_out)``, empty where ``t_out <= t_in``; its chord is the length
  between them. The grazing square root is taken of a declared softplus (``GRAZING``), so the chord and its
  derivatives stay finite at a tangent ray and beyond it.
- ``edge``: an edge function ``g`` that is negative where the ray meets the body, zero on its silhouette and positive
  beyond, vanishing there linearly. Only its zero set and its gradient matter: the pixel integration reads a straight
  edge at ``g / |grad g|`` pixels with normal ``grad g / |grad g|``, the gradient by ``jvp`` along the ray table's
  derivatives.
- ``pieces``: the edge functions ``edge`` is composed of (a capped cylinder's two caps, its side and its span), so that
  a pixel two of them cross is known to hold a corner.

The silhouettes: a ball's is its tangent cone, ``g`` the distance from the centre to the ray less the radius. A
capsule's is the ray's distance from its segment less the radius (the distance from a line to a segment is the
distance to the clamped foot of their common perpendicular, exactly, the squared distance being convex along the
segment). A capped cylinder is the convex hull of its two cap discs, so its image is their images' hull: ``g`` is the
least of each cap disc's (the in-plane distance from the cap's centre to where the ray crosses its plane, less the
radius) and the side's (the distance between the ray and the axis less the radius, beyond the span where the common
perpendicular's foot leaves it). Where that foot leaves the span inside the side's band, the ray crosses the cap disc
there, so the least is negative: the composition's zero set is the hull's outline exactly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax

from mime.core._node import jnp

type Vec = tuple[Any, Any, Any]
type Interval = tuple[Any, Any]
lax: Any = jax.lax

#: The grazing chord's smooth clamp: the half-chord's square ``h2`` (``R^2`` less the squared distance from the axis or
#: centre) enters its square root as ``w softplus(h2 / w)``, ``w = (GRAZING R)^2``. Beyond the silhouette the chord falls
#: to zero within a few ``GRAZING R``; at a tangent ray it is ``2 GRAZING R sqrt(ln 2)``, its derivative finite. A
#: declared number, small against any pixel's footprint on a body: the chord's halo lies inside the edge pixels, which
#: the coverage weighs.
GRAZING = 1e-3
#: Directions closer than this to parallel (in the sine or cosine their test reads) are taken at this value: a ray
#: along a cylinder's axis, or across its caps' planes, where a crossing goes to infinity. The value only keeps the
#: arithmetic finite; such a ray's crossing then lies far beyond the body, as it does.
PARALLEL = 1e-12
#: An interval's ends where it is empty: far enough that no body is there, finite so that no ``inf - inf`` is formed.
FAR = 1e30


def dot(a: Vec, b: Vec) -> Any:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _soft(h2: Any, radius: Any) -> Any:
    w = (GRAZING * radius) ** 2
    return w * jax.nn.softplus(h2 / w)


def _root(x: Any) -> Any:
    """``sqrt(x)`` of a clamped square, held at the dtype's smallest normal number: far beyond a silhouette the softplus
    underflows to zero, where the root's derivative would be ``inf x 0``."""
    return jnp.sqrt(jnp.maximum(x, jnp.finfo(jnp.result_type(x)).tiny))


def _safe_sqrt(q: Any, floor: Any) -> Any:
    """``sqrt(q)`` for ``q >= 0``, with a finite derivative where ``q`` is below ``floor`` (a ray through a centre or
    an axis, where the distance has a cone and the edge is far)."""
    return jnp.sqrt(jnp.maximum(q, floor))


def _guard(x: Any) -> Any:
    """``x`` kept away from zero by ``PARALLEL``, its sign kept (zero taken as positive)."""
    return jnp.where(x < 0, jnp.minimum(x, -PARALLEL), jnp.maximum(x, PARALLEL))


@dataclass(frozen=True)
class Form:
    """A closed form in its own frame: its ``interval``, ``edge`` and ``pieces`` on rays ``(o, d)``, and the radius of
    the smallest ball about its frame's origin that holds it (``enclosing``), for a window's size."""

    interval: Callable[[Mapping[str, Any], Vec, Vec], Interval]
    edge: Callable[[Mapping[str, Any], Vec, Vec], Any]
    pieces: Callable[[Mapping[str, Any], Vec, Vec], tuple[Any, ...]]
    enclosing: Callable[[Mapping[str, float]], float]
    corners: bool = False


def _ball(centre_z: Any, radius: Any, o: Vec, d: Vec) -> tuple[Any, Any, Any]:
    """A ball on the frame's z axis at ``centre_z``: the ray's parameter nearest its centre, the squared distance there
    (stable: the offset's square, not a difference of squares) and the smoothly clamped half-chord's square."""
    w = (o[0], o[1], o[2] - centre_z)
    mid = -dot(w, d)
    p = (w[0] + mid * d[0], w[1] + mid * d[1], w[2] + mid * d[2])
    q = dot(p, p)
    return mid, q, _soft(radius * radius - q, radius)


def _sphere_interval(p: Mapping[str, Any], o: Vec, d: Vec) -> Interval:
    mid, _, h2 = _ball(0.0, p["radius"], o, d)
    half = _root(h2)
    return mid - half, mid + half


def _sphere_edge(p: Mapping[str, Any], o: Vec, d: Vec) -> Any:
    _, q, _ = _ball(0.0, p["radius"], o, d)
    return _safe_sqrt(q, (GRAZING * p["radius"]) ** 2) - p["radius"]


SPHERE = Form(_sphere_interval, _sphere_edge, lambda p, o, d: (_sphere_edge(p, o, d),),
              lambda s: float(s["radius"]))


def _side(radius: Any, o: Vec, d: Vec) -> tuple[Any, Any, Any, Any]:
    """The unbounded cylinder of ``radius`` about the z axis: the parameter of the ray's closest approach to the axis,
    the squared distance there, the clamped half-chord in the parameter, and the axial coordinate of the approach."""
    across = _guard(d[0] * d[0] + d[1] * d[1])
    mid = -(o[0] * d[0] + o[1] * d[1]) / across
    mx, my = o[0] + mid * d[0], o[1] + mid * d[1]
    q = mx * mx + my * my
    half = _root(_soft(radius * radius - q, radius) / across)
    return mid, q, half, o[2] + mid * d[2]


def _span(half_length: Any, o: Vec, d: Vec) -> Interval:
    """Where the ray lies between the planes ``z = -+ half_length``."""
    along = _guard(d[2])
    a, b = (-half_length - o[2]) / along, (half_length - o[2]) / along
    return jnp.minimum(a, b), jnp.maximum(a, b)


def _cylinder_interval(p: Mapping[str, Any], o: Vec, d: Vec) -> Interval:
    mid, _, half, _ = _side(p["radius"], o, d)
    low, high = _span(p["length"] / 2.0, o, d)
    return jnp.maximum(mid - half, low), jnp.minimum(mid + half, high)


def _cap(z: Any, radius: Any, o: Vec, d: Vec) -> Any:
    """The cap disc in the plane ``z``: the in-plane distance from its centre to where the ray crosses the plane, less
    the radius."""
    t = (z - o[2]) / _guard(d[2])
    x, y = o[0] + t * d[0], o[1] + t * d[1]
    return _safe_sqrt(x * x + y * y, (GRAZING * radius) ** 2) - radius


def _cylinder_pieces(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
    radius, half_length = p["radius"], p["length"] / 2.0
    _, q, _, axial = _side(radius, o, d)
    side = _safe_sqrt(q, (GRAZING * radius) ** 2) - radius
    return _cap(-half_length, radius, o, d), _cap(half_length, radius, o, d), side, jnp.abs(axial) - half_length


def _cylinder_edge(p: Mapping[str, Any], o: Vec, d: Vec) -> Any:
    low, high, side, span = _cylinder_pieces(p, o, d)
    return jnp.minimum(jnp.minimum(low, high), jnp.maximum(side, span))


CYLINDER = Form(_cylinder_interval, _cylinder_edge, _cylinder_pieces,
                lambda s: float((s["radius"] ** 2 + (s["length"] / 2.0) ** 2) ** 0.5), corners=True)


def _capsule_interval(p: Mapping[str, Any], o: Vec, d: Vec) -> Interval:
    radius, half_height = p["radius"], p["height"] / 2.0
    mid, q, half, _ = _side(radius, o, d)
    low, high = _span(half_height, o, d)
    side_in, side_out = jnp.maximum(mid - half, low), jnp.minimum(mid + half, high)
    ins, outs, met = [side_in], [side_out], [(q < radius * radius) & (side_out > side_in)]
    for z in (-half_height, half_height):
        centre, q_ball, h2 = _ball(z, radius, o, d)
        ins.append(centre - _root(h2))
        outs.append(centre + _root(h2))
        met.append(q_ball < radius * radius)
    # The capsule is convex, so its interval is the union of its pieces': the least entry and the last exit of the
    # pieces the ray meets (a piece it misses, whose clamped chord is a halo about its nearest point, moved away); a ray
    # that meets none keeps the side's halo, as the cylinder's does.
    enter = jnp.minimum(jnp.minimum(jnp.where(met[0], ins[0], FAR), jnp.where(met[1], ins[1], FAR)),
                        jnp.where(met[2], ins[2], FAR))
    leave = jnp.maximum(jnp.maximum(jnp.where(met[0], outs[0], -FAR), jnp.where(met[1], outs[1], -FAR)),
                        jnp.where(met[2], outs[2], -FAR))
    hit = met[0] | met[1] | met[2]
    return jnp.where(hit, enter, side_in), jnp.where(hit, leave, side_out)


def _capsule_edge(p: Mapping[str, Any], o: Vec, d: Vec) -> Any:
    radius, half_height = p["radius"], p["height"] / 2.0
    b = d[2]
    e = dot(o, d)
    foot = (o[2] - e * b) / _guard(1.0 - b * b)
    s = jnp.clip(foot, -half_height, half_height)
    v = (-o[0], -o[1], s - o[2])
    along = dot(v, d)
    px, py, pz = v[0] - along * d[0], v[1] - along * d[1], v[2] - along * d[2]
    return _safe_sqrt(px * px + py * py + pz * pz, (GRAZING * radius) ** 2) - radius


CAPSULE = Form(_capsule_interval, _capsule_edge, lambda p, o, d: (_capsule_edge(p, o, d),),
               lambda s: float(s["radius"] + s["height"] / 2.0))


def chord(interval: Interval) -> Any:
    """The length of the ray inside an interval: zero where it is empty."""
    return jnp.maximum(interval[1] - interval[0], 0.0)


def overlap(a: Interval, b: Interval) -> Any:
    """The length of the ray inside both intervals."""
    return jnp.maximum(jnp.minimum(a[1], b[1]) - jnp.maximum(a[0], b[0]), 0.0)
