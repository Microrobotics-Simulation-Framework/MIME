"""Route 1, the closed form: each shape's chord along a ray and its edge function, in the shape's own frame, on
per-ray arrays (a component each: no ``(N, 3)`` intermediates).

A form gives, for rays ``o + t d`` (``|d| = 1``, ``t >= 0``: a ray starts at the pinhole, so nothing behind the camera is
imaged):

- ``interval``: where the ray's line is inside, ``(t_in, t_out)``, empty where ``t_out <= t_in``; its chord is the
  length between them, taken on the ray (``ahead``). The grazing square root is taken of a declared softplus
  (``GRAZING``), so the chord and its derivatives stay finite at a tangent ray and beyond it.
- ``pieces``: edge functions ``g``, each negative where the ray meets its piece, zero on its outline and positive beyond,
  vanishing there linearly, on the ray (not its whole line); and ``compose``, how the pieces make the body (a least for
  a union, a greatest for an intersection). The pixel integration reads each piece's straight edge at ``g / |grad g|``
  pixels with normal ``grad g / |grad g|``, the gradient by ``jvp`` along the ray table's derivatives, and composes those
  distances, which share one scale (pixels); raw values do not (a cap seen nearly edge-on changes thirteen times faster
  across the image than a side), so a least of raw values can take the wrong piece. ``edge`` is the composition of the raw
  values: the same sign everywhere, so the same outline.

The silhouettes: a ball's is its tangent cone, ``g`` the distance from the centre to the ray less the radius. A
capsule's is the ray's distance from its segment less the radius (the distance from a line to a segment is the distance
to the clamped foot of their common perpendicular, exactly, the squared distance being convex along the segment; on the
ray, where that foot is behind the pinhole, the pinhole's distance from the segment). A capped cylinder is the convex
hull of its two cap discs, so its image is their images' hull: its pieces are each cap disc (by its cone from the
pinhole, ``_cap``, bounded by the cap's ball), the side (the distance between the ray and the axis less the radius) and
the span (the axial coordinate of their common perpendicular's foot beyond the half-length), composed as the least of
the caps and of the greatest of the side and the span. Where that foot leaves the span inside the side's band, the ray
crosses the cap disc there, so the least is negative: the composition's zero set is the hull's outline exactly."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from mime.core import switch
from mime.core._node import jnp

type Vec = tuple[Any, Any, Any]
type Interval = tuple[Any, Any]
lax: Any = jax.lax

#: The grazing chord's smooth clamp: the half-chord's square ``h2`` (``R^2`` less the squared distance from the axis or
#: centre) enters its square root as ``w softplus(h2 / w)``, ``w = (GRAZING R)^2``. Near the tangent ``h2`` is about
#: ``2 R`` times the ray's distance inside the silhouette, so the clamp's width in distance is ``GRAZING^2 R / 2``
#: (5e-7 of the radius): beyond the silhouette the chord falls to zero within a few of those; at a tangent ray it is
#: ``2 GRAZING R sqrt(ln 2)``, its derivative finite. A declared number, small against any pixel's footprint on a body:
#: the chord's halo lies inside the edge pixels, which the coverage weighs. (In float32 that width is about eight units
#: in the last place of ``h2`` at the tangent, so the smooth derivative there is barely above rounding.)
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


def _on_ray(ahead: Any, on_line: Any, at_start: Any, nearest: Any) -> Any:
    """A piece's value on the ray: its value on the ray's line where the line's nearest point to the piece (at the ray
    parameter ``nearest``, in metres) is ahead of the pinhole, else its value at the pinhole."""
    return switch.where(ahead, on_line, at_start, owner="camera.ahead",
                        resolution=lambda: switch.rounding(nearest), indicator=lambda: nearest)


def _guard(x: Any) -> Any:
    """``x`` kept away from zero by ``PARALLEL``, its sign kept (zero taken as positive)."""
    return jnp.where(x < 0, jnp.minimum(x, -PARALLEL), jnp.maximum(x, PARALLEL))


type Compose = Callable[[Sequence[Any], Callable[[Any, Any], Any], Callable[[Any, Any], Any]], Any]


def _single(pieces: Sequence[Any], lesser: Callable[[Any, Any], Any], greater: Callable[[Any, Any], Any]) -> Any:
    return pieces[0]


@dataclass(frozen=True)
class Form:
    """A closed form in its own frame: its ``interval`` and ``pieces`` on rays ``(o, d)``, how the pieces compose
    (``compose(pieces, lesser, greater)``, given the least and the greatest of two pieces, whatever a piece is: a raw
    value or an edge's line), and the radius of the smallest ball about its frame's origin that holds it
    (``enclosing``), for a window's size. ``corners``: whether two pieces meet at an angle in its outline; ``turning``:
    the pieces whose own outline can turn within a pixel (a cap disc seen near edge-on, its ellipse's ends a fraction of
    a pixel round), sub-sampled wherever they cross one; ``bounds``: pairs ``(i, j)``, piece ``i``'s region lying inside
    piece ``j``'s (a cap inside its bounding ball), so that ``i``'s line is taken as the greater of the two: the same
    outline, and ``j``'s distance where ``i``'s first-order one falls short (beyond a thin cap's ends, where a cap's
    edge function changes fastest across its ellipse, not towards its end). Pieces named only as bounds are not
    composed."""

    interval: Callable[[Mapping[str, Any], Vec, Vec], Interval]
    pieces: Callable[[Mapping[str, Any], Vec, Vec], tuple[Any, ...]]
    enclosing: Callable[[Mapping[str, float]], float]
    compose: Compose = _single
    corners: bool = False
    turning: tuple[int, ...] = ()
    bounds: tuple[tuple[int, int], ...] = ()

    def edge(self, p: Mapping[str, Any], o: Vec, d: Vec) -> Any:
        """The composed edge function of raw values: its sign and its zero set are the body's."""
        return self.compose(self.pieces(p, o, d), jnp.minimum, jnp.maximum)


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


def _sphere_pieces(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
    """The distance from the centre to the ray less the radius: where the centre is behind the pinhole, the pinhole's
    distance from it."""
    mid, q, _ = _ball(0.0, p["radius"], o, d)
    on_ray = _on_ray(mid >= 0.0, q, dot(o, o), mid)
    return (_safe_sqrt(on_ray, (GRAZING * p["radius"]) ** 2) - p["radius"],)


SPHERE = Form(_sphere_interval, _sphere_pieces, lambda s: float(s["radius"]))


def _side(radius: Any, o: Vec, d: Vec) -> tuple[Any, Any, Any, Any]:
    """The unbounded cylinder of ``radius`` about the z axis: the parameter of the ray's line's closest approach to the
    axis, the squared distance there, the clamped half-chord in the parameter, and the axial coordinate of the
    approach."""
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
    return (switch.minimum(a, b, owner="camera.span.low", resolution=lambda: switch.rounding(jnp.abs(a) + jnp.abs(b)),
                           indicator=lambda: b - a),
            switch.maximum(a, b, owner="camera.span.high", resolution=lambda: switch.rounding(jnp.abs(a) + jnp.abs(b)),
                           indicator=lambda: a - b))


def _cylinder_interval(p: Mapping[str, Any], o: Vec, d: Vec) -> Interval:
    mid, _, half, _ = _side(p["radius"], o, d)
    low, high = _span(p["length"] / 2.0, o, d)
    return (switch.maximum(mid - half, low, owner="camera.interval.enter",
                           resolution=lambda: switch.rounding(jnp.abs(mid) + half), indicator=lambda: mid - half - low),
            switch.minimum(mid + half, high, owner="camera.interval.leave",
                           resolution=lambda: switch.rounding(jnp.abs(mid) + half), indicator=lambda: high - mid - half))


def _cap(z: Any, radius: Any, o: Vec, d: Vec) -> Any:
    """The cap disc in the plane ``z``, by its cone from the pinhole: ``|V| - R d_z s``, ``V`` the in-plane offset from
    the cap's centre of the ray's crossing times ``d_z`` and ``s`` the side of the plane the pinhole is on. Where the ray
    crosses the plane ahead, it is ``|d_z|`` times the in-plane distance less the radius; where it crosses behind, it is
    positive. Homogeneous in ``d``, it has no pole at the plane's vanishing line (where the crossing itself goes to
    infinity, and the crossing's distance, read to first order, puts an outline at that line however far the cap is)."""
    height = z - o[2]
    vx, vy = height * d[0] + d[2] * o[0], height * d[1] + d[2] * o[1]
    facing = switch.where(height < 0.0, -d[2], d[2], owner="camera.cap_side",
                          resolution=lambda: switch.rounding(radius), indicator=lambda: -height)
    floor = (GRAZING * radius * d[2]) ** 2 + jnp.finfo(jnp.result_type(vx)).tiny
    return _safe_sqrt(vx * vx + vy * vy, floor) - radius * facing


def _cylinder_pieces(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
    """The caps, the side and the span, each on the ray: where the line's closest approach to the axis is behind the
    pinhole, the side and the span are read at the pinhole. Then each cap's bounding ball (the ball of the cap's radius
    about its centre, which holds the cap), the cap's bound."""
    radius, half_length = p["radius"], p["length"] / 2.0
    mid, q, _, axial = _side(radius, o, d)
    ahead = mid >= 0.0
    side = _safe_sqrt(_on_ray(ahead, q, o[0] * o[0] + o[1] * o[1], mid), (GRAZING * radius) ** 2) - radius
    along = _on_ray(ahead, axial, o[2], mid)
    span = switch.absolute(along, owner="camera.span.axial", resolution=lambda: switch.rounding(half_length),
                           indicator=lambda: along) - half_length
    balls: list[Any] = []
    for z in (-half_length, half_length):
        centre, q_ball, _ = _ball(z, radius, o, d)
        start = (o[0], o[1], o[2] - z)
        balls.append(_safe_sqrt(_on_ray(centre >= 0.0, q_ball, dot(start, start), centre), (GRAZING * radius) ** 2)
                     - radius)
    return _cap(-half_length, radius, o, d), _cap(half_length, radius, o, d), side, span, balls[0], balls[1]


def _hull_of_caps(pieces: Sequence[Any], lesser: Callable[[Any, Any], Any], greater: Callable[[Any, Any], Any]) -> Any:
    return lesser(lesser(pieces[0], pieces[1]), greater(pieces[2], pieces[3]))


CYLINDER = Form(_cylinder_interval, _cylinder_pieces,
                lambda s: float((s["radius"] ** 2 + (s["length"] / 2.0) ** 2) ** 0.5), _hull_of_caps, corners=True,
                turning=(0, 1), bounds=((0, 4), (1, 5)))


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


def _capsule_pieces(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
    """The ray's distance from the segment less the radius. The line's: the segment's point at the clamped foot of
    their common perpendicular, and the line's nearest point to it. Where that point is behind the pinhole, the ray's
    nearest point is the pinhole (the squared distance is convex over the ray and the segment, so with its least off
    the ray it lies on the ray's start), and the segment's point the pinhole's clamped foot on it."""
    radius, half_height = p["radius"], p["height"] / 2.0
    b = d[2]
    e = dot(o, d)
    foot = (o[2] - e * b) / _guard(1.0 - b * b)
    s = jnp.clip(foot, -half_height, half_height)
    v = (-o[0], -o[1], s - o[2])
    along = dot(v, d)
    px, py, pz = v[0] - along * d[0], v[1] - along * d[1], v[2] - along * d[2]
    start = o[2] - jnp.clip(o[2], -half_height, half_height)
    on_ray = _on_ray(along >= 0.0, px * px + py * py + pz * pz, o[0] * o[0] + o[1] * o[1] + start * start, along)
    return (_safe_sqrt(on_ray, (GRAZING * radius) ** 2) - radius,)


CAPSULE = Form(_capsule_interval, _capsule_pieces, lambda s: float(s["radius"] + s["height"] / 2.0))


def ahead(interval: Interval) -> Interval:
    """The part of a line's interval on the ray (``t >= 0``): a ray starts at the pinhole."""
    return (switch.maximum(interval[0], 0.0, owner="camera.interval.start",
                           resolution=lambda: switch.rounding(jnp.abs(interval[0]) + jnp.abs(interval[1])),
                           indicator=lambda: interval[0]),
            switch.maximum(interval[1], 0.0, owner="camera.interval.stop",
                           resolution=lambda: switch.rounding(jnp.abs(interval[0]) + jnp.abs(interval[1])),
                           indicator=lambda: interval[1]))


def chord(interval: Interval) -> Any:
    """The length of the ray inside an interval: zero where it is empty."""
    return switch.maximum(interval[1] - interval[0], 0.0, owner="camera.interval.chord",
                          resolution=lambda: switch.rounding(jnp.abs(interval[0]) + jnp.abs(interval[1])),
                          indicator=lambda: interval[1] - interval[0])


def overlap(a: Interval, b: Interval) -> Any:
    """The length of the ray inside both intervals."""
    return jnp.maximum(jnp.minimum(a[1], b[1]) - jnp.maximum(a[0], b[0]), 0.0)
