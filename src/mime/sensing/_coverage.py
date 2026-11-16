"""Pixel integration, its own stage: how much of each pixel a body's image covers, by the camera's box filter.

- **The box edge.** Each of a form's pieces (``mime.sensing._forms``) gives, from its edge function ``g`` and its
  gradient along the image's axes (by ``jvp``), a straight line within the pixel: at ``g / |grad g|`` pixels from its
  centre, with normal ``grad g / |grad g|``. The pieces' lines are composed by the form's own composition (a least for a
  union, a greatest for an intersection) on that one scale, the distance in pixels, and the coverage is the exact area
  of the pixel's square on the body's side of the composed line: the box filter exactly where the silhouette is
  straight across the pixel. Its accuracy falls with the silhouette's curvature across a pixel (a sphere's coverage
  and its derivative per pixel, as a relative L2 over its edge's band, within 0.5 % and 1.4 % at a radius of 15 px,
  1.2 % and 2.8 % at 6, 2.8 % and 6.5 % at 3, measured against an exact oracle, ``tests/sensing/test_box_edge.py``),
  which the small-silhouette flag reports.
- **Corner sub-samples.** Where two of a body's pieces cross one pixel (a cylinder seen side-on, its cap meeting its
  side at a right angle: 24 % coverage error with one straight edge), the pixel is split into ``s x s`` sub-pixels, each
  its own box edge at its own centre, the ray there taken along the ray table's derivatives.
- **Exact disc coverage.** A ball's silhouette through a pinhole is a conic; the area of each pixel inside it is exact,
  by mapping the conic to the unit disc (an affine map, which keeps ratios of areas) and intersecting the pixel's
  quadrilateral with the disc. The pixel's corners are undistorted exactly and its sides taken as straight between
  them, so with distortion it is exact to the curvature of a pixel's side through the lens."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from mime.core import switch
from mime.core._node import jnp

#: A pixel is a corner pixel where two of a body's pieces pass within this many pixels of its centre (half its
#: diagonal: a line nearer than that crosses the square).
CORNER_REACH = 0.7072
#: Sub-pixels a side in a corner pixel.
CORNER_SAMPLES = 4
#: Below this gradient length (in the edge function's units per pixel) a pixel's edge is taken as hard: the edge
#: function is flat there (a ray through a centre, where its edge is far).
FLAT = 1e-30
#: The scale a line's distance from a pixel's centre is read on, in pixels, for the resolution of a switch on it.
PIXELS = 1.0


def square_below(a: Any, b: Any, c: Any) -> Any:
    """The area of the unit square centred on the origin where ``a x + b y < c``, for ``a >= b >= 0`` and
    ``a^2 + b^2 = 1``: zero below the far corner, a triangle up to the near side, then linear to the centre, and
    symmetric beyond it."""
    b = jnp.maximum(b, 1e-9)
    corner, flat = (a + b) / 2.0, (a - b) / 2.0

    def outside(c: Any) -> Any:
        """The share on the body's side where the pixel's centre is outside it (``c`` at most zero)."""
        triangle = (c + corner) ** 2 / (2.0 * a * b)
        beyond = switch.where(c <= -flat, triangle, 0.5 + c / a, owner="camera.edge.out_near",
                              resolution=lambda: switch.rounding(PIXELS), indicator=lambda: -flat - c)
        return switch.where(c <= -corner, 0.0, beyond, owner="camera.edge.out_far",
                            resolution=lambda: switch.rounding(PIXELS), indicator=lambda: -corner - c)

    def inside(c: Any) -> Any:
        """The same, written for the mirrored line (``c`` at most zero), where the centre is inside the body."""
        triangle = (c + corner) ** 2 / (2.0 * a * b)
        beyond = switch.where(c <= -flat, triangle, 0.5 + c / a, owner="camera.edge.in_near",
                              resolution=lambda: switch.rounding(PIXELS), indicator=lambda: -flat - c)
        return switch.where(c <= -corner, 0.0, beyond, owner="camera.edge.in_far",
                            resolution=lambda: switch.rounding(PIXELS), indicator=lambda: -corner - c)

    return switch.where(c <= 0.0, outside(jnp.minimum(c, 0.0)), 1.0 - inside(jnp.minimum(-c, 0.0)),
                        owner="camera.edge.centre", resolution=lambda: switch.rounding(PIXELS), indicator=lambda: -c)


class Line(NamedTuple):
    """An edge's straight line within a pixel: its distance from the pixel's centre in pixels (positive where the
    centre is outside the body) and its unit normal's two components, along ``u`` and ``v``."""

    distance: Any
    nu: Any
    nv: Any


def line(g: Any, gu: Any, gv: Any) -> Line:
    """The edge's line within the pixel, from ``g`` and its derivatives along ``u`` and ``v``."""
    length = jnp.sqrt(jnp.maximum(gu * gu + gv * gv, FLAT))
    return Line(g / length, gu / length, gv / length)


def lesser(a: Line, b: Line) -> Line:
    """The nearer-inside of two pieces' lines (a union's): the one at the lesser distance, its normal with it."""
    first = a.distance <= b.distance
    return Line(*(switch.where(first, x, y, owner="camera.pieces.least", resolution=lambda: switch.rounding(PIXELS),
                               indicator=lambda: b.distance - a.distance) for x, y in zip(a, b, strict=True)))


def greater(a: Line, b: Line) -> Line:
    """The farther-out of two pieces' lines (an intersection's)."""
    first = a.distance >= b.distance
    return Line(*(switch.where(first, x, y, owner="camera.pieces.greatest", resolution=lambda: switch.rounding(PIXELS),
                               indicator=lambda: a.distance - b.distance) for x, y in zip(a, b, strict=True)))


def covered(edge: Line, size: float = 1.0) -> Any:
    """The share of a square ``size`` pixels a side, centred where the line was read, on the body's side of it."""
    au, av = jnp.abs(edge.nu), jnp.abs(edge.nv)
    return square_below(jnp.maximum(au, av), jnp.minimum(au, av), -edge.distance / size)


def box(g: Any, gu: Any, gv: Any, size: float = 1.0) -> Any:
    """The share of a square ``size`` pixels a side, centred where ``g`` is evaluated, on the body's side of the edge."""
    return covered(line(g, gu, gv), size)


#: A line's distance from a cell's centre, in cells, beyond which its share of the cell is zero or one (the square's
#: half-diagonal is 0.71): the union's lines are read with their distances clamped here.
UNION_REACH = 2.0
#: The least the union's divisors are taken as, each keeping its sign (a slope along a side or a chord, a direction's
#: component): below it a segment runs within it of parallel to a line, and its share is its middle's side alone.
UNION_FLOOR = 2.0 ** -20
#: The cell's sides, each its middle and its direction, counterclockwise.
_SIDES = (((0.0, -0.5), (1.0, 0.0)), ((0.5, 0.0), (0.0, 1.0)), ((0.0, 0.5), (-1.0, 0.0)), ((-0.5, 0.0), (0.0, -1.0)))


def _outside(edge: Line, size: float) -> tuple[Any, Any, Any]:
    """The half-plane outside a line's body in a cell's own units, ``nu y_u + nv y_v >= c`` (``y`` from the cell's
    centre in cells, the normal unit), its distance clamped at ``UNION_REACH``. A line with no direction (an edge
    function flat to the floor, its distance then beyond the reach) is given the cell's ``u`` axis."""
    squared = edge.nu * edge.nu + edge.nv * edge.nv
    directed = squared > UNION_FLOOR * UNION_FLOOR
    length = jnp.sqrt(switch.domain_guard(directed, squared, 1.0))
    distance = jnp.clip(-edge.distance / (size * length), -UNION_REACH, UNION_REACH)
    return (switch.domain_guard(directed, edge.nu / length, 1.0), switch.domain_guard(directed, edge.nv / length, 0.0),
            distance)


def _share(middle: Any, slope: Any) -> Any:
    """The share of a segment on which a linear function is at least zero: its value at the segment's middle and its
    change along the segment."""
    ratio = middle / jnp.maximum(jnp.abs(slope), UNION_FLOOR) + 0.5
    return switch.clip(ratio, 0.0, 1.0, owner="camera.union.share", resolution=lambda: switch.rounding(1.0),
                       indicator=ratio)


def _both(first: tuple[Any, Any], second: tuple[Any, Any]) -> Any:
    """The share of a segment on which two linear functions (each its middle's value and its change) are both at
    least zero: each keeps one end of it, so where they keep the same end the lesser share, and where opposite ends
    what their shares overlap by."""
    a, b = _share(*first), _share(*second)
    together = first[1] * second[1]
    nested = switch.minimum(a, b, owner="camera.union.nested", resolution=lambda: switch.rounding(1.0),
                            indicator=lambda: b - a)
    apart = switch.maximum(a + b - 1.0, 0.0, owner="camera.union.apart", resolution=lambda: switch.rounding(1.0),
                           indicator=a + b - 1.0)
    return switch.where(together > 0.0, nested, apart, owner="camera.union.ends",
                        resolution=lambda: switch.rounding(1.0), indicator=together)


def _chord(nu: Any, nv: Any, c: Any, closed: bool = False) -> tuple[Any, Any, Any]:
    """A line's chord through the cell (``nu y_u + nv y_v = c``, the normal unit): its length and its middle. ``closed``:
    a line along one of the cell's sides (parallel to it within ``UNION_FLOOR``, on it) keeps its whole chord, the cell
    taken as closed, where otherwise the floor's division leaves it half (``composed_covered``'s sides meet it there)."""

    def slab(along: Any, foot: Any) -> tuple[Any, Any]:
        held = jnp.copysign(jnp.maximum(jnp.abs(along), UNION_FLOOR), along)
        if not closed:
            return -foot / held, 0.5 / jnp.abs(held)
        half = jnp.where(jnp.abs(along) > UNION_FLOOR, 0.5, 0.5 + UNION_FLOOR)
        return -foot / held, half / jnp.abs(held)

    (cu, wu), (cv, wv) = slab(-nv, c * nu), slab(nu, c * nv)
    start = switch.maximum(cu - wu, cv - wv, owner="camera.union.chord_start",
                           resolution=lambda: switch.rounding(1.0), indicator=lambda: (cu - wu) - (cv - wv))
    end = switch.minimum(cu + wu, cv + wv, owner="camera.union.chord_end", resolution=lambda: switch.rounding(1.0),
                         indicator=lambda: (cv + wv) - (cu + wu))
    length = switch.maximum(end - start, 0.0, owner="camera.union.chord_empty",
                            resolution=lambda: switch.rounding(1.0), indicator=end - start)
    middle = 0.5 * (start + end)
    return length, c * nu - middle * nv, c * nv + middle * nu


def union_covered(a: Line, b: Line, size: float = 1.0) -> Any:
    """The exact share of a square ``size`` pixels a side, centred where the lines were read, inside the union of the
    two lines' bodies: one less the area of the convex polygon outside both, by its boundary integral (half the sum
    over its sides of each side's distance from the centre times its length). Its sides are the square's, each the share
    of it outside both bodies, and each line's chord through the square, the share of it outside the other's. Exact
    for two straight lines, continuous in both; with one line beyond the square, the other's ``covered``."""
    planes = (_outside(a, size), _outside(b, size))
    outside = jnp.zeros_like(planes[0][2])
    for (mu, mv), (eu, ev) in _SIDES:
        sides = [(nu * mu + nv * mv - c, nu * eu + nv * ev) for nu, nv, c in planes]
        outside = outside + 0.25 * _both(sides[0], sides[1])
    for (nu, nv, c), (ou, ov, oc) in (planes, planes[::-1]):
        length, mu, mv = _chord(nu, nv, c)
        kept = _share(ou * mu + ov * mv - oc, length * (ov * nu - ou * nv))
        outside = outside - 0.5 * c * length * kept
    return 1.0 - outside


#: The most terms a composition's union may have for its exact share (``composed_covered``): its inclusion and exclusion
#: takes every nonempty subset of them.
MOST_TERMS = 6


class _Terms(NamedTuple):
    """A composition of half-planes as a union of intersections: each term the set of the pieces it intersects."""

    terms: tuple[frozenset[int], ...]


def terms_of(compose: Any, count: int) -> tuple[tuple[int, ...], ...]:
    """The composition ``compose(pieces, lesser, greater)`` of ``count`` pieces as a union of intersections of their
    half-planes (``lesser`` a union, ``greater`` an intersection), each term the pieces it intersects: a term holding
    another is dropped (the union holds it already), and so is a repeated one. Read once, when a form is made."""
    def union(a: _Terms, b: _Terms) -> _Terms:
        return _Terms(a.terms + b.terms)

    def intersection(a: _Terms, b: _Terms) -> _Terms:
        return _Terms(tuple(x | y for x in a.terms for y in b.terms))

    found = compose([_Terms((frozenset((i,)),)) for i in range(count)], union, intersection).terms
    kept: list[frozenset[int]] = []
    for term in sorted(set(found), key=lambda t: (len(t), sorted(t))):
        if not any(held <= term for held in kept):
            kept.append(term)
    if len(kept) > MOST_TERMS:
        raise ValueError(f"a composition of {len(kept)} terms: the exact share takes at most {MOST_TERMS}")
    return tuple(tuple(sorted(term)) for term in kept)


def _kept(middle: Any, slope: Any, later: bool = False, beyond: Any = None) -> tuple[Any, Any]:
    """Where on a segment (from -1/2 to 1/2 of its length) a linear function is at least zero, given its middle's value
    and its change along the segment: the interval from the end it keeps, ``_share`` of the segment long. Where the
    change is within ``UNION_FLOOR`` of zero and the function is zero to it too (two lines that coincide), the share is
    a half, kept from the end ``later`` names (the later of the two lines the start, the earlier the end), so that two
    coinciding lines' chords keep the two halves of one chord between them. ``beyond``: on a side of the cell, the
    line's distance from the cell's centre; a side on the line is taken as just outside the cell, kept whole where the
    body lies beyond it (the distance below zero) and not at all where the body holds the cell, so that with the line's
    whole chord (``_chord``, closed) it counts the side once, or cancels it."""
    if beyond is not None:
        middle = jnp.where(jnp.abs(slope) > UNION_FLOOR, middle, middle - UNION_FLOOR * beyond)
    share = _share(middle, slope)
    high = jnp.where(jnp.abs(slope) > UNION_FLOOR, slope > 0.0, later)
    low_end = switch.where(high, 0.5 - share, -0.5, owner="camera.composed.end",
                           resolution=lambda: switch.rounding(1.0), indicator=slope)
    high_end = switch.where(high, 0.5, share - 0.5, owner="camera.composed.end",
                            resolution=lambda: switch.rounding(1.0), indicator=slope)
    return low_end, high_end


def _within(bounds: list[tuple[Any, Any]]) -> Any:
    """The length, as a share of its segment, of the intersection of the intervals ``bounds`` on it."""
    low, high = bounds[0]
    for lower, upper in bounds[1:]:
        low_now, high_now = low, high
        low = switch.maximum(low_now, lower, owner="camera.composed.low", resolution=lambda: switch.rounding(1.0),
                             indicator=lambda: low_now - lower)
        high = switch.minimum(high_now, upper, owner="camera.composed.high", resolution=lambda: switch.rounding(1.0),
                              indicator=lambda: upper - high_now)
    return switch.maximum(high - low, 0.0, owner="camera.composed.empty", resolution=lambda: switch.rounding(1.0),
                          indicator=high - low)


def composed_covered(terms: tuple[tuple[int, ...], ...], lines: list[Line], size: float = 1.0) -> Any:
    """The exact share of a square ``size`` pixels a side, centred where the lines were read, inside a composition of
    the lines' bodies given as a union of intersections (``terms``, from ``terms_of``): by inclusion and exclusion, the
    signed sum of the areas of the intersections of every nonempty subset of the terms. Each is a convex polygon whose
    area is its boundary integral (half the sum over its sides of each side's distance from the centre times its
    length); its sides are the square's, each the share of it inside every body of the subset, and each line's chord
    through the square, the share of it inside the others'. Exact for straight lines, continuous in each; two lines that
    coincide (to ``UNION_FLOOR``) keep a half of their chord each, and a line on one of the square's sides keeps its whole
    chord while that side is counted by the side the body lies on (``_kept``'s ``beyond``)."""
    used = sorted({index for term in terms for index in term})
    planes = {index: _outside(lines[index], size) for index in used}
    side_bounds: dict[int, list[tuple[Any, Any]]] = {index: [] for index in used}
    for (mu, mv), (eu, ev) in _SIDES:
        for index in used:
            nu, nv, c = planes[index]
            side_bounds[index].append(_kept(c - nu * mu - nv * mv, -(nu * eu + nv * ev), beyond=c))
    chords = {index: _chord(*planes[index], closed=True) for index in used}
    chord_bounds: dict[tuple[int, int], tuple[Any, Any]] = {}
    for j in used:
        length, mu, mv = chords[j]
        nu, nv, _ = planes[j]
        for k in used:
            if k != j:
                ou, ov, oc = planes[k]
                chord_bounds[j, k] = _kept(oc - ou * mu - ov * mv, -length * (ov * nu - ou * nv), j > k)
    total = jnp.zeros_like(planes[used[0]][2])
    for chosen in range(1, 2 ** len(terms)):
        subset = [term for bit, term in enumerate(terms) if chosen >> bit & 1]
        held = sorted({index for term in subset for index in term})
        area = jnp.zeros_like(total)
        for side in range(len(_SIDES)):
            area = area + 0.25 * _within([side_bounds[index][side] for index in held])
        for j in held:
            others = [chord_bounds[j, k] for k in held if k != j]
            length = chords[j][0]
            kept = _within(others) if others else jnp.ones_like(length)
            area = area + 0.5 * planes[j][2] * length * kept
        total = total + (area if len(subset) % 2 else -area)
    return total


def near(edge: Line) -> Any:
    """Whether the edge's line crosses the pixel."""
    return jnp.abs(edge.distance) < CORNER_REACH


def sub_offsets(samples: int) -> list[tuple[float, float]]:
    """The centres of a pixel's ``samples x samples`` sub-pixels, in pixels from its centre."""
    steps = [(i + 0.5) / samples - 0.5 for i in range(samples)]
    return [(du, dv) for dv in steps for du in steps]


def _segment(px: Any, py: Any, qx: Any, qy: Any) -> Any:
    """The signed area between the chord from P to Q (both on the unit circle, or the same point) and the arc between
    them: ``(phi - sin phi) / 2``, ``phi`` the signed angle from P to Q, by its series where ``phi`` is small (the
    difference would cancel)."""
    phi = jnp.arctan2(px * qy - py * qx, px * qx + py * qy)
    p2 = phi * phi
    series = phi * p2 * (1.0 / 6.0 - p2 * (1.0 / 120.0 - p2 * (1.0 / 5040.0 - p2 / 362880.0)))
    return 0.5 * switch.where(jnp.abs(phi) < 0.5, series, phi - jnp.sin(phi), owner="camera.disc.series",
                              resolution=lambda: switch.rounding(1.0), indicator=lambda: 0.5 - jnp.abs(phi))


def _on_circle(x: Any, y: Any) -> tuple[Any, Any]:
    length = jnp.sqrt(jnp.maximum(x * x + y * y, jnp.finfo(jnp.result_type(x)).tiny))
    return x / length, y / length


def _edge_in_disc(ax: Any, ay: Any, bx: Any, by: Any, cx: Any, cy: Any) -> Any:
    """The edge AB's share of the signed area of its polygon inside the unit disc, as a closed chain of pieces: the
    part of AB inside the disc, and the arcs its parts outside it subtend, each piece a triangle on ``C`` (a point
    near the polygon, so that no large area cancels) plus, for an arc, the segment between its chord and the circle.
    An end outside the disc enters the chain as its projection onto the circle, where the neighbouring edge takes it
    up, so the chain closes and the triangles on ``C`` sum to the area whatever ``C`` is."""
    ex, ey = bx - ax, by - ay
    a = ex * ex + ey * ey
    b = ax * ex + ay * ey
    c = ax * ax + ay * ay - 1.0
    discriminant = b * b - a * c
    real = (discriminant > 0.0) & (a > 0.0)
    root = jnp.sqrt(switch.domain_guard(real, discriminant, 1.0))
    held = switch.domain_guard(real, a, 1.0)
    low, high = (-b - root) / held, (-b + root) / held
    crosses = real & (high > 0.0) & (low < 1.0)
    first = switch.clip(low, 0.0, 1.0, owner="camera.disc.enter", resolution=lambda: switch.rounding(1.0),
                        indicator=lambda: low)
    second = switch.clip(high, 0.0, 1.0, owner="camera.disc.exit", resolution=lambda: switch.rounding(1.0),
                         indicator=lambda: high)
    a_in, b_in = c < 0.0, bx * bx + by * by < 1.0
    hax, hay = _on_circle(ax, ay)
    hbx, hby = _on_circle(bx, by)

    sx, sy = (switch.where(a_in, ax, hax, owner="camera.disc.start", resolution=lambda: switch.rounding(1.0),
                           indicator=lambda: -c),
              switch.where(a_in, ay, hay, owner="camera.disc.start", resolution=lambda: switch.rounding(1.0),
                           indicator=lambda: -c))
    tx, ty = (switch.where(b_in, bx, hbx, owner="camera.disc.end", resolution=lambda: switch.rounding(1.0),
                           indicator=lambda: 1.0 - bx * bx - by * by),
              switch.where(b_in, by, hby, owner="camera.disc.end", resolution=lambda: switch.rounding(1.0),
                           indicator=lambda: 1.0 - bx * bx - by * by))
    # The chord's margin in the segment's own fraction: the half-chord in it where the line meets the circle, minus the
    # square root of how far it misses it where it does not, the segment's end past the chord's start and its start
    # before the chord's end; the least of them is positive where the segment crosses the circle.
    def reach() -> Any:
        margin = jnp.sign(discriminant) * jnp.sqrt(jnp.abs(discriminant)) / held
        return jnp.minimum(jnp.minimum(margin, high), 1.0 - low)

    def entered(inside: Any, outside: Any) -> Any:
        return switch.where(crosses, inside, outside, owner="camera.disc.crossing",
                            resolution=lambda: switch.rounding(1.0), indicator=reach)

    p1x, p1y = entered(ax + first * ex, sx), entered(ay + first * ey, sy)
    p2x, p2y = entered(ax + second * ex, sx), entered(ay + second * ey, sy)

    def triangle(ux: Any, uy: Any, vx: Any, vy: Any) -> Any:
        return 0.5 * ((ux - cx) * (vy - cy) - (uy - cy) * (vx - cx))

    return (triangle(sx, sy, p1x, p1y) + _segment(sx, sy, p1x, p1y) + triangle(p1x, p1y, p2x, p2y)
            + triangle(p2x, p2y, tx, ty) + _segment(p2x, p2y, tx, ty))


def disc(corners_x: Any, corners_y: Any, centre: Any, radius: Any) -> Any:
    """The share of each pixel inside a ball's silhouette: ``corners_x``, ``corners_y`` the undistorted normalised
    coordinates of the pixels' corners (``(height + 1, width + 1)``, or a window's), ``centre`` the ball's centre in
    the optical frame and ``radius`` its radius. The ball is wholly in front of the camera (its centre's depth above its
    radius), where its silhouette is an ellipse."""
    distance = jnp.sqrt(centre[0] * centre[0] + centre[1] * centre[1] + centre[2] * centre[2])
    ux, uy, uz = centre[0] / distance, centre[1] / distance, centre[2] / distance
    sine2 = (radius / distance) ** 2
    # The cone's conic in (x, y, 1), over the centre's squared distance: (x ux + y uy + uz)^2 - cos^2 (x^2 + y^2 + 1)
    # > 0 inside, cos^2 = 1 - sine2. Each coefficient is written with |u| = 1 so that no difference of near-equal
    # numbers is formed (``uz^2 - cos^2`` is ``sine2 - ux^2 - uy^2``): exact disc coverage holds in float32.
    m11, m12, m22 = sine2 - uy * uy - uz * uz, ux * uy, sine2 - ux * ux - uz * uz
    # Its centre and its Schur complement in closed form (with cos^2 = 1 - sine2: det M2 = cos^2 (uz^2 - sine2) and
    # det M = sine2 cos^4), so that no sum cancels.
    across = uz * uz - sine2
    x0, y0 = ux * uz / across, uy * uz / across
    k = sine2 * (1.0 - sine2) / across
    p11, p12, p22 = -m11 / k, -m12 / k, -m22 / k
    l11 = jnp.sqrt(p11)
    l21 = p12 / l11
    l22 = jnp.sqrt(p22 - l21 * l21)
    dx, dy = corners_x - x0, corners_y - y0
    yx, yy = l11 * dx + l21 * dy, l22 * dy
    quad = [(yx[:-1, :-1], yy[:-1, :-1]), (yx[:-1, 1:], yy[:-1, 1:]), (yx[1:, 1:], yy[1:, 1:]), (yx[1:, :-1], yy[1:, :-1])]
    centre_x = 0.25 * (quad[0][0] + quad[1][0] + quad[2][0] + quad[3][0])
    centre_y = 0.25 * (quad[0][1] + quad[1][1] + quad[2][1] + quad[3][1])
    inside = sum((_edge_in_disc(*quad[i], *quad[(i + 1) % 4], centre_x, centre_y) for i in range(4)),
                 start=jnp.zeros(()))
    area = 0.5 * sum(((quad[i][0] - centre_x) * (quad[(i + 1) % 4][1] - centre_y)
                      - (quad[i][1] - centre_y) * (quad[(i + 1) % 4][0] - centre_x) for i in range(4)),
                     start=jnp.zeros(()))
    # A pixel wholly inside the disc is covered exactly (the ellipse is convex); one with no corner inside and no
    # side crossing the circle is clear, or holds the whole disc.
    corners = [qx * qx + qy * qy < 1.0 for qx, qy in quad]
    full = corners[0] & corners[1] & corners[2] & corners[3]
    crossed = sum((jnp.where(_crosses(*quad[i], *quad[(i + 1) % 4]), 1.0, 0.0) for i in range(4)), start=jnp.zeros(()))
    clear = ~(corners[0] | corners[1] | corners[2] | corners[3]) & (crossed == 0)
    holds = _holds_origin(quad)
    def smallest(values: list[Any]) -> Any:
        return jnp.minimum(jnp.minimum(values[0], values[1]), jnp.minimum(values[2], values[3]))

    def margins() -> list[Any]:
        """Each corner's distance outside the unit circle, in the circle's units squared (negative inside)."""
        return [qx * qx + qy * qy - 1.0 for qx, qy in quad]

    held = switch.where(holds, jnp.pi / switch.domain_guard(holds, area, 1.0), 0.0, owner="camera.disc.holds",
                        resolution=lambda: switch.rounding(1.0), indicator=lambda: _origin_margin(quad))
    shut = switch.where(clear, held, inside / area, owner="camera.disc.clear",
                        resolution=lambda: switch.rounding(1.0),
                        indicator=lambda: jnp.minimum(smallest(margins()), 1.0 - crossed))
    share = switch.where(full, 1.0, shut, owner="camera.disc.full", resolution=lambda: switch.rounding(1.0),
                         indicator=lambda: -jnp.maximum(jnp.maximum(margins()[0], margins()[1]),
                                                        jnp.maximum(margins()[2], margins()[3])))
    return jnp.clip(share, 0.0, 1.0)


def _crosses(ax: Any, ay: Any, bx: Any, by: Any) -> Any:
    """Whether the segment AB meets the unit circle."""
    ex, ey = bx - ax, by - ay
    a = ex * ex + ey * ey
    b = ax * ex + ay * ey
    c = ax * ax + ay * ay - 1.0
    discriminant = b * b - a * c
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    held = switch.domain_guard(a > 0.0, a, 1.0)
    first, second = (-b - root) / held, (-b + root) / held
    return (discriminant > 0.0) & (((first >= 0.0) & (first <= 1.0)) | ((second >= 0.0) & (second <= 1.0)))


def _origin_margin(quad: list[tuple[Any, Any]]) -> Any:
    """How far inside the quadrilateral the origin lies, in the cross products of its sides (the least of them, or
    its negative where they run the other way: the origin is held where it is positive)."""
    sides = jnp.stack([quad[i][0] * quad[(i + 1) % 4][1] - quad[i][1] * quad[(i + 1) % 4][0] for i in range(4)])
    return jnp.maximum(jnp.min(sides, axis=0), jnp.min(-sides, axis=0))


def _holds_origin(quad: list[tuple[Any, Any]]) -> Any:
    """Whether a convex quadrilateral holds the origin: the origin on one side of each of its sides."""
    sides = [quad[i][0] * quad[(i + 1) % 4][1] - quad[i][1] * quad[(i + 1) % 4][0] for i in range(4)]
    return ((sides[0] > 0) & (sides[1] > 0) & (sides[2] > 0) & (sides[3] > 0)) | \
        ((sides[0] < 0) & (sides[1] < 0) & (sides[2] < 0) & (sides[3] < 0))


def sigmoid(edge: Line, size: float = 1.0, width: float = 1.0) -> Any:
    """The soft edge the design replaced, ``sigma(-d / w)`` at ``w`` pixels: the box edge gate's control."""
    return jax.nn.sigmoid(-edge.distance / (width * size))
