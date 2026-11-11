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

from mime.core._node import jnp

#: A pixel is a corner pixel where two of a body's pieces pass within this many pixels of its centre (half its
#: diagonal: a line nearer than that crosses the square).
CORNER_REACH = 0.7072
#: Sub-pixels a side in a corner pixel.
CORNER_SAMPLES = 4
#: Below this gradient length (in the edge function's units per pixel) a pixel's edge is taken as hard: the edge
#: function is flat there (a ray through a centre, where its edge is far).
FLAT = 1e-30


def square_below(a: Any, b: Any, c: Any) -> Any:
    """The area of the unit square centred on the origin where ``a x + b y < c``, for ``a >= b >= 0`` and
    ``a^2 + b^2 = 1``: zero below the far corner, a triangle up to the near side, then linear to the centre, and
    symmetric beyond it."""
    b = jnp.maximum(b, 1e-9)
    corner, flat = (a + b) / 2.0, (a - b) / 2.0

    def lower(c: Any) -> Any:
        triangle = (c + corner) ** 2 / (2.0 * a * b)
        return jnp.where(c <= -corner, 0.0, jnp.where(c <= -flat, triangle, 0.5 + c / a))

    return jnp.where(c <= 0.0, lower(jnp.minimum(c, 0.0)), 1.0 - lower(jnp.minimum(-c, 0.0)))


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
    return Line(*(jnp.where(first, x, y) for x, y in zip(a, b, strict=True)))


def greater(a: Line, b: Line) -> Line:
    """The farther-out of two pieces' lines (an intersection's)."""
    first = a.distance >= b.distance
    return Line(*(jnp.where(first, x, y) for x, y in zip(a, b, strict=True)))


#: A piece lying inside another's region (a cap inside its bounding ball) is read only where the bound's line is within
#: this many pixels: beyond it the inner piece's outline cannot reach the pixel (half its diagonal, and a margin for
#: the bound's own first-order distance).
BOUND_REACH = 1.5
#: The distance a piece is given where it cannot reach the pixel: far outside.
OUTSIDE_PX = 1e6


def beyond(inner: Line, bound: Line) -> Line:
    """``inner``'s line where ``bound``'s is within ``BOUND_REACH`` pixels, else far outside (``OUTSIDE_PX``): a piece
    whose outline lies inside another's region cannot cross a pixel beyond that region."""
    far = bound.distance > BOUND_REACH
    return Line(jnp.where(far, OUTSIDE_PX, inner.distance), jnp.where(far, 1.0, inner.nu), jnp.where(far, 0.0, inner.nv))


def covered(edge: Line, size: float = 1.0) -> Any:
    """The share of a square ``size`` pixels a side, centred where the line was read, on the body's side of it."""
    au, av = jnp.abs(edge.nu), jnp.abs(edge.nv)
    return square_below(jnp.maximum(au, av), jnp.minimum(au, av), -edge.distance / size)


def box(g: Any, gu: Any, gv: Any, size: float = 1.0) -> Any:
    """The share of a square ``size`` pixels a side, centred where ``g`` is evaluated, on the body's side of the edge."""
    return covered(line(g, gu, gv), size)


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
    return 0.5 * jnp.where(jnp.abs(phi) < 0.5, series, phi - jnp.sin(phi))


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
    root = jnp.sqrt(jnp.where(real, discriminant, 1.0))
    held = jnp.where(real, a, 1.0)
    low, high = (-b - root) / held, (-b + root) / held
    crosses = real & (high > 0.0) & (low < 1.0)
    first, second = jnp.clip(low, 0.0, 1.0), jnp.clip(high, 0.0, 1.0)
    a_in, b_in = c < 0.0, bx * bx + by * by < 1.0
    hax, hay = _on_circle(ax, ay)
    hbx, hby = _on_circle(bx, by)
    sx, sy = jnp.where(a_in, ax, hax), jnp.where(a_in, ay, hay)
    tx, ty = jnp.where(b_in, bx, hbx), jnp.where(b_in, by, hby)
    p1x, p1y = jnp.where(crosses, ax + first * ex, sx), jnp.where(crosses, ay + first * ey, sy)
    p2x, p2y = jnp.where(crosses, ax + second * ex, sx), jnp.where(crosses, ay + second * ey, sy)

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
    share = jnp.where(full, 1.0, jnp.where(clear, jnp.where(holds, jnp.pi / jnp.where(holds, area, 1.0), 0.0),
                                           inside / area))
    return jnp.clip(share, 0.0, 1.0)


def _crosses(ax: Any, ay: Any, bx: Any, by: Any) -> Any:
    """Whether the segment AB meets the unit circle."""
    ex, ey = bx - ax, by - ay
    a = ex * ex + ey * ey
    b = ax * ex + ay * ey
    c = ax * ax + ay * ay - 1.0
    discriminant = b * b - a * c
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    held = jnp.where(a > 0.0, a, 1.0)
    first, second = (-b - root) / held, (-b + root) / held
    return (discriminant > 0.0) & (((first >= 0.0) & (first <= 1.0)) | ((second >= 0.0) & (second <= 1.0)))


def _holds_origin(quad: list[tuple[Any, Any]]) -> Any:
    """Whether a convex quadrilateral holds the origin: the origin on one side of each of its sides."""
    sides = [quad[i][0] * quad[(i + 1) % 4][1] - quad[i][1] * quad[(i + 1) % 4][0] for i in range(4)]
    return ((sides[0] > 0) & (sides[1] > 0) & (sides[2] > 0) & (sides[3] > 0)) | \
        ((sides[0] < 0) & (sides[1] < 0) & (sides[2] < 0) & (sides[3] < 0))


def sigmoid(edge: Line, size: float = 1.0, width: float = 1.0) -> Any:
    """The soft edge the design replaced, ``sigma(-d / w)`` at ``w`` pixels: the box edge gate's control."""
    return jax.nn.sigmoid(-edge.distance / (width * size))
