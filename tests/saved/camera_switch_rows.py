"""The camera's switch rows (``mime.sensing._switches``), for ``test_switch_harness.py``: each owner a quarter of a step from its
switch, on the renderer's own function that evaluates it, at one pixel or one ray.

A row evaluates the function itself, never a frame: the harness needs every evaluation of an owner at the row's offset,
and a frame evaluates most of the renderer's functions at every pixel (and the pixel integration's twice, once for
each side of the line). The output is the function's value, under the group ``expected``: the camera's differentiable
output reads each of these values linearly or through a saturating step (the share of a pixel, a piece's edge function,
an interval's end), so a jump or a kink in the value is one in ``expected`` wherever that step is not saturated, and each
row is placed where it is not. The corner choice alone is a frame's: its row renders one pixel of a cylinder's corner
through ``render``, the unrolled sub-samples and the select a collector open puts in the cond's place, and its output is
that pixel's ``expected``.

Each row's placement is analytic (a line at a distance from the pixel's centre, a ray at a parameter, a corner at a
margin inside the unit circle), except the corner choice's, which a secant on the margin places."""

from __future__ import annotations

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

import mime.sensing._coverage as coverage
import mime.sensing._forms as forms
import mime.sensing._render as render_module
from mime.core import switch
from mime.sensing import Appearance, CameraPose, Pinhole, Subject
from mime.testing import SwitchRow

#: A normal's components, with ``a^2 + b^2 = 1``: the pixel square's corner reach is ``(a + b) / 2`` and its flat ``(a - b) / 2``.
A, B = 0.8, 0.6
CORNER, FLAT = (A + B) / 2, (A - B) / 2


def shared(value):
    return {"expected": jnp.asarray(value)}


# -- the box edge ---------------------------------------------------------------------------------------------------

def edge_rows(declared) -> list[SwitchRow]:
    """Each piece of the share of a pixel's square: the line's signed distance ``c`` from the centre (positive on the
    body's side), placed so that the owner's indicator is the offset."""

    def share(c):
        return shared(coverage.square_below(jnp.asarray(A), jnp.asarray(B), jnp.asarray(c)))

    return [
        SwitchRow(declared["camera.edge.out_far"], lambda o: share(-CORNER - o), 1e-3),
        SwitchRow(declared["camera.edge.out_near"], lambda o: share(-FLAT - o), 1e-3),
        SwitchRow(declared["camera.edge.in_far"], lambda o: share(CORNER + o), 1e-3),
        SwitchRow(declared["camera.edge.in_near"], lambda o: share(FLAT + o), 1e-3),
        SwitchRow(declared["camera.edge.centre"], lambda o: share(-o), 1e-3),
    ]


def pieces_rows(declared) -> list[SwitchRow]:
    """Two pieces' lines at the distance ``D`` from the pixel's centre with different normals, the second's farther by
    the offset (the nearer taken by the least, the farther by the greatest, so the first is the one the indicator
    names): one along the pixel's axis and one at a slant, where the share differs by about a thirtieth."""
    distance = 0.3

    def least(o):
        return shared(coverage.covered(coverage.lesser(coverage.Line(jnp.asarray(distance), jnp.asarray(1.0),
                                                                     jnp.asarray(0.0)),
                                                       coverage.Line(jnp.asarray(distance + o), jnp.asarray(A),
                                                                     jnp.asarray(B)))))

    def greatest(o):
        return shared(coverage.covered(coverage.greater(coverage.Line(jnp.asarray(distance + o), jnp.asarray(1.0),
                                                                      jnp.asarray(0.0)),
                                                        coverage.Line(jnp.asarray(distance), jnp.asarray(A),
                                                                      jnp.asarray(B)))))

    return [SwitchRow(declared["camera.pieces.least"], least, 1e-3),
            SwitchRow(declared["camera.pieces.greatest"], greatest, 1e-3)]


# -- the exact disc coverage -----------------------------------------------------------------------------------------

def edge_of(ax, ay, bx, by, cx=0.2, cy=0.5):
    return shared(coverage._edge_in_disc(*(jnp.asarray(v, dtype=jnp.float64) for v in (ax, ay, bx, by, cx, cy))))


def segment(o):
    """A chord's area between P on the unit circle and Q the angle ``0.5 - o`` from it."""
    phi = 0.5 - o
    return shared(coverage._segment(jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(math.cos(phi)),
                                    jnp.asarray(math.sin(phi))))


def whole_pixel_of_a_ball_on_the_axis(low, high, ball=1e-3, depth=0.05):
    """``coverage.disc`` of one pixel with corners ``low`` and ``high`` (in units of the silhouette's radius, the
    pixel's two corner coordinates along each axis) of a ball of radius ``ball`` at ``depth`` on the optical axis,
    where the conic is a circle of radius ``tan(asin(ball / depth))`` in the normalised plane."""
    rho = math.tan(math.asin(ball / depth))
    xs = jnp.asarray([[low[0], high[0]], [low[0], high[0]]]) * rho
    ys = jnp.asarray([[low[1], low[1]], [high[1], high[1]]]) * rho
    return shared(coverage.disc(xs, ys, jnp.asarray([0.0, 0.0, depth]), jnp.asarray(ball)))


def disc_rows(declared) -> list[SwitchRow]:
    # A side from A inside the circle by the offset (squared distance) out through it; and the same from the far end.
    def start(o):
        return edge_of(math.sqrt(1.0 - o), 0.0, 0.0, 2.0)

    def end(o):
        return edge_of(0.0, -2.0, math.sqrt(1.0 - o), 0.0)

    def crossing(o):
        x0 = math.sqrt(1.0 - 4.0 * o * o)
        return edge_of(x0, -1.0, x0, 1.0)

    # A side along y = 0.3, off the circle's centre (a side through it puts every piece of the chain on one line, and the
    # branches' areas equal), meeting the circle at x = +-0.954.
    y0 = 0.3
    xc = math.sqrt(1.0 - y0 * y0)

    def horizontal(ax, bx):
        return edge_of(ax, y0, bx, y0)

    def enter_low(o):  # the line enters the circle the fraction ``o`` along the side (it starts outside, near the circle)
        return horizontal(-(xc + 3.0 * o) / (1.0 - o), 3.0)

    def enter_high(o):  # the side ends just after entering: the entry's fraction is ``1 - o``
        return horizontal(-2.0, -xc + o * (2.0 - xc) / (1.0 - o))

    def exit_low(o):  # the side starts inside the circle just before its exit: the exit's fraction is ``o``
        return horizontal(xc - o * (3.0 - xc) / (1.0 - o), 3.0)

    def exit_high(o):  # the side ends just outside the circle: the exit's fraction is ``1 - o``
        return horizontal(-2.0, xc + o * (2.0 + xc) / (1.0 - o))

    # The pixels of a circle of radius ``tan(asin(1/50))`` (in the pixel's own units, ``rho``): a whole pixel inside
    # whose farthest corner lies ``o`` inside the circle (squared); a pixel outside whose nearest corner lies ``o``
    # outside it; a pixel holding the circle's centre at the offset (in the cross products of its sides).
    side = 0.2

    def full(o):
        far = math.sqrt((1.0 - o) / 2.0)
        return whole_pixel_of_a_ball_on_the_axis((far - side, far - side), (far, far))

    def clear(o):
        near = math.sqrt((1.0 + o) / 2.0)
        return whole_pixel_of_a_ball_on_the_axis((near, near), (near + side, near + side))

    def holds(o):
        half = math.sqrt(o / 2.0)
        return whole_pixel_of_a_ball_on_the_axis((-half, -half), (half, half))

    return [
        SwitchRow(declared["camera.disc.series"], segment, 1e-2),
        SwitchRow(declared["camera.disc.start"], start, 1e-3),
        SwitchRow(declared["camera.disc.end"], end, 1e-3),
        SwitchRow(declared["camera.disc.crossing"], crossing, 1e-3),
        SwitchRow(declared["camera.disc.enter.low"], enter_low, 1e-3),
        SwitchRow(declared["camera.disc.enter.high"], enter_high, 1e-3),
        SwitchRow(declared["camera.disc.exit.low"], exit_low, 1e-3),
        SwitchRow(declared["camera.disc.exit.high"], exit_high, 1e-3),
        SwitchRow(declared["camera.disc.full"], full, 1e-3),
        SwitchRow(declared["camera.disc.clear"], clear, 1e-3),
        SwitchRow(declared["camera.disc.holds"], holds, 1e-3),
    ]


# -- the closed forms --------------------------------------------------------------------------------------------------

RADIUS, HALF_LENGTH = 1e-3, 2e-3
ROUND = {"radius": jnp.asarray(RADIUS), "length": jnp.asarray(2 * HALF_LENGTH)}


def vec(*components):
    return tuple(jnp.asarray(c, dtype=jnp.float64) for c in components)


def form_rows(declared) -> list[SwitchRow]:
    def ahead(o):
        """A ball's piece on a ray whose nearest approach to the ball's centre is the offset ahead of the pinhole: the
        pinhole ``o`` behind the centre's plane, a ray along z."""
        (piece,) = forms._sphere_pieces({"radius": jnp.asarray(RADIUS)}, vec(1.5 * RADIUS, 0.0, -o), vec(0.0, 0.0, 1.0))
        return shared(piece)

    def cap_side(o):
        """A cap in the plane z = 0 seen from the offset above it, the ray tilted so that its cone is not edge-on."""
        d = vec(0.3, 0.0, math.sqrt(1.0 - 0.09))
        return shared(forms._cap(jnp.asarray(0.0), jnp.asarray(RADIUS), vec(0.5 * RADIUS, 0.0, o), d))

    def span(o, sign):
        """The planes of a cylinder whose half-length is half the offset, along a ray with a unit z component (up for
        the least's indicator, down for the greatest's)."""
        low, high = forms._span(jnp.asarray(o / 2.0), vec(0.0, 0.0, 0.0), vec(0.0, 0.0, sign))
        return shared(jnp.stack([low, high]))

    tilt = vec(0.0, math.cos(0.6), math.sin(0.6))
    start = vec(0.0, -0.01, 0.0)

    def interval_at(end, o):
        """A cylinder's interval along a tilted ray across it, the span's plane at the offset from the side's end."""
        mid, _, half, _ = forms._side(jnp.asarray(RADIUS), start, tilt)
        sin = tilt[2]
        if end == "enter":
            centre = -HALF_LENGTH - sin * (mid - half - o)
        else:
            centre = HALF_LENGTH - sin * (mid + half + o)
        low, high = forms._cylinder_interval(ROUND, vec(0.0, -0.01, centre), tilt)
        return shared(jnp.stack([low, high]))

    def axial(o):
        """The span piece on a ray across the cylinder whose nearest approach to the axis is at the offset above the middle."""
        pieces = forms._cylinder_pieces(ROUND, vec(0.0, -0.01, o), vec(0.0, 1.0, 0.0))
        return shared(jnp.stack([pieces[3]]))

    return [
        SwitchRow(declared["camera.ahead"], ahead, 1e-4),
        SwitchRow(declared["camera.cap_side"], cap_side, 1e-4),
        SwitchRow(declared["camera.span.low"], lambda o: span(o, 1.0), 1e-3),
        SwitchRow(declared["camera.span.high"], lambda o: span(o, -1.0), 1e-3),
        SwitchRow(declared["camera.span.axial"], axial, 1e-3),
        SwitchRow(declared["camera.interval.enter"], lambda o: interval_at("enter", o), 1e-4),
        SwitchRow(declared["camera.interval.leave"], lambda o: interval_at("leave", o), 1e-4),
        SwitchRow(declared["camera.interval.start"],
                  lambda o: shared(jnp.stack(forms.ahead((jnp.asarray(o), jnp.asarray(1.0))))), 1e-3),
        SwitchRow(declared["camera.interval.stop"],
                  lambda o: shared(jnp.stack(forms.ahead((jnp.asarray(-1.0), jnp.asarray(o))))), 1e-3),
        SwitchRow(declared["camera.interval.chord"],
                  lambda o: shared(forms.chord((jnp.asarray(0.5), jnp.asarray(0.5 + o)))), 1e-3),
    ]


# -- the corner choice ---------------------------------------------------------------------------------------------------

SIDE = 5.0  # the rod's top side, in pixels above the pixel's centre (the pixel lies inside it)
WIDTH_PX = 30e-3 / 900.0  # a pixel's footprint at the cylinder, at focal 900 px and 30 mm
CYLINDER_RADIUS, CYLINDER_LENGTH = 0.31e-3, 2.07e-3
LENS = Pinhole(1, 1, 900.0, 900.0, 0.0, 0.0)
LOOKING = CameraPose(jnp.asarray([0.0, -30e-3, 0.0]), jnp.asarray([math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]))
ROD = Subject("rod", "cylinder", {"radius": CYLINDER_RADIUS, "length": CYLINDER_LENGTH, "axis": (1.0, 0.0, 0.0)},
              appearance=Appearance.opaque(), moving=True)


def one_pixel(end):
    """One pixel of a rod across the line of sight, its cap's end ``end`` metres along x from the pixel's centre and its
    top side 0.2 of a pixel above it: ``expected`` there."""
    centre = jnp.asarray([end - CYLINDER_LENGTH / 2.0, 0.0, SIDE * WIDTH_PX - CYLINDER_RADIUS])
    frame = render_module.render([ROD], LENS, LOOKING, {"rod": (centre, jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    return frame.expected


@cache
def corner_end(offset: float) -> float:
    """Where the rod's end places the corner choice's indicator at ``offset`` pixels: a margin in pixels, linear in the
    end's place while one line is the nearest, so a secant on two readings either side of the reach, twice."""

    def margin(end):
        with switch.collecting() as rows:
            one_pixel(end)
        return float(np.asarray(next(r for r in rows if r.owner == "camera.corner").indicator).ravel()[0])

    first, second = -0.8 * WIDTH_PX, -0.6 * WIDTH_PX
    m1, m2 = margin(first), margin(second)
    for _ in range(2):
        first, second = second, second + (offset - m2) * (second - first) / (m2 - m1)
        m1, m2 = m2, margin(second)
    return second


def corner_row(declared) -> SwitchRow:
    return SwitchRow(declared["camera.corner"], lambda o: shared(one_pixel(corner_end(round(o, 12)))), 1e-2)


def guard_points() -> dict:
    """Where the renderer's domain guards have their condition false, each with the derivative of what it evaluates:
    the exact disc over a window of pixels some of which the ball does not touch (a side missing the circle, a pixel
    not holding its centre), in its ball's centre; and a side of no length, which a pixel of no size has."""
    rho = math.tan(math.asin(0.02))
    steps = jnp.asarray([-2.0, -0.9, 0.3, 1.4, 2.5]) * rho
    xs, ys = jnp.meshgrid(steps, steps, indexing="xy")

    def window(centre):
        return {"share": coverage.disc(xs, ys, centre, jnp.asarray(1e-3))}

    centre = jnp.asarray([0.0, 0.0, 0.05])

    def degenerate(point):
        return {"crosses": jnp.where(coverage._crosses(point[0], point[1], point[0], point[1]), 1.0, 0.0)}

    at = jnp.asarray([0.4, 0.3])
    return {"the camera's exact disc over pixels some of which the ball does not touch": (
                lambda: window(centre), lambda: jax.jacfwd(window)(centre)),
            "the camera's crossing test on a side of no length": (lambda: degenerate(at),
                                                                  lambda: jax.jacfwd(degenerate)(at))}


def camera_rows(declared) -> list[SwitchRow]:
    return [*edge_rows(declared), *pieces_rows(declared), corner_row(declared), *disc_rows(declared),
            *form_rows(declared)]


__all__ = ["camera_rows"]
