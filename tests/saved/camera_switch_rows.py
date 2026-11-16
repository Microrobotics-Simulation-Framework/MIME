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
import mime.sensing._sampled as sampled_route
from mime.core import switch
from mime.sensing import Appearance, CameraPose, Pinhole, Subject
from mime.testing import SwitchRow
from mime.units import um

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

    def sampled_ray(slope):
        # The march along a ray whose first-order distance only rises (the stand-in's slope, no dips): its least
        # sample is the first, where the parabola's vertex is not taken; no spacing crosses the surface; and there is
        # no second minimum, so the second piece is the first moved out.
        sampled = sampled_route.Sampled(_Profile({}, slope), "solid", 1.0, 1.0, _Profile.SAMPLES)  # type: ignore[arg-type]
        o = (jnp.asarray(0.0), jnp.asarray(-1e3), jnp.asarray(0.0))
        d = (jnp.asarray([0.0]), jnp.asarray([1.0]), jnp.asarray([0.0]))
        held = sampled_route.march(sampled, {}, o, d, (jnp.asarray([0.0]),) * 3)
        first, second = sampled_route.pieces(sampled, held)({}, o, d)
        return {"first": first, "second": second, "chord": sampled_route.chord(sampled, {}, o, d, held)}

    def directionless(distance):
        # A line of no direction (an edge function flat to the floor) beside a line across the cell.
        return {"share": coverage.union_covered(coverage.Line(jnp.asarray(1e15), jnp.asarray(0.0), jnp.asarray(0.0)),
                                                coverage.Line(distance, jnp.asarray(0.6), jnp.asarray(0.8)))}

    def creased_rays(shift):
        # A capped body's least's place moving with its derivative (``_moving``, which its creased pieces read their
        # regions at), on a ray along its axis beside it, where the side's profile is flat (no strict minimum: the
        # place is held, the implicit step's divisor guarded), and on one from the same pinhole through its centre;
        # its derivative in the pinhole's place across the axis.
        import dataclasses

        from mime.scene import Cylinder
        from mime.scene._creases import capped
        from mime.units import um

        kind = Cylinder(radius=um(300.0), length=um(2000.0), axis=(0.0, 0.0, 1.0))
        parameters = {"radius": jnp.asarray(300e-6), "length": jnp.asarray(2000e-6), "axis": jnp.asarray((0.0, 0.0, 1.0))}
        sampled = dataclasses.replace(sampled_route.Sampled(kind, "side", 300e-6, 1.044e-3, 32), regions=("span",),
                                      creases=4, compose=capped)
        o = (400e-6 + shift, jnp.asarray(0.0), jnp.asarray(-5e-3))
        across = math.hypot(400e-6, 5e-3)
        d = (jnp.asarray([0.0, -400e-6 / across]), jnp.asarray([0.0, 0.0]), jnp.asarray([1.0, 5e-3 / across]))
        at = jnp.asarray([5e-3, across])
        return {"place": sampled_route._moving(sampled, parameters, o, d, at, jnp.asarray([6.5e-5, 6.5e-5]))}

    slope = jnp.asarray(_Profile.SLOPE)
    at_distance = jnp.asarray(0.2)
    shift = jnp.asarray(0.0)
    return {"the camera's creased least's place on a ray along a side, its profile flat": (
                lambda: creased_rays(shift), lambda: jax.jacfwd(creased_rays)(shift)),
            "the camera's union of two branches, one of no direction": (
                lambda: directionless(at_distance), lambda: jax.jacfwd(directionless)(at_distance)),
            "the camera's sampled march along a ray whose distance only rises": (
                lambda: sampled_ray(slope), lambda: jax.jacfwd(sampled_ray)(slope)),
            "the camera's exact disc over pixels some of which the ball does not touch": (
                lambda: window(centre), lambda: jax.jacfwd(window)(centre)),
            "the camera's crossing test on a side of no length": (lambda: degenerate(at),
                                                                  lambda: jax.jacfwd(degenerate)(at))}


# -- route 2, the sampled march -------------------------------------------------------------------------------------

#: The helix the kinds' rows evaluate.
COIL = {"radius": 0.4e-3, "pitch": 0.6e-3, "turns": 2.5, "filament_radius": 0.08e-3}


class _Profile:
    """A stand-in kind for the march's rows: along the ray ``y`` (from -1 to 1 across the enclosing ball of radius one,
    looked at from far along +y), its first-order distance is a gentle slope with narrow dips centred on chosen
    samples, so that each owner's indicator is placed exactly: ``dips`` maps a sample's index to how far below the
    slope its dip reaches."""

    SAMPLES, SLOPE, WIDTH = 41, 1e-2, 0.2

    def __init__(self, dips: dict[int, float], slope: float = SLOPE):
        self.dips, self.slope = dips, slope

    def at(self, k: float) -> float:
        return -1.0 + 2.0 * k / (self.SAMPLES - 1)

    def sign_function(self, parameters, region, point):
        from mime.registry import SignValue

        y = point[1]
        k = (y + 1.0) * (self.SAMPLES - 1) / 2.0
        value = 1.0 + self.slope * k
        for index, depth in self.dips.items():
            value = value - depth * jnp.exp(-(((k - index) / self.WIDTH) ** 2))
        return SignValue(value, jnp.ones_like(value))


def _profile_marched(dips: dict[int, float], slope: float = _Profile.SLOPE):
    """What the pixel reads of the march along the stand-in's ray: its two pieces' values."""
    stand_in = _Profile(dips, slope)
    sampled = sampled_route.Sampled(stand_in, "solid", 1.0, 1.0, _Profile.SAMPLES)  # type: ignore[arg-type]
    o = (jnp.asarray(0.0), jnp.asarray(-1e3), jnp.asarray(0.0))
    d = (jnp.asarray([0.0]), jnp.asarray([1.0]), jnp.asarray([0.0]))
    held = sampled_route.march(sampled, {}, o, d, (jnp.asarray([0.0]),) * 3)
    first, second = sampled_route.pieces(sampled, held)({}, o, d)
    return shared(jnp.concatenate([first, second]))


def sampled_rows(declared) -> list[SwitchRow]:
    """The march's owners on the stand-in's profile, each indicator placed by a dip's depth: ``least`` two dips
    the offset apart; ``basin`` a second dip the offset below the lower of its neighbours (a slope keeps every other
    sample's margin at the slope's step; falling, so that every sample a basin, its branch forced, puts the second
    piece at the far end, and none at the near); ``second`` three dips, the second and the third the offset apart;
    ``branch`` the second dip of ``basin``, whose margin is the indicator."""
    deep, step = 1.5, _Profile.SLOPE
    return [
        SwitchRow(declared["camera.sampled.least"], lambda o: _profile_marched({10: deep, 20: deep - o + 10 * step}),
                  1e-3),
        SwitchRow(declared["camera.sampled.basin"], lambda o: _profile_marched({10: deep, 20: step + o}, -step), 1e-3),
        SwitchRow(declared["camera.sampled.second"],
                  lambda o: _profile_marched({10: deep, 20: 0.5, 30: 0.5 + 10 * step - o}), 1e-3),
        SwitchRow(declared["camera.sampled.branch"], lambda o: _profile_marched({10: deep, 20: step + o}), 1e-3),
    ]


def sign_rows(declared) -> list[SwitchRow]:
    """The kinds' sign functions' own switches: ``helix.end.low`` and ``.high``, a point whose nearest angle on the
    helix lies the offset inside the arc from an end (half the filament's radius from that angle's point, towards the
    axis), its value."""
    from mime.scene import Helix
    from mime.scene._helix import HelixTurn

    kind = Helix(radius=um(400.0), pitch=um(600.0), turns=2.5, filament_radius=um(80.0))
    parameters = {name: jnp.asarray(value) for name, value in COIL.items()}
    turn = HelixTurn({name: float(value) for name, value in COIL.items()})
    centroid = np.asarray(turn.centroid(), dtype=np.float64)

    def value(o, end):
        theta = o if end == "low" else float(turn.sweep) - o
        centre = np.array([float(turn.b) * (theta - 0.5 * float(turn.sweep)), float(turn.radius) * math.cos(theta),
                           float(turn.radius) * math.sin(theta)])
        point = centre + 0.5 * COIL["filament_radius"] * np.array([0.0, -math.cos(theta), -math.sin(theta)]) - centroid
        found = kind.sign_function(parameters, "solid", tuple(jnp.asarray([c]) for c in point))
        return {"sign_function": found.value}

    from mime.scene._creases import cap

    held = {"radius": jnp.asarray(300e-6), "length": jnp.asarray(2000e-6), "axis": jnp.asarray((0.0, 0.0, 1.0))}

    def cap_side(o):
        """The upper cap's piece (``mime.scene._creases.cap``, as the cylinder's measure makes it) on a ray from a pinhole
        the offset (m) above its plane, rising across it."""
        origin = (jnp.asarray([150e-6]), jnp.asarray([-400e-6]), jnp.asarray([1000e-6 + o]))
        direction = (jnp.asarray([0.0]), jnp.asarray([0.8]), jnp.asarray([0.6]))
        return {"crease_measure": cap(origin, direction, held["axis"], held["length"] / 2.0,
                                      lambda vx, vy, vz: held["radius"], "cylinder.cap_side", held["radius"])}

    return [*(SwitchRow(declared[f"helix.end.{end}"], lambda o, end=end: value(o, end), 1e-3) for end in ("low", "high")),
            SwitchRow(declared["cylinder.cap_side"], cap_side, 1e-4)]


# -- route 2's exact union of two branches ----------------------------------------------------------------------------

#: A line at 30 degrees to the cell's u axis, the slabs' coefficients of its distance along it (in cells).
UNION_ANGLE = math.radians(30.0)


def union_rows(declared) -> list[SwitchRow]:
    """The union's owners, each on the helper that evaluates it, its indicator placed: ``share``'s ratio at the offset
    from zero (``low``) or one (``high``); ``nested`` two shares keeping the same end, the second the offset larger;
    ``apart`` two at opposite ends overlapping by the offset; ``ends`` the second change the offset (its share then
    one, so the branches agree: it measures nothing); the chord's ends and its emptiness on a line at 30 degrees, its
    distance placing each (the indicator linear in it)."""
    one = jnp.asarray(1.0)

    def share(ratio):
        return shared(coverage._share(jnp.asarray(ratio - 0.5), one))

    def both(first, second):
        return shared(coverage._both(*(tuple(jnp.asarray(v) for v in part) for part in (first, second))))

    nu, nv = math.cos(UNION_ANGLE), math.sin(UNION_ANGLE)
    across = nu / nv + nv / nu

    def chord(c):
        return shared(jnp.stack(coverage._chord(jnp.asarray(nu), jnp.asarray(nv), jnp.asarray(c))))

    start_at = (0.5 / nv - 0.5 / nu) / across
    empty_at = (0.5 / nu + 0.5 / nv) / across
    return [
        SwitchRow(declared["camera.union.share.low"], share, 1e-3),
        SwitchRow(declared["camera.union.share.high"], lambda o: share(1.0 - o), 1e-3),
        SwitchRow(declared["camera.union.nested"], lambda o: both((-0.1, 1.0), (-0.1 + o, 1.0)), 1e-3),
        SwitchRow(declared["camera.union.apart"], lambda o: both((0.0, 1.0), (o, -1.0)), 1e-3),
        SwitchRow(declared["camera.union.ends"], lambda o: both((-0.1, 1.0), (0.3, o)), 1e-3),
        SwitchRow(declared["camera.union.chord_start"], lambda o: chord(start_at + o / across), 1e-3),
        SwitchRow(declared["camera.union.chord_end"], lambda o: chord(-start_at - o / across), 1e-3),
        SwitchRow(declared["camera.union.chord_empty"], lambda o: chord(empty_at - o / across), 1e-3),
    ]


def composed_rows(declared) -> list[SwitchRow]:
    """A creased body's owners, each on the helper that evaluates it: ``end`` a function along a segment whose change
    is the offset (its middle held, so its share is one either side: it measures nothing, as declared); ``low``,
    ``high`` and ``empty`` two intervals' intersection whose start, end, or length the offset places; ``body`` a
    cylinder's first-order distance at a point whose side's lies the offset (m) beyond its span's."""
    import dataclasses

    import mime.sensing._sampled as sampled_route
    from mime.scene import Cylinder
    from mime.units import um

    def kept(o):
        return shared(jnp.stack(coverage._kept(jnp.asarray(0.3), jnp.asarray(o))))

    def within(bounds):
        return shared(coverage._within([tuple(jnp.asarray(v) for v in bound) for bound in bounds]))

    kind = Cylinder(radius=um(300.0), length=um(2000.0), axis=(0.0, 0.0, 1.0))
    held = sampled_route.Sampled(kind, "side", 300e-6, 1.044e-3, 32)
    held = dataclasses.replace(held, regions=("span",))
    parameters = {"radius": jnp.asarray(300e-6), "length": jnp.asarray(2000e-6), "axis": jnp.asarray((0.0, 0.0, 1.0))}

    def body(o):
        point = (jnp.asarray([360e-6]), jnp.asarray([0.0]), jnp.asarray([1060e-6 - o]))
        return {"expected": sampled_route.body(held, parameters, *point)}

    return [
        SwitchRow(declared["camera.composed.end"], kept, 1e-3),
        SwitchRow(declared["camera.composed.low"], lambda o: within([(-0.2, 0.4), (-0.2 - o, 0.4)]), 1e-3),
        SwitchRow(declared["camera.composed.high"], lambda o: within([(-0.2, 0.3), (-0.2, 0.3 + o)]), 1e-3),
        SwitchRow(declared["camera.composed.empty"], lambda o: within([(-0.2, 0.3), (0.3 - o, 0.4)]), 1e-3),
        SwitchRow(declared["camera.sampled.body"], body, 1e-3),
    ]


def camera_rows(declared) -> list[SwitchRow]:
    return [*edge_rows(declared), *pieces_rows(declared), corner_row(declared), *disc_rows(declared),
            *form_rows(declared), *sampled_rows(declared), *union_rows(declared), *composed_rows(declared)]


__all__ = ["camera_rows"]
