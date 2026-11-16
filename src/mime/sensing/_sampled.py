"""Route 2, sampled: a body whose geometry kind declares a cheap sign function (``Geometry.sign_function``), imaged
without a closed form, by a march along each ray.

**The march.** Each ray's line through the body's enclosing ball, widened by ``MARGIN_PX`` pixels' footprints, is
sampled at ``samples`` evenly spaced points in one reduction (no loop): the marched function is the first-order
distance ``value / slope`` itself, never read at another function's least point (a Lipschitz bound marched and the
first-order distance read at its least sits off by a quarter of a pixel at any count). Its local minima among the
samples are found, the two least kept, and each refined by the parabola through its neighbours. The search is held:
at a least point the distance does not move with the point along the ray (its derivative there is zero), so its
value's derivative is the live evaluation's at the held point, exact to the first order.

**Two branches, one corner.** Each of the two minima is a piece in route 1's sense (the first-order distance at its
held point, its line in the pixel by ``jvp`` along the ray table), composed as a union (``lesser``). Where both lie
within ``CORNER_REACH`` of the pixel's centre, two branches of the outline cross the pixel (a body occluding itself,
seen as two separated minima along the ray: FL-9's lobes meet in concave notches of about 80 degrees, where one
straight edge misses a quarter of the pixel), and route 1's corner sub-samples take it, each the exact share of the
union of the two branches' half-planes, each branch read at its sub-ray's own least: the descent from its pixel's
held point at the march's spacing to a sample lower than both its neighbours (at a notch the least moves along the
sub-ray several spacings, and between the basins the held values are not convex), then the quartic's least about it.
The descent runs only on the pieces it moves, gathered (a few per cent of a frame's): every other piece's result is
the quartic about its held point, which is the descent's there. Where there is no second minimum, the second piece is
the first moved out by the enclosing radius, which never reaches the pixel.

**The reading verified.** The corner line's derivation rests on a premise: each least reaching a sub-cell is read.
Where a corner pixel's sub-ray reads both branches at one point (one merged into the other's basin, or both
descended into it), a dense window about the lost branch's held point (``VERIFY_STEPS`` a spacing) finds the basins
of its own that reach the sub-cell: none, the reading stands, verified; one, the lost branch is read there, verified;
more, the nearest is read and the others' share of the sub-cell is the pixel's bound, raised as the run flag
``corner_unverified`` (its pixel-frames: summed over an exposure's sub-frames) with ``corner_bound`` (the largest bound), never a refusal. The lost rays are a few
a frame, gathered.

**The chord** (for an attenuating body) is the samples' sign changes, each refined by linear interpolation: a body
thinner than the samples' spacing along a ray can be missed in its chord, never in its coverage. The interval is held
and read only by the flags.

**The sample count** is a declaration, ``SAMPLES`` (32, which camera truth 1 found within 1 % of the physical outline
in motion off the crease, at 160 x 120 to 640 x 480 on FL-9), raised for a slender body so that neighbouring samples
lie at most ``SPACING`` of its smallest feature apart (``samples_for``): a structural rule, never a measured time."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax

from mime.core import switch
from mime.core._node import jnp
from mime.registry import EntryKind, Geometry, Solid, implements, lookup, registered
from mime.sensing import _coverage as coverage
from mime.sensing import _forms as forms
from mime.sensing._forms import Form, Vec

lax: Any = jax.lax
#: JAX's jit and linearize, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_jit: Any = cast(Any, jax).jit
_linearize: Any = cast(Any, jax).linearize
_jvp: Any = cast(Any, jax).jvp

#: Samples a ray by default: camera truth 1's count, within 1 % of the physical silhouette in motion off the crease.
SAMPLES = 32
#: The most neighbouring samples may lie apart, as a share of a body's smallest feature (its least radius of
#: curvature): a slender body's count rises to keep it.
SPACING = 0.5
#: Pixels' footprints the march's enclosing ball is widened by, so that a ray grazing the body's outline from outside
#: is sampled where its distance is least.
MARGIN_PX = 2.0
#: Flops-equivalents a transcendental counts as in the structural cost estimate: the count that predicted the
#: renderer's time (camera optimisation 1).
TRANSCENDENTAL_WEIGHT = 20.0
#: Newton's steps on the quartic through a ray's five held values, from the parabola's least to the quartic's.
LEAST_STEPS = 2
#: Moves of a march spacing a branch's least may take from its held point, by descent, before the quartic. At a notch
#: of the outline a corner sub-ray's least lies up to several spacings from its pixel's held point: a branch's foot
#: runs along the filament ``(rho + r) / rho`` times as fast as the sub-ray moves across the image (``r`` the
#: filament's radius, ``rho`` its outline's), and between the two branches' basins the held values are not convex.
LEAST_REACH = 6
#: The share of a frame's pieces (two a ray) whose descents run together, as many at a time: a piece's descent moves
#: only where its held point is not the lowest of its three samples a spacing apart (under 3.4 % of the pieces of the
#: coils' frames camera PR 5a measured), and every other piece holds the quartic about its own point, which is what
#: the descent gives it. A buffer's size, not a bound: a frame with more takes more turns of it.
DESCENT_SHARE = 0.125
#: The dense window that verifies a corner sub-ray's reading where a branch is lost (both branches read at one point:
#: one merged into the other's basin, or both descended into it): samples a march spacing, and spacings each side of
#: the lost branch's held point (the descent's reach and one more). The premise guard reads four times as finely.
VERIFY_STEPS = 16
VERIFY_REACH = LEAST_REACH + 1
#: The profile's rounding in the window, in units in the last place of the ray's parameter: a ray's point is rounded
#: at the camera's distance, so two minima a ridge lower than this apart are one basin's rounding (float32's, about a
#: hundredth of a pixel at 512 px), not two branches.
VERIFY_FLOOR_ULPS = 4
#: The share of a frame's rays whose verifications run together, as many at a time: a buffer's size, not a bound (a
#: frame with more takes more turns of it).
VERIFY_SHARE = 1.0 / 2048
#: A corner sub-cell's reach (pixels), its half-diagonal: a line farther from its centre leaves it wholly on one side.
SUB_REACH = math.sqrt(2.0) / (2 * coverage.CORNER_SAMPLES)
#: Evaluations a ray costs besides its samples: each of the two pieces' held values (the three a spacing apart and the
#: quartic's two between), its share of the descents' (``LEAST_REACH`` moves and the quartic's two about where it
#: stops), and its value and its line's two derivatives.
PIECE_EVALUATIONS = 2 * (5 + 3) + 2 * DESCENT_SHARE * (LEAST_REACH + 2)
#: Evaluations a ray costs on a frame whose corner sub-samples run: every pixel's sub-pixels (``CORNER_SAMPLES``
#: squared, 16), each the two pieces' value and line. The verification's turns (``VERIFY_SHARE`` of the rays, each its
#: window's 225 values and lines) add about 0.6 % a turn, not counted.
CORNER_EVALUATIONS = 16 * PIECE_EVALUATIONS
#: The structural cost estimate above which the build warns, in flops-equivalents a frame: a declaration, about 33 to
#: 37 ms in float32 and 56 to 66 ms in float64 on a four-core runner at the rates G19 measured for route 2 through the
#: in-loop camera (27 to 30 and 15 to 18 G a second; camera PR 5a's predictions file holds the runs).
COST_LINE = 1.0e9


@dataclass(frozen=True)
class Sampled:
    """A body route 2 images: its geometry kind (made from its settings), the region it marches, its smallest
    feature's size and its enclosing radius (m), and the samples a ray. A body whose kind declares a crease measure
    (``Geometry.crease_measure``) also holds its other ``regions`` (read at each branch's least), its measure's count of
    pieces (``creases``) and its composition (``compose``)."""

    kind: Geometry
    region: str
    feature: float
    enclosing: float
    samples: int
    regions: tuple[str, ...] = ()
    creases: int = 0
    compose: Callable[..., Any] | None = None


def kind_named(name: str) -> type[Geometry] | None:
    """The newest registered version of the geometry kind ``name``, or ``None``."""
    found = [record for record in registered(EntryKind.GEOMETRY) if record.name == name]
    if not found:
        return None
    newest = max(found, key=lambda record: tuple(int(p) if p.isdigit() else 0 for p in record.version.split(".")))
    held = lookup(EntryKind.GEOMETRY, name, newest.version)
    return None if held is None else held.cls  # type: ignore[return-value]


def creased(kind: type[Geometry]) -> bool:
    """Whether a kind's body has creases: a solid whose body is the intersection of two or more regions."""
    return issubclass(kind, Solid) and len(kind.regions) >= 2


def has_sign(kind: type[Geometry]) -> bool:
    return implements(kind, "sign_function")


def samples_for(enclosing: float, feature: float) -> int:
    """The samples a ray for a body of ``enclosing`` radius whose smallest feature is ``feature``: ``SAMPLES``, or
    more where the march's longest line (the enclosing ball's diameter) would leave its samples more than ``SPACING``
    of the feature apart."""
    return max(SAMPLES, math.ceil(2.0 * enclosing / (SPACING * feature)) + 1)


def evaluate(sampled: Sampled, p: Mapping[str, Any], x: Any, y: Any, z: Any) -> Any:
    """The marched function: the region's first-order distance, ``value / slope``."""
    found = sampled.kind.sign_function(p, sampled.region, (x, y, z))
    return found.value / found.slope


def _region(sampled: Sampled, p: Mapping[str, Any], region: str, x: Any, y: Any, z: Any) -> Any:
    found = sampled.kind.sign_function(p, region, (x, y, z))
    return found.value / found.slope


def body(sampled: Sampled, p: Mapping[str, Any], x: Any, y: Any, z: Any) -> Any:
    """The body's first-order distance where it is creased: the greatest of its regions' (its sign is the body's), the
    marched region's where it is not."""
    found = evaluate(sampled, p, x, y, z)
    for region in sampled.regions:
        other = _region(sampled, p, region, x, y, z)
        held = found
        found = switch.maximum(held, other, owner="camera.sampled.body", resolution=lambda: switch.rounding(
            sampled.enclosing), indicator=lambda: held - other)
    return found


class March(NamedTuple):
    """The march's held result on each ray: the least sample's refined point and the least other local minimum's,
    whether there is another, the samples' points and spacing, which spacings lie inside the body, which it is entered
    or left in (each a held mask, one or zero), and the held interval the flags read; on a frame's corner sub-samples,
    which rays' pixels are corners and the ray table's derivatives along ``u`` and ``v`` (``Form.at_corners``)."""

    ta: Any
    tb: Any
    has2: Any
    t: Any
    dt: Any
    inside: Any
    entering: Any
    leaving: Any
    interval: tuple[Any, Any]
    corner: Any = None
    du: Any = None
    dv: Any = None
    interior: Any = None


def _margins(v: Any) -> Any:
    """How far each interior sample lies below the lower of its neighbours (positive at a local minimum)."""
    return jnp.minimum(v[..., :-2], v[..., 2:]) - v[..., 1:-1]


def _gap(values: Any) -> Any:
    """How far the least of each row lies below the next least (zero at a tie)."""
    low = jnp.sort(values, axis=-1)
    return low[..., 1] - low[..., 0]


def march(sampled: Sampled, p: Mapping[str, Any], o: Vec, d: Vec, du: Vec) -> March:
    """The held search along rays ``o + t d`` (``du`` the ray table's derivative along a pixel's row, for the
    footprint): the samples across the widened enclosing ball, the least sample (the first branch), the least of the
    other local minima (the second), each refined by the parabola through its neighbours."""
    o, d, du = lax.stop_gradient((o, d, du))
    held_p = {name: value if isinstance(value, int) else lax.stop_gradient(value) for name, value in p.items()}
    n = sampled.samples
    b = forms.dot(o, d)
    footprint = jnp.sqrt(forms.dot(o, o) * forms.dot(du, du))
    reach = sampled.enclosing + MARGIN_PX * footprint
    half = jnp.sqrt(jnp.maximum(b * b - (forms.dot(o, o) - reach * reach), 0.0))
    t0 = jnp.maximum(-b - half, 0.0)
    t1 = jnp.maximum(-b + half, t0)
    dt = (t1 - t0) / (n - 1)
    k = jnp.arange(n, dtype=t0.dtype)
    t = t0[..., None] + dt[..., None] * k
    v = evaluate(sampled, held_p, *(o[i] + t * d[i][..., None] for i in range(3)))

    def resolution() -> Any:
        return switch.rounding(reach)

    first = switch.argmin(v, owner="camera.sampled.least", resolution=resolution, indicator=lambda: _gap(v))
    margins = _margins(v)
    others = jnp.where(jnp.arange(1, n - 1) == first[..., None], -jnp.inf, margins)
    basins = switch.where(margins > 0.0, v[..., 1:-1], jnp.inf, owner="camera.sampled.basin", resolution=resolution,
                          indicator=lambda: margins)
    rest = jnp.where(jnp.arange(1, n - 1) == first[..., None], jnp.inf, basins)
    second = switch.argmin(rest, owner="camera.sampled.second", resolution=resolution, indicator=lambda: _gap(rest)) + 1
    has2 = switch.predicate(jnp.max(others, axis=-1) > 0.0, owner="camera.sampled.branch", resolution=resolution,
                            indicator=lambda: jnp.max(others, axis=-1))

    def refined(at: Any) -> Any:
        def near(offset: int) -> Any:
            return jnp.take_along_axis(v, (at + offset)[..., None], axis=-1, mode="clip")[..., 0]

        before, least, after = near(-1), near(0), near(1)
        curvature = before - 2.0 * least + after
        inner = (at > 0) & (at < n - 1) & (curvature > 0.0)
        shift = 0.5 * (before - after) / switch.domain_guard(inner, curvature, 1.0)
        return t0 + (at.astype(t0.dtype) + jnp.where(inner, shift, 0.0)) * dt

    ta = refined(first)
    tb = refined(second)
    if sampled.regions:
        v = body(sampled, held_p, *(o[i] + t * d[i][..., None] for i in range(3)))
    entering = (v[..., :-1] >= 0.0) & (v[..., 1:] < 0.0)
    leaving = (v[..., :-1] < 0.0) & (v[..., 1:] >= 0.0)
    inside = (v[..., :-1] < 0.0) & (v[..., 1:] < 0.0)
    across = t[..., :-1] + dt[..., None] * v[..., :-1] / jnp.where(entering | leaving, v[..., :-1] - v[..., 1:], 1.0)
    interval = (jnp.min(jnp.where(entering, across, forms.FAR), axis=-1),
                jnp.max(jnp.where(leaving, across, -forms.FAR), axis=-1))
    # A creased body's marched region reaches past its enclosing ball (a cylinder's side, a screw's thread): a least at
    # the ends of the samples is not the region's least along the ray, and the body is not there.
    interior = None if sampled.compose is None else (first > 0) & (first < n - 1)
    return March(ta, tb, has2, t, dt, inside.astype(v.dtype), entering.astype(v.dtype), leaving.astype(v.dtype),
                 interval, interior=interior)


def chord(sampled: Sampled, p: Mapping[str, Any], o: Vec, d: Vec, held: March) -> Any:
    """The length of each ray inside the body: each spacing between neighbouring samples that the march found inside,
    and where it found the body entered or left, the share inside by linear interpolation of the live values at the
    held samples. The masks are held: a sample crossing the surface moves the crossing from one spacing to its
    neighbour, where the interpolation puts it at the same place, so the chord is continuous across it."""
    v = (body if sampled.regions else evaluate)(sampled, p, *(o[i] + held.t * d[i][..., None] for i in range(3)))
    a, b = v[..., :-1], v[..., 1:]
    changes = (held.entering + held.leaving) > 0.0
    share = a / switch.domain_guard(changes, a - b, 1.0)
    within = held.inside + held.leaving * share + held.entering * (1.0 - share)
    return jnp.sum(within, axis=-1) * held.dt


def _descended(at: Any, t: Any, spacing: Any, near: tuple[Any, Any, Any] | None = None) -> tuple[Any, ...]:
    """The descent along a ray from ``t`` at the march's spacing towards its lower neighbour, at most
    ``LEAST_REACH`` moves: the sample it stops at, lower than both its neighbours (or the last it reached), its value,
    its neighbours' a spacing below and above it along the ray, the direction it went (-1 or 1), and whether it ran its
    whole reach still descending. Where the five values about a held point are not convex (between two branches'
    basins) a step of the quartic alone does not move; the descent does. ``near``: the values a spacing below, at and
    a spacing above ``t``, where already made."""
    before, middle, after = (at(t - spacing), at(t), at(t + spacing)) if near is None else near
    toward = jnp.where(before < after, -1.0, 1.0)
    path = [jnp.where(toward < 0.0, after, before), middle, jnp.where(toward < 0.0, before, after)]
    path += [at(t + toward * k * spacing) for k in range(2, LEAST_REACH + 2)]
    centre, value, behind, ahead = t, middle, path[0], path[2]
    going = jnp.ones_like(middle, dtype=bool)
    for k in range(1, LEAST_REACH + 1):
        going = going & (path[k + 1] < path[k])
        centre = jnp.where(going, t + toward * k * spacing, centre)
        value, behind, ahead = (jnp.where(going, path[k + 1], value), jnp.where(going, path[k], behind),
                                jnp.where(going, path[k + 2], ahead))
    ran = going & (path[LEAST_REACH + 2] < path[LEAST_REACH + 1])
    return (centre, value, jnp.where(toward < 0.0, ahead, behind), jnp.where(toward < 0.0, behind, ahead), toward,
            ran)


def _quartic_least(far_low: Any, low: Any, middle: Any, high: Any, far_high: Any) -> Any:
    """Where the least of five values half a spacing apart lies, in half spacings from the middle one: the quartic's
    least through them, by Newton's steps (``LEAST_STEPS``, each at most half a spacing) from the least of the
    parabola through the outer three; or the least of the inner three, where it is lower than the quartic's own value
    there (a profile the quartic does not fit: a kink in its curvature, where an arc's nearest point passes its end).
    The quartic takes out the parabola's error from the profile's asymmetry, whose effect on the derivative a held
    point leaves first order."""
    curvature = far_low - 2.0 * middle + far_high
    rising = curvature > 0.0
    at_least = jnp.where(rising, jnp.clip((far_low - far_high) / jnp.where(rising, curvature, 1.0), -2.0, 2.0), 0.0)
    c1 = (far_low - 8.0 * low + 8.0 * high - far_high) / 12.0
    c2 = (-far_low + 16.0 * low - 30.0 * middle + 16.0 * high - far_high) / 24.0
    c3 = (-far_low + 2.0 * low - 2.0 * high + far_high) / 12.0
    c4 = (far_low - 4.0 * low + 6.0 * middle - 4.0 * high + far_high) / 24.0
    for _ in range(LEAST_STEPS):
        slope = c1 + at_least * (2.0 * c2 + at_least * (3.0 * c3 + at_least * 4.0 * c4))
        turn = 2.0 * c2 + at_least * (6.0 * c3 + at_least * 12.0 * c4)
        convex = turn > 0.0
        at_least = jnp.clip(at_least + jnp.where(convex, jnp.clip(-slope / jnp.where(convex, turn, 1.0), -1.0, 1.0),
                                                 0.0), -2.0, 2.0)
    fitted = middle + at_least * (c1 + at_least * (c2 + at_least * (c3 + at_least * c4)))
    inner = jnp.where(low < high, -1.0, 1.0)
    return jnp.where(fitted <= jnp.minimum(middle, jnp.minimum(low, high)), at_least,
                     jnp.where(middle <= jnp.minimum(low, high), 0.0, inner))


def _gathered(at: Any, start: Any, other: Any, spacing: Any, near: tuple[Any, Any, Any],
              read: Any) -> tuple[Any, Any]:
    """The pieces' least points and merges (``_leasts``) with the descent gathered: each piece the quartic about its
    held point, and those whose lower neighbour a spacing away is below it (``read``: whose least is read at all)
    descend together, ``DESCENT_SHARE`` of the list a turn, until none is left. ``at(t, index)`` evaluates the pieces
    ``index`` (all where it is ``None``)."""
    before, middle, after = near
    point = start + 0.5 * _quartic_least(before, at(start - 0.5 * spacing), middle, at(start + 0.5 * spacing),
                                         after) * spacing
    merged = jnp.zeros(point.shape, dtype=bool)
    moves = (jnp.minimum(before, after) < middle) & read
    count = point.shape[0]
    width = max(1, math.ceil(DESCENT_SHARE * count))
    turns = -(-count // width)
    order = jnp.nonzero(moves, size=turns * width, fill_value=count)[0]
    moving = jnp.sum(moves)

    def turn(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        k, point, merged = state
        index = lax.dynamic_slice(order, (k * width,), (width,))

        def taken(values: Any) -> Any:
            return jnp.take(values, index, mode="clip")

        t, h = taken(start), taken(spacing)

        def along(s: Any) -> Any:
            return at(s, index)

        centre, value, below, above, toward, ran = _descended(along, t, h, (taken(before), taken(middle),
                                                                          taken(after)))
        moved = centre + 0.5 * _quartic_least(below, along(centre - 0.5 * h), value, along(centre + 0.5 * h),
                                              above) * h
        return (k + 1, point.at[index].set(moved, mode="drop"),
                merged.at[index].set(ran & (toward * (taken(other) - t) > 0.0), mode="drop"))

    def left(state: tuple[Any, Any, Any]) -> Any:
        return state[0] * width < moving

    _, point, merged = lax.while_loop(left, turn, (0, point, merged))
    return point, merged


def _leasts(sampled: Sampled, held: March, p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
    """Each branch's held point moved to its ray's least nearby (a corner sub-sample's, whose least is not its
    pixel's), on rays ``o + t d``: the descent at the march's spacing to the sample lower than both its neighbours
    (``_descended``), then the least of the quartic through the distance at five points half a spacing apart about it
    (``_quartic_least``); a branch merged into the other's basin (its descent ran its whole reach towards the other
    branch's held point: past the fold where its own least vanishes) read at the other's point; and, on a corner
    pixel's sub-ray (``March.corner``), the reading verified where a branch is lost (``_verified``). Returns the two
    read points and the bound on each ray's sub-cell (zero where verified). Held, as the march is: no derivative passes
    through it, and at the least the distance does not move with it.

    The two branches' pieces are one list. Every piece takes its five values about its held point; where the lower of
    its neighbours a spacing away is lower than it, the descent moves (otherwise it stops where it starts, and the
    quartic about the held point is its result), and those pieces descend together (``_gathered``): the descent's
    moves cost what the pieces that move need."""
    p = {name: value if isinstance(value, int) else lax.stop_gradient(value) for name, value in p.items()}
    o, d = lax.stop_gradient((o, d))
    shape = jnp.broadcast_shapes(jnp.shape(held.ta), jnp.shape(d[0]))
    rays = math.prod(shape)

    def listed(first: Any, second: Any) -> Any:
        return jnp.concatenate([jnp.broadcast_to(first, shape).reshape(-1),
                                jnp.broadcast_to(second, shape).reshape(-1)])

    start, other, spacing = listed(held.ta, held.tb), listed(held.tb, held.ta), listed(held.dt, held.dt)
    origin, direction = tuple(listed(c, c) for c in o), tuple(listed(c, c) for c in d)

    def at(t: Any, index: Any = None) -> Any:
        if index is None:
            return evaluate(sampled, p, *(origin[i] + t * direction[i] for i in range(3)))
        return evaluate(sampled, p, *(jnp.take(origin[i], index, mode="clip") + t * jnp.take(direction[i], index,
                                                                                          mode="clip")
                                      for i in range(3)))

    before, middle, after = at(start - spacing), at(start), at(start + spacing)
    if switch.collector_open():
        # A switch collector keeps what a loop's body traces beyond it, so while one is open every piece descends, as
        # the descent is written without the gathering: the same values (a piece that does not move holds the quartic
        # about its point), each switch the descent's evaluations hold recorded on the values it reads.
        centre, value, below, above, toward, ran = _descended(at, start, spacing, (before, middle, after))
        point = centre + 0.5 * _quartic_least(below, at(centre - 0.5 * spacing), value, at(centre + 0.5 * spacing),
                                              above) * spacing
        merged = ran & (toward * (other - start) > 0.0)
    else:
        point, merged = _gathered(at, start, other, spacing, (before, middle, after), listed(True, held.has2))
    has2 = jnp.broadcast_to(held.has2, shape).reshape(-1)
    a, b, a_merged, b_merged = point[:rays], point[rays:], merged[:rays], merged[rays:]
    # A branch merged into the other's basin is read at the other's point.
    first, second = jnp.where(has2 & a_merged & ~b_merged, b, a), jnp.where(b_merged, a, b)
    bound = jnp.zeros_like(first)
    if held.corner is not None:
        first, second, bound = _verified(sampled, held, p, o, d, shape, first, second, (a_merged, b_merged))
    first, second = lax.stop_gradient((first, second))
    return first.reshape(shape), second.reshape(shape), lax.stop_gradient(bound).reshape(shape)


def _ridges(values: Any, at: Any) -> Any:
    """The highest of each row's ``values`` between each place and the place ``at`` (one a row), both included."""
    index = jnp.arange(values.shape[1])[None, :]
    after = lax.cummax(jnp.where(index >= at[:, None], values, -jnp.inf), axis=1)
    before = lax.cummax(jnp.where(index <= at[:, None], values, -jnp.inf), axis=1, reverse=True)
    return jnp.where(index >= at[:, None], after, before)


def _verified(sampled: Sampled, held: March, p: Mapping[str, Any], o: Vec, d: Vec, shape: tuple[int, ...],
              first: Any, second: Any, merged: tuple[Any, Any], reads: bool = True) -> tuple[Any, Any, Any]:
    """The reading of a corner pixel's sub-rays verified where a branch is lost (both read at one point). The
    derivation's premise is that each least reaching a sub-cell (its line within ``SUB_REACH`` of the sub-cell's
    centre, the cell not wholly inside the read branch) is read. The lost branch's dense window (``VERIFY_STEPS`` a
    spacing, ``VERIFY_REACH`` spacings each side of its held point) finds the local minima that are basins of their
    own: beyond half a spacing of the read point and parted from it by a ridge above the profile's rounding
    (``VERIFY_FLOOR_ULPS`` of the ray's parameter). None reaching, the reading is verified as it is; one or more, the
    lost branch is read at the one whose line is nearest (its parabola at the window's spacing), verified where it is
    the only one; the share of the sub-cell inside each other's half-plane is the bound on what the cell's union
    leaves out (zero where verified). Where not ``reads`` (the flag's control) the least found is bounded, not read.
    The lost rays are gathered, ``VERIFY_SHARE`` of them a turn. Each of ``first``, ``second`` and ``merged`` is one
    a ray; returned with the bound, one a ray."""
    def flat(c: Any) -> Any:
        return jnp.broadcast_to(c, shape).reshape(-1)

    corner, has2, ta, tb, spacing = (flat(c) for c in (held.corner, held.has2, held.ta, held.tb, held.dt))
    origin, direction = tuple(flat(c) for c in o), tuple(flat(c) for c in d)
    along_u, along_v = tuple(flat(c) for c in held.du), tuple(flat(c) for c in held.dv)
    lost = corner & has2 & (jnp.abs(first - second) < 0.5 * spacing)
    # Which branch is lost: the merged one, or (both descended into one basin) the one whose held point is farther.
    second_lost = merged[1] | (~merged[0] & (jnp.abs(tb - second) >= jnp.abs(ta - first)))
    start = jnp.where(second_lost, tb, ta)
    count = start.shape[0]
    reach = VERIFY_REACH * VERIFY_STEPS
    steps = jnp.arange(-reach, reach + 1, dtype=start.dtype) / VERIFY_STEPS

    def lines_at(index: Any, t: Any) -> tuple[Any, coverage.Line]:
        def taken(c: Any) -> Any:
            return jnp.take(c, index, mode="clip")[:, None]

        def value(dx: Any, dy: Any, dz: Any) -> Any:
            return evaluate(sampled, p, *(taken(origin[i]) + t * (dx, dy, dz)[i] for i in range(3)))

        g, linear = _linearize(value, *(taken(c) for c in direction))
        return g, coverage.line(g, linear(*(taken(c) for c in along_u)), linear(*(taken(c) for c in along_v)))

    def verify(index: Any) -> tuple[Any, Any]:
        """The lost branch's point and the bound on the rays ``index``."""
        s, r, q, h = (jnp.take(c, index, mode="clip") for c in (start, first, second, spacing))
        t = s[:, None] + h[:, None] * steps[None, :]
        values, line = lines_at(index, t)
        _, read = lines_at(index, r[:, None])
        floor = VERIFY_FLOOR_ULPS * jnp.finfo(values.dtype).eps * jnp.abs(t)

        def parted(place: Any) -> Any:
            """Whether each sample is a basin apart from the point ``place``: beyond half a spacing, and the profile
            between them rising above both by more than its rounding (or the point beyond the window)."""
            at = jnp.round((place - s) / h * VERIFY_STEPS).astype(jnp.int32) + reach
            within = (at >= 0) & (at <= 2 * reach)
            at = jnp.clip(at, 0, 2 * reach)
            ridge = _ridges(values, at) - jnp.maximum(values, jnp.take_along_axis(values, at[:, None], axis=1))
            return (jnp.abs(t - place[:, None]) > 0.5 * h[:, None]) & ((ridge > floor) | ~within[:, None])

        low = jnp.concatenate([jnp.zeros_like(values[:, :1], dtype=bool),
                               (values[:, 1:-1] < values[:, :-2]) & (values[:, 1:-1] <= values[:, 2:]),
                               jnp.zeros_like(values[:, :1], dtype=bool)], axis=1)
        reaching = low & parted(r) & (jnp.abs(t - q[:, None]) > 0.5 * h[:, None]) & (line.distance < SUB_REACH) \
            & (read.distance >= -SUB_REACH)
        nearest = jnp.argmin(jnp.where(reaching, line.distance, jnp.inf), axis=1)

        def near(offset: int) -> Any:
            return jnp.take_along_axis(values, jnp.clip(nearest + offset, 0, 2 * reach)[:, None], axis=1)[:, 0]

        before, least, after = near(-1), near(0), near(1)
        curvature = before - 2.0 * least + after
        shift = jnp.where(curvature > 0.0, 0.5 * (before - after) / jnp.where(curvature > 0.0, curvature, 1.0), 0.0)
        found = jnp.take_along_axis(t, nearest[:, None], axis=1)[:, 0] + jnp.clip(shift, -1.0, 1.0) * h / VERIFY_STEPS
        any_found = jnp.any(reaching, axis=1)
        others = reaching & parted(found) if reads else reaching
        share = coverage.covered(line, 1.0 / coverage.CORNER_SAMPLES)
        bound = jnp.minimum(jnp.sum(jnp.where(others, share, 0.0), axis=1), 1.0)
        return jnp.where(any_found & reads, found, r), bound

    if switch.collector_open():
        # A switch collector keeps what a loop's body traces beyond it, so while one is open every ray is verified at
        # once, as the verification is written without the gathering: the same values on the lost rays.
        point, bound = verify(jnp.arange(count))
    else:
        width = max(1, math.ceil(VERIFY_SHARE * count))
        order = jnp.nonzero(lost, size=-(-count // width) * width, fill_value=count)[0]

        def turn(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            k, point, bound = state
            index = lax.dynamic_slice(order, (k * width,), (width,))
            found, left = verify(index)
            return k + 1, point.at[index].set(found, mode="drop"), bound.at[index].set(left, mode="drop")

        def pending(state: tuple[Any, Any, Any]) -> Any:
            return state[0] * width < jnp.sum(lost)

        _, point, bound = lax.while_loop(pending, turn, (0, first, jnp.zeros_like(first)))
    bound = jnp.where(lost, bound, 0.0)
    return jnp.where(lost & ~second_lost, point, first), jnp.where(lost & second_lost, point, second), bound


def _moving(sampled: Sampled, p: Mapping[str, Any], o: Vec, d: Vec, at: Any, spacing: Any) -> Any:
    """A branch's held least ``at`` along the rays with the derivative of the least's place: one Newton step on the
    marched function's slope along the ray, its value taken off (so the place is the held one), its derivative the
    implicit function's. The step's curvature is the profile's second difference at the march's ``spacing``, held: at a
    kink of the profile (a ray through the region's medial set, a cylinder's axis) the least's place moves with the
    kink, and the curvature a spacing wide keeps the step's derivative finite there; where it is not convex the place is
    held."""
    def along(t: Any) -> Any:
        return evaluate(sampled, p, *(o[i] + t * d[i] for i in range(3)))

    live = _jvp(along, (at,), (jnp.ones_like(at),))[1]
    curvature = lax.stop_gradient((along(at + spacing) - 2.0 * along(at) + along(at - spacing)) / (spacing * spacing))
    convex = curvature > 0.0
    step = (live - lax.stop_gradient(live)) / switch.domain_guard(convex, curvature, 1.0)
    return at - switch.domain_guard(convex, step, jnp.zeros_like(step))


def _creased_pieces(sampled: Sampled, held: March) -> Any:
    """A creased body's pieces on rays: the two branches' (as ``pieces``), each branch's other regions' first-order
    distances at its least (``Sampled.regions``, the least's place moving with its derivative, ``_moving``), and the
    crease measure's pieces; on a frame's corner sub-samples followed by the bound. A second branch read at the
    first's point is moved out by the enclosing radius, as one that is not there: the two would repeat one term of the
    composition."""

    def made(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
        def at(t: Any) -> Any:
            return evaluate(sampled, p, *(o[i] + t * d[i] for i in range(3)))

        at_first, at_second, bound = _leasts(sampled, held, p, o, d)
        first, second = at(at_first), at(at_second)
        second = jnp.where(held.has2 & (at_second != at_first), second, first + sampled.enclosing)
        # The least at the ends of the samples lies past the enclosing ball's margin, two footprints from the body.
        first = jnp.where(held.interior, first, first + 2.0 * sampled.enclosing)
        regions: list[Any] = []
        for place in (at_first, at_second):
            moving = _moving(sampled, p, o, d, place, held.dt)
            regions += [_region(sampled, p, region, *(o[i] + moving * d[i] for i in range(3)))
                        for region in sampled.regions]
        found = sampled.kind.crease_measure(p, o, d)
        made_pieces = (first, second, *regions, *found.pieces)
        return made_pieces if held.corner is None else (*made_pieces, bound)

    return made


def creased_compose(sampled: Sampled) -> Any:
    """A creased body's composition of its pieces (``_creased_pieces``' order): its crease measure's, given each
    branch's regions and the creases."""
    count = len(sampled.regions)
    compose = sampled.compose
    assert compose is not None

    def composed(parts: Sequence[Any], lesser: Any, greater: Any) -> Any:
        marched = [(parts[0], *parts[2:2 + count]), (parts[1], *parts[2 + count:2 + 2 * count])]
        return compose(marched, parts[2 + 2 * count:2 + 2 * count + sampled.creases], lesser, greater)

    return composed


def pieces(sampled: Sampled, held: March) -> Any:
    """The two branches' pieces on rays: each the first-order distance at its ray's own least near its held point
    (``_leasts``); the second, where there is no second minimum, the first moved out by the enclosing radius; on a
    frame's corner sub-samples (``March.corner``) followed by the bound where the reading is not verified."""
    if sampled.compose is not None:
        return _creased_pieces(sampled, held)

    def made(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
        def at(t: Any) -> Any:
            return evaluate(sampled, p, *(o[i] + t * d[i] for i in range(3)))

        at_first, at_second, bound = _leasts(sampled, held, p, o, d)
        first, second = at(at_first), at(at_second)
        # With no second minimum, the second is the first moved out by the enclosing radius.
        second = jnp.where(held.has2, second, first + sampled.enclosing)
        return (first, second) if held.corner is None else (first, second, bound)

    return made


def _union(parts: Sequence[Any], lesser: Any, greater: Any) -> Any:
    return lesser(parts[0], parts[1])


def _covered_union(lines: Sequence[Any], size: float) -> Any:
    """A cell's share of the two branches: the exact share of the union of their lines' half-planes, so that a cell
    both cross (a concave notch, its sliver) is right, and continuous where one branch's line overtakes the other's."""
    return coverage.union_covered(lines[0], lines[1], size)


def form(sampled: Sampled, held: March | None = None) -> Form:
    """Route 2's form: with ``held``, the march done once on these rays, its pieces at the held points (for the
    coverage and its corner sub-samples); without, a form whose interval marches on the rays it is given (a sampled
    body as another subject's parent)."""

    def interval(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, Any]:
        zero = jnp.zeros_like(d[0])
        return march(sampled, p, o, d, (zero, zero, zero)).interval

    def enclosing(settings: Mapping[str, float]) -> float:
        return sampled.enclosing

    if sampled.compose is not None:
        return _creased_form(sampled, held, interval, enclosing)
    if held is None:
        def unmarched(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
            raise TypeError("a sampled form's pieces are made from its march (form(sampled, held))")

        return Form(interval, unmarched, enclosing, _union, corners=True, cover=_covered_union)
    if held.corner is not None:
        return Form(interval, pieces(sampled, held), enclosing, _union, corners=True, cover=_covered_union,
                    unverified=1)

    def at_corners(corner: Any, du: Vec, dv: Vec) -> Form:
        return form(sampled, held._replace(corner=corner, du=du, dv=dv))

    return Form(interval, pieces(sampled, held), enclosing, _union, corners=True, cover=_covered_union,
                at_corners=at_corners)


def _creased_form(sampled: Sampled, held: March | None, interval: Any, enclosing: Any) -> Form:
    """A creased body's form: its pieces (``_creased_pieces``), composed by its crease measure, each cell the exact
    share of that composition of the pieces' half-planes (``coverage.composed_covered``), the creases' pieces taken as
    turning (a cap's outline turns within a pixel where it is seen near edge-on)."""
    compose = creased_compose(sampled)
    count = 2 + 2 * len(sampled.regions) + sampled.creases
    terms = coverage.terms_of(compose, count)
    turning = tuple(range(2 + 2 * len(sampled.regions), count))

    def cover(lines: Sequence[Any], size: float) -> Any:
        return coverage.composed_covered(terms, list(lines), size)

    if held is None:
        def unmarched(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
            raise TypeError("a sampled form's pieces are made from its march (form(sampled, held))")

        return Form(interval, unmarched, enclosing, compose, corners=True, turning=turning, cover=cover)
    if held.corner is not None:
        return Form(interval, pieces(sampled, held), enclosing, compose, corners=True, turning=turning, cover=cover,
                    unverified=1)

    def at_corners(corner: Any, du: Vec, dv: Vec) -> Form:
        return form(sampled, held._replace(corner=corner, du=du, dv=dv))

    return Form(interval, pieces(sampled, held), enclosing, compose, corners=True, turning=turning, cover=cover,
                at_corners=at_corners)


def faces(sampled: Sampled, p: Mapping[str, Any]) -> tuple[tuple[Any, Any, Any], ...]:
    """A creased body's planar faces (``CreaseMeasure.faces``), in its kind's frame."""
    zero = jnp.zeros(())
    return sampled.kind.crease_measure(p, (zero, zero, zero), (zero, zero, jnp.ones(()))).faces


def evaluation_cost(sampled: Sampled, parameters: Mapping[str, Any], dtype: Any) -> float:
    """The flops-equivalents of one evaluation of the marched function (flops plus ``TRANSCENDENTAL_WEIGHT`` times
    the transcendentals), from XLA's cost analysis of the lowered program on one point: a count of the program, never
    a time."""
    point = jax.ShapeDtypeStruct((), dtype)
    held = {name: (value if isinstance(value, int) else jnp.asarray(value, dtype=dtype))
            for name, value in parameters.items()}
    def one(x: Any, y: Any, z: Any) -> Any:
        return evaluate(sampled, held, x, y, z)

    cost: Mapping[str, float] = _jit(one).lower(point, point, point).cost_analysis() or {}
    return float(cost.get("flops", 0.0)) + TRANSCENDENTAL_WEIGHT * float(cost.get("transcendentals", 0.0))


def _counted(function: Any, *shapes: Any) -> float:
    cost: Mapping[str, float] = _jit(function).lower(*shapes).cost_analysis() or {}
    return float(cost.get("flops", 0.0)) + TRANSCENDENTAL_WEIGHT * float(cost.get("transcendentals", 0.0))


def crease_cost(sampled: Sampled, parameters: Mapping[str, Any], dtype: Any) -> tuple[float, float]:
    """A creased body's counts besides its marched function's, in flops-equivalents: its crease measure's pieces and its
    other regions on one ray, and the exact share of its composition in one cell (``coverage.composed_covered``)."""
    point = jax.ShapeDtypeStruct((), dtype)
    held = {name: (value if isinstance(value, int) else jnp.asarray(value, dtype=dtype))
            for name, value in parameters.items()}

    def measured(ox: Any, oy: Any, oz: Any, dx: Any, dy: Any, dz: Any) -> Any:
        found = sampled.kind.crease_measure(held, (ox, oy, oz), (dx, dy, dz)).pieces
        regions = [_region(sampled, held, region, ox, oy, oz) for region in sampled.regions]
        return sum((*found, *regions), start=jnp.zeros((), dtype))

    count = 2 + 2 * len(sampled.regions) + sampled.creases
    terms = coverage.terms_of(creased_compose(sampled), count)

    def covered(values: Any) -> Any:
        lines = [coverage.Line(values[3 * k], values[3 * k + 1], values[3 * k + 2]) for k in range(count)]
        return coverage.composed_covered(terms, lines, 1.0 / coverage.CORNER_SAMPLES)

    return (_counted(measured, *([point] * 6)),
            _counted(covered, jax.ShapeDtypeStruct((3 * count,), dtype)))


#: Evaluations of the marched function a creased body adds to each set of its pieces (a ray's, and each corner
#: sub-sample's): each branch's least's place moving with its derivative, its slope along the ray (two) and its second
#: difference (three, ``_moving``).
CREASE_EVALUATIONS = 10


def frame_cost(sampled: Sampled, parameters: Mapping[str, Any], rays: int, sub_frames: int, dtype: Any) -> float:
    """The structural cost estimate of a body's frame, in flops-equivalents: its rays (its window's) times the
    sub-frames times one evaluation's count times the evaluations a ray takes (its samples, its pieces, and the corner
    sub-samples, counted as running: a frame with a corner in view runs them on every ray); for a creased body, each set
    of pieces' crease measure, other regions and moving leasts, and each corner sub-sample's exact share of its
    composition."""
    one = evaluation_cost(sampled, parameters, dtype)
    per_ray = (sampled.samples + PIECE_EVALUATIONS + CORNER_EVALUATIONS) * one
    if sampled.compose is not None:
        measure, cover = crease_cost(sampled, parameters, dtype)
        sub_cells = coverage.CORNER_SAMPLES ** 2
        per_ray += (1 + sub_cells) * (CREASE_EVALUATIONS * one + measure) + sub_cells * cover
    return float(rays) * sub_frames * per_ray


def made_kind(kind: type[Geometry], settings: Mapping[str, Any]) -> Geometry:
    """The kind made from its settings in SI (each dimensioned one a quantity, a list a tuple)."""
    import dataclasses

    from mime.registry import registration_of
    from mime.units import Quantity

    record = registration_of(kind)
    made: dict[str, Any] = {}
    for item in dataclasses.fields(cast(Any, kind)):
        value: Any = settings[item.name]
        value = tuple(cast(list[Any], value)) if isinstance(value, list) else value
        declared = record.settings.get(item.name) if record is not None else None
        made[item.name] = Quantity(value, declared.dimension) if declared is not None and declared.dimension is not None \
            else value
    return kind(**made)


def smallest_length(kind: type[Geometry], settings: Mapping[str, Any]) -> float:
    """The smallest of a kind's length settings (each component of a vector one): the structural stand-in for its
    smallest feature that sets the samples a ray (a sphere's or capsule's radius, a helix's filament, an ellipsoid's
    least semi-axis)."""
    from mime.registry import registration_of
    from mime.units import Length

    record = registration_of(kind)
    lengths: list[float] = []
    for name, declared in (record.settings.items() if record is not None else ()):
        if declared.dimension != Length:
            continue
        value: Any = settings[name]
        held: Sequence[Any] = cast(Sequence[Any], value) if isinstance(value, (list, tuple)) else (value,)
        lengths += [abs(float(v)) for v in held]
    positive = [v for v in lengths if v > 0.0]
    return min(positive) if positive else math.inf


__all__ = ["COST_LINE", "MARGIN_PX", "SAMPLES", "SPACING", "Sampled", "body", "chord", "crease_cost", "creased",
           "creased_compose", "evaluate", "evaluation_cost", "faces", "form", "frame_cost",
           "has_sign", "kind_named", "made_kind", "march", "samples_for", "smallest_length"]
