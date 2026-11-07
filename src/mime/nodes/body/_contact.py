"""A body's contact, laid out over contact pairs: each pair a body's hull against what it touches, a fixed surface
now (a ``Wall``) and another body's hull later, whose rows join one solve. A hull holds its points in its kind's frame
and finds its farthest points by its kind's support query, which reads no wall; a wall gives its gap, its normal and
the normal's derivative at any points. A pair of two bodies would find its nearest points from the two hulls'
supports (as GJK does, the farthest point of one less the other's along a direction) and give each row a column
block per body; nothing here assumes one body and one wall."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple, cast

import numpy as np
from jax import Array, lax
from numpy.typing import NDArray

from mime.core import jnp, quaternion, switch
from mime.registry import Geometry

#: The most of the search's Newton steps, their gradients stopped, that take each group's kept start (``kept``) towards
#: the body's own deepest point near it against a wall, the point where the wall's outward normal is the body's
#: (``x = s(n(x))``, ``s`` the hull's support), before one step with them flowing, so the point's derivatives are the
#: implicit function's (``deepest``). An XLA loop's iterations (``repeated``), a step each. A declaration,
#: measured on the review of the first fix (#116): a search started within the margin of the shrunk wall converged in
#: 4 steps at the 99th percentile and in at most 9 to 11; one started farther off whose point still reached the wall,
#: in 16 at the 99th percentile, some never within 30. A search that has not converged after them holds its group's
#: nearest place (``contact.search``).
CONTACT_NEWTON = 12
#: The solve's residual (``residual``) above which a step is reported unconverged: the share of the free step's speed
#: at the held points that its sweeps left unmet. Set by measurement (phase 7o's fix round), above a healthy run's and
#: below a wedged body's: script A's reproduction arm at 5 to 20 Hz, dt 0.25 to 1 ms and spacings 0.05 to 0.2 mm peaks at
#: 2.5e-3 in either precision, with 60 sweeps moving its speed by 0.02 %; F1's arms peak at 1e-4; a 2.6 by 3.5 mm cylinder
#: wedged at its tilt limit read 6.7e-2 (the contact 0.2.0, whose searches it held unconverged; 7.9e-8 under 0.3.0,
#: and 1.6e-2 given five sweeps, the flag's test), and a 0.5 by 8 mm rod wedged with a hundredth of its transverse
#: moments 6.5e-3 under 0.3.0. **What it says:** below it, the held points' normal speeds are within this
#: share of the free step's speed, and the forces' net moment within about twice it (over 28 wedged steps, against the
#: dual's own solution, the residual stood at 1 to 27 times the speeds' error). **What else it reads:** the drive, since
#: what is left unmet is set by the rows' allowances (gap / dt) while the speed under it grows with the drive, so that
#: wedge reads 6.7e-2 at 2 nN m and 7.8e-5 at 2 uN m. **What it does not say:** how far each point's own force is: eight
#: rows of rank two or three share their forces in no unique way.
CONVERGED = 5e-3
#: A held point may stand shallower than the sample its support's steps started from, where that sample is within the
#: margin of the shrunk wall (``shallower``), by this share of the margin (of a thousandth of the hull's size at a
#: margin of zero, the bound the spacing's guard holds a margin of zero to) before the flag ``support_short`` rises:
#: the steps went the wrong way, and the body may pass the samples' own bound, which the spacing's guard keeps within
#: the margin. A declaration, far above a gap's rounding in float32 (about 1e-9 m in a tube of a few millimetres).
SUPPORT_SHORT = 1e-2
#: The sweeps' relaxation: each row's update moves its force this share of the way to its own exact value (one: Gauss-
#: Seidel; between one and two: successive over-relaxation, which a symmetric positive definite problem still
#: converges under, each update still lowering its dual).
RELAXATION = 1.0
#: The least separation of two starts of one group, in units of the hull's spacing (the median distance from a sample
#: to its nearest neighbour, ``spacing_of``): a start's neighbours lie on its own crest or rim, which its search
#: reaches anyway, and a second start among them would only repeat it. A declaration, far below the separation of two
#: of a screw's crest passes (2.1 mm on FL-9) or a rod's rims; measured on the first fix (#116), 1.5 spacings left 6
#: of 20,000 poses of FL-9 at 0.4 mm missing its deepest point, 2.5 and 4 none.
START_APART = 3.0
#: The search's stopped steps every start takes before its group keeps the deepest (``kept``): enough to rank two of the
#: body's local deepest points apart by more than a micrometre (a climb from a sample a margin's spacing from its point
#: reaches rounding in two). The kept start then takes the search's own steps from the start.
SCREEN_STEPS = 2


class Wall(NamedTuple):
    """A fixed surface as a pair's other side: the region its shape is, which a body is held inside, shrunk by
    ``margin``. Its gap at a point is positive inside, its normal points into the region, and ``curvature`` is the
    derivative of its outward normal in the point (the shape's)."""

    shape: Any
    parameters: Mapping[str, Any]
    margin: Any

    def gaps(self, points: Array) -> Array:
        return -self.shape.signed_distance(self.parameters, points) - self.margin

    def normals(self, points: Array) -> Array:
        return -self.shape.outward_normal(self.parameters, points)

    def curvature(self, points: Array) -> Array:
        return self.shape.curvature(self.parameters, points)


class Hull(NamedTuple):
    """A body's hull as a pair's side: its points in the body's frame (``points``, ``N x 3``), the rows of the
    points each group chooses its nearest from (``candidates``, padded with ``-1``), the body's pose, and, for a kind
    with a support query, the kind and its parameters (``kind`` is ``None`` for a kind held by its samples), the rows
    of the points each group's further starts are chosen from (``places``: its own convex hull's vertices against a
    convex region, whatever its candidates), the least separation of two starts (``apart``, m), how far the body
    reaches past its samples (``shortfall``, m: ``mime.scene._sampling.hull_shortfall``), the farthest of its points
    from the body's origin (``size``, m) and the starts each group's search takes (``starts``: ``start_count``)."""

    points: Array
    candidates: NDArray[np.int32]
    pose: Array
    kind: Geometry | None
    parameters: Mapping[str, Any]
    places: NDArray[np.int32]
    apart: float
    shortfall: float
    size: float
    starts: int

    def rotation(self) -> Array:
        return quaternion.rotation_matrix(self.pose[3:])


class Pair(NamedTuple):
    """A contact pair: a body's hull and what it touches, a fixed surface now (``Wall``); another body's ``Hull``
    later."""

    body: Hull
    other: Wall


class Contacts(NamedTuple):
    """The rows a solve holds, over every pair: each row's gap (m, positive where the points are apart), its row
    ``J = [n; r x n]`` in the body's twist (the normal into the free region and its moment about the centre of mass),
    where its point is in the world, where it is in the body's own frame, and how far it stands shallower than the
    sample its search started from (``shallower``: m, zero where it is as deep or deeper, and for a point held where
    it is)."""

    gaps: Array
    rows: Array
    points: Array
    local: Array
    short: Array


def spacing_of(points: np.ndarray) -> float:
    """The hull's spacing as its points have it: the median distance from a point to its nearest other (m), zero for
    fewer than two points."""
    from scipy.spatial import KDTree

    held = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(held) < 2:
        return 0.0
    nearest = np.asarray(cast(Any, KDTree(held)).query(held, k=2)[0], dtype=np.float64)
    return float(np.median(nearest[:, 1]))


def joined(found: Sequence[Contacts]) -> Contacts:
    """The rows of several pairs as one solve's."""
    if len(found) == 1:
        return found[0]
    return Contacts(*(jnp.concatenate([getattr(c, name) for c in found]) for name in Contacts._fields))


def nearest_samples(pair: Pair, margin: Any) -> Array:
    """Each group's candidate nearest the other side at the body's pose, by its index into the hull's points: the
    switch ``contact.point``."""
    hull, wall = pair.body, pair.other
    rows = np.maximum(hull.candidates, 0)
    gaps = wall.gaps(hull.pose[:3] + jnp.matmul(hull.points[rows], hull.rotation().T))
    grouped = switch.static_select(hull.candidates >= 0, gaps, jnp.inf)

    def runner_up() -> Array:
        ordered = jnp.sort(grouped, axis=-1)
        return ordered[..., 1] - ordered[..., 0] if ordered.shape[-1] > 1 else jnp.full(ordered.shape[:-1], jnp.inf)

    index = switch.argmin(grouped, owner="contact.point", resolution=lambda: switch.rounding(margin),
                          indicator=runner_up)
    return jnp.asarray(rows)[jnp.arange(len(rows)), index]


def deepest(pair: Pair, local: Array) -> Array:
    """The body's own deepest point against the wall from ``local`` (each group's, in the body's frame), where a search
    converged: one of the search's Newton steps on ``x - s`` (``x = s(R^T n(p + R x))``, ``s`` the hull's support,
    ``n`` the wall's outward normal) with the gradients flowing, its derivative ``I - S R^T N R`` (``S`` the support's
    derivative in its direction, ``N`` the normal's in the point), and the support at its direction: a point of the
    body, whose derivatives are the implicit function's (``(I - S R^T N R)^-1`` of the support's own) where ``local``
    is the fixed point. A kind with a support (a closed form, or a refinement on its own surface) is held at its own
    surface's point, which moves continuously as the body turns."""
    hull, wall = pair.body, pair.other
    return _supported(pair, _step(pair, lax.stop_gradient(local), hull.pose, hull.parameters, wall), hull.pose,
                      hull.parameters, wall)[0]


def climbed(pair: Pair, seeds: Array, count: int) -> tuple[Array, Array]:
    """``count`` of the search's Newton steps (``deepest``) from each of ``seeds`` (points of the hull in the body's
    frame, any leading axes), on stopped values (the pose, the kind's and the wall's numbers and the seeds held as
    values, so no derivative passes through them): the steps' point, and the body's own point there, the support at
    its direction (the steps' point itself need not lie on the body)."""
    hull, wall = pair.body, pair.other
    held = lax.stop_gradient((seeds, hull.pose, _numbers(hull.parameters), _numbers(wall.parameters), wall.margin))
    x, pose, parameters = held[0], held[1], {**hull.parameters, **held[2]}
    surface = Wall(wall.shape, {**wall.parameters, **held[3]}, held[4])
    x = repeated(count, lambda _, x: _step(pair, x, pose, parameters, surface), x)
    return x, _supported(pair, x, pose, parameters, surface)[0]


def _supported(pair: Pair, x: Array, pose: Array, parameters: Mapping[str, Any], surface: Wall) -> tuple[Array, Array]:
    """The support along the wall's outward normal at ``x`` (a point in the body's frame), from ``x`` (each step's
    start is the last's point, so a kind refined on its surface starts where it last arrived), and the derivative of
    ``x - s`` in ``x``."""
    kind = cast(Geometry, pair.body.kind)
    turn = quaternion.rotation_matrix(pose[3:])
    world = pose[:3] + jnp.matmul(x, turn.T)
    found = kind.support(parameters, jnp.matmul(-surface.normals(world), turn), lax.stop_gradient(x))
    bend = jnp.matmul(jnp.matmul(turn.T, surface.curvature(world)), turn)
    return found.point, jnp.eye(3, dtype=x.dtype) - jnp.matmul(found.jacobian, bend)


def _step(pair: Pair, x: Array, pose: Array, parameters: Mapping[str, Any], surface: Wall) -> Array:
    """One of the search's Newton steps on ``x - s(x)``."""
    point, slope = _supported(pair, x, pose, parameters, surface)
    return x - _solved(slope, x - point)


def starts(pair: Pair) -> tuple[Array, Array]:
    """Each group's places to start its search from, ``G x starts x 3`` in the body's frame (``Hull.starts``), and which
    of them are taken: its candidate nearest the wall, then each the nearest of its ``places`` farther than the hull's
    ``apart`` from every place taken, a group with none left leaving the rest untaken (each untaken place repeats the
    last taken). Chosen on stopped values: where the search starts, which no derivative passes through; the step jumps
    where the group's kept point changes (``kept``, ``contact.point``)."""
    hull, wall = pair.body, pair.other

    def gaps_at(rows: NDArray[np.int32]) -> tuple[Array, Array]:
        local = hull.points[np.maximum(rows, 0)]
        local, gaps = lax.stop_gradient((local, wall.gaps(hull.pose[:3] + jnp.matmul(local, hull.rotation().T))))
        return local, switch.static_select(rows >= 0, gaps, jnp.inf)

    def nearest(local: Array, gaps: Array) -> Array:
        return jnp.take_along_axis(local, jnp.argmin(gaps, axis=-1)[:, None, None], axis=1)[:, 0]

    candidates = gaps_at(hull.candidates)
    place = nearest(*candidates)
    found, taken = [place], [jnp.ones(place.shape[:1], dtype=bool)]
    local, gaps = candidates if np.array_equal(hull.places, hull.candidates) else gaps_at(hull.places)
    while True:
        gaps = jnp.where(jnp.sum((local - place[:, None, :]) ** 2, axis=-1) > hull.apart ** 2, gaps, jnp.inf)
        if len(found) == hull.starts:
            return jnp.stack(found, axis=1), jnp.stack(taken, axis=1)
        open_ = jnp.isfinite(jnp.min(gaps, axis=-1))
        place = jnp.where(open_[:, None], nearest(local, gaps), place)
        found.append(place)
        taken.append(open_)


def kept(pair: Pair, places: Array, taken: Array, margin: Any) -> Array:
    """Each group's start (of ``places``, ``starts``'s) whose body point after ``SCREEN_STEPS`` of the search's
    stopped steps (``climbed``: the support at the steps' direction, a point of the body) is deepest: the switch
    ``contact.point``, its indicator the gap of the deepest point reached apart from that one (farther than the hull's
    ``apart`` from it) less its own, so that two starts that reach one point are one."""
    hull, wall = pair.body, pair.other
    _, reached = climbed(pair, places, SCREEN_STEPS)
    gaps = jnp.where(taken, lax.stop_gradient(wall.gaps(hull.pose[:3] + jnp.matmul(reached, hull.rotation().T))),
                     jnp.inf)

    def runner_up() -> Array:
        best = jnp.take_along_axis(reached, jnp.argmin(gaps, axis=-1)[:, None, None], axis=1)
        apart = jnp.sum((reached - best) ** 2, axis=-1) > pair.body.apart ** 2
        return jnp.min(jnp.where(apart, gaps, jnp.inf), axis=-1) - jnp.min(gaps, axis=-1)

    index = switch.argmin(gaps, owner="contact.point", resolution=lambda: switch.rounding(margin),
                          indicator=runner_up)
    return jnp.take_along_axis(places, index[:, None, None], axis=1)[:, 0]


def search_tolerance(dtype: Any, size: float) -> float:
    """How near its point a search must stand to the support at its direction (``|x - s(x)|``, m) to count as
    converged: the square root of the dtype's resolution, of the hull's ``size``, so that the flowing step after it
    takes the point to rounding (Newton's steps square the error)."""
    return float(np.sqrt(np.finfo(dtype).eps)) * size


def searched(pair: Pair, seeds: Array) -> tuple[Array, Array]:
    """Each group's search from its kept start (``seeds``): its point after ``CONTACT_NEWTON`` stopped steps
    (``climbed``), and how far within ``search_tolerance`` that point stands of the support at its direction (m,
    negative where the search has not converged): ``contact.search``'s indicator. On stopped values."""
    x, reached = climbed(pair, seeds, CONTACT_NEWTON)
    return x, search_tolerance(x.dtype, pair.body.size) - jnp.sqrt(jnp.sum((x - reached) ** 2, axis=-1))


def _solved(matrix: Array, vector: Array) -> Array:
    """``matrix^-1 vector`` for 3 x 3 matrices by their adjugate, elementwise over the leading axes: a handful of
    products where a factorisation per group would cost a call each."""
    a = matrix
    cofactor = jnp.stack([jnp.cross(a[..., 1, :], a[..., 2, :]), jnp.cross(a[..., 2, :], a[..., 0, :]),
                          jnp.cross(a[..., 0, :], a[..., 1, :])], axis=-1)
    determinant = jnp.sum(a[..., 0, :] * cofactor[..., :, 0], axis=-1)
    return jnp.einsum("...ij,...j->...i", cofactor, vector) / determinant[..., None]


def _numbers(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """The entries of ``parameters`` that are numbers or arrays of them, which a stopped evaluation holds as values."""
    return {name: value for name, value in parameters.items() if not isinstance(value, (bool, int, str))}


def start_count(points: np.ndarray, places: NDArray[np.int32], apart: float, shortfall: float) -> int:
    """The starts each group's search takes (``starts``), from the hull's own shape, when it is built: the most of a
    group's ``places`` that stand within ``shortfall`` of its farthest along any of ``mime.scene``'s shortfall
    directions (its spread over the sphere; not the axes and diagonals it adds, along which a flat face, a screw's or
    a cylinder's cap, ties whole, a set every start on it climbs out of to one point), each farther than ``apart``
    from those taken before it, nearest first, as ``starts`` takes them against a flat wall: a group can hold that
    many of the body's own local deepest points that its samples' gaps do not order (a screw's crest passes and its
    rims). Against a curved wall more can stand within the shortfall, unreported: MIME-ANO-134 records how often a
    search so started missed its body's deepest point. A count of places, not of the body's distinct deepest points:
    a long tie (a crest lying along the wall, a cap facing it) counts several, which costs starts and misses
    nothing."""
    from mime.scene._sampling import SHORTFALL_DIRECTIONS, shortfall_directions

    rows = np.maximum(np.asarray(places), 0)
    valid = np.asarray(places) >= 0
    held = np.asarray(points, dtype=np.float64)[rows]
    directions = shortfall_directions()[:SHORTFALL_DIRECTIONS]
    most = 1
    for start in range(0, len(directions), 64):
        along = np.where(valid[None], np.einsum("gpi,di->dgp", held, directions[start:start + 64]), -np.inf)
        open_ = along >= np.max(along, axis=-1, keepdims=True) - shortfall
        count = np.zeros(open_.shape[:2], dtype=np.int64)
        while open_.any():
            pick = np.argmax(np.where(open_, along, -np.inf), axis=-1)
            count += open_.any(axis=-1)
            picked = np.take_along_axis(held[None], pick[..., None, None], axis=2)
            open_ &= np.sum((held[None] - picked) ** 2, axis=-1) > apart * apart
        most = max(most, int(count.max()))
    return most


def pair_contacts(pair: Pair, margin: Any) -> Contacts:
    """A pair's rows: where the hull's kind has a support query, each group's start (``starts``) whose climb goes
    deepest (``kept``), searched (``searched``) and taken to the body's own deepest point there (``deepest``), or,
    where the search has not converged, its nearest place held where it is (``contact.search``); with how far the body
    may reach past it (``shallower``). For a kind with none, each group's nearest candidate, held where it is."""
    hull = pair.body
    if hull.kind is None:
        return held_at(pair, hull.points[nearest_samples(pair, margin)])
    places, taken = starts(pair)
    x, within = searched(pair, kept(pair, places, taken, margin))
    found = held_at(pair, switch.where(
        (within >= 0.0)[:, None], deepest(pair, x), places[:, 0], owner="contact.search",
        resolution=lambda: switch.rounding(search_tolerance(x.dtype, hull.size)), indicator=within))
    return found._replace(short=shallower(pair, places[:, 0], found.gaps, within < 0.0))


def shallower(pair: Pair, nearest: Array, gaps: Array, unconverged: Array) -> Array:
    """How far the body may reach past each row's point, at ``gaps`` from the shrunk wall, where the group's sample
    nearest the wall (``nearest``) is within the margin of it (m; zero elsewhere): how far the point stands shallower
    than that sample, a point of the body, where the search went the wrong way; and, where the search has not
    converged (``unconverged``) and the group holds that sample, at least the samples' shortfall, the most a sample
    falls short of the body. The point's gap is counted up to the margin, so a row whose sample is farther off reads
    zero: there the search may end anywhere on the body (a sample on the far side of a body in a tube meets the wall's
    normal turning across the axis), and the row holds nothing. Measured (phase 7o's second fix round): no row whose
    seed is within the margin ended shallower over script A's 5 s at 0.1 mm (8,142 rows) or the round-trip case's 20
    steps, while rows seeded 0.64 mm or more off ended up to 1.5 mm shallower. A report on stopped values, which the
    flag ``support_short`` reads: no derivative passes through it (review 7o's G4)."""
    hull, wall = pair.body, pair.other
    nearest, gaps, pose, margin = lax.stop_gradient((nearest, gaps, hull.pose, wall.margin))
    started = wall.gaps(pose[:3] + jnp.matmul(nearest, quaternion.rotation_matrix(pose[3:]).T))
    wrong = jnp.maximum(jnp.minimum(gaps, margin) - started, 0.0)
    return jnp.where(unconverged & (started <= margin), jnp.maximum(wrong, hull.shortfall), wrong)


def held_at(pair: Pair, local: Array) -> Contacts:
    """A pair's rows at the body's points ``local`` (in its own frame) at its pose, none of them short of a seed."""
    hull, wall = pair.body, pair.other
    arms = jnp.matmul(local, hull.rotation().T)
    points = hull.pose[:3] + arms
    normals = wall.normals(points)
    gaps = wall.gaps(points)
    return Contacts(gaps, jnp.concatenate([normals, jnp.cross(arms, normals)], axis=-1), points, local,
                    jnp.zeros_like(gaps))


def repeated(count: int, body: Callable[[Any, Any], Any], value: Any) -> Any:
    """``value`` after ``body(i, value)`` for ``i`` from 0 to ``count - 1``, each call taking the last's value: an
    XLA loop (``lax.fori_loop``) whose bounds are fixed when the step is traced, so that it lowers to a scan, through
    which both modes of derivative pass. While a switch collector is open (``switch.collector_open``) it is unrolled,
    a call each iteration, so that the collector gathers each iteration's rows as values of the step."""
    if switch.collector_open():
        for i in range(count):
            value = body(i, value)
        return value
    return cast(Any, lax).fori_loop(0, count, body, value)  # its hints read as partly unknown


def sweeps(coupling: Array, start: Array, count: int, *, owner: str) -> Array:
    """The multipliers ``x >= 0`` with ``start + coupling x >= 0`` and ``x (start + coupling x) = 0``, by ``count``
    projected Gauss-Seidel sweeps from zero, each update exact for its own multiplier with the others held, the rows
    in their order; a row of no length (a point on the surface's axis, where it has no normal) is never moved. The
    sweeps, and each sweep's updates, are XLA loops' iterations (``repeated``)."""
    own = jnp.diagonal(coupling)
    held = switch.domain_guard(own > 0, own, 1.0)
    update = partial(_updated, coupling, start, held, owner=owner)
    rows = start.shape[0]
    return repeated(count, lambda _, found: repeated(rows, update, found), jnp.zeros_like(start))


def _updated(coupling: Array, start: Array, held: Array, k: Any, found: Array, *, owner: str) -> Array:
    """``found`` with row ``k``'s multiplier moved to its own exact value with the others held, and kept at zero or
    above: one update of the sweeps."""
    residual = start[k] + jnp.dot(coupling[k], found)
    moved = found[k] - RELAXATION * residual / held[k]
    return found.at[k].set(switch.maximum(moved, 0.0, owner=owner, indicator=moved, resolution=lambda:
                                          switch.rounding(jnp.abs(found[k]) + jnp.abs(residual / held[k]))))


def no_work(coupling: Array, start: Array, forces: Array) -> Array:
    """``forces`` scaled along their own ray to where they do no work on the motion they leave, ``t = -f.b / f.C f``:
    then ``f.(b + C f) = 0``, so their power at the held twist is ``-t f.g/dt``, never positive, whatever the sweeps
    left; and of the scalings it is the one that lowers the problem's dual most, so the energy in the system's norm
    never rises (for a symmetric system). At the solution it is one, and for the sweeps' own forces on a symmetric
    system it is at least a half: they lower the dual from zero, ``f.b + f.C f / 2 <= 0``, so the scaled forces push
    as the sweeps' do. For 0.3.0's system, whose skew part is a share ``|omega| dt / 2`` of the moments' spread, it
    stays near one. Unchanged where no force is taken. Forces that are not the sweeps' (a planted force at a point far
    from the wall) can give ``t`` below zero: it is meant for the sweeps' alone."""
    stiffness = forces @ coupling @ forces
    taken = stiffness > 0
    return forces * switch.domain_guard(taken, -(forces @ start) / switch.domain_guard(taken, stiffness, 1.0), 1.0)


def residual(coupling: Array, start: Array, forces: Array, rows: Array, moved: Array) -> Array:
    """How far the solve is from its complementarity, a report: each row's ``min(f C_kk, b + C f)`` (m/s, zero at the
    solution: a held point leaves at its allowance, a free one takes no force), its largest magnitude over the speed
    the free step gives the held points, ``|v| + |r x n| |omega|`` at the fastest (the bound of each point's speed
    along any direction): a share of the body's own motion there, not of its speed into the wall alone, which a body
    resting on the wall has next to none of. Taken on stopped values: a report, through which no derivative passes."""
    coupling, start, forces, rows, moved = lax.stop_gradient((coupling, start, forces, rows, moved))
    left = start + coupling @ forces
    unmet = jnp.minimum(forces * jnp.diagonal(coupling), left)
    linear = jnp.sqrt(jnp.sum(moved[:3] * moved[:3]))
    angular = jnp.sqrt(jnp.sum(moved[3:] * moved[3:]))
    lever = jnp.sqrt(jnp.sum(rows[:, 3:] * rows[:, 3:], axis=-1))
    speed = jnp.max(linear + lever * angular)
    return jnp.max(jnp.abs(unmet)) / (speed + float(jnp.finfo(moved.dtype).tiny))


__all__ = ["CONTACT_NEWTON", "CONVERGED", "RELAXATION", "SCREEN_STEPS", "START_APART", "SUPPORT_SHORT", "Contacts",
           "Hull", "Pair", "Wall", "climbed", "deepest", "held_at", "joined", "kept", "nearest_samples", "no_work",
           "pair_contacts", "repeated", "residual", "search_tolerance", "searched", "shallower", "spacing_of",
           "start_count", "starts", "sweeps"]
