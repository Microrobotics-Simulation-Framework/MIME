"""The exact signed distance of a union of regions that each have one (a vessel's lumens joined at its junctions, a
body's solids): the minimum, a check at each point of where the minimum is exact, and Newton at a wall, a seam or a
corner where it is not.

**Why the minimum is not enough.** Outside every region (the solid round a vessel's lumens) the distance to the union
is the least of the distances to its regions, exactly. Inside (the fluid) the minimum is the depth in the deepest
region, which is a lower bound of the depth in the union. It equals that depth wherever the deepest region's nearest
wall point lies outside every other region, so is a point of the union's boundary; where it lies inside another (a
false wall, at every seam) it reads too shallow, by 0.8 to 1.75 cells at a Y's carina at 8 to 32 cells a diameter.

**The rule.** At each point the deepest region's nearest wall point is checked against the regions that meet it.
Where it lies inside one, the depth is the least distance to a point of the union's boundary among: each region's
nearest wall point that lies outside the others; each seam's (two walls meeting: the nearest point of their
intersection, by Newton on the Lagrangian from its linearisation's); and each corner (three walls meeting). Each is
kept only where it lies outside the remaining regions, so every candidate is a point of the boundary and the least
never reads too shallow. It finds the nearest point where each region's wall has one nearest point from any point off
its medial set (balls, round cones, half-spaces: convex regions) and Newton reaches the seam's nearest point from its
linearisation's, which the tests measure against a brute force.

**The fillet.** A fillet of radius ``r`` is the union's closing by a ball of radius ``r``: every edge the solid has
between two walls is rounded to radius ``r`` (a carina, a daughter's root) and nothing else moves. Its distance is the
same rule on the regions widened by ``r``, moved back by ``r``, and it is exact on both sides. In the fluid the
closing's depth is the widened union's less ``r``. In the solid within ``r`` of the widened union the distance is ``r``
less the depth there (any point within ``r`` of the widened union's complement is within ``r`` of the closing's
boundary, and no nearer), and beyond it the distance to the union itself (the opening of the solid lies inside the
solid, and the ball about the point inside the eroded solid grows by ``r`` into it). So ``r`` is a parameter like any
other: traced, ``r = 0`` the sharp union, its cost independent of it, and never set from a grid.

**Derivatives.** The iteration counts are fixed. A seam's or a corner's point carries the implicit function theorem's
derivative: one Newton step from the converged point with its Jacobian held, so its tangent is exact in the point and
in every parameter, and the distance's derivative and its normal's are exact off the sets where the selection
switches. The selections are plain JAX: no node's step runs this yet. The first step that does (the FVM's wall) writes
them through ``mime.core.switch``, its owners declared on the shape it registers.

**Where the Newton runs.** The check runs at every point. The candidates run only at a held set of points, the plan
(:meth:`ExactUnion.plan`), chosen from the points the check fails at and a margin when the caller builds, so the
program's shapes stay fixed. A point outside the plan that fails the check is counted (``unrefined``), so walls moved
further than the plan's margin are seen, never silent."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp

type Parameters = Mapping[str, Any]
#: A value read from the parameters: a JAX function of them, so that a derivative in them flows.
type Read = Callable[[Parameters], Any]

_lax: Any = cast(Any, jax).lax
_stop: Callable[[Any], Any] = cast(Any, jax).lax.stop_gradient

#: Gauss-Newton steps onto a seam (two walls) or a corner (three) from the query point, each landing on the nearest
#: point of the constraints' linearisation: the seam's start and the corner itself.
PROJECTIONS = 12
#: Newton steps on the seam's Lagrangian from that start, to the seam's nearest point.
NEWTON_STEPS = 8
#: A candidate's point is kept where it lies on its walls and outside the remaining regions to this many units in the
#: last place of the point's scale (its distance from the origin plus the union's size): its own rounding.
KEPT_ULPS = 64.0
#: Newton's answer is taken where its walls' residuals are within this many units in the last place of the scale.
CONVERGED_ULPS = 4096.0


class Region(Protocol):
    """A region with an exact signed distance (negative inside), evaluated at points ``(..., 3)``: the distance
    ``(...)``, its gradient, the outward unit normal ``(..., 3)``, and the normal's derivative ``(..., 3, 3)``, each a
    JAX function of the parameters. Off its medial set its wall has one nearest point from any point."""

    def distance(self, parameters: Parameters, points: Array) -> Array: ...

    def normal(self, parameters: Parameters, points: Array) -> Array: ...

    def hessian(self, parameters: Parameters, points: Array) -> Array: ...


def _length(v: Array) -> Array:
    """``|v|`` over the last axis, its derivative zero at zero (where ``|v|``'s is not defined)."""
    squared = jnp.sum(v * v, axis=-1)
    positive = squared > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def _unit(v: Array) -> tuple[Array, Array]:
    """``v`` over its length (zero for a zero vector), and the length."""
    length = _length(v)
    return v / jnp.where(length > 0.0, length, 1.0)[..., None], length


def _ball(offset: Array, radius: Any) -> tuple[Array, Array, Array]:
    """A ball's distance, normal and normal's derivative at ``offset`` from its centre."""
    normal, length = _unit(offset)
    across = jnp.eye(3, dtype=offset.dtype) - normal[..., :, None] * normal[..., None, :]
    return length - radius, normal, across / jnp.where(length > 0.0, length, 1.0)[..., None, None]


@dataclass(frozen=True)
class Ball:
    """A ball: ``centre`` (three numbers) and ``radius``, each read from the parameters."""

    centre: Read
    radius: Read

    def evaluate(self, parameters: Parameters, points: Array) -> tuple[Array, Array, Array]:
        centre = jnp.asarray(self.centre(parameters), dtype=points.dtype)
        return _ball(points - centre, jnp.asarray(self.radius(parameters), dtype=points.dtype))

    def distance(self, parameters: Parameters, points: Array) -> Array:
        return self.evaluate(parameters, points)[0]

    def normal(self, parameters: Parameters, points: Array) -> Array:
        return self.evaluate(parameters, points)[1]

    def hessian(self, parameters: Parameters, points: Array) -> Array:
        return self.evaluate(parameters, points)[2]


@dataclass(frozen=True)
class PlaneSide:
    """The side of a plane its ``normal`` points away from, ``x . n <= offset``: the normal (three numbers, any length
    above zero) and the offset, each read from the parameters. Its distance is ``x . n - offset`` with ``n`` the unit
    normal, flat, so its normal's derivative is zero."""

    normal_of: Read
    offset: Read

    def _unit_normal(self, parameters: Parameters, dtype: Any) -> Array:
        normal = jnp.asarray(self.normal_of(parameters), dtype=dtype)
        return normal / jnp.sqrt(jnp.sum(normal * normal))

    def distance(self, parameters: Parameters, points: Array) -> Array:
        normal = self._unit_normal(parameters, points.dtype)
        return jnp.sum(points * normal, axis=-1) - jnp.asarray(self.offset(parameters), dtype=points.dtype)

    def normal(self, parameters: Parameters, points: Array) -> Array:
        return jnp.broadcast_to(self._unit_normal(parameters, points.dtype), points.shape)

    def hessian(self, parameters: Parameters, points: Array) -> Array:
        return jnp.zeros((*points.shape, 3), dtype=points.dtype)


@dataclass(frozen=True)
class RoundCone:
    """The balls along a segment whose radius changes linearly from ``start_radius`` at ``start`` to ``end_radius``
    at ``end``, together: their convex hull, a cone tangent to the two end balls, closed by them. A tube of a
    constant radius is a capsule. Each is read from the parameters; the radii differ by less than the segment's
    length (a ball inside the other would leave no cone).

    In the plane through the axis and a point (``t`` along the axis from ``start``, ``rho`` from it) the cone's wall
    lies at ``start_radius`` from ``start`` along its outward normal ``(sin a, cos a)``, ``sin a = (start_radius -
    end_radius) / L``. A point whose foot on that wall lies before the first tangent circle (``t cos a - rho sin a <
    0``) is nearest the start ball, one past the second (beyond ``L cos a``) the end ball, and any other the cone: each
    piece's distance and its normal agree where they meet, so the distance is continuously differentiable there. Its
    medial set is the axis between the balls' centres, where the normal is zero."""

    start: Read
    end: Read
    start_radius: Read
    end_radius: Read

    def evaluate(self, parameters: Parameters, points: Array) -> tuple[Array, Array, Array]:
        dtype = points.dtype
        a = jnp.asarray(self.start(parameters), dtype=dtype)
        b = jnp.asarray(self.end(parameters), dtype=dtype)
        ra = jnp.asarray(self.start_radius(parameters), dtype=dtype)
        rb = jnp.asarray(self.end_radius(parameters), dtype=dtype)
        axis, length = _unit(b - a)
        sine = (ra - rb) / length
        cosine = jnp.sqrt(1.0 - sine * sine)
        offset = points - a
        along = jnp.sum(offset * axis, axis=-1)
        radial_direction, rho = _unit(offset - along[..., None] * axis)
        foot = along * cosine - rho * sine
        cone = along * sine + rho * cosine - ra
        cone_normal = sine * axis + cosine * radial_direction
        projector = (jnp.eye(3, dtype=dtype) - axis[:, None] * axis[None, :]
                     - radial_direction[..., :, None] * radial_direction[..., None, :])
        cone_hessian = cosine * projector / jnp.where(rho > 0.0, rho, 1.0)[..., None, None]
        start_ball = _ball(offset, ra)
        end_ball = _ball(points - b, rb)
        before = foot < 0.0
        after = foot > length * cosine
        distance = jnp.where(before, start_ball[0], jnp.where(after, end_ball[0], cone))
        normal = jnp.where(before[..., None], start_ball[1], jnp.where(after[..., None], end_ball[1], cone_normal))
        hessian = jnp.where(before[..., None, None], start_ball[2],
                            jnp.where(after[..., None, None], end_ball[2], cone_hessian))
        return distance, normal, hessian

    def distance(self, parameters: Parameters, points: Array) -> Array:
        return self.evaluate(parameters, points)[0]

    def normal(self, parameters: Parameters, points: Array) -> Array:
        return self.evaluate(parameters, points)[1]

    def hessian(self, parameters: Parameters, points: Array) -> Array:
        return self.evaluate(parameters, points)[2]


class UnionDistance(NamedTuple):
    """The union's signed distance at the points (negative inside) and, for the caller's report: whether each point
    failed the check (its deepest region's nearest wall point inside another); how many failing points lie outside
    the plan, where the minimum stands uncorrected; and how many planned points that failed found no candidate."""

    distance: Array
    failing: Array
    unrefined: Array
    unconverged: Array


@dataclass(frozen=True)
class ExactUnion:
    """The union of ``regions``, its exact signed distance by the seam-aware rule (the module's docstring), with a
    fillet of radius ``fillet(parameters)`` (zero by default) on every edge the solid has between two walls.

    ``meets`` lists the pairs of regions whose walls meet (each by index, the smaller first): the structure, fixed
    when the union is made (a tree's junctions, and any other pair that overlaps). Only those are checked against each
    other, and their seams, and the corners of each three that all meet, are the candidates. ``size`` is a length the
    rounding allowances scale with: the union's largest radius, say."""

    regions: tuple[Region, ...]
    meets: tuple[tuple[int, int], ...]
    size: float
    fillet: Read = field(default=lambda parameters: 0.0)

    def __post_init__(self) -> None:
        count = len(self.regions)
        for i, j in self.meets:
            if not 0 <= i < j < count:
                raise ValueError(f"a union's meeting pair ({i}, {j}) names regions by index, the smaller first, "
                                 f"below {count}")
        if len(set(self.meets)) != len(self.meets):
            raise ValueError(f"a union's meeting pairs are listed once each: {self.meets}")

    @property
    def neighbours(self) -> np.ndarray:
        """Which regions meet which, ``(n, n)``, fixed."""
        table = np.zeros((len(self.regions), len(self.regions)), dtype=bool)
        for i, j in self.meets:
            table[i, j] = table[j, i] = True
        return table

    @property
    def corners(self) -> tuple[tuple[int, int, int], ...]:
        """Each three regions every two of which meet."""
        held = set(self.meets)
        count = len(self.regions)
        return tuple((i, j, k) for i in range(count) for j in range(i + 1, count) for k in range(j + 1, count)
                     if {(i, j), (i, k), (j, k)} <= held)

    # -- the check, everywhere ------------------------------------------------------------------------------------
    def _widened(self, parameters: Parameters, points: Array) -> tuple[Array, Array, Any]:
        """Each region's distance less the fillet's radius ``(n, ...)``, its normal ``(n, ..., 3)``, and the radius."""
        radius = jnp.asarray(self.fillet(parameters), dtype=points.dtype)
        distances = jnp.stack([region.distance(parameters, points) for region in self.regions]) - radius
        normals = jnp.stack([region.normal(parameters, points) for region in self.regions])
        return distances, normals, radius

    def _slack(self, points: Array) -> Array:
        """The rounding allowance at each point: ``KEPT_ULPS`` units in the last place of its scale."""
        return KEPT_ULPS * jnp.finfo(points.dtype).eps * (_length(points) + self.size)

    def check(self, parameters: Parameters, points: Array) -> tuple[Array, Array]:
        """At each point ``(N, 3)``: the least of the widened regions' distances, and how far the deepest one's nearest
        wall point lies outside the regions that meet it (below zero: inside one, where the minimum is not exact;
        infinite where it meets none)."""
        distances, normals, _ = self._widened(parameters, points)
        least = jnp.min(distances, axis=0)
        deepest = jnp.argmin(distances, axis=0)
        normal = jnp.take_along_axis(normals, deepest[None, :, None], axis=0)[0]
        foot = points - least[:, None] * normal
        at_foot = jnp.stack([region.distance(parameters, foot) for region in self.regions]) - \
            jnp.asarray(self.fillet(parameters), dtype=points.dtype)
        meets = jnp.asarray(self.neighbours)[:, deepest]
        clearance = jnp.min(jnp.where(meets, at_foot, jnp.inf), axis=0)
        return least, clearance

    def plan(self, parameters: Parameters, points: Any, margin: float) -> np.ndarray:
        """The points the candidates run at, held for a build: the indices of those in the fluid or within ``margin``
        of it whose deepest region's nearest wall point lies inside another or within ``margin`` of one, at
        ``parameters`` as built (NumPy, sorted). Walls that later move by up to about half the margin leave no
        failing point outside it; past that, :meth:`signed_distance` counts them."""
        held = jnp.asarray(points)
        least, clearance = self.check(parameters, held)
        chosen = np.asarray((least < margin) & (clearance < margin))
        return np.nonzero(chosen)[0]

    # -- the candidates, at the plan's points ---------------------------------------------------------------------
    def _walls(self, parameters: Parameters, points: Array, distances: Array, normals: Array, radius: Any) -> Array:
        """Each region's nearest wall point's distance where it lies outside every region that meets it, infinite
        elsewhere; the least of them, ``(M,)``."""
        slack = self._slack(points)
        table = self.neighbours
        best = jnp.full(points.shape[:-1], jnp.inf, dtype=points.dtype)
        for i in range(len(self.regions)):
            foot = points - distances[i][:, None] * normals[i]
            kept = jnp.ones(points.shape[:-1], dtype=bool)
            for j in np.nonzero(table[i])[0]:
                kept = kept & (self.regions[j].distance(parameters, foot) - radius >= -slack)
            best = jnp.minimum(best, jnp.where(kept, jnp.abs(distances[i]), jnp.inf))
        return best

    def _kept(self, parameters: Parameters, point: Array, own: Sequence[int], radius: Any, slack: Array) -> Array:
        """Whether ``point`` lies outside every region that meets one of ``own`` and is not one of them."""
        table = self.neighbours
        others = sorted({int(j) for i in own for j in np.nonzero(table[i])[0]} - set(own))
        kept = jnp.ones(point.shape[:-1], dtype=bool)
        for j in others:
            kept = kept & (self.regions[j].distance(parameters, point) - radius >= -slack)
        return kept

    def _seam(self, parameters: Parameters, points: Array, i: int, j: int, radius: Any) -> tuple[Array, Array]:
        """The nearest point of the widened walls ``i`` and ``j``'s intersection to each point, by Newton on the
        Lagrangian of ``|q - x|^2 / 2`` under ``phi_i = phi_j = 0`` from the linearisation's nearest point; its
        distance where it converged and lies outside the remaining regions (infinite elsewhere), and whether it
        converged. Its derivative is the implicit function theorem's."""
        regions = self.regions
        dtype = points.dtype
        eye = jnp.eye(3, dtype=dtype)

        def walls(q: Array) -> tuple[Array, Array]:
            values = jnp.stack([regions[i].distance(parameters, q), regions[j].distance(parameters, q)], -1) - radius
            gradients = jnp.stack([regions[i].normal(parameters, q), regions[j].normal(parameters, q)], -2)
            return values, gradients

        def projected(_: int, q: Array) -> Array:
            values, gradients = walls(q)
            gram = gradients @ jnp.swapaxes(gradients, -1, -2) + _ridge(dtype, 2)
            return q - (jnp.swapaxes(gradients, -1, -2) @ jnp.linalg.solve(gram, values[..., None]))[..., 0]

        def kkt(z: Array) -> tuple[Array, Array]:
            q, multipliers = z[..., :3], z[..., 3:]
            values, gradients = walls(q)
            hessians = jnp.stack([regions[i].hessian(parameters, q), regions[j].hessian(parameters, q)], -3)
            stationary = q - points + jnp.sum(multipliers[..., :, None] * gradients, axis=-2)
            residual = jnp.concatenate([stationary, values], -1)
            top = eye + jnp.sum(multipliers[..., :, None, None] * hessians, axis=-3)
            jacobian = jnp.concatenate([jnp.concatenate([top, jnp.swapaxes(gradients, -1, -2)], -1),
                                        jnp.concatenate([gradients, jnp.zeros((*q.shape[:-1], 2, 2), dtype)], -1)],
                                       -2)
            return residual, jacobian

        def stepped(_: int, z: Array) -> Array:
            residual, jacobian = kkt(z)
            return z - jnp.linalg.solve(jacobian + _ridge(dtype, 5), residual[..., None])[..., 0]

        held_points = _stop(points)
        start = _lax.fori_loop(0, PROJECTIONS, projected, held_points)
        _, gradients = walls(start)
        gram = gradients @ jnp.swapaxes(gradients, -1, -2) + _ridge(dtype, 2)
        multipliers = jnp.linalg.solve(gram, (gradients @ (held_points - start)[..., None]))[..., 0]
        reached = _stop(_lax.fori_loop(0, NEWTON_STEPS, stepped, jnp.concatenate([start, multipliers], -1)))
        reached = _finite_or(reached, jnp.concatenate([held_points, jnp.zeros((*points.shape[:-1], 2), dtype)], -1))
        residual, jacobian = kkt(reached)
        z = _implicit(reached, _stop(jacobian) + _ridge(dtype, 5), residual)
        scale = _length(held_points) + self.size
        converged = _stop(jnp.max(jnp.abs(residual), axis=-1)) <= CONVERGED_ULPS * jnp.finfo(dtype).eps * scale
        kept = converged & self._kept(parameters, reached[..., :3], (i, j), radius, self._slack(held_points))
        seam = jnp.where(kept[..., None], z[..., :3], held_points)
        return jnp.where(kept, _length(points - seam), jnp.inf), converged

    def _corner(self, parameters: Parameters, points: Array, corner: tuple[int, int, int],
                radius: Any) -> Array:
        """The point where the three widened walls of ``corner`` meet nearest each point, by Gauss-Newton from it; its
        distance where it converged and lies outside the remaining regions, infinite elsewhere. Its derivative in
        the parameters is the implicit function theorem's (the corner does not move with the point)."""
        regions = [self.regions[k] for k in corner]
        dtype = points.dtype

        def walls(c: Array) -> tuple[Array, Array]:
            values = jnp.stack([region.distance(parameters, c) for region in regions], -1) - radius
            gradients = jnp.stack([region.normal(parameters, c) for region in regions], -2)
            return values, gradients

        def projected(_: int, c: Array) -> Array:
            values, gradients = walls(c)
            gram = gradients @ jnp.swapaxes(gradients, -1, -2) + _ridge(dtype, 3)
            return c - (jnp.swapaxes(gradients, -1, -2) @ jnp.linalg.solve(gram, values[..., None]))[..., 0]

        held_points = _stop(points)
        reached = _finite_or(_stop(_lax.fori_loop(0, PROJECTIONS + NEWTON_STEPS, projected, held_points)),
                             held_points)
        values, gradients = walls(reached)
        held_gradients = _stop(gradients)
        gram = held_gradients @ jnp.swapaxes(held_gradients, -1, -2) + _ridge(dtype, 3)
        step = _implicit(jnp.zeros_like(values), gram, values)
        c = reached + (jnp.swapaxes(held_gradients, -1, -2) @ step[..., None])[..., 0]
        scale = _length(held_points) + self.size
        converged = _stop(jnp.max(jnp.abs(values), axis=-1)) <= CONVERGED_ULPS * jnp.finfo(dtype).eps * scale
        kept = converged & self._kept(parameters, reached, corner, radius, self._slack(held_points))
        c = jnp.where(kept[..., None], c, held_points)
        return jnp.where(kept, _length(points - c), jnp.inf)

    def refined_depth(self, parameters: Parameters, points: Array) -> tuple[Array, Array]:
        """The depth in the widened union at ``points`` ``(M, 3)`` by every candidate (walls, seams, corners), and
        whether any candidate was found."""
        distances, normals, radius = self._widened(parameters, points)
        best = self._walls(parameters, points, distances, normals, radius)
        for i, j in self.meets:
            best = jnp.minimum(best, self._seam(parameters, points, i, j, radius)[0])
        for corner in self.corners:
            best = jnp.minimum(best, self._corner(parameters, points, corner, radius))
        return best, jnp.isfinite(best)

    # -- the distance ------------------------------------------------------------------------------------------------
    def signed_distance(self, parameters: Parameters, points: Any, plan: np.ndarray | None = None) -> UnionDistance:
        """The union's signed distance at ``points`` ``(N, 3)``, negative inside, filleted: the candidates run at the
        points ``plan`` holds (indices, :meth:`plan`), at every point where it is ``None``."""
        held = jnp.asarray(points)
        least, clearance = self.check(parameters, held)
        radius = jnp.asarray(self.fillet(parameters), dtype=held.dtype)
        failing = (least < 0.0) & (clearance < -self._slack(held))
        index = np.arange(held.shape[0]) if plan is None else np.asarray(plan, dtype=np.int64)
        planned = np.zeros(held.shape[0], dtype=bool)
        planned[index] = True
        depth = -least
        unconverged = jnp.asarray(0)
        if index.size:
            refined, found = self.refined_depth(parameters, held[index])
            use = failing[index] & found
            depth = depth.at[index].set(jnp.where(use, refined, depth[index]))
            unconverged = jnp.sum(failing[index] & ~found)
        widened = jnp.where(least >= 0.0, least, -depth)
        unrefined = jnp.sum(failing & ~jnp.asarray(planned))
        return UnionDistance(widened + radius, failing, unrefined, unconverged)

    def distance(self, parameters: Parameters, points: Any, plan: np.ndarray | None = None) -> Array:
        """:meth:`signed_distance`'s distance alone."""
        return self.signed_distance(parameters, points, plan).distance

    def normal(self, parameters: Parameters, points: Any, plan: np.ndarray | None = None) -> Array:
        """The union's outward unit normal at ``points``: the distance's gradient in each point."""
        held = jnp.asarray(points)

        def total(at: Array) -> Array:
            return jnp.sum(self.distance(parameters, at, plan))

        return cast(Any, jax).grad(total)(held)

    def minimum(self, parameters: Parameters, points: Any) -> Array:
        """The plain minimum of the regions' distances, filleted the same way: exact outside the union, too shallow
        inside it near every seam. The control the rule is measured against."""
        held = jnp.asarray(points)
        distances, _, radius = self._widened(parameters, held)
        return jnp.min(distances, axis=0) + radius


def _implicit(reached: Array, jacobian: Array, residual: Array) -> Array:
    """One Newton step from ``reached``, a root to rounding held out of every derivative, with ``jacobian`` held: its
    value is the root's, and its tangent is the implicit function theorem's, ``-J^-1 dF``, through the residual's
    dependence on the point and the parameters."""
    return reached - jnp.linalg.solve(jacobian, residual[..., None])[..., 0]


def _finite_or(value: Array, fallback: Array) -> Array:
    """``value`` where every component of a point's row is finite, ``fallback`` elsewhere: a step that did not reach a
    seam keeps its discarded lane finite, so no NaN reaches a derivative through it."""
    return jnp.where(jnp.all(jnp.isfinite(value), axis=-1, keepdims=True), value, fallback)


def _ridge(dtype: Any, size: int) -> Array:
    """A few units in the last place on a small system's diagonal, keeping a step finite where its walls are tangent
    (no seam there: the step is discarded, its residual never converging)."""
    return 16.0 * jnp.finfo(dtype).eps * jnp.eye(size, dtype=dtype)


__all__ = ["CONVERGED_ULPS", "KEPT_ULPS", "NEWTON_STEPS", "PROJECTIONS", "Ball", "ExactUnion", "PlaneSide", "Region",
           "RoundCone", "UnionDistance"]
