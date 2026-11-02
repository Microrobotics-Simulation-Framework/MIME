"""A body's contact, laid out over contact pairs: each pair a body's hull against what it touches, a fixed surface
now (a ``Wall``) and another body's hull later, whose rows join one solve. A hull holds its points in its kind's frame
and finds its farthest points by its kind's support query, which reads no wall; a wall gives its gap, its normal and
the normal's derivative at any points. A pair of two bodies would find its nearest points from the two hulls'
supports (as GJK does, the farthest point of one less the other's along a direction) and give each row a column
block per body; nothing here assumes one body and one wall."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
from jax import Array, lax
from numpy.typing import NDArray

from mime.core import jnp, quaternion, switch
from mime.registry import Geometry

#: The Newton steps that take each group's nearest sample to the body's own deepest point near it against a wall, the
#: point where the wall's outward normal is the body's (``x = s(n(x))``, ``s`` the hull's support): its gradients
#: stopped, then one step with them flowing, so the point's derivatives are the implicit function's. A declaration:
#: from a sample at a spacing of a margin the point converges to rounding in two (phase 7o's fix round).
CONTACT_NEWTON = 5
#: The solve's residual (``residual``) above which a step is reported unconverged: the share of the free step's normal
#: speeds that its sweeps left unmet. A declaration, a thousandth: what is left goes to the projection after the step,
#: which keeps the margin but charges no velocity, so the forces read from the state are short by about that share.
CONVERGED = 1e-3
#: The sweeps' relaxation: each row's update moves its force this share of the way to its own exact value (one: Gauss-
#: Seidel; between one and two: successive over-relaxation, which a symmetric positive definite problem still
#: converges under, each update still lowering its dual).
RELAXATION = 1.0


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
    with a support query, the kind and its parameters (``kind`` is ``None`` for a kind held by its samples)."""

    points: Array
    candidates: NDArray[np.int32]
    pose: Array
    kind: Geometry | None
    parameters: Mapping[str, Any]

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
    where its point is in the world, and where it is in the body's own frame."""

    gaps: Array
    rows: Array
    points: Array
    local: Array


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


def deepest(pair: Pair, seeds: Array) -> Array:
    """Each seed (a point of the hull, in the body's frame) taken to the body's own deepest point near it against the
    wall: the fixed point ``x = s(R^T n(p + R x))`` of the hull's support ``s`` (from the seed) along the wall's outward
    normal ``n``, by Newton's steps on ``x - s`` with the derivative ``I - S R^T N R`` (``S`` the support's derivative
    in its direction, ``N`` the normal's in the point), ``CONTACT_NEWTON`` with the gradients stopped and one with them
    flowing; the point is the support at the last step's direction, a point of the body however far the steps got,
    whose derivatives are the implicit function's (``(I - S R^T N R)^-1`` of the support's own). A kind with a support
    (a closed form, or a refinement on its own surface) is held at its own surface's point, which moves continuously
    as the body turns."""
    hull, wall = pair.body, pair.other
    kind = cast(Geometry, hull.kind)
    held = lax.stop_gradient((seeds, hull.pose, _numbers(hull.parameters), _numbers(wall.parameters), wall.margin))

    def supported(x: Array, pose: Array, parameters: Mapping[str, Any], surface: Wall) -> tuple[Array, Array, Array]:
        """The support along the wall's outward normal at ``x``, from ``x`` (each step's start is the last's point,
        so a kind refined on its surface starts where it last arrived), and the derivative of ``x - s`` in ``x``."""
        turn = quaternion.rotation_matrix(pose[3:])
        world = pose[:3] + jnp.matmul(x, turn.T)
        found = kind.support(parameters, jnp.matmul(-surface.normals(world), turn), lax.stop_gradient(x))
        bend = jnp.matmul(jnp.matmul(turn.T, surface.curvature(world)), turn)
        return found.point, jnp.eye(3, dtype=x.dtype) - jnp.matmul(found.jacobian, bend), turn

    def step(x: Array, pose: Array, parameters: Mapping[str, Any], surface: Wall) -> Array:
        point, slope, _ = supported(x, pose, parameters, surface)
        return x - _solved(slope, x - point)

    x, pose, parameters = held[0], held[1], {**hull.parameters, **held[2]}
    surface = Wall(wall.shape, {**wall.parameters, **held[3]}, held[4])
    for _ in range(CONTACT_NEWTON):
        x = step(x, pose, parameters, surface)
    return supported(step(lax.stop_gradient(x), hull.pose, hull.parameters, wall), hull.pose, hull.parameters,
                     wall)[0]


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


def pair_contacts(pair: Pair, margin: Any) -> Contacts:
    """A pair's rows: each group's nearest candidate, taken to the body's own deepest point near it where the hull's
    kind has a support query (``deepest``), or held where it is."""
    hull = pair.body
    seeds = hull.points[nearest_samples(pair, margin)]
    return held_at(pair, seeds if hull.kind is None else deepest(pair, seeds))


def held_at(pair: Pair, local: Array) -> Contacts:
    """A pair's rows at the body's points ``local`` (in its own frame) at its pose."""
    hull, wall = pair.body, pair.other
    arms = jnp.matmul(local, hull.rotation().T)
    points = hull.pose[:3] + arms
    normals = wall.normals(points)
    return Contacts(wall.gaps(points), jnp.concatenate([normals, jnp.cross(arms, normals)], axis=-1), points, local)


def sweeps(coupling: Array, start: Array, count: int, *, owner: str) -> Array:
    """The multipliers ``x >= 0`` with ``start + coupling x >= 0`` and ``x (start + coupling x) = 0``, by ``count``
    projected Gauss-Seidel sweeps from zero, each update exact for its own multiplier with the others held; a row of
    no length (a point on the surface's axis, where it has no normal) is never moved."""
    own = jnp.diagonal(coupling)
    held = switch.domain_guard(own > 0, own, 1.0)
    found = jnp.zeros_like(start)
    for _ in range(count):
        for k in range(start.shape[0]):
            residual = start[k] + jnp.dot(coupling[k], found)
            moved = found[k] - RELAXATION * residual / held[k]
            found = found.at[k].set(switch.maximum(moved, 0.0, owner=owner, indicator=moved, resolution=lambda:
                                                   switch.rounding(jnp.abs(found[k]) + jnp.abs(residual / held[k]))))
    return found


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


__all__ = ["CONTACT_NEWTON", "CONVERGED", "RELAXATION", "Contacts", "Hull", "Pair", "Wall", "deepest", "held_at",
           "joined", "nearest_samples", "no_work",
           "pair_contacts", "residual", "sweeps"]
