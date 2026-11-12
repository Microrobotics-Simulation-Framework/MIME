"""Friction at a held point, solved in the contact's own sweeps: exact Coulomb, one coefficient, the elliptic cone.

Each held point carries its normal row ``[n; r x n]`` and, for each friction row kind its law declares, a pair of
rows: sliding ``[t_i; r x t_i]``, which measure the slip of the body's own point ``t_i . (v + omega x r)`` (rolling
without slip is the sliding rows sticking), each pair's force bounded in its disk by the kind's coefficient times the
point's normal force (``mu lambda_n``). The frame ``t_1, t_2`` across the normal only lays the rows out: the disk is
round, so the solution is the frame's in no way.

**The solve** (decided Fr1, measured by the friction spike): projected Gauss-Seidel over the held points, each visit
the point's own updates repeated :data:`FRICTION_REPEATS` times (its normal row's exact update, then each friction
pair's exact update within its disk), from zero, the sweep count the hull kind's declaration
(``_rigid_inertial.FRICTION_SWEEPS``). A pair's exact update is the 2 x 2 trust region (:func:`disk_update`): inside
the disk the pair sticks, its slip zero; on its boundary it slides, its slip anti-parallel to its force (maximal
dissipation), found by Newton's steps on the secular equation in variables scaled by the block's mean diagonal and the
free minimiser's length, so that no product leaves float32's range, every input of the branch not taken fed values it
cannot overflow on (``0 x inf`` in reverse mode was the spike's float32 defect). The bound ``mu lambda_n`` moves as the
sweeps move ``lambda_n``, so the problem is not convex and no dual descent guards it: after the sweeps the forces are
scaled along their own ray to where the normal block does no work (:func:`normal_work_free`), and each point's friction
power at the solved twist is checked never positive (:func:`friction_power`, the spike's guard: it caught a planted sign
flip in every draw of every body in both dtypes, where the kinetic energy missed most), the step flagged unconverged
where it is.

**Where friction's force comes from** (decided Fr4): only the hold's own normal force, ``lambda_n``, never the film's
pressure: a body riding a film above the contact's margin takes no normal force, so no friction, exactly. The film's
coupling of rotation and translation is in the resistance the rows are solved through (``M'/dt + R``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from jax import Array, lax

from mime.core import jnp, switch
from mime.nodes.body._contact import repeated

#: Each held point's own updates repeated this many times per visit of the sweeps: they take out the coupling between
#: a point's rows (a sphere's sliding against its rolling resistance, -0.845) as a per-point block solve would, where the
#: arm's stick start harms a redundant multi-point hold (the spike: a box's sweeps to 1e-3 from p99 22 to over 400) and
#: was the only start that raised the kinetic energy. A declaration, measured by the spike (three repeats: a held sphere
#: still to 3.3e-13 m/s).
FRICTION_REPEATS = 3
#: Newton's steps on a sliding pair's secular equation (:func:`disk_update`): from the free minimiser, the multiplier
#: rises monotonically to its root on a symmetric block; the spike's twelve (its round 3) left no measured difference
#: against 400 sweeps' reference.
TRUST_NEWTON = 12
#: The rows a held point carries with the sliding pair: the normal and the pair. Row kinds after it (rolling
#: resistance) add pairs of their own.
NORMAL, SLIDING = 0, (1, 2)


class Traction(NamedTuple):
    """A solve's friction: the row pairs each held point carries beside its normal row (their indices in the point's
    rows, fixed at build) and each pair's coefficient (a traced value: the sliding pair's ``mu``, dimensionless)."""

    pairs: tuple[tuple[int, int], ...]
    coefficients: tuple[Any, ...]

    @property
    def rows(self) -> int:
        return 1 + 2 * len(self.pairs)


def traction_rows(rows: Array, points: Array, centre: Array, along: Array) -> Array:
    """Each held point's rows, ``K x 3 x 6``: its normal row (``rows``, the hold's ``[n; r x n]``) and the sliding pair
    ``[t_i; r x t_i]``, ``t_1`` the direction ``along`` taken across the normal and ``t_2 = n x t_1``. ``along`` is
    never along a normal at a valid state (a tube's axis, across every normal of its wall; a plane's frame fixed at
    build), so the frame has no switch."""
    normals = rows[:, :3]
    arms = points - centre
    across = along - jnp.sum(along * normals, axis=-1, keepdims=True) * normals
    first = across / switch.nonzero_norm(across, axis=-1)[:, None]
    second = jnp.cross(normals, first)
    return jnp.stack([rows, jnp.concatenate([first, jnp.cross(arms, first)], axis=-1),
                      jnp.concatenate([second, jnp.cross(arms, second)], axis=-1)], axis=1)


def _solved2(block: Array, vector: Array) -> Array:
    """``block^-1 vector`` for a 2 x 2 ``block`` by its adjugate: a sliding pair's block is definite at every valid
    state, so its determinant is never zero."""
    a, b, c, d = block[0, 0], block[0, 1], block[1, 0], block[1, 1]
    return jnp.stack([d * vector[0] - b * vector[1], a * vector[1] - c * vector[0]]) / (a * d - b * c)


#: The block a pair's update is fed where its own is not used (a point with no normal row; a boundary not taken):
#: definite, its entries all nonzero, so that no fallback scaling (``switch.replacing_fallbacks``) makes it singular.
_DEFINITE = np.asarray([[2.0, 1.0], [1.0, 2.0]])


def disk_update(block: Array, load: Array, bound: Array) -> Array:
    """A friction pair's exact force with the others held: the ``x`` that leaves the pair's slip ``u = load + block x``
    zero within the disk ``|x| <= bound``, or on its boundary with ``u`` anti-parallel to ``x`` (``(block + nu I) x =
    -load``, ``nu >= 0``): the switch ``contact.friction``, its indicator ``|x_free|^2 - bound^2`` (the free minimiser's
    squared force over the disk's). The boundary's ``nu`` by :data:`TRUST_NEWTON` of Newton's steps on ``1/|y| - 1/R``
    from zero, in variables scaled by the block's mean diagonal and the free minimiser's length ``|x_free|`` (so the
    scaled load is bounded by the block's condition and ``R = bound/|x_free|`` lies below one), ``R`` taken above
    ``eps^2`` so that a bound at rounding is read as a bound and no step leaves the dtype's range; the force then
    ``bound`` along ``y``, on the boundary to rounding. Where the boundary is not taken its inputs are fed the unit
    load, a radius of a half and a definite block (domain guards), on which no step overflows, so no ``0 x inf``
    reaches reverse mode.
    The pair sticks only strictly inside its disk: at a bound of zero (a point its normal row leaves free) with no
    load, the boundary is taken, ``0`` times a direction from those same inputs, so the force and its derivative are
    zero (the free minimiser's derivative there would read friction at a point with no normal force)."""
    free = -_solved2(block, load)
    size = jnp.sum(free * free)
    outside = size >= bound * bound
    reached = outside & (size > 0)
    length = jnp.sqrt(switch.domain_guard(reached, size, 1.0))
    scale = 0.5 * (block[0, 0] + block[1, 1])
    scaled = switch.domain_guard(reached, block / scale, _DEFINITE.astype(load.dtype))
    target = switch.domain_guard(reached, load / (scale * length), jnp.asarray([1.0, 0.0], dtype=load.dtype))
    radius = switch.domain_guard(reached, bound / length, 0.5) + jnp.finfo(load.dtype).eps ** 2
    eye = jnp.eye(2, dtype=load.dtype)

    def newton(_: Any, nu: Array) -> Array:
        shifted = scaled + nu * eye
        y = -_solved2(shifted, target)
        reach = switch.nonzero_norm(y)
        direction = y / reach  # the step's slope over |y|^2, taken on the unit direction: no product underflows
        return nu + (reach / radius - 1.0) / jnp.dot(direction, _solved2(shifted, direction))

    nu = repeated(TRUST_NEWTON, newton, jnp.zeros((), dtype=load.dtype), loops=True)
    y = -_solved2(scaled + nu * eye, target)
    boundary = bound * y / switch.nonzero_norm(y)
    return switch.where(outside, boundary, free, owner="contact.friction", indicator=size - bound * bound,
                        resolution=lambda: switch.rounding(size + bound * bound))


def point_update(own: Array, velocity: Array, held: Array, traction: Traction) -> tuple[Array, Array]:
    """One held point's own updates with every other point's forces held: its normal row's exact update (kept at
    zero or above: ``contact.impulse``), then each friction pair's (:func:`disk_update`) within its disk at the normal
    force just found. ``own`` is the point's ``R x R`` block of the coupling, ``velocity`` its rows' velocities at the
    current forces, ``held`` its forces; both returned moved. A point whose normal row has no length (on the surface's
    axis, where it has no normal) is never moved: its normal force stays zero, and its pairs, whose second tangent has
    no length there either, are fed a definite block (domain guards), so they take no force and no 0/0 reaches the
    solve."""
    old = held[NORMAL]
    normal_row = own[NORMAL, NORMAL] > 0
    diagonal = switch.domain_guard(normal_row, own[NORMAL, NORMAL], 1.0)
    moved = old - velocity[NORMAL] / diagonal
    normal = switch.maximum(moved, 0.0, owner="contact.impulse", indicator=moved,
                            resolution=lambda: switch.rounding(jnp.abs(old) + jnp.abs(velocity[NORMAL] / diagonal)))
    velocity = velocity + own[:, NORMAL] * (normal - old)
    held = held.at[NORMAL].set(normal)
    for pair, coefficient in zip(traction.pairs, traction.coefficients, strict=True):
        index = np.asarray(pair)
        before = held[index]
        block = switch.domain_guard(normal_row, own[index[:, None], index[None, :]], _DEFINITE.astype(own.dtype))
        force = disk_update(block, velocity[index] - block @ before, coefficient * normal)
        velocity = velocity + own[:, index] @ (force - before)
        held = held.at[index].set(force)
    return velocity, held


def frictional_sweeps(coupling: Array, start: Array, traction: Traction, count: int, *, loops: bool,
                      initial: Array | None = None) -> Array:
    """The held points' forces (``K x R``: each point's normal force, then each friction pair's) after ``count``
    projected Gauss-Seidel sweeps over the points from ``initial`` (zero where none is given), each visit
    :data:`FRICTION_REPEATS` of the point's own updates (:func:`point_update`). ``coupling`` is the Delassus matrix ``J
    A^-1 J^T`` laid out ``K x R x K x R`` and ``start`` the rows' velocities at no force (``J V_free``, the normal rows'
    allowances added). XLA loops, as the contact 0.5.0's sweeps, unrolled while a switch collector is open
    (``_contact.repeated``)."""
    points = start.shape[0]

    def visit(k: Any, forces: Array) -> Array:
        velocity = start[k] + jnp.einsum("ilj,lj->i", coupling[k], forces)

        def again(_: Any, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            return point_update(coupling[k, :, k, :], carry[0], carry[1], traction)

        return forces.at[k].set(repeated(FRICTION_REPEATS, again, (velocity, forces[k]), loops=loops)[1])

    return repeated(count, lambda _, forces: repeated(points, visit, forces, loops=loops),
                    jnp.zeros_like(start) if initial is None else initial, loops=loops)


def normal_work_free(coupling: Array, start: Array, forces: Array) -> Array:
    """``forces`` scaled along their own ray to where the normal block does no work on the motion they leave: ``t =
    -f_n.b_n / f_n.(C f)_n``, so that the normal forces' power at the held twist is ``-t f_n.g/dt``, never positive,
    whatever the sweeps left; the friction scaled with them stays in its cone (a cone holds every non-negative
    multiple). At the exact solution ``t`` is one, by the normal rows' complementarity; with no friction rows it is
    ``_contact.no_work``'s scaling. Unchanged where no normal force is taken."""
    points, rows = start.shape
    flat = coupling.reshape(points * rows, points * rows)
    pushed = (flat @ forces.reshape(-1)).reshape(points, rows)
    stiffness = jnp.dot(forces[:, NORMAL], pushed[:, NORMAL])
    taken = stiffness > 0
    return forces * switch.domain_guard(taken, -jnp.dot(forces[:, NORMAL], start[:, NORMAL])
                                        / switch.domain_guard(taken, stiffness, 1.0), 1.0)


def slip(coupling: Array, start: Array, forces: Array) -> Array:
    """Each held point's rows' velocities at the solved forces, ``K x R``: the normal rows' less their allowances'
    (``start`` holds them), and each friction pair's slip at the body's point."""
    points, rows = start.shape
    return start + (coupling.reshape(points * rows, points * rows) @ forces.reshape(-1)).reshape(points, rows)


def friction_power(coupling: Array, start: Array, forces: Array, pairs: Sequence[tuple[int, int]], rows: Array,
                   moved: Array) -> Array:
    """The guard: each held point's friction power at the solved twist, ``P = sum_g l_g . u_g``, over its scale ``D =
    sum_g |l_g| (sqrt(|b_g|^2 + |u_g|^2) + c U)``, the largest over the points: never above :func:`power_floor` at a
    converged solve, whose friction never pushes the motion it leaves. ``b_g`` is the pair's free slip (in the scale
    so that a sticking point, its slip at rounding, reads its rounding, and a planted sign flip reads about one: the
    spike's normalisation), ``U`` the free step's fastest held point's speed (``rows``' levers, the twist ``moved``)
    and ``c = ROUNDING_ULPS sqrt(eps)``, so that a power at the rounding of its own arithmetic (``64 eps |l| U``) reads
    under the line where every slip is at rounding (a body at rest). Each force's length is taken on its largest
    component, so a share at the body's rounding (one of a sphere's eight groups sharing its point) does not underflow
    its own scale. Taken on stopped values: a report, through which no derivative passes."""
    coupling, start, forces, rows, moved = lax.stop_gradient((coupling, start, forces, rows, moved))
    moving = slip(coupling, start, forces)
    tiny = jnp.finfo(start.dtype).tiny
    linear = jnp.sqrt(jnp.sum(moved[:3] * moved[:3]))
    angular = jnp.sqrt(jnp.sum(moved[3:] * moved[3:]))
    lever = jnp.sqrt(jnp.max(jnp.sum(rows[:, :, 3:] * rows[:, :, 3:], axis=-1), axis=-1))
    fastest = jnp.max(linear + lever * angular)
    rounding = switch.ROUNDING_ULPS * float(np.sqrt(np.finfo(np.dtype(start.dtype)).eps))
    power = jnp.zeros(start.shape[:1], dtype=start.dtype)
    measure = jnp.zeros(start.shape[:1], dtype=start.dtype)
    for pair in pairs:
        index = np.asarray(pair)
        force, rate, free = forces[:, index], moving[:, index], start[:, index]
        largest = jnp.max(jnp.abs(force), axis=-1, keepdims=True)
        length = largest[:, 0] * jnp.sqrt(jnp.sum((force / (largest + tiny)) ** 2, axis=-1))
        power = power + jnp.sum(force * rate, axis=-1)
        measure = measure + length * (jnp.sqrt(jnp.sum(free * free + rate * rate, axis=-1)) + rounding * fastest)
    return jnp.max(power / (measure + tiny))


#: The guard's line in float64, a share of a point's friction power's scale: declared above what a converged
#: multi-point hold reads (a cylinder lying on its side, held at its eight groups along its line of contact, 3.8e-8 at
#: 4,000 sweeps, unchanged from 400: the conformance suite's draw) and far below a planted sign flip's (about one).
POWER_LINE = 1e-4


def power_floor(dtype: Any) -> float:
    """The guard's line in ``dtype``: :data:`POWER_LINE`, or the square root of the dtype's resolution where that is
    larger (float32's 3.5e-4, the friction spike's float32 line of 1e-3 to within a factor of three)."""
    return max(POWER_LINE, float(np.sqrt(np.finfo(dtype).eps)))


def cone_residual(coupling: Array, start: Array, forces: Array, traction: Traction, rows: Array,
                  moved: Array) -> Array:
    """How far the solve is from exact Coulomb, a report: the normal rows' unmet complementarity as
    ``_contact.residual`` reads it, ``min(f C_kk, b + C f)``, and each friction pair's natural-map residual ``|x -
    P(x - u / c)| c`` (``P`` the disk's projection, ``c`` the pair's mean diagonal: zero iff the pair sticks inside its
    disk or slides on it anti-parallel to its slip), in m/s, its largest over the speed the free step gives the held
    points (``_contact.residual``'s), so that ``CONVERGED`` reads it as it reads the frictionless residual. On stopped
    values."""
    coupling, start, forces, rows, moved = lax.stop_gradient((coupling, start, forces, rows, moved))
    points, count = start.shape
    left = slip(coupling, start, forces)
    diagonal = jnp.diagonal(coupling.reshape(points * count, points * count)).reshape(points, count)
    worst = jnp.abs(jnp.minimum(forces[:, NORMAL] * diagonal[:, NORMAL], left[:, NORMAL]))
    for pair, coefficient in zip(traction.pairs, traction.coefficients, strict=True):
        index = np.asarray(pair)
        own = jnp.mean(diagonal[:, index], axis=-1, keepdims=True)
        trial = forces[:, index] - left[:, index] / own
        reach = jnp.sqrt(jnp.sum(trial * trial, axis=-1, keepdims=True))
        bound = lax.stop_gradient(coefficient) * forces[:, NORMAL:NORMAL + 1]
        projected = trial * jnp.minimum(1.0, bound / (reach + jnp.finfo(start.dtype).tiny))
        worst = jnp.maximum(worst, jnp.sqrt(jnp.sum((forces[:, index] - projected) ** 2, axis=-1)) * own[:, 0])
    linear = jnp.sqrt(jnp.sum(moved[:3] * moved[:3]))
    angular = jnp.sqrt(jnp.sum(moved[3:] * moved[3:]))
    lever = jnp.sqrt(jnp.sum(rows[:, 3:] * rows[:, 3:], axis=-1))
    return jnp.max(worst) / (jnp.max(linear + lever * angular) + jnp.finfo(moved.dtype).tiny)


__all__ = ["FRICTION_REPEATS", "NORMAL", "POWER_LINE", "SLIDING", "TRUST_NEWTON", "Traction", "cone_residual",
           "disk_update", "friction_power", "frictional_sweeps", "normal_work_free", "point_update", "power_floor",
           "slip", "traction_rows"]
