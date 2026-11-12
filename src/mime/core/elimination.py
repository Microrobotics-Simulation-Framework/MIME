"""The one elimination over a solved field's exchange partners (C5 point 3; SF-4): every partner's deliveries solved
together with the field's affine response within the step, generic over what the partners are.

**What a partner gives.** A partner (an opening, a point body) delivers into the field through a few unit deliveries
(an opening's pressure, a body's force and couple); the field solves one unit field for each (extra right-hand sides
of its own step) and reads every unit field for every partner (its readings' response: an opening's flux, a body's
gathered velocity and rotation), so a partner's readings are ``y = y0 + R x`` over every partner's deliveries ``x``,
the cross terms included. Each partner states its rows over the step from them (:class:`PartnerRows`): an opening's
law's (``p = a + B q`` or ``q = Q``), a body's drag law's. A partner whose own unknowns the field cannot close (a
body whose own step solves its twist: F5's handed design) leaves them free, a column each on the right-hand side; the
answer is then affine in them, ``x = x0 + X v`` (:class:`Eliminated`), and the partner's own rows close them
(:func:`closed`) where the field holds those rows, or the partner takes the affine map (its load and response) and
closes them in its own step.

**One dense system.** The partners' rows are stacked in the partners' order and solved by one dense LU
(``mime.core.solves.dense_solve``), equilibrated (:func:`equilibrated_solve`); a single partner's rows without free
unknowns are solved as they are, so a field with openings alone eliminates them by the very operations it did before
the bodies joined (its programs keep their bits)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
from jax import Array

from mime.core import jnp
from mime.core.solves import dense_solve


class PartnerRows(NamedTuple):
    """One partner's rows over the step, ``matrix @ x = right + free @ v``: ``matrix`` over every partner's deliveries
    ``x`` (its readings' response already folded in), ``right`` the rows' right-hand side at its free unknowns ``v``
    zero, and ``free`` each free unknown's column on the right-hand side (``None`` for a partner the field closes)."""

    matrix: Array
    right: Array
    free: Array | None = None


class Eliminated(NamedTuple):
    """The deliveries solving every partner's rows: ``deliveries`` at the free unknowns zero and ``sensitivity`` each
    delivery's change per unit of each free unknown, the free unknowns in the partners' order (``None`` where no
    partner left any free)."""

    deliveries: Array
    sensitivity: Array | None = None


def _powers_of_two(largest: Array) -> Array:
    """The power of two that brings each of ``largest`` into [1/2, 1) (one where it is zero), held out of the
    derivative: a scaling by it is exact."""
    _, exponent = jnp.frexp(jax.lax.stop_gradient(jnp.where(largest > 0.0, largest, 1.0)))
    return jnp.ldexp(jnp.ones_like(largest), -exponent)


def equilibrated_solve(matrix: Array, right: Array) -> Array:
    """``matrix^-1 right`` by a dense LU (``dense_solve``) of ``matrix`` with each column and then each row scaled by a
    power of two to a largest magnitude in [1/2, 1), the scalings exact and held out of the derivative, and one step
    of refinement in the run's dtype (the residual of the scaled system solved again), which leaves each row's
    residual at its own terms' rounding (Skeel's componentwise backward stability). A partner's rows and deliveries
    carry their own units (a body's force against its couple, some ten decades apart in SI on a micrometre body): an
    LU of the matrix as it stands rounds every entry against the largest, which in float32 moved two bodies' forces by
    up to 1e-3 of themselves (#270's review, B1; the same step with these solves in float64 agreed with the monolithic
    step to 1.5e-6), and scaled without the refinement a body's spin was still 4e-4 of itself off at some draws, about
    40 times gate 1's line, where its own row mixes its spin with the other body's terms."""
    columns = _powers_of_two(jnp.max(jnp.abs(matrix), axis=0))
    scaled = matrix * columns[None, :]
    rows = _powers_of_two(jnp.max(jnp.abs(scaled), axis=1))
    system = rows[:, None] * scaled
    given = rows[:, None] * right if right.ndim == 2 else rows * right
    solved = dense_solve(system, given)
    solved = solved + dense_solve(system, given - system @ solved)
    return columns[:, None] * solved if solved.ndim == 2 else columns * solved


def exchange_system(partners: Sequence[PartnerRows]) -> Eliminated:
    """Every partner's deliveries from their rows stacked in order, by one dense LU (equilibrated); with free unknowns,
    the right-hand side of each as a further column of the one solve."""
    if not partners:
        raise ValueError("an elimination needs one partner or more")
    frees = [p.free for p in partners if p.free is not None]
    if len(partners) == 1 and not frees:
        return Eliminated(dense_solve(partners[0].matrix, partners[0].right))
    matrix = jnp.concatenate([p.matrix for p in partners], axis=0)
    right = jnp.concatenate([p.right for p in partners], axis=0)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"the partners' rows stack to a {matrix.shape[0]} x {matrix.shape[1]} system: each partner "
                         f"gives as many rows as it has deliveries, each row over every partner's deliveries")
    if not frees:
        return Eliminated(equilibrated_solve(matrix, right))
    dtype = right.dtype
    width = sum(int(free.shape[1]) for free in frees)
    columns: list[Array] = []
    start = 0
    for p in partners:
        rows = int(p.right.shape[0])
        block = jnp.zeros((rows, width), dtype=dtype)
        if p.free is not None:
            count = int(p.free.shape[1])
            block = block.at[:, start:start + count].set(p.free)
            start += count
        columns.append(block)
    solved = equilibrated_solve(matrix, jnp.concatenate([right[:, None], jnp.concatenate(columns, axis=0)], axis=1))
    return Eliminated(solved[:, 0], solved[:, 1:])


def closed(eliminated: Eliminated, own: Array, delivered: Array, right: Array) -> tuple[Array, Array]:
    """The free unknowns ``v`` closed by their partners' own rows ``own @ v + delivered @ x = right`` (a body's step:
    its stepping matrix on its twist, its load from the field among the deliveries), and the deliveries there:
    ``(own + delivered X) v = right - delivered x0``, one dense LU (equilibrated)."""
    sensitivity = eliminated.sensitivity
    if sensitivity is None:
        raise ValueError("no partner left an unknown free: nothing to close")
    free = equilibrated_solve(own + delivered @ sensitivity, right - delivered @ eliminated.deliveries)
    return free, eliminated.deliveries + sensitivity @ free


__all__ = ["Eliminated", "PartnerRows", "closed", "equilibrated_solve", "exchange_system"]
