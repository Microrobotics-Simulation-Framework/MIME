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
(``mime.core.solves.dense_solve``); a single partner's rows are solved as they are, so a field with openings alone
eliminates them by the very operations it did before the bodies joined (its programs keep their bits)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

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


def exchange_system(partners: Sequence[PartnerRows]) -> Eliminated:
    """Every partner's deliveries from their rows stacked in order, by one dense LU; with free unknowns, the
    right-hand side of each as a further column of the one solve."""
    if not partners:
        raise ValueError("an elimination needs one partner or more")
    frees = [p.free for p in partners if p.free is not None]
    if len(partners) == 1 and not frees:
        return Eliminated(dense_solve(partners[0].matrix, partners[0].right))
    matrix = jnp.concatenate([p.matrix for p in partners], axis=0)
    right = jnp.concatenate([p.right for p in partners], axis=0)
    if not frees:
        return Eliminated(dense_solve(matrix, right))
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
    solved = dense_solve(matrix, jnp.concatenate([right[:, None], jnp.concatenate(columns, axis=0)], axis=1))
    return Eliminated(solved[:, 0], solved[:, 1:])


def closed(eliminated: Eliminated, own: Array, delivered: Array, right: Array) -> tuple[Array, Array]:
    """The free unknowns ``v`` closed by their partners' own rows ``own @ v + delivered @ x = right`` (a body's step:
    its stepping matrix on its twist, its load from the field among the deliveries), and the deliveries there:
    ``(own + delivered X) v = right - delivered x0``, one dense LU."""
    sensitivity = eliminated.sensitivity
    if sensitivity is None:
        raise ValueError("no partner left an unknown free: nothing to close")
    free = dense_solve(own + delivered @ sensitivity, right - delivered @ eliminated.deliveries)
    return free, eliminated.deliveries + sensitivity @ free


__all__ = ["Eliminated", "PartnerRows", "closed", "exchange_system"]
