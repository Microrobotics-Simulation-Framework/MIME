"""The iterative allocation: commands within explicit bounds that minimise a sum of squares of rows a model makes
from them, by projected Newton, with the derivative of the converged solution by the implicit function theorem.

**The problem.** ``min 1/2 |r(u)|^2`` over ``lower <= u <= upper``, ``r`` the caller's rows (each a miss over its scale,
and any regularisation's rows), in any model: an affine one above the exact path's size, a nonlinear one (an arm's
frames from its joint positions). The bounds may be infinite.

**The method** (Bertsekas's projected Newton, 1982). In commands scaled to unit columns of the Jacobian ``J`` (``v = c
u``, ``c`` the columns' lengths, taken again each iteration), each iteration: an epsilon-active set, the commands within
a band of a bound with the gradient pushing them out (the band the lesser of :data:`ACTIVE_BAND` of the problem's size
and the stationarity's length); on the free commands Newton's step on the exact Hessian where it is positive definite
there (its least eigenvalue above ``RCOND`` of its largest), which converges quadratically near a minimiser whatever the
rows' miss, and the Gauss-Newton step ``(Jf^T Jf + floor) d = -Jf^T r`` where it is not (far from one), with a floor of
:data:`FLOOR` on the scaled normal matrix (a Levenberg term at the dtype's rank test, which keeps the step defined
where the rows do not fix the free commands); a scaled gradient step on the held ones; the box as a projection, and
Armijo's test on the projection arc (:data:`ARMIJO`, at most :data:`BACKTRACKS` halvings), the decrease measured as a
difference of squares, ``sum (r - r') (r + r') / 2``, whose subtraction's rounding is the rows' change's and not the
objective's, and a decrease within what the objective resolves passing (a miss that cannot be met, as an unreachable
target's, leaves an objective whose own rounding hides every decrease near the minimiser, where a Newton step still
lowers the stationarity). It starts from the caller's start projected into the box. On an affine model the two steps are
one, Newton's, so the iteration is exact once its active set is the solution's.

**Its guards, declared** (MIME's rule for single precision: a tolerance at or above the dtype's reachable floor, with a
stagnation stop). It stops at :data:`ITERATIONS`, where the stationarity reaches :data:`TOLERANCE` of the problem's
size (then polished: full steps go on while they are longer than the commands' rounding, at most
:data:`POLISHING_STEPS`, since the tolerance bounds the stationarity, not the distance to the minimiser at which the
derivative is taken), or where it stagnates: no step passes Armijo's test, or :data:`STALLS` steps in a row lower the objective by no
more than the objective resolves (about four units in its last place, beside the subtraction's rounding) without
halving the stationarity. The stationarity is the projected gradient's length in the scaled
commands, ``|v - P(v - grad_v)|_inf``, over the problem's size, ``|r| + sum_k c_k |u_k|`` (the rows' miss and what the
commands make, so its rounding floor is the dtype's units in the last place of it whatever the rows cancel).
``converged`` is read again at the commands returned, never from the loop's own account: the stationarity within its
tolerance there, and the free commands' scaled normal matrix of full rank (its least eigenvalue over its largest above
``RCOND^2``), so that the minimiser and its derivative are defined. Where it is not, the commands returned are the last
iterate, inside the box, and the caller says so: ``ctx.allocator`` raises its controller's flag
``allocation_unconverged``, carrying the stationarity.

**The derivative.** ``jax.lax.custom_root`` on ``F(u) = c (u - P(u - grad(u) / c^2))``, ``c`` held (a positive scaling
changes the root of no component): at the root, a free command's row is the gradient's and a held one's ``u - bound``.
The tangent solve is ``F``'s linearisation at the root, taken by autodiff of the exact gradient, so its free rows are
the exact Hessian's, ``J^T J + sum_i r_i grad^2 r_i``, never the Gauss-Newton one (which is wrong wherever the rows are
not met, as an unreachable target's); a held command's derivative is its bound's (a port's, or the slew box's around
the last command). Each closed-over value of the rows (a wanted value, a model's parameters, a point) and of the bounds
carries its tangent through ``F``. The tangent system, ``K x K``, is solved by one dense LU (``mime.core.solves``),
its columns equilibrated. It is the derivative of the converged commands: where the solve did not converge, it is the
implicit function's at the iterate, which solves nothing, and the step's flag says so.

**Its status.** On an affine model the problem is convex: the minimiser is continuous (regularised, or of full column
rank), and its derivative has kinks where a command reaches a bound or leaves it (``piecewise_kink``). On a nonlinear
one the minimiser is the local one reached from the start, which can change branch as what is wanted moves
(``piecewise_jump``). The switch is its caller's owner (:func:`mime.control.allocation_switch`), evaluated once at the
solution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import numpy as np

from mime.control._allocation import RCOND
from mime.core import switch
from mime.core.solves import dense_solve

jnp: Any = jax.numpy
lax: Any = jax.lax
#: JAX's transforms, whose hints a strict checker reads as partly unknown.
_jax: Any = jax

#: The most iterations the solve takes.
ITERATIONS = 50
#: The most halvings of a step Armijo's test takes before the solve counts the iteration as stagnated.
BACKTRACKS = 30
#: Armijo's fraction of the predicted decrease a step must make.
ARMIJO = 1e-4
#: The stationarity, over the problem's size, at which the solve has converged, by dtype: at or above the reachable floor.
TOLERANCE = {np.dtype(np.float64): 1e-10, np.dtype(np.float32): 1e-5}
#: The Levenberg floor on the scaled normal matrix (whose diagonal is one), by dtype: the square of the rank test's.
FLOOR = {dtype: value * value for dtype, value in RCOND.items()}
#: The widest band, as a share of the problem's size in the scaled commands, within which a command at a bound is held.
ACTIVE_BAND = 1e-3
#: The steps in a row that lower the objective by no more than its rounding before the solve stops as stagnated.
STALLS = 2
#: The most full steps taken within the tolerance while the step is longer than the commands' rounding (four units in
#: the last place of the problem's size, in the scaled commands), each where it passes Armijo's test. The tolerance
#: bounds the stationarity, not the distance to the minimiser, which on an ill-conditioned problem (a weak effort, whose
#: rows set the scaled Hessian's least eigenvalues near 1e-8 of its largest) can be far larger; and there the
#: derivative is sensitive to the point it is taken at, the residual's curvature at the iterate's miss moving those
#: eigenvalues by their own size. A warm start near the minimiser meets the tolerance in two steps with the commands
#: 4e-8 rad from it and the derivative 8e-5 off the minimiser's (A0's follower); Newton's steps take them to rounding
#: (quadratic convergence: 2e-6 rad, then 3e-10, then rounding), and the derivative with them.
POLISHING_STEPS = 3

type Rows = Callable[[Any], Any]


class IterativeAllocation(NamedTuple):
    """The commands ``(K,)``; each command's place, ``active`` ``(K,)``, ``-1`` held at its lower bound, ``0`` free,
    ``+1`` at its upper one; the minimised ``objective``, the sum of the squares of the rows; whether it ``converged``
    (read at the commands returned); its ``stationarity`` there, over the problem's size; and the ``iterations`` it
    took."""

    commands: Any
    active: Any
    objective: Any
    converged: Any
    stationarity: Any
    iterations: Any


class _At(NamedTuple):
    """The rows, their Jacobian, its columns' lengths, the gradient, the stationarity's vector and the size, at a
    point."""

    rows: Any
    jacobian: Any
    columns: Any
    gradient: Any
    projected: Any
    size: Any


def _at(rows: Rows, u: Any, lower: Any, upper: Any) -> _At:
    """The solve's view of the problem at ``u``: none of it is differentiated (the derivative is the root's)."""
    def twice(v: Any) -> tuple[Any, Any]:
        made = rows(v)
        return made, made

    jacobian, made = _jax.jacfwd(twice, has_aux=True)(u)
    squared = jnp.sum(jacobian * jacobian, axis=0)
    columns = jnp.sqrt(jnp.where(squared > jnp.finfo(u.dtype).tiny, squared, 1.0))
    gradient = jacobian.T @ made
    projected = columns * (u - jnp.clip(u - gradient / (columns * columns), lower, upper))
    size = jnp.sqrt(jnp.sum(made * made)) + jnp.sum(columns * jnp.abs(u)) + jnp.finfo(u.dtype).tiny
    return _At(made, jacobian, columns, gradient, projected, size)


def _held(at: _At, u: Any, lower: Any, upper: Any, band: Any) -> tuple[Any, Any]:
    """The commands held at their lower and upper bounds: within ``band`` of one (scaled), the gradient pushing out."""
    scaled = at.gradient / at.columns
    return ((at.columns * (u - lower) <= band) & (scaled > 0.0),
            (at.columns * (upper - u) <= band) & (scaled < 0.0))


def _rank(at: _At, free: Any) -> Any:
    """The free commands' scaled normal matrix's least eigenvalue over its largest (one where none is free)."""
    scaled = at.jacobian / at.columns
    mask = free.astype(scaled.dtype)
    normal = (scaled.T @ scaled) * mask[:, None] * mask[None, :] + jnp.diag(1.0 - mask)
    values = jnp.linalg.eigvalsh(normal)
    return values[0] / jnp.maximum(values[-1], jnp.finfo(scaled.dtype).tiny)


def _solved(rows: Rows, start: Any, lower: Any, upper: Any, dtype: Any) -> tuple[Any, tuple[Any, Any, Any, Any]]:
    """The forward solve, with no derivative: the commands, and ``(converged, stationarity, iterations, active)`` read
    at them."""
    held = np.dtype(dtype)
    tolerance, floor, rcond = TOLERANCE[held], FLOOR[held], RCOND[held]
    eps = float(np.finfo(held).eps)
    count = start.shape[0]
    eye = jnp.eye(count, dtype=dtype)

    def objective(u: Any) -> Any:
        made = rows(u)
        return 0.5 * jnp.sum(made * made)

    def iteration(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        u, taken, stalls, _, before, quiet, polished = carry
        at = _at(rows, u, lower, upper)
        measure = jnp.max(jnp.abs(at.projected)) / at.size
        stationary = measure <= tolerance
        # A stall: the last step's decrease within what the objective resolves, and the stationarity not halved by it.
        stalls = jnp.where(quiet & (measure >= 0.5 * before), stalls + 1, 0)
        band = jnp.minimum(ACTIVE_BAND * at.size, jnp.sqrt(jnp.sum(at.projected * at.projected)))
        low, high = _held(at, u, lower, upper, band)
        free = ~(low | high)
        mask = free.astype(dtype)
        pair = mask[:, None] * mask[None, :]
        outside = jnp.diag(1.0 - mask)
        scaled = at.jacobian / at.columns
        # Newton's step on the free commands where the exact Hessian is positive definite there (near a minimiser,
        # where it converges quadratically whatever the rows' miss), the Gauss-Newton one where it is not.
        exact = _jax.hessian(objective)(u) / (at.columns[:, None] * at.columns[None, :]) * pair + outside
        values = jnp.linalg.eigvalsh(exact)
        definite = values[0] > rcond * jnp.maximum(values[-1], 1.0)
        normal = (scaled.T @ scaled) * pair + outside + floor * eye
        newton = -dense_solve(jnp.where(definite, exact, normal), mask * at.gradient / at.columns)
        step = jnp.where(free, newton, -at.gradient / at.columns) / at.columns
        # What the objective resolves: its rows' own rounding, about eps of each row's size, makes the measured
        # decrease uncertain by about eps |r|^2 whatever the step, beside the subtraction's eps |r - r'| |r + r'|.
        resolved = 4.0 * eps * jnp.sum(at.rows * at.rows)

        def trial(alpha: Any) -> tuple[Any, Any, Any]:
            moved = jnp.clip(u + alpha * step, lower, upper)
            predicted = alpha * jnp.sum(jnp.where(free, -at.gradient * step, 0.0)) + \
                jnp.sum(jnp.where(free, 0.0, at.gradient * (u - moved)))
            after = rows(moved)
            # The decrease as a difference of squares, whose subtraction's rounding is the rows' change's.
            change = at.rows - after
            decrease = 0.5 * jnp.sum(change * (at.rows + after))
            noise = resolved + 4.0 * eps * jnp.sum(jnp.abs(change) * (jnp.abs(at.rows) + jnp.abs(after)))
            # A decrease within its noise passes too: near the floor a Newton step lowers the stationarity by more
            # than the objective can show, and the stall count above stops a step that does not.
            return moved, decrease <= noise, decrease + noise >= ARMIJO * predicted

        def halve(search: tuple[Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any]:
            alpha, tries, _, _, _ = search
            return alpha * 0.5, tries + 1, *trial(alpha * 0.5)

        def searching(search: tuple[Any, Any, Any, Any, Any]) -> Any:
            return (~search[4]) & (search[1] < BACKTRACKS)

        first = trial(jnp.ones((), dtype=dtype))
        _, _, moved, small, passed = lax.while_loop(searching, halve,
                                                    (jnp.ones((), dtype=dtype), jnp.zeros((), dtype=np.int32), *first))
        # Polishing: within the tolerance, full steps go on (each passing Armijo's test, at most POLISHING_STEPS) while
        # the step is longer than the commands' rounding at the problem's size, so that what is returned, and
        # linearised for the derivative, is the minimiser to rounding, not a point the tolerance merely admits.
        longer = jnp.max(at.columns * jnp.abs(first[0] - u)) > 4.0 * eps * at.size
        polish = stationary & first[2] & longer & (polished < POLISHING_STEPS)
        stop = (stationary & ~polish) | ~passed | (stalls >= STALLS)
        after = jnp.where(polish, first[0], jnp.where(stop, u, moved))
        return (after, taken + jnp.where(stop, 0, 1).astype(np.int32), stalls, stop, measure, small,
                polished + polish.astype(np.int32))

    first = (start, jnp.zeros((), dtype=np.int32), jnp.zeros((), dtype=np.int32), jnp.zeros((), dtype=bool),
             jnp.asarray(np.inf, dtype=dtype), jnp.zeros((), dtype=bool), jnp.zeros((), dtype=np.int32))
    def going(carry: tuple[Any, ...]) -> Any:
        return (~carry[3]) & (carry[1] < ITERATIONS)

    u, taken, _, _, _, _, _ = lax.while_loop(going, iteration, first)
    at = _at(rows, u, lower, upper)
    stationarity = jnp.max(jnp.abs(at.projected)) / at.size
    low, high = _held(at, u, lower, upper, jnp.minimum(ACTIVE_BAND * at.size,
                                                       jnp.sqrt(jnp.sum(at.projected * at.projected))))
    converged = (stationarity <= tolerance) & (_rank(at, ~(low | high)) > floor)
    active = jnp.where(low, -1, jnp.where(high, 1, 0)).astype(np.int8)
    return u, (converged, stationarity, taken, active)


def _gradient(rows: Rows, u: Any) -> Any:
    """The gradient of ``1/2 |rows(u)|^2``, exact: its linearisation is the exact Hessian's."""
    def objective(v: Any) -> Any:
        made = rows(v)
        return 0.5 * jnp.sum(made * made)

    return _jax.grad(objective)(u)


def _bounds(lower: Any, upper: Any) -> tuple[Any, Any]:
    """The bounds as the root reads them, each carrying its tangent: a held command's derivative is its bound's."""
    return lower, upper


def _root(rows: Rows, lower: Any, upper: Any) -> Callable[[Any], Any]:
    """``F(u) = c (u - P(u - grad / c^2))``, zero exactly at a stationary point in the box, ``c`` the columns' lengths
    held (a scaling that moves no root)."""

    def residual(u: Any) -> Any:
        jacobian = lax.stop_gradient(_jax.jacfwd(rows)(u))
        squared = jnp.sum(jacobian * jacobian, axis=0)
        columns = jnp.sqrt(jnp.where(squared > jnp.finfo(u.dtype).tiny, squared, 1.0))
        low, high = _bounds(lower, upper)
        return columns * (u - jnp.clip(u - _gradient(rows, u) / (columns * columns), low, high))

    return residual


def _tangent(linear: Callable[[Any], Any], rhs: Any) -> Any:
    """The tangent system ``F'(u*) x = rhs``, ``K x K``, by one dense LU of its matrix with its columns equilibrated."""
    matrix = _jax.jacfwd(linear)(rhs)
    largest = jnp.max(jnp.abs(matrix), axis=0)
    scale = jnp.where(largest > jnp.finfo(matrix.dtype).tiny, largest, 1.0)
    return dense_solve(matrix / scale[None, :], rhs) / scale


def iterative_allocation(rows: Rows, start: Any, *, lower: Any, upper: Any, owner: str) -> IterativeAllocation:
    """The commands within ``[lower, upper]`` that minimise ``1/2 |rows(u)|^2``, by projected Newton from ``start``
    (see the module), with the derivative of the converged commands by the implicit function theorem. ``rows`` maps
    commands ``(K,)`` to rows ``(M,)`` in their dtype, each a miss over its scale, and may close over traced values,
    each of which carries its derivative; ``lower`` and ``upper`` ``(K,)`` or one value, possibly infinite, ``lower <=
    upper`` the caller's. ``owner`` is the switch's owner, which the calling entry declares
    (:func:`mime.control.allocation_switch`): the active set, evaluated once at the solution."""
    first = jnp.asarray(start)
    dtype = first.dtype if jnp.issubdtype(first.dtype, jnp.floating) else jnp.zeros(()).dtype
    first = first.astype(dtype)
    low = jnp.broadcast_to(jnp.asarray(lower, dtype=dtype), first.shape)
    high = jnp.broadcast_to(jnp.asarray(upper, dtype=dtype), first.shape)
    guess = jnp.clip(first, low, high)
    def solve(_: Any, initial: Any) -> tuple[Any, tuple[Any, Any, Any, Any]]:
        return _solved(rows, initial, low, high, dtype)

    with switch.not_collected():
        commands, (converged, stationarity, taken, active) = lax.custom_root(_root(rows, low, high), guess, solve,
                                                                             _tangent, has_aux=True)
    inside = jnp.minimum(commands - low, high - commands)
    switch.predicate(active == 0, owner=owner, resolution=lambda: switch.rounding(jnp.maximum(jnp.abs(low), jnp.abs(high))),
                     indicator=lambda: inside)
    made = rows(commands)
    return IterativeAllocation(commands, active, jnp.sum(made * made), converged, stationarity, taken)


__all__ = ["ACTIVE_BAND", "ARMIJO", "BACKTRACKS", "FLOOR", "ITERATIONS", "POLISHING_STEPS", "STALLS", "TOLERANCE",
           "IterativeAllocation", "iterative_allocation"]
