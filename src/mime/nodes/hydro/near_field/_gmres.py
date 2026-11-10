"""GMRES for the near field's joint system, each body's own block its preconditioner, every solve reported.

**The solve** (:func:`gmres`). Restarted GMRES, right-preconditioned, on columns of right-hand sides at once: each
column its own solve, its own Krylov space, its own tolerance (relative to its own right-hand side) and its own stop;
the operator applied to every column together, so that one product with the joint matrix serves them all. Right
preconditioning minimises the true residual itself, so the Arnoldi estimate is the true residual's in exact
arithmetic; the basis is orthogonalised by classical Gram-Schmidt twice over (as stable as the modified form, and one
product a pass).

**Stopping is never silent** (F0's B4). A column stops on its **true** residual ``b - A x``, recomputed at each
restart from the iterate, relative to its right-hand side: a cycle ends at its ``restart`` length, or early where the
column's estimate reaches the tolerance, and the restart then recomputes the residual and starts the next cycle from
it. So the recurrence's drift from the true residual (float32's stall in the CG, which replacing its residual broke) is
bounded to one cycle: a column whose estimate claims the tolerance while its true residual does not is restarted from
the true one, never accepted. What remains is either a floor (the dtype's own rounding of ``A x`` against ``b``,
which no restart lowers) or a convergence stall of the restarted method: both stop as **stagnated**, over the last
``window`` restarts none falling below ``stagnation`` times the best before them. Or a column stops at its **cap**
(iterations); these two return the iterate of the least true residual checked. A residual that is not finite returns
**NaN**. A zero right-hand side is accepted at once, its answer zero (no iteration needed: the start is cold).

**Starts are cold** (the stall decision's item 4): every solve, primal and derivative alike, starts from zero, so a
tangent or transpose solve never starts at the primal's warm start, whose error would be set by the primal's scale.

**The report** is the FVM's (``mime.nodes.flow.fvm._solve``'s ``REPORT_FIELDS``: one report for every iterative
solve in MIME), over the columns: the most iterations a column took (the operator's applications), the largest true
relative residual (-1 where one is not finite), and how many columns stopped at the cap, stagnated, at a residual not
finite, or were accepted at a zero right-hand side. Counts add where reports are summed, so each keeps its meaning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._solve import REPORT_FIELDS, REPORT_SIZE
from mime.nodes.hydro.mrs._solve import PRECISION

_lax: Any = cast(Any, jax).lax
_vmap: Any = cast(Any, jax).vmap
_solve_triangular: Any = cast(Any, jax).scipy.linalg.solve_triangular

type Operator = Callable[[Array], Array]

#: How a column ended, as the loop carries it.
_RUNNING, _CONVERGED, _STAGNATED, _AT_CAP, _NOT_FINITE, _AT_ZERO = 0, 1, 2, 3, 4, 5

#: The Krylov basis's length before a restart, the stagnation's window in restarts and its factor, and the cap in
#: iterations: labelled defaults. Measured on MIME's sphere (pairs a radius to a twentieth of one apart, a triangle, a
#: line of four and a cube of eight): to 1e-10 in float64, 11 to 61 iterations; float32's floor the same at restarts of
#: 10, 30 and 60; a tolerance under the floor stops stagnated three restarts after reaching it, far before the cap.
RESTART = 30
STAGNATION_WINDOW = 3
STAGNATION_FACTOR = 0.5
CAP = 600


@dataclass(frozen=True)
class Stopping:
    """When GMRES stops: the relative tolerance on each column's true residual (a ``params`` leaf: the
    accuracy-bearing setting), the basis's length before a restart, the cap in iterations, and the stagnation's window
    (restarts) and factor."""

    tolerance: Any
    restart: int = RESTART
    cap: int = CAP
    window: int = STAGNATION_WINDOW
    stagnation: float = STAGNATION_FACTOR


def _norms(x: Array) -> Array:
    return jnp.sqrt(jnp.sum(x * x, axis=0))


def gmres(operator: Operator, b: Array, precondition: Operator, stopping: Stopping) -> tuple[Array, Array]:
    """``A x = b`` for each column of ``b`` (``(n, c)``) by restarted GMRES right-preconditioned by ``precondition``
    (``M``, near ``A^-1``), from a cold start, stopping as the module states; the answers ``(n, c)`` and the report
    (:data:`REPORT_FIELDS`). ``operator`` and ``precondition`` act on ``(n, c)`` arrays column by column. Each column is
    solved scaled by its largest magnitude, which changes no iterate's direction and keeps every inner product clear of
    float32's underflow."""
    size = jnp.max(jnp.abs(b), axis=0)
    scale = jnp.where(size > 0.0, size, 1.0)
    x, report = _scaled_gmres(operator, b / scale, precondition, stopping)
    return x * scale, report


def _scaled_gmres(operator: Operator, b: Array, precondition: Operator, stopping: Stopping) -> tuple[Array, Array]:
    dtype = b.dtype
    n, columns = b.shape
    m = stopping.restart
    tiny = jnp.finfo(dtype).tiny
    norm_b = _norms(b)
    threshold = stopping.tolerance * norm_b
    ended0 = jnp.where(~jnp.isfinite(norm_b), _NOT_FINITE, jnp.where(norm_b == 0.0, _AT_ZERO, _RUNNING))
    eye = jnp.eye(m, dtype=dtype)

    def cycle(x: Array, r: Array, beta: Array, running: Array, count: Array) -> tuple[Array, Array]:
        """One cycle of at most ``m`` Arnoldi steps from the residual ``r`` (norm ``beta``) in each running column;
        the correction it gives each, and the steps each took."""
        safe = jnp.where(beta > 0.0, beta, 1.0)
        basis = jnp.zeros((m + 1, n, columns), dtype=dtype).at[0].set(r / safe)
        g = jnp.zeros((m + 1, columns), dtype=dtype).at[0].set(beta)
        hessenberg = jnp.zeros((m, m, columns), dtype=dtype)
        cosines = jnp.ones((m, columns), dtype=dtype)
        sines = jnp.zeros((m, columns), dtype=dtype)
        active = running & (count < stopping.cap)

        def going(carry: tuple[Any, ...]) -> Array:
            j, active = carry[0], carry[6]
            return (j < m) & jnp.any(active)

        def step(carry: tuple[Any, ...]) -> tuple[Any, ...]:
            j, basis, g, hessenberg, cosines, sines, active, length = carry
            w = operator(precondition(basis[j]))
            h = jnp.einsum("knc,nc->kc", basis, w, precision=PRECISION)
            w = w - jnp.einsum("knc,kc->nc", basis, h, precision=PRECISION)
            again = jnp.einsum("knc,nc->kc", basis, w, precision=PRECISION)
            w = w - jnp.einsum("knc,kc->nc", basis, again, precision=PRECISION)
            h = h + again
            below = _norms(w)
            h = h.at[j + 1].set(below)
            next_vector = w / jnp.where(below > tiny, below, 1.0)

            def rotate(i: Array, held: Array) -> Array:
                upper, lower = held[i], held[i + 1]
                turned_upper = cosines[i] * upper + sines[i] * lower
                turned_lower = -sines[i] * upper + cosines[i] * lower
                apply = i < j
                return held.at[i].set(jnp.where(apply, turned_upper, upper)).at[i + 1].set(
                    jnp.where(apply, turned_lower, lower))

            h = _lax.fori_loop(0, m, rotate, h)
            top, bottom = h[j], h[j + 1]
            rho = jnp.sqrt(top * top + bottom * bottom)
            has = rho > 0.0
            cosine = jnp.where(has, top / jnp.where(has, rho, 1.0), 1.0)
            sine = jnp.where(has, bottom / jnp.where(has, rho, 1.0), 0.0)
            h = h.at[j].set(rho).at[j + 1].set(0.0)
            estimate = -sine * g[j]
            g_next = g.at[j + 1].set(estimate).at[j].set(cosine * g[j])
            keep = active[None, :]
            basis = basis.at[j + 1].set(jnp.where(keep, next_vector, basis[j + 1]))
            g = jnp.where(keep, g_next, g)
            hessenberg = hessenberg.at[:, j].set(jnp.where(keep, h[:m], hessenberg[:, j]))
            cosines = cosines.at[j].set(jnp.where(active, cosine, cosines[j]))
            sines = sines.at[j].set(jnp.where(active, sine, sines[j]))
            length = jnp.where(active, j + 1, length)
            taken = count + length
            # A column leaves the cycle where its estimate reaches its tolerance (the restart then checks the true
            # residual), where its basis broke down (the space holds the answer), where its estimate is not a number,
            # or at the cap.
            done = (jnp.abs(estimate) <= threshold) | (below <= tiny) | ~jnp.isfinite(estimate) \
                | (taken >= stopping.cap)
            return j + 1, basis, g, hessenberg, cosines, sines, active & ~done, length

        start = (jnp.asarray(0, jnp.int32), basis, g, hessenberg, cosines, sines, active,
                 jnp.zeros(columns, dtype=jnp.int32))
        _, basis, g, hessenberg, _, _, _, length = _lax.while_loop(going, step, start)
        inside = jnp.arange(m)[:, None] < length[None, :]
        square = (inside[:, None, :] & inside[None, :, :])
        upper = jnp.where(square, hessenberg, eye[:, :, None])
        def back_substituted(triangle: Array, rhs: Array) -> Array:
            return cast(Array, _solve_triangular(triangle, rhs, lower=False))

        y = _vmap(back_substituted, in_axes=(2, 1), out_axes=1)(upper, jnp.where(inside, g[:m], 0.0))
        correction = precondition(jnp.einsum("knc,kc->nc", basis[:m], y, precision=PRECISION))
        return jnp.where(length[None, :] > 0, correction, 0.0), length

    def going(carry: tuple[Any, ...]) -> Array:
        return jnp.any(carry[-1] == _RUNNING)

    def restart(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        x, r, beta, count, recent, before, checks, kept, lowest, ended = carry
        running = ended == _RUNNING
        correction, length = cycle(x, r, beta, running, count)
        x = jnp.where(running[None, :], x + correction, x)
        count = count + length
        r = b - operator(x)
        true_norm = _norms(r)
        converged = true_norm <= threshold
        held = jnp.concatenate([recent[1:], true_norm[None]])
        earlier = jnp.minimum(before, recent[0])
        full = checks + 1 >= stopping.window
        stagnated = ~converged & full & (jnp.min(held, axis=0) > stopping.stagnation * earlier)
        lower = running & (true_norm < lowest)
        status = jnp.where(~jnp.isfinite(true_norm), _NOT_FINITE, jnp.where(
            converged, _CONVERGED, jnp.where(stagnated, _STAGNATED, jnp.where(count >= stopping.cap, _AT_CAP,
                                                                              _RUNNING))))
        return (x, jnp.where(running[None, :], r, carry[1]), jnp.where(running, true_norm, beta), count,
                jnp.where(running[None, :], held, recent), jnp.where(running, earlier, before),
                jnp.where(running, checks + 1, checks), jnp.where(lower[None, :], x, kept),
                jnp.where(lower, true_norm, lowest), jnp.where(running, status, ended).astype(jnp.int32))

    zeros = jnp.zeros_like(b)
    initial = (zeros, b, norm_b, jnp.zeros(columns, dtype=jnp.int32),
               jnp.full((stopping.window, columns), jnp.inf, dtype=dtype), norm_b, jnp.zeros(columns, dtype=jnp.int32),
               zeros, norm_b, ended0.astype(jnp.int32))
    x, _, last, count, _, _, _, kept, lowest, ended = _lax.while_loop(going, restart, initial)
    # A column that converged (or was accepted) returns its last iterate; one stopped short, the iterate of its least
    # true residual checked (past a floor the iterate wanders); one whose residual went non-finite, NaN.
    short = (ended == _STAGNATED) | (ended == _AT_CAP)
    x = jnp.where(short[None, :], kept, x)
    x = jnp.where((ended == _NOT_FINITE)[None, :], jnp.nan, x)
    final = jnp.where(short, lowest, jnp.where(ended == _AT_ZERO, 0.0, last))
    relative = final / jnp.maximum(norm_b, tiny)
    finite = jnp.all(jnp.isfinite(relative)) & ~jnp.any(ended == _NOT_FINITE)
    report = jnp.stack([jnp.max(count).astype(dtype), jnp.where(finite, jnp.max(relative), -1.0).astype(dtype),
                        jnp.sum(ended == _AT_CAP).astype(dtype), jnp.sum(ended == _STAGNATED).astype(dtype),
                        jnp.sum(ended == _NOT_FINITE).astype(dtype), jnp.sum(ended == _AT_ZERO).astype(dtype)])
    return x, report


def reported_solve(matrix: Array, b: Array, tap: Array, precondition: Operator, transposed: Operator,
                   stopping: Stopping) -> tuple[Array, Array]:
    """``matrix^-1 b`` for each column of ``b`` by :func:`gmres` through ``custom_linear_solve``, with the report block
    on its right-hand side (``tap``, :data:`REPORT_SIZE` zeros, the block's input, whose cotangent is the transpose
    solve's report): the primal, the tangent and the transpose solve each take their tolerance from their own
    right-hand side, and the transpose solve the transposed preconditioner, ``transposed``. Returns the answers and
    the primal solve's report. The blocks are coupled to the field by exact zeros, so that each has a defined tangent
    whenever anything in the solve is differentiated."""

    def matvec(pair: tuple[Array, Array]) -> tuple[Array, Array]:
        x, block = pair
        y = jnp.matmul(matrix, x, precision=PRECISION)
        return y, block + 0.0 * y[0, 0]

    def solving(preconditioner: Operator) -> Callable[..., tuple[Array, Array]]:
        def solve(operator: Callable[[tuple[Array, Array]], tuple[Array, Array]],
                  pair: tuple[Array, Array]) -> tuple[Array, Array]:
            rhs, block = pair

            def field(x: Array) -> Array:
                return operator((x, block))[0]

            return gmres(field, rhs, preconditioner, stopping)

        return solve

    coupled_rhs = b + 0.0 * tap[0]
    coupled_tap = tap + 0.0 * b[0, 0]
    solved, report = _lax.custom_linear_solve(matvec, (coupled_rhs, coupled_tap), solving(precondition),
                                              transpose_solve=solving(transposed))
    return solved, report


__all__ = ["CAP", "REPORT_FIELDS", "REPORT_SIZE", "RESTART", "STAGNATION_FACTOR", "STAGNATION_WINDOW", "Stopping",
           "gmres", "reported_solve"]
