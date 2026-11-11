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

**Stall restarts** (``stall_restarts``, 0 by default, as the FVM's CG's ``Stopping.stall_restarts``): up to that many
times a stagnation begins the column's window again, from the best true residual so far, before one stops it. A run's
GMRES takes none, and with none the solve is traced as it was without them.

**A stop at the floor only** (``floor_stop``, off by default): past its stall restarts, a stagnation stops a column only
where its least true residual is at its **rounding floor**, within :data:`FLOOR_MARGIN` of ``eps || |A| |x| + |b| ||``
at the iterate of that residual (the FVM's ``rounding_floor``: the residual of an ``x`` that solves the system with each
coefficient and datum moved by one rounding unit, Oettli and Prager's componentwise backward error at ``eps``;
:data:`FLOOR_MEASURED`). Above it the stagnation is a stall of
the restarted method, not a floor, and the window begins again, the column running on to its cap. A gradient study's
recheck of a member whose solve stopped short sets it, with the cap lifted and the stall restarts raised
(:func:`recheck_stopping`), so that a recheck's solve that stops stagnated is at its floor, and one that stalls above
it converges or reaches the lifted cap: the recheck names the cause from that (``mime.study``). It needs the operator in
magnitude (``magnitude``: ``|A|`` applied to a block of columns); off, nothing of it is traced.

**Starts are cold** (the stall decision's item 4): every solve, primal and derivative alike, starts from zero, so a
tangent or transpose solve never starts at the primal's warm start, whose error would be set by the primal's scale.

**The report** is the FVM's (``mime.nodes.flow.fvm._solve``'s ``REPORT_FIELDS``: one report for every iterative
solve in MIME), over the columns: the most iterations a column took (the operator's applications), the largest true
relative residual (-1 where one is not finite), and how many columns stopped at the cap, stagnated, at a residual not
finite, or were accepted at a zero right-hand side. Counts add where reports are summed, so each keeps its meaning."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
from jax import Array

from mime.core import jnp
from mime.core import solves
from mime.core.solves import SolveSite, register
from mime.nodes.flow.fvm._solve import DEFAULT_CAP, RECHECK_CAP_FACTOR, REPORT_FIELDS, REPORT_SIZE, STALL_RESTARTS
from mime.nodes.hydro.mrs._solve import PRECISION

_lax: Any = cast(Any, jax).lax
_vmap: Any = cast(Any, jax).vmap
_solve_triangular: Any = cast(Any, jax).scipy.linalg.solve_triangular

type Operator = Callable[[Array], Array]

#: The solve site of :func:`reported_solve`: every GMRES solve of the near field runs inside it.
GMRES_SITE = register(SolveSite("near_field_gmres", "gmres", "mime.nodes.hydro.near_field._gmres.reported_solve",
                                nodes=("JointDragNode",)))

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
#: How far above ``eps || |A| |x| + |b| ||`` a stagnated column's least true residual may lie and still be at its floor
#: (``floor_stop``), and what set it.
FLOOR_MARGIN = 4.0
FLOOR_MEASURED = (
    "measured through the node's own step on a GitHub runner (probe run 42925275946), at a tolerance under every floor "
    "(1e-9 in float32, 1e-15 in float64), MIME's sphere coarse (0.4 mm spacing: pairs 1 to 0.03 mm apart, a triangle, "
    "four in a line 0.05 mm apart) and at 882 force points (pairs 1 and 0.05 mm apart, three in a line, to 7,938 "
    "unknowns): each column stopped at its floor lay at 0.33 to 2.72 times eps || |A| |x| + |b| || in either dtype. "
    "Stalls planted with a Krylov length of 1, explored on the node's joint matrix of three coarse spheres, stopped 6.7 "
    "to 6.6e5 times above it in float32 and 2e8 to 3.6e14 in float64; through Study.gradient's recheck (run "
    "42925968039) each such stall ran on past the floor test, never named floor-limited")


@dataclass(frozen=True)
class Stopping:
    """When GMRES stops: the relative tolerance on each column's true residual (a ``params`` leaf: the
    accuracy-bearing setting), the basis's length before a restart, the cap in iterations, the stagnation's window
    (restarts) and factor, how many times a stagnation begins the window again before one stops it, and whether past
    those a stagnation stops a column only at its rounding floor (``floor_stop``, the module's)."""

    tolerance: Any
    restart: int = RESTART
    cap: int = CAP
    window: int = STAGNATION_WINDOW
    stagnation: float = STAGNATION_FACTOR
    stall_restarts: int = 0
    floor_stop: bool = False


def recheck_stopping(stopping: Stopping) -> Stopping:
    """``stopping`` with the cause of a stop short removed, as a recheck re-solves a member: by the FVM's CG's rule
    (``mime.nodes.flow.fvm._solve.recheck_stopping``) the cap lifted to the larger of :data:`RECHECK_CAP_FACTOR` times
    itself and :data:`DEFAULT_CAP` and the stall restarts raised by :data:`STALL_RESTARTS`, at the same tolerance; and
    a stagnation past them stopping a column only at its rounding floor (``floor_stop``)."""
    return dataclasses.replace(stopping, cap=max(stopping.cap * RECHECK_CAP_FACTOR, DEFAULT_CAP),
                               stall_restarts=stopping.stall_restarts + STALL_RESTARTS, floor_stop=True)


def _norms(x: Array) -> Array:
    return jnp.sqrt(jnp.sum(x * x, axis=0))


def gmres(operator: Operator, b: Array, precondition: Operator, stopping: Stopping,
          magnitude: Operator | None = None) -> tuple[Array, Array]:
    """``A x = b`` for each column of ``b`` (``(n, c)``) by restarted GMRES right-preconditioned by ``precondition``
    (``M``, near ``A^-1``), from a cold start, stopping as the module states; the answers ``(n, c)`` and the report
    (:data:`REPORT_FIELDS`). ``operator``, ``precondition`` and ``magnitude`` (``|A|``, for a stop at the floor) act on
    ``(n, c)`` arrays column by column. Each column is solved scaled by its largest magnitude, which changes no
    iterate's direction and keeps every inner product clear of float32's underflow."""
    if stopping.floor_stop and magnitude is None:
        raise ValueError("a stop at the rounding floor needs the operator's magnitude")
    size = jnp.max(jnp.abs(b), axis=0)
    scale = jnp.where(size > 0.0, size, 1.0)
    x, report = _scaled_gmres(operator, b / scale, precondition, stopping, magnitude)
    return x * scale, report


def _scaled_gmres(operator: Operator, b: Array, precondition: Operator, stopping: Stopping,
                  magnitude: Operator | None) -> tuple[Array, Array]:
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
        return jnp.any(carry[9] == _RUNNING)

    allowed = int(stopping.stall_restarts)
    eps = jnp.finfo(dtype).eps

    def begun_again(again: Array, held: Array, earlier: Array, counted: Array) -> tuple[Array, Array, Array]:
        """The window begun again, from the best so far, in the columns ``again``."""
        return (jnp.where(again[None, :], jnp.inf, held),
                jnp.where(again, jnp.minimum(earlier, jnp.min(held, axis=0)), earlier), jnp.where(again, 0, counted))

    def restart(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        x, r, beta, count, recent, before, checks, kept, lowest, ended, *stalls = carry
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
        counted = checks + 1
        if allowed:  # a stall restart begins the window again from the best so far; none traces no such step
            again = stagnated & (stalls[0] < allowed)
            stagnated = stagnated & ~again
            held, earlier, counted = begun_again(again, held, earlier, counted)
            stalls = [stalls[0] + again.astype(jnp.int32)]
        lower = running & (true_norm < lowest)
        kept = jnp.where(lower[None, :], x, kept)
        lowest = jnp.where(lower, true_norm, lowest)
        if magnitude is not None and stopping.floor_stop:  # past them, a stagnation above the floor is a stall
            floor = eps * _norms(magnitude(jnp.abs(kept)) + jnp.abs(b))
            stalled = stagnated & ~(lowest <= FLOOR_MARGIN * floor)
            stagnated = stagnated & ~stalled
            held, earlier, counted = begun_again(stalled, held, earlier, counted)
        status = jnp.where(~jnp.isfinite(true_norm), _NOT_FINITE, jnp.where(
            converged, _CONVERGED, jnp.where(stagnated, _STAGNATED, jnp.where(count >= stopping.cap, _AT_CAP,
                                                                              _RUNNING))))
        return (x, jnp.where(running[None, :], r, carry[1]), jnp.where(running, true_norm, beta), count,
                jnp.where(running[None, :], held, recent), jnp.where(running, earlier, before),
                jnp.where(running, counted, checks), kept, lowest,
                jnp.where(running, status, ended).astype(jnp.int32), *stalls)

    zeros = jnp.zeros_like(b)
    initial = (zeros, b, norm_b, jnp.zeros(columns, dtype=jnp.int32),
               jnp.full((stopping.window, columns), jnp.inf, dtype=dtype), norm_b, jnp.zeros(columns, dtype=jnp.int32),
               zeros, norm_b, ended0.astype(jnp.int32), *([jnp.zeros(columns, dtype=jnp.int32)] if allowed else []))
    x, _, last, count, _, _, _, kept, lowest, ended, *_ = _lax.while_loop(going, restart, initial)
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
    right-hand side, and the transpose solve the transposed preconditioner, ``transposed``, and for a stop at the floor
    (``floor_stop``) the matrix in magnitude, transposed with it. Returns the answers and the primal solve's report. The
    blocks are coupled to the field by exact zeros, so that each has a defined tangent whenever anything in the solve
    is differentiated."""

    with solves.solving(GMRES_SITE):
        def matvec(pair: tuple[Array, Array]) -> tuple[Array, Array]:
            x, block = pair
            y = jnp.matmul(matrix, x, precision=PRECISION)
            return y, block + 0.0 * y[0, 0]

        def solving(preconditioner: Operator, transpose: bool) -> Callable[..., tuple[Array, Array]]:
            def magnitude(x: Array) -> Array:
                held = jnp.abs(matrix)
                return jnp.matmul(held.T if transpose else held, x, precision=PRECISION)

            def solve(operator: Callable[[tuple[Array, Array]], tuple[Array, Array]],
                      pair: tuple[Array, Array]) -> tuple[Array, Array]:
                rhs, block = pair

                def field(x: Array) -> Array:
                    return operator((x, block))[0]

                return gmres(field, rhs, preconditioner, stopping, magnitude if stopping.floor_stop else None)

            return solve

        coupled_rhs = b + 0.0 * tap[0]
        coupled_tap = tap + 0.0 * b[0, 0]
        solved, report = _lax.custom_linear_solve(matvec, (coupled_rhs, coupled_tap), solving(precondition, False),
                                                  transpose_solve=solving(transposed, True))
        return solved, report


__all__ = ["CAP", "FLOOR_MARGIN", "FLOOR_MEASURED", "REPORT_FIELDS", "REPORT_SIZE", "RESTART", "STAGNATION_FACTOR", "STAGNATION_WINDOW", "Stopping",
           "gmres", "recheck_stopping", "reported_solve"]
