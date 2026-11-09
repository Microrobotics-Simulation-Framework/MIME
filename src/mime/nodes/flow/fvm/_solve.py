"""The solve: the preconditioner contract, Jacobi, the guarded conjugate gradient and its report block.

**Stopping is never silent** (F0's B4 and the stall decision). The CG stops on the **true** residual, recomputed as
``b - A x`` from its iterate, relative to the right-hand side (its last iterate returned); or at a stagnation (over the last ``window`` checks of the
true residual, none fell below ``stagnation`` times the best before them: CG's residual is not monotone, least of all
against Jacobi's contrast between the fluid and a stiff interior, so a stagnation is judged against the best over a
window, never against the last check); or at its cap: these two return the better of its last iterate and the one of
the least true residual checked; or at a residual that is not finite, which **returns NaN**, never its start or a
checked iterate: a NaN that arises in a step (its drive, its operator) is not a number in the step's answer either, so
the run's own non-finite refusal names the flow. It takes at
least one iteration, so a warm start whose residual is already below the tolerance is never accepted unsolved; only an
exactly zero residual is (a zero right-hand side from a zero start among them).

**The report block** (``BACKWARD_DIAGNOSTICS``'s design, one for every iterative solve in MIME). The solve's
right-hand side carries a block of :data:`REPORT_SIZE` numbers on which the operator is the identity, and the solve
writes its own report into that block of its answer. JAX's ``custom_linear_solve`` calls the same solve for the
primal, the tangent and the transpose solve, so each writes its own: the primal's is the block of the answer, the
tangent's is that block's tangent, the transpose's is the cotangent of the block's input, a zero ``params`` leaf the
step reads. The blocks are coupled to the field by exact zeros, so that each has a defined tangent whenever anything
in the solve is differentiated, and the report is always finite (a residual that is not is written as -1, with its
count), so the zeros add nothing.

**Starts that scale.** The start is made inside the solve from its own right-hand side, the Galerkin multiple of the
previous solution ``x0 = alpha g``, ``alpha = <g, b> / <g, A g>``: linear in ``b``, so a tangent or transpose solve starts
at a guess in proportion to its own right-hand side, never at the primal's warm start, whose error would be set by the
primal's scale. On the primal it is the best multiple of the plain warm start in the operator's norm."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._grid import MACGrid, Velocity

lax: Any = jax.lax

#: What a solve reports, in its block's order: its iterations; its true relative residual (-1 where not finite); and
#: one count each for how it stopped other than converging: at its cap, stagnated, at a residual not finite, and
#: accepted unsolved at an exactly zero residual. Counts add where reports are summed, so each keeps its meaning.
REPORT_FIELDS = ("iterations", "residual", "at_cap", "stagnated", "not_finite", "accepted_at_zero")
REPORT_SIZE = len(REPORT_FIELDS)

#: How a solve ended, as its loop carries it.
_RUNNING, _CONVERGED, _STAGNATED, _AT_CAP, _NOT_FINITE, _AT_ZERO = 0, 1, 2, 3, 4, 5

type Operator = Callable[[Velocity], Velocity]


#: The stagnation stop, set from measurement (``STAGNATION_MEASURED``): over a window of checks of the true residual,
#: none fell below ``STAGNATION_FACTOR`` times the best before them. Jacobi's CG takes iterations in proportion to the
#: cells across (and so do its plateaus), so the flow's window spans ``STAGNATION_SPAN`` iterations per cell along the
#: box's longest side, and at least ``STAGNATION_LEAST_WINDOW`` checks (:func:`stagnation_window`); ``Stopping``'s own
#: default is ``STAGNATION_WINDOW`` checks.
STAGNATION_FACTOR = 0.5
STAGNATION_SPAN = 12
STAGNATION_LEAST_WINDOW = 5
STAGNATION_WINDOW = 10
STAGNATION_MEASURED = (
    "measured: every solve of the empty tube's 0.1h and 0.01h interiors at D/h 8, 16, 32 and 64, IMEX Euler under "
    "Stokes at dt 10 ms and 0.1 ms and the steady limit, in both dtypes, replayed check by check with no stagnation "
    "stop: the window clears every converging solve, by 2.6 times the least window that does at the tightest (float32, "
    "0.01h, D/h 32; 3.8 in float64 at D/h 64), and float32 at D/h 64 with 0.01h stalls above its 1e-5 (least true "
    "residuals 1.1e-5 to 1.3e-4 over 5000 iterations), flagged; replayed again at D/h 16 to 48 (131 solves a dtype, a "
    "tilted box among them), it stops no converging solve in float64, and float32 with 0.01h stalls from D/h 48: "
    "there 3 of 6 solves stopped stagnated at 1.03 to 1.21 times its 1e-5, each converging later (at 1814 to 4306 "
    "iterations), flagged; a tolerance below rounding stops stagnated before the cap, at 240 to 940 iterations at D/h "
    "8 to 32")


@dataclass(frozen=True)
class Stopping:
    """When the CG stops: the relative tolerance on the true residual (a ``params`` leaf: the accuracy-bearing
    setting, 1e-8 in float64 and 1e-5 in float32 by default), the cap, how often the true residual is checked, the
    stagnation's window (in checks) and the factor a check in it must beat the best before it by, and the least
    iterations taken from a nonzero residual."""

    tolerance: Any
    cap: int = 5000
    check_every: int = 20
    window: int = STAGNATION_WINDOW
    stagnation: float = STAGNATION_FACTOR
    least_iterations: int = 1


DEFAULT_TOLERANCE = {"float64": 1e-8, "float32": 1e-5}


def stagnation_window(cells: tuple[int, ...], check_every: int) -> int:
    """The stagnation's window, in checks, for a box of ``cells``: :data:`STAGNATION_SPAN` iterations per cell along
    its longest side, and at least :data:`STAGNATION_LEAST_WINDOW` checks."""
    return max(STAGNATION_LEAST_WINDOW, math.ceil(STAGNATION_SPAN * max(cells) / check_every))


def default_tolerance(dtype: Any) -> float:
    return DEFAULT_TOLERANCE[str(jnp.dtype(dtype))]


def _add(a: Velocity, b: Velocity, scale: Any = 1.0) -> Velocity:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


def _scaled(a: Velocity, scale: Any) -> Velocity:
    return (scale * a[0], scale * a[1], scale * a[2])


def _finite_or_zero(value: Array) -> Array:
    return jnp.where(jnp.isfinite(value), value, 0.0)


def _select(choose: Array, a: Velocity, b: Velocity) -> Velocity:
    return (jnp.where(choose, a[0], b[0]), jnp.where(choose, a[1], b[1]), jnp.where(choose, a[2], b[2]))


class Preconditioner:
    """A registered preconditioner: what it needs from the grid kind, and its map. It may change a solve's cost, never
    its answer beyond the tolerance; under CG it must be symmetric positive definite; it need not be differentiable
    (gradients pass through ``custom_linear_solve``'s rule); its transpose is used in the transpose solve."""

    name: ClassVar[str]
    version: ClassVar[str]
    needs: ClassVar[tuple[str, ...]]
    symmetric: ClassVar[bool]

    def made(self, diagonal: Velocity) -> Operator:
        raise NotImplementedError


class Jacobi(Preconditioner):
    """The operator's diagonal, inverted: any grid kind gives it. The default (the only one measured in 3-D)."""

    name = "jacobi"
    version = "0.1.0"
    needs = ("diagonal",)
    symmetric = True

    @override
    def made(self, diagonal: Velocity) -> Operator:
        def apply(r: Velocity) -> Velocity:
            return (r[0] / diagonal[0], r[1] / diagonal[1], r[2] / diagonal[2])
        return apply


PRECONDITIONERS: dict[str, Preconditioner] = {Jacobi.name: Jacobi()}
DEFAULT_PRECONDITIONER = Jacobi.name


def preconditioner_named(name: str) -> Preconditioner:
    if name not in PRECONDITIONERS:
        raise ValueError(f"no preconditioner {name!r} is registered: {sorted(PRECONDITIONERS)}")
    return PRECONDITIONERS[name]


def guarded_cg(operator: Operator, b: Velocity, precondition: Operator, start: Velocity, stopping: Stopping,
               inner: Callable[[Velocity, Velocity], Array]) -> tuple[Velocity, Array]:
    """The preconditioned CG on ``operator x = b`` from ``start``, stopping as the module states; its answer and its
    report (:data:`REPORT_FIELDS`). The system is solved scaled by the right-hand side's largest magnitude, which
    changes no iterate's direction and keeps every inner product clear of float32's underflow (a drive of 1e-12 in SI
    on micrometre cells squares below the smallest normal number, and 0/0 follows)."""
    size = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in b]))
    scale = jnp.where(size > 0.0, size, 1.0)
    x, report = _scaled_cg(operator, _scaled(b, 1.0 / scale), precondition, _scaled(start, 1.0 / scale), stopping,
                           inner)
    return _scaled(x, scale), report


def _scaled_cg(operator: Operator, b: Velocity, precondition: Operator, start: Velocity, stopping: Stopping,
               inner: Callable[[Velocity, Velocity], Array]) -> tuple[Velocity, Array]:
    dtype = b[0].dtype
    norm_b = jnp.sqrt(inner(b, b))
    threshold = stopping.tolerance * norm_b
    r0 = _add(b, operator(start), -1.0)
    z0 = precondition(r0)
    true0 = jnp.sqrt(inner(r0, r0))
    # A start already within the tolerance is accepted unsolved only where no iteration is required: at an exactly zero
    # residual, or with least_iterations 0 (the spikes' old path, kept for the stall's control).
    accepted = (stopping.least_iterations == 0) & (true0 <= threshold)
    ended0 = jnp.where(~jnp.isfinite(true0), _NOT_FINITE,
                       jnp.where(true0 == 0.0, _AT_ZERO, jnp.where(accepted, _CONVERGED, _RUNNING)))

    def going(carry: tuple[Any, ...]) -> Array:
        return carry[-1] == _RUNNING

    def iterate(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        x, r, z, d, rz, count, recent, best, checks, kept, lowest, _ = carry
        ad = operator(d)
        # A breakdown (a curvature or an rz that underflowed to zero, past the dtype's rounding floor) takes no step,
        # so the iterate stays finite and the window sees the residual stop falling: a stagnation, not a NaN.
        step = _finite_or_zero(rz / inner(d, ad))
        x = _add(x, d, step)
        r = _add(r, ad, -step)
        z = precondition(r)
        rz_next = inner(r, z)
        d = _add(z, d, _finite_or_zero(rz_next / rz))
        count = count + 1
        recursive_small = jnp.sqrt(inner(r, r)) <= threshold
        at_check = (count % stopping.check_every) == 0

        def checked(_: Any) -> tuple[Any, ...]:
            true_r = _add(b, operator(x), -1.0)
            true_norm = jnp.sqrt(inner(true_r, true_r))
            converged = (true_norm <= threshold) & (count >= stopping.least_iterations)
            # A regular check enters the window; the one it pushes out joins the best before it.
            held = jnp.concatenate([recent[1:], true_norm[None]])
            before = jnp.minimum(best, recent[0])
            full = checks + 1 >= stopping.window
            stagnated = (~converged) & at_check & full & (jnp.min(held) > stopping.stagnation * before)
            restart = recursive_small & (~converged)
            true_z = precondition(true_r)
            ended = jnp.where(~jnp.isfinite(true_norm), _NOT_FINITE,
                              jnp.where(converged, _CONVERGED, jnp.where(stagnated, _STAGNATED, _RUNNING)))
            # The iterate of the least true residual checked, which a solve stopped short returns.
            lower = true_norm < lowest
            return (_select(restart, true_r, r), _select(restart, true_z, z), _select(restart, true_z, d),
                    jnp.where(restart, inner(true_r, true_z), rz_next), jnp.where(at_check, held, recent),
                    jnp.where(at_check, before, best), jnp.where(at_check, checks + 1, checks),
                    _select(lower, x, kept), jnp.where(lower, true_norm, lowest), ended.astype(jnp.int32))

        def unchecked(_: Any) -> tuple[Any, ...]:
            ended = jnp.where(jnp.isfinite(rz_next), _RUNNING, _NOT_FINITE).astype(jnp.int32)
            return r, z, d, rz_next, recent, best, checks, kept, lowest, ended

        r, z, d, rz_next, recent, best, checks, kept, lowest, ended = lax.cond(
            recursive_small | at_check, checked, unchecked, None)
        ended = jnp.where((ended == _RUNNING) & (count >= stopping.cap), _AT_CAP, ended).astype(jnp.int32)
        return x, r, z, d, rz_next, count, recent, best, checks, kept, lowest, ended

    window = jnp.full((stopping.window,), jnp.inf, dtype=dtype)
    initial = (start, r0, z0, z0, inner(r0, z0), jnp.asarray(0, jnp.int32), window, true0,
               jnp.asarray(0, jnp.int32), start, true0, ended0.astype(jnp.int32))
    x, _, _, _, _, count, _, _, _, kept, _, ended = lax.while_loop(going, iterate, initial)
    # A solve that converged (or was accepted) returns its last iterate; one stopped short, the better of its last and
    # its best checked one: past the rounding floor the iterate wanders (a planted stall's grew to a true residual of
    # 7e-2 in float64 at D/h 16), and a cap short of the first check has only its last. One whose residual went
    # non-finite returns NaN: its start (zero, for a NaN right-hand side) would be a velocity the step never solved for.
    last = _add(b, operator(x), -1.0)
    held = _add(b, operator(kept), -1.0)
    last_norm, held_norm = inner(last, last), inner(held, held)
    better = (ended != _CONVERGED) & (ended != _AT_ZERO) & (~jnp.isfinite(last_norm) | (held_norm < last_norm))
    x = _select(better, kept, x)
    nan = jnp.full_like(x[0], jnp.nan)
    x = _select(ended == _NOT_FINITE, (nan, nan, nan), x)
    final = _select(better, held, last)
    relative = jnp.sqrt(inner(final, final)) / jnp.maximum(norm_b, jnp.finfo(dtype).tiny)
    relative = jnp.where(norm_b == 0.0, jnp.sqrt(inner(final, final)), relative)
    report = jnp.stack([count.astype(dtype), jnp.where(jnp.isfinite(relative), relative, -1.0).astype(dtype),
                        (ended == _AT_CAP).astype(dtype), (ended == _STAGNATED).astype(dtype),
                        (ended == _NOT_FINITE).astype(dtype), (ended == _AT_ZERO).astype(dtype)])
    return x, report


def scaling_start(guess: Velocity | None, b: Velocity, operator: Operator,
                  inner: Callable[[Velocity, Velocity], Array]) -> Velocity:
    """``alpha g``, ``alpha = <g, b> / <g, A g>`` (zero where ``<g, A g>`` is zero), or a zero start without a guess:
    linear in ``b``, so every solve that shares this closure starts in proportion to its own right-hand side."""
    if guess is None:
        return _scaled(b, 0.0)
    # The guess scaled to unit magnitude first: a guess of 1e-20 in float32 squares to a subnormal, and alpha overflows.
    size = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in guess]))
    unit = _scaled(guess, 1.0 / jnp.where(size > 0.0, size, 1.0))
    curvature = inner(unit, operator(unit))
    positive = curvature > 0.0
    alpha = jnp.where(positive, inner(unit, b) / jnp.where(positive, curvature, 1.0), 0.0)
    return _scaled(unit, jnp.where(jnp.isfinite(alpha), alpha, 0.0))


def projected_solve(grid: MACGrid, apply: Operator, diagonal: Velocity, rhs: Velocity, guess: Velocity | None,
                    stopping: Stopping, tap: Array, preconditioner: Preconditioner) -> tuple[Velocity, Array]:
    """``P A P u = P rhs`` in the divergence-free subspace (the null-space form of the saddle-point problem, the
    projector the grid kind's exact one), through ``custom_linear_solve`` with the report block on its right-hand
    side; ``tap`` is the block's input (zeros, :data:`REPORT_SIZE`), whose cotangent is the transpose solve's report.
    Returns the divergence-free solution and the primal solve's report."""
    project = grid.project
    inner = grid.inner
    precondition_raw = preconditioner.made(diagonal)
    projected_guess = None if guess is None else project(guess)

    def precondition(r: Velocity) -> Velocity:
        return project(precondition_raw(project(r)))

    def matvec(pair: tuple[Velocity, Array]) -> tuple[Velocity, Array]:
        x, block = pair
        y = project(apply(project(x)))
        return y, block + 0.0 * y[0][0, 0, 0]

    def solve(operator: Callable[[tuple[Velocity, Array]], tuple[Velocity, Array]],
              pair: tuple[Velocity, Array]) -> tuple[Velocity, Array]:
        b, block = pair

        def field_operator(x: Velocity) -> Velocity:
            return operator((x, block))[0]

        b = project(b)
        start = scaling_start(projected_guess, b, field_operator, inner)
        x, report = guarded_cg(field_operator, b, precondition, start, stopping, inner)
        return project(x), report

    coupled_rhs = _add(rhs, (tap[0], tap[0], tap[0]), 0.0)
    coupled_tap = tap + 0.0 * rhs[0][0, 0, 0]
    solved, report = lax.custom_linear_solve(matvec, (coupled_rhs, coupled_tap), solve, symmetric=True)
    return solved, report


__all__ = ["DEFAULT_PRECONDITIONER", "DEFAULT_TOLERANCE", "STAGNATION_FACTOR", "STAGNATION_LEAST_WINDOW",
           "STAGNATION_MEASURED", "STAGNATION_SPAN", "STAGNATION_WINDOW", "stagnation_window", "PRECONDITIONERS", "REPORT_FIELDS", "REPORT_SIZE", "Jacobi",
           "Preconditioner", "Stopping", "default_tolerance", "guarded_cg", "preconditioner_named", "projected_solve",
           "scaling_start"]
