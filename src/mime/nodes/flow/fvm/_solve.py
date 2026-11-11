"""The solve: the preconditioner contract, Jacobi, the guarded conjugate gradient and its report block.

**Stopping is never silent** (F0's B4 and the stall decision). The CG stops on the **true** residual, recomputed as
``b - A x`` from its iterate, relative to the right-hand side (its last iterate returned), at its tolerance or, where
its stopping says so (float32's default), at the solve's own rounding floor if that is higher; or at a stagnation (over
the last ``window`` checks of the true residual, none fell below ``stagnation`` times the best before them: CG's
residual is not monotone, least of all against Jacobi's contrast between the fluid and a stiff interior, so a stagnation
is judged against the best over a window, never against the last check); or at its cap: these two return the better of
its last iterate and the one of the least true residual checked; or at a residual that is not finite, which **returns
NaN**, never its start or a checked iterate: a NaN that arises in a step (its drive, its operator) is not a number in
the step's answer either, so the run's own non-finite refusal names the flow. It takes at least one iteration, so a warm
start whose residual is already below the tolerance is never accepted unsolved; only an exactly zero residual is (a zero
right-hand side from a zero start among them).

**Restarts from the true residual.** The CG restarts from the true residual (``r`` set to it, ``z`` made from it, the
direction ``z``: :func:`restarted`, through which its start is made too) where its recursive residual falls under the
tolerance before the true one, and **where it would stop stagnated**: up to ``stall_restarts`` times
(:data:`STALL_RESTARTS`) a stagnation restarts it, its window begun again, before one stops it, flagged. On stiff solves
in float32 the recurrence stalls far above the solve's rounding floor, and a restart at the stall breaks it
(:data:`STALL_RESTART_MEASURED`); a restart every hundred iterations broke the small stalls too but crippled the large
stiff solves it did not need to touch, and replacing the residual alone, the direction kept, was unreliable. A solve
that never stagnates never restarts so, and its arithmetic is unchanged. The stop judges the true residual alone,
against a bar the iterate raises only while its rounding floor is under a line (below), so no restart passes as
converged a solve whose true residual is above that line.

**The report block** (``BACKWARD_DIAGNOSTICS``'s design, one for every iterative solve in MIME). The solve's
right-hand side carries a block of :data:`REPORT_SIZE` numbers on which the operator is the identity, and the solve
writes its own report into that block of its answer. JAX's ``custom_linear_solve`` calls the same solve for the
primal, the tangent and the transpose solve, so each writes its own: the primal's is the block of the answer, the
tangent's is that block's tangent, the transpose's is the cotangent of the block's input, a zero ``params`` leaf the
step reads. The blocks are coupled to the field by exact zeros, so that each has a defined tangent whenever anything
in the solve is differentiated, and the report is always finite (a residual that is not is written as -1, with its
count), so the zeros add nothing.

**The rounding floor** (:func:`rounding_floor`). An iterate ``x`` stored and multiplied in the run's dtype cannot bring
the true residual much below ``eps || |A| |x| + |b| ||``, the operator's and the right-hand side's own rounding against
each other: the residual of an ``x`` that solves the system with each coefficient and each datum moved by one rounding
unit (Oettli and Prager's componentwise backward error at ``eps``). On fine stiff solves float32's floor sits above its
1e-5 (:data:`FLOOR_MEASURED`), so float32's default tolerance is the larger of the two, made in the program at each
check from the iterate; a tolerance given is held as given. A solve that stalls above the floor still stops stagnated,
and is flagged. Made from the iterate, the floor rises with it: on a singular or inconsistent system the iterate grows
without bound, and its true residual, even one worse than the zero start's, falls under its own floor. So the floor
counts only while it is under a line, :data:`FLOOR_STOP_TOLERANCES` times the tolerance (1e-3 for float32's 1e-5;
:data:`FLOOR_STOP_MEASURED`); past it the true residual cannot be read to the line, and the stop holds the tolerance. A
stop at the floor is accepted only where the true relative residual is itself under the line, and a solve the line
refuses carries on, to stop stagnated or at its cap, flagged. The line separates the genuine floor stops measured from
most non-convergent ones, not from all: a singular system's growing iterate can still stop under it.

**Starts that scale.** The start is made inside the solve from its own right-hand side, the Galerkin multiple of the
previous solution ``x0 = alpha g``, ``alpha = <g, b> / <g, A g>``: linear in ``b``, so a tangent or transpose solve starts
at a guess in proportion to its own right-hand side, never at the primal's warm start, whose error would be set by the
primal's scale. On the primal it is the best multiple of the plain warm start in the operator's norm."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax
from jax import Array

from mime.core import jnp
from mime.core.solves import SolveSite, register, solving
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

#: The solve site of :func:`projected_solve`: every solve of the FVM's guarded conjugate gradient runs inside it.
CG_SITE = register(SolveSite("fvm_cg", "cg", "mime.nodes.flow.fvm._solve.projected_solve",
                             nodes=("ImplicitFlowNode", "OpenFlowNode")))


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


#: How many times a stagnation restarts the CG from the true residual before one stops it (module docstring), and the
#: measurements that chose a restart at the stall.
STALL_RESTARTS = 3
STALL_RESTART_MEASURED = (
    "measured on the solid's interior spike's stenosis (75 %, two diameters long, the 0.01h interior, float32 under "
    "its default stop): without a restart the recurrence stalls above the solve's floor on every off-centre draw of "
    "its narrowing, sub-cell offset and drive tilt (344 of 344 at D/h 8 on two kinds of core, 206 of them stalling on "
    "CI's runners too; 16 of 16 at D/h 16) and at D/h 32, steady and IMEX Euler, while a centred one stalls or not by "
    "the CPU's rounding (14 of 22 at D/h 8 here); a restart where the stop would find the stagnation breaks each (379 "
    "of 379 draws at D/h 8, at 102 to 390 iterations, within 5.4e-7 of float64's answer; 8 of 8 at D/h 16 at 657 to "
    "686; at D/h 32 the steady solve at 1693 and its IMEX Euler steps at 1264 and 1240; at D/h 64 the steady solve at "
    "its floor, at 3522 or 4155 by the kind of core), one restart sufficing on every case counted; a restart every "
    "100 iterations instead broke the small stalls, but at D/h 64 kept the stiff solve from progressing (0.76 at 1540 "
    "iterations, its flux 3.7 % off, where the plain CG reached 2.0e-3) and cost converging solves up to twice the "
    "iterations; replacing the residual alone every 100 iterations, the direction kept, left 13 of 16 D/h 16 draws "
    "stalled")


@dataclass(frozen=True)
class Stopping:
    """When the CG stops: the relative tolerance on the true residual (a ``params`` leaf: the accuracy-bearing
    setting, 1e-8 in float64 and 1e-5 in float32 by default), whether it is raised to the solve's rounding floor where
    that is higher (``at_floor``: float32's default, :func:`rounding_floor`), the cap, how often the true residual is
    checked, the stagnation's window (in checks) and the factor a check in it must beat the best before it by, the
    least iterations taken from a nonzero residual, and how many times a stagnation restarts the CG from the true
    residual before one stops it (:data:`STALL_RESTARTS`; 0 never)."""

    tolerance: Any
    at_floor: bool = False
    cap: int = 5000
    check_every: int = 20
    window: int = STAGNATION_WINDOW
    stagnation: float = STAGNATION_FACTOR
    least_iterations: int = 1
    stall_restarts: int = STALL_RESTARTS


DEFAULT_TOLERANCE = {"float64": 1e-8, "float32": 1e-5}
#: The cap the node declares by default, and how a recheck (a gradient study re-solving a member whose solve stopped
#: short) lifts it: to the larger of this many times the cap it ran with and the default. A recheck also raises the stall
#: restarts by :data:`STALL_RESTARTS`. Neither changes a solve that did not reach its cap or stall, so a recheck of a
#: member moves only the solves that stopped short.
DEFAULT_CAP = 5000
RECHECK_CAP_FACTOR = 10


def recheck_stopping(stopping: Stopping) -> Stopping:
    """``stopping`` with the cause of a stop short removed, as a recheck re-solves a member: the cap lifted
    (:data:`RECHECK_CAP_FACTOR`, :data:`DEFAULT_CAP`) and the stall restarts raised, at the same tolerance."""
    return dataclasses.replace(stopping, cap=max(stopping.cap * RECHECK_CAP_FACTOR, DEFAULT_CAP),
                               stall_restarts=stopping.stall_restarts + STALL_RESTARTS)


def stagnation_window(cells: tuple[int, ...], check_every: int) -> int:
    """The stagnation's window, in checks, for a box of ``cells``: :data:`STAGNATION_SPAN` iterations per cell along
    its longest side, and at least :data:`STAGNATION_LEAST_WINDOW` checks."""
    return max(STAGNATION_LEAST_WINDOW, math.ceil(STAGNATION_SPAN * max(cells) / check_every))


def default_tolerance(dtype: Any) -> float:
    return DEFAULT_TOLERANCE[str(jnp.dtype(dtype))]


#: The dtypes whose default tolerance is raised to the solve's rounding floor where that is higher: float32, whose
#: floor on fine stiff solves sits above its 1e-5; float64's 1e-8 sits orders above its floor on every solve measured.
AT_FLOOR_BY_DEFAULT = frozenset({"float32"})
FLOOR_MEASURED = (
    "measured in float32 on the solid's interior spike's cases: the floor bounds what float32 can reach where its "
    "1e-5 cannot be, the empty tube at D/h 64 with the 0.01h interior (float64's answer rounded to float32 reads "
    "1.6e-5 to 2.1e-5, the floor 2.1e-4; the stop takes 69 iterations, its flux within 1.3e-7 of float64's, where the "
    "fixed 1e-5 stagnated at 1000), and at D/h 16 to 32 it lies above the residual each solve reaches with the 0.1h "
    "interior; with the 0.01h interior in a flow with a pressure gradient (a 75 % stenosis) the CG's recurrence "
    "stalls far above it without a periodic restart (1.0e-4 at D/h 16 and 5.9e-4 at 32, against floors of 4.7e-6 and "
    "1.8e-5), its flux within 1e-7 of float64's: stalls, not floors, which a restart at the stall breaks")
#: The line the rounding floor counts under, in tolerances: past it the stop holds the tolerance (module docstring);
#: and the measurements that set it.
FLOOR_STOP_TOLERANCES = 100
FLOOR_STOP_MEASURED = (
    "measured in float32: genuine floor stops at true residuals of 1.3e-5 to 1.6e-5 (the stenosis at D/h 32 with the "
    "0.01h interior), 4.9e-5 to 5.6e-5 (at D/h 64, 0.01h and 0.1h) and 1.9e-4 (the open tube at D/h 64, under its "
    "floor of 2.1e-4), floors growing about fourfold a doubling of cells across; a singular system (a periodic "
    "Laplacian with no penalty, its inconsistent part 1e-6 to 100 of its range part, 8 to 32 cells a side, with and "
    "without the restart, 102 solves a dtype) was passed as converged at true residuals of 2.6e-4 to 2.6e4 in float32 "
    "(82 solves) and 1.6e-4 to 4e11 in float64 with its floor on (91); under the line 80 of the 82 and 91 of the 91 "
    "end flagged, the 2 left (16 cells a side, the inconsistent part 1e-5) stopping at 2.6e-4, among the genuine "
    "stops' residuals")


def rounding_floor(b: Velocity, magnitude: Operator, inner: Callable[[Velocity, Velocity], Array]
                   ) -> Callable[[Velocity], Array]:
    """The true residual's floor at an iterate ``x``, ``eps || |A| |x| + |b| ||`` in the solve's own norm:
    ``magnitude`` is the operator with every coefficient in magnitude, applied to a field of magnitudes."""
    eps = jnp.finfo(b[0].dtype).eps
    size = (jnp.abs(b[0]), jnp.abs(b[1]), jnp.abs(b[2]))

    def floor(x: Velocity) -> Array:
        held = _add(magnitude((jnp.abs(x[0]), jnp.abs(x[1]), jnp.abs(x[2]))), size)
        return eps * jnp.sqrt(inner(held, held))

    return floor


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
               inner: Callable[[Velocity, Velocity], Array], magnitude: Operator | None = None
               ) -> tuple[Velocity, Array]:
    """The preconditioned CG on ``operator x = b`` from ``start``, stopping as the module states; its answer and its
    report (:data:`REPORT_FIELDS`). ``magnitude`` (the operator with its coefficients in magnitude) gives the rounding
    floor a stop ``at_floor`` takes. The system is solved scaled by the right-hand side's largest magnitude, which
    changes no iterate's direction and keeps every inner product clear of float32's underflow (a drive of 1e-12 in SI
    on micrometre cells squares below the smallest normal number, and 0/0 follows)."""
    if stopping.at_floor and magnitude is None:
        raise ValueError("a stop at the rounding floor needs the operator's magnitude")
    size = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in b]))
    scale = jnp.where(size > 0.0, size, 1.0)
    x, report = _scaled_cg(operator, _scaled(b, 1.0 / scale), precondition, _scaled(start, 1.0 / scale), stopping,
                           inner, magnitude)
    return _scaled(x, scale), report


def restarted(true_r: Velocity, precondition: Operator, inner: Callable[[Velocity, Velocity], Array]
              ) -> tuple[Velocity, Velocity, Array]:
    """The CG (re)started from the true residual ``true_r``: ``r``, ``z`` (the direction too) and ``<r, z>``."""
    true_z = precondition(true_r)
    return true_r, true_z, inner(true_r, true_z)


def _scaled_cg(operator: Operator, b: Velocity, precondition: Operator, start: Velocity, stopping: Stopping,
               inner: Callable[[Velocity, Velocity], Array], magnitude: Operator | None) -> tuple[Velocity, Array]:
    dtype = b[0].dtype
    norm_b = jnp.sqrt(inner(b, b))
    target = stopping.tolerance * norm_b
    floor = rounding_floor(b, magnitude, inner) if stopping.at_floor and magnitude is not None else None
    # The floor counts only while it is under its line: an iterate that grows raises its own floor, and past the line
    # the true residual cannot be read to it.
    line = FLOOR_STOP_TOLERANCES * target

    def threshold_at(x: Velocity) -> Array:
        if floor is None:
            return target
        held = floor(x)
        return jnp.where(held <= line, jnp.maximum(target, held), target)

    threshold = threshold_at(start)
    r0, z0, rz0 = restarted(_add(b, operator(start), -1.0), precondition, inner)
    true0 = jnp.sqrt(inner(r0, r0))
    # A start already within the tolerance is accepted unsolved only where no iteration is required: at an exactly zero
    # residual, or with least_iterations 0 (the spikes' old path, kept for the stall's control).
    accepted = (stopping.least_iterations == 0) & (true0 <= threshold)
    ended0 = jnp.where(~jnp.isfinite(true0), _NOT_FINITE,
                       jnp.where(true0 == 0.0, _AT_ZERO, jnp.where(accepted, _CONVERGED, _RUNNING)))

    def going(carry: tuple[Any, ...]) -> Array:
        return carry[-1] == _RUNNING

    def iterate(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        x, r, z, d, rz, count, recent, best, checks, kept, lowest, threshold, left, _ = carry
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
            reached = threshold_at(x)
            converged = (true_norm <= reached) & (count >= stopping.least_iterations)
            # A regular check enters the window; the one it pushes out joins the best before it.
            held = jnp.concatenate([recent[1:], true_norm[None]])
            before = jnp.minimum(best, recent[0])
            full = checks + 1 >= stopping.window
            stalled = (~converged) & at_check & full & (jnp.min(held) > stopping.stagnation * before)
            retry = stalled & (left > 0)
            stagnated = stalled & (~retry)
            restart = (recursive_small & (~converged)) | retry
            new_r, new_z, new_rz = restarted(true_r, precondition, inner)
            ended = jnp.where(~jnp.isfinite(true_norm), _NOT_FINITE,
                              jnp.where(converged, _CONVERGED, jnp.where(stagnated, _STAGNATED, _RUNNING)))
            # The iterate of the least true residual checked, which a solve stopped short returns.
            lower = true_norm < lowest
            return (_select(restart, new_r, r), _select(restart, new_z, z), _select(restart, new_z, d),
                    jnp.where(restart, new_rz, rz_next),
                    jnp.where(retry, jnp.full_like(recent, jnp.inf), jnp.where(at_check, held, recent)),
                    jnp.where(at_check, before, best), jnp.where(retry, 0, jnp.where(at_check, checks + 1, checks)),
                    _select(lower, x, kept), jnp.where(lower, true_norm, lowest), reached,
                    (left - retry.astype(jnp.int32)).astype(jnp.int32), ended.astype(jnp.int32))

        def unchecked(_: Any) -> tuple[Any, ...]:
            ended = jnp.where(jnp.isfinite(rz_next), _RUNNING, _NOT_FINITE).astype(jnp.int32)
            return r, z, d, rz_next, recent, best, checks, kept, lowest, threshold, left, ended

        r, z, d, rz_next, recent, best, checks, kept, lowest, threshold, left, ended = lax.cond(
            recursive_small | at_check, checked, unchecked, None)
        ended = jnp.where((ended == _RUNNING) & (count >= stopping.cap), _AT_CAP, ended).astype(jnp.int32)
        return x, r, z, d, rz_next, count, recent, best, checks, kept, lowest, threshold, left, ended

    window = jnp.full((stopping.window,), jnp.inf, dtype=dtype)
    initial = (start, r0, z0, z0, rz0, jnp.asarray(0, jnp.int32), window, true0,
               jnp.asarray(0, jnp.int32), start, true0, jnp.asarray(threshold, dtype),
               jnp.asarray(stopping.stall_restarts, jnp.int32), ended0.astype(jnp.int32))
    x, _, _, _, _, count, _, _, _, kept, _, _, _, ended = lax.while_loop(going, iterate, initial)
    # A solve that converged (or was accepted) returns its last iterate; one stopped short, the better of its last and
    # its best checked one: past the rounding floor the iterate wanders (a planted stall's grew to a true residual of
    # 7e-2 in float64 at D/h 16), and a cap short of the first check has only its last. One whose residual went
    # non-finite returns NaN: its start (zero, for a NaN right-hand side) would be a velocity the step never solved for.
    last = _add(b, operator(x), -1.0)
    held = _add(b, operator(kept), -1.0)
    last_norm, held_norm = inner(last, last), inner(held, held)
    better = (ended != _CONVERGED) & (ended != _AT_ZERO) & (~jnp.isfinite(last_norm) | (held_norm < last_norm))
    x = _select(better, kept, x)
    nans = (jnp.full_like(x[0], jnp.nan), jnp.full_like(x[1], jnp.nan), jnp.full_like(x[2], jnp.nan))
    x = _select(ended == _NOT_FINITE, nans, x)
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


def transposed(stopping: Stopping) -> Stopping:
    """The stopping of the transpose solves, which the derivative in reverse mode takes: the primal's, the very object
    (so one program serves both, the operator symmetric). A seam: a test plants a cap on the transpose solves alone
    here, through the solve the node runs, and the solve is then traced with its transpose given apart."""
    return stopping


def projected_solve(grid: MACGrid, apply: Operator, diagonal: Velocity, rhs: Velocity, guess: Velocity | None,
                    stopping: Stopping, tap: Array, preconditioner: Preconditioner,
                    magnitude: Operator | None = None) -> tuple[Velocity, Array]:
    """``P A P u = P rhs`` in the divergence-free subspace (the null-space form of the saddle-point problem, the
    projector the grid kind's exact one), through ``custom_linear_solve`` with the report block on its right-hand
    side; ``tap`` is the block's input (zeros, :data:`REPORT_SIZE`), whose cotangent is the transpose solve's report.
    ``magnitude`` is ``apply`` with its coefficients in magnitude, for a stop at the rounding floor. Returns the
    divergence-free solution and the primal solve's report."""
    with solving(CG_SITE):
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

        def solver(stops: Stopping) -> Callable[[Callable[[tuple[Velocity, Array]], tuple[Velocity, Array]],
                                                 tuple[Velocity, Array]], tuple[Velocity, Array]]:
            def solve(operator: Callable[[tuple[Velocity, Array]], tuple[Velocity, Array]],
                      pair: tuple[Velocity, Array]) -> tuple[Velocity, Array]:
                b, block = pair

                def field_operator(x: Velocity) -> Velocity:
                    return operator((x, block))[0]

                b = project(b)
                start = scaling_start(projected_guess, b, field_operator, inner)
                x, report = guarded_cg(field_operator, b, precondition, start, stops, inner, magnitude)
                return project(x), report

            return solve

        coupled_rhs = _add(rhs, (tap[0], tap[0], tap[0]), 0.0)
        coupled_tap = tap + 0.0 * rhs[0][0, 0, 0]
        other = transposed(stopping)
        same = other is stopping
        solved, report = lax.custom_linear_solve(matvec, (coupled_rhs, coupled_tap), solver(stopping),
                                                 transpose_solve=None if same else solver(other), symmetric=same)
        return solved, report


__all__ = ["AT_FLOOR_BY_DEFAULT", "DEFAULT_CAP", "DEFAULT_PRECONDITIONER", "DEFAULT_TOLERANCE", "RECHECK_CAP_FACTOR", "FLOOR_MEASURED",
           "FLOOR_STOP_MEASURED", "FLOOR_STOP_TOLERANCES", "STALL_RESTARTS", "STALL_RESTART_MEASURED",
           "STAGNATION_FACTOR",
           "STAGNATION_LEAST_WINDOW", "STAGNATION_MEASURED", "STAGNATION_SPAN", "STAGNATION_WINDOW",
           "stagnation_window", "PRECONDITIONERS", "REPORT_FIELDS", "REPORT_SIZE", "Jacobi", "Preconditioner",
           "Stopping", "default_tolerance", "guarded_cg", "preconditioner_named",
           "projected_solve", "recheck_stopping", "restarted", "rounding_floor", "scaling_start", "transposed"]
