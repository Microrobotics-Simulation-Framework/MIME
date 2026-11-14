"""The exact allocation: commands within explicit bounds that come nearest wanted quantities through an affine model,
exact by enumerating the active sets, in JAX, with the derivative of the selected set's solve.

**The problem.** ``min sum_q ((A u + c - y)_q / s_q)^2 + sum_k ((u - r)_k / t_k)^2`` over ``lower <= u <= upper``:
``A`` the model's rows per unit command, ``c`` its offset at zero command, ``y`` the wanted values, ``s`` the caller's
scales (each row's unit), ``r`` a reference command and ``t`` its scales (the regularisation, absent unless ``t`` is
given). The rows are stacked as ``M u ~ b``. Where several commands minimise it (more commands than the rows fix, as
eight coils asked for a field alone), the one nearest the reference, least ``|u - r|^2``, is returned: least effort for
a reference of zero.

**Exact by active sets.** Each command is free, at its upper bound or at its lower one: ``3^K`` sets. A set's free
commands are those that minimise the residual given its bound ones, and among them the nearest the reference; the
minimiser is the candidate inside the box of least residual, ties going to the least ``|u - r|^2``: the minimiser's free
commands are its own set's candidate, so no feasible candidate does better. A set's factors depend only on which
commands are free, so there are ``2^K`` QR factors and every set's candidate is a matrix product of its subset's. A
subset with no more free commands than rows is solved as a least squares, by the QR of ``[M * free; diag(not free)]``
(a bound column a unit row of its own) with its columns scaled to unit length; one with more, whose free columns cannot
be independent, as the least-norm solution of its rows, by the QR of its free rows' transpose. The selection runs with
no derivative; the selected set is then solved once more, differentiably, by one QR (by its free columns'
pseudo-inverse where there are more commands than rows, which gives both forms), so the derivative is that set's
whatever ``K``.

**Its guards.** Rank: a subset is rejected where a diagonal entry of its triangle falls below ``rcond`` times its
largest; its triangle is replaced by a stand-in before the solve and its sets are kept from the selection after it (the
double ``where``), so no singular solve reaches a value; the selected set passed the test, so its own solve needs no
stand-in. A column of zero
length (a command with no effect, and no regularisation) has its scale held at one, and the command is held at the
reference clipped into its bounds, never driven to a limit by a tie. The returned commands are clipped into the box:
the selected set's solve and the selection's product differ by rounding. More than :data:`EXACT_COMMANDS` commands is
refused by name: ``3^K`` grows past what a step should pay, and the iterative path is the route."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from functools import cache
from typing import Any, NamedTuple

import jax
import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.core import switch
from mime.core.solves import SolveSite, register, solving

jnp: Any = jax.numpy
lax: Any = jax.lax

#: The most commands the exact path takes: ``3^8 = 6561`` active sets from ``2^8 = 256`` QR factors. Every coil array
#: MIME ships has at most 8 coils (Navion_2 3, CardioMag and OctoMag 8), and ``allocate()`` serves them all through this
#: law. A choice against the rule set before the cost was measured (the largest bound at which the allocation costs at
#: most three times its five-command cost, which gives 6): about 0.8 to 1.0 ms a base step at 8 commands against 44 us
#: at 5, on one runner thread in float64.
EXACT_COMMANDS = 8
#: A subset is rank deficient where a diagonal entry of its triangle is below this times its largest.
RCOND = {np.dtype(np.float64): 1e-10, np.dtype(np.float32): 1e-5}
#: Two candidates' residual norms count as tied within this many units in the last place of their scale.
TIE_ULPS = 64

ALLOCATION_SITE = register(SolveSite(
    "allocation_exact", "dense", "mime.control._allocation.exact_allocation", nodes=("ControllerNode",),
    exempt="the exact allocation's QR factors of its active sets' columns or rows and their triangular solves, of order "
           "at most its commands, in the arithmetic of the law that calls it",
    bound=EXACT_COMMANDS))
LEAST_NORM_SITE = register(SolveSite(
    "allocation_least_norm", "lstsq", "mime.control._allocation.exact_allocation", nodes=("ControllerNode",),
    exempt="the selected active set's free columns' pseudo-inverse, where an allocation has more commands than rows, "
           "of order at most its commands, in the arithmetic of the law that calls it",
    bound=EXACT_COMMANDS))


class ExactAllocation(NamedTuple):
    """The commands ``(K,)``; each command's place in the selected active set, ``active`` ``(K,)``, ``-1`` at its
    lower bound, ``0`` free, ``+1`` at its upper bound (a command with no effect reads ``0``); and the minimised
    ``objective``, the sum of the scaled squares."""

    commands: Any
    active: Any
    objective: Any


def allocation_switch(owner: str, orders: Mapping[str, int]) -> switch.Declaration:
    """The switch :func:`exact_allocation` evaluates under ``owner``, for the gradient record of the entry whose law
    calls it (the law takes its owner from its caller, as ``mime.core.multibody``'s helpers do): the selection of the
    active set. ``orders`` per output group of that entry: ``0`` (a kink) where the minimiser is continuous (the problem
    regularised, or its rows of full column rank); ``-1`` where it may not be (the least-effort minimiser of an
    unregularised problem whose rows do not fix the commands can jump between faces)."""
    return switch.Declaration(
        owner=owner,
        indicator="the selected active set's residual norm less the next feasible set's, in the scaled rows "
                  "(dimensionless), and among tied ones the same of |u - reference|^2; a subset's least diagonal entry "
                  "of its triangle over its largest, less rcond; a free command's distance inside its bounds; a "
                  "command's squared column length",
        orders=orders,
        route="none: the derivative is the selected set's solve, which changes where a command reaches a bound or "
              "leaves it; a smooth allocation would be another law")


class _Sets(NamedTuple):
    """The active sets of ``K`` commands against ``R`` rows, fixed when the law is traced: each set's signs ``(3^K, K)``
    (``0`` free, ``+1`` upper, ``-1`` lower) and the same as masks; the ``2^K`` free subsets, each set's subset, and
    which subsets have more free commands than rows (``wide``, solved as least norm) and which no more (``tall``)."""

    signs: NDArray[np.int8]
    free: NDArray[np.bool_]
    up: NDArray[np.bool_]
    down: NDArray[np.bool_]
    subsets: NDArray[np.bool_]
    subset_of: NDArray[np.int32]
    wide: NDArray[np.bool_]
    tall_subsets: NDArray[np.int32]
    wide_subsets: NDArray[np.int32]


@cache
def _sets(commands: int, rows: int) -> _Sets:
    signs = np.asarray(list(itertools.product((0, 1, -1), repeat=commands)), dtype=np.int8)
    subsets = np.asarray(list(itertools.product((True, False), repeat=commands)), dtype=bool)
    # Indices in int32, whatever the run's dtype: one in int64 gathers into a float32 run's arrays as int64.
    index = ((signs != 0) * 2 ** np.arange(commands - 1, -1, -1)).sum(axis=1).astype(np.int32)
    wide = subsets.sum(axis=1) > rows
    return _Sets(signs, signs == 0, signs > 0, signs < 0, subsets, index, wide,
                 np.flatnonzero(~wide).astype(np.int32), np.flatnonzero(wide).astype(np.int32))


def _refused(message: str, action: str, code: str = "allocation.wanted") -> ArgumentRefused:
    return ArgumentRefused([Refusal(code, RefusalClass.USER, message, action)])


def _vector(value: Any, size: int, name: str, dtype: Any) -> Any:
    held = jnp.asarray(value, dtype=dtype)
    if held.ndim == 0:
        return jnp.full((size,), held, dtype=dtype)
    if held.shape != (size,):
        raise _refused(f"exact_allocation: {name} has shape {tuple(held.shape)}, not ({size},)",
                       f"give {name} one value, or {size}")
    return held


def exact_allocation(matrix: Any, wanted: Any, *, lower: Any, upper: Any, scales: Any, owner: str,
                     offset: Any = None, reference: Any = None, reference_scales: Any = None) -> ExactAllocation:
    """The commands within ``[lower, upper]`` that minimise ``sum_q ((matrix @ u + offset - wanted)_q / scales_q)^2 +
    sum_k ((u - reference)_k / reference_scales_k)^2``, exactly (see the module); among several minimisers, the nearest
    the reference. Arrays in SI, traced or not, in the matrix's dtype: ``matrix`` ``(R, K)``, each row per unit
    command; ``wanted``, ``offset`` (zero unless given) and ``scales`` ``(R,)``, each in its row's unit (``scales`` is
    required: rows in different units are weighed only through the caller's scales); ``lower``, ``upper``,
    ``reference`` (zero unless given) and ``reference_scales`` ``(K,)`` or one value, in the command's unit; without
    ``reference_scales`` there is no regularisation. ``lower <= upper`` is the caller's. ``owner`` is the switch's
    owner, which the calling entry declares (:func:`allocation_switch`). More than :data:`EXACT_COMMANDS` commands is
    refused (``allocation.exact-size``)."""
    held = jnp.asarray(matrix)
    dtype = held.dtype if jnp.issubdtype(held.dtype, jnp.floating) else jnp.zeros(()).dtype
    held = held.astype(dtype)
    if held.ndim != 2 or held.shape[1] < 1:
        raise _refused(f"exact_allocation: the matrix has shape {tuple(held.shape)}, not (rows, commands)",
                       "give the model's rows per unit command, one column a command")
    rows, commands = held.shape
    if commands > EXACT_COMMANDS:
        raise _refused(
            f"an exact allocation of {commands} commands: it enumerates 3^{commands} = {3 ** commands} active sets, "
            f"and the exact path takes at most {EXACT_COMMANDS} commands",
            "allocate fewer commands at once, or wait for the iterative path, which takes any number (a later 0.3.0 "
            "development release)", code="allocation.exact-size")
    wanted_, scales_ = (jnp.asarray(v, dtype=dtype) for v in (wanted, scales))
    offset_ = jnp.zeros((rows,), dtype=dtype) if offset is None else jnp.asarray(offset, dtype=dtype)
    for name, value in (("wanted", wanted_), ("scales", scales_), ("offset", offset_)):
        if value.shape != (rows,):
            raise _refused(f"exact_allocation: {name} has shape {tuple(value.shape)}, not ({rows},)",
                           f"give {name} one value per row of the matrix")
    low, high = _vector(lower, commands, "lower", dtype), _vector(upper, commands, "upper", dtype)
    near = jnp.zeros((commands,), dtype=dtype) if reference is None else _vector(reference, commands, "reference", dtype)
    stacked, target = held / scales_[:, None], (wanted_ - offset_) / scales_
    if reference_scales is not None:
        spread = _vector(reference_scales, commands, "reference_scales", dtype)
        stacked = jnp.concatenate([stacked, jnp.diag(1.0 / spread)])
        target = jnp.concatenate([target, near / spread])
    return _solved(stacked, target, low, high, near, owner=owner)


def _solved(stacked: Any, target: Any, low: Any, high: Any, reference: Any, *, owner: str) -> ExactAllocation:
    rows, commands = stacked.shape
    dtype = stacked.dtype
    tiny = float(np.finfo(dtype).tiny)
    squared = jnp.sum(stacked * stacked, axis=0)
    effect = switch.predicate(squared > tiny, owner=owner, resolution=lambda: switch.rounding(tiny),
                              indicator=lambda: squared - tiny)
    scale = jnp.sqrt(switch.domain_guard(effect, squared, 1.0))
    rest = switch.minimum(switch.maximum(reference, low, owner=owner, resolution=lambda: switch.rounding(low),
                                         indicator=lambda: reference - low),
                          high, owner=owner, resolution=lambda: switch.rounding(high), indicator=lambda: high - reference)
    sets = _sets(commands, rows)
    selected = _selection(*lax.stop_gradient((stacked, target, low, high, reference, rest, scale, effect)),
                          owner=owner)
    keep = jnp.asarray(sets.free, dtype=dtype)[selected]
    up, down = jnp.asarray(sets.up, dtype=dtype)[selected], jnp.asarray(sets.down, dtype=dtype)[selected]
    bound = switch.where(effect, up * high + down * low, rest, owner=owner, resolution=lambda: switch.rounding(tiny),
                         indicator=lambda: squared - tiny)
    leaves = _left(target, stacked, bound)
    if sets.wide_subsets.size:
        # More commands than rows: the selected set's free commands are the least |u - reference|^2 among those that
        # minimise its residual, its free columns' pseudo-inverse giving both forms (a least squares where they are
        # independent, a least-norm solution where they outnumber the rows).
        free_columns = stacked * keep[None, :]
        with solving(LEAST_NORM_SITE):
            inverse = jnp.linalg.pinv(free_columns, rtol=RCOND[np.dtype(dtype)])
        solved = keep * (reference + inverse @ (leaves - free_columns @ reference))
    else:
        scaled = stacked / scale
        with solving(ALLOCATION_SITE):
            q, r = jnp.linalg.qr(jnp.concatenate([scaled * keep[None, :], jnp.diag(1.0 - keep)]))
            solved = keep * jax.scipy.linalg.solve_triangular(r, q[:rows].T @ leaves, lower=False) / scale
    candidate = solved + bound
    clipped = switch.maximum(switch.minimum(candidate, high, owner=owner, resolution=lambda: switch.rounding(high),
                                            indicator=lambda: high - candidate),
                             low, owner=owner, resolution=lambda: switch.rounding(low),
                             indicator=lambda: candidate - low)
    commands_ = switch.where(effect, clipped, rest, owner=owner, resolution=lambda: switch.rounding(tiny),
                             indicator=lambda: squared - tiny)
    active = switch.where(effect, jnp.asarray(sets.signs)[selected], jnp.zeros((commands,), dtype=np.int8), owner=owner,
                          resolution=lambda: switch.rounding(tiny), indicator=lambda: squared - tiny)
    miss = stacked @ commands_ - target
    return ExactAllocation(commands_, active, jnp.sum(miss * miss))


def _left(target: Any, stacked: Any, bound: Any) -> Any:
    """What the bound commands leave the free ones to make: the target less the rows at the bound values (each free
    command's entry of ``bound`` is zero)."""
    return target - stacked @ bound


def _selection(stacked: Any, target: Any, low: Any, high: Any, reference: Any, rest: Any, scale: Any, effect: Any, *,
               owner: str) -> Any:
    """The index of the selected active set: each subset's QR and rank test, each set's candidate from its subset's
    factors, then among the feasible candidates the least residual, ties going to the least ``|u - reference|^2``. No
    derivative is taken of it, so the magnitudes its tests compare are written plainly."""
    rows, commands = stacked.shape
    dtype = stacked.dtype
    rcond = RCOND[np.dtype(dtype)]
    sets = _sets(commands, rows)
    eye = jnp.eye(commands, dtype=dtype)
    # Each subset's candidate as ``u = held + free * (a - b @ held)``, from its factors.
    offsets = jnp.zeros((len(sets.subsets), commands), dtype=dtype)
    products = jnp.zeros((len(sets.subsets), commands, commands), dtype=dtype)
    ok = jnp.zeros((len(sets.subsets),), dtype=bool)
    with solving(ALLOCATION_SITE):
        if sets.tall_subsets.size:
            free = sets.subsets[sets.tall_subsets][:, None, :]
            scaled = stacked / scale
            system = jnp.concatenate([switch.static_select(free, scaled[None], 0.0),
                                      switch.static_select(free, 0.0, eye[None])], axis=1)
            q, r = jnp.linalg.qr(system)
            diagonal = jnp.abs(jnp.diagonal(r, axis1=-2, axis2=-1))
            fit = switch.predicate(jnp.all(diagonal > rcond * jnp.max(diagonal, axis=-1, keepdims=True), axis=-1),
                                   owner=owner, resolution=lambda: rcond * eps_of(dtype),
                                   indicator=lambda: jnp.min(diagonal, axis=-1) / jnp.max(diagonal, axis=-1) - rcond)
            factors = jax.scipy.linalg.solve_triangular(
                switch.domain_guard(fit[:, None, None], r, eye[None]),
                jnp.einsum("srk,rj->skj", q[:, :rows], jnp.concatenate([target[:, None], scaled], axis=1)), lower=False)
            offsets = offsets.at[sets.tall_subsets].set(factors[:, :, 0] / scale)
            products = products.at[sets.tall_subsets].set(factors[:, :, 1:] * scale[None, None, :] / scale[None, :, None])
            ok = ok.at[sets.tall_subsets].set(fit)
        if sets.wide_subsets.size:
            free = sets.subsets[sets.wide_subsets]
            free_rows = jnp.swapaxes(switch.static_select(free[:, None, :], stacked[None], 0.0), -1, -2)
            q, r = jnp.linalg.qr(free_rows)
            diagonal = jnp.abs(jnp.diagonal(r, axis1=-2, axis2=-1))
            fit = switch.predicate(jnp.all(diagonal > rcond * jnp.max(diagonal, axis=-1, keepdims=True), axis=-1),
                                   owner=owner, resolution=lambda: rcond * eps_of(dtype),
                                   indicator=lambda: jnp.min(diagonal, axis=-1) / jnp.max(diagonal, axis=-1) - rcond)
            guarded = switch.domain_guard(fit[:, None, None], r, jnp.eye(rows, dtype=dtype)[None])
            masked = switch.static_select(free[:, None, :], stacked[None], 0.0)
            lifted = jax.scipy.linalg.solve_triangular(
                guarded, jnp.concatenate([target[None, :, None] - masked @ reference[None, :, None],
                                          jnp.broadcast_to(stacked, (len(free), rows, commands))], axis=2),
                lower=False, trans="T")
            spread = q @ lifted
            offsets = offsets.at[sets.wide_subsets].set(switch.static_select(free, reference[None], 0.0)
                                                        + spread[:, :, 0])
            products = products.at[sets.wide_subsets].set(spread[:, :, 1:])
            ok = ok.at[sets.wide_subsets].set(fit)
    held = switch.static_select(sets.up, high[None], switch.static_select(sets.down, low[None], 0.0))
    held = switch.where(effect[None], held, rest[None], owner=owner, resolution=lambda: switch.rounding(1.0),
                        indicator=lambda: effect.astype(dtype))
    candidate = switch.static_select(
        sets.free, offsets[sets.subset_of] - jnp.einsum("skj,sj->sk", products[sets.subset_of], held), held)
    tolerance = switch.rounding(low) + switch.rounding(high)
    inside = switch.predicate(
        jnp.all(switch.static_select(sets.free, jnp.logical_and(candidate >= low - tolerance,
                                                                candidate <= high + tolerance), True), axis=-1),
        owner=owner, resolution=lambda: jnp.max(tolerance),
        indicator=lambda: jnp.min(jnp.minimum(candidate - low, high - candidate), axis=-1))
    admitted = jnp.logical_and(ok[sets.subset_of], inside)
    miss = candidate @ stacked.T - target[None, :]
    residual = jnp.sqrt(jnp.sum(miss * miss, axis=-1))
    made = jnp.sqrt(jnp.sum((candidate @ stacked.T) ** 2, axis=-1))
    width = TIE_ULPS * eps_of(dtype) * (jnp.sqrt(jnp.sum(target * target)) + made)
    ranked = switch.where(admitted, residual, jnp.inf, owner=owner, resolution=lambda: width, indicator=lambda: residual)
    best = switch.argmin(ranked, owner=owner, resolution=lambda: width[jnp.argmin(ranked)],
                         indicator=lambda: jnp.diff(jnp.sort(ranked)[:2])[0])
    tied = switch.predicate(jnp.logical_and(admitted, residual <= residual[best] + width[best] + width), owner=owner,
                            resolution=lambda: width, indicator=lambda: residual[best] + width[best] + width - residual)
    effort = jnp.sum((candidate - reference[None, :]) ** 2, axis=-1)
    ranked = switch.where(tied, effort, jnp.inf, owner=owner, resolution=lambda: width, indicator=lambda: effort)
    return switch.argmin(ranked, owner=owner, resolution=lambda: switch.rounding(jnp.min(ranked)),
                         indicator=lambda: jnp.diff(jnp.sort(ranked)[:2])[0])


def eps_of(dtype: Any) -> float:
    """The dtype's unit in the last place at one."""
    return float(np.finfo(dtype).eps)


__all__ = ["ALLOCATION_SITE", "EXACT_COMMANDS", "LEAST_NORM_SITE", "RCOND", "ExactAllocation", "allocation_switch", "exact_allocation"]
