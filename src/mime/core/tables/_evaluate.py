"""Reading a table at a batch of queries, in JAX: the value, and the flags a consumer stops the run on."""

from __future__ import annotations

import functools
import itertools
import math
from typing import Any, NamedTuple, cast

import jax
import numpy as np

from mime.core import switch
from mime.core._node import jnp, run_dtype
from mime.core.tables._compile import AxisArrays, TableArrays, held_in
from mime.core.tables._table import Table

#: JAX's loop, recomputation and gather, whose hints name types from jaxlib, which ships no stubs.
_jax: Any = jax

#: How a query is located among an axis's breaks where the axis has no bucket index (``jnp.searchsorted``'s method).
SEARCH_METHOD = "scan"


class Evaluation(NamedTuple):
    """A read at a batch of queries, each array of the batch's shape (``value`` followed by the value's):

    - ``value``: the table's value; no number (NaN) where the read is refused, and where it lies outside the readable
      range with no extrapolation declared;
    - ``in_range``: whether the query lies in the readable range on every axis (OpenMaterica's ``in_range``);
    - ``refused``: whether it lies at a declared edge, or across one from the readable range, on some axis, which
      always stops the run;
    - ``accepted``: whether the run may go on: not refused, a query that is a number, and in range or under the
      declared extrapolation.

    Where the readable range is one point, a read at that point is the value there with a derivative that is NaN (a
    point has no inside to differentiate along): along that axis in reverse mode; in every direction in forward mode
    once that axis's query carries a tangent (traced by ``jvp`` or ``jacfwd``, even given a zero tangent), which no
    linear tangent rule avoids. A query on that axis that carries no tangent in the mode used (a constant, or a
    ``jit`` argument the derivative is not taken in) leaves the other axes' derivatives finite in both modes.

    A consumer stops the run wherever ``accepted`` is false; the NaN is only a second line."""

    value: Any
    in_range: Any
    refused: Any
    accepted: Any


class TableRefused(ValueError):
    """A read that stops the run, found on the host."""


class _Located(NamedTuple):
    """A query's place on one axis, at the query's own shape: its interval ``index``, the first knot (or knot cell)
    its read gathers from, and the weight of each knot of the window (or each power of ``t``)."""

    index: Any
    first: Any
    weights: Any


def evaluate(table: Table | TableArrays, *at: Any, owner: str) -> Evaluation:
    """``table`` read at the queries ``at``, one argument per axis in the table's order, broadcast together, in their
    floating dtype (the run's, where they are not floating). Locating each query among its axis's breaks is the switch
    ``owner``, which the entry that owns the table declares (``Table.value_order`` gives its order); nothing else a
    read evaluates is non-smooth.

    Each axis's position is computed from its own argument, never from a stacked array: a tangent rule is linear and
    ``0 * NaN`` is NaN, so one array holding every axis would carry the NaN of a one-point axis into the others'
    forward derivatives. Each axis is located at its own argument's shape, before the arguments are broadcast: a
    coordinate many queries share (a point's radius, across all its pairs, given as a column or a row) is located
    once."""
    count = len(table.axes if isinstance(table, Table) else table.axes)
    if len(at) != count:
        raise ValueError(f"a query is a point of the table's {count} axes, one argument each; this read passed "
                         f"{len(at)}")
    queries = [jnp.asarray(q) for q in at]
    queries = [q if jnp.issubdtype(q.dtype, jnp.floating) else q.astype(run_dtype()) for q in queries]
    dtype = jnp.result_type(*queries)
    arrays = table.arrays(dtype) if isinstance(table, Table) else table
    held_in(arrays.dtype)
    return _read(arrays, [q.astype(arrays.dtype) for q in queries], owner=owner)


def _read(arrays: TableArrays, columns: list[Any], *, owner: str) -> Evaluation:
    """The read at one query array per axis, broadcast together: each axis located at its own array's shape, so that
    a coordinate shared by many queries (a point's, across all its pairs) is located once."""
    batch = jnp.broadcast_shapes(*(q.shape for q in columns))
    count = len(arrays.axes)
    degrees = arrays.values.shape[count:2 * count] if arrays.stored_polynomials else (None,) * count
    located = [_located(axis, q, arrays, degree, owner=owner)
               for axis, q, degree in zip(arrays.axes, columns, degrees, strict=True)]
    value = _restored(_interpolated(arrays, located), arrays)
    in_range = functools.reduce(jnp.logical_and, [jnp.asarray(axis.inside)[found.index]
                                                  for axis, found in zip(arrays.axes, located, strict=True)])
    stopped = functools.reduce(jnp.logical_or, [jnp.asarray(axis.refused)[found.index]
                                                for axis, found in zip(arrays.axes, located, strict=True)])
    finite = functools.reduce(jnp.logical_and, [jnp.isfinite(q) for q in columns])
    accepted = jnp.logical_not(stopped) & (in_range | arrays.extrapolated) & finite
    return Evaluation(value.reshape(*batch, *arrays.value_shape), jnp.broadcast_to(in_range, batch),
                      jnp.broadcast_to(stopped, batch), jnp.broadcast_to(accepted, batch))


def _located(axis: AxisArrays, q: Any, arrays: TableArrays, degree: int | None, *, owner: str) -> _Located:
    """A query's interval on one axis, and the first knot (or, of ``degree`` powers of ``t``, the knot cell) and the
    weights its read takes there."""
    breaks = jnp.asarray(axis.breaks)
    index = switch.searchsorted(breaks, q, side=arrays.side, method=SEARCH_METHOD, buckets=axis.buckets, owner=owner,
                                resolution=lambda: switch.rounding(axis.scale),
                                indicator=lambda: _offset_from_nearest(breaks, q))
    position = jnp.log(switch.domain_guard(q > 0, q, 1.0)) if arrays.scheme == "loglog" else q
    position = switch.without_slope(position, jnp.asarray(axis.slopeless)[index])
    t = (position - jnp.asarray(axis.origin)[index]) / jnp.asarray(axis.width)[index]
    if arrays.stored_polynomials:
        t = t + jnp.asarray(axis.offset)[index]
        powers = [jnp.ones_like(t), t]
        while len(powers) < cast(int, degree):
            powers.append(powers[-1] * t)
        return _Located(index, jnp.asarray(axis.cell)[index], jnp.stack(powers, axis=-1))
    coefficients = jnp.asarray(axis.coefficients)[index]
    weights = coefficients[..., -1, :]
    for power in range(coefficients.shape[-2] - 2, -1, -1):
        weights = coefficients[..., power, :] + t[..., None] * weights
    return _Located(index, jnp.asarray(axis.start)[index], weights)


def _offset_from_nearest(breaks: Any, q: Any) -> Any:
    """Each query's offset from its nearest break: the location's indicator, read only inside a collector."""
    offsets = q[..., None] - breaks
    return jnp.take_along_axis(offsets, jnp.argmin(jnp.abs(offsets), axis=-1)[..., None], axis=-1)[..., 0]


def _interpolated(arrays: TableArrays, located: list[_Located]) -> Any:
    """The interpolated form at the queries. For each combination of the leading axes' window knots (or powers of
    ``t``), in a loop, one gather of the last axis's window, its knots' values contiguous, summed with the last axis's
    weights, then with the product of the leading axes' weights: a gather feeding products and sums that XLA fuses,
    so that no window outlives its iteration, and a program the size of one combination's, whatever the table's."""
    values = jnp.asarray(arrays.values)
    count = len(arrays.axes)
    sizes = tuple(int(found.weights.shape[-1]) for found in located)
    if arrays.stored_polynomials:
        cells, powers = values.shape[:count], values.shape[count:2 * count]
        slabs = values.reshape(-1, powers[-1], values.shape[-1])
        firsts = [found.first * int(np.prod(cells[a + 1:], dtype=np.int64)) * math.prod(powers[:-1])
                  for a, found in enumerate(located)]
        last = jnp.zeros((), dtype=located[-1].first.dtype)
        strides = [math.prod(powers[a + 1:-1]) for a in range(count - 1)]
    else:
        knots = values.shape[:count]
        slabs = values.reshape(-1, knots[-1], values.shape[-1])
        strides = [int(np.prod(knots[a + 1:-1], dtype=np.int64)) for a in range(count - 1)]
        firsts = [found.first * stride for found, stride in zip(located[:-1], strides, strict=True)]
        last = located[-1].first
    first = sum(firsts, start=jnp.zeros((), dtype=located[-1].first.dtype))
    corners = list(itertools.product(*(range(size) for size in sizes[:-1])))
    offsets = np.asarray([sum(k * stride for k, stride in zip(corner, strides, strict=True)) for corner in corners],
                         dtype=np.int32)
    knots_at = np.asarray(corners, dtype=np.int32).reshape(len(corners), count - 1)
    batch = jnp.broadcast_shapes(*(found.weights.shape[:-1] for found in located))

    def summed(slabs: Any, first: Any, last: Any, weights: list[Any]) -> Any:
        def along(i: Any) -> Any:
            """The last axis's window at leading combination ``i``, summed with its weights. A stencil's weights sum
            to one, so the sum is its first knot's value plus each weight times a difference from it: a derivative
            is then a sum of differences, not a difference of sums the size of the values."""
            slab = _slab(slabs, first + jnp.asarray(offsets)[i], last, sizes[-1]).astype(weights[-1].dtype)
            if arrays.stored_polynomials:
                return functools.reduce(jnp.add, [weights[-1][..., k, None] * slab[..., k, :]
                                                  for k in range(sizes[-1])])
            anchor = slab[..., 0, :]
            return anchor + functools.reduce(jnp.add, [weights[-1][..., k, None] * (slab[..., k, :] - anchor)
                                                       for k in range(1, sizes[-1])])

        def lead(i: Any) -> Any:
            return functools.reduce(jnp.multiply, [jnp.take(w, jnp.asarray(knots_at)[i, a], axis=-1)
                                                   for a, w in enumerate(weights[:-1])])[..., None]

        if count == 1:
            return along(0)
        start = jnp.zeros((*batch, values.shape[-1]), dtype=weights[-1].dtype)
        if arrays.stored_polynomials:
            def added(i: Any, total: Any) -> Any:
                return total + lead(i) * along(i)

            return _jax.lax.fori_loop(0, len(corners), recomputed(added), start)
        anchor = along(0)  # the leading weights sum to one as well

        def added_difference(i: Any, total: Any) -> Any:
            return total + lead(i) * (along(i) - anchor)

        return anchor + _jax.lax.fori_loop(1, len(corners), recomputed(added_difference), start)

    def recomputed(function: Any) -> Any:
        """``function`` recomputed in reverse mode where the table says so: its inputs kept, not its windows."""
        return _jax.checkpoint(function) if arrays.recomputed_in_reverse else function

    return recomputed(summed)(slabs, first, last, [found.weights for found in located])


def _slab(slabs: Any, row: Any, start: Any, size: int) -> Any:
    """``size`` consecutive entries of ``slabs``'s middle dimension from ``start``, in its ``row``: a window of the last
    axis, every one in bounds by construction."""
    row, start = jnp.broadcast_arrays(row, start)
    dimensions = _jax.lax.GatherDimensionNumbers(offset_dims=(row.ndim, row.ndim + 1), collapsed_slice_dims=(0,),
                                                 start_index_map=(0, 1))
    return _jax.lax.gather(slabs, jnp.stack([row, start], axis=-1), dimensions, (1, size, slabs.shape[-1]),
                           mode=_jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS)


def _restored(held: Any, arrays: TableArrays) -> Any:
    """The interpolated form back to a value: the exponential of a ``loglog`` table's, ``L L^T`` from a ``cholesky``
    table's factor."""
    if arrays.scheme == "loglog":
        return jnp.exp(held)
    if arrays.structure == "cholesky":
        size = arrays.value_shape[0]
        rows, columns = np.tril_indices(size, -1)
        factor = jnp.zeros((*held.shape[:-1], size, size), dtype=held.dtype)
        factor = factor.at[..., np.arange(size), np.arange(size)].set(jnp.exp(held[..., :size]))
        factor = factor.at[..., rows, columns].set(held[..., size:])
        return jnp.sum(factor[..., :, None, :] * factor[..., None, :, :], axis=-1)
    return held


def refuse(evaluation: Evaluation, what: str = "the table") -> None:
    """Raise :class:`TableRefused` where an evaluation's values (not tracers: a step's published flags, read on the
    host) stop the run: a read at or across a declared edge, or outside the readable range with no extrapolation
    declared."""
    refused = np.asarray(evaluation.refused)
    outside = ~np.asarray(evaluation.accepted) & ~refused
    if refused.any() or outside.any():
        raise TableRefused(f"{what}: {int(refused.sum())} reads at or across a declared edge and {int(outside.sum())} "
                           f"outside the readable range with no extrapolation declared, of {refused.size}; the run "
                           f"stops here, and nothing computed from them is kept")


__all__ = ["SEARCH_METHOD", "Evaluation", "TableRefused", "evaluate", "refuse"]
