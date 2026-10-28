"""Reading a table at a batch of queries, in JAX: the value, and the flags a consumer stops the run on."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from mime.core import switch
from mime.core._node import jnp, run_dtype
from mime.core.tables._compile import AxisArrays, TableArrays, held_in
from mime.core.tables._table import Table

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
    point has no inside to differentiate along): along that axis if its query is traced, in reverse mode; in every
    direction in forward mode, which no linear tangent rule avoids. A query on that axis that is no tracer leaves the
    other axes' derivatives finite in both modes.

    A consumer stops the run wherever ``accepted`` is false; the NaN is only a second line."""

    value: Any
    in_range: Any
    refused: Any
    accepted: Any


class TableRefused(ValueError):
    """A read that stops the run, found on the host."""


class _Located(NamedTuple):
    index: Any
    stencil: Any
    coefficients: Any
    t: Any


def evaluate(table: Table | TableArrays, *at: Any, owner: str) -> Evaluation:
    """``table`` read at the queries ``at``, one argument per axis in the table's order, broadcast together, in their
    floating dtype (the run's, where they are not floating). Locating each query among its axis's breaks is the switch
    ``owner``, which the entry that owns the table declares (``Table.value_order`` gives its order); nothing else a
    read evaluates is non-smooth.

    Each axis's position is computed from its own argument, never from a stacked array: a tangent rule is linear and
    ``0 * NaN`` is NaN, so one array holding every axis would carry the NaN of a one-point axis into the others'
    forward derivatives."""
    count = len(table.axes if isinstance(table, Table) else table.axes)
    if len(at) != count:
        raise ValueError(f"a query is a point of the table's {count} axes, one argument each; this read passed "
                         f"{len(at)}")
    queries = [jnp.asarray(q) for q in at]
    queries = [q if jnp.issubdtype(q.dtype, jnp.floating) else q.astype(run_dtype()) for q in queries]
    dtype = jnp.result_type(*queries)
    arrays = table.arrays(dtype) if isinstance(table, Table) else table
    held_in(arrays.dtype)
    batch = jnp.broadcast_shapes(*(q.shape for q in queries))
    columns = [jnp.broadcast_to(q.astype(arrays.dtype), batch).reshape(-1) for q in queries]
    located = [_located(axis, column, logarithmic=arrays.scheme == "loglog", side=arrays.side, owner=owner)
               for axis, column in zip(arrays.axes, columns, strict=True)]
    values = jnp.asarray(arrays.values)
    knots = values.shape[:count]
    strides = [int(np.prod(knots[a + 1:], dtype=np.int64)) for a in range(count)]
    corners = sum((found.stencil[(slice(None), *(slice(None) if b == a else None for b in range(count)))] * stride
                   for a, (found, stride) in enumerate(zip(located, strides, strict=True))),
                  start=jnp.zeros((), dtype=located[0].stencil.dtype))
    held = values.reshape(int(np.prod(knots, dtype=np.int64)), -1)[corners]
    for found in reversed(located):
        held = _along_last(held, found.coefficients, found.t)
    inside = jnp.stack([jnp.asarray(axis.inside)[found.index] for axis, found in zip(arrays.axes, located,
                                                                                     strict=True)])
    refused = jnp.stack([jnp.asarray(axis.refused)[found.index] for axis, found in zip(arrays.axes, located,
                                                                                       strict=True)])
    in_range = jnp.all(inside, axis=0)
    stopped = jnp.any(refused, axis=0)
    accepted = jnp.logical_not(stopped) & (in_range | arrays.extrapolated) & jnp.all(jnp.stack([jnp.isfinite(column) for column in columns]), axis=0)
    value = _restored(held, arrays)
    return Evaluation(value.reshape(*batch, *arrays.value_shape), in_range.reshape(batch), stopped.reshape(batch),
                      accepted.reshape(batch))


def _located(axis: AxisArrays, q: Any, *, logarithmic: bool, side: str, owner: str) -> _Located:
    """A query's interval on one axis, and the stencil, coefficients and ``t`` it reads there."""
    breaks = jnp.asarray(axis.breaks)
    index = switch.searchsorted(breaks, q, side=side, method=SEARCH_METHOD, buckets=axis.buckets, owner=owner,
                                resolution=lambda: switch.rounding(axis.scale),
                                indicator=lambda: _offset_from_nearest(breaks, q))
    position = jnp.log(switch.domain_guard(q > 0, q, 1.0)) if logarithmic else q
    position = switch.without_slope(position, jnp.asarray(axis.slopeless)[index])
    t = (position - jnp.asarray(axis.origin)[index]) / jnp.asarray(axis.width)[index]
    return _Located(index, jnp.asarray(axis.stencil)[index], jnp.asarray(axis.coefficients)[index], t)


def _offset_from_nearest(breaks: Any, q: Any) -> Any:
    """Each query's offset from its nearest break: the location's indicator, read only inside a collector."""
    offsets = q[:, None] - breaks[None, :]
    return jnp.take_along_axis(offsets, jnp.argmin(jnp.abs(offsets), axis=-1)[:, None], axis=-1)[:, 0]


def _along_last(held: Any, coefficients: Any, t: Any) -> Any:
    """The interpolation along the last stencil dimension of ``held`` (batch, stencil..., values): the polynomial's
    coefficients from that stencil's values, then the polynomial at ``t`` by Horner's rule. Products and sums only, so
    no matrix unit's reduced precision reaches it."""

    def across(column: Any) -> Any:
        """A batch column, broadcast over ``held``'s other dimensions."""
        return column.reshape(column.shape[0], *([1] * (held.ndim - 2)))

    powers: list[Any] = []
    for power in range(coefficients.shape[1]):
        term = across(coefficients[:, power, 0]) * held[..., 0, :]
        for knot in range(1, coefficients.shape[2]):
            term = term + across(coefficients[:, power, knot]) * held[..., knot, :]
        powers.append(term)
    result = powers[-1]
    for power in range(len(powers) - 2, -1, -1):
        result = powers[power] + across(t) * result
    return result


def _restored(held: Any, arrays: TableArrays) -> Any:
    """The interpolated form back to a value: the exponential of a ``loglog`` table's, ``L L^T`` from a ``cholesky``
    table's factor."""
    if arrays.scheme == "loglog":
        return jnp.exp(held)
    if arrays.structure == "cholesky":
        size = arrays.value_shape[0]
        rows, columns = np.tril_indices(size, -1)
        factor = jnp.zeros((held.shape[0], size, size), dtype=held.dtype)
        factor = factor.at[:, np.arange(size), np.arange(size)].set(jnp.exp(held[:, :size]))
        factor = factor.at[:, rows, columns].set(held[:, size:])
        return jnp.sum(factor[:, :, None, :] * factor[:, None, :, :], axis=-1)
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
