"""A table compiled for reading in one dtype, on the host, once.

Each axis's knots, readable range and edges, as the dtype holds them, become the **breaks** a query is located among,
and every interval between two breaks holds one rule for what a read there returns:

- **inside** the readable range, a polynomial in the query's position in its knot cell, ``t``, from 0 at the cell's
  lower knot to 1 at its upper (in logarithms for ``loglog``), given by its coefficients on a few knots' values, the
  cell's **stencil**;
- **outside** it, under a declared extrapolation, the constant the inside polynomial takes at the range's nearest end;
  with none, no number (NaN), since the run stops there;
- **refused**, at a declared edge or across one from the readable range: no number.

A readable range of one point has one inside interval, holding that point alone: there a read is the value, with no
derivative (NaN), as the interval's ``slopeless`` says.

A break is placed so that each interval holds exactly what it should in the dtype: the readable range's ends inside
it, an interior knot in the interval on the declared side, and an edge alone in an interval of its own, between it
and the next number of the dtype. So the one non-smooth operation a read evaluates is the location of the query among
the breaks, and everything else is selected by that index."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.core._node import jnp
from mime.core.tables._table import Axis, Table, TableInvalid

#: The cubic Hermite basis on ``t`` in ``[0, 1]``, its coefficients from ``t**0`` to ``t**3``: the lower knot's value,
#: the lower slope times the cell's width, the upper value and the upper slope times the width.
_HERMITE = np.array([[1.0, 0.0, -3.0, 2.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 3.0, -2.0], [0.0, 0.0, -1.0, 1.0]])
#: The linear basis: ``f0 + t (f1 - f0)``.
_LINEAR = np.array([[1.0, 0.0], [-1.0, 1.0]])


@dataclass(frozen=True, eq=False)
class AxisArrays:
    """One axis, compiled: the ``breaks``; for each of the ``len(breaks) + 1`` intervals they bound, the ``origin``
    and ``width`` that give ``t``, the ``stencil`` of knot indices and the ``coefficients`` (power of ``t`` by stencil
    knot), whether it is ``inside`` the readable range, and whether it is ``refused``. ``scale`` is the axis's largest
    knot in magnitude, the scale of its location's resolution. ``slopeless`` marks the intervals whose reads have no
    derivative: the one inside interval of a readable range of one point."""

    scale: Any
    breaks: Any
    origin: Any
    width: Any
    stencil: Any
    coefficients: Any
    inside: Any
    refused: Any
    slopeless: Any


@dataclass(frozen=True, eq=False)
class TableArrays:
    """A table compiled for reading in one dtype: its ``axes``, and its ``values`` as they are interpolated (the
    logarithms of a ``loglog`` table's, the Cholesky factor of a ``cholesky`` table's, its diagonal's logarithms first,
    then the entries below it by row), with what a read needs to know of the table, fixed at compilation. A pytree whose
    leaves are the arrays, on the device, so that a node can hold it in its ``params`` and a compiled step takes it
    without copying it."""

    axes: tuple[AxisArrays, ...]
    values: Any
    scheme: str
    structure: str
    side: str
    value_shape: tuple[int, ...]
    extrapolated: bool

    @property
    def dtype(self) -> Any:
        return self.values.dtype


_tree_util: Any = cast(Any, jax).tree_util
_tree_util.register_dataclass(AxisArrays, data_fields=["scale", "breaks", "origin", "width", "stencil", "coefficients",
                                                       "inside", "refused", "slopeless"], meta_fields=[])
_tree_util.register_dataclass(TableArrays, data_fields=["axes", "values"],
                              meta_fields=["scheme", "structure", "side", "value_shape", "extrapolated"])


def compiled(table: Table, dtype: np.dtype[Any]) -> TableArrays:
    """``table`` compiled for reading in ``dtype``."""
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"a table is read in a floating dtype, and {dtype} is not one")
    held_in(dtype)
    reachable(math.prod(len(axis.x) for axis in table.axes))
    axes = tuple(_compiled_axis(axis, table, dtype) for axis in table.axes)
    values = _held(_interpolated_values(table, dtype))
    found = TableArrays(axes, values, table.scheme, table.structure, table.side, table.value_shape,
                        table.extrapolation is not None)
    # On the device, evaluated now even while a step is being traced, so that what is cached holds no tracer.
    with cast(Any, jax).ensure_compile_time_eval():
        return _tree_util.tree_map(jnp.asarray, found)


def reachable(nodes: int) -> None:
    """Refuse a table of ``nodes`` knots, from the count a read's int32 index reaches (kept two below it)."""
    if nodes >= np.iinfo(np.int32).max:
        raise TableInvalid(f"a table of {nodes} knots, more than a read's int32 index reaches")


def held_in(dtype: Any) -> None:
    """Refuse a dtype JAX would not keep: float64 without ``jax_enable_x64``, which it would read as float32."""
    held = cast(Any, jax).dtypes.canonicalize_dtype(dtype)
    if held != np.dtype(dtype):
        raise TypeError(f"a table read in {np.dtype(dtype)} needs jax_enable_x64, without which JAX holds it in "
                        f"{held}")


def _held(held: np.ndarray) -> np.ndarray:
    held = np.ascontiguousarray(held)
    held.setflags(write=False)
    return held


def _interpolated_values(table: Table, dtype: np.dtype[Any]) -> np.ndarray:
    """The table's values as the dtype holds them, in the form that is interpolated."""
    y = np.asarray(table.y, dtype=dtype).astype(np.float64)
    if table.scheme == "loglog":
        return np.log(y).astype(dtype)
    if table.structure == "cholesky":
        size = table.value_shape[0]
        try:
            factor = np.linalg.cholesky(y.reshape(-1, size, size))
        except np.linalg.LinAlgError as failed:
            raise TableInvalid(f"the cholesky table over {[axis.name for axis in table.axes]} is positive definite "
                               f"in float64 but not once its values are rounded to {dtype}: its factor is too "
                               f"ill-conditioned for that dtype") from failed
        rows, columns = np.tril_indices(size, -1)
        held = np.concatenate([np.log(np.diagonal(factor, axis1=-2, axis2=-1)), factor[:, rows, columns]], axis=-1)
        return held.reshape(*y.shape[:len(table.axes)], -1).astype(dtype)
    return y.astype(dtype)


def _compiled_axis(axis: Axis, table: Table, dtype: np.dtype[Any]) -> AxisArrays:
    kind = cast(Any, dtype.type)

    def rounded(value: float) -> float:
        return float(kind(value))

    def up(value: float) -> float:
        return _neighbour(kind, value, math.inf)

    def down(value: float) -> float:
        return _neighbour(kind, value, -math.inf)

    x = [rounded(v) for v in axis.x]
    for value in (*x, *(rounded(e) for e in axis.edges), *(rounded(v) for v in axis.readable_range)):
        if 0.0 < abs(value) < np.finfo(kind).tiny:
            raise TableInvalid(f"axis {axis.name!r}: {value!r} is a denormal in {dtype}, which a compiled read "
                               f"flushes to zero")
    for (a, b), (a0, b0) in zip(zip(x, x[1:], strict=False), zip(axis.x, axis.x[1:], strict=False), strict=True):
        if a0 < b0 and not up(a) < b:
            raise TableInvalid(f"axis {axis.name!r}: the knots {a0!r} and {b0!r} lie within one unit in the last "
                               f"place of each other in {dtype}, so the table does not survive it")
    edges = sorted(rounded(e) for e in axis.edges)
    low, high = (rounded(v) for v in axis.readable_range)
    interior = sorted({v for v in x if low < v < high and v not in edges})
    if table.side == "right":
        wanted = [low, *interior, up(high), *(p for edge in edges for p in (edge, up(edge)))]
    else:
        wanted = [down(low), *interior, high, *(p for edge in edges for p in (down(edge), edge))]
    breaks = sorted(set(wanted))
    count = len(breaks) + 1
    logarithmic = table.scheme == "loglog"
    coordinate = [math.log(v) for v in x] if logarithmic else x
    cells = _cells(x, table.scheme)
    size = cells[next(iter(cells))][1].shape[1]
    degree = cells[next(iter(cells))][1].shape[0]

    roles: list[tuple[str, int]] = []
    for i in range(count):
        if table.side == "right":
            representative = breaks[i - 1] if i > 0 else down(breaks[0])
        else:
            representative = breaks[i] if i < len(breaks) else up(breaks[-1])
        roles.append(_role(representative, x, edges, low, high, table.side))
    inside_cells = [cell for role, cell in roles if role == "inside"]
    if not inside_cells:
        raise TableInvalid(f"axis {axis.name!r}: no point of its readable range is off its edges in {dtype}")
    ends = {"below": (low, inside_cells[0]), "above": (high, inside_cells[-1])}

    origin = np.zeros(count)
    width = np.full(count, math.inf)
    stencil = np.zeros((count, size), dtype=np.int32)
    coefficients = np.full((count, degree, size), math.nan)
    slopeless = np.zeros(count, dtype=bool)
    for i, (role, cell) in enumerate(roles):
        if role == "inside":
            stencil[i], coefficients[i] = cells[cell]
            slopeless[i] = low == high
            origin[i] = coordinate[cell]
            width[i] = coordinate[cell + 1] - coordinate[cell]
        elif role in ends:
            end, end_cell = ends[role]
            stencil[i] = cells[end_cell][0]
            if table.extrapolation is not None:
                at = math.log(end) if logarithmic else end
                t = (at - coordinate[end_cell]) / (coordinate[end_cell + 1] - coordinate[end_cell])
                coefficients[i] = 0.0
                coefficients[i, 0] = np.power(t, np.arange(degree, dtype=np.float64)) @ cells[end_cell][1]
        else:
            stencil[i] = cells[inside_cells[0]][0]
    return AxisArrays(
        scale=_held(np.asarray(max(abs(v) for v in x), dtype=dtype)),
        breaks=_held(np.asarray(breaks, dtype=dtype)),
        origin=_held(origin.astype(dtype)),
        width=_held(width.astype(dtype)),
        stencil=_held(stencil),
        coefficients=_held(coefficients.astype(dtype)),
        inside=_held(np.array([role == "inside" for role, _ in roles])),
        refused=_held(np.array([role == "refused" for role, _ in roles])),
        slopeless=_held(slopeless))


def _neighbour(kind: Any, value: float, toward: float) -> float:
    """The next number of the dtype from ``value`` toward ``toward``, with its denormals left out: a compiled read
    flushes a denormal to zero, so beside zero the next number is the smallest normal one."""
    stepped = float(np.nextafter(kind(value), kind(toward)))
    tiny = float(np.finfo(kind).tiny)
    if 0.0 < abs(stepped) < tiny:
        return math.copysign(tiny, stepped) if value == 0.0 else 0.0
    return stepped


def _role(at: float, x: list[float], edges: list[float], low: float, high: float, side: str) -> tuple[str, int]:
    """What an interval holding ``at`` returns: ``refused`` at an edge or across one from the readable range,
    ``below`` or ``above`` it, or ``inside``, with the knot cell whose polynomial it reads."""
    if at in edges or any(at < edge <= low for edge in edges) or any(high <= edge < at for edge in edges):
        return "refused", -1
    if at < low:
        return "below", -1
    if at > high:
        return "above", -1
    for k in range(len(x) - 1):
        if x[k] < x[k + 1] and (x[k] <= at < x[k + 1] if side == "right" else x[k] < at <= x[k + 1]):
            return "inside", k
    last = max(k for k in range(len(x) - 1) if x[k] < x[k + 1])
    first = min(k for k in range(len(x) - 1) if x[k] < x[k + 1])
    if side == "right" and at == x[-1]:  # a readable range of one point, at the last knot: the last cell, closed
        return "inside", last
    if side == "left" and at == x[0]:
        return "inside", first
    raise AssertionError(f"no knot cell holds {at!r}")  # unreachable: the readable range lies within [x[0], x[-1]]


def _cells(x: list[float], scheme: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Each knot cell's stencil and coefficients, by its lower knot's index: every cell of positive width."""
    found: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    coordinate = [math.log(v) for v in x] if scheme == "loglog" else x
    for k in range(len(x) - 1):
        if x[k] < x[k + 1]:
            found[k] = (_cubic_cell(coordinate, k) if scheme == "cubic"
                        else (np.array([k, k + 1], dtype=np.int32), _LINEAR.copy()))
    return found


def _piece(x: list[float], k: int) -> tuple[int, int]:
    """The first and last index of the run of strictly increasing knots holding the cell ``k``: an edge ends one."""
    first, last = k, k + 1
    while first > 0 and x[first - 1] < x[first]:
        first -= 1
    while last < len(x) - 1 and x[last] < x[last + 1]:
        last += 1
    return first, last


def _slope(x: list[float], j: int, first: int, last: int) -> dict[int, float]:
    """The derivative at knot ``j`` of the parabola through it and its neighbours in its run (the two beside it, or at
    an end of the run the next two), as weights on knot values; the secant where the run has two knots."""
    if last - first == 1:
        width = x[last] - x[first]
        return {first: -1.0 / width, last: 1.0 / width}
    trio = (j - 1, j, j + 1) if first < j < last else (first, first + 1, first + 2) if j == first else \
        (last - 2, last - 1, last)
    weights: dict[int, float] = {}
    for a in trio:
        b, c = (other for other in trio if other != a)
        weights[a] = (2.0 * x[j] - x[b] - x[c]) / ((x[a] - x[b]) * (x[a] - x[c]))
    return weights


def _cubic_cell(x: list[float], k: int) -> tuple[np.ndarray, np.ndarray]:
    """The cubic Hermite polynomial on the cell ``k`` with the slopes of :func:`_slope`: its stencil (four knot
    indices, a repeated last one carrying nothing) and coefficients (power of ``t`` by stencil knot)."""
    first, last = _piece(x, k)
    width = x[k + 1] - x[k]
    terms: dict[int, np.ndarray] = {}

    def add(index: int, basis: np.ndarray) -> None:
        terms[index] = terms.get(index, np.zeros(4)) + basis

    add(k, _HERMITE[0])
    add(k + 1, _HERMITE[2])
    for knot, basis in ((k, _HERMITE[1]), (k + 1, _HERMITE[3])):
        for index, weight in _slope(x, knot, first, last).items():
            add(index, width * weight * basis)
    indices = sorted(terms)
    stencil = np.array(indices + [indices[-1]] * (4 - len(indices)), dtype=np.int32)
    coefficients = np.zeros((4, 4))
    for column, index in enumerate(indices):
        coefficients[:, column] = terms[index]
    return stencil, coefficients


__all__ = ["AxisArrays", "TableArrays", "compiled"]
