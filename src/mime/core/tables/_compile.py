"""A table compiled for reading in one dtype, on the host, once.

Each axis's knots, readable range and edges, as the dtype holds them, become the **breaks** a query is located among,
and every interval between two breaks holds one rule for what a read there returns:

- **inside** the readable range, a polynomial in the query's position in its knot cell, ``t``, from 0 at the cell's
  lower knot to 1 at its upper (in logarithms for ``loglog``), given by its coefficients on the values of a window of
  consecutive knots, the cell's **stencil**, from its ``start``;
- **outside** it, under a declared extrapolation, the constant the inside polynomial takes at the range's nearest end;
  with none, no number (NaN), since the run stops there;
- **refused**, at a declared edge or across one from the readable range: no number.

A readable range of one point has one inside interval, holding that point alone: there a read is the value, with no
derivative (NaN), as the interval's ``slopeless`` says.

A break is placed so that each interval holds exactly what it should in the dtype: the readable range's ends inside
it, an interior knot in the interval on the declared side, and an edge alone in an interval of its own, between it
and the next number of the dtype. So the one non-smooth operation a read evaluates is the location of the query among
the breaks, and everything else is selected by that index. Where it is no slower, an axis's **bucket index** locates a
query without a search (:class:`~mime.core.switch.Buckets`), giving the search's index exactly.

With ``stored_polynomials``, each knot cell's polynomial in every axis's ``t`` is computed here, in float64, and stored
in place of the values: a read gathers its cell's coefficients and evaluates them, with no stencil to combine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.core import switch
from mime.core._node import jnp
from mime.core.tables._table import GATHERED_AT_ONCE, Axis, Table, TableInvalid

#: The cubic Hermite basis on ``t`` in ``[0, 1]``, its coefficients from ``t**0`` to ``t**3``: the lower knot's value,
#: the lower slope times the cell's width, the upper value and the upper slope times the width.
_HERMITE = np.array([[1.0, 0.0, -3.0, 2.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 3.0, -2.0], [0.0, 0.0, -1.0, 1.0]])
#: The linear basis: ``f0 + t (f1 - f0)``.
_LINEAR = np.array([[1.0, 0.0], [-1.0, 1.0]])
#: How many buckets an axis's index may hold, as multiples of its breaks' intervals: tried in order, the fewest
#: candidates per bucket kept, the fewest buckets among equals.
BUCKET_FACTORS = (1, 2, 4, 8, 16)


@dataclass(frozen=True, eq=False)
class AxisArrays:
    """One axis, compiled: the ``breaks`` and, where it locates without a search, their ``buckets``; for each of the
    ``len(breaks) + 1`` intervals they bound, the ``origin`` and ``width`` that give ``t``, the ``start`` of its
    stencil's window of consecutive knots and the ``coefficients`` (power of ``t`` by window knot), whether it is
    ``inside`` the readable range, and whether it is ``refused``. ``scale`` is the axis's largest knot in magnitude,
    the scale of its location's resolution. ``slopeless`` marks the intervals whose reads have no derivative: the one
    inside interval of a readable range of one point. Under stored polynomials, ``cell`` is each interval's knot cell
    and ``offset`` is added to its ``t``: the range's end for a constant outside, NaN where the read is refused."""

    scale: Any
    breaks: Any
    buckets: switch.Buckets | None
    origin: Any
    width: Any
    start: Any
    coefficients: Any
    inside: Any
    refused: Any
    slopeless: Any
    cell: Any
    offset: Any


@dataclass(frozen=True, eq=False)
class TableArrays:
    """A table compiled for reading in one dtype: its ``axes``, and its ``values`` as they are interpolated (the
    logarithms of a ``loglog`` table's, the Cholesky factor of a ``cholesky`` table's, its diagonal's logarithms first,
    then the entries below it by row), or each knot cell's polynomial where they are ``stored_polynomials``, with what
    a read needs to know of the table, fixed at compilation. A pytree whose leaves are the arrays, on the device, so
    that a node can hold it in its ``params`` and a compiled step takes it without copying it."""

    axes: tuple[AxisArrays, ...]
    values: Any
    scheme: str | tuple[str, ...]
    structure: str
    side: str
    value_shape: tuple[int, ...]
    extrapolated: bool
    stored_polynomials: bool = False
    recomputed_in_reverse: bool = True
    gathered_at_once: int = GATHERED_AT_ONCE

    @property
    def dtype(self) -> Any:
        """The dtype a read runs in: its breaks', whatever its values are stored in."""
        return self.axes[0].breaks.dtype


_tree_util: Any = cast(Any, jax).tree_util
_tree_util.register_dataclass(AxisArrays, data_fields=["scale", "breaks", "buckets", "origin", "width", "start",
                                                       "coefficients", "inside", "refused", "slopeless", "cell",
                                                       "offset"], meta_fields=[])
_tree_util.register_dataclass(TableArrays, data_fields=["axes", "values"],
                              meta_fields=["scheme", "structure", "side", "value_shape", "extrapolated",
                                           "stored_polynomials", "recomputed_in_reverse", "gathered_at_once"])


def compiled(table: Table, dtype: np.dtype[Any]) -> TableArrays:
    """``table`` compiled for reading in ``dtype``."""
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"a table is read in a floating dtype, and {dtype} is not one")
    held_in(dtype)
    reachable(math.prod(len(axis.x) for axis in table.axes))
    axes = tuple(_compiled_axis(axis, scheme, table, dtype) for axis, scheme in zip(table.axes, table.schemes,
                                                                                    strict=True))
    stored = np.dtype(np.float32) if table.float32_values else dtype
    values = _interpolated_values(table, dtype)
    if table.stored_polynomials:
        values = _polynomials(values.astype(np.float64), table, dtype)
        reachable(math.prod(values.shape[:-1]))
    found = TableArrays(axes, _held(values.astype(stored)), table.scheme, table.structure, table.side,
                        table.value_shape, table.extrapolation is not None, table.stored_polynomials,
                        table.recomputed_in_reverse, table.gathered_at_once)
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
    held = np.ascontiguousarray(held).reshape(np.shape(held))  # a number stays a number, not an array of one
    held.setflags(write=False)
    return held


def _interpolated_values(table: Table, dtype: np.dtype[Any]) -> np.ndarray:
    """The table's values as the dtype holds them, in the form that is interpolated, one value's components last."""
    y = np.asarray(table.y, dtype=dtype).astype(np.float64)
    knots = y.shape[:len(table.axes)]
    if table.scheme == "loglog":
        return np.log(y).astype(dtype).reshape(*knots, -1)
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
        return held.reshape(*knots, -1).astype(dtype)
    return y.astype(dtype).reshape(*knots, -1)


def float32_holds(table: Table) -> None:
    """Refuse a table stored in float32 under any read (``float32_values``) whose values, in the form it stores them
    (the logarithms of a ``loglog`` table's, the Cholesky factor of a ``cholesky`` table's), float32 holds only past
    its range, which a read returns as no number, or below its normal numbers, which a compiled read flushes to
    zero."""
    held = _interpolated_values(table, np.dtype(np.float64))
    with np.errstate(over="ignore"):
        past = int(np.sum(~np.isfinite(held.astype(np.float32))))
    magnitude: np.ndarray = np.abs(held)
    below = int(np.count_nonzero((magnitude < float(np.finfo(np.float32).tiny)) & (magnitude > 0.0)))
    problems = [f"{past} past its range, which a read returns as no number"] if past else []
    problems += [f"{below} below its normal numbers, which a compiled read flushes to zero"] if below else []
    if problems:
        raise TableInvalid(f"a table stored in float32 (float32_values) holds values, in the form it stores them, "
                           f"that float32 does not: {' and '.join(problems)}; store it in its read's dtype")


def _polynomials(values: np.ndarray, table: Table, dtype: np.dtype[Any]) -> np.ndarray:
    """Each knot cell's polynomial, its coefficients by power of each axis's ``t`` in order, then the value's
    components: ``values`` combined over each axis's cell stencils, in float64. A cell of no width (an edge's) holds
    zeros, which no read reaches."""
    held = values
    for a, (axis, scheme) in enumerate(zip(table.axes, table.schemes, strict=True)):
        x = [float(cast(Any, dtype.type)(v)) for v in axis.x]
        cells = _cells(x, scheme, table.scheme == "loglog")
        size, degree = _window(scheme, len(x)), _degree(scheme)
        starts = np.zeros(len(x) - 1, dtype=np.int64)
        coefficients = np.zeros((len(x) - 1, degree, size))
        for k, (start, matrix) in cells.items():
            starts[k], coefficients[k] = start, matrix
        windows = np.take(held, starts[:, None] + np.arange(size)[None, :], axis=a)  # (..., cells, size, ...)
        combined = np.einsum("kps,ks...->kp...", coefficients, np.moveaxis(windows, (a, a + 1), (0, 1)))
        held = np.moveaxis(combined, (0, 1), (a, -2))  # the cell in the axis's place, its powers before the value's
    return held


def _window(scheme: str, knots: int) -> int:
    """How many consecutive knots a cell's stencil spans on an axis of ``knots`` knots."""
    return min(4, knots) if scheme == "cubic" else 2


def _degree(scheme: str) -> int:
    """How many powers of ``t`` a cell's polynomial holds."""
    return 4 if scheme == "cubic" else 2


def _compiled_axis(axis: Axis, scheme: str, table: Table, dtype: np.dtype[Any]) -> AxisArrays:
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
    cells = _cells(x, scheme, logarithmic)
    size, degree = _window(scheme, len(x)), _degree(scheme)

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
    start = np.zeros(count, dtype=np.int32)
    coefficients = np.full((count, degree, size), math.nan)
    slopeless = np.zeros(count, dtype=bool)
    cell = np.full(count, inside_cells[0], dtype=np.int32)
    offset = np.full(count, math.nan)
    for i, (role, k) in enumerate(roles):
        if role == "inside":
            start[i], coefficients[i] = cells[k]
            slopeless[i] = low == high
            origin[i] = coordinate[k]
            width[i] = coordinate[k + 1] - coordinate[k]
            cell[i], offset[i] = k, 0.0
        elif role in ends:
            end, end_cell = ends[role]
            start[i] = cells[end_cell][0]
            cell[i] = end_cell
            if table.extrapolation is not None:
                at = math.log(end) if logarithmic else end
                t = (at - coordinate[end_cell]) / (coordinate[end_cell + 1] - coordinate[end_cell])
                coefficients[i] = 0.0
                coefficients[i, 0] = np.power(t, np.arange(degree, dtype=np.float64)) @ cells[end_cell][1]
                offset[i] = t
        else:
            start[i] = cells[inside_cells[0]][0]
    stored = table.stored_polynomials
    return AxisArrays(
        scale=_held(np.asarray(max(abs(v) for v in x), dtype=dtype)),
        breaks=_held(np.asarray(breaks, dtype=dtype)),
        buckets=bucket_index(breaks, dtype),
        origin=_held(origin.astype(dtype)),
        width=_held(width.astype(dtype)),
        start=None if stored else _held(start),
        coefficients=None if stored else _held(coefficients.astype(dtype)),
        inside=_held(np.array([role == "inside" for role, _ in roles])),
        refused=_held(np.array([role == "refused" for role, _ in roles])),
        slopeless=_held(slopeless),
        cell=_held(cell) if stored else None,
        offset=_held(offset.astype(dtype)) if stored else None)


def bucket_index(breaks: list[float], dtype: np.dtype[Any]) -> switch.Buckets | None:
    """The index of an axis's ``breaks`` (as ``dtype`` holds them) that locates a query among them without a search
    (:class:`~mime.core.switch.Buckets`): equal buckets in the breaks, or in their logarithms on a positive axis,
    whichever compares a query with fewer candidates; ``None`` where every choice compares more candidates than the
    search takes steps. A bucket's candidates are the breaks within a margin of it, wider than the rounding of a
    query's bucket: 64 units in the last place of the largest coordinate the buckets map, the logarithm's included.
    A coordinate whose offset from the first break the dtype cannot hold across the axis (an axis wider than the
    dtype's range), or whose scale is no normal number of the dtype (which a compiled read flushes to zero), has no
    buckets: a query's bucket would be wrong there."""
    kind = cast(Any, dtype.type)
    unit = float(np.finfo(dtype).eps)
    held = np.asarray(breaks, dtype=np.float64)
    steps = math.ceil(math.log2(len(held) + 1))
    found: tuple[int, int, switch.Buckets] | None = None
    for logarithmic in (False, True) if held[0] > 0 else (False,):
        coordinate = np.log(held) if logarithmic else held
        origin = float(kind(coordinate[0]))
        if not coordinate[-1] > origin:  # a span the dtype's origin swallows: no buckets in this coordinate
            continue
        for factor in BUCKET_FACTORS:
            count = factor * (len(held) - 1)
            with np.errstate(over="ignore"):  # a span or scale past the dtype's range, refused just below
                scale = float(kind(count / (coordinate[-1] - origin)))
            if not mapped_as_made(float(coordinate[-1]), origin, scale, kind):
                continue
            mapped = (coordinate - origin) * scale
            margin = 64 * unit * (count + max(abs(coordinate[0]), abs(coordinate[-1]), 1.0) * scale)
            low = np.concatenate([[-math.inf], np.arange(1, count) - margin])
            high = np.concatenate([np.arange(1, count) + margin, [math.inf]])
            first = np.searchsorted(mapped, low, side="left")
            candidates = int(np.max(np.searchsorted(mapped, high, side="left") - first))
            if found is not None and (candidates, count) >= found[:2]:
                continue
            first = np.minimum(first, len(held) - candidates)
            found = (candidates, count, switch.Buckets(
                origin=_held(np.asarray(origin, dtype=dtype)), scale=_held(np.asarray(scale, dtype=dtype)),
                first=_held(first.astype(np.int32)),
                candidates=_held(held[first[:, None] + np.arange(candidates)[None, :]].astype(dtype)),
                logarithmic=logarithmic))
    return found[2] if found is not None and found[0] <= steps else None


def mapped_as_made(last: float, origin: float, scale: float, kind: Any) -> bool:
    """Whether a read maps a query to the bucket these buckets were made with: the query's offset from the
    ``origin``, computed in the dtype as a read computes it, a number at the axis's ``last`` coordinate (and so at
    every coordinate between), and the ``scale`` a normal number of the dtype, which a compiled read does not flush to
    zero."""
    with np.errstate(over="ignore"):
        offset = kind(last) - kind(origin)
    return bool(np.isfinite(offset)) and float(np.finfo(kind).tiny) <= scale < math.inf


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


def _cells(x: list[float], scheme: str, logarithmic: bool) -> dict[int, tuple[int, np.ndarray]]:
    """Each knot cell's stencil window start and coefficients (power of ``t`` by window knot), by its lower knot's
    index: every cell of positive width."""
    found: dict[int, tuple[int, np.ndarray]] = {}
    coordinate = [math.log(v) for v in x] if logarithmic else x
    for k in range(len(x) - 1):
        if x[k] < x[k + 1]:
            found[k] = _cubic_cell(coordinate, k) if scheme == "cubic" else (k, _LINEAR.copy())
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


def _cubic_cell(x: list[float], k: int) -> tuple[int, np.ndarray]:
    """The cubic Hermite polynomial on the cell ``k`` with the slopes of :func:`_slope`: its window's start (four
    consecutive knots, or every knot of a shorter axis, the window kept inside the axis) and its coefficients (power
    of ``t`` by window knot, zero on a knot the stencil does not reach)."""
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
    size = _window("cubic", len(x))
    start = min(min(terms), len(x) - size)
    coefficients = np.zeros((4, size))
    for index, column in terms.items():
        coefficients[:, index - start] = column
    return start, coefficients


__all__ = ["BUCKET_FACTORS", "AxisArrays", "TableArrays", "bucket_index", "compiled", "float32_holds",
           "mapped_as_made"]
