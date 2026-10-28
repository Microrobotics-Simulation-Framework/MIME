"""A table as the evaluator reads it: OpenMaterica's table description (its record format's section 6) on as many axes
as the table has, with MIME's smooth scheme and structure-preserving variant beside its two schemes.

Every rule of OpenMaterica's form is checked when a table is made, and a table that breaks one is refused
(:class:`TableInvalid`, OpenMaterica's ``form_invalid``); so are MIME's own conditions, each named where it is
checked. Numbers are compared by value and exactly, as OpenMaterica compares them."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

type Scheme = Literal["linear", "loglog", "cubic"]
type Structure = Literal["plain", "cholesky"]
type Side = Literal["left", "right"]

#: OpenMaterica's two schemes, then MIME's smooth one: piecewise cubic, continuously differentiable.
SCHEMES: tuple[Scheme, ...] = ("linear", "loglog", "cubic")
#: ``plain`` interpolates the values as they are; ``cholesky`` interpolates the Cholesky factor of a symmetric
#: positive definite value, its diagonal as logarithms, so that every value it returns is positive definite.
STRUCTURES: tuple[Structure, ...] = ("plain", "cholesky")
#: The side whose derivative a table returns at an interior knot of its readable range.
SIDES: tuple[Side, ...] = ("left", "right")


class TableInvalid(ValueError):
    """A table refused when it is made: OpenMaterica's ``form_invalid``, or a condition of MIME's evaluator."""


def _numbers(values: object, what: str) -> tuple[float, ...]:
    held = np.asarray(values, dtype=np.float64)
    if held.ndim != 1 or not np.all(np.isfinite(held)):
        raise TableInvalid(f"{what} is not a list of finite numbers: {values!r}")
    return tuple(float(v) for v in held)


@dataclass(frozen=True)
class Axis:
    """One axis of a table: the condition it tabulates (``name``) in its ``unit``, its knots ``x``, its declared
    ``edges`` and its ``source_range``, as OpenMaterica's table gives them (``axis``, ``axis_unit``, ``x``, ``edges``,
    ``source_range``).

    ``x`` holds at least two distinct values and never decreases. An edge lies strictly between ``x[0]`` and
    ``x[-1]`` and appears in ``x`` exactly twice, consecutively, the two giving the values below and above it; no other
    knot repeats. ``source_range``, where given, is the range the source says the values hold over: two numbers, the
    first not above the second, sharing with ``[x[0], x[-1]]`` a value that is no edge. The **readable range** runs
    from ``x[0]``, or ``source_range[0]`` where that is greater, to ``x[-1]``, or ``source_range[1]`` where that is
    smaller, ends included. It may be one point: a read there is the value, with no derivative."""

    name: str
    unit: str
    x: tuple[float, ...]
    edges: tuple[float, ...] = ()
    source_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.name), str) or not self.name:
            raise TableInvalid(f"an axis's name is a condition's name, a string: {self.name!r}")
        if not isinstance(cast(object, self.unit), str) or not self.unit:
            raise TableInvalid(f"axis {self.name!r}: its unit is a string, and this one is {self.unit!r}: a table "
                               f"whose unit is not stated cannot be read at a condition")
        x = _numbers(self.x, f"axis {self.name!r}'s x")
        edges = _numbers(self.edges, f"axis {self.name!r}'s edges")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "edges", edges)
        if self.source_range is not None:
            object.__setattr__(self, "source_range", cast(tuple[float, float],
                                                          _numbers(self.source_range, f"axis {self.name!r}'s "
                                                                                      f"source_range")))
        self._check()

    def _check(self) -> None:
        x, edges, name = self.x, self.edges, self.name
        if len(set(x)) < 2:
            raise TableInvalid(f"axis {name!r}: x holds fewer than two distinct values")
        if any(b < a for a, b in zip(x, x[1:], strict=False)):
            raise TableInvalid(f"axis {name!r}: x decreases somewhere")
        if len(set(edges)) != len(edges):
            raise TableInvalid(f"axis {name!r}: an edge is given twice")
        for edge in edges:
            if not x[0] < edge < x[-1]:
                raise TableInvalid(f"axis {name!r}: the edge {edge!r} does not lie strictly between x[0] and x[-1]")
            at = [i for i, v in enumerate(x) if v == edge]
            if len(at) != 2 or at[1] != at[0] + 1:
                raise TableInvalid(f"axis {name!r}: the edge {edge!r} does not appear in x exactly twice, "
                                   f"consecutively")
        repeated = {v for v in x if x.count(v) > 1} - set(edges)
        if repeated:
            raise TableInvalid(f"axis {name!r}: x repeats {sorted(repeated)}, which are no declared edges")
        if self.source_range is not None:
            if len(self.source_range) != 2:
                raise TableInvalid(f"axis {name!r}: source_range {self.source_range!r} is not two numbers")
            first, second = self.source_range
            if first > second:
                raise TableInvalid(f"axis {name!r}: source_range {self.source_range!r} is not two numbers, the "
                                   f"first not above the second")
            low, high = max(first, x[0]), min(second, x[-1])
            if low > high or (low == high and low in edges):
                raise TableInvalid(f"axis {name!r}: source_range {self.source_range!r} shares no value that is no "
                                   f"edge with [x[0], x[-1]]")

    @property
    def readable_range(self) -> tuple[float, float]:
        """The range a read is in range over, ends included."""
        if self.source_range is None:
            return self.x[0], self.x[-1]
        return max(self.x[0], self.source_range[0]), min(self.x[-1], self.source_range[1])


@dataclass(frozen=True)
class ConstantExtrapolation:
    """A constant extrapolation past a table's readable range, declared by its consumer with its ``basis``: outside the
    range, a read returns the value at the range's nearest end, taken from inside it, with the derivative zero. Never
    across or at a declared edge. The only extrapolation the evaluator computes: any other is the consumer's law,
    computed beside it from that end value."""

    basis: str

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.basis), str) or not self.basis.strip():
            raise TableInvalid("a constant extrapolation states its basis, a sentence")


@dataclass(frozen=True, eq=False)
class Table:
    """A table: its ``axes``, and its values ``y``, whose leading dimensions are the axes' knots in order and whose
    remaining ones are the shape of one value. ``scheme`` is how it interpolates between knots, along each axis:
    ``linear``, ``loglog`` (linear in the logarithms of the axis and the value, for positive ones) or ``cubic``
    (a piecewise cubic, continuously differentiable, exact on quadratics in each axis). ``structure`` is ``plain`` or
    ``cholesky`` (each value a symmetric positive definite matrix, whose Cholesky factor is interpolated). ``side`` is
    the side whose derivative a read returns at an interior knot. ``extrapolation`` is the consumer's declared
    extrapolation past the readable range, or ``None``: then a read outside it stops the run."""

    axes: tuple[Axis, ...]
    y: Any
    scheme: Scheme = "linear"
    structure: Structure = "plain"
    side: Side = "right"
    extrapolation: ConstantExtrapolation | None = None
    _compiled: dict[str, Any] = field(default_factory=dict[str, Any], init=False, repr=False)

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)
        if not axes or not all(isinstance(cast(object, axis), Axis) for axis in axes):
            raise TableInvalid("a table has at least one axis, each an Axis")
        if len({axis.name for axis in axes}) != len(axes):
            raise TableInvalid(f"a table's axes are named once each: {[axis.name for axis in axes]}")
        if self.scheme not in SCHEMES or self.structure not in STRUCTURES or self.side not in SIDES:
            raise TableInvalid(f"scheme {self.scheme!r}, structure {self.structure!r} and side {self.side!r}: the "
                               f"schemes are {SCHEMES}, the structures {STRUCTURES}, the sides {SIDES}")
        y = np.array(self.y, dtype=np.float64)
        knots = tuple(len(axis.x) for axis in axes)
        if y.shape[:len(axes)] != knots:
            raise TableInvalid(f"y's leading dimensions are {y.shape[:len(axes)]}, and the axes have {knots} knots")
        if not np.all(np.isfinite(y)):
            raise TableInvalid("y holds a number that is not finite")
        y.setflags(write=False)
        object.__setattr__(self, "y", y)
        if self.scheme == "loglog":
            if self.structure != "plain":
                raise TableInvalid("a loglog table's values are interpolated as logarithms: it takes no structure")
            if any(min(axis.x) <= 0 for axis in axes) or np.any(y <= 0):
                raise TableInvalid("a loglog table holds only positive x and y")
        if self.structure == "cholesky":
            self._check_definite(y)

    def _check_definite(self, y: np.ndarray) -> None:
        shape = self.value_shape
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
            raise TableInvalid(f"a cholesky table's value is a square matrix, and this one's shape is {shape}")
        held = y.reshape(-1, *shape)
        skew = np.max(np.abs(held - np.swapaxes(held, -1, -2)), axis=(-1, -2))
        if np.any(skew > 64 * np.finfo(np.float64).eps * np.max(np.abs(held), axis=(-1, -2))):
            raise TableInvalid("a cholesky table's values are symmetric to their rounding, and one is not")
        try:
            np.linalg.cholesky(held)
        except np.linalg.LinAlgError as failed:
            raise TableInvalid("a cholesky table's values are positive definite, and one is not") from failed

    @property
    def value_shape(self) -> tuple[int, ...]:
        """The shape of one value."""
        return cast(tuple[int, ...], cast(np.ndarray, self.y).shape[len(self.axes):])

    @property
    def value_order(self) -> int:
        """The smoothness order of a read's value across its table's switch, as the entry that owns the table
        declares it: a knot of a ``linear`` or ``loglog`` table is a kink (0); a ``cubic`` table is continuously
        differentiable (1) across every knot inside its readable range, and its readable range's ends stop the run
        unless an extrapolation is declared, under which they are kinks."""
        if self.scheme == "cubic" and self.extrapolation is None:
            return 1
        return 0

    def arrays(self, dtype: Any) -> Any:
        """The table compiled for reading in ``dtype`` (:class:`~mime.core.tables.TableArrays`), once per dtype."""
        from mime.core.tables._compile import compiled

        key = np.dtype(dtype).name
        if key not in self._compiled:
            self._compiled[key] = compiled(self, np.dtype(dtype))
        return self._compiled[key]


def query_range_problems(table: Table, low: Sequence[float], high: Sequence[float],
                         dtype: Any = np.float64) -> list[str]:
    """Why reads of ``table`` over the box ``[low, high]`` (a bound per axis, ends included) would stop the run, where
    the consumer knows that range at build: a declared edge in it, or across one from the readable range; or, with no
    extrapolation declared, any part of it outside the readable range. Empty where none would. Compared exactly, in
    ``dtype``, the dtype the run reads in: the bounds, the edges and the readable range as it holds them."""
    if len(low) != len(table.axes) or len(high) != len(table.axes):
        raise ValueError(f"a bound per axis: the table has {len(table.axes)} axes")
    kind = cast(Any, np.dtype(dtype).type)

    def held(value: float) -> float:
        return float(kind(value))

    found: list[str] = []
    for axis, lo, hi in zip(table.axes, low, high, strict=True):
        if lo > hi or not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"axis {axis.name!r}: the query range [{lo!r}, {hi!r}] is not two finite numbers in order")
        lo, hi = held(lo), held(hi)
        start, end = (held(v) for v in axis.readable_range)
        for edge in (held(e) for e in axis.edges):
            if lo <= edge <= hi:
                found.append(f"axis {axis.name!r}: the query range [{lo!r}, {hi!r}] {axis.unit} holds the declared "
                             f"edge {edge!r}, where the table gives no value")
            elif lo < edge <= start or end <= edge < hi:
                found.append(f"axis {axis.name!r}: the query range [{lo!r}, {hi!r}] {axis.unit} lies across the "
                             f"declared edge {edge!r} from the readable range [{start!r}, {end!r}], where no "
                             f"extrapolation applies")
        if table.extrapolation is None and (lo < start or hi > end):
            found.append(f"axis {axis.name!r}: the query range [{lo!r}, {hi!r}] {axis.unit} leaves the readable range "
                         f"[{start!r}, {end!r}], and no extrapolation is declared")
    return found


__all__ = ["SCHEMES", "SIDES", "STRUCTURES", "Axis", "ConstantExtrapolation", "Scheme", "Side", "Structure", "Table",
           "TableInvalid", "query_range_problems"]
