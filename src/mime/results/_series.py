"""A run's values by time: series, the readings a window reduces to, the flags raised over a window, and the table of
columns every export is written from.

**One inclusion rule for every window**, the observations' own: a window from ``start`` to ``end`` holds the steps
whose end lies after ``start`` and not after ``end``, the half-open interval (start, end]. A step's end is its count
from the graph's start times the base step. The two ends are decided differently on purpose. The start is decided as
a node decides its ``after``, in the run's float type: the step's end and the start are each rounded to it
(:func:`~mime.results._layout.ended_after`), so in float32 a start on the grid may fall on either side of the step
that ends there, exactly as that node's does, and a user's own window (``between(s(2.5), s(5.0))``) is decided the
same way, whether or not a node shares its start. The end is a run's own end or a time the user wrote, and a node has
no end to match: a window's end within :data:`~mime.results._layout.ON_THE_GRID` of a step's end is that step's end.
A reduction follows the rule the node declares beside the output it reduces (:class:`~mime.registry.Reduction`), so a
series over an observation's window, reduced by that rule, is the output the graph reduced, to the rounding of the
node's own sums, on the grid as off it."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, override

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, MimeError, Refusal, RefusalClass
from mime.errors import public
from mime.registry import Reduction
from mime.results._layout import FlagLayout, OutputLayout, ended_after, position
from mime.units import TIME, Quantity, Unit, Values, si, unit_symbol


class NotInResult(MimeError, KeyError):
    """An observation or output a run's result does not hold. A ``KeyError``, as it was."""

    headline = "the result does not hold it"


#: Why a per-step output a run did not record is not in its result.
RECORDED_ONLY = "a run keeps a per-step output's series only where record= names it, and every output at the end always"


class NotRecorded(NotInResult):
    """A per-step output a run did not record, or a window reaching outside what it recorded. The message names the
    ``record=`` that keeps it."""






def _seconds(time: float) -> str:
    return f"{time:.6g} s"


def window_text(start: float, end: float) -> str:
    return f"({_seconds(start)}, {_seconds(end)}]"


def inside(steps: NDArray[np.int64], start: float, end: float, dt: float,
           clock: np.dtype[Any]) -> NDArray[np.bool_]:
    """Which of ``steps`` (step counts) lie in the window (start, end], by the one inclusion rule: the start as the
    run's nodes decide theirs (in its float type, ``clock``), the end on the grid."""
    return ended_after(steps, start, dt, clock) & (steps <= position(end, dt))


@dataclass(frozen=True)
class FlagTrack:
    """A flag over one run: whether it was raised at each of the run's steps (``raised``, one per step from its first
    on) when the run recorded anything, and whether it was raised at its last step, which is all a run that recorded
    nothing knows."""

    raised: NDArray[np.bool_] | None
    at_last: bool

    @property
    def was_raised(self) -> bool | None:
        """Whether it was raised at any step of the run: ``None`` when only the last step is known and it was not
        raised there."""
        if self.raised is not None:
            return bool(self.raised.any())
        return True if self.at_last else None


@dataclass(frozen=True)
class FlagRead:
    """A flag over a window: how many of the window's steps raised it (``None`` where the run did not record it per
    step) out of how many, and when it was first and last raised. ``at_last`` is whether it was raised at the run's last
    step, for a window that ends there; ``recorded_from`` is set when the window starts before the steps the run
    recorded (a run continued from an earlier one), from which time on the count holds.

    :attr:`was_raised` is whether the flag was raised anywhere in the window: ``True``, ``False``, or ``None`` when the
    run did not keep enough to say. A run that recorded nothing knows the last step alone, so a flag not raised there
    is ``None``, not ``False``: it may have been raised at any step before."""

    flag: FlagLayout
    start: float
    end: float
    raised: int | None
    steps: int | None
    first: float | None
    last: float | None
    at_last: bool | None
    recorded_from: float | None = None

    @property
    def path(self) -> str:
        return self.flag.path

    @property
    def was_raised(self) -> bool | None:
        """Whether it was raised anywhere in the window; ``None`` where that is not known."""
        if self.raised is None:
            return True if self.at_last else None
        if self.raised:
            return True
        return None if self.recorded_from is not None else False

    def text(self, *, meaning: bool = False) -> str:
        """What is known of it over the window; with ``meaning``, what it means when raised."""
        meaning_text = f" ({self.flag.meaning})" if meaning else ""
        if self.raised is None:
            known = {True: "at the run's last step it was raised, and before it is unknown",
                     False: "at the run's last step it was not raised, and before it is unknown",
                     None: "nothing is known of this window"}[self.at_last]
            return f"{self.path}: unknown, not recorded per step; {known}{meaning_text}"
        span = f"raised at {self.raised} of {self.steps} steps in {window_text(self.start, self.end)}"
        if self.raised and self.first is not None and self.last is not None:
            span += f", first at {_seconds(self.first)}, last at {_seconds(self.last)}"
        if self.recorded_from is not None:
            span += f"; not recorded before {_seconds(self.recorded_from)}"
        return f"{self.path}: {span}{meaning_text}"

    def manifest(self) -> dict[str, Any]:
        return {"flag": self.path, "entry": self.flag.owner, "raised_steps": self.raised, "steps": self.steps,
                "first": self.first, "last": self.last, "was_raised": self.was_raised,
                "raised_at_last_step": self.at_last, "recorded_from": self.recorded_from,
                "text": self.text(meaning=True)}


@dataclass(frozen=True)
class FlagSource:
    """Every flag of a run and what the run kept of each: its tracks, keyed by the flag's path, over the steps after
    ``start_step`` up to ``end_step``; ``clock`` is the run's float type, which decides a window's start."""

    flags: tuple[FlagLayout, ...]
    tracks: Mapping[str, FlagTrack]
    start_step: int
    end_step: int
    dt: float
    clock: np.dtype[Any]

    def over(self, start: float, end: float) -> tuple[FlagRead, ...]:
        """The flags of the window (start, end] that were raised in it, or that the run did not record."""
        steps = np.arange(self.start_step + 1, self.end_step + 1, dtype=np.int64)
        held = inside(steps, start, end, self.dt, self.clock)
        reaches_last = position(end, self.dt) >= self.end_step
        partial = position(start, self.dt) < self.start_step
        found: list[FlagRead] = []
        for flag in self.flags:
            track = self.tracks.get(flag.path)
            if track is None or track.raised is None:
                at_last = (track.at_last if track is not None else None) if reaches_last else None
                found.append(FlagRead(flag, start, end, None, None, None, None, at_last))
                continue
            raised = steps[held & track.raised]
            if len(raised) or partial:
                found.append(FlagRead(flag, start, end, len(raised), int(held.sum()),
                                      float(raised[0] * self.dt) if len(raised) else None,
                                      float(raised[-1] * self.dt) if len(raised) else None,
                                      bool(track.at_last) if reaches_last else None,
                                      self.start_step * self.dt if partial else None))
        return tuple(found)

    def raised_in_spans(self, steps: NDArray[np.int64], span: int) -> list[str]:
        """For each of ``steps``, the flags raised over the ``span`` steps ending there, joined by ``;``: a table's
        flags column. A flag the run did not record per step reads ``<flag>: not recorded``."""
        rows = [list[str]() for _ in steps]
        for flag in self.flags:
            track = self.tracks.get(flag.path)
            if track is None or track.raised is None:
                for row in rows:
                    row.append(f"{flag.path}: not recorded")
                continue
            counted = np.concatenate([[0], np.cumsum(track.raised, dtype=np.int64)])
            index = steps - self.start_step
            raised = counted[index] - counted[np.maximum(index - span, 0)]
            for row, count in zip(rows, raised, strict=True):
                if count:
                    row.append(flag.path)
        return [";".join(row) for row in rows]


def flags_text(flags: Sequence[FlagRead], *, meaning: bool = False) -> str:
    return "flags: " + ("; ".join(read.text(meaning=meaning) for read in flags) if flags else "none raised")


def quantity_of(values: NDArray[Any], dimension: Any) -> Quantity:
    """A state field's value as a quantity: a float for a scalar, a tuple for a vector, an array for anything larger."""
    held = np.asarray(values, dtype=np.float64)
    if held.ndim == 0:
        return Quantity(float(held), dimension)
    if held.ndim == 1:
        return Quantity(tuple(float(v) for v in held), dimension)
    return Quantity.of(held, dimension)


@dataclass(frozen=True)
class Reading:
    """One value of a run with its window and the flags raised over it: an output the graph reduced, or a series
    reduced over a window (``rule``, from ``samples`` samples). ``to(unit)`` reads it in a unit; it prints with its
    window and its flags, which never block the read."""

    name: str
    quantity: Quantity
    start: float
    end: float
    flags: tuple[FlagRead, ...]
    rule: Reduction | None = None
    samples: int | None = None

    @property
    def value(self) -> Values:
        """Its magnitude in SI."""
        return self.quantity.value

    @property
    def unit(self) -> str:
        return unit_symbol(self.quantity.dimension)

    def to(self, unit: Unit) -> Values:
        return self.quantity.to(unit)

    def text(self) -> str:
        how = f", {self.rule.value} of {self.samples} samples" if self.rule is not None and self.samples else ""
        return f"{self.name}{how} = {self.quantity} over {window_text(self.start, self.end)}; {flags_text(self.flags)}"

    @override
    def __str__(self) -> str:
        return self.text()


class Table(dict[str, NDArray[Any]]):
    """Columns by name, each name carrying its unit in brackets (``time [s]``, ``swim_speed:umr.speed [m/s]``), so
    that ``pandas.DataFrame(table)`` and ``polars.DataFrame(table)`` take it as it is and keep the units. ``units``
    maps each column to its unit alone; the ``flags`` column holds, on each row, the flags raised over the steps that
    row stands for."""

    units: dict[str, str]

    def __init__(self, columns: Mapping[str, NDArray[Any]], units: Mapping[str, str]) -> None:
        super().__init__(columns)
        self.units = dict(units)


def column_name(name: str, unit: str) -> str:
    return f"{name} [{unit}]"


def _value_columns(name: str, values: NDArray[Any], unit: str) -> dict[str, NDArray[Any]]:
    held = np.asarray(values, dtype=np.float64)
    if held.ndim == 1:
        return {column_name(name, unit): held}
    flat = held.reshape(len(held), -1)
    shape = held.shape[1:]
    return {column_name(f"{name}[{','.join(map(str, index))}]", unit): flat[:, i]
            for i, index in enumerate(np.ndindex(*shape))}


def series_table(series: Sequence[Series]) -> Table:
    """The table of ``series`` that share their samples' times: a time column, one column per scalar series and per
    component of a vector one, and the flags column."""
    first = series[0]
    for other in series[1:]:
        if not np.array_equal(other.steps, first.steps):
            raise ArgumentRefused([Refusal(
                "result.table", RefusalClass.USER,
                f"a table puts series side by side at the same times, and {other.name} is sampled at other times than "
                f"{first.name}",
                "record them in one run with one every=, run(..., record=[...], every=...), or take each series' own "
                ".table()")])
    columns: dict[str, NDArray[Any]] = {column_name("time", "s"): first.steps.astype(np.float64) * first.dt}
    units = {column_name("time", "s"): "s"}
    for held in series:
        found = _value_columns(held.name, held.raw, held.unit)
        columns.update(found)
        units.update({name: held.unit for name in found})
    columns["flags"] = np.asarray(first.flag_source.raised_in_spans(first.steps, first.every), dtype=np.str_)
    units["flags"] = ""
    return Table(columns, units)


class Series:
    """A per-step output a run recorded: its samples' times (``time``, from the graph's start), its values, and the
    window it covers, the run's steps (start, end]. ``between(start, end)`` takes a window of it by time under the one
    inclusion rule (the start decided in the run's float type, as a node decides its ``after``: in float32 a start on
    the grid may include the step that ends there, or not, as that node's does, and so does a start of the user's own,
    shared with no node); ``mean()`` and ``integral()`` reduce it as a node declares those rules; ``to(unit)`` reads its
    values in a unit; ``flags`` are those raised over its window. Each sample stands for ``every`` base steps."""

    def __init__(self, output: OutputLayout, steps: NDArray[np.int64], values: NDArray[Any], *, dt: float, every: int,
                 start: float, end: float, flags: FlagSource, clock: np.dtype[Any]) -> None:
        self.output = output
        self.steps = steps
        self.raw = values
        self.dt = dt
        self.every = every
        self.start = start
        self.end = end
        self.flag_source = flags
        self.clock = clock

    @property
    def name(self) -> str:
        return self.output.path

    @property
    def unit(self) -> str:
        return self.output.unit

    @property
    def time(self) -> Quantity:
        """Each sample's time, the end of the step it was taken at, from the graph's start."""
        return Quantity.of(self.steps.astype(np.float64) * self.dt, TIME)

    @property
    def values(self) -> Quantity:
        return Quantity.of(self.raw, self.output.dimension)

    @property
    def window(self) -> tuple[Quantity, Quantity]:
        return Quantity(self.start, TIME), Quantity(self.end, TIME)

    @property
    def flags(self) -> tuple[FlagRead, ...]:
        return self.flag_source.over(self.start, self.end)

    def __len__(self) -> int:
        return len(self.steps)

    def to(self, unit: Unit) -> Values:
        return self.values.to(unit)

    @public
    def between(self, start: Quantity, end: Quantity | None = None) -> Series:
        """The samples in the window (start, end] (``end`` the series' own when not given): the steps whose end lies
        after ``start`` and not after ``end``. The start is decided in the run's float type, as a node decides its
        ``after``, so in float32 a start on the grid may or may not take the step that ends there (a node counts it
        for about 57 % of on-grid starts at base steps such as 5e-4 s and 1e-3 s, where a float64 comparison would
        not), the same as a node with that ``after``; the end is on the grid. A window reaching outside what the run
        recorded is refused."""
        a = float(cast(float, si(start, TIME, what="a window's start")))
        b = self.end if end is None else float(cast(float, si(end, TIME, what="a window's end")))
        if not b > a:
            raise ArgumentRefused([Refusal(
                "result.window", RefusalClass.USER,
                f"a window ends after it starts; {window_text(a, b)} does not",
                "give the start first: between(s(start), s(end)), with end > start")])
        if position(a, self.dt) < position(self.start, self.dt) or position(b, self.dt) > position(self.end, self.dt):
            raise NotRecorded([Refusal(
                "result.not-recorded", RefusalClass.USER,
                f"{self.name} was recorded over {window_text(self.start, self.end)}, and the window "
                f"{window_text(a, b)} reaches outside it",
                f"read a window inside it, between(s({self.start!r}), s({self.end!r})), or record it with record=[...] "
                "in the run that covers the window",
                why=RECORDED_ONLY)])
        held = inside(self.steps, a, b, self.dt, self.clock)
        return Series(self.output, self.steps[held], self.raw[held], dt=self.dt, every=self.every, start=a, end=b,
                      flags=self.flag_source, clock=self.clock)

    def _reading(self, rule: Reduction, value: NDArray[np.float64], dimension: Any) -> Reading:
        return Reading(self.name, quantity_of(value, dimension), self.start, self.end, self.flags, rule, len(self))

    def _samples(self, rule: Reduction) -> NDArray[np.float64]:
        if not len(self):
            raise ArgumentRefused([Refusal(
                "result.window", RefusalClass.USER,
                f"no sample of {self.name} lies in {window_text(self.start, self.end)}, so it has no {rule.value}",
                "widen the window with between(...), or record more often with a smaller every=")])
        return np.asarray(self.raw, dtype=np.float64)

    @public
    def mean(self) -> Reading:
        """The mean of its samples, each standing for the same number of steps (:attr:`Reduction.MEAN`)."""
        return self._reading(Reduction.MEAN, self._samples(Reduction.MEAN).mean(axis=0), self.output.dimension)

    @public
    def integral(self) -> Reading:
        """The sum of its samples, each times the length it stands for (:attr:`Reduction.INTEGRAL`)."""
        total = self._samples(Reduction.INTEGRAL).sum(axis=0) * (self.every * self.dt)
        return self._reading(Reduction.INTEGRAL, total, self.output.dimension * TIME)

    @public
    def reduced(self, rule: Reduction) -> Reading:
        """Reduced by ``rule``, a rule a node declares for an output it reduces."""
        reductions: dict[Reduction, Callable[[], Reading]] = {Reduction.MEAN: self.mean,
                                                              Reduction.INTEGRAL: self.integral}
        return reductions[rule]()

    def table(self) -> Table:
        return series_table([self])

    def text(self) -> str:
        return (f"{self.name}: {len(self)} samples in {self.unit} over {window_text(self.start, self.end)}, one every "
                f"{self.every} base steps; {flags_text(self.flags)}")

    @override
    def __str__(self) -> str:
        return self.text()

    @override
    def __repr__(self) -> str:
        return f"<Series {self.name}: {len(self)} samples in {self.unit} over {window_text(self.start, self.end)}>"


__all__ = ["RECORDED_ONLY", "FlagRead", "FlagSource", "FlagTrack", "NotInResult", "NotRecorded", "Reading", "Series",
           "Table", "column_name", "flags_text", "inside", "quantity_of", "series_table", "window_text"]
