"""The small protocol plots, recipes and comparisons read, so that a result is not drawn from ``RunResult``'s internals:
what they need of a result is **series** (a name, a unit, a time base in seconds, values, an optional uncertainty),
**readings** (a name, a unit, a number), **flags** (where each was raised) and **validity ranges** (a numeric limit on a
series), plus the drive frequencies the run declared (for the aliasing rule). A simulated run adapts to it
(:class:`RunData`), and so does a result built by hand from plain arrays, units and a time base (:class:`Measured`): a
trajectory converted from a video, a stand-in for the measured data a later ``Dataset`` will carry. Both plot and compare
in the same code.

A study is read the same way: ``StudyData`` (members as :class:`ResultData`, the axes and each member's coordinates)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.registry import OutputKind
from mime.units import TIME, Quantity, unit_symbol

if TYPE_CHECKING:
    from mime.results._result import RunResult

UNCERTAINTY_KINDS = ("std", "bound")


@dataclass(frozen=True)
class Uncertainty:
    """An uncertainty of a series, ``amount`` at each sample (the series' unit, not negative): ``kind`` says which, ``std``
    (one standard deviation: the data's spread, an estimate) or ``bound`` (the data's stated limit)."""

    kind: str
    amount: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.kind not in UNCERTAINTY_KINDS:
            raise ArgumentRefused([Refusal("plot.uncertainty", RefusalClass.USER,
                                           f"an uncertainty is 'std' or 'bound', not {self.kind!r}",
                                           "say which: uncertainty=('std', amount)")])


@dataclass(frozen=True)
class SeriesData:
    """A series: ``time`` (s, strictly increasing), ``values`` (the first axis over time) in ``unit``; ``node`` and
    ``field`` name the state field a simulated one comes from (where a validity limit can apply to it)."""

    path: str
    unit: str
    time: NDArray[np.float64]
    values: NDArray[Any]
    uncertainty: Uncertainty | None = None
    node: str | None = None
    field: str | None = None
    dimension: Any = None


@dataclass(frozen=True)
class ReadingData:
    path: str
    unit: str
    value: NDArray[np.float64]
    observed: bool = True
    dimension: Any = None


@dataclass(frozen=True)
class FlagData:
    """A flag: where it was raised, as ``raised`` over ``steps`` (the base steps of a run; sample numbers of data) at
    ``times`` (s). ``raised`` is ``None`` where only a reduction was kept, and ``note`` says what it was."""

    path: str
    meaning: str
    steps: NDArray[np.int64]
    times: NDArray[np.float64]
    raised: NDArray[np.bool_] | None
    note: str = ""
    was_raised: bool | None = None


@dataclass(frozen=True)
class LimitData:
    """A validity range ``low`` to ``high`` of the series ``series`` (its unit), with the flag or line it is from."""

    series: str
    label: str
    low: float
    high: float
    source: str


class ResultData(Protocol):
    """What a plot or a comparison reads of one result."""

    @property
    def name(self) -> str: ...

    @property
    def drives(self) -> Sequence[tuple[float, str]]:
        """Rates (rad/s) at which the run declares a drive turning a field, with what each is from."""
        ...

    def series_data(self) -> Sequence[SeriesData]: ...

    def readings_data(self) -> Sequence[ReadingData]: ...

    def flags_data(self) -> Sequence[FlagData]: ...

    def limits_data(self) -> Sequence[LimitData]: ...

    def declared(self) -> set[tuple[str, str, str]]:
        """Each declared output as (path, kind, unit): what two arms must share to be compared."""
        ...

    def validity_lines(self) -> int: ...


class StudyData(Protocol):
    """What a study plot reads: members as results, and the axes with each member's coordinates."""

    experiment: str
    axes: tuple[str, ...]
    points: tuple[int, ...]
    groups: tuple[int, ...]

    def __len__(self) -> int: ...

    @property
    def coordinates(self) -> Mapping[str, Quantity | tuple[object, ...]]: ...

    def coordinate(self, path: str, index: int) -> object: ...

    def member_data(self, index: int) -> ResultData: ...


# ---------------------------------------------------------------------------------------------------------------


class RunData:
    """A :class:`~mime.results.RunResult` as a :class:`ResultData`."""

    def __init__(self, run: RunResult) -> None:
        self.run = run

    @property
    def name(self) -> str:
        return self.run.layout.experiment

    @property
    def drives(self) -> Sequence[tuple[float, str]]:
        return self.run.drives

    def series_data(self) -> Sequence[SeriesData]:
        return [SeriesData(s.name, s.unit, np.asarray(s.time.value, dtype=np.float64), np.asarray(s.raw),
                           None, s.output.node, s.output.field, s.output.dimension) for s in self.run.series()]

    def readings_data(self) -> Sequence[ReadingData]:
        from mime.results._series import Reading

        found: list[ReadingData] = []
        for held in self.run.layout.observations:
            for output in held.outputs:
                if output.kind is OutputKind.AT_END:
                    reading = cast(Reading, self.run.output(output))
                    found.append(ReadingData(output.path, output.unit, np.asarray(reading.quantity.value, dtype=np.float64),
                                             reading.observed, output.dimension))
        return found

    def flags_data(self) -> Sequence[FlagData]:
        run, found = self.run, []
        for flag in run.layout.flags:
            track = run.tracks.get(flag.path)
            if track is None:
                continue
            if track.raised is None:
                note = ""
                if track.whole is not None and track.whole.count:
                    note = (f"{flag.path} was raised at {track.whole.count} steps, first at {track.whole.first} and last at "
                            f"{track.whole.last}, and this run kept no per-step track, so no interval is drawn")
                found.append(FlagData(flag.path, flag.meaning, np.zeros(0, np.int64), np.zeros(0), None, note,
                                      track.was_raised))
                continue
            steps = run.start_step + 1 + np.arange(len(track.raised), dtype=np.int64)
            found.append(FlagData(flag.path, flag.meaning, steps, steps * run.dt, np.asarray(track.raised, dtype=bool),
                                  "", track.was_raised))
        return found

    def limits_data(self) -> Sequence[LimitData]:
        found: list[LimitData] = []
        for series in self.run.series():
            for flag in self.run.layout.flags:
                if flag.limits is None or (flag.node, flag.field) != (series.output.node, series.output.field):
                    continue
                for index, limit in enumerate(flag.limits):
                    if math.isfinite(limit):
                        found.append(LimitData(series.name, f"{flag.path} limit" + (f" [{index}]" if len(flag.limits) > 1 else ""),
                                               -limit, limit, flag.meaning))
        return found

    def declared(self) -> set[tuple[str, str, str]]:
        return {(o.path, o.kind.value, o.unit) for h in self.run.layout.observations for o in h.outputs}

    def validity_lines(self) -> int:
        report = self.run.report
        return len(cast(list[Any], report.get("validity", []))) if report is not None else 0


class Measured:
    """A result built by hand from plain arrays, units and a time base, to plot and compare as a simulated one does: a
    stand-in for measured data. ``time`` is a time quantity (strictly increasing); ``series`` maps a name to a quantity
    whose first axis is over time, with an optional ``uncertainty`` per name, ``(kind, quantity)`` (``kind`` ``"std"`` or
    ``"bound"``); ``readings`` maps a name to a quantity; ``flags`` maps a name to a boolean array over the samples (its
    meaning in ``meanings``); ``limits`` are ``(series, low, high, label)`` in the series' unit; ``drives`` are the
    frequencies of drives declared (a frequency quantity each)."""

    def __init__(self, name: str, time: Quantity, series: Mapping[str, Quantity],
                 readings: Mapping[str, Quantity] | None = None,
                 uncertainty: Mapping[str, tuple[str, Quantity]] | None = None,
                 flags: Mapping[str, Sequence[bool]] | None = None, meanings: Mapping[str, str] | None = None,
                 limits: Sequence[tuple[str, float, float, str]] = (), drives: Sequence[Quantity] = ()) -> None:
        if time.dimension != TIME:
            raise ArgumentRefused([Refusal("plot.time-base", RefusalClass.USER, "the time base is a time quantity",
                                           "give time=s(np.arange(n) / 30.0)")])
        self._name = name
        self._time = np.asarray(time.value, dtype=np.float64)
        if self._time.ndim != 1 or (len(self._time) > 1 and not np.all(np.diff(self._time) > 0)):
            raise ArgumentRefused([Refusal("plot.time-base", RefusalClass.USER, "the time base is a strictly increasing 1-D array",
                                           "give the samples' times in increasing order")])
        self._series = dict(series)
        for key, quantity in self._series.items():
            if np.shape(quantity.value)[:1] != self._time.shape:
                raise ArgumentRefused([Refusal("plot.time-base", RefusalClass.USER,
                                               f"{key} has {np.shape(quantity.value)[:1]} samples and the time base {self._time.shape}",
                                               "give each series one value for each time")])
        self._readings = dict(readings or {})
        self._uncertainty = dict(uncertainty or {})
        self._flags = {k: np.asarray(v, dtype=bool) for k, v in (flags or {}).items()}
        self._meanings = dict(meanings or {})
        self._limits = tuple(limits)
        self._drives = tuple((2.0 * math.pi * float(cast(float, q.value)), f"a declared drive of {float(cast(float, q.value)):g} Hz")
                             for q in drives)

    @property
    def name(self) -> str:
        return self._name

    @property
    def drives(self) -> Sequence[tuple[float, str]]:
        return self._drives

    def series_data(self) -> Sequence[SeriesData]:
        found = []
        for key, q in self._series.items():
            held = self._uncertainty.get(key)
            found.append(SeriesData(key, unit_symbol(q.dimension), self._time, np.asarray(q.value, dtype=np.float64),
                                    None if held is None else Uncertainty(held[0], np.asarray(held[1].value, dtype=np.float64)),
                                    None, None, q.dimension))
        return found

    def readings_data(self) -> Sequence[ReadingData]:
        return [ReadingData(k, unit_symbol(q.dimension), np.atleast_1d(np.asarray(q.value, dtype=np.float64)), True, q.dimension)
                for k, q in self._readings.items()]

    def flags_data(self) -> Sequence[FlagData]:
        steps = np.arange(len(self._time), dtype=np.int64)
        return [FlagData(k, self._meanings.get(k, ""), steps, self._time, v, "", bool(v.any())) for k, v in self._flags.items()]

    def limits_data(self) -> Sequence[LimitData]:
        return [LimitData(s, label, low, high, "a stated validity range") for s, low, high, label in self._limits]

    def declared(self) -> set[tuple[str, str, str]]:
        return ({(k, "per_step", unit_symbol(q.dimension)) for k, q in self._series.items()}
                | {(k, "at_end", unit_symbol(q.dimension)) for k, q in self._readings.items()})

    def validity_lines(self) -> int:
        return len(self._limits)


__all__ = ["UNCERTAINTY_KINDS", "FlagData", "LimitData", "Measured", "ReadingData", "ResultData", "RunData", "SeriesData",
           "StudyData", "Uncertainty"]
