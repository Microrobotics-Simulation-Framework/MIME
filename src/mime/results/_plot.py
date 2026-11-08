"""One backend-neutral description of a plot, built from a result and only drawn by a backend (matplotlib's ``plot()``,
plotly's ``plot_interactive()``): so a paper's figure and a card's say the same thing, and a third backend later is a
method and not a break.

What it holds, in the quantities' units (SI values; each axis label names its unit):

- **traces**: the curves, points or bars, each labelled with the output and, in a study, the member and its coordinates;
- **spans**: a flag shown as a shaded interval of the x axis, exactly where the result's own track raised it (a run's
  steps, or a study's members along a coordinate), never inferred from the values;
- **bands**: a validity range as a band of the y axis, where the report states a numeric limit on the plotted quantity
  (a flag's limits on the quantity's own field). A validity line that is prose has no band to draw and is counted in the
  notes;
- **notes**: what the figure bounds and what it estimates, and what it cannot show (a flag whose run kept no per-step
  track).

The description is data: ``to_dict()`` and ``to_json()`` (sorted keys, floats as ``repr`` writes them) are equal for
equal data, which is how a matplotlib figure is compared (by its description, never by its pixels)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._data import ResultData, SeriesData, StudyData
from mime.units import Quantity


PLOT_FORMAT = "mime-plot/1"
KINDS = ("series", "overlay", "curve", "distribution")

ESTIMATES = ("These curves are what the model computes: they estimate the quantity, and say nothing of how far the model "
             "is from the system it stands for.")
MEASURED = ("These curves are measured: they are data, with the measurement's own error (a band, where the data gives one), "
            "not a model's output. Against a model's run, the miss estimates how far the model is from them; it bounds the "
            "model's error from above only where the measurement's own error is known.")
BAND_TEXT = {"std": "the band is one standard deviation of the data (its spread: an estimate, not a bound)",
             "bound": "the band is the data's stated bound on the value"}
STUDY_ESTIMATES = ("The members are the same model run at other values: the differences between them are the model's "
                   "response to those values, an estimate, and bound nothing about the model's error.")


@dataclass(frozen=True)
class Trace:
    """One curve (``line``), set of points (``points``), curve with its points (``line+points``) or set of bars
    (``bars``, ``x`` their centres, ``width`` their width). ``member`` is the study member it belongs to, ``marked``
    which of its points have a flag raised (a study's curve)."""

    label: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    style: str = "line"
    member: int | None = None
    marked: NDArray[np.bool_] | None = None
    width: float | None = None
    #: The series' uncertainty as a band: ``y - amount`` to ``y + amount``, and which kind it is (``std`` or ``bound``).
    amount: NDArray[np.float64] | None = None
    uncertainty: str | None = None


@dataclass(frozen=True)
class Span:
    """A flag raised over ``start`` to ``end`` of the x axis (in its unit), from sample or step ``first`` to ``last``
    (inclusive; the base steps of a run, the members of a curve in the order of their coordinate)."""

    flag: str
    meaning: str
    start: float
    end: float
    first: int
    last: int
    member: int | None = None


@dataclass(frozen=True)
class Band:
    """A validity range ``low`` to ``high`` of the y axis (in its unit): a flag's numeric limit on the plotted quantity,
    with what the flag says it means."""

    label: str
    low: float
    high: float
    source: str


@dataclass(frozen=True)
class Panel:
    title: str
    x_label: str
    y_label: str
    traces: tuple[Trace, ...]
    spans: tuple[Span, ...] = ()
    bands: tuple[Band, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlotDescription:
    kind: str
    title: str
    panels: tuple[Panel, ...]
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        def number(value: float) -> float | None:
            return float(value) if math.isfinite(value) else None

        def trace(t: Trace) -> dict[str, Any]:
            return {"label": t.label, "style": t.style, "member": t.member, "width": t.width,
                    "x": [number(v) for v in t.x.tolist()], "y": [number(v) for v in t.y.tolist()],
                    "marked": None if t.marked is None else [bool(v) for v in t.marked.tolist()],
                    "uncertainty": t.uncertainty,
                    "amount": None if t.amount is None else [number(v) for v in t.amount.tolist()]}

        return {"format": PLOT_FORMAT, "kind": self.kind, "title": self.title, "notes": list(self.notes),
                "panels": [{"title": p.title, "x_label": p.x_label, "y_label": p.y_label,
                            "traces": [trace(t) for t in p.traces],
                            "spans": [{"flag": s.flag, "meaning": s.meaning, "start": number(s.start),
                                       "end": number(s.end), "first": s.first, "last": s.last, "member": s.member}
                                      for s in p.spans],
                            "bands": [{"label": b.label, "low": number(b.low), "high": number(b.high),
                                       "source": b.source} for b in p.bands],
                            "notes": list(p.notes)} for p in self.panels]}

    def to_json(self) -> str:
        """Equal descriptions are equal bytes."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=1) + "\n"

    def summary(self) -> dict[str, Any]:
        """What the figure shows, without its data: for a recipe to be read without MIME."""
        return {"kind": self.kind, "title": self.title, "notes": list(self.notes),
                "panels": [{"title": p.title, "x_label": p.x_label, "y_label": p.y_label,
                            "traces": [t.label for t in p.traces], "spans": [f"{s.flag} {s.first}..{s.last}" for s in p.spans],
                            "bands": [b.label for b in p.bands]} for p in self.panels]}


def estimates_note(data: ResultData) -> str:
    """What the figure of ``data`` bounds and estimates, by the result's own origin: a simulated one is the model's
    computation, a measured one is data."""
    return ESTIMATES if data.provenance == "simulated" else MEASURED


def refuse(code: str, category: RefusalClass, message: str, action: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(code, category, message, action)])


def label(name: str, unit: str) -> str:
    return f"{name} [{unit}]" if unit else name


# ---------------------------------------------------------------------------------------------------------------
# Flags as intervals


def runs_of(raised: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """The maximal runs of True in ``raised`` as (first, last) indices, inclusive."""
    held = np.concatenate([[False], np.asarray(raised, dtype=bool), [False]])
    edges = np.flatnonzero(held[1:] != held[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def run_spans(data: ResultData, member: int | None = None) -> tuple[list[Span], list[str]]:
    """Each flag as the intervals its own track raised it over (a step ``k`` stands at its time): ``first`` and ``last`` are
    the steps. A flag with no track per step (only its reduction) draws no interval, and is named in the notes."""
    spans: list[Span] = []
    notes: list[str] = []
    for flag in data.flags_data():
        if flag.raised is None:
            if flag.note:
                notes.append(flag.note)
            continue
        for first, last in runs_of(flag.raised):
            spans.append(Span(flag.path, flag.meaning, float(flag.times[first]), float(flag.times[last]),
                              int(flag.steps[first]), int(flag.steps[last]), member))
    return spans, notes


def validity_note(data: ResultData) -> list[str]:
    lines = data.validity_lines()
    return [f"{lines} validity lines are declared for this run (result.report); a band is drawn for those that state a "
            f"numeric limit on a plotted quantity, and the rest are prose"] if lines else []


# ---------------------------------------------------------------------------------------------------------------
# A run


def _components(series: SeriesData) -> list[tuple[str, NDArray[np.float64], NDArray[np.float64] | None]]:
    values = np.asarray(series.values, dtype=np.float64)
    amount = None if series.uncertainty is None else np.asarray(series.uncertainty.amount, dtype=np.float64)
    if values.ndim <= 1:
        return [(series.path, values, amount)]
    flat = values.reshape(len(values), -1)
    flat_amount = None if amount is None else amount.reshape(len(values), -1)
    return [(f"{series.path}[{i}]", flat[:, i], None if flat_amount is None else flat_amount[:, i])
            for i in range(flat.shape[1])]


def _trace(series: SeriesData, name: str, values: NDArray[np.float64], amount: NDArray[np.float64] | None,
           label_: str | None = None, member: int | None = None) -> Trace:
    return Trace(label_ or name, np.asarray(series.time, dtype=np.float64), values, "line", member, amount=amount,
                 uncertainty=None if amount is None or series.uncertainty is None else series.uncertainty.kind)


def describe_run(data: ResultData, outputs: Sequence[str] = (), *, title: str | None = None) -> PlotDescription:
    """The recorded series of ``data`` against time: one panel for each unit, the flags as intervals over every panel, the
    numeric limits on a series as bands, an uncertainty as a band about its curve. Refused (``plot.nothing``) where it
    holds no series."""
    held = list(data.series_data())
    if outputs:
        by_name = {series.path: series for series in held}
        absent = [name for name in outputs if name not in by_name]
        if absent:
            raise refuse("plot.output", RefusalClass.USER, f"the result holds no series {', '.join(absent)}; it holds "
                         f"{', '.join(by_name) or 'none'}", "plot an output the run recorded: record=[...] names it")
        held = [by_name[name] for name in outputs]
    if not held:
        raise refuse("plot.nothing", RefusalClass.USER, "the result holds no series, so there is no curve to draw (an output at the "
                     "end is one number)", "record one: run(..., record=[swim.speed])")
    spans, notes = run_spans(data)
    limits = list(data.limits_data())
    units = list(dict.fromkeys(series.unit for series in held))
    kinds = {series.uncertainty.kind for series in held if series.uncertainty is not None}
    panels: list[Panel] = []
    for unit in units:
        own = [series for series in held if series.unit == unit]
        traces = tuple(_trace(series, name, values, amount) for series in own for name, values, amount in _components(series))
        bands = [Band(l.label, l.low, l.high, l.source) for series in own for l in limits if l.series == series.path]
        panels.append(Panel(", ".join(series.path for series in own), label("time", "s"), label("value", unit), traces,
                            tuple(spans), tuple(dict.fromkeys(bands)), ()))
    return PlotDescription("series", title or f"{data.name}: {', '.join(s.path for s in held)}", tuple(panels),
                           (estimates_note(data), *(BAND_TEXT[k] for k in sorted(kinds)), *notes, *validity_note(data)))


# ---------------------------------------------------------------------------------------------------------------
# A study


def _member_text(study: StudyData, member: int) -> str:
    from mime.study._axes import value_text

    return f"m{study.points[member]}: " + ", ".join(f"{path} = {value_text(study.coordinate(path, member))}"
                                                     for path in study.axes)


def _output_of(path: str) -> None:
    observation, _, name = path.rpartition(".")
    if not observation or not name:
        raise refuse("plot.output", RefusalClass.USER, f"{path!r} is not an output path (<observation>.<output>)",
                     "name an output as result[swim].mean_speed has it: 'swim_speed:umr.mean_speed'")


def kind_of(study: StudyData, path: str) -> str:
    """``per_step`` or ``at_end``: how some member declares the output."""
    _output_of(path)
    for index in range(len(study)):
        for found, kind, _ in study.member_data(index).declared():
            if found == path:
                return kind
    every = sorted({p for i in range(len(study)) for p, _, _ in study.member_data(i).declared()})
    raise refuse("plot.output", RefusalClass.USER, f"no member of the study observes {path}", f"name an output of the study, `result.plot('{every[0]}')`: {', '.join(every)}")


def _reading(study: StudyData, path: str) -> tuple[NDArray[np.float64], str]:
    _output_of(path)
    values: list[float] = []
    unit = ""
    for index in range(len(study)):
        found = next((r for r in study.member_data(index).readings_data() if r.path == path), None)
        if found is None:
            if any(s.path == path for s in study.member_data(index).series_data()):
                raise refuse("plot.shape", RefusalClass.USER, f"{path} is a series, and this plot draws one number a member",
                             "plot an output at the end, or overlay the series: kind='overlay'")
            raise refuse("plot.output", RefusalClass.USER, f"member {index} does not observe {path}", "plot an output every member observes: `result.plot('swim_speed:umr.mean_speed')`")
        value = np.asarray(found.value, dtype=np.float64)
        if value.size != 1:
            raise refuse("plot.shape", RefusalClass.USER, f"{path} is a vector for each member, and this plot draws one number a member",
                         "plot a scalar output: `result.plot('swim_speed:umr.mean_speed')`")
        values.append(float(value.ravel()[0]) if found.observed else float("nan"))
        unit = found.unit
    return np.asarray(values), unit


def _coordinate(study: StudyData, against: str | None) -> tuple[str, NDArray[np.float64], str]:
    from mime.units import unit_symbol

    def varies(path: str) -> bool:
        held = study.coordinates[path]
        return len({repr(v) for v in (np.asarray(held.value).tolist() if isinstance(held, Quantity) else held)}) > 1

    paths = [p for p in study.axes if varies(p)] or list(study.axes)
    if against is None:
        if len(paths) != 1:
            raise refuse("plot.coordinate", RefusalClass.USER, f"the study varies {', '.join(paths)}; name the one to draw against",
                         f"plot(..., against={paths[0]!r})")
        against = paths[0]
    held = study.coordinates.get(against)
    if held is None:
        raise refuse("plot.coordinate", RefusalClass.USER, f"{against!r} is no axis of the study (its axes: {', '.join(study.axes)})",
                     "name an axis: plot(..., against='field.commands.frequency')")
    if not isinstance(held, Quantity) or np.asarray(held.value).ndim != 1:
        raise refuse("plot.coordinate", RefusalClass.USER, f"{against} holds names or vectors, which a curve has no axis for",
                     "draw against an axis of numbers with a unit, or overlay the members: kind='overlay'")
    return against, np.asarray(held.value, dtype=np.float64), unit_symbol(held.dimension)


def member_flags(study: StudyData) -> dict[str, list[bool | None]]:
    """Each flag any member declares, with whether each member raised it (``None`` where it kept too little to say, and
    where it does not declare the flag)."""
    names = sorted({f.path for i in range(len(study)) for f in study.member_data(i).flags_data()})
    out: dict[str, list[bool | None]] = {n: [] for n in names}
    for i in range(len(study)):
        own = {f.path: f.was_raised for f in study.member_data(i).flags_data()}
        for n in names:
            out[n].append(own.get(n))
    return out


def _flag_spans(study: StudyData, flags: dict[str, list[bool | None]], order: NDArray[np.intp],
                x: NDArray[np.float64]) -> list[Span]:
    """Each flag as the runs of consecutive members (by their coordinate) that raised it, from the first member's
    coordinate to the last's."""
    spans: list[Span] = []
    for path, readings in flags.items():
        raised = np.asarray([readings[i] is True for i in order], dtype=bool)
        meaning = next((f.meaning for i in range(len(study)) for f in study.member_data(i).flags_data() if f.path == path), "")
        for first, last in runs_of(raised):
            spans.append(Span(path, meaning, float(x[order[first]]), float(x[order[last]]), first, last))
    return spans


def _curve(study: StudyData, path: str, against: str | None, title: str | None) -> PlotDescription:
    values, unit = _reading(study, path)
    axis, x, x_unit = _coordinate(study, against)
    others = [p for p in study.axes if p != axis]

    def text(p: str, i: int) -> str:
        held = study.coordinates[p]
        return repr(np.asarray(held.value)[i]) if isinstance(held, Quantity) else repr(held[i])

    keys = [tuple(text(p, i) for p in others) for i in range(len(study))]
    flags = member_flags(study)
    raised = np.asarray([any(flags[f][i] is True for f in flags) for i in range(len(study))], dtype=bool) \
        if flags else np.zeros(len(study), dtype=bool)
    traces: list[Trace] = []
    for key in dict.fromkeys(keys):
        take = np.asarray([i for i, k in enumerate(keys) if k == key], dtype=np.intp)
        take = take[np.argsort(x[take], kind="stable")]
        shown = ", ".join(f"{p} = {v}" for p, v in zip(others, key, strict=True))
        traces.append(Trace(path + (f" ({shown})" if shown else ""), x[take], values[take], "line+points",
                            marked=raised[take]))
    spans = _flag_spans(study, flags, np.argsort(x, kind="stable"), x) if not others else []
    notes = [STUDY_ESTIMATES]
    if raised.any():
        notes.append("a point drawn open has a flag raised in its member (the shaded intervals say which)")
    if others:
        notes.append(f"{', '.join(others)} also vary: one curve for each combination; flags are marked on the points, "
                     f"and no interval is drawn across curves")
    panel = Panel(path, label(axis, x_unit), label(path, unit), tuple(traces), tuple(spans), (), ())
    return PlotDescription("curve", title or f"{study.experiment}: {path} against {axis}", (panel,), tuple(notes))


def _overlay(study: StudyData, path: str, members: Sequence[int] | None, title: str | None) -> PlotDescription:
    chosen = list(range(len(study))) if members is None else list(members)
    traces: list[Trace] = []
    spans: list[Span] = []
    notes = [STUDY_ESTIMATES]
    unit = ""
    bands: list[Band] = []
    kinds: set[str] = set()
    for index in chosen:
        data = study.member_data(index)
        series = next((s for s in data.series_data() if s.path == path), None)
        if series is None:
            raise refuse("plot.output", RefusalClass.USER, f"member {index} did not record {path}", "name it in the sweep's record=")
        unit = series.unit
        if series.uncertainty is not None:
            kinds.add(series.uncertainty.kind)
        for name, values, amount in _components(series):
            traces.append(_trace(series, name, values, amount, _member_text(study, index) + ("" if name == path else f" {name}"),
                                 index))
        found_spans, more = run_spans(data, index)
        spans += found_spans
        notes += [f"member {index}: {line}" for line in more]
        bands += [Band(l.label, l.low, l.high, l.source) for l in data.limits_data() if l.series == path]
    notes += [BAND_TEXT[k] for k in sorted(kinds)]
    panel = Panel(path, label("time", "s"), label(path, unit), tuple(traces), tuple(spans), tuple(dict.fromkeys(bands)))
    return PlotDescription("overlay", title or f"{study.experiment}: {path}, members overlaid", (panel,), tuple(notes))


def _distribution(study: StudyData, path: str, title: str | None) -> PlotDescription:
    values, unit = _reading(study, path)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise refuse("plot.nothing", RefusalClass.USER, f"{path} has no finite value in any member", "check the members' windows: `observe(..., after=s(0.0))`")
    edges = np.histogram_bin_edges(finite, bins="auto")
    counts, _ = np.histogram(finite, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2.0
    notes = [STUDY_ESTIMATES, f"{len(finite)} of {len(values)} members have a value; the bins are numpy's 'auto' rule"]
    panel = Panel(path, label(path, unit), "members", (Trace(path, centres, counts.astype(np.float64), "bars",
                                                             width=float(edges[1] - edges[0])),))
    return PlotDescription("distribution", title or f"{study.experiment}: {path} over the members", (panel,), tuple(notes))


def describe_study(study: StudyData, outputs: Sequence[str], *, kind: str | None = None, against: str | None = None,
                   members: Sequence[int] | None = None, title: str | None = None) -> PlotDescription:
    """A study's output as one of its figures: ``curve`` (an output at the end against a coordinate: the step-out curve),
    ``overlay`` (a recorded series of the members on one time axis) or ``distribution`` (an output at the end over
    members drawn from a distribution). The default is by the output: a series is overlaid, an output at the end is a
    curve."""
    if len(outputs) != 1:
        raise refuse("plot.output", RefusalClass.USER, f"a study plot draws one output, and {len(outputs)} were named",
                     "plot one: result.plot('swim_speed:umr.mean_speed')")
    path = outputs[0]
    if not len(study):
        raise refuse("plot.nothing", RefusalClass.USER, "the study has no members to draw", "run it first: study.run_to(folder)")
    output_kind = kind_of(study, path)
    if kind is None:
        kind = "overlay" if output_kind == "per_step" else "curve"
    if kind not in KINDS or kind == "series":
        raise refuse("plot.kind", RefusalClass.USER, f"a study plot is a 'curve', an 'overlay' or a 'distribution', not {kind!r}",
                     "plot(..., kind='curve')")
    if kind == "overlay":
        return _overlay(study, path, members, title)
    if kind == "distribution":
        return _distribution(study, path, title)
    return _curve(study, path, against, title)


__all__ = ["BAND_TEXT", "ESTIMATES", "MEASURED", "KINDS", "PLOT_FORMAT", "STUDY_ESTIMATES", "Band", "Panel", "PlotDescription", "Span",
           "Trace", "describe_run", "describe_study", "estimates_note", "kind_of", "label", "member_flags", "run_spans", "runs_of"]
