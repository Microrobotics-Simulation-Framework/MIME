"""A study's members against one named reference: each declared output's miss (the largest absolute miss at the end
and over a window), held to a tolerance where one is stated.

**The tolerance is never a fixed absolute.** Between members of one study, with none named, it is the study's own rule
(S2b-1's, a measurement of members against their own runs while a body is locked): :data:`TOLERANCE_EPS` machine epsilons
of the output's float type times the reference's largest magnitude over its run; a trajectory is held to it only while
neither the member nor the reference raised a flag (a flag marks the point where a pair left the declared regime, and
past it outputs are compared, not trajectories). Against anything else (a run of another study, or a member with a
tolerance named) the verdict is the stated tolerance's: a quantity per output, a spec's, and where none is stated the miss
is reported and not judged.

**What a miss says.** Against another run of the model it estimates how two runs of the model differ; it bounds the
model's error from above only against a measurement whose own error is known, and a comparison here names none. The
description of its figure says so.

A comparison is accepted across two studies whose members have the same declared outputs (path, kind and unit), since a
study's two arms may be two experiments; where both kept stage samples, the same bound things."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._data import ResultData, SeriesData, StudyData
from mime.results._plot import Band, Panel, PlotDescription, Trace, label
from mime.results._series import Table
from mime.units import TIME, Quantity, si

if TYPE_CHECKING:
    from mime.results._result import RunResult

#: The study's rule: machine epsilons of the output's float type, times the reference's largest magnitude over its run.
#: S2b-1's committed predictions (members against their own runs, locked: measured 1 to 66 eps, so these are three times
#: that), declared for this use and not fitted to it.
TOLERANCE_EPS: Mapping[str, float] = {"float32": 150.0, "float64": 300.0}
STATEMENT = ("A miss against another run of the model estimates how two runs differ. It bounds the model's error from "
             "above only against a measurement whose own error is known, and this comparison names none.")


@dataclass(frozen=True)
class Miss:
    """One member's miss on one output: ``end`` is member minus reference at the last common time (signed: of the
    component with the largest magnitude), ``largest`` the largest absolute miss over the window and ``at`` the time it
    is (s; ``None`` for an output at the end). ``tolerance`` is the bound the verdict used (``None`` if none), and
    ``verdict`` is ``within``, ``exceeds`` or ``not judged`` (with ``why``)."""

    output: str
    unit: str
    member: int
    end: float
    largest: float
    at: float | None
    tolerance: float | None
    verdict: str
    why: str
    #: How the member was put on the reference's time base: ``none: the samples coincide`` or the interpolation used.
    base: str = ""


@dataclass(frozen=True)
class Curve:
    """The arrays behind a series' miss: the common times, the reference's and the member's values there, and the miss."""

    output: str
    member: int
    time: NDArray[np.float64]
    reference: NDArray[np.float64]
    values: NDArray[np.float64]
    tolerance: float | None
    amount: NDArray[np.float64] | None = None
    uncertainty: str | None = None

    @property
    def miss(self) -> NDArray[np.float64]:
        return self.values - self.reference


@dataclass(frozen=True)
class Comparison:
    """Every member of a study against one reference, on the outputs both declare: :attr:`rows` (a :class:`Miss` for each
    member and output), :meth:`table`, :meth:`text` and the figure (:attr:`description`, :meth:`plot`,
    :meth:`plot_interactive`): the members overlaid on the reference, and a panel for the miss."""

    reference: str
    rows: tuple[Miss, ...]
    rule: str
    notes: tuple[str, ...] = ()
    curves: tuple[Curve, ...] = field(default=(), repr=False)

    def of(self, member: int, output: str | None = None) -> Miss:
        found = [row for row in self.rows if row.member == member and (output is None or row.output == output)]
        if len(found) != 1:
            raise ArgumentRefused([Refusal(
                "compare.row", RefusalClass.USER,
                f"{len(found)} rows of member {member}" + (f" for {output}" if output else "") + " (one is asked for)",
                "name the output: comparison.of(member, 'swim_speed:umr.speed'); rows lists them all")])
        return found[0]

    def table(self) -> Table:
        columns: dict[str, NDArray[Any]] = {
            "output": np.asarray([r.output for r in self.rows]), "member": np.asarray([r.member for r in self.rows]),
            "end miss": np.asarray([r.end for r in self.rows]), "largest miss": np.asarray([r.largest for r in self.rows]),
            "at [s]": np.asarray([np.nan if r.at is None else r.at for r in self.rows]),
            "tolerance": np.asarray([np.nan if r.tolerance is None else r.tolerance for r in self.rows]),
            "verdict": np.asarray([r.verdict for r in self.rows]), "unit": np.asarray([r.unit for r in self.rows])}
        return Table(columns, {"output": "", "member": "", "end miss": "", "largest miss": "", "at [s]": "s",
                               "tolerance": "", "verdict": "", "unit": ""})

    def text(self) -> str:
        lines = [f"comparison against {self.reference}: {self.rule}"]
        for row in self.rows:
            bound = "no tolerance" if row.tolerance is None else f"tolerance {row.tolerance:.3g} {row.unit}"
            lines.append(f"  member {row.member}, {row.output}: end miss {row.end:+.6g} {row.unit}, largest "
                         f"{row.largest:.6g} {row.unit}" + (f" at {row.at:.6g} s" if row.at is not None else "")
                         + f"; {bound}: {row.verdict}" + (f" ({row.why})" if row.why else "")
                         + (f"; time base: {row.base}" if row.base else ""))
        lines += [f"  {note}" for note in self.notes]
        lines.append(f"  {STATEMENT}")
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.text()

    @property
    def description(self) -> PlotDescription:
        """The members overlaid on the reference for each series compared, with a panel for the miss and the tolerance as
        its band: backend-neutral, as every figure's."""
        if not self.curves:
            raise ArgumentRefused([Refusal(
                "compare.no-series", RefusalClass.USER, "no output compared is a recorded series, so there is nothing to overlay",
                "record one in the sweep: Study(exp).sweep(..., record=[swim.speed])")])
        panels: list[Panel] = []
        for output in dict.fromkeys(curve.output for curve in self.curves):
            own = [c for c in self.curves if c.output == output]
            unit = next(r.unit for r in self.rows if r.output == output)
            first = own[0]
            traces = [Trace(f"reference ({self.reference})", first.time, first.reference, "line")]
            traces += [Trace(f"member {c.member}", c.time, c.values, "line", member=c.member, amount=c.amount,
                             uncertainty=c.uncertainty) for c in own]
            panels.append(Panel(output, label("time", "s"), label(output, unit), tuple(traces)))
            bands = tuple(Band(f"member {c.member}: tolerance", -c.tolerance, c.tolerance, self.rule)
                          for c in own if c.tolerance is not None)
            panels.append(Panel(f"{output}: miss", label("time", "s"), label(f"{output} miss", unit),
                                tuple(Trace(f"member {c.member} - reference", c.time, c.miss, "line", member=c.member)
                                      for c in own), (), bands))
        bases = sorted({r.base for r in self.rows if r.base})
        return PlotDescription("comparison", f"members against {self.reference}", tuple(panels),
                               (STATEMENT, *(f"time base: {b}" for b in bases), *self.notes))

    def plot(self, **options: Any) -> Any:
        """The comparison drawn with matplotlib (the ``plot`` extra): members overlaid, a panel for the miss."""
        from mime.results._draw import FigureOptions, draw_matplotlib

        return draw_matplotlib(self.description, FigureOptions.from_dict({**FigureOptions().to_dict(), **options}))

    def plot_interactive(self, **options: Any) -> Any:
        """The comparison drawn with plotly (the ``plot`` extra)."""
        from mime.results._draw import FigureOptions, draw_plotly

        return draw_plotly(self.description, FigureOptions.from_dict({**FigureOptions().to_dict(), **options}))


# ---------------------------------------------------------------------------------------------------------------

INTERPOLATION = "the member linearly interpolated onto the reference's times"
COINCIDENT = "none: the member has a sample at every one of the reference's times"


def _aligned(member: SeriesData, reference: SeriesData, window: tuple[float, float] | None
             ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None, str]:
    """The reference's times inside the window and inside the member's span, the reference's values there, and the member's
    on the reference's time base: its own where it has a sample at the time, else linearly interpolated (the base is
    stated: the reference's, and which interpolation)."""
    tm, tr = np.asarray(member.time, dtype=np.float64), np.asarray(reference.time, dtype=np.float64)
    vm = np.asarray(member.values, dtype=np.float64).reshape(len(tm), -1)
    vr = np.asarray(reference.values, dtype=np.float64).reshape(len(tr), -1)
    if vm.shape[1] != vr.shape[1]:
        raise ArgumentRefused([Refusal("compare.shape", RefusalClass.USER,
                                       f"{member.path} has {vm.shape[1]} components in the member and {vr.shape[1]} in the reference",
                                       "compare outputs of the same shape")])
    keep = (tr >= tm.min() * (1 - 1e-12)) & (tr <= tm.max() * (1 + 1e-12))
    if window is not None:
        keep &= (tr > window[0]) & (tr <= window[1] * (1 + 1e-9))
    times, theirs = tr[keep], vr[keep]
    on_grid = bool(np.isin(_times_key(times), _times_key(tm)).all()) if len(times) else True
    mine = np.stack([np.interp(times, tm, vm[:, k]) for k in range(vm.shape[1])], axis=1) if len(times) else vm[:0]
    if on_grid and len(times):
        _, im, _ = np.intersect1d(_times_key(tm), _times_key(times), return_indices=True)
        mine = vm[im]
    amount = None
    if member.uncertainty is not None and len(times):
        a = np.asarray(member.uncertainty.amount, dtype=np.float64).reshape(len(tm), -1)
        amount = np.stack([np.interp(times, tm, a[:, k]) for k in range(a.shape[1])], axis=1)[:, 0]
    return times, theirs, mine, amount, COINCIDENT if on_grid else INTERPOLATION


def _times_key(times: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.round(times * 1e9).astype(np.int64)


def _signed(miss: NDArray[np.float64]) -> float:
    flat = np.asarray(miss, dtype=np.float64).ravel()
    return float(flat[np.argmax(np.abs(flat))]) if flat.size else float("nan")


def _stated(tolerance: Mapping[str, Quantity] | Quantity | None, path: str, dimension: Any) -> float | None:
    if tolerance is None or dimension is None:
        return None
    held = tolerance.get(path) if isinstance(tolerance, Mapping) else tolerance
    if held is None or held.dimension != dimension:
        return None
    return float(cast(float, si(held, dimension, what=f"the tolerance of {path}")))


def _rule_bound(values: NDArray[Any], scale: float) -> float:
    """The study's rule for an output of ``values``' float type: the multiple of epsilon, times the reference's scale."""
    kind = np.asarray(values).dtype
    name = kind.name if kind.name in TOLERANCE_EPS else "float64"
    return TOLERANCE_EPS[name] * float(np.finfo(kind if kind.kind == "f" else np.float64).eps) * scale


def check_time_base(reference: ResultData, member: ResultData, extra: Sequence[tuple[float, str]]) -> None:
    """Refuses (``compare.time-base``) a reference whose samples are as far apart as half the period of a declared drive's
    frequency or farther: on its base the comparison cannot resolve that frequency (the aliasing rule's condition), and
    would call a coarse sampling of a fast motion a match or a miss it is neither."""
    rates = [*reference.drives, *member.drives, *extra]
    if not rates:
        return
    fastest = max(rates)
    hertz = fastest[0] / (2.0 * math.pi)
    for series in reference.series_data():
        gap = float(np.max(np.diff(series.time))) if len(series.time) > 1 else 0.0
        if hertz > 0.0 and gap >= 0.5 / hertz * (1.0 - 1e-9):
            raise ArgumentRefused([Refusal(
                "compare.time-base", RefusalClass.USER,
                f"the reference's {series.path} is sampled every {gap:g} s, and {fastest[1]}: {hertz:g} Hz, a period of "
                f"{1.0 / hertz:g} s that is no more than twice that spacing. The comparison is made on the reference's "
                f"time base, which cannot resolve that frequency",
                f"compare against a reference sampled at least twice as often ({0.5 / hertz:g} s apart or less), or compare an "
                f"output that does not move at the drive's frequency (a reading at the end)",
                why="a coarse reference shows a fast motion as a slower one")])


def compare(study: StudyData, reference: int | ResultData | RunResult = 0, *,
            window: tuple[Quantity, Quantity] | None = None, tolerance: Mapping[str, Quantity] | Quantity | None = None,
            drives: Sequence[Quantity] | None = None) -> Comparison:
    """See :meth:`StudyResult.compare`."""
    from mime.results._data import RunData
    from mime.results._result import RunResult

    inside = isinstance(reference, int) and not isinstance(reference, bool)
    count = len(study)
    if inside:
        if not -count <= cast(int, reference) < count:
            raise ArgumentRefused([Refusal(
                "compare.reference", RefusalClass.USER,
                f"the reference is a member number 0 to {count - 1} of this study, or a result, not {reference!r}",
                "name it: result.compare(reference=0), or result.compare(reference=other_study.member(0))")])
        ref_index = cast(int, reference) % count
        ref: ResultData = study.member_data(ref_index)
        name = f"member {ref_index}"
    else:
        ref_index = -1
        ref = RunData(reference) if isinstance(reference, RunResult) else cast(ResultData, reference)
        name = f"the result {ref.name!r}"
    stated = [(2.0 * math.pi * float(cast(float, d.value)), f"a declared drive of {float(cast(float, d.value)):g} Hz")
              for d in (drives or ())]
    seconds = None if window is None else (float(cast(float, si(window[0], TIME, what="the window's start"))),
                                           float(cast(float, si(window[1], TIME, what="the window's end"))))
    notes: list[str] = []
    rows: list[Miss] = []
    curves: list[Curve] = []
    for index in (i for i in range(count) if i != ref_index):
        member = study.member_data(index)
        _same_declared_outputs(member, ref, name)
        check_time_base(ref, member, stated)
        raised = any(f.was_raised is True for f in (*member.flags_data(), *ref.flags_data()))
        mine_series = {s.path: s for s in member.series_data()}
        ref_series = {s.path: s for s in ref.series_data()}
        ref_readings = {r.path: r for r in ref.readings_data()}
        for path, series in mine_series.items():
            if path in ref_series:
                row, curve = _series_miss(series, ref_series[path], index, seconds, inside, tolerance, raised)
                rows.append(row)
                curves.append(curve)
        for reading in member.readings_data():
            if reading.path in ref_readings:
                rows.append(_reading_miss(reading, ref_readings[reading.path], index, inside, tolerance))
        notes += [f"member {index}: {p} is a per-step output not recorded in both, so it is not compared"
                  for p in sorted({o[0] for o in member.declared() if o[1] == "per_step"} - mine_series.keys() | set())
                  if p in ref_series]
    if not rows:
        raise ArgumentRefused([Refusal("compare.nothing", RefusalClass.USER,
                                       "the members and the reference compare on no output", "give the study more than one member")])
    rule = (f"between members of one study, {TOLERANCE_EPS['float32']:g} (float32) or {TOLERANCE_EPS['float64']:g} (float64) "
            f"machine epsilons of the output's largest magnitude over the reference's run (the study's rule), a trajectory "
            f"held to it only while no flag is raised; a tolerance stated by the caller otherwise, and none means not judged"
            if inside and tolerance is None else "the tolerance stated by the caller, per output; an output with none is not judged")
    return Comparison(name, tuple(rows), rule, tuple(dict.fromkeys(notes)), tuple(curves))


def _same_declared_outputs(member: ResultData, ref: ResultData, name: str) -> None:
    """The member and the reference declare the same outputs (path, kind, unit): the condition for comparing two arms."""
    mine, theirs = member.declared(), ref.declared()
    if mine != theirs:
        differ = sorted(f"{p} ({k}, {u})" for p, k, u in mine ^ theirs)
        raise ArgumentRefused([Refusal(
            "compare.outputs", RefusalClass.USER,
            f"the member and {name} do not declare the same outputs: {', '.join(differ[:4])}{' ...' if len(differ) > 4 else ''}",
            "compare results that declare the same things; two arms of one question may be two experiments, but "
            "they declare the same outputs")])


def _verdict(miss: float, bound: float | None, judged: bool, why: str) -> tuple[str, str]:
    if not judged:
        return "not judged", why
    if bound is None:
        return "not judged", "no tolerance is stated"
    return ("within" if miss <= bound else "exceeds"), ""


def _series_miss(mine: SeriesData, theirs: SeriesData, index: int, window: tuple[float, float] | None, inside: bool,
                 tolerance: Mapping[str, Quantity] | Quantity | None, raised: bool) -> tuple[Miss, Curve]:
    times, ref_values, values, amount, base = _aligned(mine, theirs, window)
    if not len(times):
        raise ArgumentRefused([Refusal("compare.window", RefusalClass.USER,
                                       f"member {index}'s {mine.path} and the reference share no time in the window",
                                       "widen the window, or compare results that overlap in time")])
    miss = values - ref_values
    largest = np.abs(miss).max(axis=1)
    at = int(np.argmax(largest))
    bound = _stated(tolerance, mine.path, mine.dimension)
    judged, why = True, ""
    if bound is None and inside and tolerance is None:
        bound = _rule_bound(np.asarray(mine.values), float(np.max(np.abs(np.asarray(theirs.values, dtype=np.float64)))))
        judged, why = not raised, ("a flag is raised in the member or the reference, so the pair left the declared regime "
                                   "and a trajectory is not held to the locked-regime rule")
    verdict, reason = _verdict(float(largest[at]), bound, judged, why)
    row = Miss(mine.path, mine.unit, index, _signed(miss[-1]), float(largest[at]), float(times[at]), bound, verdict, reason, base)
    return row, Curve(mine.path, index, times, ref_values[:, 0], values[:, 0], bound, amount,
                      None if mine.uncertainty is None else mine.uncertainty.kind)


def _reading_miss(mine: Any, theirs: Any, index: int, inside: bool,
                  tolerance: Mapping[str, Quantity] | Quantity | None) -> Miss:
    a = np.atleast_1d(np.asarray(mine.value, dtype=np.float64))
    b = np.atleast_1d(np.asarray(theirs.value, dtype=np.float64))
    miss = a - b
    largest = float(np.abs(miss).max())
    bound = _stated(tolerance, mine.path, mine.dimension)
    if bound is None and inside and tolerance is None:
        bound = _rule_bound(a, float(np.abs(b).max()))
    verdict, reason = _verdict(largest, bound, mine.observed and theirs.observed, "an output one of them did not observe")
    return Miss(mine.path, mine.unit, index, _signed(miss), largest, None, bound, verdict, reason)


__all__ = ["STATEMENT", "TOLERANCE_EPS", "Comparison", "Curve", "Miss", "check_time_base", "compare"]
