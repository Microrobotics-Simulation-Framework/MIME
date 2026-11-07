"""A study's members against one named reference: each declared output's miss (the largest absolute miss at the end
and over a window), held to a tolerance only where one is stated or the pair is a re-run.

**A miss between members at different coordinates is reported and not judged**, unless the caller states a tolerance
(a quantity per output, a spec's). Members at other points of a sweep are meant to differ, so the sweep's own rule, which
is a numerical-equivalence bound for one point computed twice (a batched member against a fresh run of its point:
measured 1 to 66 machine epsilons), would call every pair that is not the same computation "exceeds" and say nothing.
**The rule gives a verdict only where the member and the reference share every coordinate** (a re-run, or the same point in
two environments): :data:`TOLERANCE_EPS` machine epsilons of the output's float type times the reference output's largest
magnitude over its run (a reading's scale is the largest magnitude of the quantity it reduces, where the run recorded
that), and only while both runs are in the locked regime, which is read from what the runs declare: a ``PhaseLag``'s lag
recorded, under 90 degrees at every recorded step. Where the regime cannot be read, or is not locked, there is no verdict
and the result says why. A trajectory is held to the rule only while neither run raised a flag.

**What a miss says.** Against another run of the model it estimates how two runs of the model differ; it bounds the
model's error from above only against a measurement whose own error is known, and a comparison here names none. The
description of its figure says so, and says which kind the reference is.

A comparison is accepted across two studies, since a study's two arms may be two experiments: the members declare every
output the reference does (path, kind and unit), and those are what is compared. The reference may be any result that
reads as the protocol (:mod:`mime.results._data`), a measurement included.

**The time base is the reference's.** A member sampled at other times is put on the reference's: its own sample where it
has one at the reference's time, else linearly interpolated, and the result and the figure say which, with each side's
spacing. The coarser of the two bases decides the refusal: where either is sampled as sparsely as half the period of a drive
either run declares (or the caller names, ``drives=``), the comparison is refused (``compare.time-base``): the aliasing
rule's condition. A result read from a folder keeps no drives, so a comparison of one asks for ``drives=`` and is refused
without it (``compare.drives-unknown``), never silently unchecked."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._data import ReadingData, ResultData, SeriesData, StudyData, drive_rates
from mime.results._plot import Band, Panel, PlotDescription, Trace, label
from mime.results._series import Table
from mime.units import TIME, Quantity, si

if TYPE_CHECKING:
    from mime.results._result import RunResult

#: The sweep's numerical-equivalence rule: machine epsilons of the output's float type, times the reference's largest
#: magnitude over its run. The sweep's committed predictions (members against their own runs, locked: measured 1 to 66 eps,
#: so these are three times that), declared for this use and not fitted to it; it applies to a point against its own re-run.
TOLERANCE_EPS: Mapping[str, float] = {"float32": 150.0, "float64": 300.0}
#: The lag past which a body is not locked (the sweep's gate holds each member's lag behind the field under 90 degrees).
LAG_LIMIT = math.pi / 2.0
STATEMENT = ("A miss against another run of the model estimates how two runs differ. It bounds the model's error from "
             "above only against a measurement whose own error is known, and this comparison names none.")
STATEMENT_MEASURED = ("A miss against a measurement estimates how far the model is from the data. It bounds the model's error "
                      "from above only where the measurement's own error is known, and this comparison does not subtract the "
                      "measurement's uncertainty (a band is drawn where the data gives one).")


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
    #: What a miss bounds and estimates, by what the reference is (another run of the model, or a measurement).
    statement: str = STATEMENT

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
        lines.append(f"  {self.statement}")
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
                               (self.statement, *(f"time base: {b}" for b in bases), *self.notes))

    def plot(self, **options: Any) -> Any:
        """The comparison drawn with matplotlib (the ``plot`` extra): members overlaid, a panel for the miss."""
        from mime.results._draw import FigureOptions, draw_matplotlib

        return draw_matplotlib(self.description, FigureOptions.from_dict({**FigureOptions().to_dict(), **options}))

    def plot_interactive(self, **options: Any) -> Any:
        """The comparison drawn with plotly (the ``plot`` extra)."""
        from mime.results._draw import FigureOptions, draw_plotly

        return draw_plotly(self.description, FigureOptions.from_dict({**FigureOptions().to_dict(), **options}))


# ---------------------------------------------------------------------------------------------------------------

COINCIDENT = "none: the member has a sample at every one of the reference's times"


def _gap(time: NDArray[np.float64]) -> float:
    return float(np.max(np.diff(time))) if len(time) > 1 else 0.0


def _interpolation(member_gap: float, reference_gap: float) -> str:
    return (f"the member linearly interpolated onto the reference's times (the member's samples at most {member_gap:g} s apart, "
            f"the reference's at most {reference_gap:g} s apart)")


def _aligned(member: SeriesData, reference: SeriesData, window: tuple[float, float] | None
             ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None, str]:
    """The reference's times inside the window and inside the member's span, the reference's values there, and the member's
    on the reference's time base: its own where it has a sample at the time, else linearly interpolated (the base is
    stated: the reference's, which interpolation and each side's spacing)."""
    tm, tr = np.asarray(member.time, dtype=np.float64), np.asarray(reference.time, dtype=np.float64)
    vm = np.asarray(member.values, dtype=np.float64).reshape(len(tm), -1)
    vr = np.asarray(reference.values, dtype=np.float64).reshape(len(tr), -1)
    if vm.shape[1] != vr.shape[1]:
        raise ArgumentRefused([Refusal("compare.shape", RefusalClass.USER,
                                       f"{member.path} has {vm.shape[1]} components in the member and {vr.shape[1]} in the reference",
                                       "compare outputs of the same shape: `result.compare(reference)`")])
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
    return times, theirs, mine, amount, COINCIDENT if on_grid else _interpolation(_gap(tm), _gap(tr))


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


def _rule_bound(precision: str, scale: float) -> float:
    """The sweep's rule for an output computed in ``precision`` (``float32`` or ``float64``): the multiple of epsilon, times
    the reference's scale over its run."""
    name = precision if precision in TOLERANCE_EPS else "float64"
    return TOLERANCE_EPS[name] * float(np.finfo(np.dtype(name)).eps) * scale


def _precision(values: NDArray[Any]) -> str:
    kind = np.asarray(values).dtype
    return kind.name if kind.kind == "f" else "float64"


# ---------------------------------------------------------------------------------------------------------------
# Where the sweep's rule applies


@dataclass(frozen=True)
class Basis:
    """Whether the sweep's numerical-equivalence rule applies to one member against the reference, and if not, why."""

    applies: bool
    why: str = ""


def _regime(data: ResultData) -> tuple[bool | None, str]:
    """Whether ``data`` is in the locked regime, read from what it declares: ``True`` where a ``PhaseLag``'s lag was recorded
    and stays under :data:`LAG_LIMIT` (90 degrees) at every recorded step, ``False`` where it does not, ``None`` where no lag
    was recorded (or it is not finite)."""
    lags = [s for s in data.series_data() if s.path.startswith("phase_lag:") and s.path.endswith(".lag")]
    if not lags:
        return None, "records no PhaseLag lag, so the regime cannot be read"
    worst = max(float(np.max(np.abs(np.asarray(s.values, dtype=np.float64)))) if np.size(s.values) else float("nan") for s in lags)
    if not math.isfinite(worst):
        return None, "records a lag that is not finite, so the regime cannot be read"
    return worst < LAG_LIMIT, f"its lag reaches {worst:.3g} rad" + ("" if worst < LAG_LIMIT else f", past {LAG_LIMIT:.3g} rad (90 degrees)")


def _basis(member: ResultData, reference: ResultData, member_point: Mapping[str, object],
           reference_point: Mapping[str, object] | None) -> Basis:
    from mime.study._axes import differ as values_differ
    from mime.study._axes import value_text

    if reference_point is None or not member_point:
        return Basis(False, "the reference's coordinates are not known (a measurement, or a run that is no member of a study), "
                            "so the pair is not a re-run and a miss between them is a physical difference: state a tolerance")
    differ = [f"{path}: {value_text(member_point.get(path))} against {value_text(reference_point.get(path))}"
              for path in dict.fromkeys([*member_point, *reference_point])
              if path not in member_point or path not in reference_point or values_differ(member_point[path], reference_point[path])]
    if differ:
        return Basis(False, f"the member and the reference are at different coordinates ({'; '.join(differ[:3])}), so their miss "
                            f"is a physical difference and the sweep's numerical-equivalence rule (a point against its own "
                            f"re-run) does not apply: state a tolerance")
    states = [(who, *_regime(data)) for who, data in (("the member", member), ("the reference", reference))]
    unread = [f"{who} {why}" for who, locked, why in states if locked is None]
    if unread:
        return Basis(False, "the regime cannot be read: " + "; ".join(unread) + " (declare a PhaseLag and record its lag)")
    out = [f"{who}'s regime is not locked ({why})" for who, locked, why in states if locked is False]
    if out:
        return Basis(False, "; ".join(out) + ": the rule is the locked regime's")
    return Basis(True)


# ---------------------------------------------------------------------------------------------------------------
# The refusals


def _spacing_text(time: NDArray[np.float64]) -> tuple[float, str]:
    """The largest gap of a time base, and how to say it: every so many seconds for an even base, a gap of so many for a base
    with a hole in it."""
    gap = _gap(time)
    steps = np.diff(time)
    if len(steps) > 1 and gap > 1.5 * float(np.median(steps)):
        return gap, f"has a gap of {gap:g} s between two samples"
    return gap, f"is sampled every {gap:g} s"


def check_time_base(reference: ResultData, member: ResultData, extra: Sequence[tuple[float, str]], *,
                    inside: bool = False) -> None:
    """Refuses (``compare.time-base``) a pair of which either base is as coarse as half the period of a declared drive's
    frequency or coarser, on any output both hold: the comparison is made on the reference's base with the member put on
    it, and a coarse sampling of a fast motion (on either side) would be called a match or a miss it is neither. The coarser
    of the two decides, and the refusal names which side it is."""
    rates = [*reference.drives, *member.drives, *extra]
    if not rates:
        return
    fastest = max(rates)
    hertz = fastest[0] / (2.0 * math.pi)
    if not hertz > 0.0:
        return
    mine = {series.path: series for series in member.series_data()}
    for series in reference.series_data():
        other = mine.get(series.path)
        if other is None:
            continue
        sides = [("reference", series), ("member", other)]
        side, coarser = max(sides, key=lambda pair: _gap(pair[1].time))
        gap, phrase = _spacing_text(coarser.time)
        if gap >= 0.5 / hertz * (1.0 - 1e-9):
            if inside or side == "member":
                who = "the study's members" if inside else "the member"
                route = (f"record {who} at most {0.5 / hertz:g} s apart (the sweep's every=, `Study(exp).sweep(..., every=ms(...))`), ")
            else:
                route = (f"compare against a reference sampled at least twice as often ({0.5 / hertz:g} s apart or less), "
                         f"`result.compare(reference)`, ")
            raise ArgumentRefused([Refusal(
                "compare.time-base", RefusalClass.USER,
                f"the {side}'s {series.path} {phrase}, and {fastest[1]} ({hertz:g} Hz), whose period "
                f"{1.0 / hertz:g} s is no more than twice that spacing. The comparison is made on the reference's "
                f"time base with the coarser of the two deciding, and the {side}'s cannot resolve that frequency",
                route + "or compare an output that does not move at the drive's frequency (a reading at the end)",
                why="a coarse sampling shows a fast motion as a slower one")])


def _drives_known(member: ResultData, reference: ResultData, name: str) -> None:
    unknown = [who for who, data in (("the member", member), (name, reference)) if not data.drives_known]
    if unknown:
        raise ArgumentRefused([Refusal(
            "compare.drives-unknown", RefusalClass.USER,
            f"{' and '.join(unknown)} was read from a folder, which does not keep the drive frequencies its build declared, so "
            f"the time-base rule cannot be checked and the comparison would pass it silently",
            "state the drives the runs were turned at: `result.compare(reference, drives=[Hz(50.0)])` (`drives=[]` if none "
            "turned a field)")])


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

    def point_of(index: int) -> dict[str, object]:
        return {path: study.coordinate(path, index) for path in study.axes}

    reference_point = point_of(ref_index) if inside else (None if ref.point is None else dict(ref.point))
    stated = list(drive_rates(drives, "drives")) if drives is not None else []
    seconds = None if window is None else (float(cast(float, si(window[0], TIME, what="the window's start"))),
                                           float(cast(float, si(window[1], TIME, what="the window's end"))))
    notes: list[str] = []
    rows: list[Miss] = []
    curves: list[Curve] = []
    for index in (i for i in range(count) if i != ref_index):
        member = study.member_data(index)
        _same_declared_outputs(member, ref, name)
        if drives is None:
            _drives_known(member, ref, name)
        check_time_base(ref, member, stated, inside=inside)
        basis = _basis(member, ref, point_of(index), reference_point)
        raised = any(f.was_raised is True for f in (*member.flags_data(), *ref.flags_data()))
        mine_series = {s.path: s for s in member.series_data()}
        ref_series = {s.path: s for s in ref.series_data()}
        ref_readings = {r.path: r for r in ref.readings_data()}
        for path, series in mine_series.items():
            if path in ref_series:
                row, curve = _series_miss(series, ref_series[path], index, seconds, tolerance, raised, basis)
                rows.append(row)
                curves.append(curve)
                width = int(np.prod(np.shape(series.values)[1:]))
                if width > 1:
                    notes.append(f"member {index}: {path} has {width} components; the figure draws the first, and the table's "
                                 f"miss is the largest over all of them")
        for reading in member.readings_data():
            if reading.path in ref_readings:
                rows.append(_reading_miss(reading, ref_readings[reading.path], index, tolerance, basis))
        notes += [f"member {index}: {p} is a per-step output not recorded in both, so it is not compared"
                  for p in sorted(path for path, kind, _ in member.declared()
                                  if kind == "per_step" and (path not in mine_series or path not in ref_series))]
    if not rows:
        raise ArgumentRefused([Refusal("compare.nothing", RefusalClass.USER,
                                       "the members and the reference compare on no output", "give the study more than one member: `Study(exp).sweep(grid({...: [a, b]}), duration=...)`")])
    rule = (("the tolerance stated by the caller for an output; " if tolerance is not None else "no tolerance stated; ")
            + "a miss without one is reported and not judged, except for a member at the reference's own coordinates (a re-run, "
            f"or the same point in two environments) with both runs in the locked regime (a PhaseLag lag under 90 degrees at "
            f"every recorded step): the sweep's numerical-equivalence rule, {TOLERANCE_EPS['float32']:g} (float32) or "
            f"{TOLERANCE_EPS['float64']:g} (float64) machine epsilons of the output's largest magnitude over the reference's "
            f"run, a trajectory held to it only while no flag is raised")
    return Comparison(name, tuple(rows), rule, tuple(dict.fromkeys(notes)), tuple(curves),
                      STATEMENT if ref.provenance == "simulated" else STATEMENT_MEASURED)


def _same_declared_outputs(member: ResultData, ref: ResultData, name: str) -> None:
    """Every output the reference declares is declared by the member, with the same kind and unit: the condition for
    comparing two arms (outputs only the member declares are not compared)."""
    mine, theirs = member.declared(), ref.declared()
    differ = sorted(f"{p} ({k}, {u})" for p, k, u in theirs - mine)
    if differ or not theirs:
        raise ArgumentRefused([Refusal(
            "compare.outputs", RefusalClass.USER,
            f"the member does not declare what {name} declares, with the same kind and unit: "
            f"{', '.join(differ[:4]) or 'it declares nothing'}{' ...' if len(differ) > 4 else ''}",
            "compare results that declare the same things, `result.compare(reference)`; two arms of one question may be "
            "two experiments, but the member declares every output the reference does")])


def _verdict(miss: float, bound: float | None, judged: bool, why: str) -> tuple[str, str]:
    if not judged:
        return "not judged", why
    if bound is None:
        return "not judged", "no tolerance is stated"
    return ("within" if miss <= bound else "exceeds"), ""


def _series_miss(mine: SeriesData, theirs: SeriesData, index: int, window: tuple[float, float] | None,
                 tolerance: Mapping[str, Quantity] | Quantity | None, raised: bool, basis: Basis) -> tuple[Miss, Curve]:
    times, ref_values, values, amount, base = _aligned(mine, theirs, window)
    if not len(times):
        raise ArgumentRefused([Refusal("compare.window", RefusalClass.USER,
                                       f"member {index}'s {mine.path} and the reference share no time in the window",
                                       "widen the window, `window=(s(0.0), s(0.05))`, or compare results that overlap in time")])
    miss = values - ref_values
    largest = np.abs(miss).max(axis=1)
    at = int(np.argmax(largest))
    bound = _stated(tolerance, mine.path, mine.dimension)
    judged, why = True, ""
    if bound is None:
        judged, why = False, basis.why or "no tolerance is stated"
        if basis.applies:
            bound = _rule_bound(_precision(mine.values), float(np.max(np.abs(np.asarray(theirs.values, dtype=np.float64)))))
            judged, why = not raised, ("a flag is raised in the member or the reference, so the pair left the declared regime "
                                       "and a trajectory is not held to the locked-regime rule")
    verdict, reason = _verdict(float(largest[at]), bound, judged, why)
    row = Miss(mine.path, mine.unit, index, _signed(miss[-1]), float(largest[at]), float(times[at]), bound, verdict, reason, base)
    return row, Curve(mine.path, index, times, ref_values[:, 0], values[:, 0], bound, amount,
                      None if mine.uncertainty is None else mine.uncertainty.kind)


def _reading_miss(mine: ReadingData, theirs: ReadingData, index: int,
                  tolerance: Mapping[str, Quantity] | Quantity | None, basis: Basis) -> Miss:
    a = np.atleast_1d(np.asarray(mine.value, dtype=np.float64))
    b = np.atleast_1d(np.asarray(theirs.value, dtype=np.float64))
    miss = a - b
    largest = float(np.abs(miss).max())
    bound = _stated(tolerance, mine.path, mine.dimension)
    judged, why = mine.observed and theirs.observed, "an output one of them did not observe"
    if bound is None and judged:
        judged, why = False, basis.why or "no tolerance is stated"
        if basis.applies:
            if theirs.scale is None:
                why = ("the reading's scale over the run is not kept (the run did not record the quantity it reduces), so the "
                       "rule has no scale to judge it on")
            else:
                bound, judged, why = _rule_bound(mine.precision, theirs.scale), True, ""
    verdict, reason = _verdict(largest, bound, judged, why)
    return Miss(mine.path, mine.unit, index, _signed(miss), largest, None, bound, verdict, reason)


__all__ = ["STATEMENT", "STATEMENT_MEASURED", "TOLERANCE_EPS", "Comparison", "Curve", "Miss", "check_time_base", "compare"]
