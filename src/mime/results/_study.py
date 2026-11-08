"""A study's result: one member axis, a coordinate column per axis with its units, the group each member ran in, each
declared output stacked over the members, each member as a :class:`RunResult`, and the members' flags as one union;
its long table, CSV, NPZ and ``.mat``, its manifest (the study folder's ``study.json``), the order of convergence of an
output over three levels, and reading a study folder back, a partial one included."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.errors import public
from mime.results._result import MemberOf, RunResult
from mime.results._series import Reading, Series, Table, column_name
from mime.units import Quantity, unit_symbol

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MatplotlibFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from mime.drives import Ramp
    from mime.results._compare import Comparison
    from mime.results._data import ResultData
    from mime.results._figure import FigureRecipe
    from mime.results._result import OutputRef, RunResult
    from mime.results._order import Order
    from mime.study._protocol import Protocol
    from mime.study._study import StudyPlan

#: What a member's flag reads where the member's build declares no such flag: not "not raised".
NOT_DECLARED = "not declared"


@dataclass(frozen=True)
class StackedSeries:
    """A per-step output stacked over a study's members, which share its time axis: ``time`` (samples, s) and
    ``values`` (members first, then samples, then the output's own shape), both array quantities."""

    path: str
    time: Quantity
    values: Quantity


@dataclass(frozen=True)
class Dropped:
    """A point the study did not run (``drop_refused=True``): its index, its coordinates as text, and its refusals,
    each ``[code] message``."""

    index: int
    coordinates: str
    refusals: tuple[str, ...]


def _key(value: object) -> str:
    """A coordinate value as text that is equal for equal values: a quantity by its SI value and unit."""
    if isinstance(value, Quantity):
        return f"{np.asarray(value.value, dtype=np.float64).tolist()!r} {unit_symbol(value.dimension)}"
    return repr(value)


def member_entries(point: int, member: RunResult) -> dict[str, NDArray[Any]]:
    """One member's arrays for an NPZ of a study, under ``m<point>/``: the arrays of its run's NPZ, its manifest, and
    its build report's summary (both JSON text)."""
    prefix = f"m{point}/"
    found = {prefix + key: value for key, value in member.arrays().items()}
    found[prefix + "manifest"] = np.asarray(json.dumps(member.manifest(), sort_keys=True))
    if member.report is not None:
        found[prefix + "report"] = np.asarray(json.dumps(member.report, sort_keys=True))
    if member.stage is not None:
        found.update({prefix + key: value for key, value in member.stage.arrays().items()})
    return found


class StudyOutputs:
    """One observation's outputs over a study's members (``result[swim]``): each by its name, stacked."""

    def __init__(self, result: StudyResult, id: str) -> None:
        self._result, self.id = result, id

    def __getitem__(self, name: str) -> Quantity | StackedSeries:
        return self._result.output(self.id, name)

    def __getattr__(self, name: str) -> Quantity | StackedSeries:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


@dataclass(frozen=True)
class StudyGrid:
    """A study over a full product of axes, reshaped: ``axes`` (the paths, in the order of the sweep), ``values`` (each
    axis's distinct values, in the order they first appear), ``shape`` (one length per axis) and ``index`` (an integer
    array of that shape: the member at each combination, ``-1`` where the study dropped it)."""

    axes: tuple[str, ...]
    values: Mapping[str, Quantity | tuple[object, ...]]
    shape: tuple[int, ...]
    index: NDArray[np.int64]

    def reshape(self, per_member: Quantity | NDArray[Any]) -> NDArray[Any]:
        """An array with the member axis first (``result[swim].mean_speed.value``, a coordinate, a flag column) as an
        array of ``shape`` and then its own shape, NaN where a combination was dropped."""
        held = np.asarray(per_member.value if isinstance(per_member, Quantity) else per_member, dtype=np.float64)
        out = np.full((*self.shape, *held.shape[1:]), np.nan)
        for at in np.ndindex(*self.shape):
            if self.index[at] >= 0:
                out[at] = held[self.index[at]]
        return out


class StudyResult:
    """A study's result: ``len(result)`` members, in the order of the points that ran; ``result.coordinates[path]``,
    each axis's values over the members (an array quantity, members first, or a tuple where an axis's values are not
    quantities); ``result.groups``, the group each member ran in; ``result[swim].mean_speed``, an output at the end over
    the members (an array quantity); ``result[swim].speed``, a recorded series stacked over the members where they share
    its time axis (:class:`StackedSeries`), refused otherwise with the route to :meth:`member`; :meth:`member`, one
    member as a :class:`RunResult`; :attr:`flags`, the union of the members' flags; and :attr:`dropped`, the points
    ``drop_refused=True`` left out, with their refusals. :meth:`grid` reshapes a full product of axes; :meth:`table`,
    :meth:`to_csv`, :meth:`to_npz` and :meth:`to_mat` export; :meth:`manifest` is the study folder's ``study.json``;
    :meth:`order` is an output's observed order of convergence over three levels of an axis; :meth:`read` reads a
    study folder, a partial one too (``missing`` says which members are not there)."""

    def __init__(self, experiment: str, axes: Sequence[str], coordinates: Mapping[str, Sequence[object]],
                 members: Sequence[RunResult | Callable[[], RunResult]], groups: Sequence[int], points: Sequence[int],
                 dropped: Sequence[Dropped] = (), compiles: Sequence[int] = (), *,
                 record: Mapping[str, Any] | None = None, base: Path | None = None, missing: Sequence[int] = (),
                 protocol: object = None, ramps: Sequence[Ramp | None] = ()) -> None:
        from mime.study._protocol import FromRest

        self.experiment = experiment
        self.axes = tuple(axes)
        self._coordinates = {path: tuple(values) for path, values in coordinates.items()}
        self._members: tuple[RunResult | Callable[[], RunResult], ...] = tuple(members)
        #: The group each member ran in, by member.
        self.groups = tuple(groups)
        #: The point each member is, by member: its index among the study's points.
        self.points = tuple(points)
        self.dropped = tuple(dropped)
        #: The XLA programs each group compiles: as JAX's monitoring counted them for a result run in memory, and the
        #: plan's count for one read from a folder (its ``study.json`` holds the plan's count, not a measured one).
        self.compiles = tuple(compiles)
        #: The manifest this result was made with (:meth:`manifest` is it, with the members' statuses).
        self.record: dict[str, Any] = copy.deepcopy(dict(record)) if record is not None else {}
        #: The base experiment's saved folder, where the study is on disk (the bridge ``member(i).as_experiment`` is
        #: based on it).
        self.base = base
        #: The study folder this result was read from (``None`` for one in memory): where a figure's recipe is saved.
        self.folder: Path | None = None
        #: The points a partial study folder lacks: their index among the study's points.
        self.missing = tuple(missing)
        #: How the members' commands moved in time (``mime.study.FromRest`` or ``mime.study.Ramp``), which names the
        #: quantity a step-out read from them is.
        self.protocol: Protocol = cast("Protocol", protocol) if protocol is not None else FromRest()
        #: The drive the protocol gave each member's ramped command (``None`` from rest), by member.
        self.ramps = tuple(ramps) if ramps else (None,) * len(self._members)

    @classmethod
    def of(cls, plan: StudyPlan, members: Sequence[RunResult | None], compiles: Sequence[int], *,
           input_hash: str | None = None, seconds: Sequence[float | None] = ()) -> StudyResult:
        from mime.study._axes import value_text
        from mime.study._record import document_of

        ran = [i for i, member in enumerate(members) if member is not None]
        group_of = {index: number for number, group in enumerate(plan.groups) for index in group.members}
        coordinates = {axis.path: [plan.points[i].values[axis] for i in ran] for axis in plan.axes}
        dropped = [Dropped(point.index, ", ".join(f"{axis.path} = {value_text(point.values[axis])}" for axis in plan.axes),
                           tuple(f"[{r.code}] {r.message}" for r in point.refusals)) for point in plan.refused]
        name = str(plan.groups[0].built.plans.settings.experiment) if plan.groups and plan.groups[0].built else ""
        record = document_of(plan, input_hash, compiles=compiles, seconds=seconds)
        return cls(name, [axis.path for axis in plan.axes], coordinates,
                   [cast(RunResult, members[i]) for i in ran], [group_of[i] for i in ran], ran, dropped, compiles,
                   record=record, protocol=plan.protocol, ramps=[plan.points[i].ramp for i in ran])

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[RunResult]:
        return (self._member(i) for i in range(len(self._members)))

    def _member(self, index: int) -> RunResult:
        held = self._members[index]
        if not isinstance(held, RunResult):
            held = held()
            self._members = (*self._members[:index], held, *self._members[index + 1:])
        if held.member_of is None and self.record:
            from mime.study._axes import STUDY_REASON

            values = tuple((path, self._coordinates[path][index]) for path in self.axes)
            rank = {row["point"]: row["member"] for row in self.record.get("members", [])}
            held.member_of = MemberOf(str(self.record.get("id", "")), rank.get(self.points[index], index),
                                      self.points[index], values, self.base,
                                      {str(axis["path"]): str(axis["reason"]) for axis in self.record.get("axes", [])
                                       if axis.get("reason") and axis["reason"] != STUDY_REASON}, self.protocol)
        return held

    @public
    def member(self, index: int) -> RunResult:
        """Member ``index`` as a :class:`RunResult`, with every accessor a run's result has, and ``as_experiment`` to
        make it an experiment of its own."""
        if not -len(self._members) <= index < len(self._members):
            raise ArgumentRefused([Refusal(
                "study.no-member", RefusalClass.USER, f"the study has {len(self._members)} members, and no member {index}",
                f"take one of its members, `result.member(0)` to `result.member({len(self._members) - 1})`")])
        return self._member(index % len(self._members))

    def coordinate(self, path: str, index: int) -> object:
        """Member ``index``'s value of the axis ``path``, as the sweep gave it."""
        return self._coordinates[path][index]

    @property
    def coordinates(self) -> dict[str, Quantity | tuple[object, ...]]:
        """Each axis's values over the members: an array quantity (members first) where every value is a quantity of
        one dimension, else a tuple."""
        found: dict[str, Quantity | tuple[object, ...]] = {}
        for path, values in self._coordinates.items():
            dimensions = {value.dimension for value in values if isinstance(value, Quantity)}
            if values and len(dimensions) == 1 and all(isinstance(value, Quantity) for value in values):
                quantities = cast(tuple[Quantity, ...], values)
                found[path] = Quantity.of([np.asarray(q.value, dtype=np.float64) for q in quantities],
                                          next(iter(dimensions)))
            else:
                found[path] = values
        return found

    def __getitem__(self, key: object) -> StudyOutputs:
        id = str(getattr(key, "id", key))
        if not self._members or not any(id in member.observations for member in self):
            known = sorted({held for member in self for held in member.observations})
            raise ArgumentRefused([Refusal(
                "study.no-observation", RefusalClass.USER, f"no member of the study observes {id!r}",
                f"index it by the handle `exp.observe(...)` returned, `result[swim]`, or by an id it observes: "
                f"{', '.join(known) or 'none'}")])
        return StudyOutputs(self, id)

    @public
    def output(self, id: str, name: str) -> Quantity | StackedSeries:
        """Observation ``id``'s output ``name`` over the members: what ``result[id].name`` reads."""
        held: list[Reading | Series] = []
        for index, member in enumerate(self):
            if id not in member.observations:
                raise ArgumentRefused([Refusal(
                    "study.no-observation", RefusalClass.USER,
                    f"member {index} (group {self.groups[index]}) does not observe {id!r}, so {id}.{name} has no value "
                    f"for every member", f"read each member alone: result.member({index})")])
            held.append(member[id][name])
        if all(isinstance(each, Reading) for each in held):
            readings = cast(list[Reading], held)
            values = [np.asarray(r.quantity.value, dtype=np.float64) if r.observed else
                      np.full(np.shape(r.quantity.value), np.nan) for r in readings]
            return Quantity.of(values, readings[0].quantity.dimension)
        series = cast(list[Series], held)
        times = [np.asarray(s.time.value) for s in series]
        if any(t.shape != times[0].shape or not np.array_equal(t, times[0]) for t in times[1:]):
            raise ArgumentRefused([Refusal(
                "study.series-time-axes", RefusalClass.USER,
                f"the members' series {id}.{name} do not share one time axis (another dt, or another every=), so they "
                f"are not stacked as one array", "read each member's series alone: result.member(i)[...]." + name)])
        return StackedSeries(f"{id}.{name}", series[0].time,
                             Quantity.of([np.asarray(s.values.value) for s in series], series[0].values.dimension))

    @property
    def flags(self) -> dict[str, tuple[bool | str | None, ...]]:
        """Every flag any member's build declares, with each member's: whether it was raised over its run (``True``,
        ``False``, or ``None`` where the run did not keep enough to say), or ``"not declared"`` where the member's build
        declares no such flag, never "not raised"."""
        declared = [{flag.path for flag in member.layout.flags} for member in self]
        # A run's flags are those raised over it or that it did not keep enough of to say; a flag its build declares
        # and that is not among them was kept and not raised.
        reads = [{read.path: read.was_raised for read in member.flags} for member in self]
        paths = sorted({path for held in declared for path in held})
        return {path: tuple(read.get(path, False) if path in own else NOT_DECLARED
                            for own, read in zip(declared, reads, strict=True)) for path in paths}

    @public
    def write_compare(self, reference: int = 0) -> Path:
        """Write ``compare.usda`` into the study folder: the members' recorded poses as ghosts over the reference member's
        stage, each member under its own prim ``/Compare/<member>`` with a ``visibility`` attribute and ``customData``
        naming it, its coordinates (value and unit) and whether it is the reference, one sublayer that can be muted whole.
        Needs the stage samples the sweep kept (``sweep(..., stage_every=...)``). See :mod:`mime.viz._compare`."""
        from mime.viz._compare import write_compare

        return write_compare(self, reference)

    def member_data(self, index: int) -> ResultData:
        """Member ``index`` as the protocol plots and comparisons read (``mime.results._data``)."""
        from mime.results._data import RunData

        return RunData(self.member(index))

    @public
    def compare(self, reference: int | RunResult | ResultData = 0, *, window: tuple[Quantity, Quantity] | None = None,
                tolerance: Mapping[str, Quantity] | Quantity | None = None,
                drives: Sequence[Quantity] | None = None) -> Comparison:
        """Every other member against ``reference`` (a member's number, or a run: another study's member, since two
        arms of one question may be two experiments that declare the same outputs): each declared output's miss, the
        signed miss at the end and the largest absolute miss over ``window`` (a start and an end; the whole recording
        when not given). ``tolerance`` states a tolerance (a quantity, or one per output path), a spec's, and gives the
        verdict; with none stated the miss is reported and not judged, except for a member at the reference's own
        coordinates (a re-run) with both runs in the locked regime, which the sweep's numerical-equivalence rule judges.
        The comparison is made on the reference's time base (the member's own samples where it has one at the reference's
        time, else linearly interpolated onto it, which the result and its figure say), and refused
        (``compare.time-base``) where the coarser of the two bases is sampled as sparsely as half the period of a drive
        the runs declare, or of one of ``drives`` (frequencies in Hz; a result read from a folder keeps none, so it is
        refused without ``drives=``, ``compare.drives-unknown``). The
        reference may be any result that reads as the protocol (:class:`~mime.results.Measured`). See
        :mod:`mime.results._compare`."""
        from mime.results._compare import compare

        return compare(self, reference, window=window, tolerance=tolerance, drives=drives)

    @public
    def grid(self) -> StudyGrid:
        """The study as a full product of its axes, for reshaping an output into an array with one dimension per axis
        (:meth:`StudyGrid.reshape`). Refused (``study.not-a-grid``) where the members are not every combination of the
        axes' values exactly once, naming the combinations that are not there: samples, ladders and lists of variants
        stay flat."""
        distinct: dict[str, list[object]] = {}
        for path in self.axes:
            seen: list[object] = []
            keys: list[str] = []
            for value in self._coordinates[path]:
                key = _key(value)
                if key not in keys:
                    keys.append(key)
                    seen.append(value)
            distinct[path] = seen
        shape = tuple(len(distinct[path]) for path in self.axes)
        position = {path: [_key(v) for v in distinct[path]] for path in self.axes}
        index = np.full(shape, -1, dtype=np.int64)
        for member in range(len(self)):
            at = tuple(position[path].index(_key(self._coordinates[path][member])) for path in self.axes)
            if index[at] >= 0:
                raise ArgumentRefused([Refusal(
                    "study.not-a-grid", RefusalClass.USER,
                    f"members {int(index[at])} and {member} are at the same combination of the axes "
                    f"({', '.join(f'{p} = {_key(self._coordinates[p][member])}' for p in self.axes)}), so the study is "
                    f"not a product of its axes",
                    "read it flat: result.table(), or result.member(i); a grid is a study over grid({...})")])
            index[at] = member
        absent = [at for at in np.ndindex(*shape) if index[at] < 0]
        dropped = {d.index for d in self.dropped}
        if absent and len(absent) != len(dropped) + len(self.missing):
            shown = ", ".join("(" + ", ".join(f"{p} = {position[p][i]}" for p, i in zip(self.axes, at, strict=True)) + ")"
                              for at in absent[:3])
            raise ArgumentRefused([Refusal(
                "study.not-a-grid", RefusalClass.USER,
                f"the study is not every combination of its axes' values: {len(absent)} combinations are not there "
                f"({shown}{', ...' if len(absent) > 3 else ''})",
                "read it flat: result.table(), or result.member(i); a grid is a study over grid({...})")])
        values: dict[str, Quantity | tuple[object, ...]] = {}
        for path in self.axes:
            held = distinct[path]
            quantities = [v for v in held if isinstance(v, Quantity)]
            same = len(quantities) == len(held) and len({v.dimension for v in quantities}) == 1
            values[path] = Quantity.of([np.asarray(v.value, dtype=np.float64) for v in quantities],
                                       quantities[0].dimension) if same else tuple(held)
        return StudyGrid(self.axes, values, shape, index)

    @public
    def table(self) -> Table:
        """Every member's recorded series, long: one row per member and sample, the coordinate columns first (a
        quantity's with its unit in the header, ``field.commands.frequency [Hz]``), then ``member`` and ``group``, then
        the member's own table's columns (``time [s]``, each series with its unit) and the ``flags`` column. Members
        sampled at other times (another ``dt``, another ``every=``) are in one table, each at its own samples; a column
        a member lacks is NaN there. A study that recorded no series has no table, as a run that recorded none has not."""
        tables = [member.table() for member in self]
        if not tables:
            raise ArgumentRefused([Refusal("study.no-member", RefusalClass.USER, "the study has no members to tabulate",
                                           "run it first: `study.run_to(folder)`")])
        units: dict[str, str] = {}
        columns: dict[str, list[NDArray[Any]]] = {}
        names = [name for table in tables for name in table]
        order = list(dict.fromkeys(names))
        rows = [len(next(iter(table.values()))) for table in tables]
        head: dict[str, list[NDArray[Any]]] = {}
        for path in self.axes:
            values = self._coordinates[path]
            if all(isinstance(v, Quantity) for v in values):
                unit = unit_symbol(cast(Quantity, values[0]).dimension)
                width = max(np.asarray(cast(Quantity, v).value).size for v in values)
                for component in range(width) if width > 1 else [None]:
                    name = column_name(path if component is None else f"{path}[{component}]", unit)
                    units[name] = unit
                    head[name] = [np.full(n, float(np.atleast_1d(np.asarray(cast(Quantity, v).value, dtype=np.float64))
                                                   [component or 0])) for v, n in zip(values, rows, strict=True)]
            else:
                units[path] = ""
                head[path] = [np.full(n, str(v) if isinstance(v, str) else v) for v, n in zip(values, rows, strict=True)]
        units["member"], units["group"] = "", ""
        head["member"] = [np.full(n, i, dtype=np.int64) for i, n in enumerate(rows)]
        head["group"] = [np.full(n, g, dtype=np.int64) for g, n in zip(self.groups, rows, strict=True)]
        columns.update(head)
        for name in order:
            units[name] = tables[next(i for i, t in enumerate(tables) if name in t)].units.get(name, "")
            parts: list[NDArray[Any]] = []
            for table, n in zip(tables, rows, strict=True):
                if name in table:
                    parts.append(table[name])
                else:
                    sample = tables[next(i for i, t in enumerate(tables) if name in t)][name]
                    parts.append(np.full(n, "" if sample.dtype.kind == "U" else np.nan,
                                         dtype=sample.dtype if sample.dtype.kind == "U" else np.float64))
            columns[name] = parts
        return Table({name: np.concatenate(parts) for name, parts in columns.items()}, units)

    @public
    def to_csv(self, path: str | PathLike[str]) -> Path:
        """The long table as CSV, its header the column names with their units."""
        table = self.table()
        target = Path(path)
        names = list(table)
        with target.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(names)
            for row in range(len(table[names[0]])):
                writer.writerow([str(table[name][row]) if table[name].dtype.kind == "U" else
                                 str(int(table[name][row])) if table[name].dtype.kind == "i" else
                                 repr(float(table[name][row])) for name in names])
        return target

    @public
    def arrays(self) -> dict[str, NDArray[Any]]:
        """What :meth:`to_npz` holds: each quantity coordinate in SI (``coordinates/<path>``, members first), ``groups``,
        ``points``, and every member's arrays as a run's NPZ holds them under ``m<point>/``, with its manifest
        (``m<point>/manifest``, JSON text)."""
        found: dict[str, NDArray[Any]] = {"groups": np.asarray(self.groups, dtype=np.int64),
                                          "points": np.asarray(self.points, dtype=np.int64)}
        for path, values in self.coordinates.items():
            found[f"coordinates/{path}"] = np.asarray(values.value if isinstance(values, Quantity) else values)
        for point, member in zip(self.points, self, strict=True):
            found.update(member_entries(point, member))
        return found

    @public
    def manifest(self) -> dict[str, Any]:
        """What ``study.json`` holds: the base experiment, the axes with their units and reasons, every point, the run,
        the groups with their structure keys and compiles, each member's status, overrides and content key, the points
        dropped, the seed policy and the lineage. A member that is not in this result reads ``pending``."""
        found = copy.deepcopy(self.record)
        done = set(self.points)
        for row in cast(list[dict[str, Any]], found.get("members", [])):
            row["status"] = "done" if row["point"] in done else "pending"
        return found

    @public
    def figure(self, *outputs: OutputRef | str, kind: str | None = None, against: str | None = None,
               members: Sequence[int] | None = None, name: str | None = None, **options: Any) -> FigureRecipe:
        """A figure of the study as a recipe (:class:`~mime.results.FigureRecipe`): one backend-neutral description that
        ``plot()`` (matplotlib) and ``plot_interactive()`` (plotly) draw, ``display()`` and ``web()`` show as a page, and
        ``write_recipe()`` saves beside the result's folder, to be redrawn from its data. For a study the kinds are ``curve`` (an output at the end against a coordinate: the step-out curve; ``against=`` names the axis), ``overlay`` (a recorded series of the members on one time axis; ``members=`` chooses them) and ``distribution`` (an output at the end over the members, for sampled points). ``options`` are
        ``title``, ``x_label``, ``y_label``, ``legend``, ``size`` (inches) and ``style``. Needs the ``plot`` extra."""
        from mime.results._figure import figure_of

        return figure_of(self, outputs, kind=kind, against=against, members=members, name=name, options=options)

    @public
    def plot(self, *outputs: OutputRef | str, kind: str | None = None, against: str | None = None,
             members: Sequence[int] | None = None, **options: Any) -> MatplotlibFigure:
        """The study's output as a matplotlib figure (static, for a paper; the ``plot`` extra): the :meth:`figure` of the same
        arguments, drawn. The figure is matplotlib's own, to edit with matplotlib."""
        return self.figure(*outputs, kind=kind, against=against, members=members, **options).plot()

    @public
    def plot_interactive(self, *outputs: OutputRef | str, kind: str | None = None, against: str | None = None,
                         members: Sequence[int] | None = None, **options: Any) -> PlotlyFigure:
        """The study's output as a plotly figure (interactive, for a card or a quick look; the ``plot`` extra): the
        :meth:`figure` of the same arguments, drawn. The figure is plotly's own, to edit with plotly."""
        return self.figure(*outputs, kind=kind, against=against, members=members, **options).plot_interactive()

    @public
    def to_npz(self, path: str | PathLike[str]) -> Path:
        """Every array :meth:`arrays` lists and the manifest (under ``manifest``, JSON text), in one NPZ NumPy reads."""
        target = Path(path)
        np.savez(target, allow_pickle=False, **self.arrays(),
                 **{"manifest": np.asarray(json.dumps(self.manifest(), sort_keys=True))})
        return target

    @public
    def to_mat(self, path: str | PathLike[str]) -> Path:
        """The study as a MATLAB v5 file, through ``scipy.io.savemat``, the member axis first in every array: a struct
        ``coordinates`` (each axis's values with its ``unit``), ``groups``, ``points``, a struct per observation
        holding a struct per output with its ``unit`` and ``dimension`` and its ``value`` over the members (an output at
        the end) or its ``time`` and ``values`` (members first where the members share a time axis, else a cell array
        of the members' own), ``flags`` (each flag's reading by member: ``true``, ``false``, ``unknown`` or ``not
        declared``), ``study`` (what ran), ``manifest`` (:meth:`manifest` as JSON text) and ``name_map`` from each
        MATLAB name to the study's. Names are mapped as :meth:`RunResult.to_mat` maps them. Read back by
        ``scipy.io.loadmat`` in MIME's tests; MATLAB itself has not been run on one. It holds the observations every
        member has: one that only some members have is left out (read those members alone, ``result.member(i)``)."""
        from mime.results._study_mat import write_study_mat

        return write_study_mat(self, path)

    @public
    def order(self, output: Quantity | str, *, along: object, expected: float | None = None) -> Order:
        """The observed order of convergence of ``output`` over the three levels of the axis ``along`` (a step, a
        spacing), and the value it extrapolates to. ``output`` is an output at the end over the members
        (``result[swim].mean_speed``, or its path ``"swim_speed:umr.mean_speed"``). The study is the three levels and
        nothing else: its other axes constant. Refused (``study.order``) where the levels are not in the asymptotic
        range: see :func:`mime.results.observed_order`."""
        from mime.results._order import observed_order

        values = self._series_over(output)
        levels = self.coordinates.get(str(along))
        if levels is None:
            raise ArgumentRefused([Refusal(
                "study.order", RefusalClass.USER, f"{along!s} is no axis of this study (its axes: {', '.join(self.axes)})",
                "name the axis the three levels vary: result.order(output, along=DT)")])
        others = [p for p in self.axes if p != str(along) and len({_key(v) for v in self._coordinates[p]}) > 1]
        if others:
            raise ArgumentRefused([Refusal(
                "study.order", RefusalClass.USER,
                f"the order is read from three levels of one axis, and {', '.join(others)} varies as well",
                "run a study over the one axis, `Study(exp).sweep(grid({DT: [s(4e-3), s(2e-3), s(1e-3)]}), ...)`, "
                "or one study for each value of the others")])
        if not isinstance(levels, Quantity):
            raise ArgumentRefused([Refusal("study.order", RefusalClass.USER,
                                           f"the levels {along!s} are not quantities", "sweep the step or spacing as a quantity: `grid({DT: [s(4e-3), s(2e-3), s(1e-3)]})`")])
        return observed_order(levels, values, expected=expected)

    def _series_over(self, output: Quantity | str) -> Quantity:
        if isinstance(output, Quantity):
            return output
        id, _, name = output.rpartition(".")
        held = self.output(id, name)
        if isinstance(held, StackedSeries):
            raise ArgumentRefused([Refusal("study.order", RefusalClass.USER,
                                           f"{output} is a series; the order is read from an output at the end",
                                           "name an output at the end, or reduce the series: result[swim].mean_speed")])
        return held

    @classmethod
    @public
    def read(cls, folder: str | PathLike[str]) -> StudyResult:
        """The result of a study, read from the folder ``Study.run_to`` or ``mime study run`` wrote: its layout checked,
        its manifest closed and in a format this MIME reads. A study that was stopped reads as what it holds:
        ``missing`` lists its members that are not there, and ``text()`` says how many. Members are read when first
        asked for."""
        from mime.results._study_read import read_study

        return read_study(Path(folder))

    @public
    def text(self) -> str:
        lines = [f"study of {self.experiment}: {len(self)} members over {', '.join(self.axes)}, in "
                 f"{len(set(self.groups))} group{'s' if len(set(self.groups)) != 1 else ''}"]
        raised = [path for path, values in self.flags.items() if any(value is True for value in values)]
        lines.append(f"flags raised in some member: {', '.join(raised)}" if raised else "no flag raised in any member")
        if self.missing:
            total = len(self) + len(self.missing)
            lines.append(f"PARTIAL: {len(self.missing)} of the study's {total} members are missing (not run yet, or run "
                         f"under another content key): points {', '.join(map(str, self.missing[:8]))}"
                         f"{', ...' if len(self.missing) > 8 else ''}")
        if self.dropped:
            lines.append(f"DROPPED {len(self.dropped)} refused point{'s' if len(self.dropped) != 1 else ''} "
                         f"(drop_refused=True): " + "; ".join(f"point {d.index} ({d.coordinates})" for d in self.dropped))
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.text()

    @override
    def __repr__(self) -> str:
        return f"<StudyResult {self.experiment}: {len(self)} members, {len(set(self.groups))} groups>"


__all__ = ["NOT_DECLARED", "Dropped", "StackedSeries", "StudyGrid", "StudyOutputs", "StudyResult", "member_entries"]

