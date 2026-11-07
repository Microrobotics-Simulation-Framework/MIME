"""A study's result, in memory: one member axis, a coordinate column per axis with its units, the group each member ran
in, each declared output stacked over the members, each member as a :class:`RunResult`, and the members' flags as one
union. Saving it, its long table and ``.mat`` come later (S2c), on this shape."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

import numpy as np

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.errors import public
from mime.results._result import RunResult
from mime.results._series import Reading, Series
from mime.units import Quantity

if TYPE_CHECKING:
    from mime.drives import Ramp
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


class StudyResult:
    """A study's result: ``len(result)`` members, in the order of the points that ran; ``result.coordinates[path]``,
    each axis's values over the members (an array quantity, members first, or a tuple where an axis's values are not
    quantities); ``result.groups``, the group each member ran in; ``result[swim].mean_speed``, an output at the end over
    the members (an array quantity); ``result[swim].speed``, a recorded series stacked over the members where they share
    its time axis (:class:`StackedSeries`), refused otherwise with the route to :meth:`member`; :meth:`member`, one
    member as a :class:`RunResult`; :attr:`flags`, the union of the members' flags; and :attr:`dropped`, the points
    ``drop_refused=True`` left out, with their refusals."""

    def __init__(self, experiment: str, axes: Sequence[str], coordinates: Mapping[str, Sequence[object]],
                 members: Sequence[RunResult], groups: Sequence[int], points: Sequence[int],
                 dropped: Sequence[Dropped] = (), compiles: Sequence[int] = (), protocol: object = None,
                 ramps: Sequence[Ramp | None] = ()) -> None:
        from mime.study._protocol import FromRest

        self.experiment = experiment
        self.axes = tuple(axes)
        self._coordinates = {path: tuple(values) for path, values in coordinates.items()}
        self._members = tuple(members)
        #: The group each member ran in, by member.
        self.groups = tuple(groups)
        #: The point each member is, by member: its index among the study's points.
        self.points = tuple(points)
        self.dropped = tuple(dropped)
        #: The XLA programs each group's run compiled, as JAX's monitoring counted them.
        self.compiles = tuple(compiles)
        #: How the members' commands moved in time (``mime.study.FromRest`` or ``mime.study.Ramp``), which names the
        #: quantity a step-out read from them is.
        self.protocol: Protocol = cast("Protocol", protocol) if protocol is not None else FromRest()
        #: The drive the protocol gave each member's ramped command (``None`` from rest), by member.
        self.ramps = tuple(ramps) if ramps else (None,) * len(self._members)

    @classmethod
    def of(cls, plan: StudyPlan, members: Sequence[RunResult | None], compiles: Sequence[int]) -> StudyResult:
        from mime.study._axes import value_text

        ran = [i for i, member in enumerate(members) if member is not None]
        group_of = {index: number for number, group in enumerate(plan.groups) for index in group.members}
        coordinates = {axis.path: [plan.points[i].values[axis] for i in ran] for axis in plan.axes}
        dropped = [Dropped(point.index, ", ".join(f"{axis.path} = {value_text(point.values[axis])}" for axis in plan.axes),
                           tuple(f"[{r.code}] {r.message}" for r in point.refusals)) for point in plan.refused]
        name = str(plan.groups[0].built.plans.settings.experiment) if plan.groups and plan.groups[0].built else ""
        return cls(name, [axis.path for axis in plan.axes], coordinates,
                   [cast(RunResult, members[i]) for i in ran], [group_of[i] for i in ran], ran, dropped, compiles,
                   plan.protocol, [plan.points[i].ramp for i in ran])

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[RunResult]:
        return iter(self._members)

    @public
    def member(self, index: int) -> RunResult:
        """Member ``index`` as a :class:`RunResult`, with every accessor a run's result has."""
        if not -len(self._members) <= index < len(self._members):
            raise ArgumentRefused([Refusal(
                "study.no-member", RefusalClass.USER, f"the study has {len(self._members)} members, and no member {index}",
                f"take one of its members, `result.member(0)` to `result.member({len(self._members) - 1})`")])
        return self._members[index]

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
        if not self._members or not any(id in member.observations for member in self._members):
            known = sorted({held for member in self._members for held in member.observations})
            raise ArgumentRefused([Refusal(
                "study.no-observation", RefusalClass.USER, f"no member of the study observes {id!r}",
                f"index it by the handle `exp.observe(...)` returned, `result[swim]`, or by an id it observes: "
                f"{', '.join(known) or 'none'}")])
        return StudyOutputs(self, id)

    @public
    def output(self, id: str, name: str) -> Quantity | StackedSeries:
        """Observation ``id``'s output ``name`` over the members: what ``result[id].name`` reads."""
        held: list[Reading | Series] = []
        for index, member in enumerate(self._members):
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
        declared = [{flag.path for flag in member.layout.flags} for member in self._members]
        # A run's flags are those raised over it or that it did not keep enough of to say; a flag its build declares
        # and that is not among them was kept and not raised.
        reads = [{read.path: read.was_raised for read in member.flags} for member in self._members]
        paths = sorted({path for held in declared for path in held})
        return {path: tuple(read.get(path, False) if path in own else NOT_DECLARED
                            for own, read in zip(declared, reads, strict=True)) for path in paths}

    @public
    def text(self) -> str:
        lines = [f"study of {self.experiment}: {len(self)} members over {', '.join(self.axes)}, in "
                 f"{len(set(self.groups))} group{'s' if len(set(self.groups)) != 1 else ''}"]
        raised = [path for path, values in self.flags.items() if any(value is True for value in values)]
        lines.append(f"flags raised in some member: {', '.join(raised)}" if raised else "no flag raised in any member")
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


__all__ = ["NOT_DECLARED", "Dropped", "StackedSeries", "StudyOutputs", "StudyResult"]

