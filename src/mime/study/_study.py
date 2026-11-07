"""A study: many runs of one experiment. Its points are the experiment with their values declared, each planned in full;
the points that need one compiled program are a group, built and compiled once, and its members run together as one
batched scan (``mime.build._maddening.batched_scan``), or one after another on one compiled step where a controller is
called between chunks."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from jax import monitoring

from mime.build import BuildRefused, Refusal, RefusalClass, RunSettings
from mime.build._construct import report_of
from mime.build._checks import node_lines
from mime.build._key import split_reasons, structure_key
from mime.build._maddening import (
    META,
    FlagTally,
    FlagTest,
    batched_scan,
    full_state,
    state_put_back,
    unreadable_leaves,
)
from mime.build._plan import Section
from mime.build._control import Mode
from mime.build._refusals import ArgumentRefused
from mime.control._run import automatic_chunks
from mime.errors import public
from mime.study._axes import STUDY_REASON, Axis, differ, resolve, value_text
from mime.study._point import Point, build_point, precision, same_run
from mime.study._protocol import FromRest, Protocol, protocol_of
from mime.units import Quantity
from mime.validity import Profile

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment, Experiment
    from mime.results import OutputRef
    from mime.results._study import StudyResult

type Arrays = dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class Points:
    """A study's points: the axes as written (typed paths or their strings) and each point's value of each, in order."""

    axes: tuple[object, ...]
    rows: tuple[tuple[object, ...], ...]
    #: How the rows were made: ``"grid"`` (every combination of ``lists``), ``"along"`` (``lists`` taken together) or
    #: ``"table"`` (rows as given), and each axis's list of values for the first two: what a spec file writes.
    kind: str = "table"
    lists: tuple[tuple[object, ...], ...] = ()

    def __len__(self) -> int:
        return len(self.rows)


def _values(axes: Mapping[object, Sequence[object]], what: str) -> dict[object, tuple[object, ...]]:
    found: dict[object, tuple[object, ...]] = {}
    for path, values in axes.items():
        if isinstance(values, (str, Quantity)) or not isinstance(cast(object, values), Sequence):
            raise ArgumentRefused([Refusal(
                "study.axis-values", RefusalClass.USER,
                f"{what} takes each axis's values as a list, one per point; {path} is given {values!r}",
                f"list them: {what}({{{path}: [Hz(10.0), Hz(20.0)]}}), or [Hz(f) for f in ...]")])
        found[path] = tuple(values)
    if not found:
        raise ArgumentRefused([Refusal("study.axis-values", RefusalClass.USER, f"{what} is given no axis",
                                       f"name at least one: {what}({{field.commands.frequency: [...]}})")])
    return found


@public
def grid(axes: Mapping[object, Sequence[object]]) -> Points:
    """Every combination of the axes' values, the last axis varying fastest: ``grid({field.commands.frequency:
    [Hz(10.0), Hz(20.0)], "umr.mass": [kg(8e-5), kg(9e-5)]})`` is four points."""
    held = _values(axes, "grid")
    return Points(tuple(held), tuple(itertools.product(*held.values())), "grid", tuple(held.values()))


@public
def along(axes: Mapping[object, Sequence[object]]) -> Points:
    """The axes' values taken together, the i-th of each at the i-th point: every axis lists as many values."""
    held = _values(axes, "along")
    lengths = {len(values) for values in held.values()}
    if len(lengths) != 1:
        raise ArgumentRefused([Refusal(
            "study.axis-values", RefusalClass.USER,
            f"along takes axes of one length; they list {', '.join(f'{len(v)} ({p})' for p, v in held.items())}",
            "give each axis as many values, or take every combination with grid({...})")])
    return Points(tuple(held), tuple(zip(*held.values(), strict=True)), "along", tuple(held.values()))


def _points(given: object) -> Points:
    """A sweep's points: from :func:`grid` or :func:`along`, or a table, a list of mappings of one set of axes."""
    if isinstance(given, Points):
        return given
    held: list[Mapping[object, object]] = []
    rows: Sequence[object] = cast(Sequence[object], given) if isinstance(given, (list, tuple)) else ()
    for row in rows:
        if isinstance(row, Mapping):
            held.append(cast(Mapping[object, object], row))
    if held and len(held) == len(rows):
        axes = tuple(held[0])
        if any(tuple(row) != axes for row in held):
            raise ArgumentRefused([Refusal(
                "study.axis-values", RefusalClass.USER, "a table of points names the same axes in every row",
                "give every row the same axes: `[{\"umr.mass\": kg(8e-5)}, {\"umr.mass\": kg(9e-5)}]`")])
        rows = tuple(tuple(row[axis] for axis in axes) for row in held)
        return Points(axes, rows, "along", tuple(tuple(row[i] for row in rows) for i in range(len(axes))))
    raise ArgumentRefused([Refusal(
        "call.wrong-type", RefusalClass.USER, f"sweep takes points from grid(...), along(...) or a list of rows, not "
                                              f"{type(cast(object, given)).__name__}",
        "write sweep(grid({field.commands.frequency: [...]}), duration=...)")])


@dataclass
class Group:
    """Points that need one compiled program: its structure key, its members (point indices, in order), why it is not
    the first group (empty for the first), the built experiment it runs on, how its members run (``batched``, or one
    after another where a controller is called between chunks), the batches, the leaves that vary among its members
    (``varying``), and its memory estimate (:class:`Memory`)."""

    key: str
    members: list[int]
    reasons: tuple[str, ...] = ()
    built: BuiltExperiment | None = None
    batched: bool = True
    batches: tuple[int, ...] = ()
    chunks: tuple[int, ...] = ()
    varying: tuple[tuple[str, str], ...] = ()
    memory: Memory | None = None
    steps: int = 0

    @property
    def compiles(self) -> int:
        """The programs its run compiles: one per batch size and chunk length, where its members are batched; one per
        chunk length, where they run one after another."""
        lengths = len(set(self.chunks)) or 1
        return lengths * (len(set(self.batches)) if self.batched else 1)


def _mib(count: float) -> str:
    return f"{count / 2**20:.1f} MiB" if count >= 2**20 else f"{count / 2**10:.1f} KiB"


#: How far the third measured batch size may lie off the line through the other two, as a fraction of the per-member
#: bytes between it and the nearer of them, for the line to give a batch's device memory.
LINE_MISS = 0.2


def measured_sizes(batch: int) -> tuple[tuple[int, ...], int | None]:
    """The batch sizes a group's step is measured at for batches of ``batch`` members: the two the line is drawn
    through, and the one it is checked at, the batch itself always among them. From eight members, four and eight,
    which lie where XLA's buffer assignment grows by one member's bytes a member (at one member it assigns otherwise:
    on the MRS screw in a tube the step's temporaries grow by under 3 MiB from one member to two and by about 12 MiB a
    member from three up), checked at the batch, or at three for a batch of eight. Below eight, the batch and half of
    it, checked between them, or at three for a batch of two. A batch of one is measured at one alone."""
    if batch <= 1:
        return (1,), None
    if batch >= 8:
        return (4, 8), (batch if batch > 8 else 3)
    low = batch // 2
    between = list(range(low + 1, batch))
    return (low, batch), (between[len(between) // 2] if between else batch + 1)


@dataclass(frozen=True)
class Memory:
    """A group's memory estimate, in bytes. On the device: XLA's buffer assignment of one chunk of the batched step,
    compiled and not run, at each batch size of ``at`` (members, bytes), the batch it runs among them, so that its
    bytes (``device``) are measured and not extrapolated; and the line ``fixed + members * each`` through the first two,
    with ``miss``, how far the third lies off it as a fraction of the per-member bytes between it and the nearer of the
    two. The line gives a figure a member, for choosing a batch, only where it ``holds`` (``miss`` at most
    :data:`LINE_MISS`). On the host, ``host`` a member: its own leaves and state, its series and its report.
    ``batched`` is false where members run one after another on the build's own step, whose device memory is one
    run's, not measured here."""

    batch: int
    members: int
    host: int
    at: tuple[tuple[int, int], ...] = ()
    fixed: float | None = None
    each: float | None = None
    miss: float | None = None
    batched: bool = True

    @property
    def holds(self) -> bool:
        return self.miss is not None and self.miss <= LINE_MISS and self.fixed is not None and self.fixed >= 0

    @property
    def device(self) -> int | None:
        """The device bytes of one batch, measured at its size."""
        return dict(self.at).get(self.batch)

    @property
    def total(self) -> int | None:
        device = self.device
        return None if device is None else device + self.members * self.host

    def text(self) -> str:
        host = f"{self.members} x {_mib(self.host)} on the host (each member's own leaves and state, its series and its " \
               f"report)"
        device = self.device
        if not self.batched or device is None:
            return f"memory: one run's on the device at a time, and {host}"
        head = (f"memory about {_mib(device + self.members * self.host)}: {_mib(device)} on the device for a batch of "
                f"{self.batch} (XLA's buffer assignment of one chunk of the batched step at that size, compiled and not "
                f"run)")
        if len(self.at) < 3 or self.miss is None:
            return f"{head}, and {host}"
        (a, _), (b, _), (c, _) = self.at[:3]
        off = f"{100 * self.miss:.0f} %" if math.isfinite(self.miss) else "far"
        if self.holds and self.fixed is not None and self.each is not None:
            line = (f"{_mib(self.fixed)} + {_mib(self.each)} a member on the line through {a} and {b} members, which "
                    f"lies {off} of a member's bytes off the assignment at {c}")
        else:
            line = (f"no figure a member: at {c} members XLA's assignment lies {off} of a member's bytes off the line "
                    f"through {a} and {b}")
        return f"{head}; {line}; and {host}"


@dataclass(frozen=True)
class StudyPlan:
    """What a study will do, before anything runs: its points and axes, the groups with their compiles, batches and
    memory, why each group past the first split from it, and every refused point with every reason."""

    axes: tuple[Axis, ...]
    points: tuple[Point, ...]
    groups: tuple[Group, ...]
    steps: int
    drop_refused: bool
    #: What the sweep asked besides its points: each axis's own reason (by path), the run's length, what it records and
    #: how often, the largest batch, and the base experiment's name, profile and seed.
    reasons: Mapping[str, str] = field(default_factory=dict[str, str])
    duration: Quantity | None = None
    record: tuple[str, ...] = ()
    every: Quantity | None = None
    batch: int | None = None
    experiment: str = ""
    profile: str = ""
    seed: int = 0
    #: How the members' commands move in time: from rest (the default) or a ramp of one command.
    protocol: Protocol = field(default_factory=FromRest)

    @property
    def refused(self) -> tuple[Point, ...]:
        return tuple(point for point in self.points if point.refused)

    @property
    def compiles(self) -> int:
        return sum(group.compiles for group in self.groups)

    def text(self, *, memory: bool = True) -> str:
        """The plan as it prints. ``memory=False`` leaves out each group's memory estimate, whose bytes depend on the
        machine, the jaxlib and the Python it is made on, for a text that is the same on each."""
        refused = self.refused
        planning = sum(len(group.memory.at) for group in self.groups if group.memory is not None)
        lines = [f"{len(self.points)} points over {', '.join(axis.path for axis in self.axes)}; {len(self.groups)} "
                 f"group{'s' if len(self.groups) != 1 else ''}, {self.compiles} compile{'s' if self.compiles != 1 else ''}"
                 f" to run it (planning compiled {'each group' if len(self.groups) != 1 else 'the group'}'s build, "
                 f"and its batched step {planning} time{'s' if planning != 1 else ''} for the memory estimate); "
                 f"{f'{len(refused)} refused' if refused else 'nothing refused'}"]
        for number, group in enumerate(self.groups):
            how = (f"batched in {len(group.batches)} batch{'es' if len(group.batches) != 1 else ''} of "
                   f"{', '.join(map(str, sorted(set(group.batches), reverse=True)))}" if group.batched else
                   "one after another on one compiled step (a controller is called on the host between chunks)")
            estimate = f"; {group.memory.text()}" if memory and group.memory is not None else ""
            lines.append(f"  group {number}: {len(group.members)} points, {how}, {group.compiles} compile"
                         f"{'s' if group.compiles != 1 else ''}{estimate}")
            if group.reasons:
                lines.append(f"    split from group 0 by: {'; '.join(group.reasons)}")
        if not isinstance(self.protocol, FromRest):
            lines.append(f"  protocol: {self.protocol.text()}")
        for point in refused:
            lines.append(f"  refused point {point.index} ({_coordinates(self.axes, point)}):")
            lines += [f"    [{refusal.code}] {refusal.message}" for refusal in point.refusals]
        if refused:
            lines.append("  the study runs only once nothing is refused, or with drop_refused=True, which runs the rest "
                         "and records the points it dropped" if not self.drop_refused else
                         "  drop_refused=True: the study runs the rest, and its result records these")
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.text()


def _coordinates(axes: Sequence[Axis], point: Point) -> str:
    return ", ".join(f"{axis.path} = {value_text(point.values[axis])}" for axis in axes)


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def _nbytes(arrays: Mapping[str, Mapping[str, np.ndarray]]) -> int:
    return sum(value.nbytes for fields in arrays.values() for value in fields.values())


def device_bytes(built: BuiltExperiment, points: Sequence[Point], members: Sequence[int],
                 varying: Sequence[tuple[str, str]], tests: Sequence[FlagTest], kept: Sequence[tuple[str, str]],
                 chunk: int, batch: int) -> tuple[tuple[tuple[int, int], ...], float | None, float | None, float | None]:
    """The device memory of batches of the group's members: XLA's buffer assignment of one chunk of the batched step,
    compiled and not run, at each batch size :func:`measured_sizes` gives for ``batch`` (its members taken in turn,
    over again where the group has fewer: the bytes depend on the shapes, not the values). Returns the measurements,
    and the line through the first two, ``fixed`` and ``each``, with the third's ``miss`` (``None`` for one size). A
    measurement of the program, not a count of its arrays."""
    graph = built.graph
    with state_put_back(graph):
        graph.reset_state()
        meta = dict(full_state(graph).get(META) or {})
    line, third = measured_sizes(batch)
    sizes = [*line, *([third] if third is not None else [])]
    at: list[tuple[int, int]] = []
    for size in sizes:
        chosen = [members[i % len(members)] for i in range(size)]
        params, axes = stacked_tree(points, chosen, varying)
        state = stacked_state(points, chosen, meta)
        total, _, _ = batched_scan(graph, chunk, state, params, axes, tests, kept, measure=True)
        at.append((size, int(cast(int, total))))
    if third is None:
        return tuple(at), None, None, None
    return (tuple(at), *line_through(at))


def line_through(at: Sequence[tuple[int, int]]) -> tuple[float, float, float]:
    """The line ``fixed + members * each`` through the first two of three measurements (members, bytes), and how far
    the third lies off it, as a fraction of the per-member bytes between the third and the nearer of the two."""
    (a, ta), (b, tb), (c, tc) = at
    each = (tb - ta) / (b - a)
    fixed = ta - a * each
    nearer = min((a, b), key=lambda size: abs(size - c))
    span = abs(c - nearer) * each
    off = abs(tc - (fixed + c * each))
    return fixed, each, off / span if span > 0 else (0.0 if off == 0 else float("inf"))


def report_bytes(built: BuiltExperiment, point: Point) -> int:
    """What one member's result holds on the host beyond its arrays, measured: the bytes Python allocates for its
    report's summary, the largest part, as ``tracemalloc`` counts them."""
    import tracemalloc

    held = tracemalloc.is_tracing()
    if not held:
        tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        summary = member_report(built, point)
        after = tracemalloc.get_traced_memory()[0]
    finally:
        if not held:
            tracemalloc.stop()
    del summary
    return max(0, after - before)


class Study:
    """Many runs of one experiment (``exp``). :meth:`sweep` names the points and the run; :meth:`plan` says, before
    anything runs, the groups and compiles, the refused points and the memory; :meth:`run` runs it and returns a
    :class:`~mime.results.StudyResult`. The experiment's declarations are read when the study is planned."""

    def __init__(self, experiment: Experiment) -> None:
        from mime.experiment import Experiment

        if not isinstance(cast(object, experiment), Experiment):
            raise ArgumentRefused([Refusal(
                "call.wrong-type", RefusalClass.USER, f"Study takes an Experiment, not {type(experiment).__name__}",
                "pass the experiment: Study(exp).sweep(grid({...}), duration=s(1.0))")])
        self._experiment = experiment
        self._points: Points | None = None
        self._duration: Quantity | None = None
        self._record: tuple[OutputRef | str, ...] = ()
        self._every: Quantity | None = None
        self._batch: int | None = None
        self._drop = False
        self._protocol: Protocol = FromRest()
        self._reasons: dict[str, str] = {}
        self._plan: StudyPlan | None = None

    @public
    def sweep(self, points: object, *, duration: Quantity, record: Sequence[OutputRef | str] = (),
              every: Quantity | None = None, protocol: object = None, batch: int | None = None,
              drop_refused: bool = False, reasons: Mapping[object, str] | None = None) -> Study:
        """The points (``grid({...})``, ``along({...})`` or a list of rows) and each member's run: ``duration`` from
        the start, the per-step outputs ``record`` names kept every ``every`` (as :meth:`BuiltExperiment.run` takes
        them), how its commands move in time (``protocol``: ``FromRest()``, the default, or a ``Ramp(...)`` of one
        command from its point's value), at most ``batch`` members at once (the whole group when not given), and with
        ``drop_refused`` the points nothing refuses run while the refused ones are recorded on the result. ``reasons``
        gives each axis its own reason (by the axis, or its path), one string per axis: the research profile refuses a
        study whose axis has none (``study.axis-reason``), and the exploratory profile declares every value with the
        generic study reason where an axis has none. Returns the study."""
        if batch is not None and (not isinstance(cast(object, batch), int) or batch < 1):
            raise ArgumentRefused([Refusal("study.batch", RefusalClass.USER,
                                           f"batch= is a number of members of at least one, not {batch!r}",
                                           "write batch=16, or leave it out for the whole group at once")])
        self._points, self._duration, self._record, self._every = _points(points), duration, tuple(record), every
        self._batch, self._drop, self._protocol, self._plan = batch, drop_refused, protocol_of(protocol), None
        self._reasons = {str(axis): reason for axis, reason in (reasons or {}).items()}
        return self

    @public
    def plan(self) -> StudyPlan:
        """Every point planned in full and assembled (nothing compiled), the points grouped by the compiled program
        each needs, one build per group, and every refusal at once: a study-wide one raises ``BuildRefused``; a
        refused point is listed in the plan, and :meth:`run` refuses the study for it unless ``drop_refused``."""
        if self._plan is not None:
            return self._plan
        if self._points is None or self._duration is None:
            raise ArgumentRefused([Refusal("study.no-sweep", RefusalClass.USER, "the study has no points to plan",
                                           "name them first: Study(exp).sweep(grid({...}), duration=s(1.0))")])
        self._plan = _plan(self._experiment, self._points, self._duration, self._record, self._every, self._batch,
                           self._drop, self._reasons, self._protocol)
        return self._plan

    @public
    def run(self) -> StudyResult:
        """Run every group: its members batched (one compile per batch size and chunk length) or, with a controller
        called between chunks, one after another on one compiled step. Refused with every refused point unless the
        sweep dropped them (``drop_refused=True``). Prints nothing; :meth:`plan` says what it will do."""
        from mime.study._run import run_study

        plan = self.plan()
        if plan.refused and not plan.drop_refused:
            raise BuildRefused([Refusal(
                "study.refused-points", RefusalClass.USER,
                f"{len(plan.refused)} of the study's {len(plan.points)} points are refused:\n" + "\n".join(
                    f"  point {point.index} ({_coordinates(plan.axes, point)}): " + "; ".join(
                        f"[{r.code}] {r.message}" for r in point.refusals) for point in plan.refused),
                "change the axes' values so that every point is one the experiment allows, or sweep(..., "
                "drop_refused=True) to run the rest, the dropped points recorded on the result")])
        assert self._duration is not None
        return run_study(plan, self._record, self._every)

    @public
    def run_to(self, out: str | PathLike[str], *, shard: tuple[int, int] | None = None, pilot: bool = False,
               say: Callable[[str], None] | None = None) -> StudyResult:
        """Run the study into the folder ``out``, group by group and batch by batch, and return what it holds
        (:meth:`mime.results.StudyResult.read`). The folder is a study folder (``mime.results``): the base experiment
        saved once, the spec file, a manifest that says which members are done, and each group's arrays. ``out`` is a
        new or empty folder, or one that holds this study: a folder that holds anything else is refused and left as it
        is. A run into a folder that holds this study resumes: a member that is ``done`` under the same content key is
        not run again, and every other is. The key is its point, its group's structure key, the base's input hash, the
        run, and the code: every file of the installed ``mime`` package (and of a package a class of its nodes, entries
        or law comes from), with the versions of MADDENING, OpenMaterica, JAX, jaxlib, NumPy and SciPy and the
        precision. So any change to MIME's files runs every member again. ``shard=(i, k)`` runs only the structure
        groups whose number is ``i`` modulo ``k`` (``k`` at most the study's groups), the rest left ``pending``; shards
        run one after another into one folder add up to the whole study. ``pilot`` has ``say`` the measured cost of a
        chunk of the first group this run runs, before the rest."""
        from mime.study._folder_run import run_study_to

        self.plan()
        assert self._points is not None
        return run_study_to(self, self._experiment, self._points, Path(out), shard=shard, pilot=pilot, say=say)

    @classmethod
    @public
    def from_spec(cls, spec: str | PathLike[str], *, base: str | PathLike[str] | None = None) -> Study:
        """The study a spec file (``study.mime.yaml``) describes, over the saved experiment it names (or ``base``)."""
        from mime.study._spec import study_from_spec

        return study_from_spec(Path(spec), None if base is None else Path(base))[0]

    @override
    def __repr__(self) -> str:
        return f"<Study of {self._experiment.name}: {len(self._points) if self._points is not None else 0} points>"


def _reason_refusals(settings: RunSettings, axes: Sequence[Axis], reasons: Mapping[str, str]) -> list[Refusal]:
    """The refusals of the reasons a sweep gives: a reason for a path that is no axis of it, and, in the research
    profile, each axis without a reason of its own."""
    found: list[Refusal] = []
    known = [axis.path for axis in axes]
    stray = [path for path in reasons if path not in known]
    if stray:
        found.append(Refusal("study.reason-no-axis", RefusalClass.USER,
                             f"a reason is given for {', '.join(stray)}, which no axis of the sweep is",
                             f"give each reason for an axis of the sweep, `reasons={{'<axis>': '<why>'}}`: {', '.join(known)}"))
    bare = [path for path in known if not reasons.get(path, "").strip()]
    if settings.profile == Profile.RESEARCH.value and bare:
        found.append(Refusal(
            "study.axis-reason", RefusalClass.USER,
            f"the research profile needs each axis of a study to say why it is varied, and {', '.join(bare)} "
            f"{'has' if len(bare) == 1 else 'have'} no reason",
            "give one string per axis, not one per point: Study(exp).sweep(grid({...}), duration=..., reasons={"
            + ", ".join(f"{path!r}: '<why this axis is varied, and over what range>'" for path in bare)
            + "}); in a spec file, a `reason:` under each axis",
            why="a swept value is a declared value, and the research profile asks each declared value for its basis"))
    return found


def _plan(experiment: Experiment, given: Points, duration: Quantity, record: Sequence[OutputRef | str],
          every: Quantity | None, batch: int | None, drop: bool, reasons: Mapping[str, str],
          protocol: Protocol | None = None) -> StudyPlan:
    from mime.experiment import Experiment
    from mime.saved._pins import keeping_pins

    settings, declarations, saved = experiment.declared_as()
    axes = tuple(resolve(path, declarations) for path in given.axes)
    seen = [axis.path for axis in axes]
    if len(set(seen)) != len(seen):
        raise ArgumentRefused([Refusal("study.axis-twice", RefusalClass.USER, f"an axis is named twice: {seen}",
                                       "name each axis once in `grid({...})`")])
    if refusals := _reason_refusals(settings, axes, reasons):
        raise BuildRefused(refusals)
    first: dict[str, Point] = {}
    records: dict[str, Any] = {}
    points: list[Point] = []
    with keeping_pins(saved):
        for index, row in enumerate(given.rows):
            point = build_point(index, settings, declarations, dict(zip(axes, row, strict=True)), STUDY_REASON,
                                first.get, reasons, protocol)
            points.append(point)
            if point.key is not None and point.key not in first and not point.refused:
                first[point.key] = point

        def probe(values: Mapping[Axis, object]) -> Point:
            return build_point(-1, settings, declarations, values, STUDY_REASON, first.get, reasons, protocol)

        inert = inert_axes(axes, points, probe)
        if inert:
            raise BuildRefused(inert)
        groups: list[Group] = []
        for key in first:
            built = None
            while built is None:
                candidates = [p for p in points if p.key == key and not p.refused]
                if not candidates:
                    break
                builder = candidates[0]
                try:
                    with precision(builder.declared.x64):
                        built = Experiment.declared_from(builder.declared.settings,
                                                         builder.declared.declarations).build()
                        observing = [n for p in built.plans.entries if p.section is Section.OBSERVATION
                                     for n in p.node_names()]
                        records[key] = structure_key(built.graph, observing).record
                except BuildRefused as refused:
                    # The full build judges what assembling did not: the node's own checks at the start, the post-build
                    # checks. Refused for this point, and for each point whose checked nodes it shared.
                    for point in candidates:
                        if point is builder or point.checked == builder.checked:
                            point.refusals = tuple(refused.refusals)
                    continue
                groups.append(Group(key, [p.index for p in candidates], built=built))
    for group in groups[1:]:
        group.reasons = tuple(dict.fromkeys(split_reasons(records[groups[0].key], records[group.key])))
    steps = groups[0].built.steps_in(duration) if groups and groups[0].built is not None else 0
    for group in groups:
        with precision(points[group.members[0]].declared.x64):
            _settle(group, points, axes, duration, record, every, batch)
    return StudyPlan(axes, tuple(points), tuple(groups), steps, drop, dict(reasons), duration,
                     tuple(str(getattr(each, "path", each)) for each in record), every, batch, settings.experiment,
                     settings.profile, settings.seed, protocol or FromRest())


def inert_axes(axes: Sequence[Axis], points: Sequence[Point],
               probe: Callable[[Mapping[Axis, object]], Point]) -> list[Refusal]:
    """The refusal of each axis whose values change nothing the study runs (``study.axis-inert``): every two points
    that differ in its value alone run one computation (``same_run``: one structure key, equal leaves and initial
    states, a host controller's law from equal settings). The pairs are the points' own where the sweep holds the other
    axes fixed between them (a grid); otherwise one pair is made, the first point and the same point at another of the
    axis's values (``probe``). An axis with one value, or whose pairs are refused, is not judged."""
    found: list[Refusal] = []
    held = [point for point in points if not point.refused and point.key is not None]
    for axis in axes:
        if len({value_text(point.values[axis]) for point in points}) < 2:
            continue
        others = [other for other in axes if other is not axis]
        rows: dict[tuple[str, ...], list[Point]] = {}
        for point in held:
            rows.setdefault(tuple(value_text(point.values[other]) for other in others), []).append(point)
        pairs = [(row[0], point) for row in rows.values() for point in row[1:]
                 if differ(row[0].values[axis], point.values[axis])]
        if not pairs and held:
            start = held[0]
            moved = next((point.values[axis] for point in points if differ(point.values[axis], start.values[axis])),
                         None)
            if moved is not None:
                made = probe({**start.values, axis: moved})
                if not made.refused and made.key is not None:
                    pairs = [(start, made)]
        if pairs and all(same_run(a, b) for a, b in pairs):
            a, b = pairs[0]
            found.append(Refusal(
                "study.axis-inert", RefusalClass.USER,
                f"the axis {axis.path} changes nothing the study runs: its values ({value_text(a.values[axis])} and "
                f"{value_text(b.values[axis])}, and every other pair the sweep holds) give points with one compiled "
                f"program, the same parameter leaves and the same initial state, so its members would run one "
                f"computation. Either the experiment does not use the value, or the entry that consumes it keeps it "
                f"outside its node's leaves and declared statics, where a study cannot see it",
                "sweep a value the experiment's nodes hold in a leaf or a declared static; for a value an entry keeps "
                "outside both, make each value an experiment of its own, `Experiment.based_on(...)`, and build each, "
                "which bakes the value into that build, and report the entry: a node holds every value it consumes in "
                "a leaf or a declared static"))
    return found


def _settle(group: Group, points: Sequence[Point], axes: Sequence[Axis], duration: Quantity,
            record: Sequence[OutputRef | str], every: Quantity | None, batch: int | None) -> None:
    """A group's run settled: its batches and chunks, the leaves that vary among its members, the refusal of a varying
    leaf the step does not read, and its memory estimate, measured by compiling one chunk of the batched step at two or
    three batch sizes (``device_bytes``) and running nothing."""
    built = group.built
    assert built is not None
    steps = built.steps_in(duration)
    group.steps = steps
    recorded = built.recording(record)
    built.spacing(every)
    controller = built.plans.controller
    group.batched = controller is None or controller.mode is not Mode.CALLBACK
    size = len(group.members) if batch is None else min(batch, len(group.members))
    group.batches = tuple(min(size, len(group.members) - start) for start in range(0, len(group.members), size)) \
        if group.batched else (1,) * len(group.members)
    group.chunks = automatic_chunks(steps) if group.batched else ()
    builder = points[group.members[0]]
    varying = [(node, leaf) for node, leaves in builder.tree.items() for leaf, value in leaves.items()
               if any(not _equal(points[i].tree[node][leaf], value) for i in group.members[1:])]
    group.varying = tuple(varying)
    unread = unreadable_leaves(built.graph)
    baked = [(node, leaf) for node, leaf in varying if (node, leaf) in unread and not leaf.startswith("initial_")]
    if baked:
        written = [axis.path for axis in axes
                   if len({value_text(points[i].values[axis]) for i in group.members}) > 1]
        raise BuildRefused([Refusal(
            "study.leaf-unread", RefusalClass.USER,
            f"the study's axes ({', '.join(written)}) write " + "; ".join(
                f"params['nodes'][{node!r}][{leaf!r}], which changes nothing in the compiled step: {unread[(node, leaf)]}"
                for node, leaf in baked),
            "sweep what the step reads; a value a node consumes when it is built needs a rebuild for each value: "
            "declare it in the experiment, `exp.declare(...)`, once per study, or make each value an experiment of its "
            "own, `Experiment.based_on(...)`")])
    state = builder.state
    kept = sorted({(output.node, output.field) for output in recorded})
    per_step = sum(state[node][name].nbytes for node, name in kept if node in state and name in state[node])
    second = points[group.members[1]] if len(group.members) > 1 else builder
    own = sum(value.nbytes for node, leaves in second.tree.items() for leaf, value in leaves.items()
              if value is not builder.tree[node][leaf])
    own += sum(value.nbytes for node, fields in second.state.items() for name, value in fields.items()
               if value is not builder.state[node][name])
    host = per_step * steps + report_bytes(built, builder) + own + _nbytes(state)
    largest = max(group.batches, default=1)
    if group.batched:
        at, fixed, each, miss = device_bytes(built, points, group.members, varying, built.flag_tests(), kept,
                                             max(group.chunks, default=1), largest)
        group.memory = Memory(largest, len(group.members), host, at, fixed, each, miss)
    else:
        group.memory = Memory(1, len(group.members), host, batched=False)


@dataclass
class Compiles:
    """The XLA programs compiled while it is listened to, as JAX's monitoring reports them."""

    count: int = 0
    _listening: bool = field(default=False, repr=False)

    def __enter__(self) -> Compiles:
        monitoring.register_event_duration_secs_listener(self._listen)
        return self

    def __exit__(self, *_: object) -> None:
        monitoring.unregister_event_duration_listener(self._listen)

    def _listen(self, event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            self.count += 1


def stacked_tree(points: Sequence[Point], members: Sequence[int], varying: Sequence[tuple[str, str]]
                 ) -> tuple[dict[str, Any], tuple[tuple[str, str, int | None], ...]]:
    """The batch's parameter tree and, for every leaf, its member axis: a leaf that varies among ``members`` stacked
    along a leading axis (``0``), and every other the first member's, shared (``None``) and never copied."""
    first = points[members[0]].tree
    moving = set(varying)
    nodes: dict[str, dict[str, Any]] = {}
    axes: list[tuple[str, str, int | None]] = []
    for node in sorted(first):
        nodes[node] = {}
        for leaf in sorted(first[node]):
            if (node, leaf) in moving:
                nodes[node][leaf] = np.stack([points[i].tree[node][leaf] for i in members])
                axes.append((node, leaf, 0))
            else:
                nodes[node][leaf] = first[node][leaf]
                axes.append((node, leaf, None))
    return {"nodes": nodes, "mappings": {}}, tuple(axes)


def stacked_state(points: Sequence[Point], members: Sequence[int], meta: Mapping[str, Any]) -> dict[str, Any]:
    """Each member's initial state, every field stacked along a leading member axis, and MADDENING's counters
    (``_meta``) seeded for each member from the built graph's own, as it stands before a run."""
    first = points[members[0]].state
    state: dict[str, Any] = {node: {name: np.stack([points[i].state[node][name] for i in members])
                                    for name in fields} for node, fields in first.items()}
    state[META] = {name: np.broadcast_to(np.asarray(value), (len(members), *np.shape(value))).copy()
                   for name, value in meta.items()}
    return state


def run_members(built: BuiltExperiment, points: Sequence[Point], members: Sequence[int],
                varying: Sequence[tuple[str, str]], chunks: Sequence[int], tests: Sequence[FlagTest],
                kept: Sequence[tuple[str, str]], *, batched: bool = True) -> tuple[Arrays, list[FlagTally], Arrays]:
    """``members`` of one group, run as a batch over ``chunks`` (steps each) from their own initial states on the group's
    built graph: their final states and kept fields stacked over members (kept also over steps), and each member's
    flag tally from the run's start. ``batched=False`` runs one member unvmapped, the program a batch of one is held to.
    """
    graph = built.graph
    with state_put_back(graph):
        graph.reset_state()
        meta = dict(full_state(graph).get(META) or {})
        params, axes = stacked_tree(points, members, varying)
        state: Any = stacked_state(points, members, meta)
        if not batched:
            state = {node: {name: value[0] for name, value in fields.items()} for node, fields in state.items()}
            params = {"nodes": {node: {leaf: (value[0] if (node, leaf) in set(varying) else value)
                                       for leaf, value in leaves.items()} for node, leaves in params["nodes"].items()},
                      "mappings": {}}
        tallies = [FlagTally.none(len(tests)) for _ in members]
        kept_parts: list[Any] = []
        done = 0
        for length in chunks:
            state, (counted, first, last), out = batched_scan(graph, length, state, params, axes, tests, kept,
                                                              batched=batched)
            counted, first, last = (np.asarray(each, np.int64) for each in (counted, first, last))
            if not batched:
                counted, first, last = counted[None], first[None], last[None]
            tallies = [tally.then(FlagTally(counted[i], first[i], last[i]).shifted(done))
                       for i, tally in enumerate(tallies)]
            kept_parts.append(out)
            done += length
        final = {node: {name: np.asarray(value) if batched else np.asarray(value)[None] for name, value in fields.items()}
                 for node, fields in cast(dict[str, Any], state).items() if node != META}
        stacked: Arrays = {}
        for node, field_ in kept:
            parts = [np.asarray(part[node][field_]) for part in kept_parts]
            joined = np.concatenate(parts, axis=1 if batched else 0)
            stacked.setdefault(node, {})[field_] = joined if batched else joined[None]
    return final, tallies, stacked


def member_report(built: BuiltExperiment, point: Point) -> dict[str, Any]:
    """The build report's summary of ``point``, as its own build would give it: the group's structure (nodes, edges,
    lags, external inputs, MADDENING's warnings) with the point's own plan, physics hash and computed lines."""
    from mime.experiment._experiment import report_summary, run_layout

    plans = point.plans
    assert plans is not None
    group = built.report
    computed = point.computed if point.computed is not None else node_lines(built.graph, plans)
    report = report_of(plans, [line.node for line in group.nodes], (), group.external_inputs, group.warnings,
                       point.physics_hash, group.lags, point.models, computed)
    return report_summary(report, run_layout(plans))


__all__ = ["LINE_MISS", "Compiles", "Group", "Memory", "Points", "Study", "StudyPlan", "along", "device_bytes", "grid",
           "inert_axes", "line_through", "measured_sizes", "member_report", "run_members", "stacked_state", "stacked_tree"]
