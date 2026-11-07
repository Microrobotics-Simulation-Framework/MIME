"""A study: many runs of one experiment. Its points are the experiment with their values declared, each planned in full;
the points that need one compiled program are a group, built and compiled once, and its members run together as one
batched scan (``mime.build._maddening.batched_scan``), or one after another on one compiled step where a controller is
called between chunks."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from jax import monitoring

from mime.build import BuildRefused, Refusal, RefusalClass
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
from mime.study._axes import STUDY_REASON, Axis, resolve, value_text
from mime.study._point import Point, build_point, precision
from mime.units import Quantity

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
    return Points(tuple(held), tuple(itertools.product(*held.values())))


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
    return Points(tuple(held), tuple(zip(*held.values(), strict=True)))


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
        return Points(axes, tuple(tuple(row[axis] for axis in axes) for row in held))
    raise ArgumentRefused([Refusal(
        "call.wrong-type", RefusalClass.USER, f"sweep takes points from grid(...), along(...) or a list of rows, not "
                                              f"{type(cast(object, given)).__name__}",
        "write sweep(grid({field.commands.frequency: [...]}), duration=...)")])


@dataclass
class Group:
    """Points that need one compiled program: its structure key, its members (point indices, in order), why it is not
    the first group (empty for the first), the built experiment it runs on, how its members run (``batched``, or one
    after another where a controller is called between chunks), the batches, the leaves that vary among its members
    (``varying``), and the memory estimate's two terms in bytes, ``fixed + members * each``."""

    key: str
    members: list[int]
    reasons: tuple[str, ...] = ()
    built: BuiltExperiment | None = None
    batched: bool = True
    batches: tuple[int, ...] = ()
    chunks: tuple[int, ...] = ()
    varying: tuple[tuple[str, str], ...] = ()
    fixed: int = 0
    each: int = 0
    steps: int = 0

    @property
    def compiles(self) -> int:
        """The programs its run compiles: one per batch size and chunk length, where its members are batched; one per
        chunk length, where they run one after another."""
        lengths = len(set(self.chunks)) or 1
        return lengths * (len(set(self.batches)) if self.batched else 1)


def _mib(count: int) -> str:
    return f"{count / 2**20:.1f} MiB" if count >= 2**20 else f"{count / 2**10:.1f} KiB"


@dataclass(frozen=True)
class StudyPlan:
    """What a study will do, before anything runs: its points and axes, the groups with their compiles, batches and
    memory, why each group past the first split from it, and every refused point with every reason."""

    axes: tuple[Axis, ...]
    points: tuple[Point, ...]
    groups: tuple[Group, ...]
    steps: int
    drop_refused: bool

    @property
    def refused(self) -> tuple[Point, ...]:
        return tuple(point for point in self.points if point.refused)

    @property
    def compiles(self) -> int:
        return sum(group.compiles for group in self.groups)

    def text(self) -> str:
        refused = self.refused
        lines = [f"{len(self.points)} points over {', '.join(axis.path for axis in self.axes)}; {len(self.groups)} "
                 f"group{'s' if len(self.groups) != 1 else ''}, {self.compiles} compile{'s' if self.compiles != 1 else ''}"
                 f"; {f'{len(refused)} refused' if refused else 'nothing refused'}"]
        for number, group in enumerate(self.groups):
            how = (f"batched in {len(group.batches)} batch{'es' if len(group.batches) != 1 else ''} of "
                   f"{', '.join(map(str, sorted(set(group.batches), reverse=True)))}" if group.batched else
                   "one after another on one compiled step (a controller is called on the host between chunks)")
            memory = (f"memory about {_mib(group.fixed)} + {len(group.members)} x {_mib(group.each)} "
                      f"= {_mib(group.fixed + len(group.members) * group.each)} (XLA's buffer assignment of a chunk "
                      f"of the step at one and two members, which leaves out a solver's own workspace; each point's own "
                      f"leaves and state, its series and its report, measured)")
            lines.append(f"  group {number}: {len(group.members)} points, {how}, {group.compiles} compile"
                         f"{'s' if group.compiles != 1 else ''}; {memory}")
            if group.reasons:
                lines.append(f"    split from group 0 by: {'; '.join(group.reasons)}")
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
                 chunk: int) -> tuple[int, int]:
    """The device memory a batch of the group's members needs, ``fixed + B * each`` in bytes: XLA's buffer assignment
    of one chunk of the batched step, compiled and not run, for a batch of one member and of two (the first member
    twice where the group has one). A measurement of the program, not a count of its arrays."""
    graph = built.graph
    with state_put_back(graph):
        graph.reset_state()
        meta = dict(full_state(graph).get(META) or {})
    sizes: list[int] = []
    for batch in ([members[0]], [members[0], members[1] if len(members) > 1 else members[0]]):
        params, axes = stacked_tree(points, batch, varying)
        state = stacked_state(points, batch, meta)
        total, _, _ = batched_scan(graph, chunk, state, params, axes, tests, kept, measure=True)
        sizes.append(int(cast(int, total)))
    each = max(0, sizes[1] - sizes[0])
    return sizes[0] - each, each


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
        self._plan: StudyPlan | None = None

    @public
    def sweep(self, points: object, *, duration: Quantity, record: Sequence[OutputRef | str] = (),
              every: Quantity | None = None, batch: int | None = None, drop_refused: bool = False) -> Study:
        """The points (``grid({...})``, ``along({...})`` or a list of rows) and each member's run: ``duration`` from
        the start, the per-step outputs ``record`` names kept every ``every`` (as :meth:`BuiltExperiment.run` takes
        them), at most ``batch`` members at once (the whole group when not given), and with ``drop_refused`` the
        points nothing refuses run while the refused ones are recorded on the result. Returns the study."""
        if batch is not None and (not isinstance(cast(object, batch), int) or batch < 1):
            raise ArgumentRefused([Refusal("study.batch", RefusalClass.USER,
                                           f"batch= is a number of members of at least one, not {batch!r}",
                                           "write batch=16, or leave it out for the whole group at once")])
        self._points, self._duration, self._record, self._every = _points(points), duration, tuple(record), every
        self._batch, self._drop, self._plan = batch, drop_refused, None
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
                           self._drop)
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

    @override
    def __repr__(self) -> str:
        return f"<Study of {self._experiment.name}: {len(self._points) if self._points is not None else 0} points>"


def _plan(experiment: Experiment, given: Points, duration: Quantity, record: Sequence[OutputRef | str],
          every: Quantity | None, batch: int | None, drop: bool) -> StudyPlan:
    from mime.experiment import Experiment
    from mime.saved._pins import keeping_pins

    settings, declarations, saved = experiment.declared_as()
    axes = tuple(resolve(path, declarations) for path in given.axes)
    seen = [axis.path for axis in axes]
    if len(set(seen)) != len(seen):
        raise ArgumentRefused([Refusal("study.axis-twice", RefusalClass.USER, f"an axis is named twice: {seen}",
                                       "name each axis once in `grid({...})`")])
    first: dict[str, Point] = {}
    records: dict[str, Any] = {}
    points: list[Point] = []
    with keeping_pins(saved):
        for index, row in enumerate(given.rows):
            point = build_point(index, settings, declarations, dict(zip(axes, row, strict=True)), STUDY_REASON,
                                first.get)
            points.append(point)
            if point.key is not None and point.key not in first and not point.refused:
                first[point.key] = point
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
    return StudyPlan(axes, tuple(points), tuple(groups), steps, drop)


def _settle(group: Group, points: Sequence[Point], axes: Sequence[Axis], duration: Quantity,
            record: Sequence[OutputRef | str], every: Quantity | None, batch: int | None) -> None:
    """A group's run settled: its batches and chunks, the leaves that vary among its members, the refusal of a varying
    leaf the step does not read, and its memory estimate, measured by compiling one chunk of the batched step at one
    and two members (``device_bytes``) and running nothing."""
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
    if group.batched:
        fixed, each = device_bytes(built, points, group.members, varying, built.flag_tests(), kept,
                                   max(group.chunks, default=1))
    else:
        fixed, each = sum(v.nbytes for leaves in builder.tree.values() for v in leaves.values()), _nbytes(state)
    group.fixed = fixed
    second = points[group.members[1]] if len(group.members) > 1 else builder
    own = sum(value.nbytes for node, leaves in second.tree.items() for leaf, value in leaves.items()
              if value is not builder.tree[node][leaf])
    own += sum(value.nbytes for node, fields in second.state.items() for name, value in fields.items()
               if value is not builder.state[node][name])
    group.each = each + per_step * steps + report_bytes(built, builder) + own


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


__all__ = ["Compiles", "Group", "Points", "Study", "StudyPlan", "along", "grid", "member_report", "run_members",
           "stacked_state", "stacked_tree", "device_bytes"]
