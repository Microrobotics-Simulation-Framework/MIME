"""A study's run: each group on its built experiment, its members batched over the group's compiled step, or run one
after another on it where a controller is called between chunks, each member's result read as its own run's."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from mime.build._maddening import set_node_state, state_put_back
from mime.build._watch import ContainmentPlan
from mime.control import ChunkedRun
from mime.results import RunResult
from mime.results._study import StudyResult
from mime.study._point import Point, precision
from mime.study._study import Compiles, Group, StudyPlan, member_report, run_members

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment
    from mime.results import OutputRef
    from mime.units import Quantity

type Arrays = dict[str, dict[str, np.ndarray]]


def _kept(built: BuiltExperiment, recorded: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    """The fields a member's run keeps at every step when it records anything: its recorded outputs, its flags and the
    poses the containment watch reads, as a run's own ``keep`` is."""
    if not recorded:
        return ()
    held = {(output.node, output.field) for output in recorded}
    held |= {(flag.node, flag.field) for flag in built.layout.flags}
    held |= {(plan.node, plan.field) for plan in built.plans.watches if isinstance(plan, ContainmentPlan)}
    return tuple(sorted(held))


def _staged(built: BuiltExperiment, kept: tuple[tuple[str, str], ...], stage: int | None
            ) -> tuple[tuple[str, str], ...]:
    """What the run stacks: ``kept`` and, where the members' stage is kept (``stage``, the base steps between samples),
    the bound fields at every step too."""
    if stage is None:
        return kept
    return tuple(sorted({*kept, *built.binding().fields()}))


def _with_stage(result: RunResult, built: BuiltExperiment, point: Point, stacked: Arrays | None, steps: int,
                stage: int | None) -> RunResult:
    """``result`` with the member's stage samples (``result.stage``) taken from the fields the run stacked."""
    if stage is not None:
        assert stacked is not None
        from mime.study._stage import stage_record

        result.stage = stage_record(built, point.state, stacked, steps, stage)
    return result


def _one(arrays: Arrays, index: int) -> Arrays:
    return {node: {name: value[index] for name, value in fields.items()} for node, fields in arrays.items()}


type After = Callable[[list[int]], None]


def _batched(group: Group, points: Sequence[Point], steps: int, recorded: Sequence[Any], spacing: int,
             members: list[RunResult | None], todo: Collection[int] | None, after: After | None,
             stage: int | None = None) -> None:
    built = group.built
    assert built is not None
    tests = built.flag_tests()
    kept = _kept(built, recorded)
    stacking = _staged(built, kept, stage)
    start = 0
    for size in group.batches:
        planned = group.members[start:start + size]
        start += size
        batch = [index for index in planned if todo is None or index in todo]
        if not batch:
            continue
        final, tallies, stacked = run_members(built, points, batch, group.varying, group.chunks, tests, stacking)
        for j, index in enumerate(batch):
            point = points[index]
            members[index] = _with_stage(
                built.result_of(0, steps, _one(final, j), _one(stacked, j) if kept else None, recorded,
                                spacing, tally=None if kept else tallies[j], plans=point.plans,
                                report=member_report(built, point)),
                built, point, _one(stacked, j) if stage is not None else None, steps, stage)
        if after is not None:
            after(batch)


def _one_after_another(group: Group, points: Sequence[Point], steps: int, recorded: Sequence[Any], spacing: int,
                       members: list[RunResult | None], todo: Collection[int] | None, after: After | None,
                       stage: int | None = None) -> None:
    """Members with a controller called between chunks: each from its own initial state, with its own parameter tree
    and its own law, made from its own plan, on the group's one compiled step."""
    built = group.built
    assert built is not None
    graph = built.graph
    kept = _kept(built, recorded)
    stacking = _staged(built, kept, stage)
    for index in group.members:
        if todo is not None and index not in todo:
            continue
        point = points[index]
        assert point.plans is not None
        with state_put_back(graph):
            graph.reset_state()
            for name, fields in point.state.items():
                set_node_state(graph, name, fields)
            runs = ChunkedRun(graph, point.plans, point.law)
            final, stacked = runs.run(steps, history=bool(stacking), keep=frozenset(stacking) if stacking else None,
                                      tests=built.flag_tests(),
                                      params={"nodes": point.tree, "mappings": {}})
        members[index] = _with_stage(
            built.result_of(0, steps, final, stacked if kept else None, recorded, spacing, tally=runs.tally,
                            plans=point.plans, report=member_report(built, point)),
            built, point, stacked, steps, stage)
        if after is not None:
            after([index])


def _warn_aliasing(plan: StudyPlan, group: Group, recorded: int | None, stage: int | None) -> None:
    """The aliasing warning for a group: the fastest drive any of its members declares, against what it records and
    what it keeps of the stage."""
    from mime.experiment._aliasing import largest, warn_aliasing

    built = group.built
    assert built is not None
    rates = largest([plan.points[i].plans.turning for i in group.members if plan.points[i].plans is not None])
    for spacing, what in ((recorded, "the study's recorded series"), (stage, "the study's stage samples")):
        if spacing is not None:
            warn_aliasing(rates, spacing, built.plans.dt, what)


def run_group(plan: StudyPlan, number: int, record: Sequence[OutputRef | str], every: Quantity | None,
              members: list[RunResult | None], *, todo: Collection[int] | None = None,
              after: After | None = None) -> tuple[int, float]:
    """Group ``number`` of ``plan`` run into ``members`` (by point): the members of ``todo`` only, where it is given,
    batched as the plan batched them; ``after`` is called with the points of each batch or member as it finishes.
    Returns the XLA programs the run compiled (as JAX's monitoring counts them) and the seconds it took (noisy)."""
    import time

    group = plan.groups[number]
    built = group.built
    assert built is not None
    builder = plan.points[group.members[0]]
    started = time.perf_counter()
    with precision(builder.declared.x64):
        recorded = built.recording(record)
        spacing = built.spacing(every)
        stage = built.spacing(plan.stage_every) if plan.stage_every is not None else None
        _warn_aliasing(plan, group, spacing if recorded else None, stage)
        with Compiles() as compiled:
            if group.batched:
                _batched(group, plan.points, group.steps, recorded, spacing, members, todo, after, stage)
            else:
                _one_after_another(group, plan.points, group.steps, recorded, spacing, members, todo, after, stage)
    return compiled.count, time.perf_counter() - started


def run_study(plan: StudyPlan, record: Sequence[OutputRef | str], every: Quantity | None) -> StudyResult:
    """Every group of ``plan`` run, and the members' results gathered in point order (a dropped point has none)."""
    members: list[RunResult | None] = [None] * len(plan.points)
    compiles: list[int] = []
    seconds: list[float | None] = []
    for number in range(len(plan.groups)):
        count, took = run_group(plan, number, record, every, members)
        compiles.append(count)
        seconds.append(took)
    return StudyResult.of(plan, members, tuple(compiles), seconds=seconds)


__all__ = ["run_group", "run_study"]
