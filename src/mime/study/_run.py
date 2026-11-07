"""A study's run: each group on its built experiment, its members batched over the group's compiled step, or run one
after another on it where a controller is called between chunks, each member's result read as its own run's."""

from __future__ import annotations

from collections.abc import Sequence
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


def _one(arrays: Arrays, index: int) -> Arrays:
    return {node: {name: value[index] for name, value in fields.items()} for node, fields in arrays.items()}


def _batched(group: Group, points: Sequence[Point], steps: int, recorded: Sequence[Any], spacing: int,
             members: list[RunResult | None]) -> None:
    built = group.built
    assert built is not None
    tests = built.flag_tests()
    kept = _kept(built, recorded)
    start = 0
    for size in group.batches:
        batch = group.members[start:start + size]
        start += size
        final, tallies, stacked = run_members(built, points, batch, group.varying, group.chunks, tests, kept)
        for j, index in enumerate(batch):
            point = points[index]
            members[index] = built.result_of(0, steps, _one(final, j), _one(stacked, j) if kept else None, recorded,
                                             spacing, tally=None if kept else tallies[j], plans=point.plans,
                                             report=member_report(built, point))


def _one_after_another(group: Group, points: Sequence[Point], steps: int, recorded: Sequence[Any], spacing: int,
                       members: list[RunResult | None]) -> None:
    """Members with a controller called between chunks: each from its own initial state, with its own parameter tree
    and its own law, made from its own plan, on the group's one compiled step."""
    built = group.built
    assert built is not None
    graph = built.graph
    kept = _kept(built, recorded)
    for index in group.members:
        point = points[index]
        assert point.plans is not None
        with state_put_back(graph):
            graph.reset_state()
            for name, fields in point.state.items():
                set_node_state(graph, name, fields)
            runs = ChunkedRun(graph, point.plans, point.law)
            final, stacked = runs.run(steps, history=bool(kept), keep=frozenset(kept) if kept else None,
                                      tests=built.flag_tests(),
                                      params={"nodes": point.tree, "mappings": {}})
        members[index] = built.result_of(0, steps, final, stacked, recorded, spacing, tally=runs.tally,
                                         plans=point.plans, report=member_report(built, point))


def run_study(plan: StudyPlan, record: Sequence[OutputRef | str], every: Quantity | None) -> StudyResult:
    """Every group of ``plan`` run, and the members' results gathered in point order (a dropped point has none)."""
    members: list[RunResult | None] = [None] * len(plan.points)
    compiles: list[int] = []
    for group in plan.groups:
        built = group.built
        assert built is not None
        builder = plan.points[group.members[0]]
        with precision(builder.declared.x64):
            steps = group.steps
            recorded = built.recording(record)
            spacing = built.spacing(every)
            with Compiles() as compiled:
                if group.batched:
                    _batched(group, plan.points, steps, recorded, spacing, members)
                else:
                    _one_after_another(group, plan.points, steps, recorded, spacing, members)
            compiles.append(compiled.count)
    return StudyResult.of(plan, members, tuple(compiles))


__all__ = ["run_study"]
