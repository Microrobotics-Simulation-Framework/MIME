"""Unit fields in a study (F0's item 9.2 and round 5's item 13): a field computed in the program from what it depends
on is made per member where that differs and once where it agrees, the stand-in's unit field (``unit_fields.py``, a
run-constant made from the law's ``nu``) read in the batched program a study runs (``vmap`` of the chunk over the
group's varying leaves, ``mime.build._maddening.chunk_function``): batched over the members in a sweep of ``nu``,
unbatched (made once) in a sweep of ``amplitude``. Each sweep's members match their own runs within gate 1's tolerance
(``tests/study/predictions.json``). Control: in the sweep of ``nu`` the field forced shared (``nu`` given every member
at the first member's value, as a leaf every member shares enters the batch) moves the members off their own runs
beyond that tolerance, on every run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from unit_fields import SIZE, spring_experiment

from mime import Study
from mime.build._maddening import chunk_function, full_state
from mime.study import grid
from mime.study._study import run_members, stacked_state, stacked_tree
from mime.units import s

GATE_1 = json.loads((Path(__file__).parents[1] / "study" / "predictions.json").read_text(encoding="utf-8"))["gate_1"]
TOLERANCE_EPS = float(GATE_1["tolerance_eps"]["float64"])
DURATION = s(0.4)
NUS = (0.3, 0.5, 0.8, 1.3)
AMPLITUDES = (0.1, 0.3, 0.6, 1.0)


def study_of(setting: str, values: tuple[float, ...]) -> tuple[Any, Any]:
    exp, spring = spring_experiment()
    study = Study(exp).sweep(grid({getattr(spring, setting): list(values)}), duration=DURATION)
    return study, study.plan()


def solve_shapes(plan: Any) -> list[tuple[int, ...]]:
    """The operand shapes of every LU factorisation in the group's batched chunk program, traced as the study's batched
    run vmaps it: the unit field's solve, batched or not."""
    group = plan.groups[0]
    graph = group.built.graph
    params, axes = stacked_tree(plan.points, group.members, group.varying)
    meta = dict(full_state(graph).get("_meta") or {})
    state = stacked_state(plan.points, group.members, meta)
    in_axes = {"nodes": {node: {leaf: axis for n, leaf, axis in axes if n == node} for node in params["nodes"]},
               "mappings": {}}
    run = jax.vmap(chunk_function(graph, 2, group.built.held_inputs()), in_axes=(0, in_axes))
    jaxpr = jax.make_jaxpr(run)(state, params)
    shapes: list[tuple[int, ...]] = []

    def visit(inner: Any) -> None:
        for eqn in inner.eqns:
            if eqn.primitive.name == "lu":
                shapes.append(tuple(eqn.invars[0].aval.shape))
            for value in eqn.params.values():
                for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                    held = getattr(sub, "jaxpr", sub)
                    if hasattr(held, "eqns"):
                        visit(held)

    visit(jaxpr.jaxpr)
    return shapes


def test_the_unit_field_is_made_per_member_where_its_input_differs():
    _, plan = study_of("nu", NUS)
    assert len(plan.groups) == 1
    assert solve_shapes(plan) and all(shape == (len(NUS), SIZE, SIZE) for shape in solve_shapes(plan))


def test_the_unit_field_is_made_once_where_its_input_agrees():
    _, plan = study_of("amplitude", AMPLITUDES)
    assert len(plan.groups) == 1
    assert solve_shapes(plan) and all(shape == (SIZE, SIZE) for shape in solve_shapes(plan))


def miss_eps(reference: np.ndarray, found: np.ndarray, over: np.ndarray) -> float:
    scale = float(np.max(np.abs(over)))
    delta = float(np.max(np.abs(np.asarray(reference, np.float64) - np.asarray(found, np.float64))))
    return 0.0 if delta == 0.0 else (float("inf") if scale == 0.0 else delta / (np.finfo(np.float64).eps * scale))


def worst_miss(plan: Any, finals: Any) -> float:
    """The largest miss of any member's final state from its own run (``run(override=...)`` on a fresh build), in eps of
    each field's largest magnitude over that run."""
    worst = 0.0
    for index, point in enumerate(plan.points):
        exp, spring = spring_experiment()
        built = exp.build()
        override = {axis.path: value for axis, value in point.values.items()}
        own = built.run(DURATION, history=True, override=override)
        assert own.history is not None
        for node, fields in own.final.items():
            for name, value in fields.items():
                worst = max(worst, miss_eps(value, finals[node][name][index], own.history[node][name]))
    return worst


@pytest.mark.parametrize(("setting", "values"), [("nu", NUS), ("amplitude", AMPLITUDES)])
def test_each_member_matches_its_own_run(setting: str, values: tuple[float, ...]):
    study, plan = study_of(setting, values)
    result = study.run()
    finals = {node: {name: np.stack([np.asarray(result.member(i).final[node][name]) for i in range(len(result))])
                     for name in fields} for node, fields in result.member(0).final.items()}
    assert worst_miss(plan, finals) <= TOLERANCE_EPS


def test_control_the_field_forced_shared_moves_the_members_off_their_own_runs():
    _, plan = study_of("nu", NUS)
    group = plan.groups[0]
    shared = tuple(each for each in group.varying if each[1] not in ("nu", "held_from.nu", "unit_field"))
    final, _, _ = run_members(group.built, plan.points, group.members, shared, group.chunks,
                              group.built.flag_tests(), ())
    assert worst_miss(plan, final) > TOLERANCE_EPS
