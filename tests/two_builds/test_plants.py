"""The two-builds check against the bake-check spike's plants, in the repository's harness (``mime.testing``), float64.

The toy plants (``plants.py``): a term made from the spring's ``k`` when the node is built (a Python float, a NumPy
array, a JAX array, a term of share 1e-6, S7's ``eta``) beside ``k`` read live, each pair two builds that differ only in
``k`` by a relative 1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-13, 1e-14, 1e-15 or one ulp, and twenty draws of ``k`` and of the
move (``k`` log-uniform in [0.5, 3], the move log-uniform in magnitude in [1e-15, 1e-1], either sign, new draws on every
run). The spike measured: the trace form caught every plant at every move (145 of 145), no twin (0 of 116) and no
same-value rebuild; the run form caught 104 of 145 at its 64-ulp line. Here:

- the trace form catches every plant at every move, and fires on no twin and no same-value rebuild;
- the run form, its line :data:`~mime.testing.RUN_FACTOR` times each field's measured resolution, catches every plant
  the spike's 64-ulp line catches, and fires on no twin;
- a value-keyed count fires only where the move crosses its threshold, as a ``shape``;
- the MRS drag with 30 % of its viscosity taken from the build's value at the trace (a plant in a real node) is caught
  by the trace form at every move from 1e-1 to 1e-15, and by the run form wherever the 64-ulp line caught it; its twin
  (the viscosity read live) quiet by both forms;
- conditions: the leaves closed over (the step's default parameters) make every twin fire; in float32 the check refuses
  to run.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
import pytest
from mrs_cases import BALL, free_space, mrs
from plants import BAKED, NODES, TWINS, planted

from mime.build._in_graph import program_of
from mime.build._maddening import chunk_step, fed_inputs, full_state, graph_params
from mime.nodes.hydro.mrs import _node as mrs_node
from mime.testing import TwoBuilds, two_builds_of
from mime.testing._two_builds import programs_differ, step_program
from mime.units import N, kg, kg_m2, m, m_per_s, mm, mPa_s, rad_per_s

EPS = float(np.finfo(np.float64).eps)
#: The spike's line for the run form: 64 ulps of a field's scale.
OLD_LINE = 64.0
LADDER: tuple[float | str, ...] = (1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-13, 1e-14, 1e-15, "ulp")


def draws(count: int = 20) -> list[tuple[float, float]]:
    rng = np.random.default_rng()
    return [(float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))),
             float(np.exp(rng.uniform(np.log(1e-15), np.log(1e-1)))) * float(rng.choice([-1.0, 1.0])))
            for _ in range(count)]


def old_line(found: TwoBuilds) -> bool:
    """Whether the spike's run form, 64 ulps of a field's scale, fires."""
    assert found.run is not None
    return any(gap.difference > OLD_LINE * EPS * gap.scale for gap in found.run.fields if math.isfinite(gap.scale))


@pytest.fixture(scope="module")
def pairs() -> dict[str, list[tuple[float, float, TwoBuilds]]]:
    """Every kind's pairs: the ladder at ``k`` 1.7, the draws, and a same-value rebuild (move 0)."""
    points: list[tuple[float, float | str]] = [(1.7, move) for move in LADDER]
    points += draws()
    points.append((1.7, 0.0))
    found: dict[str, list[tuple[float, float, TwoBuilds]]] = {}
    for kind in NODES:
        bases: dict[float, Any] = {}
        rows = []
        for k, move in points:
            if k not in bases:
                exp, path = planted(kind, k)
                bases[k] = (exp, path, exp.build())
            exp, path, built = bases[k]
            other = float(np.nextafter(k, np.inf)) if move == "ulp" else k * (1.0 + float(move))
            rows.append((k, (other - k) / k, two_builds_of(exp, path, to=N(other) / m(1.0), built=built)))
        found[kind] = rows
    return found


@pytest.mark.parametrize("kind", BAKED)
def test_the_trace_form_catches_every_plant_at_every_move_and_the_run_form_whatever_its_old_line_caught(
        kind: str, pairs: dict[str, list[tuple[float, float, TwoBuilds]]]):
    moved = [(k, move, found) for k, move, found in pairs[kind] if move != 0.0]
    assert len(moved) == len(LADDER) + 20
    missed = [(k, move) for k, move, found in moved if found.verdict != "bake"]
    assert not missed
    lost = [(k, move, found.run.ratio if found.run else None) for k, move, found in moved
            if old_line(found) and not (found.run is not None and found.run.fired)]
    assert not lost, "a plant the 64-ulp line caught falls under the resolution's line"


@pytest.mark.parametrize("kind", TWINS)
def test_no_twin_fires_either_form(kind: str, pairs: dict[str, list[tuple[float, float, TwoBuilds]]]):
    fired = [(k, move, found.verdict) for k, move, found in pairs[kind] if found.verdict != "quiet"]
    assert not fired
    assert all(found.run is not None and found.run.ratio == 0.0 for _, _, found in pairs[kind])


@pytest.mark.parametrize("kind", list(NODES))
def test_a_same_value_rebuild_is_quiet(kind: str, pairs: dict[str, list[tuple[float, float, TwoBuilds]]]):
    same = [found for _, move, found in pairs[kind] if move == 0.0]
    assert len(same) == 1 and same[0].verdict == "quiet" and not same[0].moved


def test_a_value_keyed_count_fires_as_a_shape_exactly_where_the_move_crosses(
        pairs: dict[str, list[tuple[float, float, TwoBuilds]]]):
    for k, move, found in pairs["count"]:
        crosses = math.ceil(4.0 * k) != math.ceil(4.0 * k * (1.0 + move))
        assert found.trace_fired == crosses, (k, move)
        if crosses:
            assert found.verdict in ("shape", "bake")


def test_condition_the_leaves_closed_over_make_every_twin_fire():
    """The trace form needs the leaves as arguments: traced with the step's default parameters (closed over, so each
    leaf's value a constant of the program) a twin fires at every move."""
    for kind in TWINS:
        exp, path = planted(kind, 1.7)
        base = exp.build()
        for move in (1e-1, 1e-6, 1e-12, "ulp"):
            other_k = float(np.nextafter(1.7, np.inf)) if move == "ulp" else 1.7 * (1.0 + float(move))
            from mime.saved._pins import keeping_pins
            from mime.study._axes import STUDY_REASON, declared, resolve
            from mime.experiment import Experiment

            settings, declarations, saved = exp.declared_as()
            with keeping_pins(saved):
                held = declared(settings, declarations, {resolve(path, declarations): N(other_k) / m(1.0)},
                                STUDY_REASON)
                other = Experiment.declared_from(held.settings, held.declarations).build()
            assert programs_differ(step_program(base), step_program(other)) == "", (kind, move)
            assert programs_differ(closed(base), closed(other)) != "", (kind, move)


def closed(built: Any) -> tuple[str, list[np.ndarray]]:
    graph = built.graph
    step = chunk_step(graph)
    graph.reset_state()
    state = jax.tree_util.tree_map(lambda leaf: leaf, full_state(graph))
    fed, params = fed_inputs(graph, built.held_inputs()), graph_params(graph)
    return program_of(jax.make_jaxpr(lambda s, e: step(s, e, params))(state, fed))


def test_condition_the_check_refuses_float32():
    exp, path = planted("float", 1.7)
    with jax.enable_x64(False):
        built = exp.build()
        with pytest.raises(ValueError, match="float64"):
            two_builds_of(exp, path, built=built)


MOVES = (1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-13, 1e-14, 1e-15)


@pytest.fixture
def viscosity(request: pytest.FixtureRequest) -> Iterator[str]:
    """The MRS drag's step with 30 % of its viscosity taken from the build's value, a Python float read at the trace
    (``plant``), or read live (``twin``)."""
    arm = request.param
    original = mrs_node.MRSDragNode._separated_step  # noqa: SLF001

    def step(self: mrs_node.MRSDragNode, p: Any, state: Any, pose: Any) -> Any:
        held = float(self.constants()["viscosity"]) if arm == "plant" else p["viscosity"]
        return original(self, {**p, "viscosity": 0.7 * p["viscosity"] + 0.3 * held}, state, pose)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mrs_node.MRSDragNode, "_separated_step", step)
        yield arm


@pytest.mark.parametrize("viscosity", ["plant", "twin"], indirect=True)
def test_a_plant_in_the_mrs_drags_step_is_caught_at_every_move_and_its_twin_never(viscosity: str):
    exp = free_space("free_ball", {"ball": BALL}, mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                     velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))
    built = exp.build()
    found = [two_builds_of(exp, "water.viscosity", to=mPa_s(1.0 + move), built=built) for move in MOVES]
    if viscosity == "plant":
        assert all(each.verdict == "bake" for each in found)
        lost = [move for move, each in zip(MOVES, found, strict=True)
                if old_line(each) and not (each.run is not None and each.run.fired)]
        assert not lost, "a plant the 64-ulp line caught falls under the resolution's line"
    else:
        assert all(each.verdict == "quiet" and each.run is not None and each.run.ratio == 0.0 for each in found)
