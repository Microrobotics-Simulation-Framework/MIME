"""The two-builds check against the bake-check spike's plants, in the repository's harness (``mime.testing``), float64
(the check refuses float32, a condition below).

The toy plants (``plants.py``): a term made from the spring's ``k`` when the node is built (a Python float, a NumPy
array, a JAX array, a term of share 1e-6, S7's ``eta``) beside ``k`` read live, each pair two builds that differ only in
``k`` by a relative 1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-13, 1e-14, 1e-15 or one ulp, twenty draws of ``k`` and of the
move (``k`` log-uniform in [0.5, 3], the move log-uniform in magnitude in [1e-15, 1e-1], either sign) and six draws in
the run form's band (the move in [1e-6, 1e-1]), new draws on every run. The spike measured: the trace form caught every
plant at every move (145 of 145), no twin (0 of 116) and no same-value rebuild; the run form caught 104 of 145 at its
64-ulp line. Here:

- the trace form catches every plant at every move, and fires on no twin and no same-value rebuild;
- the run form, its line :data:`~mime.testing.RUN_FACTOR` (k) times each field's measured resolution, catches every
  plant moved by at least 1e-6 with four times k resolutions to spare, on every run, and fires on no twin. Its line
  is thin (k measured in a window of margins 1.1 and 1.2), so the gate stands on plants far above it; the plants
  just above the spike's 64-ulp line are a probe of the line, reported (``record_property``), not a gate;
- a value-keyed count fires only where the move crosses its threshold, as a ``shape``;
- the MRS drag with 30 % of its viscosity taken from the build's value at the trace (a plant in a real node) is caught
  by the trace form at every move from 1e-1 to 1e-15, and by the run form from 1e-1 to 1e-12 with the same margin
  (the smaller moves reported); its twin (the viscosity read live) quiet by both forms;
- conditions: the leaves closed over (the step's default parameters) make every twin fire; in float32 the check is
  refused, and so are a pair whose leaves differ in shape and a value with no relative move.
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

from mime.build import ArgumentRefused
from mime.build._in_graph import program_of
from mime.build._maddening import chunk_step, fed_inputs, full_state, graph_params
from mime.nodes.hydro.mrs import _node as mrs_node
from mime.testing import TwoBuilds, run_form, two_builds_of
from mime.testing._two_builds import programs_differ, step_program
from mime.testing.toys._experiments import ADD_BATH, ADD_PROBE, toy_experiment
from mime.units import N, kg, kg_m2, m, m_per_s, mm, mPa_s, rad_per_s

EPS = float(np.finfo(np.float64).eps)
#: The spike's line for the run form: 64 ulps of a field's scale.
OLD_LINE = 64.0
LADDER: tuple[float | str, ...] = (1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-13, 1e-14, 1e-15, "ulp")
#: The smallest move the run form's gate plants: the smallest share of the load a plant makes (``tiny``'s 1e-6) moved
#: by it shows about 3e3 resolutions [measured], 45 times the gate's 4 k.
BAND = 1e-6
#: How far above its line a gating plant stands: four times k resolutions.
MARGIN = 4.0


def draws(count: int, smallest: float) -> list[tuple[float, float]]:
    rng = np.random.default_rng()
    return [(float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))),
             float(np.exp(rng.uniform(np.log(smallest), np.log(1e-1)))) * float(rng.choice([-1.0, 1.0])))
            for _ in range(count)]


def old_line(found: TwoBuilds) -> bool:
    """Whether the spike's run form, 64 ulps of a field's scale, fires."""
    assert found.run is not None
    return any(gap.difference > OLD_LINE * EPS * gap.scale for gap in found.run.fields if math.isfinite(gap.scale))


Pairs = dict[str, list[tuple[float, float, TwoBuilds]]]


@pytest.fixture(scope="module")
def pairs() -> Pairs:
    """Every kind's pairs: the ladder at ``k`` 1.7, the draws, and a same-value rebuild (move 0)."""
    points: list[tuple[float, float | str]] = [(1.7, move) for move in LADDER]
    points += draws(20, 1e-15) + draws(6, BAND)
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


def test_the_trace_form_catches_every_plant_at_every_move(pairs: Pairs):
    for kind in BAKED:
        moved = [(k, move, found) for k, move, found in pairs[kind] if move != 0.0]
        assert len(moved) == len(LADDER) + 26
        assert not [(k, move) for k, move, found in moved if found.verdict != "bake"], kind


def gated(move: float) -> bool:
    return abs(move) >= BAND * (1.0 - 1e-9)


def test_the_run_form_catches_every_plant_in_its_band_four_lines_up(pairs: Pairs):
    for kind in BAKED:
        band = [(k, move, found) for k, move, found in pairs[kind] if move != 0.0 and gated(move)]
        assert len(band) >= 3 + 6
        short = [(k, move, found.run.ratio if found.run else None) for k, move, found in band
                 if found.run is None or found.run.ratio < MARGIN]
        assert not short, f"{kind}: a gating plant stands less than four times k resolutions above the line: {short}"


def test_probe_the_plants_just_above_the_old_line(pairs: Pairs, record_property: Any):
    """Reported, not a gate: of the plants under the band that the spike's 64-ulp line catches, how many the line of
    k resolutions catches too, and the least ratio among them (above 1 fires), by kind. Each is measured against a
    resolution the run found, so the ratios are numbers."""
    for kind in BAKED:
        probed = [found for _, move, found in pairs[kind] if move != 0.0 and not gated(move) and old_line(found)]
        assert all(found.run is not None and found.run.worst is not None and found.run.worst.resolution > 0.0
                   for found in probed), kind
        ratios = [found.run.ratio for found in probed if found.run is not None]
        record_property(f"probe_{kind}_over_old_line", len(ratios))
        record_property(f"probe_{kind}_caught_by_new_line", sum(ratio > 1.0 for ratio in ratios))
        record_property(f"probe_{kind}_least_ratio", min(ratios, default=float("nan")))


def test_no_twin_fires_either_form(pairs: Pairs):
    for kind in TWINS:
        assert not [(k, move, found.verdict) for k, move, found in pairs[kind] if found.verdict != "quiet"], kind
        assert all(found.run is not None and found.run.ratio == 0.0 for _, _, found in pairs[kind]), kind


def test_a_same_value_rebuild_is_quiet(pairs: Pairs):
    for kind in NODES:
        same = [found for _, move, found in pairs[kind] if move == 0.0]
        assert len(same) == 1 and same[0].verdict == "quiet" and not same[0].moved, kind


def test_a_value_keyed_count_fires_as_a_shape_exactly_where_the_move_crosses(pairs: Pairs):
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
        with pytest.raises(ArgumentRefused) as raised:
            two_builds_of(exp, path, built=built)
    assert raised.value.codes() == ["two-builds.float32"]


def test_the_run_form_refuses_two_builds_whose_leaves_differ_in_shape():
    exp, _ = planted("float", 1.7)
    bare = toy_experiment("two_builds_bare")
    ADD_BATH.apply(bare)
    ADD_PROBE.apply(bare)
    with pytest.raises(ArgumentRefused) as raised:
        run_form(exp.build(), bare.build())
    assert raised.value.codes() == ["two-builds.leaf-shapes"]


def test_a_value_with_no_relative_move_is_refused():
    exp, path = planted("float", 1.7)
    with pytest.raises(ArgumentRefused) as raised:
        two_builds_of(exp, f"{str(path).rsplit('.', 1)[0]}.kind")
    assert raised.value.codes() == ["two-builds.not-movable"]


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
def test_a_plant_in_the_mrs_drags_step_is_caught_at_every_move_and_its_twin_never(viscosity: str,
                                                                                  record_property: Any):
    exp = free_space("free_ball", {"ball": BALL}, mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                     velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))
    built = exp.build()
    found = [two_builds_of(exp, "water.viscosity", to=mPa_s(1.0 + move), built=built) for move in MOVES]
    if viscosity == "plant":
        assert all(each.verdict == "bake" for each in found)
        short = [move for move, each in zip(MOVES, found, strict=True)
                 if move >= 1e-12 and not (each.run is not None and each.run.ratio >= MARGIN)]
        assert not short, "a gating plant stands less than four times k resolutions above the run form's line"
        for move, each in zip(MOVES, found, strict=True):
            if move < 1e-12 and each.run is not None:
                record_property(f"probe_ratio_at_{move:.0e}", each.run.ratio)
    else:
        assert all(each.verdict == "quiet" and each.run is not None and each.run.ratio == 0.0 for each in found)
