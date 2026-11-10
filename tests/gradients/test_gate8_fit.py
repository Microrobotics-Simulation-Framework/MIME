"""The first fit (``Study.fit``) under the fit-loop rule, and gate 8: validity judged at each iterate.

- **The consumer**: the gently pulled sphere's radius fitted from 1.0 mm to a pose target made at 1.15 mm, inside its
  declared range: it ends on the target, where the spike's descent by rebuilding ended 0.79 % short (each rebuild
  re-chose its force points). Its control: the same fit without the range is refused at its first iterate that needs
  another program (``fit.value-keyed``), never ending short.
- **Gate 8**: the same sphere turned by a torque that puts the rung's declared Reynolds number (0.1) at about 1.05 mm,
  fitted from 1.15 mm towards a target made at 1.0 mm, past the line: an iterate past it is judged there and not
  taken, the line's margin limits the steps, and the fit ends on it, naming it. **Control**: the same loop through
  MADDENING's ``sysid.fit`` on the built graph (Adam, the loss through the graph's parameters) ends past the line, and
  its result says nothing of it (``MIME_PLANT_FIT=sysid`` runs it in the gate's place, for the kit's rate).
- **The Fisher information at the start**: a residual the radius cannot move is named insensitive before the first
  step."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradient_rigs import RADIUS, STEPS, TURNED, gentle, pose_residual, pose_target

from mime.build import ArgumentRefused, BuildRefused
from mime.build._watch import WatchPlan
from mime.results._watch import read_watch
from mime.study import Study
from mime.units import mm, ms

NOISE = 1e-4
LINE = "reynolds:drag:ball"


@pytest.fixture(scope="module")
def consumer():
    """The consumer's fit; ``MIME_PLANT_FIT=unranged`` takes the range away (the gate kit's rate of its control)."""
    ranged = None if os.environ.get("MIME_PLANT_FIT") == "unranged" else (1.0, 1.15)
    return Study(gentle(1.0, ranged=ranged)).fit(pose_residual(pose_target(1.15)), start={RADIUS: mm(1.0)},
                                                 noise=NOISE, duration=ms(STEPS))


def test_the_fit_ends_on_a_target_inside_the_range(consumer):
    assert consumer.ended == "converged"
    assert abs(float(consumer.values[RADIUS]) / 1.15e-3 - 1.0) < 1e-6
    assert all(line.judged for iterate in consumer.iterates for line in iterate.lines)
    assert int(consumer.fim.rank) == 1 and consumer.insensitive == ()


def test_control_without_the_range_the_fit_is_refused_where_an_iterate_needs_another_program():
    with pytest.raises(BuildRefused) as raised:
        Study(gentle(1.0, ranged=None, name="unranged")).fit(pose_residual(pose_target(1.15)),
                                                            start={RADIUS: mm(1.0)}, noise=NOISE, duration=ms(STEPS))
    assert [refusal.code for refusal in raised.value.refusals] == ["fit.value-keyed"]


def sysid_radius(target) -> float:
    """MADDENING's sysid.fit on the built graph, the loss half the scaled residual's squared norm through the graph's
    parameters, the radius alone trained: Adam from 1.15 mm, 150 updates of about 2e-6 m."""
    from maddening import sysid

    built = gentle(1.15, pull=TURNED, name="sysid").build()
    graph = built.graph
    residual = pose_residual(target)

    def loss(params):
        graph.reset_state()
        scaled = residual(graph.run_scan(STEPS, params=params)) / NOISE
        return 0.5 * jnp.sum(scaled * scaled)

    mask = jax.tree.map(lambda _: False, graph.params)
    mask["nodes"]["drag:ball"]["shape_radius"] = True
    found = sysid.fit(graph, loss, mask=mask, n_iter=150, lr=2e-6)
    graph.reset_state()
    return float(found.params["nodes"]["drag:ball"]["shape_radius"])


def turning_number(radius_mm: float) -> float:
    built = gentle(radius_mm, pull=TURNED, name="watched").build()
    final = {node: {name: np.asarray(value) for name, value in fields.items()}
             for node, fields in built.run(ms(STEPS)).final.items()}
    (plan,) = [held for held in built.plans.watches if isinstance(held, WatchPlan)]
    read = read_watch(plan, final, None)
    assert read.reynolds is not None
    return read.reynolds


@pytest.fixture(scope="module")
def across():
    target = pose_target(1.0, TURNED)
    if os.environ.get("MIME_PLANT_FIT") == "sysid":
        return None, sysid_radius(target)
    return Study(gentle(1.15, pull=TURNED)).fit(pose_residual(target), start={RADIUS: mm(1.15)}, noise=NOISE,
                                                duration=ms(STEPS)), None


def test_an_iterate_past_a_line_is_judged_there_and_the_fit_ends_on_the_line(across):
    result, control = across
    assert control is None, f"the loop ran through sysid.fit and ended at {control} m, judging no line"
    assert result.ended == "at a line" and result.why == LINE
    final = result.iterates[[i.taken for i in result.iterates].index(True, 0)] if result.taken else None
    assert final is not None
    last = result.taken[-1]
    (held,) = [line for line in last.lines if line.name == LINE]
    assert 0.0 <= held.margin <= 1e-5
    assert any(LINE in [line.name for line in iterate.past] and not iterate.taken for iterate in result.iterates)
    assert 1.0e-3 < float(result.values[RADIUS]) < 1.15e-3


def test_control_the_same_loop_through_sysid_fit_ends_past_the_line_and_says_nothing():
    radius = sysid_radius(pose_target(1.0, TURNED))
    assert radius < 1.04e-3, "it went towards the target, through the line"
    assert turning_number(radius * 1e3) > 0.1, "past the line, which its result does not mention"


def test_an_axis_the_residual_cannot_see_is_named_before_the_first_step():
    result = Study(gentle(1.05)).fit(lambda final: jnp.zeros(3) * final["ball"]["pose"][0] + 1.0,
                                     start={RADIUS: mm(1.05)}, noise=NOISE, duration=ms(STEPS))
    assert int(result.fim.rank) == 0 and result.insensitive == (RADIUS,)


def test_a_fit_with_no_start_or_a_noise_that_is_not_positive_is_refused():
    with pytest.raises(ArgumentRefused) as raised:
        Study(gentle(1.05)).fit(lambda final: final["ball"]["pose"], start={}, noise=NOISE, duration=ms(STEPS))
    assert [r.code for r in raised.value.refusals] == ["fit.start"]
    with pytest.raises(ArgumentRefused) as raised:
        Study(gentle(1.05)).fit(lambda final: final["ball"]["pose"], start={RADIUS: mm(1.05)}, noise=-1.0,
                                duration=ms(STEPS))
    assert [r.code for r in raised.value.refusals] == ["fit.noise"]


def test_a_fit_starting_outside_its_range_is_refused():
    with pytest.raises(ArgumentRefused) as raised:
        Study(gentle(1.05)).fit(lambda final: final["ball"]["pose"], start={RADIUS: mm(1.2)}, noise=NOISE,
                               duration=ms(STEPS))
    assert [r.code for r in raised.value.refusals] == ["fit.start"]
