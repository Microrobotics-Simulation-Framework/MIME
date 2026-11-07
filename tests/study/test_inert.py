"""A value that reaches nothing a run computes from is refused, in a study (``study.axis-inert``) and in an override
(``override.no-effect``), each with its route: the planted spring that keeps its stiffness in a Python attribute alone
(``hidden``: no leaf, static or structure holds it) is the control both must refuse, on every run, since its points
run one computation whatever the stiffness. The spring that holds its stiffness in a leaf is accepted in both, and an
override at the experiment's own value runs as the base's own run."""

from __future__ import annotations

import numpy as np
import pytest
from planted import planted

from mime import Study
from mime.build import BuildRefused
from mime.study import along, grid
from mime.units import N, m, mm, s

K = [N(1.1) / m(1.0), N(3.9) / m(1.0)]


def _refused(kind: str, points):
    exp, spring = planted(kind)
    with pytest.raises(BuildRefused) as refused:
        Study(exp).sweep(points(spring), duration=s(0.4)).plan()
    return refused.value


def test_an_axis_a_node_keeps_outside_its_leaves_and_statics_is_refused_with_its_route():
    refused = _refused("hidden", lambda spring: grid({spring.k: K}))
    assert refused.codes() == ["study.axis-inert"]
    (refusal,) = refused.refusals
    assert "studytoy.spring:probe.k" in refusal.message and "one computation" in refusal.message
    assert "Experiment.based_on" in refusal.action and "leaf or a declared static" in refusal.action


def test_along_another_axis_the_inert_axis_is_judged_on_a_pair_made_for_it():
    """With ``along`` no two points differ in the stiffness alone, so the plan makes one: the first point at the second
    stiffness. The start moves the probe's state, so that axis is not refused."""
    starts = [mm((50.0, -20.0, 10.0)), mm((40.0, -20.0, 10.0))]
    refused = _refused("hidden", lambda spring: along({spring.k: K, "probe.initial.position": starts}))
    assert refused.codes() == ["study.axis-inert"] and "studytoy.spring:probe.k" in refused.refusals[0].message


def test_the_control_a_stiffness_the_node_holds_in_a_leaf_is_swept():
    exp, spring = planted("plain")
    plan = Study(exp).sweep(grid({spring.k: K}), duration=s(0.4)).plan()
    assert len(plan.points) == 2 and not plan.refused
    assert not np.array_equal(plan.points[0].tree["studytoy.spring:probe"]["k"],
                              plan.points[1].tree["studytoy.spring:probe"]["k"])


def test_an_axis_with_one_value_is_not_judged():
    exp, spring = planted("hidden")
    assert not Study(exp).sweep(grid({spring.k: K[1:]}), duration=s(0.4)).plan().refused


def test_an_override_that_changes_nothing_is_refused_and_one_at_the_experiments_own_value_runs():
    exp, spring = planted("hidden")
    built = exp.build()
    with pytest.raises(BuildRefused) as refused:
        built.run(s(0.4), override={spring.k: K[1]})
    assert refused.value.codes() == ["override.no-effect"]
    assert "studytoy.spring:probe.k" in refused.value.refusals[0].message
    assert "Experiment.based_on" in refused.value.refusals[0].action
    base = planted("hidden")[0].build().run(s(0.4))
    own = built.run(s(0.4), override={spring.k: N(1.7) / m(1.0)})
    assert all(np.array_equal(base.final[n][f], own.final[n][f]) for n in base.final for f in base.final[n])


def test_the_control_an_override_the_node_holds_in_a_leaf_runs_and_moves_the_run():
    exp, spring = planted("plain")
    base = planted("plain")[0].build().run(s(0.4))
    moved = exp.build().run(s(0.4), override={spring.k: K[1]})
    assert moved.overrides[0]["leaves"] == ["studytoy.spring:probe.k"]
    assert not all(np.array_equal(base.final[n][f], moved.final[n][f]) for n in base.final for f in base.final[n])


def test_a_host_controllers_entry_is_part_of_what_a_point_runs():
    """A controller called on the host between chunks makes its law from its entry, whose fields its law may read
    directly (here the write it holds before its first acts): two points that differ there alone are two computations,
    though their key, leaves and initial states are equal. The control: the same entry twice is one computation."""
    from control_cases import Ticker, spun_magnet

    from mime.study._point import build_point, same_run

    def point(held: float):
        settings, declarations, _ = spun_magnet(controller=Ticker(held=held)).declared_as()
        return build_point(0, settings, declarations, {}, "a test's point")

    first, again, other = point(-1.0), point(-1.0), point(-2.0)
    assert first.key == other.key and not first.refused
    assert same_run(first, again) and not same_run(first, other)
