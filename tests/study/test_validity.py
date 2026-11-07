"""Gate 4: validity per point. A sweep of script A's body's start across the tube's wall refuses each point that starts
the body outside the water (``state.outside-region``, a validity limit, refused in every profile), every refused point
at once with every reason; the study is refused for them unless ``drop_refused=True``, which runs the rest and records
what it dropped on the result. The control: the same sweep in a narrower tube refuses a point that was inside."""

from __future__ import annotations

import dataclasses

import pytest
from script_a import reproduction_arm

from mime import Study
from mime.build import BuildRefused, ThingDeclaration
from mime.experiment import Experiment
from mime.scene import Tube, X
from mime.study import grid
from mime.units import mm, s

STARTS = [mm((0.0, 0.0, -1.0)), mm((0.0, 0.0, -3.0)), mm((0.0, 0.0, -3.4)), mm((0.0, 0.0, -4.0))]


def _sweep(experiment, *, drop=False):
    return Study(experiment).sweep(grid({"umr.initial.position": STARTS}), duration=s(0.01), drop_refused=drop)


def _profiled(profile: str, inner=None) -> Experiment:
    settings, declarations, _ = reproduction_arm().declared_as()
    held = []
    for declaration in declarations:
        if isinstance(declaration, ThingDeclaration) and declaration.thing.name == "tube" and inner is not None:
            declaration = ThingDeclaration(dataclasses.replace(declaration.thing, geometry=Tube(
                axis=+X, inner_radius=inner, wall=mm(1.6), length=mm(100.0))))
        held.append(declaration)
    return Experiment.declared_from(dataclasses.replace(settings, profile=profile), held)


def _outside(point) -> bool:
    return any(refusal.code == "state.outside-region" for refusal in point.refusals)


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_points_outside_the_water_are_refused_in_every_profile_all_at_once(profile):
    plan = _sweep(_profiled(profile)).plan()
    assert [_outside(point) for point in plan.points] == [False, False, True, True], plan.text()
    with pytest.raises(BuildRefused) as refused:
        _sweep(_profiled(profile)).run()
    message = refused.value.refusals[0].message
    assert refused.value.codes() == ["study.refused-points"]
    assert "point 2" in message and "point 3" in message and "state.outside-region" in message


def test_drop_refused_runs_the_rest_and_records_what_it_dropped():
    result = _sweep(_profiled("exploratory"), drop=True).run()
    assert len(result) == 2 and result.points == (0, 1)
    assert [held.index for held in result.dropped] == [2, 3]
    assert all(any("state.outside-region" in line for line in held.refusals) for held in result.dropped)
    assert "DROPPED 2 refused points" in result.text()


def test_the_control_a_narrower_tube_refuses_a_point_that_was_inside():
    plan = _sweep(_profiled("exploratory", inner=mm(2.9))).plan()
    assert _outside(plan.points[1]) and not _outside(plan.points[0]), plan.text()
