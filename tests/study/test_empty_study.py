"""A study with no member is refused, never run and read as a pass (UX-1's F3). Before, an axis with no values made a
study of zero points that "ran", and a study whose every point was refused (a mistake in the base experiment, which no
axis value could mend) ran under ``drop_refused=True`` as if each were one bad point; its result read "no flag raised in
any member" over zero members, which a reader takes for a pass.

- an empty axis of ``grid`` or ``along`` is refused where it is written (``study.axis-values``), naming the axis;
- a study left with no member by ``drop_refused`` is refused by ``run``, ``run_to`` and ``gradient`` (``study.no-member``)
  with every refused point's reason, and not as ``study.refused-points`` (which says to drop the points: it was done);
- without ``drop_refused`` the refusal stays ``study.refused-points``, which names what to do;
- a result of zero members says "0 members", not "no flag raised".

The controls are the same studies with one point that stands: they run, and a mix runs the rest."""

from __future__ import annotations

import pytest
from folder_rigs import RECORDED, toy
from script_a import reproduction_arm

from mime import Study
from mime.build import BuildRefused
from mime.build._refusals import ArgumentRefused
from mime.results import StudyResult
from mime.study import along, grid
from mime.units import N, m, mm, s


def spring_study() -> Study:
    spring = toy().handle("toy.spring:probe")
    return Study(toy()).sweep(grid({spring.k: [N(1.1) / m(1.0)]}), duration=s(0.04), record=[RECORDED])


@pytest.mark.parametrize("make", [grid, along], ids=["grid", "along"])
def test_an_empty_axis_is_refused_naming_it(make):
    with pytest.raises(ArgumentRefused) as refused:
        make({"toy.spring:probe.k": []})
    (held,) = refused.value.refusals
    assert held.code == "study.axis-values" and "no values for the axis toy.spring:probe.k" in held.message
    assert "at least one value" in held.action


def test_one_empty_axis_among_full_ones_is_refused_and_control_full_axes_make_points():
    with pytest.raises(ArgumentRefused):
        grid({"a": [1.0, 2.0], "b": []})
    assert len(grid({"a": [1.0, 2.0], "b": [3.0]})) == 2
    assert len(along({"a": [1.0, 2.0], "b": [3.0, 4.0]})) == 2


#: Script A's body started across the tube's wall: the first two start inside the water, the last two outside it
#: (``state.outside-region``, a refusal of the point, in every profile).
INSIDE = [mm((0.0, 0.0, -1.0)), mm((0.0, 0.0, -3.0))]
OUTSIDE = [mm((0.0, 0.0, -3.4)), mm((0.0, 0.0, -4.0))]


def starts(positions, *, drop: bool) -> Study:
    return Study(reproduction_arm()).sweep(grid({"umr.initial.position": positions}), duration=s(0.01),
                                           drop_refused=drop)


def test_every_point_refused_is_refused_as_such_without_drop_and_as_no_member_with_it(tmp_path):
    with pytest.raises(BuildRefused) as plain:
        starts(OUTSIDE, drop=False).run()
    assert plain.value.codes() == ["study.refused-points"]
    with pytest.raises(BuildRefused) as dropped:
        starts(OUTSIDE, drop=True).run()
    assert dropped.value.codes() == ["study.no-member"]
    (held,) = dropped.value.refusals
    assert "all 2 of the study's points are refused, and drop_refused leaves none to run" in held.message
    assert "point 0" in held.message and "point 1" in held.message and "state.outside-region" in held.message
    assert "applies to every point" in held.action
    with pytest.raises(BuildRefused) as folder:
        starts(OUTSIDE, drop=True).run_to(tmp_path / "study")
    assert folder.value.codes() == ["study.no-member"] and not (tmp_path / "study").exists()


def test_control_one_point_that_stands_runs_the_rest_and_the_dropped_are_recorded():
    result = starts([*INSIDE[:1], *OUTSIDE], drop=True).run()
    assert len(result) == 1 and [held.index for held in result.dropped] == [1, 2]
    assert "DROPPED 2 refused points" in result.text() and "0 members" not in result.text()


def test_a_result_of_no_members_says_so_and_does_not_read_as_a_pass():
    empty = StudyResult("toy", ["toy.spring:probe.k"], {"toy.spring:probe.k": []}, [], [], [])
    text = empty.text()
    assert "0 members" in text and "no flag raised in any member" not in text


def test_control_a_result_of_members_that_raised_nothing_says_no_flag_raised(tmp_path):
    result = spring_study().run()
    assert len(result) == 1 and "no flag raised in any member" in result.text()
