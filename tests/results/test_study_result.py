"""``StudyResult`` in memory: one member axis, the coordinates with their units, each output at the end and each
recorded series stacked over the members, each member a ``RunResult``, and the members' flags as one union in which a
flag a member's build does not declare reads "not declared", never "not raised"."""

from __future__ import annotations

import numpy as np
import pytest
from script_a import reproduction_arm

from mime import Study
from mime.build._refusals import ArgumentRefused
from mime.results import NOT_DECLARED, RunResult, StackedSeries, StudyResult
from mime.study import DT, grid
from mime.units import FREQUENCY, Hz, mm_per_s, s

SPEED = "swim_speed:umr.speed"


@pytest.fixture(scope="module")
def result() -> StudyResult:
    exp = reproduction_arm()
    field = exp.handle("field")
    return Study(exp).sweep(grid({field.commands.frequency: [Hz(10.0), Hz(20.0), Hz(30.0)]}),  # type: ignore[union-attr]
                            duration=s(0.05), record=[SPEED], every=s(0.005)).run()


def test_one_member_axis_with_its_coordinates_in_their_units(result):
    assert len(result) == 3 and result.groups == (0, 0, 0) and result.points == (0, 1, 2)
    frequency = result.coordinates["field.commands.frequency"]
    assert frequency.dimension == FREQUENCY and np.array_equal(frequency.value, [10.0, 20.0, 30.0])


def test_outputs_stack_over_the_members_and_each_member_is_a_run_result(result):
    mean = result[SPEED.rpartition(".")[0]].mean_speed
    assert mean.value.shape == (3,)
    assert all(isinstance(result.member(i), RunResult) for i in range(3))
    assert np.allclose(mean.value, [result.member(i)["swim_speed:umr"].mean_speed.quantity.value for i in range(3)])
    speed = result["swim_speed:umr"].speed
    assert isinstance(speed, StackedSeries) and speed.values.value.shape == (3, 10)
    assert np.array_equal(speed.values.value[1], result.member(1)["swim_speed:umr"].speed.values.value)
    assert speed.values.to(mm_per_s).shape == (3, 10)


def test_every_flag_is_in_the_union_with_each_members_reading(result):
    flags = result.flags
    assert flags and all(len(values) == 3 for values in flags.values())
    assert all(value in (True, False, None) for values in flags.values() for value in values)


def test_a_flag_a_members_build_does_not_declare_reads_not_declared_and_a_bad_index_is_refused(result):
    other = StudyResult("x", ["a"], {"a": [1, 2]}, [result.member(0), result.member(1)], [0, 1], [0, 1])
    other._members[1].layout = type(other._members[1].layout)(  # noqa: SLF001 - a member whose build declares no flag
        other._members[1].layout.experiment, other._members[1].layout.dt, other._members[1].layout.observations, ())
    flags = other.flags
    assert all(values[1] == NOT_DECLARED for values in flags.values()) and flags
    with pytest.raises(ArgumentRefused):
        result.member(7)


def test_series_on_different_time_axes_are_refused_with_the_route_to_each_member():
    result = Study(reproduction_arm()).sweep(grid({DT: [s(5e-4), s(2.5e-4)]}), duration=s(0.01),
                                             record=[SPEED]).run()
    with pytest.raises(ArgumentRefused) as refused:
        _ = result["swim_speed:umr"].speed
    assert "result.member(i)" in refused.value.refusals[0].action
