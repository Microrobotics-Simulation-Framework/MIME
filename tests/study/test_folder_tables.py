"""The long table, CSV, NPZ and the grid of a study's result: the table has one row per member and sample, the coordinate
columns first with their units, ``member`` and ``group``, then the member's own columns and the flags, members of two
lengths in one table; ``grid()`` reshapes a full product and refuses a set of members that is not one; ``order`` reads
three levels of an axis and refuses what is not."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from folder_rigs import RECORDED, toy_study

from mime import Study
from mime.build._refusals import ArgumentRefused
from mime.study import DT, along
from mime.units import TIME, VELOCITY, N, Quantity, m, s

OBSERVATION = "toy.recorder:probe"


@pytest.fixture(scope="module")
def result():
    return toy_study().run()


def test_the_table_is_long_with_coordinates_first_and_two_lengths_in_one(result):
    table = result.table()
    names = list(table)
    assert names[:4] == ["toy.spring:probe.k [kg s^-2]", "dt [s]", "member", "group"]
    assert "time [s]" in names and "flags" in names and names.index("time [s]") > names.index("group")
    assert len(table["member"]) == 3 * 20 + 3 * 40
    for i in range(6):
        own = result.member(i).table()
        rows = table["member"] == i
        assert np.array_equal(table["time [s]"][rows], own["time [s]"])
        assert np.array_equal(table["toy.recorder:probe.last[1] [m]"][rows], own["toy.recorder:probe.last[1] [m]"])
        assert np.all(table["dt [s]"][rows] == result.coordinates["dt"].value[i])  # type: ignore[union-attr]
        assert np.all(table["group"][rows] == result.groups[i])
    assert table.units["dt [s]"] == "s" and table.units["toy.recorder:probe.last[1] [m]"] == "m"


def test_csv_and_npz_hold_what_the_table_and_the_members_hold(result, tmp_path):
    path = result.to_csv(tmp_path / "t.csv")
    with path.open(newline="") as file:
        rows = list(csv.reader(file))
    table = result.table()
    assert rows[0] == list(table) and len(rows) == 1 + len(table["member"])
    assert float(rows[1][0]) == table["toy.spring:probe.k [kg s^-2]"][0] and int(rows[-1][2]) == 5
    with np.load(result.to_npz(tmp_path / "s.npz")) as held:
        assert held["points"].tolist() == list(range(6)) and held["groups"].tolist() == list(result.groups)
        assert np.array_equal(held["coordinates/dt"], result.coordinates["dt"].value)  # type: ignore[union-attr]
        assert np.array_equal(held["m3/series/toy.recorder:probe.last/values"],
                              result.member(3)[OBSERVATION].last.raw)
        assert '"format": "mime-study-outputs"' in str(held["manifest"])


def test_the_grid_reshapes_a_product_and_refuses_what_is_not_one(result):
    grid = result.grid()
    assert grid.shape == (3, 2) and grid.index.tolist() == [[0, 1], [2, 3], [4, 5]]
    samples = result[OBSERVATION].samples
    assert grid.reshape(samples).shape == (3, 2) and grid.reshape(result.coordinates["dt"]).tolist()[2] == [0.02, 0.01]
    exp = toy_study().plan  # noqa: F841 - the study of the fixture is a product; this one is a list of three rows
    from folder_rigs import toy
    spring = toy().handle("toy.spring:probe")
    rows = Study(toy()).sweep(along({spring.k: [N(1.1) / m(1.0), N(1.7) / m(1.0), N(2.3) / m(1.0)],  # type: ignore[union-attr]
                                     DT: [s(0.02), s(0.01), s(0.02)]}), duration=s(0.2), record=[RECORDED]).run()
    with pytest.raises(ArgumentRefused) as refused:
        rows.grid()
    assert refused.value.codes() == ["study.not-a-grid"] and "combinations are not there" in refused.value.refusals[0].message


def test_the_order_is_read_from_three_levels_of_one_axis_and_refused_otherwise(result):
    levels = toy_study(stiffnesses=(1.7,), steps=(0.04, 0.02, 0.01)).run()
    manufactured = Quantity.of(2.5 + 0.3 * levels.coordinates["dt"].value ** 2, VELOCITY)  # type: ignore[union-attr]
    found = levels.order(manufactured, along=DT)
    assert abs(found.order - 2.0) < 1e-9 and abs(float(np.asarray(found.extrapolated.value)) - 2.5) < 1e-9
    assert levels.order(manufactured, along="dt").order == found.order
    with pytest.raises(ArgumentRefused) as flat:
        levels.order(f"{OBSERVATION}.samples", along=DT)  # a real output that does not move with the step
    assert flat.value.codes() == ["study.order"] and "one side" in flat.value.refusals[0].message
    with pytest.raises(ArgumentRefused) as others:
        result.order(Quantity.of(np.zeros(6), VELOCITY), along=DT)  # the stiffness varies too
    assert "varies as well" in others.value.refusals[0].message
    with pytest.raises(ArgumentRefused) as unknown:
        levels.order(manufactured, along="toy.spring:probe.k")
    assert unknown.value.codes() == ["study.order"] and TIME is not None
