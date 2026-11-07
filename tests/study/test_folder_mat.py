"""``StudyResult.to_mat``: gate 2 of S2c-1. The file read back by ``scipy.io.loadmat`` holds the coordinates, every
output and the flags equal to the result's, the member axis first, with the units present and the name map in the file.
MATLAB itself has not been run on a file written here, so this bounds what scipy's reader sees and nothing of MATLAB's.
Controls (each caught in every trial, the detection rate asserted): a unit dropped from a coordinate or an output, and a
member axis transposed in each array that has one."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from folder_rigs import toy_study
from scipy.io import loadmat

from mime.results._mat import matlab_name, write_variables
from mime.results._study_mat import variables_of_study

OBSERVATION = "toy.recorder:probe"


def text(value) -> str:
    """A text read back: ``loadmat`` reads an empty text as an empty array."""
    return "" if np.size(value) == 0 else str(value)


@pytest.fixture(scope="module")
def result():
    return toy_study().run()


def problems(path, result) -> list[str]:
    """Everything the file holds that is not the result's, by name: [] for a faithful file."""
    held = loadmat(path, simplify_cells=True)
    found: list[str] = []
    n = len(result)
    for axis, values in result.coordinates.items():
        entry = held["coordinates"][matlab_name(axis)]
        if not np.array_equal(np.atleast_1d(entry["value"]), np.asarray(values.value)):
            found.append(f"coordinate {axis} values")
        if text(entry["unit"]) == "":
            found.append(f"coordinate {axis} unit")
    if not np.array_equal(np.atleast_1d(held["groups"]), result.groups):
        found.append("groups")
    for name, array in (("last", None), ("samples", None)):
        output = held[matlab_name(OBSERVATION)][name]
        if text(output["unit"]) == "":
            found.append(f"{name} unit")
    samples = held[matlab_name(OBSERVATION)]["samples"]["value"]
    expected = np.asarray([result.member(i)[OBSERVATION].samples.quantity.value for i in range(n)], dtype=np.float64)
    if np.shape(samples) != expected.shape or not np.array_equal(samples, expected):
        found.append("samples values")
    last = held[matlab_name(OBSERVATION)]["last"]
    for i in range(n):
        series = result.member(i)[OBSERVATION].last
        if not (np.array_equal(np.asarray(last["time"][i]), series.time.value)
                and np.array_equal(np.asarray(last["values"][i]), series.values.value)):
            found.append(f"last member {i}")
    if json.loads(str(held["manifest"])) != json.loads(json.dumps(result.manifest())):
        found.append("manifest")
    if held["name_map"]["axes"][matlab_name("dt")] != "dt":
        found.append("name map")
    return found


def test_the_file_holds_the_coordinates_outputs_and_flags_with_their_units_member_axis_first(result, tmp_path):
    path = result.to_mat(tmp_path / "study.mat")
    assert problems(path, result) == []
    held = loadmat(path, simplify_cells=True)
    assert held["study"]["members"] == 6 and held["coordinates"]["toy_spring_probe_k"]["unit"] == "kg s^-2"
    assert held["coordinates"]["dt"]["unit"] == "s" and np.shape(held["coordinates"]["dt"]["value"]) == (6,)
    assert np.shape(held["toy_recorder_probe"]["samples"]["value"]) == (6,)
    assert held["toy_recorder_probe"]["last"]["shared_time_axis"] == 0, "ragged: the members' own, as cells"
    assert "toy_recorder_probe" in held["name_map"]["observations"]


def test_a_shared_time_axis_is_stacked_members_first(tmp_path):
    result = toy_study(steps=()).run()
    held = loadmat(result.to_mat(tmp_path / "s.mat"), simplify_cells=True)
    last = held["toy_recorder_probe"]["last"]
    assert last["shared_time_axis"] == 1 and np.shape(last["values"]) == (3, 20, 3) and np.shape(last["time"]) == (20,)
    assert np.array_equal(last["values"][1], result.member(1)[OBSERVATION].last.values.value)


def _written(tmp_path, result, change, name):
    variables = copy.deepcopy(variables_of_study(result))
    change(variables)
    return write_variables(variables, tmp_path / name)


def test_a_dropped_unit_and_a_transposed_member_axis_are_caught_every_time(result, tmp_path):
    controls = []
    for axis in result.axes:
        controls.append((f"unit of {axis}", lambda v, axis=axis: v["coordinates"][matlab_name(axis)].update(unit="")))
    for output in ("last", "samples"):
        controls.append((f"unit of {output}", lambda v, output=output: v[matlab_name(OBSERVATION)][output].update(unit="")))
    controls.append(("coordinate transposed", lambda v: v["coordinates"]["dt"].update(value=v["coordinates"]["dt"]["value"][::-1])))
    controls.append(("groups transposed", lambda v: v.update(groups=v["groups"][::-1].copy())))
    # Not "samples" reversed: it reads 10 in every member, so reversing it changes nothing, and a control that cannot
    # differ would prove nothing. The members' series are of two lengths, so reversing them is seen.
    controls.append(("series reversed", lambda v: v[matlab_name(OBSERVATION)]["last"].update(
        values=v[matlab_name(OBSERVATION)]["last"]["values"][::-1].copy(),
        time=v[matlab_name(OBSERVATION)]["last"]["time"][::-1].copy())))
    caught = 0
    for i, (name, change) in enumerate(controls):
        found = problems(_written(tmp_path, result, change, f"c{i}.mat"), result)
        caught += bool(found)
        print(name, "->", found)
    print("controls caught:", caught, "of", len(controls))
    assert caught == len(controls)


def shared_problems(path, result) -> list[str]:
    last = loadmat(path, simplify_cells=True)[matlab_name(OBSERVATION)]["last"]
    stacked = result["toy.recorder:probe"].last
    found = []
    if np.shape(last["values"]) != stacked.values.value.shape or not np.array_equal(last["values"], stacked.values.value):
        found.append("values")
    if not np.array_equal(last["time"], stacked.time.value):
        found.append("time")
    return found


def test_the_member_axis_of_a_stacked_series_is_first_and_a_transposed_one_is_caught(tmp_path):
    result = toy_study(steps=()).run()
    assert shared_problems(result.to_mat(tmp_path / "ok.mat"), result) == []
    caught = 0
    for i, swap in enumerate((lambda a: np.swapaxes(a, 0, 1), lambda a: np.swapaxes(a, 1, 2),
                              lambda a: a.transpose(2, 1, 0))):
        def change(v, swap=swap):
            v[matlab_name(OBSERVATION)]["last"]["values"] = np.ascontiguousarray(swap(v[matlab_name(OBSERVATION)]["last"]["values"]))
        found = shared_problems(_written(tmp_path, result, change, f"t{i}.mat"), result)
        caught += bool(found)
    print("transposed member axes caught:", caught, "of 3")
    assert caught == 3
