"""A study as a MATLAB v5 file (:meth:`StudyResult.to_mat`), through ``scipy.io.savemat``: the member axis first in every
array, the coordinates with their units, each declared output over the members with its unit, the flags by member, the
manifest as JSON text and the map from each MATLAB name to the study's. Names are mapped as a run's are
(:func:`mime.results._mat.names_of`); the limits and the way it is checked are the run's (``_mat.py``)."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.registry import OutputKind
from mime.results._mat import check_sizes, names_of, write_variables
from mime.results._series import Reading
from mime.units import Quantity, unit_symbol

if TYPE_CHECKING:
    from mime.results._result import RunResult
    from mime.results._study import StudyResult

#: The variables of the file that are not an observation's.
RESERVED = ("manifest", "study", "coordinates", "groups", "points", "flags", "name_map")

type Struct = dict[str, Any]


def _tri(value: bool | str | None) -> str:
    return "unknown" if value is None else value if isinstance(value, str) else ("true" if value else "false")


def _cells(arrays: list[NDArray[Any]]) -> NDArray[Any]:
    held = np.empty(len(arrays), dtype=object)
    for i, array in enumerate(arrays):
        held[i] = array
    return held


def _output(result: StudyResult, observation: str, output: Any, first: RunResult) -> Struct:
    held: Struct = {"path": output.path, "name": output.name, "unit": output.unit, "dimension": str(output.dimension),
                    "kind": output.kind.value, "description": output.description}
    members = list(result)
    if output.kind is OutputKind.AT_END:
        values = np.stack([np.asarray(r.quantity.value if r.observed else np.nan, dtype=np.float64) for r in
                           (cast(Reading, member[observation][output.name]) for member in members)])
        held["value"] = values
        return held
    recorded = [output.path in member.recorded for member in members]
    held["recorded"] = np.asarray(recorded, dtype=np.uint8)
    if not all(recorded):
        return held
    series = [member[observation][output.name] for member in members]
    times = [np.asarray(s.time.value, dtype=np.float64) for s in series]  # type: ignore[union-attr]
    values = [np.asarray(s.values.value, dtype=np.float64) for s in series]  # type: ignore[union-attr]
    held["time_unit"] = "s"
    if all(t.shape == times[0].shape and np.array_equal(t, times[0]) for t in times[1:]):
        held.update(time=times[0], values=np.stack(values), shared_time_axis=1)
    else:
        held.update(time=_cells(times), values=_cells(values), shared_time_axis=0)
    return held


def variables_of_study(result: StudyResult) -> dict[str, Any]:
    """The variables of the MATLAB file for ``result``, by MATLAB name."""
    members = list(result)
    if not members:
        raise ArgumentRefused([Refusal("study.no-member", RefusalClass.USER, "the study has no members to write",
                                       "run it first: `study.run_to(folder)`")])
    shared = [held.id for held in members[0].layout.observations
              if all(held.id in member.observations for member in members)]
    observations = names_of("observation", shared, taken=RESERVED)
    coordinate_names = names_of("axis", list(result.axes))
    flag_paths = list(result.flags)
    flags = names_of("flag", flag_paths)
    coordinates: Struct = {}
    for path, values in result.coordinates.items():
        if isinstance(values, Quantity):
            coordinates[coordinate_names[path]] = {"path": path, "unit": unit_symbol(values.dimension),
                                                   "value": np.asarray(values.value, dtype=np.float64)}
        else:
            coordinates[coordinate_names[path]] = {"path": path, "unit": "",
                                                   "value": np.asarray([str(v) for v in values], dtype=object)}
    variables: dict[str, Any] = {
        "study": {"experiment": result.experiment, "id": str(result.record.get("id", "")), "members": len(members),
                  "missing": len(result.missing), "dropped": len(result.dropped), "axes": ", ".join(result.axes),
                  "member_axis": "first: every array's first dimension is the member, in the order of study.points"},
        "coordinates": coordinates,
        "groups": np.asarray(result.groups, dtype=np.int64),
        "points": np.asarray(result.points, dtype=np.int64),
        "manifest": json.dumps(result.manifest(), sort_keys=True)}
    if flags:
        variables["flags"] = {flags[path]: {"path": path, "by_member": np.asarray([_tri(v) for v in values], dtype=object)}
                              for path, values in result.flags.items()}
    outputs: dict[str, dict[str, str]] = {}
    for held in members[0].layout.observations:
        if held.id not in observations:
            continue
        names = names_of("output", [output.name for output in held.outputs])
        outputs[observations[held.id]] = {names[output.name]: held.id + "." + output.name for output in held.outputs}
        variables[observations[held.id]] = {names[output.name]: _output(result, held.id, output, members[0])
                                            for output in held.outputs}
    variables["name_map"] = {
        "rule": "a character outside letters, digits and underscore becomes _, and a name that does not begin with a "
                "letter gains an x before it; this maps each MATLAB name to the study's",
        "observations": {mapped: original for original, mapped in observations.items()},
        "outputs": outputs, "axes": {mapped: original for original, mapped in coordinate_names.items()},
        **({"flags": {mapped: original for original, mapped in flags.items()}} if flags else {})}
    check_sizes(variables)
    return variables


def write_study_mat(result: StudyResult, path: str | PathLike[str]) -> Path:
    return write_variables(variables_of_study(result), path)


__all__ = ["RESERVED", "variables_of_study", "write_study_mat"]
