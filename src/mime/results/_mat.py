"""A run's result as a MATLAB v5 file (:meth:`RunResult.to_mat`): the declared outputs by name with their units, the
series with their time axes, the flags and what the run kept of them, the manifest as a JSON string and the map from
MATLAB's names back to the run's. Written through ``scipy.io.savemat``, which MIME already depends on, and read back
in its tests by ``scipy.io.loadmat``; MATLAB itself has not been run on a file written here.

MATLAB names are letters, digits and underscores, a letter first, at most 63 long. An id such as
``swim_speed:umr.speed`` is not one, so each name is mapped by one rule: a character outside letters, digits and
underscore becomes ``_``, and a name that does not begin with a letter gains an ``x`` in front. The map is in the file
(``name_map``), and two names that map to one, a name that maps to more than 63 characters, and an observation whose
name is one of the file's own variables are refused, naming them.

The v5 format counts a variable's bytes in 32 bits, and MATLAB reads a variable of 2 GiB or more only from version 7.3
files. ``savemat`` itself fails past 4 GiB with an ``OverflowError`` and leaves a stub of the file (measured, scipy
1.18.1, one array of 4 GiB and 8 kB), and the documented limit of a v5 variable is 2 GiB; the check here is
before anything is written, at 2 GiB a variable, and refuses naming the variable and NPZ, which has no such limit.
Version 7.3 (HDF5) is not written."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.io

from mime.registry import OutputKind
from mime.results._layout import OutputLayout
from mime.results._series import FlagRead, Reading, Series, flags_text

if TYPE_CHECKING:
    from mime.results._result import RunResult

#: The variables of the file that are not an observation's.
RESERVED = ("manifest", "run", "flags", "name_map")
#: The longest MATLAB name.
LONGEST_NAME = 63
#: The bytes a variable may hold: MATLAB reads v5 variables only below 2 GiB.
VARIABLE_BYTES = 2**31

type Struct = dict[str, Any]

#: scipy's writer, whose parameters read as partly unknown to a strict checker.
_savemat: Any = cast(Any, scipy.io).savemat


def matlab_name(name: str) -> str:
    """``name`` as a MATLAB identifier: anything outside letters, digits and underscore becomes ``_``, and a name that
    does not begin with a letter gains an ``x`` before it."""
    mapped = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return mapped if mapped[:1].isascii() and mapped[:1].isalpha() else "x" + mapped


def names_of(kind: str, originals: Sequence[str], *, taken: Sequence[str] = ()) -> dict[str, str]:
    """Each of ``originals`` (the ``kind`` of name they are) mapped to its MATLAB name, refusing a name that maps to
    more than 63 characters, to one of ``taken``, or to what another maps to."""
    found: dict[str, str] = {}
    owner: dict[str, str] = {}
    for original in originals:
        mapped = matlab_name(original)
        if len(mapped) > LONGEST_NAME:
            raise ValueError(f"the {kind} {original!r} maps to the MATLAB name {mapped!r}, {len(mapped)} characters; "
                             f"MATLAB's longest is {LONGEST_NAME}: write NPZ (to_npz), which keeps the name as it is")
        if mapped in taken:
            raise ValueError(f"the {kind} {original!r} maps to the MATLAB name {mapped!r}, which is one of the "
                             f"file's own variables ({', '.join(RESERVED)}): write NPZ (to_npz), or rename it")
        if mapped in owner and owner[mapped] != original:
            raise ValueError(f"the {kind}s {owner[mapped]!r} and {original!r} map to the same MATLAB name "
                             f"{mapped!r}: write NPZ (to_npz), which keeps them apart, or rename one")
        owner[mapped] = original
        found[original] = mapped
    return found


def _tri(known: bool | None) -> str:
    """A flag's state as text: unknown stays ``unknown``, and is never ``false``."""
    return "unknown" if known is None else ("true" if known else "false")


def _number(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)


def _flag_read(read: FlagRead) -> Struct:
    return {"path": read.path, "was_raised": _tri(read.was_raised), "raised_steps": _number(read.raised),
            "steps": _number(read.steps), "first": _number(read.first), "last": _number(read.last),
            "raised_at_last_step": _tri(read.at_last), "recorded_from": _number(read.recorded_from),
            "text": read.text(meaning=True)}


def _flags_of(reads: Sequence[FlagRead], names: Mapping[str, str]) -> Struct:
    """A window's flags, by MATLAB name: only the flags raised in it or not known for it are there."""
    return {"flags": {names[read.path]: _flag_read(read) for read in reads}} if reads else {}


def _output(result: RunResult, output: OutputLayout, names: Mapping[str, str]) -> Struct:
    held: Struct = {"path": output.path, "name": output.name, "node": output.node, "field": output.field,
                    "kind": output.kind.value, "unit": output.unit, "dimension": str(output.dimension),
                    "shape": np.asarray(output.shape, dtype=np.int64), "description": output.description}
    if output.reduces is not None:
        held.update(reduces=output.reduces)
    if output.rule is not None:
        held.update(rule=output.rule.value)
    if output.kind is OutputKind.AT_END:
        reading = cast(Reading, result.output(output))
        held.update(value=np.array(reading.value, dtype=np.float64), window=np.array([reading.start, reading.end]),
                    window_unit="s", flags_text=flags_text(reading.flags), **_flags_of(reading.flags, names))
    elif output.path in result.recorded:
        series = cast(Series, result.output(output))
        held.update(recorded=True, time=np.array(series.time.value, dtype=np.float64), time_unit="s",
                    steps=series.steps.astype(np.int64), every=series.every,
                    values=np.array(series.values.value, dtype=np.float64),
                    window=np.array([series.start, series.end]), window_unit="s",
                    flags_text=flags_text(series.flags), **_flags_of(series.flags, names))
    else:
        held.update(recorded=False)
    return held


def _flag_variable(result: RunResult, names: Mapping[str, str]) -> Struct:
    """Every flag the run's entries declare: what it means, what the run kept of it, and where it was raised."""
    held: Struct = {}
    for flag in result.layout.flags:
        track = result.tracks.get(flag.path)
        entry: Struct = {"path": flag.path, "entry": flag.owner, "label": flag.entry, "node": flag.node,
                         "field": flag.field, "meaning": flag.meaning,
                         "recorded": "every step" if track is not None and track.raised is not None
                         else "the last step",
                         "was_raised": _tri(track.was_raised if track is not None else None),
                         "raised_at_last_step": _tri(track.at_last if track is not None else None)}
        if track is not None and track.raised is not None:
            steps = result.start_step + 1 + np.flatnonzero(track.raised).astype(np.int64)
            entry.update(raised_steps=steps, raised_times=steps.astype(np.float64) * result.dt, time_unit="s")
        held[names[flag.path]] = entry
    return held


def _run_variable(result: RunResult) -> Struct:
    manifest = result.manifest()
    stamp = result.layout.stamp
    return {"experiment": result.layout.experiment, "format": manifest["format"], "version": manifest["version"],
            "dt": result.dt, "steps": result.steps, "start_step": result.start_step, "start": manifest["start"],
            "end": manifest["end"], "time_unit": "s", "every": _number(result.every),
            "history": manifest["history"], "clock": result.clock.name, "reading": manifest["reading"],
            "profile": stamp.profile if stamp is not None else "not recorded",
            "stamp": stamp.text() if stamp is not None else "profile: not recorded (a run written before profiles)"}


def _bytes_of(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_bytes_of(held) for held in cast(Mapping[str, Any], value).values())
    if isinstance(value, str):
        return 2 * len(value)
    return 8


def variables_of(result: RunResult) -> dict[str, Any]:
    """The variables of the MATLAB file for ``result``, by MATLAB name, refused where a name cannot be mapped."""
    observations = names_of("observation", [held.id for held in result.layout.observations], taken=RESERVED)
    flags = names_of("flag", [flag.path for flag in result.layout.flags])
    outputs = {held.id: names_of("output", [output.name for output in held.outputs])
               for held in result.layout.observations}
    variables: dict[str, Any] = {"run": _run_variable(result),
                                 "manifest": json.dumps(result.manifest(), sort_keys=True)}
    if flags:
        variables["flags"] = _flag_variable(result, flags)
    for held in result.layout.observations:
        variables[observations[held.id]] = {outputs[held.id][output.name]: _output(result, output, flags)
                                            for output in held.outputs}
    variables["name_map"] = {
        "rule": "a character outside letters, digits and underscore becomes _, and a name that does not begin with a "
                "letter gains an x before it; this maps each MATLAB name to the run's",
        "observations": {mapped: original for original, mapped in observations.items()},
        "outputs": {observations[held.id]: {mapped: held.id + "." + original
                                            for original, mapped in outputs[held.id].items()}
                    for held in result.layout.observations},
        **({"flags": {mapped: original for original, mapped in flags.items()}} if flags else {})}
    for name, value in variables.items():
        size = _bytes_of(value)
        if size >= VARIABLE_BYTES:
            raise ValueError(f"the variable {name} would hold {size} bytes, past the {VARIABLE_BYTES} a MATLAB v5 "
                             f"file's variable may (MATLAB reads larger ones only from version 7.3 files, which "
                             f"MIME does not write): write NPZ (to_npz), which has no such limit, or record fewer "
                             f"series or a larger every=")
    return variables


def write_mat(result: RunResult, path: str | PathLike[str]) -> Path:
    """Write ``result`` to ``path`` (as it is named, with no ``.mat`` added) as a MATLAB v5 file."""
    variables = variables_of(result)
    target = Path(path)
    _savemat(str(target), variables, appendmat=False, long_field_names=True, oned_as="column")
    return target


__all__ = ["LONGEST_NAME", "RESERVED", "VARIABLE_BYTES", "matlab_name", "names_of", "variables_of", "write_mat"]
