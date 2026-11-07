"""A study's folder: what ``Study.run_to`` writes and ``StudyResult.read`` reads.

::

    study.json                  the manifest, format ``mime-study-outputs/1``: the base experiment, the axes, the points,
                                the groups, each member's status, overrides and content key, the dropped points
    study.mime.yaml             the study's spec file: the study run again from it is this study
    experiment/                 the base experiment, a saved folder, byte for byte what ``save()`` wrote
    groups/<k>/outputs.npz      every member of group k, its arrays under ``m<point>/`` and its manifest and report
                                under ``m<point>/manifest`` and ``m<point>/report`` (JSON text)
    groups/<k>/report.json      the build report's summary of the group's first member
    groups/<k>/experiment/      the group's own build as a saved folder, for a group whose compiled program differs
                                from group 0's

The folder is readable without MIME: JSON, YAML and NPZ. Its manifest is closed (a key it does not know is refused), its
layout is checked (a file it does not know is refused), and a newer format is refused as ``result.format`` refuses one.
A write is atomic file by file (a temporary file, then a rename), so a stopped run leaves a folder whose manifest says
which members are done."""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.units import Quantity, parse_unit, si, unit_symbol

STUDY_FORMAT = "mime-study-outputs"
STUDY_VERSION = 1
STUDY_FILE = "study.json"
SPEC_FILE = "study.mime.yaml"
BASE_DIR = "experiment"
GROUPS_DIR = "groups"
GROUP_OUTPUTS = "outputs.npz"
GROUP_REPORT = "report.json"
#: A temporary file a stopped write may leave; the layout accepts it and a resumed run removes it.
TEMPORARY = ".tmp"
#: A member's arrays in a group's NPZ are under ``m<point>/``.
MEMBER = re.compile(r"m(\d+)/")

PENDING, DONE = "pending", "done"

#: The manifest's keys at each level, closed: a key not here is refused, and so is one missing.
TOP = frozenset({"format", "version", "id", "experiment", "axes", "points", "run", "drop_refused", "dropped", "seed",
                 "groups", "members", "lineage", "reading"})
EXPERIMENT_KEYS = frozenset({"name", "input_hash", "profile"})
AXIS_KEYS = frozenset({"path", "kind", "unit", "reason"})
POINT_KEYS = frozenset({"index", "values"})
RUN_KEYS = frozenset({"duration", "steps", "record", "every", "batch"})
DROPPED_KEYS = frozenset({"index", "coordinates", "refusals"})
SEED_KEYS = frozenset({"policy", "seed", "note"})
GROUP_KEYS = frozenset({"index", "key", "reasons", "members", "batches", "batched", "compiles", "seconds",
                        "experiment"})
MEMBER_KEYS = frozenset({"member", "point", "group", "status", "content_key", "overrides"})
OVERRIDE_KEYS = frozenset({"path", "value", "reason"})
LINEAGE_KEYS = frozenset({"base", "spec", "mime"})

READING = ("each member's arrays are in groups/<group>/outputs.npz under m<point>/ (NumPy reads it); m<point>/manifest "
           "is the member's outputs.json as JSON text; a value is {value, unit}: value in SI, unit its symbol; "
           "members[].status is done or pending")


def refused(code: str, category: RefusalClass, message: str, action: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(code, category, message, action)])


def encode_value(value: object, path: str) -> dict[str, Any]:
    """An axis value as study.json holds it: a quantity as its SI magnitude and its unit's symbol, a number, a boolean
    or a string as it is. Refused (``study.unsavable-value``) for anything else, naming the axis."""
    if isinstance(value, Quantity):
        magnitude = value.value
        held: object = np.asarray(magnitude, dtype=np.float64).tolist() if isinstance(magnitude, np.ndarray) else \
            si(value, value.dimension, what=path)
        return {"value": [float(v) for v in cast(tuple[float, ...], held)] if isinstance(held, tuple) else held,
                "unit": unit_symbol(value.dimension)}
    if isinstance(value, (bool, int, float, str)):
        return {"value": value, "unit": None}
    raise refused("study.unsavable-value", RefusalClass.USER,
                  f"the value of the axis {path} is a {type(value).__name__}, which a study folder cannot hold: it holds "
                  f"quantities, numbers, booleans and strings",
                  "sweep quantities (`Hz(10.0)`), numbers or names; a value that is code belongs in an experiment of its "
                  "own, `Experiment.based_on(...)`")


def decode_value(record: Mapping[str, Any]) -> object:
    """The value :func:`encode_value` wrote."""
    unit = record["unit"]
    held = record["value"]
    if unit is None:
        return held
    magnitude = tuple(float(v) for v in cast(Sequence[float], held)) if isinstance(held, list) else float(held)
    return Quantity(magnitude, parse_unit(str(unit)))


def write_json(path: Path, document: object) -> None:
    """``document`` written atomically (a temporary file, then a rename), sorted and indented, so that equal documents
    are equal bytes."""
    temporary = path.with_name(path.name + TEMPORARY)
    temporary.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8",
                         newline="\n")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------------------------------------------
# NPZ files that are the same bytes for the same arrays


#: The time every entry of a study's NPZ carries (a zip entry's header holds one): the format's earliest, so that two
#: writes of the same arrays are the same bytes.
EPOCH = (1980, 1, 1, 0, 0, 0)


def _entry(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name + ".npy", date_time=EPOCH)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def _array_bytes(array: NDArray[Any]) -> bytes:
    import io

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_entries(path: Path, new: Mapping[str, NDArray[Any]], drop_prefixes: Sequence[str] = ()) -> None:
    """The NPZ at ``path`` with ``new`` added to what it holds (the entries of ``drop_prefixes`` removed first, so that a
    member run again replaces its old arrays), its entries sorted by name and carrying no clock, so that the same
    arrays are the same bytes however the file was reached. Written to a temporary file and renamed, one old entry in
    memory at a time."""
    temporary = path.with_name(path.name + TEMPORARY)
    old = zipfile.ZipFile(path) if path.is_file() else None
    try:
        kept = {} if old is None else {info.filename.removesuffix(".npy"): info for info in old.infolist()
                                       if not any(info.filename.startswith(prefix) for prefix in drop_prefixes)}
        names = sorted({*kept, *new})
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as out:
            for name in names:
                if name in new:
                    out.writestr(_entry(name), _array_bytes(new[name]))
                else:
                    assert old is not None
                    out.writestr(_entry(name), old.read(kept[name]))
    finally:
        if old is not None:
            old.close()
    os.replace(temporary, path)


class GroupArchive:
    """A group's ``outputs.npz``, opened once and read entry by entry: the keys it holds (indexed by member), and a
    member's view of it as the archive a :class:`~mime.results.RunResult` reads from."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        self._by_member: dict[int, list[str]] = {}
        for name in self._file().namelist():
            key = name.removesuffix(".npy")
            found = MEMBER.match(key)
            if found is not None:
                self._by_member.setdefault(int(found[1]), []).append(key)

    def _file(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.path)
        return self._zip

    def points(self) -> list[int]:
        return sorted(self._by_member)

    def keys_of(self, point: int) -> list[str]:
        return self._by_member.get(point, [])

    def __getitem__(self, key: str) -> NDArray[Any]:
        with self._file().open(key + ".npy") as file:
            return np.asarray(np.lib.format.read_array(file, allow_pickle=False))

    def member(self, point: int) -> MemberArchive:
        return MemberArchive(self, point)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __del__(self) -> None:
        self.close()


class MemberArchive:
    """One member's arrays of a :class:`GroupArchive`, under the keys a run's own NPZ uses."""

    def __init__(self, group: GroupArchive, point: int) -> None:
        self._group, self._prefix = group, f"m{point}/"
        self.files: list[str] = [key.removeprefix(self._prefix) for key in group.keys_of(point)]

    def __getitem__(self, key: str) -> NDArray[Any]:
        return self._group[self._prefix + key]

    def many(self, keys: Sequence[str]) -> dict[str, NDArray[Any]]:
        return {key: self[key] for key in keys}


# ---------------------------------------------------------------------------------------------------------------
# The manifest, closed; the layout, checked


def _closed(where: str, held: Mapping[str, Any], keys: frozenset[str]) -> None:
    extra, missing = sorted(set(held) - keys), sorted(keys - set(held))
    if extra or missing:
        raise refused("study.manifest", RefusalClass.USER,
                  f"{STUDY_FILE}'s {where} " + (f"holds {', '.join(map(repr, extra))}, which this format does not"
                                                     if extra else "") + (" and " if extra and missing else "")
                      + (f"lacks {', '.join(map(repr, missing))}" if missing else ""),
                      f"a study folder's manifest is closed: write it with `Study.run_to(folder)`, and read it with the MIME "
                      f"that wrote it if it came from another")


def check_document(document: object) -> dict[str, Any]:
    """The manifest ``document`` if it is a study's, in a version this MIME reads and with exactly its keys; refused
    otherwise (``study.format`` for another format or a newer version, ``study.manifest`` for a key it does not know)."""
    if not isinstance(document, dict):
        raise refused("study.format", RefusalClass.USER, f"{STUDY_FILE} is not a JSON object",
                      "read a folder `Study.run_to(folder)` wrote: `StudyResult.read(folder)`")
    held = cast(dict[str, Any], document)
    if held.get("format") != STUDY_FORMAT or held.get("version") != STUDY_VERSION:
        raise refused("study.format", RefusalClass.USER,
                  f"a study manifest of {held.get('format')!r} version {held.get('version')!r}; this MIME reads "
                      f"{STUDY_FORMAT!r} version {STUDY_VERSION}",
                      "read it with the MIME that wrote it (`pip install mime-engine==<version>`), or run the study "
                      "again with this one: `mime study run <spec> --out <folder>`")
    _closed("top level", held, TOP)
    _closed("experiment", held["experiment"], EXPERIMENT_KEYS)
    _closed("run", held["run"], RUN_KEYS)
    _closed("seed", held["seed"], SEED_KEYS)
    _closed("lineage", held["lineage"], LINEAGE_KEYS)
    for where, keys, rows in (("axes", AXIS_KEYS, held["axes"]), ("points", POINT_KEYS, held["points"]),
                              ("dropped", DROPPED_KEYS, held["dropped"]), ("groups", GROUP_KEYS, held["groups"]),
                              ("members", MEMBER_KEYS, held["members"])):
        for row in cast(list[dict[str, Any]], rows):
            _closed(f"{where} entry", row, keys)
    for row in cast(list[dict[str, Any]], held["members"]):
        for override in cast(list[dict[str, Any]], row["overrides"]):
            _closed("override", override, OVERRIDE_KEYS)
        if row["status"] not in (PENDING, DONE):
            raise refused("study.manifest", RefusalClass.USER,
                          f"a member's status is {row['status']!r}, and a status is {PENDING!r} or {DONE!r}",
                          "write the folder with `Study.run_to(folder)`")
    return held


def layout_problems(root: Path) -> list[str]:
    """Every path under ``root`` that a study folder does not hold, as text; ``[]`` for a folder with only its own."""
    found: list[str] = []
    top = {STUDY_FILE, SPEC_FILE, BASE_DIR, GROUPS_DIR}
    for child in sorted(root.iterdir()):
        if child.name.endswith(TEMPORARY):
            continue
        if child.name not in top:
            found.append(child.name + ("/" if child.is_dir() else ""))
    groups = root / GROUPS_DIR
    if groups.is_dir():
        for child in sorted(groups.iterdir()):
            if not (child.is_dir() and child.name.isdigit()):
                found.append(f"{GROUPS_DIR}/{child.name}")
                continue
            for inside in sorted(child.iterdir()):
                if inside.name not in (GROUP_OUTPUTS, GROUP_REPORT, BASE_DIR) and not inside.name.endswith(TEMPORARY):
                    found.append(f"{GROUPS_DIR}/{child.name}/{inside.name}")
    return found


def check_layout(root: Path) -> None:
    """Refuses (``study.stray-file``) a file or folder a study folder does not hold, naming each."""
    stray = layout_problems(root)
    if stray:
        raise refused("study.stray-file", RefusalClass.USER,
                  f"{root} holds {', '.join(stray)}, which a study folder does not: it holds {STUDY_FILE}, {SPEC_FILE}, "
                      f"{BASE_DIR}/ and {GROUPS_DIR}/<k>/ ({GROUP_OUTPUTS}, {GROUP_REPORT}, {BASE_DIR}/)",
                      "move it out of the study folder, beside it and not in it, and read the folder again with "
                      "`StudyResult.read(folder)`")


def iter_groups(root: Path) -> Iterator[tuple[int, Path]]:
    groups = root / GROUPS_DIR
    if groups.is_dir():
        for child in sorted(groups.iterdir(), key=lambda each: each.name):
            if child.is_dir() and child.name.isdigit():
                yield int(child.name), child


__all__ = ["BASE_DIR", "DONE", "GROUPS_DIR", "GROUP_OUTPUTS", "GROUP_REPORT", "PENDING", "READING", "SPEC_FILE",
           "STUDY_FILE", "STUDY_FORMAT", "STUDY_VERSION", "TEMPORARY", "GroupArchive", "MemberArchive", "check_document",
           "check_layout", "decode_value", "encode_value", "iter_groups", "layout_problems", "refused", "write_entries",
           "write_json"]
