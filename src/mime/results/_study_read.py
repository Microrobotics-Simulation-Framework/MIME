"""Reading a study folder: its layout checked, its manifest closed and in a format this MIME reads, each group file that
holds a member marked done checked against the manifest (its sha256, and each such member's entries under the content key
the manifest gives it), and each member read from its group's NPZ when first asked for. A study that was stopped reads as
what it holds."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._result import RunResult, read_report
from mime.results._stage import StageRecord
from mime.results._study import Dropped, StudyResult
from mime.results._study_folder import (
    BASE_DIR,
    CONTENT_KEY,
    DONE,
    GROUP_OUTPUTS,
    GROUP_REPORT,
    GROUPS_DIR,
    STUDY_FILE,
    GroupArchive,
    check_document,
    check_layout,
    decode_value,
    file_sha256,
    temporary_of,
)


def _group_file_refused(message: str, where: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(
        "study.group-file", RefusalClass.USER, message,
        f"read the folder as its run wrote it: a group file copied from another folder, or changed, is not this study's. "
        f"Run the study into a new folder (`--out <folder>`), or move {where} out of this one and run the study into it "
        f"again, which runs that group's members again")])


def _counted(points: Sequence[int]) -> str:
    """``n member(s) (point(s) ...)``: the members a refusal names, counted."""
    plural = "s" if len(points) != 1 else ""
    shown = ", ".join(map(str, points[:8])) + (", ..." if len(points) > 8 else "")
    return f"{len(points)} member{plural} (point{plural} {shown})"


def _group_file(root: Path, group: Mapping[str, Any], done: Sequence[Mapping[str, Any]]) -> Path:
    """The file that holds ``group``'s members, checked against the manifest: each of ``done`` (its members marked
    done) has its entries in it under the content key the manifest gives it, and its sha256 is the one the manifest
    records. That is ``outputs.npz``, or the file a run staged and recorded and stopped before renaming (the next run
    renames it). Refused (``study.group-file``) otherwise, naming the group and what is wrong."""
    number = int(group["index"])
    path = root / GROUPS_DIR / str(number) / GROUP_OUTPUTS
    where = f"{GROUPS_DIR}/{number}/{GROUP_OUTPUTS}"
    staged = temporary_of(path)
    recorded = group["sha256"]
    if not path.is_file() and staged.is_file() and file_sha256(staged) == recorded:
        return staged
    if not path.is_file():
        raise _group_file_refused(
            f"{where} is not there, and {_counted([int(row['point']) for row in done])} of group {number} "
            f"{'are' if len(done) != 1 else 'is'} marked done in {STUDY_FILE}", where)
    try:
        archive = GroupArchive(path)
        try:
            absent = [int(row["point"]) for row in done
                      if f"m{row['point']}/{CONTENT_KEY}" not in archive.keys_of(int(row["point"]))
                      or "manifest" not in archive.member(int(row["point"])).files]
            other = [int(row["point"]) for row in done if int(row["point"]) not in absent
                     and str(archive[f"m{row['point']}/{CONTENT_KEY}"]) != row["content_key"]]
        finally:
            archive.close()
    except (zipfile.BadZipFile, ValueError, OSError, EOFError) as error:
        raise _group_file_refused(f"{where} cannot be read as the NPZ its run wrote ({error}): its bytes were changed "
                                  f"after the run", where) from error
    if other:
        raise _group_file_refused(
            f"{where} holds {_counted(other)} under content keys that are not those {STUDY_FILE} gives them: a group "
            f"file of another study, or of another run of this one", where)
    if absent:
        raise _group_file_refused(
            f"{_counted(absent)} marked done in {STUDY_FILE} {'have' if len(absent) != 1 else 'has'} no entries in "
            f"{where}", where)
    held = file_sha256(path)
    if held != recorded:
        if staged.is_file() and file_sha256(staged) == recorded:
            return staged
        raise _group_file_refused(
            f"{where} is not the file {STUDY_FILE} records for group {number} (its sha256 is {held}, and {STUDY_FILE} "
            f"records {recorded}): its members' arrays were changed after the run", where)
    return path


def _read_json(path: Path) -> object:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except ValueError as error:
        raise ArgumentRefused([Refusal(
            "study.format", RefusalClass.USER, f"{path} is not JSON ({error})",
            "read a folder `Study.run_to(folder)` wrote, whole: `StudyResult.read(folder)`")]) from error


def read_study(root: Path) -> StudyResult:
    """The study in ``root``: see :meth:`StudyResult.read`."""
    manifest = root / STUDY_FILE
    if not root.is_dir() or not manifest.is_file():
        raise ArgumentRefused([Refusal(
            "study.not-a-study-folder", RefusalClass.USER, f"{root} is not a study folder: it holds no {STUDY_FILE}",
            "read a folder that Study(exp).sweep(...).run_to(<folder>) or `mime study run <spec> --out <folder>` wrote: "
            "StudyResult.read(<folder>)")])
    document = check_document(_read_json(manifest))
    groups = cast(list[dict[str, Any]], document["groups"])
    check_layout(root, len(groups))
    axes = cast(list[dict[str, Any]], document["axes"])
    paths = [str(axis["path"]) for axis in axes]
    points = {int(row["index"]): [decode_value(v) for v in row["values"]]
              for row in cast(list[dict[str, Any]], document["points"])}
    rows = cast(list[dict[str, Any]], document["members"])
    done = [row for row in rows if row["status"] == DONE]
    missing = [int(row["point"]) for row in rows if row["status"] != DONE]
    files = {int(group["index"]): _group_file(root, group, [row for row in done if row["group"] == group["index"]])
             for group in groups if any(row["group"] == group["index"] for row in done)}
    archives: dict[int, GroupArchive] = {}

    def loader(row: dict[str, Any]) -> RunResult:
        group, point = int(row["group"]), int(row["point"])
        if group not in archives:
            archives[group] = GroupArchive(files[group])
        archive = archives[group]
        member = archive.member(point)
        run = RunResult.from_manifest(cast(dict[str, Any], json.loads(str(member["manifest"]))), member)
        run.stage = StageRecord.read(member.files, member.__getitem__)
        if "report" in member.files:
            run.report = cast(dict[str, Any], json.loads(str(member["report"])))
        else:
            run.report, run.report_unreadable = read_report(root / GROUPS_DIR / str(group) / GROUP_REPORT)
        return run

    members: list[Callable[[], RunResult]] = [partial(loader, row) for row in done]
    coordinates = {path: [points[int(row["point"])][i] for row in done] for i, path in enumerate(paths)}
    dropped = [Dropped(int(d["index"]), str(d["coordinates"]), tuple(map(str, d["refusals"])))
               for d in cast(list[dict[str, Any]], document["dropped"])]
    compiles = [int(g["compiles"]) for g in cast(list[dict[str, Any]], document["groups"])]
    result = StudyResult(str(document["experiment"]["name"]), paths, coordinates, members,
                         [int(row["group"]) for row in done], [int(row["point"]) for row in done], dropped, compiles,
                         record=document, base=root / BASE_DIR, missing=missing)
    result.folder = root
    return result


__all__ = ["read_study"]
