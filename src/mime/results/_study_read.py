"""Reading a study folder: its layout checked, its manifest closed and in a format this MIME reads, and each member read
from its group's NPZ when first asked for. A study that was stopped reads as what it holds."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._result import RunResult, read_report
from mime.results._study import Dropped, StudyResult
from mime.results._study_folder import (
    BASE_DIR,
    DONE,
    GROUP_OUTPUTS,
    GROUP_REPORT,
    GROUPS_DIR,
    STUDY_FILE,
    GroupArchive,
    check_document,
    check_layout,
    decode_value,
)


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
    check_layout(root)
    axes = cast(list[dict[str, Any]], document["axes"])
    paths = [str(axis["path"]) for axis in axes]
    points = {int(row["index"]): [decode_value(v) for v in row["values"]]
              for row in cast(list[dict[str, Any]], document["points"])}
    rows = cast(list[dict[str, Any]], document["members"])
    done = [row for row in rows if row["status"] == DONE]
    missing = [int(row["point"]) for row in rows if row["status"] != DONE]
    archives: dict[int, GroupArchive] = {}

    def loader(row: dict[str, Any]) -> RunResult:
        group, point = int(row["group"]), int(row["point"])
        if group not in archives:
            archives[group] = GroupArchive(root / GROUPS_DIR / str(group) / GROUP_OUTPUTS)
        archive = archives[group]
        member = archive.member(point)
        run = RunResult.from_manifest(cast(dict[str, Any], json.loads(str(member["manifest"]))), member)
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
    return StudyResult(str(document["experiment"]["name"]), paths, coordinates, members,
                       [int(row["group"]) for row in done], [int(row["point"]) for row in done], dropped, compiles,
                       record=document, base=root / BASE_DIR, missing=missing)


__all__ = ["read_study"]
