"""``Study.run_to``: a study run into its folder, group by group and batch by batch, so that a stopped run resumes.

A study is written only into an empty or new folder, or into one that holds the same study: anything else is refused,
and nothing in it is touched. ``study.json`` is written first, every member ``pending``: it marks the folder as this
study's before anything else is written into it. Then the spec file and the base experiment (written under its
temporary name and renamed), then each batch: its arrays staged into the group's NPZ, the staged file's sha256 written
into ``study.json`` with the batch's members marked ``done``, and the staged file renamed into place. Each file is
replaced atomically. A run into a folder that holds this study (the same id) first finishes or removes its own
temporaries (only those, by their fixed names), refuses a group file that is not the one ``study.json`` records, and
reuses a member only if it is ``done`` under the same content key (``mime.study._content``) and its entries in the
group's NPZ carry that key; every other member is run, in the batches the plan made (a planned batch cut by a member
that is reused runs the rest of it as a smaller batch)."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.results._study import StudyResult, member_entries
from mime.results._study_folder import (
    BASE_DIR,
    CONTENT_KEY,
    DONE,
    GROUP_OUTPUTS,
    GROUP_REPORT,
    GROUPS_DIR,
    PENDING,
    SPEC_FILE,
    STUDY_FILE,
    GroupArchive,
    check_document,
    check_layout,
    file_sha256,
    iter_groups,
    stage_entries,
    temporaries,
    temporary_of,
    write_json,
    write_text,
)
from mime.results import RunResult
from mime.study._point import precision
from mime.study._protocol import FromRest
from mime.study._record import document_of
from mime.study._run import run_group
from mime.study._study import StudyPlan, run_members

if TYPE_CHECKING:
    from mime.experiment import Experiment
    from mime.study._study import Points, Study

type Say = Callable[[str], None]


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def _base(experiment: Experiment) -> tuple[str, Callable[[Path], object]]:
    """The base experiment's input hash, and how to write it into a folder: a copy of its saved folder, or what
    ``save()`` writes of it."""
    saved = experiment.declared_as()[2]
    if saved is not None:
        source = saved.folder

        def copy(target: Path) -> None:
            shutil.copytree(source, target)

        return str(saved.lock["hashes"]["input"]), copy
    built = experiment.build()
    return str(built.record.hashes()["input"]), built.save


def _write_folder(target: Path, write: Callable[[Path], object]) -> None:
    """A folder written under its temporary name and renamed into place, so that a stopped write leaves no half folder
    at ``target``."""
    staged = temporary_of(target)
    write(staged)
    os.replace(staged, target)


def _old_document(out: Path) -> dict[str, Any] | None:
    """The manifest of the study ``out`` holds, its layout checked; ``None`` for an empty or new folder. A folder that
    holds anything and no manifest is refused, naming what it holds: a study is never written over what it did not
    write."""
    manifest = out / STUDY_FILE
    if not manifest.is_file():
        held = sorted(child.name + ("/" if child.is_dir() else "") for child in out.iterdir())
        if held:
            raise _refuse("study.folder-not-empty", RefusalClass.USER,
                          f"{out} holds {', '.join(held[:5])}{', ...' if len(held) > 5 else ''} and no study "
                          f"({STUDY_FILE}): a study is written into an empty or new folder, and never over what it did "
                          f"not write", "name a new folder, or an empty one: `--out <folder>`, `study.run_to(<folder>)`")
        return None
    try:
        document = cast(object, json.loads(manifest.read_text(encoding="utf-8")))
    except ValueError as error:
        raise _refuse("study.format", RefusalClass.USER, f"{manifest} is not JSON ({error})",
                      "name a new folder with `--out <folder>`: this one's manifest cannot be read") from error
    old = check_document(document)
    check_layout(out, len(cast(list[Any], old["groups"])))
    return old


def _pilot(plan: StudyPlan, number: int, say: Say) -> None:
    """The first chunk of group ``number``'s first batch, run twice (the first run compiles), timed: what a member
    costs."""
    group = plan.groups[number]
    built = group.built
    assert built is not None
    batch = group.members[:group.batches[0] if group.batched else 1]
    chunk = group.chunks[:1] if group.chunks else ()
    if not chunk:
        return
    took = 0.0
    with precision(plan.points[group.members[0]].declared.x64):
        tests = built.flag_tests()
        for _ in range(2):
            started = time.perf_counter()
            run_members(built, plan.points, batch, group.varying, chunk, tests, ())
            took = time.perf_counter() - started
    chunks = len(group.chunks)
    say(f"pilot: a chunk of {chunk[0]} steps of {len(batch)} members of group {number} took {took:.3g} s once compiled "
        f"(noisy), {took / len(batch):.3g} s a member; the study's {len(plan.points) - len(plan.refused)} members of "
        f"{plan.steps} steps in {chunks} chunks: about {took / len(batch) * chunks * (len(plan.points) - len(plan.refused)):.3g} s "
        f"if this chunk is typical")


def _shard_checked(plan: StudyPlan, shard: tuple[int, int] | None) -> None:
    """Refuses a shard count past the study's groups: sharding is by structure group, so such a shard runs nothing."""
    if shard is not None and shard[1] > len(plan.groups):
        count = len(plan.groups)
        raise _refuse("study.shard", RefusalClass.USER,
                      f"--shard {shard[0]}/{shard[1]} splits the study over {shard[1]} shards, and sharding is by "
                      f"structure group: this study has {count} group{'s' if count != 1 else ''}, so "
                      f"{shard[1] - count} of the shards would run no member",
                      f"shard it at most {count} way{'s' if count != 1 else ''} (`--shard i/{count}`), or run it whole: "
                      f"`mime study run <spec> --out <folder>`")


def run_study_to(study: Study, experiment: Experiment, points: Points, out: Path, *,
                 shard: tuple[int, int] | None = None, pilot: bool = False, say: Say | None = None) -> StudyResult:
    """The study run into ``out``: see :meth:`Study.run_to`."""
    from mime.study._spec import spec_text

    plan = study.plan()
    if not isinstance(plan.protocol, FromRest):
        raise _refuse("study.protocol-folder", RefusalClass.USER,
                      f"the study runs {plan.protocol.text()}, and a study folder records no protocol yet (its "
                      f"{STUDY_FILE} and {SPEC_FILE} have no place for one), so it would read back as run from rest",
                      "run it in memory, `study.run()`, and read it there with ramp_crossings(result, lag); or sweep "
                      "from rest, `sweep(..., protocol=FromRest())`, into the folder")
    _shard_checked(plan, shard)
    out.mkdir(parents=True, exist_ok=True)
    old = _old_document(out)
    input_hash, write_base = _base(experiment)
    fresh = document_of(plan, input_hash)
    files: dict[int, str | None] = {}
    if old is not None:
        _same_study(out, old, fresh)
        files = _group_files_checked(out, old)
    statuses = _statuses(out, old, fresh, files)
    document = document_of(plan, input_hash, status=statuses, group_experiments=_experiments(plan), group_files=files)
    if old is not None:
        for number, group in enumerate(cast(list[dict[str, Any]], old["groups"])):
            document["groups"][number]["seconds"] = group["seconds"]
    write_json(out / STUDY_FILE, document)
    write_text(out / SPEC_FILE, spec_text(
        points, duration=cast(Any, plan.duration), record=plan.record, every=plan.every, batch=plan.batch,
        drop_refused=plan.drop_refused, reasons=plan.reasons, base=BASE_DIR, input_hash=input_hash))
    if not (out / BASE_DIR).is_dir():
        _write_folder(out / BASE_DIR, write_base)
    owned = [number for number in range(len(plan.groups)) if shard is None or number % shard[1] == shard[0]]
    if pilot and say is not None and owned:
        _pilot(plan, owned[0], say)
    for number in owned:
        group = plan.groups[number]
        todo = {index for index in group.members if statuses.get(index, PENDING) != DONE}
        folder = out / GROUPS_DIR / str(number)
        folder.mkdir(parents=True, exist_ok=True)
        _group_files(plan, number, folder)
        if not todo:
            continue
        members: list[RunResult | None] = [None] * len(plan.points)
        row_of = {row["point"]: row for row in document["members"]}

        def finished(batch: list[int], number: int = number, folder: Path = folder,
                     members: list[RunResult | None] = members, row_of: dict[int, dict[str, Any]] = row_of) -> None:
            entries: dict[str, NDArray[Any]] = {}
            for index in batch:
                entries.update(member_entries(index, cast(RunResult, members[index])))
                entries[f"m{index}/{CONTENT_KEY}"] = np.asarray(str(row_of[index]["content_key"]))
                members[index] = None
            staged = stage_entries(folder / GROUP_OUTPUTS, entries, [f"m{index}/" for index in batch])
            for index in batch:
                row_of[index]["status"] = DONE
            document["groups"][number]["sha256"] = file_sha256(staged)
            write_json(out / STUDY_FILE, document)
            os.replace(staged, folder / GROUP_OUTPUTS)

        _, took = run_group(plan, number, plan.record, plan.every, members, todo=todo, after=finished)
        document["groups"][number]["seconds"] = took
        write_json(out / STUDY_FILE, document)
    return StudyResult.read(out)


def _same_study(out: Path, old: dict[str, Any], fresh: dict[str, Any]) -> None:
    """Refuses a folder that holds another study, or this one with its points grouped otherwise."""
    if old["id"] != fresh["id"]:
        raise _refuse("study.other-study", RefusalClass.USER,
                      f"{out} holds the study {old['id']} (of the experiment {old['experiment']['name']!r}), and "
                      f"this one is {fresh['id']}: the base's input hash, the axes, the points, the run or the "
                      f"seed policy differ",
                      "name another folder with `--out <folder>`, or delete this one if it is yours to delete: a study "
                      "is never written over another")
    held = [(group["key"], group["members"]) for group in cast(list[dict[str, Any]], old["groups"])]
    if held != [(group["key"], group["members"]) for group in cast(list[dict[str, Any]], fresh["groups"])]:
        raise _refuse("study.other-study", RefusalClass.USER,
                      f"{out} holds this study ({old['id']}) with its points in other groups: the structure keys this "
                      f"MIME gives its points are not those it ran under, so its group files are not this run's groups",
                      "name a new folder with `--out <folder>`: this one stays as it was run")


def _group_files_checked(out: Path, old: dict[str, Any]) -> dict[int, str | None]:
    """Each group's file as ``study.json`` records it (its sha256), after the run's own temporaries are finished or
    removed: a group file staged and recorded but not yet renamed is renamed into place, and every other temporary of
    the run's is removed. A group file that is not the one recorded is refused (``study.group-file``); a group whose
    recorded file is gone has none, and its members run again."""
    recorded = {int(group["index"]): cast(str | None, group["sha256"])
                for group in cast(list[dict[str, Any]], old["groups"])}
    for number, folder in iter_groups(out):
        staged = temporary_of(folder / GROUP_OUTPUTS)
        if staged.is_file() and recorded.get(number) is not None and file_sha256(staged) == recorded[number]:
            os.replace(staged, folder / GROUP_OUTPUTS)
    for path in temporaries(out):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    found: dict[int, str | None] = {}
    for number, sha in recorded.items():
        path = out / GROUPS_DIR / str(number) / GROUP_OUTPUTS
        if not path.is_file():
            found[number] = None
            continue
        held = file_sha256(path)
        if held != sha:
            raise _refuse("study.group-file", RefusalClass.USER,
                          f"{path} is not the file {STUDY_FILE} records for group {number} (its sha256 is {held}, and "
                          f"{STUDY_FILE} records {sha}): a group file of another study, or one changed after its run",
                          f"move {GROUPS_DIR}/{number}/{GROUP_OUTPUTS} out of the folder to run that group's members "
                          f"again, or name a new folder with `--out <folder>`: a group file is never written over")
        found[number] = held
    return found


def _experiments(plan: StudyPlan) -> dict[int, str]:
    """Where each group past the first keeps its own build: its compiled program differs from group 0's."""
    return {number: f"{GROUPS_DIR}/{number}/{BASE_DIR}" for number in range(1, len(plan.groups))}


def _group_files(plan: StudyPlan, number: int, folder: Path) -> None:
    """A group's ``report.json`` (its first member's build report summary) and, for a group past the first, its own
    build as a saved folder, each written once."""
    from mime.study._study import member_report

    group = plan.groups[number]
    built = group.built
    assert built is not None
    report = folder / GROUP_REPORT
    if not report.is_file():
        write_json(report, member_report(built, plan.points[group.members[0]]))
    if number >= 1 and not (folder / BASE_DIR).is_dir():
        with precision(plan.points[group.members[0]].declared.x64):
            _write_folder(folder / BASE_DIR, built.save)


def _statuses(out: Path, old: dict[str, Any] | None, fresh: dict[str, Any],
              files: dict[int, str | None]) -> dict[int, str]:
    """The points a run into ``out`` need not run: ``done`` under the same content key as this plan gives them, in a
    group whose file is the one recorded, with their entries in it under that key."""
    if old is None:
        return {}
    keys = {row["point"]: (row["content_key"], row["group"]) for row in fresh["members"]}
    found: dict[int, str] = {}
    archives: dict[int, GroupArchive] = {}
    try:
        for row in cast(list[dict[str, Any]], old["members"]):
            point, group = int(row["point"]), int(row["group"])
            if row["status"] != DONE or keys.get(point) != (row["content_key"], group) or files.get(group) is None:
                continue
            if group not in archives:
                archives[group] = GroupArchive(out / GROUPS_DIR / str(group) / GROUP_OUTPUTS)
            archive = archives[group]
            entry = f"m{point}/{CONTENT_KEY}"
            if entry in archive.keys_of(point) and str(archive[entry]) == row["content_key"]:
                found[point] = DONE
    finally:
        for archive in archives.values():
            archive.close()
    return found


__all__ = ["run_study_to"]
