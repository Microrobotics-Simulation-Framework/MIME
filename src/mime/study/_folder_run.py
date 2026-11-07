"""``Study.run_to``: a study run into its folder, group by group and batch by batch, so that a stopped run resumes.

The folder is written as the run goes: ``study.json`` first, every member ``pending``, then each batch's arrays into its
group's NPZ and its members marked ``done`` in ``study.json``, each file replaced atomically. A run into a folder that
holds this study (the same id) reuses a member only if it is ``done`` under the same content key
(``mime.study._content``) and its arrays are in the group's NPZ; every other member is run, in the batches the plan made
(a planned batch cut by a member that is reused runs the rest of it as a smaller batch)."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from numpy.typing import NDArray

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.results._study import StudyResult, member_entries
from mime.results._study_folder import (
    BASE_DIR,
    DONE,
    GROUP_OUTPUTS,
    GROUP_REPORT,
    GROUPS_DIR,
    PENDING,
    SPEC_FILE,
    STUDY_FILE,
    TEMPORARY,
    GroupArchive,
    check_document,
    check_layout,
    write_entries,
    write_json,
)
from mime.results import RunResult
from mime.study._point import precision
from mime.study._record import document_of
from mime.study._run import run_group
from mime.study._study import StudyPlan, run_members

if TYPE_CHECKING:
    from mime.experiment import Experiment
    from mime.study._study import Points, Study

type Say = Callable[[str], None]


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def _clear_temporaries(root: Path) -> None:
    for path in sorted(root.rglob(f"*{TEMPORARY}")):
        if path.is_file():
            path.unlink()


def _base(experiment: Experiment, out: Path, existing: bool) -> str:
    """The base experiment into ``out``/experiment (a copy of its saved folder, or what ``save()`` writes of it) unless
    the folder already holds it, and its lock's input hash."""
    saved = experiment.declared_as()[2]
    target = out / BASE_DIR
    if saved is not None:
        if not existing:
            shutil.copytree(saved.folder, target)
        return str(saved.lock["hashes"]["input"])
    built = experiment.build()
    if not existing:
        built.save(target)
    return str(built.record.hashes()["input"])


def _pilot(plan: StudyPlan, say: Say) -> None:
    """The first chunk of the first group's first batch, run twice (the first run compiles), timed: what a member costs."""
    group = plan.groups[0]
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
    say(f"pilot: a chunk of {chunk[0]} steps of {len(batch)} members took {took:.3g} s once compiled (noisy), "
        f"{took / len(batch):.3g} s a member; the study's {len(plan.points) - len(plan.refused)} members of "
        f"{plan.steps} steps in {chunks} chunks: about {took / len(batch) * chunks * (len(plan.points) - len(plan.refused)):.3g} s "
        f"if this chunk is typical")


def run_study_to(study: Study, experiment: Experiment, points: Points, out: Path, *,
                 shard: tuple[int, int] | None = None, pilot: bool = False, say: Say | None = None) -> StudyResult:
    """The study run into ``out``: see :meth:`Study.run_to`."""
    from mime.study._spec import spec_text

    plan = study.plan()
    out.mkdir(parents=True, exist_ok=True)
    existing = (out / STUDY_FILE).is_file()
    if not existing and (out / BASE_DIR).is_dir() and not any(child.name != BASE_DIR for child in out.iterdir()):
        shutil.rmtree(out / BASE_DIR)  # a run that stopped before its manifest: nothing was claimed
    if not existing and any(out.iterdir()):
        raise _refuse("study.folder-not-empty", RefusalClass.USER,
                      f"{out} is not empty and holds no study ({STUDY_FILE})",
                      "name a new folder, or an empty one: --out <folder>")
    _clear_temporaries(out)
    input_hash = _base(experiment, out, existing)
    fresh = document_of(plan, input_hash)
    old: dict[str, Any] | None = None
    if existing:
        old = check_document(json.loads((out / STUDY_FILE).read_text(encoding="utf-8")))
        check_layout(out)
        if old["id"] != fresh["id"]:
            raise _refuse("study.other-study", RefusalClass.USER,
                      f"{out} holds the study {old['id']} (of the experiment {old['experiment']['name']!r}), and "
                          f"this one is {fresh['id']}: the base's input hash, the axes, the points, the run or the "
                          f"seed policy differ",
                          "name another folder with `--out <folder>`, or delete this one if it is yours to delete: a study "
                          "is never written over another")
    statuses = _statuses(out, old, fresh)
    document = document_of(plan, input_hash, status=statuses, group_experiments=_experiments(plan))
    if old is not None:
        for number, group in enumerate(cast(list[dict[str, Any]], old["groups"])):
            document["groups"][number]["seconds"] = group["seconds"]
    write_json(out / STUDY_FILE, document)
    (out / SPEC_FILE).write_text(
        spec_text(points, duration=cast(Any, plan.duration), record=plan.record, every=plan.every,
                  batch=plan.batch, drop_refused=plan.drop_refused, reasons=plan.reasons, base=BASE_DIR,
                  input_hash=input_hash), encoding="utf-8", newline="\n")
    if pilot and say is not None and plan.groups:
        _pilot(plan, say)
    compiles = [0] * len(plan.groups)
    for number, group in enumerate(plan.groups):
        if shard is not None and number % shard[1] != shard[0]:
            continue
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
                members[index] = None
            write_entries(folder / GROUP_OUTPUTS, entries, [f"m{index}/" for index in batch])
            for index in batch:
                row_of[index]["status"] = DONE
            write_json(out / STUDY_FILE, document)

        compiles[number], took = run_group(plan, number, plan.record, plan.every, members, todo=todo, after=finished)
        document["groups"][number]["seconds"] = took
        write_json(out / STUDY_FILE, document)
    return StudyResult.read(out)


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
            built.save(folder / BASE_DIR)


def _statuses(out: Path, old: dict[str, Any] | None, fresh: dict[str, Any]) -> dict[int, str]:
    """The points a run into ``out`` need not run: ``done`` under the same content key as this plan gives them, with
    their arrays in the group's NPZ."""
    if old is None:
        return {}
    keys = {row["point"]: (row["content_key"], row["group"]) for row in fresh["members"]}
    found: dict[int, str] = {}
    archives: dict[int, set[int]] = {}
    for row in cast(list[dict[str, Any]], old["members"]):
        point, group = int(row["point"]), int(row["group"])
        if row["status"] != DONE or keys.get(point) != (row["content_key"], group):
            continue
        if group not in archives:
            path = out / GROUPS_DIR / str(group) / GROUP_OUTPUTS
            archives[group] = set(GroupArchive(path).points()) if path.is_file() else set()
        if point in archives[group]:
            found[point] = DONE
    return found


__all__ = ["run_study_to"]
