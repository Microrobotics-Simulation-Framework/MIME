"""Gate 4 of S2c-1: a study stopped after k members resumes and runs exactly the missing members, and the finished folder
is an uninterrupted one's, byte for byte except the group's measured seconds. The toy study has two groups of three
members in batches of one, so a stop falls on a batch boundary: a planned batch cut by a member that is reused would run
the rest as a smaller batch, which is not claimed bit for bit (the usable CPU count and the BLAS thread count do not
matter here, both runs being in one process at one count; they are printed). Control: a member whose content key's source
hash changed is run again, never reused, on each of five repeats, beside a rerun with nothing changed that runs none."""

from __future__ import annotations

import json
import os

import pytest
from folder_rigs import toy_study, tree_bytes

import mime.study._content as content
import mime.study._folder_run as folder_run
from mime.results import StudyResult
from mime.results._study_folder import STUDY_FILE


class Stop(Exception):
    pass


@pytest.fixture
def ran(monkeypatch):
    """The members each ``run_group`` call was asked to run, and a way to stop a study after some batches."""
    state = {"asked": [], "stop_after": None, "batches": 0}
    real = folder_run.run_group

    def watching(plan, number, record, every, members, *, todo=None, after=None):
        state["asked"].append(len(todo) if todo is not None else None)

        def counted(batch):
            after(batch)
            state["batches"] += 1
            if state["stop_after"] is not None and state["batches"] >= state["stop_after"]:
                raise Stop

        return real(plan, number, record, every, members, todo=todo, after=counted)

    monkeypatch.setattr(folder_run, "run_group", watching)
    return state


def comparable(folder):
    held = tree_bytes(folder)
    document = json.loads(held.pop(STUDY_FILE))
    for group in document["groups"]:
        group["seconds"] = None
    return held, document


def test_a_study_stopped_after_four_members_resumes_and_runs_exactly_the_two_missing(ran, tmp_path):
    from os import sched_getaffinity

    print("usable CPUs", len(sched_getaffinity(0)), "OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS"))
    whole = tmp_path / "whole"
    toy_study(batch=1).run_to(whole)
    ran["asked"].clear()
    ran["batches"] = 0
    stopped = tmp_path / "stopped"
    ran["stop_after"] = 4
    with pytest.raises(Stop):
        toy_study(batch=1).run_to(stopped)
    partial = StudyResult.read(stopped)
    assert len(partial) == 4 and partial.missing == (3, 5)
    assert "2 of the study's 6 members are missing" in partial.text()
    statuses = [m["status"] for m in json.loads((stopped / STUDY_FILE).read_text())["members"]]
    assert statuses.count("done") == 4 and statuses.count("pending") == 2
    ran["stop_after"], ran["batches"] = None, 0
    ran["asked"].clear()
    done = toy_study(batch=1).run_to(stopped)
    assert sum(n for n in ran["asked"] if n) == 2 and ran["batches"] == 2, ran
    assert len(done) == 6 and done.missing == ()
    (held_whole, doc_whole), (held_stopped, doc_stopped) = comparable(whole), comparable(stopped)
    assert doc_whole == doc_stopped
    assert held_whole.keys() == held_stopped.keys()
    different = [name for name in held_whole if held_whole[name] != held_stopped[name]]
    assert different == [], different


def test_a_finished_study_run_again_runs_nothing_and_a_changed_source_runs_every_member_every_time(ran, tmp_path,
                                                                                                   monkeypatch):
    out = tmp_path / "study"
    study = toy_study(steps=(), stiffnesses=(1.1, 1.7, 2.3))
    study.run_to(out)
    assert ran["asked"] == [3]
    ran["asked"].clear()
    study.run_to(out)
    assert sum(n for n in ran["asked"] if n) == 0, "nothing changed: nothing is run"
    real = content.source_hashes
    reran = []
    for repeat in range(5):
        monkeypatch.setattr(content, "source_hashes", lambda built, repeat=repeat: {**real(built), "planted": str(repeat)})
        ran["asked"].clear()
        study.run_to(out)
        reran.append(sum(n for n in ran["asked"] if n))
    print("members rerun after a changed source hash, five repeats:", reran)
    assert reran == [3] * 5, reran


def test_a_member_whose_arrays_are_not_in_the_group_file_is_run_again_though_the_manifest_says_done(ran, tmp_path):
    import zipfile

    out = tmp_path / "study"
    study = toy_study(steps=(), stiffnesses=(1.1, 1.7))
    study.run_to(out)
    path = out / "groups" / "0" / "outputs.npz"
    with zipfile.ZipFile(path) as old:
        kept = {info.filename: old.read(info) for info in old.infolist() if not info.filename.startswith("m1/")}
    path.unlink()
    with zipfile.ZipFile(path, "w") as new:
        for name, data in kept.items():
            new.writestr(name, data)
    ran["asked"].clear()
    assert len(study.run_to(out)) == 2
    assert sum(n for n in ran["asked"] if n) == 1
