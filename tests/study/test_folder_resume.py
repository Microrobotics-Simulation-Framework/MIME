"""Gate 4 of S2c-1: a study stopped after k members resumes and runs exactly the missing members, and the finished folder
is an uninterrupted one's, byte for byte except the group's measured seconds. The toy study has two groups of three
members in batches of one, so a stop falls on a batch boundary: a planned batch cut by a member that is reused would run
the rest as a smaller batch, which is not claimed bit for bit (the usable CPU count and the BLAS thread count do not
matter here, both runs being in one process at one count; they are printed). Controls: a member whose content key's source
hash changed is run again, never reused, on each of five repeats, beside a rerun with nothing changed that runs none; and
the refuting review's attempt on what the key hashes: an edit to a file of MIME that no node class is defined in
(``nodes/body/_contact.py``, ``surrogates/__init__.py``), made in a copy of the installed package the key is pointed at,
runs every member again, on each of four edits."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

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
        monkeypatch.setattr(content, "source_hashes", lambda builts=(), repeat=repeat: {**real(builts), "planted": str(repeat)})
        ran["asked"].clear()
        study.run_to(out)
        reran.append(sum(n for n in ran["asked"] if n))
    print("members rerun after a changed source hash, five repeats:", reran)
    assert reran == [3] * 5, reran


def test_the_key_hashes_every_file_of_the_installed_package_those_the_review_found_unhashed_among_them():
    import mime

    assert content.package_root() == Path(mime.__file__).parent
    held = content.source_hashes()
    assert {"mime/nodes/body/_contact.py", "mime/surrogates/__init__.py", "mime/study/_run.py", "mime/build/__init__.py",
            "mime/units.py"} <= set(held)
    assert not [name for name in held if "__pycache__" in name or name.endswith(".pyc")]
    assert set(content.versions()) >= {"mime", "maddening", "openmaterica", "jax", "jaxlib", "x64"}


def test_an_edit_to_a_file_of_mime_no_node_class_is_in_runs_every_member_again_on_every_edit(ran, tmp_path, monkeypatch):
    tree = tmp_path / "installed" / "mime"
    shutil.copytree(content.package_root(), tree, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(content, "package_root", lambda: tree)
    out = tmp_path / "study"
    study = toy_study(steps=(), stiffnesses=(1.1, 1.7))
    study.run_to(out)
    reran = []
    for edited in ("nodes/body/_contact.py", "surrogates/__init__.py") * 2:
        with (tree / edited).open("a", encoding="utf-8") as file:
            file.write("# an edit that changes the file's bytes and nothing else\n")
        ran["asked"].clear()
        study.run_to(out)
        reran.append(sum(n for n in ran["asked"] if n))
    ran["asked"].clear()
    study.run_to(out)
    unchanged = sum(n for n in ran["asked"] if n)
    print("members rerun after each edit:", reran, "after none:", unchanged)
    assert reran == [2] * 4 and unchanged == 0, (reran, unchanged)


def test_a_group_file_that_is_not_the_one_recorded_is_refused_by_a_resumed_run_and_a_group_file_moved_out_reruns(
        ran, tmp_path):
    import zipfile

    from mime.build import BuildRefused

    from folder_rigs import tree_bytes

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
    before = tree_bytes(out)
    ran["asked"].clear()
    with pytest.raises(BuildRefused) as refused:
        study.run_to(out)
    assert refused.value.codes() == ["study.group-file"] and "groups/0/outputs.npz" in refused.value.refusals[0].action
    assert tree_bytes(out) == before and ran["asked"] == []
    path.unlink()  # the route the refusal gives: the group file moved out, its members run again
    assert len(study.run_to(out)) == 2
    assert sum(n for n in ran["asked"] if n) == 2
