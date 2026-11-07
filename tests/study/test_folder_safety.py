"""What a study folder can be trusted with, from S2c-1's refuting review.

A study never deletes or changes what it did not write: a folder that holds anything and no study (the review's case, a
saved experiment alone in it, reached by ``run_to`` and by ``mime study run --out``) is refused and left byte for byte
as it was, and a run stopped after it marked the folder as its own (its ``study.json``, written first) resumes: before
its base was in place, and between recording a group file and renaming it into place. A resumed run removes only its
own temporaries, by their fixed names.

A group's arrays are tied to its manifest: each member's entries carry the content key it ran under, and ``study.json``
holds each group file's sha256. A group file from another study (the review's swap), one with a member's entries removed
(the review's deletion) and one whose values were changed with its keys intact are each refused by name, beside the
undamaged folder, which reads."""

from __future__ import annotations

import io
import json
import os
import shutil
import zipfile

import numpy as np
import pytest
from folder_rigs import RECORDED, toy, toy_study, tree_bytes

import mime.study._folder_run as folder_run
from mime import Study
from mime.build import BuildRefused
from mime.build._refusals import ArgumentRefused
from mime.cli import main as mime_main
from mime.experiment import Experiment
from mime.results import StudyResult
from mime.results._study_folder import STUDY_FILE
from mime.saved import LOCK_FILE
from mime.study import grid
from mime.study._spec import spec_text
from mime.units import N, m, s


class Stop(Exception):
    pass


def saved_alone(tmp_path):
    """A saved experiment named ``experiment``, alone in the folder ``proj``, as a study folder's own layout invites."""
    proj = tmp_path / "proj"
    toy().build().save(proj / "experiment")
    return proj


def sweep_of(experiment):
    spring = experiment.handle("toy.spring:probe")
    return Study(experiment).sweep(grid({spring.k: [N(1.1) / m(1.0), N(1.7) / m(1.0)]}), duration=s(0.2),
                                   record=[RECORDED])


def test_a_saved_experiment_alone_in_out_is_refused_by_run_to_and_by_mime_study_run_and_left_as_it_was(tmp_path,
                                                                                                         capsys):
    proj = saved_alone(tmp_path)
    before = tree_bytes(proj)
    with pytest.raises(BuildRefused) as refused:
        sweep_of(Experiment.load(proj / "experiment")).run_to(proj)
    assert refused.value.codes() == ["study.folder-not-empty"]
    assert "experiment/" in refused.value.refusals[0].message and "--out" in refused.value.refusals[0].action
    assert tree_bytes(proj) == before
    lock = json.loads((proj / "experiment" / LOCK_FILE).read_text())
    spec = tmp_path / "spec" / "study.mime.yaml"
    spec.parent.mkdir()
    points = grid({"toy.spring:probe.k": [N(1.1) / m(1.0), N(1.7) / m(1.0)]})
    spec.write_text(spec_text(points, duration=s(0.2), record=[RECORDED], every=None, batch=None, drop_refused=False,
                              reasons={}, base="experiment", input_hash=lock["hashes"]["input"]))
    assert mime_main(["study", "run", str(spec), "--base", str(proj / "experiment"), "--out", str(proj)]) == 2
    assert "study.folder-not-empty" in capsys.readouterr().err
    assert tree_bytes(proj) == before
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "notes.txt").write_text("mine")
    with pytest.raises(BuildRefused) as refused:
        sweep_of(toy()).run_to(tmp_path / "notes")
    assert refused.value.codes() == ["study.folder-not-empty"] and (tmp_path / "notes" / "notes.txt").read_text() == "mine"


def comparable(folder):
    held = tree_bytes(folder)
    document = json.loads(held.pop(STUDY_FILE))
    for group in document["groups"]:
        group["seconds"] = None
    return held, document


def test_a_run_stopped_after_its_marker_and_before_its_base_resumes_to_an_uninterrupted_runs_folder(tmp_path,
                                                                                                     monkeypatch):
    whole = tmp_path / "whole"
    toy_study(batch=1).run_to(whole)
    out = tmp_path / "stopped"
    real = folder_run._write_folder

    def stopped(target, write):
        staged = target.with_name(target.name + ".tmp")
        staged.mkdir()
        (staged / "half.yaml").write_text("half a folder")
        raise Stop

    monkeypatch.setattr(folder_run, "_write_folder", stopped)
    with pytest.raises(Stop):
        toy_study(batch=1).run_to(out)
    assert sorted(p.name for p in out.iterdir()) == ["experiment.tmp", "study.json", "study.mime.yaml"]
    monkeypatch.setattr(folder_run, "_write_folder", real)
    assert len(toy_study(batch=1).run_to(out)) == 6
    assert comparable(out) == comparable(whole) and not (out / "experiment.tmp").exists()


class _StopAtRename:
    """``os`` as the folder run sees it, stopping the run at its first rename of a staged group file: after the
    manifest recorded the staged file, before it is in place."""

    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def replace(source, target):
        if str(source).endswith("outputs.npz.tmp"):
            raise Stop
        os.replace(source, target)


def test_a_run_stopped_between_recording_a_group_file_and_renaming_it_reads_and_resumes(tmp_path, monkeypatch):
    whole = tmp_path / "whole"
    toy_study(batch=1).run_to(whole)
    out = tmp_path / "stopped"
    monkeypatch.setattr(folder_run, "os", _StopAtRename())
    with pytest.raises(Stop):
        toy_study(batch=1).run_to(out)
    monkeypatch.setattr(folder_run, "os", os)
    assert (out / "groups" / "0" / "outputs.npz.tmp").is_file() and not (out / "groups" / "0" / "outputs.npz").exists()
    partial = StudyResult.read(out)
    assert len(partial) == 1 and partial.missing == (1, 2, 3, 4, 5)
    assert len(toy_study(batch=1).run_to(out)) == 6
    assert comparable(out) == comparable(whole)


def test_a_resumed_run_removes_only_its_own_temporaries_and_a_file_named_like_one_is_stray(tmp_path):
    out = tmp_path / "study"
    study = toy_study(steps=(), stiffnesses=(1.1, 1.7))
    study.run_to(out)
    (out / "experiment" / "mine.tmp").write_text("a file of the user's, inside the base")
    study.run_to(out)
    assert (out / "experiment" / "mine.tmp").read_text() == "a file of the user's, inside the base"
    (out / "experiment" / "mine.tmp").unlink()
    (out / "notes.tmp").write_text("mine")
    for attempt in (lambda: StudyResult.read(out), lambda: study.run_to(out)):
        with pytest.raises(ArgumentRefused) as refused:
            attempt()
        assert refused.value.codes() == ["study.stray-file"] and "notes.tmp" in refused.value.refusals[0].message
    assert (out / "notes.tmp").read_text() == "mine"


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    """Two studies of the toy over other stiffnesses, A and B, each one group of three members."""
    root = tmp_path_factory.mktemp("two")
    toy_study(steps=(), stiffnesses=(1.1, 1.7, 2.3)).run_to(root / "A")
    toy_study(steps=(), stiffnesses=(5.1, 5.7, 6.3)).run_to(root / "B")
    return root


def damaged(two, tmp_path, damage):
    copy = tmp_path / "copy"
    shutil.copytree(two / "A", copy)
    damage(copy / "groups" / "0" / "outputs.npz")
    return copy


def rewritten(path, change):
    """The NPZ at ``path`` rewritten with ``change`` applied to its entries (name to bytes)."""
    with zipfile.ZipFile(path) as old:
        held = {info.filename: old.read(info) for info in old.infolist()}
    path.unlink()
    with zipfile.ZipFile(path, "w") as new:
        for name, data in change(held).items():
            new.writestr(name, data)


def _shifted(held):
    name = next(n for n in sorted(held) if n.startswith("m1/") and "final" in n and n.endswith(".npy"))
    array = np.load(io.BytesIO(held[name]), allow_pickle=False)
    buffer = io.BytesIO()
    np.save(buffer, array + np.ones_like(array), allow_pickle=False)
    return {**held, name: buffer.getvalue()}


@pytest.mark.parametrize("case, damage, words", [
    ("the review's swap: B's group file over A's",
     lambda two: lambda path: shutil.copy(two / "B" / "groups" / "0" / "outputs.npz", path), "another study"),
    ("the review's deletion: member 1's entries removed",
     lambda two: lambda path: rewritten(path, lambda held: {n: d for n, d in held.items() if not n.startswith("m1/")}),
     "1 member (point 1) marked done"),
    ("a value changed, the keys intact",
     lambda two: lambda path: rewritten(path, _shifted), "changed after the run")])
def test_a_group_file_that_is_not_the_manifests_is_refused_by_name(two, tmp_path, case, damage, words):
    assert len(StudyResult.read(two / "A")) == 3, "the undamaged folder reads"
    copy = damaged(two, tmp_path, damage(two))
    with pytest.raises(ArgumentRefused) as refused:
        StudyResult.read(copy)
    message = refused.value.refusals[0].message
    print(case, "->", message)
    assert refused.value.codes() == ["study.group-file"] and "groups/0/outputs.npz" in message and words in message
