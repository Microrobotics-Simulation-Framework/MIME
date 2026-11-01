"""A folder saved before the drag's rung ``bem`` was renamed ``mrs`` (the saved form's version 2, ``mime.document/2``,
``mime.lock/2``) loads under version 3, held to its own lock, and ``mime verify`` reproduces it.

Version 3 changes three things a hash saw: the rung's name (in the document and the resolved record), the drag node's
type in the derived graph (``BEMDragNode`` became ``MRSDragNode``), and the possessive in the sources of values declared
in the experiment (``body 'ball''s`` became ``body ball's``). A load of version 2 writes the rebuilt experiment back
into version 2's form and compares it with the lock there (``mime.saved._versions``), and says that it read ``bem`` as
``mrs``.

Both folders were written by MIME at the commit before the rename (22fd495 plus the duration allowlist, c6d5567), in
float64 with one BLAS thread as the saved tests run, from experiments of ``tests/mrs_cases.py``'s kind:

- ``previous/bem_ball``: a ball of 1 mm radius in water filling no region, ``Drag.BEM(spacing=1.5 mm, blob=0.5,
  wall_kind="free_space")``, mass and inertia declared, moving and turning;
- ``previous/possessive_ball``: the same ball under ``Gravity()`` in water with its density declared, no drag of its
  own, so that nothing but the possessive and the schema differ in it from version 3.

The controls: each folder with its document edited after its lock is refused; a version 2 document beside a version 3
lock, and a version 3 document beside a version 2 lock, are refused; the write-back is checked to be what finds the
lock's hashes (a load that did not write the form back is refused, for the document, the record and the graph); and a
version 2 folder saved again is of version 3 and holds neither the old rung nor the doubled quote.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import mrs_cases
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE
from mime.saved import _record as record_module
from mime.saved._folder import LoadRefused
from mime.saved._versions import (
    document_in_form,
    physics_in_form,
    resolved_in_form,
    upgraded_document,
    upgraded_lock,
)
from mime.units import mm

PREVIOUS = Path(__file__).parent / "previous"
FOLDERS = ("bem_ball", "possessive_ball")


def copied(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / name
    shutil.copytree(PREVIOUS / name, folder)
    return folder


def current_ball(tmp_path: Path, name: str) -> Path:
    """The experiment of ``name`` built and saved by this MIME: version 3."""
    if name == "bem_ball":
        experiment = mrs_cases.free_space(name, {"ball": mrs_cases.BALL}, mrs_cases.mrs(mm(1.5)))
    else:
        experiment = Experiment.load(PREVIOUS / name)
    experiment.build().save(tmp_path / f"current_{name}")
    return tmp_path / f"current_{name}"


@pytest.mark.parametrize("name", FOLDERS)
def test_the_fixtures_are_folders_of_version_two(name: str):
    assert yaml.safe_load((PREVIOUS / name / DOCUMENT_FILE).read_text())["schema"] == "mime.document/2"
    lock = json.loads((PREVIOUS / name / LOCK_FILE).read_text())
    assert lock["schema"] == "mime.lock/2"
    assert "'ball''s" in json.dumps(lock["resolved"]["values"])


def test_the_rung_is_named_bem_in_the_old_folder_and_read_as_mrs():
    text = (PREVIOUS / "bem_ball" / DOCUMENT_FILE).read_text()
    assert "rung: bem" in text
    experiment = Experiment.load(PREVIOUS / "bem_ball")
    (drag,) = [each for each in experiment.document()["interactions"] if each["name"] == "drag"]
    assert drag["settings"]["rung"] == "mrs"
    report = experiment.load_report
    assert report is not None
    lines = report.text().splitlines()
    assert lines[1] == ("saved form: mime.document/2, read as version 4; the rung bem of drag:ball is read as mrs: the "
                        "rung was renamed for its method, and is the same; it declares no licence, terms or "
                        "permission, being of a form before them")
    assert lines[2] == "input hash: reproduced exactly"
    assert lines[-1] == "derived graph: the same graph, hash for hash" or lines[-1].startswith("derived graph:")


def test_a_folder_without_the_rung_loads_without_saying_so():
    report = Experiment.load(PREVIOUS / "possessive_ball").load_report
    assert report is not None
    assert report.text().splitlines()[1] == ("saved form: mime.document/2, read as version 4; it declares no licence, "
                                             "terms or permission, being of a form before them")
    assert report.text().splitlines()[2] == "input hash: reproduced exactly"


@pytest.mark.parametrize("name", FOLDERS)
def test_mime_verify_reproduces_a_version_two_folder(tmp_path: Path, name: str):
    """In a fresh process, as a user runs it, with the one BLAS thread the tests' processes set."""
    done = subprocess.run([sys.executable, "-m", "mime.cli", "verify", str(copied(tmp_path, name))],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "input hash: reproduced exactly" in done.stdout
    assert "verify: reproduced 8 steps" in done.stdout


@pytest.mark.parametrize("name", FOLDERS)
def test_the_control_a_version_two_document_edited_after_its_lock_is_refused(tmp_path: Path, name: str):
    folder = copied(tmp_path, name)
    text = (folder / DOCUMENT_FILE).read_text()
    assert "a millimetre body's order" in text
    (folder / DOCUMENT_FILE).write_text(text.replace("a millimetre body's order", "a body's order"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert refused.value.codes() == ["load.input-hash"]


def test_the_control_a_version_two_drag_edited_after_its_lock_is_refused(tmp_path: Path):
    folder = copied(tmp_path, "bem_ball")
    text = (folder / DOCUMENT_FILE).read_text()
    assert "blob: 0.5" in text
    (folder / DOCUMENT_FILE).write_text(text.replace("blob: 0.5", "blob: 0.6"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert refused.value.codes() == ["load.input-hash"]


@pytest.mark.parametrize("name", FOLDERS)
def test_the_control_mixed_versions_are_refused(tmp_path: Path, name: str):
    old, new = copied(tmp_path, name), current_ball(tmp_path, name)
    mixed = tmp_path / "version_two_document_version_three_lock"
    shutil.copytree(old, mixed)
    shutil.copy(new / LOCK_FILE, mixed / LOCK_FILE)
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(mixed)
    assert "load.lock-shape" in refused.value.codes()
    other = tmp_path / "version_three_document_version_two_lock"
    shutil.copytree(new, other)
    shutil.copy(old / LOCK_FILE, other / LOCK_FILE)
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(other)
    assert "load.lock-shape" in refused.value.codes()


@pytest.mark.parametrize(("name", "writer", "code"), [
    ("bem_ball", "document_in_form", "load.input-hash"),
    ("bem_ball", "resolved_in_form", "load.input-hash"),
    ("possessive_ball", "resolved_in_form", "load.input-hash"),
])
def test_the_control_a_load_that_does_not_write_the_form_back_is_refused(tmp_path: Path, monkeypatch, name: str,
                                                                         writer: str, code: str):
    """The write-back finds the lock's hashes: without it the rebuilt experiment hashes as version 3 does (the rung's
    name in the document and the record, the doubled quote in a value's source), and the load refuses."""
    monkeypatch.setattr(record_module, writer, lambda held, version: dict(held))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(copied(tmp_path, name))
    assert code in refused.value.codes()


def test_the_control_a_load_that_does_not_write_the_graph_back_does_not_find_its_hash(tmp_path: Path, monkeypatch):
    """The drag node's type is in the derived graph: refused in the environment the lock was written in, and reported
    as a graph that differs (not the same, hash for hash) in another."""
    monkeypatch.setattr(record_module, "physics_in_form", lambda physics, version: dict(physics))
    try:
        report = Experiment.load(copied(tmp_path, "bem_ball")).load_report
    except LoadRefused as refused:
        assert "load.derived-hash" in refused.codes()
    else:
        assert report is not None and report.derived != "the same graph, hash for hash"


def test_the_write_back_is_the_inverse_of_the_upgrade():
    document = yaml.safe_load((PREVIOUS / "bem_ball" / DOCUMENT_FILE).read_text())
    upgraded, version, renamed = upgraded_document(document)
    assert version == 2 and renamed == ("drag:ball",)
    assert upgraded["schema"] == "mime.document/4"
    assert [each["settings"]["rung"] for each in upgraded["interactions"]] == ["mrs"]
    assert document_in_form(upgraded, 2) == document
    lock = json.loads((PREVIOUS / "bem_ball" / LOCK_FILE).read_text())
    held, lock_version = upgraded_lock(lock)
    assert lock_version == 2 and held["schema"] == "mime.lock/4"
    assert resolved_in_form(held["resolved"], 2) == lock["resolved"], "a version 2 record is its own write-back"
    physics = lock["derived"]["physics"]
    renamed_nodes = {**physics, "nodes": [{**node, "type": "MRSDragNode"} if node["type"] == "BEMDragNode" else node
                                          for node in physics["nodes"]]}
    assert renamed_nodes != physics and physics_in_form(renamed_nodes, 2) == physics


def test_a_version_two_folder_saved_again_is_of_the_current_version(tmp_path: Path):
    experiment = Experiment.load(copied(tmp_path, "bem_ball"))
    experiment.build().save(tmp_path / "again")
    document = (tmp_path / "again" / DOCUMENT_FILE).read_text()
    lock = (tmp_path / "again" / LOCK_FILE).read_text()
    assert yaml.safe_load(document)["schema"] == "mime.document/4"
    assert json.loads(lock)["schema"] == "mime.lock/4"
    assert "rung: mrs" in document and '"rung": "mrs"' in lock and '"type": "MRSDragNode"' in lock
    assert "bem" not in document.replace("bem_ball", "") and "BEM" not in lock and "''s" not in lock
    assert "body ball's mass and inertia" in lock
    again = Experiment.load(tmp_path / "again").load_report
    assert again is not None and again.text().splitlines()[1].startswith("saved form: mime.document/4")
