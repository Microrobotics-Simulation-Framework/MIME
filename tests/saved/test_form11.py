"""Saved form 11: a medium's fields are written under its own key ``fields``, by role (``fields: {flow: ...}``), as an
apparatus's are, where form 10 and earlier wrote a medium's flow as a dynamics aspect (``dynamics: {flow: ...}``). No
other key moves. A folder of an earlier form is read with its medium's ``dynamics.flow`` as ``fields.flow`` and the
rebuilt document written back so for its input hash; one whose medium holds ``fields`` is refused.

The folders of the tenth form:

- ``form10/periodic_flow``: the round-trip case ``periodic_tube_flow`` (the FVM on a periodic box, IMEX Euler with
  convection, eight cells across, driven by ``BodyForce``'s steady and pulsatile parts), written by MIME at the last
  commit of form 10 on its branch (9d728fc, F1a-2's first PR), float64, four usable CPUs (``taskset -c 0-3``), one BLAS
  thread, saved by ``exp.build().save(folder)``, pinned by its sha256;
- ``form10/chain_screw``: the current folder as form 10 wrote it.

A reader of version 10 refuses a folder of version 11 as ``load.newer-form``, the document and the lock each once.

**The controls.** With the write-back removed, the form-10 flow folder is refused as ``load.input-hash`` on every run
(it is a hash: the same input, the same refusal). The form left unstepped (the folder's files stating version 10) is
read by version 10's reader with no ``load.newer-form``, so the refusal is the step's."""

import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, read_folder
from mime.saved import _folder, _versions
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS, MEDIUM_FIELDS_ADDED

SAVED = Path(__file__).parent
FLOW10 = SAVED / "form10" / "periodic_flow"
PINNED = {"experiment.mime.yaml": "fb46e826cd2f0be8dcffa499eac62c7e9e29091c2e4ed6a4fd463c5351ee83ce",
          "mime.lock": "40232a5526f489d705e632aba80ff39c72d2b3c872860819fd4180cacad44cde",
          "reference.npz": "92e9160facde462c378e392e004ccbbb61c3afff768cb0c5e7184ca318422c75",
          "scene/scene.usda": "ada6dcbfd8ce8e660a83aae1ed016895ac3002524a281a3515ec0d646a665cf0"}
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM10_READS = 10


def codes_of(folder: Path) -> list[str]:
    try:
        read_folder(folder)
    except LoadRefused as refused:
        return refused.codes()
    return []


def test_version_eleven_writes_a_mediums_fields_by_role_and_reads_them_back(tmp_path: Path):
    from cases import CASES

    assert MEDIUM_FIELDS_ADDED == 11 < CURRENT
    assert DOCUMENT_SCHEMAS[MEDIUM_FIELDS_ADDED] == "mime.document/11" and LOCK_SCHEMAS[MEDIUM_FIELDS_ADDED] == \
        "mime.lock/11"
    experiment = next(case for case in CASES if case.name == "periodic_tube_flow").make()
    lock = experiment.save(tmp_path / "flow")
    assert lock["schema"] == LOCK_SCHEMAS[CURRENT]
    water = yaml.safe_load((tmp_path / "flow" / DOCUMENT_FILE).read_text(encoding="utf-8"))["things"]["water"]
    assert "dynamics" not in water and list(water["fields"]) == ["flow"] and water["fields"]["flow"]["name"] == "fvm"
    loaded = Experiment.load(tmp_path / "flow")
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()


def test_the_form_ten_flow_folder_is_the_folder_its_commit_wrote_byte_for_byte():
    found = {path.relative_to(FLOW10).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(FLOW10.rglob("*")) if path.is_file()}
    assert found == PINNED


def test_a_form_ten_folder_with_a_flow_loads_rebuilds_and_verifies(tmp_path: Path):
    folder = Path(shutil.copytree(FLOW10, tmp_path / "flow"))
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[1:3] == [
        f"saved form: mime.document/10, read as version {CURRENT}", "input hash: reproduced exactly"]
    report = loaded.build().verify()
    assert report.reproduced


def test_control_the_form_ten_flow_folder_without_the_write_back_is_refused_by_its_input_hash(tmp_path: Path):
    folder = Path(shutil.copytree(FLOW10, tmp_path / "flow"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "_flow_of_fields", lambda medium: medium)
        with pytest.raises(LoadRefused) as refused:
            Experiment.load(folder)
    assert refused.value.codes() == ["load.input-hash"]


def test_a_form_ten_document_whose_medium_holds_fields_is_refused_and_the_same_document_without_them_loads(
        tmp_path: Path):
    folder = Path(shutil.copytree(FLOW10, tmp_path / "flow"))
    assert codes_of(folder) == [], "the control: the folder as its commit wrote it"
    path = folder / DOCUMENT_FILE
    text = path.read_text(encoding="utf-8")
    assert text.count("    dynamics:\n      flow:\n") == 1
    path.write_text(text.replace("    dynamics:\n      flow:\n", "    fields:\n      flow:\n"), encoding="utf-8")
    assert codes_of(folder) == ["load.document-shape"]


@contextmanager
def reader_of(newest: int) -> Iterator[None]:
    """The loader reading the forms up to ``newest``: version 10's, this MIME before the step."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_folder, "DOCUMENT_SCHEMAS", {v: DOCUMENT_SCHEMAS[v] for v in range(1, newest + 1)})
        patch.setattr(_folder, "LOCK_SCHEMAS", {v: LOCK_SCHEMAS[v] for v in range(1, newest + 1)})
        yield


def restated(folder: Path, out: Path, version: int) -> Path:
    copy = Path(shutil.copytree(folder, out))
    path = copy / DOCUMENT_FILE
    path.write_text(re.sub(r"^schema: .*$", f"schema: {DOCUMENT_SCHEMAS[version]}", path.read_text(encoding="utf-8"),
                           count=1, flags=re.MULTILINE), encoding="utf-8")
    lock = json.loads((copy / LOCK_FILE).read_text(encoding="utf-8"))
    lock["schema"] = LOCK_SCHEMAS[version]
    (copy / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return copy


@pytest.fixture(scope="module")
def current_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from mime.testing.toys import full_toy

    root = tmp_path_factory.mktemp("form11")
    full_toy().save(root / "full_toy")
    return root / "full_toy"


def test_version_tens_reader_refuses_a_folder_of_version_eleven_as_newer_each_file_once(current_folder: Path):
    assert codes_of(current_folder) == [], "this MIME reads it"
    with reader_of(FORM10_READS):
        assert codes_of(current_folder) == [NEWER, NEWER]


def test_control_the_form_left_unstepped_is_read_by_version_tens_reader_with_no_newer_form(current_folder: Path,
                                                                                           tmp_path: Path):
    unstepped = restated(current_folder, tmp_path / "unstepped", FORM10_READS)
    with reader_of(FORM10_READS):
        assert NEWER not in codes_of(unstepped)
