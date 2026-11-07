"""Saved form 7: the document records the saved experiment it was made from (``based_on``), written only where it was
made from one, inside the document and input hashes. A document of version 6 reads under the current form as it was;
one of version 6 that holds ``based_on`` is refused, never read as declaring it. The committed folders of either form
hold the hashes their documents make: saving again under form 7 moved the ``document`` and ``input`` hashes (the
document names its form), never ``derived`` or ``observations``. A reader of version 6, that of 0.3.0.dev1, refuses a folder
of version 7 or later as ``load.newer-form``, the document and the lock each once, rather than reading a key it does not
know. The current form is version 9 (``test_form9.py``); the folder as version 7 wrote it is ``form7/chain_screw``.

**The controls.** An upgrade planted to accept ``based_on`` in an earlier document reads it, and the refusal test
fails on every run. The form left unstepped (the folder's files stating version 6) is read by version 6's reader
without a ``load.newer-form``, so the refusal is the step's."""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, ExperimentDocument, Lock, LoadRefused, conforms, read_document, \
    read_folder
from mime.saved import _folder, _versions
from mime.saved._document import document_hash, parse_yaml
from mime.saved._record import input_hash
from mime.saved._versions import BASED_ON_ADDED, CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS, upgraded_document, \
    upgraded_lock
from mime.testing.toys import full_toy

FORM6 = Path(__file__).parent / "form6" / "chain_screw"
FORM7 = Path(__file__).parent / "form7" / "chain_screw"
RUNS = 20
NEWER = "load.newer-form"
#: What 0.3.0.dev1 reads, the last release of version 6: the forms up to it (its ``_versions.py`` at the tag).
DEV1_READS = 6


def document6() -> dict[str, object]:
    return yaml.safe_load((FORM6 / DOCUMENT_FILE).read_text(encoding="utf-8"))


def test_version_seven_added_based_on_and_a_version_six_document_and_lock_read_under_the_current_form():
    assert BASED_ON_ADDED == 7 < CURRENT
    assert DOCUMENT_SCHEMAS[BASED_ON_ADDED] == "mime.document/7" and LOCK_SCHEMAS[BASED_ON_ADDED] == "mime.lock/7"
    upgraded, version, renamed = upgraded_document(document6())
    assert (version, renamed) == (6, ()) and conforms(upgraded, ExperimentDocument)[1] == []
    held, lock_version = upgraded_lock(json.loads((FORM6 / LOCK_FILE).read_text(encoding="utf-8")))
    assert lock_version == 6 and conforms(held, Lock)[1] == []


@pytest.mark.parametrize("folder", [FORM6, FORM7], ids=["form6", "form7"])
def test_a_committed_folders_lock_holds_the_hashes_its_document_makes(folder: Path):
    document = parse_yaml((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["hashes"]["document"] == document_hash(document)  # type: ignore[arg-type]
    assert lock["hashes"]["input"] == input_hash(document, lock["resolved"], lock["entries"])  # type: ignore[arg-type]


def test_saving_again_under_form_seven_moved_the_document_and_input_hashes_and_never_the_others():
    """The same experiment as version 6 and as version 7 wrote it: its document names its form, and so its document
    and input hashes moved; what it computes and observes did not."""
    six = json.loads((FORM6 / LOCK_FILE).read_text(encoding="utf-8"))["hashes"]
    seven = json.loads((FORM7 / LOCK_FILE).read_text(encoding="utf-8"))["hashes"]
    assert [key for key in ("document", "input", "derived", "observations") if six[key] != seven[key]] == \
        ["document", "input"]


def test_a_version_six_document_that_declares_a_run_length_reads(tmp_path: Path):
    """Version 5 added ``run.duration``, so a document of version 6 may hold it (the check once refused it in any
    version below the current one, which would have refused every version 6 folder that declares a run's length)."""
    held = document6()
    held["run"] = {**held["run"], "duration": {"value": 0.005, "unit": "s"}}  # type: ignore[dict-item]
    path = tmp_path / DOCUMENT_FILE
    path.write_text(yaml.safe_dump(held, sort_keys=False), encoding="utf-8")
    assert read_document(path)[1] == []


def test_a_document_made_from_none_holds_no_based_on(tmp_path: Path):
    full_toy().save(tmp_path / "toy")
    document = yaml.safe_load((tmp_path / "toy" / DOCUMENT_FILE).read_text(encoding="utf-8"))
    assert document["schema"] == DOCUMENT_SCHEMAS[CURRENT] and "based_on" not in document


def an_earlier_document_holding_based_on_is_refused(tmp_path: Path) -> None:
    held = {**document6(), "based_on": {"experiment": "a", "input_hash": "sha256:" + "ab" * 32, "mime": "0.3.0.dev1"}}
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / DOCUMENT_FILE
    path.write_text(yaml.safe_dump(held, sort_keys=False), encoding="utf-8")
    _, refusals = read_document(path)
    assert refusals and {each.code for each in refusals} == {"load.document-shape"}
    assert any("holds based_on, which version 7 of the saved form added" in each.message for each in refusals), \
        [each.message for each in refusals]


def test_an_earlier_document_holding_based_on_is_refused(tmp_path: Path):
    an_earlier_document_holding_based_on_is_refused(tmp_path)


@contextmanager
def based_on_accepted_early() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "based_on_held", lambda data: [])
        patch.setattr(_folder, "based_on_held", lambda data: [])
        yield


def test_control_an_upgrade_that_accepts_based_on_in_an_earlier_document_is_caught_every_run(tmp_path: Path):
    caught = 0
    for run in range(RUNS):
        with based_on_accepted_early():
            try:
                an_earlier_document_holding_based_on_is_refused(tmp_path / str(run))
            except AssertionError:
                caught += 1
    assert caught == RUNS


# --- the reader of 0.3.0.dev1 -------------------------------------------------------------------------------------------------


@contextmanager
def dev1s_reader() -> Iterator[None]:
    """The loader as 0.3.0.dev1 ships it: the same reader, reading the forms up to version 6 (measured against the
    released code itself as well, from the tag, when the form was stepped: both files ``load.newer-form``)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_folder, "DOCUMENT_SCHEMAS", {v: DOCUMENT_SCHEMAS[v] for v in range(1, DEV1_READS + 1)})
        patch.setattr(_folder, "LOCK_SCHEMAS", {v: LOCK_SCHEMAS[v] for v in range(1, DEV1_READS + 1)})
        yield


def codes_of(folder: Path) -> list[str]:
    try:
        read_folder(folder)
    except LoadRefused as refused:
        return refused.codes()
    return []


@pytest.fixture(scope="module")
def based_on_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("form7")
    full_toy().save(root / "full_toy")
    Experiment.based_on(root / "full_toy", name="child").save(root / "child")
    return root / "child"


def test_dev1s_reader_refuses_a_folder_of_the_current_form_as_newer_each_file_once(based_on_folder: Path):
    assert codes_of(based_on_folder) == [], "this MIME reads it"
    with dev1s_reader():
        assert codes_of(based_on_folder) == [NEWER, NEWER]


def restated(folder: Path, out: Path, version: int) -> Path:
    """``folder`` with its document and lock stating the form ``version``, as a MIME that had not stepped the form
    would have written them."""
    import shutil

    copy = Path(shutil.copytree(folder, out))
    path = copy / DOCUMENT_FILE
    path.write_text(re.sub(r"^schema: .*$", f"schema: {DOCUMENT_SCHEMAS[version]}", path.read_text(encoding="utf-8"),
                           count=1, flags=re.MULTILINE), encoding="utf-8")
    lock = json.loads((copy / LOCK_FILE).read_text(encoding="utf-8"))
    lock["schema"] = LOCK_SCHEMAS[version]
    (copy / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return copy


def test_control_the_form_left_unstepped_is_read_by_dev1s_reader_with_no_newer_form(based_on_folder: Path,
                                                                                     tmp_path: Path):
    """The folder's files stating version 6 while the document holds ``based_on``: version 6's reader does not
    refuse it as newer (it judges the key by its schema), so the refusal above is the step's and nothing else's."""
    unstepped = restated(based_on_folder, tmp_path / "unstepped", DEV1_READS)
    with dev1s_reader():
        found = codes_of(unstepped)
    assert NEWER not in found
