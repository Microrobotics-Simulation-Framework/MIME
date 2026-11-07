"""Saved form 8: the lock names the program its reference outputs were run by (``reference.program``, ``one-step`` where
the save took them from a run chunked by one step, absent where they are the history scan's) and records how many CPUs
the build's process could use (``environment.usable_cpus``) and how many threads its BLAS used
(``environment.blas_threads``). All three are outside the input hash; the document holds no new key, its schema moving
with the lock's.

A lock of version 7 reads under the current form as it was: it names no program, so ``verify`` reruns the history
scan, and it records no count, so its environment is compared without one (``form7/timed_toy``, written by MIME at the
last commit of form 7, loads and verifies; its environment's lines never name the count). One of version 7 that holds
either key is refused, never read as declaring it. The committed folders of either form hold the hashes their
documents make: saving again under form 8 moved the ``document`` and ``input`` hashes, never ``derived`` or
``observations``. A reader of version 7 (this MIME before the step) and one of version 6 (0.3.0.dev1's) refuse a folder
of version 8 or later as ``load.newer-form``, the document and the lock each once. The current form is version 10
(``test_form10.py``); the folder as version 8 wrote it is ``form8/chain_screw``.

**The controls.** An upgrade planted to accept the keys in an earlier lock reads it, and the refusal test fails on every
run. The form left unstepped (the folder's files stating version 7) is read by version 7's reader without a
``load.newer-form``, so the refusal is the step's."""

import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, Lock, LoadRefused, conforms, read_folder
from mime.saved import _folder, _versions
from mime.saved._document import document_hash, parse_yaml
from mime.saved._record import input_hash
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS, PROGRAM_ADDED, upgraded_lock
from mime.testing.toys import full_toy

SAVED = Path(__file__).parent
FORM7 = SAVED / "form7" / "chain_screw"
FORM8 = SAVED / "form8" / "chain_screw"
#: The timed toy saved by MIME at the last commit of form 7 (f2cdac6), as ``form6/timed_toy`` was by the 0.3.0.dev1
#: wheel: float64, one BLAS thread, ``test_based_on.py``'s declarations.
TIMED7 = SAVED / "form7" / "timed_toy"
TIMED7_SHA256 = {
    "experiment.mime.yaml": "e84146c969a973d8a32ea53113c51816bbaccbbd3499a1ee3adcd6568d0a7355",
    "mime.lock": "e0b384d45812b81a07e2952f28dfd4eaeb2cd646f240998b28a74f9e60a998a7",
    "reference.npz": "84743bcaeaa0948ccd58deaab4ca4461029a6f776959aec5a319348ceb570159",
    "scene/scene.usda": "78a93ae11ce3de31021b76e56c8f7d849d2ee1c1a292bdf78f83a7b136132039",
}
RUNS = 20
#: The lock's keys version 8 added.
KEYS = ["reference.program", "environment.usable_cpus", "environment.blas_threads"]
NEWER = "load.newer-form"
#: The forms each earlier reader reads: this MIME before the step, and 0.3.0.dev1.
FORM7_READS, DEV1_READS = 7, 6


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def test_version_eight_added_the_program_and_a_version_seven_lock_reads_under_the_current_form():
    assert PROGRAM_ADDED == 8 < CURRENT
    assert DOCUMENT_SCHEMAS[PROGRAM_ADDED] == "mime.document/8" and LOCK_SCHEMAS[PROGRAM_ADDED] == "mime.lock/8"
    held, version = upgraded_lock(lock_of(FORM7))
    assert version == 7 and conforms(held, Lock)[1] == []
    assert "usable_cpus" not in held["environment"] and "blas_threads" not in held["environment"]
    assert "program" not in held["reference"]


@pytest.mark.parametrize("folder", [FORM7, FORM8], ids=["form7", "form8"])
def test_a_committed_folders_lock_holds_the_hashes_its_document_makes(folder: Path):
    document = parse_yaml((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    lock = lock_of(folder)
    assert lock["hashes"]["document"] == document_hash(document)  # type: ignore[arg-type]
    assert lock["hashes"]["input"] == input_hash(document, lock["resolved"], lock["entries"])  # type: ignore[arg-type]


def test_saving_again_under_form_eight_moved_the_document_and_input_hashes_and_never_the_others():
    seven, eight = lock_of(FORM7)["hashes"], lock_of(FORM8)["hashes"]
    assert [key for key in ("document", "input", "derived", "observations") if seven[key] != eight[key]] == \
        ["document", "input"]


def test_a_save_records_the_count_and_a_save_with_no_run_names_no_program(tmp_path: Path):
    """``built.save`` straight after a build has run nothing chunked by one step: its reference outputs are the history
    scan's, and the lock names no program."""
    full_toy().save(tmp_path / "toy")
    lock = lock_of(tmp_path / "toy")
    assert lock["schema"] == LOCK_SCHEMAS[CURRENT] and "program" not in lock["reference"]
    assert isinstance(lock["environment"]["usable_cpus"], int) and lock["environment"]["usable_cpus"] >= 1
    assert isinstance(lock["environment"]["blas_threads"], int) and lock["environment"]["blas_threads"] >= 1


# --- an earlier lock holding a key version 8 added ----------------------------------------------------------------


def an_earlier_lock(tmp_path: Path, key: str) -> Path:
    """The form-7 timed toy with ``key`` (``reference.program``, ``environment.usable_cpus`` or ``environment.blas_threads``) written into its lock."""
    folder = Path(shutil.copytree(TIMED7, tmp_path))
    lock = lock_of(folder)
    parent, name = key.split(".")
    lock[parent][name] = "one-step" if name == "program" else 4
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return folder


def an_earlier_lock_holding_the_key_is_refused(tmp_path: Path, key: str) -> None:
    folder = an_earlier_lock(tmp_path, key)
    with pytest.raises(LoadRefused) as refused:
        read_folder(folder)
    assert {each.code for each in refused.value.refusals} == {"load.lock-shape"}
    assert any(f"a mime.lock/7 lock holds {key}, which version 8 of the saved form added" in each.message
               for each in refused.value.refusals), [each.message for each in refused.value.refusals]


@pytest.mark.parametrize("key", KEYS)
def test_an_earlier_lock_holding_a_key_version_eight_added_is_refused(tmp_path: Path, key: str):
    an_earlier_lock_holding_the_key_is_refused(tmp_path / "folder", key)


@contextmanager
def keys_accepted_early() -> Iterator[None]:
    """The planted upgrade: version 8's keys read in a lock of any version."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "program_keys_held", lambda data: [])
        patch.setattr(_folder, "program_keys_held", lambda data: [])
        yield


@pytest.mark.parametrize("key", KEYS)
def test_control_an_upgrade_that_accepts_the_keys_in_an_earlier_lock_is_caught_every_run(tmp_path: Path, key: str):
    caught = 0
    for run in range(RUNS):
        with keys_accepted_early():
            try:
                an_earlier_lock_holding_the_key_is_refused(tmp_path / str(run), key)
            except (AssertionError, pytest.fail.Exception):
                caught += 1
    assert caught == RUNS


# --- a folder of version 7 reads, verifies and is compared as before ----------------------------------------------


def test_the_form_seven_timed_toy_is_the_folder_form_seven_wrote_byte_for_byte():
    found = {path.relative_to(TIMED7).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(TIMED7.rglob("*")) if path.is_file()}
    assert found == TIMED7_SHA256
    assert (parse_yaml((TIMED7 / DOCUMENT_FILE).read_text(encoding="utf-8"))["schema"], lock_of(TIMED7)["schema"]) == \
        ("mime.document/7", "mime.lock/7")


def test_a_version_seven_folder_loads_and_verifies_by_the_history_scan_and_its_environment_names_no_count(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """It names no program, so ``verify`` reruns the history scan (the program is what ``_history_on`` is asked for);
    it records neither count, so no line of its environment names one, here or under another count."""
    from mime.experiment import _experiment

    folder = Path(shutil.copytree(TIMED7, tmp_path / "timed_toy"))
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[1:3] == [
        f"saved form: mime.document/7, read as version {CURRENT}", "input hash: reproduced exactly"]
    asked: list[object] = []
    real = _experiment.BuiltExperiment._history_on

    def recorded(self, graph, steps, program):
        asked.append(program)
        return real(self, graph, steps, program)

    monkeypatch.setattr(_experiment.BuiltExperiment, "_history_on", recorded)
    built = loaded.build()
    assert built.verify().reproduced and asked == [None]
    monkeypatch.setattr("mime.saved._record.usable_cpus", lambda: 1_000_003)
    monkeypatch.setattr("mime.saved._record.blas_threads", lambda: 1_000_003)
    assert not any("usable_cpus" in line or "blas_threads" in line
                   for line in _folder.environment_differences(read_folder(folder).lock))


# --- the readers before the step ----------------------------------------------------------------------------------


@contextmanager
def reader_of(newest: int) -> Iterator[None]:
    """The loader reading the forms up to ``newest``: version 7's, this MIME before the step; version 6's, 0.3.0.dev1."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_folder, "DOCUMENT_SCHEMAS", {v: DOCUMENT_SCHEMAS[v] for v in range(1, newest + 1)})
        patch.setattr(_folder, "LOCK_SCHEMAS", {v: LOCK_SCHEMAS[v] for v in range(1, newest + 1)})
        yield


def codes_of(folder: Path) -> list[str]:
    try:
        read_folder(folder)
    except LoadRefused as refused:
        return refused.codes()
    return []


@pytest.fixture(scope="module")
def current_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("form8")
    full_toy().save(root / "full_toy")
    return root / "full_toy"


@pytest.mark.parametrize("newest", [FORM7_READS, DEV1_READS], ids=["form7_reader", "dev1_reader"])
def test_an_earlier_reader_refuses_a_folder_of_version_eight_as_newer_each_file_once(current_folder: Path,
                                                                                     newest: int):
    assert codes_of(current_folder) == [], "this MIME reads it"
    with reader_of(newest):
        assert codes_of(current_folder) == [NEWER, NEWER]


def restated(folder: Path, out: Path, version: int) -> Path:
    """``folder`` with its document and lock stating the form ``version``, as a MIME that had not stepped the form
    would have written them."""
    copy = Path(shutil.copytree(folder, out))
    path = copy / DOCUMENT_FILE
    path.write_text(re.sub(r"^schema: .*$", f"schema: {DOCUMENT_SCHEMAS[version]}", path.read_text(encoding="utf-8"),
                           count=1, flags=re.MULTILINE), encoding="utf-8")
    lock = lock_of(copy)
    lock["schema"] = LOCK_SCHEMAS[version]
    (copy / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return copy


def test_control_the_form_left_unstepped_is_read_by_version_sevens_reader_with_no_newer_form(current_folder: Path,
                                                                                             tmp_path: Path):
    """The folder's files stating version 7 while its lock holds the count: version 7's reader does not refuse it as
    newer (it judges the key by its schema), so the refusal above is the step's and nothing else's."""
    unstepped = restated(current_folder, tmp_path / "unstepped", FORM7_READS)
    with reader_of(FORM7_READS):
        found = codes_of(unstepped)
    assert NEWER not in found
