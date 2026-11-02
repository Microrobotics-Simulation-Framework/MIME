"""A saved folder written by a newer MIME is refused as ``load.newer-form``: the document (``mime.document/N``) or the
lock (``mime.lock/N``) states a version above the newest this MIME reads, and the one refusal names the file, the
version found, the versions this MIME reads and the advice to upgrade, instead of the schema check's list of unknown
keys. The controls: the folder as saved loads; a version this MIME reads is never ``load.newer-form``; nor is an id of
another family, which stays the schema problem it is."""

import json
import re
from pathlib import Path

import pytest

from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, read_folder
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS
from mime.testing.toys import full_toy

NEWER = "load.newer-form"


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    full_toy().save(tmp_path)
    return tmp_path


def codes(folder: Path) -> list[str]:
    with pytest.raises(LoadRefused) as refused:
        read_folder(folder)
    return [each.code for each in refused.value.refusals]


def restate_document(folder: Path, schema: str) -> None:
    path = folder / DOCUMENT_FILE
    text = re.sub(r"^schema: .*$", f"schema: {schema}", path.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def restate_lock(folder: Path, schema: str) -> None:
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    lock["schema"] = schema
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")


def test_the_folder_as_saved_loads(saved: Path):
    assert read_folder(saved).version == CURRENT


@pytest.mark.parametrize(("restate", "family", "file", "reads"), [
    (restate_document, "mime.document", DOCUMENT_FILE, DOCUMENT_SCHEMAS),
    (restate_lock, "mime.lock", LOCK_FILE, LOCK_SCHEMAS)])
def test_a_form_one_above_the_newest_is_refused_naming_the_file_the_version_and_the_advice(
        saved: Path, restate, family: str, file: str, reads):
    restate(saved, f"{family}/{CURRENT + 1}")
    with pytest.raises(LoadRefused) as refused:
        read_folder(saved)
    (refusal,) = refused.value.refusals
    assert refusal.code == NEWER and refusal.category.value == "user"
    assert str(saved / file) in refusal.message and f"{family}/{CURRENT + 1}" in refusal.message
    assert ", ".join(reads.values()) in refusal.message
    assert "mime-engine" in refusal.action


@pytest.mark.parametrize(("restate", "family"), [(restate_document, "mime.document"), (restate_lock, "mime.lock")])
def test_a_version_this_mime_reads_or_another_family_is_never_the_newer_form_refusal(saved: Path, restate, family: str):
    other_family = f"other.{family.split('.')[1]}"
    for schema in (f"{family}/{CURRENT - 1}", f"{family}/0", f"{other_family}/{CURRENT + 1}", f"{family}/x"):
        restate(saved, schema)
        assert NEWER not in codes(saved), schema
        restate(saved, DOCUMENT_SCHEMAS[CURRENT] if family.endswith("document") else LOCK_SCHEMAS[CURRENT])
    assert read_folder(saved).version == CURRENT


def test_a_document_and_a_lock_that_are_both_newer_are_each_refused_once(saved: Path):
    restate_document(saved, f"mime.document/{CURRENT + 3}")
    restate_lock(saved, f"mime.lock/{CURRENT + 3}")
    assert codes(saved) == [NEWER, NEWER]
