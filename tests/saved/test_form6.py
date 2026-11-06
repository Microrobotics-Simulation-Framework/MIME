"""Saved form 6: the lock holds each asset's record whole and the manifests of the meshes its stage draws, both
outside the input hash. A lock of version 5 reads under the current form as it was; one of version 5 that holds a key
version 6 added is refused, never read as stating it. The committed folders of either form hold the hashes their
documents make: saving again under form 6 moves the ``document`` and ``input`` hashes (the document names its form),
never ``derived`` or ``observations``. The control: an upgrade planted to accept the keys in an earlier lock reads
them, and the refusal test fails on every run."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime.saved import Lock, conforms
from mime.saved import _versions
from mime.saved._versions import RECORDS_ADDED, records_held, upgraded_lock

FORM5 = Path(__file__).parent / "form5" / "chain_screw"
#: The same folder as version 6 wrote it (the current form is version 7: ``test_form7.py``).
FORM6 = Path(__file__).parent / "form6" / "chain_screw"
RUNS = 20


def lock5() -> dict[str, object]:
    return json.loads((FORM5 / "mime.lock").read_text(encoding="utf-8"))


def test_a_version_five_lock_reads_under_the_current_form():
    held, version = upgraded_lock(lock5())
    assert version == 5 == RECORDS_ADDED - 1 and conforms(held, Lock)[1] == []


@pytest.mark.parametrize("folder", [FORM5, FORM6], ids=["form5", "form6"])
def test_a_committed_folders_lock_holds_the_hashes_its_document_makes(folder: Path):
    from mime.saved._document import document_hash, parse_yaml
    from mime.saved._record import input_hash

    document = parse_yaml((folder / "experiment.mime.yaml").read_text(encoding="utf-8"))
    lock = json.loads((folder / "mime.lock").read_text(encoding="utf-8"))
    assert lock["hashes"]["document"] == document_hash(document)  # type: ignore[arg-type]
    assert lock["hashes"]["input"] == input_hash(document, lock["resolved"], lock["entries"])  # type: ignore[arg-type]


def test_saving_again_under_form_six_moves_the_document_and_input_hashes_and_never_the_others():
    """The same experiment, saved under form 5 and under form 6: its document names its form, and so its document
    and input hashes move; what it computes and observes does not."""
    five, six = lock5()["hashes"], json.loads((FORM6 / "mime.lock").read_text(encoding="utf-8"))["hashes"]
    assert [key for key in ("document", "input", "derived", "observations")
            if five[key] != six[key]] == ["document", "input"]  # type: ignore[index]


def test_the_version_six_lock_reads_under_the_current_form():
    held, version = upgraded_lock(json.loads((FORM6 / "mime.lock").read_text(encoding="utf-8")))
    assert version == 6 and conforms(held, Lock)[1] == []


@pytest.mark.parametrize("change", ["visuals", "record"])
def test_an_earlier_lock_holding_a_key_version_six_added_is_refused(change: str):
    held = lock5()
    if change == "visuals":
        held["visuals"] = {}
    else:
        held["assets"] = [{"name": "a", "sha256": "ab" * 32, "size": 1, "licence": "CC0-1.0",
                           "locations": ["https://example.org/a"], "kind": "derived"}]
    assert records_held(held)
    upgraded, version = upgraded_lock(held)
    assert version == 5 and conforms(upgraded, Lock)[1] != []


@contextmanager
def keys_accepted() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "records_held", lambda data: [])
        yield


def test_control_an_upgrade_that_accepts_the_keys_in_an_earlier_lock_is_caught_every_run():
    caught = 0
    for _ in range(RUNS):
        with keys_accepted():
            try:
                test_an_earlier_lock_holding_a_key_version_six_added_is_refused("visuals")
            except AssertionError:
                caught += 1
    assert caught == RUNS
