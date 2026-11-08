"""Saved form 11: one key added to the document, ``ranges``, each range declared in the experiment
(``exp.declare_range``) by its path: its ends in SI, their unit and its reason, written only where one is declared.
Nothing else changed; the lock's schema moves with the document's.

The folder of the tenth form: ``form10/chain_screw``, the current folder as form 10 wrote it (a document and a lock);
``current/chain_screw`` restated to form 11 (its schema lines and its document and input hashes, no value moved).

A reader of version 10 refuses a folder of version 11 as ``load.newer-form``, the document and the lock each once.

**The controls.** The form left unstepped (the folder's files stating version 10) is read by version 10's reader
without a ``load.newer-form``, so the refusal is the step's. A form-10 document holding ``ranges`` is refused, where the
same document without it is read."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from mime.build import RangeDeclaration
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, read_document, read_folder
from mime.saved import _folder
from mime.saved._document import document_hash, parse_yaml
from mime.saved._record import input_hash
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS, RANGES_ADDED, RANGES_KEY

SAVED = Path(__file__).parent
FORM10_CHAIN = SAVED / "form10" / "chain_screw"
CURRENT_FOLDER = SAVED / "current" / "chain_screw"
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM10_READS = 10


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def ranged() -> Experiment:
    """A sphere with ``Drag.MRS`` alone, moving from the start, its radius ranged over [1.0, 1.15] mm."""
    from mrs_cases import free_space, mrs

    from mime.scene import Sphere
    from mime.units import kg, kg_m2, m_per_s, mm

    experiment = free_space("ranged", {"ball": Sphere(radius=mm(1.0))}, mrs(mm(0.6)), mass=kg(2e-8),
                            inertia=kg_m2((2e-17, 2e-17, 2e-17)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)))
    experiment.declare_range("ball.parts.shell.radius", mm(1.0), mm(1.15), reason="the design range of the ball's radius")
    return experiment


def test_version_eleven_writes_a_declared_range_into_the_document_and_reads_it_back(tmp_path: Path):
    assert CURRENT == RANGES_ADDED == 11
    assert DOCUMENT_SCHEMAS[CURRENT] == "mime.document/11" and LOCK_SCHEMAS[CURRENT] == "mime.lock/11"
    lock = ranged().save(tmp_path / "ranged", steps=2)
    document = yaml.safe_load((tmp_path / "ranged" / DOCUMENT_FILE).read_text(encoding="utf-8"))
    assert lock["schema"] == "mime.lock/11" and document["schema"] == "mime.document/11"
    assert document[RANGES_KEY] == {"ball.parts.shell.radius": {
        "low": 0.001, "high": 0.00115, "unit": "m", "reason": "the design range of the ball's radius"}}
    loaded = Experiment.load(tmp_path / "ranged")
    assert loaded.load_report is not None and loaded.load_report.text().splitlines()[2] == \
        "input hash: reproduced exactly"
    (held,) = [d for d in loaded.declared_as()[1] if isinstance(d, RangeDeclaration)]
    assert str(held.path) == "ball.parts.shell.radius"
    assert loaded.build().graph.get_node("drag:ball").sample_count == 91, "the load builds over the range"


@pytest.mark.parametrize("folder", [FORM10_CHAIN, CURRENT_FOLDER], ids=["form10", "current"])
def test_a_committed_folders_lock_holds_the_hashes_its_document_makes(folder: Path):
    document = parse_yaml((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    lock = lock_of(folder)
    assert lock["hashes"]["document"] == document_hash(document)  # type: ignore[arg-type]
    assert lock["hashes"]["input"] == input_hash(document, lock["resolved"], lock["entries"])  # type: ignore[arg-type]


def test_saving_again_under_form_eleven_moves_the_document_and_input_hashes_and_never_the_others():
    ten, eleven = lock_of(FORM10_CHAIN), lock_of(CURRENT_FOLDER)
    assert [key for key in ("document", "input", "derived", "observations")
            if ten["hashes"][key] != eleven["hashes"][key]] == ["document", "input"]
    assert {key: value for key, value in ten.items() if key not in ("schema", "hashes")} == \
        {key: value for key, value in eleven.items() if key not in ("schema", "hashes")}


def test_a_form_ten_document_holding_ranges_is_refused_and_the_same_document_without_it_is_read(tmp_path: Path):
    text = (FORM10_CHAIN / DOCUMENT_FILE).read_text(encoding="utf-8")
    plain = tmp_path / "plain.mime.yaml"
    plain.write_text(text, encoding="utf-8")
    assert read_document(plain)[1] == [], "the control: the form-10 document as written"
    document = yaml.safe_load(text)
    document[RANGES_KEY] = {"umr.parts.shell.radius": {"low": 1e-3, "high": 2e-3, "unit": "m", "reason": "planted"}}
    held = tmp_path / "held.mime.yaml"
    held.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    assert read_document(held)[1] != []


@contextmanager
def reader_of(newest: int) -> Iterator[None]:
    """The loader reading the forms up to ``newest``: version 10's, this MIME before the step."""
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


def restated(folder: Path, out: Path, version: int) -> Path:
    copy = Path(shutil.copytree(folder, out))
    path = copy / DOCUMENT_FILE
    path.write_text(re.sub(r"^schema: .*$", f"schema: {DOCUMENT_SCHEMAS[version]}", path.read_text(encoding="utf-8"),
                           count=1, flags=re.MULTILINE), encoding="utf-8")
    lock = lock_of(copy)
    lock["schema"] = LOCK_SCHEMAS[version]
    (copy / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return copy


def test_version_tens_reader_refuses_a_folder_of_version_eleven_as_newer_each_file_once(tmp_path: Path):
    from mime.testing.toys import full_toy

    full_toy().save(tmp_path / "full_toy")
    folder = tmp_path / "full_toy"
    assert codes_of(folder) == [], "this MIME reads it"
    with reader_of(FORM10_READS):
        assert codes_of(folder) == [NEWER, NEWER]
    unstepped = restated(folder, tmp_path / "unstepped", FORM10_READS)
    with reader_of(FORM10_READS):
        assert NEWER not in codes_of(unstepped), "the control: the form left unstepped"
