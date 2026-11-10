"""Saved form 12: one key added to the document, ``ranges``, each range declared in the experiment
(``exp.declare_range``) by its path: its ends in SI, their unit and its reason, written only where one is declared.
Nothing else changed; the lock's schema moves with the document's. A folder of form 10 is read through form 11's step
(its medium's flow as ``fields.flow``) and then this one (``tests/saved/test_form11.py`` loads the form-10 flow folder
under this MIME).

The folders of the eleventh form:

- ``form11/periodic_flow``: the round-trip case ``periodic_tube_flow`` (the FVM on a periodic box, IMEX Euler with
  convection, eight cells across, driven by ``BodyForce``'s steady and pulsatile parts), written by MIME at the last
  commit of form 11 (46fcc79, F1a-2's second PR), float64, four usable CPUs (``taskset -c 0-3``), one BLAS thread,
  saved by ``exp.build().save(folder)``, pinned by its sha256;
- ``form11/chain_screw``: the current folder as form 11 wrote it (a document and a lock); ``current/chain_screw`` is
  it restated to form 12 (its schema lines and its document and input hashes, no value moved).

A reader of version 11 refuses a folder of version 12 as ``load.newer-form``, the document and the lock each once.

**The controls.** The form left unstepped (the folder's files stating version 11) is read by version 11's reader
without a ``load.newer-form``, so the refusal is the step's. A form-11 document holding ``ranges`` is refused, where the
same document without it is read."""

from __future__ import annotations

import hashlib
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
FORM11_CHAIN = SAVED / "form11" / "chain_screw"
FLOW11 = SAVED / "form11" / "periodic_flow"
CURRENT_FOLDER = SAVED / "current" / "chain_screw"
PINNED = {"experiment.mime.yaml": "f7ea736b84fb0637039741e06e9d033dbd884b767c240b1a7dd2ba0ab69cb480",
          "mime.lock": "769d6932ce38a5f95b8b83209c63c6ee02e6c8247b6037935f7e6ff54b91e7e7",
          "reference.npz": "92e9160facde462c378e392e004ccbbb61c3afff768cb0c5e7184ca318422c75",
          "scene/scene.usda": "ada6dcbfd8ce8e660a83aae1ed016895ac3002524a281a3515ec0d646a665cf0"}
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM11_READS = 11


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


def test_version_twelve_writes_a_declared_range_into_the_document_and_reads_it_back(tmp_path: Path):
    assert CURRENT == RANGES_ADDED == 12
    assert DOCUMENT_SCHEMAS[CURRENT] == "mime.document/12" and LOCK_SCHEMAS[CURRENT] == "mime.lock/12"
    lock = ranged().save(tmp_path / "ranged", steps=2)
    document = yaml.safe_load((tmp_path / "ranged" / DOCUMENT_FILE).read_text(encoding="utf-8"))
    assert lock["schema"] == "mime.lock/12" and document["schema"] == "mime.document/12"
    assert document[RANGES_KEY] == {"ball.parts.shell.radius": {
        "low": 0.001, "high": 0.00115, "unit": "m", "reason": "the design range of the ball's radius"}}
    loaded = Experiment.load(tmp_path / "ranged")
    assert loaded.load_report is not None and loaded.load_report.text().splitlines()[2] == \
        "input hash: reproduced exactly"
    (held,) = [d for d in loaded.declared_as()[1] if isinstance(d, RangeDeclaration)]
    assert str(held.path) == "ball.parts.shell.radius"
    assert loaded.build().graph.get_node("drag:ball").sample_count == 91, "the load builds over the range"


@pytest.mark.parametrize("folder", [FORM11_CHAIN, CURRENT_FOLDER], ids=["form11", "current"])
def test_a_committed_folders_lock_holds_the_hashes_its_document_makes(folder: Path):
    document = parse_yaml((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    lock = lock_of(folder)
    assert lock["hashes"]["document"] == document_hash(document)  # type: ignore[arg-type]
    assert lock["hashes"]["input"] == input_hash(document, lock["resolved"], lock["entries"])  # type: ignore[arg-type]


def test_saving_again_under_form_twelve_moves_the_document_and_input_hashes_and_never_the_others():
    eleven, twelve = lock_of(FORM11_CHAIN), lock_of(CURRENT_FOLDER)
    assert [key for key in ("document", "input", "derived", "observations")
            if eleven["hashes"][key] != twelve["hashes"][key]] == ["document", "input"]
    assert {key: value for key, value in eleven.items() if key not in ("schema", "hashes")} == \
        {key: value for key, value in twelve.items() if key not in ("schema", "hashes")}


def test_the_form_eleven_flow_folder_is_the_folder_its_commit_wrote_byte_for_byte():
    found = {path.relative_to(FLOW11).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(FLOW11.rglob("*")) if path.is_file()}
    assert found == PINNED


def test_a_form_eleven_folder_with_a_flow_loads_rebuilds_and_verifies(tmp_path: Path):
    folder = Path(shutil.copytree(FLOW11, tmp_path / "flow"))
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[1:3] == [
        f"saved form: mime.document/11, read as version {CURRENT}", "input hash: reproduced exactly"]
    assert not [d for d in loaded.declared_as()[1] if isinstance(d, RangeDeclaration)], "form 11 declares no range"
    report = loaded.build().verify()
    assert report.reproduced


def test_a_form_eleven_document_holding_ranges_is_refused_and_the_same_document_without_it_is_read(tmp_path: Path):
    text = (FORM11_CHAIN / DOCUMENT_FILE).read_text(encoding="utf-8")
    assert text.startswith("schema: mime.document/11\n") or "\nschema: mime.document/11\n" in text
    plain = tmp_path / "plain.mime.yaml"
    plain.write_text(text, encoding="utf-8")
    assert read_document(plain)[1] == [], "the control: the form-11 document as written"
    document = yaml.safe_load(text)
    document[RANGES_KEY] = {"umr.parts.shell.radius": {"low": 1e-3, "high": 2e-3, "unit": "m", "reason": "planted"}}
    held = tmp_path / "held.mime.yaml"
    held.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    assert read_document(held)[1] != []


@contextmanager
def reader_of(newest: int) -> Iterator[None]:
    """The loader reading the forms up to ``newest``: version 11's, this MIME before the step."""
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


def test_version_elevens_reader_refuses_a_folder_of_version_twelve_as_newer_each_file_once(tmp_path: Path):
    from mime.testing.toys import full_toy

    full_toy().save(tmp_path / "full_toy")
    folder = tmp_path / "full_toy"
    assert codes_of(folder) == [], "this MIME reads it"
    with reader_of(FORM11_READS):
        assert codes_of(folder) == [NEWER, NEWER]
    unstepped = restated(folder, tmp_path / "unstepped", FORM11_READS)
    with reader_of(FORM11_READS):
        assert NEWER not in codes_of(unstepped), "the control: the form left unstepped"
