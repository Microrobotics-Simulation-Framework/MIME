"""Saved form 10: no key added or removed. From the tenth form the lock's ``reference.rtol`` is each field's largest
difference over its source scale (``mime.saved._scales``) and ``atol`` is written 0.0, the tolerance of the save's
precision (1e-5 in float32, 2e-14 in float64), which the folder keeps. A folder of the ninth form or earlier is judged
as its form says, element by element with its own numbers (1e-6 and 1e-9), in ``verify``, ``verify --run`` and the
load's comparison of the derived graph; nothing of it is written back.

The folders of the ninth form, each written by MIME at the last commit of form 9 (ce43659):

- ``form9/free_ball``: a free 1 mm sphere of density 1000 kg/m^3 at the origin moving at 1 mm/s along x and turning at
  (30, -20, 10) rad/s, no gravity and no law, float64, four usable CPUs (``taskset -c 0-3``), one BLAS thread, saved
  by ``exp.build().save(folder)``, pinned by its sha256.
- ``form9/chain_screw``: the current folder as form 9 wrote it (a document and a lock).

A reader of version 9 refuses a folder of version 10 as ``load.newer-form``, the document and the lock each once.

**The controls.** The same planted rerun of ``form9/free_ball`` is reproduced under its own lock and moved under the same
lock restated as form 10: the rule is the form's. The form left unstepped (the folder's files stating version 9) is read
by version 9's reader without a ``load.newer-form``, so the refusal is the step's."""

import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, read_folder, verify_history
from mime.saved import _folder, _scales
from mime.saved._document import document_hash, parse_yaml
from mime.saved._record import input_hash
from mime.saved._scales import FLOAT64_RTOL, tolerance_for
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS, SOURCE_SCALE_ADDED

SAVED = Path(__file__).parent
FREE9 = SAVED / "form9" / "free_ball"
FORM9_CHAIN = SAVED / "form9" / "chain_screw"
CURRENT_FOLDER = SAVED / "current" / "chain_screw"
PINNED = {"experiment.mime.yaml": "f113c23f78f9c3a5116e3b9dc5ac0924b6e3d1478f58b07a0ee86a63d8412cd7",
          "mime.lock": "e023d1da1868263a4915d3782fd47f8bb6b74e5a4d66acd9ddab7ff49f7d5791",
          "reference.npz": "e9feb3f36a9855ae226d1b70ac94f06fb8bf75511cb776452d1896ac3d541b91",
          "scene/scene.usda": "d51cb2b2367c4e4d4fb4a63b1db3db91b3bfd1225b4aeaef5a163d37ab3bcea4"}
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM9_READS = 9


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def test_version_ten_writes_its_precisions_tolerance_with_no_absolute_part(tmp_path: Path):
    from mime.testing.toys import full_toy

    assert CURRENT == SOURCE_SCALE_ADDED == _scales.SCALED_FROM == 10
    assert DOCUMENT_SCHEMAS[CURRENT] == "mime.document/10" and LOCK_SCHEMAS[CURRENT] == "mime.lock/10"
    lock = full_toy().save(tmp_path / "toy")
    assert lock["schema"] == "mime.lock/10"
    assert (lock["reference"]["rtol"], lock["reference"]["atol"]) == (tolerance_for(lock["environment"]["x64"]), 0.0)
    assert tolerance_for(False) == 1e-5 and tolerance_for(True) == 2e-14


@pytest.mark.parametrize("folder", [FORM9_CHAIN, CURRENT_FOLDER], ids=["form9", "current"])
def test_a_committed_folders_lock_holds_the_hashes_its_document_makes(folder: Path):
    document = parse_yaml((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    lock = lock_of(folder)
    assert lock["hashes"]["document"] == document_hash(document)  # type: ignore[arg-type]
    assert lock["hashes"]["input"] == input_hash(document, lock["resolved"], lock["entries"])  # type: ignore[arg-type]


def test_saving_again_under_form_ten_moves_the_document_and_input_hashes_and_the_tolerance_and_never_the_others():
    nine, ten = lock_of(FORM9_CHAIN), lock_of(CURRENT_FOLDER)
    assert [key for key in ("document", "input", "derived", "observations")
            if nine["hashes"][key] != ten["hashes"][key]] == ["document", "input"]
    assert (nine["reference"]["rtol"], nine["reference"]["atol"]) == (1e-6, 1e-9)
    assert (ten["reference"]["rtol"], ten["reference"]["atol"]) == (FLOAT64_RTOL, 0.0)
    assert {key: value for key, value in nine["reference"].items() if key not in ("rtol", "atol")} == \
        {key: value for key, value in ten["reference"].items() if key not in ("rtol", "atol")}


def test_the_form_nine_folder_is_the_folder_its_commit_wrote_byte_for_byte():
    found = {path.relative_to(FREE9).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(FREE9.rglob("*")) if path.is_file()}
    assert found == PINNED


def test_a_form_nine_folder_loads_with_the_tolerance_it_was_written_with_and_verifies_element_by_element(
        tmp_path: Path):
    folder = Path(shutil.copytree(FREE9, tmp_path / "free_ball"))
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[1:3] == [
        f"saved form: mime.document/9, read as version {CURRENT}", "input hash: reproduced exactly"]
    reference = read_folder(folder).lock["reference"]
    assert (reference["rtol"], reference["atol"]) == (1e-6, 1e-9)
    report = loaded.build().verify()
    print(f"\n{report.text()}")
    assert report.reproduced and not report.scaled and (report.rtol, report.atol) == (1e-6, 1e-9)


def planted(form, change: float) -> dict[str, dict[str, np.ndarray]]:
    """The folder's reference outputs as a rerun, every element moved by ``change`` of itself."""
    held: dict[str, dict[str, np.ndarray]] = {}
    for key, value in form.reference.items():
        if key.startswith("history/") and value.dtype.kind == "f":
            node, name = key.removeprefix("history/").split("/")
            held.setdefault(node, {})[name] = value * (1.0 + change)
    return held


def test_a_form_nine_folder_is_judged_element_by_element_and_the_same_rerun_as_form_ten_on_its_source_scale(
        monkeypatch: pytest.MonkeyPatch):
    """1e-7 of each element is within the ninth form's rtol of 1e-6 element by element, and beyond the tenth form's 2e-14
    of each field's own largest magnitude: the folder is judged by its own form's rule."""
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: ["usable_cpus: built under 4"])
    form = read_folder(FREE9)
    rerun = planted(form, 1e-7)
    kept = verify_history(form, rerun)
    assert kept.reproduced and not kept.scaled and kept.tolerance == "within rtol 1e-06, atol 1e-09", kept.text()
    ten = replace(form, version=10, lock={**form.lock, "reference": {**form.lock["reference"], "rtol": FLOAT64_RTOL,
                                                                      "atol": 0.0}})
    restated = verify_history(ten, rerun)
    print(f"\n{restated.text()}")
    assert restated.scaled and not restated.reproduced
    assert {line.split(": ")[0] for line in restated.moved} == {"ball.pose", "ball.twist"}


def test_a_folder_keeps_the_tolerance_it_was_written_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mime.testing.toys import full_toy

    lock = full_toy().save(tmp_path / "toy")
    written = lock["reference"]["rtol"]
    monkeypatch.setattr(_scales, "FLOAT32_RTOL", 0.5)
    monkeypatch.setattr(_scales, "FLOAT64_RTOL", 0.5)
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: ["usable_cpus: built under 4"])
    form = read_folder(tmp_path / "toy")
    report = verify_history(form, planted(form, 0.0))
    assert report.rtol == written != 0.5 and report.reproduced


def test_from_form_ten_the_derived_graphs_numbers_are_judged_on_their_own_largest_magnitude():
    """A list of numbers against its largest: rounding of a zero entry is within; element by element with no absolute part
    it would not be. And a number against its own magnitude, as before."""
    out: list[str] = []
    _folder._numbers_beyond({"a": [1.0, 0.0]}, {"a": [1.0, 1e-20]}, FLOAT64_RTOL, 0.0, "physics", out, scaled=True)
    assert out == []
    _folder._numbers_beyond({"a": [1.0, 0.0]}, {"a": [1.0, 1e-20]}, FLOAT64_RTOL, 0.0, "physics", out, scaled=False)
    assert out == ["physics.a[1]: 0.0 -> 1e-20"]
    out.clear()
    _folder._numbers_beyond({"a": [1.0, 0.0], "b": 2.0}, {"a": [1.0 + 1e-10, 0.0], "b": 2.0 * (1 + 1e-10)},
                            FLOAT64_RTOL, 0.0, "physics", out, scaled=True)
    assert out == ["physics.a: max |difference| 1.000e-10", "physics.b: 2.0 -> 2.0000000002"]


@contextmanager
def reader_of(newest: int) -> Iterator[None]:
    """The loader reading the forms up to ``newest``: version 9's, this MIME before the step."""
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


@pytest.fixture(scope="module")
def current_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from mime.testing.toys import full_toy

    root = tmp_path_factory.mktemp("form10")
    full_toy().save(root / "full_toy")
    return root / "full_toy"


def test_version_nines_reader_refuses_a_folder_of_version_ten_as_newer_each_file_once(current_folder: Path):
    assert codes_of(current_folder) == [], "this MIME reads it"
    with reader_of(FORM9_READS):
        assert codes_of(current_folder) == [NEWER, NEWER]


def test_control_the_form_left_unstepped_is_read_by_version_nines_reader_with_no_newer_form(current_folder: Path,
                                                                                            tmp_path: Path):
    unstepped = restated(current_folder, tmp_path / "unstepped", FORM9_READS)
    with reader_of(FORM9_READS):
        assert NEWER not in codes_of(unstepped)
