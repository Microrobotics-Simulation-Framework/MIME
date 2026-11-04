"""What a load refuses, one hand edit each, for the refusals the specification states (``docs/specification/
saved_form.md``) that no other test raised: a document or a lock that cannot be read, a lock that holds a key twice,
reference outputs changed or holding a pickled object, a quantity in a unit MIME does not write, a setting the entry
does not have or one it needs left out, a material the document does not hold, an id that is not the one its bindings
give, and dynamics a thing's kind does not take.

A folder's files are edited in a copy of the committed current-form folder and read by ``read_folder``, no build; a
document's are edited in script A's document (or a controlled one's, for the controller's rate) and read by
``Experiment.from_document``, which rebuilds the declarations and builds nothing. Each case asserts the code it names
and no other, beside the unedited input, which passes. The lock's one reader (``mime._lock_json``) is held to be the
only place a lock is parsed.
"""

import ast
import json
import pickle
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from chain_cases import tracked_rpm
from script_a import script_a

from mime.assets import folder_assets
from mime.build import BuildRefused
from mime.env._check import lock_x64
from mime.env._modal_script import _read_lock  # pyright: ignore[reportPrivateUsage]
from mime.experiment import Experiment
from mime.push._stage import publication_of
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, dump_document, read_document
from mime.saved._folder import read_folder
from mime.saved._terms import _lock_of  # pyright: ignore[reportPrivateUsage]

SAVED = Path(__file__).parent / "stage_writers" / "chain_screw"
SOURCE = Path(__file__).resolve().parents[2] / "src" / "mime"
REPEATED = ('"schema": "mime.lock/4"', '"schema": "mime.lock/4",\n "schema": "mime.lock/4"')


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return Path(shutil.copytree(SAVED, tmp_path / "saved"))


def codes(read: Callable[[], object]) -> list[str]:
    with pytest.raises((LoadRefused, BuildRefused)) as refused:
        read()
    return refused.value.codes()


def edited(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_the_unedited_folder_reads(folder: Path):
    """The baseline the folder's controls discriminate against."""
    assert read_folder(folder).lock["schema"] == "mime.lock/5", "a version 4 lock, read under the current form"
    assert lock_x64(folder) in (True, False) and _read_lock(folder)["schema"] == "mime.lock/4"
    assert folder_assets(folder) == () and _lock_of(folder) != {}


def test_a_document_that_cannot_be_read_is_refused_naming_it(folder: Path):
    (folder / DOCUMENT_FILE).unlink()
    with pytest.raises(LoadRefused) as refused:
        read_folder(folder)
    assert refused.value.codes() == ["load.missing-file"]
    assert str(folder / DOCUMENT_FILE) in refused.value.refusals[0].message


def test_a_lock_that_is_not_json_is_refused(folder: Path):
    (folder / LOCK_FILE).write_text('{"schema": "mime.lock/4",', encoding="utf-8")
    assert codes(lambda: read_folder(folder)) == ["load.lock"]


def test_a_lock_holding_a_key_twice_is_refused_by_the_load_and_by_every_reader_of_the_lock(folder: Path):
    """Design #221 (2): the repeat is the lock's integrity failing, so the load refuses it as ``load.lock``, and every
    other reader of a lock (an asset fetch, a run's precision, a Modal script) refuses it by its own code. The control:
    Python's ``json`` reads the same bytes without a word, keeping the last value."""
    edited(folder / LOCK_FILE, *REPEATED)
    text = (folder / LOCK_FILE).read_text(encoding="utf-8")
    assert json.loads(text)["schema"] == "mime.lock/4", "the control: json alone keeps the last value silently"
    with pytest.raises(LoadRefused) as refused:
        read_folder(folder)
    assert refused.value.codes() == ["load.lock"]
    assert "the key 'schema' appears twice in one mapping" in refused.value.refusals[0].message
    assert codes(lambda: folder_assets(folder)) == ["asset.lock"]
    assert codes(lambda: lock_x64(folder)) == ["env.lock"]
    assert codes(lambda: _read_lock(folder)) == ["env.lock"]
    assert _lock_of(folder) == {} and publication_of(text.encode("utf-8")) is None


def lock_parsed_with_json(source: str, path: str) -> list[str]:
    """Each place in ``source`` that parses a lock with ``json`` itself (a ``json.load`` or ``json.loads`` whose
    argument's text names a lock), not through the one reader. A lock read into a variable named otherwise is not
    seen; the readers today name it."""
    found = []
    for call in ast.walk(ast.parse(source)):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in ("load", "loads")
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
                and any("lock" in (ast.get_source_segment(source, argument) or "").lower() for argument in call.args)):
            found.append(f"{path}:{call.lineno}")
    return found


def test_every_module_reads_the_lock_through_the_one_reader():
    found = [place for path in sorted(SOURCE.rglob("*.py")) if path.name != "_lock_json.py"
             for place in lock_parsed_with_json(path.read_text(encoding="utf-8"), str(path.relative_to(SOURCE)))]
    assert found == []
    planted = 'held = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))\nother = json.loads(text)\n'
    assert lock_parsed_with_json(planted, "planted.py") == ["planted.py:1"], "the control: a lock parsed by json"


def reference_of(folder: Path) -> Path:
    return folder / read_folder(folder).lock["reference"]["file"]


def test_reference_outputs_whose_arrays_changed_are_refused(folder: Path):
    path = reference_of(folder)
    with np.load(path, allow_pickle=False) as held:
        arrays = {key: np.asarray(held[key]) for key in held.files}
    first = sorted(arrays)[0]
    arrays[first] = arrays[first] + 1
    np.savez(path, allow_pickle=False, **arrays)
    assert codes(lambda: read_folder(folder)) == ["load.reference-changed"]


class Planted:
    """An object whose unpickling writes the file ``target``: code a pickle runs."""

    def __init__(self, target: str) -> None:
        self.target = target

    def __reduce__(self):
        return (Path.write_text, (Path(self.target), "ran"))


def test_reference_outputs_holding_a_pickled_object_are_refused_and_run_nothing(folder: Path, tmp_path: Path):
    """The reference outputs are read with pickling off: an array of objects is refused as ``load.reference``, and
    the code its pickle carries never runs. The control: read with pickling on, it runs."""
    target = tmp_path / "ran.txt"
    path = reference_of(folder)
    with zipfile.ZipFile(path, "a") as archive, archive.open("planted.npy", "w") as member:
        np.lib.format.write_array(member, np.array([Planted(str(target))], dtype=object), allow_pickle=True)
    assert codes(lambda: read_folder(folder)) == ["load.reference"]
    assert not target.exists()
    with np.load(path, allow_pickle=True) as held:
        held["planted"]
    assert target.read_text(encoding="utf-8") == "ran", "the control: a reader with pickling on runs the payload"
    assert pickle  # the payload is a pickle, written by numpy's own writer above


def document_refusals(text: str, tmp_path: Path) -> list[str]:
    """The codes reading the document ``text`` is refused with: its read, or the rebuild of its declarations."""
    (tmp_path / DOCUMENT_FILE).write_text(text, encoding="utf-8")
    document, refusals = read_document(tmp_path / DOCUMENT_FILE)
    if refusals:
        return [refusal.code for refusal in refusals]
    with pytest.raises(LoadRefused) as refused:
        Experiment.from_document(tmp_path / DOCUMENT_FILE)
    return refused.value.codes()


#: Each hand edit of script A's document, with the one code it is refused with: (old text, new text, code).
EDITS = {
    "a-setting-in-another-unit": ("            unit: T\n", "            unit: furlong\n", "load.units"),
    "an-initial-state-in-another-unit": ("        unit: m/s\n", "        unit: furlong\n", "load.units"),
    "a-condition-in-another-unit": ("conditions: {}\n", "conditions:\n  temperature:\n    value: 310.0\n"
                                    "    plus_minus: 1.0\n    unit: furlong\n", "load.units"),
    "the-step-in-another-unit": ("    unit: s\n  seed: 0\n", "    unit: furlong\n  seed: 0\n", "load.units"),
    "a-setting-the-entry-does-not-have": ("        axis: [1.0, 0.0, 0.0]\n",
                                          "        axis: [1.0, 0.0, 0.0]\n        colour: 1.0\n",
                                          "load.setting-unknown"),
    "a-setting-the-entry-needs-left-out": ("        inner_radius:\n          value: 0.003175\n          unit: m\n", "",
                                           "load.setting"),
    "a-group-no-package-registers": ("      group: lubrication\n", "      group: unregistered\n", "load.setting"),
    "a-material-the-document-does-not-hold": ("    fills: tube.lumen\n",
                                              "    fills: tube.lumen\n    material: unobtainium\n",
                                              "load.document-material"),
    "an-id-its-bindings-do-not-give": ("- id: gravity:umr\n", "- id: gravity:other\n", "load.id"),
    "dynamics-a-medium-does-not-take": ("    fills: tube.lumen\n",
                                        "    fills: tube.lumen\n    dynamics:\n      motion:\n"
                                        "        name: rigid_inertial\n        version: 0.3.0\n",
                                        "load.document-thing"),
    "a-document-that-is-not-yaml": ("experiment: script_a\n", "experiment: [script_a\n", "load.document-yaml"),
}


@pytest.fixture(scope="module")
def script_a_text() -> str:
    return dump_document(script_a().document())


def test_the_unedited_document_rebuilds(script_a_text: str, tmp_path: Path):
    """The baseline the document's controls discriminate against."""
    (tmp_path / DOCUMENT_FILE).write_text(script_a_text, encoding="utf-8")
    assert Experiment.from_document(tmp_path / DOCUMENT_FILE).document()["experiment"] == "script_a"


@pytest.mark.parametrize("edit", EDITS)
def test_a_document_edited_by_hand_is_refused_by_its_code(script_a_text: str, tmp_path: Path, edit: str):
    old, new, code = EDITS[edit]
    assert script_a_text.count(old) == 1, old
    assert document_refusals(script_a_text.replace(old, new), tmp_path) == [code]


def test_the_controllers_rate_in_another_unit_is_refused(tmp_path: Path):
    text = dump_document(tracked_rpm().document())
    old = "  rate:\n    value: 0.002\n    unit: s\n"
    assert text.count(old) == 1, text
    assert document_refusals(text.replace(old, old.replace("unit: s", "unit: furlong")), tmp_path) == ["load.units"]
