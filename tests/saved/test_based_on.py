"""``Experiment.based_on``: a saved experiment edited as a new one, which records the experiment it was made from.

The new experiment holds the parent's declarations under its own name, with no lock; a declaration taken from the
parent, made again and changed, replaces it in a script as in a notebook; saving it writes a new folder with its own
lock, whose document records the parent as ``based_on`` (its name, its input hash, the MIME version that saved it, and
its DOI where its lock names one); and the parent's folder is never written. An experiment based on one that was itself
based on another records its immediate parent, and the chain is followed through each parent's own document.

**The controls.** The parent's folder is held by every file's bytes, size and modification time and every folder's
modification time, before and after (``tests/folder_state.py``): a write planted inside ``based_on`` (a byte appended to the parent's lock, a file
added beside it, the lock rewritten with its own bytes) fails that check on every run. The recorded input hash is held
to the parent's lock and away from the child's, so recording the child's own hash fails; the child's lock is held away
from the parent's (its hashes, no ``publication``), so reusing the parent's lock fails. The form's step is held in
``test_form7.py``."""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import jax
import pytest
import yaml
from folder_state import untouched
from script_a0 import a0

from mime.build import ArgumentRefused, BuildRefused
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, card_text, read_folder
from mime.saved import _based_on
from mime.saved._card import card_facts, card_of
from mime.testing.toys import full_toy
from mime.testing.toys._entries import ToyBody, ToySpring
from mime.things import Body, State
from mime.units import A_m2, N, kg, m, m_per_s

#: The runs of a control that must fail on every one.
RUNS = 20
#: The record a published parent's lock names (outside its input hash), as ``mime push`` writes it.
PUBLICATION = {"service": "zenodo", "record": "4242", "doi": "10.5281/zenodo.4242", "concept": "4241", "version": 1}


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def document_of(folder: Path) -> dict:
    return yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("parent") / "full_toy"
    full_toy().save(folder)
    return folder


def stiffer(experiment: Experiment) -> None:
    """The edit the toy tests make: the spring the parent declared, stiffer."""
    experiment.attach(ToySpring(k=N(3.4) / m(1.0)), body="probe")


def based_on_edited_saved(parent: Path, out: Path, name: str = "stiffer_toy") -> Path:
    """The parent, based on, edited, built and saved into ``out``."""
    child = Experiment.based_on(parent, name=name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stiffer(child)
    child.build().save(out)
    return out


# --- what it returns ---------------------------------------------------------------------------------------------------


def test_the_new_experiment_holds_the_parents_declarations_under_its_name_and_no_lock(parent: Path):
    child = Experiment.based_on(parent, name="copy")
    held = child.document()
    saved = document_of(parent)
    assert held["experiment"] == child.name == "copy" and saved["experiment"] == "full_toy"
    assert {k: v for k, v in held.items() if k not in ("experiment", "based_on")} == \
        {k: v for k, v in saved.items() if k != "experiment"}, "every declaration is the parent's"
    assert child.load_report is None and child._saved is None
    built = child.build()
    assert built.load_report is None, "no lock is checked: the experiment is a new one"
    with pytest.raises(ArgumentRefused) as refused:
        built.verify()
    assert refused.value.codes() == ["verify.nothing-saved"]


def test_saving_writes_a_new_folder_whose_document_records_the_parent_and_whose_lock_is_its_own(parent: Path,
                                                                                               tmp_path: Path):
    child = based_on_edited_saved(parent, tmp_path / "child")
    parent_lock, child_lock = lock_of(parent), lock_of(child)
    assert document_of(child)["based_on"] == {"experiment": "full_toy", "input_hash": parent_lock["hashes"]["input"],
                                              "mime": parent_lock["environment"]["mime"]}
    assert child_lock["hashes"]["input"] != parent_lock["hashes"]["input"], "the child's own hash is another"
    assert child_lock["hashes"]["derived"] != parent_lock["hashes"]["derived"], "the edit is in the child's graph"
    assert "publication" not in child_lock
    loaded = Experiment.load(child)
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()
    assert loaded.document()["based_on"] == document_of(child)["based_on"]


def test_the_record_names_the_parents_doi_where_its_lock_names_the_record_it_was_published_as(parent: Path,
                                                                                               tmp_path: Path):
    published = Path(shutil.copytree(parent, tmp_path / "full_toy"))
    lock = lock_of(published)
    lock["publication"] = PUBLICATION
    (published / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    child = based_on_edited_saved(published, tmp_path / "child")
    assert document_of(child)["based_on"] == {"experiment": "full_toy", "input_hash": lock["hashes"]["input"],
                                              "mime": lock["environment"]["mime"], "doi": PUBLICATION["doi"]}
    assert "publication" not in lock_of(child), "the child is not the parent's record: push makes it one of its own"


def test_the_record_edited_by_hand_after_its_lock_was_written_is_refused(parent: Path, tmp_path: Path):
    child = based_on_edited_saved(parent, tmp_path / "child")
    path = child / DOCUMENT_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("  experiment: full_toy\n", "  experiment: another\n"),
                    encoding="utf-8")
    assert document_of(child)["based_on"]["experiment"] == "another"
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(child)
    assert refused.value.codes() == ["load.input-hash"]


# --- editing -----------------------------------------------------------------------------------------------------------


def test_a_declaration_taken_from_the_parent_made_again_replaces_it_in_a_script_with_a_note(parent: Path):
    child = Experiment.based_on(parent, name="edited")
    with pytest.warns(UserWarning) as caught:
        stiffer(child)
        child.add_body(Body("probe", dynamics=ToyBody(mass=kg(2.6)), initial=State(
            position=m((0.0, 0.0, 0.0)), velocity=m_per_s((0.0, 0.0, 0.0)))))
    notes = [str(each.message) for each in caught]
    assert all("it replaces the one this experiment took from 'full_toy', which it is based on" in note
               for note in notes) and len(notes) == 2
    assert "toy.spring:probe" in notes[1], "the body's note names the laws that follow it to the new one"
    assert child.plan().replaced == tuple(notes), "recorded, as a notebook's replacements are"
    settings = {entry["id"]: entry.get("settings", {}) for entry in child.document()["interactions"]}
    assert settings["toy.spring:probe"]["k"]["value"] == 3.4
    assert child.document()["things"]["probe"]["dynamics"]["motion"]["settings"]["mass"]["value"] == 2.6
    child.build()


def test_a_declaration_made_after_based_on_follows_the_scripts_rule(parent: Path):
    child = Experiment.based_on(parent, name="edited")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stiffer(child)
        child.attach(ToySpring(k=N(5.1) / m(1.0)), body="probe")
    with pytest.raises(BuildRefused) as refused:
        child.build()
    assert "id.duplicate" in refused.value.codes(), "the script's own declaration, changed, is refused as in any script"


def test_in_a_notebook_both_a_taken_declaration_and_one_made_after_are_replaced(parent: Path,
                                                                                monkeypatch: pytest.MonkeyPatch):
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    child = Experiment.based_on(parent, name="edited")
    with pytest.warns(UserWarning) as caught:
        stiffer(child)
        child.attach(ToySpring(k=N(5.1) / m(1.0)), body="probe")
    first, second = (str(each.message) for each in caught)
    assert "took from 'full_toy'" in first and "in a notebook or IPython the last one run wins" in second
    settings = {entry["id"]: entry.get("settings", {}) for entry in child.document()["interactions"]}
    assert settings["toy.spring:probe"]["k"]["value"] == 5.1
    child.build()


def test_a_loaded_experiment_edited_in_a_notebook_is_refused_by_its_lock_pointing_to_based_on(
        parent: Path, monkeypatch: pytest.MonkeyPatch):
    """The lock's refusal of a loaded experiment that a notebook's cell changed names the call that keeps the change,
    and that call does."""
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    loaded = Experiment.load(parent)
    with pytest.warns(UserWarning):
        stiffer(loaded)
    with pytest.raises(LoadRefused) as refused:
        loaded.build()
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.input-hash" and 'Experiment.based_on("<folder>", name="<a new name>")' in refusal.action
    child = Experiment.based_on(parent, name="stiffer_toy")
    with pytest.warns(UserWarning):
        stiffer(child)
    assert child.build().load_report is None


# --- the parent's folder -------------------------------------------------------------------------------------------------


def the_whole_use(parent: Path, out: Path) -> None:
    """Everything an experiment based on ``parent`` does: made, edited, built, saved, loaded, carded."""
    child = based_on_edited_saved(parent, out / "child")
    Experiment.load(child).build()
    card_text(child)


def test_the_parents_folder_is_never_written(parent: Path, tmp_path: Path):
    with untouched(parent):
        the_whole_use(parent, tmp_path)


@contextmanager
def writing_into_the_parent(how: Callable[[Path], None]) -> Iterator[None]:
    """``based_on`` planted to write into the folder it is based on, as it reads it."""
    real = _based_on.based_on_of

    def writing(form):  # type: ignore[no-untyped-def]
        how(form.folder)
        return real(form)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_based_on, "based_on_of", writing)
        yield


PLANTED_WRITES: dict[str, Callable[[Path], None]] = {
    "a byte appended to its lock": lambda folder: (folder / LOCK_FILE).write_text(
        (folder / LOCK_FILE).read_text(encoding="utf-8") + " ", encoding="utf-8"),
    "a file added beside it": lambda folder: (folder / "notes.txt").write_text("mine", encoding="utf-8"),
    "its lock rewritten with its own bytes": lambda folder: (folder / LOCK_FILE).write_bytes(
        (folder / LOCK_FILE).read_bytes()),
}


@pytest.mark.parametrize("write", PLANTED_WRITES, ids=list(PLANTED_WRITES))
def test_control_a_write_planted_into_the_parent_is_caught_every_run(parent: Path, tmp_path: Path, write: str):
    caught = 0
    for run in range(RUNS):
        copy = Path(shutil.copytree(parent, tmp_path / f"run{run}" / "full_toy"))
        try:
            with writing_into_the_parent(PLANTED_WRITES[write]), untouched(copy):
                Experiment.based_on(copy, name="child")
        except AssertionError:
            caught += 1
    assert caught == RUNS


def test_a_parent_that_does_not_load_is_refused_with_the_loads_reasons_and_left_as_it_was(parent: Path,
                                                                                          tmp_path: Path):
    edited = Path(shutil.copytree(parent, tmp_path / "full_toy"))
    path = edited / DOCUMENT_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("seed: 0", "seed: 1"), encoding="utf-8")
    with untouched(edited), pytest.raises(LoadRefused) as refused:
        Experiment.based_on(edited, name="child")
    assert refused.value.codes() == ["load.input-hash"]


# --- a chain -----------------------------------------------------------------------------------------------------------


def chain_of(folder: Path, folders: dict[str, Path]) -> list[str]:
    """The names an experiment's ``based_on`` leads through, each found among ``folders`` by its name and held to
    its input hash, followed through each parent's own document."""
    names = [document_of(folder)["experiment"]]
    held = document_of(folder).get("based_on")
    while held is not None:
        found = folders[held["experiment"]]
        assert lock_of(found)["hashes"]["input"] == held["input_hash"]
        names.append(held["experiment"])
        held = document_of(found).get("based_on")
    return names


def test_an_experiment_based_on_one_that_was_itself_based_on_another_records_its_immediate_parent(
        parent: Path, tmp_path: Path):
    second = based_on_edited_saved(parent, tmp_path / "second", name="second")
    third = Experiment.based_on(second, name="third")
    third.save(tmp_path / "third")
    recorded = document_of(tmp_path / "third")["based_on"]
    assert recorded["experiment"] == "second" and recorded["input_hash"] == lock_of(second)["hashes"]["input"]
    assert recorded["input_hash"] != lock_of(parent)["hashes"]["input"], "the immediate parent, not the first"
    assert chain_of(tmp_path / "third", {"second": second, "full_toy": parent}) == ["third", "second", "full_toy"]
    assert Experiment.load(tmp_path / "third").load_report is not None


# --- the card ------------------------------------------------------------------------------------------------------------


def test_the_card_names_the_parent_by_its_input_hash_or_its_doi(parent: Path, tmp_path: Path):
    child = based_on_edited_saved(parent, tmp_path / "child")
    hash_line = f"- **Based on full_toy (input hash {lock_of(parent)['hashes']['input']})**"
    assert hash_line in card_text(child).splitlines()
    published = Path(shutil.copytree(parent, tmp_path / "published" / "full_toy"))
    lock = lock_of(published)
    lock["publication"] = PUBLICATION
    (published / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    of_published = based_on_edited_saved(published, tmp_path / "of_published")
    assert f"- **Based on full_toy (DOI {PUBLICATION['doi']})**" in card_text(of_published).splitlines()
    assert "Based on" not in card_text(parent), "an experiment made from none names none"


def test_push_s_description_is_the_card_without_the_line_until_push_states_the_relation(parent: Path,
                                                                                         tmp_path: Path):
    """``mime push`` writes ``card_of(facts, publication)`` as a record's description: what it sends is unchanged by
    ``based_on`` until the relation is rehearsed on the sandbox."""
    child = based_on_edited_saved(parent, tmp_path / "child")
    facts = card_facts(child)
    pushed, shown = card_of(facts, None), card_of(facts, None, based_on=True)
    assert "Based on" not in pushed and "Based on full_toy" in shown
    assert [line for line in shown.splitlines() if "Based on" not in line] == pushed.splitlines()


# --- the round trip on A0 ------------------------------------------------------------------------------------------------


@pytest.fixture
def float32() -> Iterator[None]:
    """A0 in the precision its tutorial runs in, not the module's float64: nothing here claims bits across builds,
    and float64 doubles the round trip's compiles."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


def test_a0_based_on_with_a_stronger_field_saves_loads_and_rebuilds_against_its_own_lock(tmp_path: Path,
                                                                                         float32: None):
    """Tutorial A0's rig (the AR4 arm carrying the rotor magnet, closed by its controller), saved; an experiment
    based on it with the rotor magnet's moment, the field's strength, raised; built, saved, loaded and rebuilt
    against its own lock, which names A0 as its parent."""
    saved = tmp_path / "script_a0"
    a0().save(saved)
    with untouched(saved):
        child = Experiment.based_on(saved, name="a0_stronger_field")
        with pytest.warns(UserWarning, match="took from 'script_a0'"):
            child.declare("rpm.magnetic_moment", A_m2(25.0), reason="a stronger rotor magnet, 25 A m^2")
        child.build().save(tmp_path / "stronger")
        loaded = Experiment.load(tmp_path / "stronger")
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()
    rebuilt = read_folder(tmp_path / "stronger")
    parent_lock = lock_of(saved)
    assert rebuilt.document["based_on"] == {"experiment": "script_a0", "input_hash": parent_lock["hashes"]["input"],
                                            "mime": parent_lock["environment"]["mime"]}
    assert rebuilt.document["declared"]["rpm.magnetic_moment"]["value"] == 25.0
    assert document_of(saved)["declared"]["rpm.magnetic_moment"]["value"] == pytest.approx(18.89)
    assert rebuilt.lock["hashes"]["derived"] != parent_lock["hashes"]["derived"]
