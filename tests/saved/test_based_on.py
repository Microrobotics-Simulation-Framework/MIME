"""``Experiment.based_on``: a saved experiment edited as a new one, which records the experiment it was made from.

The new experiment holds the parent's declarations under its own name, with no lock; a declaration taken from the
parent, made again and changed, replaces it in a script as in a notebook; saving it writes a new folder with its own
lock, whose document records the parent as ``based_on`` (its name, its input hash, the MIME version that saved it, and
its DOI where its lock names one); and the parent's folder is never written. An experiment based on one that was itself
based on another records its immediate parent, and the chain is followed through each parent's own document. A DOI
that is not a DOI's shape is not recorded, and the warning says why.

**A parent saved by the released 0.3.0.dev1** (saved form 6, the folder every user of that release holds):
``form6/timed_toy`` was written by the ``mime-engine==0.3.0.dev1`` wheel from PyPI, in its own environment, in float64
with one BLAS thread as the saved tests run, from the full toy with its run's length declared::

    jax.config.update("jax_enable_x64", True)
    experiment = declare_in_order(lambda: Experiment("timed_toy", scene=Scene.new(up=Z, units="m"),
                                                     gravity=standard_gravity(), dt=DT, duration=s(0.04)),
                                  full_declarations())  # mime.testing.toys._experiments, as 0.3.0.dev1 ships it
    experiment.build().save(folder)

Its files are pinned by their sha256. Like every folder of an earlier form here, it is re-stamped at a release's version
bump (``restamp_version.py``), which moves the pins of its document and lock and nothing else. A child of it saves as
the current form, loads against its own lock, rebuilds and reproduces its reference steps: a child whose lock was hashed in the
parent's form is refused at that load. Which MIME each string names is held apart from the MIME running: with the
running MIME a release (patched in, never in the repository) and the parent's lock edited to name an earlier one
(outside the input hash), the record names the parent's and the child's lock the running one, three strings that all
differ (the installed distribution's is the third).

**The controls.** The parent's folder is held by every file's bytes, size and modification time and every folder's
modification time, before and after (``tests/folder_state.py``): a write planted inside ``based_on`` (a byte appended to the parent's lock, a file
added beside it, the lock rewritten with its own bytes) fails that check on every run. The recorded input hash is held
to the parent's lock and away from the child's, so recording the child's own hash fails; the child's lock is held away
from the parent's (its hashes, no ``publication``), so reusing the parent's lock fails. The form's step is held in
``test_form7.py``."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import jax
import pytest
import yaml
from packaging.version import Version
from folder_state import untouched
from script_a0 import a0

import mime
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
    assert refusal.code == "load.input-hash" and "edited after it was loaded" in refusal.message
    assert f'Experiment.based_on({str(parent)!r}, name="<a new name>")' in refusal.action
    child = Experiment.based_on(parent, name="stiffer_toy")
    with pytest.warns(UserWarning):
        stiffer(child)
    assert child.build().load_report is None


def test_remove_names_a_declaration_taken_from_the_parent_by_where_it_came_from(parent: Path):
    """``exp.remove`` refuses while something names what it removes, citing where each was declared: a declaration
    taken from the parent was declared in it, and the refusal says so, by the parent's name."""
    child = Experiment.based_on(parent, name="edited")
    with pytest.raises(ArgumentRefused) as refused:
        child.remove("probe")
    (refusal,) = refused.value.refusals
    assert refusal.code == "remove.named"
    assert "toy.spring:probe (taken from 'full_toy', the saved experiment this one is based on)" in refusal.message
    child.remove("toy.spring:probe")
    assert "toy.spring:probe" not in {entry["id"] for entry in child.document()["interactions"]}


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

    def writing(form, *rest):  # type: ignore[no-untyped-def]
        how(form.folder)
        return real(form, *rest)

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


# --- a parent saved by the released 0.3.0.dev1 (form 6) -----------------------------------------------------------------

RELEASED = Path(__file__).parent / "form6" / "timed_toy"
#: Its files as the 0.3.0.dev1 wheel wrote them (the document's and the lock's move only with ``restamp_version.py``).
RELEASED_SHA256 = {
    "experiment.mime.yaml": "66a4cc6a8afbe4ee98c729c117c4a7df7d13e565ffb33dd070f60c2873e0c03a",
    "mime.lock": "a1cda39cf969f2cc2d2bc826339e6c132317258640dd6afe0ff8e67264082004",
    "reference.npz": "84743bcaeaa0948ccd58deaab4ca4461029a6f776959aec5a319348ceb570159",
    "scene/scene.usda": "78a93ae11ce3de31021b76e56c8f7d849d2ee1c1a292bdf78f83a7b136132039",
}
#: The run's length the released parent declares, which its child keeps.
RELEASED_DURATION = {"value": 0.04, "unit": "s"}


def test_the_released_parent_is_the_folder_the_wheel_wrote_byte_for_byte():
    found = {path.relative_to(RELEASED).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(RELEASED.rglob("*")) if path.is_file()}
    assert found == RELEASED_SHA256
    document, lock = document_of(RELEASED), lock_of(RELEASED)
    assert (document["schema"], lock["schema"]) == ("mime.document/6", "mime.lock/6")
    assert document["run"]["duration"] == RELEASED_DURATION and lock["reference"]["steps"] > 0


def test_a_child_of_a_folder_the_released_wheel_saved_saves_in_the_current_form_loads_rebuilds_and_reproduces(
        tmp_path: Path):
    parent = Path(shutil.copytree(RELEASED, tmp_path / "timed_toy"))
    loaded = Experiment.load(parent)
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[1:3] == [
        "saved form: mime.document/6, read as version 10", "input hash: reproduced exactly"]
    assert loaded.build().verify().reproduced, "the parent, its run's length declared in form 6, loads and verifies"
    with untouched(parent):
        child = based_on_edited_saved(parent, tmp_path / "child", name="stiffer_timed_toy")
    parent_lock, document = lock_of(parent), document_of(child)
    assert (document["schema"], lock_of(child)["schema"]) == ("mime.document/11", "mime.lock/11")
    assert document["based_on"] == {"experiment": "timed_toy", "input_hash": parent_lock["hashes"]["input"],
                                    "mime": parent_lock["environment"]["mime"]}
    assert document["run"]["duration"] == RELEASED_DURATION, "the run's length is the parent's"
    reloaded = Experiment.load(child)
    assert reloaded.load_report is not None and "input hash: reproduced exactly" in reloaded.load_report.text()
    assert reloaded.build().verify().reproduced


def the_next_development_release(version: str) -> str:
    """The development release after ``version``: ``0.3.0.dev1`` -> ``0.3.0.dev2``."""
    held = Version(version)
    assert held.dev is not None and held.local is None, version
    return f"{held.base_version}.dev{held.dev + 1}"


def test_a_parent_another_development_release_saved_is_refused_by_based_on_as_its_load_refuses_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The released parent under the next development release (``__version__`` patched for the test): refused by
    name, ``load.other-release`` with both versions (design #232), before anything is made of it, and left as it was.
    The control is the test above: the same folder under the MIME that saved it is based on."""
    parent = Path(shutil.copytree(RELEASED, tmp_path / "timed_toy"))
    saved_by = lock_of(parent)["environment"]["mime"]
    later = the_next_development_release(saved_by)
    monkeypatch.setattr(mime, "__version__", later)
    with untouched(parent), pytest.raises(LoadRefused) as refused:
        Experiment.based_on(parent, name="child")
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.other-release"
    assert f"saved by MIME {saved_by}, and this is MIME {later}" in refusal.message
    assert not (tmp_path / "child").exists()


#: The running MIME (a release, patched in for the test) and the MIME a parent's lock is edited to name (an earlier
#: one): neither is the installed distribution's development release, and neither is a development release, which
#: makes no compatibility promise and whose folders another refuses.
RUNNING, PARENT_SAVED_BY = "0.3.0", "0.2.1"


def test_the_record_names_the_mime_of_the_parents_lock_and_the_childs_lock_names_the_one_running(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mime, "__version__", RUNNING)
    parent = tmp_path / "full_toy"
    full_toy().save(parent)
    lock = lock_of(parent)
    lock["environment"]["mime"] = PARENT_SAVED_BY
    (parent / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    assert len({RUNNING, PARENT_SAVED_BY, importlib.metadata.version("mime-engine")}) == 3
    child = based_on_edited_saved(parent, tmp_path / "child")
    assert document_of(child)["based_on"]["mime"] == PARENT_SAVED_BY, "the parent's, not the running MIME's"
    assert lock_of(child)["environment"]["mime"] == RUNNING, "the running MIME's, not the parent's"
    assert Experiment.load(child).load_report is not None


# --- a DOI that is not a DOI's shape ---------------------------------------------------------------------------------------


def published_as(parent: Path, root: Path, doi: str) -> Path:
    """The parent with its lock naming the record ``doi`` (outside the input hash), as a lock edited by hand may."""
    folder = Path(shutil.copytree(parent, root / "full_toy"))
    lock = lock_of(folder)
    lock["publication"] = {**PUBLICATION, "doi": doi}
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return folder


#: What a lock's ``publication.doi`` may hold that is no DOI, with why it is not recorded.
NOT_DOIS = {
    "a newline and a YAML line": ("10.1/x\n- evil: 1", "control or format character"),
    "a terminal's escape": ("10.1234/\x1b[31mred", "control or format character"),
    "a right-to-left override": ("10.1234/abc\u202edef", "control or format character"),
    "a Markdown link": ("x**) [a](http://e.example) **(", "not a DOI's shape"),
    "5000 characters": ("1" * 5000, "5000 characters long, and a DOI recorded is at most 256"),
    "nothing": ("", "not a DOI's shape"),
    "a space": ("10.5281/zenodo 4242", "not a DOI's shape"),
    "a registrant of one digit": ("10.1/x", "not a DOI's shape"),
    "one character over the limit": ("10.1234/" + "a" * 249, "257 characters long"),
}


@pytest.mark.parametrize("doi, why", NOT_DOIS.values(), ids=list(NOT_DOIS))
def test_a_doi_that_is_not_a_dois_shape_is_not_recorded_and_the_warning_says_why(parent: Path, tmp_path: Path,
                                                                                 doi: str, why: str):
    with pytest.warns(UserWarning, match="is not recorded") as caught:
        child = Experiment.based_on(published_as(parent, tmp_path, doi), name="child")
    (note,) = [str(each.message) for each in caught if "is not recorded" in str(each.message)]
    assert why in note and "\n" not in note and "\x1b" not in note and len(note) < 400
    assert "doi" not in child.document()["based_on"]
    child.save(tmp_path / "child")
    assert f"- **Based on full_toy (input hash {lock_of(parent)['hashes']['input']})**" in \
        card_text(tmp_path / "child").splitlines()


def test_the_control_a_doi_of_the_longest_length_recorded_is_recorded(parent: Path, tmp_path: Path):
    longest = "10.1234/" + "a" * 248
    assert len(longest) == _based_on.DOI_LIMIT
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        child = Experiment.based_on(published_as(parent, tmp_path, longest), name="child")
    assert child.document()["based_on"]["doi"] == longest


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
