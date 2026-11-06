"""A saved folder written by another development release is refused by its own refusal, ``load.other-release``, which
names both versions and says how to save again (design #232); ``load.input-hash`` keeps its meaning, a document changed
after its lock was written, and is raised for it only when the versions agree. An experiment loaded and then edited
(in a notebook, a cell run after the load) is refused in words about the edit, not about a document.

The controls: the version check removed (the folder is then refused for the version in its input hash and the message
says the document was changed, or loads), and the checks in the other order (the hash first), each fail the assertion
the real check passes. They are run through the same assertion, so they cannot pass for a reason of their own."""

import builtins
import json

import pytest

import mime
from mime.build import RefusalClass
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, read_folder
from mime.saved import _folder as saved_folder
from mime.testing.toys import ToyBody, full_toy
from mime.testing.toys._experiments import PROBE_START
from mime.things import Body

EARLIER = "0.3.0.dev0"
CURRENT = mime.__version__


def _edit_lock(folder, change):
    lock = json.loads((folder / LOCK_FILE).read_text())
    change(lock)
    (folder / LOCK_FILE).write_text(json.dumps(lock))


def _record_saved_by(folder, version):
    """The lock of a real folder, with the MIME that saved it recorded as ``version``."""
    _edit_lock(folder, lambda lock: lock["environment"].update(mime=version))


def assert_refused_as_another_release(folder, saved_by):
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.other-release" and refusal.category is RefusalClass.USER, refusal
    assert f"saved by MIME {saved_by}, and this is MIME {mime.__version__}" in refusal.message
    assert "development releases do not load each other's folders" in refusal.message
    assert "changed" not in refusal.message
    assert f"built.save({str(folder)!r})" in refusal.action
    assert f"pip install mime-engine=={saved_by}" in refusal.action
    return refusal


@pytest.fixture
def saved(tmp_path):
    full_toy().save(tmp_path)
    return tmp_path


def test_the_control_a_folder_saved_by_this_release_loads(saved):
    assert Experiment.load(saved).load_report is not None


@pytest.mark.parametrize("version", [EARLIER, "0.3.0.dev99"])
def test_a_lock_recording_another_development_release_is_refused_by_its_own_refusal(saved, version):
    _record_saved_by(saved, version)
    assert_refused_as_another_release(saved, version)


def test_a_folder_saved_under_an_earlier_release_is_refused_by_its_own_refusal(tmp_path, monkeypatch):
    """The folder as the earlier release wrote it: the version in the document, the resolved record and the lock."""
    with monkeypatch.context() as earlier:
        earlier.setattr(mime, "__version__", EARLIER)
        full_toy().save(tmp_path)
    assert_refused_as_another_release(tmp_path, EARLIER)


def test_a_document_edited_under_one_version_is_still_refused_as_changed(saved):
    path = saved / DOCUMENT_FILE
    path.write_text(path.read_text().replace("  binds:\n    body: probe\nobservations",
                                             "  binds:\n    body: probe\n  settings:\n    k:\n"
                                             "      value: 2.5\n      unit: kg s^-2\nobservations"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(saved)
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.input-hash" and "the document was changed after its lock was written" in refusal.message


def test_only_a_development_release_that_differs_is_refused_this_way(saved):
    form = read_folder(saved)
    assert saved_folder.other_release_refusal(form) is None
    assert saved_folder.other_release_refusal(form, running="1.0.0") is not None  # the saved CURRENT is a dev release
    _record_saved_by(saved, "1.0.0")
    form = read_folder(saved)
    assert saved_folder.other_release_refusal(form, running="1.0.1") is None  # neither is: not this refusal's
    assert saved_folder.other_release_refusal(form, running="1.0.0") is None
    assert saved_folder.other_release_refusal(form, running="1.1.0.dev1") is not None


LOCAL = "0.3.0.dev2+g1234abc"
DIRTY = "0.3.0.dev2+g1234abc.d20261106"


@pytest.mark.parametrize("saved_by, running, refused", [
    (LOCAL, LOCAL, False), (DIRTY, DIRTY, False),  # a checkout's own folders
    (LOCAL, DIRTY, True), (DIRTY, LOCAL, True),  # the same commit, clean and dirty: the string is in the hash
    (LOCAL, "0.3.0.dev2", True), ("0.3.0.dev2", LOCAL, True),
    (LOCAL, "0.3.0.dev2+g9999999", True),  # another commit
    ("0.3.0.dev1+g1234abc", LOCAL, True), (LOCAL, "0.3.0.dev1+g1234abc", True),  # another release, same commit
    (LOCAL, "0.3.0", True), ("0.3.0", DIRTY, True)])
def test_a_version_with_a_local_part_is_judged_on_the_whole_string(saved, saved_by, running, refused):
    _record_saved_by(saved, saved_by)
    found = saved_folder.other_release_refusal(read_folder(saved), running=running)
    assert (found is not None) == refused
    if found is not None:
        assert f"saved by MIME {saved_by}, and this is MIME {running}" in found.message
        assert f"pip install mime-engine=={saved_by}" in found.action


@pytest.mark.parametrize("version", [LOCAL, DIRTY])
def test_a_checkout_with_a_local_version_loads_its_own_folders_and_refuses_the_release_it_was_cut_from(
        tmp_path, monkeypatch, version):
    own, released = tmp_path / "own", tmp_path / "released"
    with monkeypatch.context() as checkout:
        checkout.setattr(mime, "__version__", version)
        full_toy().save(own)
        assert Experiment.load(own).load_report is not None
        with monkeypatch.context() as earlier:
            earlier.setattr(mime, "__version__", "0.3.0.dev2")
            full_toy().save(released)
        assert_refused_as_another_release(released, "0.3.0.dev2")
    with monkeypatch.context() as release:  # and the release refuses the checkout's
        release.setattr(mime, "__version__", "0.3.0.dev2")
        assert_refused_as_another_release(own, version)


def test_a_folder_unpacked_from_a_published_record_says_so_and_names_its_page(saved):
    _edit_lock(saved, lambda lock: lock.update(publication={
        "service": "zenodo-sandbox", "record": "123", "doi": "10.5072/zenodo.123", "concept": "122", "version": 2}))
    _record_saved_by(saved, EARLIER)
    refusal = assert_refused_as_another_release(saved, EARLIER)
    assert "came from `mime pull`, the published record 10.5072/zenodo.123 (version 2)" in refusal.message
    assert "https://sandbox.zenodo.org/records/123" in refusal.action
    assert f"ask its author for a record saved by MIME {CURRENT}," in refusal.action
    assert f"load it as it is under the MIME that saved it, `pip install mime-engine=={EARLIER}`" in refusal.action
    _edit_lock(saved, lambda lock: lock.pop("publication"))
    assert "mime pull" not in assert_refused_as_another_release(saved, EARLIER).message


def _version_check_removed(monkeypatch):
    monkeypatch.setattr(saved_folder, "other_release_refusal", lambda form, running=None: None)


def _hash_first(monkeypatch):
    """The checks in the other order: what the hash check refuses is raised before the version is judged."""
    import mime.experiment._experiment as experiment_module

    real_check = experiment_module.check_load
    real_refusal = saved_folder.other_release_refusal
    _version_check_removed(monkeypatch)

    def hash_first(form, record, stages=None, **kwargs):
        report = real_check(form, record, stages, **kwargs)  # raises what the hash refuses, before the version is judged
        refusal = real_refusal(form)
        if refusal is not None:
            raise LoadRefused([refusal])
        return report

    monkeypatch.setattr(experiment_module, "check_load", hash_first)


def _release_check_after_the_entries(monkeypatch):
    """The old place: the version is judged inside the build, after the entries resolved, not at the top of the load."""
    import mime.experiment._experiment as experiment_module

    real_check = experiment_module.check_load
    real_refusal = saved_folder.other_release_refusal
    _version_check_removed(monkeypatch)

    def inside_the_build(form, record, stages=None, **kwargs):
        refusal = real_refusal(form)
        if refusal is not None:
            raise LoadRefused([refusal])
        return real_check(form, record, stages, **kwargs)

    monkeypatch.setattr(experiment_module, "check_load", inside_the_build)


@pytest.mark.parametrize("break_it", [_version_check_removed, _hash_first])
def test_the_controls_the_version_check_removed_and_the_checks_in_the_other_order_fail_the_assertion(
        tmp_path, monkeypatch, break_it):
    with monkeypatch.context() as earlier:
        earlier.setattr(mime, "__version__", EARLIER)
        full_toy().save(tmp_path)
    assert_refused_as_another_release(tmp_path, EARLIER)  # the real check passes it
    break_it(monkeypatch)
    with pytest.raises(AssertionError):
        assert_refused_as_another_release(tmp_path, EARLIER)
    with pytest.raises(LoadRefused) as refused:  # and the message the maintainer saw returns
        Experiment.load(tmp_path)
    assert {r.code for r in refused.value.refusals} == {"load.input-hash"}
    assert "the document was changed after its lock was written" in refused.value.refusals[0].message


def _entry_this_mime_no_longer_registers(folder):
    """script_a's drag recorded at a version no installed package registers, as a document from before a change."""
    document = (folder / DOCUMENT_FILE).read_text()
    assert "name: drag\n  version: 0.4.0" in document
    (folder / DOCUMENT_FILE).write_text(document.replace("name: drag\n  version: 0.4.0", "name: drag\n  version: 0.9.9"))
    _edit_lock(folder, lambda lock: [record.update(version="0.9.9") for record in lock["entries"]
                                     if record["name"] == "drag"])


def _body_this_mime_cannot_build(folder):
    """A bind to a body the document does not have: a document that no longer builds."""
    document = (folder / DOCUMENT_FILE).read_text()
    broken = document.replace("  binds:\n    body: probe\n    field: source.push",
                              "  binds:\n    body: ghost\n    field: source.push", 1)
    assert broken != document
    (folder / DOCUMENT_FILE).write_text(broken)


def _folder_of_script_a(tmp_path):
    from script_a import script_a

    script_a().save(tmp_path)
    return tmp_path


def _folder_of_full_toy(tmp_path):
    full_toy().save(tmp_path)
    return tmp_path


FOLDERS_THAT_FAIL_EARLIER = [(_folder_of_script_a, _entry_this_mime_no_longer_registers, "load.version"),
                             (_folder_of_full_toy, _body_this_mime_cannot_build, "load.id")]


@pytest.mark.parametrize("make, break_it, otherwise", FOLDERS_THAT_FAIL_EARLIER)
def test_a_folder_from_another_release_whose_document_no_longer_resolves_is_still_refused_as_another_release(
        tmp_path, make, break_it, otherwise):
    """What a release between two others may do: change an entry's version, or the rules a document is built by. The
    folder is refused for the MIME that saved it, not for an entry to install or a reference to repair."""
    folder = make(tmp_path)
    break_it(folder)
    _record_saved_by(folder, EARLIER)
    assert_refused_as_another_release(folder, EARLIER)


@pytest.mark.parametrize("make, break_it, otherwise", FOLDERS_THAT_FAIL_EARLIER)
def test_the_control_the_same_folders_of_this_release_are_refused_for_what_they_are_and_the_late_check_hides_nothing_else(
        tmp_path, monkeypatch, make, break_it, otherwise):
    """The documents really fail what they are named for (so the test above had something to hide), and the release
    check in its old place, after the entries, is what lets that hide it."""
    folder = make(tmp_path)
    break_it(folder)
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert otherwise in {r.code for r in refused.value.refusals}
    _record_saved_by(folder, EARLIER)
    assert_refused_as_another_release(folder, EARLIER)
    _release_check_after_the_entries(monkeypatch)
    with pytest.raises(AssertionError):
        assert_refused_as_another_release(folder, EARLIER)
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert otherwise in {r.code for r in refused.value.refusals}


def test_an_experiment_edited_after_it_was_loaded_is_refused_in_words_about_the_edit(saved):
    experiment = Experiment.load(saved)
    experiment.add_body(Body("second", dynamics=ToyBody(), initial=PROBE_START))
    with pytest.raises(LoadRefused) as refused:
        experiment.build()
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.input-hash" and refusal.category is RefusalClass.USER
    assert "edited after it was loaded" in refusal.message and "second" in refusal.message
    assert "notebook" not in refusal.message and "changed after its lock was written" not in refusal.message
    assert f"Experiment.load({str(saved)!r})" in refusal.action
    assert f"Experiment.based_on({str(saved)!r}, name=\"<a new name>\")" in refusal.action


def _edit(experiment):
    experiment.add_body(Body("second", dynamics=ToyBody(), initial=PROBE_START))


def test_the_actions_the_edited_refusal_names_work_and_the_one_it_rules_out_is_refused(saved, tmp_path):
    """Load the folder again; or make the edit in a new experiment based on the folder and save that. Saving the
    loaded, edited experiment is what cannot be done, and it cannot."""
    edited = Experiment.load(saved)
    _edit(edited)
    with pytest.raises(LoadRefused):
        edited.build()
    with pytest.raises(LoadRefused) as refused:
        edited.save(tmp_path / "from_the_loaded")
    assert refused.value.refusals[0].code == "load.input-hash"
    again = Experiment.load(saved)  # drop the edit
    assert again.load_report is not None and "second" not in (saved / DOCUMENT_FILE).read_text()
    based = Experiment.based_on(saved, name="with_a_second_body")  # keep it: the edit in a new experiment
    _edit(based)
    based.save(tmp_path / "from_based_on")
    kept = Experiment.load(tmp_path / "from_based_on")
    assert kept.load_report is not None and "second" in (tmp_path / "from_based_on" / DOCUMENT_FILE).read_text()


def edited_after_load_action(saved):
    experiment = Experiment.load(saved)
    _edit(experiment)
    with pytest.raises(LoadRefused) as refused:
        experiment.build()
    return refused.value.refusals[0].action


def test_the_action_names_based_on_now_that_it_exists(saved, monkeypatch):
    """``Experiment.based_on`` exists (U15): the action names it, with the name it requires, and says how to drop the
    edit. The control: with the switch set back, the action names what worked before it and not ``based_on``."""
    assert saved_folder._BASED_ON_EXISTS is True
    action = edited_after_load_action(saved)
    assert f"Experiment.based_on({str(saved)!r}, name=\"<a new name>\")" in action
    assert f"Experiment.load({str(saved)!r})" in action and "cannot be built or saved" not in action
    monkeypatch.setattr(saved_folder, "_BASED_ON_EXISTS", False)
    before = edited_after_load_action(saved)
    assert "Experiment.based_on" not in before and "cannot be built or saved" in before


def test_the_edit_after_a_load_in_a_notebook_or_ipython_says_so(saved, monkeypatch):
    monkeypatch.setattr(builtins, "get_ipython", lambda: object(), raising=False)
    experiment = Experiment.load(saved)
    experiment.add_body(Body("second", dynamics=ToyBody(), initial=PROBE_START))
    with pytest.raises(LoadRefused) as refused:
        experiment.build()
    (refusal,) = refused.value.refusals
    assert "edited after it was loaded" in refusal.message
    assert "a notebook or IPython: a line or cell run after the load changed it" in refusal.message


def test_the_control_a_document_edited_on_disk_is_not_an_edit_after_a_load(saved):
    """The load's own check, before any edit in memory, still speaks of the document."""
    path = saved / DOCUMENT_FILE
    path.write_text(path.read_text().replace("  binds:\n    body: probe\nobservations",
                                             "  binds:\n    body: probe\n  settings:\n    k:\n"
                                             "      value: 2.5\n      unit: kg s^-2\nobservations"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(saved)
    assert "edited after it was loaded" not in refused.value.refusals[0].message
