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
    assert f"saved by MIME {saved_by}, and this is MIME {CURRENT}" in refusal.message
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


def test_a_folder_unpacked_from_a_published_record_says_so_and_names_its_page(saved):
    _edit_lock(saved, lambda lock: lock.update(publication={
        "service": "zenodo-sandbox", "record": "123", "doi": "10.5072/zenodo.123", "concept": "122", "version": 2}))
    _record_saved_by(saved, EARLIER)
    refusal = assert_refused_as_another_release(saved, EARLIER)
    assert "came from `mime pull`, the published record 10.5072/zenodo.123 (version 2)" in refusal.message
    assert "https://sandbox.zenodo.org/records/123" in refusal.action
    _edit_lock(saved, lambda lock: lock.pop("publication"))
    assert "mime pull" not in assert_refused_as_another_release(saved, EARLIER).message


def _version_check_removed(monkeypatch):
    monkeypatch.setattr(saved_folder, "other_release_refusal", lambda form, running=None: None)


def _hash_first(monkeypatch):
    """The checks in the other order: what the hash check refuses is raised before the version is judged."""
    import mime.experiment._experiment as experiment_module

    real = experiment_module.check_load

    def hash_first(form, record, stages=None, **kwargs):
        with monkeypatch.context() as without:
            _version_check_removed(without)
            report = real(form, record, stages, **kwargs)  # raises what the hash refuses, before the version is judged
        refusal = saved_folder.other_release_refusal(form)
        if refusal is not None:
            raise LoadRefused([refusal])
        return report

    monkeypatch.setattr(experiment_module, "check_load", hash_first)


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


def test_an_experiment_edited_after_it_was_loaded_is_refused_in_words_about_the_edit(saved):
    experiment = Experiment.load(saved)
    experiment.add_body(Body("second", dynamics=ToyBody(), initial=PROBE_START))
    with pytest.raises(LoadRefused) as refused:
        experiment.build()
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.input-hash" and refusal.category is RefusalClass.USER
    assert "edited after it was loaded" in refusal.message and "second" in refusal.message
    assert "notebook" not in refusal.message and "changed after its lock was written" not in refusal.message
    assert "built.save(" in refusal.action and "Experiment.load(" in refusal.action


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
