"""The CI control of ``mime scaffold`` (design §5): each of the five kinds scaffolded with ``--yes`` into a temporary folder
passes its own check (``scaffold_rig.failures_of``): an experiment runs and its folder verifies; a bundle, a study and a block get a
``mime.toml`` that ``mime check`` exits 0 on with exactly the kind's to-do list; a plugin is unpacked into a temporary target,
where a subprocess lists its entry with ``mime registry --json`` and runs its own tests.

**Caught n of 5.** One misspelt key planted in each kind's template (a bundle's and a study's text is push's one writer's,
and the plant edits its output) fails that kind's check, on every run: the plants are deterministic edits, so one run
shows the count (one test a kind: five that pass is 5 of 5) and none is rated.

**A plugin's namespace is never one that is taken.** A folder named for a word MIME's own entries begin with
(``helix-probes``) must not default to that namespace: scaffold refuses it, and the same scaffold with the refusal planted
away writes a plugin that breaks ``mime registry`` for the whole environment (the control's own ``plugin_failures`` sees
the exit). The words are read from the registry, never listed here; a namespace an installed distribution declares is
refused too (the plugin's target planted on this process's path, as below).

**The plugin is never installed here.** The control's plugin lives in a temporary target that only its subprocesses see;
a test that installed it into this environment would leave a namespace in the registry every later test of the shard
reads (the registry walk, the golden counts, the refusal table). Its own test: after the control has run, this process's
entry points, import path and imported packages are what they were; and the same check, with the plugin's target planted
on this process's path, fails. (It does not read the registry itself: nothing the plant does reaches the registry's table
without importing the plugin into this process for good, which is what the test is there to prevent; an import is what the
imported-packages half sees.)"""

import importlib.metadata
from pathlib import Path

import pytest
from scaffold_rig import (KINDS, NAMESPACE, PLUGIN_PLANTS, PluginBase, PluginClean, environment, failures_of, planted,
                          plugin_planted_failures, unpacked)

from mime.scaffold import _scaffold
from mime.build import BuildRefused
from mime.scaffold import _values


@pytest.mark.parametrize("kind", [pytest.param(each, marks=pytest.mark.fixture_first) if each == "plugin" else each
                                  for each in KINDS])
def test_each_kind_scaffolded_with_yes_passes_its_own_check(kind: str, tmp_path: Path, request: pytest.FixtureRequest):
    """The plugin's check is run once for the module (``clean_plugin``), by the first test that reads it: its lanes run the
    plugin's own tests in subprocesses, which is most of the module's time. ``fixture_first`` (tests/_fixture_first.py) keeps the
    two tests that read it on one pytest-xdist worker, so that CI pays it once, not once for each worker the tests reach."""
    if kind == "plugin":
        assert request.getfixturevalue("clean_plugin").failures == []
    else:
        assert failures_of(kind, tmp_path) == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_planted_defect_in_the_kinds_template_is_caught(kind: str, tmp_path: Path):
    """One test a kind, so the count is the five that pass: caught 5 of 5."""
    with pytest.MonkeyPatch.context() as patch:
        planted(kind)(patch)
        found = failures_of(kind, tmp_path / kind)
    assert found, f"a misspelt key in the {kind} template passed its check"


@pytest.mark.parametrize("which", PLUGIN_PLANTS)
def test_a_planted_slip_in_another_file_of_the_plugin_is_caught(which: str, clean_plugin_base: PluginBase, tmp_path: Path):
    """The plugin's other files: a misspelt key of the records' entry point, a planted error that does nothing (the plugin's
    own in-process control fails), the variable that plants it misspelt (the planted run finds no failure, and says only the one
    property should fail), an import of mime in the engine (as a statement, through `importlib.import_module` and through `__import__`) and a misspelt command of the README each fail the stage of the
    plugin's check that is there to catch it, with the plugin's own message. The clean plugin is scaffolded and built once for
    the module; each plant is a copy of it with one file written again, and only its stage runs."""
    assert clean_plugin_base.failures == [], "the clean plugin must pass the stages before the plant's, or the plant proves nothing"
    plant = PLUGIN_PLANTS[which]
    found = plugin_planted_failures(clean_plugin_base.folder, clean_plugin_base.target, plant, tmp_path)
    assert found, f"{which} passed the plugin's check"
    assert plant.message in "\n".join(found), f"{which} failed the check, but not by the plugin's own message: {found}"


@pytest.mark.fixture_first
def test_the_plugin_control_leaves_this_process_registry_unchanged(clean_plugin: PluginClean):
    """What the clean plugin's control did to this process, read at its start (before the scaffold) and its end (after the
    build and the subprocess runs), and now: the same, and no distribution of the plugin installed."""
    assert clean_plugin.failures == []
    assert clean_plugin.after == clean_plugin.before
    assert environment() == clean_plugin.before
    assert not [each for each in importlib.metadata.distributions() if each.metadata["Name"] == "scafx-probes"]


def test_the_unchanged_check_fails_when_the_plugin_is_put_on_this_processs_path(tmp_path: Path):
    """The control of the test above: a plugin installed where this process reads is caught, on every run."""
    plugin = tmp_path / "scafx-probes"
    _scaffold.scaffold("plugin", plugin, yes=True, sets=[f"plugin.namespace={NAMESPACE}"], say=lambda line: None)
    before = environment()
    target = unpacked(plugin, tmp_path / "target")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(target))
        after = environment()
    assert after != before
    assert any(each.startswith(f"{NAMESPACE}=") for each in after[0])


def test_a_namespace_an_installed_distribution_declares_is_refused_and_a_free_one_is_not(tmp_path: Path):
    plugin = tmp_path / "scafx-probes"
    _scaffold.scaffold("plugin", plugin, yes=True, sets=[f"plugin.namespace={NAMESPACE}"], say=lambda line: None)
    assert _values.namespace_clash(NAMESPACE) is None, "free until a distribution declares it"
    target = unpacked(plugin, tmp_path / "target")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(target))
        clash = _values.namespace_clash(NAMESPACE)
    assert clash is not None and "declares" in clash and "scafx-probes" in clash
    assert _values.namespace_clash(NAMESPACE) is None, "and free again once it is off the path"


# A folder named for a word MIME's own entries begin with: its derived namespace is that word, which the registry refuses
# to load (``register_namespace``): one such plugin breaks ``mime registry`` for the whole environment.
RESERVED_FOLDER = "helix-probes"


def test_a_plugin_folder_named_for_one_of_mimes_words_is_refused_not_defaulted(tmp_path: Path):
    assert "helix" in _values.taken_namespaces(), "the word is read from the registry, and MIME's entries begin with it"
    with pytest.raises(BuildRefused) as refused:
        _scaffold.scaffold("plugin", tmp_path / RESERVED_FOLDER, yes=True, sets=[], say=lambda line: None)
    refusal = refused.value.refusals[0]
    assert refusal.code == "scaffold.set" and "helix" in refusal.message and "lab_helix" in refusal.message
    assert not (tmp_path / RESERVED_FOLDER).exists()
    assert failures_of("plugin", tmp_path / "again", RESERVED_FOLDER, sets=[]) != [], "the control reports the refusal"
    assert failures_of("plugin", tmp_path / "free", RESERVED_FOLDER, sets=["plugin.namespace=lab_helix"], lanes=False) == [], (
        "a free namespace given by --set is accepted, and the plugin is listed (its lanes are the clean plugin's, run once)")


def test_the_reserved_namespace_control_fails_when_the_refusal_is_planted_away(tmp_path: Path):
    """Without the refusal the scaffold writes ``helix``, and the control's own check sees ``mime registry`` fail."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_values, "taken_namespaces", dict)
        found = failures_of("plugin", tmp_path, RESERVED_FOLDER, sets=[])
    assert any("mime registry --json exited" in each for each in found), found
