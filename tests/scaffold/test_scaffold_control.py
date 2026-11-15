"""The CI control of ``mime scaffold`` (design §5): each of the five kinds scaffolded with ``--yes`` into a temporary folder
passes its own check (``scaffold_rig.failures_of``): an experiment runs and its folder verifies; a bundle, a study and a block get a
``mime.toml`` that ``mime check`` exits 0 on with exactly the kind's to-do list; a plugin is unpacked into a temporary target,
where a subprocess lists its entry with ``mime registry --json`` and runs its own tests.

**Caught n of 5.** One misspelt key planted in each kind's template (a bundle's and a study's text is push's one writer's,
and the plant edits its output) fails that kind's check, on every run: the plants are deterministic edits, so one run
shows the count (one test a kind: five that pass is 5 of 5) and none is rated.

**The plugin is never installed here.** The control's plugin lives in a temporary target that only its subprocesses see;
a test that installed it into this environment would leave a namespace in the registry every later test of the shard
reads (the registry walk, the golden counts, the refusal table). Its own test: after the control has run, this process's
entry points, registry and import path are what they were; and the same check, with the plugin's target planted on this
process's path, fails."""

import importlib.metadata
import sys
from pathlib import Path

import pytest
from scaffold_rig import KINDS, NAMESPACE, failures_of, planted, unpacked

from mime import scaffold
from mime.registry import registered


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_scaffolded_with_yes_passes_its_own_check(kind: str, tmp_path: Path):
    assert failures_of(kind, tmp_path) == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_planted_defect_in_the_kinds_template_is_caught(kind: str, tmp_path: Path):
    """One test a kind, so the count is the five that pass: caught 5 of 5."""
    with pytest.MonkeyPatch.context() as patch:
        planted(kind)(patch)
        found = failures_of(kind, tmp_path / kind)
    assert found, f"a misspelt key in the {kind} template passed its check"


def entry_points() -> list[str]:
    return sorted(f"{each.name}={each.value}" for each in importlib.metadata.entry_points(group="mime.entries"))


def environment() -> tuple[list[str], list[tuple[str, str, str]], list[str], bool]:
    """What every later test in this process would read: the ``mime.entries`` entry points, the registry, the import path
    and whether the plugin's package is imported."""
    return (entry_points(), [(r.kind.value, r.name, r.version) for r in registered()], list(sys.path),
            any(name.startswith("scafx_") for name in sys.modules))


def test_the_plugin_control_leaves_this_process_registry_unchanged(tmp_path: Path):
    before = environment()
    assert failures_of("plugin", tmp_path) == []
    assert environment() == before
    assert not [each for each in importlib.metadata.distributions() if each.metadata["Name"] == "scafx-probes"]


def test_the_unchanged_check_fails_when_the_plugin_is_put_on_this_processs_path(tmp_path: Path):
    """The control of the test above: a plugin installed where this process reads is caught, on every run."""
    plugin = tmp_path / "scafx-probes"
    scaffold.scaffold("plugin", plugin, yes=True, sets=[f"plugin.namespace={NAMESPACE}"], say=lambda line: None)
    before = environment()
    target = unpacked(plugin, tmp_path / "target")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(target))
        after = environment()
    assert after != before
    assert any(each.startswith(f"{NAMESPACE}=") for each in after[0])
