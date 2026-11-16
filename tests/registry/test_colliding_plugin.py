"""One distribution that cannot register its namespace does not take the registry of the whole environment with it (the
coordinator's G11, from #313's review): a plugin whose namespace is the first word of one of MIME's own entries
(``helix``) made ``mime registry --json`` exit 3 with an uncoded ``ValueError`` and list no entry at all.

It is now a ``load.namespace-import`` refusal naming the distribution and its module and what to do about it; every other
namespace is loaded and listed, so ``mime registry --json`` prints the rest of the registry and exits 2, as any input
the user must fix does (an environment's reason, never 3, which is MIME's own defect). ``load_namespaces()`` without a
list to collect into raises the same refusal, so the docs build and check stop on it by name.

The plugin is a distribution written into a temporary target that only the subprocesses see (``PYTHONPATH``), as
``tests/scaffold/scaffold_rig.py`` does for the scaffolded plugin: nothing is installed in the environment this suite
runs in, whose registry every later test reads. The controls are a plugin that registers a namespace of its own, which
loads and is listed, and the same registry with no plugin at all."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

COLLIDES = 'from mime.vocabulary import register_namespace\nregister_namespace("helix", "0.1.0")\n'
BREAKS = 'raise RuntimeError("this module is broken")\n'
FINE = 'from mime.vocabulary import register_namespace\nregister_namespace("harmlessplugin", "0.1.0")\n'


def distribution(target: Path, name: str, namespace: str, source: str) -> None:
    """A distribution ``name`` of one module, whose ``mime.entries`` entry point names ``namespace``, written where
    ``importlib.metadata`` finds it for a process whose ``PYTHONPATH`` names ``target``."""
    module = name.replace("-", "_")
    (target / f"{module}.py").write_text(source, encoding="utf-8")
    info = target / f"{module}-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1.0\n", encoding="utf-8")
    (info / "entry_points.txt").write_text(f"[mime.entries]\n{namespace} = {module}\n", encoding="utf-8")


def mime(target: Path | None, *words: str) -> subprocess.CompletedProcess[str]:
    held = os.environ.get("PYTHONPATH")
    path = os.pathsep.join([*([str(target)] if target is not None else []), *([held] if held else [])])
    return subprocess.run([sys.executable, "-m", "mime.cli", *words], capture_output=True, text=True, check=False,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path})


@pytest.fixture(scope="module")
def baseline() -> dict:
    done = mime(None, "registry", "--json")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_plugin_whose_namespace_collides_is_refused_by_name_and_the_rest_of_the_registry_is_listed(tmp_path, baseline):
    distribution(tmp_path, "colliding-plugin", "helix", COLLIDES)
    done = mime(tmp_path, "registry", "--json")
    assert done.returncode == 2, done.stderr
    assert "load.namespace-import" in done.stderr and "distribution colliding-plugin provides namespace 'helix'" in done.stderr
    assert "first word of MIME's entries" in done.stderr and "pip uninstall colliding-plugin" in done.stderr
    assert "[internal]" not in done.stderr and "error.uncoded" not in done.stderr
    held = json.loads(done.stdout)
    assert held["entries"] == baseline["entries"], "every entry the environment holds without the plugin is still listed"
    assert len(held["entries"]) >= 70 and "labx" in held["namespaces"] and "helix" not in held["namespaces"]
    assert list(held["not_loaded"]) == ["helix"] and held["not_loaded"]["helix"].startswith("colliding-plugin: ")


def test_a_plugin_whose_module_fails_for_any_other_reason_is_the_same_refusal(tmp_path, baseline):
    distribution(tmp_path, "broken-plugin", "brokenplugin", BREAKS)
    done = mime(tmp_path, "registry", "--json")
    assert done.returncode == 2 and "distribution broken-plugin provides namespace 'brokenplugin'" in done.stderr
    assert "this module is broken" in done.stderr and json.loads(done.stdout)["entries"] == baseline["entries"]


def test_the_listing_on_a_terminal_and_plain_lists_the_rest_and_refuses_the_same(tmp_path):
    distribution(tmp_path, "colliding-plugin", "helix", COLLIDES)
    done = mime(tmp_path, "--plain", "registry")
    assert done.returncode == 2 and "dynamics  rigid_inertial" in done.stdout
    assert "not loaded: helix (colliding-plugin: " in done.stdout and "load.namespace-import" in done.stderr


def test_two_bad_distributions_are_two_refusals_and_a_good_one_among_them_still_loads(tmp_path):
    distribution(tmp_path, "colliding-plugin", "helix", COLLIDES)
    distribution(tmp_path, "broken-plugin", "brokenplugin", BREAKS)
    distribution(tmp_path, "harmless-plugin", "harmlessplugin", FINE)
    done = mime(tmp_path, "registry", "--json")
    held = json.loads(done.stdout)
    assert done.returncode == 2 and sorted(held["not_loaded"]) == ["brokenplugin", "helix"]
    assert "harmlessplugin" in held["namespaces"], "the namespace that registers is registered"
    assert done.stderr.count("load.namespace-import") >= 2 and "2 reasons" in done.stderr


def test_control_a_plugin_that_registers_a_namespace_of_its_own_is_listed_and_the_exit_is_zero(tmp_path):
    distribution(tmp_path, "harmless-plugin", "harmlessplugin", FINE)
    done = mime(tmp_path, "registry", "--json")
    held = json.loads(done.stdout)
    assert done.returncode == 0, done.stderr
    assert "harmlessplugin" in held["namespaces"] and held["not_loaded"] == {}


def test_load_namespaces_with_no_list_to_collect_into_raises_the_refusal_naming_the_distribution(tmp_path):
    distribution(tmp_path, "colliding-plugin", "helix", COLLIDES)
    program = ("from mime.build import BuildRefused\nfrom mime.docs import load_namespaces\n"
               "try:\n    load_namespaces()\nexcept BuildRefused as error:\n"
               "    print(error.codes(), error.refusals[0].message)\n")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=False,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(tmp_path)})
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("['load.namespace-import'] distribution colliding-plugin provides namespace 'helix'")
