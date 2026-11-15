"""``mime scaffold``'s tests run in a private home (the sharing tests' ``homes.isolate``): the defaults it reads and writes,
the tokens it searches for and the acceptances it must not touch are in a folder of the test's own."""

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sharing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from homes import isolate  # noqa: E402
from scaffold_rig import PluginBase, PluginClean, plugin_base, plugin_clean  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def module_environment_restored() -> Iterator[None]:
    """The process's environment as the module found it, put back after the module. The module-scoped fixtures below
    first run inside a test, after ``private_home`` has set that test's own home and assets cache: the monkeypatch each
    opens there takes those as the values to restore, so without this the module's last teardown left the *first test's*
    ``MIME_ASSETS_DIR`` (a folder with no cache in it) for every later test of the worker: an ``asset.missing`` in
    whichever file came next, only in a shard where one did (``tests/scaffold/test_environment_restored.py``)."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


@pytest.fixture(autouse=True)
def private_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = isolate(monkeypatch, tmp_path / "home")
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "default-cache"))
    yield home


@pytest.fixture(scope="module")
def clean_plugin_base(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PluginBase]:
    """The clean plugin scaffolded, built and listed once for the module (a build and a registry listing, in a home of its
    own: the per-test home below is not yet there at module scope). The plants copy its folder and re-write one file."""
    root = tmp_path_factory.mktemp("clean-plugin")
    with pytest.MonkeyPatch.context() as patch:
        isolate(patch, root / "home")
        patch.setenv("MIME_ASSETS_DIR", str(root / "default-cache"))
        yield plugin_base(root)


@pytest.fixture(scope="module")
def clean_plugin(clean_plugin_base: PluginBase) -> Iterator[PluginClean]:
    """The clean plugin's whole check, run once for the module: the three tests that read its outcome assert on it, and on
    this process's registry after it (the stages that run the plugin's own tests, in subprocesses, are most of its time)."""
    with pytest.MonkeyPatch.context() as patch:
        isolate(patch, clean_plugin_base.folder.parent / "home")
        patch.setenv("MIME_ASSETS_DIR", str(clean_plugin_base.folder.parent / "default-cache"))
        yield plugin_clean(clean_plugin_base)
