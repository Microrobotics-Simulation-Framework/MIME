import importlib
import importlib.metadata
import pkgutil
import sys

import pytest

import mime

# The subpackages of the 0.3 package layout; the manifest, src/mime/README.md, says what each is for.
LAYOUT = {
    "build",
    "cli",
    "control",
    "core",
    "data",
    "docs",
    "dynamics",
    "experiment",
    "fields",
    "gradients",
    "interactions",
    "mappings",
    "materials",
    "nodes",
    "observations",
    "registry",
    "saved",
    "scene",
    "study",
    "surrogates",
    "testing",
    "things",
    "validity",
    "viz",
    "vocabulary",
}


def _subpackages() -> set[str]:
    return {info.name for info in pkgutil.iter_modules(mime.__path__) if info.ispkg}


def test_the_subpackages_are_the_layouts():
    assert _subpackages() == LAYOUT


@pytest.mark.parametrize("name", sorted(LAYOUT))
def test_every_subpackage_imports_and_states_its_role(name):
    module = importlib.import_module(f"mime.{name}")
    assert module.__doc__ and module.__doc__.strip()


def test_the_version_is_the_distributions():
    assert mime.__version__ == importlib.metadata.version("mime-engine")


def test_mime_testing_without_the_verify_extra_names_it(monkeypatch):
    for name in [name for name in sys.modules if name == "mime.testing" or name.startswith("maddening.testing")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "hypothesis", None)
    with pytest.raises(ImportError, match=r"mime-engine\[verify\]"):
        importlib.import_module("mime.testing")
