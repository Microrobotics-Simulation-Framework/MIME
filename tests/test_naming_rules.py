"""Two of the naming rules in CONTRIBUTING.md, enforced: (2) no public class of ``mime.fields``, ``mime.interactions``,
``mime.dynamics``, ``mime.observations``, ``mime.drives`` or ``mime.scene`` ends with its package's category word (the
namespace already says it: ``fields.UniformRotating``, not ``fields.UniformRotatingField``); and (6) no subpackage
exports a class of another subpackage under a name it also defines itself, so that two classes of one name are each
imported from their own module.

Each check is a function of plain data (the exported classes with the subpackage each was defined in, and the class
names each subpackage defines), so each control hands it a planted violation, which it must report on every run. A
control that found nothing would pass for a scan that found nothing, so the real scan must first find classes to judge."""

import ast
import importlib
import inspect
from pathlib import Path

import mime

PACKAGE = Path(mime.__file__).parent

#: Each package's category word(s): a public class named ``<stem><word>`` repeats what ``mime.<package>`` says.
CATEGORY_WORDS = {
    "fields": ("Field",),
    "interactions": ("Interaction",),
    "dynamics": ("Dynamics",),
    "observations": ("Observation",),
    "drives": ("Drive",),
    "scene": ("Solid",),
}


def exported_classes(package: str) -> dict[str, str]:
    """The classes ``mime.<package>`` exports (its ``__all__``), each with the subpackage of ``mime`` that defines it."""
    module = importlib.import_module(f"mime.{package}")
    found: dict[str, str] = {}
    for name in module.__all__:
        value = getattr(module, name)
        if inspect.isclass(value):
            found[name] = value.__module__.split(".")[1] if value.__module__.startswith("mime.") else value.__module__
    return found


def defined_classes(package: str) -> set[str]:
    """The names of the classes defined at module level in any file of ``mime/<package>`` (read from the source)."""
    root = PACKAGE / package
    files = [root] if root.suffix == ".py" else sorted(root.rglob("*.py"))
    path = PACKAGE / f"{package}.py"
    if not root.exists() and path.exists():
        files = [path]
    return {node.name for file in files for node in ast.parse(file.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef)}


def subpackages() -> list[str]:
    return sorted(p.name for p in PACKAGE.iterdir() if p.is_dir() and (p / "__init__.py").exists())


def names_ending_with_the_category_word(exports: dict[str, tuple[str, ...]], words: dict[str, tuple[str, ...]]) -> list[str]:
    """Rule 2: ``exports`` maps a package to its public class names. A name equal to the word itself is a base class
    that names the category, not a class repeating it."""
    return [f"mime.{package}.{name} ends with {word!r}, which mime.{package} already says"
            for package, names in sorted(exports.items()) for name in names for word in words.get(package, ())
            if name.endswith(word) and name != word]


def reexports_that_shadow(exports: dict[str, dict[str, str]], defined: dict[str, set[str]]) -> list[str]:
    """Rule 6: ``exports`` maps a package to its exported classes and the subpackage defining each."""
    return [f"mime.{package} exports mime.{home}.{name}, and mime.{package} defines a class {name} of its own"
            for package, classes in sorted(exports.items()) for name, home in sorted(classes.items())
            if home != package and name in defined.get(package, set())]


def test_no_public_class_ends_with_its_packages_category_word():
    exports = {package: tuple(exported_classes(package)) for package in CATEGORY_WORDS}
    assert sum(len(names) for names in exports.values()) >= 10, "the scan found too few classes to judge"
    assert all(exports.values()), f"a package exports no class, so it could not fail: {exports}"
    assert names_ending_with_the_category_word(exports, CATEGORY_WORDS) == []


def test_the_category_word_check_reports_a_planted_name_on_every_run():
    for package, words in CATEGORY_WORDS.items():
        planted = {package: ("Fine", f"Planted{words[0]}")}
        assert names_ending_with_the_category_word(planted, CATEGORY_WORDS) == [
            f"mime.{package}.Planted{words[0]} ends with {words[0]!r}, which mime.{package} already says"]
    real = {package: tuple(exported_classes(package)) for package in CATEGORY_WORDS}
    assert names_ending_with_the_category_word({**real, "fields": (*real["fields"], "PointDipoleField")},
                                               CATEGORY_WORDS), "the real exports, with the old name added, pass"


def test_no_subpackage_exports_a_class_of_another_under_a_name_it_defines():
    exports = {package: exported_classes(package) for package in subpackages()
               if hasattr(importlib.import_module(f"mime.{package}"), "__all__")}
    defined = {package: defined_classes(package) for package in exports}
    reexported = [(package, name) for package, classes in exports.items() for name, home in classes.items()
                  if home != package]
    assert reexported, "no re-export of a class was found, so the scan could not fail"
    assert reexports_that_shadow(exports, defined) == []


def test_the_reexport_check_reports_a_planted_shadow_on_every_run():
    planted = {"scene": {"Cylinder": "core", "Tube": "scene"}}
    assert reexports_that_shadow(planted, {"scene": {"Cylinder", "Tube"}}) == [
        "mime.scene exports mime.core.Cylinder, and mime.scene defines a class Cylinder of its own"]
    assert reexports_that_shadow(planted, {"scene": {"Tube"}}) == [], "a re-export under a name not defined is allowed"
    real = {"scene": {**exported_classes("scene"), "Cylinder": "core"}}
    assert "Tube" in defined_classes("scene"), "the source scan found no class of the scene, so it could not fail"
    assert reexports_that_shadow(real, {"scene": defined_classes("scene") | {"Cylinder"}}), "a planted shadow passes"
