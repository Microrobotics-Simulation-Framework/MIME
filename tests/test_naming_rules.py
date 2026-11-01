"""Three of the naming rules in CONTRIBUTING.md, enforced: (2) no public class of ``mime.fields``, ``mime.interactions``,
``mime.dynamics``, ``mime.observations``, ``mime.drives`` or ``mime.scene`` ends with its package's category word (the
namespace already says it: ``fields.UniformRotating``, not ``fields.UniformRotatingField``); and (6) no subpackage
exports a class of another subpackage under a name it also defines itself, so that two classes of one name are each
imported from their own module.

Each check is a function of plain data (the exported classes with the subpackage each was defined in, and the class
names each subpackage defines), so each control hands it a planted violation, which it must report on every run. A
control that found nothing would pass for a scan that found nothing, so the real scan must first find classes to judge.

And (5, for the one name that broke it) the method of regularised Stokeslets is named for what it is: the rung `mrs`,
`Drag.MRS`, `MRSDragNode`, `mime.nodes.hydro.mrs`. It was named `bem`, `Drag.BEM`, `BEMDragNode` for the boundary-element
method, which it is not, and the generic name is kept free for a true boundary-element rung. No file of the repository
holds the old tokens (`BEM`, `Bem`, `bem`, case-sensitive) outside the listed exceptions, each with its reason; a new
exception is a line in ``OLD_NAME_FILES`` or ``OLD_NAME_LINES`` with a reason, never a looser pattern."""

import ast
import importlib
import inspect
import re
import subprocess
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


# -- rule 5: the old name of the method of regularised Stokeslets ---------------------------------------------------------

OLD_NAME = re.compile("BE" + "M|Be" + "m|be" + "m")
#: Keys of data files whose bytes a sha256 pins: kept, since renaming a key means writing the file again.
KEPT_KEYS = re.compile("R_BE" + "Ms|v_BE" + "M_mm_s")
#: A file whose own text may hold the old tokens, with why. A path is a regular expression matched whole.
OLD_NAME_FILES = {
    r"CHANGELOG\.md": "history: the entries of the release say what they said when they were written",
    r"src/mime/saved/_versions\.py": "the reader of the saved form's version 2, which still accepts the old rung",
    r"tests/saved/previous/.*": "saved folders of versions 1 and 2, written before the rename",
    r"tests/saved/test_form_v2\.py": "the tests of that reader",
    r"tests/test_naming_rules\.py": "this scan",
    r"scripts/sim2real/.*": "phase 7j-a's scripts and data: their arms and outputs are named as the frozen "
                            "preregistration and predictions key on them",
    r"tests/docs/pages_0\.2_prose\.txt": "the 0.2 line's pages, quoted as they were",
    r"tests/differential/references/(mrs_convergence|port_mrs_provenance|port_mrs_full_provenance)\.json":
        "a frozen record of what ran under the old name, its bytes pinned",
}
#: A line of a file that may hold the old tokens, with why: a path expression, a line expression and the reason.
OLD_NAME_LINES = [
    (r".*", r"formerly (the )?(" + OLD_NAME.pattern + ")", "history, stated where it is told"),
    (r"tests/oracles/references/dejongh_mrs_offsets\.json", r'^ *"(what|generator|copied_from)"',
     "where the file came from, in the 0.2 line's words"),
    (r"src/mime/data/__init__\.py", r"regularised-Stokeslet " + "BE" + "M resistances",
     "how the 0.2 line trained the weights, in its words"),
]
SKIPPED_DIRECTORIES = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", "dist", "build", "site"}


def old_name_findings(path: str, text: str) -> list[str]:
    """Each line of ``text`` (the file ``path``, below the repository's root) that holds the old tokens and is no
    listed exception, as ``path:line: text``."""
    if any(re.fullmatch(pattern, path) for pattern in OLD_NAME_FILES):
        return []
    found: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not OLD_NAME.search(KEPT_KEYS.sub("", line)):
            continue
        if any(re.fullmatch(where, path) and re.search(what, line) for where, what, _ in OLD_NAME_LINES):
            continue
        found.append(f"{path}:{number}: {line.strip()[:120]}")
    return found


def repository_files() -> list[Path]:
    """The files git tracks, or every file below the root when git is not there."""
    root = PACKAGE.parents[1]
    try:
        listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
        return [root / name for name in listed.decode().split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        return [path for path in root.rglob("*") if path.is_file()
                and not SKIPPED_DIRECTORIES & set(path.relative_to(root).parts)]


def test_no_file_holds_the_old_name_of_the_method_of_regularised_stokeslets_outside_the_listed_exceptions():
    root = PACKAGE.parents[1]
    files = repository_files()
    assert len(files) > 500, "the scan found too few files to judge"
    names = [str(path.relative_to(root)) for path in files if OLD_NAME.search(path.name)]
    assert [name for name in names if not any(re.fullmatch(pattern, name) for pattern in OLD_NAME_FILES)] == []
    found: list[str] = []
    for path in files:
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found += old_name_findings(path.relative_to(root).as_posix(), text)
    assert found == []


def test_the_old_name_scan_reports_a_planted_old_name_on_every_run():
    planted = "exp.attach(Drag.BE" + "M(), body=ball, medium=water)\nthe be" + "m rung\n"
    assert len(old_name_findings("docs/anywhere.md", planted)) == 2
    assert len(old_name_findings("src/mime/nodes/hydro/mrs/_node.py", "class BE" + "MDragNode:")) == 1
    assert old_name_findings("src/mime/saved/_versions.py", planted) == [], "a listed file may hold it"
    kept = 'MRS["R_BE' + 'Ms"] and point["v_BE' + 'M_mm_s"]'
    assert old_name_findings("tests/oracles/test_drag.py", kept) == [], "a pinned file's keys are kept"
    assert old_name_findings("docs/x.md", "the rung (formerly the be" + "m rung)") == []
    two = "formerly the be" + "m rung\nand a be" + "m rung"
    assert len(old_name_findings("docs/x.md", two)) == 1, "a line that says 'formerly' licenses only itself"


def test_the_old_name_scan_reads_the_real_listed_files():
    """Each listed file exists and holds the old name, so a stale entry (a file renamed or cleaned) is found and
    removed rather than kept as a licence."""
    root = PACKAGE.parents[1]
    files = {path.relative_to(root).as_posix(): path for path in repository_files()}
    for pattern in OLD_NAME_FILES:
        held = [name for name in files if re.fullmatch(pattern, name)]
        assert held, f"{pattern} matches no file"
        assert any(OLD_NAME.search(files[name].read_bytes().decode("utf-8", errors="replace")) for name in held), \
            f"{pattern} holds no old name: remove it from the exceptions"
