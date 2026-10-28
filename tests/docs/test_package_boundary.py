"""The package boundary (C1). ``mime.docs`` may import ``mime.testing``, and ``mime.testing`` never imports
``mime.docs``; what ``mime registry --json`` imports from ``mime.docs`` imports nothing from ``mime.testing``; and
``mime.cli`` imports ``mime.docs`` only inside a command's function, which a scan of the whole file could not tell
from an import at module level. So without the ``verify`` extra, ``import mime.cli``, ``mime verify`` and ``mime
registry --json`` work, and ``mime docs check`` exits naming the extra, with no traceback.

A relative import is resolved against its module's package, so ``from .. import docs`` in ``mime.cli`` is an import of
``mime.docs`` and ``from . import docs`` there is not.

The controls: each scan finds a planted import of the kind it forbids, and passes the kind it allows; and each check,
run on a copy of the package with a relative import planted at module level, names it."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from script_a import script_a

import mime
from mime._extras import VERIFY_EXTRA_MISSING
from mime.units import m_per_s

PACKAGE = Path(mime.__file__).parent
#: What ``mime registry --json`` imports from ``mime.docs``.
EXPORT_MODULES = ("__init__.py", "_export.py", "_lists.py", "_refusal_scan.py", "_symbols.py")


def _named(node: ast.AST, package: str) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if not isinstance(node, ast.ImportFrom):
        return set()
    module = node.module or ""
    if node.level:
        parts = package.split(".")
        assert node.level <= len(parts), f"from {'.' * node.level}{module} reaches above {package!r}"
        module = ".".join([*parts[:len(parts) - node.level + 1], *filter(None, [module])])
    return {module, *(f"{module}.{alias.name}" for alias in node.names)}


def imports(source: str, *, in_functions: bool, package: str = "") -> set[str]:
    """The modules ``source`` imports at module level (class bodies included), or inside functions; a relative import
    resolved against ``package``, the package of the module ``source`` is."""
    found: set[str] = set()

    def walk(node: ast.AST, inside: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = inside or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if nested == in_functions:
                found.update(_named(child, package))
            walk(child, nested)

    walk(ast.parse(source), False)
    return found


def of(names: set[str], package: str) -> set[str]:
    return {name for name in names if name == package or name.startswith(f"{package}.")}


def package_of(path: Path, package: Path) -> str:
    """The package a module of ``package``'s source (``src/mime``) resolves its relative imports against."""
    return ".".join(path.relative_to(package.parent).with_suffix("").parts[:-1])


def module_imports(path: Path, package: Path, *, in_functions: bool) -> set[str]:
    return imports(path.read_text(encoding="utf-8"), in_functions=in_functions, package=package_of(path, package))


def every_import(path: Path, package: Path = PACKAGE) -> set[str]:
    return module_imports(path, package, in_functions=False) | module_imports(path, package, in_functions=True)


def cli_imports_of_docs(package: Path) -> dict[str, set[str]]:
    """Each file of ``mime.cli`` that imports from ``mime.docs`` at module level, with what it imports."""
    return {path.name: found for path in sorted((package / "cli").glob("*.py"))
            if (found := of(module_imports(path, package, in_functions=False), "mime.docs"))}


def docs_imports_in_testing(package: Path) -> dict[str, set[str]]:
    """Each file of ``mime.testing`` that imports from ``mime.docs`` anywhere, with what it imports."""
    return {str(path.relative_to(package)): found for path in sorted((package / "testing").rglob("*.py"))
            if (found := of(every_import(path, package), "mime.docs"))}


def export_imports_beyond(package: Path) -> dict[str, set[str]]:
    """Each module the export imports from ``mime.docs`` that imports what the export may not: anything of
    ``mime.testing``, or a module of ``mime.docs`` beyond the export's own, which this test does not scan."""
    within = {f"mime.docs.{name.removesuffix('.py')}" for name in EXPORT_MODULES if name != "__init__.py"}
    found: dict[str, set[str]] = {}
    for name in EXPORT_MODULES:
        modules = every_import(package / "docs" / name, package)
        beyond = of(modules, "mime.testing") | ({n for n in of(modules, "mime.docs") if n.count(".") == 2} - within)
        if beyond:
            found[name] = beyond
    return found


def test_the_cli_imports_mime_docs_only_inside_a_function():
    assert {path.name for path in (PACKAGE / "cli").glob("*.py")} == {"__init__.py", "__main__.py"}
    assert cli_imports_of_docs(PACKAGE) == {}
    assert of(module_imports(PACKAGE / "cli" / "__init__.py", PACKAGE, in_functions=True), "mime.docs"), \
        "the scan found no import of mime.docs inside a function, so it could not tell one at module level"


def test_the_function_scan_tells_an_import_in_a_function_from_one_at_module_level():
    top = "from mime.docs._check import run_check\n\ndef f():\n    return run_check\n"
    guarded = "import sys\nif sys:\n    import mime.docs\n"
    inside = "def f():\n    from mime.docs._check import run_check\n    return run_check\n"
    method = "class C:\n    def m(self):\n        import mime.docs\n"
    assert of(imports(top, in_functions=False), "mime.docs") and of(imports(guarded, in_functions=False), "mime.docs")
    assert not of(imports(inside, in_functions=False), "mime.docs")
    assert not of(imports(method, in_functions=False), "mime.docs")
    assert of(imports(inside, in_functions=True), "mime.docs")


def test_mime_testing_imports_nothing_from_mime_docs():
    paths = sorted((PACKAGE / "testing").rglob("*.py"))
    assert any(of(every_import(path), "mime") for path in paths), "the scan found no import at all"
    assert docs_imports_in_testing(PACKAGE) == {}


def test_what_the_registry_export_imports_from_mime_docs_imports_nothing_from_mime_testing():
    assert all((PACKAGE / "docs" / name).is_file() for name in EXPORT_MODULES)
    assert of(every_import(PACKAGE / "docs" / "_check.py"), "mime.testing"), \
        "the scan does not see the docs check's own import of mime.testing, so it could not see one here"
    assert export_imports_beyond(PACKAGE) == {}


def test_the_scans_find_a_planted_import():
    assert of(imports("from mime.docs import read_list\n", in_functions=False), "mime.docs")
    assert of(imports("import mime.testing.toys\n", in_functions=False), "mime.testing")
    assert not of(imports("import mime.testingly\nfrom mime import docs_of\n", in_functions=False), "mime.docs")


def test_a_relative_import_is_resolved_against_its_modules_package():
    def found(source: str, package: str) -> set[str]:
        return of(imports(source, in_functions=False, package=package), "mime.docs")

    assert found("from . import docs\n", "mime") == {"mime.docs"}
    assert found("from .docs import x\n", "mime") == {"mime.docs", "mime.docs.x"}
    assert found("from .. import docs\n", "mime.cli") == {"mime.docs"}
    assert found("from ..docs._lists import read_list\n", "mime.testing") == {"mime.docs._lists",
                                                                              "mime.docs._lists.read_list"}
    assert found("from . import docs\n", "mime.cli") == set(), "mime.cli's own docs is not mime.docs"
    assert of(imports("from . import docs\n", in_functions=False, package="mime.cli"), "mime.cli") == {
        "mime.cli", "mime.cli.docs"}


def planted_copy(tmp_path: Path, module: str, line: str) -> Path:
    """A copy of the package's source with ``line`` added at module level of ``module`` (``cli/__init__.py``)."""
    copy = tmp_path / "mime"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__", "*.npz", "*.usd*"))
    target = copy / module
    target.write_text(target.read_text(encoding="utf-8") + f"\n{line}\n", encoding="utf-8")
    return copy


@pytest.mark.parametrize(("module", "line", "named"), [
    ("cli/__init__.py", "from .. import docs as _d", {"mime.docs"}),
    ("cli/__init__.py", "from ..docs._lists import read_list", {"mime.docs._lists", "mime.docs._lists.read_list"}),
    ("cli/__main__.py", "from ..docs import _check", {"mime.docs", "mime.docs._check"}),
], ids=["from .. import docs", "from ..docs._lists import", "from ..docs import"])
def test_c1_a_relative_import_of_mime_docs_at_module_level_of_the_cli_fails(tmp_path, module, line, named):
    assert cli_imports_of_docs(planted_copy(tmp_path, module, line)) == {Path(module).name: named}


@pytest.mark.parametrize(("module", "line", "named"), [
    ("testing/_nodes.py", "from .. import docs", {"mime.docs"}),
    ("testing/toys/_nodes.py", "from ...docs._lists import read_list", {"mime.docs._lists",
                                                                        "mime.docs._lists.read_list"}),
], ids=["from .. import docs", "from ...docs._lists import"])
def test_c1_a_relative_import_of_mime_docs_in_mime_testing_fails(tmp_path, module, line, named):
    assert docs_imports_in_testing(planted_copy(tmp_path, module, line)) == {module: named}


@pytest.mark.parametrize(("module", "line", "named"), [
    ("_export.py", "from .. import testing", {"mime.testing"}),
    ("_symbols.py", "from ..testing import installed_entries", {"mime.testing", "mime.testing.installed_entries"}),
    ("_lists.py", "from . import _check", {"mime.docs._check"}),
    ("_refusal_scan.py", "from ._check import check", {"mime.docs._check"}),
], ids=["from .. import testing", "from ..testing import", "from . import", "from ._check import"])
def test_c1_a_relative_import_beyond_the_exports_modules_fails(tmp_path, module, line, named):
    assert export_imports_beyond(planted_copy(tmp_path, f"docs/{module}", line)) == {module: named}


#: A prelude that removes the ``verify`` extra from a fresh process: Hypothesis is what it installs.
WITHOUT_VERIFY = "import sys\nsys.modules['hypothesis'] = None\nfrom mime.cli import main\n"


def without_verify(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", WITHOUT_VERIFY + f"sys.exit(main({list(argv)!r}))"],
                          capture_output=True, text=True, check=False)


def test_without_the_verify_extra_the_cli_imports_and_the_export_runs():
    done = without_verify("registry", "--json")
    assert done.returncode == 0, done.stderr
    exported = json.loads(done.stdout)
    assert exported["not_loaded"].get("toy") == VERIFY_EXTRA_MISSING
    assert {e["namespace"] for e in exported["entries"]} == {"mime"} and len(exported["entries"]) >= 12


def test_without_the_verify_extra_mime_verify_runs(tmp_path):
    script_a("boundary", velocity=m_per_s((1e-3, 0.0, 0.0)), drag=False).save(tmp_path)
    done = without_verify("verify", str(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "reproduced" in done.stdout


def test_without_the_verify_extra_mime_docs_check_names_it_without_a_traceback():
    done = without_verify("docs", "check")
    assert done.returncode == 2
    assert VERIFY_EXTRA_MISSING in done.stderr and "Traceback" not in done.stderr + done.stdout


def test_without_the_verify_extra_mime_docs_build_names_it_without_a_traceback(tmp_path):
    done = without_verify("docs", "build", "--out", str(tmp_path / "pages"))
    assert done.returncode == 2
    assert VERIFY_EXTRA_MISSING in done.stderr and "Traceback" not in done.stderr + done.stdout
    assert not (tmp_path / "pages").exists()


def test_the_control_with_the_extra_the_export_loads_every_namespace():
    done = subprocess.run([sys.executable, "-m", "mime.cli", "registry", "--json"], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0 and json.loads(done.stdout)["not_loaded"] == {}
