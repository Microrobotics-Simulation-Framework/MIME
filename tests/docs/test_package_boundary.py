"""The package boundary (C1). ``mime.docs`` may import ``mime.testing``, and ``mime.testing`` never imports
``mime.docs``; what ``mime registry --json`` imports from ``mime.docs`` imports nothing from ``mime.testing``; and
``mime.cli`` imports ``mime.docs`` only inside a command's function, which a scan of the whole file could not tell
from an import at module level. So without the ``verify`` extra, ``import mime.cli``, ``mime verify`` and ``mime
registry --json`` work, and ``mime docs check`` exits naming the extra, with no traceback.

The controls: each scan finds a planted import of the kind it forbids, and passes the kind it allows."""

import ast
import json
import subprocess
import sys
from pathlib import Path

from script_a import script_a

import mime
from mime._extras import VERIFY_EXTRA_MISSING
from mime.units import m_per_s

PACKAGE = Path(mime.__file__).parent
#: What ``mime registry --json`` imports from ``mime.docs``.
EXPORT_MODULES = ("__init__.py", "_export.py", "_lists.py", "_refusal_scan.py", "_symbols.py")


def _named(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return {node.module, *(f"{node.module}.{alias.name}" for alias in node.names)}
    return set()


def imports(source: str, *, in_functions: bool) -> set[str]:
    """The modules ``source`` imports at module level (class bodies included), or inside functions."""
    found: set[str] = set()

    def walk(node: ast.AST, inside: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = inside or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if nested == in_functions:
                found.update(_named(child))
            walk(child, nested)

    walk(ast.parse(source), False)
    return found


def of(names: set[str], package: str) -> set[str]:
    return {name for name in names if name == package or name.startswith(f"{package}.")}


def every_import(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    return imports(source, in_functions=False) | imports(source, in_functions=True)


def test_the_cli_imports_mime_docs_only_inside_a_function():
    sources = {path: path.read_text(encoding="utf-8") for path in sorted((PACKAGE / "cli").glob("*.py"))}
    assert {path.name for path in sources} == {"__init__.py", "__main__.py"}
    assert all(not of(imports(source, in_functions=False), "mime.docs") for source in sources.values())
    assert of(imports(sources[PACKAGE / "cli" / "__init__.py"], in_functions=True), "mime.docs"), \
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
    assert {str(path.relative_to(PACKAGE)): of(every_import(path), "mime.docs") for path in paths
            if of(every_import(path), "mime.docs")} == {}


def test_what_the_registry_export_imports_from_mime_docs_imports_nothing_from_mime_testing():
    paths = [PACKAGE / "docs" / name for name in EXPORT_MODULES]
    assert all(path.is_file() for path in paths)
    assert of(every_import(PACKAGE / "docs" / "_check.py"), "mime.testing"), \
        "the scan does not see the docs check's own import of mime.testing, so it could not see one here"
    assert {path.name: of(every_import(path), "mime.testing") for path in paths
            if of(every_import(path), "mime.testing")} == {}
    within = {f"mime.docs.{name.removesuffix('.py')}" for name in EXPORT_MODULES if name != "__init__.py"}
    assert {path.name: sorted(modules - within) for path in paths
            if (modules := {n for n in of(every_import(path), "mime.docs") if n.count(".") == 2}) - within} == {}, \
        "a module the export imports reaches another of mime.docs's, which this test does not scan"


def test_the_scans_find_a_planted_import():
    assert of(imports("from mime.docs import read_list\n", in_functions=False), "mime.docs")
    assert of(imports("import mime.testing.toys\n", in_functions=False), "mime.testing")
    assert not of(imports("import mime.testingly\nfrom mime import docs_of\n", in_functions=False), "mime.docs")


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


def test_the_control_with_the_extra_the_export_loads_every_namespace():
    done = subprocess.run([sys.executable, "-m", "mime.cli", "registry", "--json"], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0 and json.loads(done.stdout)["not_loaded"] == {}
