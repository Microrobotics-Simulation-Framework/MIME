"""Each outside API has its importers: only ``mime.scene`` and ``mime.viz`` import ``pxr`` (OpenUSD), and only
``mime.materials`` imports ``materica`` (OpenMaterica), by an import statement or by ``importlib.import_module``. The
control: the scan finds each API's own importers, and finds a planted import in another package.

Only one module imports ``rich`` (``mime/cli/_display.py``, what ``mime`` draws on a terminal), and a command that
draws nothing never loads it. The controls: the scan finds a second importer planted in another module, and a fresh
process that draws loads ``rich`` while one that does not, does not.

Nothing in MIME imports ``modal``: the script ``mime modal`` writes is its only importer. The scan finds no importer,
and a fresh process that imports ``mime`` and every submodule holds no ``modal`` module; the controls find a planted
import in source, and a planted module that imports a stand-in ``modal`` in a fresh process.

And ``mime.core`` (with ``mime.core.multibody``, which nodes, dynamics entries and controllers import) imports nothing
from MIME above it. The control: the scan finds core's imports of itself, and a planted import of ``mime.build``."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mime

PACKAGE = Path(mime.__file__).parent
PXR_IMPORTERS = {"scene", "viz"}


def imports_api(source: str, api: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == api for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == api:
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                and node.args and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).split(".")[0] == api):
            return True
    return False


def importers(api: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if imports_api(path.read_text(encoding="utf-8"), api):
            relative = path.relative_to(PACKAGE)
            found.setdefault(relative.parts[0], []).append(str(relative))
    return found


def test_only_the_scene_and_viz_import_pxr():
    found = importers("pxr")
    assert "scene" in found, "the scan found no importer at all, so it could not fail"
    assert set(found) <= PXR_IMPORTERS, {k: v for k, v in found.items() if k not in PXR_IMPORTERS}


def test_only_the_materials_seam_imports_openmaterica():
    found = importers("materica")
    assert "materials" in found, "the scan found no importer at all, so it could not fail"
    assert set(found) == {"materials"}, {k: v for k, v in found.items() if k != "materials"}


def test_the_scan_finds_a_planted_openmaterica_import():
    for source in ("import materica", "from materica.core.record import RecordStore",
                   "import importlib\nm = importlib.import_module('materica')"):
        assert imports_api(source, "materica"), source
    assert not imports_api("import importlib.metadata\nv = importlib.metadata.version('openmaterica')", "materica")


def test_the_scan_finds_a_planted_import():
    for source in ("import pxr", "from pxr import Usd", "import pxr.Usd as usd",
                   "import importlib\nUsd = importlib.import_module('pxr.Usd')"):
        assert imports_api(source, "pxr"), source
    assert not imports_api("import pxrish\nfrom mime.scene import Scene", "pxr")


def mime_imports(source: str) -> set[str]:
    """Every ``mime`` module ``source`` imports, by an import statement or ``importlib.import_module``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[0] == "mime"}
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "mime" and node.level == 0:
            found.add(str(node.module))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                and node.args and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).split(".")[0] == "mime"):
            found.add(str(node.args[0].value))
    return found


def imports_above_core(source: str) -> set[str]:
    return {name for name in mime_imports(source) if name != "mime.core" and not name.startswith("mime.core.")}


def test_core_imports_nothing_from_mime_above_it():
    sources = {path: path.read_text(encoding="utf-8") for path in sorted((PACKAGE / "core").rglob("*.py"))}
    assert any(mime_imports(source) for source in sources.values()), "the scan found no import at all"
    assert any("multibody" in str(path) for path in sources), "the scan did not reach mime.core.multibody"
    above = {str(path.relative_to(PACKAGE)): imports_above_core(source) for path, source in sources.items()}
    assert {path: names for path, names in above.items() if names} == {}


def test_the_scan_finds_a_planted_import_from_above_core():
    for source in ("from mime.build import BuildRefused", "import mime.nodes", "from mime import units",
                   "import importlib\nb = importlib.import_module('mime.build')"):
        assert imports_above_core(source), source
    assert not imports_above_core("from mime.core._node import jnp\nfrom mime.core import quaternion")


def test_nothing_in_mime_imports_modal():
    assert importers("modal") == {}


def test_the_scan_finds_a_planted_modal_import():
    for source in ("import modal", "from modal import App", "import modal.experimental as x",
                   "import importlib\nm = importlib.import_module('modal')"):
        assert imports_api(source, "modal"), source
    assert not imports_api("import modality\nfrom mime.env import modal_script", "modal")


IMPORT_EVERY_MODULE = """
import importlib, json, pkgutil, sys
import mime
names = [i.name for i in pkgutil.walk_packages(mime.__path__, "mime.") if not i.name.endswith("__main__")]
for name in names + sys.argv[1:]:
    importlib.import_module(name)
print(json.dumps({"imported": len(names), "modal": sorted(m for m in sys.modules if m.split(".")[0] == "modal")}))
"""


def imported_modal(*planted: str, path: str = "") -> dict:
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [path, os.environ.get("PYTHONPATH")]))}
    done = subprocess.run([sys.executable, "-c", IMPORT_EVERY_MODULE, *planted], capture_output=True, text=True,
                          env=environment, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_importing_mime_and_every_submodule_imports_no_modal():
    held = imported_modal()
    assert held["imported"] > 100 and held["modal"] == []


def test_a_module_that_imports_modal_is_found_by_the_fresh_process(tmp_path):
    (tmp_path / "modal").mkdir()
    (tmp_path / "modal" / "__init__.py").write_text("")
    (tmp_path / "planted_modal_user.py").write_text("import modal\n")
    assert imported_modal("planted_modal_user", path=str(tmp_path))["modal"] == ["modal"]


#: The one module that may import ``rich``, relative to the package.
RICH_IMPORTER = "cli/_display.py"


def rich_importers(package: Path) -> list[str]:
    return [path.relative_to(package).as_posix() for path in sorted(package.rglob("*.py"))
            if imports_api(path.read_text(encoding="utf-8"), "rich")]


def test_only_the_display_module_imports_rich():
    assert rich_importers(PACKAGE) == [RICH_IMPORTER], "the scan found no importer, or a second one"


def test_the_scan_finds_a_planted_rich_import_and_a_second_importer(tmp_path):
    for source in ("import rich", "from rich.console import Console", "import rich.live as live",
                   "import importlib\nm = importlib.import_module('rich.text')"):
        assert imports_api(source, "rich"), source
    assert not imports_api("import richard\nfrom mime.cli import _display", "rich")
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "_display.py").write_text("from rich.console import Console\n")
    (tmp_path / "errors").mkdir()
    (tmp_path / "errors" / "_shape.py").write_text("def f():\n    import rich\n")
    assert rich_importers(tmp_path) == [RICH_IMPORTER, "errors/_shape.py"] != [RICH_IMPORTER]


LOADS_RICH = """
import io, sys
import mime.cli as cli
for words in ({words}):
    try:
        cli.main(words)
    except SystemExit:
        pass
print("rich" in sys.modules)
"""


def loaded_rich(words: str, draw: bool = False) -> bool:
    body = LOADS_RICH.format(words=words) + ("from mime.cli import _display\n_display._console(sys.stdout)\n"
                                              "print('rich' in sys.modules)\n" if draw else "")
    done = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, timeout=300, check=False,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr
    return done.stdout.split()[-1] == "True"


def test_a_command_that_draws_nothing_never_loads_rich():
    """Help, a bare ``mime``, a mistyped command and a refused duration, piped: none of them touches ``rich``."""
    assert not loaded_rich('["--help"], [], ["verfy"], ["run", "--help"], ["run", "nowhere", "--duration", "10x", '
                           '"--out", "nowhere-out"]')


@pytest.mark.parametrize("command", ["verify", "registry", "env check"])
def test_a_table_command_piped_never_loads_rich(command, tmp_path):
    """The three commands that draw a table on a terminal, piped (here a process whose standard output is a pipe): none of
    them loads ``rich``."""
    words = {"verify": lambda: ["verify", str(saved_toy(tmp_path))], "registry": lambda: ["registry"],
             "env check": lambda: ["env", "check", "--variant", "cpu"]}[command]()
    assert not loaded_rich(repr(words) + ",")


def saved_toy(tmp_path: Path) -> Path:
    import jax

    from mime.testing.toys import full_toy

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        full_toy().save(tmp_path / "toy")
    finally:
        jax.config.update("jax_enable_x64", held)
    return tmp_path / "toy"


def test_the_control_a_process_that_draws_loads_rich():
    assert loaded_rich('["--help"]', draw=True)
