"""Each outside API has its importers: only ``mime.scene`` and ``mime.viz`` import ``pxr`` (OpenUSD), by an import
statement or by ``importlib.import_module``. The control: the scan finds the scene's own importers, and finds a
planted import in another package."""

import ast
from pathlib import Path

import mime

PACKAGE = Path(mime.__file__).parent
PXR_IMPORTERS = {"scene", "viz"}


def imports_pxr(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "pxr" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pxr":
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                and node.args and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).split(".")[0] == "pxr"):
            return True
    return False


def pxr_importers() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if imports_pxr(path.read_text(encoding="utf-8")):
            relative = path.relative_to(PACKAGE)
            found.setdefault(relative.parts[0], []).append(str(relative))
    return found


def test_only_the_scene_and_viz_import_pxr():
    found = pxr_importers()
    assert "scene" in found, "the scan found no importer at all, so it could not fail"
    assert set(found) <= PXR_IMPORTERS, {k: v for k, v in found.items() if k not in PXR_IMPORTERS}


def test_the_scan_finds_a_planted_import():
    for source in ("import pxr", "from pxr import Usd", "import pxr.Usd as usd",
                   "import importlib\nUsd = importlib.import_module('pxr.Usd')"):
        assert imports_pxr(source), source
    assert not imports_pxr("import pxrish\nfrom mime.scene import Scene")
