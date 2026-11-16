"""``mime.toml``'s names (its file, its format, its family, the file it replaced, a block's tables) are defined once, in
``mime._manifest_file``; ``mime.push``, ``mime.blocks``, ``mime.results`` and ``mime.saved`` read them from there. The
control: the scan finds each kind of copy planted in a module, and the template a block folder is scaffolded from states the
format the constant holds."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import mime
from mime import _manifest_file
from mime.scaffold import _templates

SOURCE = Path(mime.__file__).resolve().parent
HOME = Path(_manifest_file.__file__).resolve()
#: The family ("toml") is left out of the scan: it is a word too common to find a copy of.
NAMES = ("MANIFEST_FILE", "MANIFEST_FORMAT", "OLD_FILE")


def copies(source: str) -> list[str]:
    """What ``source`` defines again: a string that is one of the names' values, or the block tables written out."""
    held = {getattr(_manifest_file, name) for name in NAMES}
    tables = tuple(_manifest_file.BLOCK_TABLES)
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in held:
            found.append(node.value)
        elif isinstance(node, ast.Tuple) and tuple(getattr(each, "value", None) for each in node.elts) == tables:
            found.append("the block tables")
    return found


def test_no_module_defines_a_name_of_mime_toml_again():
    found = {path.relative_to(SOURCE).as_posix(): copies(path.read_text(encoding="utf-8"))
             for path in sorted(SOURCE.rglob("*.py")) if path.resolve() != HOME}
    assert {where: held for where, held in found.items() if held} == {}


def test_the_scan_finds_each_kind_of_copy():
    assert copies('MANIFEST_FILE = "mime.toml"') == ["mime.toml"]
    assert copies('format = {"format": "mime.toml/1"}') == ["mime.toml/1"]
    assert copies('SHARING_FILES = ("mime.toml", "sharing.yaml")') == ["mime.toml", "sharing.yaml"]
    tables = ", ".join(repr(each) for each in _manifest_file.BLOCK_TABLES)
    assert copies(f"TABLES = ({tables})") == ["the block tables"]
    assert copies('f"{root} holds no {MANIFEST_FILE}"') == []


def test_every_former_home_reads_the_one_definition():
    from mime import blocks, push
    from mime.blocks import _check, _container, _definition
    from mime.push import _manifest
    from mime.results import _study_folder
    from mime.saved import _bundle

    assert push.MANIFEST_FILE is _manifest_file.MANIFEST_FILE and blocks.MANIFEST_FILE is _manifest_file.MANIFEST_FILE
    assert push.MANIFEST_FORMAT is _manifest_file.MANIFEST_FORMAT
    assert _manifest.BLOCK_TABLES is _manifest_file.BLOCK_TABLES is _definition.BLOCK_TABLES
    assert _study_folder.MANIFEST_FILE is _check.MANIFEST_FILE is _container.MANIFEST_FILE
    assert _bundle.SHARING_FILES == (_manifest_file.MANIFEST_FILE, _manifest_file.OLD_FILE)
    assert (_manifest_file.MANIFEST_FILE, _manifest_file.MANIFEST_FORMAT, _manifest_file.OLD_FILE) == (
        "mime.toml", "mime.toml/1", "sharing.yaml"), "a changed name is a changed saved form: decide it on purpose"


def test_the_block_template_states_the_format_the_constant_holds():
    template = Path(_templates.__file__).parent / "templates" / "block" / "mime.toml.txt"
    stated = re.findall(r'^format = "([^"]+)"$', template.read_text(encoding="utf-8"), re.MULTILINE)
    assert stated == [_manifest_file.MANIFEST_FORMAT]
