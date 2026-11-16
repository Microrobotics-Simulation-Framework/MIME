"""The checks every packer makes alike are one function each, in ``mime.saved._container``: the output outside the folder,
a file the kind's folder does not keep, two names that are one, and a file over the size limit (the message of its
refusal: each kind's action differs).

``pack`` (a saved folder), ``pack_study`` and ``pack_block`` call all four, ``pack_studies`` the first and the third. Three
tests hold it. The packers' modules hold the shared function itself and no copy of the code or the text it writes. Each
failure planted in the folder is refused by every packer, with the code its check owns. And each shared check, replaced in
a packer's module by one that always refuses with a message of its own, is the refusal the packer makes: the packer runs it
and does not make the judgement again beside it.
"""

from __future__ import annotations

import ast
import importlib
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from push_cases import saved_toy
from study_rigs import copy_of, write_study

import mime.blocks._container as block_packer
from mime.build import BuildRefused, MimeError
from mime.build import RefusalClass
from mime.saved import _bundle, _container, _studies, pack, pack_studies

ROOT = Path(__file__).resolve().parents[2]
PLANTED = "planted by the test"
SOURCE = Path(_container.__file__).resolve().parents[1]
SHARED = ("inside_problem", "stray_refusals", "collision_refusals", "over_limit")


@dataclass(frozen=True)
class Packer:
    """One packer: how it is called on a folder, the module whose namespace holds the shared checks it calls, and the
    module whose size limit it reads."""

    name: str
    run: Callable[[Path, Path | None], object]
    module: ModuleType
    uses: tuple[str, ...]


def pack_study_folder(folder: Path, out: Path | None) -> object:
    return pack(folder, out=out)


def pack_block_folder(folder: Path, out: Path | None) -> object:
    return block_packer.pack_block(folder, out=out, cases=False)


def pack_publishing_folder(folder: Path, out: Path | None) -> object:
    return pack_studies(folder, out=out)


PACKERS = {
    "bundle": Packer("bundle", lambda folder, out: pack(folder, out=out), _bundle, SHARED),
    "study": Packer("study", pack_study_folder, _studies, SHARED),
    "block": Packer("block", pack_block_folder, block_packer, SHARED),
    "studies": Packer("studies", pack_publishing_folder, _studies, ("inside_problem", "collision_refusals")),
}


def codes_and_text(packer: Packer, folder: Path, out: Path | None = None) -> tuple[list[str], str]:
    try:
        packer.run(folder, out)
    except MimeError as refused:
        return [each.code for each in refused.refusals], "\n".join(each.message for each in refused.refusals)
    raise AssertionError(f"{packer.name}: not refused")


@pytest.fixture(scope="module")
def written_study(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_study(tmp_path_factory.mktemp("study") / "wall_scan")


@pytest.fixture(scope="module")
def written_saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("saved"))


def block_folder(root: Path) -> Path:
    try:
        importlib.import_module("mime_lab_fixture")
    except ImportError:
        if os.environ.get("CI"):
            pytest.fail("the lab fixture package is not installed; every CI lane installs it from ./lab_fixture")
        pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")
    return Path(shutil.copytree(ROOT / "lab_fixture" / "blocks" / "nudged_probe", root / "nudged_probe"))


@pytest.fixture
def folders(written_saved: Path, written_study: Path, tmp_path: Path) -> dict[str, Path]:
    """A clean folder for each packer, each a copy of its own to plant a failure in."""
    publishing = tmp_path / "studies"
    publishing.mkdir()
    copy_of(written_study, publishing / "wall_a")
    copy_of(written_study, publishing / "wall_b")
    return {"bundle": copy_of(written_saved, tmp_path / "saved" / "full_toy"),
            "study": copy_of(written_study, tmp_path / "study" / "wall_scan"),
            "block": block_folder(tmp_path), "studies": publishing}


def plant_stray(packer: Packer, folder: Path) -> None:
    """A file the kind's folder does not keep: at a study's top (``study.stray-file`` refuses that before the packer's
    own check) it would not reach the shared one, so a study's goes inside its saved experiment."""
    place = folder / "experiment" if packer.name == "study" else folder
    (place / ".env").write_bytes(b"SECRET=hunter2\n")


def plant_collision(packer: Packer, folder: Path) -> None:
    """Two files whose names are one on a case-insensitive filesystem (in a study, inside its saved experiment, as for a
    stray file); for a publishing folder, two study folders."""
    if packer.name == "studies":
        copy_of(folder / "wall_a", folder / "Wall_A")
        return
    place = folder / "experiment" if packer.name == "study" else folder
    for name in ("records/Notes.json", "records/notes.json"):
        (place / name).parent.mkdir(parents=True, exist_ok=True)
        (place / name).write_bytes(b"{}")


@pytest.mark.parametrize("name", ["bundle", "study", "block", "studies"])
def test_a_clean_folder_is_packed_by_every_packer(name: str, folders: dict[str, Path], tmp_path: Path):
    """The premise of the planted failures: nothing is refused until a failure is planted."""
    out = tmp_path / "out" if name == "studies" else tmp_path / "out" / f"{name}.container"
    assert PACKERS[name].run(folders[name], out) is not None


@pytest.mark.parametrize("name", ["bundle", "study", "block", "studies"])
def test_an_output_inside_the_folder_is_refused_by_every_packer(name: str, folders: dict[str, Path]):
    folder = folders[name]
    for out in (folder / "inside", folder):
        codes, _ = codes_and_text(PACKERS[name], folder, out)
        assert codes == ["bundle.output-inside"], (name, out)


@pytest.mark.parametrize("name", ["bundle", "study", "block"])
def test_a_file_the_folder_does_not_keep_is_refused_by_every_packer(name: str, folders: dict[str, Path]):
    plant_stray(PACKERS[name], folders[name])
    codes, text = codes_and_text(PACKERS[name], folders[name])
    assert "bundle.stray-file" in codes and ".env" in text, (name, codes)


@pytest.mark.parametrize("name", ["bundle", "study", "block", "studies"])
def test_two_names_that_are_one_are_refused_by_every_packer(name: str, folders: dict[str, Path]):
    plant_collision(PACKERS[name], folders[name])
    codes, text = codes_and_text(PACKERS[name], folders[name])
    assert "bundle.name-collision" in codes and "one name on a case-insensitive filesystem" in text, (name, codes)


@pytest.mark.parametrize("name", ["bundle", "study", "block"])
def test_a_file_over_the_limit_is_refused_by_every_packer(name: str, folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
    """The limit is each packer's own read of ``INLINE_LIMIT_BYTES`` (the tests of the tiers move it the same way)."""
    packer = PACKERS[name]
    monkeypatch.setattr(packer.module, "INLINE_LIMIT_BYTES", 10)
    codes, text = codes_and_text(packer, folders[name])
    assert "bundle.file-too-large" in codes and "over the 10 bytes a" in text, (name, codes, text)


def shared_check_stub(name: str) -> Callable[..., Any]:
    """A replacement for the shared check ``name`` that always refuses, with a message of its own."""
    from mime.saved._container import refuse

    refusal = refuse("bundle.stray-file", RefusalClass.USER, PLANTED, "planted `mime pack <folder>`")
    stubs: dict[str, Callable[..., Any]] = {
        "inside_problem": lambda *args, **kwargs: BuildRefused([refusal]),
        "stray_refusals": lambda *args, **kwargs: [refusal],
        "collision_refusals": lambda *args, **kwargs: [refusal],
        "over_limit": lambda *args, **kwargs: PLANTED,
    }
    return stubs[name]


@pytest.mark.parametrize(("name", "check"), [(name, check) for name, packer in PACKERS.items() for check in packer.uses])
def test_each_packer_runs_each_shared_check_it_names(name: str, check: str, folders: dict[str, Path],
                                                     monkeypatch: pytest.MonkeyPatch):
    packer = PACKERS[name]
    assert getattr(packer.module, check) is getattr(_container, check), f"{name} holds its own {check}"
    monkeypatch.setattr(packer.module, check, shared_check_stub(check))
    _, text = codes_and_text(packer, folders[name])
    assert PLANTED in text, f"{name} made its judgement without {check}"


def literal_copies(source: str) -> list[str]:
    """The codes of the shared checks that ``source`` writes: a refusal of one is built in ``mime.saved._container``
    alone."""
    held = {"bundle.output-inside", "bundle.name-collision"}
    return sorted({node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant)
                   and isinstance(node.value, str) and node.value in held})


def test_no_packer_writes_a_shared_checks_code_again():
    """``bundle.output-inside`` and ``bundle.name-collision`` are refused in one place. (``bundle.stray-file`` and
    ``bundle.file-too-large`` are made where the action is, and a packer's own judgements, of a case's files and of what
    only its kind keeps, make them too.)"""
    found = {}
    for path in sorted(SOURCE.rglob("*.py")):
        if path == Path(_container.__file__).resolve():
            continue
        copies = literal_copies(path.read_text(encoding="utf-8"))
        if copies:
            found[path.relative_to(SOURCE).as_posix()] = copies
    assert found == {}, found


def test_the_scan_finds_a_planted_copy():
    planted = 'refuse("bundle.output-inside", RefusalClass.USER, "inside", "outside")'
    assert literal_copies(planted) == ["bundle.output-inside"]
    assert literal_copies('x = "bundle.stray-file"') == []
