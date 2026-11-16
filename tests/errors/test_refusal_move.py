"""The refusal classes live in ``mime._refusals`` so that ``mime.units``'s errors can be MIME's; the old module
``mime.build._refusals`` is a shim. What must not change by the move: every old import path gives the same object, a
refusal pickles and loads (in a fresh process, and from a pickle written under the old path), and nothing records the
classes' qualified path in a place that a reader would then find wrong."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import mime._refusals as home
import mime.build as build_package
import mime.build._refusals as shim
from mime.build import ArgumentRefused, InternalError, RunFailed, UnrecognisedError
from mime.errors._boundary import public
from mime.units import UnitError, m

ROOT = Path(__file__).resolve().parents[2]
NAMES = tuple(name for name in shim.__all__)


def test_every_old_import_path_gives_the_object_of_the_new_one():
    assert NAMES and set(NAMES) <= set(dir(home))
    for name in NAMES:
        assert getattr(shim, name) is getattr(home, name), name
    for name in ("ArgumentRefused", "ArgumentTypeRefused", "BuildRefused", "InternalError", "MimeError", "Refusal",
                 "RefusalClass", "RunFailed", "UnrecognisedError"):
        assert getattr(build_package, name) is getattr(home, name), name


def test_the_shim_holds_every_public_name_the_old_module_held():
    old_names = {"DOCS", "ERROR_PAGES", "GUIDE_PAGES", "ISSUES", "MIME_DEFECT", "SUBJECT_NAME", "TABLE_FILE",
                 "REFUSAL_TABLE", "ArgumentRefused", "ArgumentTypeRefused", "BuildRefused", "Explained", "InternalError",
                 "MimeError", "Refusal", "RefusalClass", "RunFailed", "Subject", "UncataloguedRefusal",
                 "UnrecognisedError", "catalogue_problem", "described", "docs_version", "entry_defect", "guide_page_of",
                 "page_of", "passed_through", "plain_possessive", "read_table", "standing_for", "_made",
                 "_rebuilt_plain"}
    assert old_names <= {name for name in dir(shim)}


def _standing_for_one_from_below() -> Exception:
    class Done(Exception):
        pass

    try:
        public(lambda: (_ for _ in ()).throw(KeyError("gone")))()
    except Exception as error:  # noqa: BLE001 - the error MIME's boundary makes
        return error
    raise AssertionError("the boundary raised nothing")


def _a_run_failed() -> RunFailed:
    refusal = home.Refusal("run.chunks", home.RefusalClass.USER, "the run failed", "run it again")
    return RunFailed((refusal,))


def _errors() -> list[Exception]:
    refusal = home.Refusal("units.label", home.RefusalClass.USER, "not a label", "write one")
    return [ArgumentRefused((refusal,)), UnitError("cannot add", dimension=m(1.0).dimension), _a_run_failed(),
            InternalError((home.Refusal("entry.defect", home.RefusalClass.INTERNAL, "a check", "report it"),))]


def _same(first: BaseException, second: BaseException) -> bool:
    return (type(first) is type(second) and str(first) == str(second)
            and getattr(first, "codes", lambda: None)() == getattr(second, "codes", lambda: None)())


def test_a_refusal_pickles_and_loads_as_the_same_class_with_the_same_words():
    for error in _errors():
        back = pickle.loads(pickle.dumps(error))  # noqa: S301 - the test's own object
        assert _same(error, back)
        assert pickle.dumps(error).count(b"mime._refusals") >= 1 or isinstance(error, UnitError)


def test_a_pickle_written_under_the_old_path_still_loads(monkeypatch: pytest.MonkeyPatch):
    for name in ("ArgumentRefused", "MimeError", "UnrecognisedError", "RunFailed", "Refusal", "RefusalClass"):
        monkeypatch.setattr(getattr(home, name), "__module__", "mime.build._refusals")
    monkeypatch.setattr(home._made, "__module__", "mime.build._refusals")
    monkeypatch.setattr(home._rebuilt_plain, "__module__", "mime.build._refusals")
    errors = [*_errors(), _standing_for_one_from_below()]
    written = [pickle.dumps(error) for error in errors]
    monkeypatch.undo()
    assert all(b"mime.build._refusals" in blob for blob, error in zip(written, errors)
               if not isinstance(error, UnitError)), "the control: the written bytes name the old path"
    for blob, error in zip(written, errors):
        back = pickle.loads(blob)  # noqa: S301 - the test's own bytes
        assert _same(error, back)


def test_a_pickle_loads_in_a_fresh_process_that_imported_nothing_of_mime():
    errors = [*_errors(), _standing_for_one_from_below()]
    blob = pickle.dumps(errors)
    program = ("import pickle, sys\nassert 'mime' not in sys.modules\n"
               "held = pickle.loads(sys.stdin.buffer.read())\n"
               "print([type(e).__name__ + ':' + str(e) for e in held])\n")
    ran = subprocess.run([sys.executable, "-I", "-c", program], input=blob, capture_output=True, check=False,
                         cwd=ROOT, timeout=120)
    assert ran.returncode == 0, ran.stderr.decode()
    assert ran.stdout.decode().strip() == str([type(e).__name__ + ":" + str(e) for e in errors])


def test_the_classes_report_the_new_module_and_the_old_path_is_the_shim_alone():
    assert {cls.__module__ for cls in (ArgumentRefused, UnrecognisedError, RunFailed, UnitError.__mro__[1])} == {
        "mime._refusals"}
    assert importlib.import_module("mime.build._refusals") is shim and shim.__name__ == "mime.build._refusals"


#: The files that may name the old path: the shim itself, the changelog that records the move, the guides that say the
#: shim exists, and this test.
_MAY_NAME_THE_OLD_PATH = {"src/mime/build/_refusals.py", "CHANGELOG.md", "src/mime/errors/README.md",
                          "src/mime/README.md",
                          "tests/docs/golden_numbers.json",  # a history of which files constructed codes, as measured then
                          "tests/errors/test_refusal_move.py"}
_OLD_PATH = "mime.build._refusals"


def _files_naming_the_old_path(root: Path, relatives: list[str]) -> list[str]:
    found: list[str] = []
    for relative in relatives:
        if relative in _MAY_NAME_THE_OLD_PATH or relative.endswith(".py"):
            continue
        path = root / relative
        if path.is_file() and _OLD_PATH.encode() in path.read_bytes():
            found.append(relative)
    return found


def test_no_recorded_text_names_the_old_qualified_path_of_a_class():
    """Saved forms, golden texts, docs and data record a class by its qualified path: none names the old one. The
    source's own imports are code (they work through the shim) and are not text recorded for a reader."""
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    assert len(tracked) > 500, "the control: the scan sees the repository's files"
    assert _files_naming_the_old_path(ROOT, tracked) == []


def test_the_control_the_scan_finds_a_planted_old_path(tmp_path: Path):
    (tmp_path / "golden.txt").write_text(f"raised {_OLD_PATH}.ArgumentRefused", encoding="utf-8")
    (tmp_path / "clean.txt").write_text("raised mime._refusals.ArgumentRefused", encoding="utf-8")
    assert _files_naming_the_old_path(tmp_path, ["golden.txt", "clean.txt"]) == ["golden.txt"]
