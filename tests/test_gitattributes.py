"""Every tracked file keeps the bytes the repository holds, on a checkout on any platform.

Git on Windows ships with ``core.autocrlf=true``, which writes a text file out with CRLF line ends; a file that code or
a test pins by sha256 (the packaged records and tables, the saved-form fixtures and goldens, the reference arrays) then
hashes differently and is refused. ``.gitattributes`` says, for each tracked file, either ``eol: lf`` (a text file is
written with LF whatever the user's configuration) or ``text: unset`` (a binary is never touched). A file with
neither would be silently rewritten on a Windows checkout, so a new data folder cannot be left out.

The control: in a repository whose attributes leave a file out, the same check reports it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def uncovered(repository: Path, paths: list[str]) -> list[str]:
    """The ``paths`` that ``repository``'s attributes leave to the user's line-end configuration."""
    given = subprocess.run(["git", "-C", str(repository), "check-attr", "-z", "--stdin", "eol", "text"],
                           input="".join(path + "\0" for path in paths), capture_output=True, text=True, check=True)
    fields = given.stdout.split("\0")[:-1]
    held: dict[str, dict[str, str]] = {}
    for path, attribute, value in zip(fields[0::3], fields[1::3], fields[2::3], strict=True):
        held.setdefault(path, {})[attribute] = value
    return [path for path, attributes in held.items() if attributes["eol"] != "lf" and attributes["text"] != "unset"]


def tracked(repository: Path) -> list[str]:
    done = subprocess.run(["git", "-C", str(repository), "ls-files", "-z"], capture_output=True, text=True, check=True)
    return done.stdout.split("\0")[:-1]


@pytest.fixture(scope="module")
def git_checkout() -> Path:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    return ROOT


def test_every_tracked_file_keeps_its_bytes_on_a_windows_checkout(git_checkout: Path):
    files = tracked(git_checkout)
    assert ".gitattributes" in files and "src/mime/data/records/water.json" in files
    assert uncovered(git_checkout, files) == [], ".gitattributes leaves these to the user's core.autocrlf"


def test_a_file_the_rules_leave_out_is_reported(tmp_path: Path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitattributes").write_text("*.json text eol=lf\n*.npz binary\n", encoding="utf-8")
    (tmp_path / "record.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "weights.npz").write_bytes(b"\x00\x01")
    (tmp_path / "table.csv").write_text("a,b\n", encoding="utf-8")
    assert uncovered(tmp_path, ["record.json", "weights.npz", "table.csv"]) == ["table.csv"]
