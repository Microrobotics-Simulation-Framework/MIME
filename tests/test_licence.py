"""MIME is Apache-2.0 from the 0.3 line on. The 0.2 line's licence is named in two places only: NOTICE's history
note and the changelog. Every other tracked file is searched for it, and the search has a control: a planted mention
outside those two places is found."""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: The SHA-256 of the Apache License 2.0 text as apache.org publishes it (LICENSE-2.0.txt).
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
#: The two files allowed to name the earlier licence.
HISTORY = {"NOTICE", "CHANGELOG.md"}
#: Spelt in pieces, so that this file, which the search reads too, holds no mention of its own.
EARLIER_ID, EARLIER_NAME = "G" + "PL", "Lesser " + "General Public"
EARLIER_LICENCE = re.compile(rf"\bL?{EARLIER_ID}|{EARLIER_NAME}", re.IGNORECASE)


def tracked_files(root: Path) -> list[str]:
    git = shutil.which("git")
    if git is None or not (root / ".git").exists():
        pytest.skip("not a git checkout: the tracked files are unknown")
    done = subprocess.run([git, "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [name for name in done.stdout.decode().split("\0") if name]


def mentions_of_the_earlier_licence(root: Path, files: list[str]) -> list[str]:
    found: list[str] = []
    for name in files:
        path = root / name
        if name in HISTORY or not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        found += [f"{name}:{number}: {line.strip()}" for number, line in enumerate(text.splitlines(), 1)
                  if EARLIER_LICENCE.search(line)]
    return found


def test_the_licence_is_the_apache_2_text_exactly():
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == APACHE_2_SHA256


def test_notice_names_the_copyright_and_the_0_2_lines_licence():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Copyright" in notice and "Nicholas Ehsan Roy" in notice
    assert "Apache License, Version 2.0" in notice and "pre-0.3" in notice and EARLIER_LICENCE.search(notice)


def test_no_tracked_file_but_notice_and_the_changelog_names_the_earlier_licence():
    assert mentions_of_the_earlier_licence(ROOT, tracked_files(ROOT)) == []


def test_a_planted_mention_outside_notice_and_the_changelog_is_found(tmp_path):
    (tmp_path / "NOTICE").write_text(f"Releases of the 0.2 line were distributed under the L{EARLIER_ID}.\n")
    (tmp_path / "CHANGELOG.md").write_text(f"- moved from L{EARLIER_ID}-3.0-or-later\n")
    (tmp_path / "README.md").write_text(f"A line.\nLicensed under the GNU {EARLIER_NAME} License.\n")
    (tmp_path / "pyproject.toml").write_text(f'license = "L{EARLIER_ID}-3.0-or-later"\n')
    (tmp_path / "COPYING").write_text(f"{EARLIER_ID.lower()}v3\n")
    found = mentions_of_the_earlier_licence(tmp_path, ["NOTICE", "CHANGELOG.md", "README.md", "pyproject.toml",
                                                       "COPYING"])
    assert found == [f"README.md:2: Licensed under the GNU {EARLIER_NAME} License.",
                     f'pyproject.toml:1: license = "L{EARLIER_ID}-3.0-or-later"', f"COPYING:1: {EARLIER_ID.lower()}v3"]
