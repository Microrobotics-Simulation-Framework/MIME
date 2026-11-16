"""The fold of ``changes.d/`` fragments into ``CHANGELOG.md`` (``scripts/assemble_changelog.py``): each kind lands in its
section, the fold is stable and idempotent, nothing else in the file moves, and a fragment it cannot use is refused by
name before anything is written. The real changelog is folded in memory, so a fragment waiting in the tree is known to
fold."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_assemble_changelog", ROOT / "scripts" / "assemble_changelog.py")
assert spec is not None and spec.loader is not None
assemble = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = assemble
spec.loader.exec_module(assemble)

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- **An old addition** (a line).
  - A nested line.

### Changed

- **An old change.**

### Fixed

- **An old fix.**

### Known Anomalies

- `MIME-ANO-1`: an old anomaly.

## [0.3.0.dev1] - 2026-11-05

### Added

- **A released addition.**

### Changed

- **A released change.**
"""
KIND_SECTION = {"added": "Added", "changed": "Changed", "deprecated": "Deprecated", "removed": "Removed",
                "fixed": "Fixed", "security": "Security", "anomaly": "Known Anomalies"}


def checkout(tmp_path: Path, text: str = CHANGELOG, **fragments: str) -> Path:
    """A checkout holding ``text`` and a fragment per ``<slug>__<kind>=entry`` (``__`` standing for the dot)."""
    (tmp_path / "CHANGELOG.md").write_text(text, encoding="utf-8")
    (tmp_path / "changes.d").mkdir()
    (tmp_path / "changes.d" / "README.md").write_text("How to write a fragment.\n", encoding="utf-8")
    for name, entry in fragments.items():
        (tmp_path / "changes.d" / f"{name.replace('__', '.')}.md").write_text(entry, encoding="utf-8")
    return tmp_path


def sections_of_unreleased(text: str) -> dict[str, list[str]]:
    """Each ``###`` section of the Unreleased block mapped to its top-level bullets' first lines."""
    block = text.split("## [Unreleased]")[1].split("\n## [")[0]
    found: dict[str, list[str]] = {}
    section = ""
    for line in block.splitlines():
        if line.startswith("### "):
            section = line[4:]
            found[section] = []
        elif line.startswith("- "):
            found[section].append(line)
    return found


def fold_in(root: Path, *arguments: str) -> int:
    return assemble.main(["--root", str(root), *arguments])


def test_each_kind_lands_in_its_section_after_the_old_entries(tmp_path: Path):
    root = checkout(tmp_path, a__added="- **New addition.**\n", b__changed="- **New change.**\n",
                    c__fixed="- **New fix.**\n", d__anomaly="- `MIME-ANO-2`: a new anomaly.\n")
    assert fold_in(root) == 0
    found = sections_of_unreleased((root / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert found == {"Added": ["- **An old addition** (a line).", "- **New addition.**"],
                     "Changed": ["- **An old change.**", "- **New change.**"],
                     "Fixed": ["- **An old fix.**", "- **New fix.**"],
                     "Known Anomalies": ["- `MIME-ANO-1`: an old anomaly.", "- `MIME-ANO-2`: a new anomaly."]}
    assert [path.name for path in (root / "changes.d").iterdir()] == ["README.md"], "the folded ones are deleted"


def test_the_released_sections_are_not_touched(tmp_path: Path):
    root = checkout(tmp_path, a__added="- **New addition.**\n", b__changed="- **New change.**\n")
    fold_in(root)
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.split("## [0.3.0.dev1]")[1] == CHANGELOG.split("## [0.3.0.dev1]")[1]
    removed = text.replace("- **New addition.**\n", "").replace("- **New change.**\n", "")
    assert removed == CHANGELOG, "the fold only adds the entries' own lines"


def test_a_continuation_of_a_multi_line_entry_comes_with_it(tmp_path: Path):
    entry = "- **Two lines.** The first,\n  the second.\n  - A nested one.\n"
    root = checkout(tmp_path, a__fixed=entry)
    fold_in(root)
    assert "- **An old fix.**\n" + entry + "\n### Known Anomalies" in (root / "CHANGELOG.md").read_text(encoding="utf-8")


def test_entries_of_a_section_are_in_slug_order_whatever_the_directory_order(tmp_path: Path):
    root = checkout(tmp_path, zeta__added="- **Z.**\n", alpha__added="- **A.**\n", mid__added="- **M.**\n")
    fold_in(root)
    added = sections_of_unreleased((root / "CHANGELOG.md").read_text(encoding="utf-8"))["Added"]
    assert added[1:] == ["- **A.**", "- **M.**", "- **Z.**"]


def test_a_second_fold_changes_nothing(tmp_path: Path):
    root = checkout(tmp_path, a__added="- **New addition.**\n", d__anomaly="- `MIME-ANO-2`: new.\n")
    assert fold_in(root) == 0
    once = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert fold_in(root) == 0
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == once


def test_a_hand_edit_elsewhere_survives_the_fold_and_a_refold(tmp_path: Path):
    root = checkout(tmp_path, a__added="- **New addition.**\n")
    fold_in(root)
    path = root / "CHANGELOG.md"
    path.write_text(path.read_text(encoding="utf-8").replace("An old change.", "An old change, edited by hand.")
                    .replace("A released addition.", "A released addition, edited too."), encoding="utf-8")
    (root / "changes.d" / "b.fixed.md").write_text("- **A later fix.**\n", encoding="utf-8")
    fold_in(root)
    text = path.read_text(encoding="utf-8")
    assert "An old change, edited by hand." in text and "A released addition, edited too." in text
    assert sections_of_unreleased(text)["Fixed"] == ["- **An old fix.**", "- **A later fix.**"]


@pytest.mark.parametrize("kind", KIND_SECTION)
def test_a_section_the_block_lacks_is_made_in_its_place(tmp_path: Path, kind: str):
    sections = list(KIND_SECTION.values())
    others = [name for name in sections if name != KIND_SECTION[kind]]
    text = "# Changelog\n\n## [Unreleased]\n\n" + "".join(f"### {name}\n\n- **{name} entry.**\n\n" for name in others) \
        + "## [0.1.0] - 2026-01-01\n\n### Added\n\n- **Old.**\n"
    root = checkout(tmp_path, text)
    (root / "changes.d" / f"new.{kind}.md").write_text(f"- **New {kind}.**\n", encoding="utf-8")
    assert fold_in(root) == 0
    result = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    found = sections_of_unreleased(result)
    assert list(found) == sections, "the sections keep the changelog's order"
    assert found[KIND_SECTION[kind]] == [f"- **New {kind}.**"]
    assert result.split("## [0.1.0]")[1] == text.split("## [0.1.0]")[1]
    assert "\n\n\n" not in result
    assert fold_in(root) == 0 and (root / "CHANGELOG.md").read_text(encoding="utf-8") == result


def test_check_fails_while_a_fragment_waits_and_passes_after_the_fold(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = checkout(tmp_path, a__added="- **New addition.**\n")
    assert fold_in(root, "--check") == 1
    assert "changes.d/a.added.md is waiting" in capsys.readouterr().err
    assert (root / "changes.d" / "a.added.md").exists() and (root / "CHANGELOG.md").read_text() == CHANGELOG
    fold_in(root)
    assert fold_in(root, "--check") == 0


def test_dry_run_prints_the_result_and_changes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = checkout(tmp_path, a__added="- **New addition.**\n")
    assert fold_in(root, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert "- **New addition.**" in printed
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG
    assert (root / "changes.d" / "a.added.md").exists()
    fold_in(root)
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == printed


@pytest.mark.parametrize(("name", "message"), [
    ("no-kind.md", "no-kind.md: a fragment is named <slug>.<kind>.md"),
    ("slug.unknown.md", "slug.unknown.md: a fragment is named"),
    ("Upper.added.md", "Upper.added.md: a fragment is named"),
    ("slug.added.txt", "slug.added.txt: a fragment is named"),
    ("a.b.added.md", "a.b.added.md: a fragment is named"),
])
def test_a_malformed_fragment_name_is_refused_by_name_and_nothing_is_written(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, message: str):
    root = checkout(tmp_path, ok__added="- **Fine.**\n")
    (root / "changes.d" / name).write_text("- **Entry.**\n", encoding="utf-8")
    for arguments in ([], ["--check"], ["--dry-run"]):
        assert fold_in(root, *arguments) == 2
        assert message in capsys.readouterr().err
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG
    assert (root / "changes.d" / "ok.added.md").exists(), "a refused fold deletes no fragment"


@pytest.mark.parametrize("entry", ["", "A line that is not a bullet.\n", "\n\n"])
def test_a_fragment_that_is_not_a_bullet_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str):
    root = checkout(tmp_path, bad__fixed=entry)
    assert fold_in(root) == 2
    assert "bad.fixed.md: a fragment holds the entry as the changelog reads it" in capsys.readouterr().err
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG


def test_a_changelog_without_one_unreleased_block_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = checkout(tmp_path, "# Changelog\n\n## [0.1.0] - 2026-01-01\n", a__added="- **A.**\n")
    assert fold_in(root) == 2
    assert "0 '## [Unreleased]' headings" in capsys.readouterr().err


def test_no_directory_or_no_fragment_is_nothing_to_fold(tmp_path: Path):
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    assert fold_in(tmp_path, "--check") == 0 and fold_in(tmp_path) == 0
    assert (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG


def test_the_trees_own_fragments_are_named_well_and_fold_into_the_real_changelog():
    fragments = assemble.read_fragments(ROOT / "changes.d")
    real = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    folded = assemble.fold(real, fragments)
    assert all(entry in folded for _, _, entry in fragments)
    assert assemble.fold(folded, []) == folded
    assert assemble.fold(real, fragments).count("## [Unreleased]") == 1


def test_the_real_changelog_gets_each_kind_in_its_own_section_and_nothing_else_moves(tmp_path: Path):
    shutil.copy(ROOT / "CHANGELOG.md", tmp_path / "CHANGELOG.md")
    (tmp_path / "changes.d").mkdir()
    in_it = ("added", "changed", "fixed", "anomaly")  # the sections the real block holds, so no heading is made
    entries = {kind: f"- **A planted {kind} entry.**\n" for kind in in_it}
    for kind, entry in entries.items():
        (tmp_path / "changes.d" / f"planted.{kind}.md").write_text(entry, encoding="utf-8")
    original = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert fold_in(tmp_path) == 0
    folded = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    found = sections_of_unreleased(folded)
    for kind, section in ((kind, KIND_SECTION[kind]) for kind in in_it):
        assert found[section][-1] == f"- **A planted {kind} entry.**"
    for entry in entries.values():
        folded = folded.replace(entry, "", 1)
    assert folded == original


def test_the_kinds_of_keep_a_changelog_are_made_in_their_places_among_the_existing_sections(tmp_path: Path):
    root = checkout(tmp_path, a__security="- **S.**\n", b__removed="- **R.**\n", c__deprecated="- **D.**\n")
    assert fold_in(root) == 0
    found = sections_of_unreleased((root / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert list(found) == ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security", "Known Anomalies"]
    assert [found[name] for name in ("Deprecated", "Removed", "Security")] == [["- **D.**"], ["- **R.**"], ["- **S.**"]]


#: The reviewer's two inputs (a heading line; an unbalanced fence) and the variants the same rules refuse.
BAD_BODIES = {
    "a-heading-that-fakes-the-block": "- thing\n## [Unreleased]\n## 9.9 fake\n",
    "a-level-3-heading": "- thing\n### Added\n",
    "an-indented-heading": "- thing\n   # not a bullet\n",
    "an-unbalanced-fence": "- thing\n```\ncode\n",
    "an-unbalanced-indented-fence": "- thing\n  ```python\n  code\n  ```\n  ```\n",
}


@pytest.mark.parametrize("body", BAD_BODIES.values(), ids=list(BAD_BODIES))
def test_a_fragment_body_that_would_change_how_the_next_fold_reads_the_file_is_refused_and_nothing_is_written(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str):
    root = checkout(tmp_path, ok__added="- **Fine.**\n", bad__added=body)
    for arguments in ([], ["--dry-run"]):
        assert fold_in(root, *arguments) == 2
        assert "bad.added.md: a fragment" in capsys.readouterr().err
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG
    assert sorted(path.name for path in (root / "changes.d").iterdir()) == ["README.md", "bad.added.md", "ok.added.md"]


def test_a_balanced_fence_in_a_fragment_folds_and_leaves_the_file_foldable_again(tmp_path: Path):
    entry = "- **With code.**\n  ```python\n  x = 1\n  ```\n"
    root = checkout(tmp_path, a__fixed=entry)
    assert fold_in(root) == 0
    (root / "changes.d" / "b.fixed.md").write_text("- **After.**\n", encoding="utf-8")
    assert fold_in(root) == 0
    assert sections_of_unreleased((root / "CHANGELOG.md").read_text(encoding="utf-8"))["Fixed"][-1] == "- **After.**"


UNBALANCED = CHANGELOG.replace("- **An old fix.**", "- **An old fix.**\n\n```\nan unclosed fence")
DUPLICATED = CHANGELOG + "\n## [Unreleased]\n"


@pytest.mark.parametrize(("text", "message"), [
    (UNBALANCED, "code fences are not balanced"),
    (DUPLICATED, "2 '## [Unreleased]' headings"),
    (CHANGELOG.replace("## [Unreleased]", "## [Next]"), "0 '## [Unreleased]' headings"),
    (CHANGELOG.replace("\n", "\r\n", 3), "mixes"),
], ids=["an-unbalanced-fence", "two-unreleased-blocks", "none", "mixed-line-endings"])
def test_a_changelog_the_fold_cannot_read_is_refused_and_nothing_is_written(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, message: str):
    root = checkout(tmp_path, a__added="- **A.**\n")
    (root / "CHANGELOG.md").write_bytes(text.encode())
    assert fold_in(root) == 2
    assert message in capsys.readouterr().err
    assert (root / "CHANGELOG.md").read_bytes() == text.encode()
    assert (root / "changes.d" / "a.added.md").exists()


def test_crlf_line_endings_are_kept_byte_for_byte(tmp_path: Path):
    root = checkout(tmp_path, a__added="- **New.**\n  and more.\n", b__security="- **Sec.**\n")
    (root / "CHANGELOG.md").write_bytes(CHANGELOG.replace("\n", "\r\n").encode())
    assert fold_in(root) == 0
    raw = (root / "CHANGELOG.md").read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b""), "no bare line feed"
    lf = CHANGELOG.replace("\n", "\r\n").encode()
    assert raw.replace(b"- **New.**\r\n  and more.\r\n", b"").replace(b"### Security\r\n\r\n- **Sec.**\r\n\r\n", b"") == lf
    (tmp_path / "lf").mkdir()
    again = checkout(tmp_path / "lf", a__added="- **New.**\n  and more.\n", b__security="- **Sec.**\n")
    fold_in(again)
    assert raw.decode().replace("\r\n", "\n") == (again / "CHANGELOG.md").read_text(encoding="utf-8")
