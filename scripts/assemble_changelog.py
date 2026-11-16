#!/usr/bin/env python
"""Fold the changelog fragments of ``changes.d/`` into ``CHANGELOG.md``'s Unreleased block. Standard library only.

A change writes ``changes.d/<slug>.<kind>.md`` instead of editing ``CHANGELOG.md``, so two changes never touch the same
line. The file holds the entry exactly as the changelog should read it (a ``- `` bullet and its continuation lines). The
kind names the section of the Unreleased block: ``added``, ``changed``, ``deprecated``, ``removed``, ``fixed``,
``security`` or ``anomaly`` (Known Anomalies).

    python scripts/assemble_changelog.py             # fold, delete the folded fragments
    python scripts/assemble_changelog.py --dry-run   # print the changelog as it would read, change nothing
    python scripts/assemble_changelog.py --check     # exit 1 if any fragment is waiting

Entries land at the end of their section, in the order of kind and then slug; a section the block lacks is made in its
place among the others. Nothing else in the file is touched, so a hand edit anywhere survives, and a second fold finds
nothing to do. The file's line endings are kept byte for byte. Refused (exit 2) before anything is written: a name
that is not ``<slug>.<kind>.md`` with a known kind; a fragment that is not a bullet, that holds a heading line or an
odd number of code-fence lines (either would change how the next fold reads the file); a changelog without exactly one
``## [Unreleased]`` heading, with unbalanced code fences, or with mixed line endings.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: The kind of a fragment and the section of the Unreleased block it lands in, in the order the sections are kept.
SECTIONS = {"added": "Added", "changed": "Changed", "deprecated": "Deprecated", "removed": "Removed", "fixed": "Fixed",
            "security": "Security", "anomaly": "Known Anomalies"}
NAME = re.compile(r"^(?P<slug>[a-z0-9][a-z0-9_-]*)\.(?P<kind>[a-z]+)\.md$")
#: Files of the directory that are not fragments.
NOT_FRAGMENTS = {"README.md"}
UNRELEASED = "## [Unreleased]"
#: A line a Markdown reader takes as a heading (up to three spaces of indent); the fold's own parser reads column 0.
HEADING = re.compile(r"^ {0,3}#")


class Refused(Exception):
    """A fragment or a changelog the fold cannot use; the message names it."""


def read_fragments(directory: Path) -> list[tuple[str, str, str]]:
    """``(kind, slug, entry)`` for each fragment, in the fold's order: by the order of the kinds, then by slug."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if path.name in NOT_FRAGMENTS or path.name.startswith(".") or not path.is_file():
            continue
        match = NAME.match(path.name)
        if match is None or match["kind"] not in SECTIONS:
            raise Refused(f"{path.name}: a fragment is named <slug>.<kind>.md, the slug lower-case letters, digits, "
                          f"'-' and '_', the kind one of {', '.join(SECTIONS)}")
        entry = path.read_text(encoding="utf-8").strip("\n")
        if not entry.startswith("- "):
            raise Refused(f"{path.name}: a fragment holds the entry as the changelog reads it, so it starts with '- '")
        lines = entry.split("\n")
        if heading := next((line for line in lines if HEADING.match(line)), None):
            raise Refused(f"{path.name}: a fragment holds no heading line, and this one has {heading!r}")
        if sum(_is_fence(line) for line in lines) % 2:
            raise Refused(f"{path.name}: a fragment's code fences are balanced, and this one's are not")
        found.append((match["kind"], match["slug"], entry))
    return sorted(found, key=lambda held: (list(SECTIONS).index(held[0]), held[1]))


def _is_fence(line: str) -> bool:
    return line.lstrip(" ").startswith(("```", "~~~"))


def _heading_lines(lines: list[str]) -> list[int]:
    """The indices of the lines that are headings, outside code fences."""
    fenced = False
    found: list[int] = []
    for index, line in enumerate(lines):
        if _is_fence(line):
            fenced = not fenced
        elif not fenced and (line.startswith("## ") or line.startswith("### ")):
            found.append(index)
    return found


def fold(text: str, fragments: list[tuple[str, str, str]]) -> str:
    """``text`` with each fragment's entry at the end of its section of the Unreleased block."""
    lines = text.split("\n")
    headings = _heading_lines(lines)
    starts = [index for index in headings if lines[index].startswith(UNRELEASED)]
    if len(starts) != 1:
        raise Refused(f"CHANGELOG.md has {len(starts)} '{UNRELEASED}' headings, not one")
    if sum(_is_fence(line) for line in lines) % 2:
        raise Refused("CHANGELOG.md's code fences are not balanced")
    if not fragments:
        return text
    after = [index for index in headings if index > starts[0] and lines[index].startswith("## ")]
    end = after[0] if after else len(lines)
    for kind in SECTIONS:
        entries = [entry for held, _, entry in fragments if held == kind]
        if not entries:
            continue
        block = [index for index in _heading_lines(lines) if starts[0] < index < end and lines[index].startswith("### ")]
        heading = f"### {SECTIONS[kind]}"
        own = [index for index in block if lines[index] == heading]
        if own:
            following = [index for index in block if index > own[0]]
            stop = following[0] if following else end
            while lines[stop - 1] == "":
                stop -= 1
            added = [line for entry in entries for line in entry.split("\n")]
            lines[stop:stop] = added
            end += len(added)
        else:
            later = [index for index in block if lines[index][4:] in
                     [SECTIONS[k] for k in list(SECTIONS)[list(SECTIONS).index(kind) + 1:]]]
            where = later[0] if later else end
            made = [heading, "", *[line for entry in entries for line in entry.split("\n")], ""]
            if where == end:  # the block's last section: a blank line before it, the next heading's own after it
                while lines[where - 1] == "":
                    where -= 1
                made = ["", *made[:-1]]
                lines[where:where] = made
                end += len(made)
                if end < len(lines) and lines[end - 1] != "":  # no blank line was there: keep one
                    lines.insert(end, "")
                    end += 1
            else:
                lines[where:where] = made
                end += len(made)
    return "\n".join(lines)


def fold_bytes(raw: bytes, fragments: list[tuple[str, str, str]]) -> bytes:
    """``fold`` of a file's bytes, its line endings (all ``\\n`` or all ``\\r\\n``) kept."""
    text = raw.decode("utf-8")
    crlf = "\r\n" in text
    if crlf and "\n" in text.replace("\r\n", ""):
        raise Refused("CHANGELOG.md mixes \\r\\n and \\n line endings")
    folded = fold(text.replace("\r\n", "\n"), fragments)
    return (folded.replace("\n", "\r\n") if crlf else folded).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if any fragment is waiting")
    parser.add_argument("--dry-run", action="store_true", help="print the changelog as it would read, change nothing")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout (default: this one)")
    arguments = parser.parse_args(argv)
    directory, changelog = arguments.root / "changes.d", arguments.root / "CHANGELOG.md"
    try:
        fragments = read_fragments(directory)
        if arguments.check:
            for kind, slug, _ in fragments:
                print(f"changes.d/{slug}.{kind}.md is waiting to be folded into CHANGELOG.md", file=sys.stderr)
            return 1 if fragments else 0
        folded = fold_bytes(changelog.read_bytes(), fragments)
    except Refused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    if arguments.dry_run:
        sys.stdout.flush()
        sys.stdout.buffer.write(folded)
        return 0
    if fragments:
        changelog.write_bytes(folded)
        for kind, slug, _ in fragments:
            (directory / f"{slug}.{kind}.md").unlink()
    print(f"folded {len(fragments)} fragment(s) into CHANGELOG.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
