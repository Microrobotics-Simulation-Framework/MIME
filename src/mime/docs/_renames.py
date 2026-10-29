"""The rename map: ``docs/ratchets/renames.txt``, one ``<kind> <old> -> <new>  # <date>, <log>`` line per registered
entry that was renamed. A ratchet that only shrinks compares the head's list with the base's, and a renamed line would
read as a new one; the map lets the comparison carry the base's line over to the new name instead.

A map line excuses a line only when all hold: the old name is registered in the base and the new name in the head,
the old name is registered in the head no longer, and the line's reason is unchanged. A line that fails any of these
excuses nothing, so the head's line is read as new and fails. (The old name is read from the base's sources, as a
registration's ``name = "<old>"``.) The map is also the start of the migration table: what each renamed name was, and
when."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from mime.docs._lists import Line, RatchetList

ARROW = " -> "
NAME = r"[a-z][a-z0-9_.]*"


@dataclass(frozen=True)
class Rename:
    kind: str
    old: str
    new: str
    number: int
    reason: str


def renames_of(held: RatchetList) -> list[Rename]:
    """The map's lines as renames (a malformed one is the list's own problem)."""
    found: list[Rename] = []
    for line in held.lines:
        match = re.fullmatch(rf"(\S+) ({NAME}){re.escape(ARROW)}({NAME})", line.item)
        if match:
            found.append(Rename(match.group(1), match.group(2), match.group(3), line.number, line.reason))
    return found


def registered_in_base(root: Path, base: str, old: str) -> bool:
    """Whether the base revision's sources register an entry named ``old`` (``name = "old"`` in ``src/mime``)."""
    done = subprocess.run(["git", "-C", str(root), "grep", "-q", "-F", "-e", f'name = "{old}"', base, "--", "src/mime"],
                          capture_output=True, text=True, check=False)
    return done.returncode == 0


def carried(item: str, renames: Collection[Rename]) -> str:
    """``item`` (``<kind> <name> <version> ...``) under the name an accepted rename gave its entry, or itself."""
    kind, _, rest = item.partition(" ")
    name, _, tail = rest.partition(" ")
    for rename in renames:
        if rename.kind == kind and rename.old == name:
            return f"{kind} {rename.new} {tail}".rstrip()
    return item


def rename_problems(root: Path, base: str | None, before: RatchetList | None, held: RatchetList, mapped: RatchetList,
                    head_entries: Collection[tuple[str, str]]) -> tuple[list[str], list[Rename], dict[str, Line]]:
    """The map's failures, the renames it accepts against ``base``, and the base's lines carried to their new names (new
    item to the base's line). A new name the head does not register, or an old name it still does, is a failure of the
    map; an old name the base does not register accepts nothing (and a line that relies on it fails as new), and a carried line whose reason changed is a failure."""
    problems = list(mapped.problems)
    accepted: list[Rename] = []
    for rename in renames_of(mapped):
        where = f"{mapped.path}:{rename.number}: {rename.kind} {rename.old}{ARROW}{rename.new}"
        if (rename.kind, rename.new) not in head_entries:
            problems.append(f"{where}: {rename.new!r} is not a {rename.kind} the head registers; fix the map, or "
                            f"delete the line")
        elif (rename.kind, rename.old) in head_entries:
            problems.append(f"{where}: {rename.old!r} is still registered at the head, so it was not renamed; "
                            f"delete the line")
        elif base is not None and before is not None and registered_in_base(root, base, rename.old):
            accepted.append(rename)
    moved: dict[str, Line] = {}
    if before is not None:
        for line in before.lines:
            moved[carried(line.item, accepted)] = line
    for line in held.lines:
        was = moved.get(line.item)
        if was is not None and was.item != line.item and was.reason != line.reason:
            problems.append(f"{held.path}:{line.number}: {line.item!r} is the base's {was.item!r} renamed, and its "
                            f"reason changed from {was.reason!r} to {line.reason!r}; a renamed line keeps its reason")
    return problems, accepted, moved
