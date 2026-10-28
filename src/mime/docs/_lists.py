"""The ratchets, kept as lists: one committed file per ratchet, ``docs/ratchets/<name>.txt``, one item per line as
``item  # reason``. No count is stored: a list's length is derived from it and printed.

Every list holds the items a check would otherwise fail on (a code no test names, a rung no case builds), so for every
list a hit not on it fails, and a listed item that is no longer a hit fails, saying which line to delete: growth is a
line added with its reason, and paying down is forced. A line with no reason, a duplicate, a malformed item and a bare
count each fail. A floor runs the other way: it lists what must stay present, and fails on a listed item that is
missing."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

#: Where an item's reason starts: a ``#`` after whitespace, followed by a space or the end of the line. A ``#`` with
#: anything else after it (``#2``) belongs to the item.
_REASON = re.compile(r"\s#(?: |$)")
#: A line that stores a count rather than an item: a bare number, or a word set to one (``count: 12``, ``n = 3``).
_BARE_COUNT = re.compile(r"\d+|[A-Za-z_ ]+\s*[:=]\s*\d+")


@dataclass(frozen=True)
class Line:
    number: int
    item: str
    reason: str


@dataclass(frozen=True)
class RatchetList:
    """A list as read: its items with their line numbers and reasons, and every way its lines are malformed."""

    path: Path
    lines: tuple[Line, ...]
    problems: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.path.stem

    def items(self) -> set[str]:
        return {line.item for line in self.lines}

    def __len__(self) -> int:
        return len(self.lines)


def read_list(path: Path, item_problem: Callable[[str], str | None] = lambda item: None) -> RatchetList:
    """The list at ``path``. A line starting with ``#`` is a comment and a blank line is nothing; every other line is an
    item and its reason, the item's whitespace collapsed to single spaces. ``item_problem`` says what is wrong with
    an item's form, or ``None``. A missing file is a problem, not an empty list."""
    if not path.is_file():
        return RatchetList(path, (), (f"{path}: the list does not exist; create it, empty if nothing is listed",))
    return list_from_text(path, path.read_text(encoding="utf-8"), item_problem)


def list_from_text(path: Path, content: str,
                   item_problem: Callable[[str], str | None] = lambda item: None) -> RatchetList:
    """The list whose text is ``content``, read as :func:`read_list` reads a file and placed at ``path``: a list as
    another revision holds it."""
    lines: list[Line] = []
    problems: list[str] = []
    seen: dict[str, int] = {}
    for number, text in enumerate(content.splitlines(), start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        where = f"{path}:{number}"
        found = _REASON.search(text)
        item = " ".join((text[:found.start()] if found else text).split())
        reason = text[found.end():].strip() if found else ""
        if _BARE_COUNT.fullmatch(item):
            problems.append(f"{where}: {item!r} is a count, and a list stores no count: list the items themselves")
            continue
        if not reason:
            problems.append(f"{where}: {item!r} has no reason; write it as '{item}  # <reason>'")
        form = item_problem(item)
        if form is not None:
            problems.append(f"{where}: {item!r} is malformed: {form}")
        if item in seen:
            problems.append(f"{where}: {item!r} is listed twice (first at line {seen[item]}); delete one")
            continue
        seen[item] = number
        lines.append(Line(number, item, reason))
    return RatchetList(path, tuple(lines), tuple(problems))


def list_problems(held: RatchetList, hits: Collection[str], what: str) -> list[str]:
    """Every failure of a list against the hits its check computed: a hit not listed (``what`` says what a hit is), a
    listed item that is no longer a hit, and the list's own malformed lines."""
    found = list(held.problems)
    listed = held.items()
    for item in sorted(set(hits) - listed):
        found.append(f"{item}: {what}, and it is not on {held.path}; fix it, or add the line '{item}  # <reason>'")
    for line in held.lines:
        if line.item not in hits:
            found.append(f"{held.path}:{line.number}: {line.item!r} now passes; delete this line")
    return found


def floor_problems(held: RatchetList, present: Collection[str], what: str) -> list[str]:
    """Every failure of a floor: a listed item that is no longer present (``what`` says what an item is), and the
    list's own malformed lines. An item present and not listed is not a failure here."""
    found = list(held.problems)
    for line in held.lines:
        if line.item not in present:
            found.append(f"{held.path}:{line.number}: the pinned {what} {line.item!r} is missing; restore it, or "
                         f"delete the line with its reason logged")
    return found


def base_list(root: Path, base: str, path: Path,
              item_problem: Callable[[str], str | None] = lambda item: None) -> tuple[RatchetList | None, str | None]:
    """The list ``path`` (relative to the repository at ``root``) as the revision ``base`` holds it (``None`` when that
    revision has none), or why the revision cannot be read."""
    try:
        known = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
                               capture_output=True, text=True, check=False)
        shown = subprocess.run(["git", "-C", str(root), "show", f"{base}:./{path.as_posix()}"],
                               capture_output=True, text=True, check=False)
    except OSError as error:
        return None, f"the base revision {base!r} cannot be read: {error}"
    if known.returncode != 0:
        return None, (f"the base revision {base!r} cannot be read in {root} ({shown.stderr.strip()}); fetch it, or "
                      f"pass another --base")
    if shown.returncode != 0:
        return None, None
    return list_from_text(Path(f"{base}:{path.as_posix()}"), shown.stdout, item_problem), None
