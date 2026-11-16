"""The order of a list kept sorted: the key of a line of each list that is, and the check that names a line out of
order, a duplicate and a line the key cannot read. Used by ``test_sorted_lists.py`` and ``test_refusal_floor.py``."""

from bisect import bisect_right
from collections.abc import Callable


def duration_key(line: str) -> tuple[str, ...]:
    """A node id, which the line starts with, up to the `` # `` before its reason."""
    return (line.partition(" # ")[0].strip(),)


def switch_key(line: str) -> tuple[str, str, str, int]:
    """``file function family occurrence``, the first four words of a line, the occurrence as a number (``sqrt 2``
    before ``sqrt 10``); a line that does not start so raises ``ValueError``."""
    file, function, family, occurrence = line.split()[:4]
    return file, function, family, int(occurrence)


def items(text: str) -> list[tuple[int, str]]:
    """``(line number, line)`` for every line that is no comment and no blank."""
    return [(number, line) for number, line in enumerate(text.splitlines(), start=1)
            if line.strip() and not line.startswith("#")]


def order_problems(name: str, text: str, key: Callable[[str], tuple]) -> list[str]:
    """What is wrong with the order of the list ``text`` (named ``name``): a line that sorts before the one above it
    (the place it belongs is named), a line whose key an earlier line has, a line the key cannot read."""
    found: list[str] = []
    kept: list[tuple[tuple, int]] = []          # the keys in order, and each one's line number, of the lines in place
    for number, line in items(text):
        try:
            here = key(line)
        except ValueError:
            found.append(f"{name}:{number}: {line[:60]!r} does not start as the list's lines do")
            continue
        twin = next((n for k, n in kept if k == here), None)
        if twin is not None:
            found.append(f"{name}:{number}: has the key of line {twin}; delete one of the two")
        elif kept and here < kept[-1][0]:
            place = kept[bisect_right([k for k, _ in kept], here)][1]
            found.append(f"{name}:{number}: sorts before the line above it; move it up to just above line {place}")
        else:
            kept.append((here, number))
    return found


def code_key(line: str) -> tuple[str, ...]:
    """A refusal code, which a line of the floor is, and which a line of the removals starts with."""
    return (line.partition(" # ")[0].strip(),)


def entry_key(line: str) -> tuple[str, ...]:
    """An entry's kind, name and version, which a line of the entries floor is and a line of its removals starts with."""
    return tuple(line.partition(" # ")[0].split())
