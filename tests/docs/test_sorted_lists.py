"""The two lists that grow with almost every change, ``tests/duration_allowlist.txt`` and
``docs/ratchets/raw_switches.txt``, are kept sorted and without a duplicate, so that two changes that each add a line
add them in different places and a branch taking the release in has no conflict to resolve (both used to be appended to
at the end: 10 and 6 of the last 45 pull requests conflicted on them). Their readers take them as sets, so the order
carries nothing; this check is what keeps a line added at the end from coming back.

A line out of order fails naming the line and the place it belongs; a duplicate names both lines. The controls plant
each fault in a copy of the real list and show the check naming it, and a list read in a shuffled order failing, every
run (there is no draw: the plants are fixed)."""

from bisect import bisect_right
from collections.abc import Callable
from pathlib import Path

import pytest
from checkout import ROOT

DURATIONS = Path("tests/duration_allowlist.txt")
SWITCHES = Path("docs/ratchets/raw_switches.txt")


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


def test_the_duration_allowlist_is_sorted_by_node_id_without_a_duplicate():
    text = (ROOT / DURATIONS).read_text(encoding="utf-8")
    assert items(text) and order_problems(str(DURATIONS), text, duration_key) == []


def test_the_raw_switches_list_is_sorted_by_site_without_a_duplicate():
    text = (ROOT / SWITCHES).read_text(encoding="utf-8")
    assert items(text) and order_problems(str(SWITCHES), text, switch_key) == []


# -- the controls: each fault planted in a copy of the real list is named ------------------------------------------

#: Each list's path and key, and a line that sorts before every line of the real list (a plant that is no copy of a
#: real line, so it is no duplicate).
LISTS = {"durations": (DURATIONS, duration_key, "a_planted.py::test_it # kept: a plant"),
         "switches": (SWITCHES, switch_key, "a_planted.py f where 1  # a plant")}


def real(name: str) -> tuple[str, list[str], Callable[[str], tuple]]:
    path, key, _ = LISTS[name]
    return (str(path), (ROOT / path).read_text(encoding="utf-8").splitlines(), key)


@pytest.mark.parametrize("name", LISTS)
def test_control_a_line_added_at_the_end_that_sorts_earlier_is_named_with_its_place(name):
    where, lines, key = real(name)
    moved = next(number for number, line in items("\n".join(lines)) if number > 40)         # a line from the front
    taken = lines[moved - 1]
    assert order_problems(where, "\n".join(lines), key) == []
    planted = [line for number, line in enumerate(lines, start=1) if number != moved] + [taken]
    problems = order_problems(where, "\n".join(planted), key)
    assert problems == [f"{where}:{len(planted)}: sorts before the line above it; move it up to just above line "
                        f"{moved}"]


@pytest.mark.parametrize("name", LISTS)
def test_control_a_line_that_sorts_before_all_planted_in_the_middle_is_named_at_the_middle(name):
    where, lines, key = real(name)
    first_number = items("\n".join(lines))[0][0]
    middle = next(number for number, _ in items("\n".join(lines)) if number > len(lines) // 2)
    planted = [*lines[:middle - 1], LISTS[name][2], *lines[middle - 1:]]
    assert order_problems(where, "\n".join(planted), key) == [
        f"{where}:{middle}: sorts before the line above it; move it up to just above line {first_number}"]


@pytest.mark.parametrize("name", LISTS)
def test_control_a_duplicate_is_named_with_both_lines(name):
    where, lines, key = real(name)
    number, first = items("\n".join(lines))[100]
    planted = [*lines[:number], first, *lines[number:]]
    assert order_problems(where, "\n".join(planted), key) == [
        f"{where}:{number + 1}: has the key of line {number}; delete one of the two"]


@pytest.mark.parametrize("name", LISTS)
def test_control_the_real_list_read_in_reverse_fails_on_every_line_but_the_first(name):
    where, lines, key = real(name)
    header = [line for line in lines if line.startswith("#")]
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    problems = order_problems(where, "\n".join([*header, *reversed(body)]), key)
    assert len(problems) == len(body) - 1 and all("sorts before the line above it" in p for p in problems)


def test_control_a_line_the_switch_key_cannot_read_is_named():
    where = str(SWITCHES)
    assert order_problems(where, "core/_a.py f where 1  # a\nbroken\n", switch_key) == [
        f"{where}:2: 'broken' does not start as the list's lines do"]


def test_the_occurrence_sorts_as_a_number():
    assert order_problems("x", "a.py f sqrt 2  # r\na.py f sqrt 10  # r\n", switch_key) == []
    assert order_problems("x", "a.py f sqrt 10  # r\na.py f sqrt 2  # r\n", switch_key) == [
        "x:2: sorts before the line above it; move it up to just above line 1"]
