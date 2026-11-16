"""The refusal table's floor: no code leaves the table unrecorded, with no total for two changes to both edit.

The table was held by a pinned count (686 codes, in three places), which every change adding a code had to move, so any
two such changes conflicted, and a count says nothing about which code went. The table is already held to the source by
the set difference both ways (``test_refusal_table.py``: a row for every code constructed, none for a code that is not).
What that cannot see is a code and its construction deleted together. ``refusal_floor.txt`` lists the table's codes as
they were when the pin was dropped, never edited; a code on it that the table no longer holds fails unless
``refusal_removed.txt`` logs it with a reason, and a code logged there that the table holds again fails. A code added
later needs no line anywhere, and a change that adds one edits only the table and the source.

The controls plant a silently removed code, a removal that is logged, a logged code that is back, a log line with no
reason, and a floor line out of order, each against the real files, and each fails or passes by name on every run (there
is no draw)."""

from collections.abc import Collection
from pathlib import Path

import pytest
from checkout import ROOT
from sorted_lists import code_key, order_problems

from mime.build import REFUSAL_TABLE, read_table
from mime.docs import RatchetList
from mime.docs._lists import list_from_text, read_list

FLOOR = Path(__file__).parent / "refusal_floor.txt"
REMOVED = Path(__file__).parent / "refusal_removed.txt"
TABLE = ROOT / "src/mime/build/refusal_table.txt"


def floor_codes(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def removal_problems(table: Collection[str], floor: Collection[str], removed: RatchetList) -> list[str]:
    """Why the table does not hold the floor: a code of the floor that is not in the table and not logged as removed,
    a code logged as removed that is in the table, and what is malformed in the log."""
    found = list(removed.problems)
    found += [f"{removed.path}:{line.number}: {line.item!r} is logged as removed and the table holds it; delete this "
              f"line" for line in removed.lines if line.item in table]
    found += [f"{code}: is on the floor ({FLOOR.name}) and not in the refusal table; restore its row, or log the "
              f"removal as '{code}  # <reason>' in {REMOVED.name}" for code in sorted(set(floor) - set(table) - removed.items())]
    return found


def real_removed() -> RatchetList:
    return read_list(REMOVED)


def test_the_table_holds_every_code_of_its_floor_or_logs_its_removal():
    assert floor_codes(FLOOR.read_text(encoding="utf-8")) and removal_problems(
        REFUSAL_TABLE, floor_codes(FLOOR.read_text(encoding="utf-8")), real_removed()) == []


def test_the_floor_and_the_log_are_sorted_without_a_duplicate_and_hold_codes():
    assert order_problems(FLOOR.name, FLOOR.read_text(encoding="utf-8"), code_key) == []
    assert order_problems(REMOVED.name, REMOVED.read_text(encoding="utf-8"), code_key) == []
    codes = floor_codes(FLOOR.read_text(encoding="utf-8"))
    assert all(code.count(" ") == 0 and "." in code for code in codes)


# -- the controls ---------------------------------------------------------------------------------------------------

def table_without(code: str) -> set[str]:
    """The codes of the real table's file with the row of ``code`` deleted (the data planted, as a change deleting
    the code and its construction together leaves it)."""
    text = TABLE.read_text(encoding="utf-8")
    kept = [line for line in text.splitlines(keepends=True) if line.split()[:1] != [code]]
    assert len(kept) == len(text.splitlines()) - 1
    return set(read_table("".join(kept)))


FIRST_CODE = floor_codes(FLOOR.read_text(encoding="utf-8"))[0]


def logged_with_the_real_removals(planted: str) -> RatchetList:
    """The log the controls plant a line into: the planted lines, then the removals really logged (a control is about
    its one planted code, and a real removal is no problem of it)."""
    real = [line for line in REMOVED.read_text(encoding="utf-8").splitlines(keepends=True) if not line.startswith("#")]
    return list_from_text(REMOVED, planted + "".join(real))


def test_control_a_code_deleted_with_its_construction_and_not_logged_fails_naming_it():
    floor = floor_codes(FLOOR.read_text(encoding="utf-8"))
    assert removal_problems(table_without(FIRST_CODE), floor, real_removed()) == [
        f"{FIRST_CODE}: is on the floor ({FLOOR.name}) and not in the refusal table; restore its row, or log the "
        f"removal as '{FIRST_CODE}  # <reason>' in {REMOVED.name}"]


@pytest.mark.parametrize("code", [FIRST_CODE, floor_codes(FLOOR.read_text(encoding="utf-8"))[-1]],
                         ids=["the first code", "the last code"])
def test_control_a_logged_removal_passes(code):
    log = logged_with_the_real_removals(f"{code}  # removed with the planted feature\n")
    assert removal_problems(table_without(code), floor_codes(FLOOR.read_text(encoding="utf-8")), log) == []


def test_control_a_logged_code_the_table_holds_again_fails_naming_the_line():
    log = logged_with_the_real_removals(f"# header\n{FIRST_CODE}  # removed once\n")
    assert removal_problems(set(REFUSAL_TABLE), floor_codes(FLOOR.read_text(encoding="utf-8")), log) == [
        f"{REMOVED}:2: {FIRST_CODE!r} is logged as removed and the table holds it; delete this line"]


def test_control_a_log_line_with_no_reason_is_a_problem():
    log = logged_with_the_real_removals(f"{FIRST_CODE}\n")
    problems = removal_problems(table_without(FIRST_CODE), floor_codes(FLOOR.read_text(encoding="utf-8")), log)
    assert problems and f"{FIRST_CODE!r} has no reason" in problems[0]


def test_control_the_first_code_appended_at_the_end_of_the_floor_is_named_with_its_place():
    lines = FLOOR.read_text(encoding="utf-8").splitlines()
    first = next(number for number, line in enumerate(lines, start=1) if not line.startswith("#"))
    planted = "\n".join([*lines[:first - 1], *lines[first:], lines[first - 1]]) + "\n"
    assert order_problems(FLOOR.name, planted, code_key) == [
        f"{FLOOR.name}:{len(lines)}: sorts before the line above it; move it up to just above line {first}"]
