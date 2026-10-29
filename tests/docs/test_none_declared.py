"""W7 as a list. An empty equations, assumptions or references field prints ``none declared`` and is counted
(``test_pages.py``); ``mime docs check`` also holds the fields of MIME's own to ``docs/ratchets/none_declared.txt``, as
it holds the README start list: today's cases are listed with the shared reason ``present at D1b``, a new empty field
fails until it has its own line and reason, a listed field that is now filled fails until its line is deleted, and,
against a base revision, a line giving the shared reason that the base's list does not hold fails.

The toys' and a lab's entries are outside the list, by item 5's namespace rules: their fields are counted in the build's
summary and never listed. The controls run the real command over a copy of the checkout; each plant is fixed, so each
fails on every run."""

import re

from checkout import ROOT, copy_checkout, docs_check, edit, failures
from test_docs_check import paged
from test_numbers import committed

from mime.docs._check import NONE_DECLARED_START, read_lists

LIST = "docs/ratchets/none_declared.txt"
#: A spring of MIME's whose references are empty (it prints ``none declared``), and one whose references are filled.
EMPTY, FILLED = "docsplant_mime_empty", "docsplant_mime_known"
EMPTY_ITEM, FILLED_ITEM = f"interaction {EMPTY} 0.1.0 references", f"interaction {FILLED} 0.1.0 references"


def listing(root, line):
    path = root / LIST
    path.write_text(path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
    return path


#: The lines added after the start, each a cut with its own reason, logged with the maintainer: the free-space wall
#: kind's references (phase 7h), a definition that rests on no work.
ADDED = {"wall_kind free_space 0.1.0 references"}


def test_the_start_list_holds_exactly_the_empty_fields_of_mimes_own_entries():
    held = read_lists(ROOT)["none_declared"]
    start = [line for line in held.lines if line.item not in ADDED]
    assert len(start) >= 5 and {line.reason for line in start} == {NONE_DECLARED_START}
    assert all(line.reason != NONE_DECLARED_START for line in held.lines if line.item in ADDED)
    assert all(item.endswith(" references") for item in held.items()), "no MIME node has NodeMeta yet"


def test_a_new_empty_field_of_mimes_fails_naming_it_until_it_has_its_own_line(tmp_path):
    root = paged(copy_checkout(tmp_path), EMPTY)
    (problem,) = failures(docs_check(root, plant="mime_empty"))
    assert problem == (f"{EMPTY_ITEM}: a field of MIME's prints 'none declared', and it is not on {root / LIST}; fix "
                       f"it, or add the line '{EMPTY_ITEM}  # <reason>'")
    listing(root, f"{EMPTY_ITEM}  # a plant, with its own reason")
    done = docs_check(root, plant="mime_empty")
    assert done.returncode == 0, done.stdout
    assert re.search(r"none declared in MIME's pages: 7 fields, 5 of the list's 7 lines at the start reason", done.stdout)


def test_a_listed_field_that_is_now_filled_fails_until_its_line_is_deleted(tmp_path):
    root = paged(copy_checkout(tmp_path), FILLED)
    path = listing(root, f"{FILLED_ITEM}  # a plant, filled")
    number = len(path.read_text(encoding="utf-8").splitlines())
    assert failures(docs_check(root, plant="mime_known")) == [
        f"{path}:{number}: {FILLED_ITEM!r} now passes; delete this line"]
    edit(path, f"{FILLED_ITEM}  # a plant, filled\n", "")
    assert docs_check(root, plant="mime_known").returncode == 0


def test_a_field_of_a_lab_or_a_toy_is_outside_the_list_and_counted(tmp_path):
    done = docs_check(copy_checkout(tmp_path), plant="lab_numbered")
    assert done.returncode == 0, done.stdout
    assert "none declared in MIME's pages: 6 fields" in done.stdout
    assert re.search(r"none declared (\d+)", done.stdout) and int(re.search(r"none declared (\d+)",
                                                                           done.stdout).group(1)) > 5


def test_a_new_line_with_the_shared_start_reason_fails_against_a_base_that_does_not_hold_it(tmp_path):
    root = committed(paged(copy_checkout(tmp_path), EMPTY))
    path = listing(root, f"{EMPTY_ITEM}  # {NONE_DECLARED_START}")
    number = len(path.read_text(encoding="utf-8").splitlines())
    assert failures(docs_check(root, plant="mime_empty", base="HEAD")) == [
        f"{path}:{number}: {EMPTY_ITEM!r} is not on the list at HEAD, and {NONE_DECLARED_START!r} is the reason of the "
        f"cases that stood when the list landed, which only shrinks; fix the empty field, or give the line its own "
        f"reason (logged)"]
    edit(path, f"{EMPTY_ITEM}  # {NONE_DECLARED_START}", f"{EMPTY_ITEM}  # its own reason")
    done = docs_check(root, plant="mime_empty", base="HEAD")
    assert done.returncode == 0, done.stdout
    assert re.search(r"compared with HEAD's, 6 lines there and 7 here", done.stdout)


def test_a_malformed_item_and_a_missing_list_fail(tmp_path):
    root = copy_checkout(tmp_path)
    listing(root, "node Bare something  # not a field")
    assert any("is malformed" in problem for problem in failures(docs_check(root)))
    (root / LIST).unlink()
    assert any("the list does not exist" in problem for problem in failures(docs_check(root)))
