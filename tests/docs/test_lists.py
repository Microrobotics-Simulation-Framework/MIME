"""The ratchets kept as lists, and W4 on the first of them, the refusal codes no test names: a listed code a test now
names fails, naming the line to delete; a code no test names that is not listed fails; a line with no reason, a bare
count, a duplicate and a malformed line each fail. Each arm runs the real refusal check over a doctored copy of the
checkout. The list's length is derived from it and printed, never stored.

W4's extension to the ledger (``docs/unpaged.txt``) and the README start list (``docs/ratchets/readme_numbers.txt``):
a line with an empty reason, and a bare count, each fail through the real ``mime docs check``. Their other two arms
are the ledger's W2 and W3 (``test_walk.py``) and the start list's N1 arms (``test_numbers.py``)."""

import pytest
from checkout import ROOT, copy_checkout, docs_check, failures

from mime.docs import floor_problems, list_problems, read_list
from mime.docs._check import LISTS, list_path, read_lists, refusal_check

FIRST = read_list(list_path(ROOT, "refusal_codes"), LISTS["refusal_codes"]).lines[0]


def doctored(tmp_path, first: str | None = None):
    """A copy of the checkout, the refusal list's first item's line replaced by ``first`` when given."""
    root = copy_checkout(tmp_path)
    path = list_path(root, "refusal_codes")
    if first is not None:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[FIRST.number - 1] = f"{first}\n"
        path.write_text("".join(lines), encoding="utf-8")
    return root, path


def test_the_refusal_list_holds_every_code_no_test_names_and_prints_its_length():
    held = read_lists(ROOT)["refusal_codes"]
    report = refusal_check(ROOT)
    assert report.problems == []
    assert f"refusal table: 196 codes; {len(held)} that no test names" in report.summary
    assert len(held) > 0 and all(line.reason == "no test names it" for line in held.lines)


def test_w4_a_listed_code_a_test_now_names_fails_naming_the_line_to_delete(tmp_path):
    root, path = doctored(tmp_path)
    (root / "tests" / "test_planted_names.py").write_text(f'def test_it():\n    assert "{FIRST.item}"\n')
    assert refusal_check(root).problems == [f"{path}:{FIRST.number}: {FIRST.item!r} now passes; delete this line"]


def test_w4_a_code_no_test_names_and_not_listed_fails(tmp_path):
    root, path = doctored(tmp_path, f"planted  # {FIRST.reason}")
    problems = refusal_check(root).problems
    assert f"{FIRST.item}: a refusal code no test names, and it is not on {path}; fix it, or add the line " \
           f"'{FIRST.item}  # <reason>'" in problems
    assert f"{path}:{FIRST.number}: 'planted' is malformed: an item is a refusal code" in problems


def test_w4_a_line_with_an_empty_reason_fails(tmp_path):
    root, path = doctored(tmp_path, f"{FIRST.item}  #")
    assert refusal_check(root).problems == [f"{path}:{FIRST.number}: {FIRST.item!r} has no reason; write it as "
                                            f"'{FIRST.item}  # <reason>'"]


def test_w4_a_bare_count_fails(tmp_path):
    root, path = doctored(tmp_path)
    count = len(read_list(path).lines)
    path.write_text(path.read_text(encoding="utf-8") + f"{count}\n", encoding="utf-8")
    (problem,) = refusal_check(root).problems
    assert problem.endswith(f"'{count}' is a count, and a list stores no count: list the items themselves")


def test_w4_a_duplicated_line_fails(tmp_path):
    root, path = doctored(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + f"{FIRST.item}  # again\n", encoding="utf-8")
    (problem,) = refusal_check(root).problems
    assert f"{FIRST.item!r} is listed twice (first at line {FIRST.number}); delete one" in problem


def test_the_reader_keeps_a_hash_that_is_part_of_an_item_and_skips_comments(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("# a comment\n\nREADME.md  0.95  #2  # a reason # with a hash\nlast   item\t# its reason\n")
    held = read_list(path)
    assert [(line.item, line.reason) for line in held.lines] == [("README.md 0.95 #2", "a reason # with a hash"),
                                                                ("last item", "its reason")]
    assert held.problems == ()


def test_a_missing_list_is_a_failure_not_an_empty_list(tmp_path):
    (problem,) = read_list(tmp_path / "absent.txt").problems
    assert problem.endswith("the list does not exist; create it, empty if nothing is listed")


def test_a_floor_fails_on_a_listed_item_missing_and_not_on_one_present_and_unlisted(tmp_path):
    path = tmp_path / "floor.txt"
    path.write_text("Node x  # pinned at landing\n")
    held = read_list(path)
    assert floor_problems(held, {"Node x", "Node y"}, "symbol row") == []
    (problem,) = floor_problems(held, {"Node y"}, "symbol row")
    assert problem.endswith("the pinned symbol row 'Node x' is missing; restore it, or delete the line with its "
                            "reason logged")
    assert list_problems(held, {"Node x"}, "a hit") == []


def first_item(path) -> tuple[int, str]:
    first = read_list(path).lines[0]
    return first.number, first.item


@pytest.mark.parametrize("listed", ["docs/unpaged.txt", "docs/ratchets/readme_numbers.txt"],
                         ids=["the-ledger", "the-readme-start-list"])
def test_w4_a_line_with_an_empty_reason_fails_on_the_ledger_and_the_start_list(tmp_path, listed):
    root = copy_checkout(tmp_path)
    path = root / listed
    number, item = first_item(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[number - 1] = f"{item}  #\n"
    path.write_text("".join(lines), encoding="utf-8")
    assert failures(docs_check(root)) == [f"{path}:{number}: {item!r} has no reason; write it as '{item}  # <reason>'"]


@pytest.mark.parametrize("listed", ["docs/unpaged.txt", "docs/ratchets/readme_numbers.txt"],
                         ids=["the-ledger", "the-readme-start-list"])
def test_w4_a_bare_count_fails_on_the_ledger_and_the_start_list(tmp_path, listed):
    root = copy_checkout(tmp_path)
    path = root / listed
    count = len(read_list(path).lines)
    path.write_text(path.read_text(encoding="utf-8") + f"{count}\n", encoding="utf-8")
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert failures(docs_check(root)) == [f"{path}:{lines}: '{count}' is a count, and a list stores no count: list the "
                                          f"items themselves"]
