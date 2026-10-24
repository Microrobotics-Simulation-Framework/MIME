"""The number rule, and N1, its control. Every plant runs the real rule, ``number_report``, over a doctored copy of the
checkout (``checkout.py``); the wiring arms run the real ``mime docs check`` in a fresh process.

- Keying, on a line of a page source listed with its own reason: a changed number fails, naming the file and the new
  token; a wording edit passes; an added number fails; two numbers swapped fail.
- Occurrences: a reworded copy of a listed line fails as an unlisted ``#2``, naming the file and the tokens; with its
  own ``#2`` line it passes; then the original's number edited fails as new tokens, and the orphaned ``#2`` line fails,
  naming the file and the tokens and saying that the remaining line takes ``#1``'s entry, whose reason was written for
  another. A copy placed before the original makes the original the unlisted ``#2``: the index forces a visit to the
  list, not the copy's re-justification, and the test holds that as the rule's known reach.
- Idioms: ``4D`` fails; ``3D`` and ``6 x 6`` pass; ``6 x 6`` beside ``0.95`` fails on ``0.95`` alone; "takes 2
  iterations" fails.
- Words: a number spelled in words is invisible to the failing rule, an expected gap asserted as one, while the warning
  fires on "sixteen per cent" and "three times", stays silent on "one screen", and never changes the exit status.
- The README start list: a README line with tokens not on it fails; a listed line whose tokens change fails, until
  its number goes or it takes its own reason; a wording edit passes; a line with another reason fails.
- Mutants of the 0.2 line's node pages (``pages_0.2_prose.txt``): every line, with a number inserted in each of the
  three digit forms (an order, a ratio, a percent), and, listed with such a number, the number changed, a number added
  and two swapped, each mutant in a page of its own, must fail on every run; the listed lines and their rewordings
  pass. The test prints the detection rate per form."""

import re
from pathlib import Path

import pytest
from checkout import ROOT, copy_checkout, docs_check, edit

from mime.docs._numbers import (ALLOWLIST, IDIOMS, README_START, TOKEN_GRAMMAR, Occurrence, idiom_pattern,
                                number_report, number_tokens, occurrences, read_idioms)

PAGE = "docs/entries/interaction/drag.md"
LINE = "Its error against the BEM is 16% at an offset of 0.3 of the radius."
LISTED = f"{PAGE}  16% 0.3  #1  # planted for the control\n"
HERE = Path(__file__).parent


def with_page(tmp_path, *lines: str, listed: str = LISTED) -> Path:
    """A copy of the checkout with the page source ``PAGE`` holding ``lines`` (after its front matter), its ledger line
    deleted, and ``listed`` appended to the number allowlist."""
    root = copy_checkout(tmp_path)
    edit(root / "docs/unpaged.txt", "entry interaction drag  # pre-D1-min; paid at D2\n", "")
    path = root / PAGE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ndescribes: 0.2.0\n---\n\n" + "".join(f"{line}\n" for line in lines), encoding="utf-8")
    allowlist = root / ALLOWLIST
    allowlist.write_text(allowlist.read_text(encoding="utf-8") + listed, encoding="utf-8")
    return root


def problems(root: Path) -> list[str]:
    return number_report(root).problems


def unlisted(line: int, tokens: str, index: int = 1, copy: bool = False) -> str:
    """The failure of a line of ``PAGE`` whose key is listed nowhere."""
    said = (f"; it is occurrence #{index} of these tokens in its file, and a copy of a listed line takes its own line"
            if copy else "")
    return (f"{PAGE}:{line}: the number token(s) {tokens} in prose{said}. A number is a claim record's (see "
            f"MIME-ANO-nnn); or list the line as '{PAGE}  {tokens}  #{index}  # <reason>' in {ALLOWLIST}")


def orphaned(root: Path, item: str, count: int, tokens: str, index: int) -> str:
    lines = (root / ALLOWLIST).read_text(encoding="utf-8").splitlines()
    (number,) = [n for n, line in enumerate(lines, start=1) if line.split("  # ")[0].split() == item.split()]
    where = f"{root / ALLOWLIST}:{number}: {item!r} matches no line"
    if count == 0:
        return f"{where}: no line of {PAGE} carries the tokens {tokens} now; delete this line"
    assert count == 1, "the plants orphan an entry beside one remaining line"
    return (f"{where}: {count} line(s) of {PAGE} carry the tokens {tokens}, keyed #1 to #{count}, not #{index}; delete "
            f"this line. Lines with the same tokens are numbered in line order, so once one is edited or removed the "
            f"remaining line now takes the #1 entry, whose reason was written for another line: read it again (the "
            f"index forces this visit; only you know which line restates which)")


# The front matter is lines 1 to 3 and a blank line 4, so a page's first line is line 5.
FIRST = 5


def test_the_rule_passes_the_checkout_and_prints_its_token_grammar(tmp_path):
    report = number_report(copy_checkout(tmp_path))
    assert report.problems == []
    assert f"number tokens: {TOKEN_GRAMMAR}" in report.summary


def test_n1_a_listed_line_passes_and_a_wording_edit_keeps_its_entry(tmp_path):
    assert problems(with_page(tmp_path, LINE)) == []
    reworded = LINE.replace("Its error against the BEM", "Measured against the BEM, the error")
    assert problems(with_page(tmp_path / "reworded", reworded)) == []


@pytest.mark.parametrize(("changed", "tokens"), [
    (LINE.replace("16%", "17%"), "17% 0.3"),
    (LINE.replace("0.3 of", "0.3 of 2.97, of"), "16% 0.3 2.97"),
    (LINE.replace("16%", "0.3").replace("0.3 of", "16% of"), "0.3 16%"),
], ids=["a-changed-number", "an-added-number", "two-numbers-swapped"])
def test_n1_a_new_key_fails_naming_the_file_and_its_tokens_and_orphans_the_old_entry(tmp_path, changed, tokens):
    root = with_page(tmp_path, changed)
    assert problems(root) == [unlisted(FIRST, tokens), orphaned(root, f"{PAGE} 16% 0.3 #1", 0, "16% 0.3", 1)]


def test_n1_a_copy_of_a_listed_line_is_an_unlisted_second_occurrence_until_it_has_its_own_line(tmp_path):
    copy = LINE.replace("Its error", "As above, its error")
    assert problems(with_page(tmp_path, LINE, copy)) == [unlisted(FIRST + 1, "16% 0.3", 2, copy=True)]
    second = f"{PAGE}  16% 0.3  #2  # a restatement, planted\n"
    assert problems(with_page(tmp_path / "listed", LINE, copy, listed=LISTED + second)) == []
    root = with_page(tmp_path / "edited", LINE.replace("16%", "17%"), copy, listed=LISTED + second)
    assert problems(root) == [unlisted(FIRST, "17% 0.3"), orphaned(root, f"{PAGE} 16% 0.3 #2", 1, "16% 0.3", 2)]


def test_n1_a_copy_placed_before_the_original_fails_and_the_original_is_the_line_named(tmp_path):
    """The copy takes ``#1`` and the original's entry with it: the failure names the original, now ``#2``, so the
    edit forces a visit to the list but not the copy's own reason (no key of file, tokens and position can)."""
    copy = LINE.replace("Its error", "As below, its error")
    assert problems(with_page(tmp_path, copy, LINE)) == [unlisted(FIRST + 1, "16% 0.3", 2, copy=True)]


@pytest.mark.parametrize(("line", "tokens"), [
    ("A 4D field.", "4D"),
    ("The 6 x 6 resistance is good to 0.95 of the BEM's.", "0.95"),
    ("The solve takes 2 iterations.", "2"),
], ids=["4D-is-no-idiom", "6x6-beside-0.95-fails-on-0.95", "a-count-that-is-a-result"])
def test_n1_a_form_that_is_no_idiom_fails(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


@pytest.mark.parametrize("line", ["A 3D mesh.", "The 6 x 6 resistance.", "A 2D slice of the 3D field."])
def test_n1_an_idiom_passes(tmp_path, line):
    assert problems(with_page(tmp_path, line, listed="")) == []


def test_the_idiom_list_holds_exact_digit_forms_each_with_a_reason():
    idioms, found = read_idioms(Path(__file__).resolve().parents[2] / IDIOMS)
    assert idioms == ("2D", "3D", "6 x 6") and found == []


def test_n1_a_number_in_a_code_fence_is_exempt_and_outside_one_is_not(tmp_path):
    fenced = ["```python", "tolerance = 0.95", "```"]
    assert problems(with_page(tmp_path, *fenced, listed="")) == []
    assert problems(with_page(tmp_path / "unfenced", fenced[1], listed="")) == [unlisted(FIRST, "0.95")]


WORDS = ["It is sixteen per cent off at the wall.", "It converges three times faster.", "It fits on one screen."]


def test_n1_words_are_the_failing_rules_gap_and_the_warning_fires_on_them(tmp_path):
    report = number_report(with_page(tmp_path, *WORDS, listed=""))
    assert report.problems == [], "the digit rule sees no digit in a number spelled in words: an expected gap"
    assert [(warning.line, warning.phrase) for warning in report.warnings
            if warning.path.as_posix() == PAGE] == [(FIRST, "sixteen per cent"), (FIRST + 1, "three times")]


def test_n1_the_warning_never_changes_the_exit_status(tmp_path):
    done = docs_check(with_page(tmp_path, *WORDS, listed=""))
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"warning: {PAGE}:{FIRST}: 'sixteen per cent' is a number spelled in words" in done.stdout
    assert f"warning: {PAGE}:{FIRST + 1}: 'three times' is a number spelled in words" in done.stdout
    assert "'one screen'" not in done.stdout


def test_n1_an_unlisted_number_fails_through_the_real_command(tmp_path):
    done = docs_check(with_page(tmp_path, LINE.replace("16%", "17%")))
    assert done.returncode == 1 and f"FAIL: {unlisted(FIRST, '17% 0.3')}" in done.stdout


# -- the README start list ----------------------------------------------------------------------------------------

IDIOM = idiom_pattern(read_idioms(ROOT / IDIOMS)[0])


def single(root: Path) -> tuple[int, Occurrence, str]:
    """The README start list's first line keying one token that occurs once in its README line, on no other line of its
    file: the list line's number, the occurrence it keys, and that README line's text."""
    lines = (root / README_START).read_text(encoding="utf-8").splitlines()
    for number, text in enumerate(lines, start=1):
        if not text or text.startswith("#"):
            continue
        file, *tokens, index = text.split("  # ")[0].split()
        found = occurrences(root, Path(file), IDIOM)
        (occurrence,) = [held for held in found if held.key == f"{file} {' '.join(tokens)} {index}"]
        line = (root / file).read_text(encoding="utf-8").splitlines()[occurrence.line - 1]
        alone = [held for held in found if held.tokens == occurrence.tokens] == [occurrence]
        if len(tokens) == 1 and line.count(tokens[0]) == 1 and alone:
            return number, occurrence, line
    raise AssertionError("no line of the start list keys one token that occurs once")


def replace_line(root: Path, occurrence: Occurrence, new: str) -> None:
    path = root / occurrence.path
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[occurrence.line - 1] = new + "\n"
    path.write_text("".join(lines), encoding="utf-8")


def delete_line(path: Path, number: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:number - 1] + lines[number:]), encoding="utf-8")


def test_n1_a_readme_line_not_on_the_start_list_fails(tmp_path):
    root = copy_checkout(tmp_path)
    readme = root / "src/mime/README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nIt is 16% faster.\n", encoding="utf-8")
    (problem,) = problems(root)
    assert problem.startswith("src/mime/README.md:") and "the number token(s) 16% in prose" in problem


def test_n1_a_listed_readme_lines_wording_edit_passes(tmp_path):
    root = copy_checkout(tmp_path)
    _, occurrence, line = single(root)
    replace_line(root, occurrence, line.replace(" ", "  ", 1) + " (reworded)")
    assert problems(root) == []


def test_n1_a_listed_readme_line_whose_number_changes_fails_until_it_has_its_own_reason(tmp_path):
    root = copy_checkout(tmp_path)
    number, occurrence, line = single(root)
    (token,) = occurrence.tokens
    replace_line(root, occurrence, line.replace(token, changed(token)))
    file = occurrence.path.as_posix()
    found = problems(root)
    assert len(found) == 2
    assert found[0].startswith(f"{file}:{occurrence.line}: the number token(s) {changed(token)} in prose")
    assert found[1].startswith(f"{root / README_START}:{number}: '{file} {token} #{occurrence.index}' matches no "
                               f"line")
    delete_line(root / README_START, number)
    (root / ALLOWLIST).write_text((root / ALLOWLIST).read_text(encoding="utf-8")
                                  + f"{file}  {changed(token)}  #1  # its own reason, planted\n", encoding="utf-8")
    assert problems(root) == []


def test_n1_a_listed_readme_line_whose_number_goes_passes_once_its_line_is_deleted(tmp_path):
    root = copy_checkout(tmp_path)
    number, occurrence, line = single(root)
    (token,) = occurrence.tokens
    replace_line(root, occurrence, line.replace(token, "see MIME-ANO-114"))
    (problem,) = problems(root)
    assert problem.startswith(f"{root / README_START}:{number}: ") and "matches no line" in problem
    delete_line(root / README_START, number)
    assert problems(root) == []


def test_n1_the_start_list_takes_no_reason_but_its_own(tmp_path):
    root = copy_checkout(tmp_path)
    number, *_ = single(root)
    text = (root / README_START).read_text(encoding="utf-8").splitlines(keepends=True)
    text[number - 1] = text[number - 1].replace("# present at D1-min", "# a later reason")
    (root / README_START).write_text("".join(text), encoding="utf-8")
    (problem,) = problems(root)
    assert problem.startswith(f"{root / README_START}:{number}: ") and "holds only lines present at D1-min" in problem


# -- mutants of the 0.2 pages -------------------------------------------------------------------------------------

CORPUS = [line for line in (HERE / "pages_0.2_prose.txt").read_text(encoding="utf-8").splitlines()
          if line and not line.startswith("# ")]
#: Each digit form's two numbers, by the line's position, written as the 0.2 pages wrote them.
FORMS = {
    "order": lambda i: (f"order {(1, 2, 3, 0.95, 1.02, 2.97)[i % 6]}", f"order {(4, 0.5, 2.5)[i % 3]}"),
    "ratio": lambda i: (f"a ratio of {(0.33, 1.5, 3, 12.5)[i % 4]}", f"{(3, 10, 2)[i % 3]}:1"),
    "percent": lambda i: (f"{(16, 1, 0.5, 97.2)[i % 4]}%", f"{(5, 43, 6.5)[i % 3]}%"),
}


def inserted(line: str, phrase: str) -> str:
    """``line`` with ``(phrase)`` put between its words, after its middle one."""
    words = line.split(" ")
    at = max(1, len(words) // 2)
    return " ".join([*words[:at], f"({phrase})", *words[at:]])


def changed(phrase: str) -> str:
    """``phrase`` with its last digit changed."""
    at = max(index for index, char in enumerate(phrase) if char.isdigit())
    return phrase[:at] + str((int(phrase[at]) + 1) % 10) + phrase[at + 1:]


def mutant_pages() -> dict[str, tuple[str, str, bool]]:
    """Each page's name to its line, the line its allowlist entry is the key of, and whether it must fail. Every corpus
    line takes a number inserted unlisted in each form; and, in one form by turns, the line listed with its form's
    first number inserted, which passes, as does its rewording, while the number changed and a number added fail; and
    listed with both numbers, which passes, while the two swapped fail."""
    pages: dict[str, tuple[str, str, bool]] = {}
    for i, line in enumerate(CORPUS):
        for form, numbers in FORMS.items():
            pages[f"insert_{form}_{i}"] = (inserted(line, numbers(i)[0]), line, True)
        form = list(FORMS)[i % len(FORMS)]
        first, second = FORMS[form](i)
        one, both = inserted(line, first), inserted(line, f"{first} and {second}")
        pages[f"listed_{form}_{i}"] = (one, one, False)
        pages[f"reworded_{form}_{i}"] = (one.replace(" ", "  ", 1) + " (reworded)", one, False)
        pages[f"changed_{form}_{i}"] = (inserted(line, changed(first)), one, True)
        pages[f"added_{form}_{i}"] = (both, one, True)
        pages[f"both_{form}_{i}"] = (both, both, False)
        pages[f"swapped_{form}_{i}"] = (inserted(line, f"{second} and {first}"), both, True)
    return pages


def test_n1_mutants_of_the_0_2_pages_fail_every_one_and_the_listed_lines_pass(tmp_path, capsys):
    pages = mutant_pages()
    root = copy_checkout(tmp_path)
    directory = root / "docs/entries/interaction"
    directory.mkdir(parents=True)
    listed: list[str] = []
    for name, (line, keyed, _) in pages.items():
        (directory / f"{name}.md").write_text(line + "\n", encoding="utf-8")
        if tokens := number_tokens(keyed, IDIOM):
            listed.append(f"docs/entries/interaction/{name}.md  {' '.join(tokens)}  #1  # planted for N1\n")
    allowlist = root / ALLOWLIST
    allowlist.write_text(allowlist.read_text(encoding="utf-8") + "".join(listed), encoding="utf-8")
    caught = {name for problem in number_report(root).problems
              for name in re.findall(r"docs/entries/interaction/([a-z]+_[a-z]+_\d+)\.md", problem)}
    rates: dict[str, tuple[int, int]] = {}
    for name, (_, _, fails) in pages.items():
        kind, form, _ = name.split("_")
        label = f"{'caught' if fails else 'passed'}, {kind} {form}"
        hit, total = rates.get(label, (0, 0))
        rates[label] = (hit + ((name in caught) == fails), total + 1)
    with capsys.disabled():
        print("\nN1 on the 0.2 pages' prose: " + "; ".join(f"{label} {hit}/{total}"
                                                         for label, (hit, total) in sorted(rates.items())))
    assert {name for name, (_, _, fails) in pages.items() if fails} == caught
