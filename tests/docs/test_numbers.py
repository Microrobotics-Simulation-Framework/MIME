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

import os
import re
import subprocess
from pathlib import Path

import pytest
from checkout import ROOT, copy_checkout, docs_check, edit, failures

from mime.docs._numbers import (ALLOWLIST, IDIOMS, README_START, TOKEN_GRAMMAR, Occurrence, number_report,
                                number_tokens, occurrences, read_idioms, reading_of)

PAGE = "docs/entries/interaction/drag.md"
LINE = "Its error against the BEM is 16% at an offset of 0.3 of the radius."
LISTED = f"{PAGE}  16% 0.3  #1  # planted for the control\n"
HERE = Path(__file__).parent


def with_page(tmp_path, *lines: str, listed: str = LISTED) -> Path:
    """A copy of the checkout with the page source ``PAGE`` holding ``lines`` (after its front matter), in place of the
    drag's own, and ``listed`` appended to the number allowlist."""
    root = copy_checkout(tmp_path)
    path = root / PAGE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ndescribes: 0.3.0\n---\n\n" + "".join(f"{line}\n" for line in lines), encoding="utf-8")
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


@pytest.mark.parametrize("line", ["A 3D mesh.", "The 6 x 6 resistance.", "A 2D slice of the 3D field.",
                                  "The resistance is positive where V^T R V > 0.", "It holds where v.Rv >= 0.",
                                  "The drag is linear as Re -> 0.", "The tail sits at L/2 of the body."])
def test_n1_an_idiom_passes(tmp_path, line):
    assert problems(with_page(tmp_path, line, listed="")) == []


@pytest.mark.parametrize(("line", "tokens"), [
    ("It holds where x > 0.5.", "0.5"),
    ("It holds where x >= 0.5.", "0.5"),
    ("The drag is linear as Re -> 0.1.", "0.1"),
    ("The tail sits at L/2.5 of the body.", "2.5"),
    ("It holds where x > 0 mm.", "0"),
    ("It holds where x > 10.", "10"),
    ("It holds where x >= 05.", "05"),
    ("The tail sits at L/20 of the body.", "20"),
    ("The tail sits at 3L/2.", "3L 2"),
], ids=["gt-0.5", "ge-0.5", "to-0.1", "L-over-2.5", "gt-0-before-a-unit", "gt-10", "ge-05", "L-over-20", "3L-over-2"])
def test_n1_a_near_miss_of_a_structural_idiom_is_no_idiom_and_is_keyed(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


def test_the_idiom_list_holds_exact_digit_forms_each_with_a_reason():
    idioms, found = read_idioms(Path(__file__).resolve().parents[2] / IDIOMS)
    assert idioms == ("2D", "3D", "6 x 6", "> 0", ">= 0", "-> 0", "L/2") and found == []


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

READING = reading_of(ROOT)[0]


def single(root: Path) -> tuple[int, Occurrence, str]:
    """The README start list's first line keying one token that occurs once in its README line, on no other line of its
    file, and that changed (``changed``) is on no line of its file either: the list line's number, the occurrence it
    keys, and that README line's text."""
    lines = (root / README_START).read_text(encoding="utf-8").splitlines()
    for number, text in enumerate(lines, start=1):
        if not text or text.startswith("#"):
            continue
        file, *tokens, index = text.split("  # ")[0].split()
        found = occurrences(root, Path(file), READING)
        (occurrence,) = [held for held in found if held.key == f"{file} {' '.join(tokens)} {index}"]
        line = (root / file).read_text(encoding="utf-8").splitlines()[occurrence.line - 1]
        alone = [held for held in found if held.tokens == occurrence.tokens] == [occurrence]
        free = all(changed(tokens[0]) not in held.tokens for held in found) if len(tokens) == 1 else False
        if len(tokens) == 1 and line.count(tokens[0]) == 1 and alone and free:
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
    directory.mkdir(parents=True, exist_ok=True)
    listed: list[str] = []
    for name, (line, keyed, _) in pages.items():
        (directory / f"{name}.md").write_text(line + "\n", encoding="utf-8")
        if tokens := number_tokens(keyed, READING):
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


# -- each exemption as narrow as its rule ----------------------------------------------------------------------------

@pytest.mark.parametrize(("line", "tokens"), [
    ("MIME 0.95 mm of the gap.", "0.95"),
    ("MIME 16% error at the wall.", "16%"),
    ("Python 3 s a step.", "3"),
    ("A 0.2 line width.", "0.2"),
    ("The 0.2 line of the pipe.", "0.2"),
    ("MIME 5 iterations.", "5"),
    ("MADDENING 3 times slower.", "3"),
    ("OpenMaterica 0.95 of the bound.", "0.95"),
    ("MIME 0.3 mm off.", "0.3"),
], ids=["MIME-0.95-mm", "MIME-16%", "Python-3-s", "a-0.2-line-width", "the-0.2-line", "MIME-5", "MADDENING-3-times",
        "OpenMaterica-0.95", "a-release-before-a-unit"])
def test_a_number_after_a_products_name_that_is_no_release_the_code_reads_fails(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


@pytest.mark.parametrize("line", ["MIME 0.3 targets MADDENING 0.4.x.", "MADDENING 0.5's tool, on Python 3.12.",
                                  "OpenMaterica 0.1.0.dev3 and usd-core 26.5."])
def test_a_release_the_code_reads_passes_after_its_products_name(tmp_path, line):
    assert problems(with_page(tmp_path, line, listed="")) == []


def test_the_releases_are_read_from_the_pins(tmp_path):
    """The control that they are read, not written in: with MADDENING's upper bound moved, the old bound is a number
    and the new one a release."""
    root = with_page(tmp_path, "Until MADDENING 0.5.", "Until MADDENING 0.6.", listed="")
    assert problems(root) == [unlisted(FIRST + 1, "0.6")]
    pins = root / "pyproject.toml"
    pins.write_text(pins.read_text(encoding="utf-8").replace(">=0.4.2,<0.5", ">=0.4.2,<0.6"), encoding="utf-8")
    assert [problem for problem in problems(root) if problem.startswith(PAGE)] == [unlisted(FIRST, "0.5")]


@pytest.mark.parametrize(("line", "tokens"), [
    ("The phase 90 degrees ahead.", "90"),
    ("A phase 45deg lag.", "45deg"),
    ("A phase 5mm off.", "5mm"),
    ("Its phase 10 to 20 mm.", "10 20"),
    ("Its phase 3 or 4 mm.", "4"),
    ("Its phase 3 and 4 mm.", "4"),
    ("The phase 0.37 of the drive.", "0.37"),
], ids=["90-degrees", "45deg", "5mm", "10-to-20-mm", "3-or-4-mm", "3-and-4-mm", "a-decimal"])
def test_a_phase_name_is_a_digit_with_at_most_a_letter_and_a_digit_before_no_unit(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


def test_a_phase_name_passes(tmp_path):
    assert problems(with_page(tmp_path, "Since phase 7a2, after phase 4c3.", "Phases 4 and 7, and phases 1, 2, and 3.",
                              listed="")) == []


@pytest.mark.parametrize(("line", "tokens"), [
    ("Its RMS-5 mm.", "5"), ("An SNR-30 signal.", "30"), ("A T-0.5 junction.", "0.5"), ("N-95 masks.", "95"),
    ("ABC-0.95 of it.", "0.95"), ("See MIME-ANO-0.95.", "0.95"), ("See MIME-ANO-1140.", "1140"),
    ("See MIME-NODE-10.", "10"),
], ids=["RMS-5", "SNR-30", "T-0.5", "N-95", "ABC-0.95", "an-id-with-a-decimal", "an-id-of-four-digits",
        "an-id-of-two-digits"])
def test_an_identifier_is_one_of_mimes_prefixes_and_three_digits(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


def test_mimes_identifiers_and_a_numbered_reference_pass(tmp_path):
    assert problems(with_page(tmp_path, "See MIME-ANO-114, MIME-NODE-104, MIME-VER-033 and MADD-ANO-001, as #146 "
                                        "has it.", listed="")) == []


@pytest.mark.parametrize(("line", "tokens"), [
    ("## 0.95 of the radius", "0.95"), ("# 95 percent", "95"), ("95. next", "95")],
    ids=["a-heading's-decimal", "a-heading's-percent", "a-list-item-out-of-its-count"])
def test_a_headings_number_and_a_list_items_out_of_its_count_are_tokens(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


def test_an_ordered_lists_count_passes_and_a_number_out_of_it_fails(tmp_path):
    counted = ["1. Install.", "   Then:", "   1. one", "   2. two", "2. Run.", "", "3. Check."]
    assert problems(with_page(tmp_path, *counted, listed="")) == []
    assert problems(with_page(tmp_path / "skipped", "1. Install.", "3. Check.", listed="")) == [
        unlisted(FIRST + 1, "3")]


def test_an_unclosed_code_fence_fails_naming_the_line_it_opened_on(tmp_path):
    assert problems(with_page(tmp_path, "A line.", "```python", "x = 1", "Prose, 0.97 mm.", listed="")) == [
        f"{PAGE}:{FIRST + 1}: the code fence opened here (```) is never closed, so the rule read none of the file "
        f"after it; close it"]


def test_a_line_opening_with_inline_triple_backticks_fails_and_is_read_as_prose(tmp_path):
    """CommonMark: a backtick fence's info string holds no backtick, so this line is inline code, and the lines after
    it are prose."""
    assert problems(with_page(tmp_path, "```0.95``` is the drag.", "Prose, 0.97 mm.", listed="")) == [
        f"{PAGE}:{FIRST}: it opens with ``` and holds another backtick after it, so it is inline code, not a code "
        f"fence; write inline code with single backticks, or open the fence on a line of its own",
        unlisted(FIRST, "0.95"), unlisted(FIRST + 1, "0.97")]


def test_a_superscript_exponent_is_part_of_its_token_so_a_changed_exponent_is_a_new_key(tmp_path):
    line = "The tolerance is 12 × 10⁻³ of the radius."
    listed = f"{PAGE}  12 10⁻³  #1  # planted for the control\n"
    assert problems(with_page(tmp_path, line, listed=listed)) == []
    root = with_page(tmp_path / "changed", line.replace("³", "⁶"), listed=listed)
    assert problems(root) == [unlisted(FIRST, "12 10⁻⁶"),
                              orphaned(root, f"{PAGE} 12 10⁻³ #1", 0, "12 10⁻³", 1)]


@pytest.mark.parametrize(("line", "tokens"), [
    ("Half of it, ½ mm.", "½"), ("A third, ⅓.", "⅓"), ("It is 0·95 of it.", "0·95"),
    ("It takes 1½ turns.", "1½")], ids=["a-half", "a-third", "a-middle-dot-decimal", "one-and-a-half"])
def test_a_vulgar_fraction_and_a_middle_dot_decimal_are_tokens(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


def test_the_warning_reads_cm_m_s_and_n_and_leaves_other_letters_out(tmp_path):
    report = number_report(with_page(tmp_path, "A gap of five cm.", "It moves at ten m/s.", "It takes forty s.",
                                     "A load of two N.", "One A matrix.", listed=""))
    assert report.problems == []
    assert [(warning.line, warning.phrase) for warning in report.warnings if warning.path.as_posix() == PAGE] == [
        (FIRST, "five cm"), (FIRST + 1, "ten m"), (FIRST + 2, "forty s"), (FIRST + 3, "two N")]


@pytest.mark.parametrize(("line", "tokens"), [
    ("A 13D field.", "13D"), ("A 3Dx field.", "3Dx"), ("A 1.3D field.", "1.3D"), ("A 6 x 6 mm plate.", "6 6"),
    ("A 6 x 6.5 block.", "6 6.5")], ids=["13D", "3Dx", "1.3D", "6x6-mm", "6x6.5"])
def test_an_idiom_inside_a_number_or_a_word_or_before_a_unit_or_a_decimal_is_no_idiom(tmp_path, line, tokens):
    assert problems(with_page(tmp_path, line, listed="")) == [unlisted(FIRST, tokens)]


def test_a_malformed_idiom_line_fails(tmp_path):
    root = copy_checkout(tmp_path)
    path = root / IDIOMS
    path.write_text(path.read_text(encoding="utf-8") + "XD  # a shape, planted\n4D\n", encoding="utf-8")
    count = len(path.read_text(encoding="utf-8").splitlines())
    assert problems(root) == [f"{path}:{count - 1}: 'XD' is malformed: an idiom is a digit form",
                              f"{path}:{count}: '4D' has no reason; write it as '4D  # <reason>'"]


def test_a_key_on_both_lists_fails(tmp_path):
    root = copy_checkout(tmp_path)
    _, occurrence, _ = single(root)
    allowlist = root / ALLOWLIST
    allowlist.write_text(allowlist.read_text(encoding="utf-8") + f"{occurrence.path.as_posix()}  "
                         f"{' '.join(occurrence.tokens)}  #{occurrence.index}  # planted on both\n", encoding="utf-8")
    assert problems(root) == [f"{occurrence.key!r} is on both {root / ALLOWLIST} and {root / README_START}; keep one"]


def test_a_page_sources_line_on_the_readme_start_list_fails(tmp_path):
    root = with_page(tmp_path, LINE, listed="")
    start = root / README_START
    start.write_text(start.read_text(encoding="utf-8") + f"{PAGE}  16% 0.3  #1  # present at D1-min\n",
                     encoding="utf-8")
    count = len(start.read_text(encoding="utf-8").splitlines())
    assert problems(root) == [f"{start}:{count}: '{PAGE} 16% 0.3 #1' is malformed: the README start list holds lines "
                              f"of the package READMEs (src/**/README*.md) only"]


def test_the_allowlist_takes_its_own_reason_not_the_start_lists(tmp_path):
    root = with_page(tmp_path, LINE, listed=f"{PAGE}  16% 0.3  #1  # present at D1-min\n")
    count = len((root / ALLOWLIST).read_text(encoding="utf-8").splitlines())
    assert problems(root) == [f"{root / ALLOWLIST}:{count}: '{PAGE} 16% 0.3 #1' gives the start list's reason, "
                              f"'present at D1-min'; a line of {ALLOWLIST} gives its own"]


@pytest.mark.parametrize(("listed", "problem"), [
    (f"{PAGE}  16% 0.3  #1  #\n", f"'{PAGE} 16% 0.3 #1' has no reason; write it as '{PAGE} 16% 0.3 #1  # <reason>'"),
    (LISTED + "1\n", "'1' is a count, and a list stores no count: list the items themselves"),
], ids=["an-empty-reason", "a-bare-count"])
def test_w4_an_empty_reason_and_a_bare_count_fail_on_the_allowlist(tmp_path, listed, problem):
    """W4's extension to the allowlist, through the real command."""
    root = with_page(tmp_path, LINE, listed=listed)
    count = len((root / ALLOWLIST).read_text(encoding="utf-8").splitlines())
    assert failures(docs_check(root)) == [f"{root / ALLOWLIST}:{count}: {problem}"]


# -- the start list against its base revision ------------------------------------------------------------------------

def committed(root: Path) -> Path:
    """``root`` made a git repository with its files committed, the base revision a control compares with."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    env |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    git = ["git", "-C", str(root), "-c", "user.name=Author", "-c", "user.email=author@example.org"]
    for command in (["init", "-q", "-b", "base"], ["add", "-A"], ["commit", "-q", "-m", "the base"]):
        subprocess.run([*git, *command], check=True, env=env, capture_output=True)
    return root


def grown(root: Path) -> tuple[Path, int]:
    """``root`` with a README line carrying a new number, put on the start list with the shared reason; the start
    list's line number."""
    readme = root / "src/mime/README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nIt is 16% faster.\n", encoding="utf-8")
    start = root / README_START
    start.write_text(start.read_text(encoding="utf-8") + "src/mime/README.md  16%  #1  # present at D1-min\n",
                     encoding="utf-8")
    return start, len(start.read_text(encoding="utf-8").splitlines())


def test_q7_the_start_list_compared_with_its_base_passes_unchanged(tmp_path):
    done = docs_check(committed(copy_checkout(tmp_path)), base="HEAD")
    assert done.returncode == 0, done.stdout + done.stderr
    assert re.search(r"^docs/ratchets/readme_numbers\.txt: compared with HEAD's, (\d+) lines there and \1 here$",
                     done.stdout, re.MULTILINE)


def test_q7_a_start_list_line_the_base_does_not_hold_fails_through_the_real_command(tmp_path):
    root = committed(copy_checkout(tmp_path))
    start, number = grown(root)
    assert failures(docs_check(root, base="HEAD")) == [
        f"{start}:{number}: 'src/mime/README.md 16% #1' is not on the start list at HEAD, which only shrinks; a README "
        f"line's number takes its own line in {ALLOWLIST}, with its own reason logged, or goes (see MIME-ANO-nnn)"]


def test_q7_the_new_number_passes_with_its_own_line_in_the_allowlist(tmp_path):
    root = committed(copy_checkout(tmp_path))
    start, number = grown(root)
    delete_line(start, number)
    (root / ALLOWLIST).write_text((root / ALLOWLIST).read_text(encoding="utf-8")
                                  + "src/mime/README.md  16%  #1  # its own reason, planted\n", encoding="utf-8")
    assert number_report(root, base="HEAD").problems == []


def test_q7_a_base_that_cannot_be_read_fails_and_a_base_without_the_list_is_said(tmp_path):
    root = committed(copy_checkout(tmp_path / "repository"))
    (problem,) = number_report(root, base="no-such-revision").problems
    assert problem.startswith("the base revision 'no-such-revision' cannot be read in ")
    plain = copy_checkout(tmp_path / "plain")
    (problem,) = number_report(plain, base="HEAD").problems
    assert problem.startswith("the base revision 'HEAD' cannot be read in "), "a copy that is no repository"
    (root / README_START).rename(tmp_path / "held")
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    subprocess.run(["git", "-C", str(root), "-c", "user.name=A", "-c", "user.email=a@example.org", "commit", "-q",
                    "-am", "no start list"], check=True, env=env, capture_output=True)
    (tmp_path / "held").rename(root / README_START)
    report = number_report(root, base="HEAD")
    assert report.problems == []
    assert f"{README_START}: HEAD holds no start list, so it is compared with nothing: this change adds it" in \
        report.summary
