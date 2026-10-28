r"""The citation parser (4g) and its control N2.

``scripts/check_citations.py`` read ``\[@(\w+)``: for ``[@Cox1970; @Goldman1967; @Typo9999]`` it saw one key and the
typo passed; bare ``@Key``, ``[-@Key]`` and ``[see @Key]`` were not seen; ``[@de-Jongh2025]`` was read as ``de``; its
bibliography regex would not extract ``de-Jongh2025``; it printed a raw count and passed when ``docs/`` held nothing.
The parser reads Pandoc's grammar, counts the citations it checked and the ones it did not find, and fails on none at
all when there are pages.

N2 is ``[@a; @TYPO]``, and its control is the old pattern, which sees one key of the two, so that the plant discriminates:
a test that the old pattern passes it, and one that the parser fails it, naming the typo. The parser's tests are
deterministic, so each fails on every run."""

import re

import pytest
from checkout import ROOT, copy_checkout, docs_check, edit, failures

from mime.docs._citations import bibliography_keys, citation_report, citations_in

OLD_PATTERN = re.compile(r"\[@(\w+)")
KEYS = {"Cox1970", "Goldman1967", "de-Jongh2025", "Key"}


def keys(text: str) -> list[str]:
    return [citation.key for citation in citations_in("page.md", text)]


def test_n2_a_typo_in_a_list_of_citations_fails_naming_it():
    text = "As shown [@Cox1970; @Goldman1967; @Typo9999].\n"
    report = citation_report({"page.md": text}, KEYS)
    assert report.problems == ["page.md:1: [@Typo9999] is not an entry of docs/bibliography.bib; add the work to the "
                               "bibliography, or correct the key"]
    assert (report.checked, report.not_found) == (2, 1)
    assert "citations: 3 in 1 pages and sources, 2 checked against 4 bibliography entries, 1 not found" in \
        report.summary[0]


def test_n2_the_control_the_old_pattern_sees_one_key_of_the_two_and_passes_the_typo():
    text = "As shown [@a; @TYPO].\n"
    assert OLD_PATTERN.findall(text) == ["a"], "the old pattern must miss the second key, or the plant proves nothing"
    assert keys(text) == ["a", "TYPO"]
    report = citation_report({"page.md": text}, {"a"})
    assert [p for p in report.problems if "TYPO" in p] and not [p for p in report.problems if "[@a]" in p]


@pytest.mark.parametrize(("text", "found"), [
    ("[@Key]", ["Key"]),
    ("[@a; @b; @c]", ["a", "b", "c"]),
    ("[see @Key, p. 33]", ["Key"]),
    ("[-@Key]", ["Key"]),
    ("In text @Key says so.", ["Key"]),
    ("In text @Key.", ["Key"]),
    ("In text @Key, then @Other.", ["Key", "Other"]),
    ("@Key [p. 33]", ["Key"]),
    ("[@de-Jongh2025]", ["de-Jongh2025"]),
    ("[@a:b/c.d#e]", ["a:b/c.d#e"]),
    ("[@_key2]", ["_key2"]),
    ("@{any key/at all}", ["any key/at all"]),
    ("[@Key1; @Key2, ch. 3; -@Key3]", ["Key1", "Key2", "Key3"]),
], ids=["one", "list", "prefix-and-locator", "suppressed-author", "bare", "bare-before-a-full-stop", "bare-and-a-comma",
        "bare-with-a-locator", "hyphen", "internal-punctuation", "underscore", "braced", "mixed"])
def test_the_citation_forms_of_pandocs_grammar_are_read(text, found):
    assert keys(text) == found


@pytest.mark.parametrize("text", [
    "write to someone@example.org",
    "the handle `@Key` in code",
    "``a `@Key` span``",
    "[a link](https://example.org/@Key)",
    "see https://example.org/@Key now",
    "an escaped \\@Key",
    "a trailing @ alone and a@",
], ids=["email", "code-span", "double-backtick-span", "link-target", "url", "escaped", "bare-at"])
def test_text_that_is_not_a_citation_is_not_read_as_one(text):
    assert keys(text) == []


def test_a_citation_in_a_code_fence_is_not_read_and_the_line_after_it_is():
    text = "Before.\n```\n[@InFence]\n```\nAfter [@Key].\n"
    found = citations_in("page.md", text)
    assert [(c.key, c.line) for c in found] == [("Key", 5)]


def test_the_bibliography_reader_extracts_hyphenated_keys_and_skips_what_is_not_a_reference():
    bib = ("@article{Cox1970,\n  title = {x},\n}\n@book{de-Jongh2025,\n}\n@comment{not a reference,\n}\n"
           "@string{journal = \"x\"}\n@misc{ spaced:key ,\n}\n")
    assert bibliography_keys(bib) == {"Cox1970", "de-Jongh2025", "spaced:key"}
    assert bibliography_keys((ROOT / "docs" / "bibliography.bib").read_text(encoding="utf-8")) >= {"Purcell1977"}


def test_pages_that_cite_nothing_fail_rather_than_pass_on_nothing_and_no_pages_is_no_failure():
    """The old script passed when ``docs/`` held nothing. Pages that exist and cite nothing mean the parser read nothing."""
    (problem,) = citation_report({"a.md": "no citation here\n", "b.md": "nor here\n"}, KEYS).problems
    assert problem.startswith("2 pages and sources were read and none holds a citation")
    assert citation_report({}, KEYS).problems == []
    assert citation_report({"a.md": "cites [@Key]\n"}, KEYS).problems == []


def test_a_page_outside_mime_is_counted_and_not_judged():
    report = citation_report({"lab.md": "[@LabKey1999]\n", "mime.md": "[@Key]\n"}, KEYS, exempt={"lab.md"})
    assert report.problems == [] and (report.checked, report.not_found) == (1, 0)
    assert "1 in pages outside MIME, counted, not checked" in report.summary[0]


def test_each_failure_names_the_page_and_the_line_it_is_on():
    report = citation_report({"a.md": "x\n\n[@Gone1]\n", "b.md": "[@Key]\n[@Gone2]\n"}, KEYS)
    assert [problem.split(" ")[0] for problem in report.problems] == ["a.md:3:", "b.md:2:"]


def test_the_command_fails_a_planted_typo_in_a_hand_written_page_and_passes_it_without(tmp_path):
    """Through the real command: the concepts note, hand-written, cites a work; a typo fails it, naming the page."""
    root = copy_checkout(tmp_path)
    note = root / "docs" / "concepts.md"
    note.write_text(note.read_text(encoding="utf-8") + "\nSee [@CoxBrenner1967; @NoSuchWork].\n", encoding="utf-8")
    (problem,) = failures(docs_check(root))
    assert re.fullmatch(r"pages/concepts\.md:\d+: \[@NoSuchWork\] is not an entry of docs/bibliography\.bib; add the "
                        r"work to the bibliography, or correct the key", problem)
    edit(note, "; @NoSuchWork", "")
    assert docs_check(root).returncode == 0
