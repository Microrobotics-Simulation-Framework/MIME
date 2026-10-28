"""The number rule over the strings the generator renders from code (item 2: validity lines, an interface's description,
``NodeMeta`` text, a gradient record's text). A page prints them as the code states them, so a number in one is in the
documentation as much as a number in a hand-written file is; the rule reads them as it reads a file's lines, keyed by the
page's name under ``generated/``, the number tokens and an occurrence index.

- Each kind of string a generator renders is reached: a number planted in it, in each of three digit forms (an order, a
  ratio, a percent), fails the rule, naming the page and the string, on every run (the generators call ``prose`` for each
  of them, which this test holds by planting in the real generators, not in the rule alone).
- Every string a page records is on the page, verbatim, so the rule reads what the reader reads.
- Keying: a listed string passes, a reworded one keeps its entry, a changed or added number is a new key, a copy is
  an unlisted ``#2``, and an entry matching no string is an orphan; a number spelled in words is a warning and never a failure.
- Namespace rules: through the real command, a number in a validity line of MIME's own entry fails and, listed, passes; the
  same line in a lab's entry is counted and not judged."""

import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from checkout import ROOT, copy_checkout, docs_check, failures
from maddening.core.compliance.metadata import ValidatedRegime
from planted_nodes import META, planted
from test_docs_check import paged

from mime.docs._build import build_pages
from mime.docs._context import Context
from mime.docs._entry_page import entry_page
from mime.docs._family_page import family_page
from mime.docs._markdown import cell
from mime.docs._node_page import node_page
from mime.docs._numbers import number_report, rendered_occurrences
from mime.testing import Family, SignRule

FORMS = {"an order": "of order 2 in the step", "a ratio": "a ratio of 3 to 1", "a percent": "off by up to 16%"}
EMPTY = Context((), {}, {}, {})


def minimal_root(tmp_path: Path) -> Path:
    """What the rule reads of a checkout: its idiom list, the two lists, and ``pyproject.toml`` for the releases."""
    (tmp_path / "docs" / "ratchets").mkdir(parents=True)
    shutil.copyfile(ROOT / "docs" / "number_idioms.txt", tmp_path / "docs" / "number_idioms.txt")
    for name in ("number_allowlist", "readme_numbers"):
        (tmp_path / "docs" / "ratchets" / f"{name}.txt").write_text("", encoding="utf-8")
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    return tmp_path


def entry(**declared: object) -> dict:
    classvars = {"differentiability": "exact", "references": ["Sola2017"], **declared}
    return {"kind": "interaction", "name": "planted.law", "version": "0.1.0", "namespace": "mime", "defined_in": "x.Y",
            "fields": {}, "settings": {},
            "class_vars": {name: {"type": "t", "declared_by": "x", "default": None, "value": value}
                           for name, value in classvars.items()}}


def meta(**changed: object) -> type:
    return planted("Texted", meta=replace(META, **changed))


def family(description: str) -> Family:
    return Family(name="planted", rig=lambda law: None, cases=(), sign=SignRule(description, lambda response: True))


#: Each kind of string the generators render, by a function from a text to the page it is printed on.
KINDS = {
    "a validity line": lambda t: entry_page(entry(validity=[t]), EMPTY),
    "a gradient record's where": lambda t: entry_page(entry(differentiability="piecewise", gradient={
        "where": t, "reason": "r", "route": "q"}), EMPTY),
    "a gradient record's reason": lambda t: entry_page(entry(differentiability="piecewise", gradient={
        "where": "w", "reason": t, "route": "q"}), EMPTY),
    "a gradient record's route": lambda t: entry_page(entry(differentiability="piecewise", gradient={
        "where": "w", "reason": "r", "route": t}), EMPTY),
    "a support row's note": lambda t: entry_page(entry(rungs=["a"], builds={"a": []}, support=[
        {"rung": "a", "when": {}, "note": t}]), EMPTY),
    "a reserved rung's reason": lambda t: entry_page(entry(rungs=["a"], reserved_rungs={"a": t}, builds={}), EMPTY),
    "a node's description": lambda t: node_page(meta(description=t), "m.Texted", EMPTY),
    "a node's discretisation": lambda t: node_page(meta(discretization=t), "m.Texted", EMPTY),
    "a node's assumption": lambda t: node_page(meta(assumptions=(t,)), "m.Texted", EMPTY),
    "a node's limitation": lambda t: node_page(meta(limitations=(t,)), "m.Texted", EMPTY),
    "a validated regime's note": lambda t: node_page(meta(validated_regimes=(ValidatedRegime("x", notes=t),)),
                                                       "m.Texted", EMPTY),
    "a family's sign rule": lambda t: family_page(family(t), EMPTY),
}


@pytest.mark.parametrize("form", FORMS, ids=list(FORMS))
@pytest.mark.parametrize("kind", KINDS, ids=list(KINDS))
def test_a_number_planted_in_each_kind_of_rendered_string_fails_naming_the_page_and_the_string(tmp_path, kind, form):
    """The control, per kind and per digit form, on every run: the page's recorded strings are read by the real rule, so
    this fails if a generator prints a string without recording it, or if the rule misses the form."""
    text = f"valid for a body {FORMS[form]}"
    page = KINDS[kind](text)
    assert text in "".join(t for _, t in page.rendered), f"{kind}: the page did not record the planted string"
    rendered = [(f"generated/{page.path}", label, t) for label, t in page.rendered]
    report = number_report(minimal_root(tmp_path), rendered=rendered)
    tokens = {"an order": "2", "a ratio": "3 1", "a percent": "16%"}[form]
    found = [p for p in report.problems if f"generated/{page.path}:" in p and f"the number token(s) {tokens} in prose" in p]
    assert len(found) == 1 and "text from the code" in found[0], report.problems


def test_a_planted_string_without_a_number_passes_in_every_kind(tmp_path):
    for kind, make in KINDS.items():
        page = make("valid for a body in a straight tube")
        rendered = [(f"generated/{page.path}", label, t) for label, t in page.rendered]
        assert number_report(minimal_root(tmp_path / re.sub(r"\W", "", kind)), rendered=rendered).problems == [], kind


def test_every_string_a_page_records_is_on_the_page_as_it_is_printed():
    """The rule reads the recorded strings, so a page must print each as recorded (a table cell collapses whitespace and
    escapes a pipe, which is how it reads)."""
    built = build_pages(ROOT)
    missing = [(path, label) for path, page in sorted(built.pages.items()) for label, text in page.rendered
               if text not in page.text and cell(text) not in page.text]
    assert missing == []
    assert sum(len(page.rendered) for page in built.pages.values()) > 30, "too few strings to be evidence"


def test_the_vocabulary_pages_record_the_sign_and_description_of_every_interface_of_mimes():
    from mime.vocabulary import lookup, names

    built = build_pages(ROOT)
    labels = {label for label, _ in built.pages["pages/vocabulary.md"].rendered}
    for name in names():
        held = lookup(name)
        assert held is not None
        if held.namespace == "mime":
            assert (f"{name} sign" in labels or not held.sign) and (f"{name} description" in labels
                                                                   or not held.description), name
        else:
            assert f"{name} description" not in labels, "another namespace's text is printed and not judged"


# -- keying: a rendered string is keyed like a line of a file ---------------------------------------------------------

PAGE = "generated/entries/interaction/planted.md"


def judged(tmp_path: Path, strings: list[str], listed: str = "") -> list[str]:
    root = minimal_root(tmp_path)
    (root / "docs" / "ratchets" / "number_allowlist.txt").write_text(listed, encoding="utf-8")
    rendered = [(PAGE, f"validity line {n}", text) for n, text in enumerate(strings, start=1)]
    return number_report(root, rendered=rendered).problems


LISTED = f"{PAGE}  0.95  #1  # planted for the control\n"


def test_a_listed_string_passes_and_a_reworded_one_keeps_its_entry(tmp_path):
    assert judged(tmp_path / "a", ["clamped at 0.95 of the room"], LISTED) == []
    assert judged(tmp_path / "b", ["held to 0.95 of the room, reworded"], LISTED) == []


@pytest.mark.parametrize(("string", "tokens"), [
    ("clamped at 0.9 of the room", "0.9"),
    ("clamped at 0.95 of the room and 3 of the width", "0.95 3"),
    ("clamped at 0.95", "0.95"),
], ids=["a-changed-number", "an-added-number", "a-reworded-string-with-the-same-tokens"])
def test_a_changed_or_added_number_is_a_new_key_and_the_old_entry_is_orphaned(tmp_path, string, tokens):
    problems = judged(tmp_path, [string], LISTED)
    if tokens == "0.95":
        assert problems == []
        return
    assert any(f"the number token(s) {tokens} in prose" in p for p in problems)
    assert any("matches no line" in p and "0.95" in p for p in problems)


def test_a_copy_of_a_listed_string_is_an_unlisted_second_occurrence(tmp_path):
    problems = judged(tmp_path, ["clamped at 0.95 of the room", "also clamped at 0.95"], LISTED)
    (problem,) = problems
    assert "0.95 in prose; it is occurrence #2 of these tokens in its file" in problem and "validity line 2" in problem
    assert judged(tmp_path / "own", ["clamped at 0.95 of the room", "also clamped at 0.95"],
                  LISTED + f"{PAGE}  0.95  #2  # planted\n") == []


def test_a_number_spelled_in_words_is_a_warning_and_never_a_failure_in_a_rendered_string(tmp_path):
    root = minimal_root(tmp_path)
    report = number_report(root, rendered=[(PAGE, "validity line 1", "off by sixteen per cent, three times")])
    assert report.problems == []
    assert {w.phrase for w in report.warnings} == {"sixteen per cent", "three times"}


def test_without_the_rendered_text_a_listed_key_of_a_generated_page_is_not_judged(tmp_path):
    root = minimal_root(tmp_path)
    (root / "docs" / "ratchets" / "number_allowlist.txt").write_text(LISTED, encoding="utf-8")
    assert number_report(root).problems == []
    assert any("matches no line" in p for p in number_report(root, rendered=[]).problems)


def test_the_occurrence_index_counts_a_pages_strings_in_the_order_they_come():
    found = rendered_occurrences([(PAGE, "a", "x 0.95"), (PAGE, "b", "y 0.95"), ("generated/other.md", "a", "z 0.95")],
                                 None)
    assert [(o.path.as_posix(), o.tokens, o.index) for o in found] == [
        (PAGE, ("0.95",), 1), (PAGE, ("0.95",), 2), ("generated/other.md", ("0.95",), 1)]


# -- the namespace rules, through the real command ---------------------------------------------------------------------

def test_a_number_in_a_validity_line_of_mimes_own_entry_fails_and_listed_passes_and_a_labs_is_counted(tmp_path):
    root = paged(copy_checkout(tmp_path), "docsplant_mime_numbered")
    (problem,) = failures(docs_check(root, plant="mime_numbered"))
    assert problem.startswith("generated/entries/interaction/docsplant_mime_numbered.md:") and \
        "(validity line 1, text from the code): the number token(s) 0.95 in prose" in problem
    listing = root / "docs" / "ratchets" / "number_allowlist.txt"
    listing.write_text(listing.read_text(encoding="utf-8") +
                       "generated/entries/interaction/docsplant_mime_numbered.md  0.95  #1  # planted\n",
                       encoding="utf-8")
    done = docs_check(root, plant="mime_numbered")
    assert done.returncode == 0, done.stdout
    lab = docs_check(copy_checkout(tmp_path / "lab"), plant="lab_numbered")
    assert lab.returncode == 0, lab.stdout
    assert re.search(r"text from the code of pages outside MIME, counted, not judged by the number rule: [1-9]\d* "
                     r"strings", lab.stdout)
