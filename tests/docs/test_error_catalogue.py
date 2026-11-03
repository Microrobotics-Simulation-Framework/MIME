"""The refusal catalogue: a page for every code of the refusal table, and the pages of the codes on the refusal list
saying that no test names them.

The list reads the tests' text, so what a page of a listed code says is what the list knows: no file under ``tests/``
names the code. The catalogue's hand-written meanings wait for a lab's reading of the pages (the usability check), so
none is written here.

The controls: a code planted on the list through the generator's argument says it, and an unlisted one does not (the
generator), and a line deleted from a copy of the committed list takes the sentence off that page and no other
(the wiring from the list to the build, which a generator test with its own argument cannot see). The pages are
deterministic, so each control fails on every run."""

import re

from checkout import ROOT, copy_checkout, docs_run

from mime.build import REFUSAL_TABLE
from mime.docs import read_list
from mime.docs._error_pages import ERRORS_INDEX, NO_TEST, error_pages, error_path

PACKAGE = ROOT / "src" / "mime"
LIST = "docs/ratchets/refusal_codes.txt"


def build(root, out) -> tuple[dict[str, str], str]:
    """The pages of the catalogue the real command writes for the checkout ``root`` in a fresh process (a registry other
    tests have planted entries in would add their pages), by path under ``pages/errors/``, and its summary."""
    done = docs_run("build", "--root", str(root), "--out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    return ({path.relative_to(out / "pages" / "errors").as_posix(): path.read_text(encoding="utf-8")
             for path in sorted((out / "pages" / "errors").glob("*.md"))}, done.stdout)


def test_every_code_has_a_catalogue_entry_and_the_listed_ones_say_no_test_names_them(tmp_path):
    listed = read_list(ROOT / LIST).items()
    assert listed <= set(REFUSAL_TABLE), f"the list holds codes the table does not: {sorted(listed - set(REFUSAL_TABLE))}"
    pages, summary = build(ROOT, tmp_path)
    assert set(pages) == {f"{code}.md" for code in REFUSAL_TABLE} | {"index.md"}
    for code in REFUSAL_TABLE:
        assert (NO_TEST in pages[f"{code}.md"]) == (code in listed), code
    assert re.search(rf"{NO_TEST} {len(listed)}\b", summary), summary
    assert all(f"[`{code}`]({code}.md) ({NO_TEST})" in pages["index.md"] for code in listed)
    assert "no test raises" not in "".join(pages.values()), "a test can raise a code it does not name"
    assert "never raises" in pages["index.md"], "the index says what the mark does not claim"


def test_a_planted_listed_code_says_it_and_an_unlisted_one_does_not():
    planted, other = sorted(REFUSAL_TABLE)[:2]
    pages = {page.path: page.text for page in error_pages(PACKAGE, {planted})}
    assert NO_TEST in pages[error_path(planted)] and NO_TEST not in pages[error_path(other)]
    assert not any(NO_TEST in page.text for page in error_pages(PACKAGE) if page.path != ERRORS_INDEX)


def test_a_line_deleted_from_a_copy_of_the_list_takes_the_sentence_off_that_page_and_no_other(tmp_path):
    copy = copy_checkout(tmp_path / "copy")
    listed = sorted(read_list(copy / LIST).items())
    gone, kept = listed[0], listed[1]
    path = copy / LIST
    path.write_text("".join(line + "\n" for line in path.read_text(encoding="utf-8").splitlines()
                            if line.split("#")[0].split() != [gone]), encoding="utf-8")
    pages, summary = build(copy, tmp_path / "out")
    assert NO_TEST not in pages[f"{gone}.md"] and NO_TEST in pages[f"{kept}.md"]
    assert re.search(rf"{NO_TEST} {len(listed) - 1}\b", summary), summary
