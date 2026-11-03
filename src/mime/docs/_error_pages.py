"""A page for every code of MIME's refusal table, generated from the table and the scan of the source: the code's
class, what its user can do about it, and each place MIME gives it, with what its message says happened and what its
action says to do, a value filled in from the experiment written ``{…}``. An index lists the codes by class. Every
error's last line names its page (``mime.build.page_of``)."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from mime.build import TABLE_FILE, RefusalClass, read_table
from mime.docs._markdown import Hatch, Page
from mime.docs._refusal_scan import Site, scan_package

#: Where the pages are, under the pages directory: ``errors/<code>.md``, and their index.
ERRORS_DIRECTORY = "errors"
ERRORS_INDEX = f"{ERRORS_DIRECTORY}/index.md"
#: The hand-written note on reading an error and reaching its full traceback, a page beside the concepts note.
READING_ERRORS = "errors.md"
#: What a page says of a code on the refusal list, and the hatch it counts: no file under ``tests/`` names the code,
#: which is all the list reads. A test can raise a code without naming it (two listed codes were found raised by passing
#: tests that assert something else), so the claim is about names and never about what is raised.
NO_TEST = "no test names this code"
_NO_TEST_TEXT = (f"**On the refusal list: {NO_TEST}**, since no file under `tests/` names it. A test may still raise "
                 f"it without naming it, and none asserts it: what its message and action say is read from the source "
                 f"below. The hook that records which codes a test raises has not landed (D2's), so until it does the "
                 f"list says names and never raises. The list (`docs/ratchets/refusal_codes.txt`) only shrinks.")

CLASS_TEXT: Mapping[RefusalClass, str] = {
    RefusalClass.USER: "the script can change it: the error says what happened in the experiment's names, and what "
                       "to write instead",
    RefusalClass.ENVIRONMENT: "something the run needs is missing or different where it runs (a package, a file, a "
                              "device, the network): the error names what to install, fetch or attach",
    RefusalClass.INTERNAL: "a check on MIME's own construction, which no supported script should reach: a MIME defect, "
                           "to report with the file the error names; or an error MIME raised with no code yet "
                           "(error.uncoded), keeping its own class, to report so that it gets one",
    RefusalClass.UNRECOGNISED: "an error from below (the script's own code, JAX, MADDENING or another library) that "
                               "MIME has no translation for: it is shown at the script's own line, and is never called "
                               "MIME's defect",
}
_ORDER = (RefusalClass.USER, RefusalClass.ENVIRONMENT, RefusalClass.INTERNAL, RefusalClass.UNRECOGNISED)


def error_path(code: str) -> str:
    return f"{ERRORS_DIRECTORY}/{code}.md"


def _quoted(texts: Sequence[str] | None) -> str:
    if texts is None:
        return "(built where it is given, so not read here)"
    if not texts:
        return "(none)"
    return " or ".join(f"`{' '.join(text.replace('{}', '{…}').replace('`', chr(39)).split())}`" for text in texts)


def _places(sites: Sequence[Site]) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for site in sorted(sites, key=lambda s: (s.path, s.line)):
        said, do = _quoted(site.messages), _quoted(site.actions)
        if (site.path, said, do) in seen:
            continue
        seen.add((site.path, said, do))
        lines += [f"- in `mime/{site.path}`:", f"  - what happened: {said}", f"  - what to do: {do}"]
    return lines


def error_page(code: str, classes: frozenset[RefusalClass], sites: Sequence[Site], *, unraised: bool = False) -> Page:
    """The page of ``code``; ``unraised`` where the refusal list holds it, no test naming it."""
    held = [c for c in _ORDER if c in classes]
    lines = [f"# The error `{code}`", ""]
    lines += [f"**{c.value}**: {CLASS_TEXT[c]}." for c in held] + [""]
    if unraised:
        lines += [_NO_TEST_TEXT, ""]
    if sites:
        lines += ["Where MIME gives it, with what its message says happened and what its action says to do (`{…}` is "
                  "filled in from the experiment):", ""]
        lines += _places(sites)
    else:
        lines.append("MIME gives it where its code is passed by name, so its words are read where it is raised.")
    lines += ["", f"Reading an error and reaching its full traceback: [reading an error](../{READING_ERRORS}). Every "
                  f"code: [the index](index.md)."]
    return Page(error_path(code), "\n".join(lines) + "\n", hatches=(Hatch(NO_TEST, f"code {code}"),) if unraised else ())


def index_of_errors(table: Mapping[str, frozenset[RefusalClass]], unraised: Collection[str] = ()) -> Page:
    lines = ["# MIME's errors", "", "Every code MIME gives, by its class. Each error's last line names its page. A code "
             f"marked *{NO_TEST}* is one no file under `tests/` names; a test may raise it without naming it, and until "
             f"the hook that records what a test raises lands (D2's), the mark says names, never raises.", ""]
    for held in _ORDER:
        codes = sorted(code for code, classes in table.items() if held in classes)
        lines += [f"## {held.value}", "", CLASS_TEXT[held][:1].upper() + CLASS_TEXT[held][1:] + ".", ""]
        lines += [f"- [`{code}`]({code}.md)" + (f" ({NO_TEST})" if code in unraised else "") for code in codes] \
            or ["- none"]
        lines.append("")
    return Page(ERRORS_INDEX, "\n".join(lines))


def error_pages(package: Path, unraised: Collection[str] = ()) -> list[Page]:
    """A page for every code of the refusal table under ``package`` (``src/mime``), and their index. ``unraised`` is
    the codes of the refusal list, which no test names: their pages say so."""
    table = read_table((package / "build" / TABLE_FILE).read_text(encoding="utf-8"))
    by_code: dict[str, list[Site]] = {}
    for site in scan_package(package).sites():
        if site.code is not None:
            by_code.setdefault(site.code, []).append(site)
    pages = [error_page(code, classes, by_code.get(code, []), unraised=code in unraised)
             for code, classes in sorted(table.items())]
    return [*pages, index_of_errors(table, unraised)]


__all__ = ["CLASS_TEXT", "ERRORS_DIRECTORY", "ERRORS_INDEX", "NO_TEST", "READING_ERRORS", "error_page", "error_pages",
           "error_path", "index_of_errors"]
