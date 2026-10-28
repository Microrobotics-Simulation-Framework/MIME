"""The citation check: every citation in a page resolves to an entry of ``docs/bibliography.bib``.

A citation is read by Pandoc's grammar, which the pages are written in: ``@key`` (in text), ``[@key]``,
``[@a; @b]``, ``[see @key, p. 3]`` and ``[-@key]`` (the author suppressed). A key begins with a letter, a digit or
``_`` and holds letters, digits, ``_`` and the internal punctuation ``:.#$%&-+?<>~/`` (one followed by a letter, a digit
or ``_``, so a full stop after a key ends the sentence and not the key); ``@{any key}`` braces a key that holds
anything else. An ``@`` after a word character is not one (``a@b.org``), and neither is any text in a code fence, an
inline code span, a link's target or a URL.

It reads the built pages (the inlined page sources among them) and every other hand-written ``.md`` under ``docs/``,
as the script it replaces read every one. That script's pattern read ``[@(\\w+)`` and so saw one key of ``[@a; @b]``,
none of ``@key``, ``[-@key]`` and ``[see @key]``, and the first word of a key with a hyphen. This counts the citations
checked and the citations not found, and fails on none at all when there are pages, which the old script passed."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from mime.docs._numbers import PAGE_DIRECTORIES, TOP_LEVEL_SOURCES, prose_lines

_KEY_START = r"[A-Za-z0-9_]"
_KEY = rf"{_KEY_START}(?:\w|[:.#$%&\-+?<>~/](?=\w))*"
_CITATION = re.compile(rf"(?<![\w@\\])-?@(?:\{{(?P<braced>[^{{}}\n]+)\}}|(?P<key>{_KEY}))")
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")
_LINK_TARGET = re.compile(r"\]\([^)\n]*\)")
_URL = re.compile(r"<?\b[a-z][a-z0-9+.-]*://[^\s<>)\]]+>?")
_BIB_ENTRY = re.compile(r"^\s*@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s{}]+)\s*,", re.MULTILINE)
#: BibTeX entries that are not references.
_NOT_REFERENCES = frozenset({"comment", "string", "preamble"})
BIBLIOGRAPHY = Path("docs/bibliography.bib")
DOCS = Path("docs")


@dataclass(frozen=True)
class Citation:
    """One citation: its key, and where it stands (a source's name and a line)."""

    key: str
    source: str
    line: int

    def where(self) -> str:
        return f"{self.source}:{self.line}"


def bibliography_keys(text: str) -> set[str]:
    """The keys of the references in the text of a ``.bib`` file."""
    return {match["key"] for match in _BIB_ENTRY.finditer(text) if match["type"].lower() not in _NOT_REFERENCES}


def hand_written_files(docs: Path) -> dict[str, str]:
    """The text of every hand-written Markdown file under ``docs`` that is not a page source (the sources are inlined in
    the pages the build makes, and read there): the README, the guides, a validation record. Named by its path under the
    repository, as ``docs/validation/README.md``. A file or a directory whose name starts with ``_`` is a template, and
    ``_build`` is the build's output: neither is read. This is the coverage ``scripts/check_citations.py`` had, which
    walked every ``.md`` under ``docs/``."""
    held: dict[str, str] = {}
    for path in sorted(docs.rglob("*.md")):
        relative = path.relative_to(docs)
        if any(part.startswith("_") for part in relative.parts) or relative.parts[0] in PAGE_DIRECTORIES \
                or relative.as_posix() in TOP_LEVEL_SOURCES:
            continue
        held[(DOCS / relative).as_posix()] = path.read_text(encoding="utf-8")
    return held


def citations_in(source: str, text: str) -> list[Citation]:
    """Each citation in ``text`` (the page named ``source``), in order, outside code and links."""
    found: list[Citation] = []
    for number, line in prose_lines(text):
        line = _LINK_TARGET.sub("](...)", _URL.sub(" ", _CODE_SPAN.sub(" ", line)))
        found += [Citation(match["braced"] or match["key"], source, number) for match in _CITATION.finditer(line)]
    return found


@dataclass
class CitationReport:
    """What the check found: every citation with the key it names, those the bibliography holds and those it does not."""

    problems: list[str]
    summary: list[str]
    checked: int = 0
    not_found: int = 0

    @property
    def total(self) -> int:
        return self.checked + self.not_found


def citation_report(sources: Mapping[str, str], keys: Collection[str], *, bibliography: Path = BIBLIOGRAPHY,
                    exempt: Collection[str] = ()) -> CitationReport:
    """Check the citations in ``sources`` (page text by name) against the bibliography's ``keys``. A source named in
    ``exempt`` is read and counted but not judged (a lab's pages, whose bibliography is its own). With sources and no
    citation in any, the check fails: a pass on nothing is not a pass."""
    report = CitationReport([], [])
    cited: set[str] = set()
    uncounted = 0
    for name in sorted(sources):
        found = citations_in(name, sources[name])
        if name in exempt:
            uncounted += len(found)
            continue
        for citation in found:
            cited.add(citation.key)
            if citation.key in keys:
                report.checked += 1
            else:
                report.not_found += 1
                report.problems.append(f"{citation.where()}: [@{citation.key}] is not an entry of {bibliography}; "
                                       f"add the work to the bibliography, or correct the key")
    if sources and report.total == 0:
        report.problems.append(f"{len(sources)} pages and sources were read and none holds a citation, so the "
                               f"check examined nothing; a page that cites no work is a page whose entry declares "
                               f"none, and MIME's entries cite at least one (is the parser reading the pages?)")
    report.summary.append(f"citations: {report.total} in {len(sources)} pages and sources, {report.checked} checked "
                          f"against {len(keys)} bibliography entries, {report.not_found} not found"
                          + (f"; {uncounted} in pages outside MIME, counted, not checked" if uncounted else "")
                          + f"; bibliography entries not cited: {len(set(keys) - cited)}")
    return report


__all__ = ["BIBLIOGRAPHY", "Citation", "CitationReport", "bibliography_keys", "citation_report", "citations_in",
           "hand_written_files"]
