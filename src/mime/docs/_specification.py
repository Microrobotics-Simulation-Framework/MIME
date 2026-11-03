"""The normative statements of a hand-written specification, and the check that each cites its cases.

A statement is a list item that opens with its id in bold brackets, ``- **[document.encoding]** ...``, runs to the
next item, heading or blank line, and ends with its cases: ``*Cases:*`` and the pytest node ids that hold it, each in
backticks, or ``*Cases:* not yet covered.`` Every other line is prose, and states nothing. The check fails on a
statement without its cases, on an id used twice, and on a case that does not resolve (``mime.docs._evidence``: a test
that is not collected fails). A statement is covered when one of its cases counts as evidence (``Resolution.counts``:
at collection, a test CI's lanes run); one whose every case is marked ``skip``, ``xfail``, ``slow`` or ``gpu`` cites
only tests CI does not run, and is not covered. The statements not covered are a ratchet's list
(``docs/ratchets/not_yet_covered.txt``), so that a new one is listed with its reason and one that gains a case CI runs
leaves the list."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mime.docs._evidence import Collection, lane_counts, resolve
from mime.docs._lists import RatchetList, list_problems

#: The hand-written specifications, under the checkout.
SPECIFICATIONS = (Path("docs/specification/saved_form.md"),)
STATEMENT_ID = re.compile(r"[a-z]+(?:[.-][a-z]+)*")
_OPENS = re.compile(r"- \*\*\[(?P<id>[^\]]*)\]\*\* ")
_CASES = re.compile(r"\*Cases:\* (?P<cases>.+)$")
_CASE = re.compile(r"`([^`]+)`")
NOT_YET_COVERED = "not yet covered."


@dataclass(frozen=True)
class Statement:
    key: str
    line: int
    text: str
    cases: tuple[str, ...]
    covered: bool


def statements(text: str, path: Path) -> tuple[list[Statement], list[str]]:
    """The statements of the specification ``text`` (read from ``path``), and every way one is malformed."""
    found: list[Statement] = []
    problems: list[str] = []
    held: list[tuple[int, str]] = []

    def close() -> None:
        if not held:
            return
        number, opening = held[0]
        body = " ".join(line.strip() for _, line in held)
        held.clear()
        match = _OPENS.match(opening)
        assert match is not None
        statement_id = match.group("id")
        if not STATEMENT_ID.fullmatch(statement_id):
            problems.append(f"{path}:{number}: the statement id {statement_id!r} is lower-case words joined by '.' "
                            f"or '-'")
        cases = _CASES.search(body)
        if cases is None:
            problems.append(f"{path}:{number}: the statement [{statement_id}] ends with no '*Cases:*': cite the tests "
                            f"that hold it, or write '*Cases:* {NOT_YET_COVERED}'")
            return
        written = cases.group("cases").strip()
        ids = tuple(_CASE.findall(written))
        covered = written != NOT_YET_COVERED
        if covered and (not ids or _CASE.sub("", written).strip(" ,.;") != ""):
            problems.append(f"{path}:{number}: the cases of [{statement_id}] are pytest node ids in backticks, "
                            f"separated by commas, or '{NOT_YET_COVERED}'")
        found.append(Statement(statement_id, number, body[:cases.start()].strip(), ids, covered))

    for number, line in enumerate(text.splitlines(), start=1):
        if _OPENS.match(line):
            close()
            held.append((number, line))
        elif held and line.strip() and not line.lstrip().startswith(("-", "#")) and line.startswith(" "):
            held.append((number, line))
        else:
            close()
    close()
    return found, problems


@dataclass
class SpecificationReport:
    problems: list[str]
    summary: list[str]
    statements: dict[Path, list[Statement]]


def specification_report(root: Path, collection: Collection | None,
                         not_yet_covered: RatchetList) -> SpecificationReport:
    """Every specification of the checkout at ``root``: its statements' form, their cases resolved against
    ``collection`` (not resolved where it is ``None``), and the statements not covered against their list: those not
    yet covered, and those whose every case is a test CI does not run."""
    problems: list[str] = []
    summary: list[str] = []
    every: dict[Path, list[Statement]] = {}
    uncovered: set[str] = set()
    unrun: dict[str, str] = {}
    for relative in SPECIFICATIONS:
        path = root / relative
        if not path.is_file():
            problems.append(f"{relative}: the specification is missing")
            continue
        held, malformed = statements(path.read_text(encoding="utf-8"), relative)
        every[relative] = held
        problems += malformed
        if not held:
            problems.append(f"{relative}: it holds no statement, and a specification states something")
        for statement_id, count in Counter(s.key for s in held).items():
            if count > 1:
                problems.append(f"{relative}: the statement id [{statement_id}] is used {count} times")
        cited = [case for s in held for case in s.cases]
        resolutions = resolve(cited, root, collection) if collection is not None else {}
        for statement in held:
            problems += [f"{relative}:{statement.line}: [{statement.key}] cites {resolutions[case].text()}"
                         for case in statement.cases if case in resolutions and resolutions[case].problem is not None]
        uncovered |= {s.key for s in held if not s.covered}
        for statement in held:
            found = [resolutions[case] for case in statement.cases if case in resolutions]
            if statement.covered and found and all(r.problem is None and not r.counts for r in found):
                unrun[statement.key] = (f"{relative}:{statement.line}: [{statement.key}] cites only tests CI does not "
                                        f"run ({'; '.join(r.text() for r in found)}), so it is not covered")
        resolved = (f"({lane_counts(resolutions.values())})" if collection is not None
                    else "(not resolved: --skip-evidence)")
        covered = sum(s.covered and s.key not in unrun for s in held)
        summary.append(f"{relative}: {len(held)} normative statements, {covered} covered by a test CI runs, citing "
                       f"{len(set(cited))} tests {resolved}; {sum(s.key in unrun for s in held)} citing only tests CI "
                       f"does not run; {sum(not s.covered for s in held)} not yet covered")
    listed = not_yet_covered.items()
    problems += [f"{text}: cite a test CI runs, or add the line '{key}  # <reason>' to {not_yet_covered.path}"
                 for key, text in sorted(unrun.items()) if key not in listed]
    problems += [problem for problem in list_problems(not_yet_covered, uncovered | set(unrun),
                                                      "a normative statement not yet covered")
                 if not any(problem.startswith(f"{key}: ") for key in set(unrun) - listed)]
    return SpecificationReport(problems, summary, every)


__all__ = ["NOT_YET_COVERED", "SPECIFICATIONS", "STATEMENT_ID", "SpecificationReport", "Statement",
           "specification_report", "statements"]
