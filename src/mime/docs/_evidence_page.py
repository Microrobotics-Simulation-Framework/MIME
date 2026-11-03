"""The evidence counts, v0: every claim of the anomaly registry and every normative statement of a specification, with
its tests' outcomes in the default lane's JUnit report, read through the resolver (``mime.docs._evidence``) and never
resolved again here.

**What the page is.** Counts and a table, not a verdict. A row's status is what its tests did in the default lane:
``passed``, ``failed``, ``skipped``, ``not run`` (marked ``skip``, ``xfail``, ``slow`` or ``gpu``: CI's default lane does
not run it), ``no test`` (a claim that is declared, or a statement not yet covered) or ``not read`` (no lane report was
read, the evidence was not resolved, or the test is one the lane runs and no report holds, as when a shard's report is
missing). Only ``passed`` counts as a test that ran and passed. A row with several tests (a bound's held-out test is one
of them) has the worst of their statuses. The device is declared by the workflow, ``cpu`` (``JAX_PLATFORMS`` in
``ci.yml``, which a test holds): a JUnit report records none, so every row that ran has the one value, and the page says
so; a row that did not run has none.

**What it bounds, and what it does not.** It bounds that every cited test is collected by name and what its outcome was
in a report. It does not check a test's body, and nothing ties a cited test to the claim that cites it; the page prints
that line under its counts. A bound, an estimate and an acceptance are kept in groups of their own, so an acceptance
test (agreement with a physical reference that grounds sign, frame and units and claims no accuracy) never prints under
an accuracy heading. A ``rule`` (a claim of that role, or a specification's statement) has no basis: it prints with its
cases (decision #222). The page is called a report only once it holds a claim with a ``measured`` basis.

**Held, so that the page cannot silently shrink or be edited.** The source (``docs/reference/evidence_counts.md``)
opens with the fixed sentence, exactly (decision M, #115). The floor (``docs/ratchets/evidence_floor.txt``) lists the
claims and statements that must stay present, a floor of the counts' N; against ``--base`` it may only grow, and an item
leaves it only with a reason logged in ``evidence_floor_removed.txt``. The page's own text is checked against the
claims and statements it was made from (:func:`page_problems`), and ends with a digest of itself, so a copy edited
by hand fails (:func:`generation_problems`); a generated page is never committed, and a copy under ``docs/`` fails.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mime.docs._evidence import (
    CLAIM_COLUMNS,
    DEFAULT_LANE,
    ROLES,
    Claim,
    Collection,
    Resolution,
    claim_rows,
    junit_outcomes,
    read_claims,
    resolve,
)
from mime.docs._lists import RatchetList, base_list, floor_problems, read_list
from mime.docs._markdown import Page, Writer, code, link, plural
from mime.docs._specification import SPECIFICATIONS, Statement, statements
from mime.docs._specification_pages import SAVED_FORM_PAGE

PAGE = "reference/evidence_counts.md"
SOURCE = Path("docs/reference/evidence_counts.md")
FLOOR = Path("docs/ratchets/evidence_floor.txt")
#: The floor's removals, one ``<item>  # <reason>`` line each: the only way a pinned item leaves the floor.
FLOOR_REMOVED = Path("docs/ratchets/evidence_floor_removed.txt")
REGISTRY = Path("docs/validation/known_anomalies.yaml")

#: Decision M (#115), word for word: the page opens with it, and a test holds it.
FIXED_SENTENCE = "This document is not a regulatory record or certificate; bounds and estimates are as labelled."
#: Printed under the counts: the empty-body and same-name-helper gap of the resolver, printed and not hidden.
BODY_LINE = ("A test's body is not checked: a cited test is matched by name, and nothing ties it to the claim or "
             "statement that cites it.")
#: The device of the default lane: ``JAX_PLATFORMS`` in ``ci.yml``'s workflow environment (a test holds it).
LANE_DEVICE = "cpu"
#: What the Device column of a row that ran says: the value is the workflow's declaration, not an observation.
DEVICE_NOTE = f"{LANE_DEVICE} (declared by the workflow)"
#: The last line of a page, before its digest.
DIGEST_LABEL = "Generation digest (sha256 of the text above):"

PASSED, FAILED, SKIPPED, NOT_RUN, NO_TEST, NOT_READ, MISSING = (
    "passed", "failed", "skipped", "not run", "no test", "not read", "missing")
#: The statuses a row can print, in the order the counts and the table give them.
STATUSES = (PASSED, FAILED, SKIPPED, NOT_RUN, NO_TEST, NOT_READ)
#: Which one a row with several tests takes: the first of these any test has.
_WORST = (MISSING, FAILED, NOT_RUN, SKIPPED, NOT_READ)
#: A lane's outcome of a test (``junit_outcomes``), as a status.
_STATUS_OF_OUTCOME = {"passed": PASSED, "failed": FAILED, "error": FAILED, "skipped": SKIPPED, "xfailed": NOT_RUN,
                      "not in the lane's report": NOT_READ}

#: The groups of claims, in the order the page gives them: the heading of each, and what the role means.
ROLE_GROUPS: Mapping[str, tuple[str, str]] = {
    "bound": ("Bounds", "An upper bound on an error over its stated domain."),
    "estimate": ("Estimates", "A value with an uncertainty that bounds nothing; a maximum over finitely many points "
                              "is one."),
    "acceptance": ("Acceptance tests", "Agreement with a physical reference that grounds sign, frame and units; it "
                                       "claims no accuracy."),
    "limit": ("Limits", "The stated edge of applicability."),
    "rule": ("Rules", "A normative sentence; it has no basis, only its cases."),
}
CLAIM_HEADING = "Claims"
STATEMENT_HEADING = "Specification statements"
COUNTS_HEADING = "Counts"
OUTCOME_COLUMNS = ("Outcome", "Device")
RULE_COLUMNS = ("Rule", "Quantity", "Outcome", "Device", "Its cases")
STATEMENT_COLUMNS = ("Statement", "Specification", "Outcome", "Device", "Its cases")
TABLE_COLUMNS = ("Group", "Rows", *STATUSES)


@dataclass(frozen=True)
class Inputs:
    """What the page's evidence is resolved with. ``resolve`` false resolves nothing (``--skip-evidence``: every row
    reads "not read"); otherwise the ids are resolved against ``collection``, or against a collection of the files the
    ids name when there is none, and ``reports`` are the default lane's JUnit reports (``None``: no lane report)."""

    resolve: bool = False
    collection: Collection | None = None
    reports: tuple[Path, ...] | None = None


def lane_reports(directory: Path) -> tuple[Path, ...]:
    """The default lane's JUnit reports under ``directory``: every ``*.xml`` in it, a sharded lane's reports together."""
    return tuple(sorted(directory.rglob("*.xml")))


def status_of(ids: Sequence[str], resolutions: Mapping[str, Resolution], *, resolved: bool) -> str:
    """What ``ids``' tests did in the default lane, the worst of them: ``no test`` for none, ``not read`` where nothing
    was resolved, no report was read or the report holds no such test (a lane's test no report has is not read, and is
    not "not run", the word for what CI does not run), ``not run`` for a test CI does not run, ``skipped`` and
    ``failed`` from the report, ``passed`` when every one did."""
    if not ids:
        return NO_TEST
    if not resolved:
        return NOT_READ
    states: list[str] = []
    for held in ids:
        found = resolutions[held]
        if found.problem is not None:
            states.append(MISSING)
        elif found.expected_to_fail or not found.runs_in_ci:
            states.append(NOT_RUN)
        elif found.outcomes is None:
            states.append(NOT_READ)
        else:
            states.append(_STATUS_OF_OUTCOME[dict(found.outcomes)[DEFAULT_LANE]])
    return next((state for state in _WORST if state in states), PASSED)


def device_of(status: str) -> str:
    """The device a row's outcome is for: the workflow's declaration where a report says it ran (a report records no
    device), ``not read`` where none was read, none otherwise."""
    if status in (PASSED, FAILED, SKIPPED):
        return DEVICE_NOTE
    return NOT_READ if status == NOT_READ else "none"


@dataclass(frozen=True)
class Row:
    """A claim or a statement with its evidence ids and their status."""

    key: str
    kind: str
    role: str
    ids: tuple[str, ...]
    status: str


def _cases(row_ids: Sequence[str], resolutions: Mapping[str, Resolution]) -> str:
    return "; ".join(resolutions[i].text() if i in resolutions else i for i in row_ids) or "none"


def title_of(claims: Sequence[Claim]) -> str:
    """Counts, until the page holds a claim with a ``measured`` basis; then a report."""
    return "Evidence report" if any(held.basis == "measured" for held in claims) else "Evidence counts"


def _count_line(noun: str, rows: Sequence[Row]) -> str:
    held = Counter(row.status for row in rows)
    return (f"{noun} {len(rows)}; with a test that ran and passed in the default lane {held[PASSED]}; not run "
            f"{held[NOT_RUN]}; failed {held[FAILED]}; skipped {held[SKIPPED]}; covered by no test {held[NO_TEST]}; "
            f"outcome not read {held[NOT_READ]}.")


def _lane_line(inputs: Inputs, outcomes: Mapping[str, str] | None, reports: int) -> str:
    if not inputs.resolve:
        return ("Evidence not resolved (`--skip-evidence`): no test was collected, so every row's outcome is not "
                "read.")
    if outcomes is None:
        return ("Outcome not read (no lane report): this build was given none, so no row says a test ran. CI builds "
                "this page with the default lane's report.")
    return (f"Outcomes are the default lane's, read from its JUnit reports ({plural(len(outcomes), 'test')} in "
            f"{plural(reports, 'report')}); "
            f"the workflow declares the lane's device, {LANE_DEVICE}, and a report records none, so the Device column "
            f"is that one declared value for every row that ran.")


def row_ids(held: Claim) -> tuple[str, ...]:
    """The tests a claim's row rests on: its evidence and, for a bound, its held-out test, which the row's status takes
    the worse of like any other."""
    return (*held.evidence, *((held.held_out,) if held.held_out is not None else ()))


def claim_evidence(claims: Sequence[Claim]) -> list[str]:
    """Every id the claims' rows rest on (:func:`row_ids`), in order, each once."""
    return list(dict.fromkeys(i for held in claims for i in row_ids(held)))


def build_page(claims: Sequence[Claim], specs: Mapping[Path, Sequence[Statement]], source: str,
               resolutions: Mapping[str, Resolution], inputs: Inputs, outcomes: Mapping[str, str] | None,
               floor: tuple[int, int]) -> str:
    """The page's text: its title, the source (the fixed sentence and the framing), the counts and their table, the
    claims by role, the specification's statements, and the page's digest. ``floor`` is the count of claims and of
    statements the floor list pins."""
    resolved = inputs.resolve
    claim_rows_ = [Row(held.key, "claim", held.role, row_ids(held), status_of(row_ids(held), resolutions,
                                                                             resolved=resolved)) for held in claims]
    statement_rows = [Row(statement.key, "statement", "rule", statement.cases,
                          status_of(statement.cases, resolutions, resolved=resolved))
                      for held in specs.values() for statement in held]
    w = Writer()
    w.heading(1, title_of(claims))
    w.add(source.strip("\n"), "")
    w.heading(2, COUNTS_HEADING)
    w.paragraph(_count_line("Claims", claim_rows_))
    w.paragraph(_count_line("Statements", statement_rows))
    w.paragraph(BODY_LINE)
    groups = [(ROLE_GROUPS[role][0], [r for r in claim_rows_ if r.role == role]) for role in ROLES]
    groups.append((STATEMENT_HEADING, statement_rows))
    w.table(TABLE_COLUMNS, [[name, str(len(rows)), *(str(sum(r.status == s for r in rows)) for s in STATUSES)]
                            for name, rows in groups])
    w.paragraph(f"The floor on those counts is pinned in `{FLOOR.as_posix()}`: at least {plural(floor[0], 'claim')} "
                f"and {plural(floor[1], 'statement')} must stay, and a pinned one that is missing fails the build.")
    w.paragraph(_lane_line(inputs, outcomes, len(inputs.reports or ())))
    if not any(held.basis == "measured" for held in claims):
        w.paragraph("It holds no claim with a measured basis, so it is a count and not a report.")
    w.heading(2, CLAIM_HEADING)
    by_key = {held.key: held for held in claims}
    for role in ROLES:
        heading, meaning = ROLE_GROUPS[role]
        w.heading(3, heading)
        w.paragraph(meaning)
        members = [r for r in claim_rows_ if r.role == role]
        if not members:
            w.paragraph("None.")
        elif role == "rule":
            w.table(RULE_COLUMNS, [[code(r.key), by_key[r.key].quantity, r.status, device_of(r.status),
                                    _cases(r.ids, resolutions)] for r in sorted(members, key=lambda r: r.key)])
        else:
            rows = claim_rows([by_key[r.key] for r in members], resolutions)
            states = {r.key: r.status for r in members}
            w.table((*CLAIM_COLUMNS, *OUTCOME_COLUMNS),
                    [[code(row[0]), *row[1:], states[row[0]], device_of(states[row[0]])] for row in rows])
    w.heading(2, STATEMENT_HEADING)
    w.paragraph("Rules: each has its cases and no basis. The statement is in the specification it names.")
    for path, held in sorted(specs.items()):
        if not held:
            continue
        status = {row.key: row.status for row in statement_rows}
        w.table(STATEMENT_COLUMNS,
                [[link(PAGE, SAVED_FORM_PAGE, code(s.key)), code(path.as_posix()), status[s.key],
                  device_of(status[s.key]), _cases(s.cases, resolutions) if s.cases else "not yet covered"]
                 for s in held])
    text = "\n".join(w.lines).rstrip("\n") + "\n"
    return text + f"\n{DIGEST_LABEL} `{digest(text)}`\n"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generation_problems(text: str) -> list[str]:
    """Whether the page is as it was generated: its last line holds the digest of the text above it, so a page edited
    by hand, or cut short, fails."""
    body, found, last = text.rstrip("\n").rpartition(f"\n{DIGEST_LABEL} ")
    if not found:
        return ["the page ends with no generation digest: it was not generated by mime docs build, or was cut short"]
    if last != f"`{digest(body.rstrip(chr(10)) + chr(10))}`":
        return ["the page is not what mime docs build generated (its digest does not match its text): it was edited "
                "by hand; build it again"]
    return []


def committed_copy_problems(docs: Path) -> list[str]:
    """A generated page is built and never committed: a file under ``docs/`` that ends as a generated page does is a
    copy, which a hand can edit, and fails (naming it, and saying whether it was also edited)."""
    found: list[str] = []
    for path in sorted(docs.rglob("*.md")):
        relative = path.relative_to(docs)
        if any(part.startswith("_") for part in relative.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if f"\n{DIGEST_LABEL} " in text:
            edited = generation_problems(text)
            found.append(f"{path}: a copy of a generated page; pages are built and never committed, delete it"
                         + (" (it is also edited by hand)" if edited else ""))
    return found


_ROW = re.compile(r"\| (?:\[)?`([^`]+)`")


def page_problems(text: str, claims: Sequence[Claim], specs: Mapping[Path, Sequence[Statement]]) -> list[str]:
    """The page's text against the claims and statements it was made from: it opens with the fixed sentence, the title
    says counts until a claim is measured, every claim is on it once and under the heading of its role, every statement
    once, and the digest matches. A fault here is a generator that is wrong, or a page edited."""
    found: list[str] = []
    lines = text.splitlines()
    title = title_of(claims)
    if not lines or lines[0] != f"# {title}":
        found.append(f"the page's title is not '{title}': it says counts until it holds a claim with a measured basis")
    elif lines[1:3] != ["", FIXED_SENTENCE]:
        found.append(f"the fixed sentence is not the page's first line after its title: {FIXED_SENTENCE!r}")
    heading_of_role = {role: heading for role, (heading, _) in ROLE_GROUPS.items()}
    section: str | None = None
    group: str | None = None
    seen: dict[str, list[str | None]] = {}
    for line in lines:
        if line.startswith("## "):
            section, group = line[3:], None
        elif line.startswith("### "):
            group = line[4:]
        elif section in (CLAIM_HEADING, STATEMENT_HEADING) and (match := _ROW.match(line)):
            seen.setdefault(match.group(1), []).append(group if section == CLAIM_HEADING else STATEMENT_HEADING)
    for held in claims:
        where = seen.get(held.key, [])
        if where != [heading_of_role[held.role]]:
            found.append(f"the claim {held.key} ({held.role}) is "
                         + (f"under {', '.join(repr(w) for w in where)}" if where else "not on the page")
                         + f", not once under '{heading_of_role[held.role]}'")
    for held_statements in specs.values():
        found += [f"the statement {s.key} is on the page {len(seen.get(s.key, []))} times, not once"
                  for s in held_statements if len(seen.get(s.key, [])) != 1]
    return found + generation_problems(text)


@dataclass
class EvidencePages:
    """What the evidence page's build made: its page (none where the checkout has no registry and no specification),
    the problems that stop a build, and the problems of resolving evidence, which ``mime docs check`` reports itself
    (the anomaly script and the specification check) and a build reports too."""

    pages: list[Page]
    problems: list[str]
    unresolved: list[str]
    summary: str = ""


def floor_form(item: str) -> str | None:
    """What is wrong with a floor item's form: it is 'claim <anomaly id>.<claim id>' or 'statement <statement id>'."""
    return None if re.fullmatch(r"(?:claim [A-Z]+-[A-Z]+-\d+\.[a-z][a-z0-9_]*|statement [a-z]+(?:[.-][a-z]+)*)",
                                item) else "an item is 'claim <anomaly id>.<claim id>' or 'statement <statement id>'"


def floor_base_problems(root: Path, base: str | None, floor: RatchetList, removed: RatchetList) -> tuple[list[str], str]:
    """The floor against the base revision's, which it may only grow from: an item on the base's floor and not on
    this one fails unless ``removed`` (``evidence_floor_removed.txt``) logs it with a reason, and an item logged as
    removed that is still on the floor fails. Also the summary line."""
    found = list(removed.problems)
    found += [f"{removed.path}:{line.number}: {line.item!r} is logged as removed from the floor and is still on it; "
              f"delete one of the two lines" for line in removed.lines if line.item in floor.items()]
    if base is None:
        return found, "evidence floor: not compared with a base revision (pass --base; CI passes origin/v0.3)"
    before, problem = base_list(root, base, FLOOR, floor_form)
    if problem is not None:
        return [*found, problem], f"evidence floor: not compared with {base}, which cannot be read"
    if before is None:
        return found, f"evidence floor: {base} holds no such list, so it is compared with nothing: this change adds it"
    gone = sorted(before.items() - floor.items() - removed.items())
    found += [f"{FLOOR}: the pinned {item!r} is on the floor at {base} and not here; the floor only grows: restore "
              f"the line, or log the removal as '{item}  # <reason>' in {FLOOR_REMOVED}" for item in gone]
    return found, (f"evidence floor: compared with {base}'s, {len(before)} lines there and {len(floor)} here, "
                   f"{len(before.items() - floor.items())} removed, {len(removed)} logged")


def evidence_pages(root: Path, inputs: Inputs) -> EvidencePages:
    """The evidence counts page of the checkout at ``root``, its evidence resolved as ``inputs`` says."""
    registry, source = root / REGISTRY, root / SOURCE
    problems: list[str] = []
    unresolved: list[str] = []
    claims: list[Claim] = []
    if registry.is_file():
        try:
            claims = sorted(read_claims(registry).values(), key=lambda held: held.key)
        except ValueError as error:
            problems.append(f"{REGISTRY}: {error}")
    specs: dict[Path, list[Statement]] = {}
    for relative in SPECIFICATIONS:
        if (root / relative).is_file():
            specs[relative], _ = statements((root / relative).read_text(encoding="utf-8"), relative)
    if not registry.is_file() and not specs:
        return EvidencePages([], problems, unresolved)
    if not source.is_file():
        problems.append(f"{SOURCE}: the evidence counts page's source is missing, and with it the page and its fixed "
                        f"sentence; restore it (the checkout holds a registry or a specification to count)")
        return EvidencePages([], problems, unresolved)
    text = source.read_text(encoding="utf-8")
    first = text.strip("\n").split("\n\n", 1)[0].strip()
    if first != FIXED_SENTENCE:
        problems.append(f"{SOURCE}: the page opens with the fixed sentence, exactly: {FIXED_SENTENCE!r}; it opens "
                        f"with {first[:120]!r}")
    floor: RatchetList = read_list(root / FLOOR, floor_form)
    present = {f"claim {held.key}" for held in claims} | {f"statement {s.key}" for held in specs.values()
                                                          for s in held}
    problems += floor_problems(floor, present, "claim or statement")
    pinned = (sum(item.startswith("claim ") for item in floor.items()),
              sum(item.startswith("statement ") for item in floor.items()))
    ids = claim_evidence(claims) + [case for held in specs.values() for s in held for case in s.cases]
    resolutions: dict[str, Resolution] = {}
    outcomes: Mapping[str, str] | None = None
    if inputs.resolve:
        if inputs.reports is not None:
            if not inputs.reports:
                problems.append("no JUnit report was found where the default lane's reports were said to be; a "
                                "build given a directory of them needs one")
                return EvidencePages([], problems, unresolved)
            outcomes = junit_outcomes(inputs.reports)
        lanes = {DEFAULT_LANE: inputs.reports} if inputs.reports else None
        resolutions = resolve(ids, root, inputs.collection, lanes=lanes)
        unresolved += [f"evidence counts: {found.text()}" for found in resolutions.values()
                       if found.problem is not None]
    if problems or unresolved:
        return EvidencePages([], problems, unresolved)
    page = build_page(claims, specs, text, resolutions, inputs, outcomes, pinned)
    wrong = page_problems(page, claims, specs)
    if wrong:
        return EvidencePages([], [f"evidence counts: {problem}" for problem in wrong], unresolved)
    held = Counter(status_of(row_ids(each), resolutions, resolved=inputs.resolve) for each in claims)
    kept = Counter(status_of(s.cases, resolutions, resolved=inputs.resolve) for each in specs.values() for s in each)
    summary = (f"evidence counts: {plural(len(claims), 'claim')} ({held[PASSED]} with a test that ran and passed in "
               f"the default lane), {plural(sum(len(each) for each in specs.values()), 'statement')} "
               f"({kept[PASSED]}), the floor {pinned[0]} and {pinned[1]}"
               + ("" if inputs.resolve else "; evidence not resolved (--skip-evidence)")
               + ("" if inputs.reports is not None or not inputs.resolve else "; outcomes not read (no lane report)"))
    return EvidencePages([Page(PAGE, page)], problems, unresolved, summary)


__all__ = ["BODY_LINE", "DEVICE_NOTE", "DIGEST_LABEL", "FIXED_SENTENCE", "FLOOR", "FLOOR_REMOVED", "LANE_DEVICE", "PAGE", "REGISTRY", "ROLE_GROUPS",
           "SOURCE", "STATUSES", "EvidencePages", "Inputs", "build_page", "committed_copy_problems", "device_of",
           "evidence_pages", "floor_base_problems", "floor_form", "generation_problems", "lane_reports", "page_problems", "status_of", "title_of"]
