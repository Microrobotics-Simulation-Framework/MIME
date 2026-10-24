"""``mime docs check``: every check of the documentation's sources over a checkout of MIME, and the two compliance
scripts, unchanged. It builds no graph: the checks that need a built node (the ``builds`` comparison, a row's name and
units) run in the node walk, where the cases are built, and this checks those lists' form.

Each check fails naming what is wrong; the summary prints each list's length, every entry whose derivative is not
exact, and the counts the checks are over."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mime.build import RefusalClass, read_table
from mime.docs._export import completeness_problems, entry_record, load_namespaces
from mime.docs._lists import RatchetList, list_problems, read_list
from mime.docs._refusal_scan import CODE_SHAPE, derive, name_rule, scan_package
from mime.docs._symbols import FrozenId, read_frozen_ids, static_problems
from mime.gradients import GradientStatus
from mime.registry import (
    NOT_ESTABLISHED,
    Entry,
    EntryKind,
    GradientRecord,
    Interaction,
    NotEstablished,
    Registered,
    registered,
)
from mime.testing import family_named
from mime.vocabulary import names as interface_names

_KINDS = "|".join(kind.value.replace(" ", "_") for kind in EntryKind)
_NODE_ITEM = re.compile(r"[A-Za-z_][A-Za-z0-9_]* \S+")
_ENTRY = rf"(?:{_KINDS}) [a-z][a-z0-9_.]* \S+"


def _form(pattern: str, what: str) -> Callable[[str], str | None]:
    compiled = re.compile(pattern)
    return lambda item: None if compiled.fullmatch(item) else f"an item is {what}"


#: Every list this check reads, with the form of its items.
LISTS: Mapping[str, Callable[[str], str | None]] = {
    "refusal_codes": _form(CODE_SHAPE.pattern, "a refusal code"),
    "refusal_names": _form(CODE_SHAPE.pattern, "a code-shaped name"),
    "symbol_rows": _form(_NODE_ITEM.pattern, "a node class's name and a row's LaTeX"),
    "symbol_indices": _form(_NODE_ITEM.pattern, "a node class's name and a symbol"),
    "symbol_units": _form(_NODE_ITEM.pattern, "a node class's name and a row's LaTeX"),
    "builds_unchecked": _form(rf"{_ENTRY}(?: \S+)?", "an entry's kind, name and version, then its rung if it has one"),
    "not_established": _form(rf"{_ENTRY} (?:gradient\.(?:where|reason|route)|references)",
                             "an entry's kind, name and version, then gradient.where, .reason, .route or references"),
}
FROZEN_IDS = Path("docs/data/node_ids_0.2.yaml")
BIBLIOGRAPHY = Path("docs/bibliography.bib")
PACKAGE = Path("src/mime")
TESTS = Path("tests")
SCRIPTS = ("scripts/check_citations.py", "scripts/check_anomalies.py")


def list_path(root: Path, name: str) -> Path:
    return root / "docs" / "ratchets" / f"{name}.txt"


def read_lists(root: Path) -> dict[str, RatchetList]:
    return {name: read_list(list_path(root, name), form) for name, form in LISTS.items()}


def entry_item(record: Registered, tail: str = "") -> str:
    """An entry as the lists name it: its kind (with ``_`` for a space), name and version, then ``tail``."""
    return " ".join(part for part in (record.kind.value.replace(" ", "_"), record.name, record.version, tail) if part)


def _mimes(record: Registered) -> bool:
    return record.cls.__module__.split(".")[0] == "mime"


@dataclass
class Report:
    problems: list[str] = field(default_factory=list[str])
    summary: list[str] = field(default_factory=list[str])

    def text(self) -> str:
        lines = [*self.summary, ""]
        lines += [f"FAIL: {problem}" for problem in self.problems] or ["every check passed"]
        return "\n".join(lines)


def _built_rungs(cls: type[Entry]) -> set[str]:
    if issubclass(cls, Interaction) and cls.rungs:
        return set(cls.rungs) - set(cls.reserved_rungs)
    return {""}


def entry_problems(records: list[Registered], not_established: RatchetList, report: Report, *,
                   bibliography: Collection[str]) -> None:
    """The four docs-driven fields of every entry: ``builds`` keyed by each rung it builds, a gradient record for each
    status but ``exact`` (with its reason and route where it is ``cut`` or ``unverified``), ``references`` as keys,
    each of a MIME entry's in ``bibliography``, and an interaction's ``family`` registered; and every value of MIME's
    declared not established, listed. Another namespace's values not established, and its keys, are counted and not
    checked: its list and its bibliography are its own."""
    hits: set[str] = set()
    outside: list[str] = []
    keys = {True: 0, False: 0}
    unbuilt: list[str] = []
    inexact: list[tuple[GradientStatus, str]] = []
    for record in records:
        cls, label, own = record.cls, f"{record.kind.value} {record.name} {record.version}", _mimes(record)
        builds = dict(cls.builds)
        if not builds:
            if own:
                report.problems.append(f"{label}: it declares no builds; declare each rung's node classes, "
                                       f"builds = {{{', '.join(repr(r) for r in sorted(_built_rungs(cls)))}: (...)}}")
            else:
                unbuilt.append(label)
        elif set(builds) != _built_rungs(cls):
            report.problems.append(f"{label}: builds names the rungs {sorted(builds)}, and it builds "
                                   f"{sorted(_built_rungs(cls))} ('' for an entry without rungs)")
        for rung, nodes in builds.items():
            held = cast(object, nodes)  # an entry's own declaration, checked like a script's
            if not (isinstance(held, tuple) and all(isinstance(n, type) for n in cast(tuple[object, ...], held))):
                report.problems.append(f"{label}: builds[{rung!r}] is not a tuple of node classes")
        status, record_ = cls.differentiability, cls.gradient
        if status is GradientStatus.EXACT and record_ is not None:
            report.problems.append(f"{label}: it is exact and declares a gradient record; an exact entry has none")
        if status is not GradientStatus.EXACT:
            if not isinstance(record_, GradientRecord):
                report.problems.append(f"{label}: it is {status.value} and declares no gradient record (where, "
                                       f"reason, route to exact)")
            else:
                inexact.append((status, f"{label}: {status.value}; where: {_said(record_.where)}; why: "
                                        f"{_said(record_.reason)}; route to exact: {_said(record_.route)}"))
                if status in (GradientStatus.CUT, GradientStatus.UNVERIFIED) and \
                        NOT_ESTABLISHED in (record_.reason, record_.route):
                    report.problems.append(f"{label}: it is {status.value}, and a {status.value} entry states its "
                                           f"reason and its route to exact")
                unknown = {entry_item(record, f"gradient.{name}") for name in ("where", "reason", "route")
                           if getattr(record_, name) is NOT_ESTABLISHED}
                if own:
                    hits |= unknown
                else:
                    outside += sorted(unknown)
        references = cls.references
        if isinstance(references, NotEstablished):
            if own:
                hits.add(entry_item(record, "references"))
            else:
                outside.append(entry_item(record, "references"))
        elif not all(isinstance(key, str) and key and len(key.split()) == 1
                     for key in cast(tuple[object, ...], references)):
            report.problems.append(f"{label}: references {references!r} are not bibliography keys")
        else:
            keys[own] += len(references)
            if own:
                report.problems += [f"{label}: references {key!r}, which docs/bibliography.bib has no entry for"
                                    for key in references if key not in bibliography]
        if issubclass(cls, Interaction) and cls.family is not None:
            try:
                family_named(cls.family)
            except KeyError as error:
                report.problems.append(f"{label}: family: {error.args[0]}")
    report.problems += list_problems(not_established, hits, "a value declared not established")
    report.summary.append(f"not established outside MIME: {len(outside)}"
                          + "".join(f"\n  - {item}" for item in outside))
    report.summary.append(f"references: {keys[True]} keys of MIME's entries, checked against docs/bibliography.bib; "
                          f"{keys[False]} of other namespaces' entries, counted, not checked (their bibliographies "
                          f"are their own)")
    report.summary.append(f"gradients: {len(inexact)} entries not exact"
                          + "".join(f"\n  - {line}" for _, line in inexact))
    cut = [line for status, line in inexact if status in (GradientStatus.CUT, GradientStatus.UNVERIFIED)]
    report.summary.append(f"gradients cut or unverified: {len(cut)}" + "".join(f"\n  - {line}" for line in cut))
    if unbuilt:
        report.summary.append(f"builds not declared, outside MIME: {len(unbuilt)} ({', '.join(unbuilt)})")


def _said(value: str | NotEstablished) -> str:
    return "not established" if isinstance(value, NotEstablished) else value


def node_classes(records: list[Registered]) -> set[type]:
    """The node classes the entries' ``builds`` name: the nodes the docs walk."""
    return {node for record in records for nodes in dict(record.cls.builds).values() for node in nodes}


_NAMED = r"(?<![A-Za-z0-9_.-]){}(?![A-Za-z0-9_-]|\.[A-Za-z0-9_])"


def unnamed_codes(codes: list[str], tests: Path) -> set[str]:
    """The codes no test file's text names."""
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(tests.rglob("*.py")))
    return {code for code in codes if not re.search(_NAMED.format(re.escape(code)), text)}


def refusal_problems(root: Path, lists: Mapping[str, RatchetList], entries: set[str], interfaces: set[str],
                     report: Report) -> None:
    """The refusal table against the scan of the source: a row for every code constructed and none for a code that
    is not, each with the classes it is constructed in; every code-shaped literal a code or a name; and the codes no
    test names, listed."""
    scan = scan_package(root / PACKAGE)
    derived = derive(scan, name_rule(entries, interfaces), lists["refusal_names"].items())
    table_file = root / PACKAGE / "build" / "refusal_table.txt"
    try:
        table = read_table(table_file.read_text(encoding="utf-8"))
    except ValueError as error:
        report.problems.append(f"{table_file}: {error}")
        return
    order = [RefusalClass.USER, RefusalClass.ENVIRONMENT, RefusalClass.INTERNAL]

    def row(code: str) -> str:
        return f"{code}  {' '.join(c.value for c in order if c in derived.codes[code])}"

    for code in sorted(set(derived.codes) - set(table)):
        report.problems.append(f"{code}: constructed at {', '.join(derived.where[code])} and not in {table_file}; "
                               f"add the row '{row(code)}'")
    for code in sorted(set(table) - set(derived.codes)):
        report.problems.append(f"{table_file}: {code} is constructed nowhere in the source; delete its row")
    for code in sorted(set(table) & set(derived.codes)):
        if table[code] != derived.codes[code]:
            constructed = sorted(c.value for c in derived.codes[code])
            report.problems.append(f"{table_file}: {code} is constructed as {constructed} and the table says "
                                   f"{sorted(c.value for c in table[code])}; write '{row(code)}'")
    for site in derived.unknown_class:
        report.problems.append(f"{site.where}: {site.callee}({site.code!r}) names no class and the scan knows none "
                               f"for it; pass a RefusalClass, or add the helper to the scan's implied classes")
    for site in scan.built:
        what = "a (code, message, action) tuple" if site.form == "tuple" else f"{site.callee}(...)"
        report.problems.append(f"{site.where}: {what} builds its code where it is used, so the scan cannot read it; "
                               f"write each code it can be as a literal (a lookup in a mapping of literal codes)")
    unclassified = {literal.text: literal for literal in derived.unclassified}
    for text, literal in sorted(unclassified.items()):
        report.problems.append(f"{literal.path}:{literal.line}: {text!r} is shaped like a refusal code and the scan "
                               f"finds it neither constructed nor a name; if it is a name, add '{text}  # <what it "
                               f"names>' to {lists['refusal_names'].path}")
    names = lists["refusal_names"]
    report.problems += list(names.problems)
    for line in names.lines:
        if line.item not in derived.listed:
            report.problems.append(f"{names.path}:{line.number}: {line.item!r} is no longer a code-shaped literal of "
                                   f"the source that needs the list; delete this line")
    unnamed = unnamed_codes(sorted(table), root / TESTS)
    report.problems += list_problems(lists["refusal_codes"], unnamed, "a refusal code no test names")
    report.summary.append(f"refusal table: {len(table)} codes; {len(unnamed)} that no test names")


def refusal_check(root: Path) -> Report:
    """The refusal checks alone over the checkout at ``root``."""
    report = Report()
    load_namespaces()
    refusal_problems(root, read_lists(root), {record.name for record in registered()}, set(interface_names()), report)
    return report


def bibliography_keys(root: Path) -> set[str]:
    """The keys of the checkout's bibliography, read by ``scripts/check_citations.py``'s own reader; none when the
    checkout has no bibliography."""
    spec = importlib.util.spec_from_file_location("mime_docs_check_citations", root / SCRIPTS[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"{root / SCRIPTS[0]} cannot be loaded")
    script: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    path = root / BIBLIOGRAPHY
    return {str(key) for key in script.extract_bib_keys(str(path))} if path.exists() else set()


def run_scripts(root: Path, report: Report) -> None:
    for script in SCRIPTS:
        done = subprocess.run([sys.executable, script], cwd=root, capture_output=True, text=True, check=False)
        output = (done.stdout + done.stderr).strip().splitlines()
        if done.returncode != 0:
            report.summary.append(f"{script}: exit {done.returncode}\n" + "\n".join(output))
            report.problems.append(f"{script} failed (exit {done.returncode}); its output is above")
        else:
            report.summary.append(f"{script}: {output[-1] if output else 'no output'}")


def check(root: Path) -> Report:
    """Every check over the checkout at ``root``, with MIME and every installed namespace loaded."""
    report = Report()
    missing = [str(root / path) for path in (PACKAGE, TESTS, FROZEN_IDS, *map(Path, SCRIPTS))
               if not (root / path).exists()]
    if missing:
        report.problems.append(f"{root} is not a checkout of MIME: {', '.join(missing)} missing; run it from the "
                               f"repository's root, or pass --root")
        return report
    not_loaded = load_namespaces()
    records = registered()
    lists = read_lists(root)
    by_namespace: dict[str, int] = {}
    for record in records:
        by_namespace[record.namespace] = by_namespace.get(record.namespace, 0) + 1
        report.problems += completeness_problems(record.cls, entry_record(record))
    report.summary.append(f"registry: {len(records)} entries ("
                          + ", ".join(f"{name} {count}" for name, count in sorted(by_namespace.items()))
                          + f"); not loaded: {', '.join(sorted(not_loaded)) or 'none'}")
    entry_problems(records, lists["not_established"], report, bibliography=bibliography_keys(root))
    nodes = node_classes(records)
    frozen: dict[str, FrozenId] = {}
    try:
        frozen = read_frozen_ids(root / FROZEN_IDS)
    except ValueError as error:
        report.problems.append(str(error))
    tables = static_problems(nodes, rows=lists["symbol_rows"], indices=lists["symbol_indices"], frozen=frozen)
    report.problems += list(tables.problems)
    report.summary.append(f"nodes the entries build: {len(nodes)}; with NodeMeta and a symbol table: "
                          f"{len(tables.documented)}; undocumented: {len(tables.undocumented)}")
    documented = set(tables.documented)
    for name in ("symbol_units", "builds_unchecked"):
        report.problems += list(lists[name].problems)
    for line in lists["symbol_units"].lines:
        if line.item.split(" ", 1)[0] not in documented:
            report.problems.append(f"{lists['symbol_units'].path}:{line.number}: {line.item!r} names no node with "
                                   f"a symbol table; delete this line")
    entries = {record.name for record in records}
    interfaces = set(interface_names())
    refusal_problems(root, lists, entries, interfaces, report)
    report.summary.append("lists: " + ", ".join(f"{name} {len(held)}" for name, held in lists.items()))
    run_scripts(root, report)
    return report


def run_check(root: Path) -> int:
    started = time.perf_counter()
    report = check(root.resolve())
    print(report.text())
    print(f"mime docs check: {'failed' if report.problems else 'passed'} in {time.perf_counter() - started:.1f} s")
    return 1 if report.problems else 0


__all__ = ["BIBLIOGRAPHY", "LISTS", "Report", "bibliography_keys", "check", "entry_item", "list_path", "node_classes",
           "read_lists", "refusal_check", "run_check", "unnamed_codes"]
