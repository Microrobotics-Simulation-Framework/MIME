"""The docs walk: the hand-written half of the documentation, over a docs root. Every registered entry will have a
generated page, so "has a page" tests nothing; what the walk checks is

- (a) that a hand-written source exists where one is required: an entry's file, ``entries/<kind>/<name>.md``; a
  mechanism page for each target kind, ``mechanisms/<kind>.md``; a ``NodeMeta`` and a symbol table on each node class
  of MIME's that an entry builds; a card for each packaged asset, ``assets/<name>.md``;
- (b) that every page source's id resolves (its entry, target kind, node, family or asset exists), a relative link to
  a page source names one that does, and an entry's file names the registered version it describes in YAML front
  matter that parses. The links read are an inline link or image (``[x](target)``, with a title or the target in
  angle brackets), a reference definition (``[x]: target``) and an HTML ``href``; a link inside a code fence is
  read too.

MIME's namespace is walked in full. ``toy`` is exempt from the hand-written half by namespace, its entries being made
up for the tests; another namespace's entries and nodes without a source here are counted, not failed, unless it
opts in (``held``). The ledger, ``unpaged.txt``, lists what merged before the walk without its source, one item a
line with its reason: an item missing its source and not listed fails, and so does a listed item that has its source
now, so the ledger only shrinks."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

from mime.data import PACKAGED
from mime.docs._lists import RatchetList, read_list
from mime.docs._numbers import PAGE_DIRECTORIES
from mime.docs._symbols import meta_of, rows_of
from mime.registry import EntryKind, Registered
from mime.testing import family_named
from mime.vocabulary import MIME_NAMESPACE, TargetKind

LEDGER = "unpaged.txt"
#: The namespace whose entries are made up for MIME's own tests, and so need no hand-written source.
EXEMPT = frozenset({"toy"})
_KIND_DIRECTORIES = {kind.value.replace(" ", "_"): kind for kind in EntryKind}
_MECHANISMS = {kind.value.replace(" ", "_"): kind for kind in TargetKind}
_LEDGER_ITEM = re.compile(rf"entry (?:{'|'.join(_KIND_DIRECTORIES)}) [a-z][a-z0-9_.]*|mechanism [a-z_]+|"
                          r"node [A-Za-z_][A-Za-z0-9_]*|asset [a-z][a-z0-9_]*")
#: The link forms the walk reads: an inline link or image's target (bare, or in angle brackets, a title after it), a
#: reference definition's, and an HTML ``href``.
_LINKS = (re.compile(r"\]\(\s*<([^>\n]+)>"), re.compile(r"\]\(\s*([^)\s<][^)\s]*)"),
          re.compile(r"^ {0,3}\[[^\]\n]+\]:\s*<?([^\s>]+)>?", re.MULTILINE),
          re.compile(r"\bhref\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE))


def ledger_form(item: str) -> str | None:
    return None if _LEDGER_ITEM.fullmatch(item) else \
        "an item is 'entry <kind> <name>', 'mechanism <target kind>', 'node <class>' or 'asset <name>'"


def kind_directory(kind: EntryKind) -> str:
    return kind.value.replace(" ", "_")


def entry_source(record: Registered) -> str:
    return f"entries/{kind_directory(record.kind)}/{record.name}.md"


def is_mimes_node(node: type) -> bool:
    """A node class of MIME's own: defined under ``mime``, outside ``mime.testing``, where the toys' nodes are."""
    module = node.__module__
    return module.split(".")[0] == "mime" and not (module == "mime.testing" or module.startswith("mime.testing."))


def is_documented(node: type) -> bool:
    return meta_of(node) is not None and bool(rows_of(node))


@dataclass(frozen=True)
class Required:
    """A hand-written source the walk requires: the ledger's item, what supplies it, and whether it is there."""

    item: str
    source: str
    present: bool


@dataclass
class WalkReport:
    problems: list[str] = field(default_factory=list[str])
    summary: list[str] = field(default_factory=list[str])
    required: list[Required] = field(default_factory=list[Required])


def _front_matter(text: str) -> object:
    """The YAML front matter of ``text`` (``None`` without one); a front matter that is not YAML raises
    ``yaml.YAMLError``."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return yaml.safe_load("\n".join(lines[1:end]))
    return None


def _described(text: str) -> list[object] | str | None:
    """What an entry's file says it describes, its front matter's ``describes`` as YAML read it (``None`` if it says
    nothing), or, when the front matter is not YAML, what is wrong with it."""
    try:
        held = _front_matter(text)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        at = f" (the file's line {mark.line + 2})" if mark is not None else ""
        return f"its front matter is not YAML{at}: {getattr(error, 'problem', None) or error}; fix it"
    described = cast(dict[object, object], held).get("describes") if isinstance(held, dict) else None
    values = cast(list[object], described) if isinstance(described, list) else [described]
    return values if described is not None and values else None


class _Ids:
    """What a page source's id can name: the registered entries, the target kinds, the nodes the entries build, the
    families and the packaged assets."""

    def __init__(self, records: Sequence[Registered], nodes: Iterable[type]) -> None:
        self.entries: dict[tuple[str, str], list[Registered]] = {}
        for record in records:
            self.entries.setdefault((kind_directory(record.kind), record.name), []).append(record)
        self.nodes = {node.__name__ for node in nodes}

    def problem(self, relative: Path) -> str | None:
        """Why the page source at ``relative`` (under the docs root) names nothing that exists, or ``None``."""
        parts = relative.parts
        if relative.suffix != ".md":
            return "is not a page source (a page source is a .md file)"
        if parts[0] == "entries":
            if len(parts) != 3 or parts[1] not in _KIND_DIRECTORIES:
                return "is not an entry's file, entries/<kind>/<name>.md, kind one of " + ", ".join(_KIND_DIRECTORIES)
            if (parts[1], relative.stem) not in self.entries:
                return f"names the {parts[1].replace('_', ' ')} {relative.stem}, which no entry registers"
            return None
        if len(parts) != 2:
            return f"is not a page source, {parts[0]}/<id>.md"
        stem = relative.stem
        if parts[0] == "mechanisms" and stem not in _MECHANISMS:
            return f"names the target kind {stem}, which is none of " + ", ".join(_MECHANISMS)
        if parts[0] == "nodes" and stem not in self.nodes:
            return f"names the node {stem}, which no entry builds"
        if parts[0] == "assets" and stem not in PACKAGED:
            return f"names the asset {stem}, which MIME does not package"
        if parts[0] == "families":
            try:
                family_named(stem)
            except KeyError:
                return f"names the family {stem}, which is not registered"
        return None


def page_sources(docs_root: Path, directories: Collection[str]) -> list[Path]:
    """Every file under the page-source directories of ``docs_root``, relative to it."""
    found = {path for directory in directories for path in (docs_root / directory).rglob("*") if path.is_file()}
    return sorted(path.relative_to(docs_root) for path in found)


def docs_walk(docs_root: Path, records: Sequence[Registered], nodes: Iterable[type], *,
              held: Collection[str] = (MIME_NAMESPACE,)) -> WalkReport:
    """The walk over the docs root ``docs_root``, for the registered entries ``records`` and the node classes their
    ``builds`` name (``nodes``). The namespaces in ``held`` are walked in full: MIME's, and a lab's that opts in."""
    report = WalkReport()
    built = sorted(set(nodes), key=lambda node: (node.__name__, node.__module__))
    owned = [record for record in records if record.namespace in held]
    if not owned:
        report.problems.append(f"the registry holds no entry of {', '.join(sorted(held))}, so the walk required no "
                               f"entry's source and checked nothing")
    by_entry: dict[tuple[str, str], Registered] = {}
    for record in owned:
        by_entry.setdefault((kind_directory(record.kind), record.name), record)
    for (kind, name), record in sorted(by_entry.items()):
        report.required.append(Required(f"entry {kind} {name}", entry_source(record),
                                         (docs_root / entry_source(record)).is_file()))
    if MIME_NAMESPACE in held:
        for mechanism in sorted(_MECHANISMS):
            source = f"mechanisms/{mechanism}.md"
            report.required.append(Required(f"mechanism {mechanism}", source, (docs_root / source).is_file()))
        for asset in sorted(PACKAGED):
            source = f"assets/{asset}.md"
            report.required.append(Required(f"asset {asset}", source, (docs_root / source).is_file()))
    lab_built = {node for record in owned if record.namespace != MIME_NAMESPACE
                 for builds in dict(record.cls.builds).values() for node in builds}
    for node in built:
        if (MIME_NAMESPACE in held and is_mimes_node(node)) or node in lab_built:
            report.required.append(Required(f"node {node.__name__}", f"NodeMeta and a symbol table on "
                                            f"{node.__module__}.{node.__qualname__}", is_documented(node)))
    ledger = _ledger_problems(docs_root, report)
    ids = _Ids(records, built)
    sources = page_sources(docs_root, PAGE_DIRECTORIES)
    for relative in sources:
        problem = ids.problem(relative)
        if problem is not None:
            report.problems.append(f"{docs_root / relative}: {problem}")
            continue
        text = (docs_root / relative).read_text(encoding="utf-8")
        if relative.parts[0] == "entries":
            report.problems += _version_problems(docs_root / relative, text, ids.entries[relative.parts[1],
                                                                                         relative.stem])
        for target in (target for link in _LINKS for target in link.findall(text)):
            path = target.split("#", 1)[0]
            if not path.endswith(".md") or re.match(r"[a-z][a-z0-9+.-]*:", path):
                continue
            resolved = (docs_root / relative).parent.joinpath(path).resolve()
            try:
                linked = resolved.relative_to(docs_root.resolve())
            except ValueError:
                continue
            if linked.parts and linked.parts[0] in PAGE_DIRECTORIES and (why := ids.problem(linked)) is not None:
                report.problems.append(f"{docs_root / relative}: its link {target!r} {why}")
    outside: dict[str, list[str]] = {}
    for record in records:
        if record.namespace not in held and record.namespace not in EXEMPT and \
                not (docs_root / entry_source(record)).is_file():
            outside.setdefault(record.namespace, []).append(f"{record.kind.value} {record.name} {record.version}")
    undocumented = sorted(node.__name__ for node in built if not is_mimes_node(node) and node not in lab_built
                          and not is_documented(node) and not _exempt_node(node, records))
    exempt = sorted({record.name for record in records if record.namespace in EXEMPT})
    with_source = sum(1 for required in report.required if required.present)
    report.summary += [
        f"docs walk ({docs_root}): {len(report.required)} hand-written sources required of "
        f"{', '.join(sorted(held))} ({_counts(report.required)}); {with_source} present; page sources: "
        f"{len(sources)}",
        f"ledger ({LEDGER}): {len(ledger)} lines",
        f"exempt from the hand-written half by namespace ({', '.join(sorted(EXEMPT))}): {len(exempt)} entries",
        "outside MIME, without a hand-written source here, counted, not failed: "
        + ("; ".join(f"{namespace} {len(found)} ({', '.join(found)})" for namespace, found in sorted(outside.items()))
           or "none")
        + f"; nodes without NodeMeta: {len(undocumented)}" + (f" ({', '.join(undocumented)})" if undocumented else ""),
    ]
    return report


def _exempt_node(node: type, records: Sequence[Registered]) -> bool:
    """A node only exempt namespaces' entries build."""
    builders = [record for record in records for builds in dict(record.cls.builds).values() if node in builds]
    return bool(builders) and all(record.namespace in EXEMPT for record in builders)


def _counts(required: Sequence[Required]) -> str:
    kinds: dict[str, int] = {}
    for item in required:
        kind = item.item.split(" ", 1)[0]
        kinds[kind] = kinds.get(kind, 0) + 1
    plural = {"entry": "entries"}
    return ", ".join(f"{count} {plural.get(kind, kind + 's') if count != 1 else kind}" for kind, count in kinds.items())


def _version_problems(path: Path, text: str, records: list[Registered]) -> list[str]:
    described = _described(text)
    versions = sorted(record.version for record in records)
    name = records[0].name
    if described is None:
        return [f"{path}: names no version it describes; start it with front matter, '---', 'describes: "
                f"{versions[-1]}', '---'"]
    if isinstance(described, str):
        return [f"{path}: {described}"]
    unquoted = [value for value in described if not isinstance(value, str)]
    if unquoted:
        return [f"{path}: describes {unquoted[0]!r}, which YAML reads as a {type(unquoted[0]).__name__}, not a "
                f"version; quote it, as 'describes: \"{versions[-1]}\"'"]
    found = [f"{path}: describes {name} {version}, which is not registered (registered: {', '.join(versions)}); "
             f"describe the registered version, keeping the old one's history in the text"
             for version in described if version not in versions]
    found += [f"{path}: does not name {name} {version}, the registered version; describe it"
              for version in versions if version not in described]
    return found


def _ledger_problems(docs_root: Path, report: WalkReport) -> RatchetList:
    ledger = read_list(docs_root / LEDGER, ledger_form)
    report.problems += list(ledger.problems)
    by_item = {required.item: required for required in report.required}
    listed = ledger.items()
    for required in report.required:
        if not required.present and required.item not in listed:
            report.problems.append(f"{required.item}: its hand-written source ({required.source}) is missing and "
                                   f"{ledger.path} does not list it; write it (the ledger holds what merged before "
                                   f"the walk, and a line added needs its reason logged)")
    for line in ledger.lines:
        required = by_item.get(line.item)
        if required is None:
            report.problems.append(f"{ledger.path}:{line.number}: {line.item!r} is nothing the walk requires a "
                                   f"source for; delete this line")
        elif required.present:
            report.problems.append(f"{ledger.path}:{line.number}: {line.item!r} has its source now "
                                   f"({required.source}); delete this line")
    return ledger


__all__ = ["EXEMPT", "LEDGER", "Required", "WalkReport", "docs_walk", "entry_source", "is_documented",
           "is_mimes_node", "kind_directory", "ledger_form", "page_sources"]
