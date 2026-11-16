"""``mime show <block folder>``: what a block is and what it needs, offline. Nothing is installed, downloaded or run.

**Not gated, by design.** ``show`` is the command that says the block's licence and its conditions, which a person needs
in order to decide whether to accept them; asking first would hide what the question is about. It reads the definition
and names what the fragment adds (its section names, counted), binds no socket, builds nothing and runs no case: the
gates guard where the block's content is first used (a load, ``verify``, ``pull``), never where it is described.

Its id, version, licence and tier; its record's title and creators where ``[record]`` reads; its content hash; each of
its requirements against what is installed, with the line that installs what is missing; its sockets; its asset
records with their tiers and whether each is cached; what it adds to an experiment, computed from its fragment and never
stored; its validity lines, what to cite with it, and its cases. Every text from the block is made plain before it is
printed: it is a stranger's. The lines are the output; a terminal draws the same words as tables
(:func:`mime.cli._display.show_block`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mime._terminal import plain
from mime.assets._cache import is_cached, size_text
from mime.blocks._definition import CASES, FRAGMENT_FILE, Definition, definition_of
from mime.blocks._fragment import read_fragment
from mime.build import BuildRefused
from mime.env._check import DistributionRow

NOTHING_RUN = "Nothing was installed, downloaded or run."


@dataclass(frozen=True)
class Shown:
    """A block as ``mime show`` says it: its head lines, its requirements against what is installed (and the line that
    installs what is missing), its sockets, its assets (name, size, cached, licence, tier), and its closing lines."""

    head: tuple[str, ...]
    requires: tuple[DistributionRow, ...]
    install: str | None
    sockets: tuple[tuple[str, str], ...]
    assets: tuple[tuple[str, str, str, str, str], ...]
    tail: tuple[str, ...]

    def lines(self) -> list[str]:
        lines = list(self.head)
        lines.append("requires (distribution, wanted, found, verdict):")
        lines += [f"  {plain(row.name)}  {plain(row.wanted)}  {plain(row.found or '-')}  {row.verdict}"
                  for row in self.requires]
        if self.install is not None:
            lines += ["! its requirements are not all met here: its cases cannot run, and it cannot be used",
                      f"  {self.install}"]
        lines.append("sockets (you bind these): " + (", ".join(f"{name} ({kind})" for name, kind in self.sockets)
                                                     or "none"))
        if self.assets:
            lines.append("assets (name, size, cached, licence, tier):")
            lines += [f"  {'  '.join(plain(cell) for cell in row)}" for row in self.assets]
        return [*lines, *self.tail]


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'s' if number != 1 else ''}"


def adds(doc: Mapping[str, Any]) -> str:
    """What a fragment adds to an experiment, in words: each kind of thing by name, laws, observations, a controller,
    declared values."""
    things = cast(Mapping[str, Mapping[str, Any]], doc.get("things", {}))
    parts = [f"{_count(sum(1 for t in things.values() if t.get('kind') == kind), kind)} "
             f"({', '.join(plain(name) for name, t in sorted(things.items()) if t.get('kind') == kind)})"
             for kind in ("structure", "medium", "body", "apparatus")
             if any(t.get("kind") == kind for t in things.values())]
    parts += [_count(len(doc.get("interactions", [])), "law")] if doc.get("interactions") else []
    parts += [_count(len(doc.get("observations", [])), "observation")] if doc.get("observations") else []
    parts += [f"a controller ({plain(str(doc['controller']['name']))})"] if doc.get("controller") else []
    parts += [_count(len(doc.get("declared", {})), "declared value")] if doc.get("declared") else []
    return "Adds " + (", ".join(parts) if parts else "nothing (an empty fragment)") + "."


def _asset_rows(definition: Definition) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for ref in definition.assets:
        cached = "cached" if is_cached(ref) else ("by request" if ref.by_request else "not cached")
        rows.append((ref.name, size_text(ref.size), cached, ref.licence, ref.tier))
    return rows


def _cases(folder: Path) -> list[str]:
    held = folder / CASES
    return sorted(each.name for each in held.iterdir() if each.is_dir()) if held.is_dir() else []


def shown(folder: str | Path) -> Shown:
    """What ``mime show`` says of the block folder ``folder``: refused (nothing shown) where its definition does not
    read, as ``mime check`` would find."""
    from mime.blocks._use import requirement_rows, requires_install_line

    root = Path(folder)
    definition = definition_of(root)
    verdict = definition.verdict
    conditions = f", {', '.join(verdict.conditions)}" if verdict.conditions else ""
    head = [f"block {plain(definition.id)} {plain(definition.version)}, {plain(definition.licence)} "
            f"({verdict.tier}{conditions}: {definition.tier} into the document that uses it)"]
    if definition.tier == "referenced":
        head[0] = (f"block {plain(definition.id)} {plain(definition.version)}, {plain(definition.licence)} (used by "
                   f"reference: its content stays in its container, never in the document that uses it)")
    record = _record(root)
    if record is not None:
        head.append(f"  {plain(record[0])}" + (f". Creators: {plain(record[1])}" if record[1] else ""))
    head.append(f"  content {definition.content_hash()}")
    rows = requirement_rows(definition)
    unmet = [row for row in rows if not row.agrees]
    tail: list[str] = []
    try:
        fragment = read_fragment(root, definition.sockets)
    except BuildRefused as refused:
        tail.append(f"! its fragment is refused ({', '.join(sorted({each.code for each in refused.refusals}))}): "
                    f"`mime check {root}` lists why")
    else:
        tail.append(adds(cast(Mapping[str, Any], fragment.doc)))
    tail += [f"Validity (the author's): {plain(each.line)} ({plain(each.reason)})" for each in definition.validity]
    if record is not None and record[2]:
        tail.append(f"Also cite: {plain(record[2])}")
    cases = _cases(root)
    tail.append(f"Cases: {len(cases)}" + (f" ({', '.join(plain(each) for each in cases)})" if cases else
                                          " (none yet: `mime pack` refuses a block with none)"))
    if definition.shows:
        tail.append(f"The author says its cases show: {plain(definition.shows)}")
    tail.append(NOTHING_RUN)
    return Shown(tuple(head), tuple(rows), requires_install_line(unmet) if unmet else None,
                 tuple(sorted(definition.sockets.items())), tuple(_asset_rows(definition)), tuple(tail))


def _record(root: Path) -> tuple[str, str, str] | None:
    """``[record]``'s title, creators and what to cite with it, as words, where it reads."""
    from mime.blocks._definition import read_definition

    definition, _ = read_definition(root)
    record = definition.record if definition is not None else None
    if record is None:
        return None
    creators = "; ".join(person.words() for person in record.creators)
    return record.title, creators, "; ".join(each.words() for each in record.also_cite)


def data(folder: str | Path) -> dict[str, object]:
    """``mime show --json`` of a block folder: the same facts as data."""
    held = shown(folder)
    definition = definition_of(Path(folder))
    return {"format": "mime.block-show/1", "id": definition.id, "version": definition.version,
            "licence": definition.licence, "tier": definition.tier, "content": definition.content_hash(),
            "requires": [{"distribution": row.name, "wanted": row.wanted, "found": row.found, "verdict": row.verdict}
                         for row in held.requires],
            "sockets": dict(held.sockets),
            "assets": [dict(zip(("name", "size", "cached", "licence", "tier"), row, strict=True))
                       for row in held.assets],
            "cases": _cases(Path(folder)), "fragment": FRAGMENT_FILE}


__all__ = ["NOTHING_RUN", "Shown", "adds", "data", "shown"]
