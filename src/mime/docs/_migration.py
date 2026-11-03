"""The 0.2 to 0.3 migration table: each public name of the 0.2 line, with the entry a 0.3 experiment writes in its
place or ``removed`` with the reason (``docs/data/migration_0.2.yaml``), and the checks that hold it.

The 0.2 names are a list frozen from the tag ``pre-0.3`` (``docs/data/names_0.2.txt``, written by
``scripts/names_0_2.py`` and held to its digest here), so a row naming anything else, a typo included, fails. Each name
has one row; a ``to`` names a registered entry by its kind and name, and a ``rung`` one of its rungs. The renames of
entries during 0.3's own development (``docs/ratchets/renames.txt``) are the table's second part."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from mime.registry import Registered

RECORD = Path("docs/data/migration_0.2.yaml")
NAMES = Path("docs/data/names_0.2.txt")
#: The digest of the frozen list, as scripts/names_0_2.py wrote it from the tag pre-0.3.
NAMES_SHA256 = "fdab13bc4c6220eedacf56a1ab88b34f1ad57d71fc3a9a255a35706dd343852e"
_FIELDS = frozenset({"name", "to", "rung", "note", "removed"})


@dataclass(frozen=True)
class Row:
    name: str
    module: str
    to: str | None
    rung: str | None
    note: str | None
    removed: str | None


def frozen_names(text: str) -> dict[str, str]:
    """The frozen list's names, each with its module."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            name, module = line.split()
            found[name] = module
    return found


def migration_report(root: Path, records: Iterable[Registered]) -> tuple[list[Row], list[str]]:
    """The table's rows over the checkout at ``root``, against the registered entries ``records``, and every way it
    breaks its rules."""
    problems: list[str] = []
    names_text = (root / NAMES).read_bytes() if (root / NAMES).is_file() else b""
    if hashlib.sha256(names_text).hexdigest() != NAMES_SHA256:
        problems.append(f"{NAMES}: it is not the list frozen from the tag pre-0.3 (its digest is not "
                        f"{NAMES_SHA256}); restore it, as scripts/names_0_2.py writes it")
    names = frozen_names(names_text.decode("utf-8"))
    try:
        data = yaml.safe_load((root / RECORD).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return [], [*problems, f"{RECORD}: cannot be read: {error}"]
    raw = cast(list[object], cast(dict[str, object], data).get("migration", []) if isinstance(data, dict) else [])
    entries: dict[str, set[str]] = {}
    for record in records:
        cls = record.cls
        rungs = set(cast(Iterable[str], getattr(cls, "rungs", ()) or ())) | set(
            cast(Iterable[str], getattr(cls, "reserved_rungs", ()) or ()))
        entries.setdefault(f"{record.kind.value.replace(' ', '_')} {record.name}", set()).update(rungs)
    rows: list[Row] = []
    for index, held in enumerate(raw):
        where = f"{RECORD}: row {index + 1}"
        if not isinstance(held, dict):
            problems.append(f"{where} is not a mapping of name, to or removed, rung and note")
            continue
        row = cast(Mapping[str, object], held)
        unknown = sorted(set(row) - _FIELDS)
        name = str(row.get("name"))
        if unknown:
            problems.append(f"{where} ({name}): unknown fields {unknown}")
        if name not in names:
            problems.append(f"{where}: {name!r} is not a name of the 0.2 line ({NAMES}, frozen from the tag pre-0.3)")
        to, removed = row.get("to"), row.get("removed")
        if (to is None) == (removed is None):
            problems.append(f"{where} ({name}): give either 'to', the 0.3 entry, or 'removed', the reason")
        if to is not None and str(to) not in entries:
            problems.append(f"{where} ({name}): 'to: {to}' names no registered entry (an entry's kind and name, as "
                            f"'interaction drag')")
        rung = row.get("rung")
        if rung is not None and (to is None or str(rung) not in entries.get(str(to), set())):
            problems.append(f"{where} ({name}): {to} has no rung {rung!r}")
        if removed is not None and not str(removed).strip():
            problems.append(f"{where} ({name}): 'removed' states its reason")
        rows.append(Row(name, names.get(name, ""), None if to is None else str(to),
                        None if rung is None else str(rung), None if row.get("note") is None else str(row["note"]),
                        None if removed is None else str(removed)))
    counted = Counter(row.name for row in rows)
    problems += [f"{RECORD}: {name} has {count} rows; a name has one" for name, count in sorted(counted.items())
                 if count > 1]
    problems += [f"{RECORD}: {name} ({module}) has no row; say what it is in 0.3, or why it was removed"
                 for name, module in sorted(names.items()) if name not in counted]
    return rows, problems


__all__ = ["NAMES", "NAMES_SHA256", "RECORD", "Row", "frozen_names", "migration_report"]
