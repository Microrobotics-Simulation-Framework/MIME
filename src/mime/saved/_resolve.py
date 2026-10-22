"""Finding a document's entries among installed packages, and nowhere else.

A package that registers entries declares a ``mime.entries`` entry point per namespace: its name is the namespace
and its value the module whose import registers the namespace's entries. A load imports those modules for the
namespaces a document uses, and never a module the document itself names, so a document cannot make a load run code
of its choosing. MIME's own namespace needs no entry point: its entries register when MIME is imported.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from mime.build import Refusal, RefusalClass
from mime.registry import EntryKind, Registered, lookup, versions_of
from mime.saved._schema import EntryRecord
from mime.vocabulary import MIME_NAMESPACE

ENTRY_POINT_GROUP = "mime.entries"
MIME_DISTRIBUTION = "mime-engine"


def namespace_providers() -> dict[str, importlib.metadata.EntryPoint]:
    return {point.name: point for point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}


@dataclass(frozen=True)
class Provenance:
    """Where an entry came from: the distribution that provides its namespace, and that distribution's version."""

    distribution: str
    version: str


def provenance_of(record: Registered) -> Provenance | str:
    """The distribution behind a registered entry, or why a load elsewhere could not find it."""
    if record.namespace == MIME_NAMESPACE:
        return Provenance(MIME_DISTRIBUTION, importlib.metadata.version(MIME_DISTRIBUTION))
    point = namespace_providers().get(record.namespace)
    if point is None:
        return (f"no installed package declares a {ENTRY_POINT_GROUP!r} entry point for namespace "
                f"{record.namespace!r}, so a load could not find {record.name}; declare one in the package that "
                f"registers it: [project.entry-points.\"{ENTRY_POINT_GROUP}\"] {record.namespace} = \"<module>\"")
    package = point.value.split(".", 1)[0].split(":", 1)[0]
    module = record.cls.__module__
    if module.split(".", 1)[0] != package:
        return (f"{record.name} is defined in module {module}, outside {package}, the package whose entry point "
                f"provides namespace {record.namespace!r}; a load imports only {point.value}, so it would not find "
                f"it")
    dist = point.dist
    if dist is None:
        return f"the entry point for namespace {record.namespace!r} belongs to no distribution"
    return Provenance(dist.metadata["Name"], dist.version)


def install_line(kind: str, name: str, version: str, locked: Iterable[EntryRecord]) -> str:
    """What to install to provide an entry, from the lock's record of where it came from."""
    for record in locked:
        if (record["kind"], record["name"], record["version"]) == (kind, name, version):
            return (f"install the package it came from: pip install "
                    f"'{record['distribution']}=={record['distribution_version']}'")
    return (f"install the package that registers {name} {version}: its namespace is "
            f"{name.split('.', 1)[0]!r}, and the lock names no distribution for it")


def import_namespaces(names: Iterable[str], locked: Iterable[EntryRecord]) -> list[Refusal]:
    """Import the provider of every namespace the entry ``names`` may belong to (the first word of each name).
    Never installs anything: a provider that fails to import is a refusal with the line to install."""
    providers = namespace_providers()
    refusals: list[Refusal] = []
    locked = list(locked)
    for word in sorted({name.split(".", 1)[0] for name in names}):
        point = providers.get(word)
        if point is None:
            continue
        try:
            point.load()
        except Exception as error:  # noqa: BLE001 - a provider that cannot import is a refusal naming it
            dist = point.dist
            pin = next((r for r in locked if r["name"].split(".", 1)[0] == word), None)
            line = (f"pip install '{pin['distribution']}=={pin['distribution_version']}'" if pin else
                    f"reinstall {dist.metadata['Name'] if dist else 'the package providing it'}")
            refusals.append(Refusal("load.namespace-import", RefusalClass.ENVIRONMENT,
                                    f"namespace {word!r} is provided by {point.value}, which fails to import: "
                                    f"{error!r}", line))
    return refusals


_KINDS = {kind.value: kind for kind in EntryKind}


def resolve_entry(kind: str, name: str, version: str, where: str, locked: Iterable[EntryRecord]) -> \
        tuple[Registered | None, Refusal | None]:
    """The registered entry a document names, or the refusal that says what to install."""
    entry_kind = _KINDS[kind]
    found = lookup(entry_kind, name, version)
    if found is not None:
        return found, None
    held = versions_of(entry_kind, name)
    line = install_line(kind, name, version, locked)
    if held:
        message = (f"{where} names {kind} {name} {version}, and the installed packages register {name} at "
                   f"{', '.join(held)} only")
        return None, Refusal("load.version", RefusalClass.ENVIRONMENT, message, line)
    return None, Refusal("load.unknown-entry", RefusalClass.ENVIRONMENT,
                         f"{where} names {kind} {name} {version}, which no installed package registers", line)


def entry_records(used: Iterable[Registered]) -> tuple[list[EntryRecord], list[str]]:
    """The lock's record of every entry used, sorted, and the reasons any of them could not be found by a load."""
    records: dict[tuple[str, str, str], EntryRecord] = {}
    problems: list[str] = []
    for record in used:
        key = (record.kind.value, record.name, record.version)
        if key in records:
            continue
        found = provenance_of(record)
        if isinstance(found, str):
            problems.append(found)
            continue
        records[key] = {"kind": cast(Any, record.kind.value), "name": record.name, "version": record.version,
                        "distribution": found.distribution, "distribution_version": found.version}
    return [records[key] for key in sorted(records)], problems


__all__ = ["ENTRY_POINT_GROUP", "MIME_DISTRIBUTION", "Provenance", "entry_records", "import_namespaces",
           "install_line", "namespace_providers", "provenance_of", "resolve_entry"]
