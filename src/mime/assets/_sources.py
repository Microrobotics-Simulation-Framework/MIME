"""Where records come from, and which one wins.

In this order: **(a)** the experiment's own declaration, or its lock when it is loaded (the pins, :func:`pinned`);
**(b)** the manifest beside the description in use (``visuals.json`` next to a URDF; ``mime.assets._manifest``);
**(c)** installed packages, each through a ``mime.assets`` entry point (its name the source's, its value a package whose
data holds JSON files of records and manifests); **(d)** MIME's own package data, for MIME's own assets only (the wall
tables). (c) and (d) are the shared sources.

**The pin wins.** An experiment that pinned a hash keeps it: :func:`reference` returns the pinned record whatever a
shared source now holds under the name, and says so (each line goes to the build report). Two shared sources that
hold one name under different hashes, or the same bytes under different terms (its licence and custom licences, credit,
notice, role, kind or derivation: a disagreement on what may be done with the file), with no pin to settle it, are
refused (``asset.conflict``), naming both, whichever source is read first. A new location is never a conflict: the hash
names the file, so the same bytes under the same terms in two sources are one record with both sources' locations.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import warnings
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cache
from types import MappingProxyType
from typing import cast

from mime.assets._adapters import MIME_SOURCE, SOURCE_GROUP
from mime.assets._manifest import Manifest, is_manifest_data, read_manifest_data
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

#: MIME's own records, in ``mime.data``.
ASSETS_FOLDER = "assets"


@dataclass(frozen=True)
class Sourced:
    """A record, and the source it was read from (``mime``, or a ``mime.assets`` entry point's name)."""

    ref: AssetRef
    source: str


@dataclass(frozen=True)
class Shared:
    """Everything the shared sources hold: each name's records, one per source, and every manifest by source."""

    records: Mapping[str, tuple[Sourced, ...]]
    manifests: tuple[tuple[str, Manifest], ...]


def _read_file(data: object, where: str, source: str) -> tuple[list[AssetRef], list[Manifest]]:
    if is_manifest_data(data):
        manifest = read_manifest_data(data, where)
        return [], [manifest]
    if not isinstance(data, dict) or not isinstance(cast(dict[str, object], data).get("assets"), list):
        raise BuildRefused([Refusal(
            "asset.manifest", RefusalClass.ENVIRONMENT,
            f"{where} (the source {source}) is neither a manifest nor a list of records under \"assets\"",
            "correct the package's file, or uninstall the package that ships it: `pip uninstall <package>`")])
    listed = cast(list[object], cast(dict[str, object], data)["assets"])
    return [AssetRef.from_record(cast(dict[str, object], each)) for each in listed if isinstance(each, dict)], []


def _source_files(module: str) -> list[tuple[str, object]]:
    """Every JSON file of a source's package data, with where it is, in name order: ``assets/`` for MIME's own, the
    package's own folder for an entry point's."""
    root = importlib.resources.files(module)
    folder = root / ASSETS_FOLDER if module == "mime.data" else root
    found: list[tuple[str, object]] = []
    for entry in sorted(folder.iterdir(), key=lambda held: held.name):
        if entry.name.endswith(".json") and entry.is_file():
            found.append((f"{module}/{entry.name}", json.loads(entry.read_text(encoding="utf-8"))))
    return found


def _sources() -> list[tuple[str, str]]:
    """Each shared source with its module: MIME's own first, then every ``mime.assets`` entry point by name."""
    found = [(MIME_SOURCE, "mime.data")]
    found += sorted((point.name, point.value.partition(":")[0])
                    for point in importlib.metadata.entry_points(group=SOURCE_GROUP) if point.name != MIME_SOURCE)
    return found


@cache
def shared() -> Shared:
    """What every shared source holds, read once per process."""
    by_name: dict[str, list[Sourced]] = {}
    manifests: list[tuple[str, Manifest]] = []
    for source, module in _sources():
        try:
            files = _source_files(module)
        except (ImportError, OSError, ValueError) as error:
            warnings.warn(f"the asset source {source!r} (module {module}) cannot be read, and its records are left out: "
                          f"{error}; reinstall or uninstall the package that declares it", RuntimeWarning, stacklevel=2)
            continue
        for where, data in files:
            refs, held = _read_file(data, where, source)
            manifests += [(source, each) for each in held]
            for ref in [*refs, *(each for manifest in held for each in manifest.assets)]:
                by_name.setdefault(ref.name, []).append(Sourced(ref, source))
    return Shared(MappingProxyType({name: tuple(held) for name, held in by_name.items()}), tuple(manifests))


#: What two records of the same bytes must agree on to be one record: its terms, and where its truth comes from.
TERMS_KEYS = ("licence", "licences", "credit", "notice", "role", "kind", "derivation")


def terms_of(ref: AssetRef) -> tuple[object, ...]:
    """What a record says may be done with its file, and where its truth comes from (:data:`TERMS_KEYS`)."""
    record: dict[str, object] = dict(ref.to_record())
    return tuple(repr(record.get(key)) for key in TERMS_KEYS)


def agree(held: Sequence[Sourced]) -> bool:
    """Whether a name's records in several sources hold the same bytes under the same terms."""
    return len({each.ref.sha256 for each in held}) == 1 and len({terms_of(each.ref) for each in held}) == 1


def merged(held: Sequence[Sourced]) -> AssetRef | None:
    """One record from a name's records in several sources, where they hold the same bytes under the same terms
    (:func:`agree`): the first's, with every source's locations in source order; ``None`` where two disagree, on the
    bytes or on the terms, whichever source comes first."""
    if not agree(held):
        return None
    locations = tuple(dict.fromkeys(location for each in held for location in each.ref.locations))
    return replace(held[0].ref, locations=locations)


def shared_records() -> Mapping[str, AssetRef]:
    """Every name the shared sources agree on, with its record: a name two sources hold under different hashes, or
    under different terms, is left out (:func:`conflicts`)."""
    found: dict[str, AssetRef] = {}
    for name, held in shared().records.items():
        ref = merged(held)
        if ref is not None:
            found[name] = ref
    return MappingProxyType(found)


def conflicts() -> Mapping[str, tuple[Sourced, ...]]:
    """Each name two shared sources hold under different hashes, or as the same bytes under different terms, with
    each source's record."""
    return MappingProxyType({name: held for name, held in shared().records.items() if not agree(held)})


def _side(each: Sourced) -> str:
    ref = each.ref
    stated = ", ".join(f"{key} {value}" for key, value in zip(TERMS_KEYS, terms_of(ref), strict=True)
                       if value != "None")
    return f"{each.source} holds sha256 {ref.sha256} ({stated})"


def conflict_refusal(name: str, held: Sequence[Sourced]) -> BuildRefused:
    sides = "; ".join(_side(each) for each in held)
    what = "as different files" if len({each.ref.sha256 for each in held}) > 1 else \
        "as the same file under different terms"
    return BuildRefused([Refusal(
        "asset.conflict", RefusalClass.ENVIRONMENT,
        f"two installed sources hold the asset {name!r} {what} ({sides}), and nothing pins one",
        "pin the one the experiment is written against, AssetRef(..., sha256=...) declared in the script or kept in "
        "its saved lock; or uninstall the package whose record is wrong")])


# --- pins -----------------------------------------------------------------------------------------------------------


@dataclass
class _Pins:
    refs: Mapping[str, AssetRef]
    notes: list[str]


_pins: ContextVar[_Pins | None] = ContextVar("mime_asset_pins", default=None)


@contextmanager
def pinned(refs: Iterable[AssetRef]) -> Generator[list[str]]:
    """Inside this block, a name of ``refs`` resolves to its pinned record (:func:`resolve_name`), whatever a shared
    source holds; yields the lines that say where a shared source now holds another version, for the build report.
    Pins nest: an inner block's records stand over an outer one's."""
    outer = _pins.get()
    held = dict(outer.refs) if outer is not None else {}
    held.update({ref.name: ref for ref in refs})
    notes: list[str] = outer.notes if outer is not None else []
    token = _pins.set(_Pins(MappingProxyType(held), notes))
    try:
        yield notes
    finally:
        _pins.reset(token)


def active_pins() -> Mapping[str, AssetRef]:
    held = _pins.get()
    return held.refs if held is not None else MappingProxyType({})


def _note(line: str) -> None:
    held = _pins.get()
    if held is not None and line not in held.notes:
        held.notes.append(line)


def resolve_name(name: str, shared_now: Mapping[str, AssetRef]) -> AssetRef | None:
    """The record a name stands for: its pin where it has one (with a shared source's locations of the same bytes
    after its own, and a note where a shared source holds other bytes); else the shared sources' record; ``None`` for a
    name nothing holds. Two shared sources that disagree, with no pin, are refused."""
    pin = active_pins().get(name)
    held = shared().records.get(name, ())
    if pin is not None:
        same = [each for each in held if each.ref.sha256 == pin.sha256]
        others = [each for each in held if each.ref.sha256 != pin.sha256]
        for each in others:
            _note(f"asset {name}: the experiment keeps its pin, sha256 {pin.sha256}; the source {each.source} now "
                  f"holds another version, sha256 {each.ref.sha256} (save the experiment again from its script to "
                  f"take it)")
        for each in same:
            if terms_of(each.ref) != terms_of(pin):
                _note(f"asset {name}: the experiment keeps its pin's terms (licence {pin.licence}); the source "
                      f"{each.source} holds the same file under other terms (licence {each.ref.licence})")
        locations = tuple(dict.fromkeys((*pin.locations, *(loc for each in same for loc in each.ref.locations))))
        return replace(pin, locations=locations)
    found = shared_now.get(name)
    if found is not None:
        return found
    if name in conflicts():
        raise conflict_refusal(name, conflicts()[name])
    return None


def shared_manifests() -> tuple[tuple[str, Manifest], ...]:
    """Every manifest the shared sources hold, with its source."""
    return shared().manifests


__all__ = ["ASSETS_FOLDER", "TERMS_KEYS", "Shared", "Sourced", "active_pins", "agree", "conflict_refusal", "conflicts",
           "merged", "pinned", "resolve_name", "shared", "shared_manifests", "shared_records", "terms_of"]
