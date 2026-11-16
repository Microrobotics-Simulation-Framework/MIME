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

**Names.** A record's name is dotted like an entry's: a package's records are ``<source>.<name>`` (the source being its
``mime.assets`` entry point's name, its namespace), MIME's own have no namespace. The earlier spelling with an
underscore, ``<source>_<name>``, is still found under either spelling for this release (:func:`spellings`), so a
package that ships it keeps working while it moves, and written so it says once which spelling to use. A namespace a
resolver owns is the resolver's: its dotted names are never respelled onto a record of another source, MIME's own
included (``openmaterica.example_water`` asks the ``openmaterica`` resolver, never finds MIME's
``openmaterica_example_water``).

**Resolvers.** A ``mime.assets`` entry point whose value names an object (``module:object``) is a resolver, not a
folder of records: asked for a name in its namespace that no pin and no folder holds, it answers with the record or
``None`` (:class:`Resolver`). A DOI, a hub, OpenMaterica or a scan's derived layer becomes a source this way.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import re
import warnings
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cache
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from mime.assets._adapters import MIME_SOURCE, SOURCE_GROUP
from mime.assets._manifest import Manifest, is_manifest_data, read_manifest_data
from mime.assets._reference import PACKAGE_PREFIX, AssetRef
from mime._refusals import BuildRefused, MimeError, Refusal, RefusalClass

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
    """Each shared source of records with its module: MIME's own first, then every ``mime.assets`` entry point that
    names a package (a resolver names an object, :func:`resolvers`), by name."""
    found = [(MIME_SOURCE, "mime.data")]
    found += sorted((point.name, point.value) for point in importlib.metadata.entry_points(group=SOURCE_GROUP)
                    if point.name != MIME_SOURCE and ":" not in point.value)
    return found


@runtime_checkable
class Resolver(Protocol):
    """A source that answers by name (a ``mime.assets`` entry point whose value is ``module:object``): asked for a name
    in its namespace (``<source>.<name>``) that no pin and no folder of records holds, :meth:`record` gives that
    name's record, or ``None`` where it has none. Its record is held as any other: by its sha256, judged by its
    licence, fetched through the locations it names."""

    def record(self, name: str) -> AssetRef | None: ...


@cache
def resolvers() -> Mapping[str, object]:
    """Each resolver an installed package declares, by its source's name: the object its entry point names, loaded
    once per process. A resolver that cannot be loaded is left out, with a warning, as a folder that cannot be read is."""
    found: dict[str, object] = {}
    for point in sorted(importlib.metadata.entry_points(group=SOURCE_GROUP), key=lambda held: held.name):
        if ":" not in point.value or point.name == MIME_SOURCE:
            continue
        try:
            found[point.name] = point.load()
        except Exception as error:  # noqa: BLE001 - a package's import may fail in any way; its records are left out
            warnings.warn(f"the asset resolver {point.name!r} ({point.value}) cannot be loaded, and its records are left "
                          f"out: {error}; reinstall or uninstall the package that declares it", RuntimeWarning,
                          stacklevel=2)
    return MappingProxyType(found)


def _resolver_refusal(source: str, name: str, said: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.resolver", RefusalClass.ENVIRONMENT, f"the asset resolver {source!r}, asked for {name!r}, {said}",
        f"correct the resolver, or uninstall the package that declares it: `pip uninstall <package>`")])


def resolved_by_a_resolver(name: str) -> AssetRef | None:
    """The record a resolver gives for ``name``, where the resolver of its namespace has one; ``None`` where no
    resolver holds the namespace or it has no record of the name. A resolver that answers with anything but a record of
    that name, or that fails, is refused (``asset.resolver``)."""
    source = name.partition(".")[0]
    held = resolvers().get(source)
    if held is None or "." not in name:
        return None
    if not isinstance(held, Resolver):
        raise _resolver_refusal(source, name, f"is {type(held).__name__}, which has no record(name) method")
    try:
        answer = cast(object, held.record(name))
    except MimeError:
        raise
    except (Exception, SystemExit) as error:  # noqa: BLE001 - a package's resolver may fail in any way; it is refused by name
        raise _resolver_refusal(source, name, f"failed: {type(error).__name__}: {error}") from error
    if answer is None:
        return None
    if not isinstance(answer, AssetRef) or answer.name != name:
        said = f"the record {answer.name!r}" if isinstance(answer, AssetRef) else type(answer).__name__
        raise _resolver_refusal(source, name, f"answered {said}, not a record of that name")
    return answer


@cache
def _namespaces() -> tuple[str, ...]:
    """Every installed folder of records's source name but MIME's, longest first: the namespaces whose names have an
    earlier spelling. A namespace a resolver owns is not among them: its names are the resolver's own."""
    owned = {point.name for point in importlib.metadata.entry_points(group=SOURCE_GROUP) if ":" in point.value}
    names = {point.name for point in importlib.metadata.entry_points(group=SOURCE_GROUP)
             if point.name != MIME_SOURCE and point.name not in owned}
    return tuple(sorted(names, key=lambda name: (-len(name), name)))


def spellings(name: str) -> tuple[str, ...]:
    """``name``, then its earlier spelling where it is in a source's namespace: ``<source>.<name>`` and
    ``<source>_<name>`` find each other for this release, while packages move their records onto dots. A namespace a
    resolver owns has none (:func:`_namespaces`)."""
    for source in _namespaces():
        for written, other in ((f"{source}.", f"{source}_"), (f"{source}_", f"{source}.")):
            if name.startswith(written) and len(name) > len(written):
                return name, other + name.removeprefix(written)
    return (name,)


_WARNED: set[str] = set()


def _earlier_spelling_said(name: str, found: str) -> None:
    """Say once per name that a record found under its dotted spelling (``<source>.<name>``) was asked for under the
    earlier one (``<source>_<name>``)."""
    if name in _WARNED or not any(name.startswith(f"{source}_") and found.startswith(f"{source}.")
                                  for source in _namespaces()):
        return
    _WARNED.add(name)
    warnings.warn(f"the record {found!r} was asked for as {name!r}, the earlier spelling: write {found!r} (a package's "
                  f"records are <namespace>.<name>; the underscore spelling is read for this release only)",
                  FutureWarning, stacklevel=5)


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
    (:func:`agree`): the first's, with every source's locations in source order and the first route to request it that
    any states (a route is no term: two sources may name different ones); ``None`` where two disagree, on the bytes or
    on the terms, whichever source comes first."""
    if not agree(held):
        return None
    locations = tuple(dict.fromkeys(location for each in held for location in each.ref.locations))
    request = next((each.ref.request for each in held if each.ref.request is not None), None)
    return replace(held[0].ref, locations=locations, request=request)


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


def own_record(name: str) -> AssetRef | None:
    """MIME's own record of ``name`` whose bytes the installed package ships (a location ``package:mime/<path>``), or
    ``None``. A record of MIME's own that names only remote locations (a wall table) is not one: a lock keeps its pin of
    the version it was written against (the pin wins), and that version is its own bytes, not MIME's."""
    held = next((each.ref for each in shared().records.get(name, ()) if each.source == MIME_SOURCE), None)
    shipped = f"{PACKAGE_PREFIX}{MIME_SOURCE}/"
    return held if held is not None and any(location.startswith(shipped) for location in held.locations) else None


def own_namespaces() -> frozenset[str]:
    """The first word of every name of MIME's own records (``navion`` of ``navion_2``): words no plugin may take."""
    return frozenset(re.split(r"[._]", name, maxsplit=1)[0] for name, held in shared().records.items()
                     if any(each.source == MIME_SOURCE for each in held))


def claims_an_own_name(ref: AssetRef) -> BuildRefused | None:
    """The refusal (``asset.packaged-name``) of a record that claims a name MIME ships a file under (:func:`own_record`)
    as other bytes or other terms: a lock, a pin or a script's record never stands in for what MIME ships under its name. A record that
    restates MIME's own one changes nothing."""
    own = own_record(ref.name)
    if own is None or (own.sha256 == ref.sha256 and terms_of(own) == terms_of(ref)):
        return None
    return BuildRefused([Refusal(
        "asset.packaged-name", RefusalClass.USER,
        f"the record {ref.name!r} (sha256 {ref.sha256}, licence {ref.licence}) takes a name MIME's own records hold: "
        f"MIME ships {ref.name!r} as sha256 {own.sha256} (licence {own.licence}), and no lock, pin or script replaces "
        f"it", f"give the record a name of its own, <namespace>.<name>, and `mime assets fetch` it under that name")])


def _without_own(refs: Iterable[AssetRef]) -> list[AssetRef]:
    """``refs`` as pins: a record that restates MIME's own is no pin; one that claims its name for other bytes or terms
    is refused (:func:`claims_an_own_name`)."""
    kept: list[AssetRef] = []
    for ref in refs:
        refused = claims_an_own_name(ref)
        if refused is not None:
            raise refused
        if own_record(ref.name) is None:
            kept.append(ref)
    return kept


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
    held.update({ref.name: ref for ref in _without_own(refs)})
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
    """The record a name stands for, under either of its spellings (:func:`spellings`), each tried in turn: its pin
    where it has one (with a shared source's locations of the same bytes after its own, and a note where a shared
    source holds other bytes); else the shared sources' record; else the record the resolver of its namespace gives;
    ``None`` for a name nothing holds. Two shared sources that disagree, with no pin, are refused."""
    for spelling in spellings(name):
        found = _resolve_spelling(spelling, shared_now)
        if found is not None:
            _earlier_spelling_said(name, spelling)
            return found
    return resolved_by_a_resolver(name)


def _resolve_spelling(name: str, shared_now: Mapping[str, AssetRef]) -> AssetRef | None:
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


__all__ = ["ASSETS_FOLDER", "TERMS_KEYS", "Resolver", "Shared", "Sourced", "active_pins", "agree", "claims_an_own_name",
           "conflict_refusal", "conflicts", "own_namespaces", "own_record", "merged", "pinned", "resolve_name", "resolved_by_a_resolver", "resolvers", "shared",
           "shared_manifests", "shared_records", "spellings", "terms_of"]
