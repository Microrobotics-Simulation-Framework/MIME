"""``mime assets sources``: for each asset, its terms, its tier and where it can be fetched from now: availability made
visible. Each location is looked at without downloading the file (a ranged request for its
first byte, or the adapter's open and close), in the order a fetch tries them, and said reachable or not, and why;
offline, each says it was not looked at."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mime._lock_json import read_lock
from mime.assets._cache import is_cached, size_text
from mime.assets._fetch import probe
from mime.assets._known import reference
from mime.assets._locations import read_locations
from mime.assets._lock import folder_assets
from mime.assets._manifest import BESIDE_NAME, read_manifest, read_manifest_data
from mime.assets._reference import AssetRef

#: The lock's key that holds the manifests of the meshes the stage draws.
_VISUALS = "visuals"


def assets_of(target: str | os.PathLike[str]) -> list[AssetRef]:
    """The records ``target`` names: a saved folder's (its lock's assets, then the records of its meshes' manifests),
    a folder holding a manifest (``visuals.json``), a manifest file, or one asset by name."""
    path = Path(target)
    if path.is_dir() and (path / "mime.lock").is_file():
        found = list(folder_assets(path))
        try:
            held: object = read_lock(path / "mime.lock")
        except (OSError, ValueError):
            held = {}
        visuals: object = cast(dict[str, object], held).get(_VISUALS, {}) if isinstance(held, dict) else {}
        for name, value in sorted(cast(dict[str, object], visuals).items()) if isinstance(visuals, dict) else []:
            found += read_manifest_data(value, f"{path} ({_VISUALS}: {name})").assets
        return found
    if path.is_dir() and (path / BESIDE_NAME).is_file():
        return list(read_manifest(path / BESIDE_NAME).assets)
    if path.is_file() and path.suffix == ".json":
        return list(read_manifest(path).assets)
    return [reference(str(target))]


def _terms_line(ref: AssetRef) -> str:
    verdict = ref.verdict
    custom = ref.custom.get(ref.licence)
    licence = f"{ref.licence} ({custom.title}, {custom.url})" if custom is not None else ref.licence
    kind = ref.kind or "not stated"
    if ref.derivation is not None:
        kind += f", its derivation record sha256 {ref.derivation}"
    parts = [f"licence {licence}", f"tier {verdict.tier} ({verdict.words()})", f"role {ref.role or 'not stated'}",
             f"kind {kind}"]
    if ref.credit:
        parts.append(f"credit {ref.credit}")
    return "; ".join(parts)


@dataclass(frozen=True)
class SourceLocation:
    """One location of an asset, looked at: its address, where it came from (``this machine's locations file``) when not
    the record, and whether it was reachable (``None``: not looked at) with why."""

    location: str
    where: str
    reachable: bool | None
    why: str

    def line(self) -> str:
        state = {True: "reachable", False: "not reachable", None: "not looked at"}[self.reachable]
        return f"    {self.location}{self.where}: {state}: {self.why}"


@dataclass(frozen=True)
class SourceView:
    """What ``mime assets sources`` found of one asset: its record, whether it is cached, and each location looked at in
    the order a fetch tries them."""

    ref: AssetRef
    cached: bool
    locations: tuple[SourceLocation, ...]

    @property
    def reachable(self) -> str:
        """The locations that answered, as a table cell: ``2 of 3``, or ``not looked at`` where none was."""
        looked = [each for each in self.locations if each.reachable is not None]
        if not looked:
            return "not looked at"
        return f"{sum(1 for each in looked if each.reachable)} of {len(looked)}"

    def head(self) -> str:
        return (f"{self.ref.name}  {size_text(self.ref.size)}  {'cached' if self.cached else 'not cached'}  "
                f"sha256 {self.ref.sha256}")

    def detail(self) -> list[str]:
        """Every line of the asset's text after the first: its terms, what a referenced tier means, its locations and
        its adapters."""
        ref = self.ref
        lines = [f"  terms: {_terms_line(ref)}"]
        if ref.tier == "referenced":
            lines.append("  its bytes never travel inside a bundle, a push or a package: only this record does")
        lines.append("  locations, in the order a fetch tries them:")
        lines += [each.line() for each in self.locations]
        for scheme, package in ref.adapters:
            lines.append(f"  the scheme {scheme} is read by an adapter from {package or 'a script'}")
        return lines


def sources_view(refs: Sequence[AssetRef]) -> list[SourceView]:
    """Each asset looked at now: its cache state and each of its locations, in the order a fetch tries them, probed
    without downloading."""
    from mime.assets._fetch import tried_locations

    extra = read_locations()
    views: list[SourceView] = []
    for ref in refs:
        mine = set(extra.extra(ref.name, ref.sha256))
        found: list[SourceLocation] = []
        for location in tried_locations(ref):
            seen = probe(location, ref)
            where = " (this machine's locations file)" if location in mine and location not in ref.locations else ""
            found.append(SourceLocation(seen.location, where, seen.reachable, seen.why))
        views.append(SourceView(ref, is_cached(ref), tuple(found)))
    return views


def sources_text(refs: Sequence[AssetRef]) -> str:
    """Each asset's terms, tier, cache state and locations, as ``mime assets sources`` prints them."""
    lines: list[str] = []
    for view in sources_view(refs):
        lines += [view.head(), *view.detail()]
    return "\n".join(lines) if lines else "no asset"


__all__ = ["SourceLocation", "SourceView", "assets_of", "sources_text", "sources_view"]
