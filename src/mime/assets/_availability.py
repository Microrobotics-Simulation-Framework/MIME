"""``mime assets sources``: for each asset, its terms, its tier and where it can be fetched from now: availability made
visible. Each location is looked at without downloading the file (a ranged request for its
first byte, or the adapter's open and close), in the order a fetch tries them, and said reachable or not, and why;
offline, each says it was not looked at."""

from __future__ import annotations

import os
from collections.abc import Sequence
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


def sources_text(refs: Sequence[AssetRef]) -> str:
    """Each asset's terms, tier, cache state and locations, as ``mime assets sources`` prints them."""
    from mime.assets._fetch import tried_locations

    extra = read_locations()
    lines: list[str] = []
    for ref in refs:
        lines.append(f"{ref.name}  {size_text(ref.size)}  {'cached' if is_cached(ref) else 'not cached'}  "
                     f"sha256 {ref.sha256}")
        lines.append(f"  terms: {_terms_line(ref)}")
        if ref.tier == "referenced":
            lines.append("  its bytes never travel inside a bundle, a push or a package: only this record does")
        mine = set(extra.extra(ref.name, ref.sha256))
        lines.append("  locations, in the order a fetch tries them:")
        for location in tried_locations(ref):
            found = probe(location, ref)
            where = " (this machine's locations file)" if location in mine and location not in ref.locations else ""
            state = {True: "reachable", False: "not reachable", None: "not looked at"}[found.reachable]
            lines.append(f"    {found.location}{where}: {state}: {found.why}")
        for scheme, package in ref.adapters:
            lines.append(f"  the scheme {scheme} is read by an adapter from {package or 'a script'}")
    return "\n".join(lines) if lines else "no asset"


__all__ = ["assets_of", "sources_text"]
