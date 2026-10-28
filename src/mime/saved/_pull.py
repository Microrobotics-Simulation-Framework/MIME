"""``pull``: get a bundle by DOI, URL or file, unpack it into a new folder, and say what is still needed.

The bundle is fetched through ``mime.assets`` as any content-addressed file is: stored in the cache under its hash, the
network used only because this was asked for, and offline refused. Its assets are fetched the same way afterwards,
unless declined. Nothing here installs or runs a package: the packages the lock names that are not installed are
listed with the command to install them.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mime import assets
from mime.assets._commands import fetch_command, folder_argument
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.saved._bundle import BUNDLE_SUFFIX, Unpacked, bundle_name, unpack
from mime.saved._schema import Lock

#: Assets whose download is larger than this are asked about first (an interactive terminal), or need ``--yes``. A
#: choice: about what a session on a laptop's connection should not start without being told.
ASSET_CONFIRM_BYTES = 1_000_000_000

_DOI = re.compile(r"10\.\d{4,9}/\S+")
_DOI_PREFIXES = ("doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/")


@dataclass(frozen=True)
class Source:
    """Where a bundle comes from: a ``file`` on this machine, a ``url`` to download, or a ``doi`` (with the record's file
    when the source names one)."""

    kind: Literal["file", "url", "doi"]
    value: str
    file: str | None = None


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def parse_source(text: str) -> Source:
    """A DOI (``10.5281/zenodo.1``, ``doi:...`` or ``https://doi.org/...``, optionally ending ``/<file>.mimebundle``), an
    ``https://`` URL, or the path of a file."""
    if Path(text).is_file():
        return Source("file", text)
    bare = next((text.removeprefix(prefix) for prefix in _DOI_PREFIXES if text.startswith(prefix)), text)
    if _DOI.fullmatch(bare):
        doi, _, last = bare.rpartition("/")
        return Source("doi", doi, last) if last.endswith(BUNDLE_SUFFIX) and doi.count("/") >= 1 else Source("doi", bare)
    if text.startswith(("https://", "http://")):
        problem = assets.location_problem(text)
        if problem is not None:
            raise _refuse("pull.source", RefusalClass.USER, f"{text!r}: {problem}",
                          "give an https:// address, a DOI, or a file")
        return Source("url", text)
    raise _refuse("pull.source", RefusalClass.USER,
                  f"{text!r} is not a file on this machine, a DOI or an https:// address",
                  "give the bundle's DOI (10.5281/zenodo.1234567), its address, or its file")


def _held(value: object) -> Any:
    return value


def _offline_refusal(what: str) -> BuildRefused:
    return _refuse("asset.offline", RefusalClass.ENVIRONMENT,
                   f"{what} is not on this machine, and {assets.OFFLINE_VARIABLE} is set, so nothing is downloaded",
                   f"unset {assets.OFFLINE_VARIABLE}, or give the bundle as a file")


def _record_file(doi: str, wanted: str | None) -> tuple[str, str]:
    """The bundle file of the repository record ``doi`` names, and the repository's checksum of it, through pooch's DOI
    support. The record must hold one ``.mimebundle`` file, or ``wanted`` names it."""
    import pooch

    try:
        with tempfile.TemporaryDirectory() as scratch:
            library = _held(pooch)
            listing = library.create(path=scratch, base_url=f"doi:{doi}/", registry={})
            listing.load_registry_from_doi()
            registry: dict[str, str] = dict(listing.registry)
    except Exception as error:  # noqa: BLE001 - pooch and requests raise unrelated types for a record that can't be read
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT, f"the record doi:{doi} could not be read: {error}",
                      "check the DOI and the network") from error
    bundles = sorted(name for name in registry if name.endswith(BUNDLE_SUFFIX))
    chosen = [wanted] if wanted in bundles else bundles
    if len(chosen) != 1:
        held = ", ".join(sorted(registry)) or "none"
        raise _refuse("pull.record", RefusalClass.ENVIRONMENT,
                      f"the record doi:{doi} lists {len(bundles)} bundle files (its files: {held})",
                      f"name the one to pull: doi:{doi}/<file>{BUNDLE_SUFFIX}")
    checksum = registry[chosen[0]]
    return chosen[0], checksum if ":" in checksum else f"md5:{checksum}"


def _bundle_file(source: Source, say: Callable[[str], None] | None) -> Path:
    if source.kind == "file":
        return Path(source.value)
    if assets.offline():
        raise _offline_refusal(source.value)
    if source.kind == "url":
        return assets.locate(assets.retrieve(source.value, name="bundle", report=say))
    file, checksum = _record_file(source.value, source.file)
    return assets.locate(assets.retrieve(f"doi:{source.value}/{file}", name="bundle", known_hash=checksum, report=say))


def packages_needed(lock: Lock) -> list[tuple[str, str, str | None]]:
    """The distributions the lock's entries came from that are not installed, or not at the version that built them:
    each with the version wanted and the one installed, if any."""
    wanted = {entry["distribution"]: entry["distribution_version"] for entry in lock["entries"]}
    needed: list[tuple[str, str, str | None]] = []
    for name, version in sorted(wanted.items()):
        try:
            installed: str | None = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            needed.append((name, version, installed))
    return needed


def _packages_line(needed: list[tuple[str, str, str | None]]) -> str:
    if not needed:
        return "Packages: all installed"
    names = ", ".join(f"{name} {version} needed" + (f" ({installed} installed)" if installed else "")
                      for name, version, installed in needed)
    return f"Packages: {names}: pip install " + " ".join(f'"{name}=={version}"' for name, version, _ in needed)


@dataclass(frozen=True)
class _AssetOutcome:
    cached: int
    downloaded: int
    downloaded_bytes: int
    not_downloaded: int
    stopped: BuildRefused | None = None


def _assets_line(outcome: _AssetOutcome, folder: Path) -> str:
    parts = [text for text in (
        f"{outcome.cached} cached" if outcome.cached else "",
        f"{outcome.downloaded} downloaded ({assets.size_text(outcome.downloaded_bytes)})" if outcome.downloaded else "",
        f"{outcome.not_downloaded} not downloaded: run `{fetch_command(folder=folder)}`"
        if outcome.not_downloaded else "") if text]
    return "Assets: " + (", ".join(parts) if parts else "none referenced")


def _fetch_assets(folder: Path, *, wanted: bool, yes: bool, confirm: Callable[[str], bool] | None,
                  say: Callable[[str], None] | None) -> _AssetOutcome:
    refs = assets.folder_assets(folder)
    absent = [ref for ref in refs if not assets.is_cached(ref)]
    cached = len(refs) - len(absent)
    if not wanted or not absent:
        return _AssetOutcome(cached, 0, 0, len(absent))
    total = sum(ref.size for ref in absent)
    if say is not None:
        say(f"Assets to download: {len(absent)} file{'s' if len(absent) != 1 else ''}, {assets.size_text(total)}")
    if total > ASSET_CONFIRM_BYTES and not yes:
        if confirm is None:
            return _AssetOutcome(cached, 0, 0, len(absent), _refuse(
                "pull.assets-large", RefusalClass.USER, f"the assets are {assets.size_text(total)}, over the "
                f"{assets.size_text(ASSET_CONFIRM_BYTES)} MIME asks about, and this is not a terminal to ask in",
                f"run `{fetch_command(folder=folder)}` to download them"))
        if not confirm(f"Download {assets.size_text(total)} of assets now?"):
            return _AssetOutcome(cached, 0, 0, len(absent))
    stored = 0
    try:
        for ref in refs:
            assets.fetch(ref, report=say)
            stored += ref in absent
    except BuildRefused as refused:
        return _AssetOutcome(cached, stored, sum(ref.size for ref in absent[:stored]), len(absent) - stored, refused)
    return _AssetOutcome(cached, len(absent), total, 0)


def pull(source: str | os.PathLike[str], *, into: str | os.PathLike[str] | None = None, fetch_assets: bool = True,
         yes: bool = False, confirm: Callable[[str], bool] | None = None,
         say: Callable[[str], None] | None = None) -> Path:
    """Get the bundle ``source`` names (a DOI, an address or a file), check its manifest's hashes, and unpack it into
    the new folder ``into`` (by default ``./<experiment name>/``); then, unless ``fetch_assets`` is false, fetch its
    assets as ``mime assets fetch <folder>`` does. Assets over :data:`ASSET_CONFIRM_BYTES` are fetched only with ``yes``
    or when ``confirm`` (asked with a question) says so; with neither, the folder is complete except for them and a
    ``BuildRefused`` says the command that fetches them. ``say`` receives progress lines and, last, the summary.
    Installs nothing."""
    bundle = _bundle_file(parse_source(os.fspath(source)), say)
    destination = Path(into) if into is not None else Path(bundle_name(bundle))
    unpacked: Unpacked = unpack(bundle, destination)
    needed = packages_needed(unpacked.lock)
    outcome = _fetch_assets(destination, wanted=fetch_assets, yes=yes, confirm=confirm, say=say)
    lines = [f"\N{CHECK MARK} Bundle {unpacked.manifest['name']!r} unpacked to {folder_argument(destination)}",
             "  - Manifest, scene and records: OK",
             f"  - {_assets_line(outcome, destination)}",
             f"  - {_packages_line(needed)}",
             f"Next: mime verify {folder_argument(destination)}"]
    if say is not None:
        say("\n".join(lines))
    if outcome.stopped is not None:
        raise outcome.stopped
    return destination


__all__ = ["ASSET_CONFIRM_BYTES", "Source", "packages_needed", "parse_source", "pull"]
