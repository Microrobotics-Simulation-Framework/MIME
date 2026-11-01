"""Bundles written by hand, for the tests of what ``pull`` refuses: a well-formed one from files and a name (format 2,
with its NOTICE), and a rewrite of a real one with a member changed, added, removed or made to look like a link."""

import json
import stat
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

from mime.saved._bundle import FOLDER_PREFIX, FORMAT, MANIFEST_MEMBER, NOTICE_MEMBER, bundle_hash, sha256_of

#: The NOTICE a bundle written by hand carries.
NOTICE = b"NOTICE: made by hand\n"


def manifest_of(files: Mapping[str, bytes], name: str = "made_by_hand", assets: list[dict[str, object]] | None = None,
                *, notice: bytes = NOTICE, referenced: Mapping[str, object] | None = None) -> dict[str, object]:
    manifest: dict[str, object] = {
        "format": FORMAT, "name": name,
        "files": {member: {"sha256": sha256_of(data), "size": len(data)} for member, data in sorted(files.items())},
        "referenced": dict(referenced or {}), "notice": {"sha256": sha256_of(notice), "size": len(notice)},
        "assets": assets or []}
    manifest["bundle_hash"] = bundle_hash(manifest)
    return manifest


def member(archive: zipfile.ZipFile, name: str, data: bytes, *, mode: int = 0o100644,
           compress: int = zipfile.ZIP_STORED) -> None:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = compress
    info.create_system = 3
    info.external_attr = mode << 16
    archive.writestr(info, data)


def write_bundle(path: Path, files: Mapping[str, bytes], name: str = "made_by_hand",
                 assets: list[dict[str, object]] | None = None, *, notice: bytes = NOTICE,
                 referenced: Mapping[str, object] | None = None) -> Path:
    """A bundle as ``pack`` writes one, of exactly ``files`` (names relative to the folder) and its NOTICE."""
    manifest = json.dumps(manifest_of(files, name, assets, notice=notice, referenced=referenced), sort_keys=True,
                          indent=1).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        member(archive, MANIFEST_MEMBER, manifest)
        member(archive, NOTICE_MEMBER, notice)
        for each, data in sorted(files.items()):
            member(archive, FOLDER_PREFIX + each, data)
    return path


def contents(path: Path) -> dict[str, bytes]:
    """The folder files of a bundle, by name."""
    with zipfile.ZipFile(path) as archive:
        return {info.filename.removeprefix(FOLDER_PREFIX): archive.read(info) for info in archive.infolist()
                if info.filename.startswith(FOLDER_PREFIX)}


def rewrite(source: Path, target: Path, change: Callable[[dict[str, bytes]], None],
            manifest_change: Callable[[dict[str, object]], None] | None = None, *, keep_hash: bool = True) -> Path:
    """The bundle ``source`` with ``change`` applied to its files (a dict of name to bytes, edited in place), and the
    manifest left as it was (so that what changed no longer matches it), or edited by ``manifest_change``."""
    with zipfile.ZipFile(source) as archive:
        manifest: dict[str, object] = json.loads(archive.read(MANIFEST_MEMBER))
        notice = archive.read(NOTICE_MEMBER) if NOTICE_MEMBER in archive.namelist() else None
    files = contents(source)
    change(files)
    if manifest_change is not None:
        manifest_change(manifest)
        if not keep_hash:
            manifest["bundle_hash"] = bundle_hash(manifest)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        member(archive, MANIFEST_MEMBER, json.dumps(manifest, sort_keys=True, indent=1).encode("utf-8"))
        if notice is not None:
            member(archive, NOTICE_MEMBER, notice)
        for each, data in files.items():
            member(archive, FOLDER_PREFIX + each, data)
    return target


def symlink_mode() -> int:
    return stat.S_IFLNK | 0o777


def lock_with_assets(files: dict[str, bytes], *references: object) -> list[dict[str, object]]:
    """Put ``references`` (``AssetRef``) in the lock among ``files`` as the build that records an asset would, and return
    their records."""
    records = [each.to_record() for each in references]  # type: ignore[attr-defined]
    lock = json.loads(files["mime.lock"])
    lock["assets"] = records
    files["mime.lock"] = (json.dumps(lock, indent=1, sort_keys=True) + "\n").encode("utf-8")
    return records
