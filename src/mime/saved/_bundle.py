"""A bundle: one file holding a saved experiment folder, to be shared and checked on the other side.

It is a zip with nothing compressed, so the same folder always makes the same bytes on any machine (a compressor's
output is not the same across zlib builds). The first member is ``manifest.json``: the format, the experiment's name,
every file's sha256 and size, the asset references of the lock (a large file is referenced, never copied), and the
bundle's own hash, the sha256 of the rest of the manifest. The folder's files follow under ``experiment/``, sorted, with
a fixed timestamp and mode.

``pack`` writes one after checking that the folder loads, and refuses to carry a path of the machine that wrote it or
any token MIME holds. ``unpack`` is the reading side: it checks every name and member kind, then every hash, and writes
only into a new folder, through its own file opens and never an extraction, so that no path can leave the folder and no
link or special file can be made.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from typing_extensions import ReadOnly, TypedDict

from mime.assets import folder_assets, size_text
from mime.auth import held_values
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.saved._folder import read_folder
from mime.saved._schema import Lock, RemoteAssetRecord, conforms

BUNDLE_SUFFIX = ".mimebundle"
FORMAT = "mime.bundle/1"
MANIFEST_MEMBER = "manifest.json"
FOLDER_PREFIX = "experiment/"

#: The most one file may weigh and still travel inside a bundle. A choice, not a measurement: a bundle is read and
#: hashed member by member, and one a person is to download, check and keep next to their experiment should stay
#: small. A larger file is an asset (its reference travels, with its sha256, size, licence and locations) and is
#: fetched on request.
INLINE_LIMIT_BYTES = 50_000_000

_MANIFEST_LIMIT_BYTES = 16_000_000
_CHUNK = 1 << 20
_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MEMBER_MODE = 0o100644 << 16
_REGULAR = 0o100000

#: An absolute path of a machine, which no bundle may carry: the rule of the documentation generator's pages
#: (``mime.docs._build``, which a test holds this to), as bytes since a file is searched as it is stored.
_MACHINE_PATH = re.compile(rb"(?<![\w./-])/(?:home|Users|root|tmp|var|mnt|opt|srv|private|runner)/\S")
_WINDOWS_HOME = re.compile(rb"(?<![A-Za-z])[A-Za-z]:[\\/]+Users[\\/]", re.IGNORECASE)
_DRIVE = re.compile(r"[A-Za-z]:")
_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


class BundleFile(TypedDict, closed=True):
    sha256: ReadOnly[str]
    size: ReadOnly[int]


class Manifest(TypedDict, closed=True):
    """What a bundle says it holds. ``bundle_hash`` is the sha256 of the manifest without it."""

    format: ReadOnly[Literal["mime.bundle/1"]]
    name: ReadOnly[str]
    files: ReadOnly[dict[str, BundleFile]]
    assets: ReadOnly[list[RemoteAssetRecord]]
    bundle_hash: ReadOnly[str]


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_hash(manifest: dict[str, object]) -> str:
    """The hash that names a bundle's content: the manifest without its own hash, in a canonical text."""
    body = {key: value for key, value in manifest.items() if key != "bundle_hash"}
    return sha256_of(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def member_problem(name: str) -> str | None:
    """Why ``name`` cannot be a file inside the folder a bundle unpacks to, or ``None``: a name is a relative path of
    plain parts, as ``/`` writes it."""
    if not name or name != name.strip():
        return "is empty or has spaces at its ends"
    if any(ord(each) < 32 or each in "\\\x7f" for each in name):
        return "has a backslash or a control character"
    if name.startswith("/") or _DRIVE.match(name):
        return "is an absolute path"
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "has an empty, '.' or '..' part"
    if PurePosixPath(name).as_posix() != name:
        return "is not written in its plain form"
    return None


def _target(root: Path, name: str) -> Path:
    """Where the member ``name`` is written under ``root``: its name checked, and the path it makes checked to stay
    inside ``root``."""
    problem = member_problem(name)
    if problem is not None:
        raise BuildRefused([_refuse("bundle.unsafe", RefusalClass.USER, f"a member's name {name!r} {problem}",
                                     "this is not a bundle MIME packed; do not unpack it")])
    target = root / name
    if root.resolve() not in target.resolve().parents:
        raise BuildRefused([_refuse(
            "bundle.unsafe", RefusalClass.USER, f"the member {name!r} would be written outside the folder",
            "this is not a bundle MIME packed; do not unpack it")])
    return target


def _scan(name: str, data: bytes, secrets_held: Sequence[str]) -> list[Refusal]:
    found: list[Refusal] = []
    home = os.path.expanduser("~")
    machine = _MACHINE_PATH.search(data) or _WINDOWS_HOME.search(data)
    if machine is None and len(home) > 1 and home.encode("utf-8") in data:
        machine = re.search(re.escape(home.encode("utf-8")), data)
    if machine is not None:
        # The path is not quoted: it is the user's, and the message is read by others.
        found.append(_refuse(
            "bundle.machine-path", RefusalClass.USER,
            f"{name} holds a path of this machine (at byte {machine.start()})",
            "write the path as a name below the folder, or remove it, and save the experiment again"))
    if any(secret.encode("utf-8") in data for secret in secrets_held):
        found.append(_refuse("bundle.credential", RefusalClass.USER, f"{name} holds a token MIME has stored for you",
                              "remove it (a token never belongs in an experiment), and revoke it at the service if "
                              "it was shared"))
    return found


def _files_of(folder: Path) -> list[tuple[str, Path]]:
    """Every file below ``folder``, by its name as the bundle writes it, sorted. A link or a special file is refused."""
    found: list[tuple[str, Path]] = []
    refusals: list[Refusal] = []
    for directory, subdirectories, names in os.walk(folder):
        for entry in [*subdirectories, *names]:
            path = Path(directory) / entry
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                refusals.append(_refuse(
                    "bundle.special-file", RefusalClass.USER, f"{path.relative_to(folder).as_posix()} is a link or a "
                    f"special file", "a bundle holds plain files; copy the file's content in"))
            elif path.is_file():
                found.append((path.relative_to(folder).as_posix(), path))
    if refusals:
        raise BuildRefused(refusals)
    return sorted(found)


def pack(folder: str | os.PathLike[str], *, out: str | os.PathLike[str] | None = None) -> Path:
    """Write the bundle of the saved experiment in ``folder`` (by default ``<folder>.mimebundle`` beside it) and return
    its path. The folder must load (``Experiment.load``: a ``LoadRefused`` otherwise). Refused, with every reason, when
    a file holds a path of this machine or a token MIME has stored, when a file is over :data:`INLINE_LIMIT_BYTES` and
    is not an asset the lock references, or when the folder holds a link or a special file. The same folder makes the
    same bytes."""
    from mime.experiment import Experiment

    root = Path(folder)
    Experiment.load(root)
    name = _experiment_name(root)
    target = Path(out) if out is not None else root.with_name(root.name + BUNDLE_SUFFIX)
    if root.resolve() in (target.resolve(), *target.resolve().parents):
        raise BuildRefused([_refuse(
            "bundle.output-inside", RefusalClass.USER, "the bundle would be written inside the folder it packs",
            "write it beside the folder (--out)")])
    references = {ref.sha256: ref for ref in folder_assets(root)}
    stored = held_values()
    refusals: list[Refusal] = []
    members: dict[str, bytes] = {}
    for member, path in _files_of(root):
        size = path.stat().st_size
        if size > INLINE_LIMIT_BYTES:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest not in references:
                refusals.append(_refuse(
                    "bundle.file-too-large", RefusalClass.USER,
                    f"{member} has {size_text(size)}, over the {size_text(INLINE_LIMIT_BYTES)} a bundle carries",
                    "reference it as an asset in the lock (name, sha256, size, licence and locations), which travels "
                    "as a reference and is fetched on request"))
            continue
        data = path.read_bytes()
        refusals += _scan(member, data, stored)
        members[member] = data
    if refusals:
        raise BuildRefused(refusals)
    manifest: dict[str, object] = {
        "format": FORMAT, "name": name,
        "files": {member: {"sha256": sha256_of(data), "size": len(data)} for member, data in sorted(members.items())},
        "assets": [ref.to_record() for ref in references.values()]}
    manifest["bundle_hash"] = bundle_hash(manifest)
    text = (json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
    refusals = _scan(MANIFEST_MEMBER, text, stored)
    if refusals:
        raise BuildRefused(refusals)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
            ordered = [(MANIFEST_MEMBER, text), *((FOLDER_PREFIX + m, d) for m, d in sorted(members.items()))]
            for member, data in ordered:
                info = zipfile.ZipInfo(member, _TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = _MEMBER_MODE
                archive.writestr(info, data)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _experiment_name(folder: Path) -> str:
    """The experiment's name, as the folder's document records it, written as a file name."""
    return _NAME_SAFE.sub("_", read_folder(folder).document["experiment"]) or "experiment"


@dataclass(frozen=True)
class Unpacked:
    """What an unpacked bundle was: its manifest, its lock, and the folder it was written to."""

    manifest: Manifest
    lock: Lock
    folder: Path


def _member_kind_problems(archive: zipfile.ZipFile) -> list[Refusal]:
    found: list[Refusal] = []
    seen: set[str] = set()
    for info in archive.infolist():
        problem = None
        mode = info.external_attr >> 16
        if info.filename in seen:
            problem = "is listed twice"
        elif info.is_dir() or (mode != 0 and stat.S_IFMT(mode) != _REGULAR):
            problem = "is a directory, a link or a special file"
        elif info.flag_bits & 0x1:
            problem = "is encrypted"
        elif info.compress_type != zipfile.ZIP_STORED:
            problem = "is compressed (MIME packs nothing compressed)"
        elif info.filename != MANIFEST_MEMBER and not info.filename.startswith(FOLDER_PREFIX):
            problem = f"is not {MANIFEST_MEMBER} or below {FOLDER_PREFIX}"
        else:
            problem = member_problem(info.filename) or member_problem(info.filename.removeprefix(FOLDER_PREFIX))
        seen.add(info.filename)
        if problem is not None:
            found.append(_refuse("bundle.unsafe", RefusalClass.USER, f"the member {info.filename!r} {problem}",
                                  "this is not a bundle MIME packed; do not unpack it"))
    return found


def bundle_name(bundle: str | os.PathLike[str]) -> str:
    """The experiment name the bundle's manifest gives, written as a file name: where ``pull`` unpacks it by default."""
    try:
        with zipfile.ZipFile(bundle) as archive:
            name = read_manifest(archive)["name"]
    except zipfile.BadZipFile as error:
        raise _unreadable(error) from error
    return name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) else "bundle"


def _unreadable(error: Exception) -> BuildRefused:
    return BuildRefused([_refuse(
        "bundle.unreadable", RefusalClass.USER, f"the bundle cannot be read as a bundle ({error})",
        "fetch it again; a download that stopped early looks like this")])


def read_manifest(archive: zipfile.ZipFile) -> Manifest:
    """The manifest, checked for its shape, its format and its own hash."""
    try:
        info = archive.getinfo(MANIFEST_MEMBER)
    except KeyError:
        raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER, f"the bundle has no {MANIFEST_MEMBER}",
                                     "this is not a bundle MIME packed")]) from None
    if info.file_size > _MANIFEST_LIMIT_BYTES:
        raise BuildRefused([_refuse(
            "bundle.manifest", RefusalClass.USER, f"{MANIFEST_MEMBER} is {size_text(info.file_size)}",
            "this is not a bundle MIME packed")])
    try:
        data: object = json.loads(archive.read(info))
    except (ValueError, zipfile.BadZipFile) as error:
        raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER, f"{MANIFEST_MEMBER} cannot be read ({error})",
                                     "this is not a bundle MIME packed")]) from error
    manifest, problems = conforms(data, Manifest)
    if manifest is None:
        raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER, f"{MANIFEST_MEMBER}: {problem}",
                                     "this is not a bundle of a format this MIME reads") for problem in problems])
    if bundle_hash(cast("dict[str, object]", data)) != manifest["bundle_hash"]:
        raise BuildRefused([_refuse("bundle.changed", RefusalClass.USER, "the manifest does not match its own hash",
                                     "the bundle was altered after it was packed; fetch it again from its source")])
    return manifest


def _verify_member(name: str, digest: str, size: int, expected: BundleFile) -> Refusal | None:
    if size != expected["size"] or digest != expected["sha256"]:
        return _refuse(
            "bundle.changed", RefusalClass.USER,
            f"{name} is not the file the manifest lists (its size or its hash differs)",
            "the bundle was altered after it was packed; fetch it again from its source")
    return None


def unpack(bundle: str | os.PathLike[str], into: str | os.PathLike[str]) -> Unpacked:
    """Check the bundle and write its folder to ``into``, which must not exist (or be an empty folder). Nothing is
    written outside ``into``: a member's name is checked, links and special files are refused, and every file is hashed
    against the manifest while it is written to a hidden folder beside ``into``, which is renamed to ``into`` only when
    every hash matched and the folder reads as a saved experiment. Raises ``BuildRefused``."""
    destination = Path(into)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise BuildRefused([_refuse(
            "pull.destination", RefusalClass.USER, f"{destination} exists and is not an empty folder",
            "unpack into a new folder (--into)")])
    staging = destination.with_name(f".{destination.name}.partial-{secrets.token_hex(8)}")
    try:
        with zipfile.ZipFile(bundle) as archive:
            refusals = _member_kind_problems(archive)
            if refusals:
                raise BuildRefused(refusals)
            manifest = read_manifest(archive)
            listed = {FOLDER_PREFIX + name for name in manifest["files"]}
            held = {info.filename for info in archive.infolist()} - {MANIFEST_MEMBER}
            if listed != held:
                raise BuildRefused([_refuse(
                    "bundle.changed", RefusalClass.USER,
                    f"the bundle's files are not the manifest's: {len(held - listed)} unlisted, "
                    f"{len(listed - held)} missing", "the bundle was altered after it was packed; fetch it again")])
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            problems: list[Refusal] = []
            for name, expected in sorted(manifest["files"].items()):
                info = archive.getinfo(FOLDER_PREFIX + name)
                target = _target(staging, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with archive.open(info) as source, target.open("xb") as sink:
                    while chunk := source.read(_CHUNK):
                        size += len(chunk)
                        if size > expected["size"]:
                            break
                        digest.update(chunk)
                        sink.write(chunk)
                found = _verify_member(name, digest.hexdigest(), size, expected)
                if found is not None:
                    problems.append(found)
            if problems:
                raise BuildRefused(problems)
        form = read_folder(staging)
        if manifest["assets"] != [ref.to_record() for ref in folder_assets(staging)]:
            raise BuildRefused([_refuse(
                "bundle.manifest", RefusalClass.USER, "the manifest's asset references are not the lock's",
                "this is not a bundle MIME packed")])
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    except zipfile.BadZipFile as error:
        raise _unreadable(error) from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return Unpacked(manifest, form.lock, destination)


__all__ = ["BUNDLE_SUFFIX", "FORMAT", "INLINE_LIMIT_BYTES", "Manifest", "Unpacked", "bundle_hash", "bundle_name",
           "member_problem", "pack", "read_manifest", "unpack"]
