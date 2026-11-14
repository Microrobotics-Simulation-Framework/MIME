"""A container: one file holding a folder MIME wrote, to share it and check it on the other side.

Each container has a **kind**, which fixes its suffix, the family of its format ids (``mime.<family>/<n>``), the ones this MIME
reads, the prefix its folder's files sit under, the manifest key that holds its own hash, and its **layout check**: why a
name below the prefix is not a file of the kind's folder. A container is a zip with nothing compressed, so the same folder
always makes the same bytes on any machine. ``manifest.json`` is its first member and the ``NOTICE`` its second; the
folder's files follow under the prefix, sorted, with a fixed timestamp and mode.

What a kind judges before it writes (which files a folder may carry, their terms, the scans) is the kind's own code; this
module writes the zip and reads one back. The reading side checks the format first, then every member's name and kind
against the kind's layout, the manifest's shape and its own hash, and every file's hash while it is written into a
staging folder. It writes through its own file opens and never through an extraction, so that no path can leave the
folder, no link or special file can be made, and nothing the layout does not name is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import unicodedata
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from typing_extensions import ReadOnly, TypedDict

from mime._terminal import plain
from mime.assets import size_text
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.saved._schema import shape_problems
from mime.saved._versions import newer_form_refusal

MANIFEST_MEMBER = "manifest.json"
NOTICE_MEMBER = "NOTICE"
MANIFEST_LIMIT_BYTES = 16_000_000
CHUNK = 1 << 20
_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MEMBER_MODE = 0o100644 << 16
_REGULAR = 0o100000
_DRIVE = re.compile(r"[A-Za-z]:")
#: Names Windows reserves for devices, with or without an extension.
_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$",
                           *(f"{base}{digit}" for base in ("COM", "LPT") for digit in "0123456789¹²³")})
_NAME_LIMIT_BYTES = 255


class BundleFile(TypedDict, closed=True):
    """A carried file, as every kind's manifest lists it: its sha256 and size."""

    sha256: ReadOnly[str]
    size: ReadOnly[int]


class ReferencedFile(TypedDict, closed=True):
    """A file that travels as a reference: the asset of the lock that holds its bytes, and their sha256 and size."""

    name: ReadOnly[str]
    sha256: ReadOnly[str]
    size: ReadOnly[int]


@dataclass(frozen=True)
class ContainerKind:
    """A kind of container: its file suffix, its format family and the format ids this MIME reads (the newest first),
    the prefix its folder's files sit under, the manifest key holding its own hash, its layout check (why a name below
    the prefix is not a file of its folder, or ``None``), and how a message names one (``a bundle``, ``the bundle``).

    The four checks the reading side makes of a name and a file (a name's own form, its place in the layout, where it
    is written, its hash) are read through the kind: ``None`` is this module's own. The bundle's are read from its
    module when called, which is where its controls plant a skipped check."""

    name: str
    suffix: str
    family: str
    reads: tuple[str, ...]
    prefix: str
    hash_key: str
    layout: Callable[[str], str | None]
    article: str
    the: str
    member_check: Callable[[str], str | None] | None = None
    name_check: Callable[[str], str | None] | None = None
    target_of: Callable[[Path, str], Path] | None = None
    verify_of: Callable[[str, str, int, BundleFile], Refusal | None] | None = None

    def member_problem(self, name: str) -> str | None:
        """Why ``name`` cannot be a file inside the folder the container unpacks to (:func:`member_problem`)."""
        return (self.member_check or member_problem)(name)

    def name_problem(self, name: str) -> str | None:
        """Why ``name`` is not a file of this kind's folder: a name every platform accepts, at a place its layout keeps
        one."""
        if self.name_check is not None:
            return self.name_check(name)
        return member_problem(name) or self.layout(name)

    def target(self, root: Path, name: str) -> Path:
        """Where the member ``name`` is written under ``root`` (:func:`target`)."""
        return self.target_of(root, name) if self.target_of is not None else target(root, name, self)

    def verify(self, name: str, digest: str, size: int, expected: BundleFile) -> Refusal | None:
        """``bundle.changed`` where a member is not the file the manifest lists (:func:`verify_member`)."""
        if self.verify_of is not None:
            return self.verify_of(name, digest, size, expected)
        return verify_member(name, digest, size, expected, self)



def refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def _unsafe(kind: ContainerKind, message: str) -> Refusal:
    return refuse("bundle.unsafe", RefusalClass.USER, message,
                  f"this is not {kind.article} `mime pack` wrote; do not unpack it: ask its sender to pack the folder "
                  f"again with `mime pack <folder>`")


def _not_written(kind: ContainerKind, message: str) -> Refusal:
    return refuse("bundle.manifest", RefusalClass.USER, message,
                  f"this is not {kind.article} `mime pack` wrote: ask its sender to pack the folder again with `mime "
                  f"pack <folder>`")


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def container_hash(manifest: Mapping[str, object], key: str) -> str:
    """The hash that names a container's content: the manifest without its own hash (``key``), in a canonical text."""
    body = {name: value for name, value in manifest.items() if name != key}
    return sha256_of(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _part_problem(part: str) -> str | None:
    """Why one part of a path cannot be a name on every filesystem a container may unpack to (Linux, macOS, Windows)."""
    if part.startswith("."):
        return "has a part that starts with a dot (a hidden file, a .git directory)"
    if plain(part) != part:
        return "has a control, format or separator character"
    if any(character in '<>:"|?*' for character in part):
        return "has a character Windows does not allow in a name"
    if part != part.rstrip(". "):
        return "has a part that ends with a dot or a space"
    if part.split(".", 1)[0].rstrip(" ").upper() in _DEVICE_NAMES:
        return "has a part Windows reserves for a device"
    if len(part.encode("utf-8")) > _NAME_LIMIT_BYTES:
        return f"has a part longer than {_NAME_LIMIT_BYTES} bytes"
    return None


def member_problem(name: str) -> str | None:
    """Why ``name`` cannot be a file inside the folder a container unpacks to, or ``None``: a name is a relative path of
    plain parts, as ``/`` writes it, none hidden, none a name a filesystem treats specially."""
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
    return next((problem for part in parts if (problem := _part_problem(part)) is not None), None)


def _collision_key(name: str) -> str:
    """The name as a case-insensitive filesystem that normalises Unicode (macOS, Windows) would read it."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def collisions(names: Sequence[str]) -> list[tuple[str, str]]:
    """The pairs of names that are one name on a case-insensitive filesystem: a container unpacks to the same folder on
    every platform, or is not one."""
    seen: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for name in sorted(set(names)):
        key = _collision_key(name)
        if key in seen:
            pairs.append((seen[key], name))
        seen[key] = seen.get(key, name)
    return pairs


def target(root: Path, name: str, kind: ContainerKind) -> Path:
    """Where the member ``name`` is written under ``root``: its name checked, and the path it makes checked to stay
    inside ``root``."""
    problem = kind.name_problem(name)
    if problem is not None:
        raise BuildRefused([_unsafe(kind, f"a member's name {name!r} {problem}")])
    path = root / name
    if root.resolve() not in path.resolve().parents:
        raise BuildRefused([_unsafe(kind, f"the member {name!r} would be written outside the folder")])
    return path


def files_of(folder: Path, kind: ContainerKind) -> list[tuple[str, Path]]:
    """Every file below ``folder``, by its name as the container writes it, sorted. A link or a special file is
    refused."""
    found: list[tuple[str, Path]] = []
    refusals: list[Refusal] = []
    for directory, subdirectories, names in os.walk(folder):
        for entry in [*subdirectories, *names]:
            path = Path(directory) / entry
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                refusals.append(refuse(
                    "bundle.special-file", RefusalClass.USER, f"{plain(path.relative_to(folder).as_posix())} is a link "
                    f"or a special file", f"{kind.article} holds plain files: copy the file's content in, then `mime "
                                          f"pack <folder>` again"))
            elif path.is_file():
                found.append((path.relative_to(folder).as_posix(), path))
    if refusals:
        raise BuildRefused(refusals)
    return sorted(found)


def write_container(path: Path, manifest: bytes, notice: bytes, members: Mapping[str, bytes],
                    kind: ContainerKind) -> Path:
    """The container at ``path``: ``manifest`` first, then ``notice``, then each of ``members`` under the kind's prefix,
    sorted, each stored with a fixed timestamp and mode; written to a hidden file beside ``path`` and renamed onto it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
            ordered = [(MANIFEST_MEMBER, manifest), (NOTICE_MEMBER, notice),
                       *((kind.prefix + member, data) for member, data in sorted(members.items()))]
            for member, data in ordered:
                info = zipfile.ZipInfo(member, _TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = _MEMBER_MODE
                archive.writestr(info, data)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def member_kind_problems(archive: zipfile.ZipFile, kind: ContainerKind) -> list[Refusal]:
    """Every member that is not one a ``mime pack`` of ``kind`` writes: listed twice, a directory, link or special file,
    encrypted, compressed, outside the manifest, the NOTICE and the prefix, or a name the kind's layout does not hold;
    and every pair of names one on a case-insensitive filesystem."""
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
        elif info.filename not in (MANIFEST_MEMBER, NOTICE_MEMBER) and not info.filename.startswith(kind.prefix):
            problem = f"is not {MANIFEST_MEMBER}, {NOTICE_MEMBER} or below {kind.prefix}"
        else:
            problem = kind.member_problem(info.filename) or (
                None if info.filename in (MANIFEST_MEMBER, NOTICE_MEMBER)
                else kind.name_problem(info.filename.removeprefix(kind.prefix)))
        seen.add(info.filename)
        if problem is not None:
            found.append(_unsafe(kind, f"the member {info.filename!r} {problem}"))
    found += [refuse("bundle.unsafe", RefusalClass.USER,
                     f"the members {one!r} and {other!r} are one name on a case-insensitive filesystem",
                     f"this is not {kind.article} `mime pack` wrote (it could not unpack the same on macOS and Windows): "
                     f"ask its sender to pack the folder again with `mime pack <folder>`")
              for one, other in collisions([info.filename for info in archive.infolist()])]
    return found


def unreadable(error: Exception, kind: ContainerKind) -> BuildRefused:
    return BuildRefused([refuse(
        "bundle.unreadable", RefusalClass.USER, f"{kind.the} cannot be read as {kind.article} ({plain(str(error))})",
        "fetch it again, `mime pull <doi | url>`: a download that stopped early looks like this")])


def format_of(data: object) -> object:
    return cast("dict[str, object]", data).get("format") if isinstance(data, dict) else None


def stated_format(archive: zipfile.ZipFile) -> object:
    """The format a container's manifest states, read before any other check; ``None`` where it cannot be read here,
    which the checks after refuse with their reasons."""
    try:
        info = archive.getinfo(MANIFEST_MEMBER)
    except KeyError:
        return None
    if info.file_size > MANIFEST_LIMIT_BYTES or info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
        return None
    try:
        return format_of(json.loads(archive.read(info)))
    except (ValueError, RecursionError, zipfile.BadZipFile, NotImplementedError, OSError):
        return None


def unknown_format(archive: zipfile.ZipFile, kind: ContainerKind) -> BuildRefused | None:
    """A container of ``kind`` in a format this MIME does not read (a later one), found before any other check, so that
    its member names (a later format may add a member) are never judged as a hostile container's: the advice is to
    install the MIME that packed it. ``None`` for a known format, and for a manifest that cannot be read here."""
    held = stated_format(archive)
    return BuildRefused([refusal]) if (refusal := newer_form_refusal(held, kind.family, kind.reads,
                                                                      kind.the)) is not None else None


def read_manifest(archive: zipfile.ZipFile, kind: ContainerKind, shape: Callable[[object], type[object]],
                  held: Callable[[dict[str, object]], list[Refusal]] | None = None) -> dict[str, object]:
    """The manifest of a container of ``kind``, checked for its format first (a later one is refused with the advice to
    install the MIME that packed it), then for the shape ``shape`` gives for the format it states, then by ``held`` (what
    its format does not hold, though its shape reads), and its own hash."""
    later = unknown_format(archive, kind)
    if later is not None:
        raise later
    try:
        info = archive.getinfo(MANIFEST_MEMBER)
    except KeyError:
        raise BuildRefused([_not_written(kind, f"{kind.the} has no {MANIFEST_MEMBER}")]) from None
    if info.file_size > MANIFEST_LIMIT_BYTES:
        raise BuildRefused([_not_written(kind, f"{MANIFEST_MEMBER} is {size_text(info.file_size)}")])
    try:
        data: object = json.loads(archive.read(info))
        problems = shape_problems(data, shape(format_of(data)))
        if problems:
            raise BuildRefused([refuse("bundle.manifest", RefusalClass.USER, f"{MANIFEST_MEMBER}: {plain(problem)}",
                                       f"this is not {kind.article} of a format this MIME reads: install the MIME that "
                                       f"packed it (`pip install mime-engine==<version>`)") for problem in problems])
        manifest = cast(dict[str, object], data)
        changed = container_hash(manifest, kind.hash_key) != manifest[kind.hash_key]
        refused = held(manifest) if held is not None else []
        if refused:
            raise BuildRefused(refused)
    except RecursionError:
        raise BuildRefused([_not_written(kind, f"{MANIFEST_MEMBER} is nested too deeply to be a manifest")]) from None
    except (ValueError, zipfile.BadZipFile) as error:
        raise BuildRefused([_not_written(kind, f"{MANIFEST_MEMBER} cannot be read ({plain(str(error))})")]) from error
    if changed:
        raise BuildRefused([refuse("bundle.changed", RefusalClass.USER, "the manifest does not match its own hash",
                                   f"{kind.the} was altered after it was packed: fetch it again from its source, "
                                   f"`mime pull <doi | url>`")])
    return manifest


def verify_member(name: str, digest: str, size: int, expected: BundleFile, kind: ContainerKind) -> Refusal | None:
    if size != expected["size"] or digest != expected["sha256"]:
        return refuse("bundle.changed", RefusalClass.USER,
                      f"{name} is not the file the manifest lists (its size or its hash differs)",
                      f"{kind.the} was altered after it was packed: fetch it again from its source, `mime pull <doi | "
                      f"url>`")
    return None


def members_listed(archive: zipfile.ZipFile, files: Mapping[str, BundleFile], kind: ContainerKind,
                   notice: bool = True) -> BuildRefused | None:
    """``bundle.changed`` where the container's members are not exactly the manifest's files (under the prefix) and its
    NOTICE."""
    listed: set[str] = {kind.prefix + name for name in files} | ({NOTICE_MEMBER} if notice else set[str]())
    held = {info.filename for info in archive.infolist()} - {MANIFEST_MEMBER}
    if listed == held:
        return None
    return BuildRefused([refuse(
        "bundle.changed", RefusalClass.USER,
        f"{kind.the}'s files are not the manifest's: {len(held - listed)} unlisted, {len(listed - held)} missing",
        f"{kind.the} was altered after it was packed: fetch it again from its source, `mime pull <doi | url>`")])


def notice_of(archive: zipfile.ZipFile, expected: BundleFile, kind: ContainerKind) -> tuple[str, list[Refusal]]:
    """The ``NOTICE`` a container carries, held to its hash in the manifest."""
    data = archive.read(NOTICE_MEMBER)
    problem = kind.verify(NOTICE_MEMBER, sha256_of(data), len(data), expected)
    return data.decode("utf-8", errors="replace"), [problem] if problem is not None else []


def destination_problem(into: str | os.PathLike[str]) -> BuildRefused | None:
    """Why a container cannot be unpacked to ``into`` (it must not exist, or be an empty folder), or ``None``: checked
    before anything is downloaded, and again before anything is written."""
    destination = Path(os.path.abspath(into))
    if not destination.name:
        return BuildRefused([refuse("pull.destination", RefusalClass.USER, f"{destination} is not a folder to create",
                                    "unpack into a new folder (--into)")])
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        return BuildRefused([refuse(
            "pull.destination", RefusalClass.USER, f"{destination} exists and is not an empty folder",
            "unpack into a new folder (--into)")])
    return None


def write_members(archive: zipfile.ZipFile, files: Mapping[str, BundleFile], staging: Path,
                  kind: ContainerKind) -> list[Refusal]:
    """Every file of the manifest written under ``staging`` while it is hashed; the refusals of those whose size or hash
    differs."""
    problems: list[Refusal] = []
    for name, expected in sorted(files.items()):
        info = archive.getinfo(kind.prefix + name)
        path = kind.target(staging, name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with archive.open(info) as source, path.open("xb") as sink:
                while chunk := source.read(CHUNK):
                    size += len(chunk)
                    if size > expected["size"]:
                        break
                    digest.update(chunk)
                    sink.write(chunk)
        except (FileExistsError, NotADirectoryError, IsADirectoryError):
            raise BuildRefused([_unsafe(
                kind, f"the member {name!r} would be written where another member, or a folder, already is")]) from None
        found = kind.verify(plain(name), digest.hexdigest(), size, expected)
        if found is not None:
            problems.append(found)
    return problems


def staging_of(destination: Path) -> Path:
    """The hidden folder beside ``destination`` a container is unpacked into before it is renamed onto it."""
    return destination.with_name(f".{destination.name}.partial-{secrets.token_hex(8)}")


__all__ = ["CHUNK", "MANIFEST_LIMIT_BYTES", "MANIFEST_MEMBER", "NOTICE_MEMBER", "BundleFile", "ContainerKind",
           "ReferencedFile", "collisions", "container_hash", "destination_problem", "files_of", "format_of",
           "member_kind_problems", "member_problem", "members_listed", "notice_of", "read_manifest", "refuse",
           "sha256_of", "staging_of", "stated_format", "target", "unknown_format", "unreadable", "verify_member",
           "write_container", "write_members"]
