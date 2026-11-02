"""A bundle: one file holding a saved experiment folder, to be shared and checked on the other side.

It is a zip with nothing compressed, so the same folder always makes the same bytes on any machine (a compressor's
output is not the same across zlib builds). The first member is ``manifest.json``: the format, the experiment's name,
every carried file's sha256 and size, each file that travels as a reference with the asset that holds it, the
``NOTICE``'s hash, the asset references of the lock, and the bundle's own hash, the sha256 of the rest of the manifest.
The ``NOTICE`` follows (each file's licence, credit, verdict and declared permission, ``mime.saved._card``), then the
folder's files under ``experiment/``, sorted, with a fixed timestamp and mode.

**Format 2** (``mime.bundle/2``) carries a file only where its terms let it be redistributed: a file that may not be
redistributed travels as a reference to its origin, **whatever its size**, and never inside a bundle, nor its bytes at
another path; so does a file over the size a bundle carries that has an origin (``mime.saved._terms``). Each reference's
asset is held to the file's terms, and an asset MIME ships to its shipped licence: the lock is outside the hashed
record. Format 1 (``mime.bundle/1``) still unpacks. A bundle of a format this MIME does not read (a later one) is
refused first, before any other check, as ``load.newer-form``, with the advice to install the MIME that wrote it; a MIME
before format 2 (a development checkout) instead refuses one as not a bundle ``mime pack`` wrote, except ``mime pull`` without
``--into``, which reads the manifest first.

``pack`` writes one after checking that the folder loads, and refuses to carry a path of the machine that wrote it,
any token MIME holds, or any file a saved folder does not name (an ``.env``, a ``.git`` directory, an editor's file).
``unpack`` is the reading side: it checks every name and member kind against the saved form's own layout, then every
hash, and writes only into a new folder, through its own file opens and never an extraction, so that no path can leave
the folder, no link or special file can be made, and nothing but what the saved form names is written.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import shutil
import stat
import unicodedata
import zipfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from typing_extensions import ReadOnly, TypedDict

from mime._terminal import plain
from mime.assets import folder_assets, size_text
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token, held_tokens
from mime.saved._folder import layout_problem, named_files, read_folder
from mime.saved._schema import Lock, RemoteAssetRecord, conforms
from mime.saved._terms import INLINE_LIMIT_BYTES, FileTerms, FolderTerms, folder_terms, place_referenced
from mime.saved._versions import newer_form_refusal

BUNDLE_SUFFIX = ".mimebundle"
FORMAT = "mime.bundle/2"
FORMAT_1 = "mime.bundle/1"
MANIFEST_MEMBER = "manifest.json"
NOTICE_MEMBER = "NOTICE"
FOLDER_PREFIX = "experiment/"

#: The record's credit and description beside the saved form (``mime push --init`` writes it): neither carried nor
#: refused, the one file of a folder that is not the saved form's and is not a stray (design #199, section 8 item 7).
SHARING_FILE = "sharing.yaml"
_MANIFEST_LIMIT_BYTES = 16_000_000
_CHUNK = 1 << 20
_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MEMBER_MODE = 0o100644 << 16
_REGULAR = 0o100000

#: An absolute path of a machine, which no bundle may carry: the rule of the documentation generator's pages
#: (``mime.docs._build``, which a test holds this to), as bytes since a file is searched as it is stored. A pack
#: refuses these and the forms below.
_MACHINE_PATH = re.compile(rb"(?<![\w./-])/(?:home|Users|root|tmp|var|mnt|opt|srv|private|runner)/\S")
_WINDOWS_HOME = re.compile(rb"(?<![A-Za-z])[A-Za-z]:[\\/]+Users[\\/]", re.IGNORECASE)
#: What a pack refuses beyond the generator's rule, each a way a machine's path was seen to hide from it: the same
#: roots behind ``file://`` (the third slash is preceded by a slash, which the rule's look-behind excludes), the roots
#: of a mounted disk (``/Volumes``, ``/media``) and of a session (``/run/user``), any Windows drive written with a
#: backslash (or a slash and a name), and a UNC share. Not flagged, on purpose: ``/data``, ``/scratch`` and
#: ``/work``, which are as likely a rooted name inside a record as a cluster's disk, and ``~/`` and ``$HOME/``,
#: which are not absolute. The strings are as a file stores them: a backslash is doubled in JSON and in a quoted YAML
#: string.
_FILE_URL = re.compile(rb"file:/{2,}(?:home|Users|root|tmp|var|mnt|opt|srv|private|runner|Volumes|media|run/user)/\S")
_MOUNTED = re.compile(rb"(?<![\w./-])/(?:Volumes|media|run/user)/\S")
_DRIVE_PATH = re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:(?:\\{1,2}[^\s\\]|/[^\s/\\])")
_UNC_PATH = re.compile(rb"(?<![\\\w])(?:\\{4}[\w.-]+\\{2}[\w.$-]+|\\{2}[\w.-]+\\[\w.$-]+)")
#: The first two are long literals, searched in every file. A drive letter and a UNC share are a few characters, which
#: random bytes spell once in a million positions, so they are searched only in a file that is text (valid UTF-8 with no
#: NUL): a binary array of a few megabytes would otherwise be refused for a path it does not hold.
_MORE_PATHS = (_FILE_URL, _MOUNTED)
_TEXT_PATHS = (_DRIVE_PATH, _UNC_PATH)
_DRIVE = re.compile(r"[A-Za-z]:")
_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")

#: The most a compressed file may expand to while a pack searches inside it: a file that expands further cannot be
#: searched, and is refused rather than carried unsearched.
EXPANSION_LIMIT_BYTES = 200_000_000
_NESTING_LIMIT = 4
#: Names Windows reserves for devices, with or without an extension.
_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$",
                           *(f"{base}{digit}" for base in ("COM", "LPT") for digit in "0123456789\u00b9\u00b2\u00b3")})
_NAME_LIMIT_BYTES = 255


class BundleFile(TypedDict, closed=True):
    sha256: ReadOnly[str]
    size: ReadOnly[int]


class ReferencedFile(TypedDict, closed=True):
    """A file that travels as a reference: the asset of the lock that holds its bytes, and their sha256 and size."""

    name: ReadOnly[str]
    sha256: ReadOnly[str]
    size: ReadOnly[int]


class Manifest(TypedDict, closed=True):
    """What a bundle says it holds. ``bundle_hash`` is the sha256 of the manifest without it. ``files`` are the files
    carried; ``referenced`` those that travel as references, never carried; together they are every file the saved
    folder names."""

    format: ReadOnly[Literal["mime.bundle/2"]]
    name: ReadOnly[str]
    files: ReadOnly[dict[str, BundleFile]]
    referenced: ReadOnly[dict[str, ReferencedFile]]
    notice: ReadOnly[BundleFile]
    assets: ReadOnly[list[RemoteAssetRecord]]
    bundle_hash: ReadOnly[str]


class ManifestFormat1(TypedDict, closed=True):
    """A format 1 manifest: every file carried, no ``NOTICE``."""

    format: ReadOnly[Literal["mime.bundle/1"]]
    name: ReadOnly[str]
    files: ReadOnly[dict[str, BundleFile]]
    assets: ReadOnly[list[RemoteAssetRecord]]
    bundle_hash: ReadOnly[str]


def referenced_of(manifest: Manifest | ManifestFormat1) -> dict[str, ReferencedFile]:
    """The files a manifest says travel as references: none in format 1."""
    return dict(manifest["referenced"]) if manifest["format"] == FORMAT else {}


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_hash(manifest: dict[str, object]) -> str:
    """The hash that names a bundle's content: the manifest without its own hash, in a canonical text."""
    body = {key: value for key, value in manifest.items() if key != "bundle_hash"}
    return sha256_of(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _part_problem(part: str) -> str | None:
    """Why one part of a path cannot be a name on every filesystem a bundle may unpack to (Linux, macOS, Windows)."""
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
    """Why ``name`` cannot be a file inside the folder a bundle unpacks to, or ``None``: a name is a relative path of
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


def name_problem(name: str) -> str | None:
    """Why ``name`` is not a file of a saved folder: a name every platform accepts, at a place the saved form keeps one."""
    return member_problem(name) or layout_problem(name)


def _collision_key(name: str) -> str:
    """The name as a case-insensitive filesystem that normalises Unicode (macOS, Windows) would read it."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def collisions(names: Sequence[str]) -> list[tuple[str, str]]:
    """The pairs of names that are one name on a case-insensitive filesystem: a bundle unpacks to the same folder on
    every platform, or is not a bundle."""
    seen: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for name in sorted(set(names)):
        key = _collision_key(name)
        if key in seen:
            pairs.append((seen[key], name))
        seen[key] = seen.get(key, name)
    return pairs


def _target(root: Path, name: str) -> Path:
    """Where the member ``name`` is written under ``root``: its name checked, and the path it makes checked to stay
    inside ``root``."""
    problem = name_problem(name)
    if problem is not None:
        raise BuildRefused([_refuse("bundle.unsafe", RefusalClass.USER, f"a member's name {name!r} {problem}",
                                     "this is not a bundle `mime pack` wrote; do not unpack it: ask its sender to pack "
                                     "the folder again with `mime pack <folder>`")])
    target = root / name
    if root.resolve() not in target.resolve().parents:
        raise BuildRefused([_refuse(
            "bundle.unsafe", RefusalClass.USER, f"the member {name!r} would be written outside the folder",
            "this is not a bundle `mime pack` wrote; do not unpack it: ask its sender to pack the folder again with "
            "`mime pack <folder>`")])
    return target


class _Unscannable(Exception):
    """A compressed file that expands past :data:`EXPANSION_LIMIT_BYTES`, or cannot be read inside."""


def _take(budget: list[int], amount: int) -> None:
    budget[0] -= amount
    if budget[0] < 0:
        raise _Unscannable


def _gunzipped(data: bytes, budget: list[int]) -> bytes | None:
    """What the gzip stream (or the several, joined) in ``data`` holds, or ``None`` when it is not gzip."""
    chunks: list[bytes] = []
    rest = data
    while rest.startswith(b"\x1f\x8b"):
        stream = zlib.decompressobj(wbits=31)
        try:
            chunk = stream.decompress(rest, budget[0] + 1)
        except zlib.error:
            return b"".join(chunks) if chunks else None
        _take(budget, len(chunk))
        chunks.append(chunk)
        if stream.unconsumed_tail:
            raise _Unscannable
        rest = stream.unused_data
    return b"".join(chunks)


def _inside(data: bytes, budget: list[int], depth: int = 0) -> Iterator[bytes]:
    """What ``data`` holds in compressed form: the inside of a gzip stream, and of a compressed member of a zip archive
    (a stored member is already in ``data``), each searched again for another layer."""
    if depth > _NESTING_LIMIT:
        raise _Unscannable
    if data.startswith(b"\x1f\x8b"):
        content = _gunzipped(data, budget)
        if content is not None:
            yield content
            yield from _inside(content, budget, depth + 1)
    elif data.startswith(b"PK\x03\x04") and zipfile.is_zipfile(io.BytesIO(data)):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    with archive.open(info) as member:
                        chunks: list[bytes] = []
                        while chunk := member.read(_CHUNK):
                            _take(budget, len(chunk))
                            chunks.append(chunk)
                    content = b"".join(chunks)
                    if info.compress_type != zipfile.ZIP_STORED:
                        yield content
                    yield from _inside(content, budget, depth + 1)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError) as error:
            raise _Unscannable from error


def _is_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _scan_one(name: str, data: bytes, tokens: Sequence[Token]) -> list[Refusal]:
    found: list[Refusal] = []
    home = os.path.expanduser("~")
    patterns = (*_MORE_PATHS, *(_TEXT_PATHS if _is_text(data) else ()))
    machine = (_MACHINE_PATH.search(data) or _WINDOWS_HOME.search(data)
               or next((hit for pattern in patterns if (hit := pattern.search(data)) is not None), None))
    if machine is None and len(home) > 1 and home.encode("utf-8") in data:
        machine = re.search(re.escape(home.encode("utf-8")), data)
    if machine is not None:
        # The path is not quoted: it is the user's, and the message is read by others.
        found.append(_refuse(
            "bundle.machine-path", RefusalClass.USER,
            f"{plain(name)} holds a path of this machine (at byte {machine.start()})",
            "write the path as a name below the folder, or remove it, then save the experiment again, "
            "built.save(folder), and `mime pack <folder>`"))
    if any(each.occurs_in(data) for each in tokens):
        found.append(_refuse("bundle.credential", RefusalClass.USER,
                              f"{plain(name)} holds a token MIME has stored for you",
                              "remove the token from that file first (a token never belongs in an experiment), save "
                              "the experiment again, built.save(folder), and pack it again, `mime pack <folder>`; keep "
                              "the stored token until the pack passes, since the pack finds only tokens MIME holds; "
                              "then revoke it at the service if it was shared"))
    return found


def _scan(name: str, data: bytes, tokens: Sequence[Token]) -> list[Refusal]:
    """Every reason ``data`` cannot be carried: it holds a path of this machine or a token MIME holds (as written, or
    percent-encoded, JSON-escaped, in base64, in hex or in UTF-16), in the file or inside a gzip stream or a compressed
    zip member. The search is for the forms listed, not for every encoding a person could invent, which is why a bundle
    holds only the files the saved form names."""
    found = _scan_one(name, data, tokens)
    budget = [EXPANSION_LIMIT_BYTES]
    try:
        for content in _inside(data, budget):
            found += [each for each in _scan_one(f"{name} (inside it)", content, tokens)
                      if each.code not in {seen.code for seen in found}]
    except _Unscannable:
        found.append(_refuse(
            "bundle.unscannable", RefusalClass.USER,
            f"{plain(name)} is compressed in a way MIME cannot search (it expands past "
            f"{size_text(EXPANSION_LIMIT_BYTES)}, is nested more than {_NESTING_LIMIT} deep, or cannot be read)",
            "store the file uncompressed, as built.save(folder) writes it, then `mime pack <folder>` again"))
    return found


def scan_text(name: str, data: bytes, tokens: Sequence[Token]) -> list[Refusal]:
    """The scans a bundle's files are held to, for text that is not one of them: the NOTICE, the card and the metadata
    ``mime push`` sends, so a token pasted into a description is caught there too."""
    return _scan(name, data, tokens)


def _files_of(folder: Path) -> list[tuple[str, Path]]:
    """Every file below ``folder``, by its name as the bundle writes it, sorted. A link or a special file is refused."""
    found: list[tuple[str, Path]] = []
    refusals: list[Refusal] = []
    for directory, subdirectories, names in os.walk(folder):
        for entry in [*subdirectories, *names]:
            path = Path(directory) / entry
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                refusals.append(_refuse(
                    "bundle.special-file", RefusalClass.USER, f"{plain(path.relative_to(folder).as_posix())} is a link "
                    f"or a special file", "a bundle holds plain files: copy the file's content in, then `mime pack "
                                          "<folder>` again"))
            elif path.is_file():
                found.append((path.relative_to(folder).as_posix(), path))
    if refusals:
        raise BuildRefused(refusals)
    return sorted(found)


def _scanned(member: str, path: Path, tokens: Sequence[Token], refusals: list[Refusal]) -> bytes | None:
    """The bytes of ``path`` when nothing in them is refused; otherwise the reasons are added to ``refusals`` and the bytes
    are not returned, so that the caller's frame never holds a file that holds a token."""
    data = path.read_bytes()
    found = _scan(member, data, tokens)
    refusals += found
    return None if found else data


def _stray_files(files: Sequence[tuple[str, Path]], named: set[str]) -> list[Refusal]:
    """A refusal for each file the saved form does not name, or whose name some platform cannot hold, and for each pair
    of names one platform reads as one."""
    found = [_refuse("bundle.stray-file", RefusalClass.USER,
                     f"{plain(member)} is not a file the saved experiment names: it {problem or 'is not listed in the lock'}",
                     "a bundle carries what the saved folder holds and nothing else (an .env, a .git directory and an "
                     "editor's file stay behind): remove the file, or keep the folder somewhere it is not, then `mime "
                     "pack <folder>` again")
             for member, _ in files if (problem := name_problem(member)) is not None or member not in named]
    found += [_refuse("bundle.name-collision", RefusalClass.USER,
                      f"{plain(one)} and {plain(other)} are one name on a case-insensitive filesystem",
                      "rename one, so that the bundle unpacks to the same folder on macOS and Windows as on Linux; "
                      "then `mime pack <folder>`")
              for one, other in collisions([member for member, _ in files])]
    return found


def _reference_problem(file: str, problem: str) -> Refusal:
    return _refuse("bundle.lock-terms", RefusalClass.USER,
                   f"{plain(file)}: {problem}, so the lock is not the one its document's terms make",
                   "save the experiment again, built.save(folder), which writes the lock from the terms; then `mime "
                   "pack <folder>`")


def referenced_members(judged: FolderTerms, lock: Lock) -> tuple[dict[str, ReferencedFile], list[Refusal]]:
    """The files that travel as references, each with its asset, judged twice and held to agree: by each file's terms
    now (a file that may not be redistributed always travels as one, whatever its size) and by the lock's
    ``referenced_files``; and each one's asset held to its terms, the asset's licence their licence and its locations
    their origin, primary first, since the lock is outside the hashed record and the document's terms are inside it. A
    file that must travel as a reference and has no origin is refused (``bundle.no-location``); a file the lock lists
    whose terms give no origin, any other disagreement, or an asset whose bytes are not the file's, as
    ``bundle.lock-terms``; and a carried file whose bytes are a referenced file's, at whatever path and with whatever
    terms, as ``bundle.referenced-copy``, naming both."""
    listed = lock.get("referenced_files", {})
    assets = {held["name"]: held for held in lock.get("assets", [])}
    found: dict[str, ReferencedFile] = {}
    refusals: list[Refusal] = []
    for file in judged.files:
        must = not file.verdict.carried
        held = assets.get(listed[file.path]) if file.path in listed else None
        if must and file.path not in listed and not file.origin:
            refusals.append(_refuse(
                "bundle.no-location", RefusalClass.USER,
                f"{plain(file.path)} may not be redistributed (licence {plain(file.licence or 'none')}: "
                f"{file.verdict.words()}), so it travels as a reference, and its terms give no origin to fetch it from",
                "give its terms the locations of its exact bytes, pinned: asset(path, terms=Terms(..., "
                "origin=(\"https://...\",))), then save the experiment again, built.save(folder); or declare the rights "
                "holder's permission: exp.permission(Permission(...))"))
        elif must and file.path not in listed:
            refusals.append(_reference_problem(file.path, "it may not be redistributed and the lock does not say it "
                                                          "travels as a reference"))
        elif file.path in listed and held is None:
            refusals.append(_reference_problem(file.path, f"the lock says it travels as the asset {listed[file.path]!r}, "
                                                          f"which its assets do not name"))
        elif file.path in listed and not file.origin:
            refusals.append(_reference_problem(file.path, "the lock says it travels as a reference, and no terms of "
                                                          "it give an origin to fetch it from (save writes a reference "
                                                          "only from terms that give one)"))
        elif held is not None and (held["sha256"], held["size"]) != (file.sha256, file.size):
            refusals.append(_reference_problem(file.path, f"the lock's asset {held['name']!r} holds other bytes than "
                                                          f"the file"))
        elif held is not None and not must and file.size <= INLINE_LIMIT_BYTES:
            refusals.append(_reference_problem(file.path, "the lock says it travels as a reference, and its terms let "
                                                          "it travel in the bundle"))
        elif held is not None and _restated(held, file, lock.get("publication")):
            refusals.append(_reference_problem(
                file.path, f"the lock's asset {held['name']!r} states the licence {plain(held['licence'])!r} and the "
                           f"locations {', '.join(plain(each) for each in held['locations'])}, and the file's terms "
                           f"the licence {plain(file.licence or 'none')!r} and the origin "
                           f"{', '.join(plain(each) for each in file.origin)}"))
        elif held is not None:
            found[file.path] = {"name": held["name"], "sha256": held["sha256"], "size": held["size"]}
    refusals += _copy_problems(judged, set(listed))
    return found, refusals


def _restated(held: RemoteAssetRecord, file: FileTerms, publication: Mapping[str, object] | None = None) -> bool:
    """Whether a reference's asset, as the lock holds it, states another licence or other locations than the file's
    terms give (their licence, and their origin, primary first). A lock ``mime push`` wrote into a published bundle
    names the record that holds the file first (its ``publication``): that one location is the record's, before the
    origin."""
    locations = tuple(held["locations"])
    if publication is not None and locations and _own_location(locations[0], publication):
        locations = locations[1:]
    return (held["licence"], locations) != (file.licence, file.origin)


def _own_location(location: str, publication: Mapping[str, object]) -> bool:
    """Whether ``location`` is the published record's own, as ``mime push`` names it: its DOI and a file, or the
    file's address in its service's records API."""
    return location.startswith(f"doi:{publication.get('doi')}/") or \
        re.fullmatch(rf"https?://[^/]+(/[^/]+)*/records/{re.escape(str(publication.get('record')))}/files/[^/]+/content",
                     location) is not None


def _copy_problems(judged: FolderTerms, listed: set[str]) -> list[Refusal]:
    """``bundle.referenced-copy`` for each file that would be carried and holds the bytes of a file that travels as a
    reference (a mesh copied for a second prim, or under another name): terms bind by path, so the copy is refused
    rather than carried, whatever its terms say, and the same bytes are never in a bundle at another path."""
    travelling = {file.path for file in judged.files if not file.verdict.carried or file.path in listed}
    by_bytes = {file.sha256: file for file in judged.files if file.path in travelling}
    found: list[Refusal] = []
    for file in judged.files:
        original = by_bytes.get(file.sha256)
        if file.path in travelling or original is None:
            continue
        found.append(_refuse(
            "bundle.referenced-copy", RefusalClass.USER,
            f"{plain(file.path)} holds the same bytes as {plain(original.path)} (sha256 {file.sha256}), which travels "
            f"as a reference ({original.verdict.words()}, by {original.verdict.by}), so the bundle would carry those "
            f"bytes at a second path ({file.verdict.words()}, by {file.verdict.by})",
            f"declare {plain(file.path)} with the terms of {plain(original.path)}, asset(path, terms=Terms(...)), or "
            f"keep one copy and reference it at each prim; then save the experiment again, built.save(folder), and "
            f"`mime pack <folder>`"))
    return found


def shipped_asset_problems(lock: Lock) -> list[Refusal]:
    """``bundle.lock-terms`` for each asset of the lock that is one MIME ships (``mime.assets.known``) whose licence
    the lock states otherwise than MIME ships it: the lock is outside the hashed record, so the licence a bundle's
    manifest repeats is re-derived from its source where MIME knows it. An asset that is neither MIME's nor a referenced
    file's (``referenced_members``) has no source MIME knows, and is not judged here."""
    from mime.assets import known

    shipped = {ref.sha256: ref for ref in known().values()}
    found: list[Refusal] = []
    for held in lock.get("assets", []):
        ref = shipped.get(held["sha256"])
        if ref is not None and held["licence"] != ref.licence:
            found.append(_refuse(
                "bundle.lock-terms", RefusalClass.USER,
                f"the lock's asset {plain(held['name'])!r} states the licence {plain(held['licence'])!r}, and MIME "
                f"ships it ({ref.name}) under {ref.licence!r}, so the lock is not the one its build writes",
                "save the experiment again, built.save(folder), which writes the lock's assets as MIME ships them; "
                "then `mime pack <folder>`"))
    return found


def pack(folder: str | os.PathLike[str], *, out: str | os.PathLike[str] | None = None) -> Path:
    """Write the bundle of the saved experiment in ``folder`` (by default ``<folder>.mimebundle`` beside it) and return
    its path. The folder must load (``Experiment.load``: a ``LoadRefused`` otherwise). A file that may not be
    redistributed is never carried, whatever its size: it travels as a reference to its origin, as the lock names it,
    and a file over :data:`INLINE_LIMIT_BYTES` likewise. Refused, with every reason, when a file holds a path of this
    machine or a token MIME has stored, when a file is not one the saved form names (or its name is not one every
    platform holds; ``sharing.yaml``, push's credit, is neither carried nor refused), when a file that may not be redistributed has no origin, when the lock's references are not the
    ones the terms make (or a reference's asset does not state its terms' licence and origin), when a file that would
    be carried holds the bytes of one that travels as a reference, when an asset MIME ships is stated under another
    licence, when a licence names an id the installed packaging does not list, when a file is over the size limit and
    travels as no reference, or when the folder holds a link or a special file. The same folder makes the same bytes.
    The experiment's own licence may be absent: ``pack`` never refuses that, and the ``NOTICE`` says so."""
    from mime.experiment import Experiment
    from mime.saved._card import notice_text

    root = Path(os.path.abspath(folder))
    Experiment.load(root)
    form = read_folder(root)
    name = _NAME_SAFE.sub("_", form.document["experiment"]) or "experiment"
    target = Path(out) if out is not None else root.with_name(root.name + BUNDLE_SUFFIX)
    if root.resolve() in (target.resolve(), *target.resolve().parents):
        raise BuildRefused([_refuse(
            "bundle.output-inside", RefusalClass.USER, "the bundle would be written inside the folder it packs",
            "write it beside the folder (--out)")])
    references = {ref.sha256: ref for ref in folder_assets(root)}
    stored = held_tokens()
    files = [(member, path) for member, path in _files_of(root) if member != SHARING_FILE]
    named = named_files(form)
    refusals: list[Refusal] = _stray_files(files, named)
    if refusals:
        raise BuildRefused(refusals)
    judged = folder_terms(root, form.document, form.lock, named, form.version)
    refusals = judged.unlisted_refusals("mime pack")
    if refusals:
        raise BuildRefused(refusals)
    referenced, refusals = referenced_members(judged, form.lock)
    refusals += shipped_asset_problems(form.lock)
    members: dict[str, bytes] = {}
    for member, path in files:
        if member in referenced:
            continue
        size = path.stat().st_size
        if size > INLINE_LIMIT_BYTES:
            refusals.append(_refuse(
                "bundle.file-too-large", RefusalClass.USER,
                f"{member} has {size_text(size)}, over the {size_text(INLINE_LIMIT_BYTES)} a bundle carries",
                "give its terms an origin, asset(path, terms=Terms(..., origin=(\"https://...\",))), and save the "
                "experiment again, built.save(folder): it then travels as a reference that `mime assets fetch "
                "<folder>` fetches on request"))
            continue
        carried = _scanned(member, path, stored, refusals)
        if carried is not None:
            members[member] = carried
    notice = notice_text(judged, form.document, form.lock).encode("utf-8")
    refusals += _scan(NOTICE_MEMBER, notice, stored)
    if refusals:
        raise BuildRefused(refusals)
    manifest: dict[str, object] = {
        "format": FORMAT, "name": name,
        "files": {member: {"sha256": sha256_of(data), "size": len(data)} for member, data in sorted(members.items())},
        "referenced": dict(sorted(referenced.items())),
        "notice": {"sha256": sha256_of(notice), "size": len(notice)},
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
            ordered = [(MANIFEST_MEMBER, text), (NOTICE_MEMBER, notice),
                       *((FOLDER_PREFIX + m, d) for m, d in sorted(members.items()))]
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


@dataclass(frozen=True)
class Unpacked:
    """What an unpacked bundle was: its manifest, its lock, and the folder it was written to; its ``NOTICE`` (empty
    in format 1); and the files that travel as references, those placed from the asset cache and those still to be
    fetched (``mime assets fetch <folder>`` fetches and places them)."""

    manifest: Manifest | ManifestFormat1
    lock: Lock
    folder: Path
    notice: str = ""
    placed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


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
        elif info.filename not in (MANIFEST_MEMBER, NOTICE_MEMBER) and not info.filename.startswith(FOLDER_PREFIX):
            problem = f"is not {MANIFEST_MEMBER}, {NOTICE_MEMBER} or below {FOLDER_PREFIX}"
        else:
            problem = member_problem(info.filename) or (
                None if info.filename in (MANIFEST_MEMBER, NOTICE_MEMBER)
                else name_problem(info.filename.removeprefix(FOLDER_PREFIX)))
        seen.add(info.filename)
        if problem is not None:
            found.append(_refuse("bundle.unsafe", RefusalClass.USER, f"the member {info.filename!r} {problem}",
                                  "this is not a bundle `mime pack` wrote; do not unpack it: ask its sender to pack "
                                  "the folder again with `mime pack <folder>`"))
    found += [_refuse("bundle.unsafe", RefusalClass.USER,
                      f"the members {one!r} and {other!r} are one name on a case-insensitive filesystem",
                      "this is not a bundle `mime pack` wrote (it could not unpack the same on macOS and Windows): ask "
                      "its "
                      "sender to pack the folder again with `mime pack <folder>`")
              for one, other in collisions([info.filename for info in archive.infolist()])]
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
        "bundle.unreadable", RefusalClass.USER, f"the bundle cannot be read as a bundle ({plain(str(error))})",
        "fetch it again, `mime pull <doi | url>`: a download that stopped early looks like this")])


def _format_of(data: object) -> object:
    return cast("dict[str, object]", data).get("format") if isinstance(data, dict) else None


def unknown_format(archive: zipfile.ZipFile) -> BuildRefused | None:
    """A bundle of a format this MIME does not read (a later one), found before any other check, so that its member
    names (a later format may add a member) are never judged as a hostile bundle's: the advice is to install the MIME
    that packed it. ``None`` for a known format, and for a manifest that cannot be read here, which the checks after
    refuse with their reasons."""
    try:
        info = archive.getinfo(MANIFEST_MEMBER)
    except KeyError:
        return None
    if info.file_size > _MANIFEST_LIMIT_BYTES or info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
        return None
    try:
        held = _format_of(json.loads(archive.read(info)))
    except (ValueError, RecursionError, zipfile.BadZipFile, NotImplementedError, OSError):
        return None
    return BuildRefused([refusal]) if (refusal := newer_form_refusal(held, "bundle", (FORMAT, FORMAT_1),
                                                                      "the bundle")) is not None else None


def read_manifest(archive: zipfile.ZipFile) -> Manifest | ManifestFormat1:
    """The manifest, checked for its format first (2, or 1 which every file carried: a later one is refused with the
    advice to install the MIME that packed it), then its shape and its own hash."""
    later = unknown_format(archive)
    if later is not None:
        raise later
    try:
        info = archive.getinfo(MANIFEST_MEMBER)
    except KeyError:
        raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER, f"the bundle has no {MANIFEST_MEMBER}",
                                     "this is not a bundle `mime pack` wrote: ask its sender to pack the folder again "
                                     "with `mime pack <folder>`")]) from None
    if info.file_size > _MANIFEST_LIMIT_BYTES:
        raise BuildRefused([_refuse(
            "bundle.manifest", RefusalClass.USER, f"{MANIFEST_MEMBER} is {size_text(info.file_size)}",
            "this is not a bundle `mime pack` wrote: ask its sender to pack the folder again with `mime pack "
            "<folder>`")])
    try:
        data: object = json.loads(archive.read(info))
        older = _format_of(data) == FORMAT_1
        manifest: Manifest | ManifestFormat1 | None
        manifest, problems = conforms(data, ManifestFormat1) if older else conforms(data, Manifest)
        if manifest is None:
            raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER, f"{MANIFEST_MEMBER}: {plain(problem)}",
                                         "this is not a bundle of a format this MIME reads: install the MIME that "
                                         "packed it (`pip install mime-engine==<version>`)") for problem in problems])
        changed = bundle_hash(cast("dict[str, object]", data)) != manifest["bundle_hash"]
    except RecursionError:
        raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER,
                                     f"{MANIFEST_MEMBER} is nested too deeply to be a manifest",
                                     "this is not a bundle `mime pack` wrote: ask its sender to pack the folder again "
                                     "with `mime pack <folder>`")]) from None
    except (ValueError, zipfile.BadZipFile) as error:
        raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER,
                                     f"{MANIFEST_MEMBER} cannot be read ({plain(str(error))})",
                                     "this is not a bundle `mime pack` wrote: ask its sender to pack the folder again "
                                     "with `mime pack <folder>`")]) from error
    if changed:
        raise BuildRefused([_refuse("bundle.changed", RefusalClass.USER, "the manifest does not match its own hash",
                                     "the bundle was altered after it was packed: fetch it again from its source, "
                                     "`mime pull <doi | url>`")])
    return manifest


def _verify_member(name: str, digest: str, size: int, expected: BundleFile) -> Refusal | None:
    if size != expected["size"] or digest != expected["sha256"]:
        return _refuse(
            "bundle.changed", RefusalClass.USER,
            f"{name} is not the file the manifest lists (its size or its hash differs)",
            "the bundle was altered after it was packed: fetch it again from its source, `mime pull <doi | url>`")
    return None


def destination_problem(into: str | os.PathLike[str]) -> BuildRefused | None:
    """Why a bundle cannot be unpacked to ``into`` (it must not exist, or be an empty folder), or ``None``: checked
    before anything is downloaded, and again before anything is written."""
    destination = Path(os.path.abspath(into))
    if not destination.name:
        return BuildRefused([_refuse("pull.destination", RefusalClass.USER, f"{destination} is not a folder to create",
                                      "unpack into a new folder (--into)")])
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        return BuildRefused([_refuse(
            "pull.destination", RefusalClass.USER, f"{destination} exists and is not an empty folder",
            "unpack into a new folder (--into)")])
    return None


def _write_members(archive: zipfile.ZipFile, manifest: Manifest | ManifestFormat1, staging: Path) -> list[Refusal]:
    """Every file of the manifest written under ``staging`` while it is hashed; the refusals of those whose size or hash
    differs."""
    problems: list[Refusal] = []
    for name, expected in sorted(manifest["files"].items()):
        info = archive.getinfo(FOLDER_PREFIX + name)
        target = _target(staging, name)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with archive.open(info) as source, target.open("xb") as sink:
                while chunk := source.read(_CHUNK):
                    size += len(chunk)
                    if size > expected["size"]:
                        break
                    digest.update(chunk)
                    sink.write(chunk)
        except (FileExistsError, NotADirectoryError, IsADirectoryError):
            raise BuildRefused([_refuse(
                "bundle.unsafe", RefusalClass.USER,
                f"the member {name!r} would be written where another member, or a folder, already is",
                "this is not a bundle `mime pack` wrote; do not unpack it: ask its sender to pack the folder again "
                "with `mime pack <folder>`")]) from None
        found = _verify_member(plain(name), digest.hexdigest(), size, expected)
        if found is not None:
            problems.append(found)
    return problems


def _notice_of(archive: zipfile.ZipFile, manifest: Manifest | ManifestFormat1) -> tuple[str, list[Refusal]]:
    """The ``NOTICE`` a format 2 bundle carries, held to its hash in the manifest."""
    if manifest["format"] != FORMAT:
        return "", []
    data = archive.read(NOTICE_MEMBER)
    expected = manifest["notice"]
    problem = _verify_member(NOTICE_MEMBER, sha256_of(data), len(data), expected)
    return data.decode("utf-8", errors="replace"), [problem] if problem is not None else []


def _referenced_problems(manifest: Manifest | ManifestFormat1, form_lock: Lock) -> list[Refusal]:
    """The manifest's references against the unpacked lock: each path a name the layout holds, carried by no member,
    and the asset the lock names for it holding the same bytes."""
    referenced = referenced_of(manifest)
    listed = form_lock.get("referenced_files", {})
    assets = {held["name"]: held for held in form_lock.get("assets", [])}
    action = ("this is not a bundle `mime pack` wrote: ask its sender to pack the folder again with `mime pack "
              "<folder>`")
    problems: list[Refusal] = []
    for path, held in sorted(referenced.items()):
        why = name_problem(path)
        if why is not None or path in manifest["files"]:
            problems.append(_refuse("bundle.unsafe", RefusalClass.USER,
                                    f"the reference {path!r} {why or 'is carried as a file too'}", action))
            continue
        asset = assets.get(held["name"])
        if listed.get(path) != held["name"] or asset is None or (asset["sha256"], asset["size"]) != (held["sha256"],
                                                                                                   held["size"]):
            problems.append(_refuse("bundle.manifest", RefusalClass.USER,
                                    f"the manifest's reference {plain(path)} is not the lock's", action))
    if set(listed) != set(referenced):
        problems.append(_refuse("bundle.manifest", RefusalClass.USER,
                                "the manifest's references are not the files the lock says travel as references",
                                action))
    return problems


def _carried_reference_problems(manifest: Manifest | ManifestFormat1) -> list[Refusal]:
    """What no bundle ``pack`` writes holds, judged from the manifest before anything is written (so the same on every
    filesystem): a carried member with a reference's bytes, and a carried member whose name is a reference's in
    another case."""
    referenced = referenced_of(manifest)
    action = ("this is not a bundle `mime pack` wrote; do not unpack it: ask its sender to pack the folder again with "
              "`mime pack <folder>`")
    carried = {held["sha256"]: name for name, held in manifest["files"].items()}
    problems = [_refuse("bundle.unsafe", RefusalClass.USER,
                        f"the bundle carries the bytes of the reference {path!r} as {carried[held['sha256']]!r}",
                        action)
                for path, held in sorted(referenced.items()) if held["sha256"] in carried]
    problems += [_refuse("bundle.unsafe", RefusalClass.USER,
                         f"{one!r} and {other!r}, one carried and one a reference, are one name on a case-insensitive "
                         f"filesystem", action)
                 for one, other in collisions([*manifest["files"], *referenced])
                 if (one in referenced) != (other in referenced)]
    return problems


def unpack(bundle: str | os.PathLike[str], into: str | os.PathLike[str]) -> Unpacked:
    """Check the bundle and write its folder to ``into``, which must not exist (or be an empty folder). Nothing is
    written outside ``into``: a member's name is checked against the saved form's layout (no hidden file, no name a
    platform treats specially, no two names one on a case-insensitive filesystem), links and special files are refused,
    and every file is hashed against the manifest while it is written to a hidden folder beside ``into``, which is
    renamed to ``into`` only when every hash matched, the folder reads as a saved experiment, and its files are exactly
    the ones the saved form names, those carried and those that travel as references together. Each file that travels
    as a reference is placed from the asset cache where it is cached (copied and hashed); the others are left for
    ``mime assets fetch <folder>``, and a load of the folder until then is refused as ``asset.missing`` with that
    command. Raises ``BuildRefused``."""
    refused = destination_problem(into)
    if refused is not None:
        raise refused
    destination = Path(os.path.abspath(into))
    staging = destination.with_name(f".{destination.name}.partial-{secrets.token_hex(8)}")
    try:
        with zipfile.ZipFile(bundle) as archive:
            later = unknown_format(archive)
            if later is not None:
                raise later
            refusals = _member_kind_problems(archive)
            if refusals:
                raise BuildRefused(refusals)
            manifest = read_manifest(archive)
            listed = {FOLDER_PREFIX + name for name in manifest["files"]}
            listed |= {NOTICE_MEMBER} if manifest["format"] == FORMAT else set()
            held = {info.filename for info in archive.infolist()} - {MANIFEST_MEMBER}
            if listed != held:
                raise BuildRefused([_refuse(
                    "bundle.changed", RefusalClass.USER,
                    f"the bundle's files are not the manifest's: {len(held - listed)} unlisted, "
                    f"{len(listed - held)} "
                    "missing", "the bundle was altered after it was packed: fetch it again from its source, `mime pull "
                               "<doi | url>`")])
            notice, problems = _notice_of(archive, manifest)
            unsafe = _carried_reference_problems(manifest)
            if unsafe:
                raise BuildRefused([*problems, *unsafe])
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            problems += _write_members(archive, manifest, staging)
            if problems:
                raise BuildRefused(problems)
        form = read_folder(staging, absent_referenced=True)
        carried, referenced = set(manifest["files"]), set(referenced_of(manifest))
        if carried | referenced != named_files(form):
            extra, missing = (carried | referenced) - named_files(form), named_files(form) - (carried | referenced)
            raise BuildRefused([_refuse(
                "bundle.unsafe", RefusalClass.USER,
                f"the bundle's files are not the ones the saved experiment names: {len(extra)} it does not name "
                f"({', '.join(repr(each) for each in sorted(extra)[:3])}), {len(missing)} it names and the bundle "
                f"lacks", "this is not a bundle `mime pack` wrote; do not unpack it: ask its sender to pack the folder "
                          "again with `mime pack <folder>`")])
        problems = _referenced_problems(manifest, form.lock)
        if problems:
            raise BuildRefused(problems)
        if manifest["assets"] != [ref.to_record() for ref in folder_assets(staging)]:
            raise BuildRefused([_refuse(
                "bundle.manifest", RefusalClass.USER, "the manifest's asset references are not the lock's",
                "this is not a bundle `mime pack` wrote: ask its sender to pack the folder again with `mime pack "
                "<folder>`")])
        placed = place_referenced(staging)
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    except zipfile.BadZipFile as error:
        raise _unreadable(error) from error
    except RecursionError:
        raise BuildRefused([_refuse("bundle.manifest", RefusalClass.USER,
                                     "a file of the bundle is nested too deeply to be read",
                                     "this is not a bundle `mime pack` wrote: ask its sender to pack the folder again "
                                     "with `mime pack <folder>`")]) from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return Unpacked(manifest, form.lock, Path(into), notice, placed.placed,
                    tuple(path for path, _, _ in placed.missing))


__all__ = ["BUNDLE_SUFFIX", "EXPANSION_LIMIT_BYTES", "FORMAT", "FORMAT_1", "INLINE_LIMIT_BYTES", "MANIFEST_MEMBER",
           "NOTICE_MEMBER", "Manifest", "ManifestFormat1", "ReferencedFile", "Unpacked", "bundle_hash", "bundle_name",
           "collisions", "destination_problem", "member_problem", "name_problem", "pack", "read_manifest",
           "referenced_members", "referenced_of", "shipped_asset_problems", "unknown_format", "unpack"]
