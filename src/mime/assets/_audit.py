"""The guard: no file whose bytes may not be redistributed leaves the machine inside an archive or a tree.

A wheel and an sdist are redistributions (a plugin's wheel is archived on Zenodo by sha256), and so is a repository's
tree at a tag (a GitHub release's source archive, the snapshot Zenodo's GitHub integration takes). :func:`audit` reads
a wheel, an sdist, a zip or tar archive, or a directory, hashes every file in it, and finds:

- **a referenced file**: any file whose sha256 is a record's whose tier is ``referenced``, from every record known
  here (the shared sources', each manifest, lock and record file inside the target, and the manifests given);
- **a derived file of one**: a file a record names with its derivation record, whose inputs make it referenced
  (``mime.assets.record_terms``: one referenced input is enough, whatever licence the record declares), and a mesh MIME
  baked from a referenced source into the local asset cache (its record names the output's sha256), both by hash;
- **a mesh with no record**: any file in a mesh or scene format whose sha256 no record names at all, the format read
  from its **bytes** (:func:`mesh_format`: binary and ASCII STL, USD's crate and text, OBJ, PLY, glTF and its binary,
  and the others listed there), its extension a second hint only. A mesh a record names under an open or a conditional
  licence may travel; so may a saved folder's own stage and layers, which its lock names.

**What it opens, by content, never by name**: a zip (a wheel, a bundle, an ``.npz``, a zip renamed ``.dat``) and a tar,
and a gzip, bzip2 or xz stream, one inside another to :data:`DEPTH_LIMIT` layers (a zip in a tar in a gzip in a wheel);
and in a text file, every base64 block that decodes to :data:`BASE64_MIN_BYTES` or more (a data URI, a mesh pasted
into JSON), which it reads as a file of its own. Nothing is skipped by name, inside an archive or in a tree: the one
exception is a checkout's own ``.git`` directory at the top of a tree, which the report says it left out; a link in a
tree is not followed (an archive of the tree holds the link, not the bytes), and the report counts them.

**Its limits refuse, never skip**: a file it must look inside that is over :data:`MEMBER_LIMIT_BYTES`, more than
:data:`EXPANDED_LIMIT_BYTES` expanded from one target in all, more than :data:`COUNT_LIMIT` files, a layer deeper than
:data:`DEPTH_LIMIT`, and an archive or a stream it cannot read to its end (corrupt, encrypted, a compression it does not
know) are each refused as ``asset.audit``, so that a file the guard did not read is never one it passed.

**What a hash guard cannot do**: a file converted (an STL written out as USD by any tool, a mesh decimated or
re-encoded) has other bytes, and only the mesh check sees it, as a mesh with no record; a target that ships its own
record calling a mesh open is believed for a mesh no other record knows. Nothing found is deleted or changed; ``mime
assets audit`` exits 1 when anything is found.
"""

from __future__ import annotations

import base64
import binascii
import bz2
import hashlib
import io
import json
import lzma
import os
import re
import struct
import tarfile
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal, Protocol, cast

from mime._lock_json import lock_data
from mime.assets._derivation import DerivationRecord, by_sha256, read_derivation, record_terms
from mime.assets._manifest import Manifest, is_manifest_data, read_manifest, read_manifest_data
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

#: The file name extensions of mesh, scene and CAD formats: a second hint beside a file's bytes, for the formats whose
#: bytes say nothing plain (an FBX in ASCII, a 3MF, an IGES).
MESH_EXTENSIONS = frozenset({".stl", ".obj", ".ply", ".dae", ".gltf", ".glb", ".fbx", ".3mf", ".off", ".step", ".stp",
                             ".iges", ".igs", ".usd", ".usda", ".usdc", ".usdz", ".vtk", ".vtp", ".vtu", ".msh",
                             ".wrl", ".x3d", ".blend", ".abc"})
#: How many layers of archives, compressed streams and base64 blocks, one inside another, the guard opens: an sdist's
#: gzip and its tar are two. A deeper one is refused.
DEPTH_LIMIT = 6
#: The most bytes one file the guard must look inside may hold (an archive, a compressed stream, a text it searches for
#: base64), which it holds in memory to read. A larger one is refused; a larger file of any other kind is hashed as it
#: is read.
MEMBER_LIMIT_BYTES = 1 << 30
#: The most bytes the guard expands from one target in all (archives' members, compressed streams, base64 blocks): a
#: zip bomb is refused, never read to its end.
EXPANDED_LIMIT_BYTES = 16 << 30
#: The most files the guard reads in one target, those inside archives and base64 blocks among them.
COUNT_LIMIT = 1_000_000
#: A base64 block in a text file is decoded and read as a file of its own when it decodes to at least this many bytes.
BASE64_MIN_BYTES = 512
#: The most a file inside the target may weigh for the guard to read the records it holds.
_RECORDS_LIMIT_BYTES = 50_000_000
_HEAD = 1 << 20
_CHUNK = 1 << 20
#: A run of base64's characters (either alphabet), its lines broken or not, long enough to decode to the floor above.
_BASE64_RUN = re.compile(rb"[A-Za-z0-9+/_-](?:[A-Za-z0-9+/_=-]|\r?\n){%d,}" % (BASE64_MIN_BYTES * 4 // 3 - 1))
_OBJ_VERTEX = re.compile(rb"^v[ \t]+[-+0-9.eE]+[ \t]+[-+0-9.eE]+[ \t]+[-+0-9.eE]+", re.MULTILINE)
_OBJ_FACE = re.compile(rb"^f[ \t]+\d", re.MULTILINE)
#: What the report says of the one directory a tree's audit leaves out.
GIT_LEFT_OUT = ("the checkout's own .git directory at the top of the tree (its history, which no archive of the tree "
                "holds)")
_COMPRESSED_SUFFIXES = {"gzip": (".tgz", ".gz"), "bzip2": (".tbz2", ".bz2"), "xz": (".txz", ".xz")}

type Kind = Literal["referenced", "derived", "unrecorded mesh"]
type Container = Literal["zip", "tar", "gzip", "bzip2", "xz"]


@dataclass(frozen=True)
class Finding:
    """A file the guard found: its path in the target, what it is, and what names it."""

    path: str
    kind: Kind
    what: str

    def line(self) -> str:
        return f"{self.path}: {self.kind}, {self.what}"


@dataclass(frozen=True)
class AuditReport:
    """What :func:`audit` read and found: the target, how many files it read (archives' members, compressed streams'
    contents and base64 blocks among them), how many records and derived files it held each against, every finding,
    and what it left out (a checkout's own ``.git``, links it did not follow)."""

    target: str
    files: int
    records: int
    derived: int
    findings: tuple[Finding, ...] = field(default=())
    left_out: tuple[str, ...] = field(default=())

    def text(self) -> str:
        head = (f"{self.target}: {self.files} files read against {self.records} records and {self.derived} derived "
                f"files")
        lines = [f"  left out: {each}" for each in self.left_out]
        if not self.findings:
            return "\n".join([f"{head}; no file that may not be redistributed, and no mesh without a record", *lines])
        return "\n".join([f"{head}; {len(self.findings)} found:", *(f"  {each.line()}" for each in self.findings),
                          *lines])


def _gltf_json(data: bytes) -> bool:
    if data.lstrip()[:1] != b"{" or b'"asset"' not in data[:_HEAD]:
        return False
    try:
        held: object = json.loads(data)
    except (ValueError, RecursionError):
        return False
    if not isinstance(held, dict):
        return False
    record = cast(dict[str, object], held)
    asset = record.get("asset")
    return isinstance(asset, dict) and "version" in asset and \
        any(key in record for key in ("meshes", "accessors", "buffers", "nodes"))


_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"#usda ", "USD text"), (b"# vtk datafile version", "VTK"), (b"$meshformat", "Gmsh"), (b"#vrml", "VRML"),
    (b"iso-10303-21;", "STEP"), (b"off\n", "OFF"), (b"off\r\n", "OFF"), (b"coff\n", "OFF"))
_MARKERS: tuple[tuple[bytes, str], ...] = ((b"<vtkfile", "VTK XML"), (b"<x3d", "X3D"), (b"<collada", "COLLADA"))
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PXR-USDC", "USD crate"), (b"Kaydara FBX Binary", "FBX"), (b"BLENDER", "Blender"), (b"Ogawa", "Alembic"))


#: The bytes after a binary STL's last facet that a reader ignores, and the guard with it: a copy with a few bytes
#: appended is still the mesh.
_STL_TRAILING_BYTES = 200


def _binary_stl(head: bytes, size: int) -> bool:
    """Whether a file of ``size`` bytes beginning with ``head`` is a binary STL: its facet count, at byte 80, gives its
    size, give or take a few bytes appended."""
    if len(head) < 84:
        return False
    facets = struct.unpack_from("<I", head, 80)[0]
    return facets > 0 and 0 <= size - (84 + 50 * facets) <= _STL_TRAILING_BYTES or (facets == 0 and size == 84)


def mesh_format(name: str, data: bytes) -> str | None:
    """The mesh or scene format a file is in, read from its bytes first: USD's crate and text, glTF (JSON) and its
    binary, PLY, binary and ASCII STL, OBJ, OFF, VTK (legacy and XML), Gmsh, VRML, X3D, COLLADA, FBX's binary, STEP,
    Blender and Alembic; then by its extension (:data:`MESH_EXTENSIONS`); ``None`` for none."""
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    for magic, form in _MAGIC:
        if data.startswith(magic):
            return form
    if data.startswith(b"glTF") and len(data) >= 12 and struct.unpack_from("<I", data, 4)[0] == 2:
        return "glTF binary"
    if (data.startswith(b"ply\n") or data.startswith(b"ply\r\n")) and b"format " in data[:256]:
        return "PLY"
    if _binary_stl(data, len(data)):
        return "binary STL"
    if head.startswith(b"solid") and b"facet normal" in head:
        return "ASCII STL"
    for signature, form in _SIGNATURES:
        if head.startswith(signature):
            return form
    for marker, form in _MARKERS:
        if head.startswith(b"<") and marker in head[:1024]:
            return form
    window = data[:65536]
    if b"\x00" not in window and len(_OBJ_VERTEX.findall(window)) >= 3 and _OBJ_FACE.search(window):
        return "OBJ"
    if _gltf_json(data):
        return "glTF"
    return _extension_format(name)


def _extension_format(name: str) -> str | None:
    """A format named by a file's extension alone (a ``.usdz``, which is a zip)."""
    suffix = Path(name).suffix.lower()
    return suffix.removeprefix(".").upper() if suffix in MESH_EXTENSIONS else None


def container_of(data: bytes) -> Container | None:
    """What ``data`` is a container of, read from its bytes: a zip (a wheel, a bundle, an ``.npz``; by its first
    record, or by its directory at its end where something stands before it), a tar, or a gzip, bzip2 or xz stream;
    ``None`` for none. A file that begins as a zip and does not read as one is refused where it is opened, never passed
    as plain bytes."""
    if data.startswith(b"\x1f\x8b\x08"):
        return "gzip"
    if data.startswith(b"BZh") and data[3:4].isdigit() and data[4:10] in (b"1AY&SY", b"\x17rE8P\x90"):
        return "bzip2"
    if data.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if len(data) >= 262 and data[257:262] == b"ustar":
        return "tar"
    if data.startswith((b"PK\x03\x04", b"PK\x05\x06")) or zipfile.is_zipfile(io.BytesIO(data)):
        return "zip"
    return None


def _refused(message: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.audit", RefusalClass.USER, message,
        "the guard passes no file it could not read: unpack the file and audit what it holds (mime assets audit "
        "<directory>), or leave it out of what is shipped")])


# --- what bytes are held against -----------------------------------------------------------------------------------


@dataclass
class KnownBytes:
    """What the guard, ``pack`` and ``push`` hold a file against: each record's sha256 (any tier), the referenced ones,
    and the derived files that are referenced by their inputs (by a record's derivation, or baked into the cache from a
    referenced mesh), each with what says so."""

    any: set[str] = field(default_factory=set[str])
    referenced: dict[str, str] = field(default_factory=dict[str, str])
    derived: dict[str, str] = field(default_factory=dict[str, str])

    def found(self, sha256: str) -> tuple[Kind, str] | None:
        """Whether bytes of ``sha256`` may not travel, and what says so: ``referenced`` or ``derived``; ``None``."""
        if sha256 in self.referenced:
            return "referenced", self.referenced[sha256]
        if sha256 in self.derived:
            return "derived", self.derived[sha256]
        return None


def _shared_records() -> list[tuple[AssetRef, str]]:
    from mime.assets._sources import shared

    return [(each.ref, f"from {each.source}") for sourced in shared().records.values() for each in sourced]


def _baked(known: KnownBytes) -> None:
    """The baked meshes in the local asset cache whose records say their tier is referenced."""
    from mime.assets._cache import cache_dir

    folder = cache_dir() / "visuals"
    if not folder.is_dir():
        return
    for path in folder.glob("*.json"):
        try:
            held: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(held, dict) or not str(cast(dict[str, object], held).get("format", "")).startswith(
                "mime.baked_mesh/"):
            continue
        record = cast(dict[str, object], held)
        output = record.get("output")
        digest = cast(dict[str, object], output).get("sha256") if isinstance(output, dict) else None
        if isinstance(digest, str):
            known.any.add(digest)
            # A bake an earlier MIME wrote says "no-redistribution", or nothing: only a stated open or conditional
            # tier lets a baked mesh travel.
            if record.get("tier") not in ("open", "conditional"):
                known.derived.setdefault(digest, f"baked from {record.get('set')}'s {record.get('file')} (tier "
                                                 f"referenced)")


def known_bytes(records: Iterable[tuple[AssetRef, str]] = (), *,
                derivations: Mapping[str, DerivationRecord] | None = None, cache: bool = True,
                also: Iterable[str] = ()) -> KnownBytes:
    """Every record in reach, judged as every exit judges it (``mime.assets.record_terms``): the shared sources' and
    ``records`` (each with where it was read), with the derivation records at hand (``derivations``, by the sha256 of
    their bytes, beside those MIME ships and those in the asset cache); the meshes baked into the local cache from a
    referenced source (``cache``); and ``also``, the sha256s of files recorded otherwise (a saved folder's own stage).
    Of two records of the same bytes, any that makes them referenced does."""
    held = [*_shared_records(), *records]
    by_hash = by_sha256(ref for ref, _ in held)
    known = KnownBytes(set(also))
    for ref, where in held:
        known.any.add(ref.sha256)
        verdict = record_terms(ref, by_hash, derivations=derivations)
        if verdict.tier != "referenced":
            continue
        if ref.verdict.tier == "referenced":
            known.referenced.setdefault(ref.sha256, f"the record {ref.name} ({ref.licence}, tier referenced; {where})")
        else:
            known.derived.setdefault(ref.sha256, f"the record {ref.name} ({ref.licence}; {where}), referenced by "
                                                 f"{verdict.by}")
    if cache:
        _baked(known)
    return known


# --- reading a target -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Seen:
    """A file the guard read: its path in the target, its sha256, and the mesh format its bytes are in."""

    path: str
    sha256: str
    form: str | None


@dataclass
class _Reading:
    """One target's read: every file seen, the records, derivation records and saved stages it holds, what it left
    out, and what its limits have counted."""

    seen: list[_Seen] = field(default_factory=list[_Seen])
    records: list[tuple[AssetRef, str]] = field(default_factory=list[tuple[AssetRef, str]])
    derivations: dict[str, DerivationRecord] = field(default_factory=dict[str, DerivationRecord])
    stages: set[str] = field(default_factory=set[str])
    left_out: list[str] = field(default_factory=list[str])
    links: int = 0
    expanded: int = 0

    def add(self, path: str, sha256: str, form: str | None) -> None:
        if len(self.seen) >= COUNT_LIMIT:
            raise _refused(f"{path}: the target holds more than {COUNT_LIMIT} files, more than the guard reads in one "
                           f"target")
        self.seen.append(_Seen(path, sha256, form))

    def expand(self, path: str, amount: int) -> None:
        self.expanded += amount
        if self.expanded > EXPANDED_LIMIT_BYTES:
            raise _refused(f"{path}: the target expands past {EXPANDED_LIMIT_BYTES} bytes in all, more than the guard "
                           f"reads")


def _inside(name: str, data: bytes) -> list[Manifest | AssetRef]:
    """The manifests and records a file of the target holds (a manifest, a lock, a file of records under
    ``assets``), which the guard learns before it judges."""
    if len(data) > _RECORDS_LIMIT_BYTES or data.lstrip()[:1] != b"{":
        return []
    try:
        held: object = lock_data(data)
    except (ValueError, RecursionError):
        return []
    if is_manifest_data(held):
        try:
            return [read_manifest_data(held, name)]
        except BuildRefused:
            return []
    found: list[Manifest | AssetRef] = []
    if isinstance(held, dict):
        record = cast(dict[str, object], held)
        listed = record.get("assets")
        for each in cast(list[object], listed) if isinstance(listed, list) else []:
            if isinstance(each, dict):
                try:
                    found.append(AssetRef.from_record(cast(dict[str, object], each)))
                except BuildRefused:
                    continue
        visuals = record.get("visuals")
        for value in list(cast(dict[str, object], visuals).values()) if isinstance(visuals, dict) else []:
            try:
                found.append(read_manifest_data(value, name))
            except BuildRefused:
                continue
    return found


def _stage_files(data: bytes) -> set[str]:
    """The sha256s a saved folder's lock names for its own stage and layers: the author's files, whose terms its
    document states, recorded there."""
    if len(data) > _RECORDS_LIMIT_BYTES or b'"stage"' not in data or data.lstrip()[:1] != b"{":
        return set()
    try:
        held: object = lock_data(data)
    except (ValueError, RecursionError):
        return set()
    stage = cast(dict[str, object], held).get("stage") if isinstance(held, dict) else None
    if not isinstance(stage, dict):
        return set()
    record = cast(dict[str, object], stage)
    found = {record["sha256"]} if isinstance(record.get("sha256"), str) else set[object]()
    layers = record.get("layers")
    found |= set(cast(dict[str, object], layers).values()) if isinstance(layers, dict) else set()
    return {each for each in found if isinstance(each, str)}


def _learn(reading: _Reading, path: str, data: bytes) -> None:
    for held in _inside(path, data):
        for ref in held.assets if isinstance(held, Manifest) else [held]:
            reading.records.append((ref, f"in {path}"))
    reading.stages |= _stage_files(data)
    if data.lstrip()[:1] == b"{" and len(data) <= _RECORDS_LIMIT_BYTES:
        derivation = read_derivation(data)
        if derivation is not None:
            reading.derivations[hashlib.sha256(data).hexdigest()] = derivation


class _Stream(Protocol):
    """A gzip, bzip2 or xz decompressor, as the guard reads one."""

    @property
    def eof(self) -> bool: ...

    @property
    def unused_data(self) -> bytes: ...

    def decompress(self, data: bytes, max_length: int, /) -> bytes: ...


def _decompressor(kind: Container) -> _Stream:
    if kind == "gzip":
        return zlib.decompressobj(wbits=31)
    if kind == "bzip2":
        return bz2.BZ2Decompressor()
    return lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)


def _expanded(path: str, kind: Container, data: bytes, reading: _Reading) -> bytes:
    """What a gzip, bzip2 or xz stream holds (several joined, as each format allows), held to the limits: a stream
    that expands past :data:`MEMBER_LIMIT_BYTES`, or that ends before its end, is refused."""
    out = bytearray()
    rest = data
    try:
        while rest:
            stream = _decompressor(kind)
            pending = rest
            while not stream.eof:
                chunk = stream.decompress(pending, MEMBER_LIMIT_BYTES - len(out) + 1)
                pending = cast(bytes, getattr(stream, "unconsumed_tail", b""))
                out += chunk
                reading.expand(path, len(chunk))
                if len(out) > MEMBER_LIMIT_BYTES:
                    raise _refused(f"{path}: its {kind} stream expands past {MEMBER_LIMIT_BYTES} bytes, more than the "
                                   f"guard holds to look inside")
                if not chunk and not pending and cast(bool, getattr(stream, "needs_input", True)):
                    raise _refused(f"{path}: its {kind} stream ends before its end, so the guard cannot read all of it")
            rest = stream.unused_data
            if rest and container_of(rest) != kind:
                break  # bytes after the last stream, which each format's reader ignores
    except (OSError, EOFError, ValueError, zlib.error, lzma.LZMAError) as error:
        raise _refused(f"{path}: its {kind} stream cannot be read ({type(error).__name__}: {error})") from None
    return bytes(out)


def _unwrapped(path: str, kind: Container) -> str:
    """The name of what a compressed stream holds: its own name less the compression's extension (``.tgz`` read as
    ``.tar``), or the stream's kind where it has no such extension."""
    last = path.rpartition("!")[2].rpartition("/")[2]
    for suffix in _COMPRESSED_SUFFIXES[kind]:
        if last.lower().endswith(suffix) and len(last) > len(suffix):
            return last[: -len(suffix)] + (".tar" if suffix.startswith(".t") else "")
    return f"({kind} stream)"


def _large(reading: _Reading, path: str, stream: IO[bytes], size: int) -> None:
    """A file over :data:`MEMBER_LIMIT_BYTES`: hashed as it is read, its format read from its head; refused where it
    is an archive, a compressed stream or a text, which the guard would have to hold to look inside."""
    head = stream.read(_HEAD)
    if container_of(head) is not None or b"\x00" not in head:
        raise _refused(f"{path}: {size} bytes, more than the {MEMBER_LIMIT_BYTES} the guard holds to look inside an "
                       f"archive, a compressed stream or a text")
    digest = hashlib.sha256(head)
    while chunk := stream.read(_CHUNK):
        digest.update(chunk)
    reading.add(path, digest.hexdigest(), _head_format(path, head, size))


def _head_format(path: str, head: bytes, size: int) -> str | None:
    """The mesh format of a file read by its head alone: a binary STL by the whole file's size."""
    if _binary_stl(head, size):
        return "binary STL"
    form = mesh_format(path, head)
    return _extension_format(path) if form == "binary STL" else form  # the head's own length matched, not the file's


def _streamed_digest(where: Path) -> str:
    digest = hashlib.sha256()
    with where.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _member(reading: _Reading, path: str, stream: IO[bytes], size: int, depth: int) -> None:
    """A member of an archive, ``size`` bytes, read from ``stream``."""
    reading.expand(path, size)
    if size > MEMBER_LIMIT_BYTES:
        _large(reading, path, stream, size)
    else:
        _file(reading, path, stream.read(), depth, f"{path}!")


def _zip(reading: _Reading, path: str, source: IO[bytes] | Path, depth: int, prefix: str) -> None:
    try:
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    with archive.open(info) as member:
                        _member(reading, prefix + info.filename, member, info.file_size, depth)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError,
            OSError, ValueError) as error:
        raise _refused(f"{path}: a zip archive the guard cannot read ({type(error).__name__}: {error})") from None


def _tar(reading: _Reading, path: str, source: IO[bytes] | Path, depth: int, prefix: str) -> None:
    try:
        with tarfile.open(source, mode="r:*") if isinstance(source, Path) else \
                tarfile.open(fileobj=source, mode="r:") as archive:
            for info in archive:
                if info.isfile():
                    stream = archive.extractfile(info)
                    if stream is not None:
                        with stream:
                            _member(reading, prefix + info.name, stream, info.size, depth)
                elif info.issym() or info.islnk():
                    reading.links += 1
    except (tarfile.TarError, EOFError, OSError, zlib.error, lzma.LZMAError, ValueError) as error:
        raise _refused(f"{path}: a tar archive the guard cannot read ({type(error).__name__}: {error})") from None


def _decoded(run: bytes) -> bytes | None:
    held = re.sub(rb"\s", b"", run).translate(bytes.maketrans(b"-_", b"+/")).rstrip(b"=")
    held = held[: len(held) - len(held) % 4] if len(held) % 4 == 1 else held
    try:
        return base64.b64decode(held + b"=" * (-len(held) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None


def _base64_blocks(data: bytes) -> Iterator[tuple[int, bytes]]:
    """Each base64 block of a text that decodes to :data:`BASE64_MIN_BYTES` or more, with the byte it starts at. A
    block whose lines are broken is read whole, and from its second line too where its first is of another length (a
    word before it, ``base64`` on the line above), since the guard cannot tell where a block begins."""
    for match in _BASE64_RUN.finditer(data):
        run = match.group()
        lines = run.splitlines()
        starts = [(match.start(), run)]
        if len(lines) > 2 and len(lines[0]) != len(lines[1]):
            skip = run.index(b"\n") + 1
            starts.append((match.start() + skip, run[skip:]))
        for start, held in starts:
            decoded = _decoded(held)
            if decoded is not None and len(decoded) >= BASE64_MIN_BYTES:
                yield start, decoded


def _deeper(path: str, what: str, depth: int) -> None:
    if depth + 1 > DEPTH_LIMIT:
        raise _refused(f"{path}: {what} {depth + 1} layers deep, deeper than the {DEPTH_LIMIT} layers the guard opens")


def _file(reading: _Reading, path: str, data: bytes, depth: int, prefix: str) -> None:
    """One file in memory: hashed, its format read, its records learnt, and what it holds read in turn, each file
    inside named ``prefix`` and its own name (``a.whl!pkg/m.stl``)."""
    kind = container_of(data)
    # An archive or a stream is not itself a mesh, whatever its first bytes hold: what it holds is read in turn.
    reading.add(path, hashlib.sha256(data).hexdigest(), mesh_format(path, data) if kind is None else
                _extension_format(path) if kind == "zip" else None)
    _learn(reading, path, data)
    if kind is not None:
        _deeper(path, f"a {kind}", depth)
        if kind == "zip":
            _zip(reading, path, io.BytesIO(data), depth + 1, prefix)
        elif kind == "tar":
            _tar(reading, path, io.BytesIO(data), depth + 1, prefix)
        else:
            inner = _expanded(path, kind, data, reading)
            _file(reading, prefix + _unwrapped(path, kind), inner, depth + 1, prefix)
        return
    if b"\x00" not in data:
        for start, decoded in _base64_blocks(data):
            _deeper(path, "a base64 block", depth)
            reading.expand(path, len(decoded))
            block = f"{path}#base64@{start}"
            _file(reading, block, decoded, depth + 1, f"{block}!")


def _disk_file(reading: _Reading, path: str, where: Path, prefix: str) -> None:
    """A file on disk: read whole where the guard may hold it; otherwise a zip or a tar is read member by member from
    the file, and anything else hashed as it is read. A file that cannot be read is refused."""
    try:
        _read_disk_file(reading, path, where, prefix)
    except OSError as error:
        raise _refused(f"{path}: cannot be read ({type(error).__name__}: {error})") from None


def _read_disk_file(reading: _Reading, path: str, where: Path, prefix: str) -> None:
    size = where.stat().st_size
    if size <= MEMBER_LIMIT_BYTES:
        _file(reading, path, where.read_bytes(), 0, prefix)
        return
    with where.open("rb") as stream:
        head = stream.read(_HEAD)
    kind = container_of(head) or ("zip" if zipfile.is_zipfile(where) else None)
    if kind == "zip" or (kind is not None and tarfile.is_tarfile(where)):
        reading.add(path, _streamed_digest(where), _extension_format(path))
        if kind == "zip":
            _zip(reading, path, where, 1, prefix)
        else:
            _tar(reading, path, where, 2 if kind != "tar" else 1, prefix)
    else:
        with where.open("rb") as stream:
            _large(reading, path, stream, size)


def _unreadable_directory(error: OSError) -> None:
    raise _refused(f"{error.filename}: a directory that cannot be read ({type(error).__name__}: {error.strerror})")


def _tree(reading: _Reading, root: Path) -> None:
    checkout = (root / ".git").exists()
    for directory, folders, files in os.walk(root, onerror=_unreadable_directory, followlinks=False):
        here = Path(directory)
        if here == root and checkout and ".git" in folders:
            folders.remove(".git")
            reading.left_out.append(GIT_LEFT_OUT)
        folders.sort()
        reading.links += sum((here / name).is_symlink() for name in folders)
        for name in sorted(files):
            path = here / name
            if path.is_symlink():
                reading.links += 1
            elif path.is_file():
                relative = path.relative_to(root).as_posix()
                _disk_file(reading, relative, path, f"{relative}!")


def audit(target: str | os.PathLike[str], *, manifests: Sequence[str | os.PathLike[str]] = (),
          cache: bool = True) -> AuditReport:
    """Read ``target`` (a wheel, an sdist, a zip or tar archive, or a directory), hash every file in it, archives'
    members, compressed streams' contents and text files' base64 blocks among them, and find every file whose bytes
    are a referenced record's, a derived file's of one (by its record's derivation, or a mesh baked from one in the
    local cache with ``cache``), and every mesh with no record. Every record known here counts: the shared sources',
    each manifest, lock and record file inside the target, and ``manifests``. Refused (``asset.audit``) where the target
    is not there, or holds a file the guard cannot read to its end within its limits. A file inside the target is
    named by its path there, a file inside it after a ``!`` (``bundle.zip!x/a.stl``)."""
    path = Path(target)
    if not path.exists():
        raise BuildRefused([Refusal(
            "asset.audit", RefusalClass.USER, f"{path} is not there to audit",
            "name a wheel, an sdist, an archive or a directory: mime assets audit dist/*.whl")])
    given: list[tuple[AssetRef, str]] = []
    for each in manifests:
        given += [(ref, f"in {each}") for ref in read_manifest(Path(each)).assets]
    reading = _Reading()
    if path.is_dir():
        _tree(reading, path)
    else:
        _disk_file(reading, path.name, path, "")
    known = known_bytes([*given, *reading.records], derivations=reading.derivations, cache=cache, also=reading.stages)
    findings: list[Finding] = []
    for seen in reading.seen:
        held = known.found(seen.sha256)
        if held is not None:
            findings.append(Finding(seen.path, held[0], held[1]))
        elif seen.sha256 not in known.any and seen.form is not None:
            findings.append(Finding(seen.path, "unrecorded mesh", f"a {seen.form} file no record names"))
    links = [f"{reading.links} links, not followed (an archive of a tree holds a link, not the bytes it points to)"]
    return AuditReport(str(path), len(reading.seen), len(known.any), len(known.derived), tuple(findings),
                       (*reading.left_out, *(links if reading.links else [])))


__all__ = ["BASE64_MIN_BYTES", "COUNT_LIMIT", "DEPTH_LIMIT", "EXPANDED_LIMIT_BYTES", "GIT_LEFT_OUT",
           "MEMBER_LIMIT_BYTES", "MESH_EXTENSIONS", "AuditReport", "Finding", "KnownBytes", "audit", "container_of",
           "known_bytes", "mesh_format"]
