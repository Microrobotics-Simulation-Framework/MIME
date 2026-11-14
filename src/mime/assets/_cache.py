"""The cache: one directory, one file per sha256, and beside it a record of the full verification made of it.

A build must not re-hash a file of about nine hundred megabytes, so :func:`locate` makes a cheap check: the file's size
is the reference's, and its size and modification time are the ones recorded at the full verification. That catches a
truncated, replaced or edited file; it does not catch a flipped byte that leaves both unchanged, which only
:func:`check` (a full re-hash) finds, and which it then marks, so that :func:`locate` refuses the file until a fetch
repairs it. A caller that holds the repository's checksum of the file (a bundle's ``md5:...``) can pass it: the check
then also reads the file once and compares its digest, which does catch that flipped byte (and costs a local read of the
whole file, no download). With no checksum the check hashes nothing and guards a file's size and its modification time,
not its bytes.

The record also says who wrote the file. A file MIME downloaded may be replaced by a new download; a file a person
placed in the directory is never deleted or overwritten, whatever its bytes.

A record with no location to fetch it from (its own, or this machine's locations file's) is had by request alone: a
build that needs it and does not find it is refused as ``asset.by-request`` (:func:`by_request`), which says where to
ask for it and where to put it, never as a file to download.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mime._terminal import plain
from mime.assets._commands import fetch_command
from mime.assets._exclusive import LOCK_SUFFIX, exclusive
from mime.assets._reference import AssetRef, BundleContainer
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.terms import NO_ASSERTION

ASSETS_DIR_VARIABLE = "MIME_ASSETS_DIR"
OFFLINE_VARIABLE = "MIME_OFFLINE"
RECORD_SUFFIX = ".verified"
CHECKSUM_SUFFIX = ".checksum"
INCOMING_INFIX = ".incoming-"
DOWNLOADED = "downloaded by MIME"
PLACED = "placed by a person"
_CHUNK = 8 * 1024 * 1024
_ON = frozenset({"1", "true", "yes", "on"})
_OFF = frozenset({"", "0", "false", "no", "off"})

type Cached = Literal["ok", "not cached", "corrupt"]


def cache_dir() -> Path:
    """The one directory: ``$MIME_ASSETS_DIR`` when set, which replaces the default, else ``assets`` under the
    platform's cache directory for MIME."""
    chosen = os.environ.get(ASSETS_DIR_VARIABLE)
    if chosen:
        return Path(chosen)
    import pooch

    return Path(str(pooch.os_cache("mime"))) / "assets"


def where_text() -> str:
    """How a message names the directory: by the variable when it chose it, else as the cache directory."""
    return f"`{ASSETS_DIR_VARIABLE}`" if os.environ.get(ASSETS_DIR_VARIABLE) else "the cache directory"


def offline() -> bool:
    """Whether ``MIME_OFFLINE`` is set: ``1``, ``true``, ``yes`` or ``on`` in any case. Unset, empty, ``0``, ``false``,
    ``no`` and ``off`` mean online; any other value is refused, never read as online, since a switch that closes the
    network must not fail open."""
    raw = os.environ.get(OFFLINE_VARIABLE, "")
    value = raw.strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise BuildRefused([Refusal(
        "asset.offline-setting", RefusalClass.USER,
        f"{OFFLINE_VARIABLE} is {raw!r}, which MIME does not read as on or off",
        "set MIME_OFFLINE=1 (or true, yes, on) to download nothing, or MIME_OFFLINE=0 (or false, no, off, or unset it) "
        "to allow downloads")])


def file_path(ref: AssetRef) -> Path:
    return cache_dir() / ref.sha256


def record_path(ref: AssetRef) -> Path:
    return cache_dir() / (ref.sha256 + RECORD_SUFFIX)


def lock_path(ref: AssetRef) -> Path:
    return cache_dir() / (ref.sha256 + LOCK_SUFFIX)


_CHECKSUM = re.compile(r"[a-z0-9]+:[0-9a-f]+")


def checksum_path(checksum: str) -> Path | None:
    """Where the cache keeps what a repository's ``checksum`` (``md5:...``) of a bundle was found to be, or ``None`` for
    a text that is no checksum's shape."""
    return cache_dir() / (checksum.replace(":", "-") + CHECKSUM_SUFFIX) if _CHECKSUM.fullmatch(checksum) else None


def remember_checksum(checksum: str, ref: AssetRef) -> None:
    """Record that the file a repository's ``checksum`` names is ``ref``'s (its sha256 and size), after a download held to
    that checksum was stored under the sha256, so that the next call for the same checksum finds it in the cache. The
    record is written to a temporary name of its own and renamed into place."""
    path = checksum_path(checksum)
    if path is None:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=cache_dir())
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"checksum": checksum, "sha256": ref.sha256, "size": ref.size}, indent=2) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def remembered_checksum(checksum: str, name: str, location: str) -> AssetRef | None:
    """The reference, named ``name`` and located at ``location``, to the file a repository's ``checksum`` was recorded to
    name (:func:`remember_checksum`), or ``None`` where none is recorded or the record does not read."""
    path = checksum_path(checksum)
    if path is None:
        return None
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
        if held["checksum"] != checksum or not isinstance(held["sha256"], str) or isinstance(held["size"], bool) \
                or not isinstance(held["size"], int):
            return None
        return BundleContainer(name, held["sha256"], held["size"], NO_ASSERTION, (location,))
    except (OSError, ValueError, KeyError, TypeError, BuildRefused):
        return None


def sha256_of_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class _Seen:
    """The record of a full check: the file as it was then, who wrote it, and whether the check failed."""

    size: int
    mtime_ns: int
    origin: str
    failed: bool


def write_record(ref: AssetRef, origin: str, *, failed: bool = False) -> None:
    """Record the outcome of a full hash of the cached file: ``failed`` when its bytes are not ``ref.sha256``. The
    caller holds the asset's lock and has hashed the file; the record is written to a temporary name of its own and
    renamed into place."""
    seen = file_path(ref).stat()
    text = json.dumps({"sha256": ref.sha256, "size": seen.st_size, "mtime_ns": seen.st_mtime_ns, "origin": origin,
                       "verified": "full sha256 of the file's bytes", "result": "failed" if failed else "ok"}, indent=2)
    descriptor, temporary = tempfile.mkstemp(prefix=record_path(ref).name + ".", suffix=".partial", dir=cache_dir())
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
        os.replace(temporary, record_path(ref))
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _recorded(ref: AssetRef) -> _Seen | None:
    try:
        held = json.loads(record_path(ref).read_text(encoding="utf-8"))
        if held["sha256"] != ref.sha256:
            return None
        return _Seen(int(held["size"]), int(held["mtime_ns"]), str(held["origin"]), held["result"] != "ok")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def mime_wrote(ref: AssetRef) -> bool:
    """Whether the cached file is one MIME downloaded (its record says so); a file with no record, or a record that
    says a person placed it, is not MIME's to replace."""
    recorded = _recorded(ref)
    return recorded is not None and recorded.origin == DOWNLOADED


def size_text(size: int) -> str:
    """A size as the messages and the listing print it: bytes, kB or MB (decimal)."""
    if size >= 1_000_000:
        return f"{size / 1e6:.0f} MB"
    return f"{size / 1e3:.0f} kB" if size >= 1_000 else f"{size} bytes"


def digest_problem(path: Path, checksum: str) -> str | None:
    """Why the file at ``path`` is not the one a repository's ``checksum`` (``<algorithm>:<digits>``) names, or ``None``
    when its digest is that. Reads the whole file, once. A checksum of an algorithm this machine does not have, or of
    no checksum's shape, cannot be compared and is a problem too, never a pass. A file that is gone by the time it is
    opened is a problem (a download replaces it); one that cannot be read is refused, as ``asset.missing``, with its
    permissions as the route, since neither the comparison nor a download could go on."""
    algorithm, _, digits = checksum.partition(":")
    try:
        with path.open("rb") as stream:
            computed = hashlib.file_digest(stream, lambda: hashlib.new(algorithm.lower(), usedforsecurity=False))
            found = computed.hexdigest()
    except (ValueError, TypeError):
        return f"the checksum {algorithm!r} is not one this machine computes"
    except FileNotFoundError:
        return "isn't downloaded"
    except OSError as error:
        raise BuildRefused([Refusal(
            "asset.missing", RefusalClass.ENVIRONMENT,
            f"the file {path.name} in {where_text()} cannot be read ({error.strerror or type(error).__name__})",
            f"give your user read permission on {path}, or delete it and its {RECORD_SUFFIX!r} record and download it "
            f"again (`mime assets fetch`, or the pull you ran)")]) from error
    return None if found == digits.lower() else "its bytes are not the ones the checksum names"


def cheap_problem(ref: AssetRef, checksum: str | None = None) -> str | None:
    """Why the cached file cannot be trusted without a re-hash, or ``None`` when its size is the reference's and it is
    the file, unchanged, that a full verification recorded as right. The reason names neither the hash nor the cache
    path.

    With ``checksum``, the repository's checksum of the file that the caller holds (``md5:...``), a file that passes all
    that is also read once and its digest compared, so that a byte changed in place with the size and the modification
    time kept, which nothing else here sees, is found: it is a problem, and the caller's way to the right bytes is a
    fetch, which hashes the file in full and replaces one MIME wrote. Without it the shortcut hashes nothing and does
    not guard the bytes."""
    try:
        seen = file_path(ref).stat()
    except OSError:
        return "isn't downloaded"
    if seen.st_size != ref.size:
        return f"is downloaded, but the file has {size_text(seen.st_size)}, not {size_text(ref.size)}"
    recorded = _recorded(ref)
    if recorded is None:
        return "is downloaded, but has no record of a full check"
    if (recorded.size, recorded.mtime_ns) != (seen.st_size, seen.st_mtime_ns):
        return "is downloaded, but the file changed after its full check"
    if recorded.failed:
        return "is downloaded, but its full check found other bytes than the reference names"
    if checksum is not None:
        mismatch = digest_problem(file_path(ref), checksum)
        if mismatch is not None:
            return f"is downloaded, but {mismatch}"
    return None


def is_cached(ref: AssetRef) -> bool:
    """Whether :func:`locate` would find the file (its licence's acceptance, which it also asks, aside)."""
    return cheap_problem(ref) is None


#: What a message says of a record that names no route to request its file by.
NO_ROUTE = "no route given"


def request_words(ref: AssetRef) -> str:
    """Where ``ref``'s file may be asked for, as a message, a NOTICE and a plan say it: its ``request``, or that it names
    none."""
    return plain(ref.request) if ref.request else NO_ROUTE


def no_location(ref: AssetRef) -> bool:
    """Whether nothing names a location to fetch ``ref`` from: the record names none and this machine's locations file
    adds none (its mirrors included)."""
    if ref.locations:
        return False
    from mime.assets._fetch import tried_locations

    return not tried_locations(ref)


def named_by_the_sources(ref: AssetRef, folder: str | os.PathLike[str] | None = None) -> bool:
    """Whether ``mime assets fetch`` can be told ``ref`` by its name, or by the experiment ``folder`` the caller has: the
    folder's command always works; a name only where the sources (or the experiment's pins) hold this record. A record
    that sits only in a saved folder's lock or a description's manifest is ``asset.unknown`` by name, so a message
    never names that command for it."""
    if folder is not None:
        return True
    from mime.assets._known import reference

    try:
        return reference(ref.name).sha256 == ref.sha256
    except BuildRefused:
        return False


def by_request(ref: AssetRef, *, folder: str | os.PathLike[str] | None = None) -> BuildRefused:
    """``asset.by-request``: ``ref``'s file is not on this machine and no location names where to fetch it, so it is had
    by request alone. Says where to ask for it, and where a copy goes once held: MIME takes it only if its sha256 is the
    record's."""
    where = f"copy it to {cache_dir() / ref.name} (or into a mirror folder your locations file names)"
    return BuildRefused([Refusal(
        "asset.by-request", RefusalClass.ENVIRONMENT,
        f"MIME cannot fetch {ref.name!r} ({size_text(ref.size)}): its publisher gives no download location (licence "
        f"{plain(ref.licence)}). Request it from: {request_words(ref)}",
        f"once you hold the file, {where}, then run `{fetch_command(ref.name, folder=folder)}`: MIME takes it only if "
        f"its sha256 is the record's" if named_by_the_sources(ref, folder) else
        f"once you hold the file, {where}, then run the command that needed it again (`mime assets fetch <folder>` "
        f"for a saved folder, `mime visuals fetch <description folder>` for a description): MIME takes it only if "
        f"its sha256 is the record's")])


def _found(ref: AssetRef, folder: str | os.PathLike[str] | None, checksum: str | None) -> Path:
    """The cached file of ``ref`` after the cheap check, or ``asset.missing``, or ``asset.by-request`` where it is not
    there and no location names where to fetch it (:func:`by_request`); no licence is asked about. Private: the only ways
    out of this module to a file's bytes are :func:`locate` and :func:`locate_bundle_container`."""
    problem = cheap_problem(ref, checksum)
    if problem is not None and not file_path(ref).is_file() and no_location(ref):
        raise by_request(ref, folder=folder)
    if problem is not None:
        raise BuildRefused([Refusal(
            "asset.missing", RefusalClass.ENVIRONMENT,
            f"the experiment needs {ref.name!r} ({size_text(ref.size)}), which {problem}",
            f"Run: {fetch_command(ref.name, folder=folder)}" if named_by_the_sources(ref, folder) else
            "run `mime assets fetch <folder>` for the saved folder, or `mime visuals fetch <description folder>` for "
            "the description, that references it")])
    return file_path(ref)


def locate(ref: AssetRef, *, folder: str | os.PathLike[str] | None = None, checksum: str | None = None) -> Path:
    """The cached file of ``ref``, checked cheaply (size, and the record of the full verification made when it was
    stored; with the caller's ``checksum``, also the file's digest, :func:`cheap_problem`). Never downloads: a file that
    is not there, or that fails the check, is refused as ``asset.missing``, and one not there that no location names as
    ``asset.by-request`` (:func:`by_request`). The action is the exact command that stores it: for everything the
    experiment ``folder`` needs when the folder is known, else for this asset by name.

    **The first use of a gated file is gated here too** (:func:`~mime.assets.require_accepted`): bytes placed in the cache
    by hand, or recorded as checked by ``mime assets check`` or a refused resolve, never pass with no acceptance on
    record. A licence with conditions beyond credit that nothing accepted is asked about once, or refused where nobody
    can be asked (``asset.licence-not-accepted``), naming the routes a build has (a build's command has no
    ``--accept-licence``).

    **A gated file had by request** meets the two in this order: while its bytes are not under their sha256 (none
    placed, or a copy placed under its name and not yet taken), ``asset.by-request``, and no licence is asked about,
    since nothing is read (``mime assets fetch``, which takes a placed copy by its sha256, asks then); once they are,
    the gate, as for any file, and the by-request words never show for a file that is here."""
    found = _found(ref, folder, checksum)
    from mime.assets._gates import require_accepted

    require_accepted([ref], accept_with=fetch_command(ref.name, folder=folder))
    return found


def locate_bundle_container(ref: AssetRef, *, folder: str | os.PathLike[str] | None = None,
                            checksum: str | None = None) -> Path:
    """:func:`locate` for the container file of a bundle a pull has just downloaded
    (:class:`~mime.assets._reference.BundleContainer`, which only that download makes), whose own terms are judged when
    it is unpacked. **Any other reference is gated as :func:`locate` gates it**: a gated asset's record is refused here
    as there, whatever it is named. So is a container whose bytes are those of a gated record the shared sources
    (installed manifests, the locations file) name: the file is the record's first, and a pull's reading it as a
    bundle does not make it any less so. A record declared only in an experiment's script is not known here."""
    if not isinstance(ref, BundleContainer):
        return locate(ref, folder=folder, checksum=checksum)
    from mime.assets._gates import require_accepted
    from mime.assets._sources import shared_records

    require_accepted([known for known in shared_records().values() if known.sha256 == ref.sha256],
                     accept_with=fetch_command(ref.name, folder=folder))
    return _found(ref, folder, checksum)


#: Where the cache keeps a file under its record's name, for a reader that needs a file name (a USD layer, a URDF):
#: ``named/<sha256>/<name>``, a hard link to the file kept under its sha256 (a copy where links are not possible).
NAMED_FOLDER = "named"


def named_path(ref: AssetRef) -> Path:
    """Where the cache keeps ``ref``'s file under its record's name. Declaring reads nothing: the file is linked there
    when it is first read (:func:`named_file`)."""
    return cache_dir() / NAMED_FOLDER / ref.sha256 / ref.name


def named_file(ref: AssetRef) -> Path:
    """``ref``'s cached file under its record's name, checked as :func:`locate` checks it (never downloaded: a file not
    in the cache is refused as ``asset.missing`` with the command)."""
    import shutil

    stored = locate(ref)
    path = named_path(ref)
    if path.is_file() and path.stat().st_size == ref.size and path.stat().st_ino == stored.stat().st_ino:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        temporary.hardlink_to(stored)
    except OSError:
        shutil.copyfile(stored, temporary)
    os.replace(temporary, path)
    return path


def hash_problem(ref: AssetRef) -> str | None:
    """Why the file at the cache path is not ``ref``'s bytes, or ``None`` when it is. A file of another size is not
    hashed: its size already says it is another file, or a copy still in progress. The result is recorded, with the
    origin the record already holds (a file with no record is taken to be a person's). The caller holds the lock."""
    path = file_path(ref)
    held = _recorded(ref)
    origin = held.origin if held is not None else PLACED
    seen = path.stat().st_size
    if seen != ref.size:
        problem = f"its size differs: {size_text(seen)}, not {size_text(ref.size)}"
    elif sha256_of_file(path) != ref.sha256:
        problem = "its hash differs"
    else:
        write_record(ref, origin)
        return None
    write_record(ref, origin, failed=True)
    return problem


def check(ref: AssetRef) -> Cached:
    """Hash the cached file in full: the integrity of a stored file, no experiment is run. A match records the
    verification (so :func:`locate` accepts the file). A difference is recorded too: :func:`locate` refuses the file
    until a fetch repairs it, and the file is left where it is (a fetch replaces it only if MIME downloaded it). Waits
    for a fetch of the same asset in another process."""
    if not file_path(ref).is_file():
        return "not cached"
    with exclusive(lock_path(ref)):
        if not file_path(ref).is_file():
            return "not cached"
        return "ok" if hash_problem(ref) is None else "corrupt"
