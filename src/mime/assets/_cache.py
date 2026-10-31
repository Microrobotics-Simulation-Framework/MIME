"""The cache: one directory, one file per sha256, and beside it a record of the full verification made of it.

A build must not re-hash a file of about nine hundred megabytes, so :func:`locate` makes a cheap check: the file's size
is the reference's, and its size and modification time are the ones recorded at the full verification. That catches a
truncated, replaced or edited file; it does not catch a flipped byte that leaves both unchanged, which only
:func:`check` (a full re-hash) finds, and which it then marks, so that :func:`locate` refuses the file until a fetch
repairs it.

The record also says who wrote the file. A file MIME downloaded may be replaced by a new download; a file a person
placed in the directory is never deleted or overwritten, whatever its bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mime.assets._commands import fetch_command
from mime.assets._exclusive import LOCK_SUFFIX, exclusive
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

ASSETS_DIR_VARIABLE = "MIME_ASSETS_DIR"
OFFLINE_VARIABLE = "MIME_OFFLINE"
RECORD_SUFFIX = ".verified"
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


def cheap_problem(ref: AssetRef) -> str | None:
    """Why the cached file cannot be trusted without a re-hash, or ``None`` when its size is the reference's and it is
    the file, unchanged, that a full verification recorded as right. The reason names neither the hash nor the cache
    path."""
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
    return None


def is_cached(ref: AssetRef) -> bool:
    """Whether :func:`locate` would return the file."""
    return cheap_problem(ref) is None


def locate(ref: AssetRef, *, folder: str | os.PathLike[str] | None = None) -> Path:
    """The cached file of ``ref``, checked cheaply (size, and the record of the full verification made when it was
    stored). Never downloads: a file that is not there, or that fails the check, is refused as ``asset.missing``. The
    action is the exact command that stores it: for everything the experiment ``folder`` needs when the folder is
    known, else for this asset by name."""
    problem = cheap_problem(ref)
    if problem is not None:
        raise BuildRefused([Refusal(
            "asset.missing", RefusalClass.ENVIRONMENT,
            f"the experiment needs {ref.name!r} ({size_text(ref.size)}), which {problem}",
            f"Run: {fetch_command(ref.name, folder=folder)}")])
    return file_path(ref)


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
