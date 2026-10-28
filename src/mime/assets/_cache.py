"""The cache: one file per sha256, and beside it a record of the full verification made when it was stored.

A build must not re-hash a file of about nine hundred megabytes, so :func:`locate` makes a cheap check: the file's size
is the reference's, and its size and modification time are the ones recorded at the full verification. That catches a
truncated, replaced or edited file; it does not catch a flipped byte that leaves both unchanged, which only
:func:`check` (a full re-hash) finds.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mime.assets._commands import fetch_command
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

ASSETS_DIR_VARIABLE = "MIME_ASSETS_DIR"
OFFLINE_VARIABLE = "MIME_OFFLINE"
RECORD_SUFFIX = ".verified"
_CHUNK = 8 * 1024 * 1024

type Cached = Literal["ok", "not cached", "corrupt"]


def cache_dir() -> Path:
    """``$MIME_ASSETS_DIR``, or ``assets`` under the platform's cache directory for MIME."""
    chosen = os.environ.get(ASSETS_DIR_VARIABLE)
    if chosen:
        return Path(chosen)
    import pooch

    return Path(str(pooch.os_cache("mime"))) / "assets"


def offline() -> bool:
    """``MIME_OFFLINE=1``: nothing is downloaded."""
    return os.environ.get(OFFLINE_VARIABLE, "").strip().lower() in {"1", "true", "yes"}


def file_path(ref: AssetRef) -> Path:
    return cache_dir() / ref.sha256


def record_path(ref: AssetRef) -> Path:
    return cache_dir() / (ref.sha256 + RECORD_SUFFIX)


def sha256_of_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class _Seen:
    """The file as it was when its full hash was checked."""

    size: int
    mtime_ns: int


def write_record(ref: AssetRef) -> None:
    """Record that the cached file has been hashed in full and is ``ref.sha256``; call it only after that check."""
    seen = file_path(ref).stat()
    text = json.dumps({"sha256": ref.sha256, "size": seen.st_size, "mtime_ns": seen.st_mtime_ns,
                       "verified": "full sha256 of the file's bytes"}, indent=2)
    partial = record_path(ref).with_suffix(RECORD_SUFFIX + ".partial")
    partial.write_text(text + "\n", encoding="utf-8")
    partial.replace(record_path(ref))


def _recorded(ref: AssetRef) -> _Seen | None:
    try:
        held = json.loads(record_path(ref).read_text(encoding="utf-8"))
        if held["sha256"] != ref.sha256:
            return None
        return _Seen(int(held["size"]), int(held["mtime_ns"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def size_text(size: int) -> str:
    """A size as the messages and the listing print it: bytes, kB or MB (decimal)."""
    if size >= 1_000_000:
        return f"{size / 1e6:.0f} MB"
    return f"{size / 1e3:.0f} kB" if size >= 1_000 else f"{size} bytes"


def cheap_problem(ref: AssetRef) -> str | None:
    """Why the cached file cannot be trusted without a re-hash, or ``None`` when its size is the reference's and it is
    the file, unchanged, that a full verification recorded. The reason names neither the hash nor the cache path."""
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


def check(ref: AssetRef) -> Cached:
    """Hash the cached file in full: the integrity of a stored file, no experiment is run. A match records the
    verification (so :func:`locate` accepts the file); a difference is reported and the file left where it is."""
    if not file_path(ref).is_file():
        return "not cached"
    if sha256_of_file(file_path(ref)) != ref.sha256:
        return "corrupt"
    write_record(ref)
    return "ok"
