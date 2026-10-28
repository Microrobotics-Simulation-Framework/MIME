"""Resolving: the one lookup of a content-addressed file, shared by every caller that needs one stored.

``resolve`` serves an asset now; a bundle (a packed experiment) is a content-addressed file too, and is resolved by the
same function, with the same cache, offline switch, hash check, refusal codes and size report.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mime.assets._cache import (
    DOWNLOADED,
    INCOMING_INFIX,
    OFFLINE_VARIABLE,
    cache_dir,
    cheap_problem,
    file_path,
    hash_problem,
    lock_path,
    mime_wrote,
    offline,
    sha256_of_file,
    size_text,
    where_text,
    write_record,
)
from mime.assets._commands import check_command, fetch_command
from mime.assets._exclusive import exclusive
from mime.assets._reference import AssetRef, location_problem
from mime.build import BuildRefused, Refusal, RefusalClass

#: The part of the message pooch raises for a download whose hash differs (``pooch.hashes.hash_matches``). Any other
#: ``ValueError`` out of pooch is a failed location, not a wrong file.
_POOCH_MISMATCH = "does not match the known hash"


def _held(value: object) -> Any:
    return value


def _download(location: str, ref: AssetRef, directory: Path) -> Path:
    """One location through pooch into ``directory``, which pooch fills through a temporary file moved into place only
    if its sha256 is the reference's (a ``doi:`` location is resolved by pooch to the repository's file URL)."""
    import pooch

    # pooch types its path argument as a bare PathLike, which a strict checker reads as unknown.
    hash_ = f"sha256:{ref.sha256}"
    return Path(str(_held(pooch).retrieve(url=location, known_hash=hash_, fname=ref.sha256, path=directory)))


def _placed_file(ref: AssetRef, why: str) -> BuildRefused:
    path = file_path(ref)
    return BuildRefused([Refusal(
        "asset.placed-file", RefusalClass.ENVIRONMENT,
        f"{ref.name}: the file {path.name} in {where_text()} does not match: {why}. MIME does not delete or "
        f"overwrite a file it did not write",
        f"if it is still being copied, wait for the copy; otherwise remove {path}, then run `{fetch_command(ref.name)}` "
        f"again")])


def _offline(ref: AssetRef, present: bool) -> BuildRefused:
    state = "is cached, but its bytes are not the reference's," if present else "is not cached,"
    return BuildRefused([Refusal(
        "asset.offline", RefusalClass.ENVIRONMENT,
        f"{ref.name} {state} and {OFFLINE_VARIABLE} is set, so nothing is downloaded (the cache is {cache_dir()})",
        f"fetch it on a machine with network access and copy the file to {file_path(ref)}, then run "
        f"`{fetch_command(ref.name)}` again (it hashes the file and downloads nothing), or unset "
        f"{OFFLINE_VARIABLE}")])


def _clear_orphans(ref: AssetRef) -> None:
    """Remove the download folders a process killed mid-download left. The caller holds the asset's lock, and every
    download of the asset runs under it, so none of them is in use."""
    for orphan in cache_dir().glob(f"{ref.sha256}{INCOMING_INFIX}*"):
        shutil.rmtree(orphan, ignore_errors=True)


def _place(ref: AssetRef, arrived: Path, *, replace: bool) -> None:
    """Move the verified download to the cache path. A file MIME wrote is replaced; where there is none the move fails
    if one has appeared (a person's copy that started meanwhile), and that file is left alone."""
    final = file_path(ref)
    if replace:
        arrived.replace(final)
        return
    try:
        final.hardlink_to(arrived)
    except FileExistsError:
        raise _placed_file(ref, "it appeared while the download was running") from None
    except OSError:  # a filesystem without hard links: the check above, then the rename
        if final.exists():
            raise _placed_file(ref, "it appeared while the download was running") from None
        arrived.replace(final)


def _store(ref: AssetRef, report: Callable[[str], None] | None, *, replace: bool) -> Path:
    """The locations in order, each into a folder of its own; the first whose file is the reference's is moved into
    place and recorded."""
    failures: list[str] = []
    mismatched = False
    for location in ref.locations:
        problem = location_problem(location)
        if problem is not None:  # a reference built around the check: the location is never contacted
            failures.append(f"{location}: not used, {problem}")
            continue
        if report is not None:
            report(f"{ref.name}: {size_text(ref.size)} from {location}")
        incoming = Path(tempfile.mkdtemp(prefix=f"{ref.sha256}{INCOMING_INFIX}", dir=cache_dir()))
        try:
            try:
                arrived = _download(location, ref, incoming)
            except Exception as error:  # noqa: BLE001 - pooch, requests and the DOI resolvers raise unrelated types
                wrong_bytes = isinstance(error, ValueError) and _POOCH_MISMATCH in str(error)
                mismatched = mismatched or wrong_bytes
                failures.append(f"{location}: {'other bytes than the reference names' if wrong_bytes else error}")
                continue
            if sha256_of_file(arrived) != ref.sha256:
                mismatched = True
                failures.append(f"{location}: stored other bytes than the reference names")
                continue
            _place(ref, arrived, replace=replace)
            write_record(ref, DOWNLOADED)
            return file_path(ref)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
    tried = "; ".join(failures)
    if mismatched:
        raise BuildRefused([Refusal(
            "asset.hash-mismatch", RefusalClass.ENVIRONMENT,
            f"no location served the bytes with sha256 {ref.sha256} for {ref.name}: {tried}",
            "check the reference's sha256 and the locations; nothing was stored")])
    raise BuildRefused([Refusal(
        "asset.unreachable", RefusalClass.ENVIRONMENT, f"{ref.name} could not be downloaded from any location: {tried}",
        f"check the network, or copy the file to {file_path(ref)} and run `{check_command()}`")])


def resolve(ref: AssetRef, *, report: Callable[[str], None] | None = None, full: bool = False) -> Path:
    """The file of ``ref`` in the cache, stored first if it is not there. In this order:

    1. **The cache**, keyed by the hash: a file whose full verification is recorded and unchanged is returned, with no
       network (not with ``full``, which hashes the file again whatever the record says).
    2. **A file already in the cache directory** (``MIME_ASSETS_DIR``, which a cluster fills by hand) that is not yet
       recorded: hashed in full, no network, kept if its sha256 is the reference's. A file with other bytes is never
       deleted or overwritten when a person placed it (``asset.placed-file``); one MIME downloaded is replaced by step 3.
    3. **The locations, primary then backups**: the first location is the primary, the permanent archive once there is
       one, and the rest are backups tried only when the one before fails (unreachable, an error, or other bytes). Each
       goes through pooch into a folder of its own, which checks the sha256; the file is hashed in full again before it
       is moved into place and recorded.

    Steps 2 and 3 hold a lock on the asset: a second process waits, then finds the file stored. With
    ``MIME_OFFLINE`` set step 3 is refused (``asset.offline``). ``report`` is called with a line before each download,
    naming the size and the location."""
    is_offline = offline()
    if not full and cheap_problem(ref) is None:
        return file_path(ref)
    if is_offline and not cache_dir().is_dir():
        raise _offline(ref, present=False)
    with exclusive(lock_path(ref)):
        if not full and cheap_problem(ref) is None:
            return file_path(ref)
        _clear_orphans(ref)
        replace = False
        if file_path(ref).is_file():
            why = hash_problem(ref)
            if why is None:
                return file_path(ref)
            if not mime_wrote(ref):
                raise _placed_file(ref, why)
            replace = True
        if is_offline:
            raise _offline(ref, present=replace)
        return _store(ref, report, replace=replace)


def fetch(ref: AssetRef, *, report: Callable[[str], None] | None = None) -> Path:
    """Store ``ref`` in the cache and return its file: :func:`resolve` with the file hashed in full, so that a file
    the cheap check passes, a flipped byte among them, is still found and repaired."""
    return resolve(ref, report=report, full=True)
