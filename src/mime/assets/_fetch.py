"""Resolving: the one lookup of a content-addressed file, shared by every caller that needs one stored.

``resolve`` serves an asset now; a bundle (a packed experiment) is a content-addressed file too, and is resolved by the
same function, with the same cache, offline switch, hash check, refusal codes and size report.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mime.assets._cache import (
    OFFLINE_VARIABLE,
    cache_dir,
    check,
    cheap_problem,
    file_path,
    offline,
    sha256_of_file,
    size_text,
    write_record,
)
from mime.assets._commands import check_command, fetch_command
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

#: The part of the message pooch raises for a download whose hash differs (``pooch.hashes.hash_matches``). Any other
#: ``ValueError`` out of pooch is a failed location, not a wrong file.
_POOCH_MISMATCH = "does not match the known hash"


def _held(value: object) -> Any:
    return value


def _download(location: str, ref: AssetRef, directory: Path) -> Path:
    """One location through pooch, which downloads to a temporary file and moves it into place only if its sha256 is
    the reference's (a ``doi:`` location is resolved by pooch to the repository's file URL)."""
    import pooch

    # pooch types its path argument as a bare PathLike, which a strict checker reads as unknown.
    hash_ = f"sha256:{ref.sha256}"
    return Path(str(_held(pooch).retrieve(url=location, known_hash=hash_, fname=ref.sha256, path=directory)))


def resolve(ref: AssetRef, *, report: Callable[[str], None] | None = None) -> Path:
    """The file of ``ref`` in the cache, stored first if it is not there. In this order:

    1. **The cache**, keyed by the hash: a file whose full verification is recorded and unchanged is returned, with no
       network.
    2. **A file already in the cache directory** (``MIME_ASSETS_DIR``, which a cluster fills by hand) that is not yet
       recorded: hashed in full, no network, kept if its sha256 is the reference's.
    3. **The locations, primary then backups**: the first location is the primary, the permanent archive once there is
       one, and the rest are backups tried only when the one before fails (unreachable, an error, or other bytes). Each
       goes through pooch, which checks the sha256, and the stored file is hashed in full again before it is recorded.

    With ``MIME_OFFLINE=1`` step 3 is refused (``asset.offline``). ``report`` is called with a line before each
    download, naming the size and the location."""
    if cheap_problem(ref) is None:
        return file_path(ref)
    if file_path(ref).is_file() and check(ref) == "ok":
        return file_path(ref)
    file_path(ref).unlink(missing_ok=True)  # the cache is keyed by hash: bytes that are not the hash are no use
    if offline():
        raise BuildRefused([Refusal(
            "asset.offline", RefusalClass.ENVIRONMENT,
            f"{ref.name} is not cached, and {OFFLINE_VARIABLE} is set, so nothing is downloaded (the cache is "
            f"{cache_dir()})",
            f"fetch it on a machine with network access and copy the file to {file_path(ref)}, then run "
            f"`{fetch_command(ref.name)}` again (it hashes the file and downloads nothing), or unset "
            f"{OFFLINE_VARIABLE}")])
    cache_dir().mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    mismatched = False
    for location in ref.locations:
        if report is not None:
            report(f"{ref.name}: {size_text(ref.size)} from {location}")
        try:
            _download(location, ref, cache_dir())
        except Exception as error:  # noqa: BLE001 - pooch, requests and the DOI resolvers raise unrelated types
            wrong_bytes = isinstance(error, ValueError) and _POOCH_MISMATCH in str(error)
            mismatched = mismatched or wrong_bytes
            failures.append(f"{location}: {'other bytes than the reference names' if wrong_bytes else error}")
            continue
        if sha256_of_file(file_path(ref)) == ref.sha256:
            write_record(ref)
            return file_path(ref)
        file_path(ref).unlink(missing_ok=True)
        mismatched = True
        failures.append(f"{location}: stored other bytes than the reference names")
    tried = "; ".join(failures)
    if mismatched:
        raise BuildRefused([Refusal(
            "asset.hash-mismatch", RefusalClass.ENVIRONMENT,
            f"no location served the bytes with sha256 {ref.sha256} for {ref.name}: {tried}",
            "check the reference's sha256 and the locations; nothing was stored")])
    raise BuildRefused([Refusal(
        "asset.unreachable", RefusalClass.ENVIRONMENT, f"{ref.name} could not be downloaded from any location: {tried}",
        f"check the network, or copy the file to {file_path(ref)} and run `{check_command()}`")])


def fetch(ref: AssetRef, *, report: Callable[[str], None] | None = None) -> Path:
    """Store ``ref`` in the cache and return its file: :func:`resolve`, under the name the command uses."""
    return resolve(ref, report=report)
