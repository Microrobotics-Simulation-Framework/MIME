"""Resolving: the one lookup of a content-addressed file, shared by every caller that needs one stored.

``resolve`` serves an asset now; a bundle (a packed experiment) is a content-addressed file too, and is resolved by the
same function, with the same cache, offline switch, hash check, refusal codes and size report.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mime.assets._cache import (
    DOWNLOADED,
    INCOMING_INFIX,
    OFFLINE_VARIABLE,
    PLACED,
    RECORD_SUFFIX,
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
from mime.assets._exclusive import LOCK_SUFFIX, exclusive
from mime.assets._reference import AssetRef, location_problem
from mime.build import BuildRefused, Refusal, RefusalClass

#: The part of the message pooch raises for a download whose hash differs (``pooch.hashes.hash_matches``). Any other
#: ``ValueError`` out of pooch is a failed location, not a wrong file.
_POOCH_MISMATCH = "does not match the known hash"


def _held(value: object) -> Any:
    return value


#: The most a file is taken from a location when nothing has said how large it is (a bundle at an address, a record
#: that does not state a size). A choice, not a measurement: a person who knows better raises it (``mime pull
#: --max-bundle-bytes``). A file of a stated size is held to that size instead.
RETRIEVE_LIMIT_BYTES = 500_000_000


class _SizeExceeded(Exception):
    """A location sent more than the size the download is held to."""

    def __init__(self, limit: int, received: int) -> None:
        super().__init__(f"more than {limit} bytes")
        self.limit = limit
        self.received = received


class _Capped:
    """The file a download is written to, which refuses the chunk that takes it past ``limit``: a download stops where it
    is, whatever the server goes on to send, and what it wrote is discarded with its temporary file."""

    def __init__(self, stream: Any, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self.received = 0

    def write(self, chunk: bytes) -> int:
        self.received += len(chunk)
        if self.received > self._limit:
            raise _SizeExceeded(self._limit, self.received)
        return int(self._stream.write(chunk))

    def flush(self) -> None:
        self._stream.flush()


class _CappedDownload:
    """A pooch downloader that is the one pooch would choose for the location, with its output held to a size."""

    def __init__(self, inner: Any, limit: int) -> None:
        self._inner = inner
        self._limit = limit

    def __call__(self, url: str, output_file: str | Path, pooch: object, check_only: bool = False) -> Any:
        if check_only:
            return self._inner(url, output_file, pooch, check_only=True)
        with Path(output_file).open("w+b") as stream:
            self._inner(url, _Capped(stream, self._limit), pooch)
        return None


def _download_held_to(location: str, known_hash: str | None, fname: str, directory: Path, limit: int) -> Path:
    """One location through pooch into ``directory``, which pooch fills through a temporary file moved into place only
    if its hash is ``known_hash`` (``sha256:...`` for an asset, the repository's ``md5:...`` for a bundle fetched by DOI,
    ``None`` for none). A ``doi:`` location is resolved by pooch to the repository's file URL. The download stops, and
    raises :class:`_SizeExceeded`, once it has passed ``limit`` bytes."""
    import pooch

    # pooch types its path argument as a bare PathLike, which a strict checker reads as unknown.
    library = _held(pooch)
    downloader = _CappedDownload(library.downloaders.choose_downloader(location), limit)
    return Path(str(library.retrieve(url=location, known_hash=known_hash, fname=fname, path=directory,
                                     downloader=downloader)))


def _download(location: str, ref: AssetRef, directory: Path) -> Path:
    """One location of ``ref`` into ``directory``, held to the reference's sha256 and its declared size."""
    return _download_held_to(location, f"sha256:{ref.sha256}", ref.sha256, directory, ref.size)


def _placed_file(ref: AssetRef, why: str, path: Path | None = None) -> BuildRefused:
    path = path or file_path(ref)
    return BuildRefused([Refusal(
        "asset.placed-file", RefusalClass.ENVIRONMENT,
        f"{ref.name}: the file {path.name} in {where_text()} does not match: {why}. MIME does not delete or "
        f"overwrite a file it did not write",
        f"if it is still being copied, wait for the copy; otherwise remove {path}, then run `{fetch_command(ref.name)}` "
        f"again")])


_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _file_name(location: str) -> str | None:
    """The name of the file a location serves: the last part of an ``https://`` URL's path, or the file of a
    ``doi:`` location."""
    last = (location.rpartition("/")[2] if location.startswith("doi:") else urlsplit(location).path.rpartition("/")[2])
    ok = _FILE_NAME.fullmatch(last) and not last.endswith((RECORD_SUFFIX, LOCK_SUFFIX, ".partial")) \
        and INCOMING_INFIX not in last
    return last if ok else None


def _names_a_person_may_use(ref: AssetRef) -> list[tuple[str, bool]]:
    """The names, besides the sha256, under which a file copied in by hand is looked for, each with whether the name is
    this asset's alone. The asset's own name, alone or with the extension its locations' files have, is its alone;
    the file name a location serves is not (``r30/wall_R2.500.npz`` and ``r40/wall_R2.500.npz`` are two assets), so a
    file under that name belongs to the asset only if its bytes are the reference's."""
    served = [name for name in map(_file_name, ref.locations) if name is not None]
    own = [ref.name, *(ref.name + Path(name).suffix for name in served)]
    names = [(name, True) for name in dict.fromkeys(own)] + [(name, False) for name in dict.fromkeys(served)]
    return [(name, own_name) for name, own_name in names if name != ref.sha256]


def _link_into_place(ref: AssetRef, placed: Path) -> None:
    """Make the file a person placed under another name the cache's file for ``ref``: a hard link, so the person's file
    stays where and as it was, and a copy on a filesystem without them."""
    final = file_path(ref)
    try:
        final.hardlink_to(placed.resolve())
    except FileExistsError:
        raise _placed_file(ref, "it appeared while the file was being adopted") from None
    except OSError:
        incoming = Path(tempfile.mkdtemp(prefix=f"{ref.sha256}{INCOMING_INFIX}", dir=cache_dir()))
        try:
            copy = incoming / ref.sha256
            shutil.copyfile(placed, copy)
            _place(ref, copy, replace=False)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)


def _adopt_named_file(ref: AssetRef) -> bool:
    """Take a file a person copied into the cache directory under the asset's name, or under the name its location
    serves (``wall_R2.500.npz``), when its full hash is the reference's: hashed once, linked as the sha256 and recorded
    as a person's. A file under the asset's own name with other bytes, or another size, is refused as
    ``asset.placed-file``; under a location's file name it is another asset's or no asset's, and is left alone. No file
    is deleted, overwritten or renamed. The caller holds the asset's lock."""
    for name, own in _names_a_person_may_use(ref):
        path = cache_dir() / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size == ref.size and sha256_of_file(path) == ref.sha256:
            _link_into_place(ref, path)
            write_record(ref, PLACED)
            return True
        if own or size == ref.size:
            raise _placed_file(ref, f"its size differs: {size_text(size)}, not {size_text(ref.size)}"
                               if size != ref.size else "its hash differs", path)
    return False


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


def _install(ref: AssetRef, arrived: Path, *, replace: bool) -> Path:
    _place(ref, arrived, replace=replace)
    write_record(ref, DOWNLOADED)
    return file_path(ref)


def _store(ref: AssetRef, report: Callable[[str], None] | None, *, replace: bool) -> Path:
    """The locations in order, each into a folder of its own; the first whose file is the reference's is moved into
    place and recorded."""
    failures: list[str] = []
    mismatched = oversize = False
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
                too_many = isinstance(error, _SizeExceeded)
                mismatched = mismatched or wrong_bytes
                oversize = oversize or too_many
                failures.append(f"{location}: " + (
                    f"sent more than the {size_text(ref.size)} the reference declares, and the download was stopped"
                    if too_many else "other bytes than the reference names" if wrong_bytes else str(error)))
                continue
            if sha256_of_file(arrived) != ref.sha256:
                mismatched = True
                failures.append(f"{location}: stored other bytes than the reference names")
                continue
            return _install(ref, arrived, replace=replace)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
    tried = "; ".join(failures)
    if mismatched:
        raise BuildRefused([Refusal(
            "asset.hash-mismatch", RefusalClass.ENVIRONMENT,
            f"no location served the bytes with sha256 {ref.sha256} for {ref.name}: {tried}",
            "check the reference's sha256 and the locations; nothing was stored")])
    if oversize:
        raise BuildRefused([Refusal(
            "asset.size-exceeded", RefusalClass.ENVIRONMENT,
            f"{ref.name} is declared as {size_text(ref.size)}, and no location served that size: {tried}",
            "check the reference's size and the locations; nothing was stored")])
    raise BuildRefused([Refusal(
        "asset.unreachable", RefusalClass.ENVIRONMENT, f"{ref.name} could not be downloaded from any location: {tried}",
        f"check the network, or copy the file to {file_path(ref)} and run `{check_command()}`")])


def resolve(ref: AssetRef, *, report: Callable[[str], None] | None = None, full: bool = False) -> Path:
    """The file of ``ref`` in the cache, stored first if it is not there. In this order:

    1. **The cache**, keyed by the hash: a file whose full verification is recorded and unchanged is returned, with no
       network (not with ``full``, which hashes the file again whatever the record says).
    2. **A file already in the cache directory** (``MIME_ASSETS_DIR``, which a cluster fills by hand) that is not yet
       recorded, named by its sha256, by the asset's name or by the file name a location serves: hashed in full, no
       network, kept if its sha256 is the reference's. A file with other bytes is never deleted or overwritten when a
       person placed it (``asset.placed-file``); one MIME downloaded is replaced by step 3.
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
        stored, replace = _look_at_what_is_there(ref)
        if stored:
            return file_path(ref)
        if is_offline:
            raise _offline(ref, present=replace)
        return _store(ref, report, replace=replace)


def _look_at_what_is_there(ref: AssetRef) -> tuple[bool, bool]:
    """Whether the cache directory already holds ``ref``'s bytes, and when it does not, whether a file MIME wrote is
    there to be replaced. A file a person placed with other bytes is refused. The caller holds the asset's lock."""
    if not file_path(ref).is_file():
        return _adopt_named_file(ref), False
    why = hash_problem(ref)
    if why is None:
        return True, False
    if not mime_wrote(ref):
        raise _placed_file(ref, why)
    return False, True


def fetch(ref: AssetRef, *, report: Callable[[str], None] | None = None) -> Path:
    """Store ``ref`` in the cache and return its file: :func:`resolve` with the file hashed in full, so that a file
    the cheap check passes, a flipped byte among them, is still found and repaired."""
    return resolve(ref, report=report, full=True)


#: A download of a bundle whose hash is not yet known goes to a folder with this prefix; one a killed process left is
#: removed by the next after a day.
_UNHASHED = "unhashed" + INCOMING_INFIX
_STALE_SECONDS = 24 * 3600


def _clear_stale_unhashed() -> None:
    for old in cache_dir().glob(f"{_UNHASHED}*"):
        if time.time() - old.stat().st_mtime > _STALE_SECONDS:
            shutil.rmtree(old, ignore_errors=True)


def retrieve(location: str, *, name: str, known_hash: str | None = None, max_bytes: int = RETRIEVE_LIMIT_BYTES,
             exceeded_action: str | None = None, report: Callable[[str], None] | None = None) -> AssetRef:
    """Store the file at one ``location`` whose sha256 nobody has said yet (a bundle named by its DOI), and return the
    reference to it that the hash makes: the file is in the cache under that hash, and :func:`locate` finds it.

    ``known_hash`` is the repository's own checksum of the file (``md5:...`` from Zenodo's record), which pooch holds the
    download to; it guards the transfer, and nothing else vouches for the bytes until someone pins their sha256 in a
    later :func:`resolve`. The location must be an allowed form, and ``MIME_OFFLINE`` refuses it. The download stops
    once it has passed ``max_bytes`` (the size a record states, or :data:`RETRIEVE_LIMIT_BYTES`), refused as
    ``asset.size-exceeded`` with ``exceeded_action`` as what to do."""
    problem = location_problem(location)
    if problem is not None:
        raise BuildRefused([Refusal(
            "asset.reference", RefusalClass.USER, f"{name}: location {location!r}: {problem}",
            "give an https:// URL or a doi: URL")])
    if offline():
        raise BuildRefused([Refusal(
            "asset.offline", RefusalClass.ENVIRONMENT,
            f"{name} is not cached, and {OFFLINE_VARIABLE} is set, so nothing is downloaded",
            f"unset {OFFLINE_VARIABLE}, or pass the file you copied here")])
    cache_dir().mkdir(parents=True, exist_ok=True)
    _clear_stale_unhashed()
    incoming = Path(tempfile.mkdtemp(prefix=_UNHASHED, dir=cache_dir()))
    try:
        if report is not None:
            report(f"{name}: from {location}")
        try:
            arrived = _download_held_to(location, known_hash, name, incoming, max_bytes)
        except Exception as error:  # noqa: BLE001 - pooch, requests and the DOI resolvers raise unrelated types
            if isinstance(error, _SizeExceeded):
                raise BuildRefused([Refusal(
                    "asset.size-exceeded", RefusalClass.ENVIRONMENT,
                    f"{name} from {location} is larger than the {size_text(max_bytes)} it may be, and the download was "
                    f"stopped there", exceeded_action or "check the address; nothing was stored")]) from None
            if isinstance(error, ValueError) and _POOCH_MISMATCH in str(error):
                raise BuildRefused([Refusal(
                    "asset.hash-mismatch", RefusalClass.ENVIRONMENT,
                    f"{name} from {location} has other bytes than the repository's checksum names",
                    "try again; if it repeats the record's file is not the one it lists")]) from error
            raise BuildRefused([Refusal(
                "asset.unreachable", RefusalClass.ENVIRONMENT, f"{name} could not be downloaded from {location}: {error}",
                "check the network and the address")]) from error
        ref = AssetRef(name, sha256_of_file(arrived), arrived.stat().st_size, "not stated", (location,))
        with exclusive(lock_path(ref)):
            if cheap_problem(ref) is None:
                return ref
            stored, replace = _look_at_what_is_there(ref)
            if not stored:
                _install(ref, arrived, replace=replace)
        return ref
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
