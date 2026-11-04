"""Resolving: the one lookup of a content-addressed file, shared by every caller that needs one stored.

``resolve`` serves an asset: its locations are opened by adapters (``mime.assets._adapters``), and the resolver keeps
the rest, the cache, the offline switch, the hash check, the size cap, redirects with a credential bound to its host,
timeouts, retries with backoff and resuming. A bundle (a packed experiment) is a content-addressed file too, whose hash
is not known before it arrives: ``retrieve`` stores it, through pooch's downloader, into the same cache.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import math
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, override
from urllib.parse import unquote, urlsplit

from mime._transfers import Transfers
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
from mime.assets._adapters import (
    LOOPBACK_HOSTS,
    Adapter,
    AdapterError,
    Credential,
    CredentialHook,
    LoginRefused,
    NotFound,
    Opened,
    Redirect,
    Request,
    Transient,
    adapter_for,
    host_of,
    stored_credential,
)
from mime.assets._locations import read_locations
from mime.assets._reference import (
    BUILT_IN_SCHEMES,
    KEY_HIDDEN,
    AssetRef,
    carries_key,
    location_problem,
    redacted,
    scheme_of,
)
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token
from mime.terms import NO_ASSERTION

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


#: Told how each download is going: the files to come, how much of each has arrived, when each is whole and checked,
#: and what is under way between them. Set for the downloads made inside :func:`watching_downloads` (a terminal's
#: progress); nothing else reads it.
_watcher: ContextVar[Transfers | None] = ContextVar("mime_download_watcher", default=None)
#: How many bytes arrive between two reports of a download's progress.
_REPORT_BYTES = 128 * 1024


@contextmanager
def watching_downloads(progress: Transfers) -> Generator[None]:
    """Inside this block ``progress`` is told how each download is going: its bytes so far (when it starts, about every
    128 kB, and as it ends), when it is whole, and what is under way while no bytes arrive. The download itself is as
    it was. A fault in ``progress`` is read, inside the download, as the download failing: a display catches its own."""
    token = _watcher.set(progress)
    try:
        yield
    finally:
        _watcher.reset(token)


def announce_downloads(files: Sequence[tuple[str, int | None]]) -> None:
    """The files about to be downloaded, with their sizes, to whoever is watching: how many there are to count to."""
    watching = _watcher.get()
    if watching is not None:
        watching.files(files)


def downloads_doing(words: str) -> None:
    """What is under way while no bytes arrive (a lookup, a hash, an unpacking), to whoever is watching."""
    watching = _watcher.get()
    if watching is not None:
        watching.doing(words)


class _Capped:
    """The file a download is written to, which refuses the chunk that takes it past ``limit``: a download stops where it
    is, whatever the server goes on to send, and what it wrote is discarded with its temporary file."""

    def __init__(self, stream: Any, limit: int, label: str = "", expected: int | None = None) -> None:
        self._stream: Any = stream
        self._limit = limit
        self._label = label
        self._expected = expected
        self._report = _watcher.get()
        self._told = 0
        self.received = 0
        if self._report is not None:
            self._report.sending(label, 0, expected)

    def write(self, chunk: bytes) -> int:
        self.received += len(chunk)
        if self.received > self._limit:
            raise _SizeExceeded(self._limit, self.received)
        written = int(self._stream.write(chunk))
        if self._report is not None and self.received - self._told >= _REPORT_BYTES:
            self._told = self.received
            self._report.sending(self._label, self.received, self._expected)
        return written

    def flush(self) -> None:
        self._stream.flush()

    def stated(self, size: int) -> None:
        """The store states the whole file is ``size`` bytes: refused before a byte is taken where that is past the
        limit, as a stream that goes past it would be."""
        if size > self._limit:
            raise _SizeExceeded(self._limit, size)

    def restart(self) -> None:
        """The download starts again from its first byte (a store that ignored a range): nothing received yet."""
        self.received = 0

    def finish(self) -> None:
        """The download ended: its last report, whatever was told since the one before."""
        if self._report is not None:
            if self.received != self._told:
                self._report.sending(self._label, self.received, self._expected)
            self._report.doing(f"checking the hash of {self._label}")


class _CappedDownload:
    """A pooch downloader that is the one pooch would choose for the location, with its output held to a size."""

    def __init__(self, inner: Any, limit: int, label: str = "", expected: int | None = None) -> None:
        self._inner = inner
        self._limit = limit
        self._label = label
        self._expected = expected

    def __call__(self, url: str, output_file: str | Path, pooch: object, check_only: bool = False) -> Any:
        if check_only:
            return self._inner(url, output_file, pooch, check_only=True)
        with Path(output_file).open("w+b") as stream:
            capped = _Capped(stream, self._limit, self._label, self._expected)
            self._inner(url, capped, pooch)
            capped.finish()
        return None


#: A downloader as pooch calls one: the address, the file to write (an object with ``write``), and pooch.
type Fetcher = Callable[[str, Any, object], None]

#: A ``doi:`` location under DataCite's test prefix: never resolved through doi.org, which does not resolve it.
TEST_DOI = "doi:10.5072/"
#: The names doi.org answers to.
_DOI_HOSTS = frozenset({"doi.org", "www.doi.org", "dx.doi.org"})


def under_test_prefix(location: str) -> bool:
    """Whether ``location`` would ask doi.org for a DOI under DataCite's test prefix, in any spelling of its address:
    its host read, with a port or none, and its path decoded."""
    if location.lower().startswith(TEST_DOI):
        return True
    parts = urlsplit(location)
    return parts.scheme.lower() in ("http", "https") and (parts.hostname or "").lower() in _DOI_HOSTS and \
        unquote(parts.path).lstrip("/").startswith(TEST_DOI.removeprefix("doi:"))


class _KeysHidden(logging.Filter):
    """pooch's own log lines (it logs each address it downloads), every address in them with its key hidden
    (:func:`~mime.assets._reference.redacted`)."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redacted(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redacted(each) if isinstance(each, str) else each for each in record.args)
        return True


_KEYS_HIDDEN = _KeysHidden()


def _download_held_to(location: str, known_hash: str | None, fname: str, directory: Path, limit: int,
                      label: str = "", expected: int | None = None, fetcher: Fetcher | None = None) -> Path:
    """One location through pooch into ``directory``, which pooch fills through a temporary file moved into place only
    if its hash is ``known_hash`` (``sha256:...`` for an asset, the repository's ``md5:...`` for a bundle fetched by DOI,
    ``None`` for none). A ``doi:`` location is resolved by pooch to the repository's file URL. The download stops, and
    raises :class:`_SizeExceeded`, once it has passed ``limit`` bytes. ``fetcher``, where given, is the downloader in
    place of the one pooch would choose for the location."""
    import pooch

    if under_test_prefix(location):
        raise ValueError(f"{location} is under DataCite's test prefix (the Zenodo sandbox's), which doi.org does not "
                         f"resolve, so MIME does not ask doi.org for it")
    # pooch types its path argument as a bare PathLike, which a strict checker reads as unknown.
    library = _held(pooch)
    logger: logging.Logger = library.get_logger()
    if _KEYS_HIDDEN not in logger.filters:
        logger.addFilter(_KEYS_HIDDEN)
    inner = fetcher if fetcher is not None else library.downloaders.choose_downloader(location)
    downloader = _CappedDownload(inner, limit, label or fname, expected)
    arrived = Path(str(library.retrieve(url=location, known_hash=known_hash, fname=fname, path=directory,
                                        downloader=downloader)))
    watching = _watcher.get()
    if watching is not None:
        watching.sent(label or fname)
    return arrived


#: How often a location is asked again after an answer worth retrying (a timeout, a cut, a 429 or a 5xx), and the
#: first wait, doubled at each retry; a ``Retry-After`` the store states is waited instead, up to the cap.
RETRIES = 3
BACKOFF_SECONDS = 0.5
RETRY_AFTER_CAP_SECONDS = 30.0
#: How long an answer, or the next block of one, may take.
TIMEOUT_SECONDS = 60.0
#: How many redirects one location may lead through.
MAX_REDIRECTS = 10
_BLOCK = 1 << 20


class _WrongBytes(Exception):
    """The stream's bytes, hashed as they arrived, are not the reference's."""


class _Unusable(Exception):
    """A location MIME does not ask: no adapter reads its scheme, or a redirect leads to a location MIME refuses."""


#: The hook the resolver asks for a host's credential; the machine's credential store unless a caller sets another.
_credential_hook: ContextVar[CredentialHook] = ContextVar("mime_credential_hook", default=stored_credential)


@contextmanager
def credentials_from(hook: CredentialHook) -> Generator[None]:
    """Inside this block, the resolver asks ``hook`` for a host's credential in place of the machine's store."""
    token = _credential_hook.set(hook)
    try:
        yield
    finally:
        _credential_hook.reset(token)


#: The longest wait a store's answer is read as meaning: a ``Retry-After`` past a day is no wait a busy server means,
#: and is read as none stated.
LONGEST_MEANT_SECONDS = 86_400.0


def wait_seconds(attempt: int, stated: object) -> float:
    """How long to wait before asking again: the wait a store stated (``Retry-After``, or an adapter's
    :class:`Transient`), up to :data:`RETRY_AFTER_CAP_SECONDS`, where it is a number of seconds from zero to
    :data:`LONGEST_MEANT_SECONDS`; otherwise (none stated, or a NaN, an infinity, a negative or an enormous number, or
    no number at all) the default backoff, doubling from :data:`BACKOFF_SECONDS` at each attempt. Always finite, and
    never more than the cap or the default."""
    default = BACKOFF_SECONDS * 2 ** (attempt - 1)
    if isinstance(stated, bool) or not isinstance(stated, int | float):
        return default
    try:
        held = float(stated)
    except OverflowError:
        return default
    if not math.isfinite(held) or not 0 <= held <= LONGEST_MEANT_SECONDS:
        return default
    return min(held, RETRY_AFTER_CAP_SECONDS)


def _wait(attempt: int, error: BaseException) -> None:
    seconds = wait_seconds(attempt, error.retry_after if isinstance(error, Transient) else None)
    downloads_doing(f"waiting {seconds:.1f} s before asking again")
    time.sleep(seconds)


# --- credentials kept out of what is said ---------------------------------------------------------------------------


#: The credentials the resolver has handed an adapter and whose attempt is still under way, in every thread: while any
#: is held, every log record this process makes has them hidden, and every text the resolver says of an attempt.
_handed: list[Token] = []
_handed_lock = threading.Lock()
#: The credentials handed in the attempts of the location in hand (this context's), whose texts the resolver scrubs.
_handed_here: ContextVar[list[Token] | None] = ContextVar("mime_credentials_handed", default=None)
_factory: list[Callable[..., logging.LogRecord]] = []


def hidden(text: str) -> str:
    """``text`` with every credential handed in the attempt in hand hidden, in every text form a token can take
    (``Token.hidden_in``), and every address's access key (:func:`~mime.assets._reference.redacted`): what the
    resolver says of an adapter's error, a redirect's target or a line it logs, whatever the adapter wrote."""
    for token in (*(_handed_here.get() or ()), *_snapshot()):
        text = token.hidden_in(text, KEY_HIDDEN)
    return redacted(text)


def _snapshot() -> tuple[Token, ...]:
    with _handed_lock:
        return tuple(_handed)


def _scrubbed_record(*arguments: Any, **keywords: Any) -> logging.LogRecord:
    """A log record whose message, and the traceback it carries, hold no credential an attempt under way was handed:
    a line an adapter, or a library it wraps, logs is scrubbed where it is made."""
    record = _factory[0](*arguments, **keywords)
    if not _snapshot():
        return record
    try:
        message = record.getMessage()
    except Exception:  # noqa: BLE001 - a record whose arguments do not fit its message is logging's to report
        return record
    record.msg, record.args = hidden(message), ()
    if record.exc_info and not record.exc_text:
        record.exc_text = hidden(logging.Formatter().formatException(record.exc_info))
    return record


def _scrub_log_records() -> None:
    """Make every log record through :func:`_scrubbed_record`, once per process."""
    with _handed_lock:
        if _factory:
            return
        _factory.append(logging.getLogRecordFactory())
        logging.setLogRecordFactory(_scrubbed_record)


@contextmanager
def _handing() -> Generator[list[Token]]:
    """The attempts of one location: the credentials :func:`_request` hands in them are held, and hidden from every
    text said or logged, until the block ends."""
    held: list[Token] = []
    token = _handed_here.set(held)
    try:
        yield held
    finally:
        _handed_here.reset(token)
        with _handed_lock:
            for each in held:
                if each in _handed:
                    _handed.remove(each)


def _hand(credential: Credential | None) -> Credential | None:
    """``credential``, handed to an adapter: held for :func:`hidden` until the attempts of :func:`_handing` end."""
    held = _handed_here.get()
    if credential is not None and held is not None:
        _scrub_log_records()
        held.append(credential.token)
        with _handed_lock:
            _handed.append(credential.token)
    return credential


def _scrubbed(error: BaseException) -> BaseException:
    """``error`` as the resolver says it, its text with every credential handed hidden: MIME's own kinds of adapter
    error kept (a missing file, a refused login, an answer worth retrying), any other an :class:`AdapterError` naming
    its type. Made outside any handler, so that it is chained to nothing that quotes the original."""
    if isinstance(error, _WrongBytes | _SizeExceeded):
        return error
    text = hidden(str(error))
    if isinstance(error, _Unusable):
        return _Unusable(text)
    if isinstance(error, NotFound) or _nothing_there(error):
        return NotFound(text)
    if isinstance(error, LoginRefused):
        return LoginRefused(text)
    if isinstance(error, Transient):
        return Transient(text, error.retry_after)
    if isinstance(error, AdapterError):
        return AdapterError(text)
    return AdapterError(f"{type(error).__name__}: {text}" if text else type(error).__name__)


# --- redirects ------------------------------------------------------------------------------------------------------


def redirect_problem(answering: str, target: str) -> str | None:
    """Why a redirect from ``answering`` to ``target`` is not followed, or ``None``. **A redirect never changes the
    scheme to a local one**: it leads to ``https``; to plain ``http`` only on this machine and only from a location on
    this machine; or, for an adapter's own scheme (not one MIME reads itself), to that same scheme, which is the
    store's own business. A ``file:`` mirror, a ``package:`` file, a ``doi:``, or any other scheme is refused: a remote
    server never chooses what MIME reads on this machine, or which store it asks with a credential."""
    to, held = scheme_of(target), scheme_of(answering)
    if to == "https":
        return None
    if to == "http":
        here = {host_of(answering), host_of(target)} <= LOOPBACK_HOSTS
        return None if here else ("a redirect leads to plain http only on this machine, from a location on this "
                                  "machine")
    if to == held and to not in BUILT_IN_SCHEMES:
        return None
    return (f"a redirect leads to https (or, on this machine, to http), never to a {to}: location: a remote server "
            f"never chooses what MIME reads here")


def _request(ref: AssetRef, adapter: Adapter, location: str, start: int, validator: str | None) -> Request:
    """The request of one attempt: the credential of the location's own host, and only where the adapter needs one."""
    credential = None
    if adapter.needs_credential:
        host = host_of(location)
        held = _credential_hook.get()(host) if host else None
        credential = _hand(held if held is not None and held.host == host else None)
    names = tuple(name for name, _ in _names_a_person_may_use(ref))
    return Request(ref.name, ref.sha256, ref.size, names, start, None, validator if start else None, credential,
                   TIMEOUT_SECONDS, read_locations().ca_bundle)


def _transfer(location: str, ref: AssetRef, target: Path, *, check_hash: bool = True) -> None:
    """Stream the file at ``location`` into ``target``: through its adapter, each redirect's target checked as a
    location is and asked with the credential of its own host alone, the bytes held to the reference's size as they
    arrive and hashed as they arrive (``check_hash``); an answer worth retrying asked again with backoff, and an
    interrupted stream resumed from the bytes it had where the adapter reads a range and the store names the version it
    sent, else started again."""
    current, hops, attempt = location, 0, 0
    written, validator = 0, None
    digest = hashlib.sha256()
    schemes = [scheme for scheme, _ in ref.adapters]
    with target.open("w+b") as out:
        capped = _Capped(out, ref.size, ref.name, ref.size)
        while True:
            adapter = adapter_for(current)
            if adapter is None:
                raise _Unusable(f"no adapter reads the scheme {scheme_of(current)!r} here: install the package the "
                                f"record names for it"
                                + (f" ({dict(ref.adapters)[scheme_of(current)]})" if scheme_of(current) in schemes
                                   and dict(ref.adapters)[scheme_of(current)] else ""))
            resume = written if written and adapter.ranged and validator is not None else 0
            if written and not resume:
                out.seek(0)
                out.truncate()
                written, digest = 0, hashlib.sha256()
                capped.restart()
            try:
                opened = adapter.open(current, _request(ref, adapter, current, resume, validator))
            except Transient as error:
                attempt += 1
                if attempt > RETRIES:
                    raise
                _wait(attempt, error)
                continue
            if isinstance(opened, Redirect):
                hops += 1
                problem = redirect_problem(current, opened.location) or location_problem(opened.location, schemes)
                if hops > MAX_REDIRECTS:
                    raise _Unusable(f"more than {MAX_REDIRECTS} redirects")
                if problem is not None:
                    raise _Unusable(f"it redirects to {hidden(opened.location)!r}, which MIME does not ask: {problem}")
                current = opened.location
                continue
            try:
                if opened.size is not None:
                    capped.stated(opened.size)
                if opened.start != resume:
                    out.seek(0)
                    out.truncate()
                    written, digest = 0, hashlib.sha256()
                    capped.restart()
                while chunk := opened.stream.read(_BLOCK):
                    capped.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                if opened.size is not None and written < opened.size:
                    raise Transient(f"the stream ended at {written} of the {opened.size} bytes the store stated")
            except (Transient, OSError, http.client.HTTPException) as error:
                attempt += 1
                validator = opened.validator
                if attempt > RETRIES:
                    raise Transient(f"the download was cut ({type(error).__name__}) and did not recover") from None
                _wait(attempt, error)
                continue
            finally:
                opened.close()
            break
        capped.finish()
    if check_hash and digest.hexdigest() != ref.sha256:
        raise _WrongBytes()


#: How long ``mime assets sources`` waits for one location's answer.
PROBE_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class Probe:
    """What looking at one location found: whether it answered with the file (``True``), did not (``False``), or was
    not asked (``None``), and why, in words."""

    location: str
    reachable: bool | None
    why: str


def probe(location: str, ref: AssetRef) -> Probe:
    """Look at one location of ``ref`` without downloading it: its adapter asked for the first byte where it reads a
    range (an ``https`` server is sent ``Range: bytes=0-0``), else opened and closed at once; each redirect followed as
    a download follows it, with the credential of its own host alone. Never with ``MIME_OFFLINE`` set."""
    shown = redacted(location)
    schemes = [scheme for scheme, _ in ref.adapters]
    problem = location_problem(location, schemes)
    if problem is not None:
        return Probe(shown, False, f"not asked: {problem}")
    if offline():
        return Probe(shown, None, f"not looked at: {OFFLINE_VARIABLE} is set")
    with _handing():
        return _probed(location, ref, shown, schemes)


def _probed(location: str, ref: AssetRef, shown: str, schemes: Sequence[str]) -> Probe:
    """:func:`probe`'s looking, each text it says with every credential handed hidden (:func:`hidden`)."""
    current = location
    for _ in range(MAX_REDIRECTS + 1):
        adapter = adapter_for(current)
        if adapter is None:
            return Probe(shown, False, f"no adapter reads the scheme {scheme_of(current)!r} here")
        request = replace(_request(ref, adapter, current, 0, None), end=1 if adapter.ranged else None,
                          timeout=PROBE_TIMEOUT_SECONDS)
        failed: str | None = None
        opened: Opened | Redirect | None = None
        try:
            opened = adapter.open(current, request)
        except NotFound as error:
            failed = f"no file there ({error})"
        except LoginRefused as error:
            failed = f"the login was refused ({error})"
        except Exception as error:  # noqa: BLE001 - an adapter, or a library it wraps, may raise any type
            failed = str(error) or type(error).__name__
        if opened is None:  # said here, outside the handler: the adapter's error is chained to nothing said
            return Probe(shown, False, hidden(failed or "the adapter opened nothing"))
        if isinstance(opened, Redirect):
            problem = redirect_problem(current, opened.location) or location_problem(opened.location, schemes)
            if problem is not None:
                return Probe(shown, False, f"it redirects to a location MIME does not ask: {problem}")
            current = opened.location
            continue
        try:
            size = opened.size
        finally:
            opened.close()
        if size is not None and size != ref.size:
            return Probe(shown, False, f"it answers with {size_text(size)}, and the record says {size_text(ref.size)}")
        stated = f", {size_text(size)} as the record says" if size is not None else ", its size not stated"
        via = "" if current == location else f" (through {hidden(current)})"
        return Probe(shown, True, f"answered{stated}{via}; its hash is checked when it is fetched")
    return Probe(shown, False, f"more than {MAX_REDIRECTS} redirects")


def _download(location: str, ref: AssetRef, directory: Path) -> Path:
    """One location of ``ref`` into ``directory``, held to the reference's sha256 and its declared size."""
    target = directory / ref.sha256
    _transfer(location, ref, target)
    watching = _watcher.get()
    if watching is not None:
        watching.sent(ref.name)
    return target


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


def _nothing_there(error: BaseException) -> bool:
    """Whether the server answered that there is no file at the address (404, 410), which a network that works says as
    plainly as one that does not: the address is wrong, or the file was moved."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status in (404, 410)


def _install(ref: AssetRef, arrived: Path, *, replace: bool) -> Path:
    _place(ref, arrived, replace=replace)
    write_record(ref, DOWNLOADED)
    return file_path(ref)


def tried_locations(ref: AssetRef) -> tuple[str, ...]:
    """The locations of ``ref`` in the order they are tried: the locations file's for it (by sha256, by name, then its
    mirrors), then the record's own, primary first; each once."""
    return tuple(dict.fromkeys((*read_locations().extra(ref.name, ref.sha256), *ref.locations)))


def _store(ref: AssetRef, report: Callable[[str], None] | None, *, replace: bool) -> Path:
    """The locations in order, each into a folder of its own; the first whose file is the reference's is moved into
    place and recorded."""
    failures: list[str] = []
    mismatched = oversize = False
    not_there = 0
    schemes = [scheme for scheme, _ in ref.adapters]
    for location in tried_locations(ref):
        problem = location_problem(location, schemes)
        if problem is not None:  # a reference built around the check: the location is never contacted
            failures.append(f"{redacted(location)}: not used, {problem}")
            continue
        if report is not None:
            report(f"{ref.name}: {size_text(ref.size)} from {redacted(location)}")
        incoming = Path(tempfile.mkdtemp(prefix=f"{ref.sha256}{INCOMING_INFIX}", dir=cache_dir()))
        try:
            with _handing():
                failed: BaseException | None = None
                arrived: Path | None = None
                try:
                    arrived = _download(location, ref, incoming)
                except Exception as error:  # noqa: BLE001 - an adapter, or a library it wraps, may raise any type
                    failed = error
                if failed is not None or arrived is None:  # judged here, outside the handler, as the resolver says it
                    error = _scrubbed(failed or AdapterError("the adapter served nothing"))
                    del failed
                    wrong_bytes = isinstance(error, _WrongBytes)
                    too_many = isinstance(error, _SizeExceeded)
                    not_there += isinstance(error, NotFound)
                    mismatched = mismatched or wrong_bytes
                    oversize = oversize or too_many
                    failures.append(hidden(f"{location}: " + (
                        f"sent more than the {size_text(ref.size)} the reference declares, and the download was "
                        f"stopped" if too_many else "other bytes than the reference names" if wrong_bytes
                        else f"refused the login ({error})" if isinstance(error, LoginRefused)
                        else str(error) or type(error).__name__)))
                    continue
            if sha256_of_file(arrived) != ref.sha256:
                mismatched = True
                failures.append(f"{redacted(location)}: stored other bytes than the reference names")
                continue
            return _install(ref, arrived, replace=replace)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
    tried = "; ".join(failures)
    if mismatched:
        raise BuildRefused([Refusal(
            "asset.hash-mismatch", RefusalClass.ENVIRONMENT,
            f"no location served the bytes with sha256 {ref.sha256} for {ref.name}: {tried}",
            "check the reference's sha256 and the locations (`mime assets list`); nothing was stored")])
    if oversize:
        raise BuildRefused([Refusal(
            "asset.size-exceeded", RefusalClass.ENVIRONMENT,
            f"{ref.name} is declared as {size_text(ref.size)}, and no location served that size: {tried}",
            "check the reference's size and the locations (`mime assets list`); nothing was stored")])
    if failures and not_there == len(failures):
        where = "its location" if len(failures) == 1 else "any of its locations"
        raise BuildRefused([Refusal(
            "asset.unreachable", RefusalClass.ENVIRONMENT,
            f"{ref.name} is not at {where}: the server answered, and has no file at that address ({tried})",
            f"the address is wrong, or the file was moved: check the locations in the reference (`mime assets list`), "
            f"or copy the file to {file_path(ref)} and run `{check_command()}`")])
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
    3. **The locations**: this machine's locations file's for the asset first (a lab's mirror), then the record's own,
       primary then backups, each tried only when the one before fails (unreachable, an error, or other bytes). Each
       is opened by its adapter and streamed into a folder of its own, held to the record's size and hashed as it
       arrives; the file is hashed in full again before it is moved into place and recorded.

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


#: What to do about a download :func:`retrieve` stopped at its limit, unless its caller says otherwise.
LIMIT_ACTION = ("check the address, and if the file is that large on purpose raise the limit: "
                "`mime pull --max-bundle-bytes <bytes>`; nothing was stored")


def _download_refused(error: Exception, name: str, shown: str, max_bytes: int,
                      exceeded_action: str = LIMIT_ACTION) -> tuple[BuildRefused, Exception | None]:
    """The refusal of a download that failed with ``error``, saying the address as ``shown`` (its key hidden, where it
    carries one), and the error it is chained to (``None`` for none)."""
    if isinstance(error, _SizeExceeded):
        return BuildRefused([Refusal(
            "asset.size-exceeded", RefusalClass.ENVIRONMENT,
            f"{name} from {shown} is larger than the {size_text(max_bytes)} it may be, and the download was "
            f"stopped there", exceeded_action)]), None
    if isinstance(error, ValueError) and _POOCH_MISMATCH in str(error):
        return BuildRefused([Refusal(
            "asset.hash-mismatch", RefusalClass.ENVIRONMENT,
            f"{name} from {shown} has other bytes than the repository's checksum names",
            "try again with `mime assets fetch <name>`; if it repeats, the record's file is not the one it "
            "lists")]), error
    if _nothing_there(error):
        return BuildRefused([Refusal(
            "asset.unreachable", RefusalClass.ENVIRONMENT,
            f"{name} is not at {shown}: the server answered, and has no file at that address ({redacted(str(error))})",
            "the address is wrong, or the file was moved: check it, then `mime assets fetch <name>` again"
        )]), error
    return BuildRefused([Refusal(
        "asset.unreachable", RefusalClass.ENVIRONMENT,
        f"{name} could not be downloaded from {shown}: {redacted(str(error))}",
        "check the network and the address, then `mime assets fetch <name>` again")]), error


def retrieve(location: str, *, name: str, known_hash: str | None = None, max_bytes: int = RETRIEVE_LIMIT_BYTES,
             exceeded_action: str = LIMIT_ACTION, report: Callable[[str], None] | None = None,
             size: int | None = None, fetcher: Fetcher | None = None) -> AssetRef:
    """Store the file at one ``location`` whose sha256 nobody has said yet (a bundle named by its DOI), and return the
    reference to it that the hash makes: the file is in the cache under that hash, and :func:`locate` finds it.

    ``known_hash`` is the repository's own checksum of the file (``md5:...`` from Zenodo's record), which pooch holds the
    download to; it guards the transfer, and nothing else vouches for the bytes until someone pins their sha256 in a
    later :func:`resolve`. The location must be an allowed form, and ``MIME_OFFLINE`` refuses it. The download stops
    once it has passed ``max_bytes`` (the size a record states, or :data:`RETRIEVE_LIMIT_BYTES`), refused as
    ``asset.size-exceeded`` with ``exceeded_action`` as what to do. ``size``, where the record states one, is what a
    terminal's progress counts towards (:func:`watching_downloads`); it holds the download to nothing. ``fetcher``, where
    given, downloads the location in place of pooch's own downloader (``mime pull --signed-in``: the token goes to the
    service's own host alone), as pooch calls a downloader, and a refusal it raises is raised as it is."""
    shown, keyed = redacted(location), carries_key(location)
    problem = location_problem(location)
    if problem is not None:
        raise BuildRefused([Refusal(
            "asset.reference", RefusalClass.USER, f"{name}: location {shown!r}: {problem}",
            "give an https:// URL or a doi: URL in the reference's locations=(...)")])
    if offline():
        raise BuildRefused([Refusal(
            "asset.offline", RefusalClass.ENVIRONMENT,
            f"{name} is not cached, and {OFFLINE_VARIABLE} is set, so nothing is downloaded",
            f"unset {OFFLINE_VARIABLE} (`unset {OFFLINE_VARIABLE}`), or pass the file you copied here")])
    cache_dir().mkdir(parents=True, exist_ok=True)
    _clear_stale_unhashed()
    incoming = Path(tempfile.mkdtemp(prefix=_UNHASHED, dir=cache_dir()))
    try:
        if report is not None:
            report(f"{name}: from {shown}")
        announce_downloads([(name, size)])
        arrived: Path | None = None
        refused: tuple[BuildRefused, Exception | None] | None = None
        try:
            arrived = _download_held_to(location, known_hash, name, incoming, max_bytes, name, size, fetcher)
        except Exception as error:  # noqa: BLE001 - pooch, requests and the DOI resolvers raise unrelated types
            if isinstance(error, BuildRefused) and fetcher is not None:
                raise
            refused = _download_refused(error, name, shown, max_bytes, exceeded_action)
        if refused is not None:
            # Raised here, outside the handler: for an address that carries a key, the library's error, which quotes the
            # address whole, is chained to nothing, as cause or as context.
            raise refused[0] from (None if keyed else refused[1])
        assert arrived is not None
        ref = AssetRef(name, sha256_of_file(arrived), arrived.stat().st_size, NO_ASSERTION, (location,))
        with exclusive(lock_path(ref)):
            if cheap_problem(ref) is None:
                return ref
            stored, replace = _look_at_what_is_there(ref)
            if not stored:
                _install(ref, arrived, replace=replace)
        return ref
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
