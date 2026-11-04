"""Adapters: how the bytes at a location are opened. The library knows how to fetch and check a file, never what or from
where.

An adapter does one thing: it opens a byte stream for a location (with the file's size where the store states it),
under the credential the resolver hands it. Everything else is the resolver's (:mod:`mime.assets._fetch`), so an adapter
written by hand behaves as one that wraps a multi-protocol library: the hash check, the size cap, the cache and the
files placed by hand, the tiers, redirects (each one's target checked as a location is, and a credential never carried
to another host), timeouts, retries with backoff, and resuming an interrupted download where the adapter can read a
byte range.

MIME's own adapters read ``https://`` (and ``http://`` on this machine), ``doi:`` (a DOI's record, resolved to its
file's address), ``package:<source>/<path>`` (bytes inside an installed package that declares a ``mime.assets`` entry
point) and ``file:///<directory>/`` (a mirror: a local or mounted directory, its files placed by sha256 or by name). A
package adds a scheme with a ``mime.asset_adapters`` entry point (its name the scheme, its value ``module:object``, an
:class:`Adapter`); a script adds one with :func:`register_adapter`.
"""

from __future__ import annotations

import http.client
import importlib.metadata
import importlib.resources
import math
import os
import ssl
import stat
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, cast, override, runtime_checkable
from urllib.parse import unquote, urljoin, urlsplit

from mime.credentials import Token

if TYPE_CHECKING:
    from mime.assets._reference import AssetRef

#: The entry-point group a package adds an adapter through: its name the scheme, its value ``module:object``.
ADAPTER_GROUP = "mime.asset_adapters"
#: The entry-point group a package's records come from (``mime.assets._sources``); a ``package:`` location names one.
SOURCE_GROUP = "mime.assets"
#: The source name of MIME's own package data.
MIME_SOURCE = "mime"
#: The hosts of this machine, as :func:`host_of` reads them.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class Credential:
    """What the machine's credential store holds for one host: its token (masked whenever it is printed), bound to
    that host. The resolver hands an adapter the credential of the host it is asked about, and of no other."""

    host: str
    token: Token


@dataclass(frozen=True)
class Request:
    """What the resolver asks of an adapter, once per attempt: the asset (its name, sha256, size, and the names a file
    of it may be placed under, which a mirror looks for), the byte range wanted (``start`` and ``end``, the end
    exclusive, ``None`` to the end of the file), the store's validator of the bytes already read (``If-Range``), the
    credential bound to the location's host (``None`` unless the adapter needs one and the store holds one), the time an
    answer may take, and the certificate bundle an institution's ``https`` server needs, where one is set."""

    name: str
    sha256: str
    size: int | None
    names: tuple[str, ...] = ()
    start: int = 0
    end: int | None = None
    validator: str | None = None
    credential: Credential | None = None
    timeout: float = 60.0
    ca_bundle: Path | None = None


@dataclass
class Opened:
    """A stream an adapter opened: ``stream.read(n)`` returns bytes (empty at its end), ``size`` the whole file's size
    where the store states it, ``start`` the byte of the file the stream begins at (the request's ``start`` where the
    store read the range, ``0`` where it sent the whole file), and ``validator`` what the store says names this
    version of the file (an ``ETag``, or a ``Last-Modified`` date), for a resumed read to ask for the same bytes."""

    stream: IO[bytes]
    size: int | None = None
    start: int = 0
    validator: str | None = None
    #: Closed with the stream (a response's connection).
    closing: list[Callable[[], object]] = field(default_factory=list[Callable[[], object]])

    def close(self) -> None:
        self.stream.close()
        for each in self.closing:
            each()


@dataclass(frozen=True)
class Redirect:
    """The location is answered by another: the resolver checks it as a location, and asks again there."""

    location: str


class AdapterError(Exception):
    """A location that failed. Its message says why, and never holds a credential."""


class NotFound(AdapterError):
    """The store answered, and has no such file there (a 404 or 410, a path that is not there)."""


class LoginRefused(AdapterError):
    """The store wants a credential, or refused the one it was given (a 401 or 403)."""


class Transient(AdapterError):
    """Worth asking again: a timeout, a connection cut, a 429 or a 5xx. ``retry_after`` is the wait the store asked
    for, in seconds, where it said one."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@runtime_checkable
class Adapter(Protocol):
    """How one scheme's locations are opened. ``scheme`` is the location's scheme (the text before its ``:``);
    ``needs_credential`` says whether the resolver should ask the machine's credential store for the location's host;
    ``ranged`` says whether :meth:`open` reads a byte range (``Request.start`` and ``end``), which lets the resolver
    resume an interrupted download and lets ``mime assets sources`` look without downloading. :meth:`open` returns
    the stream, or a :class:`Redirect`; it raises :class:`NotFound`, :class:`LoginRefused`, :class:`Transient`, or
    :class:`AdapterError` for anything else. A stream that is cut raises :class:`Transient` or an ``OSError``."""

    @property
    def scheme(self) -> str: ...

    @property
    def needs_credential(self) -> bool: ...

    @property
    def ranged(self) -> bool: ...

    def open(self, location: str, request: Request) -> Opened | Redirect: ...


@dataclass(frozen=True)
class _Registered:
    adapter: Adapter
    package: str | None


_registered: dict[str, _Registered] = {}
_loaded: list[bool] = []


def register_adapter(adapter: Adapter, *, package: str | None = None) -> None:
    """Use ``adapter`` for its scheme's locations in this process. ``package`` is the distribution that provides it,
    which a saved lock names so that ``mime pull`` lists it; an adapter a script registers has none. A scheme MIME reads
    itself is refused."""
    from mime.assets._reference import BUILT_IN_SCHEMES

    scheme = adapter.scheme
    if scheme in BUILT_IN_SCHEMES:
        raise ValueError(f"the scheme {scheme!r} is MIME's own; an adapter adds a scheme MIME does not read")
    _registered[scheme] = _Registered(adapter, package)


def _load_entry_points() -> None:
    if _loaded:
        return
    _loaded.append(True)
    for point in importlib.metadata.entry_points(group=ADAPTER_GROUP):
        if point.name in _registered:
            continue
        loaded: object = point.load()
        adapter = loaded() if isinstance(loaded, type) else loaded
        if isinstance(adapter, Adapter) and adapter.scheme == point.name:
            dist = point.dist
            _registered[point.name] = _Registered(adapter, dist.metadata["Name"] if dist is not None else None)


def registered_schemes() -> frozenset[str]:
    """The schemes an adapter is registered for beyond MIME's own: by entry point, or by :func:`register_adapter`."""
    _load_entry_points()
    return frozenset(_registered)


def adapter_package(scheme: str) -> str | None:
    """The distribution that provides the adapter of ``scheme``, where one is registered with one."""
    _load_entry_points()
    held = _registered.get(scheme)
    return held.package if held is not None else None


def adapter_for(location: str) -> Adapter | None:
    """The adapter that opens ``location``: MIME's own for its schemes, else the one registered for its scheme."""
    from mime.assets._reference import MIRROR_PREFIX, PACKAGE_PREFIX, scheme_of

    if location.startswith(("https://", "http://")):
        return HTTPS
    if location.startswith("doi:"):
        return DOI
    if location.startswith(PACKAGE_PREFIX):
        return PACKAGE
    if location.startswith(MIRROR_PREFIX):
        return MIRROR
    _load_entry_points()
    held = _registered.get(scheme_of(location))
    return held.adapter if held is not None else None


def with_adapters(ref: AssetRef) -> AssetRef:
    """``ref`` (an ``AssetRef``) naming, for each location of a scheme MIME does not read itself, the package that
    provides its adapter here (an empty name for one a script registered), so that a saved lock says what ``mime
    pull`` should list. A scheme the record already names keeps its package."""
    from dataclasses import replace

    from mime.assets._reference import BUILT_IN_SCHEMES, scheme_of

    named = dict(ref.adapters)
    for location in ref.locations:
        scheme = scheme_of(location)
        if scheme not in BUILT_IN_SCHEMES and scheme not in named:
            named[scheme] = adapter_package(scheme) or ""
    return ref if named == dict(ref.adapters) else replace(ref, adapters=tuple(sorted(named.items())))


def host_of(location: str) -> str:
    """The host a location names, lowercase (empty for one with none, such as ``package:``)."""
    return (urlsplit(location).hostname or "").lower()


# --- the credentials hook -----------------------------------------------------------------------------------------


def stored_credential(host: str) -> Credential | None:
    """The credential the machine's store holds for ``host`` (``mime credentials set host:<host>``), bound to it, or
    ``None``. Never read from a record, a location or the locations file."""
    from mime import credentials

    if not host or not credentials.is_host_service(credentials.host_service(host)):
        return None
    found = credentials.token(credentials.host_service(host))
    return Credential(host, found) if found is not None else None


#: The hook the resolver asks for a host's credential: the machine's store by default.
type CredentialHook = Callable[[str], Credential | None]


# --- https --------------------------------------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is answered as what it is: the resolver checks its target and asks there itself."""

    @override
    def redirect_request(self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: Message,
                         newurl: str) -> None:
        return None


_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_TRANSIENT = frozenset({408, 425, 429, 500, 502, 503, 504})


def tls_context(ca_bundle: Path | None) -> ssl.SSLContext:
    """The certificates an ``https`` location is checked against: the system's, certifi's (which a download through
    pooch used, so a machine whose system store is empty still reaches public hosts), and the bundle an institution's
    server needs, where one is set (``MIME_CA_BUNDLE``, or ``ca_bundle`` in the locations file)."""
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(cafile=certifi.where())
    except (ImportError, OSError):
        pass
    if ca_bundle is not None:
        context.load_verify_locations(cafile=str(ca_bundle))
    return context


def retry_after_seconds(text: str | None) -> float | None:
    """The wait a ``Retry-After`` header asks for, in seconds: a number of seconds, or an HTTP date (the time from
    now until then, no wait where it has passed); ``None`` for anything else, so that a value that is not a finite
    number of seconds that is not negative (``nan``, ``inf``, ``-5``, a date that does not parse) leaves the default
    backoff.
    The resolver bounds whatever it is given (``mime.assets._fetch.wait_seconds``)."""
    held = (text or "").strip()
    if not held:
        return None
    try:
        seconds = float(held)
    except ValueError:
        seconds = None
    except OverflowError:
        return None
    if seconds is None:
        import datetime
        import email.utils

        try:
            when = email.utils.parsedate_to_datetime(held)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if when.tzinfo is None:
            return None
        seconds = (when - datetime.datetime.now(datetime.UTC)).total_seconds()
        seconds = max(seconds, 0.0)
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _retry_after(headers: Message | None) -> float | None:
    if headers is None:
        return None
    return retry_after_seconds(str(headers.get("Retry-After", "") or ""))


def _content_range(text: str | None) -> tuple[int, int | None] | None:
    """A ``Content-Range`` header's first byte and the whole file's size: ``bytes 100-199/1000``."""
    if not text or not text.startswith("bytes "):
        return None
    span, _, total = text.removeprefix("bytes ").partition("/")
    first = span.partition("-")[0]
    try:
        return int(first), (int(total) if total.isdigit() else None)
    except ValueError:
        return None


class HttpsAdapter:
    """``https://`` through the standard library, and ``http://`` to this machine. Redirects are returned to the
    resolver, never followed here. A proxy the environment names (``HTTPS_PROXY``, with ``NO_PROXY``) is used for
    ``https`` as ``urllib`` uses it; this machine is reached directly. A byte range is asked for with ``Range`` and,
    resuming, ``If-Range`` on the validator the first answer gave; a server that ignores them sends the whole file,
    which the resolver starts again from."""

    scheme = "https"
    needs_credential = False
    ranged = True

    def open(self, location: str, request: Request) -> Opened | Redirect:
        local = host_of(location) in LOOPBACK_HOSTS
        handlers: list[urllib.request.BaseHandler] = [_NoRedirect(),
                                                      urllib.request.HTTPSHandler(context=tls_context(request.ca_bundle))]
        if local:
            handlers.insert(0, urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        headers = {"Accept-Encoding": "identity"}
        if request.start or request.end is not None:
            last = "" if request.end is None else str(request.end - 1)
            headers["Range"] = f"bytes={request.start}-{last}"
            if request.validator is not None:
                headers["If-Range"] = request.validator
        asked = urllib.request.Request(location, method="GET", headers=headers)
        try:
            response = opener.open(asked, timeout=request.timeout)
        except urllib.error.HTTPError as error:
            status, held = error.code, error.headers
            error.close()
            if status in _REDIRECTS:
                target = held.get("Location")
                if not target:
                    raise AdapterError(f"the server answered {status} with no Location") from None
                return Redirect(urljoin(location, str(target)))
            if status in (404, 410):
                raise NotFound(f"the server answered {status}: no file at that address") from None
            if status in (401, 403):
                raise LoginRefused(f"the server answered {status}: it wants a credential, or refused it") from None
            if status in _TRANSIENT:
                raise Transient(f"the server answered {status}", _retry_after(held)) from None
            raise AdapterError(f"the server answered {status}") from None
        except (TimeoutError, ConnectionResetError, http.client.RemoteDisconnected) as error:
            raise Transient(f"{type(error).__name__}: {error}") from None
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise AdapterError(f"its certificate is not trusted here ({reason.verify_message}): an institution's "
                                   f"server needs its certificate bundle, MIME_CA_BUNDLE=<the .pem file>") from None
            if isinstance(reason, TimeoutError | ConnectionResetError):
                raise Transient(f"{type(reason).__name__}: {reason}") from None
            raise AdapterError(f"{reason}") from None
        status = response.status
        length = response.headers.get("Content-Length")
        validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
        if status == 206:
            found = _content_range(response.headers.get("Content-Range"))
            start, size = found if found is not None else (0, None)
            return Opened(cast(IO[bytes], response), size, start, validator)
        size = int(length) if length is not None and length.isdigit() else None
        return Opened(cast(IO[bytes], response), size, 0, validator)


# --- doi ----------------------------------------------------------------------------------------------------------


#: How long one answer of the DOI step may take (doi.org's, then the repository's API), as any other location's.
DOI_TIMEOUT_SECONDS = 60.0


def doi_file_url(doi: str, file: str, *, timeout: float = DOI_TIMEOUT_SECONDS) -> str:
    """The address of ``file`` in the record a DOI names, as its repository (Zenodo, figshare, Dataverse) lists it:
    pooch's DOI resolution, which asks doi.org and then the repository's API, each request held to ``timeout``
    seconds (pooch passes it to every ``requests.get`` of the step)."""
    import pooch

    library: Any = pooch
    repository = library.downloaders.doi_to_repository(doi, timeout=timeout)
    return str(repository.download_url(file))


def _doi_error(error: Exception) -> AdapterError:
    """What the DOI step's failure is, in MIME's words: a timeout, a cut, a 429 or a 5xx is worth asking again; a
    404 or 410 is no such record; anything else fails the location."""
    response: object = getattr(error, "response", None)
    status: object = getattr(response, "status_code", None)
    words = f"{type(error).__name__}: {error}"
    kinds = {kind.__name__ for kind in type(error).__mro__}  # requests' Timeout and ConnectionError, by name
    if isinstance(error, TimeoutError | ConnectionError) or kinds & {"Timeout", "ConnectionError"} or \
            status in _TRANSIENT:
        held: object = getattr(response, "headers", None)
        stated: object = cast(Mapping[str, object], held).get("Retry-After") if isinstance(held, Mapping) else None
        wait = retry_after_seconds(str(stated)) if stated is not None else None
        return Transient(f"the DOI step did not answer ({words})", wait)
    if status in (404, 410):
        return NotFound(f"the DOI step found no such record ({words})")
    return AdapterError(words)


class DoiAdapter:
    """``doi:<doi>/<file>``: the DOI's record resolved to its file's address, which the resolver then asks through
    ``https``. A DOI under DataCite's test prefix (the Zenodo sandbox's) is never sent to doi.org, which does not
    resolve it."""

    scheme = "doi"
    needs_credential = False
    ranged = False

    def open(self, location: str, request: Request) -> Opened | Redirect:
        from mime.assets._fetch import under_test_prefix

        if under_test_prefix(location):
            raise AdapterError(f"{location} is under DataCite's test prefix (the Zenodo sandbox's), which doi.org does "
                               f"not resolve, so MIME does not ask doi.org for it")
        doi, _, file = location.removeprefix("doi:").rpartition("/")
        failed: AdapterError | None = None
        try:
            return Redirect(doi_file_url(doi, file, timeout=request.timeout))
        except ValueError as error:
            failed = NotFound(str(error))
        except Exception as error:  # noqa: BLE001 - pooch and requests raise unrelated types for a lost network
            failed = _doi_error(error)
        raise failed


# --- package: -----------------------------------------------------------------------------------------------------


def source_modules() -> Mapping[str, str]:
    """Each source of records by name, with the module whose package data holds them: MIME's own (``mime``, its
    ``mime.data``), then every ``mime.assets`` entry point an installed package declares."""
    found = {MIME_SOURCE: "mime.data"}
    for point in importlib.metadata.entry_points(group=SOURCE_GROUP):
        found.setdefault(point.name, point.value.partition(":")[0])
    return found


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


class PackageAdapter:
    """``package:<source>/<path>``: a file inside an installed package, through ``importlib.resources``. The source
    is a ``mime.assets`` entry point's name (or ``mime``), so a record never makes MIME import a module it names: only
    a package that declared itself a source of records is read. A regular file inside the package only."""

    scheme = "package"
    needs_credential = False
    ranged = True

    def open(self, location: str, request: Request) -> Opened | Redirect:
        source, _, path = location.removeprefix("package:").partition("/")
        module = source_modules().get(source)
        if module is None:
            raise NotFound(f"no installed package declares the source {source!r} (a {SOURCE_GROUP!r} entry point)")
        try:
            root = importlib.resources.files(module)
        except ModuleNotFoundError as error:
            raise NotFound(f"the source {source!r} names the module {module}, which cannot be found: {error}") from None
        held = root.joinpath(*path.split("/"))
        if isinstance(root, Path) and isinstance(held, Path):
            if held.is_symlink() or not held.is_file() or not _inside(held, root):
                raise NotFound(f"the package {module} holds no regular file {path}")
        elif not held.is_file():
            raise NotFound(f"the package {module} holds no file {path}")
        stream = held.open("rb")
        try:
            size: int | None = os.fstat(stream.fileno()).st_size
        except (OSError, ValueError, AttributeError):
            size = None  # a package inside an archive: its file has no descriptor
        if request.start:
            stream.seek(request.start)
        return Opened(stream, size, request.start)


# --- file:// mirrors ----------------------------------------------------------------------------------------------


def mirror_root(location: str) -> Path:
    """The directory a ``file:///<path>/`` location names, as this machine writes a path."""
    return Path(urllib.request.url2pathname(unquote(urlsplit(location).path)))


class MirrorAdapter:
    """``file:///<directory>/``: a mirror, a local or mounted directory (a lab's share), holding files named by their
    sha256 or by a name the asset may be placed under. Only a regular file directly inside the directory is read: a
    link is never followed, and nothing is run."""

    scheme = "file"
    needs_credential = False
    ranged = True

    def open(self, location: str, request: Request) -> Opened | Redirect:
        root = mirror_root(location)
        if not root.is_dir():
            raise NotFound(f"the mirror {root} is not a directory here")
        names = [request.sha256, *request.names]
        for index, name in enumerate(names):
            candidate = root / name
            try:
                seen = os.lstat(candidate)
            except OSError:
                continue
            if not stat.S_ISREG(seen.st_mode):
                continue
            if index and request.size is not None and seen.st_size != request.size:
                continue  # a file of another size under a name another asset may share is not this asset's
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
            try:
                descriptor = os.open(candidate, flags)
            except OSError:
                continue
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                continue
            stream = os.fdopen(descriptor, "rb")
            if request.start:
                stream.seek(request.start)
            return Opened(stream, seen.st_size, request.start)
        raise NotFound(f"the mirror {root} holds no file named {request.sha256} or {' or '.join(request.names) or '-'}")


HTTPS = HttpsAdapter()
DOI = DoiAdapter()
PACKAGE = PackageAdapter()
MIRROR = MirrorAdapter()


def describe_adapters() -> Sequence[tuple[str, str]]:
    """Every scheme a location may have here, with what reads it: MIME's own, then each registered adapter's package."""
    own = [("https", "MIME (and http:// on this machine)"), ("doi", "MIME, through the DOI's repository"),
           ("package", "MIME, inside an installed package that declares a mime.assets entry point"),
           ("file", "MIME, a mirror directory")]
    _load_entry_points()
    return own + [(scheme, held.package or "registered by a script") for scheme, held in sorted(_registered.items())]


__all__ = ["ADAPTER_GROUP", "DOI", "DOI_TIMEOUT_SECONDS", "HTTPS", "LOOPBACK_HOSTS", "MIME_SOURCE", "MIRROR",
           "PACKAGE", "SOURCE_GROUP", "Adapter", "AdapterError", "Credential", "CredentialHook", "DoiAdapter",
           "HttpsAdapter", "LoginRefused", "MirrorAdapter", "NotFound", "Opened", "PackageAdapter", "Redirect",
           "Request", "Transient", "adapter_for", "adapter_package", "describe_adapters", "doi_file_url", "host_of",
           "mirror_root", "register_adapter", "registered_schemes", "retry_after_seconds", "source_modules",
           "stored_credential", "tls_context", "with_adapters"]
