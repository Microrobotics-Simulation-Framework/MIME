"""The records API client: InvenioRDM's, as Zenodo serves it, matched to what the sandbox probe of 2026-11-01 measured.

What the probe found and this codes against:

- **Every request asks for** ``Accept: application/vnd.inveniordm.v1+json``. Without it the same URLs answer in the
  legacy deposit shape (one licence, ``conceptrecid``), so a 2xx answer in any other content type is refused here
  rather than read: a lost header is noticed, never read as a draft with its other licences dropped. Two addresses
  are asked otherwise (measured on the sandbox, 2026-11-02, for the first): a published record's file listing answers
  only ``application/json``, and a file's bytes are asked for as any type.
- **A failure can arrive as a 200.** Every 2xx body is checked for ``errors``: a file transfer that failed is a 200
  whose ``errors`` lists ``File upload transfer failed.``, and a draft saved without a required field is a 200 whose
  ``errors`` name it. The caller decides what each means; none is read as success. The one exception, read only
  where a draft is made or its metadata written (:func:`pending_aside`): a draft whose metadata is written before its
  files are uploaded is saved with ``errors`` naming ``files.enabled`` (measured on the sandbox, 2026-11-02), which says
  only that its files are still to come.
- **No resume.** An interrupted upload deletes the file's entry, ``Content-Range`` is silently ignored (a resumed tail
  is stored as a complete, correctly checksummed, truncated file), and a second ``PUT`` to a completed file is a 200
  with ``errors`` that deletes it. So a file is sent whole, into an entry started for it, and never ``PUT`` twice.
- **Rate limits** are on every answer (``X-RateLimit-Limit``, ``-Remaining``, ``-Reset``), and ``Retry-After`` is on
  ordinary 2xx answers too, counting down to the reset: only a ``429`` (or a ``503``) means wait, and then for its
  ``Retry-After``. Requests are paced (``pause`` seconds apart) to stay well under the limit.
- **A gateway's 502 or 504 can come after the service acted**, so a request that changes the service is never sent
  again on one: it is answered as a lost answer is (:class:`Dropped`), and the caller reads the draft or its files
  again and goes on from what is there. Only a request that changes nothing, or that is the same however often it is
  sent (writing the metadata, a commit), is sent again on a ``5xx``.

The token travels in the ``Authorization`` header alone, built inside the call that sends it, to the service's own
host: a redirect is never followed (``urllib`` would carry the header to wherever it points), a service on this
machine is reached without the environment's proxy (to which plain ``http`` would carry the header in clear), and
whatever stops a request inside the standard library (an error, an interrupt) is caught at the transport and raised
again as a fresh error from a frame that holds no token: the library's own frames hold the request's headers, and
none of them reaches a traceback, so a traceback captured with its locals shows no token. A pull let in to a record that
is not public (:func:`fetched`: the token stored for its service, or a share link's key) travels the same way.
"""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from email.message import Message as HTTPMessage
from typing import IO, BinaryIO, Protocol, cast, override, runtime_checkable
from urllib.parse import quote, urlsplit

from mime._terminal import plain
from mime._transfers import Transfers
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token
from mime.push._owners import field_owner
from mime.push._service import Service, loopback

#: The content type of InvenioRDM's own serialisation, which every request asks for.
RDM_TYPE = "application/vnd.inveniordm.v1+json"
#: A published record's file listing (``GET /records/<id>/files``) answers only this: the sandbox refused the records
#: API's type there with a 406 (the sandbox session, 2026-11-02), though a draft's listing answers in it.
RECORD_FILES_TYPE = "application/json"
#: A file's bytes, whatever their type.
ANY_TYPE = "*/*"
#: Seconds between two requests' starts: about 120 a minute, under the 133 the sandbox's answers state.
PAUSE_SECONDS = 0.5
#: How often a request is sent again after a ``429``, a ``5xx`` or, for one that changes nothing, a dropped connection.
RETRIES = 4
#: The longest ``Retry-After`` or reset push waits for before it stops and says to run again.
LONGEST_WAIT_SECONDS = 300.0
_TIMEOUT_SECONDS = 120.0
_SAFE = frozenset({"GET", "DELETE"})
#: What a gateway answers when it cannot tell whether the service behind it acted.
_GATEWAY = frozenset({502, 503, 504})
#: The answers that redirect, read for their ``Location`` and never followed with the token.
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
#: The longest part of an answer a refusal quotes.
_QUOTED = 400

type Json = dict[str, object]
type Watch = Callable[[str, str, int, Mapping[str, str], bytes], None]
type Owner = Callable[[str], str]


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


@dataclass(frozen=True)
class Answer:
    """What a service answered: the status, the headers (names in lower case) and the body's bytes."""

    status: int
    headers: Mapping[str, str]
    body: bytes = b""

    def json(self) -> object:
        return json.loads(self.body) if self.body.strip() else None


class Dropped(Exception):
    """The connection failed or dropped before an answer was read, or a gateway answered for the service: whether the
    service acted is not known."""


class Gone(Exception):
    """The service answered ``404``: the draft, the record or the file is not there."""


@dataclass
class Errors(Exception):
    """A 2xx answer whose body lists ``errors``: what the service could not do, though it said 2xx."""

    errors: list[object]
    body: Json = field(default_factory=lambda: cast(Json, {}))

    def words(self) -> str:
        found: list[str] = []
        for each in self.errors:
            if isinstance(each, dict):
                held = cast(dict[str, object], each)
                messages = held.get("messages", [])
                said = "; ".join(str(m) for m in cast(list[object], messages)) if isinstance(messages, list) else ""
                found.append(f"{held.get('field', '?')}: {said}")
            else:
                found.append(str(each))
        return _quoted("; ".join(found))


class Transport(Protocol):
    """Sends one request and returns the answer, or raises :class:`Dropped`. ``token`` is put in the
    ``Authorization`` header by the transport itself, so the text exists only inside the call that sends it."""

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer: ...


@runtime_checkable
class Reporting(Protocol):
    """A transport that tells how much of a file it has sent: the real one. A test's transport has no ``on_sent``, and is
    sent what it was always sent."""

    on_sent: Callable[[int, int], None] | None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is answered as what it is, never followed: following it would carry the token to its target."""

    @override
    def redirect_request(self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str,
                         headers: HTTPMessage, newurl: str) -> None:
        return None


class HttpTransport:
    """The real transport: the standard library's HTTP client, over ``https``, a file streamed from disk with its
    length stated (never chunked), redirects not followed. A proxy the environment names is used as ``urllib``
    uses it (for ``https`` the proxy sees only an encrypted tunnel to the service's own host), except for a service
    on this machine, which is reached directly (``direct``). Whatever the library raises is caught here and raised
    again, fresh, from this frame: a lost or cut answer as :class:`Dropped`, an interrupt as the same interrupt."""

    def __init__(self, timeout: float = _TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self.opener = urllib.request.build_opener(NoRedirect())
        self.direct = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        #: Told how much of a file has gone out, and its size, while a file is streamed (a terminal's progress). Nothing
        #: is read from, or changes in, what is sent.
        self.on_sent: Callable[[int, int], None] | None = None

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer:
        failure: str | None = None
        stopped: BaseException | None = None
        opener = self.direct if loopback(url) else self.opener
        try:
            return _exchange(opener, self.timeout, method, url, headers, body, token, self.on_sent)
        except Exception as error:  # noqa: BLE001 - a request that failed, or whose answer was lost or cut short
            failure = type(error).__name__
        except BaseException as error:  # an interrupt: raised again below, by its type alone
            stopped = _fresh(error)
        # Raised outside the handlers, so the error chains to none whose frames held the request's headers.
        if stopped is not None:
            raise stopped
        raise Dropped(failure)


class _Counted:
    """A file as ``http.client`` reads it to send, reporting how much has been read. It reports about every
    :data:`_REPORT_BYTES` and at the end, so that the report costs nothing next to the send."""

    def __init__(self, stream: BinaryIO, total: int, report: Callable[[int, int], None]) -> None:
        self._stream = stream
        self._total = total
        self._report = report
        self._read = 0
        self._told = 0

    def read(self, size: int = -1) -> bytes:
        block = self._stream.read(size)
        self._read += len(block)
        if self._read - self._told >= _REPORT_BYTES or (not block and self._told != self._read):
            self._told = self._read
            self._report(self._read, self._total)
        return block


#: How much of a file is read between two reports of its progress.
_REPORT_BYTES = 256 * 1024


def _exchange(opener: urllib.request.OpenerDirector, timeout: float, method: str, url: str,
              headers: Mapping[str, str], body: bytes | Path | None, token: Token,
              on_sent: Callable[[int, int], None] | None = None) -> Answer:
    """One request and its whole answer, through ``opener``: every error the library raises passes out of here to
    :meth:`HttpTransport.send`, which raises none of them on. ``on_sent`` is told how much of a file body has gone out."""
    stream = body.open("rb") if isinstance(body, Path) else None
    try:
        data: bytes | BinaryIO | _Counted | None = stream if stream is not None else body if isinstance(body, bytes) \
            else None
        if stream is not None and on_sent is not None and isinstance(body, Path):
            on_sent(0, body.stat().st_size)
            data = _Counted(stream, body.stat().st_size, on_sent)
        request = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
        if isinstance(body, Path):
            request.add_header("Content-Length", str(body.stat().st_size))
        request.add_header("Authorization", "Bearer " + token.reveal())
        try:
            with opener.open(request, timeout=timeout) as response:
                return Answer(response.status, {k.lower(): v for k, v in response.headers.items()}, response.read())
        except urllib.error.HTTPError as error:
            return Answer(error.code, {k.lower(): v for k, v in error.headers.items()}, error.read())
    finally:
        if stream is not None:
            stream.close()


# --- a pull let in: signed in, or by a share link's key ---------------------------------------------------------------

#: The records API's parameter a share link's key is sent as: InvenioRDM's secret links read ``token`` from a request's
#: arguments, as the record's page link carries it. **Recalled, not measured: a guess until a sandbox session makes a
#: share link and pulls it.**
LINK_PARAMETER = "token"
#: How much of a file is read at a time, and handed on, while it is downloaded.
_BLOCK = 256 * 1024


@dataclass(frozen=True)
class Access:
    """What lets a pull read a record that is not public: the token ``mime credentials`` stored for its service
    (``token``, in the ``Authorization`` header, as push sends it), or a share link's key (``key``, as the records API's
    :data:`LINK_PARAMETER`). Each is a :class:`~mime.credentials.Token`, masked wherever it is printed."""

    token: Token | None = None
    key: Token | None = None


def _keyed(url: str, key: Token | None) -> str:
    return url if key is None else f"{url}{'&' if '?' in url else '?'}{LINK_PARAMETER}={quote(key.reveal(), safe='')}"


def _let_in(opener: urllib.request.OpenerDirector, timeout: float, url: str, accept: str, access: Access,
            into: Callable[[bytes], object] | None) -> tuple[Answer, BaseException | None]:
    """One GET and its answer, through ``opener``, the token and the key put on the request here and nowhere else: every
    error the library raises passes out of here to :func:`fetched`. A 200's body is handed to ``into`` block by block
    where it is given; an error ``into`` raises is returned, never raised through the library's frames."""
    request = urllib.request.Request(_keyed(url, access.key), method="GET", headers={"Accept": accept})
    if access.token is not None:
        request.add_header("Authorization", "Bearer " + access.token.reveal())
    try:
        with opener.open(request, timeout=timeout) as response:
            headers = {k.lower(): v for k, v in response.headers.items()}
            if into is None or response.status != 200:
                return Answer(response.status, headers, response.read()), None
            while block := response.read(_BLOCK):
                try:
                    into(block)
                except BaseException as error:  # noqa: BLE001 - the caller's own (a size passed, a display's): its to raise
                    return Answer(response.status, headers), error
            return Answer(response.status, headers), None
    except urllib.error.HTTPError as error:
        return Answer(error.code, {k.lower(): v for k, v in error.headers.items()}, error.read()), None


def fetched(service: Service, path: str, access: Access, *, accept: str = RDM_TYPE,
            into: Callable[[bytes], object] | None = None, timeout: float = _TIMEOUT_SECONDS) -> Answer:
    """One GET of ``path`` below ``service``'s API for a pull let in by ``access``: the token or the key travels as push's
    token does, to the service's own host alone (``service`` pairs a real service's credential with its host,
    :meth:`Service.problem`), a redirect read and never followed (:class:`NoRedirect`), a service on this machine
    reached without the environment's proxy, and whatever stops the request inside the standard library raised again,
    fresh, from this frame (a lost or cut answer as :class:`Dropped`, an interrupt as itself), so no traceback holds the
    library's frames. A 200's body is handed to ``into`` block by block, where given, and the answer holds none; an
    error ``into`` raises (a size passed) is raised as it is."""
    problem = service.problem()
    if problem is not None:
        raise BuildRefused([_refuse("pull.source", RefusalClass.USER, problem,
                                    "pull from a service by its name, `mime pull --from zenodo-sandbox <record>`")])
    url = service.api + path
    transport = HttpTransport(timeout)
    opener = transport.direct if loopback(url) else transport.opener
    failure: str | None = None
    stopped: BaseException | None = None
    try:
        answer, raised = _let_in(opener, timeout, url, accept, access, into)
    except Exception as error:  # noqa: BLE001 - a request that failed, or whose answer was lost or cut short
        failure = type(error).__name__
    except BaseException as error:  # an interrupt: raised again below, by its type alone
        stopped = _fresh(error)
    else:
        if raised is not None:
            raise raised
        return answer
    if stopped is not None:
        raise stopped
    raise Dropped(failure)


def public_into(url: str, into: Callable[[bytes], object], timeout: float = _TIMEOUT_SECONDS) -> None:
    """A file where its service stores it (a redirect read from the records API), fetched with nothing of a token or a
    key: over ``https`` (or to this machine, without the proxy), its own redirects followed, since there is nothing they
    could carry. Its body is handed to ``into`` block by block. :class:`Dropped` where it could not be fetched, saying
    nothing of the address, which may hold a signature of the store's."""
    parts = urlsplit(url)
    if not (parts.scheme == "https" and parts.hostname) and not (parts.scheme == "http" and loopback(url)):
        raise Dropped("a redirect to an address that is not https://")
    opener = urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if loopback(url) else []))
    try:
        with opener.open(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
            while block := response.read(_BLOCK):
                into(block)
    except (OSError, ValueError, http.client.HTTPException) as error:
        raise Dropped(type(error).__name__) from None


def _fresh(error: BaseException) -> BaseException:
    """A new exception of ``error``'s type, holding nothing of it (an exit's code alone), with no traceback yet."""
    try:
        return type(error)(*error.args) if isinstance(error, SystemExit) else type(error)()
    except Exception:  # noqa: BLE001 - a type that needs arguments: an interrupt is what stopped it
        return KeyboardInterrupt()


class Client:
    """The records API of one service, with one token: each method one request (or a request and its retries), its
    answer checked as the module's docstring says. ``pause`` spaces the requests (:data:`PAUSE_SECONDS` by
    default); ``sleep`` and ``now`` are the
    clock's, which a test replaces; ``watch`` is shown each answer (method, address, status, headers, body), never
    the request's headers, which hold the token. ``owner`` says who owns a metadata field a refusal names
    (:func:`mime.push._owners.field_owner`; a push gives the one of its ``sharing.yaml``)."""

    def __init__(self, service: Service, token: Token, transport: Transport | None = None, *,
                 pause: float | None = None, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.time, watch: Watch | None = None) -> None:
        problem = service.problem()
        if problem is not None:
            raise BuildRefused([_refuse("push.service", RefusalClass.USER, problem,
                                        "push to a service by its name, `mime push <folder> --to zenodo-sandbox`")])
        self.service = service
        self._token = token
        self.transport: Transport = transport if transport is not None else HttpTransport()
        self.pause = PAUSE_SECONDS if pause is None else pause
        self.sleep = sleep
        self.now = now
        self.watch = watch
        self.owner: Owner = field_owner
        #: Told what goes up and what is under way: a terminal's progress (:class:`mime._transfers.Transfers`). The bytes
        #: of a file are told only with the real transport, which reports them; a transport of a test's is sent what it
        #: was always sent.
        self.progress: Transfers | None = None
        self._uploading: str | None = None
        self._doing = ""
        self._last = -float("inf")
        self.requests = 0

    # --- one request -----------------------------------------------------------------------------------------------

    def _wait(self, seconds: float, why: str) -> None:
        if seconds > LONGEST_WAIT_SECONDS:
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"{self.service.name} asks push to wait {seconds:.0f} s ({why}), longer than push waits",
                "run the same command again later, `mime push <folder> --to <service>`: what was uploaded and "
                "verified is not sent again")])
        if seconds > 0:
            self.doing(f"waiting {seconds:.0f} s, {why}", resume=False)
            self.sleep(seconds)
            self.doing(self._doing)

    def _paced(self) -> None:
        gap = self._last + self.pause - self.now()
        if gap > 0:
            self.doing("waiting between requests" + (f", then {self._doing}" if self._doing else ""), resume=False)
            self.sleep(gap)
            self.doing(self._doing)
        self._last = self.now()

    # --- what a terminal is told ---------------------------------------------------------------------------------

    def doing(self, words: str, *, resume: bool = True) -> None:
        """Tell the progress what is under way. A wait is told too and then the words it interrupted again (``resume``
        false: these are not the words to come back to)."""
        if resume:
            self._doing = words
        if self.progress is not None:
            self.progress.doing(words)

    def announce(self, files: list[tuple[str, int | None]]) -> None:
        """The files about to go up, before the first."""
        if self.progress is not None:
            self.progress.files(files)

    def uploaded(self, key: str) -> None:
        """The file ``key`` is whole and verified."""
        if self.progress is not None:
            self.progress.sent(key)

    def _send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None) -> Answer:
        self._paced()
        self.requests += 1
        answer = self._transported(method, url, headers, body)
        if self.watch is not None:
            self.watch(method, url, answer.status, answer.headers, answer.body)
        return answer

    def _transported(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None) -> Answer:
        transport, progress, key = self.transport, self.progress, self._uploading
        if not (isinstance(transport, Reporting) and isinstance(body, Path) and progress is not None
                and key is not None):
            return self.transport.send(method, url, headers, body, self._token)
        transport.on_sent = lambda sent, total: progress.sending(key, sent, total)
        try:
            return transport.send(method, url, headers, body, self._token)
        finally:
            transport.on_sent = None

    def _after_limits(self, answer: Answer) -> None:
        """A spent window (``X-RateLimit-Remaining: 0``) is waited out to its reset before the next request."""
        if answer.headers.get("x-ratelimit-remaining", "").strip() == "0":
            reset = _number(answer.headers.get("x-ratelimit-reset"))
            if reset is not None:
                self._wait(reset - self.now(), "its rate limit is spent until its reset")

    def call(self, method: str, path: str, *, payload: object = None, content: Path | None = None,
             expect: frozenset[int] = frozenset({200}), retry: bool | None = None, drops: bool = False,
             raw: bool = False, once: bool = False, accept: str = RDM_TYPE) -> Answer:
        """One request to ``path`` below the service's API. Sent again after a ``429``, and a ``503`` that states its
        ``Retry-After`` (the service did not act), each for its ``Retry-After`` or a growing pause; where ``retry``
        (by default, for a request that changes nothing), also after a dropped connection and any ``502``, ``503`` or
        ``504``. Where it is not, a request whose answer is lost (a dropped connection, a gateway's ``5xx``, which can
        come after the service acted) is never sent again: it is raised as :class:`Dropped` where ``drops`` (the
        caller reads the service's state again and goes on from it), and refused otherwise (``push.service``: the
        next run reads the draft first). An answer not in ``expect`` is refused by what it means; a 2xx is checked
        for the records API's own content type, and for ``errors`` (:class:`Errors`), unless ``raw`` (a file's bytes,
        or a redirect ``expect`` names, whose ``Location`` the caller reads and never follows). ``once`` (the publish, the
        one request that cannot be undone) is sent again only after a ``429``, which refuses it unread: a ``503``, whatever
        it states, is a lost answer. ``accept`` is the one content type asked for and taken: the records API's, but for
        the few addresses that answer only another (:data:`RECORD_FILES_TYPE`)."""
        url = self.service.api + path
        headers = {"Accept": accept}
        body: bytes | Path | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
        elif content is not None:
            headers["Content-Type"] = "application/octet-stream"
            body = content
        again = method in _SAFE if retry is None else retry
        status: int | None = None
        for attempt in range(RETRIES + 1):
            try:
                answer = self._send(method, url, headers, body)
            except Dropped as dropped:
                if again and attempt < RETRIES:
                    self._wait(2.0 ** attempt, "its connection dropped")
                    continue
                if drops:
                    raise dropped
                raise self.lost(method, path, "the connection dropped") from None
            self._after_limits(answer)
            status = answer.status
            after = _number(answer.headers.get("retry-after"))
            if status == 429 or (status == 503 and after is not None and not once) or (again and status in _GATEWAY):
                if attempt == RETRIES:
                    break
                self._wait(after if after is not None else 2.0 ** attempt, f"it answered {status}")
                continue
            if status in _GATEWAY:
                if drops:
                    raise Dropped(f"a gateway's {status}")
                raise self.lost(method, path, f"it answered {status}, which a gateway gives when it cannot tell "
                                               f"whether the service acted")
            if raw and status in expect:
                return answer
            return self._checked(method, path, answer, expect, accept)
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"{self.service.name} answered {method} {plain(path)} with {status} {RETRIES + 1} times",
            "run the same command again later, `mime push <folder> --to <service>`: what was uploaded and "
            "verified is not sent again")])

    def lost(self, method: str, path: str, why: str) -> BuildRefused:
        """The refusal of a request whose answer was lost: whether the service acted is not known."""
        return BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"the answer of {self.service.name} to {method} {plain(path)} was lost ({why}), so whether it was done is "
            f"not known, and push does not send it again", "run the same command again, `mime push <folder> --to "
                                                          "<service>`: it reads the draft first, and what was "
                                                          "uploaded and verified is not sent again")])

    def _checked(self, method: str, path: str, answer: Answer, expect: frozenset[int], accept: str) -> Answer:
        status = answer.status
        said = _message(answer)
        if status in expect and 200 <= status < 300:
            if answer.body.strip():
                kind = answer.headers.get("content-type", "").split(";")[0].strip().lower()
                if kind != accept:
                    raise BuildRefused([_refuse(
                        "push.service", RefusalClass.ENVIRONMENT,
                        f"{self.service.name} answered {method} {plain(path)} in {plain(kind or 'no content type')}, "
                        f"not {accept}, the type it was asked for: without that header the same address can answer in "
                        f"the legacy deposit shape, which holds one licence, so push reads nothing from it",
                        "run the same command again, `mime push <folder> --to <service>`; if it says this again, a "
                        "proxy between here and the service drops the Accept header")])
                held = answer.json()
                errors = errors_of(held)
                if errors:
                    raise Errors(errors, cast(Json, held))
            return answer
        if status in (404, 410):
            raise Gone(said)
        if status == 401:
            raise BuildRefused([_refuse(
                "push.token", RefusalClass.ENVIRONMENT,
                f"{self.service.name} does not accept the token (401: {said})",
                f"make a new personal access token on the service's site and store it, `mime credentials set "
                f"{self.service.credential}`")])
        if status == 403:
            raise BuildRefused([_refuse(
                "push.token-scope", RefusalClass.ENVIRONMENT,
                f"{self.service.name} refused {method} {plain(path)} with this token (403: {said}): {_needs(path)}",
                f"make a token with the scopes deposit:write and deposit:actions on the service's site, then `mime "
                f"credentials set {self.service.credential}`")])
        if 300 <= status < 400:
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"{self.service.name} answered {method} {plain(path)} with a redirect ({status}), which push never "
                f"follows: a token goes to the service's own host only",
                "run the same command again later, `mime push <folder> --to <service>`")])
        if 400 <= status < 500:
            try:
                errors = errors_of(answer.json())
            except ValueError:
                errors = []
            raise BuildRefused(owned_refusals(self.service, f"{self.service.name} refused {method} {plain(path)} "
                                                            f"({status})", errors, self.owner, said))
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"{self.service.name} answered {method} {plain(path)} with {status}: {said}",
            "run the same command again later, `mime push <folder> --to <service>`: what was uploaded and verified "
            "is not sent again")])

    # --- the records API -------------------------------------------------------------------------------------------

    def create_draft(self, payload: Json) -> tuple[Json, list[object]]:
        """Make a draft: the draft, and the ``errors`` its answer lists (a draft is saved with them, and a field it
        names is wrong or missing)."""
        try:
            return _draft(self, self.call("POST", "/records", payload=payload, expect=frozenset({201}),
                                          retry=False)), []
        except Errors as answered:
            return _checked_draft(self, answered.body), pending_aside(answered.errors)

    def read_draft(self, record: str) -> tuple[Json, list[object]] | None:
        """The draft ``record`` and the ``errors`` it is held with, or ``None`` when there is no draft."""
        try:
            return _draft(self, self.call("GET", f"/records/{_part(record)}/draft")), []
        except Gone:
            return None
        except Errors as answered:
            return _checked_draft(self, answered.body), answered.errors

    def read_record(self, record: str) -> Json | None:
        """The published record ``record``, or ``None``."""
        try:
            held = self._coded("the record's reading", lambda: self.call("GET", f"/records/{_part(record)}")).json()
        except Gone:
            return None
        return cast(Json, held) if isinstance(held, dict) else None

    def update_draft(self, record: str, payload: Json) -> tuple[Json, list[object]]:
        """Write the draft's metadata: the draft, and the ``errors`` its answer lists (it is saved with them)."""
        try:
            return _draft(self, self._present(f"the draft {record}", lambda: self.call(
                "PUT", f"/records/{_part(record)}/draft", payload=payload, retry=True))), []
        except Errors as answered:
            return _checked_draft(self, answered.body), pending_aside(answered.errors)

    def reserve_doi(self, record: str) -> Json:
        """Reserve the draft's DOI: the draft. :class:`Dropped` where the answer was lost (the caller reads the draft
        again, to see whether it was done)."""
        return _draft(self, self._present(f"the draft {record}", lambda: self.call(
            "POST", f"/records/{_part(record)}/draft/pids/doi", expect=frozenset({200, 201}), retry=False,
            drops=True)))

    def delete_draft(self, record: str) -> bool:
        """Delete the draft ``record``; whether it was there."""
        try:
            self._coded("the draft's deletion", lambda: self.call("DELETE", f"/records/{_part(record)}/draft",
                                                                  expect=frozenset({204})))
        except Gone:
            return False
        return True

    def files(self, record: str) -> dict[str, Json]:
        """The draft's file entries, by key."""
        return _entries(self, self._present(f"the draft {record}", lambda: self._coded(
            "the listing of the draft's files", lambda: self.call("GET", f"/records/{_part(record)}/draft/files"))))

    def start_file(self, record: str, key: str) -> None:
        """Start the entry ``key``. :class:`Dropped` where the answer was lost (the caller reads the files again);
        :class:`Errors` where the answer lists them."""
        self._present(f"the draft {record}", lambda: self.call(
            "POST", f"/records/{_part(record)}/draft/files", payload=[{"key": key}], expect=frozenset({200, 201}),
            retry=False, drops=True))

    def _coded(self, what: str, request: Callable[[], Answer]) -> Answer:
        """``request``'s answer; one that is a 2xx and lists ``errors`` is refused, with a code, never left as an error
        MIME does not recognise."""
        try:
            return request()
        except Errors as answered:
            words = answered.words()
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"{self.service.name} answered {what} as done, and listed errors: {words}",
            "run the same command again, `mime push <folder> --to <service>`; if it says this again, look at the draft "
            "on the service's site")])

    def _present(self, what: str, request: Callable[[], Answer], *, publishing: bool = False) -> Answer:
        """``request``'s answer; a ``404`` where the thing must be there (deleted on the site while push ran) is
        refused, never left as an error MIME does not recognise."""
        try:
            return request()
        except Gone:
            pass
        if publishing:
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"{what} is no longer on {self.service.name} (deleted on the site, or published there, while push ran?)",
                "run the same command again, `mime push <folder> --to <service> --publish`, which reads the record first "
                "and publishes nothing twice")])
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"{what} is no longer on {self.service.name} (deleted on the site while push ran?)",
            "run the same command again, `mime push <folder> --to <service>`, which makes a new draft where the old "
            "one is gone")])

    def send_content(self, record: str, key: str, path: Path) -> Json:
        """The whole file ``path`` into the started entry ``key``: one ``PUT``, never sent again by this call (a second
        ``PUT`` to a completed file deletes it)."""
        self._uploading = key
        try:
            return _entry(self, self.call("PUT", f"/records/{_part(record)}/draft/files/{_part(key)}/content",
                                          content=path, retry=False, drops=True))
        finally:
            self._uploading = None

    def commit_file(self, record: str, key: str) -> Json:
        return _entry(self, self.call("POST", f"/records/{_part(record)}/draft/files/{_part(key)}/commit",
                                      expect=frozenset({200, 201}), retry=True))

    def publish(self, record: str) -> Json:
        """Publish the draft ``record``: the published record. Sent again only after a ``429``, which refuses it unread
        (publishing is the one step that cannot be undone): any other answer lost is :class:`Dropped`, and the caller
        reads the record before it does anything else. A 2xx that lists ``errors`` is :class:`Errors`, for the caller
        to read the record too."""
        return _draft(self, self._present(f"the draft {record}", lambda: self.call(
            "POST", f"/records/{_part(record)}/draft/actions/publish", expect=frozenset({200, 202}), retry=False,
            drops=True, once=True), publishing=True))

    def new_version(self, record: str) -> Json:
        """A new version of the published record ``record``: the draft of it (the concept's open one where there is
        one). Never sent again: an answer lost is :class:`Dropped`; a record that is gone, :class:`Gone`; a 2xx that
        lists ``errors``, :class:`Errors`."""
        return _draft(self, self.call("POST", f"/records/{_part(record)}/versions", expect=frozenset({200, 201}),
                                      retry=False, drops=True))

    def import_files(self, record: str) -> dict[str, Json]:
        """Link every file of the version before into the new version's draft ``record``: its file entries, by key."""
        return _entries(self, self._present(f"the draft {record}", lambda: self._coded(
            "the import of the previous version's files", lambda: self.call(
                "POST", f"/records/{_part(record)}/draft/actions/files-import", expect=frozenset({200, 201}),
                retry=False))))

    def record_files(self, record: str) -> dict[str, Json]:
        """The published record's file entries, by key."""
        return _entries(self, self._present(f"the record {record}", lambda: self._coded(
            "the listing of the record's files", lambda: self.call("GET", f"/records/{_part(record)}/files",
                                                                   accept=RECORD_FILES_TYPE))))

    def download(self, record: str, key: str) -> bytes:
        """The bytes of the published record's file ``key``. Where the service answers with a redirect to where it
        stores the file, the redirect is read, never followed with the token: on the service's own host it is asked
        through this client; elsewhere, with no token at all (a published file is public)."""
        answer = self._present(f"the file {plain(key)} of the record {record}", lambda: self.call(
            "GET", f"/records/{_part(record)}/files/{_part(key)}/content", raw=True,
            expect=frozenset({200, *_REDIRECTS}), accept=ANY_TYPE))
        if answer.status == 200:
            return answer.body
        target = answer.headers.get("location", "")
        own = urlsplit(self.service.api)
        found = urlsplit(target)
        if (found.scheme, found.netloc) == (own.scheme, own.netloc) and found.path.startswith(own.path + "/"):
            path = target[len(f"{own.scheme}://{own.netloc}{own.path}"):]
            return self.call("GET", path, raw=True, accept=ANY_TYPE).body
        return _public_bytes(self.service, target)

    def latest(self, record: str) -> str:
        """The id of the latest published version of ``record``'s concept: the service's redirect read, never
        followed."""
        answer = self._present(f"the record {record}", lambda: self.call(
            "GET", f"/records/{_part(record)}/versions/latest", expect=frozenset({200, 301, 302, 303}), raw=True))
        if answer.status == 200:
            held = answer.json()
            found = cast(dict[str, object], held).get("id") if isinstance(held, dict) else None
        else:
            target = answer.headers.get("location", "")
            matched = re.fullmatch(re.escape(self.service.api) + r"/records/(\d+)", target)
            found = matched.group(1) if matched is not None else None
        if not isinstance(found, (str, int)) or not str(found).isdigit():
            raise _shape(self, "the latest version", "a redirect to a record of its own API, or the record")
        return str(found)

    def delete_file(self, record: str, key: str) -> None:
        try:
            self._coded(f"the deletion of {plain(key)}", lambda: self.call(
                "DELETE", f"/records/{_part(record)}/draft/files/{_part(key)}", expect=frozenset({204})))
        except Gone:
            pass


def _needs(path: str) -> str:
    return ("publishing needs the scope deposit:actions, and a token with deposit:write alone makes drafts" if
            path.endswith("/actions/publish") else "a draft needs the scope deposit:write")


def _public_bytes(service: Service, url: str) -> bytes:
    """A published file where its service stores it, fetched with no token over ``https`` (or to this machine), its
    redirects followed: none of them can carry a token, since none is sent."""
    parts = urlsplit(url)
    if not (parts.scheme == "https" and parts.hostname) and not (parts.scheme == "http" and loopback(url)):
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"{service.name} answered a published file's download with a redirect to {plain(url)!r}, which is not an "
            f"https:// address", "run the same command again later, `mime push <folder> --to <service> --publish`: it "
                                  "checks the record and publishes nothing again")])
    opener = urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if loopback(url) else []))
    try:
        with opener.open(urllib.request.Request(url, method="GET"), timeout=_TIMEOUT_SECONDS) as response:
            return response.read()
    except (OSError, ValueError) as error:
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"a published file of {service.name} could not be downloaded from where it is stored "
            f"({type(error).__name__})",
            "run the same command again later, `mime push <folder> --to <service> --publish`: it checks the record and "
            "publishes nothing again")]) from None


def errors_of(held: object) -> list[object]:
    """The ``errors`` a 2xx answer's body lists: what the service could not do, though it answered 2xx."""
    if isinstance(held, dict):
        errors = cast(dict[str, object], held).get("errors")
        if isinstance(errors, list):
            return cast(list[object], errors)
    return []


def pending_aside(errors: list[object]) -> list[object]:
    """A draft write's ``errors`` but the note that its files are still to come (:func:`files_pending`): read only where
    a draft is made or its metadata written, which push does before every upload; any other answer listing it, the
    publish's among them, is refused as before."""
    return [each for each in errors if not files_pending(each)]


def files_pending(error: object) -> bool:
    """Whether ``error`` is the service's note that the draft holds no file yet (``files.enabled``: "Missing uploaded
    files."), which the sandbox lists on a draft's metadata written before its files are uploaded. A draft is saved
    with it, push uploads and verifies every file next, and the service refuses to publish a draft without its files, so
    a 2xx answer listing it is not a refusal."""
    return isinstance(error, dict) and cast(dict[str, object], error).get("field") == "files.enabled"


def owned_refusals(service: Service, head: str, errors: list[object], owner: Owner, said: str) -> list[Refusal]:
    """A refusal for each field the service names in ``errors``, by who owns it (``owner``): the author's in
    sharing.yaml or the experiment, to correct there; MIME's, its defect, said as one. Errors that name no field, or
    none at all, are MIME's: what the service refused is what MIME built."""
    found: list[Refusal] = []
    for each in errors:
        held = cast(dict[str, object], each) if isinstance(each, dict) else {}
        field_name = str(held.get("field", ""))
        words = Errors([each]).words()
        who = owner(field_name) if field_name else "mime"
        if who == "sharing":
            found.append(_refuse(
                "push.rejected", RefusalClass.USER, f"{head}: {words}",
                f"correct it in sharing.yaml, then `mime push <folder> --to {service.name}`, which writes the draft's "
                f"metadata again"))
        elif who == "experiment":
            found.append(_refuse(
                "push.rejected", RefusalClass.USER, f"{head}: {words}",
                f"correct the licence or terms where the experiment is built, exp.licence(...) or Terms(...), save it "
                f"again, built.save(folder), then `mime push <folder> --to {service.name}`"))
        else:
            found.append(_refuse(
                "push.rejected", RefusalClass.INTERNAL,
                f"{head}, in a field MIME builds, not one you write: {words}. This is MIME's defect, not yours",
                "report it with this message"))
    return found or [_refuse(
        "push.rejected", RefusalClass.INTERNAL,
        f"{head}, naming no field: {said}. What it refused is what MIME built, so this is MIME's defect, not yours",
        "report it with this message")]


def _quoted(text: str) -> str:
    shown = plain(text)
    return shown if len(shown) <= _QUOTED else shown[:_QUOTED] + "..."


def _part(text: str) -> str:
    return quote(text, safe="")


def _number(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _message(answer: Answer) -> str:
    try:
        held = answer.json()
    except ValueError:
        return f"{len(answer.body)} bytes that are not JSON"
    if isinstance(held, dict):
        found = cast(dict[str, object], held)
        message = str(found.get("message", ""))
        errors = found.get("errors")
        if isinstance(errors, list) and errors:
            message += " " + Errors(cast(list[object], errors)).words()
        return _quoted(message.strip()) or "no message"
    return "no message"


def _shape(client: Client, what: str, expected: str) -> BuildRefused:
    return BuildRefused([_refuse(
        "push.service", RefusalClass.ENVIRONMENT,
        f"{client.service.name} answered with {what} in a shape push does not read ({expected} expected): the records "
        f"API may have changed", "run the same command again, `mime push <folder> --to <service>`; if it says this "
                                 "again, report it with `mime push <folder> --to <service> --status`")])


def _draft(client: Client, answer: Answer) -> Json:
    return _checked_draft(client, answer.json())


def _checked_draft(client: Client, held: object) -> Json:
    """A draft as the records API serialises one: its ``id``, its ``parent`` (the concept) with an ``id``, and its
    ``pids``. The legacy shape (``conceptrecid``, no ``parent``) is refused."""
    if not isinstance(held, dict):
        raise _shape(client, "a draft", "an object")
    draft = cast(Json, held)
    parent = draft.get("parent")
    if "conceptrecid" in draft or not isinstance(parent, dict) or not isinstance(draft.get("pids"), dict) or \
            not isinstance(draft.get("id"), (str, int)):
        raise _shape(client, "a draft", "the records API's id, parent and pids, not the legacy deposit's conceptrecid")
    return draft


def _entries(client: Client, answer: Answer) -> dict[str, Json]:
    """A listing's file entries, by key."""
    held = answer.json()
    entries = cast(dict[str, object], held).get("entries") if isinstance(held, dict) else None
    if not isinstance(entries, list):
        raise _shape(client, "a record's files", "an object with a list of entries")
    found: dict[str, Json] = {}
    for each in cast(list[object], entries):
        if not isinstance(each, dict) or not isinstance(cast(dict[str, object], each).get("key"), str):
            raise _shape(client, "a file entry", "an object with its key")
        entry = cast(Json, each)
        found[cast(str, entry["key"])] = entry
    return found


def _entry(client: Client, answer: Answer) -> Json:
    held = answer.json()
    if not isinstance(held, dict) or not isinstance(cast(dict[str, object], held).get("key"), str):
        raise _shape(client, "a file entry", "an object with its key")
    return cast(Json, held)


__all__ = ["ANY_TYPE", "LINK_PARAMETER", "LONGEST_WAIT_SECONDS", "PAUSE_SECONDS", "RDM_TYPE", "RECORD_FILES_TYPE",
           "RETRIES", "Access", "Answer", "Client", "Dropped", "Errors", "Gone", "HttpTransport", "Json", "NoRedirect",
           "Owner", "Transport", "Watch", "errors_of", "fetched", "owned_refusals", "public_into"]
