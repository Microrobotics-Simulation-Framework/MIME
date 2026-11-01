"""The records API client: InvenioRDM's, as Zenodo serves it, matched to what the sandbox probe of 2026-11-01 measured.

What the probe found and this codes against:

- **Every request asks for** ``Accept: application/vnd.inveniordm.v1+json``. Without it the same URLs answer in the
  legacy deposit shape (one licence, ``conceptrecid``), so a 2xx answer in any other content type is refused here
  rather than read: a lost header is noticed, never read as a draft with its other licences dropped.
- **A failure can arrive as a 200.** Every 2xx body is checked for ``errors``: a file transfer that failed is a 200
  whose ``errors`` lists ``File upload transfer failed.``, and a draft saved without a required field is a 200 whose
  ``errors`` name it. The caller decides what each means; none is read as success.
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
none of them reaches a traceback, so a traceback captured with its locals shows no token.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from email.message import Message as HTTPMessage
from typing import IO, BinaryIO, Protocol, cast, override
from urllib.parse import quote

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import Token
from mime.push._owners import field_owner
from mime.push._service import Service, loopback

#: The content type of InvenioRDM's own serialisation, which every request asks for.
RDM_TYPE = "application/vnd.inveniordm.v1+json"
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

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer:
        failure: str | None = None
        stopped: BaseException | None = None
        opener = self.direct if loopback(url) else self.opener
        try:
            return _exchange(opener, self.timeout, method, url, headers, body, token)
        except Exception as error:  # noqa: BLE001 - a request that failed, or whose answer was lost or cut short
            failure = type(error).__name__
        except BaseException as error:  # an interrupt: raised again below, by its type alone
            stopped = _fresh(error)
        # Raised outside the handlers, so the error chains to none whose frames held the request's headers.
        if stopped is not None:
            raise stopped
        raise Dropped(failure)


def _exchange(opener: urllib.request.OpenerDirector, timeout: float, method: str, url: str,
              headers: Mapping[str, str], body: bytes | Path | None, token: Token) -> Answer:
    """One request and its whole answer, through ``opener``: every error the library raises passes out of here to
    :meth:`HttpTransport.send`, which raises none of them on."""
    stream = body.open("rb") if isinstance(body, Path) else None
    try:
        data: bytes | BinaryIO | None = stream if stream is not None else body if isinstance(body, bytes) else None
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
            self.sleep(seconds)

    def _paced(self) -> None:
        gap = self._last + self.pause - self.now()
        if gap > 0:
            self.sleep(gap)
        self._last = self.now()

    def _send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None) -> Answer:
        self._paced()
        self.requests += 1
        answer = self.transport.send(method, url, headers, body, self._token)
        if self.watch is not None:
            self.watch(method, url, answer.status, answer.headers, answer.body)
        return answer

    def _after_limits(self, answer: Answer) -> None:
        """A spent window (``X-RateLimit-Remaining: 0``) is waited out to its reset before the next request."""
        if answer.headers.get("x-ratelimit-remaining", "").strip() == "0":
            reset = _number(answer.headers.get("x-ratelimit-reset"))
            if reset is not None:
                self._wait(reset - self.now(), "its rate limit is spent until its reset")

    def call(self, method: str, path: str, *, payload: object = None, content: Path | None = None,
             expect: frozenset[int] = frozenset({200}), retry: bool | None = None, drops: bool = False,
             raw: bool = False) -> Answer:
        """One request to ``path`` below the service's API. Sent again after a ``429``, and a ``503`` that states its
        ``Retry-After`` (the service did not act), each for its ``Retry-After`` or a growing pause; where ``retry``
        (by default, for a request that changes nothing), also after a dropped connection and any ``502``, ``503`` or
        ``504``. Where it is not, a request whose answer is lost (a dropped connection, a gateway's ``5xx``, which can
        come after the service acted) is never sent again: it is raised as :class:`Dropped` where ``drops`` (the
        caller reads the service's state again and goes on from it), and refused otherwise (``push.service``: the
        next run reads the draft first). An answer not in ``expect`` is refused by what it means; a 2xx is checked
        for the records API's own content type, and for ``errors`` (:class:`Errors`), unless ``raw`` (a file's bytes,
        or a redirect ``expect`` names, whose ``Location`` the caller reads and never follows)."""
        url = self.service.api + path
        headers = {"Accept": RDM_TYPE}
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
            if status == 429 or (status == 503 and after is not None) or (again and status in _GATEWAY):
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
            return self._checked(method, path, answer, expect)
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

    def _checked(self, method: str, path: str, answer: Answer, expect: frozenset[int]) -> Answer:
        status = answer.status
        said = _message(answer)
        if status in expect and 200 <= status < 300:
            if answer.body.strip():
                kind = answer.headers.get("content-type", "").split(";")[0].strip().lower()
                if kind != RDM_TYPE:
                    raise BuildRefused([_refuse(
                        "push.service", RefusalClass.ENVIRONMENT,
                        f"{self.service.name} answered {method} {plain(path)} in {plain(kind or 'no content type')}, "
                        f"not the records API's {RDM_TYPE}: without that header the same address answers in the "
                        f"legacy deposit shape, which holds one licence, so push reads nothing from it",
                        "run the same command again, `mime push <folder> --to <service>`; if it says this again, a "
                        "proxy between here and the service drops the Accept header")])
                held = answer.json()
                errors = errors_of(held)
                if errors:
                    raise Errors(errors, cast(Json, held))
            return answer
        if status == 404:
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
                f"{self.service.name} refused {method} {plain(path)} with this token (403: {said}): a draft needs "
                f"the scope deposit:write",
                f"make a token with the scope deposit:write on the service's site, then `mime credentials set "
                f"{self.service.credential}`")])
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
            return _checked_draft(self, answered.body), answered.errors

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
            return _checked_draft(self, answered.body), answered.errors

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

    def _present(self, what: str, request: Callable[[], Answer]) -> Answer:
        """``request``'s answer; a ``404`` where the thing must be there (deleted on the site while push ran) is
        refused, never left as an error MIME does not recognise."""
        try:
            return request()
        except Gone:
            pass
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"{what} is no longer on {self.service.name} (deleted on the site while push ran?)",
            "run the same command again, `mime push <folder> --to <service>`, which makes a new draft where the old "
            "one is gone")])

    def send_content(self, record: str, key: str, path: Path) -> Json:
        """The whole file ``path`` into the started entry ``key``: one ``PUT``, never sent again by this call (a second
        ``PUT`` to a completed file deletes it)."""
        return _entry(self, self.call("PUT", f"/records/{_part(record)}/draft/files/{_part(key)}/content",
                                      content=path, retry=False, drops=True))

    def commit_file(self, record: str, key: str) -> Json:
        return _entry(self, self.call("POST", f"/records/{_part(record)}/draft/files/{_part(key)}/commit",
                                      expect=frozenset({200, 201}), retry=True))

    def publish(self, record: str) -> Json:
        """Publish the draft ``record``: the published record. Never sent again by this call: an answer lost is
        :class:`Dropped`, and the caller reads the record before it does anything else (publishing is the one step
        that cannot be undone)."""
        return _draft(self, self._present(f"the draft {record}", lambda: self.call(
            "POST", f"/records/{_part(record)}/draft/actions/publish", expect=frozenset({200, 202}), retry=False,
            drops=True)))

    def new_version(self, record: str) -> Json:
        """A new version of the published record ``record``: the draft of it (the concept's open one where there is
        one). Never sent again: an answer lost is :class:`Dropped`."""
        return _draft(self, self._present(f"the record {record}", lambda: self.call(
            "POST", f"/records/{_part(record)}/versions", expect=frozenset({200, 201}), retry=False, drops=True)))

    def import_files(self, record: str) -> dict[str, Json]:
        """Link every file of the version before into the new version's draft ``record``: its file entries, by key."""
        return _entries(self, self._present(f"the draft {record}", lambda: self._coded(
            "the import of the previous version's files", lambda: self.call(
                "POST", f"/records/{_part(record)}/draft/actions/files-import", expect=frozenset({200, 201}),
                retry=False))))

    def record_files(self, record: str) -> dict[str, Json]:
        """The published record's file entries, by key."""
        return _entries(self, self._present(f"the record {record}", lambda: self._coded(
            "the listing of the record's files", lambda: self.call("GET", f"/records/{_part(record)}/files"))))

    def download(self, record: str, key: str) -> bytes:
        """The bytes of the published record's file ``key``."""
        return self._present(f"the file {plain(key)} of the record {record}", lambda: self.call(
            "GET", f"/records/{_part(record)}/files/{_part(key)}/content", raw=True)).body

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


def errors_of(held: object) -> list[object]:
    """The ``errors`` a 2xx answer's body lists: what the service could not do, though it answered 2xx."""
    if isinstance(held, dict):
        errors = cast(dict[str, object], held).get("errors")
        if isinstance(errors, list):
            return cast(list[object], errors)
    return []


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


__all__ = ["LONGEST_WAIT_SECONDS", "PAUSE_SECONDS", "RDM_TYPE", "RETRIES", "Answer", "Client", "Dropped", "Errors",
           "Gone", "HttpTransport", "Json", "NoRedirect", "Owner", "Transport", "Watch", "errors_of", "owned_refusals"]
