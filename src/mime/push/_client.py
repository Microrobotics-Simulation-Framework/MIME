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

The token travels in the ``Authorization`` header alone, built inside the call that sends it, to the service's own
host: a redirect is never followed (``urllib`` would carry the header to wherever it points), and no frame of this
module holds the token's text under a name, so a traceback captured with its locals shows none.
"""

from __future__ import annotations

import http.client
import json
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
from mime.push._service import Service

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

type Json = dict[str, object]
type Watch = Callable[[str, str, int, Mapping[str, str], bytes], None]


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
    """The connection failed or dropped before an answer was read: whether the service acted is not known."""


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
        return plain("; ".join(found))


class Transport(Protocol):
    """Sends one request and returns the answer, or raises :class:`Dropped`. ``token`` is put in the
    ``Authorization`` header by the transport itself, so the text exists only inside the call that sends it."""

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is answered as what it is, never followed: following it would carry the token to its target."""

    @override
    def redirect_request(self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str,
                         headers: HTTPMessage, newurl: str) -> None:
        return None


class HttpTransport:
    """The real transport: the standard library's HTTP client, over ``https``, a file streamed from disk with its
    length stated (never chunked), redirects not followed. A proxy the environment names is used as ``urllib``
    uses it: for ``https`` the proxy sees only an encrypted tunnel to the service's own host."""

    def __init__(self, timeout: float = _TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self.opener = urllib.request.build_opener(_NoRedirect())

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer:
        failure: str | None = None
        stream = body.open("rb") if isinstance(body, Path) else None
        try:
            data: bytes | BinaryIO | None = stream if stream is not None else body if isinstance(body, bytes) else None
            request = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
            if isinstance(body, Path):
                request.add_header("Content-Length", str(body.stat().st_size))
            request.add_header("Authorization", "Bearer " + token.reveal())
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    return Answer(response.status, {k.lower(): v for k, v in response.headers.items()},
                                  response.read())
            except urllib.error.HTTPError as error:
                return Answer(error.code, {k.lower(): v for k, v in error.headers.items()}, error.read())
            except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as error:
                failure = type(error).__name__
        finally:
            if stream is not None:
                stream.close()
        raise Dropped(failure)  # raised outside the handler, so it chains to no error whose frames held the request


class Client:
    """The records API of one service, with one token: each method one request (or a request and its retries), its
    answer checked as the module's docstring says. ``pause`` spaces the requests (:data:`PAUSE_SECONDS` by
    default); ``sleep`` and ``now`` are the
    clock's, which a test replaces; ``watch`` is shown each answer (method, address, status, headers, body), never
    the request's headers, which hold the token."""

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
             expect: frozenset[int] = frozenset({200}), retry: bool | None = None, drops: bool = False) -> Answer:
        """One request to ``path`` below the service's API, sent again after a ``429`` or a ``5xx`` (for its
        ``Retry-After``, or a growing pause), and after a dropped connection when ``retry`` (by default, for a
        request that changes nothing). A connection that stays dropped is refused (``push.service``), or raised as
        :class:`Dropped` where ``drops`` (the caller clears and starts again). An answer not in ``expect`` is
        refused by what it means; a 2xx is checked for the records API's own content type, and for ``errors``
        (:class:`Errors`)."""
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
                status = None
                break
            self._after_limits(answer)
            status = answer.status
            if status == 429 or (status in (502, 503, 504) and attempt < RETRIES):
                if attempt == RETRIES:
                    break
                after = _number(answer.headers.get("retry-after"))
                self._wait(after if after is not None else 2.0 ** attempt, f"it answered {status}")
                continue
            return self._checked(method, path, answer, expect)
        if status is not None:
            raise BuildRefused([_refuse(
                "push.service", RefusalClass.ENVIRONMENT,
                f"{self.service.name} answered {method} {plain(path)} with {status} {RETRIES + 1} times",
                "run the same command again later, `mime push <folder> --to <service>`: what was uploaded and "
                "verified is not sent again")])
        raise BuildRefused([_refuse(
            "push.service", RefusalClass.ENVIRONMENT,
            f"the connection to {self.service.name} dropped during {method} {plain(path)}, so whether it was done is "
            f"not known", "run the same command again, `mime push <folder> --to <service>`: it reads the draft "
                          "first, and what was uploaded and verified is not sent again")])

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
            raise BuildRefused([_refuse(
                "push.rejected", RefusalClass.USER,
                f"{self.service.name} refused {method} {plain(path)} ({status}): {said}",
                f"correct what it names in sharing.yaml or the experiment, then `mime push <folder> --to "
                f"{self.service.name}` again")])
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
            held = self.call("GET", f"/records/{_part(record)}").json()
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
        return _draft(self, self._present(f"the draft {record}", lambda: self.call(
            "POST", f"/records/{_part(record)}/draft/pids/doi", expect=frozenset({200, 201}), retry=False)))

    def delete_draft(self, record: str) -> bool:
        """Delete the draft ``record``; whether it was there."""
        try:
            self.call("DELETE", f"/records/{_part(record)}/draft", expect=frozenset({204}))
        except Gone:
            return False
        return True

    def files(self, record: str) -> dict[str, Json]:
        """The draft's file entries, by key."""
        held = self._present(f"the draft {record}",
                             lambda: self.call("GET", f"/records/{_part(record)}/draft/files")).json()
        entries = cast(dict[str, object], held).get("entries") if isinstance(held, dict) else None
        if not isinstance(entries, list):
            raise _shape(self, "the draft's files", "an object with a list of entries")
        found: dict[str, Json] = {}
        for each in cast(list[object], entries):
            if not isinstance(each, dict) or not isinstance(cast(dict[str, object], each).get("key"), str):
                raise _shape(self, "a file entry", "an object with its key")
            entry = cast(Json, each)
            found[cast(str, entry["key"])] = entry
        return found

    def start_file(self, record: str, key: str) -> None:
        self._present(f"the draft {record}", lambda: self.call(
            "POST", f"/records/{_part(record)}/draft/files", payload=[{"key": key}], expect=frozenset({200, 201}),
            retry=False))

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

    def delete_file(self, record: str, key: str) -> None:
        try:
            self.call("DELETE", f"/records/{_part(record)}/draft/files/{_part(key)}", expect=frozenset({204}))
        except Gone:
            pass


def errors_of(held: object) -> list[object]:
    """The ``errors`` a 2xx answer's body lists: what the service could not do, though it answered 2xx."""
    if isinstance(held, dict):
        errors = cast(dict[str, object], held).get("errors")
        if isinstance(errors, list):
            return cast(list[object], errors)
    return []


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
        return plain(message.strip()[:400]) or "no message"
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


def _entry(client: Client, answer: Answer) -> Json:
    held = answer.json()
    if not isinstance(held, dict) or not isinstance(cast(dict[str, object], held).get("key"), str):
        raise _shape(client, "a file entry", "an object with its key")
    return cast(Json, held)


__all__ = ["LONGEST_WAIT_SECONDS", "PAUSE_SECONDS", "RDM_TYPE", "RETRIES", "Answer", "Client", "Dropped", "Errors",
           "Gone", "HttpTransport", "Json", "Transport", "Watch", "errors_of"]
