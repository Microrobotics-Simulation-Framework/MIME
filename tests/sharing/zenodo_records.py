"""An offline fake of Zenodo's records API (InvenioRDM's), the one every push test in the default lanes uses: no test
leaves the machine, and CI never holds an upload token.

Its behaviour is the sandbox's as the probe of 2026-11-01 measured it (``zenodo_probe_answers.json`` holds the shapes
of those answers, and ``test_push_fidelity.py`` replays the probe's requests here and compares them):

- answered in InvenioRDM's shape only to a request that asks for it (``Accept: application/vnd.inveniordm.v1+json``);
  otherwise a draft is answered in the legacy deposit shape (``conceptrecid``, one ``license``), as ``application/json``;
- a draft made with its concept (``parent.id``, the record's id less one) and no DOI; ``POST .../draft/pids/doi``
  reserves ``<prefix>/zenodo.<id>``;
- a file entry started ``pending``; its ``PUT`` completes it (``md5:<hex>`` and its size; ``Content-Range`` ignored);
  ``commit`` idempotent; a ``PUT`` to a completed file a 200 whose ``errors`` say the transfer failed, and the entry
  deleted; a ``PUT`` cut off deletes the entry; a ``PUT`` or ``commit`` to an entry that is not there a 404;
- a ``LicenseRef-`` id in the rights a 400 naming no field; a draft saved without its publication date, title,
  creators or resource type a 2xx whose ``errors`` name the field;
- rate-limit headers on every answer, ``Retry-After`` on 2xx answers too.

**What it cannot show** (the sandbox lane's to check, by hand): anything the probe did not measure. Its guesses, each
marked where it is made: a draft updated without ``pids`` losing its reserved DOI (InvenioRDM's documented
behaviour, not probed; push always sends the draft's ``pids`` back, so either way is safe); a second reservation
answering the same DOI; a file entry started twice refused with a 400; a commit of a pending entry refused; the
answer to a token that is not accepted (401) or lacks a scope (403); the content type of a file answer to a request
without the records API's ``Accept``; and publishing, versions, communities and the vocabularies' contents (only the
SPDX ids ``packaging`` lists, in lower case, are known here).

Faults are planted per request (:class:`Fault`): a dropped connection before or after the service acted, an upload cut
off part way, a status (``429`` with its headers, a ``5xx``), a stored file whose bytes differ from what was sent, a
transfer that fails as a 200 with ``errors``, and a kill of the pushing process (:class:`Killed`, which nothing in push
catches) before or after the service acted.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import re
import threading
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression

from mime.credentials import Token
from mime.push import RDM_TYPE, ZENODO, ZENODO_SANDBOX, Answer, Dropped, Service

FAKE_SITE = "https://zenodo-records.invalid"
FAKE_API = FAKE_SITE + "/api"
LIMIT = 133
_FILE = re.compile(r"/records/(\d+)/draft/files/([^/]+)(/content|/commit)?")


class Killed(BaseException):
    """The pushing process killed at a request: nothing in push catches it, as nothing catches a SIGKILL."""


@dataclass
class Fault:
    """A planted failure at the requests ``matches`` picks (method, path below the API), the ``times`` next of them:
    ``drop-before`` (the connection drops and the service never acts), ``drop-after`` (it acts, then the answer is
    lost), ``cut`` (an upload's body stops part way: the entry is deleted), ``status`` (answered ``status`` with
    ``headers`` and ``body``), ``corrupt`` (a file stored with one byte changed), ``transfer-failed`` (a 200 whose
    ``errors`` say the transfer failed, the entry deleted), ``kill-before`` and ``kill-after``."""

    kind: str
    matches: Callable[[str, str], bool]
    times: int = 1
    status: int = 500
    headers: Mapping[str, str] = field(default_factory=lambda: {})
    body: object = None


@dataclass
class Entry:
    key: str
    data: bytes | None = None
    status: str = "pending"
    checksum: str | None = None


@dataclass
class Record:
    id: str
    parent: str
    body: dict[str, object]
    pids: dict[str, object] = field(default_factory=lambda: {})
    files: dict[str, Entry] = field(default_factory=lambda: {})
    published: bool = False
    revision: int = 1


def in_vocabulary(identifier: str) -> bool:
    """Whether Zenodo's licence vocabulary, as the fake knows it, holds ``identifier``: an SPDX id ``packaging`` lists,
    in lower case (the sandbox's ``cc-by-4.0`` and ``mit``), and never a ``LicenseRef-`` (a 400 on the sandbox)."""
    if identifier != identifier.lower() or identifier.startswith("licenseref-") or " " in identifier:
        return False
    try:
        return str(canonicalize_license_expression(identifier)).lower() == identifier
    except InvalidLicenseExpression:
        return False


#: What the sandbox says of a draft without a field DataCite needs to register its DOI: the publisher's words are the
#: sandbox lane's (2026-11-01), the others the probe's (``publication_date``, measured; the rest assumed alike).
REQUIRED = {"resource_type": "Missing data for required field.", "title": "Missing data for required field.",
            "publication_date": "Missing data for required field.", "creators": "Missing data for required field.",
            "publisher": "Missing publisher field required for DOI registration."}


def required_errors(metadata: Mapping[str, object]) -> list[object]:
    """The ``errors`` a draft is saved with: one for each field DataCite needs that its metadata lacks."""
    return [{"field": f"metadata.{name}", "messages": [message]} for name, message in REQUIRED.items()
            if not metadata.get(name)]


def _json(status: int, body: object, kind: str = "application/json") -> tuple[int, str, bytes]:
    return status, kind, json.dumps(body).encode("utf-8")


def _missing(message: str) -> tuple[int, str, bytes]:
    return _json(404, {"status": 404, "message": message})


class FakeRecords:
    """The fake service: its drafts and published records, every request it got (method, path, whether it asked for
    the records API's shape, status), and the faults planted in it. ``token`` is the one it accepts."""

    def __init__(self, token: str | Token, *, api: str = FAKE_API, site: str = FAKE_SITE, prefix: str = "10.5072",
                 first: int = 629525, real_address: bool = False) -> None:
        real = {service.api for service in (ZENODO, ZENODO_SANDBOX)}
        if (api in real or site in {service.site for service in (ZENODO, ZENODO_SANDBOX)}) and not real_address:
            raise ValueError(f"a fake at a real service's address ({api}): a push to it would be recorded as that "
                             f"service's, so the fake has an address of its own (real_address=True only for a test "
                             f"of the real service's identity, in this process)")
        self._token = token
        self.api = api
        self.site = site
        self.prefix = prefix
        self.next_id = first
        self.records: dict[str, Record] = {}
        self.requests: list[tuple[str, str, bool, int | str]] = []
        self.faults: list[Fault] = []
        self.clock = 1_793_495_499
        self.lock = threading.Lock()

    def service(self, name: str = "zenodo-sandbox", *, production: bool = False, credential: str | None = None
                ) -> Service:
        """A service whose API is this fake's, injected through the Python call. It may take a real service's name
        (its token is then that service's), and it is never that service: push keeps its records under another
        namespace (``mime.push._state.namespace``), so nothing it records is read for the real one."""
        return Service(name, self.api, self.site, credential or name, production, f"the fake of {name}")

    # --- what the tests read ---------------------------------------------------------------------------------------

    def drafts(self) -> list[Record]:
        return [record for record in self.records.values() if not record.published]

    def stored(self) -> list[bytes]:
        """Every file's bytes the service holds, and every member of each that is a zip (a bundle's files)."""
        found: list[bytes] = []
        for record in self.records.values():
            for entry in record.files.values():
                if entry.data is None:
                    continue
                found.append(entry.data)
                if entry.data[:4] == b"PK\x03\x04":
                    with zipfile.ZipFile(io.BytesIO(entry.data)) as archive:
                        found += [archive.read(info) for info in archive.infolist()]
        return found

    def stored_sha256s(self) -> set[str]:
        return {hashlib.sha256(each).hexdigest() for each in self.stored()}

    def publish(self, record: str) -> None:
        """Mark a draft published, as a later MIME's publish would: push's P2a never does."""
        self.records[record].published = True

    def plant(self, kind: str, matches: Callable[[str, str], bool], **given: object) -> Fault:
        fault = Fault(kind, matches, **given)  # type: ignore[arg-type]
        self.faults.append(fault)
        return fault

    # --- one request -----------------------------------------------------------------------------------------------

    def _fault(self, method: str, path: str) -> Fault | None:
        for fault in self.faults:
            if fault.times > 0 and fault.matches(method, path):
                fault.times -= 1
                return fault
        return None

    def handle(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> Answer:
        """One request as the service answers it, faults applied. Raises :class:`Dropped` for a lost answer and
        :class:`Killed` for a killed process."""
        with self.lock:
            return self._handle(method, url, {k.lower(): v for k, v in headers.items()}, body)

    def _handle(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> Answer:
        assert url.startswith(self.api + "/"), f"a request to {url}, outside the fake's API"
        path = urlsplit(url).path.removeprefix(urlsplit(self.api).path)
        rdm = headers.get("accept") == RDM_TYPE
        fault = self._fault(method, path)
        kind = fault.kind if fault is not None else None
        if kind == "kill-before":
            self.requests.append((method, path, rdm, "killed before"))
            raise Killed(f"{method} {path}")
        if kind == "drop-before":
            self.requests.append((method, path, rdm, "dropped before"))
            raise Dropped("ConnectionResetError")
        if kind == "status":
            assert fault is not None
            self.requests.append((method, path, rdm, fault.status))
            return self._answer(fault.status, "application/json",
                                json.dumps(fault.body or {"status": fault.status, "message": "planted"}).encode(),
                                fault.headers)
        if headers.get("authorization") != "Bearer " + (self._token.reveal() if isinstance(self._token, Token)
                                                         else self._token):
            self.requests.append((method, path, rdm, 403))
            return self._answer(*_json(403, {"status": 403, "message": "Permission denied."}))
        if kind == "cut" and method == "PUT" and path.endswith("/content"):
            found = _FILE.fullmatch(path)
            if found is not None and found.group(1) in self.records:
                self.records[found.group(1)].files.pop(unquote(found.group(2)), None)
            self.requests.append((method, path, rdm, "cut"))
            raise Dropped("BrokenPipeError")
        status, content_type, data = self._route(method, path, rdm, body or b"", kind)
        self.requests.append((method, path, rdm, status))
        if kind == "kill-after":
            raise Killed(f"{method} {path}")
        if kind == "drop-after":
            raise Dropped("TimeoutError")
        return self._answer(status, content_type, data)

    def _answer(self, status: int, content_type: str, data: bytes, extra: Mapping[str, str] | None = None) -> Answer:
        self.clock += 1
        reset = (self.clock // 60 + 1) * 60
        headers = {"content-type": content_type, "x-ratelimit-limit": str(LIMIT),
                   "x-ratelimit-remaining": str(LIMIT - 1 - len(self.requests) % 60),
                   "x-ratelimit-reset": str(reset), "retry-after": str(reset - self.clock)}
        headers.update({k.lower(): v for k, v in (extra or {}).items()})
        return Answer(status, headers, data)

    # --- the routes ------------------------------------------------------------------------------------------------

    def _route(self, method: str, path: str, rdm: bool, body: bytes, kind: str | None) -> tuple[int, str, bytes]:
        if path == "/records" and method == "POST":
            return self._create(body, rdm)
        found = re.fullmatch(r"/records/(\d+)(/draft(/pids/doi|/files)?)?", path)
        if found is not None:
            record = self.records.get(found.group(1))
            if record is None or (found.group(2) and record.published) or (not found.group(2) and not
                                                                            record.published):
                return _missing("The persistent identifier does not exist.")
            if not found.group(2):
                return self._draft_answer(record, rdm, status=200)
            if found.group(3) == "/pids/doi" and method == "POST":
                record.pids.setdefault("doi", {"identifier": f"{self.prefix}/zenodo.{record.id}",
                                               "provider": "datacite", "client": "datacite"})  # a second: the same (a guess)
                metadata = record.body.get("metadata")
                return self._draft_answer(record, rdm, status=201, errors=required_errors(
                    metadata if isinstance(metadata, dict) else {}))  # type: ignore[arg-type]
            if found.group(3) == "/files":
                if method == "GET":
                    return self._files_answer(record, rdm, 200)
                if method == "POST":
                    return self._start(record, body, rdm)
            if found.group(3) is None:
                if method == "GET":
                    return self._draft_answer(record, rdm, status=200)
                if method == "PUT":
                    return self._update(record, body, rdm)
                if method == "DELETE":
                    del self.records[record.id]
                    return 204, "text/html; charset=utf-8", b""
        found = _FILE.fullmatch(path)
        if found is not None and found.group(1) in self.records and not self.records[found.group(1)].published:
            return self._file(self.records[found.group(1)], unquote(found.group(2)), found.group(3), method, body,
                              rdm, kind)
        return _missing("The requested URL was not found on the server.")

    def _validated(self, body: bytes) -> tuple[dict[str, object] | None, tuple[int, str, bytes] | None, list[object]]:
        try:
            held = json.loads(body)
        except ValueError:
            return None, _json(400, {"status": 400, "message": "The request body is not JSON."}), []
        metadata = held.get("metadata") if isinstance(held, dict) else None
        if not isinstance(metadata, dict):
            return None, _json(400, {"status": 400, "message": "A validation error occurred.",
                                     "errors": [{"field": "metadata", "messages": ["Missing data."]}]}), []
        for right in metadata.get("rights", []):
            identifier = right.get("id") if isinstance(right, dict) else None
            if identifier is not None and not in_vocabulary(str(identifier)):
                return None, _json(400, {"status": 400, "message": f"Invalid value {identifier}."}), []
        return held, None, required_errors(metadata)

    def _create(self, body: bytes, rdm: bool) -> tuple[int, str, bytes]:
        held, refused, errors = self._validated(body)
        if refused is not None:
            return refused
        assert held is not None
        record = Record(str(self.next_id), str(self.next_id - 1), held)
        self.next_id += 2
        self.records[record.id] = record
        return self._draft_answer(record, rdm, status=201, errors=errors)

    def _update(self, record: Record, body: bytes, rdm: bool) -> tuple[int, str, bytes]:
        held, refused, errors = self._validated(body)
        if refused is not None:
            return refused
        assert held is not None
        record.body = held
        record.revision += 2
        if "pids" not in held:
            record.pids = {}  # a guess: InvenioRDM discards a reserved DOI a draft is updated without (not probed)
        return self._draft_answer(record, rdm, status=200, errors=errors)

    def _start(self, record: Record, body: bytes, rdm: bool) -> tuple[int, str, bytes]:
        keys = [each.get("key") for each in json.loads(body)]
        for key in keys:
            if key in record.files:
                return _json(400, {"status": 400, "message": f"File with key {key} already exists."})  # a guess
        for key in keys:
            record.files[str(key)] = Entry(str(key))
        return self._files_answer(record, rdm, 201)

    def _file(self, record: Record, key: str, action: str | None, method: str, body: bytes, rdm: bool,
              kind: str | None) -> tuple[int, str, bytes]:
        entry = record.files.get(key)
        if entry is None:
            return _missing(f"Record '{record.id}' has no file '{key}'.")
        if action is None:
            if method == "DELETE":
                del record.files[key]
                return 204, "text/html; charset=utf-8", b""
            return self._entry_answer(record, entry, rdm)
        if action == "/content" and method == "PUT":
            if entry.status == "completed" or kind == "transfer-failed":
                del record.files[key]
                return self._entry_answer(record, entry, rdm, errors=["File upload transfer failed."])
            stored = body if kind != "corrupt" else bytes([body[0] ^ 1]) + body[1:] if body else b"x"
            entry.data, entry.status = stored, "completed"
            entry.checksum = "md5:" + hashlib.md5(stored, usedforsecurity=False).hexdigest()
            return self._entry_answer(record, entry, rdm)
        if action == "/commit" and method == "POST":
            if entry.status != "completed":
                return _json(400, {"status": 400, "message": "The file has not been uploaded."})  # a guess
            return self._entry_answer(record, entry, rdm)
        return _missing("The requested URL was not found on the server.")

    # --- the answers' shapes, as the probe recorded them -----------------------------------------------------------

    def _links(self, record: Record) -> dict[str, str]:
        base = f"{self.api}/records/{record.id}"
        return {"self": f"{base}/draft", "self_html": f"{self.site}/uploads/{record.id}",
                "preview_html": f"{self.site}/records/{record.id}?preview=1", "reserve_doi": f"{base}/draft/pids/doi",
                "files": f"{base}/draft/files", "record": base, "publish": f"{base}/draft/actions/publish",
                "versions": f"{base}/versions"}

    def _entry_body(self, record: Record, entry: Entry, errors: list[object] | None = None) -> dict[str, object]:
        base = f"{self.api}/records/{record.id}/draft/files/{entry.key}"
        held: dict[str, object] = {
            "key": entry.key, "mimetype": "application/octet-stream", "metadata": None, "access": {"hidden": False},
            "links": {"self": base, "content": f"{base}/content", "commit": f"{base}/commit"},
            "transfer": {"type": "L"}, "status": entry.status}
        if entry.status == "completed" and entry.data is not None:
            held.update({"checksum": entry.checksum, "size": len(entry.data), "storage_class": "L"})
        if errors:
            held["errors"] = errors
        return held

    def _entry_answer(self, record: Record, entry: Entry, rdm: bool,
                      errors: list[object] | None = None) -> tuple[int, str, bytes]:
        return _json(200, self._entry_body(record, entry, errors), RDM_TYPE if rdm else "application/json")

    def _files_answer(self, record: Record, rdm: bool, status: int) -> tuple[int, str, bytes]:
        return _json(status, {"enabled": True, "links": {"self": f"{self.api}/records/{record.id}/draft/files"},
                              "entries": [self._entry_body(record, entry) for entry in record.files.values()],
                              "default_preview": None, "order": []}, RDM_TYPE if rdm else "application/json")

    def _draft_answer(self, record: Record, rdm: bool, *, status: int,
                      errors: list[object] | None = None) -> tuple[int, str, bytes]:
        metadata = dict(record.body.get("metadata", {}))  # type: ignore[arg-type]
        rights = [dict(each) for each in metadata.get("rights", [])]  # type: ignore[union-attr]
        for each in rights:
            if "id" in each:
                each.setdefault("title", {"en": f"the vocabulary's title of {each['id']}"})
                each.setdefault("props", {"scheme": "spdx", "url": f"https://spdx.org/licenses/{each['id']}"})
        if not rdm:
            legacy: dict[str, object] = {
                "id": int(record.id), "conceptrecid": record.parent, "title": metadata.get("title", ""),
                "metadata": {"title": metadata.get("title", ""), "license": {"id": rights[0]["id"]} if rights and
                             "id" in rights[0] else {}, "access_right": "open", "creators": [],
                             "resource_type": {"type": "dataset"}},
                "files": [], "state": "unsubmitted", "submitted": record.published, "links": self._links(record),
                "status": "published" if record.published else "draft"}
            return _json(status, legacy)
        metadata["rights"] = rights
        files = {entry.key: {"key": entry.key, "checksum": entry.checksum, "size": len(entry.data or b"")}
                 for entry in record.files.values() if entry.status == "completed"}
        held: dict[str, object] = {
            "id": record.id, "links": self._links(record), "revision_id": record.revision,
            "parent": {"id": record.parent, "communities": {}, "pids": {}}, "versions": {"index": 1},
            "is_published": record.published, "is_draft": not record.published, "pids": dict(record.pids),
            "metadata": metadata, "access": record.body.get("access", {}),
            "files": {"enabled": True, "order": [], "count": len(files), "entries": files},
            "status": "published" if record.published else "draft"}
        if errors:
            held["errors"] = errors
        return _json(status, held, RDM_TYPE)


class FakeTransport:
    """The in-process transport to a :class:`FakeRecords`: the request goes to its ``handle`` with the token in the
    ``Authorization`` header, as the real transport sends it."""

    def __init__(self, fake: FakeRecords) -> None:
        self.fake = fake

    def send(self, method: str, url: str, headers: Mapping[str, str], body: bytes | Path | None,
             token: Token) -> Answer:
        data = body.read_bytes() if isinstance(body, Path) else body
        return self.fake.handle(method, url, {**headers, "Authorization": "Bearer " + token.reveal()}, data)


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def _any(self) -> None:
        fake = self.server.fake
        length = int(self.headers.get("Content-Length", "0"))
        fault = next((each for each in fake.faults if each.times > 0 and each.kind == "cut"
                      and each.matches(self.command, urlsplit(self.path).path.removeprefix("/api"))), None)
        if fault is not None:
            self.rfile.read(min(length, 1024))
            try:
                fake.handle(self.command, fake.api + self.path.removeprefix("/api"), dict(self.headers.items()), b"")
            except Dropped:
                pass
            self.close_connection = True
            self.connection.shutdown(2)
            return
        body = self.rfile.read(length) if length else b""
        try:
            answer = fake.handle(self.command, fake.api + self.path.removeprefix("/api"), dict(self.headers.items()),
                                 body)
        except (Dropped, Killed):
            self.close_connection = True
            self.connection.shutdown(2)
            return
        self.send_response(answer.status)
        for name, value in answer.headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(answer.body)))
        self.end_headers()
        self.wfile.write(answer.body)

    do_GET = do_PUT = do_POST = do_DELETE = _any  # noqa: N815

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _Server(http.server.ThreadingHTTPServer):
    fake: FakeRecords


@contextmanager
def served(token: str, **given: object) -> Iterator[FakeRecords]:
    """A :class:`FakeRecords` behind a real HTTP server on this machine, for the push that goes through the real
    transport (the command line's, ``HttpTransport``)."""
    server = _Server(("127.0.0.1", 0), _Handler)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    fake = FakeRecords(token, api=host + "/api", site=host, **given)  # type: ignore[arg-type]
    server.fake = fake
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.005}, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
