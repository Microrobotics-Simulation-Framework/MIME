"""The records API client against the offline fake: what the sandbox probe found, coded against.

Each guard has a control that plants the defect it guards against, and the control must be caught on every run:

- **the Accept header dropped**: the fake answers in the legacy shape, as the sandbox does, and the client refuses it
  (``push.service``) rather than read a draft whose other licences the legacy shape drops;
- **a 200 carrying ``errors`` taken as success**: a draft saved with a field missing is refused (``push.rejected``),
  and with the check planted away the same push reports a draft that is not what was sent;
- **an upload whose md5 differs accepted**: a stored byte changed is caught, the entry cleared and the file sent again;
  every time, and the push stops (``push.checksum``); with the comparison planted away, the corrupt file is accepted;
- **a custom licence sent with an ``id``**: the fake refuses it with the sandbox's 400, and the rights the client sends
  carry a ``LicenseRef-`` licence as a title and a link only;
- ``Retry-After`` is waited for on a ``429`` and never on a 2xx (the sandbox sends it on both); a spent window waits
  for its reset; a redirect is never followed; a token goes only to the service's fixed host, whatever the
  environment names;
- **a request that changes the service sent again after a gateway's 502 or 504**, which can come after the service
  acted: the draft is made once (or the next push says one may be orphaned), a file started once and uploaded once;
- **the token in a traceback's locals through the real transport**: an interrupt while a request waits on a server on
  this machine, captured with its locals, holds no token; with the transport's boundary planted away, the standard
  library's own frame is caught holding it;
- **a token sent in clear to the environment's proxy**: a service on this machine is reached directly; with the
  direct opener planted away, the proxy is caught receiving the header.
"""

from __future__ import annotations

import http.server
import json
import signal
import socket
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from push_cases import client, plan_of, run_until_drafted, with_token
from zenodo_records import FakeRecords, FakeTransport

from mime import push
from mime.build import BuildRefused, RefusalClass
from mime.credentials import Token
from mime.push import _client, _draft
from mime.push._plan import rights_entries
from mime.saved._terms import FolderTerms
from mime.terms import CustomLicence


def codes(action) -> list[str]:  # noqa: ANN001
    with pytest.raises(BuildRefused) as refused:
        action()
    return refused.value.codes()


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    return with_token(monkeypatch)


def a_draft(fake: FakeRecords, tmp_path: Path, files: dict[str, bytes] | None = None) -> push.Drafted:
    plan = plan_of(fake, files or {"a.bin": b"a" * 300, "b.bin": b"b" * 500}, tmp_path / "files")
    return push.send(plan, None, client(fake))


# --- the Accept header ---------------------------------------------------------------------------------------------


def test_every_request_asks_for_the_records_apis_shape(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    a_draft(fake, tmp_path)
    assert len(fake.requests) >= 8 and all(rdm for _, _, rdm, _ in fake.requests), fake.requests


def drop_the_accept_header(monkeypatch: pytest.MonkeyPatch) -> None:
    real = FakeTransport.send

    def dropped(self, method, url, headers, body, token):  # noqa: ANN001, ANN202
        return real(self, method, url, {k: v for k, v in headers.items() if k != "Accept"}, body, token)

    monkeypatch.setattr(FakeTransport, "send", dropped)


def test_control_without_the_accept_header_the_legacy_shape_is_refused_not_read(token: str, tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch):
    drop_the_accept_header(monkeypatch)
    fake = FakeRecords(token)
    with pytest.raises(BuildRefused) as refused:
        a_draft(fake, tmp_path)
    assert refused.value.codes() == ["push.service"]
    assert "legacy deposit shape" in refused.value.refusals[0].message
    assert fake.requests[0][2] is False, "the planted defect did not reach the fake"


def test_a_legacy_draft_in_the_records_apis_content_type_is_refused_by_its_shape(token: str):
    answer = push.Answer(201, {"content-type": push.RDM_TYPE},
                         json.dumps({"id": 1, "conceptrecid": "0", "metadata": {"license": {"id": "mit"}}}).encode())
    with pytest.raises(BuildRefused) as refused:
        _client._draft(client(FakeRecords(token)), answer)
    assert refused.value.codes() == ["push.service"] and "conceptrecid" in refused.value.refusals[0].message


# --- a 200 that carries errors ---------------------------------------------------------------------------------------


def missing_title(plan: push.Plan) -> push.Plan:
    from dataclasses import replace

    metadata = dict(plan.payload["metadata"])  # type: ignore[arg-type]
    metadata.pop("title")
    return replace(plan, payload={**plan.payload, "metadata": metadata})


def test_a_draft_saved_with_errors_in_a_2xx_is_refused_naming_the_field(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    plan = missing_title(plan_of(fake, {"a.bin": b"a" * 10}, tmp_path / "files"))
    with pytest.raises(BuildRefused) as refused:
        push.send(plan, None, client(fake))
    assert refused.value.codes() == ["push.rejected"]
    assert "metadata.title" in refused.value.refusals[0].message
    assert len(fake.drafts()) == 1, "the draft the service kept is the one push records, not a second"


def test_control_a_2xx_with_errors_taken_as_success_is_caught(token: str, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_client, "errors_of", lambda held: [])
    fake = FakeRecords(token)
    plan = missing_title(plan_of(fake, {"a.bin": b"a" * 10}, tmp_path / "files"))
    drafted = push.send(plan, None, client(fake))
    assert "title" not in fake.records[drafted.record].body["metadata"], \
        "with the check planted away, push reports a draft the service holds without its title"


def test_a_transfer_that_fails_as_a_200_is_started_again_and_verified(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("transfer-failed", lambda method, path: method == "PUT" and path.endswith("/a.bin/content"))
    drafted = a_draft(fake, tmp_path)
    entries = fake.records[drafted.record].files
    assert entries["a.bin"].data == b"a" * 300 and entries["a.bin"].status == "completed"
    assert [path for method, path, _, _ in fake.requests if method == "PUT" and "a.bin" in path] == \
        ["/records/629525/draft/files/a.bin/content"] * 2


# --- the md5 -----------------------------------------------------------------------------------------------------


def test_an_upload_whose_md5_differs_is_cleared_and_sent_again(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("corrupt", lambda method, path: method == "PUT" and path.endswith("/b.bin/content"))
    drafted = a_draft(fake, tmp_path)
    assert fake.records[drafted.record].files["b.bin"].data == b"b" * 500


def test_an_upload_corrupted_every_time_is_refused_and_left_out_of_the_draft(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("corrupt", lambda method, path: method == "PUT" and path.endswith("/b.bin/content"), times=99)
    assert codes(lambda: a_draft(fake, tmp_path)) == ["push.checksum"]
    assert "b.bin" not in fake.drafts()[0].files


def test_control_an_upload_whose_md5_differs_accepted_is_caught(token: str, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_draft, "_entry_verified", lambda entry, file: entry.get("status") == "completed")
    fake = FakeRecords(token)
    fake.plant("corrupt", lambda method, path: method == "PUT" and path.endswith("/b.bin/content"), times=99)
    drafted = a_draft(fake, tmp_path)
    assert fake.records[drafted.record].files["b.bin"].data != b"b" * 500, \
        "with the comparison planted away, the corrupt file is accepted"


# --- licences ------------------------------------------------------------------------------------------------------


CUSTOM = CustomLicence("LicenseRef-Example-1.0", "Example Licence 1.0", "https://example.org/licence-1.0")


def terms_with_custom() -> FolderTerms:
    return FolderTerms((), "CC-BY-4.0", {CUSTOM.id: CUSTOM}, (), 4)


def test_spdx_ids_go_as_the_vocabularys_ids_and_a_custom_licence_as_a_title_and_link_without_an_id():
    found = rights_entries(["CC-BY-4.0", CUSTOM.id, "MIT"], terms_with_custom())
    assert found[0] == {"id": "cc-by-4.0"} and found[2] == {"id": "mit"}
    assert "id" not in found[1] and found[1]["title"] == {"en": "Example Licence 1.0"}
    assert found[1]["link"] == "https://example.org/licence-1.0"


def test_the_fake_accepts_the_rights_push_sends_and_keeps_all_three(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    plan = plan_of(fake, {"a.bin": b"a"}, tmp_path / "files")
    rights = rights_entries(["CC-BY-4.0", CUSTOM.id, "MIT"], terms_with_custom())
    from dataclasses import replace

    plan = replace(plan, payload={**plan.payload, "metadata": {**plan.payload["metadata"], "rights": rights}})  # type: ignore[dict-item]
    drafted = push.send(plan, None, client(fake))
    assert len(fake.records[drafted.record].body["metadata"]["rights"]) == 3  # type: ignore[index]


def test_control_a_custom_licence_sent_with_an_id_is_refused_by_the_fake_as_by_the_sandbox(
        token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mime.push import _plan

    real = _plan.rights_entries
    monkeypatch.setattr(_plan, "rights_entries", lambda ids, judged: [
        {**each, "id": CUSTOM.id} if "link" in each else each for each in real(ids, judged)])
    fake = FakeRecords(token)
    plan = plan_of(fake, {"a.bin": b"a"}, tmp_path / "files")
    from dataclasses import replace

    rights = _plan.rights_entries(["CC-BY-4.0", CUSTOM.id], terms_with_custom())
    plan = replace(plan, payload={**plan.payload, "metadata": {**plan.payload["metadata"], "rights": rights}})  # type: ignore[dict-item]
    with pytest.raises(BuildRefused) as refused:
        push.send(plan, None, client(fake))
    assert refused.value.codes() == ["push.rejected"] and "Invalid value LicenseRef-Example-1.0" in \
        refused.value.refusals[0].message


# --- rate limits, redirects, the host --------------------------------------------------------------------------------


def test_a_429_is_waited_for_its_retry_after_and_a_2xx_retry_after_never(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("status", lambda method, path: path == "/records", status=429, headers={"Retry-After": "7"})
    waited: list[float] = []
    held = client(fake)
    held.sleep = waited.append
    seen: list[str] = []
    held.watch = lambda method, url, status, headers, body: seen.append(f"{status} {headers.get('retry-after')}")
    push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / "files"), None, held)
    assert waited == [7.0], "a 429's Retry-After is waited for once, and no 2xx's"
    assert sum(1 for each in seen if each.startswith("2") and each.split()[1] != "None") >= 6, \
        "the fake sends Retry-After on its 2xx answers, as the sandbox does, so the test can tell"


def test_requests_are_spaced_by_the_pause(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    clock = [100.0]
    waited: list[float] = []

    def sleep(seconds: float) -> None:
        waited.append(seconds)
        clock[0] += seconds

    held = client(fake, now=lambda: clock[0])
    held.pause, held.sleep = 0.5, sleep
    push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / "files"), None, held)
    assert held.requests >= 6 and waited == [0.5] * (held.requests - 1)
    assert _client.PAUSE_SECONDS == 0.5 and push.Client(fake.service(), held._token).pause == 0.5  # type: ignore[attr-defined]


def test_a_spent_window_is_waited_out_to_its_reset(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    held = client(fake, now=lambda: 1000.0)
    waited: list[float] = []
    held.sleep = waited.append
    fake.plant("status", lambda method, path: path == "/records", status=429,
               headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1030", "Retry-After": "1"})
    push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / "files"), None, held)
    assert waited[:2] == [30.0, 1.0]


def test_a_wait_longer_than_push_waits_is_refused(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("status", lambda method, path: path == "/records", status=429, headers={"Retry-After": "3600"})
    assert codes(lambda: push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / "files"), None, client(fake))) == \
        ["push.service"]


def test_a_5xx_is_sent_again_and_one_that_stays_is_refused(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("status", lambda method, path: method == "GET", status=503, times=2)
    push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / "a"), None, client(fake))
    fake.plant("status", lambda method, path: method == "GET", status=502, times=99)
    assert codes(lambda: push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / "b"), None, client(fake))) == \
        ["push.service"]


def test_a_redirect_is_never_followed(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("status", lambda method, path: path == "/records", status=307,
               headers={"Location": "https://elsewhere.invalid/api/records"})
    assert codes(lambda: push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / "files"), None, client(fake))) == \
        ["push.service"]
    assert all(path.startswith("/records") for _, path, _, _ in fake.requests)


def test_the_real_transport_follows_no_redirect():
    import urllib.request

    held = push.HttpTransport()
    for opener in (held.opener, held.direct):
        refusing = [each for each in opener.handlers if isinstance(each, urllib.request.HTTPRedirectHandler)]
        assert refusing and all(each.redirect_request(None, None, 307, "", None, "https://x.invalid") is None  # type: ignore[arg-type]
                                for each in refusing)


def test_401_and_403_name_the_token_and_its_scope(token: str, tmp_path: Path):
    for status, code in ((401, "push.token"), (403, "push.token-scope")):
        fake = FakeRecords(token)
        fake.plant("status", lambda method, path: path == "/records", status=status)
        assert codes(lambda: push.send(plan_of(fake, {"a.bin": b"a"}, tmp_path / code), None, client(fake))) == \
            [code]


PLANTED_HOSTS = ("ZENODO_API", "ZENODO_URL", "ZENODO_SANDBOX_URL", "MIME_ZENODO_API", "MIME_ZENODO_URL",
                 "MIME_PUSH_SERVICE", "MIME_PUSH_API", "INVENIO_API", "ZENODO_HOST")


def test_no_variable_moves_a_tokens_host(monkeypatch: pytest.MonkeyPatch):
    """The control: each variable a person might expect to redirect the host is planted, MIME is imported afresh in a
    process of its own, and the services' hosts stay MIME's constants; and a client sends to its service's fixed
    address alone."""
    import subprocess
    import sys

    for name in PLANTED_HOSTS:
        monkeypatch.setenv(name, "https://attacker.invalid/api")
    shown = subprocess.run([sys.executable, "-c", "import mime.push as p; print(p.SERVICES['zenodo'].api, "
                            "p.SERVICES['zenodo-sandbox'].api, p.ZENODO.site, p.ZENODO_SANDBOX.site)"],
                           capture_output=True, text=True, check=True, timeout=300)
    assert shown.stdout.split() == ["https://zenodo.org/api", "https://sandbox.zenodo.org/api", "https://zenodo.org",
                                    "https://sandbox.zenodo.org"]
    sent: list[str] = []

    class Seen:
        def send(self, method, url, headers, body, token):  # noqa: ANN001, ANN201
            sent.append(url)
            raise push.Dropped("stopped here")

    from mime.credentials import Token

    held = push.Client(push.ZENODO_SANDBOX, Token("zenodo-sandbox", "x" * 40), Seen(), pause=0,
                       sleep=lambda seconds: None)
    with pytest.raises(BuildRefused):
        held.read_draft("1")
    assert sent and all(url.startswith("https://sandbox.zenodo.org/api/") for url in sent)


def test_a_service_reached_in_clear_is_refused_before_any_request():
    from mime.credentials import Token

    plain_http = push.Service("zenodo", "http://zenodo.example/api", "http://zenodo.example", "zenodo", True, "x")
    assert codes(lambda: push.Client(plain_http, Token("zenodo", "x" * 40))) == ["push.service"]


def test_a_draft_deleted_on_the_site_while_push_runs_is_refused_not_unrecognised(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    real = fake._route

    def deleting(method, path, rdm, body, kind):  # noqa: ANN001, ANN202
        if method == "PUT" and path.endswith("/content"):
            fake.records.clear()
        return real(method, path, rdm, body, kind)

    fake._route = deleting  # type: ignore[method-assign]
    assert codes(lambda: a_draft(fake, tmp_path)) == ["push.service"]


def test_a_service_pairs_a_real_credential_with_its_own_host_alone():
    """A ``Service`` made in Python cannot send one service's token elsewhere: a real credential goes to its own host,
    or to a stand-in on this machine or at a ``.invalid`` address; a real host takes its own credential alone."""
    token = Token("zenodo", "x" * 40)
    elsewhere = ("https://elsewhere.example/api", "https://elsewhere.example")
    for crossed in (push.Service("x", *elsewhere, "zenodo", True, "x"),
                    push.Service("x", push.ZENODO.api, push.ZENODO.site, "zenodo-sandbox", True, "x"),
                    push.Service("x", push.ZENODO_SANDBOX.api, push.ZENODO_SANDBOX.site, "zenodo", True, "x"),
                    push.Service("x", push.ZENODO.api, push.ZENODO.site, "zenodo", False, "x")):
        assert codes(lambda: push.Client(crossed, token)) == ["push.service"], crossed
    for allowed in (push.ZENODO, push.ZENODO_SANDBOX, FakeRecords("t").service("zenodo", production=True),
                    push.Service("x", "http://127.0.0.1:9/api", "http://127.0.0.1:9", "zenodo", True, "x"),
                    push.Service("x", *elsewhere, "other", False, "x")):
        assert allowed.problem() is None, allowed


def test_the_same_token_held_for_both_services_is_refused_before_a_request(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch):
    from push_cases import a_token

    from mime import credentials

    same = a_token()
    monkeypatch.setenv(credentials.SERVICES["zenodo"], same)
    monkeypatch.setenv(credentials.SERVICES["zenodo-sandbox"], same)
    fake = FakeRecords(same)
    plan = plan_of(fake, {"a.bin": b"a"}, tmp_path / "files")
    with pytest.raises(BuildRefused) as refused:
        _draft._token(plan.service)  # type: ignore[attr-defined]
    assert refused.value.codes() == ["push.token"] and "the same token" in refused.value.refusals[0].message
    monkeypatch.setenv(credentials.SERVICES["zenodo"], a_token())
    assert _draft._token(plan.service).length == 40  # type: ignore[attr-defined]


# --- who owns a refused field, in any answer --------------------------------------------------------------------------


def refused_by(fake: FakeRecords, tmp_path: Path, status: int, body: dict[str, object],
               contributors: int = 0) -> list[tuple[str, RefusalClass, str]]:
    """The refusals of a first push whose making of the draft is answered ``status`` with ``body``, for a sharing.yaml
    naming ``contributors`` contributors."""
    from dataclasses import replace

    from mime.push._sharing import Person

    fake.plant("status", lambda method, path: method == "POST" and path == "/records", status=status, body=body)
    plan = plan_of(fake, {"a.bin": b"a"}, tmp_path / f"files-{status}-{len(fake.faults)}")
    plan = replace(plan, sharing=replace(plan.sharing, contributors=tuple(
        Person("Doe", "Jane", role="DataCurator") for _ in range(contributors))))
    with pytest.raises(BuildRefused) as refused:
        push.send(plan, None, client(fake))
    return [(r.code, r.category, r.action) for r in refused.value.refusals]


def test_a_4xx_naming_fields_mime_builds_or_no_field_is_mimes_defect(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    found = refused_by(fake, tmp_path, 400, {"status": 400, "message": "A validation error occurred.", "errors": [
        {"field": "metadata.resource_type", "messages": ["Invalid value."]},
        {"field": "metadata.publisher", "messages": ["Missing data."]}]})
    assert [(code, category) for code, category, _ in found] == [("push.rejected", RefusalClass.INTERNAL)] * 2
    assert not any("sharing.yaml" in action for _, _, action in found)
    found = refused_by(fake, tmp_path, 400, {"status": 400, "message": "Invalid value bsd-3-clause-clear."})
    assert [(code, category) for code, category, _ in found] == [("push.rejected", RefusalClass.INTERNAL)]


def test_a_4xx_naming_a_field_the_author_writes_is_theirs_to_correct(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    found = refused_by(fake, tmp_path, 400, {"status": 400, "message": "A validation error occurred.", "errors": [
        {"field": "metadata.creators.0.person_or_org.family_name", "messages": ["Missing data."]},
        {"field": "metadata.rights.0.id", "messages": ["Invalid value."]}]})
    assert [(code, category) for code, category, _ in found] == [("push.rejected", RefusalClass.USER)] * 2
    assert "sharing.yaml" in found[0][2] and "exp.licence" in found[1][2]


def test_contributors_mime_adds_and_its_keyword_are_mimes_and_the_authors_are_theirs(token: str, tmp_path: Path):
    """The contributors after sharing.yaml's are the rights holders MIME adds from the files' terms, and the first
    keyword is MIME's own: a refusal of either is MIME's defect, in a 2xx's ``errors`` as in a 4xx's."""
    fake = FakeRecords(token)
    for contributors, field, category in ((0, "metadata.contributors.0.role", RefusalClass.INTERNAL),
                                          (1, "metadata.contributors.1.role", RefusalClass.INTERNAL),
                                          (1, "metadata.contributors.0.role", RefusalClass.USER),
                                          (0, "metadata.subjects.0.subject", RefusalClass.INTERNAL),
                                          (0, "metadata.subjects.1.subject", RefusalClass.USER)):
        found = refused_by(fake, tmp_path, 400, {"status": 400, "message": "A validation error occurred.",
                                                 "errors": [{"field": field, "messages": ["Invalid value."]}]},
                           contributors)
        assert [category_ for _, category_, _ in found] == [category], (contributors, field, found)


# --- a gateway's 502 or 504 after the service acted -------------------------------------------------------------------


def after_acting(fake: FakeRecords, method: str, ends: str, status: int = 504) -> None:
    fake.plant("status-after", lambda m, path: m == method and path.endswith(ends), status=status)


def test_a_504_to_the_drafts_making_is_never_sent_again_and_the_orphan_is_said(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    after_acting(fake, "POST", "/records")
    plan = plan_of(fake, {"a.bin": b"a"}, tmp_path / "files")
    with pytest.raises(BuildRefused) as refused:
        push.send(plan, None, client(fake))
    assert refused.value.codes() == ["push.service"] and "push does not send it again" in \
        refused.value.refusals[0].message
    assert [path for method, path, _, _ in fake.requests if method == "POST"] == ["/records"]
    said: list[str] = []
    drafted, stops = run_until_drafted(plan, fake, said)
    assert stops == 0 and drafted.record == "629527" and len(fake.drafts()) == 2
    assert any("may hold a draft MIME has no record of" in line and "urn:uuid:" in line for line in said)


def test_a_504_to_a_files_upload_after_it_was_stored_uploads_it_once(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    after_acting(fake, "PUT", "/a.bin/content", 502)
    drafted = a_draft(fake, tmp_path)
    assert fake.records[drafted.record].files["a.bin"].data == b"a" * 300
    assert [path for method, path, _, _ in fake.requests if method == "PUT" and "a.bin" in path] == \
        ["/records/629525/draft/files/a.bin/content"]


def test_a_504_to_a_files_start_after_it_was_started_starts_it_once(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    after_acting(fake, "POST", "/draft/files")
    drafted = a_draft(fake, tmp_path)
    assert set(fake.records[drafted.record].files) == {"a.bin", "b.bin"}
    starts = [status for method, path, _, status in fake.requests if method == "POST" and path.endswith("/draft/files")]
    assert len(starts) == 2 and 400 not in starts, starts


def test_a_504_to_the_dois_reservation_reads_the_draft_and_goes_on(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    after_acting(fake, "POST", "/pids/doi")
    drafted = a_draft(fake, tmp_path)
    assert drafted.doi == "10.5072/zenodo.629525"
    assert sum(1 for method, path, _, _ in fake.requests if path.endswith("/pids/doi")) == 1


def test_a_503_with_its_retry_after_is_sent_again_whatever_the_method(token: str, tmp_path: Path):
    fake = FakeRecords(token)
    fake.plant("status", lambda method, path: method == "POST" and path == "/records", status=503,
               headers={"Retry-After": "2"})
    drafted = a_draft(fake, tmp_path)
    assert drafted.record == "629525" and len(fake.drafts()) == 1


def test_control_a_504_resent_whatever_the_method_makes_a_second_draft(token: str, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch):
    """The planted defect, the gateway's answer read as the service's (sent again for every method), is caught making
    two drafts of one push, with no word of the first."""
    real = _client.Client.call

    def resending(self, method, path, **given):  # noqa: ANN001, ANN202
        given["retry"] = True
        return real(self, method, path, **given)

    monkeypatch.setattr(_client.Client, "call", resending)
    fake = FakeRecords(token)
    after_acting(fake, "POST", "/records")
    a_draft(fake, tmp_path)
    assert len(fake.drafts()) == 2, "the planted resend is not seen making a second draft"


# --- errors in a 2xx, wherever they come ------------------------------------------------------------------------------


def with_errors_to(fake: FakeRecords, method: str, ends: str) -> None:
    real = fake._route

    def route(m, path, rdm, body, kind):  # noqa: ANN001, ANN202
        status, content_type, data = real(m, path, rdm, body, kind)
        if m == method and path.endswith(ends) and status < 300 and data:
            held = json.loads(data)
            held["errors"] = [{"field": "entries", "messages": ["planted"]}]
            return status, push.RDM_TYPE, json.dumps(held).encode()
        return status, content_type, data

    fake._route = route  # type: ignore[method-assign]


@pytest.mark.parametrize(("method", "ends"), [("POST", "/draft/files"), ("GET", "/draft/files")])
def test_errors_in_a_2xx_to_a_files_start_or_listing_are_coded(token: str, tmp_path: Path, method: str, ends: str):
    fake = FakeRecords(token)
    with_errors_to(fake, method, ends)
    with pytest.raises(BuildRefused) as refused:
        a_draft(fake, tmp_path)
    assert refused.value.codes() and all(code.startswith("push.") for code in refused.value.codes())


def test_errors_in_a_2xx_to_a_deletion_or_a_records_reading_are_coded(token: str):
    fake = FakeRecords(token)
    held = client(fake)
    for status, call in ((204, lambda: held.delete_draft("1")), (204, lambda: held.delete_file("1", "a.bin")),
                         (200, lambda: held.read_record("1"))):
        def answered(method, url, headers, body, token, status=status):  # noqa: ANN001, ANN202
            return push.Answer(status, {"content-type": push.RDM_TYPE}, b'{"errors": ["planted"]}')

        held.transport.send = answered  # type: ignore[method-assign]
        refused = codes(call)
        assert refused == ["push.service"], (status, refused)


# --- the real transport: no token in a traceback, no proxy for this machine -------------------------------------------


def captured_with_locals(error: BaseException) -> str:
    return "".join(traceback.TracebackException.from_exception(error, capture_locals=True).format())


@contextmanager
def stalling() -> Iterator[tuple[int, threading.Event]]:
    """A server on this machine that reads a request and never answers: its port, and an event set once the request
    is in."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    arrived = threading.Event()
    held: list[socket.socket] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
        except OSError:
            return
        held.append(connection)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = connection.recv(65536)
            if not chunk:
                break
            data += chunk
        arrived.set()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1], arrived
    finally:
        listener.close()
        for each in held:
            each.close()
        thread.join(timeout=5)


def interrupted_while_waiting(token: Token, send: Callable[[str], object]) -> BaseException:
    """The error an interrupt raises while ``send`` waits on the stalling server for its answer: a real SIGALRM,
    delivered once the request is in, raising KeyboardInterrupt in whatever frame waits, as Ctrl-C would."""

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    try:
        with stalling() as (port, arrived):
            def deliver() -> None:
                if arrived.wait(10):
                    signal.pthread_kill(threading.main_thread().ident or 0, signal.SIGALRM)

            threading.Thread(target=deliver, daemon=True).start()
            try:
                send(f"http://127.0.0.1:{port}")
            except BaseException as error:  # noqa: BLE001 - the interrupt is what this returns
                return error
            raise AssertionError("the request returned without being interrupted")
    finally:
        signal.signal(signal.SIGALRM, previous)


unix_signals = pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="POSIX signals")


@unix_signals
def test_an_interrupt_through_the_real_transport_leaves_no_token_in_a_tracebacks_locals():
    held = Token("zenodo-sandbox", "a-real-transport-token-" + "q" * 20)
    error = interrupted_while_waiting(held, lambda base: push.HttpTransport(timeout=20).send(
        "GET", base + "/api/records/1/draft", {"Accept": push.RDM_TYPE}, None, held))
    assert isinstance(error, KeyboardInterrupt), repr(error)
    assert error.__context__ is None and error.__cause__ is None
    assert not held.occurs_in(captured_with_locals(error).encode("utf-8"))


@unix_signals
def test_control_without_the_transports_boundary_the_librarys_frame_is_caught_holding_the_token():
    held = Token("zenodo-sandbox", "a-real-transport-token-" + "q" * 20)
    transport = push.HttpTransport(timeout=20)
    error = interrupted_while_waiting(held, lambda base: _client._exchange(  # type: ignore[attr-defined]
        transport.direct, 20, "GET", base + "/api/records/1/draft", {"Accept": push.RDM_TYPE}, None, held))
    assert isinstance(error, KeyboardInterrupt)
    assert held.occurs_in(captured_with_locals(error).encode("utf-8")), \
        "the planted defect (the library's frames left in the traceback) is not seen"


def test_an_error_answer_cut_short_is_a_lost_answer_with_no_token_in_its_traceback():
    held = Token("zenodo-sandbox", "a-real-transport-token-" + "q" * 20)

    class Cut(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.wfile.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n"
                             b"Content-Length: 1000\r\n\r\n{\"status\": 500, \"mess")
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Cut)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(push.Dropped) as dropped:
            push.HttpTransport(timeout=10).send("GET", f"http://127.0.0.1:{server.server_address[1]}/api/x",
                                                {}, None, held)
    finally:
        server.shutdown()
        server.server_close()
    assert dropped.value.__context__ is None and "IncompleteRead" in str(dropped.value)
    assert not held.occurs_in(captured_with_locals(dropped.value).encode("utf-8"))


@contextmanager
def proxy() -> Iterator[tuple[str, list[bytes]]]:
    """A proxy on this machine that records every request it gets and answers 502."""
    seen: list[bytes] = []

    class Proxy(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append(f"{self.requestline}\r\n{self.headers}".encode())
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def through_the_proxy(monkeypatch: pytest.MonkeyPatch, transport: Callable[[], push.HttpTransport]) -> list[bytes]:
    """What the environment's proxy saw of one request to a service on this machine (one that is not listening)."""
    held = Token("zenodo-sandbox", "a-real-transport-token-" + "q" * 20)
    for name in ("no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    with proxy() as (address, seen):
        monkeypatch.setenv("http_proxy", address)
        monkeypatch.setenv("HTTP_PROXY", address)
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()
        try:
            transport().send("GET", f"http://127.0.0.1:{port}/api/records/1/draft", {}, None, held)
        except push.Dropped:
            pass
    return [each for each in seen if held.occurs_in(each)] or seen


def test_a_service_on_this_machine_is_reached_without_the_environments_proxy(monkeypatch: pytest.MonkeyPatch):
    assert through_the_proxy(monkeypatch, lambda: push.HttpTransport(timeout=5)) == []


def test_control_with_the_direct_opener_planted_away_the_proxy_gets_the_token_in_clear(
        monkeypatch: pytest.MonkeyPatch):
    def proxied() -> push.HttpTransport:
        held = push.HttpTransport(timeout=5)
        held.direct = held.opener
        return held

    seen = through_the_proxy(monkeypatch, proxied)
    held = Token("zenodo-sandbox", "a-real-transport-token-" + "q" * 20)
    assert seen and all(held.occurs_in(each) for each in seen), "the planted proxy's request is not seen"
