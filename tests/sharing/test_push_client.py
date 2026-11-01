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
  environment names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from push_cases import client, plan_of, with_token
from zenodo_records import FakeRecords, FakeTransport

from mime import push
from mime.build import BuildRefused
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

    handler = push.HttpTransport().opener.handlers
    refusing = [each for each in handler if isinstance(each, urllib.request.HTTPRedirectHandler)]
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
    """The control: each variable a person might expect to redirect the host is planted, and the services' hosts and
    the address a client sends to stay MIME's constants."""
    for name in PLANTED_HOSTS:
        monkeypatch.setenv(name, "https://attacker.invalid/api")
    import importlib

    reloaded = importlib.reload(importlib.import_module("mime.push._service"))
    assert reloaded.SERVICES["zenodo"].api == "https://zenodo.org/api"
    assert reloaded.SERVICES["zenodo-sandbox"].api == "https://sandbox.zenodo.org/api"
    sent: list[str] = []

    class Seen:
        def send(self, method, url, headers, body, token):  # noqa: ANN001, ANN201
            sent.append(url)
            raise push.Dropped("stopped here")

    from mime.credentials import Token

    held = push.Client(reloaded.SERVICES["zenodo-sandbox"], Token("zenodo-sandbox", "x" * 40), Seen(), pause=0,
                       sleep=lambda seconds: None)
    with pytest.raises(BuildRefused):
        held.read_draft("1")
    assert sent and all(url.startswith("https://sandbox.zenodo.org/api/") for url in sent)


def test_a_service_reached_in_clear_is_refused_before_any_request():
    from mime.credentials import Token

    plain_http = push.Service("zenodo", "http://zenodo.example/api", "http://zenodo.example", "zenodo", True, "x")
    assert codes(lambda: push.Client(plain_http, Token("zenodo", "x" * 40))) == ["push.service"]
