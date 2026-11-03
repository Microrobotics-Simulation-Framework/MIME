"""``mime pull`` of a record that is not public: ``--signed-in`` sends the token ``mime credentials`` stored for the
service, and a record's share link the key it carries. Either travels as push's token does: to the service's own API
host alone, never on a redirect (the file's store is asked with nothing of either), never through a proxy to a service
on this machine, never in a traceback's locals, a refusal, a log, a printed line or a file. Without the flag pull sends
no token, as before, and its refusal names the flag where a token is stored, and how to store one where none is.

**The form a share link's key travels in on the records API (the ``token`` parameter) is a guess** until a sandbox
session makes a link and pulls it: the fake takes it so (``zenodo_records``), and nothing here measures Zenodo.

Controls, each caught on every run:

- **a redirect to another host**: the store the fake redirects a download to receives no token and no key; with the
  redirect followed by urllib (the token) or the key put on the store's address (the key), the store is caught
  receiving it;
- **a planted proxy**: a proxy named by the environment receives nothing of a pull from a service on this machine; with
  the direct opener planted away, it is caught receiving the token;
- **the library's frames in a traceback**: an interrupt while a signed-in or linked request waits leaves neither in a
  traceback's locals; with the boundary planted away, the library's frame is caught holding each;
- **a link read by its spelling** (the review of #112, F3): with no host recognised as a service's, a link on the
  service's host by another port is caught sent, key and all, to whatever listens there;
- **an address said whole**: with the redaction planted away, an address elsewhere is caught printing its ``token``
  parameter, in the line, the refusal, pooch's log and MIME's traceback file;
- **a link held as text** (F7): with pull keeping a link as the text it was given, a refused pull's traceback is caught
  holding the key in a library frame's locals.

A link is read with no echo by ``mime pull --link -``, or from ``MIME_SHARE_LINK`` by ``--link env``, so that its key is
in neither the shell's history nor the process's arguments; ``--link`` with anything else is refused without its value.
"""

from __future__ import annotations

import http.server
import logging
import os
import socket
import threading
import traceback
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from push_cases import copy_of, saved_toy, with_token, write_sharing
from test_push_client import captured_with_locals, interrupted_while_waiting, proxy, unix_signals
from zenodo_records import FakeRecords, served

from mime import push
from mime.build import BuildRefused
from mime.cli import main
from mime.credentials import Token
from mime.assets import _fetch
from mime.push import _client, _records
from mime.saved import _pull, pull

NUMBER = "629525"
DOI = f"10.5072/zenodo.{NUMBER}"
READER = "a-readers-token-" + "r" * 24
KEY = "eyJhbGciOiJIUzUxMiJ9.a-share-links-key." + "k" * 24


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = saved_toy(tmp_path_factory.mktemp("toy"))
    write_sharing(folder)
    return folder


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeRecords]:
    """The fake behind a server on this machine, standing in for the sandbox, its record 629525 published and then
    restricted on its page, record and files; tracebacks written into the test's own folder."""
    token = with_token(monkeypatch)
    monkeypatch.setattr(push._client, "PAUSE_SECONDS", 0.0)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.delenv("MIME_TRACEBACK", raising=False)
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path / "tracebacks"))
    with served(token) as fake:
        monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", fake.service())  # type: ignore[index]
        yield fake


@pytest.fixture
def restricted(toy: Path, sandbox: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRecords:
    """The record published, then restricted, shared with the reader's account and by a link; the token stored for
    the sandbox is now the reader's."""
    folder = copy_of(toy, tmp_path)
    service = push.SERVICES["zenodo-sandbox"]
    push.push(folder, service, pause=0)
    push.publish(folder, service, confirm=DOI, pause=0)
    sandbox.records[NUMBER].body["access"] = {"record": "restricted", "files": "restricted"}
    sandbox.readers.add(READER)
    sandbox.links[NUMBER] = KEY
    monkeypatch.setenv("ZENODO_SANDBOX_TOKEN", READER)
    sandbox.seen.clear()
    return sandbox


def everything_written(root: Path) -> bytes:
    """Every file below ``root`` (the test's home, cache, state, tracebacks and folders), read whole."""
    return b"\n".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())


def pulled(words: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(["pull", *words])
    held = capsys.readouterr()
    return code, held.out + held.err


# --- without the flag, nothing is sent; the refusal names the route ------------------------------------------------


def test_without_the_flag_no_token_is_sent_and_the_refusal_names_the_flag(restricted: FakeRecords, tmp_path: Path):
    with pytest.raises(BuildRefused) as refused:
        pull(NUMBER, into=tmp_path / "pulled", from_service="zenodo-sandbox")
    assert refused.value.codes() == ["pull.restricted"]
    assert f"mime pull --from zenodo-sandbox {NUMBER} --signed-in" in refused.value.refusals[0].action
    assert restricted.seen and all(authorization is None for _, _, _, authorization in restricted.seen)


@pytest.mark.parametrize(("access", "said", "not_said"), [
    ({"record": "restricted", "files": "restricted"}, "it is a draft, or a restricted record", "its files"),
    ({"record": "public", "files": "restricted"}, "its files are restricted", "a restricted record")])
def test_without_a_token_the_refusal_says_whether_the_record_or_its_files_were_kept(
        restricted: FakeRecords, tmp_path: Path, access: dict[str, str], said: str, not_said: str):
    restricted.records[NUMBER].body["access"] = access
    with pytest.raises(BuildRefused) as refused:
        pull(NUMBER, into=tmp_path / "pulled", from_service="zenodo-sandbox")
    message = refused.value.refusals[0].message
    assert refused.value.codes() == ["pull.restricted"] and said in message and not_said not in message


def test_with_no_token_stored_the_refusal_says_how_to_store_one(restricted: FakeRecords, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZENODO_SANDBOX_TOKEN")
    with pytest.raises(BuildRefused) as refused:
        pull(NUMBER, into=tmp_path / "pulled", from_service="zenodo-sandbox")
    action = refused.value.refusals[0].action
    assert "`mime credentials set zenodo-sandbox`, then pull it signed in" in action and "--signed-in" in action
    assert f"{restricted.site}/records/{NUMBER}, then `mime pull <the .mimebundle file>`" in action


# --- signed in -------------------------------------------------------------------------------------------------------


def test_signed_in_pulls_a_record_shared_with_the_account_and_the_token_is_nowhere_but_its_header(
        restricted: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    held = Token("zenodo-sandbox", READER)
    code, said = pulled(["--from", "zenodo-sandbox", NUMBER, "--into", str(tmp_path / "pulled"), "--signed-in"],
                        capsys)
    assert code == 0, said
    assert main(["verify", str(tmp_path / "pulled")]) == 0
    assert not held.occurs_in(said.encode()) and not held.occurs_in(everything_written(tmp_path))
    assert {authorization for _, _, _, authorization in restricted.seen} == {f"Bearer {READER}"}
    assert all(path.startswith(f"/records/{NUMBER}") for _, path, _, _ in restricted.seen)


def test_signed_in_needs_from_and_a_stored_token_and_sends_nothing_without_either(
        restricted: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(BuildRefused) as refused:
        pull(DOI.replace("5072", "5281"), into=tmp_path / "a", signed_in=True)
    assert refused.value.codes() == ["pull.source"] and "--from" in refused.value.refusals[0].message
    monkeypatch.delenv("ZENODO_SANDBOX_TOKEN")
    with pytest.raises(BuildRefused) as refused:
        pull(NUMBER, into=tmp_path / "b", from_service="zenodo-sandbox", signed_in=True)
    assert refused.value.codes() == ["pull.token"]
    assert "mime credentials set zenodo-sandbox" in refused.value.refusals[0].action
    assert restricted.seen == []


def test_a_token_the_record_is_not_shared_with_is_refused_by_what_it_means(restricted: FakeRecords, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZENODO_SANDBOX_TOKEN", "a-stranger-" + "s" * 30)
    with pytest.raises(BuildRefused) as refused:
        pull(NUMBER, into=tmp_path / "pulled", from_service="zenodo-sandbox", signed_in=True)
    assert refused.value.codes() == ["pull.restricted"]
    assert "the token stored for it" in refused.value.refusals[0].message
    assert "mime credentials status zenodo-sandbox" in refused.value.refusals[0].action


# --- a share link -------------------------------------------------------------------------------------------------


def test_a_share_link_pulls_by_its_key_which_is_never_printed_logged_or_written(
        restricted: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    held = Token("a share link", KEY)
    link = f"{restricted.site}/records/{NUMBER}?token={KEY}"
    code, said = pulled([link, "--into", str(tmp_path / "pulled")], capsys)
    assert code == 0, said
    assert f"the record {NUMBER} on zenodo-sandbox" in said
    assert not held.occurs_in(said.encode()) and not held.occurs_in(everything_written(tmp_path))
    assert restricted.seen and all(query == f"token={KEY}" and authorization is None
                                   for _, _, query, authorization in restricted.seen)


def test_a_revoked_link_is_refused_saying_so_and_not_its_key(restricted: FakeRecords, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]):
    held = Token("a share link", KEY)
    restricted.links[NUMBER] = "another-key-" + "o" * 20
    code, said = pulled([f"{restricted.site}/records/{NUMBER}?token={KEY}", "--into", str(tmp_path / "x")], capsys)
    assert code == 2 and "pull.restricted" in said and "share link's key" in said and "a guess" in said
    assert not held.occurs_in(said.encode()) and not held.occurs_in(everything_written(tmp_path))


def test_a_key_on_the_services_host_that_is_not_a_records_link_is_refused_without_it(
        restricted: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    held = Token("a share link", KEY)
    for given in (f"{restricted.site}/records/{NUMBER}/files/full_toy.mimebundle?token={KEY}",
                  f"{restricted.site}/records/{NUMBER}?token={KEY}&token=again-{KEY}"):
        code, said = pulled([given, "--into", str(tmp_path / "x")], capsys)
        assert code == 2 and "pull.source" in said and not held.occurs_in(said.encode()), said
    code, said = pulled([f"{restricted.site}/records/{NUMBER}?token={KEY}", "--signed-in", "--into",
                         str(tmp_path / "y")], capsys)
    assert code == 2 and "pull.source" in said and not held.occurs_in(said.encode())
    assert restricted.seen == [] and not held.occurs_in(everything_written(tmp_path))


def test_an_address_elsewhere_with_a_token_parameter_is_no_share_link():
    assert _records.share_link(f"https://example.org/records/{NUMBER}?token={KEY}") is None
    assert _records.share_link(f"{push.ZENODO.site}/records/{NUMBER}") is None
    found = _records.share_link(f"{push.ZENODO.site}/records/{NUMBER}?preview=1&token={KEY}")
    assert found is not None and (found.service.name, found.record) == ("zenodo", NUMBER)
    assert found.key.reveal() == KEY and KEY not in repr(found)


# --- the controls: a redirect elsewhere, a proxy, a traceback --------------------------------------------------------


@contextmanager
def store(fake: FakeRecords) -> Iterator[tuple[str, list[tuple[str, str | None]]]]:
    """Another host (a server on another port of this machine) that holds the record's files, as a service's store
    does: its address, and each request it got, its path and parameters and its Authorization header."""
    seen: list[tuple[str, str | None]] = []

    class Store(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append((self.path, self.headers.get("Authorization")))
            record, key = self.path.split("?")[0].strip("/").split("/")[-2:]
            data = fake.records[record].files[key].data or b""
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/store", seen
    finally:
        server.shutdown()
        server.server_close()


def through_the_store(fake: FakeRecords, tmp_path: Path, how: str) -> list[tuple[str, str | None]]:
    """What the store saw of a pull signed in (``how`` ``token``) or by the link (``key``), the record's download
    redirected to it."""
    with store(fake) as (address, seen):
        fake.stored_at = address
        if how == "token":
            pull(NUMBER, into=tmp_path / "pulled", from_service="zenodo-sandbox", signed_in=True)
        else:
            pull(f"{fake.site}/records/{NUMBER}?token={KEY}", into=tmp_path / "pulled")
    return seen


def secret_of(how: str) -> Token:
    return Token("held", READER if how == "token" else KEY)


@pytest.mark.parametrize("how", ["token", "key"])
def test_a_redirect_to_another_host_receives_neither_the_token_nor_the_key(how: str, restricted: FakeRecords,
                                                                            tmp_path: Path):
    seen = through_the_store(restricted, tmp_path, how)
    assert seen and all(not secret_of(how).occurs_in(f"{path} {header}".encode()) for path, header in seen)
    assert (tmp_path / "pulled").is_dir()


def test_control_a_redirect_followed_by_urllib_carries_the_token_to_the_store(
        restricted: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_client.NoRedirect, "redirect_request", urllib.request.HTTPRedirectHandler.redirect_request)
    seen = through_the_store(restricted, tmp_path, "token")
    assert any(header == f"Bearer {READER}" for _, header in seen), "the planted defect (a redirect followed) is not seen"


def test_control_a_key_put_on_the_stores_address_is_caught_there(restricted: FakeRecords, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch):
    real = _records.public_into
    monkeypatch.setattr(_records, "public_into", lambda url, into: real(f"{url}?token={KEY}", into))
    seen = through_the_store(restricted, tmp_path, "key")
    assert any(secret_of("key").occurs_in(path.encode()) for path, _ in seen), "the planted defect is not seen"


def through_the_proxy(fake: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """What a proxy the environment names saw of a signed-in pull from the service on this machine."""
    for name in ("no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    with proxy() as (address, seen):
        monkeypatch.setenv("http_proxy", address)
        monkeypatch.setenv("HTTP_PROXY", address)
        try:
            pull(NUMBER, into=tmp_path / "pulled", from_service="zenodo-sandbox", signed_in=True)
        except BuildRefused:
            pass
    return seen


def test_a_planted_proxy_receives_nothing_of_a_signed_in_pull(restricted: FakeRecords, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch):
    assert through_the_proxy(restricted, tmp_path, monkeypatch) == []
    assert (tmp_path / "pulled").is_dir()


def test_control_with_the_direct_opener_planted_away_the_proxy_receives_the_token(
        restricted: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_client, "loopback", lambda url: False)
    seen = through_the_proxy(restricted, tmp_path, monkeypatch)
    assert seen and any(Token("held", READER).occurs_in(each) for each in seen), "the planted proxy is not seen"


def stand_in(base: str) -> push.Service:
    return push.Service("zenodo-sandbox", base + "/api", base, "zenodo-sandbox", False, "a stand-in")


@unix_signals
def test_an_interrupt_while_a_signed_in_pull_waits_leaves_no_token_in_a_tracebacks_locals():
    access = _client.Access(token=Token("zenodo-sandbox", READER))
    error = interrupted_while_waiting(Token("held", READER), lambda base: _client.fetched(
        stand_in(base), f"/records/{NUMBER}", access, timeout=20))
    assert isinstance(error, KeyboardInterrupt) and error.__context__ is None and error.__cause__ is None
    assert not secret_of("token").occurs_in(captured_with_locals(error).encode("utf-8"))


@unix_signals
def test_control_without_the_boundary_the_librarys_frame_is_caught_holding_the_token():
    access = _client.Access(token=Token("zenodo-sandbox", READER))
    transport = push.HttpTransport(timeout=20)
    error = interrupted_while_waiting(Token("held", READER), lambda base: _client._let_in(  # pyright: ignore[reportPrivateUsage]
        transport.direct, 20, f"{base}/api/records/{NUMBER}", push.RDM_TYPE, access, None))
    assert isinstance(error, KeyboardInterrupt)
    assert secret_of("token").occurs_in(captured_with_locals(error).encode("utf-8")), "the planted defect is not seen"


#: A path the standard library refuses as it puts the request line together: it raises naming the whole address,
#: share link's key and all, from a frame that holds it. (No frame of the library holds the address while a request
#: waits for its answer, so an interrupt there could not tell the boundary from its absence for a key: this can.)
_REFUSED_PATH = f"/records/{NUMBER} refused"


def test_an_address_the_library_refuses_leaves_no_key_in_the_refusal_or_its_traceback():
    error: BaseException | None = None
    try:
        _client.fetched(stand_in("http://127.0.0.1:9"), _REFUSED_PATH, _client.Access(key=Token("x", KEY)), timeout=5)
    except push.Dropped as dropped:
        error = dropped
    assert isinstance(error, push.Dropped) and error.__context__ is None and "InvalidURL" in str(error)
    assert not secret_of("key").occurs_in(captured_with_locals(error).encode("utf-8"))


def test_control_without_the_boundary_the_librarys_error_is_caught_naming_the_key():
    transport = push.HttpTransport(timeout=5)
    with pytest.raises(Exception) as raised:
        _client._let_in(transport.direct, 5, f"http://127.0.0.1:9/api{_REFUSED_PATH}",  # pyright: ignore[reportPrivateUsage]
                        push.RDM_TYPE, _client.Access(key=Token("x", KEY)), None)
    assert secret_of("key").occurs_in(captured_with_locals(raised.value).encode("utf-8")), "the planted defect is not seen"


def test_nothing_listens_where_a_pull_from_a_closed_port_goes_and_the_refusal_holds_no_secret(tmp_path: Path):
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    service = push.Service("zenodo-sandbox", f"http://127.0.0.1:{port}/api", f"http://127.0.0.1:{port}",
                           "zenodo-sandbox", False, "a stand-in")
    for access in (_client.Access(token=Token("x", READER)), _client.Access(key=Token("x", KEY))):
        with pytest.raises(BuildRefused) as refused:
            push.public_record(service, NUMBER, access)
        assert refused.value.codes() == ["pull.record"]
        text = captured_with_locals(refused.value)
        assert not Token("x", READER).occurs_in(text.encode()) and not Token("x", KEY).occurs_in(text.encode())
    del tmp_path


# --- a link read by its components (F3), its key never said (F3, F7), and read out of sight (--link) --------------------


#: Every way of spelling a real service's address that is not the share link its page gives, with a key: each refused,
#: by its components, before any request (none of these is ever asked: the refusal comes first).
_NOT_THE_PAGES_FORM = {
    "http for https": f"http://zenodo.org/records/{NUMBER}?token={KEY}",
    "www": f"https://www.zenodo.org/records/{NUMBER}?token={KEY}",
    "a user": f"https://me@zenodo.org/records/{NUMBER}?token={KEY}",
    "another port": f"https://zenodo.org:8443/records/{NUMBER}?token={KEY}",
    "the sandbox by another name": f"https://www.sandbox.zenodo.org/records/{NUMBER}?token={KEY}",
    "the DOI resolver": f"https://doi.org/10.5281/zenodo.{NUMBER}?token={KEY}",
    "the DOI resolver's other name": f"http://dx.doi.org/10.5281/zenodo.{NUMBER}?token={KEY}",
}


@pytest.mark.parametrize("form", sorted(_NOT_THE_PAGES_FORM))
def test_a_link_spelled_otherwise_than_its_services_page_is_refused_by_its_components_saying_none_of_its_key(
        form: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    def no_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("a request was made")

    monkeypatch.setattr(_records, "fetched", no_request)
    monkeypatch.setattr(_fetch, "_download_held_to", no_request)
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path / "tracebacks"))
    held = Token("held", KEY)
    with pytest.raises(BuildRefused) as refused:
        _records.share_link(_NOT_THE_PAGES_FORM[form])
    assert refused.value.codes() == ["pull.source"] and not held.occurs_in(str(refused.value).encode())
    assert not held.occurs_in(captured_with_locals(refused.value).encode())
    code, said = pulled([_NOT_THE_PAGES_FORM[form], "--into", str(tmp_path / "x")], capsys)
    assert code == 2 and "pull.source" in said and not held.occurs_in(said.encode()), said
    assert not held.occurs_in(everything_written(tmp_path))


def test_a_link_in_its_services_own_scheme_host_and_port_is_read_however_it_writes_them():
    for given in (f"https://zenodo.org:443/records/{NUMBER}?token={KEY}",
                  f"https://zenodo.org./records/{NUMBER}?token={KEY}",
                  f"HTTPS://ZENODO.ORG/records/{NUMBER}?token={KEY}",
                  f"https://zenodo.org/api/records/{NUMBER}?token={KEY}"):
        found = _records.share_link(given)
        assert found is not None and (found.service.name, found.record) == ("zenodo", NUMBER), given
        assert found.key.reveal() == KEY


def through_another_port(fake: FakeRecords, tmp_path: Path,
                         capsys: pytest.CaptureFixture[str]) -> tuple[int, str, list[tuple[str, str | None]]]:
    """A link on the fake service's host by another port, where another server listens: what pull said, and what that
    server received."""
    with store(fake) as (address, seen):
        port = urlsplit(address).port
        code, said = pulled([f"http://127.0.0.1:{port}/records/{NUMBER}?token={KEY}", "--into",
                             str(tmp_path / "x")], capsys)
    return code, said, seen


def test_a_link_on_its_services_host_by_another_port_is_refused_and_nothing_is_sent(
        restricted: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code, said, seen = through_another_port(restricted, tmp_path, capsys)
    assert code == 2 and "pull.source" in said and "names the port" in said, said
    assert seen == [] and restricted.seen == [] and not Token("held", KEY).occurs_in(said.encode())


def test_control_with_no_host_read_as_a_services_the_link_by_another_port_is_caught_sent_with_its_key(
        restricted: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_records, "_on", lambda host: None)
    _, _, seen = through_another_port(restricted, tmp_path, capsys)
    assert any(secret_of("key").occurs_in(path.encode()) for path, _ in seen), "the planted defect is not seen"


@contextmanager
def pooch_log() -> Iterator[list[str]]:
    """pooch's own log lines, as its handlers receive them: after the logger's filters."""
    lines: list[str] = []

    class Kept(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    import pooch

    logger = pooch.get_logger()
    handler = Kept()
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)


def elsewhere_said(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                   monkeypatch: pytest.MonkeyPatch) -> tuple[int, str, list[str], bytes]:
    """An address elsewhere carrying a ``token`` parameter, where nothing listens, pulled: the exit code, what was
    said, pooch's log, and every file written (MIME's traceback file among them)."""
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path / "tracebacks"))
    with pooch_log() as logged:
        code, said = pulled([f"http://localhost:{port}/full_toy.mimebundle?token={KEY}", "--into",
                             str(tmp_path / "x")], capsys)
    return code, said, logged, everything_written(tmp_path)


def test_an_address_elsewhere_is_said_with_its_key_hidden_in_lines_refusals_the_log_and_the_traceback_file(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    code, said, logged, written = elsewhere_said(tmp_path, capsys, monkeypatch)
    held = Token("held", KEY)
    assert code == 2 and "asset.unreachable" in said and "token=<hidden>" in said, said
    assert logged and all("token=<hidden>" in line for line in logged if "token" in line)
    assert not held.occurs_in(said.encode()) and not any(held.occurs_in(line.encode()) for line in logged)
    assert list((tmp_path / "tracebacks").iterdir()) and not held.occurs_in(written)


def test_control_with_the_redaction_planted_away_the_address_elsewhere_is_caught_saying_its_key(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    for module in (_fetch, _pull):
        monkeypatch.setattr(module, "redacted", lambda text: text)
    _, said, _, _ = elsewhere_said(tmp_path, capsys, monkeypatch)
    assert Token("held", KEY).occurs_in(said.encode()), "the planted defect is not seen"


def below_the_caller(error: BaseException) -> str:
    """A traceback with its locals from the first frame below the test's own, which holds the link itself."""
    exception = traceback.TracebackException.from_exception(error, capture_locals=True)
    exception.stack = traceback.StackSummary.from_list(list(exception.stack)[1:])
    return "".join(exception.format())


def refused_link_pull(fake: FakeRecords, tmp_path: Path, source: Callable[[str], object]) -> BuildRefused:
    """A pull of the record by its link, the link revoked: the refusal."""
    fake.links[NUMBER] = "another-key-" + "o" * 20
    try:
        pull(source(f"{fake.site}/records/{NUMBER}?token={KEY}"), into=tmp_path / "x")  # type: ignore[arg-type]
    except BuildRefused as refused:
        return refused
    raise AssertionError("a revoked link was pulled")


@pytest.mark.parametrize("held", [True, False], ids=["held", "text"])
def test_a_refused_link_pull_leaves_its_key_in_no_frames_locals_below_the_caller(
        held: bool, restricted: FakeRecords, tmp_path: Path):
    refused = refused_link_pull(restricted, tmp_path,
                                (lambda link: Token(_records.SHARE_LINK, link)) if held else (lambda link: link))
    assert refused.codes() == ["pull.restricted"]
    assert not secret_of("key").occurs_in(below_the_caller(refused).encode())


def test_control_with_the_link_kept_as_text_a_library_frame_is_caught_holding_its_key(
        restricted: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_pull, "_kept_secret", lambda source: source.reveal() if isinstance(source, Token) else
                        os.fspath(source))
    real = _pull._bundle_from  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(_pull, "_bundle_from", lambda source, *rest: real(  # type: ignore[misc]
        Token("x", source) if isinstance(source, str) else source, *rest))
    refused = refused_link_pull(restricted, tmp_path, lambda link: link)
    assert secret_of("key").occurs_in(below_the_caller(refused).encode()), "the planted defect is not seen"


def test_link_env_and_link_dash_read_the_link_out_of_sight_and_it_is_said_nowhere(
        restricted: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    link = f"{restricted.site}/records/{NUMBER}?token={KEY}"
    monkeypatch.setenv("MIME_SHARE_LINK", link)
    code, said = pulled(["--link", "env", "--into", str(tmp_path / "from-env")], capsys)
    assert code == 0, said
    monkeypatch.delenv("MIME_SHARE_LINK")
    asked: list[str] = []
    monkeypatch.setattr("getpass.getpass", lambda prompt: asked.append(prompt) or f" {link}\n")
    code, typed = pulled(["--link", "-", "--into", str(tmp_path / "typed")], capsys)
    assert code == 0, typed
    assert asked == ["share link (not shown): "]
    assert (tmp_path / "from-env").is_dir() and (tmp_path / "typed").is_dir()
    held = Token("held", KEY)
    assert not held.occurs_in(f"{said}{typed}".encode()) and not held.occurs_in(everything_written(tmp_path))
    assert restricted.seen and all(query == f"token={KEY}" for _, _, query, _ in restricted.seen)


def test_link_is_refused_without_its_value_where_it_is_not_dash_or_env(
        restricted: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    link = f"{restricted.site}/records/{NUMBER}?token={KEY}"
    held = Token("held", KEY)
    for words in (["--link", link], ["--link", "env", link], ["--link", "-", NUMBER]):
        with pytest.raises(SystemExit) as stopped:
            main(["pull", *words, "--into", str(tmp_path / "x")])
        said = capsys.readouterr()
        assert stopped.value.code == 2 and not held.occurs_in(f"{said.out}{said.err}".encode()), said
    monkeypatch.delenv("MIME_SHARE_LINK", raising=False)
    code, said = pulled(["--link", "env", "--into", str(tmp_path / "y")], capsys)
    assert code == 2 and "pull.source" in said and "MIME_SHARE_LINK holds no share link" in said
    monkeypatch.setattr("getpass.getpass", lambda prompt: "")
    code, said = pulled(["--link", "-", "--into", str(tmp_path / "z")], capsys)
    assert code == 2 and "no share link was typed" in said
    assert restricted.seen == []
