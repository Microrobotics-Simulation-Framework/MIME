"""What the resolver keeps for every adapter alike, against what a server or an adapter says (design #229 item 4, the
review of its first round): every server is on this machine.

- **A redirect never changes the scheme to a local one**: a location redirected to a ``file:`` mirror, a ``package:``
  file, a ``doi:`` or another adapter's scheme is refused, the local file never read and the credentialed store never
  asked; a remote location may lead to ``https`` only, and to plain ``http`` only from this machine to this machine.
  ``mime assets sources`` follows the same rule.
- **A credential is hidden from whatever an adapter writes**: an adapter that puts the token it was handed into its
  error (of every kind it may raise), into a log line or a logged traceback, or into a redirect's target, leaves it in
  no refusal, no line ``mime assets sources`` prints, no log record and no file the cache holds.
- **The DOI step has a timeout**, the resolver's, which pooch passes to every request of the step; a step that times
  out is asked again with backoff, as any location is.
- **A ``Retry-After`` that is not a wait a store means** (``nan``, ``inf``, ``-5``, more than a day, a number past a
  float, a word) leaves the default backoff, and a wait the store asks for is held to its cap.

The controls, each failing on every run: a resolver that follows any redirect reads the local mirror's file and the
package's file for a remote server; a resolver that hides only an address's key leaves the token in the refusal; a DOI
step without the timeout is caught by the timeout it was not handed; a wait read as any float sleeps a NaN."""

import hashlib
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime import assets, credentials
from mime.assets import AdapterError, LoginRefused, NotFound, Opened, Redirect, Request, Transient, _adapters, _fetch
from mime.build import BuildRefused
from mime.credentials import Token
from mime.testing import HARNESS_TOKEN, LoopbackStore

PAYLOAD = b"bytes kept on this machine only, which a remote server must not choose to read. " * 40
RUNS = 20


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MIME_OFFLINE", raising=False)
    monkeypatch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "locations.toml"))
    (tmp_path / "locations.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(_fetch, "BACKOFF_SECONDS", 0.0)
    return tmp_path / "cache"


def ref_of(data: bytes, *locations: str, name: str = "kept", adapters: tuple[tuple[str, str], ...] = ()
           ) -> assets.AssetRef:
    return assets.AssetRef(name, hashlib.sha256(data).hexdigest(), len(data), "CC0-1.0", locations,
                           adapters=adapters)


def refused_text(action) -> tuple[list[str], str]:  # type: ignore[no-untyped-def]
    try:
        action()
    except BuildRefused as refused:
        return refused.codes(), str(refused) + repr(refused.refusals)
    raise AssertionError("not refused")


# --- a redirect never changes the scheme to a local one -------------------------------------------------------------


def local_targets(tmp_path: Path) -> dict[str, tuple[str, bytes]]:
    """Each local location a redirect might lead to, with the bytes it would serve: a mirror directory holding the
    payload by its sha256, a file of MIME's own package, a DOI, and the lab fixture's credentialed store."""
    mirror = tmp_path / "private"
    mirror.mkdir(exist_ok=True)
    (mirror / hashlib.sha256(PAYLOAD).hexdigest()).write_bytes(PAYLOAD)
    import importlib.resources

    package = importlib.resources.files("mime.data").joinpath("__init__.py").read_bytes()
    return {"file": (mirror.as_uri() + "/", PAYLOAD), "package": ("package:mime/__init__.py", package),
            "doi": ("doi:10.5281/zenodo.1/kept.bin", PAYLOAD), "labstore": ("labstore://127.0.0.1:1/kept", PAYLOAD)}


@pytest.mark.parametrize("scheme", ["file", "package", "doi", "labstore"])
def test_a_redirect_to_a_local_scheme_is_refused_and_the_local_file_never_read(cache, tmp_path, scheme):
    target, data = local_targets(tmp_path)[scheme]
    with LoopbackStore() as server:
        server.redirects["/hop"] = target
        ref = ref_of(data, server.url("/hop"), adapters=(("labstore", "mime-lab-fixture"),))
        codes, text = refused_text(lambda: assets.fetch(ref))
        assert codes == ["asset.unreachable"] and f"never to a {scheme}: location" in text
        assert not assets.is_cached(ref)
        probed = assets.probe(ref.locations[0], ref)
        assert probed.reachable is False and "never to a" in probed.why


def test_a_redirect_among_this_machines_http_and_to_https_is_followed(cache):
    with LoopbackStore() as server, LoopbackStore("localhost") as other:
        other.files["/moved"] = PAYLOAD
        server.redirects["/hop"] = other.url("/moved")
        assert assets.fetch(ref_of(PAYLOAD, server.url("/hop"))).read_bytes() == PAYLOAD


@pytest.mark.parametrize("answering,target,followed", [
    ("https://files.example.org/a", "https://cdn.example.org/a", True),
    ("https://files.example.org/a", "http://127.0.0.1:8000/a", False),
    ("https://files.example.org/a", "http://localhost/a", False),
    ("http://127.0.0.1:8000/a", "http://localhost:9000/a", True),
    ("http://127.0.0.1:8000/a", "http://files.example.org/a", False),
    ("https://files.example.org/a", "file:///srv/mirror/", False),
    ("https://files.example.org/a", "package:mime/__init__.py", False),
    ("https://files.example.org/a", "doi:10.5281/zenodo.1/a", False),
    ("https://files.example.org/a", "labstore://127.0.0.1:1/a", False),
    ("labstore://127.0.0.1:1/a", "labstore://localhost:2/a", True),
    ("labstore://127.0.0.1:1/a", "file:///srv/mirror/", False),
    ("file:///srv/mirror/", "file:///srv/other/", False),
    ("doi:10.5281/zenodo.1/a", "https://zenodo.org/records/1/files/a", True),
])
def test_the_redirect_rule_reads_the_scheme_and_where_each_end_is(answering, target, followed):
    """A remote location leads to https only; plain http only from this machine to this machine; an adapter's own
    scheme to itself (its store's business, the credential still bound to its host); nothing to a local scheme."""
    assert (_fetch.redirect_problem(answering, target) is None) == followed


@contextmanager
def any_redirect_followed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "redirect_problem", lambda answering, target: None)
        yield


def test_control_a_resolver_that_follows_any_redirect_reads_the_local_files_every_run(cache, tmp_path):
    caught = 0
    for run in range(RUNS):
        with any_redirect_followed(), pytest.MonkeyPatch.context() as patch:
            patch.setenv("MIME_ASSETS_DIR", str(tmp_path / f"cache{run}"))
            try:
                test_a_redirect_to_a_local_scheme_is_refused_and_the_local_file_never_read(None, tmp_path, "file")
                test_a_redirect_to_a_local_scheme_is_refused_and_the_local_file_never_read(None, tmp_path, "package")
            except AssertionError:
                caught += 1
    assert caught == RUNS


# --- a credential hidden from whatever an adapter writes ------------------------------------------------------------


class Echoing:
    """A plugin's adapter with a bug: it writes the token it was handed into whatever it says."""

    scheme = "echoing"
    needs_credential = True
    ranged = False

    def __init__(self, how: str) -> None:
        self.how = how

    def open(self, location: str, request: Request) -> Opened | Redirect:
        token = request.credential.token.reveal() if request.credential is not None else "no token"
        said = f"the backend says: bearer {token} (as %22{token}%22 and in JSON \\u0022{token}\\u0022)"
        logging.getLogger("a_plugin").warning("asked with %s", token)
        if self.how == "logged":
            try:
                raise RuntimeError(f"traceback holding {token}")
            except RuntimeError:
                logging.getLogger("a_plugin").exception("failed for %s", token)
            raise NotFound("no file there")
        raise {"login": LoginRefused(said), "missing": NotFound(said), "busy": Transient(said, 0.0),
               "failed": AdapterError(said), "other": RuntimeError(said), "redirect": None}[self.how] or \
            AssertionError("unreachable")


class RedirectingEchoing(Echoing):
    def open(self, location: str, request: Request) -> Opened | Redirect:
        token = request.credential.token.reveal() if request.credential is not None else "no token"
        return Redirect(f"file:///srv/{token}/")


ECHOES = ["login", "missing", "busy", "failed", "other", "logged", "redirect"]


def echoed(how: str, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Where the token went when an echoing adapter was asked: every text MIME said, logged and kept."""
    adapter = RedirectingEchoing(how) if how == "redirect" else Echoing(how)
    _adapters.register_adapter(adapter)
    try:
        credentials.store(credentials.host_service("127.0.0.1"), Token("host:127.0.0.1", HARNESS_TOKEN))
        ref = ref_of(PAYLOAD, "echoing://127.0.0.1/kept", adapters=(("echoing", ""),))
        with caplog.at_level(logging.DEBUG):
            _, text = refused_text(lambda: assets.fetch(ref))
            probed = assets.probe(ref.locations[0], ref)
        cached = " ".join(path.read_bytes().decode("utf-8", "replace") for path in (tmp_path / "cache").rglob("*")
                          if path.is_file())
        said = {"the refusal": text, "the probe": probed.why + probed.location, "the log": caplog.text,
                "the cache": cached}
        return [where for where, held in said.items() if HARNESS_TOKEN in held]
    finally:
        _adapters._registered.pop("echoing", None)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("how", ECHOES)
def test_a_token_an_adapter_echoes_is_in_no_refusal_line_log_or_cache(cache, tmp_path, caplog, how):
    assert echoed(how, tmp_path, caplog) == []


def test_every_form_of_the_token_is_hidden():
    token = Token("host:x", HARNESS_TOKEN)
    text = f"{HARNESS_TOKEN} %22{HARNESS_TOKEN}%22 " + HARNESS_TOKEN.encode().hex() + " " + \
        __import__("base64").b64encode(HARNESS_TOKEN.encode()).decode()
    hidden = token.hidden_in(text, "<hidden>")
    assert HARNESS_TOKEN not in hidden and HARNESS_TOKEN.encode().hex() not in hidden and "<hidden>" in hidden
    assert not token.occurs_in(hidden.encode())


@contextmanager
def keys_alone_hidden() -> Iterator[None]:
    """The plant: a resolver that hides an address's key alone, as the first round did."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "hidden", assets._reference.redacted)  # pyright: ignore[reportPrivateUsage]
        yield


def test_control_a_resolver_that_hides_keys_alone_leaves_the_token_every_run(cache, tmp_path, caplog):
    caught = 0
    for run in range(RUNS):
        with keys_alone_hidden(), pytest.MonkeyPatch.context() as patch:
            patch.setenv("MIME_ASSETS_DIR", str(tmp_path / f"cache{run}"))
            caught += "the refusal" in echoed("failed", tmp_path, caplog)
    assert caught == RUNS


# --- the DOI step's timeout -----------------------------------------------------------------------------------------


@contextmanager
def doi_org_answering(asked: list[dict[str, object]], *, times_out: bool) -> Iterator[None]:
    """``requests.get`` as pooch's DOI step calls it, faked: each call's keywords kept, and a timeout raised or a
    record of no known repository returned. No request leaves the machine."""
    import requests

    def get(url: str, **keywords: object) -> object:
        asked.append({"url": url, **keywords})
        if times_out:
            raise requests.Timeout(f"{url} did not answer in time")
        raise requests.ConnectionError("refused here: no request leaves the machine in a test")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests, "get", get)
        yield


def test_the_doi_step_is_held_to_the_resolvers_timeout_and_asked_again_when_it_times_out(cache):
    asked: list[dict[str, object]] = []
    ref = ref_of(PAYLOAD, "doi:10.5281/zenodo.1234567/kept.bin")
    with doi_org_answering(asked, times_out=True):
        codes, text = refused_text(lambda: assets.fetch(ref))
    assert codes == ["asset.unreachable"] and "did not recover" not in text
    assert len(asked) == _fetch.RETRIES + 1, "a step that timed out is asked again, as any location is"
    assert all(each["timeout"] == _fetch.TIMEOUT_SECONDS for each in asked)
    assert all(str(each["url"]).startswith("https://doi.org/10.5281/zenodo.1234567") for each in asked)


def test_control_a_doi_step_without_the_timeout_is_caught_every_run(cache, monkeypatch):
    import pooch

    monkeypatch.setattr(_adapters, "doi_file_url", lambda doi, file, timeout: str(
        pooch.downloaders.doi_to_repository(doi).download_url(file)))  # pyright: ignore[reportUnknownMemberType]
    caught = 0
    for _ in range(RUNS):
        try:
            test_the_doi_step_is_held_to_the_resolvers_timeout_and_asked_again_when_it_times_out(cache)
        except (AssertionError, KeyError):
            caught += 1
    assert caught == RUNS


# --- Retry-After ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stated,expected", [
    ("1", 1.0), ("0", 0.0), ("2.5", 2.5), ("nan", None), ("NaN", None), ("inf", None), ("-inf", None), ("-5", None),
    ("1e999", None), ("9" * 400, None), ("soon", None), ("", None), ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0),
    ("Wed, 21 Oct 2015 07:28:00", None)])
def test_a_retry_after_is_read_as_seconds_or_an_http_date_and_nothing_else(stated, expected):
    assert _adapters.retry_after_seconds(stated) == expected


@pytest.mark.parametrize("stated,expected", [
    (None, "default"), (math.nan, "default"), (math.inf, "default"), (-math.inf, "default"), (-5.0, "default"),
    (1e308, "default"), (10 ** 400, "default"), (2 * 86_400.0, "default"), ("5", "default"), (True, "default"),
    (3.0, 3.0), (0, 0.0), (600, "cap"), (86_400.0, "cap")])
def test_any_stated_wait_is_bounded_and_one_that_is_not_a_meant_wait_leaves_the_default(stated, expected):
    held = _fetch.wait_seconds(2, stated)
    assert 0.0 <= held <= max(_fetch.RETRY_AFTER_CAP_SECONDS, _fetch.BACKOFF_SECONDS * 2) and math.isfinite(held)
    named = {"default": _fetch.BACKOFF_SECONDS * 2, "cap": _fetch.RETRY_AFTER_CAP_SECONDS}
    assert held == named.get(expected, expected)


@pytest.mark.parametrize("stated", ["nan", "inf", "-5", "1e999", "99999999999", "soon",
                                    "Wed, 21 Oct 2015 07:28:00 GMT"])
def test_a_busy_server_with_any_retry_after_is_asked_again_and_the_wait_is_bounded(cache, monkeypatch, stated):
    waits: list[float] = []
    monkeypatch.setattr(_fetch.time, "sleep", waits.append)
    with LoopbackStore() as server:
        server.files["/blob"] = PAYLOAD
        server.statuses["/blob"] = [503]
        server.retry_after["/blob"] = stated
        assert assets.fetch(ref_of(PAYLOAD, server.url("/blob"))).read_bytes() == PAYLOAD
    assert len(waits) == 1 and 0.0 <= waits[0] <= _fetch.RETRY_AFTER_CAP_SECONDS


def test_control_a_wait_read_as_any_float_sleeps_a_nan_every_run(cache, monkeypatch):
    def as_any_float(attempt: int, stated: object) -> float:
        return float(stated) if isinstance(stated, float) else _fetch.BACKOFF_SECONDS  # type: ignore[arg-type]

    monkeypatch.setattr(_fetch, "wait_seconds", as_any_float)
    monkeypatch.setattr(_adapters, "retry_after_seconds", lambda text: float(text) if text else None)
    caught = 0
    for _ in range(RUNS):
        try:
            test_a_busy_server_with_any_retry_after_is_asked_again_and_the_wait_is_bounded(cache, monkeypatch, "nan")
        except AssertionError:
            caught += 1
    assert caught == RUNS
