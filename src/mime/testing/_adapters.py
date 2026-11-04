"""The conformance harness for an asset adapter, as ``verify_node`` is for a node: an adapter's author runs
:func:`verify_adapter` in their own tests, and it drives the adapter through MIME's own resolver against loopback stores
(``mime.testing._loopback``), so the checks are the ones every download is held to.

What it checks: a file is fetched whole and held to its hash; a store that sends more than the record's size is cut at
the declared size and nothing is stored; a missing file and a refused login are told apart (the adapter raises
``NotFound`` and ``LoginRefused``, and the refusals say which); and, for an adapter that needs a credential, the
credential is sent to its own host, never in an error's text, and never to another host that a redirect leads to.

An adapter whose locations are not ``http(s)`` addresses gives ``location_for``, which turns the loopback store's
address into a location of its own scheme (``labstore://127.0.0.1:8080/x`` for ``http://127.0.0.1:8080/x``); one
that cannot follow a redirect through the store says so with ``redirects=False``, and that check is reported as not
run, never as passed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from mime.assets import (
    Adapter,
    AssetRef,
    Credential,
    adapter_for,
    credentials_from,
    fetch,
    register_adapter,
    registered_schemes,
)
from mime.assets._adapters import _registered  # pyright: ignore[reportPrivateUsage]
from mime.assets._reference import BUILT_IN_SCHEMES, scheme_of
from mime.build import BuildRefused
from mime.credentials import Token
from mime.testing._loopback import LoopbackStore

#: The token the harness hands the adapter for the store's own host: it must reach that host and no other.
HARNESS_TOKEN = "harness-token-7c1e0d95b2a64f3e"
_OTHER_HOST = "localhost"


@dataclass
class AdapterReport:
    """What :func:`verify_adapter` found: the checks that passed, those not run (with why), and every problem."""

    passed: list[str] = field(default_factory=list[str])
    not_run: list[str] = field(default_factory=list[str])
    problems: list[str] = field(default_factory=list[str])

    @property
    def conforms(self) -> bool:
        return not self.problems


def _record(name: str, data: bytes, location: str, size: int | None = None) -> AssetRef:
    return AssetRef(name, hashlib.sha256(data).hexdigest(), size if size is not None else len(data), "CC0-1.0",
                    (location,))


@contextmanager
def _environment(cache: Path) -> Generator[None]:
    saved = {key: os.environ.get(key) for key in ("MIME_ASSETS_DIR", "MIME_OFFLINE", "MIME_ASSET_LOCATIONS",
                                                  "NO_PROXY", "no_proxy")}
    os.environ["MIME_ASSETS_DIR"] = str(cache)
    os.environ["MIME_ASSET_LOCATIONS"] = str(cache / "no-locations-file.toml")
    os.environ.pop("MIME_OFFLINE", None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    (cache / "no-locations-file.toml").parent.mkdir(parents=True, exist_ok=True)
    (cache / "no-locations-file.toml").write_text("", encoding="utf-8")
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _refusal(action: Callable[[], object]) -> BuildRefused | None:
    try:
        action()
    except BuildRefused as refused:
        return refused
    return None


def verify_adapter(adapter: Adapter, *, location_for: Callable[[str], str] = lambda url: url,
                   redirects: bool = True) -> AdapterReport:
    """Hold ``adapter`` to what MIME's resolver needs of every adapter, against loopback stores; nothing leaves the
    machine. Returns the report: :attr:`AdapterReport.conforms` is the claim."""
    report = AdapterReport()
    scheme = adapter.scheme
    added = scheme not in BUILT_IN_SCHEMES and scheme not in registered_schemes()
    if added:
        register_adapter(adapter)
    with tempfile.TemporaryDirectory(prefix="mime-adapter-") as work, _environment(Path(work) / "cache"), \
            LoopbackStore() as store, LoopbackStore(_OTHER_HOST) as other:
        try:
            _checks(adapter, store, other, location_for, redirects, report)
        finally:
            if added:
                _registered.pop(scheme, None)
    return report


def _checks(adapter: Adapter, store: LoopbackStore, other: LoopbackStore, location_for: Callable[[str], str],
            redirects: bool, report: AdapterReport) -> None:
    def at(path: str, server: LoopbackStore = store) -> str:
        location = location_for(server.url(path))
        if scheme_of(location) != adapter.scheme and adapter_for(location) is not adapter:
            raise ValueError(f"location_for gave {location!r}, which is not of the adapter's scheme {adapter.scheme!r}")
        return location

    def body(name: str) -> bytes:
        """Bytes of each check's own, so that no check finds another's file in the cache and asks nothing."""
        return f"the bytes of the file {name}, whole. ".encode() * 300

    whole = body("whole")
    store.files["/whole"] = whole
    held = _record("whole", whole, at("/whole"))
    refused = _refusal(lambda: fetch(held))
    if refused is None:
        report.passed.append("a file is fetched whole and held to its hash")
    else:
        report.problems.append(f"a file served whole was refused: {refused}")

    whole = body("longer")
    store.files["/longer"] = whole
    store.extra["/longer"] = b"x" * 5000
    longer = _record("longer", whole + b"x" * 5000, at("/longer"), size=len(whole))
    refused = _refusal(lambda: fetch(longer))
    if refused is not None and "asset.size-exceeded" in refused.codes():
        report.passed.append("a stream longer than the record's size is cut at the declared size")
    else:
        report.problems.append(f"a stream longer than the record's size was not cut there: {refused}")

    missing = _record("missing", b"nothing", at("/missing"))
    refused = _refusal(lambda: fetch(missing))
    if refused is not None and "no file at that address" in str(refused):
        report.passed.append("a missing file is said to be missing")
    else:
        report.problems.append(f"a missing file was not said to be missing (NotFound): {refused}")

    whole = body("private")
    store.files["/private"] = whole
    store.tokens["/private"] = "a-token-nobody-holds-here"
    private = _record("private", whole, at("/private"))
    refused = _refusal(lambda: fetch(private))
    if refused is not None and "refused the login" in str(refused) and "no file at that address" not in str(refused):
        report.passed.append("a refused login is told apart from a missing file")
    else:
        report.problems.append(f"a refused login was not told apart from a missing file (LoginRefused): {refused}")

    if not adapter.needs_credential:
        report.not_run.append("the credential checks: the adapter needs no credential")
        return
    host = store.host

    def hook(asked: str) -> Credential | None:
        return Credential(asked, Token(f"host:{asked}", HARNESS_TOKEN)) if asked == host else None

    whole = body("own")
    store.files["/own"] = whole
    store.tokens["/own"] = HARNESS_TOKEN
    own = _record("own", whole, at("/own"))
    with credentials_from(hook):
        refused = _refusal(lambda: fetch(own))
    if refused is None:
        report.passed.append("the credential reaches its own host")
    else:
        report.problems.append(f"the credential of the store's own host did not let the file in: {refused}")

    whole = body("wrong")
    store.tokens["/wrong"] = "another-token"
    store.files["/wrong"] = whole
    wrong = _record("wrong", whole, at("/wrong"))
    with credentials_from(hook):
        refused = _refusal(lambda: fetch(wrong))
    if refused is not None and HARNESS_TOKEN not in str(refused) and HARNESS_TOKEN not in repr(refused.refusals):
        report.passed.append("a credential is never in an error's text")
    else:
        report.problems.append("a refused credential's text holds the credential, or the login was not refused")

    if not redirects:
        report.not_run.append("the redirect to another host: the adapter says it follows no redirect")
        return
    whole = body("hop")
    other.files["/moved"] = whole
    store.redirects["/hop"] = other.url("/moved")
    hop = _record("hop", whole, at("/hop"))
    with credentials_from(hook):
        refused = _refusal(lambda: fetch(hop))
    sent = [seen for seen in other.seen if any(HARNESS_TOKEN in value for value in seen.headers.values())]
    if not other.seen:
        report.problems.append(f"the redirect to another host was not followed: {refused}")
    elif sent:
        report.problems.append("the credential was sent to another host, across a redirect")
    else:
        report.passed.append("a credential is never sent to another host across a redirect")


__all__ = ["HARNESS_TOKEN", "AdapterReport", "verify_adapter"]
