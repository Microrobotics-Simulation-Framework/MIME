"""Open locations, the hash the trust (design #229): the adapters MIME reads itself, an adapter a package or a script
adds, and what the resolver keeps for every adapter alike. Every server is on this machine.

- **https** with an institution's CA: a server whose certificate a test-made CA signed is refused without the CA's
  bundle and fetched with it (``MIME_CA_BUNDLE``); the standard proxy variables are honoured (a loopback ``CONNECT``
  proxy carries the request, and ``NO_PROXY`` takes it away again).
- **A credential bound to its host**: a fake adapter that needs one (the lab fixture's ``labstore``) is handed the
  token the machine's store holds for the location's host, and a redirect to a second loopback host name gets none;
  the token is in no refusal.
- **Resuming**: a stream cut short is resumed with ``Range`` and ``If-Range``, and a server that ignores the range is
  downloaded afresh; the full hash at the end is the check either way. **Retries** with backoff follow a ``503``.
- **package:** reads bytes inside an installed package that declares a ``mime.assets`` entry point, and no other;
  **file://** mirrors read a regular file by its sha256 or a name, never a link out of the mirror.
- **The registry**: an adapter a script registers with one call reads its scheme; a scheme nothing reads is refused
  where a record is written, and a lock names each adapter's package, which ``mime pull`` lists.

Each control fails on every run: the resolver planted to hand every hop the first host's credential sends the token
to the second host; an https adapter planted to read no range downloads afresh where the server would have resumed."""

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime import assets, credentials
from mime.assets import _adapters, _fetch
from mime.build import BuildRefused
from mime.credentials import Token
from mime.testing import HARNESS_TOKEN, ConnectProxy, LoopbackStore, certificates, verify_adapter

PAYLOAD = b"an asset's bytes, served on this machine. " * 2000
RUNS = 20


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MIME_OFFLINE", raising=False)
    monkeypatch.delenv("MIME_CA_BUNDLE", raising=False)
    monkeypatch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "locations.toml"))
    (tmp_path / "locations.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(_fetch, "BACKOFF_SECONDS", 0.0)
    return tmp_path / "cache"


def ref_of(data: bytes, *locations: str, name: str = "blob") -> assets.AssetRef:
    return assets.AssetRef(name, hashlib.sha256(data).hexdigest(), len(data), "CC0-1.0", locations)


def codes_and_text(action) -> tuple[list[str], str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes(), str(refused)
    raise AssertionError("not refused")


# --- https: an institution's CA, and the proxy variables ------------------------------------------------------------


@pytest.fixture(scope="module")
def tls(tmp_path_factory: pytest.TempPathFactory):
    return certificates(tmp_path_factory.mktemp("tls"), ("localhost", "assets.example.test", "127.0.0.1"))


def test_a_server_signed_by_an_institutions_ca_is_refused_without_its_bundle_and_fetched_with_it(cache, tls,
                                                                                               monkeypatch):
    with LoopbackStore("127.0.0.1", tls.server_context()) as server:
        server.files["/blob"] = PAYLOAD
        ref = ref_of(PAYLOAD, server.url("/blob", host="localhost"))
        found, text = codes_and_text(lambda: assets.fetch(ref))
        assert found == ["asset.unreachable"] and "MIME_CA_BUNDLE" in text and "not trusted" in text
        monkeypatch.setenv("MIME_CA_BUNDLE", str(tls.ca))
        assert assets.fetch(ref).read_bytes() == PAYLOAD


def test_the_proxy_variables_are_honoured_and_no_proxy_takes_the_proxy_away(cache, tls, monkeypatch):
    with LoopbackStore("127.0.0.1", tls.server_context()) as server:
        server.files["/blob"] = PAYLOAD
        proxy = ConnectProxy(server.port)
        try:
            monkeypatch.setenv("MIME_CA_BUNDLE", str(tls.ca))
            monkeypatch.setenv("HTTPS_PROXY", proxy.url)
            monkeypatch.setenv("https_proxy", proxy.url)
            monkeypatch.setenv("NO_PROXY", "")
            monkeypatch.setenv("no_proxy", "")
            # A name only the proxy resolves: the request can reach the server through it alone.
            ref = ref_of(PAYLOAD, f"https://assets.example.test:{server.port}/blob")
            assert assets.fetch(ref).read_bytes() == PAYLOAD
            assert proxy.asked == [f"assets.example.test:{server.port}"]
            other = ref_of(PAYLOAD + b"!", f"https://assets.example.test:{server.port}/blob", name="other")
            monkeypatch.setenv("NO_PROXY", "assets.example.test")
            monkeypatch.setenv("no_proxy", "assets.example.test")
            found, _ = codes_and_text(lambda: assets.fetch(other))
            assert found == ["asset.unreachable"] and len(proxy.asked) == 1, "NO_PROXY sent it past the proxy"
        finally:
            proxy.close()


# --- a credential bound to its host ---------------------------------------------------------------------------------


def labstore(url: str) -> str:
    return "labstore://" + url.partition("://")[2]


def test_the_lab_stores_adapter_is_registered_by_its_entry_point():
    assert "labstore" in assets.registered_schemes()
    assert assets.adapter_package("labstore") == "mime-lab-fixture"


def test_the_machines_store_hands_the_token_to_its_own_host_and_never_across_a_redirect(cache):
    credentials.store(credentials.host_service("127.0.0.1"), Token("host:127.0.0.1", HARNESS_TOKEN))
    with LoopbackStore() as store, LoopbackStore("localhost") as other:
        store.files["/private"] = PAYLOAD
        store.tokens["/private"] = HARNESS_TOKEN
        assert assets.fetch(ref_of(PAYLOAD, labstore(store.url("/private")))).read_bytes() == PAYLOAD
        moved = PAYLOAD + b"moved"
        other.files["/moved"] = moved
        store.redirects["/hop"] = other.url("/moved")
        assert assets.fetch(ref_of(moved, labstore(store.url("/hop")), name="moved")).read_bytes() == moved
        assert other.seen and not [seen for seen in other.seen if "Authorization" in seen.headers]
        store.tokens["/wrong"] = "another"
        store.files["/wrong"] = PAYLOAD + b"wrong"
        found, text = codes_and_text(lambda: assets.fetch(ref_of(PAYLOAD + b"wrong", labstore(store.url("/wrong")),
                                                                 name="wrong")))
        assert found == ["asset.unreachable"] and "refused the login" in text and HARNESS_TOKEN not in text


def test_the_lab_stores_adapter_and_mimes_https_pass_the_conformance_harness(cache):
    from mime_lab_fixture.assets import LabStore

    for report in (verify_adapter(_adapters.HTTPS), verify_adapter(LabStore(), location_for=labstore)):
        assert report.conforms, report.problems
    assert len(verify_adapter(LabStore(), location_for=labstore).passed) == 7


@contextmanager
def every_hop_given_the_first_hosts_credential() -> Iterator[None]:
    """The plant: a resolver that asks the store once, for the first host, and hands that credential to every hop."""
    with pytest.MonkeyPatch.context() as patch:
        honest = _fetch._request  # pyright: ignore[reportPrivateUsage]
        first: list[object] = []

        def carried(ref, adapter, location, start, validator):  # type: ignore[no-untyped-def]
            held = honest(ref, adapter, location, start, validator)
            if held.credential is not None and not first:
                first.append(held.credential)
            from dataclasses import replace

            return replace(held, credential=first[0]) if first else held

        patch.setattr(_fetch, "_request", carried)
        yield


def test_control_a_resolver_that_carries_a_credential_across_a_redirect_is_caught_every_run(tmp_path, monkeypatch):
    from mime_lab_fixture.assets import LabStore

    monkeypatch.setattr(_fetch, "BACKOFF_SECONDS", 0.0)
    caught = 0
    for _ in range(RUNS):
        with every_hop_given_the_first_hosts_credential():
            report = verify_adapter(LabStore(), location_for=labstore)
        caught += any("sent to another host" in problem for problem in report.problems)
    assert caught == RUNS


# --- resuming and retrying ------------------------------------------------------------------------------------------


def test_a_cut_stream_is_resumed_with_a_range_and_the_full_hash_checked(cache):
    with LoopbackStore() as server:
        server.files["/blob"] = PAYLOAD
        server.cuts["/blob"] = 20_000
        assert assets.fetch(ref_of(PAYLOAD, server.url("/blob"))).read_bytes() == PAYLOAD
        second = server.seen[1].headers
        assert second.get("Range") == "bytes=20000-" and second.get("If-Range", "").startswith('"')


def test_a_server_that_ignores_the_range_is_downloaded_afresh(cache):
    with LoopbackStore() as server:
        server.files["/blob"] = PAYLOAD
        server.cuts["/blob"] = 20_000
        server.ignore_range = True
        assert assets.fetch(ref_of(PAYLOAD, server.url("/blob"))).read_bytes() == PAYLOAD
        assert len(server.seen) == 2


@contextmanager
def range_never_asked() -> Iterator[None]:
    """The plant: an https adapter that says it reads no range, so a cut download starts again from its first byte."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_adapters.HttpsAdapter, "ranged", False)
        yield


def resumed(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
        patch.setattr(_fetch, "BACKOFF_SECONDS", 0.0)
        with LoopbackStore() as server:
            server.files["/blob"] = PAYLOAD
            server.cuts["/blob"] = 20_000
            assets.fetch(ref_of(PAYLOAD, server.url("/blob")))
            assert server.seen[1].headers.get("Range") == "bytes=20000-"


def test_control_a_resolver_that_never_resumes_is_caught_every_run(tmp_path):
    caught = 0
    for run in range(RUNS):
        with range_never_asked():
            try:
                resumed(tmp_path / str(run))
            except AssertionError:
                caught += 1
    resumed(tmp_path / "unplanted")
    assert caught == RUNS


def test_a_503_is_asked_again_after_its_wait_and_a_404_is_not(cache):
    with LoopbackStore() as server:
        server.files["/blob"] = PAYLOAD
        server.statuses["/blob"] = [503, 503]
        assert assets.fetch(ref_of(PAYLOAD, server.url("/blob"))).read_bytes() == PAYLOAD
        assert len(server.seen) == 3
        found, text = codes_and_text(lambda: assets.fetch(ref_of(b"gone", server.url("/gone"), name="gone")))
        assert found == ["asset.unreachable"] and "no file at that address" in text and len(server.seen) == 4


# --- package: and file:// mirrors -----------------------------------------------------------------------------------


def test_a_package_location_reads_bytes_inside_a_package_that_declares_itself_a_source(cache):
    held = assets.reference("labx_marker.usda")
    assert held.locations == ("package:labx/marker.usda",)
    assert hashlib.sha256(assets.fetch(held).read_bytes()).hexdigest() == held.sha256
    found, text = codes_and_text(lambda: assets.fetch(ref_of(b"x", "package:os/x", name="x")))
    assert found == ["asset.unreachable"] and "no installed package declares the source 'os'" in text


def test_a_mirror_reads_a_regular_file_by_its_sha256_or_its_name_and_never_a_link_out(cache, tmp_path):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    ref = ref_of(PAYLOAD, mirror.as_uri() + "/", name="table.npz")
    (mirror / ref.sha256).write_bytes(PAYLOAD)
    assert assets.fetch(ref).read_bytes() == PAYLOAD
    named = ref_of(PAYLOAD + b"2", mirror.as_uri() + "/", name="named.npz")
    (mirror / "named.npz").write_bytes(PAYLOAD + b"2")
    assert assets.fetch(named).read_bytes() == PAYLOAD + b"2"
    outside = tmp_path / "outside.npz"
    outside.write_bytes(PAYLOAD + b"3")
    linked = ref_of(PAYLOAD + b"3", mirror.as_uri() + "/", name="linked.npz")
    os.symlink(outside, mirror / linked.sha256)
    found, _ = codes_and_text(lambda: assets.fetch(linked))
    assert found == ["asset.unreachable"]


# --- the registry ---------------------------------------------------------------------------------------------------


class Reversed:
    """A script's adapter: ``rev://<text>`` serves the text's bytes reversed."""

    scheme = "rev"
    needs_credential = False
    ranged = False

    def open(self, location: str, request: assets.Request) -> assets.Opened:
        import io

        return assets.Opened(io.BytesIO(location.removeprefix("rev://")[::-1].encode()))


def test_a_scheme_nothing_reads_is_refused_and_one_call_registers_a_scripts_adapter(cache, monkeypatch):
    found, _ = codes_and_text(lambda: ref_of(b"cba", "rev://abc"))
    assert found == ["asset.reference"]
    monkeypatch.setitem(_adapters._registered, "rev", _adapters._Registered(Reversed(), None))  # pyright: ignore
    assert assets.fetch(ref_of(b"cba", "rev://abc")).read_bytes() == b"cba"
    assert dict(assets.with_adapters(ref_of(b"cba", "rev://abc")).adapters) == {"rev": ""}
    labs = ref_of(b"x", "labstore://127.0.0.1:1/x")
    assert dict(assets.with_adapters(labs).adapters) == {"labstore": "mime-lab-fixture"}
    with pytest.raises(ValueError, match="MIME's own"):
        assets.register_adapter(type("Https", (), {"scheme": "https"})())  # type: ignore[arg-type]


def test_pull_lists_the_adapter_package_a_lock_names_and_installs_nothing():
    from mime.saved._pull import _adapters_line, adapters_needed

    lock = {"assets": [{"name": "a", "sha256": "ab" * 32, "size": 1, "licence": "CC0-1.0",
                        "locations": ["s3x://bucket/a"], "adapters": {"s3x": "mime-storage"}},
                       {"name": "b", "sha256": "cd" * 32, "size": 1, "licence": "CC0-1.0",
                        "locations": ["labstore://h/b"], "adapters": {"labstore": "mime-lab-fixture"}}]}
    needed = adapters_needed(lock)  # type: ignore[arg-type]
    assert needed == [("s3x", "mime-storage")], "the installed lab store's adapter is not listed"
    line = _adapters_line(needed)
    assert line is not None and "pip install mime-storage" in line and "placed in the asset cache" in line


def test_a_file_placed_by_its_hash_is_found_without_the_adapter_its_location_needs(cache, monkeypatch):
    """A lock naming a scheme no installed adapter reads still loads its file from the cache: a lab copies it in by
    hand, and the record's package is only listed, never needed."""
    record = {**ref_of(PAYLOAD, "https://example.org/x").to_record(), "locations": ["s3x://bucket/table.npz"],
              "adapters": {"s3x": "mime-storage"}}
    ref = assets.AssetRef.from_record(record)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ref.sha256).write_bytes(PAYLOAD)
    monkeypatch.setenv("MIME_OFFLINE", "1")
    assert assets.fetch(ref).read_bytes() == PAYLOAD and assets.locate(ref).is_file()
    monkeypatch.delenv("MIME_OFFLINE")
    other = assets.AssetRef.from_record({**record, "sha256": hashlib.sha256(b"not here").hexdigest(), "size": 8})
    found, text = codes_and_text(lambda: assets.fetch(other))
    assert found == ["asset.unreachable"] and "no adapter reads the scheme 's3x' here" in text and "mime-storage" in text
