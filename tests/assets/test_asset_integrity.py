"""What happens to a file after it is first stored, and who may touch it: the review's findings F1 to F5 and F8 of PR #44.

Each scenario is a function of the rig and asserts. Each control runs it under a plant twenty times and every run must
fail: a plant that is caught only most of the time would be a flaky control. Nothing leaves the machine.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pooch
import pytest

from mime import assets
from mime.assets import _cache, _fetch
from mime.build import BuildRefused
from assets_rig import Rig, Server, reference, run_with_rig

PAYLOAD = b"a wall table, in miniature" * 1000
RUNS = 20
HERE = Path(__file__).resolve().parent


def caught(action: Callable[[], object]) -> BuildRefused:
    try:
        action()
    except BuildRefused as refused:
        return refused
    raise AssertionError("not refused")


def stored(rig: Rig, ref: assets.AssetRef) -> Path:
    return rig.cache / ref.sha256


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    held = Rig(tmp_path, monkeypatch)
    yield held
    held.server.close()


def flip_a_byte_keeping_size_and_time(path: Path) -> None:
    """What silent storage corruption looks like: other bytes, the same size, the same modification time."""
    seen = path.stat()
    flipped = bytearray(path.read_bytes())
    flipped[7] ^= 1
    path.write_bytes(bytes(flipped))
    os.utime(path, ns=(seen.st_atime_ns, seen.st_mtime_ns))


def planted_runs(tmp_path: Path, scenario: Callable[[Rig], None], plant: Callable[[], object]) -> int:
    """How many of ``RUNS`` runs of ``scenario`` under the plant fail, and so detect it."""
    detected = 0
    for run in range(RUNS):
        try:
            with cast("Iterator[None]", plant()):
                run_with_rig(scenario, tmp_path / str(run))
        except (AssertionError, BuildRefused):
            detected += 1
    return detected


def every_run_passes(tmp_path: Path, scenario: Callable[[Rig], None]) -> None:
    for run in range(RUNS):
        run_with_rig(scenario, tmp_path / f"clean{run}")


# --- F1: a corrupt cached file is found, refused by locate, and repaired by fetch ------------------------------------


def corrupt_file_is_found_refused_and_repaired(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    path = assets.fetch(ref)
    flip_a_byte_keeping_size_and_time(path)
    assert assets.locate(ref) == path, "the cheap check is expected to pass: this is the case it cannot see"
    assert assets.check(ref) == "corrupt"
    refused = caught(lambda: assets.locate(ref, folder="run"))
    (refusal,) = refused.refusals
    assert refusal.code == "asset.missing" and refusal.action == "Run: mime assets fetch ./run/"
    assert not assets.is_cached(ref)
    assert rig.requests == 1
    assert assets.fetch(ref) == path and path.read_bytes() == PAYLOAD and rig.requests == 2
    assert assets.locate(ref) == path and assets.check(ref) == "ok"


def fetch_hashes_a_cached_file_before_calling_it_verified(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    path = assets.fetch(ref)
    flip_a_byte_keeping_size_and_time(path)
    assert assets.fetch(ref) == path
    assert path.read_bytes() == PAYLOAD and rig.requests == 2, "fetch reported a file it had not hashed"


def a_clean_fetch_of_a_cached_file_makes_no_second_request(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    path = assets.fetch(ref)
    assert assets.fetch(ref) == path and assets.fetch(ref) == path and rig.requests == 1


@contextmanager
def check_that_leaves_no_mark() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        honest = _cache.write_record
        patch.setattr(_cache, "write_record", lambda ref, origin, *, failed=False: None if failed else honest(ref, origin))
        yield


@contextmanager
def fetch_that_trusts_the_cheap_check() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assets, "fetch", lambda ref, *, report=None: _fetch.resolve(ref, report=report))
        yield


def test_a_corrupt_file_is_found_by_check_refused_by_locate_and_repaired_by_fetch(rig: Rig):
    corrupt_file_is_found_refused_and_repaired(rig)


def test_fetch_hashes_a_cached_file_before_it_calls_it_verified(rig: Rig):
    fetch_hashes_a_cached_file_before_calling_it_verified(rig)


def test_a_fetch_of_an_unchanged_cached_file_makes_no_second_request(rig: Rig):
    a_clean_fetch_of_a_cached_file_makes_no_second_request(rig)


def test_the_command_says_hashed_in_full_only_when_it_was(rig: Rig, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]):
    from mime.cli import main
    from mime.assets import _known

    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"), name="served_blob")
    monkeypatch.setattr(assets, "known", lambda: {ref.name: ref})
    monkeypatch.setattr(_known, "known", lambda: {ref.name: ref})
    assert main(["assets", "fetch", "served_blob"]) == 0
    flip_a_byte_keeping_size_and_time(stored(rig, ref))
    assert main(["assets", "fetch", "served_blob"]) == 0
    assert "hashed in full" in capsys.readouterr().out
    assert stored(rig, ref).read_bytes() == PAYLOAD and rig.requests == 2


def test_control_a_check_that_leaves_no_mark_is_detected_every_run(tmp_path: Path):
    every_run_passes(tmp_path, corrupt_file_is_found_refused_and_repaired)
    assert planted_runs(tmp_path, corrupt_file_is_found_refused_and_repaired, check_that_leaves_no_mark) == RUNS


def test_control_a_fetch_that_trusts_the_cheap_check_is_detected_every_run(tmp_path: Path):
    every_run_passes(tmp_path, fetch_hashes_a_cached_file_before_calling_it_verified)
    assert planted_runs(tmp_path, fetch_hashes_a_cached_file_before_calling_it_verified,
                        fetch_that_trusts_the_cheap_check) == RUNS


# --- F2: several processes fetching at once --------------------------------------------------------------------------


PROCESSES = 24


def run_concurrently(tmp_path: Path, *, unlocked: bool) -> tuple[list[str], dict[str, int], Server]:
    """``PROCESSES`` Python processes fetch the same two small assets from one slow local server, released together.
    Returns each process's one-line result, the requests per path, and the server (closed)."""
    server = Server()
    server.delay = 1.0
    payloads = {"/one": b"the first table" * 700, "/two": b"the second table" * 900}
    server.files.update(payloads)
    refs = [reference(payload, server.url(path), name=path.strip("/")) for path, payload in payloads.items()]
    cache, go = tmp_path / "cache", tmp_path / "go"
    environment = {**os.environ, "MIME_ASSETS_DIR": str(cache), "JAX_PLATFORMS": "cpu", "NO_PROXY": "127.0.0.1",
                   "no_proxy": "127.0.0.1"}
    environment.pop("MIME_OFFLINE", None)
    records = json.dumps([ref.to_record() for ref in refs])
    command = [sys.executable, str(HERE / "concurrent_fetch.py")]
    children: list[subprocess.Popen[str]] = []
    try:
        for index in range(PROCESSES):
            ordered = records if index % 2 == 0 else json.dumps([ref.to_record() for ref in reversed(refs)])
            child = subprocess.Popen([*command, str(tmp_path / f"ready{index}"), str(go), *(["--unlocked"] * unlocked)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, env=environment)
            assert child.stdin is not None
            child.stdin.write(ordered)
            child.stdin.close()
            children.append(child)
        waited = 0.0
        while sum((tmp_path / f"ready{index}").exists() for index in range(PROCESSES)) < PROCESSES:
            assert waited < 240, "the processes did not start"
            time.sleep(0.05)
            waited += 0.05
        go.touch()
        results = []
        for child in children:
            out, err = child.communicate(timeout=240)
            results.append(out.strip() or f"no output: {err[-300:]}")
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
        server.close()
    requested = {path: server.requests.count(f"GET {path}") for path in payloads}
    return results, requested, server


def test_twenty_four_processes_fetching_the_same_assets_all_succeed_and_download_each_once(tmp_path: Path,
                                                                                          monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIME_OFFLINE", raising=False)
    results, requested, _ = run_concurrently(tmp_path, unlocked=False)
    assert results == ["ok"] * PROCESSES
    assert requested == {"/one": 1, "/two": 1}, requested
    cache = tmp_path / "cache"
    leftovers = sorted(entry.name for entry in cache.iterdir() if ".incoming-" in entry.name or entry.name.endswith(".partial"))
    assert leftovers == []


def test_control_without_the_lock_the_same_processes_fail_or_download_twice(tmp_path: Path):
    results, requested, _ = run_concurrently(tmp_path, unlocked=True)
    assert results != ["ok"] * PROCESSES or any(count > 1 for count in requested.values()), (results, requested)


# --- F3: a file a person placed is never deleted or overwritten --------------------------------------------------------


def placed_files_are_left_alone(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"), name="wall_R2.500")
    rig.cache.mkdir(parents=True)
    wrong_same_size = b"x" * len(PAYLOAD)
    for contents in (wrong_same_size, PAYLOAD[: len(PAYLOAD) // 2], PAYLOAD + b"y", b""):
        for offline in ("0", "1"):
            stored(rig, ref).write_bytes(contents)
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("MIME_OFFLINE", offline)
                refused = caught(lambda: assets.fetch(ref))
            (refusal,) = refused.refusals
            assert refusal.code == "asset.placed-file", refusal.code
            assert stored(rig, ref).name in refusal.message and "does not match" in refusal.message
            assert "`MIME_ASSETS_DIR`" in refusal.message and "did not write" in refusal.message
            assert str(stored(rig, ref)) in (refusal.action or "")
            assert stored(rig, ref).read_bytes() == contents, "a file a person placed was changed or removed"
            assert rig.requests == 0, "the download started over a file a person placed"
            assert assets.check(ref) == "corrupt" and stored(rig, ref).read_bytes() == contents
            assert caught(lambda: assets.locate(ref)).codes() == ["asset.missing"]
            assert stored(rig, ref).read_bytes() == contents


def a_placed_file_that_is_right_is_accepted_and_never_replaced(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    rig.cache.mkdir(parents=True)
    stored(rig, ref).write_bytes(PAYLOAD)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MIME_OFFLINE", "1")
        assert assets.fetch(ref) == stored(rig, ref) and assets.locate(ref) == stored(rig, ref)
    stored(rig, ref).write_bytes(b"q" * len(PAYLOAD))
    assert caught(lambda: assets.fetch(ref)).codes() == ["asset.placed-file"]
    assert stored(rig, ref).read_bytes() == b"q" * len(PAYLOAD) and rig.requests == 0


@contextmanager
def everything_is_mimes_to_replace() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "mime_wrote", lambda ref: True)
        yield


def test_a_file_a_person_placed_with_other_bytes_or_half_copied_is_refused_and_left_alone(rig: Rig):
    placed_files_are_left_alone(rig)


def test_a_right_file_a_person_placed_is_accepted_and_a_later_wrong_one_is_not_replaced(rig: Rig):
    a_placed_file_that_is_right_is_accepted_and_never_replaced(rig)


def test_a_file_that_appears_while_the_download_runs_is_not_overwritten(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    honest = _fetch._download  # pyright: ignore[reportPrivateUsage]

    def person_copies_meanwhile(location: str, held: assets.AssetRef, directory: Path) -> Path:
        arrived = honest(location, held, directory)
        stored(rig, held).write_bytes(b"half a copy")
        return arrived

    monkeypatch.setattr(_fetch, "_download", person_copies_meanwhile)
    assert caught(lambda: assets.fetch(ref)).codes() == ["asset.placed-file"]
    assert stored(rig, ref).read_bytes() == b"half a copy"


def test_control_a_resolver_that_replaces_whatever_is_there_is_detected_every_run(tmp_path: Path):
    every_run_passes(tmp_path, placed_files_are_left_alone)
    assert planted_runs(tmp_path, placed_files_are_left_alone, everything_is_mimes_to_replace) == RUNS


def test_the_directory_is_one(rig: Rig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """``MIME_ASSETS_DIR`` replaces the default; a file in the default is not looked at, one in the variable's
    directory is accepted by its name (the sha256) and its hash."""
    default = tmp_path / "platform_default"
    monkeypatch.setattr(pooch, "os_cache", lambda project: default)  # pyright: ignore[reportUnknownLambdaType]
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    (default / "assets").mkdir(parents=True)
    (default / "assets" / ref.sha256).write_bytes(PAYLOAD)
    assert not assets.is_cached(ref) and assets.cache_dir() == rig.cache
    rig.cache.mkdir()
    stored(rig, ref).write_bytes(PAYLOAD)
    monkeypatch.setenv("MIME_OFFLINE", "1")
    assert assets.fetch(ref) == stored(rig, ref) and rig.requests == 0


# --- F4: the loopback-only http rule ---------------------------------------------------------------------------------


def free_loopback_server(host: str) -> Server:
    try:
        return Server(host)
    except OSError:
        pytest.skip(f"cannot listen on {host}")


REVIEW_STRING = "http://127.0.0.2:{port}\\@127.0.0.1/x"
REFUSED_LOCATIONS = [
    REVIEW_STRING.format(port=8000),
    "http://127.0.0.2:8000%5C@127.0.0.1/x",
    "http://127.0.0.1:8000\\@evil.example/x",
    "http://evil.example\\@127.0.0.1/x",
    "http://user@127.0.0.1/x",
    "http://user:password@127.0.0.1:8000/x",
    "http://127.0.0.1@evil.example/x",
    "http://127.0.0.1%40evil.example/x",
    "http://127.0.0.1:8000/a b",
    "http://127.0.0.1:8000/a\tb",
    "http://127.0.0.1:8000/a\nb",
    "http://127.0.0.2:8000/x",
    "http://localhost.evil.example/x",
    "http://[::1].evil.example/x",
    "http://[::1]%5C@evil.example/x",
    "http://127.0.0.1:99999999/x",
    "https://example.org\\@evil.example/x",
    "https://user@example.org/x",
    "https://example.org/a\\b",
    "https://example.org%5C@evil.example/x",
    "http://",
    "https://",
]
ACCEPTED_LOCATIONS = [
    "http://127.0.0.1:8000/x",
    "http://localhost:8000/x",
    "http://[::1]:8000/x",
    "http://[::1]/x",
    "http://127.0.0.1:8000/a%5Cb",
    "https://huggingface.co/datasets/NickERoy/mime-wall-tables/resolve/1f7b5a4e/r40/wall_R1.526.npz",
    "https://example.org:8443/x?y=1",
]


@pytest.mark.parametrize("location", REFUSED_LOCATIONS)
def test_a_location_the_two_parsers_could_read_differently_is_refused(location: str):
    assert assets.location_problem(location) is not None, location
    with pytest.raises(BuildRefused, match="asset.reference"):
        reference(PAYLOAD, location)


@pytest.mark.parametrize("location", ACCEPTED_LOCATIONS)
def test_the_loopback_and_https_forms_are_accepted(location: str):
    assert assets.location_problem(location) is None, location


def old_rule(location: str) -> str | None:
    """The rule before the fix: the standard library's parse of the host, and nothing more."""
    from urllib.parse import urlsplit

    parts = urlsplit(location)
    if parts.scheme == "https" and parts.hostname:
        return None
    if parts.scheme == "http" and parts.hostname in {"127.0.0.1", "localhost", "::1"}:
        return None
    return "refused"


def the_review_string_is_never_contacted(rig: Rig) -> None:
    """A reference built around the check (as a lock read without it might be) still never reaches the other host."""
    other = free_loopback_server("127.0.0.2")
    try:
        other.files["/blob"] = PAYLOAD
        location = f"http://127.0.0.2:{other.server_address[1]}\\@127.0.0.1/blob"
        ref = reference(PAYLOAD, rig.server.url("/blob"))
        built_around = object.__new__(assets.AssetRef)
        for field, value in (("name", ref.name), ("sha256", ref.sha256), ("size", ref.size), ("licence", ref.licence),
                             ("locations", (location,))):
            object.__setattr__(built_around, field, value)
        refused = caught(lambda: assets.fetch(built_around))
        assert refused.codes() == ["asset.unreachable"] and "not used" in str(refused)
        assert other.requests == [] and rig.requests == 0, (other.requests, rig.requests)
    finally:
        other.close()


def the_old_rule_accepted_the_review_string() -> None:
    assert old_rule(REVIEW_STRING.format(port=8000)) is None
    assert assets.location_problem(REVIEW_STRING.format(port=8000)) is not None


@contextmanager
def the_check_before_contact_removed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "location_problem", lambda location: None)
        yield


def test_a_refused_location_in_a_reference_built_around_the_check_is_never_contacted(rig: Rig):
    the_review_string_is_never_contacted(rig)
    the_old_rule_accepted_the_review_string()


def test_control_a_resolver_that_does_not_check_before_contact_is_detected_every_run(tmp_path: Path):
    every_run_passes(tmp_path, the_review_string_is_never_contacted)
    assert planted_runs(tmp_path, the_review_string_is_never_contacted, the_check_before_contact_removed) == RUNS


def test_control_the_old_rule_accepted_the_locations_that_only_the_parser_difference_hides():
    """The refusals above are not vacuous: the standard library's parse of the host alone accepted these."""
    hidden = [REFUSED_LOCATIONS[index] for index in (0, 1, 3, 4, 5)]
    assert all(old_rule(location) is None for location in hidden), hidden


def test_without_the_check_the_review_string_does_reach_the_other_host(rig: Rig):
    """The control for the contact test: the request does arrive at 127.0.0.2 when nothing stops it."""
    other = free_loopback_server("127.0.0.2")
    try:
        built_around = object.__new__(assets.AssetRef)
        location = f"http://127.0.0.2:{other.server_address[1]}\\@127.0.0.1/blob"
        for field, value in (("name", "blob"), ("sha256", hashlib.sha256(PAYLOAD).hexdigest()), ("size", len(PAYLOAD)),
                             ("licence", "CC0-1.0"), ("locations", (location,))):
            object.__setattr__(built_around, field, value)
        with the_check_before_contact_removed():
            with pytest.raises(BuildRefused):
                assets.fetch(built_around)
        assert other.requests and rig.requests == 0, other.requests
    finally:
        other.close()


# --- F5: the offline switch ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "yes", "on", "ON", " on "])
def test_the_usual_true_spellings_close_the_network(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("MIME_OFFLINE", value)
    assert assets.offline()


@pytest.mark.parametrize("value", ["", "0", "false", "False", "no", "off", "OFF"])
def test_the_usual_false_spellings_and_unset_leave_it_open(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("MIME_OFFLINE", value)
    assert not assets.offline()
    monkeypatch.delenv("MIME_OFFLINE")
    assert not assets.offline()


@pytest.mark.parametrize("value", ["enabled", "y", "2", "t", "o n", "truee", "-1"])
def test_an_unrecognised_value_is_refused_and_is_not_online(rig: Rig, monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("MIME_OFFLINE", value)
    assert caught(assets.offline).codes() == ["asset.offline-setting"]
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    assert caught(lambda: assets.fetch(ref)).codes() == ["asset.offline-setting"]
    assert rig.requests == 0 and not stored(rig, ref).exists()


def offline_spellings_never_reach_the_network(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    for value, code in (("on", "asset.offline"), ("ON", "asset.offline"), ("enabled", "asset.offline-setting")):
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("MIME_OFFLINE", value)
            assert caught(lambda: assets.fetch(ref)).codes() == [code]
    assert rig.requests == 0 and not stored(rig, ref).exists()


@contextmanager
def the_old_offline_switch() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "offline", lambda: os.environ.get("MIME_OFFLINE", "").strip().lower() in {"1", "true", "yes"})
        yield


def test_control_the_old_switch_that_failed_open_is_detected_every_run(tmp_path: Path):
    every_run_passes(tmp_path, offline_spellings_never_reach_the_network)
    assert planted_runs(tmp_path, offline_spellings_never_reach_the_network, the_old_offline_switch) == RUNS


# --- F8: which layer refuses a wrong body ----------------------------------------------------------------------------


def served_by_redirect(rig: Rig) -> assets.AssetRef:
    """A location that answers 302 to a path that serves other bytes: the redirect hides the wrong body from the
    location as written."""
    rig.server.files["/wrong"] = b"other bytes than the reference names" * 50
    rig.server.redirects["/blob"] = "/wrong"
    return reference(PAYLOAD, rig.server.url("/blob"))


def pooch_refuses_the_redirected_body(rig: Rig) -> None:
    ref = served_by_redirect(rig)
    refusal = caught(lambda: assets.fetch(ref))
    assert refusal.codes() == ["asset.hash-mismatch"]
    assert "stored other bytes" not in str(refusal) and "does not match" not in str(refusal)
    assert "other bytes than the reference names" in str(refusal), "pooch's own check did not refuse it"
    assert rig.server.requests == ["GET /blob", "GET /wrong"] and not stored(rig, ref).exists()


def the_second_layer_refuses_when_pooch_is_not_told_the_hash(rig: Rig) -> None:
    ref = served_by_redirect(rig)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "_download", download_without_the_hash)
        refusal = caught(lambda: assets.fetch(ref))
    assert refusal.codes() == ["asset.hash-mismatch"]
    assert "stored other bytes than the reference names" in str(refusal), "the re-hash did not refuse it"
    assert not stored(rig, ref).exists() and not list(rig.cache.glob("*.incoming-*"))


def pooch_is_given_the_references_hash(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    handed: list[object] = []
    honest = pooch.retrieve

    def spy(*args: object, **kwargs: object) -> object:
        handed.append(kwargs.get("known_hash"))
        return honest(*args, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pooch, "retrieve", spy)
        assets.fetch(ref)
    assert handed == [f"sha256:{ref.sha256}"], handed


def download_without_the_hash(location: str, held: assets.AssetRef, directory: Path) -> Path:
    return Path(str(pooch.retrieve(url=location, known_hash=None, fname=held.sha256, path=directory)))


@contextmanager
def pooch_not_told_the_hash() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "_download", download_without_the_hash)
        yield


@contextmanager
def the_rehash_removed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "sha256_of_file", lambda path: hashlib.sha256(PAYLOAD).hexdigest())
        yield


def test_pooch_alone_refuses_a_redirected_wrong_body(rig: Rig):
    pooch_refuses_the_redirected_body(rig)


def test_the_rehash_alone_refuses_it_when_pooch_is_not_told_the_hash(rig: Rig):
    the_second_layer_refuses_when_pooch_is_not_told_the_hash(rig)


def test_pooch_is_handed_the_hash(rig: Rig):
    pooch_is_given_the_references_hash(rig)


def test_control_removing_pooch_hash_alone_is_visible_every_run(tmp_path: Path):
    """The layer that refused is named by the message, so a change of layer shows without the other being removed."""
    every_run_passes(tmp_path, pooch_refuses_the_redirected_body)
    every_run_passes(tmp_path, pooch_is_given_the_references_hash)
    assert planted_runs(tmp_path, pooch_refuses_the_redirected_body, pooch_not_told_the_hash) == RUNS
    assert planted_runs(tmp_path, pooch_is_given_the_references_hash, pooch_not_told_the_hash) == RUNS


def test_control_removing_the_rehash_alone_is_visible_every_run(tmp_path: Path):
    every_run_passes(tmp_path, the_second_layer_refuses_when_pooch_is_not_told_the_hash)
    assert planted_runs(tmp_path, the_second_layer_refuses_when_pooch_is_not_told_the_hash, the_rehash_removed) == RUNS
