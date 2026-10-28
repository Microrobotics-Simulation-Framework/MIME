"""``mime.assets``: a reference names a file by sha256; ``fetch`` stores it in a cache keyed by the hash; ``locate``,
which a build calls, never downloads.

Nothing here leaves the machine: the files are served by a local HTTP server in a thread, which counts its requests.
Each scenario is a function of the rig and asserts; the controls run it under a plant (a skipped hash check, a
``locate`` that downloads, a fallback that is disabled) twenty times, and each time it must fail.
"""

import dataclasses
import hashlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pooch
import pytest

from mime import assets
from mime.assets import _fetch, _known
from mime.build import BuildRefused
from assets_rig import Rig, dead_url, reference, run_with_rig

PAYLOAD = b"a wall table, in miniature" * 1000


def codes_of(action: Callable[[], object]) -> list[str]:
    """The refusal codes ``action`` raises; ``AssertionError`` if it does not refuse (a plain assertion, so that a plant
    which stops the refusal is a failed assertion, which the controls look for)."""
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def stored(rig: Rig, ref: assets.AssetRef) -> Path:
    return rig.cache / ref.sha256


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    held = Rig(tmp_path, monkeypatch)
    yield held
    held.server.close()


# --- scenarios -------------------------------------------------------------------------------------------------------


def fetch_then_hit(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    assert not assets.is_cached(ref)
    path = assets.fetch(ref)
    assert path == stored(rig, ref) and path.read_bytes() == PAYLOAD
    assert rig.requests == 1 and assets.is_cached(ref) and assets.check(ref) == "ok"
    assert assets.fetch(ref) == path and assets.locate(ref) == path
    assert rig.requests == 1, "a verified cache hit made a request"


def wrong_hash_is_refused(rig: Rig) -> None:
    rig.server.files["/blob"] = b"other bytes than the reference names" * 50
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    assert codes_of(lambda: assets.fetch(ref)) == ["asset.hash-mismatch"]
    assert rig.requests == 1
    assert not stored(rig, ref).exists() and not assets.is_cached(ref), "a wrong file was kept"


def dead_first_location_falls_through(rig: Rig, dead: str) -> None:
    rig.server.files["/good"] = PAYLOAD
    ref = reference(PAYLOAD, dead, rig.server.url("/good"))
    assert assets.fetch(ref) == stored(rig, ref)
    assert stored(rig, ref).read_bytes() == PAYLOAD and assets.is_cached(ref)


def locate_never_downloads(rig: Rig) -> None:
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    assert codes_of(lambda: assets.locate(ref)) == ["asset.missing"]
    assert rig.requests == 0 and not stored(rig, ref).exists(), "locate downloaded"
    assets.fetch(ref)
    before = rig.requests
    assert assets.locate(ref) == stored(rig, ref)
    assert rig.requests == before


# --- the behaviours --------------------------------------------------------------------------------------------------


def test_a_fetch_stores_the_file_by_its_hash_and_a_verified_hit_makes_no_request(rig: Rig):
    fetch_then_hit(rig)


def test_a_wrong_hash_is_refused_and_nothing_is_stored(rig: Rig):
    wrong_hash_is_refused(rig)


@pytest.mark.parametrize("kind", ["not found", "connection refused"])
def test_a_dead_first_location_falls_to_the_second(rig: Rig, kind: str):
    dead_first_location_falls_through(rig, rig.server.url("/gone") if kind == "not found" else dead_url())


def test_a_wrong_first_location_falls_to_a_right_second(rig: Rig):
    rig.server.files["/wrong"] = b"stale mirror" * 100
    rig.server.files["/good"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/wrong"), rig.server.url("/good"))
    assert assets.fetch(ref).read_bytes() == PAYLOAD
    assert rig.server.requests == ["GET /wrong", "GET /good"]


def test_every_location_dead_is_refused_naming_each(rig: Rig):
    first, second = dead_url("/a"), rig.server.url("/missing")
    ref = reference(PAYLOAD, first, second)
    with pytest.raises(BuildRefused) as refused:
        assets.fetch(ref)
    assert refused.value.codes() == ["asset.unreachable"]
    assert first in str(refused.value) and second in str(refused.value)


def test_offline_refuses_and_makes_no_request(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    monkeypatch.setenv("MIME_OFFLINE", "1")
    assert assets.offline()
    assert codes_of(lambda: assets.fetch(ref)) == ["asset.offline"]
    assert rig.requests == 0 and not stored(rig, ref).exists()


def test_offline_still_accepts_a_file_copied_into_the_cache(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    rig.cache.mkdir()
    stored(rig, ref).write_bytes(PAYLOAD)
    assert codes_of(lambda: assets.locate(ref)) == ["asset.missing"]
    monkeypatch.setenv("MIME_OFFLINE", "1")
    assert assets.fetch(ref) == stored(rig, ref) and assets.locate(ref) == stored(rig, ref)
    assert rig.requests == 0


def test_locate_never_makes_a_request(rig: Rig):
    locate_never_downloads(rig)


def test_the_first_location_is_the_primary_and_the_backups_are_tried_only_if_it_fails(rig: Rig):
    rig.server.files["/primary"] = PAYLOAD
    rig.server.files["/backup"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/primary"), rig.server.url("/backup"))
    lines: list[str] = []
    assert assets.resolve(ref, report=lines.append) == stored(rig, ref)
    assert rig.server.requests == ["GET /primary"], "a backup was asked while the primary served"
    assert len(lines) == 1 and "/primary" in lines[0] and "26 kB" in lines[0]


def test_resolve_takes_the_cache_first_then_a_file_already_there_then_the_locations(rig: Rig):
    """The one lookup: the order is the cache (no network), a file in the directory (hashed, no network), then the
    network."""
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    rig.cache.mkdir()
    stored(rig, ref).write_bytes(PAYLOAD)
    assert assets.resolve(ref) == stored(rig, ref) and rig.requests == 0
    assert assets.resolve(ref) == stored(rig, ref) and rig.requests == 0
    other = reference(b"another bundle" * 40, rig.server.url("/other"), name="another")
    rig.server.files["/other"] = b"another bundle" * 40
    assert assets.resolve(other).read_bytes() == b"another bundle" * 40 and rig.server.requests == ["GET /other"]


def missing_line_is_exact(rig: Rig) -> None:
    """The refusal is one message and one copy-pasteable action, in the experiment's terms: the asset's name and size,
    never its hash or the cache's path; scoped to the experiment's folder where it is known."""
    ref = reference(PAYLOAD, rig.server.url("/blob"), name="wall_R2.500")
    named = caught(lambda: assets.locate(ref))
    (refusal,) = named.refusals
    assert refusal.code == "asset.missing"
    assert refusal.message == "the experiment needs 'wall_R2.500' (26 kB), which isn't downloaded"
    assert refusal.action == "Run: mime assets fetch wall_R2.500"
    (scoped,) = caught(lambda: assets.locate(ref, folder="fl9_water")).refusals
    assert scoped.message == refusal.message
    assert scoped.action == "Run: mime assets fetch ./fl9_water/", scoped.action
    for text in (str(named), str(scoped)):
        assert ref.sha256 not in text and str(rig.cache) not in text


def caught(action: Callable[[], object]) -> BuildRefused:
    try:
        action()
    except BuildRefused as refused:
        return refused
    raise AssertionError("not refused")


def test_the_missing_refusal_is_one_exact_line_scoped_to_the_folder_when_known(rig: Rig):
    missing_line_is_exact(rig)


def test_the_folder_in_the_command_is_written_so_it_cannot_be_read_as_a_name(tmp_path: Path):
    from mime.assets._commands import fetch_command

    assert fetch_command("a", "b") == "mime assets fetch a b"
    assert fetch_command("a", folder="run") == "mime assets fetch ./run/"
    assert fetch_command(folder="./run/") == "mime assets fetch ./run/"
    assert fetch_command(folder="../run") == "mime assets fetch ../run/"
    assert fetch_command(folder=tmp_path / "my run") == f"mime assets fetch '{tmp_path / 'my run'}/'"


def test_a_doi_location_reaches_pooch_as_it_is_written(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    handed: list[str] = []

    def downloader(self: object, url: str, output_file: str, pooch_instance: object) -> None:
        handed.append(url)
        Path(output_file).write_bytes(PAYLOAD)

    monkeypatch.setattr(pooch.DOIDownloader, "__call__", downloader)
    location = "doi:10.5281/zenodo.1234567/cylinder_wall.npz"
    ref = reference(PAYLOAD, location)
    assert assets.fetch(ref) == stored(rig, ref)
    assert handed == [location] and rig.requests == 0


def test_the_cheap_check_catches_a_changed_size_and_a_touched_file(rig: Rig):
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    path = assets.fetch(ref)
    path.write_bytes(PAYLOAD[:-1])
    with pytest.raises(BuildRefused, match="is downloaded, but the file has"):
        assets.locate(ref)
    path.write_bytes(PAYLOAD)
    with pytest.raises(BuildRefused, match="changed after its full check"):
        assets.locate(ref)
    before = rig.requests
    assert assets.fetch(ref) == path and rig.requests == before, "the same bytes were not kept"
    assert assets.locate(ref) == path


def test_a_file_mime_downloaded_that_now_has_other_bytes_is_replaced_by_a_download(rig: Rig):
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    path = assets.fetch(ref)
    path.write_bytes(b"x" * len(PAYLOAD))
    assert assets.fetch(ref).read_bytes() == PAYLOAD and rig.requests == 2


def test_only_check_finds_a_flipped_byte_that_leaves_size_and_time_alone(rig: Rig):
    """The cheap check's limit, written as a test: locate trusts size and time, check re-hashes."""
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    path = assets.fetch(ref)
    seen = path.stat()
    flipped = bytearray(PAYLOAD)
    flipped[7] ^= 1
    path.write_bytes(bytes(flipped))
    os.utime(path, ns=(seen.st_atime_ns, seen.st_mtime_ns))
    assert assets.locate(ref) == path
    assert assets.check(ref) == "corrupt" and path.exists()
    assert assets.check(dataclasses.replace(ref, sha256="0" * 64)) == "not cached"


def test_no_file_in_the_cache_reads_as_not_cached(rig: Rig):
    assert assets.check(reference(PAYLOAD, "https://example.org/blob")) == "not cached"


# --- the reference ---------------------------------------------------------------------------------------------------


def test_a_reference_round_trips_through_its_lock_record():
    ref = reference(PAYLOAD, "https://example.org/blob", "doi:10.5281/zenodo.1/blob.npz")
    record = ref.to_record()
    assert list(record) == ["name", "sha256", "size", "licence", "locations"]
    assert record["locations"] == ["https://example.org/blob", "doi:10.5281/zenodo.1/blob.npz"]
    assert assets.AssetRef.from_record(record) == ref


@pytest.mark.parametrize("change", [
    {"sha256": "ABC"}, {"sha256": "0" * 63}, {"size": 0}, {"size": True}, {"name": "../escape"}, {"name": ""},
    {"licence": ""}, {"locations": []}, {"locations": ["ftp://example.org/blob"]},
    {"locations": ["http://example.org/blob"]}, {"locations": ["doi:no-registrant/file"]},
    {"locations": ["doi:10.5281/zenodo.1"]}, {"locations": ["https:///blob"]}, {"locations": [3]},
    {"extra": 1},
])
def test_a_malformed_reference_is_refused_whole(change: dict[str, object]):
    record = dict(reference(PAYLOAD, "https://example.org/blob").to_record())
    record.update(change)
    assert codes_of(lambda: assets.AssetRef.from_record(record)) == ["asset.reference"]


def test_a_missing_key_in_a_record_is_refused():
    record = dict(reference(PAYLOAD, "https://example.org/blob").to_record())
    del record["size"]
    assert codes_of(lambda: assets.AssetRef.from_record(record)) == ["asset.reference"]


def test_local_http_is_accepted_and_a_remote_one_is_not():
    assert assets.location_problem("http://127.0.0.1:8000/blob") is None
    assert assets.location_problem("http://localhost/blob") is None
    assert assets.location_problem("http://example.org/blob") is not None
    assert assets.location_problem("https://example.org/blob") is None
    assert assets.location_problem("http://[::1]:8000/blob") is None


def test_the_cache_is_one_directory_the_variable_replaces_the_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "somewhere"))
    assert assets.cache_dir() == tmp_path / "somewhere"
    monkeypatch.delenv("MIME_ASSETS_DIR")
    assert assets.cache_dir() == Path(str(pooch.os_cache("mime"))) / "assets"


# --- the commands ----------------------------------------------------------------------------------------------------


@pytest.fixture
def commanded(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> assets.AssetRef:
    """One shipped-style reference served by the rig, in place of the shipped ones, for the CLI to name."""
    rig.server.files["/blob"] = PAYLOAD
    ref = reference(PAYLOAD, rig.server.url("/blob"), name="served_blob")
    held = {ref.name: ref}
    monkeypatch.setattr(assets, "known", lambda: held)
    monkeypatch.setattr(_known, "known", lambda: held)
    return ref


def test_list_fetch_and_check_commands(commanded: assets.AssetRef, rig: Rig, capsys: pytest.CaptureFixture[str]):
    from mime.cli import main

    assert main(["assets", "list"]) == 0
    assert "served_blob  26 kB  missing  CC0-1.0" in capsys.readouterr().out
    assert main(["assets", "fetch", "--all"]) == 0
    assert "hashed in full" in capsys.readouterr().out and rig.requests == 1
    assert main(["assets", "list"]) == 0
    assert "served_blob  26 kB  cached" in capsys.readouterr().out
    assert main(["assets", "fetch", "served_blob"]) == 0 and rig.requests == 1
    assert main(["assets", "check"]) == 0
    assert "served_blob  ok" in capsys.readouterr().out
    stored(rig, commanded).write_bytes(b"y" * len(PAYLOAD))
    assert main(["assets", "check"]) == 1
    assert "served_blob  corrupt" in capsys.readouterr().out


def test_the_commands_refuse_with_exit_two(commanded: assets.AssetRef, rig: Rig, capsys: pytest.CaptureFixture[str],
                                           monkeypatch: pytest.MonkeyPatch):
    from mime.cli import main

    assert main(["assets", "fetch", "nothing_of_that_name"]) == 2
    assert "asset.unknown" in capsys.readouterr().err
    monkeypatch.setenv("MIME_OFFLINE", "1")
    assert main(["assets", "fetch", "served_blob"]) == 2
    assert "asset.offline" in capsys.readouterr().err and rig.requests == 0
    with pytest.raises(SystemExit):
        main(["assets", "fetch"])
    with pytest.raises(SystemExit):
        main(["assets", "fetch", "served_blob", "--all"])



# --- the controls ----------------------------------------------------------------------------------------------------


@contextmanager
def skipped_hash_check(ref: assets.AssetRef) -> Iterator[None]:
    """pooch is not given the hash, and the full hash after it says what the reference says."""
    with pytest.MonkeyPatch.context() as patch:
        def unchecked(location: str, held: assets.AssetRef, directory: Path) -> Path:
            return Path(pooch.retrieve(url=location, known_hash=None, fname=held.sha256, path=directory))

        patch.setattr(_fetch, "_download", unchecked)
        patch.setattr(_fetch, "sha256_of_file", lambda path: ref.sha256)
        yield


@contextmanager
def locate_that_downloads() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        honest = assets.locate

        def downloading(ref: assets.AssetRef) -> Path:
            try:
                return honest(ref)
            except BuildRefused:
                return assets.fetch(ref)

        patch.setattr(assets, "locate", downloading)
        yield


@contextmanager
def fallback_disabled() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        honest = assets.fetch
        patch.setattr(assets, "fetch", lambda ref: honest(dataclasses.replace(ref, locations=ref.locations[:1])))
        yield


@contextmanager
def folder_dropped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        honest = assets.locate
        patch.setattr(assets, "locate", lambda ref, *, folder=None: honest(ref))
        yield


RUNS = 20


def planted_runs(tmp_path: Path, scenario: Callable[[Rig], None], plant: Callable[[], object]) -> int:
    """How many of ``RUNS`` runs of ``scenario`` under the plant fail, and so detect it."""
    detected = 0
    for run in range(RUNS):
        try:
            with cast("Iterator[None]", plant()):
                run_with_rig(scenario, tmp_path / str(run))
        except (AssertionError, BuildRefused):  # a plant may fail the scenario's assertion or make it refused
            detected += 1
    return detected


def test_the_scenarios_pass_unplanted_twenty_times(tmp_path: Path):
    dead = dead_url()
    for run, scenario in enumerate([fetch_then_hit, wrong_hash_is_refused, locate_never_downloads] * RUNS):
        run_with_rig(scenario, tmp_path / str(run))
        run_with_rig(lambda rig: dead_first_location_falls_through(rig, dead), tmp_path / f"d{run}")


def test_control_a_skipped_hash_check_is_detected_every_run(tmp_path: Path):
    ref = reference(PAYLOAD, "https://example.org/unused")
    assert planted_runs(tmp_path, wrong_hash_is_refused, lambda: skipped_hash_check(ref)) == RUNS


def test_control_a_locate_that_downloads_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, locate_never_downloads, locate_that_downloads) == RUNS


def test_control_a_disabled_fallback_is_detected_every_run(tmp_path: Path):
    dead = dead_url()
    assert planted_runs(tmp_path, lambda rig: dead_first_location_falls_through(rig, dead), fallback_disabled) == RUNS


def test_control_a_refusal_that_drops_the_folder_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, missing_line_is_exact, folder_dropped) == RUNS


def test_the_hash_a_download_is_held_to_is_the_references(rig: Rig):
    """The control for the first: against the same server and bytes, the unplanted fetch stores and the planted one
    would have stored the wrong file."""
    rig.server.files["/blob"] = b"other bytes than the reference names" * 50
    ref = reference(PAYLOAD, rig.server.url("/blob"))
    with skipped_hash_check(ref):
        path = assets.fetch(ref)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != ref.sha256
