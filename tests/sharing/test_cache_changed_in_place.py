"""MIME-ANO-142: a cached file changed in place, its size and modification time kept.

The cache's cheap check (size, and the size and modification time a full check recorded) cannot see a byte flipped in
the middle of a file with both restored. Where the caller holds the repository's checksum of the file (a bundle fetched by
its DOI), the check now also reads the file once and compares its digest, and a download that follows replaces a cached
file whose full hash fails. The network is the pull tests' fake repository on a local server; Zenodo is never contacted.

Controls (each fails on every run, since the flip and the plants are fixed, not drawn): an unchanged file is served with
no request, and with no hash when the caller holds no checksum; a flipped file with no checksum is served, as the anomaly's
entry says; and with the digest comparison, or the replacement, planted out, the reproduction fails.
"""

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from assets_rig import Rig, reference
from fake_zenodo import DOI, RECORD, zenodo

from mime import assets
from mime.assets import _cache, _fetch
from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.saved import pack, pull
from mime.testing.toys import full_toy

BUNDLE_ADDRESS = "/files/full_toy.mimebundle/content"


@pytest.fixture(scope="module")
def saved_toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("toy") / "full_toy"
    full_toy().save(folder)
    return folder


@pytest.fixture
def bundle_bytes(saved_toy: Path, tmp_path: Path) -> bytes:
    return pack(saved_toy, out=tmp_path / "full_toy.mimebundle").read_bytes()


def bundle_downloads(rig: Rig) -> int:
    return sum(1 for request in rig.server.requests if request.endswith(BUNDLE_ADDRESS))


#: Where the bytes inverted by :func:`flip_in_place` are, as offsets in a file of the given size.
FLIPS: dict[str, Callable[[int], list[int]]] = {
    "middle": lambda size: [size // 2],
    "start": lambda size: [0],
    "end": lambda size: [size - 1],
    "two bytes": lambda size: [size // 3, 2 * size // 3],
    "first and last": lambda size: [0, size - 1],
}


def flip_in_place(path: Path, where: str = "middle") -> None:
    """The bytes at ``FLIPS[where]`` inverted, the file's size and ``st_mtime_ns`` as they were."""
    seen = path.stat()
    with path.open("r+b") as stream:
        for offset in FLIPS[where](seen.st_size):
            stream.seek(offset)
            byte = stream.read(1)
            stream.seek(offset)
            stream.write(bytes([byte[0] ^ 0xFF]))
    os.utime(path, ns=(seen.st_atime_ns, seen.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (seen.st_size, seen.st_mtime_ns), "the flip left a trace the check sees"


def cached_bundle(rig: Rig, good: bytes) -> Path:
    return rig.cache / hashlib.sha256(good).hexdigest()


def work_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> Path:
    work = tmp_path / name
    work.mkdir()
    monkeypatch.chdir(work)
    return work


# --- the gate: the review's reproduction ---------------------------------------------------------------------------------


@pytest.mark.parametrize("where", FLIPS)
def test_a_bundle_flipped_in_place_is_downloaded_once_more_and_the_good_file_is_served(
        rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundle_bytes: bytes, where: str):
    work_folder(tmp_path, monkeypatch, "work")
    held = cached_bundle(rig, bundle_bytes)
    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        first = Experiment.based_on(DOI, name="child").document()
        assert bundle_downloads(rig) == 1 and held.read_bytes() == bundle_bytes
        flip_in_place(held, where)
        assert held.read_bytes() != bundle_bytes
        again = Experiment.based_on(DOI, name="child").document()
        assert bundle_downloads(rig) == 2, "one download repaired it"
        assert held.read_bytes() == bundle_bytes and again == first
        pulled = pull(DOI, into=tmp_path / "pulled")
        assert bundle_downloads(rig) == 2, "the repaired file is served from the cache"
    bundle = assets.AssetRef("bundle", hashlib.sha256(bundle_bytes).hexdigest(), len(bundle_bytes), "CC0-1.0",
                             ("https://example.org/b",))
    assert (pulled / "mime.lock").is_file() and assets.check(bundle) == "ok"
    assert not [each for each in rig.cache.iterdir() if ".incoming-" in each.name or each.name.endswith(".partial")]


@pytest.mark.parametrize("where", FLIPS)
def test_pull_of_the_doi_after_a_flip_downloads_once_and_unpacks(rig: Rig, tmp_path: Path, bundle_bytes: bytes,
                                                                 where: str):
    held = cached_bundle(rig, bundle_bytes)
    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        pull(DOI, into=tmp_path / "first")
        flip_in_place(held, where)
        folder = pull(DOI, into=tmp_path / "second")
    assert bundle_downloads(rig) == 2 and held.read_bytes() == bundle_bytes and (folder / "mime.lock").is_file()


def test_the_download_is_installed_atomically_over_the_failing_file(rig: Rig, tmp_path: Path, bundle_bytes: bytes,
                                                                    monkeypatch: pytest.MonkeyPatch):
    held = cached_bundle(rig, bundle_bytes)
    replaced: list[tuple[Path, Path]] = []
    real = Path.replace

    def spy(self: Path, target: str | os.PathLike[str]) -> Path:
        replaced.append((self, Path(target)))
        return real(self, target)

    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        pull(DOI, into=tmp_path / "first")
        flip_in_place(held)
        monkeypatch.setattr(Path, "replace", spy)
        pull(DOI, into=tmp_path / "second")
    assert held in [target for _, target in replaced], "the download was renamed over the old file in one step"
    assert held.read_bytes() == bundle_bytes


def test_a_flipped_file_a_person_placed_is_refused_and_left_alone(rig: Rig, tmp_path: Path, bundle_bytes: bytes):
    """Step 2 replaces only a file MIME wrote: with no record (as a person's copy has) the flipped file is refused."""
    held = cached_bundle(rig, bundle_bytes)
    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        pull(DOI, into=tmp_path / "first")
        flip_in_place(held)
        flipped = held.read_bytes()
        (rig.cache / (held.name + _cache.RECORD_SUFFIX)).unlink()
        with pytest.raises(BuildRefused) as refused:
            pull(DOI, into=tmp_path / "second")
    assert refused.value.codes() == ["asset.placed-file"] and held.read_bytes() == flipped


def test_a_bare_url_bundle_flipped_in_place_is_repaired_by_the_download_that_runs_each_time(rig: Rig):
    """Step 2's side effect: a bundle named by a bare ``https://`` address (no checksum, so downloaded every time) now
    also repairs a cached file of those bytes that was changed in place, where before the cached file was kept."""
    payload = os.urandom(4000)
    rig.server.files["/bundle"] = payload
    address = rig.server.url("/bundle")
    held = rig.cache / assets.retrieve(address, name="bundle").sha256
    flip_in_place(held)
    assert held.read_bytes() != payload
    assets.retrieve(address, name="bundle")
    assert rig.requests == 2 and held.read_bytes() == payload


def test_an_unreadable_cached_bundle_is_refused_with_its_file_and_the_route_never_a_bare_error(
        rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundle_bytes: bytes):
    """Permissions that forbid the read (the error the operating system gives, planted for that one file so that the
    test means the same as a user without read permission, whoever runs it)."""
    work_folder(tmp_path, monkeypatch, "work")
    held = cached_bundle(rig, bundle_bytes)
    real = Path.open

    def forbidden(self: Path, *arguments, **keywords):  # type: ignore[no-untyped-def]
        if self == held:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *arguments, **keywords)

    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        Experiment.based_on(DOI, name="child")
        monkeypatch.setattr(Path, "open", forbidden)
        with pytest.raises(BuildRefused) as refused:
            Experiment.based_on(DOI, name="child")
        assert bundle_downloads(rig) == 1, "an unreadable file is not worked around by a download"
    assert refused.value.codes() == ["asset.missing"]
    text = str(refused.value)
    assert held.name in text and "Permission denied" in text and "permission" in text and "delete" in text


def test_a_cached_bundle_that_is_gone_when_opened_is_a_problem_not_an_error(tmp_path: Path):
    """The file vanishing between the size check and the read: a problem, which the caller answers by downloading."""
    assert _cache.digest_problem(tmp_path / "nothing", "md5:" + "0" * 32) == "isn't downloaded"


# --- controls ------------------------------------------------------------------------------------------------------------


def counting_digests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the files the cache reads in full (a sha256 or a digest of the caller's algorithm)."""
    reads: list[str] = []
    real = hashlib.file_digest

    def counted(stream, digest, **keywords):  # type: ignore[no-untyped-def]
        reads.append(stream.name)
        return real(stream, digest, **keywords)

    monkeypatch.setattr(hashlib, "file_digest", counted)
    return reads


def test_control_an_unchanged_bundle_is_served_with_no_request_and_one_read_for_the_checksum(
        rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundle_bytes: bytes):
    work_folder(tmp_path, monkeypatch, "work")
    held = cached_bundle(rig, bundle_bytes)
    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        Experiment.based_on(DOI, name="child")
        reads = counting_digests(monkeypatch)
        Experiment.based_on(DOI, name="child")
        assert bundle_downloads(rig) == 1
        assert [read for read in reads if read == str(held)] == [str(held)], "the caller's checksum costs one read"


def test_control_an_unchanged_file_with_no_caller_checksum_is_served_with_no_hash_and_no_request(
        rig: Rig, monkeypatch: pytest.MonkeyPatch):
    """The shortcut's point."""
    payload = os.urandom(5000)
    rig.server.files["/t/blob.bin"] = payload
    ref = reference(payload, rig.server.url("/t/blob.bin"))
    assets.fetch(ref)
    requests, reads = rig.requests, counting_digests(monkeypatch)
    assert assets.locate(ref) == rig.cache / ref.sha256
    assert rig.requests == requests and reads == []


def test_control_a_flipped_file_with_no_caller_checksum_is_served_as_the_entry_states(rig: Rig):
    """What the shortcut does not guard: with no checksum to compare, only ``check`` finds the flip. With the checksum
    the same file is refused."""
    payload = os.urandom(5000)
    rig.server.files["/t/blob.bin"] = payload
    ref = reference(payload, rig.server.url("/t/blob.bin"))
    stored = assets.fetch(ref)
    flip_in_place(stored)
    assert assets.locate(ref) == stored, "served: the shortcut does not guard the bytes"
    digits = hashlib.md5(payload).hexdigest()
    with pytest.raises(BuildRefused) as refused:
        assets.locate(ref, checksum=f"md5:{digits}")
    assert refused.value.codes() == ["asset.missing"]
    assert assets.check(ref) == "corrupt"
    assets.fetch(ref)
    assert assets.locate(ref, checksum=f"md5:{digits}") == stored and stored.read_bytes() == payload


def test_control_a_checksum_of_an_algorithm_not_here_is_a_problem_never_a_pass(rig: Rig):
    payload = os.urandom(2000)
    rig.server.files["/t/blob.bin"] = payload
    ref = reference(payload, rig.server.url("/t/blob.bin"))
    assets.fetch(ref)
    with pytest.raises(BuildRefused):
        assets.locate(ref, checksum="nosuchalgorithm:00")
    with pytest.raises(BuildRefused):
        assets.locate(ref, checksum="md5:" + "0" * 32)
    with pytest.raises(BuildRefused):
        assets.locate(ref, checksum="shake_128:" + "0" * 32)


def test_control_with_the_digest_comparison_planted_out_the_flipped_bundle_is_served(
        rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundle_bytes: bytes):
    """The reproduction fails without the fix's step 1: the planted run serves the changed file, no download."""
    work_folder(tmp_path, monkeypatch, "work")
    held = cached_bundle(rig, bundle_bytes)
    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        Experiment.based_on(DOI, name="child")
        flip_in_place(held)
        monkeypatch.setattr(_cache, "digest_problem", lambda path, checksum: None)
        with pytest.raises(BuildRefused):
            Experiment.based_on(DOI, name="child")
        assert bundle_downloads(rig) == 1 and held.read_bytes() != bundle_bytes


def test_control_with_the_replacement_planted_out_the_download_does_not_repair_the_cache(
        rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundle_bytes: bytes):
    """The reproduction fails without step 2: the file is downloaded again and the cache still holds the changed one."""
    work_folder(tmp_path, monkeypatch, "work")
    held = cached_bundle(rig, bundle_bytes)
    with zenodo(rig, {"full_toy.mimebundle": bundle_bytes}):
        Experiment.based_on(DOI, name="child")
        flip_in_place(held)
        monkeypatch.setattr(_fetch, "_look_at_what_is_there", lambda ref: (True, False))
        with pytest.raises(BuildRefused):
            Experiment.based_on(DOI, name="child")
        assert bundle_downloads(rig) == 2 and held.read_bytes() != bundle_bytes


def test_control_the_fake_repository_is_the_only_server(rig: Rig):
    """Nothing here reaches Zenodo: the record's address is the rig's local server."""
    assert rig.server.url("/x").startswith("http://127.0.0.") and RECORD
