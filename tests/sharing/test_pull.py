"""``mime pull``: a bundle by file, address or DOI is checked, unpacked into a new folder and verified, its assets
fetched on request, and what is still needed said in names and sizes.

Nothing here leaves the machine: the bundle and the assets are served by a local HTTP server in a thread, and Zenodo's
record API is a fake of it (``requests.get`` is told that ``doi.org`` and ``zenodo.org`` are that server, so pooch's
own DOI code runs against the fake record). The controls plant the check each refusal depends on (the hash check
skipped, the path check skipped, the link check skipped) and the refusals must stop being refusals, twenty runs out of
twenty.
"""

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import pytest
import requests
from assets_rig import Rig, reference
from bundles import contents, lock_with_assets, member, rewrite, write_bundle

from mime import assets
from mime.build import BuildRefused
from mime.cli import main
from mime.experiment import Experiment
from mime.saved import _bundle, _pull, pack, pull
from mime.testing.toys import full_toy

TABLE = os.urandom(2000)
SMALL = os.urandom(900)


@pytest.fixture(scope="module")
def saved_toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("toy") / "full_toy"
    full_toy().save(folder)
    return folder


@pytest.fixture
def bundle(saved_toy: Path, tmp_path: Path) -> Path:
    return pack(saved_toy, out=tmp_path / "full_toy.mimebundle")


def codes_of(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def listing(path: Path) -> list[str]:
    return sorted(str(each.relative_to(path)) for each in path.rglob("*"))


def with_assets(saved_toy: Path, tmp_path: Path, *references: assets.AssetRef) -> Path:
    """A bundle of the toy whose lock references ``references``."""
    files = {path.relative_to(saved_toy).as_posix(): path.read_bytes() for path in saved_toy.rglob("*") if path.is_file()}
    records = lock_with_assets(files, *references)
    return write_bundle(tmp_path / "with_assets.mimebundle", files, "full_toy", records)


def table_reference(rig: Rig, payload: bytes = TABLE, name: str = "table") -> assets.AssetRef:
    rig.server.files[f"/t/{name}.bin"] = payload
    return reference(payload, rig.server.url(f"/t/{name}.bin"), name=name)


def said(**keywords: object) -> tuple[list[str], dict[str, object]]:
    lines: list[str] = []
    return lines, {"say": lines.append, **keywords}


# --- the round trip ------------------------------------------------------------------------------------------------------


def verified(folder: Path) -> None:
    experiment = Experiment.load(folder)
    report = experiment.build().verify()
    assert report.reproduced and report.same_environment, report.text()


def test_a_bundle_by_file_unpacks_to_the_folder_it_was_packed_from_and_verifies_bit_for_bit(saved_toy: Path,
                                                                                            bundle: Path, tmp_path: Path):
    folder = pull(bundle, into=tmp_path / "out")
    assert {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()} == \
        {p.relative_to(saved_toy).as_posix(): p.read_bytes() for p in saved_toy.rglob("*") if p.is_file()}
    verified(folder)
    assert not [each for each in tmp_path.iterdir() if each.name.startswith(".")], "a hidden staging folder was left"


def test_a_bundle_by_address_is_fetched_through_the_cache_and_verifies(bundle: Path, rig: Rig, tmp_path: Path):
    rig.server.files["/b/full_toy.mimebundle"] = bundle.read_bytes()
    address = rig.server.url("/b/full_toy.mimebundle")
    folder = pull(address, into=tmp_path / "out")
    verified(folder)
    assert rig.requests == 1
    held = hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert (rig.cache / held).read_bytes() == bundle.read_bytes(), "the bundle is a content-addressed file in the cache"
    other = pull(address, into=tmp_path / "again")
    assert other.is_dir() and rig.requests == 2, "an address is fetched again: nothing says its hash in advance"


def test_the_default_folder_is_named_for_the_experiment_in_the_working_directory(bundle: Path, tmp_path: Path,
                                                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert pull(bundle) == Path("full_toy") and (tmp_path / "full_toy" / "mime.lock").is_file()


def test_a_folder_that_exists_and_is_not_empty_is_refused_and_left_alone(bundle: Path, tmp_path: Path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "mine.txt").write_text("mine", encoding="utf-8")
    assert codes_of(lambda: pull(bundle, into=tmp_path / "out")) == ["pull.destination"]
    assert listing(tmp_path / "out") == ["mine.txt"]
    (tmp_path / "out" / "mine.txt").unlink()
    assert pull(bundle, into=tmp_path / "out").is_dir(), "an empty folder is a place to unpack to"


def test_a_source_that_is_nothing_is_refused(tmp_path: Path):
    assert codes_of(lambda: pull("no_such_file.mimebundle")) == ["pull.source"]
    assert codes_of(lambda: pull("http://example.org/b.mimebundle")) == ["pull.source"]
    assert codes_of(lambda: pull("ftp://example.org/b")) == ["pull.source"]


def test_a_file_that_is_not_a_bundle_is_refused(tmp_path: Path):
    for name, data in (("text.mimebundle", b"not a zip"), ("empty.mimebundle", b"")):
        (tmp_path / name).write_bytes(data)
        assert codes_of(lambda: pull(tmp_path / name, into=tmp_path / "out")) == ["bundle.unreadable"]
    assert not (tmp_path / "out").exists()


# --- a DOI -----------------------------------------------------------------------------------------------------------------------


RECORD = 4242
DOI = f"10.5281/zenodo.{RECORD}"


@contextmanager
def zenodo(rig: Rig, files: dict[str, bytes], *, legacy: bool = True, checksums: dict[str, str] | None = None
           ) -> Iterator[None]:
    """A fake of Zenodo's record API on the rig's server. ``requests.get`` is told that ``doi.org`` and ``zenodo.org`` are
    that server, and answers a request to the landing page with the address pooch expects to be redirected to."""
    listed = []
    for name, data in files.items():
        checksum = (checksums or {}).get(name, hashlib.md5(data).hexdigest())
        if legacy:
            listed.append({"key": name, "checksum": f"md5:{checksum}", "size": len(data),
                           "links": {"self": f"https://zenodo.org/api/records/{RECORD}/files/{name}/content"}})
            rig.server.files[f"/zenodo/api/records/{RECORD}/files/{name}/content"] = data
        else:
            listed.append({"filename": name, "checksum": checksum, "filesize": len(data)})
            rig.server.files[f"/zenodo/records/{RECORD}/files/{name}?download=1"] = data
    rig.server.files[f"/zenodo/api/records/{RECORD}"] = json.dumps({"id": RECORD, "files": listed}).encode()
    rig.server.files["/landing"] = b"<html></html>"
    real = requests.get

    def routed(url: str, *args: object, **keywords: object) -> requests.Response:
        if url == f"https://doi.org/{DOI}":
            local, reported = rig.server.url("/landing"), f"https://zenodo.org/records/{RECORD}"
        elif url.startswith("https://zenodo.org/"):
            local, reported = rig.server.url("/zenodo/" + url.removeprefix("https://zenodo.org/")), url
        else:
            return real(url, *args, **keywords)  # type: ignore[arg-type]
        response = real(local, *args, **keywords)  # type: ignore[arg-type]
        response.url = reported
        return response

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests, "get", routed)
        yield


@pytest.mark.parametrize("legacy", [True, False], ids=["record as zenodo.org serves it", "record in the other layout"])
@pytest.mark.parametrize("form", ["{doi}", "doi:{doi}", "https://doi.org/{doi}", "doi:{doi}/full_toy.mimebundle"])
def test_a_bundle_by_doi_is_fetched_with_no_account_and_verifies(bundle: Path, rig: Rig, tmp_path: Path, legacy: bool,
                                                                 form: str):
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes(), "README.md": b"hello"}, legacy=legacy):
        folder = pull(form.format(doi=DOI), into=tmp_path / "out")
    verified(folder)
    assert (rig.cache / hashlib.sha256(bundle.read_bytes()).hexdigest()).is_file()


def test_the_repositorys_checksum_holds_the_download(bundle: Path, rig: Rig, tmp_path: Path):
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}, checksums={"full_toy.mimebundle": "0" * 32}):
        assert codes_of(lambda: pull(DOI, into=tmp_path / "out")) == ["asset.hash-mismatch"]
    assert not (tmp_path / "out").exists() and not list(rig.cache.glob("[0-9a-f]" * 64))


def test_a_record_with_two_bundles_or_none_asks_for_the_name(bundle: Path, rig: Rig, tmp_path: Path):
    data = bundle.read_bytes()
    with zenodo(rig, {"a.mimebundle": data, "b.mimebundle": data}):
        assert codes_of(lambda: pull(DOI, into=tmp_path / "x")) == ["pull.record"]
        assert pull(f"doi:{DOI}/b.mimebundle", into=tmp_path / "y").is_dir()
    with zenodo(rig, {"notes.txt": b"x"}):
        assert codes_of(lambda: pull(DOI, into=tmp_path / "z")) == ["pull.record"]


def test_a_record_that_cannot_be_read_is_refused_with_the_cause(rig: Rig, tmp_path: Path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests, "get", lambda url, *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")))
        try:
            pull(DOI, into=tmp_path / "out")
        except BuildRefused as refused:
            assert refused.codes() == ["pull.record"] and "down" in str(refused)
        else:
            raise AssertionError("not refused")


def test_a_doi_is_not_resolved_offline(bundle: Path, rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MIME_OFFLINE", "1")
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        assert codes_of(lambda: pull(DOI, into=tmp_path / "out")) == ["asset.offline"]
        assert codes_of(lambda: pull(rig.server.url("/b.mimebundle"), into=tmp_path / "out")) == ["asset.offline"]
    assert rig.requests == 0
    assert pull(bundle, into=tmp_path / "out").is_dir(), "a file on this machine needs no network"


# --- bundles that are not what pack writes -------------------------------------------------------------------------------------


def flipped(files: dict[str, bytes]) -> None:
    data = bytearray(files["reference.npz"])
    data[len(data) // 2] ^= 1
    files["reference.npz"] = bytes(data)


def added(files: dict[str, bytes]) -> None:
    files["extra.txt"] = b"unlisted"


def removed(files: dict[str, bytes]) -> None:
    del files["scene/scene.usda"]


def named(name: str) -> Callable[[dict[str, bytes]], None]:
    def change(files: dict[str, bytes]) -> None:
        files[name] = b"escaped"
    return change


def listing_in_manifest(name: str) -> Callable[[dict[str, object]], None]:
    def change(manifest: dict[str, object]) -> None:
        files = manifest["files"]
        assert isinstance(files, dict)
        files[name] = {"sha256": hashlib.sha256(b"escaped").hexdigest(), "size": len(b"escaped")}
    return change


def renamed(manifest: dict[str, object]) -> None:
    manifest["name"] = "other"


def relisted_after_a_change(manifest: dict[str, object]) -> None:
    """The manifest as a forger who recomputed its own hash writes it, but who left the file's hash alone."""


ALTERED: list[tuple[str, Callable[[dict[str, bytes]], None], Callable[[dict[str, object]], None] | None, bool, str]] = [
    ("a file changed, the manifest as it was", flipped, None, True, "bundle.changed"),
    ("a file changed, the manifest's own hash recomputed", flipped, relisted_after_a_change, False, "bundle.changed"),
    ("the manifest changed, its hash as it was", lambda files: None, renamed, True, "bundle.changed"),
    ("a file the manifest does not list", added, None, True, "bundle.changed"),
    ("a file the manifest lists, missing", removed, None, True, "bundle.changed"),
]

HOSTILE_NAMES = ["../evil.txt", "a/../../evil.txt", "/etc/evil.txt", "C:evil.txt", "..\\evil.txt", "a//b.txt",
                 "./evil.txt", "a/./b.txt", "dir/"]


def unaltered_state(tmp_path: Path) -> tuple[list[str], Path]:
    return listing(tmp_path), tmp_path


@pytest.mark.parametrize(("label", "change", "manifest_change", "keep_hash", "code"), ALTERED,
                         ids=[each[0] for each in ALTERED])
def test_an_altered_bundle_is_refused_by_its_hashes_and_leaves_nothing(bundle: Path, tmp_path: Path, label: str,
                                                                       change: Callable[[dict[str, bytes]], None],
                                                                       manifest_change: Callable[..., None] | None,
                                                                       keep_hash: bool, code: str):
    altered = rewrite(bundle, tmp_path / "altered.mimebundle", change, manifest_change, keep_hash=keep_hash)
    before = listing(tmp_path)
    assert codes_of(lambda: pull(altered, into=tmp_path / "out")) == [code]
    assert listing(tmp_path) == before, "the refused pull left something behind"


def test_a_flipped_byte_inside_a_stored_member_is_refused(bundle: Path, tmp_path: Path):
    data = bytearray(bundle.read_bytes())
    start = data.index(b"schema: mime.document")
    data[start + 3] ^= 1
    (tmp_path / "flipped.mimebundle").write_bytes(bytes(data))
    assert codes_of(lambda: pull(tmp_path / "flipped.mimebundle", into=tmp_path / "out")) in (
        ["bundle.unreadable"], ["bundle.changed"])
    assert not (tmp_path / "out").exists()


def test_a_truncated_bundle_is_refused(bundle: Path, tmp_path: Path):
    """Cut short, the end of the outer archive is gone; zipfile may then find the end of the npz stored inside it, and
    the members it lists are not a bundle's. Either way nothing is unpacked."""
    (tmp_path / "cut.mimebundle").write_bytes(bundle.read_bytes()[:-30])
    assert set(codes_of(lambda: pull(tmp_path / "cut.mimebundle", into=tmp_path / "out"))) <= {
        "bundle.unreadable", "bundle.unsafe", "bundle.manifest"}
    assert not (tmp_path / "out").exists()


def escaping_bundle(bundle: Path, tmp_path: Path, name: str) -> Path:
    """A bundle whose manifest lists a file called ``name`` and whose hash is recomputed, so that only its name is wrong."""
    return rewrite(bundle, tmp_path / "hostile.mimebundle", named(name), listing_in_manifest(name), keep_hash=False)


def named_member(bundle: Path, tmp_path: Path, name: str) -> Path:
    """The same, with the member written under the exact name (``rewrite`` prefixes the folder's)."""
    target = tmp_path / "hostile.mimebundle"
    files = contents(bundle)
    manifest = json.loads(zipfile.ZipFile(bundle).read("manifest.json"))
    manifest["files"]["x"] = {"sha256": hashlib.sha256(b"escaped").hexdigest(), "size": 7}
    manifest["bundle_hash"] = _bundle.bundle_hash(manifest)
    with zipfile.ZipFile(target, "w") as archive:
        member(archive, "manifest.json", json.dumps(manifest).encode())
        for each, data in files.items():
            member(archive, "experiment/" + each, data)
        member(archive, name, b"escaped")
    return target


def a_hostile_name_is_refused(root: Path, bundle: Path, name: str) -> None:
    hostile = escaping_bundle(bundle, root, name)
    before = listing(root)
    destination = root / "inner" / "out"
    try:
        pull(hostile, into=destination)
    except BuildRefused as refused:
        assert set(refused.codes()) <= {"bundle.unsafe", "bundle.changed"}, refused.codes()
        assert listing(root) == before, "the refused pull left files"
        assert not (destination.parent / "evil.txt").exists() and not (root / "evil.txt").exists()
        assert not Path("/etc/evil.txt").exists()
        return
    raise AssertionError(f"{name!r} was unpacked")


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_a_member_named_to_escape_the_folder_is_refused(bundle: Path, tmp_path: Path, name: str):
    a_hostile_name_is_refused(tmp_path, bundle, name)


def a_link_member_is_refused(root: Path, bundle: Path, mode: int) -> None:
    target = root / "linked.mimebundle"
    files = contents(bundle)
    manifest = json.loads(zipfile.ZipFile(bundle).read("manifest.json"))
    with zipfile.ZipFile(target, "w") as archive:
        member(archive, "manifest.json", json.dumps(manifest).encode())
        for each, data in files.items():
            member(archive, "experiment/" + each, data, mode=mode if each == "mime.lock" else 0o100644)
    before = listing(root)
    try:
        pull(target, into=root / "out")
    except BuildRefused as refused:
        assert refused.codes() == ["bundle.unsafe"] and "mime.lock" in str(refused)
        assert listing(root) == before
        return
    raise AssertionError(f"a member of mode {mode:o} was unpacked")


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o644, stat.S_IFCHR | 0o644, stat.S_IFDIR | 0o755,
                                  stat.S_IFSOCK | 0o644], ids=["link", "pipe", "device", "directory", "socket"])
def test_a_member_that_is_a_link_or_a_special_file_is_refused(bundle: Path, tmp_path: Path, mode: int):
    a_link_member_is_refused(tmp_path, bundle, mode)


def test_a_compressed_a_doubled_and_an_encrypted_member_are_refused(bundle: Path, tmp_path: Path):
    files = contents(bundle)
    manifest = zipfile.ZipFile(bundle).read("manifest.json")
    compressed = tmp_path / "compressed.mimebundle"
    with zipfile.ZipFile(compressed, "w") as archive:
        member(archive, "manifest.json", manifest)
        for each, data in files.items():
            member(archive, "experiment/" + each, data, compress=zipfile.ZIP_DEFLATED)
    assert codes_of(lambda: pull(compressed, into=tmp_path / "a")) == ["bundle.unsafe"] * len(files)
    doubled = tmp_path / "doubled.mimebundle"
    with zipfile.ZipFile(doubled, "w") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        member(archive, "manifest.json", manifest)
        for each, data in files.items():
            member(archive, "experiment/" + each, data)
        member(archive, "experiment/mime.lock", b"another")
    assert codes_of(lambda: pull(doubled, into=tmp_path / "b")) == ["bundle.unsafe"]
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_a_bundle_whose_assets_are_not_its_locks_is_refused(saved_toy: Path, tmp_path: Path, rig: Rig):
    files = {p.relative_to(saved_toy).as_posix(): p.read_bytes() for p in saved_toy.rglob("*") if p.is_file()}
    records = lock_with_assets(files, table_reference(rig))
    forged = write_bundle(tmp_path / "forged.mimebundle", files, "full_toy", [{**records[0], "locations": ["https://evil.example/t"]}])
    assert codes_of(lambda: pull(forged, into=tmp_path / "out")) == ["bundle.manifest"]
    assert not (tmp_path / "out").exists()


# --- the controls: a refusal that depends on one check --------------------------------------------------------------------------


def a_changed_file_is_refused(root: Path, bundle: Path) -> None:
    altered = rewrite(bundle, root / "altered.mimebundle", flipped)
    assert codes_of(lambda: pull(altered, into=root / "out")) == ["bundle.changed"]
    assert not (root / "out").exists()


@contextmanager
def hash_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_verify_member", lambda name, digest, size, expected: None)
        yield


@contextmanager
def path_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "member_problem", lambda name: None)
        patch.setattr(_bundle, "_target", lambda root, name: root / name)
        yield


@contextmanager
def link_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_member_kind_problems", lambda archive: [])
        yield


RUNS = 20


def planted_runs(tmp_path: Path, scenario: Callable[[Path], None], plant: Callable[[], object]) -> int:
    detected = 0
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir(parents=True)
        try:
            with plant():  # type: ignore[attr-defined]
                scenario(root)
        except AssertionError:
            detected += 1
        except OSError:  # the unplanted check would have refused before this
            detected += 1
    return detected


def test_control_a_skipped_hash_check_is_detected_every_run(bundle: Path, tmp_path: Path):
    assert planted_runs(tmp_path, lambda root: a_changed_file_is_refused(root, bundle), hash_check_skipped) == RUNS


def test_control_a_skipped_path_check_is_detected_every_run(bundle: Path, tmp_path: Path):
    runs = [lambda root, name=name: a_hostile_name_is_refused(root, bundle, name) for name in ("../evil.txt", "a/../../evil.txt")]
    assert planted_runs(tmp_path / "one", runs[0], path_check_skipped) == RUNS
    assert planted_runs(tmp_path / "two", runs[1], path_check_skipped) == RUNS


def test_control_a_skipped_link_check_is_detected_every_run(bundle: Path, tmp_path: Path):
    assert planted_runs(tmp_path, lambda root: a_link_member_is_refused(root, bundle, stat.S_IFLNK | 0o777),
                        link_check_skipped) == RUNS


def test_the_scenarios_pass_unplanted_twenty_times(bundle: Path, tmp_path: Path):
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        a_changed_file_is_refused(root, bundle)
        (root / "a").mkdir()
        a_hostile_name_is_refused(root / "a", bundle, "../evil.txt")
        (root / "b").mkdir()
        a_link_member_is_refused(root / "b", bundle, stat.S_IFLNK | 0o777)


# --- what pull reports ---------------------------------------------------------------------------------------------------------------


def test_a_package_the_lock_names_that_is_not_installed_is_listed_with_the_command_and_nothing_is_installed(
        saved_toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = {p.relative_to(saved_toy).as_posix(): p.read_bytes() for p in saved_toy.rglob("*") if p.is_file()}
    lock = json.loads(files["mime.lock"])
    for entry in lock["entries"]:
        entry["distribution_version"] = "9.9.9"
    lock["entries"][0].update(distribution="mime-srl", distribution_version="0.2.1")
    files["mime.lock"] = json.dumps(lock).encode()
    needing = write_bundle(tmp_path / "needing.mimebundle", files, "full_toy")
    spawned: list[object] = []
    for name in ("run", "Popen", "check_call", "check_output", "call"):
        monkeypatch.setattr(subprocess, name, lambda *a, **k: spawned.append(a))
    monkeypatch.setattr(os, "system", lambda *a: spawned.append(a))
    lines, keywords = said()
    pull(needing, into=tmp_path / "out", **keywords)  # type: ignore[arg-type]
    installed = metadata.version("mime-engine")
    assert lines[-1].splitlines()[3] == (f'  - Packages: mime-engine 9.9.9 needed ({installed} installed), mime-srl 0.2.1 '
                                         f'needed: pip install "mime-engine==9.9.9" "mime-srl==0.2.1"')
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.version("mime-srl")
    assert spawned == [], "pull started a process"


def test_the_summary_in_names_and_sizes(saved_toy: Path, tmp_path: Path, rig: Rig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cached, fresh = table_reference(rig, SMALL, "cached_table"), table_reference(rig)
    assets.fetch(cached)
    bundle = with_assets(saved_toy, tmp_path, cached, fresh)
    lines, keywords = said()
    pull(bundle, into="fl9_water", **keywords)  # type: ignore[arg-type]
    summary = lines[-1]
    assert summary == "\n".join([
        "\N{CHECK MARK} Bundle 'full_toy' unpacked to ./fl9_water/",
        "  - Manifest, scene and records: OK",
        "  - Assets: 1 cached, 1 downloaded (2 kB)",
        "  - Packages: all installed",
        "Next: mime verify ./fl9_water/"])
    assert lines[0] == "Assets to download: 1 file, 2 kB"
    for secret in (cached.sha256, fresh.sha256, str(rig.cache), cached.locations[0]):
        assert secret not in summary, "the summary names a hash, a cache path or an address"
    verified(Path("fl9_water"))


def test_the_summary_with_no_assets_says_how_to_fetch_them(saved_toy: Path, tmp_path: Path, rig: Rig,
                                                           monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cached, fresh = table_reference(rig, SMALL, "cached_table"), table_reference(rig)
    assets.fetch(cached)
    before = rig.requests
    lines, keywords = said(fetch_assets=False)
    pull(with_assets(saved_toy, tmp_path, cached, fresh), into="fl9_water", **keywords)  # type: ignore[arg-type]
    assert rig.requests == before, "--no-assets made a request"
    assert lines == ["\n".join([
        "\N{CHECK MARK} Bundle 'full_toy' unpacked to ./fl9_water/",
        "  - Manifest, scene and records: OK",
        "  - Assets: 1 cached, 1 not downloaded: run `mime assets fetch ./fl9_water/`",
        "  - Packages: all installed",
        "Next: mime verify ./fl9_water/"])]
    assert not assets.is_cached(fresh)


def test_the_summary_when_the_bundle_references_no_asset(bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    lines, keywords = said()
    pull(bundle, into="out", **keywords)  # type: ignore[arg-type]
    assert lines == ["\n".join(["\N{CHECK MARK} Bundle 'full_toy' unpacked to ./out/",
                                "  - Manifest, scene and records: OK", "  - Assets: none referenced",
                                "  - Packages: all installed", "Next: mime verify ./out/"])]


def test_an_asset_that_cannot_be_fetched_leaves_the_folder_and_says_which_command_finishes_it(
        saved_toy: Path, tmp_path: Path, rig: Rig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    ref = table_reference(rig)
    del rig.server.files["/t/table.bin"]
    lines, keywords = said()
    assert codes_of(lambda: pull(with_assets(saved_toy, tmp_path, ref), into="out", **keywords)) == ["asset.unreachable"]  # type: ignore[arg-type]
    assert "  - Assets: 1 not downloaded: run `mime assets fetch ./out/`" in lines[-1]
    assert (tmp_path / "out" / "mime.lock").is_file()


# --- the size an asset download is asked about -------------------------------------------------------------------------------


@pytest.fixture
def large(saved_toy: Path, tmp_path: Path, rig: Rig, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, assets.AssetRef]:
    """A bundle whose one asset is over the (lowered) threshold."""
    monkeypatch.setattr(_pull, "ASSET_CONFIRM_BYTES", 1000)
    monkeypatch.chdir(tmp_path)
    ref = table_reference(rig)
    return with_assets(saved_toy, tmp_path, ref), ref


def test_the_threshold_is_a_gigabyte():
    assert _pull.ASSET_CONFIRM_BYTES == 1_000_000_000


def test_a_download_over_the_threshold_is_asked_about_and_goes_ahead_when_accepted(large: tuple[Path, assets.AssetRef],
                                                                                   rig: Rig):
    bundle, ref = large
    asked: list[str] = []
    lines, keywords = said(confirm=lambda question: asked.append(question) or True)
    pull(bundle, into="out", **keywords)  # type: ignore[arg-type]
    assert asked == ["Download 2 kB of assets now?"] and assets.is_cached(ref)
    assert "1 downloaded (2 kB)" in lines[-1]


def test_a_refused_confirmation_downloads_nothing_and_says_how_to_continue(large: tuple[Path, assets.AssetRef], rig: Rig):
    bundle, ref = large
    before = rig.requests
    lines, keywords = said(confirm=lambda question: False)
    folder = pull(bundle, into="out", **keywords)  # type: ignore[arg-type]
    assert folder.is_dir() and rig.requests == before and not assets.is_cached(ref)
    assert "1 not downloaded: run `mime assets fetch ./out/`" in lines[-1]


def test_yes_goes_ahead_without_asking(large: tuple[Path, assets.AssetRef], rig: Rig):
    bundle, ref = large

    def never(question: str) -> bool:
        raise AssertionError("asked although --yes was given")

    pull(bundle, into="out", yes=True, confirm=never)
    assert assets.is_cached(ref)


def test_without_a_terminal_and_without_yes_it_stops_with_the_command(large: tuple[Path, assets.AssetRef], rig: Rig):
    bundle, ref = large
    before = rig.requests
    lines, keywords = said()
    with pytest.raises(BuildRefused) as stopped:
        pull(bundle, into="out", **keywords)  # type: ignore[arg-type]
    assert stopped.value.codes() == ["pull.assets-large"]
    assert "mime assets fetch ./out/" in str(stopped.value) and "2 kB" in str(stopped.value)
    assert rig.requests == before and not assets.is_cached(ref)
    assert (Path("out") / "mime.lock").is_file() and "not downloaded" in lines[-1], "the folder and its summary are kept"


def test_an_asset_under_the_threshold_is_not_asked_about(saved_toy: Path, tmp_path: Path, rig: Rig,
                                                         monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_pull, "ASSET_CONFIRM_BYTES", len(TABLE))
    monkeypatch.chdir(tmp_path)
    ref = table_reference(rig)

    def never(question: str) -> bool:
        raise AssertionError("asked about a download at the threshold")

    pull(with_assets(saved_toy, tmp_path, ref), into="out", confirm=never)
    assert assets.is_cached(ref)


def test_a_cached_asset_is_not_counted_in_the_size_asked_about(large: tuple[Path, assets.AssetRef], rig: Rig):
    bundle, ref = large
    assets.fetch(ref)

    def never(question: str) -> bool:
        raise AssertionError("asked about an asset already here")

    lines, keywords = said(confirm=never)
    pull(bundle, into="out", **keywords)  # type: ignore[arg-type]
    assert "Assets: 1 cached" in lines[-1]


# --- the commands --------------------------------------------------------------------------------------------------------------


def test_the_command_pulls_and_prints_the_summary(bundle: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert main(["pull", str(bundle), "--into", "out", "--no-assets"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "\N{CHECK MARK} Bundle 'full_toy' unpacked to ./out/"
    assert out.splitlines()[-1] == "Next: mime verify ./out/"
    assert main(["verify", "out"]) == 0
    assert "reproduced 8 steps bit for bit" in capsys.readouterr().out


def test_the_command_exits_two_on_a_refusal_and_names_the_command_for_a_large_download(
        large: tuple[Path, assets.AssetRef], capsys: pytest.CaptureFixture[str]):
    bundle, ref = large
    assert main(["pull", str(bundle), "--into", "out"]) == 2
    captured = capsys.readouterr()
    assert "pull.assets-large" in captured.err and "mime assets fetch ./out/" in captured.err
    assert "Bundle 'full_toy' unpacked" in captured.out
    assert main(["pull", str(bundle), "--into", "again", "--yes"]) == 0
    assert assets.is_cached(ref)


def test_the_command_asks_on_a_terminal(large: tuple[Path, assets.AssetRef], capsys: pytest.CaptureFixture[str],
                                        monkeypatch: pytest.MonkeyPatch):
    bundle, ref = large

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Terminal("n\n"))
    assert main(["pull", str(bundle), "--into", "no"]) == 0
    assert not assets.is_cached(ref) and "Download 2 kB of assets now? [y/N]" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", Terminal("y\n"))
    assert main(["pull", str(bundle), "--into", "yes"]) == 0
    assert assets.is_cached(ref)
