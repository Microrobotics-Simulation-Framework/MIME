"""A file that travels as a reference, on the reading side (bundle format 2): ``unpack``, ``mime pull`` and ``mime
assets fetch <folder>`` place it from the asset cache, copied and hashed, never over another file and never outside the
folder, judged in the folder's real path (a folder reached through a link, as every temporary folder on macOS is, is
placed in like any other); a file left unplaced is reported with why, ``mime assets fetch <folder>`` exits two while
one is, and a folder still missing one is refused as ``asset.missing`` with the command that fetches it. A bundle's
references, its NOTICE and its lock are held to one another (no carried file holds a reference's bytes, or its name in
another case), a format 1 bundle still unpacks, and a bundle of a later format is refused first, with the advice to
install the MIME that packed it.

Nothing leaves the machine: the referenced file's origin is the assets tests' local server. The macOS and Windows smoke
runs this file (``tests/platforms.txt``).
"""

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from assets_rig import Rig
from bundles import member, rewrite, write_bundle
from terms_cases import NO_REDISTRIBUTION, V3_FIXTURE, layered, same_reference_run, write_mesh

from mime import assets
from mime.assets import AssetRef
from mime.build import BuildRefused
from mime.cli import main
from mime.experiment import Experiment
from mime.saved import _bundle, _terms, pack, place_referenced, pull, unpack
from mime.saved._bundle import FORMAT_1, MANIFEST_MEMBER, ManifestFormat1, bundle_hash, sha256_of
from mime.saved._schema import conforms
from mime.terms import Terms

LAYER = "scene/layers/Arm.usda"


def codes_of(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def referenced_folder(root: Path, rig: Rig) -> tuple[Path, bytes, AssetRef]:
    """A saved folder whose mesh may not be redistributed, its origin the rig's server; the mesh's bytes and the asset
    its lock names."""
    mesh = write_mesh(root / "sources", "Arm", 5000)
    data = mesh.read_bytes()
    rig.server.files["/upstream/Arm.usda"] = data
    terms = Terms(NO_REDISTRIBUTION, "Robot Maker Ltd", "presentation", (rig.server.url("/upstream/Arm.usda"),))
    with same_reference_run():
        layered("referenced", [(mesh, terms)], licence="CC-BY-4.0").build().save(root / "referenced")
    lock = json.loads((root / "referenced" / "mime.lock").read_text())
    (record,) = lock["assets"]
    return root / "referenced", data, AssetRef.from_record(record)


def test_a_referenced_file_is_placed_from_the_cache_when_it_is_cached(tmp_path: Path, rig: Rig):
    folder, data, ref = referenced_folder(tmp_path, rig)
    bundle = pack(folder, out=tmp_path / "x.mimebundle")
    assets.fetch(ref)
    unpacked = unpack(bundle, tmp_path / "unpacked")
    assert unpacked.placed == (LAYER,) and unpacked.missing == ()
    assert (tmp_path / "unpacked" / LAYER).read_bytes() == data
    assert Experiment.load(tmp_path / "unpacked").load_report is not None


def test_a_folder_missing_a_referenced_file_is_refused_with_the_command_that_fetches_and_places_it(tmp_path: Path,
                                                                                                    rig: Rig,
                                                                                                    monkeypatch):
    folder, data, _ = referenced_folder(tmp_path, rig)
    bundle = pack(folder, out=tmp_path / "x.mimebundle")
    unpacked = unpack(bundle, tmp_path / "unpacked")
    assert unpacked.missing == (LAYER,) and not (tmp_path / "unpacked" / LAYER).exists()
    assert rig.requests == 0, "unpack used the network"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BuildRefused) as refused:
        Experiment.load("unpacked")
    assert refused.value.codes() == ["asset.missing"]
    assert refused.value.refusals[0].action == "Run: mime assets fetch ./unpacked/"
    assert main(["assets", "fetch", "./unpacked/"]) == 0
    assert (tmp_path / "unpacked" / LAYER).read_bytes() == data
    assert Experiment.load("unpacked").load_report is not None


def test_pull_fetches_and_places_a_referenced_file_and_says_so(tmp_path: Path, rig: Rig, monkeypatch):
    folder, data, _ = referenced_folder(tmp_path, rig)
    bundle = pack(folder, out=tmp_path / "x.mimebundle")
    monkeypatch.chdir(tmp_path)
    lines: list[str] = []
    pull(bundle, into="pulled", say=lines.append)
    assert (tmp_path / "pulled" / LAYER).read_bytes() == data
    assert ("  - Referenced files: 1, not carried in the bundle (their terms are in `mime card ./pulled/ --notice`); "
            "each placed from the asset cache") in lines[-1].splitlines()


def test_pull_without_assets_says_how_to_place_a_referenced_file(tmp_path: Path, rig: Rig, monkeypatch):
    folder, _, _ = referenced_folder(tmp_path, rig)
    bundle = pack(folder, out=tmp_path / "x.mimebundle")
    monkeypatch.chdir(tmp_path)
    lines: list[str] = []
    pull(bundle, into="pulled", fetch_assets=False, say=lines.append)
    assert not (tmp_path / "pulled" / LAYER).exists() and rig.requests == 0
    assert any(line.endswith("1 not placed: run `mime assets fetch ./pulled/`") for line in lines[-1].splitlines())


def test_a_summary_of_a_bundle_that_carries_every_file_has_no_referenced_line(tmp_path: Path):
    with same_reference_run():
        layered("plain", [(write_mesh(tmp_path / "sources", "Arm", 900), None)]).build().save(tmp_path / "plain")
    lines: list[str] = []
    pull(pack(tmp_path / "plain", out=tmp_path / "plain.mimebundle"), into=tmp_path / "pulled", say=lines.append)
    assert not any("Referenced files" in line for line in lines[-1].splitlines())


def test_placing_never_writes_over_another_file_or_outside_the_folder(tmp_path: Path, rig: Rig):
    folder, data, ref = referenced_folder(tmp_path, rig)
    assets.fetch(ref)
    copy = Path(shutil.copytree(folder, tmp_path / "copy"))
    (copy / LAYER).write_bytes(b"another file")
    placed = place_referenced(copy)
    assert placed.missing == ((LAYER, ref.name, _terms.OTHER_FILE),) and (copy / LAYER).read_bytes() == b"another file"
    lock = json.loads((copy / "mime.lock").read_text())
    lock["referenced_files"] = {"../outside.usda": ref.name, "scene/layers/../../x.usda": ref.name}
    (copy / "mime.lock").write_text(json.dumps(lock))
    assert place_referenced(copy) == type(placed)(missing=(("../outside.usda", ref.name, _terms.OUTSIDE),
                                                           ("scene/layers/../../x.usda", ref.name, _terms.OUTSIDE)))
    assert not (tmp_path / "outside.usda").exists() and not (tmp_path / "x.usda").exists()
    (copy / LAYER).unlink()
    lock["referenced_files"] = {LAYER: ref.name}
    (copy / "mime.lock").write_text(json.dumps(lock))
    assert place_referenced(copy).placed == (LAYER,) and (copy / LAYER).read_bytes() == data


def test_mime_assets_fetch_exits_two_while_a_referenced_file_is_not_placed(tmp_path: Path, rig: Rig,
                                                                          capsys: pytest.CaptureFixture[str]):
    folder, data, _ = referenced_folder(tmp_path, rig)
    (folder / LAYER).write_bytes(b"another file")
    assert main(["assets", "fetch", str(folder)]) == 2
    err = capsys.readouterr().err
    assert "asset.missing" in err and _terms.OTHER_FILE in err
    (folder / LAYER).unlink()
    assert main(["assets", "fetch", str(folder)]) == 0 and (folder / LAYER).read_bytes() == data


def placed_through(alias: Path, tmp_path: Path, rig: Rig) -> None:
    """The scenario: a bundle with a referenced file, its asset cached, unpacked into a folder below ``alias``, a path
    that is not the folder's real one; the file is placed and the folder loads."""
    folder, data, ref = referenced_folder(tmp_path, rig)
    bundle = pack(folder, out=tmp_path / "x.mimebundle")
    assets.fetch(ref)
    unpacked = unpack(bundle, alias / "unpacked")
    assert unpacked.placed == (LAYER,) and unpacked.missing == (), unpacked
    assert (alias / "unpacked" / LAYER).read_bytes() == data
    assert Experiment.load(alias / "unpacked").load_report is not None
    (alias / "unpacked" / LAYER).unlink()
    assert place_referenced(alias / "unpacked").placed == (LAYER,)


def test_a_referenced_file_is_placed_in_a_folder_reached_through_a_linked_parent(tmp_path: Path, rig: Rig):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    placed_through(alias, tmp_path, rig)
    assert (real / "unpacked" / LAYER).is_file()


def test_a_referenced_file_is_placed_in_the_platforms_temporary_folder_as_it_is_named(tmp_path: Path, rig: Rig):
    """``tempfile``'s folder as the platform names it, unresolved: on macOS a link (``/var`` into ``/private``), on
    Windows often a short name, which only resolving makes the folder's real path."""
    with tempfile.TemporaryDirectory(prefix="mime-placing-") as named:
        assert os.path.isabs(named)
        placed_through(Path(named), tmp_path, rig)


# --- what unpack refuses ------------------------------------------------------------------------------------------


@pytest.fixture
def referenced_bundle(tmp_path: Path, rig: Rig) -> Path:
    folder, _, _ = referenced_folder(tmp_path, rig)
    return pack(folder, out=tmp_path / "x.mimebundle")


def test_a_reference_that_is_carried_too_is_refused(referenced_bundle: Path, tmp_path: Path, rig: Rig):
    """A bundle that carries a file it also says travels as a reference, its very bytes: not one ``pack`` writes."""
    data = rig.server.files["/upstream/Arm.usda"]

    def carried(files: dict[str, bytes]) -> None:
        files[LAYER] = data

    def listed(manifest: dict[str, object]) -> None:
        held = dict(manifest["files"])  # type: ignore[arg-type]
        held[LAYER] = {"sha256": sha256_of(data), "size": len(data)}
        manifest["files"] = held

    target = rewrite(referenced_bundle, tmp_path / "carried.mimebundle", carried, listed, keep_hash=False)
    assert "bundle.unsafe" in codes_of(lambda: unpack(target, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_a_bundle_carrying_a_references_bytes_or_name_at_another_path_is_refused(referenced_bundle: Path,
                                                                                  tmp_path: Path, rig: Rig):
    """Bytes ``pack`` never carries: the reference's at another layer's path, and another file under the reference's
    name in another case (one name on macOS and Windows)."""
    data = rig.server.files["/upstream/Arm.usda"]
    for name, content, said in (("scene/layers/Copy.usda", data, "carries the bytes of the reference"),
                                ("scene/layers/arm.usda", b"#usda 1.0\n", "one name on a case-insensitive")):
        def added(files: dict[str, bytes], name: str = name, content: bytes = content) -> None:
            files[name] = content
            lock = json.loads(files["mime.lock"])
            lock["stage"]["layers"][name] = sha256_of(content)
            files["mime.lock"] = (json.dumps(lock, indent=1, sort_keys=True) + "\n").encode()

        def listed(manifest: dict[str, object], name: str = name, content: bytes = content) -> None:
            held = dict(manifest["files"])  # type: ignore[arg-type]
            held[name] = {"sha256": sha256_of(content), "size": len(content)}
            manifest["files"] = held

        target = rewrite(referenced_bundle, tmp_path / f"{name.rsplit('/', 1)[1]}.mimebundle", added, listed,
                         keep_hash=False)
        with zipfile.ZipFile(target) as archive:
            files = {info.filename: archive.read(info) for info in archive.infolist()}
        manifest = json.loads(files["manifest.json"])
        manifest["files"]["mime.lock"] = {"sha256": sha256_of(files["experiment/mime.lock"]),
                                          "size": len(files["experiment/mime.lock"])}
        manifest["bundle_hash"] = bundle_hash(manifest)
        with zipfile.ZipFile(target, "w") as archive:
            member(archive, MANIFEST_MEMBER, json.dumps(manifest, sort_keys=True, indent=1).encode())
            for each, held in files.items():
                if each != MANIFEST_MEMBER:
                    member(archive, each, held)
        with pytest.raises(BuildRefused) as refused:
            unpack(target, tmp_path / "out")
        assert refused.value.codes() == ["bundle.unsafe"] and said in str(refused.value), str(refused.value)
        assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", [
    lambda held: {**held, LAYER: {**held[LAYER], "name": "another"}},
    lambda held: {**held, LAYER: {**held[LAYER], "sha256": "0" * 64}},
    lambda held: {},
    lambda held: {**held, "scene/layers/Hand.usda": held[LAYER]},
])
def test_references_that_are_not_the_locks_are_refused(referenced_bundle: Path, tmp_path: Path,
                                                       change: Callable[[dict[str, object]], dict[str, object]]):
    def references(manifest: dict[str, object]) -> None:
        manifest["referenced"] = change(dict(manifest["referenced"]))  # type: ignore[arg-type]

    target = rewrite(referenced_bundle, tmp_path / "changed.mimebundle", lambda files: None, references,
                     keep_hash=False)
    found = codes_of(lambda: unpack(target, tmp_path / "out"))
    assert set(found) <= {"bundle.manifest", "bundle.unsafe"} and found, found
    assert not (tmp_path / "out").exists()


def test_a_reference_outside_the_layout_is_refused(referenced_bundle: Path, tmp_path: Path):
    def outside(manifest: dict[str, object]) -> None:
        held = dict(manifest["referenced"])  # type: ignore[arg-type]
        manifest["referenced"] = {"../../outside.usda": held[LAYER]}

    target = rewrite(referenced_bundle, tmp_path / "outside.mimebundle", lambda files: None, outside, keep_hash=False)
    assert "bundle.unsafe" in codes_of(lambda: unpack(target, tmp_path / "out"))
    assert not (tmp_path / "outside.usda").exists()


def test_an_altered_or_missing_notice_is_refused(referenced_bundle: Path, tmp_path: Path):
    altered = tmp_path / "altered.mimebundle"
    with zipfile.ZipFile(referenced_bundle) as source, zipfile.ZipFile(altered, "w") as target:
        for info in source.infolist():
            data = source.read(info)
            member(target, info.filename, data.replace(b"Robot Maker", b"Robot Taker") if info.filename == "NOTICE"
                   else data)
    assert codes_of(lambda: unpack(altered, tmp_path / "a")) == ["bundle.changed"]
    missing = tmp_path / "missing.mimebundle"
    with zipfile.ZipFile(referenced_bundle) as source, zipfile.ZipFile(missing, "w") as target:
        for info in source.infolist():
            if info.filename != "NOTICE":
                member(target, info.filename, source.read(info))
    assert codes_of(lambda: unpack(missing, tmp_path / "m")) == ["bundle.changed"]


def test_unpack_keeps_the_notice_and_writes_it_into_no_folder(referenced_bundle: Path, tmp_path: Path):
    unpacked = unpack(referenced_bundle, tmp_path / "out")
    assert unpacked.notice.startswith("NOTICE: referenced") and not (tmp_path / "out" / "NOTICE").exists()


# --- format 1 -------------------------------------------------------------------------------------------------------


def format_one_bundle(folder: Path, target: Path) -> Path:
    """A bundle as MIME wrote one before format 2: every file under ``experiment/``, no NOTICE."""
    files = {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*"))
             if path.is_file()}
    manifest: dict[str, object] = {"format": FORMAT_1, "name": folder.name,
                                   "files": {name: {"sha256": sha256_of(data), "size": len(data)}
                                             for name, data in sorted(files.items())}, "assets": []}
    manifest["bundle_hash"] = bundle_hash(manifest)
    with zipfile.ZipFile(target, "w") as archive:
        member(archive, MANIFEST_MEMBER, json.dumps(manifest, sort_keys=True, indent=1).encode())
        for name, data in sorted(files.items()):
            member(archive, "experiment/" + name, data)
    return target


def test_a_format_one_bundle_still_unpacks_and_loads(tmp_path: Path):
    bundle = format_one_bundle(V3_FIXTURE, tmp_path / "old.mimebundle")
    unpacked = unpack(bundle, tmp_path / "out")
    assert unpacked.manifest["format"] == "mime.bundle/1" and unpacked.notice == ""
    assert Experiment.load(tmp_path / "out").load_report is not None


def test_a_format_one_reader_refuses_a_format_two_manifest(referenced_bundle: Path):
    manifest = json.loads(zipfile.ZipFile(referenced_bundle).read(MANIFEST_MEMBER))
    held, problems = conforms(manifest, ManifestFormat1)
    assert held is None and any("format" in each for each in problems)
    assert _bundle.FORMAT == "mime.bundle/2"


def a_later_format(path: Path) -> Path:
    """A bundle of format 3, as a later MIME might write one: a member this MIME does not know beside its manifest."""
    write_bundle(path, {"experiment.mime.yaml": b"schema: mime.document/9\n"})
    with zipfile.ZipFile(path) as archive:
        files = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = {**json.loads(files[MANIFEST_MEMBER]), "format": "mime.bundle/3", "signature": {"by": "a later MIME"}}
    manifest["bundle_hash"] = bundle_hash(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        member(archive, MANIFEST_MEMBER, json.dumps(manifest, sort_keys=True, indent=1).encode())
        for name, data in files.items():
            if name != MANIFEST_MEMBER:
                member(archive, name, data)
        member(archive, "SIGNATURE", b"signed")
    return path


def test_a_bundle_of_a_later_format_is_refused_first_with_the_advice_to_install_the_mime_that_packed_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # pull with no --into unpacks into the working directory
    bundle = a_later_format(tmp_path / "later.mimebundle")
    for action in (lambda: unpack(bundle, tmp_path / "a"), lambda: pull(bundle, into=tmp_path / "b"),
                   lambda: pull(bundle)):
        with pytest.raises(BuildRefused) as refused:
            action()
        (refusal,) = refused.value.refusals
        assert refusal.code == "load.newer-form" and "mime.bundle/3" in refusal.message, refusal
        assert "install the MIME that wrote it" in refusal.action and "do not unpack" not in refusal.action
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_referenced_bytes_never_reach_the_bundle(referenced_bundle: Path, tmp_path: Path, rig: Rig):
    data = rig.server.files["/upstream/Arm.usda"]
    assert data not in referenced_bundle.read_bytes()
    assert hashlib.sha256(data).hexdigest() in json.dumps(
        json.loads(zipfile.ZipFile(referenced_bundle).read(MANIFEST_MEMBER))["referenced"])
