"""A file declared by its record, and where its bytes may go (design #229 items 3, 5 and 7).

An experiment states the whole record of a one-off file (a scan one study uses) through ``asset(AssetRef(...))``: the
record's sha256 pins the file, its licence, credit and role are the file's terms, and its locations its origin. The
build reads it from the asset cache, never downloading it. Saved, the lock holds the record whole (its custom licence
among it), and only the record's own locations: never the locations file's. Its tier decides every exit **per asset**:
a referenced record's bytes stay out of the bundle, whose manifest names it as fetched and from where, and out of
``mime push``'s upload, while its record travels in the bundle's lock; an open record's bytes travel inside.

The controls: a save that forgets the declared record writes a record without its licence, and the lock test catches
it every run; a push planted to upload every asset is caught by the plan's upload list."""

import hashlib
import json
from pathlib import Path

import pytest
from push_cases import copy_of, with_token, write_manifest
from terms_cases import NO_REDISTRIBUTION, mesh_bytes, same_reference_run
from zenodo_records import FakeRecords

from mime import assets, push
from mime.assets import AssetRef
from mime.build import BuildRefused
from mime.push import _plan
from mime.saved import LOCK_FILE, pack, read_folder
from mime.scene import asset
from mime.testing._graph import declare_in_order
from mime.testing.toys._experiments import ADD_PROBE, ATTACH_SPRING, toy_experiment

LAB_MIRROR = "https://files.lab.example.org/internal/scan.usda"


def scan_record(data: bytes, licence: str = NO_REDISTRIBUTION.id) -> AssetRef:
    custom = (NO_REDISTRIBUTION,) if licence == NO_REDISTRIBUTION.id else ()
    return AssetRef("vessel_scan.usda", hashlib.sha256(data).hexdigest(), len(data), licence,
                    ("https://example.org/scans/6a3ebb1/vessel_scan.usda",), credit="A Hospital's imaging lab",
                    role="results", kind="measured", licences=custom, citation="A. Lab, a vessel scan, 2026")


def placed(cache: Path, data: bytes) -> None:
    """The scan's bytes copied into the cache by hand, as a lab without the network does, then checked in full (the
    fetch downloads nothing: ``MIME_OFFLINE`` is set). A person who placed it has accepted its licence (a first use of a
    cached file asks)."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / hashlib.sha256(data).hexdigest()).write_bytes(data)
    record = scan_record(data)
    assets.fetch(record, accept_licences={each.id for each in assets.licences_of(record)})


def saved_with(record: AssetRef, where: Path) -> Path:
    experiment = declare_in_order(lambda: toy_experiment("scanned"), (ADD_PROBE, ATTACH_SPRING), None)
    experiment._settings.scene.reference(asset(record), at="/World/vessel_scan")
    experiment.licence("CC-BY-4.0")
    with same_reference_run():
        experiment.build().save(where)
    return where


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    held = tmp_path / "cache"
    monkeypatch.setenv("MIME_ASSETS_DIR", str(held))
    monkeypatch.setenv("MIME_OFFLINE", "1")
    locations = tmp_path / "locations.toml"
    locations.write_text(f'[assets]\n"vessel_scan.usda" = ["{LAB_MIRROR}"]\n', encoding="utf-8")
    monkeypatch.setenv("MIME_ASSET_LOCATIONS", str(locations))
    return held


def test_a_file_declared_by_its_record_is_read_from_the_cache_and_never_downloaded(cache, tmp_path):
    data = mesh_bytes("vessel_scan", 4000)
    record = scan_record(data)
    with pytest.raises(BuildRefused) as refused:
        saved_with(record, tmp_path / "missing")
    assert "asset.missing" in refused.value.codes() and "mime assets fetch vessel_scan.usda" in str(refused.value)
    held = asset(record)
    assert held.sha256 == record.sha256 and held.terms is not None and held.terms.origin == record.locations
    with pytest.raises(ValueError, match="takes its sha256 and its terms from the record"):
        asset(record, sha256=record.sha256)


def test_the_lock_holds_the_declared_record_whole_and_never_this_machines_locations(cache, tmp_path):
    data = mesh_bytes("vessel_scan", 4000)
    placed(cache, data)
    folder = saved_with(scan_record(data), tmp_path / "scanned")
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["referenced_files"] == {"scene/layers/vessel_scan.usda": "vessel_scan.usda"}
    (record,) = lock["assets"]
    assert record == dict(scan_record(data).to_record())
    assert record["licences"][NO_REDISTRIBUTION.id]["redistribution"] == "not_permitted"
    assert LAB_MIRROR not in (folder / LOCK_FILE).read_text(encoding="utf-8")
    assert assets.tried_locations(AssetRef.from_record(record))[0] == LAB_MIRROR, "this machine still tries it first"


def test_the_bundle_carries_a_referenced_records_bytes_never_and_its_manifest_says_where_to_fetch_them(cache,
                                                                                                    tmp_path):
    data = mesh_bytes("vessel_scan", 4000)
    placed(cache, data)
    folder = saved_with(scan_record(data), tmp_path / "scanned")
    bundle = pack(folder, out=tmp_path / "scanned.mimebundle")
    import zipfile

    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = [archive.read(name) for name in archive.namelist()]
    assert "scene/layers/vessel_scan.usda" not in manifest["files"]
    assert manifest["referenced"]["scene/layers/vessel_scan.usda"]["name"] == "vessel_scan.usda"
    (record,) = manifest["assets"]
    assert record["locations"] == ["https://example.org/scans/6a3ebb1/vessel_scan.usda"]
    assert not [each for each in members if data in each], "the referenced bytes are in no member"


def test_an_open_records_bytes_travel_inside_the_bundle(cache, tmp_path):
    data = mesh_bytes("vessel_scan", 4000)
    placed(cache, data)
    folder = saved_with(scan_record(data, licence="CC-BY-4.0"), tmp_path / "open")
    bundle = pack(folder, out=tmp_path / "open.mimebundle")
    import zipfile

    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert "scene/layers/vessel_scan.usda" in manifest["files"] and manifest["referenced"] == {}


def test_a_pushed_folder_never_uploads_a_referenced_asset_and_its_record_travels(cache, tmp_path, monkeypatch):
    data = mesh_bytes("vessel_scan", 4000)
    placed(cache, data)
    folder = saved_with(scan_record(data), tmp_path / "scanned")
    write_manifest(folder)
    token = with_token(monkeypatch)
    fake = FakeRecords(token)
    plan = _plan.make_plan(copy_of(folder, tmp_path / "push"), fake.service(), push.read_manifest(folder).record,
                           tmp_path / "staged")
    uploaded = {file.sha256 for file in plan.files}
    assert hashlib.sha256(data).hexdigest() not in uploaded
    choices = {choice.ref.name: choice for choice in plan.assets}
    assert not choices["vessel_scan.usda"].upload and choices["vessel_scan.usda"].verdict.status == "not_permitted"
    bundle = next(file for file in plan.files if file.key.endswith(".mimebundle"))
    import zipfile

    with zipfile.ZipFile(bundle.path) as archive:
        lock = json.loads(archive.read("experiment/mime.lock"))
    assert [held["name"] for held in lock["assets"]] == ["vessel_scan.usda"], "the record travels in the bundle"


def upload_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plant: a push whose choice uploads every asset of the lock."""
    real = _plan.asset_choices
    monkeypatch.setattr(_plan, "asset_choices", lambda refs, judged, lock, own=frozenset(): [
        _plan.AssetChoice(each.ref, each.verdict, True, "planted") for each in real(refs, judged, lock, own)])


def test_a_push_planted_to_upload_every_asset_is_refused_by_its_payload_check(cache, tmp_path, monkeypatch):
    """Whatever chose the uploads, the record's files are held to every record in reach: the referenced scan's bytes
    are refused as ``push.referenced-copy`` before anything is sent."""
    upload_everything(monkeypatch)
    with pytest.raises(BuildRefused) as refused:
        test_a_pushed_folder_never_uploads_a_referenced_asset_and_its_record_travels(cache, tmp_path, monkeypatch)
    assert refused.value.codes() == ["push.referenced-copy"] and "vessel_scan.usda" in str(refused.value)


def test_control_a_push_planted_to_upload_every_asset_with_no_payload_check_is_caught(cache, tmp_path, monkeypatch):
    upload_everything(monkeypatch)
    monkeypatch.setattr(_plan, "upload_problems", lambda files, lock: [])
    with pytest.raises(AssertionError):
        test_a_pushed_folder_never_uploads_a_referenced_asset_and_its_record_travels(cache, tmp_path, monkeypatch)


def test_control_a_save_that_forgets_the_declared_record_writes_a_record_without_its_licence(cache, tmp_path,
                                                                                         monkeypatch):
    from mime.saved import _record

    monkeypatch.setattr(_record, "declared_records", lambda settings, declarations: {})
    with pytest.raises(AssertionError):
        test_the_lock_holds_the_declared_record_whole_and_never_this_machines_locations(cache, tmp_path)


def test_the_saved_folder_reloads_with_its_record(cache, tmp_path):
    data = mesh_bytes("vessel_scan", 4000)
    placed(cache, data)
    folder = saved_with(scan_record(data), tmp_path / "scanned")
    from mime.experiment import Experiment

    assert Experiment.load(folder).load_report is not None
    assert read_folder(folder).lock["assets"][0]["kind"] == "measured"
