"""A file had by request alone, through every exit (saved form 14, lane P's option 4): its record names no location, and
may name where to request it. MIME never refuses it and never forces a location (the maintainer's rule: encouraged,
never forced). A build refuses only when it needs the bytes and does not find them (``asset.by-request``); ``pack``
carries the record and warns (``ByRequestWarning``); ``push`` lists it, never uploads it and warns; the NOTICE, the
availability statement and the card say plainly that results depending on it are not reproducible from public sources,
and where to request it; one of role presentation gets the softer line.

**The controls.** The folder loads with the file placed and is refused as ``asset.by-request`` without it; with the
refusal's branch removed (``MIME_PLANT_BY_REQUEST=missing``) it says ``asset.missing`` and the test fails. The
statement's sentence appears whenever a results asset has no location, over folder terms drawn from ``MIME_SEED``; with
the sentence's parts removed (``MIME_PLANT_BY_REQUEST=silent``) the test fails. A bundle or a manifest file of the
format before the step that holds a record with no location is refused, and the same without it is read."""

import hashlib
import json
import os
import random
import shutil
import warnings
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from push_cases import copy_of, with_token, write_manifest
from terms_cases import mesh_bytes, same_reference_run
from zenodo_records import FakeRecords

from mime import assets, push
from mime.assets import AssetRef, _acceptances, _cache
from mime.build import BuildRefused
from mime.push import _plan
from mime.saved import LOCK_FILE, ByRequestWarning, LoadRefused, card_text, notice, pack, place_referenced, unpack
from mime.saved import _card
from mime.saved._bundle import FORMAT_3, MANIFEST_MEMBER, bundle_hash
from mime.saved._terms import FolderTerms, LockAsset
from mime.scene import asset
from mime.terms import CustomLicence, licence_verdict
from mime.testing._graph import declare_in_order
from mime.testing.toys._experiments import ADD_PROBE, ATTACH_SPRING, toy_experiment

REQUEST = "mailto:imaging@hospital.example.org"
LICENCE = CustomLicence("LicenseRef-Imaging-Lab-Restricted", "The imaging lab's data licence 1.0",
                        "https://example.org/imaging-lab/LICENSE.txt", "not_permitted")
LAYER = "scene/layers/vessel_scan.usda"
NOT_PUBLIC = "Not reproducible from public sources"


def scan(data: bytes, *, request: str | None = REQUEST) -> AssetRef:
    return AssetRef("vessel_scan.usda", hashlib.sha256(data).hexdigest(), len(data), LICENCE.id, (),
                    credit="A Hospital's imaging lab", role="results", kind="measured", licences=(LICENCE,),
                    request=request)


@contextmanager
def own_cache(root: Path) -> Iterator[Path]:
    root.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MIME_ASSETS_DIR", str(root / "cache"))
        patch.setenv("MIME_OFFLINE", "1")
        (root / "locations.toml").write_text("", encoding="utf-8")
        patch.setenv("MIME_ASSET_LOCATIONS", str(root / "locations.toml"))
        yield root / "cache"


def placed(cache: Path, ref: AssetRef, data: bytes) -> None:
    """The file copied in by hand under its name, taken by a fetch, its licence accepted by the person who placed it."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ref.name).write_bytes(data)
    _acceptances.remember(assets.licences_of(ref))
    assets.fetch(ref)


def saved_by_request(root: Path, data: bytes, *, request: str | None = REQUEST) -> Path:
    with own_cache(root) as cache:
        placed(cache, scan(data, request=request), data)
        experiment = declare_in_order(lambda: toy_experiment("scanned"), (ADD_PROBE, ATTACH_SPRING), None)
        experiment._settings.scene.reference(asset(scan(data, request=request)), at="/World/vessel_scan")  # noqa: SLF001
        experiment.licence("CC-BY-4.0")
        with same_reference_run():
            experiment.build().save(root / "scanned")
    return root / "scanned"


@pytest.fixture(scope="module")
def data() -> bytes:
    return mesh_bytes("vessel_scan", 4000)


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory, data: bytes) -> Path:
    return saved_by_request(tmp_path_factory.mktemp("by_request"), data)


def test_the_lock_holds_the_record_with_no_location_and_its_route(folder: Path):
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    (record,) = lock["assets"]
    assert record["locations"] == [] and record["request"] == REQUEST
    assert lock["referenced_files"] == {LAYER: "vessel_scan.usda"}


def test_the_folder_loads_with_the_file_placed_and_is_refused_by_request_without_it(folder: Path, data: bytes,
                                                                                    tmp_path: Path):
    """The folder as a bundle unpacks it, the file that travels as the record not in it: with nothing in the cache,
    its load is refused by request; once the file is placed in the cache and in the folder (as ``mime assets fetch
    <folder>`` places it), it loads."""
    plant = os.environ.get("MIME_PLANT_BY_REQUEST", "")
    copy = Path(shutil.copytree(folder, tmp_path / "copy"))
    (copy / LAYER).unlink()
    with own_cache(tmp_path / "empty"), pytest.MonkeyPatch.context() as patch:
        if plant == "missing":
            patch.setattr(_cache, "no_location", lambda ref: False)
        with pytest.raises((BuildRefused, LoadRefused)) as refused:
            from mime.experiment import Experiment

            Experiment.load(copy)
    assert refused.value.codes() == ["asset.by-request"], refused.value.codes()
    assert f"Request it from: {REQUEST}" in str(refused.value)
    with own_cache(tmp_path / "held") as cache:
        placed(cache, scan(data), data)
        from mime.experiment import Experiment

        assert place_referenced(copy).placed == (LAYER,)
        loaded = Experiment.load(copy)
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()


def test_pack_carries_the_record_never_the_bytes_and_warns_never_refuses(folder: Path, data: bytes, tmp_path: Path):
    with pytest.warns(ByRequestWarning) as warned:
        bundle = pack(folder, out=tmp_path / "scanned.mimebundle")
    (said,) = [str(each.message) for each in warned if each.category is ByRequestWarning]
    assert said.startswith(f"{LAYER} (licence {LICENCE.id}: not permitted, referenced) has no download location: the "
                           f"bundle carries its record (its sha256 and size), never its bytes, and the results that "
                           f"depend on it are not reproducible from public sources; request it from: {REQUEST}. MIME "
                           f"does not recommend this")
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read(MANIFEST_MEMBER))
        members = [archive.read(name) for name in archive.namelist()]
    assert manifest["format"] == "mime.bundle/4" and manifest["assets"] == [scan(data).to_record()]
    assert LAYER in manifest["referenced"] and not [each for each in members if data in each]
    unpacked = unpack(bundle, tmp_path / "unpacked")
    assert unpacked.missing == (LAYER,)


def test_the_notice_and_the_availability_say_it_plainly(folder: Path):
    text = notice(folder)
    assert (f"referenced: not carried in a bundle; the asset vessel_scan.usda, with no download location (by request); "
            f"request it from: {REQUEST}") in text
    said = text.split("Availability\n------------\n")[1]
    assert said.startswith(f"{NOT_PUBLIC}: the results depend on 1 file its publisher does not make available for "
                           f"download: {LAYER} (licence {LICENCE.id}; request it from: {REQUEST}). With the file in "
                           f"place, `mime verify <folder>` checks it by its sha256.")
    assert "from this bundle alone" not in said


def test_the_card_says_it_and_lists_it_by_request(folder: Path, data: bytes, tmp_path: Path):
    with own_cache(tmp_path) as cache:
        placed(cache, scan(data), data)
        text = card_text(folder)
    assert f"{NOT_PUBLIC}: the results depend on 1 file" in text.split("## Availability")[1]
    assert f"| `{LAYER}` | a third party's | {LICENCE.id} | not permitted | by request |" in text


def test_a_lock_asset_with_no_location_and_a_presentation_one(folder: Path, tmp_path: Path):
    """An asset the lock names that no file stands for (a lab's record an entry reads; a mesh the stage draws): its
    NOTICE lines, and the statement's sentences, the softer one for role presentation; pack warns of each."""
    copy = Path(shutil.copytree(folder, tmp_path / "held"))
    lock = json.loads((copy / LOCK_FILE).read_text(encoding="utf-8"))
    lab = AssetRef("lab_table", "ab" * 32, 179, "CC-BY-4.0", (), role="results", request="doi:10.5281/zenodo.123")
    look = AssetRef("lab_look", "cd" * 32, 4_000_000, "CC-BY-4.0", (), role="presentation")
    lock["assets"] = [*lock["assets"], lab.to_record(), look.to_record()]
    (copy / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    text = notice(copy)
    assert "  available from: no download location (by request); request it from: doi:10.5281/zenodo.123" in text
    assert "  available from: no download location (by request); request it from: no route given" in text
    said = text.split("Availability\n------------\n")[1]
    assert f"{NOT_PUBLIC}: the results depend on 2 files their publishers do not make available for download" in said
    assert "lab_table (licence CC-BY-4.0; request it from: doi:10.5281/zenodo.123)" in said
    assert ("Showing the experiment needs 1 file its publisher does not make available for download (role "
            "presentation: no result depends on it): lab_look (licence CC-BY-4.0; request it from: no route given)."
            ) in said
    with pytest.warns(ByRequestWarning) as warned:
        pack(copy, out=tmp_path / "held.mimebundle")
    lines = sorted(str(each.message) for each in warned if each.category is ByRequestWarning)
    assert len(lines) == 3 and any("the asset lab_look (licence CC-BY-4.0) has no download location" in line and
                                   "showing the experiment needs it, though no result depends on it" in line and
                                   "no route to request it is given: one is strongly recommended" in line
                                   for line in lines)


def drawn_terms(seed: int) -> FolderTerms:
    """Folder terms with one to four lock assets, each drawn with or without a location, of role results,
    presentation or none, with or without a route."""
    chosen = random.Random(seed)
    held: list[LockAsset] = []
    for index in range(chosen.randint(1, 4)):
        located = chosen.random() < 0.5
        ref = AssetRef(f"asset_{index}", f"{index:064x}", 1000 + index, "CC-BY-4.0",
                       ("https://example.org/a",) if located else (),
                       role=chosen.choice(["results", "presentation", None]),
                       request=chosen.choice([None, REQUEST]))
        held.append(LockAsset(ref, licence_verdict(ref.licence, {}), False))
    return FolderTerms((), "CC-BY-4.0", {}, (), 14, tuple(held), "0.3.0")


def test_the_statement_says_not_public_whenever_results_depend_on_a_file_no_one_can_download():
    """Over drawn folder terms (``MIME_SEED``, and twenty seeds here): the sentence appears exactly where an asset of
    a role that is not presentation has no location, and the softer one exactly where one of role presentation has
    none. ``MIME_PLANT_BY_REQUEST=silent`` removes the sentences' parts: a planted run fails here."""
    plant = os.environ.get("MIME_PLANT_BY_REQUEST", "")
    base = int(os.environ.get("MIME_SEED", "0"))
    with pytest.MonkeyPatch.context() as patch:
        if plant == "silent":
            patch.setattr(_card, "_by_request_parts", lambda judged: ([], []))
        for seed in range(base, base + 20):
            judged = drawn_terms(seed)
            said = _card.availability(judged)
            results = [each for each in judged.assets if each.ref.by_request and each.ref.role != "presentation"]
            shown = [each for each in judged.assets if each.ref.by_request and each.ref.role == "presentation"]
            assert said.startswith(NOT_PUBLIC) == bool(results), (seed, said)
            assert ("Showing the experiment needs" in said) == bool(shown), (seed, said)
            assert all(each.ref.name in said for each in [*results, *shown]), (seed, said)
            assert ("from this bundle alone." in said and not results) or "from this bundle alone." not in said


def test_push_lists_it_never_uploads_it_and_warns(folder: Path, data: bytes, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    copy = Path(shutil.copytree(folder, tmp_path / "pushed"))
    lock = json.loads((copy / LOCK_FILE).read_text(encoding="utf-8"))
    open_record = AssetRef("lab_table", "ab" * 32, 179, "CC0-1.0", (), role="results", request=REQUEST)
    lock["assets"] = [*lock["assets"], open_record.to_record()]
    (copy / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(copy)
    fake = FakeRecords(with_token(monkeypatch))
    with own_cache(tmp_path) as cache:
        placed(cache, scan(data), data)
        plan = _plan.make_plan(copy_of(copy, tmp_path / "push"), fake.service(), push.read_manifest(copy).record,
                               tmp_path / "staged")
    choices = {choice.ref.name: choice for choice in plan.assets}
    assert not choices["lab_table"].upload, "a record with no location is never uploaded, whatever its licence"
    assert choices["lab_table"].why == "no download location (by request): travels as its record, never uploaded"
    assert hashlib.sha256(data).hexdigest() not in {file.sha256 for file in plan.files}
    lines = [str(line) for line in plan.lines(None)]
    assert any(line.startswith(f"    {LAYER}  ") and f"no download location (by request), request it from: {REQUEST}"
               in line for line in lines), lines
    assert (f"    the asset lab_table  179 bytes  licence CC0-1.0: no download location (by request): travels as its "
            f"record, never uploaded, request it from: {REQUEST}") in lines
    (warning,) = [each for each in plan.warnings(None) if "no one can download" in each]
    assert warning == (f"results depend on 2 files no one can download ({LAYER}, request it from: {REQUEST}; the asset "
                       f"lab_table, request it from: {REQUEST}): not reproducible from public sources, as the card "
                       f"says")


def test_a_bundle_of_format_three_holding_a_record_with_no_location_is_refused(folder: Path, tmp_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ByRequestWarning)
        bundle = pack(folder, out=tmp_path / "scanned.mimebundle")

    def as_format_three(source: Path, target: Path, *, strip: bool) -> Path:
        with zipfile.ZipFile(source) as archive:
            members = {info.filename: (info, archive.read(info)) for info in archive.infolist()}
        manifest = json.loads(members[MANIFEST_MEMBER][1])
        manifest["format"] = FORMAT_3
        if strip:
            manifest["assets"] = []
        manifest.pop("bundle_hash")
        manifest["bundle_hash"] = bundle_hash(manifest)
        info = members[MANIFEST_MEMBER][0]
        members[MANIFEST_MEMBER] = (info, (json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n")
                                    .encode())
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
            for info, held in members.values():
                archive.writestr(info, held)
        return target

    with pytest.raises(BuildRefused) as refused:
        unpack(as_format_three(bundle, tmp_path / "three.mimebundle", strip=False), tmp_path / "three")
    assert refused.value.codes() == ["bundle.manifest"]
    assert "holds assets[0].request, assets[0].locations (empty), which mime.bundle/4 added" in str(refused.value)
    with pytest.raises(BuildRefused) as control:
        unpack(as_format_three(bundle, tmp_path / "stripped.mimebundle", strip=True), tmp_path / "stripped")
    assert "mime.bundle/4 added" not in str(control.value), "the control: the format check, not the step's, refuses it"


def manifest_of(format_: str, ref: AssetRef) -> dict[str, object]:
    return {"format": format_, "name": "scans", "title": "scans", "credit": "a lab", "notice": "a notice",
            "archives": [], "assets": [dict(ref.to_record())], "files": {}}


def test_a_manifest_file_before_the_third_format_holding_a_record_with_no_location_is_refused(data: bytes):
    for earlier in ("mime.asset_manifest/2", "mime.asset_manifest/1"):
        with pytest.raises(BuildRefused) as refused:
            assets.read_manifest_data(manifest_of(earlier, scan(data, request=None)), "scans.json")
        assert refused.value.codes() == ["asset.manifest"] and "which mime.asset_manifest/3 added" in str(refused.value)
        located = AssetRef("vessel_scan.usda", scan(data).sha256, len(data), "CC-BY-4.0", ("https://example.org/s",))
        assert [ref.sha256 for ref in assets.read_manifest_data(manifest_of(earlier, located), "scans.json").assets] \
            == [located.sha256], "the control: the same manifest with a located record is read"
    held = assets.read_manifest_data(manifest_of("mime.asset_manifest/3", scan(data)), "scans.json")
    assert held.assets[0].by_request and held.to_json()["format"] == "mime.asset_manifest/3"
    lock_form = assets.read_manifest_data(manifest_of("mime.asset_manifest/1", scan(data)), "the lock", lock=True)
    assert lock_form.assets[0].request == REQUEST, "a lock's manifest holds what the lock's form allows"


def test_a_lock_of_a_later_form_is_refused_before_its_records_are_read(folder: Path, tmp_path: Path):
    copy = Path(shutil.copytree(folder, tmp_path / "later"))
    assert [ref.name for ref in assets.folder_assets(copy)] == ["vessel_scan.usda"], "the control: this form reads"
    lock = json.loads((copy / LOCK_FILE).read_text(encoding="utf-8"))
    lock["schema"] = "mime.lock/15"
    lock["assets"][0]["a_later_key"] = True
    (copy / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        assets.folder_assets(copy)
    assert refused.value.codes() == ["load.newer-form"] and "mime.lock/15" in str(refused.value)
