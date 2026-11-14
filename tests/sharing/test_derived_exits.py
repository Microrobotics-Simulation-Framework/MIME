"""A derived file takes the strictest terms of its inputs wherever bytes leave the machine (design #229 item 8): a
record that declares an open licence (``CC0-1.0``) and names its derivation record, whose input is a referenced file,
is referenced at every exit. Its derivation record is read by hash (here from the asset cache, where a file of that
hash is kept); one that cannot be read, and an input no record names, make it referenced too: terms nobody can check
never travel.

- **save**: the lock lists the derived file in ``referenced_files``, its record whole (its derivation among it);
- **pack**: no member of the bundle holds its bytes, and the manifest names it as fetched;
- **push**: it is never chosen for upload, and no file the record would hold has its bytes;
- **a run's folder**: the run's own saved form lists it as a reference, and neither the recording nor the run's
  experiment packed carries its bytes;
- **the guard**: a tree holding its bytes and its record is a finding, ``derived``.

``pack`` and ``push`` also hold their payload's bytes against every record in reach, whatever the folder's own records
say: a file the author brings as their own (no terms) whose bytes are a referenced record's in an installed source, a
derived file's of one, or a mesh baked into the local cache from a referenced source, is refused
(``bundle.referenced-copy``).

The controls, each failing on every run (nothing here is drawn, so a run is every run): every exit with the derivation
planted away (a record judged by its licence alone) carries the derived bytes; a pack with its payload check planted
away carries the author's copy."""

import hashlib
import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from push_cases import copy_of, with_token, write_sharing
from terms_cases import NO_REDISTRIBUTION, mesh_bytes, same_reference_run, write_mesh
from zenodo_records import FakeRecords

from mime import assets, push
from mime.assets._acceptances import remember
from mime.assets import AssetRef, DerivationInput, DerivationRecord, _derivation, _sources
from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.push import _plan
from mime.saved import LOCK_FILE, _bundle, pack
from mime.saved._card import judged_folder
from mime.scene import asset
from mime.testing._graph import declare_in_order
from mime.testing.toys._experiments import ADD_PROBE, ATTACH_SPRING, toy_experiment

CLOSED = mesh_bytes("closed_scan", 3000)
DERIVED = mesh_bytes("derived_scan", 2000)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def closed_record() -> AssetRef:
    return AssetRef("closed_scan.usda", sha(CLOSED), len(CLOSED), NO_REDISTRIBUTION.id,
                    ("https://example.org/closed/6a3ebb1/closed_scan.usda",), credit="a maker", role="results",
                    kind="authored", licences=(NO_REDISTRIBUTION,))


def derivation_text(*inputs: AssetRef) -> bytes:
    record = DerivationRecord("derived_scan.usda", sha(DERIVED), "a test's builder", "1", "0" * 40, {}, False,
                              tuple(DerivationInput("source", each.name, each.sha256) for each in inputs), {}, (), {})
    return record.text().encode("utf-8")


def derived_record(derivation: bytes) -> AssetRef:
    """The derived file's own record, as its author declares it: an open licence, its kind, its derivation by hash."""
    return AssetRef("derived_scan.usda", sha(DERIVED), len(DERIVED), "CC0-1.0",
                    ("https://example.org/derived/6a3ebb1/derived_scan.usda",), credit="the author", role="results",
                    kind="derived", derivation=sha(derivation))


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    held = tmp_path / "cache"
    held.mkdir()
    monkeypatch.setenv("MIME_ASSETS_DIR", str(held))
    monkeypatch.setenv("MIME_OFFLINE", "1")
    monkeypatch.setenv("MIME_LICENCE_ACCEPTANCES", str(tmp_path / "licences.json"))
    (tmp_path / "locations.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "locations.toml"))
    return held


def placed(cache: Path, data: bytes, record: AssetRef | None = None) -> None:
    """``data`` in the cache under its sha256, as a lab without the network places a file; checked in full where it
    is an asset's. A person who placed it has accepted its licence (a first use of a cached file asks)."""
    (cache / sha(data)).write_bytes(data)
    if record is not None:
        assets.fetch(record, accept_licences={each.id for each in assets.licences_of(record)})
        remember(assets.licences_of(record))  # the person who placed it accepted: a build's read of it is gated


@pytest.fixture
def records(cache: Path) -> tuple[AssetRef, AssetRef]:
    """The closed scan and the file derived from it, both in the cache, with the derivation record beside them."""
    closed = closed_record()
    derivation = derivation_text(closed)
    derived = derived_record(derivation)
    placed(cache, CLOSED, closed)
    placed(cache, DERIVED, derived)
    placed(cache, derivation)
    return closed, derived


def experiment(*declared: AssetRef | Path) -> Experiment:
    held = declare_in_order(lambda: toy_experiment("derived"), (ADD_PROBE, ATTACH_SPRING), None)
    for index, each in enumerate(declared):
        held._settings.scene.reference(asset(each), at=f"/World/layer{index}")
    held.licence("CC-BY-4.0")
    return held


def saved(held: Experiment, where: Path) -> Path:
    with same_reference_run():
        held.build().save(where)
    return where


def members(bundle: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(bundle) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def holding(found: dict[str, bytes], data: bytes) -> list[str]:
    return [name for name, held in found.items() if data in held]


DERIVED_PATH = "scene/layers/derived_scan.usda"


# --- each exit --------------------------------------------------------------------------------------------------------


def save_lists_it_as_a_reference(tmp_path: Path, derived: AssetRef) -> Path:
    folder = saved(experiment(closed_record(), derived), tmp_path / "saved")
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["referenced_files"].get(DERIVED_PATH) == "derived_scan.usda", lock.get("referenced_files")
    record = next(each for each in lock["assets"] if each["name"] == "derived_scan.usda")
    assert record == dict(derived.to_record()), "the lock holds the derived file's record whole"
    return folder


def pack_carries_none_of_its_bytes(tmp_path: Path, derived: AssetRef) -> None:
    folder = save_lists_it_as_a_reference(tmp_path, derived)
    bundle = pack(folder, out=tmp_path / "saved.mimebundle")
    found = members(bundle)
    assert not holding(found, DERIVED) and not holding(found, CLOSED)
    assert json.loads(found["manifest.json"])["referenced"][DERIVED_PATH]["name"] == "derived_scan.usda"


def push_never_uploads_it(tmp_path: Path, derived: AssetRef, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = save_lists_it_as_a_reference(tmp_path, derived)
    write_sharing(folder)
    fake = FakeRecords(with_token(monkeypatch))
    plan = _plan.make_plan(copy_of(folder, tmp_path / "push"), fake.service(), push.read_sharing(folder),
                           tmp_path / "staged")
    choice = next(each for each in plan.assets if each.ref.name == "derived_scan.usda")
    assert not choice.upload and choice.verdict.tier == "referenced"
    assert sha(DERIVED) not in {file.sha256 for file in plan.files}
    bundle = next(file for file in plan.files if file.key.endswith(".mimebundle"))
    assert not holding(members(bundle.path), DERIVED)


def a_run_folder_lists_it_and_carries_none_of_its_bytes(tmp_path: Path, derived: AssetRef) -> None:
    from mime.units import s
    from mime.viz import RECORDING_FILE

    built = experiment(closed_record(), derived).build()
    with same_reference_run():
        built.run_to(tmp_path / "run", s(0.08), stage_every=s(0.04))
    run = tmp_path / "run"
    lock = json.loads((run / "experiment" / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["referenced_files"].get(DERIVED_PATH) == "derived_scan.usda", lock.get("referenced_files")
    assert DERIVED not in (run / RECORDING_FILE).read_bytes()
    assert not holding(members(pack(run / "experiment", out=tmp_path / "run.mimebundle")), DERIVED)


def the_guard_finds_it(tmp_path: Path, derived: AssetRef) -> None:
    held = {"assets": [dict(closed_record().to_record()), dict(derived.to_record())]}
    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "records.json").write_text(json.dumps(held), encoding="utf-8")
    (root / "pkg" / "derived.bin").write_bytes(DERIVED)
    found = [(each.path, each.kind) for each in assets.audit(root).findings]
    assert found == [("pkg/derived.bin", "derived")], found


@pytest.mark.parametrize("exit_", [save_lists_it_as_a_reference, pack_carries_none_of_its_bytes,
                                   a_run_folder_lists_it_and_carries_none_of_its_bytes, the_guard_finds_it])
def test_a_derived_open_record_over_a_referenced_input_never_travels_as_bytes(records, tmp_path, exit_):
    exit_(tmp_path, records[1])


def test_push_never_uploads_a_derived_open_record_over_a_referenced_input(records, tmp_path, monkeypatch):
    push_never_uploads_it(tmp_path, records[1], monkeypatch)


def test_a_derivation_record_that_cannot_be_read_makes_the_file_referenced(cache, tmp_path):
    closed = closed_record()
    placed(cache, CLOSED, closed)
    derived = derived_record(derivation_text(closed))  # its derivation record kept nowhere
    placed(cache, DERIVED, derived)
    assert assets.record_terms(derived, {closed.sha256: closed}).by == _derivation.UNREAD_DERIVATION_BY
    pack_carries_none_of_its_bytes(tmp_path, derived)


def test_an_input_no_record_names_makes_the_file_referenced(cache, tmp_path):
    """The closed scan is declared nowhere and no source holds its record: its terms are nobody's to state."""
    derivation = derivation_text(closed_record())
    derived = derived_record(derivation)
    placed(cache, DERIVED, derived)
    placed(cache, derivation)
    folder = saved(experiment(derived), tmp_path / "alone")
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["referenced_files"] == {"scene/layers/derived_scan.usda": "derived_scan.usda"}
    assert not holding(members(pack(folder, out=tmp_path / "alone.mimebundle")), DERIVED)


def test_a_derived_record_over_open_inputs_travels_inside(cache, tmp_path):
    """The check's other side: inputs that may travel let the derived file travel, so the rule is the inputs' terms,
    not the word derived."""
    open_input = AssetRef("open_scan.usda", sha(CLOSED), len(CLOSED), "CC-BY-4.0",
                          ("https://example.org/open/6a3ebb1/open_scan.usda",), credit="a lab", role="results")
    placed(cache, CLOSED, open_input)
    derivation = derivation_text(open_input)
    derived = derived_record(derivation)
    placed(cache, DERIVED, derived)
    placed(cache, derivation)
    folder = saved(experiment(open_input, derived), tmp_path / "open")
    found = members(pack(folder, out=tmp_path / "open.mimebundle"))
    assert f"experiment/{DERIVED_PATH}" in found and DERIVED in found[f"experiment/{DERIVED_PATH}"]


@contextmanager
def derivation_ignored() -> Iterator[None]:
    """The plant: every exit judges a record by its own licence, never reading its derivation (the review's F1)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_derivation, "_record_terms", lambda ref, known, derivations, chain: ref.verdict)
        yield


@pytest.mark.parametrize("exit_", [save_lists_it_as_a_reference, pack_carries_none_of_its_bytes,
                                   a_run_folder_lists_it_and_carries_none_of_its_bytes, the_guard_finds_it])
def test_control_an_exit_that_reads_no_derivation_carries_the_derived_bytes(records, tmp_path, exit_):
    with derivation_ignored(), pytest.raises(AssertionError):
        exit_(tmp_path, records[1])


def test_control_a_push_that_reads_no_derivation_uploads_the_derived_bytes(records, tmp_path, monkeypatch):
    with derivation_ignored(), pytest.raises(AssertionError):
        push_never_uploads_it(tmp_path, records[1], monkeypatch)


# --- the payload's bytes, whatever the folder's records say ----------------------------------------------------------


@contextmanager
def installed(*refs: AssetRef) -> Iterator[None]:
    """An installed source holding ``refs``, beside the real ones."""
    real = _sources.shared()
    held = {key: list(value) for key, value in real.records.items()}
    for ref in refs:
        held.setdefault(ref.name, []).append(_sources.Sourced(ref, "labz"))
    planted = _sources.Shared({key: tuple(value) for key, value in held.items()}, real.manifests)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_sources, "shared", lambda: planted)
        yield


def own_copy(tmp_path: Path, data: bytes) -> Path:
    """A file the author brings as their own (no terms), holding ``data``."""
    path = write_mesh(tmp_path / "mine", "my_layer", 10)
    path.write_bytes(data)
    return path


def baked_into_the_cache(cache: Path, data: bytes) -> None:
    folder = cache / "visuals"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "k.json").write_text(json.dumps({"format": "mime.baked_mesh/3", "set": "maker", "file": "m/a.stl",
                                               "output": {"sha256": sha(data)}, "tier": "referenced"}),
                                   encoding="utf-8")


def copy_is_refused(tmp_path: Path, data: bytes) -> None:
    folder = saved(experiment(own_copy(tmp_path, data)), tmp_path / "copied")
    with pytest.raises(BuildRefused) as refused:
        pack(folder, out=tmp_path / "copied.mimebundle")
    assert refused.value.codes() == ["bundle.referenced-copy"] and "my_layer.usda" in str(refused.value)


def test_an_own_copy_of_a_referenced_record_an_installed_source_holds_is_refused(cache, tmp_path):
    with installed(closed_record()):
        copy_is_refused(tmp_path, CLOSED)


def test_an_own_copy_of_a_derived_file_whose_input_is_referenced_is_refused(records, tmp_path):
    closed, derived = records
    with installed(closed, derived):
        copy_is_refused(tmp_path, DERIVED)


def test_an_own_copy_of_a_mesh_baked_from_a_referenced_source_is_refused(cache, tmp_path):
    baked_into_the_cache(cache, CLOSED)
    copy_is_refused(tmp_path, CLOSED)


def test_an_own_file_no_record_names_is_carried(cache, tmp_path):
    folder = saved(experiment(own_copy(tmp_path, DERIVED)), tmp_path / "own")
    assert holding(members(pack(folder, out=tmp_path / "own.mimebundle")), DERIVED)


def test_control_a_pack_with_no_payload_check_carries_the_copy(cache, tmp_path, monkeypatch):
    monkeypatch.setattr(_bundle, "payload_problems", lambda members, judged, lock: [])
    baked_into_the_cache(cache, CLOSED)
    with pytest.raises(pytest.fail.Exception):
        copy_is_refused(tmp_path, CLOSED)


def test_push_never_uploads_an_asset_whose_bytes_an_installed_source_says_are_referenced(cache, tmp_path, monkeypatch):
    """The lock's record says the scan is open; an installed source holds the same bytes as referenced. Push takes
    the stricter, and never chooses it for upload."""
    open_one = AssetRef("closed_scan.usda", sha(CLOSED), len(CLOSED), "CC-BY-4.0",
                        ("https://example.org/closed/6a3ebb1/closed_scan.usda",), credit="a maker", role="results")
    placed(cache, CLOSED, open_one)
    folder = saved(experiment(open_one), tmp_path / "open")
    write_sharing(folder)
    fake = FakeRecords(with_token(monkeypatch))
    with installed(closed_record()):
        with pytest.raises(BuildRefused) as refused:
            _plan.make_plan(copy_of(folder, tmp_path / "push"), fake.service(), push.read_sharing(folder),
                            tmp_path / "staged")
        assert "bundle.referenced-copy" in refused.value.codes(), "the bundle would carry the scan"
        judged, _, lock = judged_folder(folder)
        choices = _plan.asset_choices([open_one], judged, lock)
    assert len(choices) == 1 and not choices[0].upload and choices[0].verdict.tier == "referenced"
