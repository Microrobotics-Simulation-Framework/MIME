"""Saved form 14: an asset record may name no location (``locations: []``), its file had by request alone, and a route
to request its file by (``request``). The lock alone moves, in two places: its ``assets`` and the records of its
``visuals`` manifests. Both are outside every hash, so a folder of an earlier form is read as it is, with nothing
written back; one whose lock holds either addition is refused, never read as declaring it.

The folder of the thirteenth form, ``form13/scanned``, is written by ``write_form13.py`` under MIME at the last commit of
form 13 (316937a8), float64, four usable CPUs (``taskset -c 0-3``), one BLAS thread, and pinned by its sha256: the toy
with a scene layer declared by its record (a record that may not be redistributed, with a location: the lock's
``assets`` holds it whole and its ``referenced_files`` names the layer) and a hub whose description's mesh a manifest
lists (the lock's ``visuals``). It holds what the step moves, in both places, as form 13's own code wrote it. The
layer itself is not committed: its record says it may not be redistributed, which the repository's audit holds the tree
to (``tests/assets/test_audit.py``). Each test writes it into its copy of the folder from the bytes it was made from
(``write_form13.scan_bytes``), pinned by the same sha256.

No ``form13/chain_screw`` is frozen: ``form12/chain_screw`` and ``current/chain_screw`` (form 13 before this step)
differ by their schema strings and the two hashes those strings move, nothing else, so a form-13 chain folder is the
current one's bytes before its restamp, and it holds no asset record.

**The controls.** Each place the step adds to, edited into the form-13 lock, is refused (``load.lock-shape``), and with
the step's check removed (:data:`~mime.saved._versions.STEP_14_LOCK_HELD` emptied) it is read as declaring it; the
unedited folder reads with none. Version 13's reader refuses a folder of version 14 as ``load.newer-form``, the document
and the lock each once, and the form left unstepped is read by it with none."""

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from test_form11 import codes_of, reader_of, restated

from mime import assets
from mime.assets import AssetRef
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, read_folder
from mime.saved import _versions
from mime.saved._terms import lock_records
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, HASH_ONLY_ADDED, LOCK_SCHEMAS
from mime.terms import CustomLicence

SAVED = Path(__file__).parent
SCANNED13 = SAVED / "form13" / "scanned"
PINNED = {"descriptions/c26b837a85da3a41-hub.urdf": "c26b837a85da3a41d0b45d0940234553f8e06988f74bee3509cbb4d4b01f8ef5",
          "experiment.mime.yaml": "fa3d4823cce56bf2736cbd004eb5da247720548ce6bec82ca7c2c86d94173699",
          "mime.lock": "a6ac5a0f36625432096baf11c878554dbe0d9565f66ac7888be49dcf8649eacf",
          "reference.npz": "dbb09bddea94816016c1947e206cb267d0fe3918f3f687a11e6e976573ae638a",
          "scene/layers/vessel_scan.usda": "6905a2b17a4f82fd46002c15b6570f3d228d563dd995cd4170fe0d73f3ad9340",
          "scene/scene.usda": "409c78650a58d9b414b1901a7dcc053276e5afffd54e9e87924a2911c989c910"}
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM13_READS = 13
SCAN_LAYER = "scene/layers/vessel_scan.usda"
REQUEST = "mailto:imaging@hospital.example.org"


def scanned13(out: Path) -> Path:
    """The frozen form-13 folder copied to ``out``, its scene layer written from the bytes it was made from."""
    from write_form13 import scan_bytes

    folder = Path(shutil.copytree(SCANNED13, out))
    (folder / SCAN_LAYER).parent.mkdir(parents=True, exist_ok=True)
    (folder / SCAN_LAYER).write_bytes(scan_bytes())
    return folder


@contextmanager
def own_cache(root: Path) -> Iterator[Path]:
    """An asset cache, a locations file and a licence acceptances file of the test's own, with no network."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MIME_ASSETS_DIR", str(root / "cache"))
        patch.setenv("MIME_OFFLINE", "1")
        (root / "locations.toml").write_text("", encoding="utf-8")
        patch.setenv("MIME_ASSET_LOCATIONS", str(root / "locations.toml"))
        patch.setenv("MIME_LICENCE_ACCEPTANCES", str(root / "licences.json"))
        yield root / "cache"


def placed(cache: Path, ref: AssetRef, data: bytes) -> None:
    """``data`` copied into the cache under the asset's own name, taken by a fetch (which downloads nothing), and its
    licence accepted here, as the person who placed it has."""
    from mime.assets import _acceptances

    cache.mkdir(parents=True, exist_ok=True)
    (cache / ref.name).write_bytes(data)
    _acceptances.remember(assets.licences_of(ref))
    assets.fetch(ref)


def scan_record(folder: Path, **changes: object) -> AssetRef:
    """The scan's record as ``folder``'s lock holds it, with ``changes``."""
    from dataclasses import replace

    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    return replace(AssetRef.from_record(lock["assets"][0]), **changes)  # type: ignore[arg-type]


def test_the_form_thirteen_folder_is_the_folder_its_commit_wrote_byte_for_byte(tmp_path: Path):
    folder = scanned13(tmp_path / "scanned")
    found = {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(folder.rglob("*")) if path.is_file()}
    assert found == PINNED


def test_the_form_thirteen_folder_holds_a_record_in_each_place_the_step_moves():
    """The premise of every test below: the frozen lock holds a record in its ``assets`` (whole: credit, role, kind,
    custom licence, citation), a file that travels as it, and a record inside a ``visuals`` manifest."""
    lock = json.loads((SCANNED13 / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["schema"] == "mime.lock/13" and lock["referenced_files"] == {SCAN_LAYER: "vessel_scan.usda"}
    (record,) = lock["assets"]
    assert set(record) == {"name", "sha256", "size", "licence", "locations", "credit", "role", "kind", "licences",
                           "citation"} and record["locations"]
    (visual,) = lock["visuals"]["hub_meshes"]["assets"]
    assert visual["locations"] and visual["role"] == "presentation"
    assert _versions.hash_only_held(lock) == []


def test_the_step_leaves_the_form_thirteen_folder_as_it_was():
    """Nothing is written back: the document reads as form 13's but for its schema, and its document, resolved record
    and derived graph written back into form 13 are what its lock holds; every record of the lock reads as written."""
    document = yaml.safe_load((SCANNED13 / DOCUMENT_FILE).read_text(encoding="utf-8"))
    lock = json.loads((SCANNED13 / LOCK_FILE).read_text(encoding="utf-8"))
    upgraded, version, _ = _versions.upgraded_document(document)
    assert version == 13 and {**upgraded, "schema": document["schema"]} == document
    assert _versions.document_in_form(upgraded, 13) == document
    assert _versions.resolved_in_form(lock["resolved"], 13) == lock["resolved"]
    assert _versions.physics_in_form(lock["derived"]["physics"], 13) == lock["derived"]["physics"]
    held, lock_version = _versions.upgraded_lock(lock)
    assert lock_version == 13 and {**held, "schema": lock["schema"]} == lock
    records = [*lock["assets"], *lock["visuals"]["hub_meshes"]["assets"]]
    assert [dict(ref.to_record()) for ref in lock_records(lock)] == records


def test_the_form_thirteen_folder_loads_rebuilds_and_verifies(tmp_path: Path):
    folder = scanned13(tmp_path / "scanned")
    with own_cache(tmp_path) as cache:
        placed(cache, scan_record(folder), (folder / SCAN_LAYER).read_bytes())
        loaded = Experiment.load(folder)
        assert loaded.load_report is not None
        assert loaded.load_report.text().splitlines()[1:3] == [
            f"saved form: mime.document/13, read as version {CURRENT}", "input hash: reproduced exactly"]
        assert loaded.build().verify().reproduced


#: Each place the step adds to, as an edit of a lock: a record's ``request``, and its locations emptied, in the lock's
#: ``assets`` and in a ``visuals`` manifest.
EDITS = {
    "assets: request": lambda lock: lock["assets"][0].update(request=REQUEST),
    "assets: no location": lambda lock: lock["assets"][0].update(locations=[]),
    "visuals: request": lambda lock: lock["visuals"]["hub_meshes"]["assets"][0].update(request=REQUEST),
    "visuals: no location": lambda lock: lock["visuals"]["hub_meshes"]["assets"][0].update(locations=[])}
HELD_AT = {"assets: request": "assets[0].request", "assets: no location": "assets[0].locations (empty)",
           "visuals: request": "visuals.hub_meshes.assets[0].request",
           "visuals: no location": "visuals.hub_meshes.assets[0].locations (empty)"}


def edited(tmp_path: Path, edit: str) -> Path:
    folder = scanned13(tmp_path / "edited")
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    EDITS[edit](lock)
    (folder / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return folder


@pytest.mark.parametrize("edit", EDITS)
def test_a_form_thirteen_lock_holding_what_form_fourteen_added_is_refused(tmp_path: Path, edit: str):
    assert codes_of(scanned13(tmp_path / "as_written")) == [], "the control: the folder as its commit wrote it"
    folder = edited(tmp_path, edit)
    with pytest.raises(LoadRefused) as refused:
        read_folder(folder)
    assert refused.value.codes()[0] == "load.lock-shape"
    assert f"a mime.lock/13 lock holds {HELD_AT[edit]}, which version {HASH_ONLY_ADDED} of the saved form added" in \
        str(refused.value)


@pytest.mark.parametrize("edit", EDITS)
def test_control_without_the_steps_check_an_edited_form_thirteen_lock_is_read(tmp_path: Path, edit: str):
    """The refusal above is the step's check, and nothing else's: with it removed, the edited lock is read."""
    folder = edited(tmp_path, edit)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "STEP_14_LOCK_HELD", ())
        assert read_folder(folder).version == 13


SCAN_LICENCE = CustomLicence("LicenseRef-Imaging-Lab-Restricted", "The imaging lab's data licence 1.0",
                             "https://example.org/imaging-lab/LICENSE.txt", "not_permitted")


@pytest.fixture(scope="module")
def by_request_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The toy with a scene layer declared by a record with no location and a route to request it, saved in this
    form."""
    from write_form13 import scan_bytes

    from mime.scene import asset
    from mime.testing._graph import declare_in_order
    from mime.testing.toys._experiments import ADD_PROBE, ATTACH_SPRING, toy_experiment

    root = tmp_path_factory.mktemp("form14")
    data = scan_bytes()
    record = AssetRef("vessel_scan.usda", hashlib.sha256(data).hexdigest(), len(data), SCAN_LICENCE.id, (),
                      credit="A Hospital's imaging lab", role="results", kind="measured", licences=(SCAN_LICENCE,),
                      request=REQUEST)
    with own_cache(root) as cache:
        placed(cache, record, data)
        experiment = declare_in_order(lambda: toy_experiment("scanned"), (ADD_PROBE, ATTACH_SPRING), None)
        experiment._settings.scene.reference(asset(record), at="/World/vessel_scan")  # noqa: SLF001
        experiment.licence("CC-BY-4.0")
        experiment.build().save(root / "scanned")
    return root / "scanned"


def test_version_fourteen_writes_a_record_with_no_location_and_reads_it_back(by_request_folder: Path, tmp_path: Path):
    assert HASH_ONLY_ADDED == CURRENT == 14
    assert DOCUMENT_SCHEMAS[CURRENT] == "mime.document/14" and LOCK_SCHEMAS[CURRENT] == "mime.lock/14"
    lock = json.loads((by_request_folder / LOCK_FILE).read_text(encoding="utf-8"))
    (record,) = lock["assets"]
    assert lock["schema"] == "mime.lock/14" and record["locations"] == [] and record["request"] == REQUEST
    assert lock["referenced_files"] == {SCAN_LAYER: "vessel_scan.usda"}
    folder = Path(shutil.copytree(by_request_folder, tmp_path / "scanned"))
    with own_cache(tmp_path) as cache:
        placed(cache, scan_record(folder), (folder / SCAN_LAYER).read_bytes())
        loaded = Experiment.load(folder)
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()


def test_version_thirteens_reader_refuses_a_folder_of_version_fourteen_as_newer_each_file_once(
        by_request_folder: Path):
    assert codes_of(by_request_folder) == [], "this MIME reads it"
    with reader_of(FORM13_READS):
        assert codes_of(by_request_folder) == [NEWER, NEWER]


def test_control_the_form_left_unstepped_is_read_by_version_thirteens_reader_with_no_newer_form(
        by_request_folder: Path, tmp_path: Path):
    unstepped = restated(by_request_folder, tmp_path / "unstepped", FORM13_READS)
    with reader_of(FORM13_READS):
        assert NEWER not in codes_of(unstepped)


def test_the_planted_controls_for_the_probe(by_request_folder: Path, tmp_path: Path):
    """The target ``detection_rate.py`` runs on a probe (``MIME_PLANT_FORM14``): unplanted, version 13's reader refuses
    this form's folder as newer, each file once, and an edited form-13 lock is refused; ``unstepped`` writes the folder
    as form 13 (the step not taken), ``unchecked`` removes the step's check of an earlier lock, the edit drawn from
    ``MIME_SEED``. A planted run fails here, which the probe counts as caught."""
    plant = os.environ.get("MIME_PLANT_FORM14", "")
    seed = int(os.environ.get("MIME_SEED", "0"))
    folder = restated(by_request_folder, tmp_path / "read", FORM13_READS) if plant == "unstepped" else \
        by_request_folder
    with reader_of(FORM13_READS):
        assert codes_of(folder) == [NEWER, NEWER]
    edit = list(EDITS)[seed % len(EDITS)]
    with pytest.MonkeyPatch.context() as patch:
        if plant == "unchecked":
            patch.setattr(_versions, "STEP_14_LOCK_HELD", ())
        assert codes_of(edited(tmp_path, edit))[:1] == ["load.lock-shape"]
