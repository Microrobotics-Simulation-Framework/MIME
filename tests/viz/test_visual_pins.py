"""A saved arm keeps its meshes' pins (design #229, "the pin wins"). Its stage names each mesh by its manifest's name
and sha256, found beside the description it was built from; saved, the folder holds the description and not the
manifest, and the lock holds the manifest's records (never their bytes), so ``mime visuals fetch <folder>`` fetches
them wherever the folder goes. Loaded, the rebuild names each mesh as the saved stage does, whatever manifest is
beside the description now, and the build report says where one now names another file.

The control, failing on every run: a rebuild that ignores the saved stage's pins writes a stage without the meshes'
names, and the load refuses it."""

import json
import shutil
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from chain_cases import kinematic_rpm
from visual_cases import ar4_manifest

from mime.experiment import Experiment
from mime.saved import LOCK_FILE, LoadRefused
from mime.saved import _pins
from mime.scene import VisualPins
from mime.viz import lock_manifests, manifests_for

#: The load is deterministic (no draw), so one run of the control is every run.
RUNS = 1


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("arm") / "kinematic_rpm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kinematic_rpm().build().save(folder, steps=2)
    return folder


def test_the_lock_holds_the_manifest_and_the_folder_holds_none(saved: Path):
    lock = json.loads((saved / LOCK_FILE).read_text(encoding="utf-8"))
    assert list(lock["visuals"]) == ["ar4_mk5"] and lock["schema"] == "mime.lock/7"
    held = lock["visuals"]["ar4_mk5"]
    assert len(held["assets"]) == 18 and {record["licence"] for record in held["assets"]} == {
        "LicenseRef-Annin-Robotics-NonCommercial"}
    assert not list(saved.rglob("*.json")) or all(path.name != "visuals.json" for path in saved.rglob("*.json"))
    assert [manifest.name for manifest in lock_manifests(saved)] == ["ar4_mk5"]
    assert [manifest.name for manifest in manifests_for(saved)] == ["ar4_mk5"]
    assert lock_manifests(saved)[0].to_json()["assets"] == ar4_manifest().to_json(lock=True)["assets"]
    assert held["format"] == "mime.asset_manifest/1" and "conditions" not in json.dumps(lock["visuals"]), (
        "the lock (saved form 6) holds the manifest as /1: its form has no key for a licence's condition lines")


def test_the_saved_arm_loads_with_its_stage_rebuilt_from_its_pins(saved: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment = Experiment.load(saved)
    assert experiment.load_report is not None


def test_a_manifest_beside_the_saved_description_that_names_another_file_is_noted_and_the_pin_kept(saved: Path,
                                                                                                tmp_path: Path):
    folder = tmp_path / "beside"
    shutil.copytree(saved, folder)
    held = ar4_manifest().to_json()
    held["assets"][0]["sha256"] = "ab" * 32
    (folder / "descriptions" / "visuals.json").write_text(json.dumps(held), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = Experiment.load(folder).build()
    lines = [line for line in built.report.assets if "keeps the mesh its saved stage names" in line]
    assert len(lines) == 1 and "ab" * 32 in lines[0]
    assert "assets kept at their pins" in built.report.text()


@contextmanager
def pins_ignored() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_pins, "saved_visual_pins", lambda form: VisualPins({}, {}))
        yield


def test_control_a_rebuild_that_ignores_the_saved_stages_pins_is_refused_every_run(saved: Path):
    caught = 0
    for _ in range(RUNS):
        with pins_ignored(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                Experiment.load(saved)
            except LoadRefused as refused:
                caught += "load.stage" in " ".join(refused.codes()) or "stage" in str(refused)
    assert caught == RUNS
