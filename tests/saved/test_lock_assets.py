"""A saved experiment records the remote assets it reads: the build that reads the first one writes the lock's
``assets`` key (``mime.assets``'s README, "What a folder's lock references"). A tube experiment's ``Drag.MRS`` reads
the corrected tube table through its wall kind, so its lock names ``unit_tube_R0.997``: ``mime assets fetch <folder>``
stores it, and the script ``mime modal`` writes fetches it onto the assets Volume before a run. An experiment that
reads none writes no key.

**The control**, failing on every run: a lock without the key, its experiment reading the table, is refused when the
experiment is loaded, naming the asset (``load.assets``); so is a lock naming the table under another hash.

The tube experiment reads the cached asset (``mime assets fetch unit_tube_R0.997``; CI stores it once per job)."""

import json
import shutil
from pathlib import Path

import pytest
from mrs_cases import BALL, mrs, mrs_tube, free_space

import mime.env._modal_script as modal_generator
from mime import assets
from mime.experiment import Experiment
from mime.nodes.hydro.mrs._tube import UNIT_TUBE
from mime.saved import LOCK_FILE, LoadRefused
from mime.units import mm

TABLE = assets.reference(UNIT_TUBE)


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tube") / "mrs_tube"
    mrs_tube().build().save(folder, steps=2)
    return folder


def lock_of(folder: Path) -> dict[str, object]:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def edited(saved: Path, target: Path, change) -> Path:
    shutil.copytree(saved, target)
    lock = lock_of(target)
    change(lock)
    (target / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return target


def test_a_tube_experiments_lock_names_the_tube_table_and_its_folder_references_it(saved: Path):
    assert lock_of(saved)["assets"] == [TABLE.to_record()]
    assert assets.folder_assets(saved) == (TABLE,)


def test_the_saved_tube_experiment_loads_and_its_modal_script_fetches_its_assets(saved: Path, tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch):
    Experiment.load(saved)
    monkeypatch.setattr(modal_generator, "installed_checkout", lambda: None)
    written = modal_generator.modal_script(saved, out=tmp_path / "modal", wheel_sha256=lambda version: "ab" * 32)
    assert "\nASSETS = True" in written.script.read_text(encoding="utf-8")


def test_an_experiment_reading_no_remote_asset_writes_no_key(tmp_path: Path):
    folder = tmp_path / "ball"
    free_space("ball", {"ball": BALL}, mrs(mm(1.5))).build().save(folder, steps=2)
    assert "assets" not in lock_of(folder) and assets.folder_assets(folder) == ()


def test_control_a_lock_without_the_tables_name_is_refused_when_loaded_naming_it(saved: Path, tmp_path: Path):
    folder = edited(saved, tmp_path / "missing", lambda lock: lock.pop("assets"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    print(f"\n{refused.value}")
    assert refused.value.codes() == ["load.assets"]
    assert UNIT_TUBE in str(refused.value) and "mime assets fetch" in str(refused.value)


def test_control_a_lock_naming_the_table_under_another_hash_is_refused_when_loaded(saved: Path, tmp_path: Path):
    other = assets.reference("unit_tube_R0.997_1667_modes")

    def rehash(lock: dict[str, object]) -> None:
        held = lock["assets"]
        assert isinstance(held, list)
        held[0] = {**TABLE.to_record(), "sha256": other.sha256, "size": other.size}

    with pytest.raises(LoadRefused) as refused:
        Experiment.load(edited(saved, tmp_path / "rehashed", rehash))
    assert refused.value.codes() == ["load.assets"] and other.sha256 in str(refused.value)
