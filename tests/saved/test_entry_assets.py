"""Every entry kind can declare the assets it reads, and a saved experiment's lock records them as it records an
interaction plan's ``Plan.assets``: each once, by its bytes (``Entry.assets``, a class variable, and ``assets_read``, which
an entry whose settings choose the file overrides). The lab fixture's ``labx.finished_gantry`` (a dynamics) reads one of
two records by its ``finish`` setting, and ``labx.marked_rotating`` (a field provider) reads the marker; before this, only
an interaction's plan could name an asset, so a lab's material file had to be named on a contact's plan.

**The control, failing on every run:** a lock that omits an asset only an entry declares is refused when loaded
(``load.assets``, naming it); and the experiment on the non-commercial finish records that record and not the open one,
which a collection that read the class variable (the open file) instead of ``assets_read`` would get the wrong way round.

Bits: no simulation output depends on the declaration, and an experiment whose entries declare none writes no key
(``tests/saved/test_lock_assets.py``)."""

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from cases import entry_assets_rig

from mime import assets
from mime.experiment import Experiment
from mime.saved import LOCK_FILE, LoadRefused, notice, pack

lab = pytest.importorskip("mime_lab_fixture")
MARKER, NC = assets.reference("labx_marker.usda"), assets.reference("labx_finish_nc.usda")


def rig(*, gantry: object | None = None, field: object | None = None) -> Experiment:
    """The case's rig, its gantry the lab's plain one unless given, so that the provider is the only entry declaring."""
    return entry_assets_rig(gantry if gantry is not None else lab.Gantry(), field)


def unmarked_field():
    from mime.fields import UniformRotating
    return UniformRotating(rotation_axis=(0.0, 0.0, 1.0), initial_direction=(1.0, 0.0, 0.0))


def saved(tmp_path: Path, experiment: Experiment) -> Path:
    folder = tmp_path / "rig"
    experiment.build().save(folder, steps=2)
    return folder


def lock_assets(folder: Path) -> list[dict[str, object]]:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8")).get("assets", [])


def names(folder: Path) -> list[object]:
    return [held["name"] for held in lock_assets(folder)]


def test_a_dynamics_entry_declaring_a_record_has_it_in_the_lock(tmp_path: Path):
    folder = saved(tmp_path, rig(gantry=lab.FinishedGantry(), field=unmarked_field()))
    assert lock_assets(folder) == [MARKER.to_record()]


def test_a_field_provider_declaring_a_record_has_it_in_the_lock(tmp_path: Path):
    folder = saved(tmp_path, rig())
    assert lock_assets(folder) == [MARKER.to_record()]


def test_one_record_declared_by_two_entries_is_listed_once(tmp_path: Path):
    folder = saved(tmp_path, rig(gantry=lab.FinishedGantry()))
    assert names(folder) == ["labx_marker.usda"]


def test_an_experiment_whose_entries_declare_none_writes_no_key(tmp_path: Path):
    folder = saved(tmp_path, rig(field=unmarked_field()))
    assert "assets" not in json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def test_the_bundles_notice_names_an_asset_only_an_entry_declares(tmp_path: Path):
    """#289's NOTICE reads the lock's assets: the record a dynamics declares is named, with its licence, and the bundle
    does not carry it."""
    folder = saved(tmp_path, rig(gantry=lab.FinishedGantry(), field=unmarked_field()))
    with zipfile.ZipFile(pack(folder, out=tmp_path / "rig.mimebundle")) as archive:
        held = archive.read("NOTICE").decode()
    assert held == notice(folder)
    assert "labx_marker.usda\n  sha256 " + MARKER.sha256 in held and "CC-BY-4.0" in held
    assert "None named." not in held


def test_an_entry_choosing_its_record_by_a_setting_puts_only_the_chosen_one_in_the_lock_and_notice(tmp_path: Path):
    """The control for ``assets_read``: the class variable names the open file, so a collection that read it would
    record the open file for the non-commercial finish too, and the non-commercial licence would be named by a run on
    the open one if both were declared."""
    open_folder = saved(tmp_path / "open", rig(gantry=lab.FinishedGantry(finish="marker"), field=unmarked_field()))
    nc_folder = saved(tmp_path / "nc", rig(gantry=lab.FinishedGantry(finish="nc"), field=unmarked_field()))
    assert names(open_folder) == ["labx_marker.usda"] and names(nc_folder) == ["labx_finish_nc.usda"]
    assert lab.FinishedGantry.assets == ("labx_marker.usda",), "the class variable is the default, not the choice"
    wanted = {open_folder: ("labx_marker.usda", "CC-BY-4.0", "labx_finish_nc.usda", "CC-BY-NC-4.0"),
              nc_folder: ("labx_finish_nc.usda", "CC-BY-NC-4.0", "labx_marker.usda", "CC-BY-4.0")}
    for folder, (read, licence, unread, other) in wanted.items():
        held = notice(folder)
        assert read in held and licence in held
        assert unread not in held and f"{other}: " not in held and f"{other}\n" not in held, held


def test_control_a_lock_without_an_asset_only_an_entry_declares_is_refused_when_loaded_naming_it(tmp_path: Path):
    folder = saved(tmp_path, rig())
    broken = tmp_path / "broken"
    shutil.copytree(folder, broken)
    lock = json.loads((broken / LOCK_FILE).read_text(encoding="utf-8"))
    lock.pop("assets")
    (broken / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(broken)
    assert refused.value.codes() == ["load.assets"]
    assert "labx_marker.usda" in str(refused.value)
