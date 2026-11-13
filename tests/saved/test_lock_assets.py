"""A saved experiment records the remote assets it reads: the build that reads the first one writes the lock's
``assets`` key (``mime.assets``'s README, "What a folder's lock references"). A tube experiment's ``Drag.MRS`` reads
the corrected tube table through its wall kind, so its lock names ``unit_tube_R0.997``: ``mime assets fetch <folder>``
stores it, and the script ``mime modal`` writes fetches it onto the assets Volume before a run. An experiment that
reads none writes no key.

**The control**, failing on every run: a lock without the key, its experiment reading the table, is refused when the
experiment is loaded, naming the asset (``load.assets``). A lock naming the table under another hash is the
experiment's pin, and the pin wins: the load reads that hash (design #229), and a shared source holding another version
of the name is reported in the build report, never taken.

The tube experiment reads the cached asset (``mime assets fetch unit_tube_R0.997``; CI stores it once per job)."""

import json
import shutil
import zipfile
from pathlib import Path

import pytest
from mrs_cases import BALL, mrs, mrs_tube, free_space

import mime.env._modal_script as modal_generator
from mime import assets
from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.nodes.hydro.mrs._tube import UNIT_TUBE
from mime.saved import LOCK_FILE, LoadRefused, card_text, notice, pack
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


def test_the_tube_experiments_bundle_says_what_a_reuser_needs_of_mimes_own_table(saved: Path, tmp_path: Path):
    """The bundle's NOTICE and the card name the table (the folder holds no file of it, and a bundle never carries it):
    its licence, where it is available from, and what a reuser needs, MIME at the version that saved the folder.
    Before the fix both read only the folder's files, and said "None named" and "from this bundle alone"."""
    bundle = pack(saved, out=tmp_path / "mrs_tube.mimebundle")
    with zipfile.ZipFile(bundle) as archive:
        held = archive.read("NOTICE").decode()
    assert held == notice(saved)
    version = lock_of(saved)["environment"]["mime"]  # type: ignore[index]
    assert "unit_tube_R0.997\n  sha256 " + TABLE.sha256 in held and "MIME's own record; licence CC0-1.0" in held
    assert f"needs: MIME {version}, the version that saved the folder" in held
    assert "Licences\n--------\nCC0-1.0: https://spdx.org/licenses/CC0-1.0.html\n" in held
    assert "None named." not in held and "from this bundle alone" not in held
    assert f"given 1 asset the lock names that the bundle does not carry, which `mime assets fetch <folder>` fetches: " \
           f"unit_tube_R0.997 (not carried in a bundle; licence CC0-1.0;" in held
    card = card_text(saved)
    assert "| `unit_tube_R0.997` | MIME's own record | CC0-1.0 | permitted | referenced |" in card
    assert "from this bundle alone" not in card


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


def test_the_pin_wins_a_lock_naming_the_table_under_another_hash_reads_that_hash(saved: Path, tmp_path: Path):
    """The lock is the experiment's pin (design #229): a load reads the table under the hash its lock names, never the
    one the installed MIME ships under that name, and a table nothing has stored is refused as missing, with the command
    that fetches exactly what the lock names."""
    other = {**TABLE.to_record(), "sha256": "ab" * 32, "size": 1234}

    def rehash(lock: dict[str, object]) -> None:
        held = lock["assets"]
        assert isinstance(held, list)
        held[0] = other

    with pytest.raises(BuildRefused) as refused:
        Experiment.load(edited(saved, tmp_path / "rehashed", rehash))
    assert refused.value.codes() == ["asset.missing"] and "mime assets fetch unit_tube_R0.997" in str(refused.value)
    assert "1 kB" in str(refused.value), "the size the lock pins, not the shipped table's"


def test_a_shared_source_holding_another_version_of_the_table_is_reported_and_the_pin_kept(saved: Path,
                                                                                         monkeypatch: pytest.MonkeyPatch):
    from mime.assets import _sources

    real = _sources.shared()
    newer = assets.AssetRef(UNIT_TUBE, "cd" * 32, 99, "CC0-1.0", ("https://example.org/newer.npz",))
    records = {**real.records, UNIT_TUBE: (*real.records[UNIT_TUBE], _sources.Sourced(newer, "labx"))}
    monkeypatch.setattr(_sources, "shared", lambda: _sources.Shared(records, real.manifests))
    built = Experiment.load(saved).build()
    assert any(f"asset {UNIT_TUBE}: the experiment keeps its pin, sha256 {TABLE.sha256}" in line and "labx" in line
               for line in built.report.assets), built.report.assets
