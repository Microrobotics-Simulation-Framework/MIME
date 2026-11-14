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

import dataclasses
import json
import shutil
import zipfile
from functools import partial
from pathlib import Path

import pytest

from cases import entry_assets_rig
from control_cases import Ticker, spun_magnet
from mrs_cases import free_space, mrs

from mime import assets
from mime.drives import Constant
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.interactions.magnetic import MagneticTorque
from mime.observations import SwimSpeed
from mime.registry import (
    WallForm,
    WallKind,
    register_controller,
    register_drive,
    register_geometry,
    register_interaction,
    register_observation,
    register_wall_kind,
)
from mime.saved import LOCK_FILE, LoadRefused, notice, pack
from mime.saved._record import remote_assets
from mime.units import kg, kg_m2, m_per_s, mm, rad, rad_per_s
from mime.vocabulary import register_namespace

lab = pytest.importorskip("mime_lab_fixture")
MARKER, NC = assets.reference("labx_marker.usda"), assets.reference("labx_finish_nc.usda")


def rig(*, gantry: object | None = None, field: object | None = None) -> Experiment:
    """The case's rig, its gantry the lab's plain one unless given (a class), so that the provider is the only entry
    declaring."""
    return entry_assets_rig(gantry if gantry is not None else lab.Gantry, field)


def unmarked_field():
    from mime.fields import UniformRotating
    from mime.scene import X
    return UniformRotating(rotation_axis=X, initial_direction=(0.0, 1.0, 0.0))


def saved(tmp_path: Path, experiment: Experiment) -> Path:
    folder = tmp_path / "rig"
    experiment.build().save(folder, steps=2)
    return folder


def lock_assets(folder: Path) -> list[dict[str, object]]:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8")).get("assets", [])


def names(folder: Path) -> list[object]:
    return [held["name"] for held in lock_assets(folder)]


def test_a_dynamics_entry_declaring_a_record_has_it_in_the_lock(tmp_path: Path):
    folder = saved(tmp_path, rig(gantry=lab.FinishedGantry, field=unmarked_field()))
    assert lock_assets(folder) == [MARKER.to_record()]


def test_a_field_provider_declaring_a_record_has_it_in_the_lock(tmp_path: Path):
    folder = saved(tmp_path, rig())
    assert lock_assets(folder) == [MARKER.to_record()]


def test_one_record_declared_by_two_entries_is_listed_once(tmp_path: Path):
    folder = saved(tmp_path, rig(gantry=lab.FinishedGantry))
    assert names(folder) == ["labx_marker.usda"]


def test_an_experiment_whose_entries_declare_none_writes_no_key(tmp_path: Path):
    folder = saved(tmp_path, rig(field=unmarked_field()))
    assert "assets" not in json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def test_the_bundles_notice_names_an_asset_only_an_entry_declares(tmp_path: Path):
    """#289's NOTICE reads the lock's assets: the record a dynamics declares is named, with its licence, and the bundle
    does not carry it."""
    folder = saved(tmp_path, rig(gantry=lab.FinishedGantry, field=unmarked_field()))
    with zipfile.ZipFile(pack(folder, out=tmp_path / "rig.mimebundle")) as archive:
        held = archive.read("NOTICE").decode()
    assert held == notice(folder)
    assert "labx_marker.usda\n  sha256 " + MARKER.sha256 in held and "CC-BY-4.0" in held
    assert "None named." not in held


def test_an_entry_choosing_its_record_by_a_setting_puts_only_the_chosen_one_in_the_lock_and_notice(tmp_path: Path):
    """The control for ``assets_read``: the class variable names the open file, so a collection that read it would
    record the open file for the non-commercial finish too, and the non-commercial licence would be named by a run on
    the open one if both were declared."""
    open_folder = saved(tmp_path / "open", rig(gantry=partial(lab.FinishedGantry, finish="marker"),
                                                          field=unmarked_field()))
    nc_folder = saved(tmp_path / "nc", rig(gantry=partial(lab.FinishedGantry, finish="nc"), field=unmarked_field()))
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


# -- every geometry a build uses, and every kind that is collected --------------------------------------------------

MARKER_NAME = "labx_marker.usda"
register_namespace("entryassets", "0.0.1")


def marked(base: type, name: str, register):
    """``base`` registered again under ``name``, reading the lab's marker: a test's own entry, which only reads."""
    return register(dataclasses.dataclass(frozen=True, kw_only=True)(type(base.__name__ + "Marked", (base,), {
        "name": name, "version": "0.1.0", "assets": (MARKER_NAME,), "__module__": __name__})))


MarkedBead = marked(lab.Bead, "entryassets.bead", register_geometry)
MarkedConstant = marked(Constant, "entryassets.constant", register_drive)
MarkedTicker = marked(Ticker, "entryassets.ticker", register_controller)
MarkedSwimSpeed = marked(SwimSpeed, "entryassets.swim_speed", register_observation)
MarkedTorque = marked(MagneticTorque, "entryassets.torque", register_interaction)


def read_names(experiment: Experiment) -> list[str]:
    return [held.name for held in remote_assets(experiment.plan())]


def bead_body(bead) -> Experiment:
    """A bead as a body's part, the ordinary way a shape is a body's (the planner's part line, not a structure's)."""
    return free_space("bead", {"bead": bead}, mrs(mm(0.45)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


def test_a_geometry_kind_used_as_a_bodys_part_solid_has_its_record_in_the_lock(tmp_path: Path):
    """A body's shape is a ``Part``, and the planner's part lines are not among a structure's geometries: collection
    that read only the structures' left the lock without the record."""
    folder = saved(tmp_path, bead_body(MarkedBead(radius=mm(0.8))))
    assert lock_assets(folder) == [MARKER.to_record()]


def test_a_geometry_kind_used_as_a_structures_geometry_is_collected_beside_a_part_solid():
    """The structure's route held already; held here so that a collection that moves to the parts' route keeps it. (The
    test's own kind is no installed package's, so the saved lock itself is the part-solid test's.)"""
    from mime.things import Structure

    experiment = bead_body(lab.Bead(radius=mm(0.8)))
    assert read_names(experiment) == []
    experiment.add_structure(Structure("anchor", geometry=MarkedBead(radius=mm(0.8))))
    assert read_names(experiment) == [MARKER_NAME]


def test_a_geometry_kind_used_as_an_apparatus_parts_solid_has_its_record_collected():
    assert read_names(spun_magnet(magnet_solid=lab.Bead(radius=mm(0.8)))) == []
    assert read_names(spun_magnet(magnet_solid=MarkedBead(radius=mm(0.8)))) == [MARKER_NAME]


#: Each kind other than a dynamics, a provider and a geometry: the experiment holding MIME's own entry of it, and the
#: one holding the marked entry (``spun_magnet``: a motor, a body, a torque law, a controller).
KINDS = {
    "drive": (lambda: spun_magnet(drive_angle=True, uncontrolled=True),
              lambda: spun_magnet(drive=MarkedConstant(rad(0.25)), uncontrolled=True)),
    "controller": (lambda: spun_magnet(), lambda: spun_magnet(controller=MarkedTicker())),
    "observation": (lambda: spun_magnet(observe=True),
                    lambda: _observed(MarkedSwimSpeed("umr", along=(1.0, 0.0, 0.0)))),
    "interaction": (lambda: spun_magnet(), lambda: spun_magnet(torque=MarkedTorque())),
}


def _observed(observation) -> Experiment:
    experiment = spun_magnet()
    experiment.observe(observation)
    return experiment


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_other_kind_declaring_a_record_is_collected_and_MIMEs_own_entry_of_it_is_not(kind: str):
    """The control for each kind: the same experiment with MIME's own entry of the kind collects nothing, and with the
    marked one the record, so a collection that drops the kind fails its row."""
    own, marked_one = KINDS[kind]
    assert read_names(own()) == []
    assert read_names(marked_one()) == [MARKER_NAME]


# -- a wall kind has no instance in a build: a declaration there is refused, never silent ---------------------------


@pytest.mark.parametrize("attributes", [{"assets": (MARKER_NAME,)}, {"assets_read": lambda self: (MARKER_NAME,)}],
                         ids=["assets", "assets_read"])
def test_a_wall_kind_declaring_assets_is_refused_at_registration_naming_the_route_that_works(attributes):
    """A wall kind is never an instance in a plan, so its ``assets`` would be ignored: registration refuses it, naming
    the interaction's ``Plan.assets``."""
    with pytest.raises(TypeError, match=r"Plan\.assets.*wall_correction_named"):
        register_wall_kind(dataclasses.dataclass(frozen=True, kw_only=True)(type("AssetWall", (WallKind,), {
            "name": "entryassets.wall", "version": "0.1.0", "differentiability": GradientStatus.EXACT,
            "form": WallForm.ANALYTIC, "matches": None, "__module__": __name__, **attributes})))
