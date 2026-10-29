"""A chain's description files in a saved folder: two apparatuses whose arms are described by files of one name in
different folders keep each its own copy, so that the folder loads and builds what was saved."""

from pathlib import Path

from chain_cases import DESCRIPTIONS, experiment

from mime.dynamics import Arm
from mime.experiment import Experiment
from mime.scene import asset
from mime.things import Apparatus
from mime.units import rad

PLANAR = DESCRIPTIONS / "three_link_planar.urdf"


def arm_named_alike(root: Path, folder: str, upper: str) -> Path:
    """The planar arm in ``root/folder/arm.urdf``, its first joint's upper limit written as ``upper``."""
    path = root / folder / "arm.urdf"
    path.parent.mkdir()
    text = PLANAR.read_text()
    assert 'upper="3.1416"' in text
    path.write_text(text.replace('upper="3.1416"', f'upper="{upper}"', 1))
    return path


def two_arms(root: Path) -> Experiment:
    exp = experiment("two_arms")
    for name, folder, upper in (("one", "a", "3.1416"), ("two", "b", "2.0")):
        exp.add_apparatus(Apparatus(name, chain={"arm": Arm.Kinematic(
            description=asset(arm_named_alike(root, folder, upper)), initial_positions=rad((0.0, 0.0, 0.0)))},
            commands={"arm.joint_positions": rad((0.1, 0.2, 0.3))}))
    return exp


def test_two_description_files_of_one_name_are_saved_each_and_load_and_build_as_saved(tmp_path):
    saved = tmp_path / "saved"
    built = two_arms(tmp_path).build()
    built.save(saved)
    copies = sorted(path.name for path in (saved / "descriptions").iterdir())
    assert len(copies) == 2 and all(name.endswith("-arm.urdf") for name in copies)
    rebuilt = Experiment.load(saved).build()
    assert rebuilt.report.derived_hash == built.report.derived_hash
    assert rebuilt.verify().reproduced
