"""A block used: the labx lab's probe (``lab_fixture/blocks/nudged_probe/``), a part of an experiment another lab
published, in an experiment of your own; then saved as the block's case, and loaded once the block is gone. The user
guide (``docs/guide/blocks.md``) includes this file by its regions and prints its output, which
``tests/docs/test_tutorials.py`` and ``mime docs check`` regenerate. It works on a copy of the block, in a folder of its own
that it removes, and writes nothing into the block it reads."""

import shutil
import tempfile
from pathlib import Path

import yaml

from mime.experiment import Experiment
from mime.push import check
from mime.scene import Scene, Z
from mime.testing.toys import ToyRelaxation
from mime.things import Medium
from mime.units import kg, s, standard_gravity

LAB = Path(__file__).resolve().parents[1] / "lab_fixture" / "blocks" / "nudged_probe"

with tempfile.TemporaryDirectory() as scratch:
    block = Path(shutil.copytree(LAB, Path(scratch) / "nudged_probe"))

    # region: use
    experiment = Experiment("probe_in_water", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(),
                            dt=s(0.02))
    water = experiment.add_medium(Medium("water", flow=ToyRelaxation()))
    nudged = experiment.use(block, medium=water)
    print(f"the block provides {nudged.probe.name!r}, and binds its socket medium to {water.name!r}")
    experiment.declare("probe.mass", kg(2.5), reason="our unit, weighed")  # no warning: recorded in plan().replaced
    print(experiment.plan().replaced[0])
    experiment.licence("CC-BY-4.0")
    print(yaml.safe_dump({"blocks": experiment.document()["blocks"]}, sort_keys=False, width=110), end="")
    # endregion use

    # region: case
    experiment.save(block / "cases" / "in_water", steps=20)
    checked = check(block)
    print(f"mime check: {len(checked.findings)} findings, {len(checked.of('to do'))} to do")
    # endregion case

    # region: without
    saved = Path(scratch) / "probe_in_water"
    experiment.save(saved, steps=20)
    shutil.rmtree(block)
    loaded = Experiment.load(saved)
    assert loaded.load_report is not None
    print(f"with the block's folder gone: {loaded.load_report.text().splitlines()[2]}")
    # endregion without
