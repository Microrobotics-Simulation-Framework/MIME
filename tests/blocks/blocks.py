"""What the block tests share: the lab fixture's block (``lab_fixture/blocks/nudged_probe``) copied for a test to edit,
an experiment of the user's to use it in, an asset cache of the test's own, the seed and the plant.

A plant (``MIME_PLANT_P4A``) removes the check a test is the control of, so that the test fails; ``detection_rate.py``
runs each test with and without its plant, ``MIME_SEED`` drawing what the test plants or edits."""

from __future__ import annotations

import importlib
import os
import random
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime.experiment import Experiment
from mime.testing.toys import ToyRelaxation
from mime.testing.toys._experiments import toy_experiment
from mime.things import Medium, MediumHandle

ROOT = Path(__file__).resolve().parents[2]
LABX = ROOT / "lab_fixture" / "blocks" / "nudged_probe"
PLANT = "MIME_PLANT_P4A"
#: Conditional licences on MIME's list, and open ones.
CONDITIONAL = ("CC-BY-NC-4.0", "CC-BY-SA-4.0", "CC-BY-ND-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-ND-4.0")
OPEN = ("CC-BY-4.0", "CC0-1.0", "MIT", "Apache-2.0")


def plant() -> str:
    return os.environ.get(PLANT, "")


def seed() -> int:
    return int(os.environ.get("MIME_SEED", "0"))


def rng() -> random.Random:
    return random.Random(seed())


def lab_fixture() -> None:
    """The lab fixture package, which the block's [requires] names; installed in every CI lane."""
    try:
        importlib.import_module("mime_lab_fixture")
    except ImportError:
        if os.environ.get("CI"):
            pytest.fail("the lab fixture package is not installed; every CI lane installs it from ./lab_fixture")
        pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")


def block_copy(root: Path, name: str = "nudged_probe") -> Path:
    """The lab fixture's block, copied below ``root`` for a test to edit."""
    lab_fixture()
    return Path(shutil.copytree(LABX, root / name))


def edited(folder: Path, file: str, old: str, new: str) -> None:
    """``old`` replaced by ``new`` in the block's ``file``, which must hold it."""
    path = folder / file
    text = path.read_text(encoding="utf-8")
    assert old in text, (file, old)
    path.write_text(text.replace(old, new), encoding="utf-8")


def host(name: str = "host") -> tuple[Experiment, MediumHandle]:
    """An experiment of the user's: the toy's scene and step, and a medium of its own, ``water``."""
    experiment = toy_experiment(name)
    water = experiment.add_medium(Medium("water", flow=ToyRelaxation()))
    return experiment, water


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
