"""One study folder for the plot, recipe, comparison and overlay tests: script A's reproduction arm over three frequencies,
recorded every 5 ms for 50 ms with its stage kept at the same spacing, written by ``Study.run_to``. A test that changes a
file works on a copy."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest
from script_a import reproduction_arm

from mime import Study
from mime.results import StudyResult
from mime.study import grid
from mime.units import Hz, s

SPEED = "swim_speed:umr.speed"
FREQUENCIES = (10.0, 20.0, 30.0)
STAGE_EVERY = 0.005


@pytest.fixture(scope="session")
def study_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("figures") / "study"
    exp = reproduction_arm()
    field = exp.handle("field")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Study(exp).sweep(grid({field.commands.frequency: [Hz(f) for f in FREQUENCIES]}),  # type: ignore[union-attr]
                         duration=s(0.05), record=[SPEED], every=s(STAGE_EVERY), stage_every=s(STAGE_EVERY)).run_to(out)
    return out


@pytest.fixture
def copy_of_study(study_folder: Path, tmp_path: Path) -> Path:
    target = tmp_path / "study"
    shutil.copytree(study_folder, target)
    return target


@pytest.fixture(scope="session")
def study(study_folder: Path) -> StudyResult:
    return StudyResult.read(study_folder)
