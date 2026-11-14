"""The studies the study-container tests pack: the full toy over its spring's stiffness and the run's step, two groups
(the second step compiles a program of its own, so group 1 keeps its own build), written once with a figure and a
comparison layer beside the data; and a publishing folder of two of them."""

from __future__ import annotations

import shutil
from pathlib import Path

from mime import Study
from mime.study import DT, grid
from mime.testing.toys import ToyRecorder
from mime.testing.toys._experiments import full_toy
from mime.units import N, m, s

RECORDED = "toy.recorder:probe.last"


def toy_study(stiffnesses: tuple[float, ...] = (1.1, 1.7), steps: tuple[float, ...] = (0.02, 0.01)) -> Study:
    exp = full_toy()
    exp.remove("toy.recorder:probe")
    exp.observe(ToyRecorder(body="probe", rate=s(0.04)))
    exp.licence("CC-BY-4.0")
    spring = exp.handle("toy.spring:probe")
    axes = {spring.k: [N(k) / m(1.0) for k in stiffnesses], DT: [s(dt) for dt in steps]}  # type: ignore[union-attr]
    return Study(exp).sweep(grid(axes), duration=s(0.2), record=[RECORDED])


def write_study(folder: Path, *, figure: bool = True) -> Path:
    """A toy study run into ``folder``, with a figure's recipe."""
    result = toy_study().run_to(folder)
    if figure:
        result.figure(RECORDED, name="trace").write_recipe()
    return folder


def copy_of(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


__all__ = ["RECORDED", "copy_of", "toy_study", "write_study"]
