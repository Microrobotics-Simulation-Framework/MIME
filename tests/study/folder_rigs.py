"""The studies the folder tests write: the full toy over its spring's stiffness and the run's step (two groups, series of
two lengths: ragged), and script A over the field's frequency and the step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from script_a import reproduction_arm

from mime import Study
from mime.results import RunResult, StudyResult
from mime.study import DT, grid
from mime.testing.toys import ToyRecorder
from mime.testing.toys._experiments import full_toy
from mime.units import Hz, N, m, ms, s

RECORDED = "toy.recorder:probe.last"
SPEED = "swim_speed:umr.speed"
STIFFNESSES = (1.1, 1.7, 2.3)


def toy():
    exp = full_toy()
    exp.remove("toy.recorder:probe")
    exp.observe(ToyRecorder(body="probe", rate=s(0.04)))
    return exp


def toy_study(steps=(0.02, 0.01), stiffnesses=STIFFNESSES, batch=None, duration=0.4, reasons=None) -> Study:
    exp = toy()
    spring = exp.handle("toy.spring:probe")
    axes = {spring.k: [N(k) / m(1.0) for k in stiffnesses]}  # type: ignore[union-attr]
    if steps:
        axes[DT] = [s(dt) for dt in steps]
    return Study(exp).sweep(grid(axes), duration=s(duration), record=[RECORDED], batch=batch, reasons=reasons)


def script_a_study(frequencies=(10.0, 30.0), steps=(5e-4, 2.5e-4), duration=0.02) -> Study:
    exp = reproduction_arm()
    field = exp.handle("field")
    return Study(exp).sweep(grid({field.commands.frequency: [Hz(f) for f in frequencies],  # type: ignore[union-attr]
                                  DT: [s(dt) for dt in steps]}), duration=s(duration), record=[SPEED])


def same_run(a: RunResult, b: RunResult) -> list[str]:
    """What differs between two runs, by name: every final field, every recorded series' steps and values, every flag
    track, the manifests' values, exactly (``np.array_equal``)."""
    found = [f"final {n}.{f}" for n in a.final for f in a.final[n] if not np.array_equal(a.final[n][f], b.final[n][f])]
    found += [f"array {k}" for k, v in a.arrays().items() if k not in b.arrays() or not np.array_equal(v, b.arrays()[k])]
    found += [] if a.manifest() == b.manifest() else ["manifest"]
    return found


def same_study(a: StudyResult, b: StudyResult) -> list[str]:
    """What differs between two studies' results: members, groups, points, coordinates (SI values and units), every
    member's run, exactly."""
    found: list[str] = []
    if len(a) != len(b) or a.groups != b.groups or a.points != b.points:
        return ["shape"]
    for path, held in a.coordinates.items():
        other = b.coordinates[path]
        left, right = (held.value, other.value) if hasattr(held, "value") else (held, other)
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            found.append(f"coordinate {path}")
    for i in range(len(a)):
        found += [f"member {i}: {each}" for each in same_run(a.member(i), b.member(i))]
    return found


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


__all__ = ["RECORDED", "SPEED", "STIFFNESSES", "Hz", "ms", "same_run", "same_study", "script_a_study", "toy",
           "toy_study", "tree_bytes"]
