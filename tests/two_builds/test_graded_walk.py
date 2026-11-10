"""Graded sampling under the two-builds check: a graded sampling's choice is discrete (the farthest points, Lloyd's cells,
each cell's centred point) and a graded density is no scale of the body's, so a move of a setting far below the spacing
can tip a near-tie and choose other force points, a bake in the step's program. The graded path chooses at its settings,
the spacing and the contact rounded to ``FIGURES`` (``mime.scene._grading``), and moves the points with the body's own
shape, so that a move of a relative 1e-6 of a value written to three figures keeps the choice.

The walk, on the two bodies the refuting review found re-choosing: FL-9 graded towards its features at 0.6 mm in free
space, and a helix's contact graded above a plane at 0.03 mm; each number of the kind and of the drag a study can vary,
moved by a relative 1e-6 in turn, the trace form (``run=False``): quiet, or not discriminating where the move reaches no
leaf. **The control**, on the same bodies: the rounding made exact (``FIGURES = 17``, every float its own), the pairs
measured to re-choose fire (``CONTROL``), so a quiet walk is the rounding, not a form that cannot see a re-choice.
Float64 (the check refuses float32)."""

from __future__ import annotations

from typing import Any

import pytest
from mrs_cases import fl9, free_space, mrs

import mime.scene._grading as grading
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.interactions import Drag
from mime.scene import Helix, Plane, Scene, Z
from mime.testing import two_builds_of
from mime.things import Body, Medium, Part, State, Structure
from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s


def screw() -> Experiment:
    return free_space("graded_screw_walk", {"ball": fl9()},
                      mrs(mm(0.6), blob=0.25, quadrature_refinement=8, grading="features"))


def helix() -> Experiment:
    experiment = Experiment("graded_helix_walk", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=(0.0, 0.0, 1.0), offset=m(0.0))))
    experiment.add_medium(Medium("water", fills=floor.half_space))
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(
        Helix(radius=mm(0.15), pitch=mm(0.6), turns=2.5, filament_radius=mm(0.03)))},
        initial=State(position=mm((0.0, 0.0, 0.3)), orientation=(1.0, 0.0, 0.0, 0.0),
                      velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-9), reason="a sub-millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-17, 1e-17, 1e-17)), reason="a sub-millimetre body's order")
    experiment.attach(Drag.MRS(spacing=mm(0.03), blob=0.25, wall_kind="plane", contact=(0.0, 0.0, -1.0),
                               contact_gap=mm(0.015)), body="ball", medium="water")
    return experiment


#: Each body, and the numbers walked: its kind's a study can vary (the helix's turns is held as built, a rebuild) and
#: its drag's.
WALKED: dict[str, tuple[Any, tuple[str, ...]]] = {
    "screw": (screw, ("ball.parts.shell.radius", "ball.parts.shell.length", "ball.parts.shell.wavenumber",
                      "ball.parts.shell.modulation", "drag:ball.blob")),
    "helix": (helix, ("ball.parts.shell.radius", "ball.parts.shell.pitch", "ball.parts.shell.filament_radius",
                      "drag:ball.blob", "drag:ball.contact_gap", "drag:ball.spacing")),
}
#: The pairs measured to choose other force points with the rounding made exact (the fix round's probe).
CONTROL = {"screw": "ball.parts.shell.length", "helix": "ball.parts.shell.filament_radius"}


@pytest.mark.parametrize("body", sorted(WALKED))
def test_a_graded_sampling_keeps_its_choice_under_every_move_of_the_walk(body: str):
    make, paths = WALKED[body]
    experiment = make()
    built = experiment.build()
    verdicts = {path: two_builds_of(experiment, path, built=built, run=False).verdict for path in paths}
    assert set(verdicts.values()) <= {"quiet", "not-discriminating"}, verdicts
    assert "quiet" in verdicts.values(), "a move reached a leaf, so the quiet arm is evidence"


@pytest.mark.parametrize("body", sorted(CONTROL))
def test_control_with_the_rounding_exact_the_measured_pair_fires(body: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grading, "FIGURES", 17)
    make, _ = WALKED[body]
    experiment = make()
    assert two_builds_of(experiment, CONTROL[body], run=False).verdict == "bake"
