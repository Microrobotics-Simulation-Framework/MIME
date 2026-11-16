"""G24, the static mesh's refusals, each by name with its route, and the derivatives that pass with a mesh in view.

- **A derivative in any of the detector's nine numbers** while the camera traces a mesh is refused (``camera.mesh-derivative``) by
  ``Study.gradient`` in forward and in reverse mode and by ``Study.fit``, before anything is compiled, naming ``hide=``
  with the mesh's path: the mesh's part is traced under ``stop_gradient``, so the derivative would leave it out.
- **A derivative through the poses** (the settling bead's density, which moves its silhouette through the dynamics) with
  the mesh in view is not refused, and agrees with central differences of the same run as the in-loop camera's forward
  gate holds it (``test_in_loop_forward.py``'s tolerance: the box edge's allowance per unit of step plus the objective's
  rounding), in both modes (reverse against forward).
- **A moving mesh** (time-sampled points) is refused (``camera.mesh-moving``), **a camera that may move** with a mesh in
  view (``camera.mesh-mounted``), and **an open mesh named in ``absorbers``** (``camera.mesh-open``); the open mesh not
  named there is imaged (by its display opacity, or opaque).

**The control** (``CAMERA_PLANT=no-mesh-refusal``): the camera's refusal removed. The forward derivative in ``fy`` is then
taken, and on the mesh's top wall (rows 5 to 16, which the bead never reaches) it is exactly zero, where central
differences of the frame see the wall's edge and chord move: the comparison fails on every run (a code patch, no draw).
It shows the refusal is what stands between a user and that silent zero."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, STATED_MEMORY, TOO_LITTLE_MEMORY, ballistic, planted, stated_memory
from mesh_cases import VESSEL, box, placed, rotation, tube, tube_mesh_scene, with_vessel
from scenery_cases import write_gprims
from test_in_loop_forward import BALL, KINK, SPACING, objective
from test_in_loop_mounts import moving_gantry

from mime import Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.observations import Camera, Mounted, World
from mime.scene import Scene, Sphere, Tube, X, Z
from mime.assets import asset
from mime.sensing._node import INTRINSICS, CameraNode
from mime.study import Study, grid
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

if planted("no-mesh-refusal"):
    CameraNode.derivative_leaves_problem = lambda self, leaves: None  # type: ignore[method-assign]

#: The turn that lays a tube drawn along z along x, the closed-form tube's axis.
ALONG_X = rotation((2 ** -0.5, 0.0, 2 ** -0.5, 0.0))
#: The mesh tube's top wall: rows 5 to 16 of the 160 x 120 frame, every column (its edge near row 7).
WALL = (slice(5, 17), slice(None))
ACKNOWLEDGED = {"camera:": "a derivative checked here"}


def detector_scene(fy: float = 750.0, around: int = 64) -> Experiment:
    return tube_mesh_scene(around=around, detector=dataclasses.replace(DETECTOR, fy=fy))


def wall_of(final) -> object:
    return jnp.sum(final["camera:"]["expected"][WALL])


@pytest.mark.parametrize("how", ["forward", "reverse", "fit"])
def test_a_derivative_in_the_detector_with_a_mesh_in_view_is_refused_by_name(dtype, how):
    """Every one of the detector's nine numbers, each alone: a leaf is refused whichever it is, not only ``fy``."""
    names = {name: f"camera:.detector.{name}" for name in INTRINSICS}
    # the refusal comes before anything is made from the mesh, so a coarse tube (8 around) serves, unless the control runs
    study = Study(detector_scene(around=64 if planted("no-mesh-refusal") else 8)).sweep(
        grid({leaf: [float(getattr(DETECTOR, name))] for name, leaf in names.items()}), duration=ms(1))
    if planted("no-mesh-refusal") and how == "forward":
        found = study.gradient(wall_of, wrt=[names["fy"]], mode="forward", acknowledged=ACKNOWLEDGED)
        derivative = float(np.asarray(found.members[0].gradient[names["fy"]]).reshape(-1)[0])
        h = 1.0
        levels = []
        for fy in (750.0 + h, 750.0 - h):
            built = detector_scene(fy).build()
            built.run(ms(1))
            levels.append(float(wall_of(built.current_state())))
        central = (levels[0] - levels[1]) / (2 * h)
        assert abs(derivative - central) <= 1e-2 * abs(central), (derivative, central)
        return
    for name, leaf in names.items():
        start = float(getattr(DETECTOR, name))
        with pytest.raises(BuildRefused) as refused:
            if how == "fit":
                study.fit(lambda final: final["camera:"]["expected"].ravel(), start={leaf: start},
                                            noise=1.0, duration=ms(1), iterations=1, acknowledged=ACKNOWLEDGED)
            else:
                study.gradient(wall_of, wrt=[leaf], mode=how, acknowledged=ACKNOWLEDGED)
        if planted("no-mesh-refusal"):
            continue
        assert refused.value.codes() == ["camera.mesh-derivative"], name
        (refusal,) = refused.value.refusals
        assert f"detector's {name} while" in refusal.message and VESSEL in refusal.message, name
        assert f"hide=({VESSEL!r},)" in refusal.action


def settling(density: float = 1100.0) -> Experiment:
    """``test_in_loop_forward``'s settling bead in the tube's water, the closed-form tube hidden from the camera and its
    64-segment mesh imaged in its place."""
    exp = Experiment("settling", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10), length=um(480))))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((-50.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((2e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(20)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(density), reason="a bead 1.1 times water's density")
    exp.add_medium(Medium("water", fills="tube.lumen"))
    exp.declare("water.density", kg_per_m3(1000.0), reason="water")
    exp.attach(Gravity(), body="ball", medium="water")
    exp.observe(Camera(detector=DETECTOR, mount=World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS),
                       frame_period=ms(1), hide=("tube",), absorbers=((VESSEL, 3000.0),)))
    points, faces = tube(60e-6, 70e-6, 480e-6, 64)
    return with_vessel(exp, placed(points, np.zeros(3), ALONG_X), faces)


def bead(final) -> object:
    return objective(BALL, True, final["camera:"]["expected"])


@pytest.mark.memory_heavy
def test_a_derivative_through_the_poses_with_a_mesh_in_view_passes_in_both_modes(dtype, monkeypatch):
    """Reverse mode runs for real, on a stated memory the camera's check admits it on (the host's is not read: see
    ``in_loop_cases.STATED_MEMORY``); the refusal on too little is the next test's."""
    stated_memory(monkeypatch, STATED_MEMORY)
    study = Study(settling()).sweep(grid({"ball.parts.steel.density": [kg_per_m3(1100.0)]}), duration=ms(5))
    forward = float(list(study.gradient(bead, mode="forward", acknowledged=ACKNOWLEDGED).members[0].gradient.values())[0])
    h = 0.25
    levels = []
    for density in (1100.0 + h, 1100.0 - h):
        built = settling(density).build()
        built.run(ms(5))
        levels.append(float(bead(built.current_state())))
    central = (levels[0] - levels[1]) / (2 * h)
    size = max(abs(v) for v in levels)
    tolerance = KINK * h + SPACING * float(np.spacing(np.asarray(size, dtype=dtype))) / (2 * h * abs(forward))
    assert abs(forward) > 0.0 and abs(forward - central) <= tolerance * abs(forward), (forward, central, tolerance)
    reverse = float(list(study.gradient(bead, mode="reverse", acknowledged=ACKNOWLEDGED).members[0].gradient.values())[0])
    assert reverse == pytest.approx(forward, rel=1e-3 if dtype == "float32" else 1e-9)


def test_reverse_mode_with_a_mesh_in_view_is_refused_on_too_little_memory_and_admitted_on_the_stated_memory(dtype, monkeypatch):
    """The study the test above differentiates, on a stated memory: admitted on ``STATED_MEMORY`` and refused by name,
    before anything is compiled, on ``TOO_LITTLE_MEMORY`` (the control: the stated memory is what decides, and
    ``Study.gradient`` reads the planted one). Forward mode keeps no tape and is refused on neither."""
    from mime.study import _gradient

    def compiled(*_: object, **__: object) -> None:
        raise AssertionError("the derivative was compiled")

    node = settling().build().graph.get_node("camera:")
    stated_memory(monkeypatch, STATED_MEMORY)
    assert node.derivative_problem("reverse", 1, 1, 5) is None
    stated_memory(monkeypatch, TOO_LITTLE_MEMORY)
    problem = node.derivative_problem("reverse", 1, 1, 5)
    assert problem is not None and problem.code == "camera.reverse-memory"
    assert node.derivative_problem("forward", 1, 1, 5) is None
    monkeypatch.setattr(_gradient, "prepare", compiled)
    study = Study(settling()).sweep(grid({"ball.parts.steel.density": [kg_per_m3(1100.0)]}), duration=ms(5))
    with pytest.raises(BuildRefused) as refused:
        study.gradient(bead, mode="reverse", acknowledged=ACKNOWLEDGED)
    text = str(refused.value)
    assert "camera.reverse-memory" in text and "mime.sensing.render" in text and "a stated memory" in text


def refused(exp: Experiment) -> BuildRefused:
    with pytest.raises(BuildRefused) as error:
        exp.plan()
    return error.value


def test_a_moving_mesh_is_refused_by_name_with_its_route(tmp_path):
    exp = ballistic()
    held = exp._settings.scene  # noqa: SLF001
    held.reference(asset(write_gprims(tmp_path / "extras.usda")), at="/World/Extras")
    error = refused(exp)
    codes = [r for r in error.refusals if r.code == "camera.mesh-moving"]
    assert len(codes) == 1 and "/World/Extras/Animated" in codes[0].message
    assert "hide=('/World/Extras/Animated',)" in codes[0].action


def test_a_camera_that_may_move_with_a_mesh_in_view_is_refused_by_name_with_its_route():
    exp = moving_gantry(tube_mesh_scene(mount=Mounted(frame="stage.gantry.carriage", orientation=LOOK_ACROSS)))
    error = refused(exp)
    assert error.codes() == ["camera.mesh-mounted"]
    assert "stage.gantry.carriage" in error.refusals[0].message
    assert "mount=World(" in error.refusals[0].action and f"hide=({VESSEL!r},)" in error.refusals[0].action


def open_tube() -> tuple[np.ndarray, np.ndarray]:
    """The tube wall without its end rings: open at both ends."""
    points, faces = tube(60e-6, 70e-6, 480e-6, 32)
    keep = [i for i in range(len(faces)) if (i // 2) % 4 not in (2, 3)]
    return placed(points, np.zeros(3), ALONG_X), faces[keep]


def test_an_open_mesh_named_in_absorbers_is_refused_and_otherwise_imaged():
    points, faces = open_tube()
    named = with_vessel(ballistic(hide=("tube",), absorbers=((VESSEL, 3000.0),)), points, faces)
    error = refused(named)
    assert error.codes() == ["camera.mesh-open"]
    assert "edges" in error.refusals[0].message and "absorbers=((name, per_metre),) without" in error.refusals[0].action
    imaged = with_vessel(ballistic(hide=("tube",), absorbers=()), points, faces)
    assert f"{VESSEL} (" in imaged.build().report.text()


def test_a_closed_box_is_not_refused_as_open():
    points, faces = box((50e-6, 50e-6, 50e-6))
    exp = with_vessel(ballistic(hide=("tube",), absorbers=((VESSEL, 3000.0),)), points + np.array([0.0, 0.0, 2e-4]),
                      faces)
    assert "camera.mesh" not in exp.build().report.text()
