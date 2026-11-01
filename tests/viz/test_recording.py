"""The recording layer in a run folder (``run_to(..., stage_every=)``, and a live session's ``record``), on A0's apparatus closed by ``ArmTracking`` (``live_cases.tracked``):
its poses and vectors are the run's states at its samples' steps exactly, read back through usd-core; it opens with
no finding of usd-core's validators; it is crate 0.8.0, which OpenUSD 24.08 reads; and the folder moved elsewhere still
composes its stage. Its controls: the same poses compared one sample off differ, and a recording whose sublayer is
written as an absolute path into the folder no longer composes once the folder moves. Two checks hold the binding to
what it stands for rather than to its own rows: the recorded rotor turns on its base by the motor's angle in the state
(a binding that swapped the two frames fails it), and on a stage in millimetres a recorded body, composed with the
stage's metres per unit, is where the run put it (a recording that wrote metres into it fails)."""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from live_cases import built
from pxr import Sdf, Usd

from mime.build import ArgumentRefused
from mime.scene import VECTOR_VALUE, check_stage
from mime.units import ms
from mime.viz import RECORDING_FILE, crate_version, recording_layer

EVERY = 6  # base steps between samples: three of the rig's control periods


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    folder = tmp_path_factory.mktemp("rec") / "run"
    held = built()
    written = held.run_to(folder, ms(120), stage_every=ms(EVERY), history=True)
    history = written.result.history
    assert history is not None
    return folder, written, held.binding(), history


def recorded_values(path: Path, binding, samples: int):
    stage = Usd.Stage.Open(str(path))
    found = {}
    for entry in binding.entries:
        prim = stage.GetPrimAtPath(entry.prim)
        if entry.kind == "pose":
            found[entry.prim] = [np.array([*prim.GetAttribute("xformOp:translate").Get(code),
                                           prim.GetAttribute("xformOp:orient").Get(code).GetReal(),
                                           *prim.GetAttribute("xformOp:orient").Get(code).GetImaginary()])
                                 for code in range(samples)]
        else:
            found[entry.prim] = [np.array(prim.GetAttribute(VECTOR_VALUE).Get(code)) for code in range(samples)]
    return found


def state_rows(binding, history, steps, shift=0):
    rows = {}
    for entry in binding.entries:
        values = []
        for step in steps:
            if step == 0:
                values.append(None)
                continue
            held = np.asarray(history[entry.node][entry.field][step - 1 + shift], dtype=np.float64)
            values.append(held if entry.index is None else held[entry.index])
        rows[entry.prim] = values
    return rows


def misfits(found, rows) -> list[str]:
    return [f"{prim} sample {code}" for prim, values in found.items() for code, value in enumerate(values)
            if rows[prim][code] is not None and not np.array_equal(value, rows[prim][code])]


def test_the_recordings_poses_and_vectors_are_the_runs_states_exactly(recorded):
    folder, written, binding, history = recorded
    assert written.recording == folder / RECORDING_FILE
    layer = Sdf.Layer.FindOrOpen(str(written.recording))
    data = layer.customLayerData["mime:recording"]
    steps = list(data["steps"])
    assert steps == list(range(0, 121, EVERY)) and data["every_steps"] == EVERY and data["samples"] == len(steps)
    assert layer.timeCodesPerSecond == pytest.approx(1.0 / (EVERY * 0.001))
    assert {entry.kind for entry in binding.entries} == {"pose", "vector"}
    found = recorded_values(written.recording, binding, len(steps))
    assert misfits(found, state_rows(binding, history, steps)) == []
    # The control: one sample off, every pose of the body differs.
    off = misfits(found, state_rows(binding, history, steps[:-1] + [0], shift=1))
    assert {line for line in off if line.startswith("/World/umr ")} == {f"/World/umr sample {c}"
                                                                       for c in range(1, len(steps) - 1)}


def test_it_opens_with_no_finding_and_as_crate_0_8_0(recorded):
    folder, written, _, _ = recorded
    assert check_stage(written.recording) == []
    assert crate_version(written.recording) == (0, 8, 0)
    stage = Usd.Stage.Open(str(written.recording))
    assert [Path(layer.identifier).name for layer in stage.GetLayerStack()][1:] == [RECORDING_FILE, "scene.usda"]
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/World") and stage.HasAuthoredTimeCodeRange()


def test_the_folder_moved_still_composes_its_stage(recorded, tmp_path):
    folder, written, binding, _ = recorded
    moved = tmp_path / "elsewhere" / "run"
    shutil.copytree(folder, moved)
    stage = Usd.Stage.Open(str(moved / RECORDING_FILE))
    assert Path(stage.GetLayerStack()[-1].identifier).resolve() == (moved / "experiment/scene/scene.usda").resolve()
    assert all(stage.GetPrimAtPath(entry.prim).IsValid() for entry in binding.entries)
    # The control: a recording whose sublayer names the original folder's stage by its absolute path loses it once
    # that folder is gone.
    absolute = tmp_path / "absolute"
    shutil.copytree(folder, absolute)
    layer = Sdf.Layer.FindOrOpen(str(absolute / RECORDING_FILE))
    layer.subLayerPaths[0] = str((absolute / "experiment/scene/scene.usda").resolve())
    layer.Export(str(absolute / "pinned.usda"))
    shutil.copytree(absolute, tmp_path / "absolute_moved")
    shutil.rmtree(absolute)
    pinned = Usd.Stage.Open(str(tmp_path / "absolute_moved" / "pinned.usda"))
    assert not pinned.GetPrimAtPath("/World/tube").IsValid()


def test_a_spacing_that_is_not_whole_chunks_is_refused():
    with pytest.raises(ArgumentRefused, match="run.stage-every"):
        built().run_to("unused", ms(12), stage_every=ms(3))


def test_a_live_sessions_recording_is_its_run_folders(tmp_path):
    session = built().live(pace=None, out=tmp_path / "live")
    session.step(2)
    session.record(True)
    session.step(3)
    folder = session.stop()
    path = folder / RECORDING_FILE
    data = Sdf.Layer.FindOrOpen(str(path)).customLayerData["mime:recording"]
    assert list(data["steps"]) == [session.chunk_steps * n for n in range(2, 6)]
    assert check_stage(path) == []


def test_the_layer_names_its_experiment(recorded):
    _, written, _, _ = recorded
    data = Sdf.Layer.FindOrOpen(str(written.recording)).customLayerData["mime:recording"]
    assert data["experiment"] == "tracked" and data["input_hash"].startswith("sha256:")
    assert data["stage"] == "experiment/scene/scene.usda"
    assert recording_layer.__module__ == "mime.viz._recording"


def frame_prims(stage) -> dict[str, str]:
    """Each chain frame's prim by the frame's name, as the stage writes it on the prim (``mime:frame``): what a client
    finds in the stage alone, with no binding."""
    return {str(prim.GetAttribute("mime:frame").Get()): str(prim.GetPath()) for prim in stage.Traverse()
            if prim.GetAttribute("mime:frame") and str(prim.GetPath()).startswith("/World/rpm/motor/")}


def test_the_recorded_rotor_turns_on_its_base_by_the_motors_angle(recorded):
    """The binding against the kinematics, not against itself: in the recording, composed as a client composes it,
    the rotor frame is the base frame turned about its z by the motor's angle in the run's state, at every sample.
    A binding that swapped the two frames' rows records the base turned by minus the angle, and fails. The control
    of the oracle: the angle grows over the run (the motor turns), so the two frames are told apart."""
    from pxr import Gf, UsdGeom

    _, written, _, history = recorded
    stage = Usd.Stage.Open(str(written.recording))
    prims = frame_prims(stage)
    assert set(prims) == {"base", "rotor"}
    steps = list(Sdf.Layer.FindOrOpen(str(written.recording)).customLayerData["mime:recording"]["steps"])
    angles = []
    for code, step in enumerate(steps):
        if step == 0:
            continue
        cache = UsdGeom.XformCache(code)
        base, rotor = (cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prims[name])) for name in ("base", "rotor"))
        x_base, z_base = (np.array(base.TransformDir(Gf.Vec3d(*axis))) for axis in ((1, 0, 0), (0, 0, 1)))
        x_rotor, z_rotor = (np.array(rotor.TransformDir(Gf.Vec3d(*axis))) for axis in ((1, 0, 0), (0, 0, 1)))
        assert np.allclose(z_rotor, z_base, atol=1e-6)
        turned = math.atan2(float(np.dot(np.cross(x_base, x_rotor), z_base)), float(np.dot(x_base, x_rotor)))
        angle = float(history["rpm.motor"]["angle"][step - 1])
        assert math.remainder(turned - angle, 2 * math.pi) == pytest.approx(0.0, abs=1e-5), (step, turned, angle)
        angles.append(angle)
    assert max(angles) - min(angles) > 0.5


def mm_stage_experiment():
    from mime import Body, Experiment, State, Structure
    from mime.dynamics import RigidInertial
    from mime.scene import Scene, Tube, X, Z
    from mime.units import kg, kg_m2, m, m_per_s, mm, rad_per_s, standard_gravity

    exp = Experiment("mm_stage", scene=Scene.new(up=Z, units="mm"), gravity=standard_gravity(), dt=ms(1))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3), wall=mm(1), length=mm(9))))
    exp.add_body(Body("probe", dynamics=RigidInertial(), initial=State(
        position=m((0.001, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("probe.mass", kg(1e-4), reason="a probe of a millimetre body's order")
    exp.declare("probe.inertia", kg_m2((1e-10,) * 3), reason="a probe of a millimetre body's order")
    return exp


def test_a_recording_on_a_stage_in_millimetres_places_each_body_where_the_run_put_it(tmp_path):
    """A frame and the binding carry metres; the stage's translate is in its own units. Composed with the stage's
    metres per unit, as a client reads it, the recorded body is where the run's state put it, in metres, at every
    sample. A recording that wrote metres straight into a millimetre stage places it a thousand times too far, and
    fails. The control: the body is off the origin (x = 1 mm), so the two differ."""
    from pxr import UsdGeom

    held = mm_stage_experiment().build()
    entry = next(entry for entry in held.binding().entries if entry.kind == "pose")
    start = np.asarray(held.current_state()[entry.node][entry.field][:3], dtype=np.float64)
    written = held.run_to(tmp_path / "run", ms(20), stage_every=ms(10), history=True)
    stage = Usd.Stage.Open(str(written.recording))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    assert scale == pytest.approx(0.001)
    assert entry.unit.startswith("m;")
    steps = list(Sdf.Layer.FindOrOpen(str(written.recording)).customLayerData["mime:recording"]["steps"])
    assert steps == [0, 10, 20]
    for code, step in enumerate(steps):
        world = UsdGeom.XformCache(code).GetLocalToWorldTransform(stage.GetPrimAtPath(entry.prim))
        placed = np.array(world.ExtractTranslation()) * scale
        state = np.asarray(written.result.history[entry.node][entry.field][step - 1][:3], dtype=np.float64) \
            if step else start
        assert np.allclose(placed, state, rtol=1e-12, atol=0.0), (step, placed, state)
    assert abs(state[0]) > 0
