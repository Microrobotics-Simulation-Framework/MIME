"""Run folders whose recordings are written from chosen states, not from a run: a recording is the binding's fields
at each sample, so a test of what the view draws needs no physics (``mime.viz.write_recording`` over the folder's
``experiment/scene/scene.usda``, as ``run_to`` writes it); and the GLB read back."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, m_per_s2, mPa_s, ms, rad_per_s, um
from mime.viz import RUN_STAGE, StageSamples, write_recording
from mime.viz._gltf import read_glb

#: The settling ball's sphere: 10 um of steel, as ``examples/sphere_settling.py`` declares it, with no drag (the view
#: draws the stage and the recording; what moved the ball is not its concern).
BALL_RADIUS = 10e-6


def ball() -> Experiment:
    exp = Experiment("ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.1))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    held = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(10.0)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.attach(Gravity(), body=held, medium=water)
    return exp


type States = Mapping[str, Mapping[str, Any]]


def recorded(built: Any, folder: Path, count: int, state_at: Callable[[int, States], States], every: int = 1) -> Path:
    """A run folder of ``built`` with ``count`` samples ``every`` steps apart, sample ``k`` the state
    ``state_at(k, start)`` makes from the built experiment's start; its recording's path."""
    (folder / RUN_STAGE).parent.mkdir(parents=True, exist_ok=True)
    (folder / RUN_STAGE).write_text(built.stage_text, encoding="utf-8")
    start = {node: {name: np.asarray(value) for name, value in fields.items()}
             for node, fields in built.current_state().items()}
    samples = StageSamples(built.binding(), every)
    for k in range(count):
        samples.take(k * every, state_at(k, start))
    return write_recording(folder, built.binding(), samples, built.recording_settings())


def with_pose(start: States, node: str, pose: np.ndarray, index: int | None = None) -> dict[str, dict[str, Any]]:
    """``start`` with ``node``'s pose (its ``frame_poses`` row ``index`` where given) set to ``pose``."""
    held = {name: dict(fields) for name, fields in start.items()}
    if index is None:
        held[node]["pose"] = np.asarray(pose, dtype=np.float64)
    else:
        rows = np.array(held[node]["frame_poses"], dtype=np.float64, copy=True)
        rows[index] = pose
        held[node]["frame_poses"] = rows
    return held


def glb_parts(data: bytes) -> tuple[dict[str, Any], bytes]:
    return read_glb(data)


def accessor(document: Mapping[str, Any], binary: bytes, index: int) -> np.ndarray:
    """An accessor's values, read from the buffer by its view."""
    held = document["accessors"][index]
    view = document["bufferViews"][held["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[held["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[held["type"]]
    values = np.frombuffer(binary, dtype=dtype, count=held["count"] * width, offset=view["byteOffset"])
    return values.reshape(held["count"], width) if width > 1 else values


def node_by_usd(document: Mapping[str, Any], path: str) -> tuple[int, dict[str, Any]]:
    for index, node in enumerate(document["nodes"]):
        if node.get("extras", {}).get("usd") == path:
            return index, node
    raise KeyError(path)
