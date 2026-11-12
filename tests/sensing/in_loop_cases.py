"""Shared pieces of the in-loop camera's tests: the planted errors their controls switch on (``CAMERA_PLANT``, read by
``.github/probe/detection_rate.py --plant-var CAMERA_PLANT``), a ballistic sphere beside a tube seen by a camera in the
world (the done case's scene without its drag), and a bare camera node for the clock."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from mime import Body, Experiment, State, Structure
from mime.dynamics import RigidInertial
from mime.observations import Camera, Detector, Mounted, World
from mime.scene import Scene, Sphere, Tube, X, Z
from mime.sensing import _scene as scene
from mime.sensing._node import CameraNode
from mime.things import Part
from mime.units import Quantity, kg_per_m3, m, m_per_s, ms, rad_per_s, um

#: The planted error a control switches on, by name; empty when nothing is planted.
PLANT = os.environ.get("CAMERA_PLANT", "")
#: The seed of a control's draw (``DETECTION_SEED``, which detection_rate.py sets for each run).
SEED = int(os.environ.get("DETECTION_SEED", "0") or 0)

SQRT_HALF = math.sqrt(0.5)
#: A camera 1 mm from the tube's axis on -y, looking across it (+y), the image's right along the tube (+x).
LOOK_ACROSS = (SQRT_HALF, -SQRT_HALF, 0.0, 0.0)
DETECTOR = Detector(width=160, height=120, fx=750.0, fy=750.0, cx=79.5, cy=59.5)
RADIUS_UM = 20.0
#: The balls' places along the tube (um) and the speed they move at (m/s): 2 um a step at a 1 ms base step, 1.5 px.
PLACES_UM = (-50.0, 50.0)
SPEED = 2e-3


def planted(name: str) -> bool:
    return PLANT == name


def scene_without_camera(balls: int = 1) -> Experiment:
    """``balls`` steel balls moving along a tube at 2 mm/s: the done case's scene without its drag, and no camera."""
    exp = Experiment("ballistic", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10), length=um(480))))
    for index in range(balls):
        name = "ball" if balls == 1 else f"ball{index}"
        exp.add_body(Body(name, dynamics=RigidInertial(), initial=State(
            position=um((PLACES_UM[index], 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
            velocity=m_per_s((SPEED, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0))),
            parts={"steel": Part(Sphere(radius=um(RADIUS_UM)))}))
        exp.declare(f"{name}.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    return exp


def ballistic(period: Quantity = ms(3), *, sub_frames: int = 1, balls: int = 1, mount: Any = None,
              detector: Detector = DETECTOR, hide: tuple[str, ...] = (), absorbers: tuple[tuple[str, float], ...] = (
                  ("tube", 3000.0),), **camera: Any) -> Experiment:
    """``balls`` steel balls moving along a tube at 2 mm/s, seen by a camera that sees the whole scene."""
    exp = scene_without_camera(balls)
    exp.observe(Camera(detector=detector, mount=mount or World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS),
                       frame_period=period, sub_frames=sub_frames, hide=hide, absorbers=absorbers, **camera))
    return exp


def dark_centre(image: np.ndarray, backlight: float) -> tuple[float, float, float]:
    """The centroid ``(u, v)`` of the darkness ``backlight - image`` over the frame, and its total."""
    dark = np.clip(backlight - np.asarray(image, dtype=np.float64), 0.0, None)
    total = float(dark.sum())
    v, u = np.indices(dark.shape)
    return float((dark * u).sum() / total), float((dark * v).sum() / total), total


def spec_of(width: int = 16, height: int = 12) -> dict[str, Any]:
    """A bare camera's JSON: one moving sphere ``ball`` of 1 mm radius, 0.1 m from the camera, no tube."""
    from mime.sensing import Subject

    ball = Subject("ball", "sphere", {"radius": 1e-3}, moving=True)
    return {"width": width, "height": height, "source": "declared", "subjects": [scene.subject_spec(ball, 8)],
            "body_subjects": {"ball": ["ball"]},
            "response": {"full_scale": 1000.0, "gain": 1.0, "offset": 0.0, "read_noise": 2.0, "shot_noise": True,
                         "bits": 12, "psf": 0.0}}


def bare_node(period: int, sub_frames: int = 1, seed: int = 7) -> CameraNode:
    """A camera node on its own: the sphere ``ball`` in the world, the camera at the origin looking along +z."""
    return CameraNode("camera", 1e-3, spec=scene.dumps(spec_of()), fx=20.0, fy=20.0, cx=7.5, cy=5.5, k1=0.0, k2=0.0,
                      p1=0.0, p2=0.0, k3=0.0, bodies=["ball"],
                      initial_poses=(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0), period=period, sub_frames=sub_frames,
                      seed=seed, placement=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))


def step(node: CameraNode, state: dict[str, Any], pose: Any) -> dict[str, Any]:
    """One step of ``node`` with the ball at ``pose``."""
    return node.update(state, {"pose_ball": jnp.asarray(pose), "mount_poses": jnp.zeros((1, 7)),
                               "mount_rates": jnp.zeros((1, 3))}, 1e-3)


def plant_node(patches: dict[str, Callable[..., Any]]) -> None:
    """Replace methods of the camera node (a control's planted error in the target)."""
    for name, function in patches.items():
        setattr(CameraNode, name, function)


def cost_case(width: int = 320, camera: bool = True, period_steps: int = 1, moving_mount: bool = False,
              frames_every: int = 0) -> Experiment:
    """The scene at ``width`` x ``3 width / 4`` pixels for the cost probe: no camera (the baseline), a camera that renders
    every ``period_steps`` base steps, or one on a gantry's carriage (``moving_mount``), which renders the tube in every
    frame. ``frames_every`` is unused and kept at zero."""
    del frames_every
    if not camera:
        return scene_without_camera()
    scale = width / DETECTOR.width
    detector = Detector(width=width, height=width * 3 // 4, fx=DETECTOR.fx * scale, fy=DETECTOR.fy * scale,
                        cx=(width - 1) / 2, cy=(width * 3 // 4 - 1) / 2)
    exp = ballistic(ms(period_steps), detector=detector,
                    mount=Mounted(frame="stage.gantry.carriage", orientation=LOOK_ACROSS) if moving_mount else None)
    if moving_mount:
        from test_in_loop_mounts import moving_gantry

        moving_gantry(exp)
    return exp
