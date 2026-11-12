"""The calibration scene: a 3 x 3 grid of spheres on a plane seen by a tilted camera, rendered live by the in-loop camera,
whose detector numbers are the fit's axes."""

from __future__ import annotations

import math

import numpy as np

from mime import Body, Experiment
from mime import State
from mime.dynamics import RigidInertial
from mime.observations import Camera, Detector, World
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s

WIDTH, HEIGHT = 48, 36
GRID = 2
#: The truth the fit is to recover: focal length, principal point and radial distortion (pixels, pixels, 1, 1).
TRUTH = {"fx": 46.5, "fy": 45.75, "cx": 24.3, "cy": 17.3, "k1": -0.12, "k2": 0.03}
RADIUS, SPACING, DISTANCE, TILT = 7e-3, 20e-3, 70e-3, math.radians(25.0)
PATHS = {name: f"camera:.detector.{name}" for name in ("fx", "fy", "cx", "cy", "k1", "k2")}


def quaternion_of(rotation: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(1.0 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2], 0.0)) / 2.0
    if w > 1e-6:
        return (w, (rotation[2, 1] - rotation[1, 2]) / (4 * w), (rotation[0, 2] - rotation[2, 0]) / (4 * w),
                (rotation[1, 0] - rotation[0, 1]) / (4 * w))
    raise ValueError("a rotation near a half turn")


def pose() -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """The camera 70 mm from the grid's centre, its optical axis 25 degrees off the plane's normal, the image's right
    along +x."""
    axis = np.array([0.0, math.sin(TILT), -math.cos(TILT)])  # looks down at the plane z = 0
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(axis, right)
    rotation = np.stack([right, down, axis], axis=1)  # optical axes in the world, as columns
    position = -DISTANCE * axis
    return (float(position[0]), float(position[1]), float(position[2])), quaternion_of(rotation)


def grid(detector: Detector) -> Experiment:
    exp = Experiment("sphere_grid", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    for i in range(GRID):
        for j in range(GRID):
            name = f"s{i}{j}"
            exp.add_body(Body(name, dynamics=RigidInertial(), initial=State(
                position=m((SPACING * (i - (GRID - 1) / 2), SPACING * (j - (GRID - 1) / 2), 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0))),
                parts={"steel": Part(Sphere(radius=m(RADIUS)))}))
            exp.declare(f"{name}.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    position, orientation = pose()
    exp.observe(Camera(detector=detector, mount=World(position=m(position), orientation=orientation),
                       frame_period=ms(1)))
    return exp


def detector(**changes: float) -> Detector:
    held = {**TRUTH, **changes}
    return Detector(width=WIDTH, height=HEIGHT, fx=held["fx"], fy=held["fy"], cx=held["cx"], cy=held["cy"],
                    k1=held["k1"], k2=held["k2"])
