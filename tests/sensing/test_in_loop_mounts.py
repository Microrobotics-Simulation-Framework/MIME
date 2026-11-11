"""Mounts: a camera fixed in the world (``World``) and one on a frame of an apparatus's chain (``Mounted``, the lab
fixture's gantry, which a drive moves) each put a body's silhouette centroid where ``project`` puts the body's pose through
the mount's transform, within 0.1 px; the build reports whether the static structure's cache can hold (only where the
camera and the structure share the world's frame); and a camera that may move never reads a cache.

**The control** (``CAMERA_PLANT=stale-cache``) makes the build hold the cache under the moving mount: the frame then
shows the tube and the ball from where the camera started, and the centroid is off by many pixels, every run."""

from __future__ import annotations

import importlib

import numpy as np
from in_loop_cases import DETECTOR, LOOK_ACROSS, RADIUS_UM, ballistic, dark_centre, planted

from mime import Apparatus
from mime.drives import Ramp
from mime.observations import Mounted
from mime.sensing import CameraPose, project
from mime.sensing._node import CameraNode
from mime.units import m, ms, s

if planted("stale-cache"):
    CameraNode.holds_cache = lambda self: True  # type: ignore[method-assign]

BACKLIGHT = 1000.0


def moving_gantry(exp, rate: float = 0.004):
    """A gantry 1 mm from the tube's axis whose carriage a ramp moves along x at ``rate`` m/s."""
    gantry = importlib.import_module("mime_lab_fixture").Gantry
    exp.add_apparatus(Apparatus("stage", chain={"gantry": gantry(travel=(1.0, 0.0, 0.0), at=m((0.0, -1e-3, 0.0)))},
                                commands={"gantry.position": Ramp(m(0.0), to=m(1.0), rate=m(rate) / s(1.0))}))
    return exp


def centroid_error(state, camera_pose: CameraPose) -> float:
    camera = DETECTOR.pinhole()
    wanted = np.asarray(project(camera, camera_pose, np.asarray(state["ball"]["pose"])[:3])[0])
    u, v, _ = dark_centre(np.asarray(state["camera:"]["image"]), 0.0)
    # the darkness is taken against the open tube's counts: the tube is the same in both, the ball's shadow is the rest
    return float(np.hypot(u - wanted[0], v - wanted[1]))


def ball_only_centroid(image: np.ndarray) -> tuple[float, float]:
    """The centroid of the pixels darker than a fifth of the backlight: the ball's shadow (the tube's absorption is far
    lighter)."""
    dark = (np.asarray(image) < 0.2 * BACKLIGHT).astype(np.float64)
    v, u = np.indices(dark.shape)
    return float((dark * u).sum() / dark.sum()), float((dark * v).sum() / dark.sum())


def test_a_world_mount_puts_the_silhouette_where_project_does(dtype):
    built = ballistic(ms(1)).build()
    built.run(ms(3))
    state = built.current_state()
    pose = CameraPose(np.array([0.0, -1e-3, 0.0]), np.array(LOOK_ACROSS))
    wanted = np.asarray(project(DETECTOR.pinhole(), pose, np.asarray(state["ball"]["pose"])[:3])[0])
    u, v = ball_only_centroid(state["camera:"]["image"])
    assert np.hypot(u - wanted[0], v - wanted[1]) < 0.1, (u, v, wanted)
    assert "static cache: the camera and the static structure share the world's frame" in built.report.text()


def test_a_camera_on_a_moving_frame_follows_it_and_holds_no_cache(dtype):
    exp = moving_gantry(ballistic(ms(1), mount=Mounted(frame="stage.gantry.carriage", orientation=LOOK_ACROSS)))
    built = exp.build()
    assert "static cache: the camera is mounted on stage.gantry.carriage, taken to move" in built.report.text()
    built.run(ms(8))
    state = built.current_state()
    carriage = np.asarray(state["stage.gantry"]["frame_poses"])[1]
    assert carriage[0] > 1e-5, carriage  # the camera has moved by many pixels
    pose = CameraPose(carriage[:3], np.array(LOOK_ACROSS))  # the frame's axes are the world's: no turn
    wanted = np.asarray(project(DETECTOR.pinhole(), pose, np.asarray(state["ball"]["pose"])[:3])[0])
    u, v = ball_only_centroid(state["camera:"]["image"])
    assert np.hypot(u - wanted[0], v - wanted[1]) < 0.1, (u, v, wanted, carriage[:3])
