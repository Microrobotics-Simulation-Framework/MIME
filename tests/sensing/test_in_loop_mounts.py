"""Mounts: a camera fixed in the world (``World``) and one on a frame of an apparatus's chain (``Mounted``, the lab
fixture's gantry, which a drive moves) each put a body's silhouette centroid where ``project`` puts the body's pose through
the mount's transform, within 0.1 px; the build reports whether the static structure's cache can hold (only where the
camera and the structure share the world's frame); and a camera that may move never reads a cache.

**The control** (``CAMERA_PLANT=stale-cache``) makes the build hold the cache under the moving mount: the frame then
shows the tube from where the camera started. The carriage moves across the tube's axis (a camera moving along the axis
would see the same tube: the first version of this gate moved it along x and its control passed 30 of 30), so the tube's
silhouette is off by the carriage's travel, several pixels, every run. The ball is a moving subject and is drawn at the
camera's present pose whatever the cache holds, so the ball's centroid alone cannot tell a stale cache."""

from __future__ import annotations

import importlib

import numpy as np
from in_loop_cases import DETECTOR, LOOK_ACROSS, RADIUS_UM, ballistic, dark_centre, planted

from mime import Apparatus
from mime.drives import Ramp
from mime.observations import Mounted
from mime.sensing import CameraPose, Exposure, Response, frame_key, project, render, static_cache
from mime.sensing._node import CameraNode
from mime.units import m, ms, s

if planted("stale-cache"):
    CameraNode.holds_cache = lambda self: True  # type: ignore[method-assign]

BACKLIGHT = 1000.0


def moving_gantry(exp, rate: float = 0.004, travel: tuple[float, float, float] = (1.0, 0.0, 0.0)):
    """A gantry 1 mm from the tube's axis whose carriage a ramp moves along ``travel`` (x, along the tube, by default) at
    ``rate`` m/s."""
    gantry = importlib.import_module("mime_lab_fixture").Gantry
    exp.add_apparatus(Apparatus("stage", chain={"gantry": gantry(travel=travel, at=m((0.0, -1e-3, 0.0)))},
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


def offline_expected(built, state, camera_pose: CameraPose) -> np.ndarray:
    """The counts before noise of the offline render of the scene with the camera at ``camera_pose``, its static structure
    cached at that pose (a cache made where the camera is now)."""
    node = built.graph.get_node("camera:")
    camera, held = node._held  # noqa: SLF001 - the detector and the subjects the build made
    ball = np.asarray(state["ball"]["pose"])
    rows = (ball[None, :3], ball[None, 3:])
    replay = render(list(held), camera, camera_pose, {"ball": rows}, exposure=Exposure(1), response=Response(),
                    key=frame_key(node.constants()["seed"], 0), cache=static_cache(list(held), camera, camera_pose))
    return np.asarray(replay.expected, dtype=np.float64)


def test_a_camera_on_a_moving_frame_follows_it_and_holds_no_cache(dtype):
    exp = moving_gantry(ballistic(ms(1), mount=Mounted(frame="stage.gantry.carriage", orientation=LOOK_ACROSS)),
                        rate=0.0006, travel=(0.0, 0.0, 1.0))
    built = exp.build()
    assert "static cache: the camera is mounted on stage.gantry.carriage, taken to move" in built.report.text()
    built.run(ms(8))
    state = built.current_state()
    carriage = np.asarray(state["stage.gantry"]["frame_poses"])[1]
    assert carriage[2] > 4e-6, carriage  # the camera has moved across the tube by several pixels
    pose = CameraPose(carriage[:3], np.array(LOOK_ACROSS))  # the frame's axes are the world's: no turn
    wanted = np.asarray(project(DETECTOR.pinhole(), pose, np.asarray(state["ball"]["pose"])[:3])[0])
    u, v = ball_only_centroid(state["camera:"]["image"])
    assert np.hypot(u - wanted[0], v - wanted[1]) < 0.1, (u, v, wanted, carriage[:3])
    # the tube too: its image is the offline render's from where the camera is now, not from where it started
    gap = np.abs(offline_expected(built, state, pose) - np.asarray(state["camera:"]["expected"], dtype=np.float64))
    assert gap.max() < 1.0, (gap.max(), carriage[:3])
