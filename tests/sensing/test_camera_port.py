"""A measurement port on a camera's frames (``Measurement(of="camera:.expected")``) and the camera's projection
(``Camera.project``), what a controller's model of its camera computes, against OpenCV's documented ``projectPoints``
written in NumPy. A port on the camera's image may add no noise (its sensor model already does: R3,
``port.modelled-noise``); one on its expected image may, and the build says the sensor model is bypassed; a port's
rate is a whole multiple of the camera's frame period, which decimates it; its age counts from the step the frame was
made at.

**The controls** (``PORT_PLANT``, rated by ``.github/probe/detection_rate.py``): the projection without its lens
distortion (``project_undistorted``), with its focal lengths swapped (``project_swapped_focal``) and turned by the
inverse rotation (``project_inverse_rotation``): each must fail the projection's test on every run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from in_loop_cases import ballistic
from port_cases import planted

from mime import Apparatus, Measurement
from mime.build import BuildRefused
from mime.core import run_dtype
from mime.things import Gaussian
from mime.units import DIMENSIONLESS, Unit, ms

COUNTS = Unit("1", 1.0, DIMENSIONLESS, "counts")


def camera_with(port: Measurement):
    exp = ballistic(ms(3))
    exp.add_apparatus(Apparatus("imaging", ports={"frames": port}))
    return exp


def test_a_camera_port_decimates_at_a_whole_multiple_and_counts_age_from_the_frame():
    port = Measurement(of="camera:.expected", rate=ms(6), latency=ms(2), noise=Gaussian(sigma=COUNTS(3.0)),
                       reason="the capture card's transfer, measured")
    built = camera_with(port).build()
    (line,) = built.report.ports
    assert "every 6 steps" in line and "which this read bypasses" in line and "sensor response" in line
    history = built.run(ms(14), history=True).history
    camera = history["camera:"]
    ages = np.asarray(history["measure:imaging"]["frames_age"])
    steps = np.arange(14)
    made = np.where(steps >= 2, 6 * ((steps - 2) // 6), 0)
    assert list(ages) == list((steps - made).astype(float))
    frames = np.asarray(history["measure:imaging"]["frames"])
    expected = np.asarray(camera["expected"])
    assert float(np.abs(frames[8] - expected[6]).mean()) < 4 * 3.0


def test_noise_on_a_cameras_image_is_refused_and_on_its_expected_image_builds():
    with pytest.raises(BuildRefused) as refused:
        camera_with(Measurement(of="camera:.image", noise=Gaussian(sigma=COUNTS(3.0)), reason="r")).build()
    assert [r.code for r in refused.value.refusals] == ["port.modelled-noise"]
    camera_with(Measurement(of="camera:.expected", noise=Gaussian(sigma=COUNTS(3.0)), reason="r")).build()


def test_a_camera_port_off_its_frame_period_is_refused():
    with pytest.raises(BuildRefused) as refused:
        camera_with(Measurement(of="camera:.expected", rate=ms(4), reason="r")).build()
    assert [r.code for r in refused.value.refusals] == ["port.clock"] and "whole multiple" in str(refused.value)


def _rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def opencv_project(points: np.ndarray, intrinsics: dict, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """OpenCV's ``projectPoints`` (its documentation's pinhole and Brown-Conrady equations), in NumPy: a point into
    the camera's frame by the inverse of the camera's turn, divided by its depth, distorted, then scaled to pixels."""
    local = (points - position) @ _rotation(orientation)
    x, y = local[:, 0] / local[:, 2], local[:, 1] / local[:, 2]
    k1, k2, p1, p2, k3 = (intrinsics[k] for k in ("k1", "k2", "p1", "p2", "k3"))
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([intrinsics["fx"] * xd + intrinsics["cx"], intrinsics["fy"] * yd + intrinsics["cy"]], axis=-1)


@contextmanager
def planted_projection(camera) -> Iterator[None]:
    import mime.observations._camera as module

    held = module.projected
    try:
        if planted("project_undistorted"):
            def undistorted(pinhole, pose, points):
                from dataclasses import replace

                from mime.sensing import BrownConrady

                return held(replace(pinhole, distortion=BrownConrady()), pose, points)

            module.projected = undistorted
        if planted("project_swapped_focal"):
            def swapped(pinhole, pose, points):
                from dataclasses import replace

                return held(replace(pinhole, fx=pinhole.fy, fy=pinhole.fx), pose, points)

            module.projected = swapped
        if planted("project_inverse_rotation"):
            def inverse(pinhole, pose, points):
                from mime.sensing._detector import CameraPose

                w, x, y, z = (float(c) for c in pose.orientation)
                return held(pinhole, CameraPose(pose.position, (w, -x, -y, -z)), points)

            module.projected = inverse
        yield
    finally:
        module.projected = held


@settings(deadline=None)
@given(st.floats(400.0, 900.0), st.floats(1.05, 1.3), st.floats(0.05, 0.2), st.floats(-0.05, 0.05),
       st.floats(-2e-3, 2e-3), st.integers(0, 2**31 - 1))
def test_the_cameras_projection_is_opencvs_project_points(fx, aspect, k1, k2, p1, seed):
    """Drawn over a detector's numbers (its focal lengths 5 % to 30 % apart, a barrel or pincushion lens) and a turned,
    placed camera with points 5 to 20 cm before it, out to its field's corners; within 1e-3 px of OpenCV's equations
    in float64 (2e-2 px in float32, on a detector of a few hundred pixels: 64 units in the last place of its size)."""
    from mime.observations import Camera, Detector, World
    from mime.units import m

    rng = np.random.default_rng(seed)
    intrinsics = {"fx": fx, "fy": fx * aspect, "cx": 80.0 + rng.uniform(-3, 3), "cy": 60.0 + rng.uniform(-3, 3),
                  "k1": k1, "k2": k2, "p1": p1, "p2": -p1 / 2, "k3": 0.0}
    q = rng.normal(size=4)
    orientation = q / np.linalg.norm(q)
    position = rng.uniform(-0.05, 0.05, size=3)
    camera = Camera(detector=Detector(width=160, height=120, **intrinsics),
                    mount=World(position=m(tuple(position)), orientation=tuple(float(c) for c in orientation)),
                    frame_period=ms(3))
    depth = rng.uniform(0.05, 0.2, size=32)
    pixels = rng.uniform((0.0, 0.0), (160.0, 120.0), size=(32, 2))
    normalised = np.stack([(pixels[:, 0] - intrinsics["cx"]) / intrinsics["fx"],
                           (pixels[:, 1] - intrinsics["cy"]) / intrinsics["fy"], np.ones(32)], axis=-1)
    points = position + (normalised * depth[:, None]) @ _rotation(orientation).T
    with planted_projection(camera):
        found, depths = camera.project(points)
    expected = opencv_project(points, intrinsics, position, orientation)
    limit = 1e-3 if np.dtype(run_dtype()) == np.float64 else 2e-2
    assert float(np.max(np.abs(np.asarray(found, dtype=np.float64) - expected))) <= limit
    np.testing.assert_allclose(np.asarray(depths, dtype=np.float64), depth, rtol=1e-5)


def test_a_mounted_camera_projects_only_from_a_pose_given():
    from mime.observations import Camera, Detector, Mounted

    camera = Camera(detector=Detector(width=16, height=12, fx=20.0, fy=20.0, cx=7.5, cy=5.5),
                    mount=Mounted(frame="rpm.arm.link_6"), frame_period=ms(3))
    with pytest.raises(TypeError, match="mounted on a frame moves"):
        camera.project(np.zeros((1, 3)))
    pixels, depth = camera.project(np.array([[0.0, 0.0, 0.1]]), pose=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    np.testing.assert_allclose(np.asarray(pixels), [[7.5, 5.5]], atol=1e-5)
