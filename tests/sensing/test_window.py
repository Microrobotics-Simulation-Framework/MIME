"""Exact disc coverage's window: a sphere's silhouette's bounding box about its projected centre, each axis clipped to
the image on its own, so that nothing of the silhouette that lies in the image is lost; ``window_overflow`` raised
whenever some of it lies outside the window. Judged against the same exact coverage over the whole image (the disc's
own arithmetic, unwindowed), in both dtypes: the review's two cases (a 17 px sphere in a 48 x 36 image, 4.4 % lost; a
300 px sphere in 640 x 480, 20 % lost, both unflagged), a sphere at the image's corner, and a declared window too
small for its sphere (flagged).

**The control** (``CAMERA_PLANT=square-window``) cuts the window to the image's shorter side each way, as the review
found it: the 640 x 480 case loses about a fifth of its silhouette, every run."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import planted

import mime.sensing._coverage as coverage
import mime.sensing._render as render_module
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

DISTANCE = 0.03


def _square(window, camera):
    side = min(2 * max(window) + 1, camera.height, camera.width)
    return side, side


def _camera(width: int, height: int, focal: float) -> tuple[Pinhole, CameraPose]:
    s = math.sqrt(0.5)
    return (Pinhole(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2),
            CameraPose(jnp.asarray([0.0, -DISTANCE, 0.0]), jnp.asarray([s, -s, 0.0, 0.0])))


def windowed_and_whole(width: int, height: int, focal: float, centre: tuple[float, float, float], radius: float,
                       window: int | None, monkeypatch: pytest.MonkeyPatch) -> tuple[np.ndarray, np.ndarray, bool]:
    """The sphere's coverage through its window, the same exact coverage over the whole image, and the flag."""
    if planted("square-window"):
        monkeypatch.setattr(render_module, "_window_size", _square)
    camera, pose = _camera(width, height, focal)
    ball = Subject("ball", "sphere", {"radius": radius}, appearance=Appearance.opaque(), moving=True, window=window)
    frame = render([ball], camera, pose, {"ball": (jnp.asarray(centre), jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    dtype = frame.transmission.dtype
    corners_x, corners_y = render_module._corners(camera, None, (height, width))
    # The ball's centre in the optical frame as the render takes it, so that the arithmetic is the same.
    optical = render_module._rotation(pose.orientation, dtype).T @ (jnp.asarray(centre, dtype=dtype)
                                                                    - jnp.asarray(pose.position, dtype=dtype))
    whole = coverage.disc(corners_x, corners_y, optical, jnp.asarray(radius, dtype=dtype))
    return (1.0 - np.asarray(frame.transmission, dtype=np.float64), np.asarray(whole, dtype=np.float64),
            bool(frame.flags["window_overflow:ball"]))


CASES = {
    "17px-in-48x36": (48, 36, 1020.0, (0.0, 0.0, 0.0), 0.5e-3),
    "300px-in-640x480": (640, 480, 18000.0, (0.0, 0.0, 0.0), 0.5e-3),
    # Centred on the first pixel's centre: three quarters of the silhouette outside the image.
    "at-the-corner": (160, 120, 600.0, (-79.5 * DISTANCE / 600.0, 0.0, 59.5 * DISTANCE / 600.0), 0.5e-3),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_window_holds_the_silhouette(dtype, case, monkeypatch):
    windowed, whole, overflow = windowed_and_whole(*CASES[case], None, monkeypatch)
    assert whole.sum() > 50
    # The same arithmetic on the same corners, but for the corners' undistortion, an iteration bracketed by the
    # corners' own extent: a window's and the whole image's agree to its convergence, a few units in the last place of
    # a corner's coordinate, which moves an edge by that times the focal length: 8 eps |x| f at most.
    width, _, focal = CASES[case][:3]
    largest = (width / 2 + 0.5) / focal
    assert float(np.max(np.abs(windowed - whole))) <= 8 * float(np.finfo(dtype).eps) * largest * focal
    assert not overflow


def test_a_window_too_small_is_flagged(dtype, monkeypatch):
    """A declared window of half-size 5 on a 15 px sphere: the silhouette's rim is cut off, and the flag says so."""
    windowed, whole, overflow = windowed_and_whole(160, 120, 900.0, (0.0, 0.0, 0.0), 0.5e-3, 5, monkeypatch)
    assert whole.sum() - windowed.sum() > 100
    assert overflow
