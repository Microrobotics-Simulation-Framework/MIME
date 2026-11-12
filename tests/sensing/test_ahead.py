"""A body straddling the camera's plane, and a medium's chord through it. A line through the pinhole meets a ball that
contains the pinhole at two parameters, one behind it (``t < 0``) and one ahead; the ray is the part with ``t >= 0``, so
the chord a pixel's light crosses runs from the pinhole to the far side, ``t_far``, not the whole line's ``t_far - t_near``.
``forms.ahead`` is the clip, and no other test needs it: ``_on_ray`` already culls a body wholly behind the camera, and a
body ahead has both parameters positive, so only a body straddling the camera's plane, with the pinhole inside it (the
camera immersed in a medium), reads it.

The test: a tinted ball with the pinhole inside, off its centre, seen by the camera across script A's tube's scene, transmits
``exp(-mu t_far)`` at every pixel, with ``t_far`` written apart from ``mime.sensing`` (the quadratic's far root, in
extended precision), in both dtypes.

**The control** (``CAMERA_PLANT=ahead-unclipped``) makes ``ahead`` return the line's whole interval: the chord then runs
from ``t_near < 0`` and every pixel's transmission is too small by the factor ``exp(-mu |t_near|)``, about 0.35 here."""

from __future__ import annotations

import numpy as np
from camera_cases import across, planted

import mime.sensing._forms as forms
from mime.sensing import Appearance, Subject, render

MU = 20.0
RADIUS = 0.05
#: The ball's centre relative to the pinhole (m): the pinhole is 4 cm inside the ball's surface at its nearest.
OFFSET = np.array([0.004, -0.003, 0.002])


def far_root(width: int) -> np.ndarray:
    """``t_far`` of each pixel's ray from the pinhole, through a pinhole camera without distortion written afresh (as
    ``test_media.oracle_rays``): the camera at the origin of its own frame, looking along +y, the image's x along +x
    and its y along -z; the ball's centre at ``OFFSET`` from the pinhole."""
    height = width * 3 // 4
    focal = 0.1 / (13.3e-3 / width)
    v, u = np.mgrid[0:height, 0:width].astype(np.longdouble)
    x, y = (u - (width - 1) / 2) / focal, (v - (height - 1) / 2) / focal
    d = np.stack([x, np.ones_like(x), -y], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    w = -OFFSET.astype(np.longdouble)  # the pinhole relative to the centre
    b = d @ w
    c = float(w @ w) - RADIUS ** 2
    assert c < 0, "the pinhole is inside the ball"
    return -b + np.sqrt(b * b - c)


def test_the_chord_of_a_medium_the_camera_is_in_starts_at_the_pinhole(dtype, monkeypatch):
    if planted("ahead-unclipped"):
        monkeypatch.setattr(forms, "ahead", lambda interval: interval)
    width = 80
    camera, pose = across(width)
    centre = np.asarray(pose.position) + OFFSET
    fog = Subject("fog", "sphere", {"radius": RADIUS, "centre": [float(c) for c in centre]},
                  appearance=Appearance.attenuating(MU))
    found = np.asarray(render([fog], camera, pose).transmission, dtype=np.float64)
    far = far_root(width)
    expected = np.exp(-MU * far).astype(np.float64)
    # Judged on the scale it came from: the exponent mu t_far is rounded to a few eps of itself, so the relative error is
    # that many eps times (1 + mu t_far); the line takes 64 of them.
    line = 64 * float(np.finfo(dtype).eps) * (1.0 + MU * float(np.max(far)))
    assert float(np.max(np.abs(found / expected - 1.0))) <= line
