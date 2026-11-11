"""G1, the projection: ``mime.sensing.project`` against OpenCV's ``projectPoints`` on drawn cameras (Brown-Conrady's
five coefficients), poses and points, in both dtypes. The goldens were written once by ``make_projection_goldens.py``
(OpenCV 4.10) and are pinned by their sha256.

The tolerance is 64 units in the last place of the pixel's scale (``fx`` plus its coordinate): the formulas are the
same, so the differences are rounding. **The control** (``CAMERA_PLANT=k1-sign``) flips the sign of ``k1``; it fails
every distorted case, by pixels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import planted

from mime.sensing import BrownConrady, CameraPose, Pinhole, project

GOLDENS = Path(__file__).with_name("projection_goldens.json")
SHA256 = "790dcab72e5b98efde651cf0fdfa730103f3a8e3edad1396ca2b29c8fcd53468"
ULPS = 64.0


def cases() -> list[dict[str, object]]:
    return json.loads(GOLDENS.read_text(encoding="utf-8"))["cases"]


def test_the_goldens_are_the_pinned_ones():
    assert hashlib.sha256(GOLDENS.read_bytes()).hexdigest() == SHA256


def worst_error(case: dict[str, object], dtype: str) -> tuple[float, float]:
    """The largest pixel error against OpenCV's, and the tolerance it is judged by."""
    k1, k2, p1, p2, k3 = (float(c) for c in case["distortion"])
    if planted("k1-sign"):
        k1 = -k1
    camera = Pinhole(int(case["width"]), int(case["height"]), case["fx"], case["fy"], case["cx"], case["cy"],
                     BrownConrady(k1, k2, p1, p2, k3))
    pose = CameraPose(jnp.asarray(case["position"]), jnp.asarray(case["orientation"]))
    pixels, depth = project(camera, pose, jnp.asarray(case["points"]))
    expected = np.asarray(case["pixels"])
    assert pixels.dtype == jnp.dtype(dtype) and bool(jnp.all(depth > 0))
    scale = float(case["fx"]) + np.abs(expected)
    eps = float(np.finfo(dtype).eps)
    return float(np.max(np.abs(np.asarray(pixels, dtype=np.float64) - expected) / scale)), ULPS * eps


@pytest.mark.parametrize("index", range(16))
def test_projection_agrees_with_opencv(dtype, index):
    error, tolerance = worst_error(cases()[index], dtype)
    assert error <= tolerance, f"case {index}: {error:.3g} of the pixel's scale, against {tolerance:.3g}"
