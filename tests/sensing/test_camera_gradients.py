"""G9, gradients on aggregates: the total darkness and the darkness centroid, as autodiff derivatives in the sphere's
position, the focal length, the principal point's u and the first radial coefficient, against central differences of
the same render, through a distorted lens, in both dtypes:

- **the sphere's darkening** (the frame without it less the frame with it), Landers' sphere in mid-lumen of script A's
  translucent tube seen across it;
- **an opaque tube's darkness**: the cylinders' box edges in the intrinsics.

Per-pixel derivatives are G15's (the box edge's own differences disagree per pixel by 3 to 15 %, where an edge crosses
a pixel's corner); aggregates integrate those jumps away. A translucent silhouette is not an aggregate's edge here: its
chord rises as a square root, sampled at the pixel's centre and smoothed by the grazing softplus over a few
thousandths of the radius (G6), so central differences there need steps under that width, which float32 cannot take.

The steps move the image by about 1e-4 px in float64 and 1e-2 px in float32. **The control**
(``CAMERA_PLANT=held-rays``) holds the ray table out of the derivative: the intrinsics' derivatives then miss the
rays' own motion, every run."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import TUBE, planted, tube_subjects

import mime.sensing._render as render_module
from mime.sensing import Appearance, BrownConrady, CameraPose, Pinhole, Subject, render

WIDTH, HEIGHT = 160, 120
FOCAL = 0.1 / (13.3e-3 / WIDTH)
START = np.array([0.4e-3, 0.0, -0.9e-3])
THETA = np.concatenate([START, [FOCAL, (WIDTH - 1) / 2, -0.05]])
FOOTPRINT = 0.1 / FOCAL
#: Each parameter's scale: a pixel's footprint at the tube, a pixel's worth of focal length at the image's edge, a pixel,
#: and the first radial coefficient's effect at the image's edge (about two pixels a unit).
SCALES = np.array([FOOTPRINT, FOOTPRINT, FOOTPRINT, FOCAL / WIDTH, 1.0, 0.5])


def _camera(theta: jax.Array) -> tuple[Pinhole, CameraPose]:
    s = np.sqrt(0.5)
    return (Pinhole(WIDTH, HEIGHT, theta[3], theta[3], theta[4], (HEIGHT - 1) / 2, BrownConrady(k1=theta[5], k2=0.01)),
            CameraPose(jnp.asarray([0.0, -0.1, 0.0]), jnp.asarray([s, -s, 0.0, 0.0])))


def _moments(dark: jax.Array) -> jax.Array:
    v, u = jnp.mgrid[0:HEIGHT, 0:WIDTH]
    total = jnp.sum(dark)
    return jnp.stack([total, jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])


@jax.jit
def darkening(theta: jax.Array) -> jax.Array:
    camera, pose = _camera(theta)
    held = tube_subjects(Appearance.attenuating(200.0), Appearance.attenuating(20.0))
    without = render(held, camera, pose).transmission
    # The sphere's exact disc coverage, its default, in a declared window (a derivative cannot size one).
    held.append(Subject("umr", "sphere", {"radius": 0.7e-3}, appearance=Appearance.opaque(), parent="water",
                        moving=True, window=16))
    within = render(held, camera, pose, {"umr": (theta[:3], jnp.asarray((1.0, 0.0, 0.0, 0.0)))}).transmission
    return _moments(without - within)


@jax.jit
def opaque_tube(theta: jax.Array) -> jax.Array:
    camera, pose = _camera(theta)
    held = [Subject("tube", "tube", TUBE, "wall", Appearance.opaque())]
    return _moments(1.0 - render(held, camera, pose).transmission)


@pytest.mark.parametrize("case", ["darkening", "opaque_tube"])
def test_aggregate_derivatives_agree_with_central_differences(dtype, case, monkeypatch):
    if planted("held-rays"):
        original = render_module.ray_table
        monkeypatch.setattr(render_module, "ray_table",
                            lambda camera, pose: jax.lax.stop_gradient(original(camera, pose)))
        darkening.clear_cache()
        opaque_tube.clear_cache()
    function = {"darkening": darkening, "opaque_tube": opaque_tube}[case]
    parameters = range(6) if case == "darkening" else range(3, 6)
    fine = dtype == "float64"
    steps = SCALES * (1e-4 if fine else 1e-2)
    theta = jnp.asarray(THETA)
    jacobian = np.asarray(jax.jacfwd(function)(theta), dtype=np.float64)
    size = np.max(np.abs(jacobian[:, list(parameters)] * SCALES[None, list(parameters)]), axis=1)
    worst = 0.0
    for k in parameters:
        step = jnp.zeros(6).at[k].set(steps[k])
        difference = (np.asarray(function(theta + step), dtype=np.float64)
                      - np.asarray(function(theta - step), dtype=np.float64)) / (2 * steps[k])
        # Each aggregate judged against the size of its own derivatives over the parameters (per parameter scale), so
        # that a component near zero is not judged against itself.
        worst = max(worst, float(np.max(np.abs(jacobian[:, k] - difference) * SCALES[k] / size)))
    if planted("held-rays"):
        darkening.clear_cache()
        opaque_tube.clear_cache()
    assert worst <= (1e-4 if fine else 3e-2), worst
