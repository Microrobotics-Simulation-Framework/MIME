"""G5, the response's noise: over 400 keys, the counts of a uniformly lit 16x16 frame have the mean
``gain x full_scale x T + offset`` and the variance ``gain^2 (full_scale T + read_noise^2) + 1/12`` (Gaussian shot and
read noise, then rounding to whole counts), each within four standard errors, in both dtypes; the same key gives the same
frame (the re-render contract), another key another; saturation is reported, and full scale displays white.

**The control** (``CAMERA_PLANT=gain``) judges the variance with a gain 10 % high: caught every run (the variance's
standard error is about 0.5 % at this sample size)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import SEED, planted

from mime.sensing import Response, for_display, respond

KEYS = 400
SHAPE = (16, 16)


@pytest.mark.parametrize("level", [0.2, 0.7])
def test_counts_have_the_responses_mean_and_variance(dtype, level):
    response = Response(full_scale=2000.0, gain=0.8, offset=12.0, read_noise=6.0, bits=12)
    transmission = jnp.full(SHAPE, level)
    # The keys follow the run's seed (DETECTION_SEED), so a control's rate is measured over fresh draws.
    frames = np.stack([np.asarray(respond(transmission, response, jax.random.key(SEED * KEYS + seed))[1],
                                  dtype=np.float64) for seed in range(KEYS)])
    electrons = 2000.0 * level
    gain = 0.8 * (1.1 if planted("gain") else 1.0)
    mean, variance = 0.8 * electrons + 12.0, gain ** 2 * (electrons + 36.0) + 1.0 / 12.0
    n = frames.size
    assert abs(frames.mean() - mean) <= 4.0 * np.sqrt(variance / n)
    assert abs(frames.var(ddof=1) - variance) <= 4.0 * variance * np.sqrt(2.0 / (n - 1))


def test_the_same_key_gives_the_same_frame_and_another_another(dtype):
    response = Response(full_scale=500.0, read_noise=2.0)
    transmission = jnp.full(SHAPE, 0.5)
    first = respond(transmission, response, jax.random.key(7))[1]
    assert bool(jnp.all(first == respond(transmission, response, jax.random.key(7))[1]))
    assert bool(jnp.any(first != respond(transmission, response, jax.random.key(8))[1]))
    expected, image, saturated = respond(transmission, response, None)
    assert bool(jnp.all(image == jnp.round(expected))) and float(saturated) == 0.0


def test_saturation_is_reported(dtype):
    response = Response(full_scale=5000.0, bits=12)
    _, image, saturated = respond(jnp.full(SHAPE, 1.0), response, None)
    assert float(saturated) == 1.0 and float(jnp.max(image)) == 4095.0
    assert float(jnp.max(for_display(image, response))) == 1.0
