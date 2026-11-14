"""Measurement ports: the measure stage (``mime.nodes.control.MeasureStageNode``) against a NumPy oracle of the design's
order (sample, delay, noise, quantum) with its age, and its seeded noise. A port on a camera's frames and the camera's
projection are in ``tests/sensing/test_camera_port.py``.

**The controls** (``PORT_PLANT``, rated by ``.github/probe/detection_rate.py``): the stage's clock one step short of its
latency (``latency_short``, in the clock both stages share), and the age counted from the delivery rather than the
sample (``stale_age``): each must fail the first test on every run."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from port_cases import planted, sampled_and_delayed

import mime.nodes.control._ports as stages
from mime.core import run_dtype
from mime.nodes.control import MeasureStageNode

STEPS = 20
DT = 1e-3


def measure_stage(period: int, latency: int, *, shape=(4,), sigma=None, quantum=None, start=0.0, start_age=1,
                  seed=11, name="probe") -> MeasureStageNode:
    leaves = {f"initial_{name}": start, f"initial_age_{name}": start_age}
    if sigma is not None:
        leaves[f"sigma_{name}"] = sigma
    if quantum is not None:
        leaves[f"quantum_{name}"] = quantum
    return MeasureStageNode("measure:probe", DT, ports=[name], units={name: "m"}, shapes={name: shape},
                            periods={name: period}, latencies={name: latency}, aged={name: False}, seeds={name: seed},
                            noisy={name: sigma is not None}, quantised={name: quantum is not None}, **leaves)


def measured(node: MeasureStageNode, values: np.ndarray, name: str = "probe") -> tuple[np.ndarray, np.ndarray]:
    state = node.initial_state()
    out, ages = [], []
    for value in values:
        state = node.fresh_update(state, {name: jnp.asarray(value)}, DT)
        out.append(np.asarray(state[name], dtype=np.float64))
        ages.append(float(state[f"{name}_age"]))
    return np.stack(out), np.asarray(ages)


@contextmanager
def planted_clock() -> Iterator[None]:
    """The stages' shared clock as the plant leaves it: one step short of its latency, or its age counted from the
    delivery; as it is with nothing planted."""
    held = stages.held_sample
    advance = MeasureStageNode.advance
    try:
        if planted("latency_short"):
            def short(buffer, value, count, period, latency):
                return held(buffer, value, count + (1 if latency else 0), period, latency)

            stages.held_sample = short
        if planted("stale_age"):
            def stale(self, p, state, inputs, dt):
                out = advance(self, p, state, inputs, dt)
                return {**out, **{f"{k}_age": jnp.zeros_like(out[f"{k}_age"]) for k in self._keys()}}

            MeasureStageNode.advance = stale
        yield
    finally:
        stages.held_sample, MeasureStageNode.advance = held, advance


@settings(deadline=None)
@given(st.integers(1, 5), st.integers(0, 7), st.integers(0, 2**31 - 1))
def test_the_measure_stage_samples_and_delays_with_the_age_of_what_it_delivers(period, latency, seed):
    values = np.random.default_rng(seed).uniform(-1e-3, 1e-3, size=(STEPS, 4))
    start = 2.5e-4
    with planted_clock():
        out, ages = measured(measure_stage(period, latency, start=start), values)
    np.testing.assert_array_equal(out, sampled_and_delayed(values, period, latency, np.full(4, start))
                                  .astype(np.dtype(run_dtype())).astype(np.float64))
    expected_ages = [n - (period * ((n - latency) // period)) if n >= latency else n + 1 for n in range(STEPS)]
    assert list(ages) == expected_ages


def test_a_quantum_rounds_after_the_delay_and_noise_is_held_with_its_sample():
    values = np.linspace(0.0, 1e-3, STEPS)[:, None] * np.ones((1, 4))
    out, _ = measured(measure_stage(1, 2, quantum=1e-4), values)
    np.testing.assert_allclose(out[2:], np.round(values[:-2] / 1e-4) * 1e-4, rtol=0, atol=1e-9)
    noisy, _ = measured(measure_stage(3, 0, sigma=1e-5), values)
    clean, _ = measured(measure_stage(3, 0), values)
    drawn = noisy - clean
    for start in range(0, STEPS - 2, 3):
        np.testing.assert_array_equal(drawn[start], drawn[start + 1])
        np.testing.assert_array_equal(drawn[start], drawn[start + 2])
    assert not np.array_equal(drawn[0], drawn[3])


def test_the_noise_is_seeded_reproducible_and_of_its_declared_spread():
    """Two stages with one seed draw the same noise; another seed, other noise; over 20 steps of a 50 by 50 source
    (50 000 draws) the noise's mean is within 4 standard errors of zero and its spread within 4 of sigma."""
    sigma, shape = 2.0, (50, 50)
    values = np.zeros((STEPS, *shape))
    first, _ = measured(measure_stage(1, 0, shape=shape, sigma=sigma), values)
    again, _ = measured(measure_stage(1, 0, shape=shape, sigma=sigma), values)
    other, _ = measured(measure_stage(1, 0, shape=shape, sigma=sigma, seed=12), values)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)
    count = first.size
    assert abs(float(first.mean())) <= 4 * sigma / math.sqrt(count)
    assert abs(float(first.std()) - sigma) <= 4 * sigma / math.sqrt(2 * count)
