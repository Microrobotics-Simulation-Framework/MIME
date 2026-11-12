"""G4, the in-loop camera's clock: at the step with count n (from zero), ``new_frame`` is one exactly when n is a multiple
of the frame period, ``frame_step`` is the last such n (``-1`` before the first step), and ``age`` is ``n - frame_step``;
the held image changes only on a frame step, and the exposure's ring holds the last ``sub_frames`` poses, the newest the
pose read at the step. Exact, over drawn periods (1 to 4 steps), sub-frames (1 or 2, at most the period) and numbers of
steps, in both dtypes.

**The control** (``CAMERA_PLANT=latency``) makes the frame condition ``(n + 1) mod P``, one step late: caught every run
(the draws always include a period above one)."""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from in_loop_cases import bare_node, planted, plant_node, step

from mime.sensing._node import CameraNode

if planted("latency"):
    plant_node({"frame_due": lambda self, count, period: ((count + 1) % period) == 0})


@functools.cache
def node_for(period: int, sub_frames: int, wide: bool) -> CameraNode:
    del wide  # the dtype the node is made in: a node made in one dtype is not reused in the other
    return bare_node(period, sub_frames)


def pose_at(n: int) -> np.ndarray:
    return np.array([0.001 * n, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(1, 4), st.integers(1, 2), st.integers(1, 40))
def test_the_clock_is_the_counts(dtype, period, sub_frames, steps):
    sub_frames = min(sub_frames, period)
    node = node_for(period, sub_frames, dtype == "float64")
    state = node.initial_state()
    assert int(state["frame_step"]) == -1
    last_image = np.asarray(state["image"])
    for n in range(steps):
        state = step(node, state, pose_at(n))
        due = n % period == 0
        assert int(state["steps"]) == n + 1
        assert int(state["new_frame"]) == int(due), (n, period)
        assert int(state["frame_step"]) == (n // period) * period, (n, period)
        assert int(state["age"]) == n - (n // period) * period, (n, period)
        image = np.asarray(state["image"])
        if not due:
            assert np.array_equal(image, last_image)
        last_image = image
        wanted = np.stack([pose_at(max(n - back, 0)) for back in range(sub_frames - 1, -1, -1)])
        held = np.asarray(state["ring"])[:, 0, :]
        # Before the run has taken that many steps the ring holds the initial pose (the body's start).
        initial = np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
        for row, (back, pose) in enumerate(zip(range(sub_frames - 1, -1, -1), wanted, strict=True)):
            expect = pose if n - back >= 0 else initial
            assert np.allclose(held[row], expect, atol=1e-6), (n, row)
