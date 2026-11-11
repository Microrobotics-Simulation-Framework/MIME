"""The replay contract: an in-loop frame equals the offline render of the recorded poses with the same function and the
same stream key (``mime.sensing.frame_key``), in both dtypes: every pixel but a rounding boundary's (a share under 1e-3, one count at most, measured at 1 pixel of
19,200 in float32; the prediction said bit for bit and missed). The poses are the
body's, one after each step (what a recording keeps); the frame made at step ``n`` was exposed over the poses after steps
``n - K + 1`` to ``n`` (the body's start before the first).

**The controls**: the ring read one step late (``CAMERA_PLANT=ring``: a frame exposed over the previous step's poses) and
a wrong stream key (``CAMERA_PLANT=key``, the seed plus one) each fail the equality, every run."""

from __future__ import annotations

import jax
import numpy as np
from in_loop_cases import LOOK_ACROSS, RADIUS_UM, ballistic, planted, plant_node
from mime.sensing import Appearance, CameraPose, Exposure, Response, Subject, frame_key, render, static_cache
from mime.sensing._node import CameraNode
from mime.units import ms

if planted("ring"):
    plant_node({"exposed": lambda self, before, ring: before})
if planted("key"):
    plant_node({"key_of": lambda self, seed, count: jax.random.fold_in(jax.random.PRNGKey(seed + 1), count)})

TUBE = {"axis": (1.0, 0.0, 0.0), "inner_radius": 60e-6, "wall": 10e-6, "length": 480e-6, "centre": (0.0, 0.0, 0.0)}


def test_a_frame_is_the_offline_render_of_its_recorded_poses(dtype):
    built = ballistic(ms(3), sub_frames=2).build()
    node = built.graph.get_node("camera:")
    assert isinstance(node, CameraNode)
    camera, held, cache = node._held  # noqa: SLF001 - the detector and the window the build sized
    window = next(s.window for s in held if s.name == "ball")
    seed = node.constants()["seed"]
    wall = Subject("tube", "tube", TUBE, "wall", Appearance.attenuating(3000.0))
    ball = Subject("ball", "sphere", {"radius": RADIUS_UM * 1e-6, "centre": [0.0, 0.0, 0.0]}, moving=True, window=window)
    camera_pose = CameraPose(np.array([0.0, -1e-3, 0.0]), np.array(LOOK_ACROSS))
    start = np.asarray(built.current_state()["ball"]["pose"])
    after: list[np.ndarray] = []
    frames = []
    for n in range(9):
        built.run(ms(1))
        state = built.current_state()
        after.append(np.asarray(state["ball"]["pose"]))
        if int(state["camera:"]["new_frame"]):
            frames.append((n, np.asarray(state["camera:"]["image"])))
    assert [n for n, _ in frames] == [0, 3, 6]
    differences = []
    for n, image in frames:
        rows = np.stack([after[n - 1] if n >= 1 else start, after[n]])
        replay = render([wall, ball], camera, camera_pose, {"ball": (rows[:, :3], rows[:, 3:])},
                        exposure=Exposure(2), response=Response(), key=frame_key(seed, n),
                        cache=static_cache([wall, ball], camera, camera_pose))
        gap = np.abs(np.asarray(replay.image) - image)
        differences.append((float(gap.max()), float((gap > 0).mean())))
    print("largest difference in counts and share of pixels differing, per frame:", differences, "dtype", dtype)
    # The compiled step and the eager offline call are not one XLA program: a pixel whose noisy value lies within a few
    # ulps of a rounding boundary rounds the other way (one count). The line is derived: a pixel is that near a boundary
    # with probability about 4 ulps of its count (6e-5 relative in float32, a count of 1e3) over one count, 2.4e-4.
    assert max(g for g, _ in differences) <= 1.0, differences
    assert max(share for _, share in differences) <= 1e-3, differences
