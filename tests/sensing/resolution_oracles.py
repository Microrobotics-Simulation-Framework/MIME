"""The oracles ``discriminating_range.py`` scans (``--oracle resolution_oracles:replay_flags --path tests/sensing``): each is
``function(size, seed) -> bool``, True when it flags a planted error of ``size``.

``replay_flags`` is the replay gate's comparison of the counts before noise (``test_in_loop_replay.py``: the largest
difference from the offline render is at most 64 eps of the full scale) on the frames of a built camera, with an error of
``size`` counts added at one pixel; the seed picks the frame of three and the pixel. The run's dtype is the process's
(``JAX_ENABLE_X64``)."""

from __future__ import annotations

import functools

import jax
import numpy as np
from in_loop_cases import LOOK_ACROSS, RADIUS_UM, ballistic
from mime.sensing import Appearance, CameraPose, Exposure, Response, Subject, frame_key, render, static_cache
from mime.sensing._node import CameraNode
from mime.units import ms

TUBE = {"axis": (1.0, 0.0, 0.0), "inner_radius": 60e-6, "wall": 10e-6, "length": 480e-6, "centre": (0.0, 0.0, 0.0)}


@functools.cache
def moved_frames() -> tuple[list[np.ndarray], float]:
    """The absolute difference of ``expected`` from the offline render's, for each of the three frames, and the floor."""
    built = ballistic(ms(3), sub_frames=2).build()
    node = built.graph.get_node("camera:")
    assert isinstance(node, CameraNode)
    camera, held, _ = node._held  # noqa: SLF001
    window = next(s.window for s in held if s.name == "ball")
    seed = node.constants()["seed"]
    wall = Subject("tube", "tube", TUBE, "wall", Appearance.attenuating(3000.0))
    ball = Subject("ball", "sphere", {"radius": RADIUS_UM * 1e-6, "centre": [0.0, 0.0, 0.0]}, moving=True, window=window)
    camera_pose = CameraPose(np.array([0.0, -1e-3, 0.0]), np.array(LOOK_ACROSS))
    start = np.asarray(built.current_state()["ball"]["pose"])
    after: list[np.ndarray] = []
    moved: list[np.ndarray] = []
    for n in range(9):
        built.run(ms(1))
        state = built.current_state()
        after.append(np.asarray(state["ball"]["pose"]))
        if int(state["camera:"]["new_frame"]):
            rows = np.stack([after[n - 1] if n >= 1 else start, after[n]])
            replay = render([wall, ball], camera, camera_pose, {"ball": (rows[:, :3], rows[:, 3:])}, exposure=Exposure(2),
                            response=Response(), key=frame_key(seed, n),
                            cache=static_cache([wall, ball], camera, camera_pose))
            moved.append(np.abs(np.asarray(replay.expected, dtype=np.float64)
                                - np.asarray(state["camera:"]["expected"], dtype=np.float64)))
    wide = jax.config.jax_enable_x64
    return moved, 64 * float(np.finfo(np.float64 if wide else np.float32).eps) * 1000.0


def replay_flags(size: float, seed: int) -> bool:
    moved, floor = moved_frames()
    rng = np.random.default_rng(seed)
    frame = moved[int(rng.integers(len(moved)))].copy()
    row, column = int(rng.integers(frame.shape[0])), int(rng.integers(frame.shape[1]))
    frame[row, column] += size
    return bool(frame.max() > floor)
