"""The replay contract: an in-loop frame equals the offline render of the recorded poses with the same function and the
same stream key (``mime.sensing.frame_key``), in both dtypes: every pixel but a rounding boundary's (a share under 1e-3, one count at most, measured at 1 pixel of
19,200 in float32; the prediction said bit for bit and missed). The poses are the
body's, one after each step (what a recording keeps); the frame made at step ``n`` was exposed over the poses after steps
``n - K + 1`` to ``n`` (the body's start before the first).

**The controls**: the ring read one step late (``CAMERA_PLANT=ring``: a frame exposed over the previous step's poses) and
a wrong stream key (``CAMERA_PLANT=key``, the seed plus one) and a shift of one count over a band of rows
(``CAMERA_PLANT=shift``, planted in the node's image) each fail it, every run."""

from __future__ import annotations

import jax
import numpy as np
from in_loop_cases import LOOK_ACROSS, RADIUS_UM, ballistic, planted, plant_node
from mime.sensing import Appearance, CameraPose, Exposure, Response, Subject, frame_key, render, static_cache
from mime.sensing._node import CameraNode
from mime.units import ms

if planted("ring"):
    plant_node({"exposed": lambda self, before, ring: before})
if planted("shift"):
    held_render = CameraNode._rendered  # noqa: SLF001

    def shifted(self, *arguments):
        image, expected, flags = held_render(self, *arguments)
        return image.at[:12].add(1.0), expected, flags

    CameraNode._rendered = shifted  # noqa: SLF001
if planted("key"):
    plant_node({"key_of": lambda self, seed, count: jax.random.fold_in(jax.random.PRNGKey(seed + 1), count)})

TUBE = {"axis": (1.0, 0.0, 0.0), "inner_radius": 60e-6, "wall": 10e-6, "length": 480e-6, "centre": (0.0, 0.0, 0.0)}


def test_a_frame_is_the_offline_render_of_its_recorded_poses(dtype):
    built = ballistic(ms(3), sub_frames=2).build()
    node = built.graph.get_node("camera:")
    assert isinstance(node, CameraNode)
    camera, held = node._held  # noqa: SLF001 - the detector and the window the build sized
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
            frames.append((n, np.asarray(state["camera:"]["image"]), np.asarray(state["camera:"]["expected"], dtype=np.float64)))
    assert [n for n, _, _ in frames] == [0, 3, 6]
    wide = dtype == "float64"
    floor = 64 * float(np.finfo(np.float64 if wide else np.float32).eps) * 1000.0
    for n, image, expected in frames:
        rows = np.stack([after[n - 1] if n >= 1 else start, after[n]])
        replay = render([wall, ball], camera, camera_pose, {"ball": (rows[:, :3], rows[:, 3:])},
                        exposure=Exposure(2), response=Response(), key=frame_key(seed, n),
                        cache=static_cache([wall, ball], camera, camera_pose))
        moved = np.abs(np.asarray(replay.expected, dtype=np.float64) - expected)
        flipped = int((np.asarray(replay.image) != image).sum())
        # The mechanism: the replay and the in-loop frame are two compiled programs, so the noisy value (the counts
        # before noise and rounding, the shot term and the read term added) differs in its last bits, up to one unit in
        # the last place of its own size at any pixel, and by more at the edge pixels, where the counts before noise were
        # measured to differ by 3.3e-3 counts at most (28 float32 epsilons of the full scale; the gaps below). A pixel
        # whose noisy value has a rounding boundary (a half count) between the two values rounds to another count: a
        # boundary falls in a gap of width g with probability g, boundaries one a count apart. So the flips expected
        # are the sum of the gaps, which is at most the sum of each pixel's unit in the last place plus the measured
        # gaps in `expected`; the line is that and four standard deviations of a Poisson count of it, plus one.
        unit = float(np.sum(np.spacing(np.abs(np.asarray(replay.image, dtype=np.float64 if wide else np.float32)))))
        assert moved.max() <= floor, (n, moved.max(), floor)
        expected_flips = float(moved.sum()) + unit
        assert flipped <= expected_flips + 4.0 * np.sqrt(expected_flips) + 1.0, (n, flipped, expected_flips)
        assert np.abs(np.asarray(replay.image) - image).max() <= 1.0
        if wide:
            assert flipped == 0, (n, flipped)
        print(f"frame {n} {dtype}: {flipped} pixels flipped, expected {expected_flips:.4g}, largest gap {moved.max():.3g}")
