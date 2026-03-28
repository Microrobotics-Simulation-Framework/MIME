"""USDRecorderObserver — records time-sampled USD animation for .usdc export.

Records simulation state as USD time samples on the StageBridge's stage.
On save(), sets stage time metadata and exports to a replayable .usdc file
openable in usdview, MICROROBOTICA, and Omniverse.

Can coexist with StreamingObserver on the same bridge — default-time
writes (for live Hydra rendering) and time-sampled writes (for
recording) are independent storage on the same USD attributes.

Usage (standalone recording):
    bridge = StageBridge()
    bridge.register_robot("body", CylinderGeometry(2e-3, 10e-3))

    recorder = USDRecorderObserver(bridge, "output.usdc", live=False)
    runner = PolicyRunner(gm, ci, observers=[recorder])
    runner.run(n_steps, dt)
    recorder.save()

Usage (alongside live viewport):
    recorder = USDRecorderObserver(bridge, "output.usdc", live=True)
    runner.add_observer(recorder)
    runner.add_observer(streaming_observer)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pxr import Usd
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


class USDRecorderObserver:
    """PolicyRunner observer that records time-sampled USD animation.

    Parameters
    ----------
    bridge : StageBridge
        Writes simulation state to the USD stage.
    output_path : str
        Path for the exported file. Use .usdc for binary crate format.
    sampling_interval : int
        Record a time sample every ``sampling_interval`` simulation steps.
        Default 1 (every step).
    fps : float
        Playback frame rate written to the stage. Controls playback
        speed in usdview/Omniverse. Default 24.0.
    live : bool
        If True (default), also fires ``bridge.update(state)`` at default
        time each step for live Hydra rendering coexistence. If False,
        only time-sampled writes occur (standalone recording).
    flow_extractor : callable, optional
        ``(state_dict) -> np.ndarray | None`` that extracts velocity
        magnitude from the state dict for flow field recording. When
        provided, flow mesh colors are recorded as time-sampled
        ``primvars:displayColor``.
    flow_prim_path : str
        USD prim path for the flow cross-section mesh.
    flow_colormap : str
        Matplotlib colormap name for flow visualization.
    """

    def __init__(
        self,
        bridge: Any,
        output_path: str,
        sampling_interval: int = 1,
        fps: float = 24.0,
        live: bool = True,
        flow_extractor: Optional[Callable] = None,
        flow_prim_path: str = "/World/Analysis/FlowField",
        flow_colormap: str = "viridis",
    ):
        self.bridge = bridge
        self.output_path = output_path
        self.sampling_interval = max(1, sampling_interval)
        self.fps = fps
        self.live = live
        self.flow_extractor = flow_extractor
        self.flow_prim_path = flow_prim_path
        self.flow_colormap = flow_colormap
        self._step_count = 0
        self._sample_count = 0
        # Value-change filtering: last written robot transform values
        # Key: node_name, Value: (position_np, orientation_np)
        self._last_robot_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(
        self,
        t: float,
        dt: float,
        true_state: dict[str, dict[str, Any]],
        observed_state: dict[str, dict[str, Any]],
        external_inputs: dict[str, dict[str, Any]],
        applied_inputs: dict[str, dict[str, Any]],
    ) -> None:
        """Called after each PolicyRunner step."""
        # Update bridge at default time (for live Hydra coexistence)
        if self.live:
            self.bridge.update(true_state)
        self._step_count += 1

        # Record time sample at decimated rate
        if self._step_count % self.sampling_interval == 0:
            time_code = Usd.TimeCode(self._sample_count) if _HAS_USD else self._sample_count

            # Value-change filtering: build a filtered state dict
            # containing only entries whose robot transforms changed.
            # bridge.update() handles partial dicts gracefully — it
            # skips missing node_names via state.get() → continue.
            filtered = {}
            for key, node_state in true_state.items():
                pos = node_state.get("position")
                ori = node_state.get("orientation")
                if pos is not None and ori is not None:
                    # Robot state — filter unchanged transforms
                    last = self._last_robot_values.get(key)
                    pos_np = np.asarray(pos)
                    ori_np = np.asarray(ori)
                    if last is not None:
                        last_pos, last_ori = last
                        if (np.allclose(pos_np, last_pos)
                                and np.allclose(ori_np, last_ori)):
                            continue  # unchanged — skip write
                    self._last_robot_values[key] = (pos_np.copy(), ori_np.copy())
                # Non-robot state (fields, etc.) — always include
                filtered[key] = node_state

            self.bridge.update(filtered, time_code=time_code)

            # Record flow field if extractor provided
            if self.flow_extractor is not None:
                velocity_mag = self.flow_extractor(true_state)
                if velocity_mag is not None:
                    self.bridge.update_flow_cross_section(
                        velocity_mag, self.flow_prim_path,
                        self.flow_colormap, time_code=time_code,
                    )

            self._sample_count += 1

    def save(self) -> None:
        """Export the recorded animation to disk.

        Sets stage time metadata (start/end time codes, FPS) and writes
        simulation metadata to ``customLayerData`` for reproducibility,
        then exports via ``bridge.export()``.
        """
        stage = self.bridge.stage
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(max(0, self._sample_count - 1))
        stage.SetTimeCodesPerSecond(self.fps)
        stage.SetFramesPerSecond(self.fps)

        # Simulation metadata for reproducibility
        stage.GetRootLayer().customLayerData = {
            "mime:sampling_interval": self.sampling_interval,
            "mime:fps": self.fps,
            "mime:total_steps": self._step_count,
            "mime:total_samples": self._sample_count,
        }

        self.bridge.export(self.output_path)
        logger.info(
            "USDRecorderObserver: saved %d samples (%d steps) to %s",
            self._sample_count, self._step_count, self.output_path,
        )

    @property
    def sample_count(self) -> int:
        """Number of time samples recorded so far."""
        return self._sample_count
