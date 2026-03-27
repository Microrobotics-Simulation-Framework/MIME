"""StreamingObserver — renders + streams each simulation frame via Selkies.

Combines StageBridge (USD state write) + HydraStormViewport (render to
pixels) + SelkiesSession (H.264 encode + WebRTC stream) into a single
PolicyRunner observer callback.

The observer fires after each simulation step. To achieve a target frame
rate, a ``frame_skip`` parameter controls how often rendering + streaming
occurs (e.g. frame_skip=200 means render every 200 LBM steps → ~2 fps
at 64³).

Usage:
    bridge = StageBridge()
    viewport = HydraStormViewport(width=1280, height=720)
    selkies = SelkiesSession(secret="...")
    selkies.start(StreamConfig(width=1280, height=720, fps=2))

    observer = StreamingObserver(bridge, viewport, selkies, frame_skip=200)
    runner.add_observer(observer)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class StreamingObserver:
    """PolicyRunner observer that renders + streams each frame via Selkies.

    Parameters
    ----------
    bridge : StageBridge
        Writes simulation state to the USD stage.
    viewport : HydraStormViewport
        Renders the USD stage to RGBA pixels.
    selkies : SelkiesSession
        Pushes rendered pixels to the WebRTC stream.
    frame_skip : int
        Render and stream every ``frame_skip`` steps. Default 1 (every step).
        Compute from target FPS and step time:
        ``frame_skip = max(1, int(1.0 / (target_fps * step_time)))``
    """

    def __init__(
        self,
        bridge: Any,
        viewport: Any,
        selkies: Any,
        frame_skip: int = 1,
    ):
        self.bridge = bridge
        self.viewport = viewport
        self.selkies = selkies
        self.frame_skip = max(1, frame_skip)
        self._frame_count = 0
        self._render_count = 0

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
        # Always update the USD stage (cheap — batched attribute writes)
        self.bridge.update(true_state)
        self._frame_count += 1

        # Render + stream at the target frame rate
        if self._frame_count % self.frame_skip == 0:
            pixels = self.viewport.render()
            if pixels is not None and pixels.size > 0:
                self.selkies.update_framebuffer_cpu(
                    pixels.tobytes(),
                    self.viewport.width,
                    self.viewport.height,
                )
                self._render_count += 1

                if self._render_count % 10 == 0:
                    logger.debug(
                        "StreamingObserver: rendered %d frames "
                        "(skip=%d, sim_step=%d)",
                        self._render_count, self.frame_skip, self._frame_count,
                    )

    @property
    def render_count(self) -> int:
        """Number of frames rendered and streamed so far."""
        return self._render_count

    @staticmethod
    def compute_frame_skip(
        step_time_s: float,
        target_fps: float = 2.0,
    ) -> int:
        """Compute frame_skip from step time and target FPS.

        Parameters
        ----------
        step_time_s : float
            Physical time per simulation step [s].
        target_fps : float
            Target streaming frame rate.

        Returns
        -------
        int
            Number of steps between rendered frames.
        """
        if step_time_s <= 0 or target_fps <= 0:
            return 1
        return max(1, int(1.0 / (target_fps * step_time_s)))
