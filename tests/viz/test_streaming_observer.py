"""Tests for StreamingObserver — frame skip logic and observer protocol."""

import numpy as np
import pytest

from mime.viz.streaming_observer import StreamingObserver


class MockBridge:
    def __init__(self):
        self.update_count = 0

    def update(self, state):
        self.update_count += 1


class MockViewport:
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        self.render_count = 0

    def render(self):
        self.render_count += 1
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)


class MockSelkies:
    def __init__(self):
        self.frames_pushed = 0
        self.last_width = 0
        self.last_height = 0

    def update_framebuffer_cpu(self, pixels, width, height):
        self.frames_pushed += 1
        self.last_width = width
        self.last_height = height


class TestStreamingObserverBasic:
    def test_every_step_without_skip(self):
        bridge = MockBridge()
        viewport = MockViewport()
        selkies = MockSelkies()
        obs = StreamingObserver(bridge, viewport, selkies, frame_skip=1)

        for i in range(10):
            obs(float(i), 0.001, {}, {}, {}, {})

        assert bridge.update_count == 10
        assert viewport.render_count == 10
        assert selkies.frames_pushed == 10

    def test_frame_skip(self):
        bridge = MockBridge()
        viewport = MockViewport()
        selkies = MockSelkies()
        obs = StreamingObserver(bridge, viewport, selkies, frame_skip=5)

        for i in range(20):
            obs(float(i), 0.001, {}, {}, {}, {})

        # Bridge updates every step
        assert bridge.update_count == 20
        # Viewport renders every 5 steps
        assert viewport.render_count == 4
        assert selkies.frames_pushed == 4

    def test_render_count_property(self):
        bridge = MockBridge()
        viewport = MockViewport()
        selkies = MockSelkies()
        obs = StreamingObserver(bridge, viewport, selkies, frame_skip=3)

        for i in range(9):
            obs(float(i), 0.001, {}, {}, {}, {})

        assert obs.render_count == 3

    def test_frame_dimensions_passed_to_selkies(self):
        bridge = MockBridge()
        viewport = MockViewport(width=1280, height=720)
        selkies = MockSelkies()
        obs = StreamingObserver(bridge, viewport, selkies, frame_skip=1)

        obs(0.0, 0.001, {}, {}, {}, {})

        assert selkies.last_width == 1280
        assert selkies.last_height == 720


class TestComputeFrameSkip:
    def test_64_cubed_2fps(self):
        # 64³: ~0.005s/step → frame_skip = 100
        skip = StreamingObserver.compute_frame_skip(0.005, target_fps=2.0)
        assert skip == 100

    def test_32_cubed_2fps(self):
        # 32³: ~0.001s/step → frame_skip = 500
        skip = StreamingObserver.compute_frame_skip(0.001, target_fps=2.0)
        assert skip == 500

    def test_128_cubed_2fps(self):
        # 128³: ~0.012s/step → frame_skip = 41
        skip = StreamingObserver.compute_frame_skip(0.012, target_fps=2.0)
        assert skip == 41

    def test_very_fast_step(self):
        # Minimum 1
        skip = StreamingObserver.compute_frame_skip(0.0001, target_fps=1.0)
        assert skip == 10000

    def test_zero_step_time(self):
        skip = StreamingObserver.compute_frame_skip(0.0, target_fps=2.0)
        assert skip == 1

    def test_zero_fps(self):
        skip = StreamingObserver.compute_frame_skip(0.005, target_fps=0.0)
        assert skip == 1
