"""Tests for USDRecorderObserver — mock-based logic + real-USD integration."""

import os
import tempfile

import numpy as np
import pytest

from mime.viz.usd_recorder import USDRecorderObserver


# ---------------------------------------------------------------------------
# Mock helpers (no pxr dependency)
# ---------------------------------------------------------------------------

class MockBridge:
    """Tracks update() and update_flow_cross_section() calls."""

    def __init__(self):
        self.update_calls = []  # list of (state, time_code_kwarg)
        self.flow_calls = []    # list of (vel_mag, prim_path, colormap, time_code)

    def update(self, state, time_code=None):
        self.update_calls.append((state, time_code))

    def update_flow_cross_section(
        self, velocity_magnitude, prim_path, colormap, time_code=None,
    ):
        self.flow_calls.append((velocity_magnitude, prim_path, colormap, time_code))


def _make_state(px=0.0, py=0.0, pz=0.0):
    """Helper: create a state dict with a single robot."""
    return {"body": {
        "position": np.array([px, py, pz]),
        "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
    }}


def _call_n(recorder, n, state_fn=None):
    """Call the recorder n times with optional per-step state function."""
    for i in range(n):
        state = state_fn(i) if state_fn else _make_state(px=i * 0.001)
        recorder(float(i), 0.001, state, state, {}, {})


# ---------------------------------------------------------------------------
# Mock-based tests (tests 1-8)
# ---------------------------------------------------------------------------

class TestRecorderLogic:
    def test_records_every_step(self):
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", sampling_interval=1, live=False)
        _call_n(rec, 5)

        assert rec.sample_count == 5
        # Every call should have a time_code (non-None)
        tc_calls = [tc for _, tc in bridge.update_calls if tc is not None]
        assert len(tc_calls) == 5

    def test_sampling_interval(self):
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", sampling_interval=5, live=False)
        _call_n(rec, 20)

        assert rec.sample_count == 4
        tc_calls = [tc for _, tc in bridge.update_calls if tc is not None]
        assert len(tc_calls) == 4

    def test_sample_count_property(self):
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", sampling_interval=3, live=False)

        assert rec.sample_count == 0
        _call_n(rec, 6)
        assert rec.sample_count == 2
        _call_n(rec, 3)
        assert rec.sample_count == 3

    def test_live_true_fires_default_update(self):
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", sampling_interval=1, live=True)
        _call_n(rec, 3)

        # Each step: one default-time update (tc=None) + one time-sampled update
        default_calls = [(s, tc) for s, tc in bridge.update_calls if tc is None]
        tc_calls = [(s, tc) for s, tc in bridge.update_calls if tc is not None]
        assert len(default_calls) == 3
        assert len(tc_calls) == 3

    def test_live_false_skips_default_update(self):
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", sampling_interval=1, live=False)
        _call_n(rec, 3)

        # No default-time calls at all
        default_calls = [(s, tc) for s, tc in bridge.update_calls if tc is None]
        assert len(default_calls) == 0
        # Only time-sampled calls
        assert len(bridge.update_calls) == 3

    def test_flow_extractor_called_on_sample_frames(self):
        bridge = MockBridge()
        vel_mag = np.ones((8, 8), dtype=np.float32)
        rec = USDRecorderObserver(
            bridge, "/tmp/out.usdc", sampling_interval=2, live=False,
            flow_extractor=lambda state: vel_mag,
            flow_prim_path="/World/Flow",
            flow_colormap="jet",
        )
        _call_n(rec, 6)

        # 3 samples (steps 2, 4, 6)
        assert rec.sample_count == 3
        assert len(bridge.flow_calls) == 3
        for vm, pp, cm, tc in bridge.flow_calls:
            assert pp == "/World/Flow"
            assert cm == "jet"
            assert tc is not None
            assert np.array_equal(vm, vel_mag)

    def test_flow_extractor_none_skips_flow(self):
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", live=False)
        _call_n(rec, 5)

        assert len(bridge.flow_calls) == 0

    def test_value_change_filtering_skips_unchanged(self):
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", sampling_interval=1, live=False)

        same_state = _make_state(px=0.001)

        # First call: writes (new value)
        rec(0.0, 0.001, same_state, same_state, {}, {})
        assert rec.sample_count == 1
        first_state_written = bridge.update_calls[0][0]
        assert "body" in first_state_written

        # Second call: identical state → robot filtered out
        rec(1.0, 0.001, same_state, same_state, {}, {})
        assert rec.sample_count == 2
        second_state_written = bridge.update_calls[1][0]
        assert "body" not in second_state_written

        # Third call: changed position → robot included again
        changed_state = _make_state(px=0.999)
        rec(2.0, 0.001, changed_state, changed_state, {}, {})
        assert rec.sample_count == 3
        third_state_written = bridge.update_calls[2][0]
        assert "body" in third_state_written

    def test_value_change_filtering_includes_non_robot_state(self):
        """Non-robot state (fields, etc.) should always be included."""
        bridge = MockBridge()
        rec = USDRecorderObserver(bridge, "/tmp/out.usdc", sampling_interval=1, live=False)

        state = {
            "body": {
                "position": np.array([0.001, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            },
            "ext_field": {
                "field_vector": np.array([0.0, 0.0, 0.01]),
            },
        }

        # First call: both included
        rec(0.0, 0.001, state, state, {}, {})
        assert "body" in bridge.update_calls[0][0]
        assert "ext_field" in bridge.update_calls[0][0]

        # Second call: body unchanged (filtered), but ext_field still included
        rec(1.0, 0.001, state, state, {}, {})
        assert "body" not in bridge.update_calls[1][0]
        assert "ext_field" in bridge.update_calls[1][0]


# ---------------------------------------------------------------------------
# Real-USD integration tests (tests 9-13)
# ---------------------------------------------------------------------------

class TestRecorderUSDIntegration:
    @pytest.fixture(autouse=True)
    def _require_usd(self):
        pytest.importorskip("pxr")

    @pytest.fixture
    def bridge(self):
        from mime.viz.stage_bridge import StageBridge
        b = StageBridge()
        b.register_robot("body")
        return b

    @pytest.fixture
    def tmp_usdc(self, tmp_path):
        return str(tmp_path / "recording.usdc")

    def _record_frames(self, bridge, output_path, n=10, fps=10.0):
        rec = USDRecorderObserver(
            bridge, output_path,
            sampling_interval=1, fps=fps, live=False,
        )
        for i in range(n):
            state = {"body": {
                "position": np.array([0.001 * i, 0.0, 0.0]),
                "orientation": np.array([
                    np.cos(i * 0.1), np.sin(i * 0.1), 0.0, 0.0,
                ]),
            }}
            rec(float(i) * 0.001, 0.001, state, state, {}, {})
        return rec

    def test_save_sets_time_metadata(self, bridge, tmp_usdc):
        from pxr import Usd
        rec = self._record_frames(bridge, tmp_usdc, n=20, fps=12.0)
        rec.save()

        stage = Usd.Stage.Open(tmp_usdc)
        assert stage.GetStartTimeCode() == 0.0
        assert stage.GetEndTimeCode() == 19.0
        assert stage.GetTimeCodesPerSecond() == 12.0
        assert stage.GetFramesPerSecond() == 12.0

    def test_save_exports_usdc_file(self, bridge, tmp_usdc):
        rec = self._record_frames(bridge, tmp_usdc)
        rec.save()

        assert os.path.exists(tmp_usdc)
        assert os.path.getsize(tmp_usdc) > 0

    def test_save_writes_custom_layer_data(self, bridge, tmp_usdc):
        from pxr import Usd
        rec = self._record_frames(bridge, tmp_usdc, n=15)
        rec.save()

        stage = Usd.Stage.Open(tmp_usdc)
        data = stage.GetRootLayer().customLayerData
        assert data["mime:sampling_interval"] == 1
        assert data["mime:fps"] == 10.0
        assert data["mime:total_steps"] == 15
        assert data["mime:total_samples"] == 15

    def test_coexists_with_default_updates(self, bridge, tmp_usdc):
        from pxr import Usd, UsdGeom, Gf

        rec = USDRecorderObserver(
            bridge, tmp_usdc,
            sampling_interval=1, fps=10.0, live=True,
        )

        state = {"body": {
            "position": np.array([0.005, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
        }}
        rec(0.0, 0.001, state, state, {}, {})

        # Read back from in-memory stage
        prim = bridge.stage.GetPrimAtPath("/World/Robot")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()

        # Default-time value (from live update)
        default_val = ops[0].Get(Usd.TimeCode.Default())
        assert abs(default_val[0] - 0.005) < 1e-6

        # Time-sampled value (from recording)
        sampled_val = ops[0].Get(Usd.TimeCode(0))
        assert abs(sampled_val[0] - 0.005) < 1e-6

    def test_time_samples_have_correct_values(self, bridge, tmp_usdc):
        from pxr import Usd, UsdGeom

        rec = self._record_frames(bridge, tmp_usdc, n=5)
        rec.save()

        stage = Usd.Stage.Open(tmp_usdc)
        prim = stage.GetPrimAtPath("/World/Robot")
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()

        # Verify each frame has the correct position
        for i in range(5):
            val = ops[0].Get(Usd.TimeCode(i))
            expected_x = 0.001 * i
            assert abs(val[0] - expected_x) < 1e-6, (
                f"Frame {i}: expected x={expected_x}, got {val[0]}"
            )
