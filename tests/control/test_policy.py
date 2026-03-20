"""Tests for ControlPolicy, ControlPrimitive, ControlSequence, SequentialPolicy."""

import pytest

from mime.control.policy import (
    ControlPolicy,
    ControlPrimitive,
    ControlSequence,
    ExternalInputs,
    PolicyState,
    SequentialPolicy,
)
from mime.control.primitives import (
    HoldField,
    RampField,
    RotateField,
    StepDown,
    SweepFrequency,
)


# -- Concrete test policy --------------------------------------------------

class ConstantPolicy(ControlPolicy):
    """Returns a fixed command every step. For testing."""

    def __init__(self, node: str, field: str, value: float):
        self.node = node
        self.field = field
        self.value = value

    def __call__(self, t, observed_state, policy_state):
        count = policy_state.get("count", 0) + 1
        return (
            {self.node: {self.field: self.value}},
            {"count": count},
        )

    def initial_policy_state(self):
        return {"count": 0}


class ScalingPolicy(ControlPolicy):
    """Scales a field from observed_state. For testing feedback."""

    def __init__(self, read_node: str, read_field: str,
                 write_node: str, write_field: str, gain: float):
        self.read_node = read_node
        self.read_field = read_field
        self.write_node = write_node
        self.write_field = write_field
        self.gain = gain

    def __call__(self, t, observed_state, policy_state):
        val = observed_state.get(self.read_node, {}).get(self.read_field, 0.0)
        return (
            {self.write_node: {self.write_field: self.gain * val}},
            policy_state,
        )

    def initial_policy_state(self):
        return {}


# -- ControlPolicy tests ---------------------------------------------------

class TestControlPolicy:
    def test_constant_policy_returns_correct_inputs(self):
        policy = ConstantPolicy("field", "frequency_hz", 20.0)
        ext, ps = policy(0.0, {}, policy.initial_policy_state())
        assert ext == {"field": {"frequency_hz": 20.0}}
        assert ps["count"] == 1

    def test_policy_state_accumulates(self):
        policy = ConstantPolicy("field", "freq", 10.0)
        ps = policy.initial_policy_state()
        for i in range(5):
            _, ps = policy(float(i), {}, ps)
        assert ps["count"] == 5

    def test_feedback_policy_reads_observed_state(self):
        policy = ScalingPolicy("robot", "position", "field", "gradient", 2.0)
        obs = {"robot": {"position": 3.0}}
        ext, _ = policy(0.0, obs, {})
        assert ext == {"field": {"gradient": 6.0}}

    def test_feedback_policy_handles_missing_node(self):
        policy = ScalingPolicy("robot", "position", "field", "gradient", 2.0)
        ext, _ = policy(0.0, {}, {})
        assert ext == {"field": {"gradient": 0.0}}


# -- SequentialPolicy tests ------------------------------------------------

class TestSequentialPolicy:
    def test_pipe_operator_creates_sequential(self):
        a = ConstantPolicy("field", "freq", 10.0)
        b = ConstantPolicy("field", "strength", 5.0)
        combined = a | b
        assert isinstance(combined, SequentialPolicy)
        assert len(combined.policies) == 2

    def test_sequential_merges_inputs(self):
        a = ConstantPolicy("field", "freq", 10.0)
        b = ConstantPolicy("field", "strength", 5.0)
        combined = a | b
        ext, _ = combined(0.0, {}, combined.initial_policy_state())
        assert ext == {"field": {"freq": 10.0, "strength": 5.0}}

    def test_sequential_later_overrides(self):
        a = ConstantPolicy("field", "freq", 10.0)
        b = ConstantPolicy("field", "freq", 20.0)
        combined = a | b
        ext, _ = combined(0.0, {}, combined.initial_policy_state())
        assert ext == {"field": {"freq": 20.0}}

    def test_sequential_different_nodes(self):
        a = ConstantPolicy("field_a", "freq", 10.0)
        b = ConstantPolicy("field_b", "freq", 20.0)
        combined = a | b
        ext, _ = combined(0.0, {}, combined.initial_policy_state())
        assert "field_a" in ext and "field_b" in ext

    def test_triple_pipe_flattens(self):
        a = ConstantPolicy("a", "x", 1.0)
        b = ConstantPolicy("b", "x", 2.0)
        c = ConstantPolicy("c", "x", 3.0)
        combined = a | b | c
        assert isinstance(combined, SequentialPolicy)
        assert len(combined.policies) == 3

    def test_sequential_maintains_sub_states(self):
        a = ConstantPolicy("field", "freq", 10.0)
        b = ConstantPolicy("field", "strength", 5.0)
        combined = a | b
        ps = combined.initial_policy_state()
        _, ps = combined(0.0, {}, ps)
        _, ps = combined(1.0, {}, ps)
        # Both sub-policies should have count=2
        assert ps["_sub_states"][0]["count"] == 2
        assert ps["_sub_states"][1]["count"] == 2


# -- ControlPrimitive tests ------------------------------------------------

class TestControlPrimitives:
    def test_rotate_field(self):
        p = RotateField(frequency=20.0, field_strength=15.0, duration=1.0)
        result = p.external_inputs(0.5, 0.001, {})
        assert result["frequency_hz"] == 20.0
        assert result["field_strength_mt"] == 15.0

    def test_sweep_frequency_start(self):
        p = SweepFrequency(f_start=10.0, f_end=50.0, duration=4.0)
        result = p.external_inputs(0.0, 0.001, {})
        assert result["frequency_hz"] == pytest.approx(10.0)

    def test_sweep_frequency_midpoint(self):
        p = SweepFrequency(f_start=10.0, f_end=50.0, duration=4.0)
        result = p.external_inputs(2.0, 0.001, {})
        assert result["frequency_hz"] == pytest.approx(30.0)

    def test_sweep_frequency_end(self):
        p = SweepFrequency(f_start=10.0, f_end=50.0, duration=4.0)
        result = p.external_inputs(4.0, 0.001, {})
        assert result["frequency_hz"] == pytest.approx(50.0)

    def test_sweep_frequency_past_end_clamps(self):
        p = SweepFrequency(f_start=10.0, f_end=50.0, duration=4.0)
        result = p.external_inputs(10.0, 0.001, {})
        assert result["frequency_hz"] == pytest.approx(50.0)

    def test_hold_field(self):
        p = HoldField(frequency=30.0, field_strength=8.0, duration=2.0)
        result = p.external_inputs(1.0, 0.001, {})
        assert result["frequency_hz"] == 30.0
        assert result["field_strength_mt"] == 8.0

    def test_ramp_field_starts_at_zero(self):
        p = RampField(frequency=10.0, target_strength=20.0, duration=2.0)
        result = p.external_inputs(0.0, 0.001, {})
        assert result["field_strength_mt"] == pytest.approx(0.0)

    def test_ramp_field_reaches_target(self):
        p = RampField(frequency=10.0, target_strength=20.0, duration=2.0)
        result = p.external_inputs(2.0, 0.001, {})
        assert result["field_strength_mt"] == pytest.approx(20.0)

    def test_ramp_field_midpoint(self):
        p = RampField(frequency=10.0, target_strength=20.0, duration=2.0)
        result = p.external_inputs(1.0, 0.001, {})
        assert result["field_strength_mt"] == pytest.approx(10.0)

    def test_step_down(self):
        p = StepDown(frequency=5.0, field_strength=10.0, duration=1.0)
        result = p.external_inputs(0.0, 0.001, {})
        assert result["frequency_hz"] == 5.0


# -- ControlSequence tests -------------------------------------------------

class TestControlSequence:
    def test_basic_sequence(self):
        seq = ControlSequence([
            RampField(frequency=10.0, target_strength=20.0, duration=1.0,
                      target_node="field"),
            HoldField(frequency=10.0, field_strength=20.0, duration=2.0,
                      target_node="field"),
        ])
        assert seq.total_duration == pytest.approx(3.0)

    def test_sequence_ramp_phase(self):
        seq = ControlSequence([
            RampField(frequency=10.0, target_strength=20.0, duration=1.0,
                      target_node="field"),
            HoldField(frequency=10.0, field_strength=20.0, duration=2.0,
                      target_node="field"),
        ])
        ps = seq.initial_policy_state()
        # At t=0, we're in the ramp phase
        ps["sequence_start_t"] = 0.0
        ext, ps = seq(0.0, {}, ps)
        assert "field" in ext
        assert ext["field"]["field_strength_mt"] == pytest.approx(0.0)

    def test_sequence_hold_phase(self):
        seq = ControlSequence([
            RampField(frequency=10.0, target_strength=20.0, duration=1.0,
                      target_node="field"),
            HoldField(frequency=10.0, field_strength=20.0, duration=2.0,
                      target_node="field"),
        ])
        ps = {"sequence_start_t": 0.0, "active_index": -1, "dt": 0.001}
        # At t=1.5, we should be in the hold phase
        ext, ps = seq(1.5, {}, ps)
        assert ext["field"]["field_strength_mt"] == pytest.approx(20.0)

    def test_sequence_past_end_returns_empty(self):
        seq = ControlSequence([
            HoldField(frequency=10.0, field_strength=20.0, duration=1.0,
                      target_node="field"),
        ], loop=False)
        ps = {"sequence_start_t": 0.0, "active_index": -1, "dt": 0.001}
        ext, _ = seq(2.0, {}, ps)
        assert ext == {}

    def test_sequence_loop(self):
        seq = ControlSequence([
            HoldField(frequency=10.0, field_strength=20.0, duration=1.0,
                      target_node="field"),
            HoldField(frequency=30.0, field_strength=20.0, duration=1.0,
                      target_node="field"),
        ], loop=True)
        ps = {"sequence_start_t": 0.0, "active_index": -1, "dt": 0.001}
        # At t=2.5, looped -> elapsed 0.5 -> first primitive
        ext, _ = seq(2.5, {}, ps)
        assert ext["field"]["frequency_hz"] == pytest.approx(10.0)

    def test_sequence_lifecycle_hooks(self):
        hooks_fired = []

        class TrackingPrimitive(ControlPrimitive):
            def __init__(self, name, duration, target_node="field"):
                super().__init__(duration=duration, target_node=target_node)
                self._name = name

            def external_inputs(self, t_local, dt, observed_state):
                return {"frequency_hz": 0.0}

            def on_start(self, observed_state):
                hooks_fired.append(f"{self._name}_start")

            def on_end(self, observed_state):
                hooks_fired.append(f"{self._name}_end")

        seq = ControlSequence([
            TrackingPrimitive("A", 1.0),
            TrackingPrimitive("B", 1.0),
        ])
        ps = {"sequence_start_t": 0.0, "active_index": -1, "dt": 0.001}

        # First call -> A starts
        _, ps = seq(0.0, {}, ps)
        assert hooks_fired == ["A_start"]

        # Still in A
        _, ps = seq(0.5, {}, ps)
        assert hooks_fired == ["A_start"]

        # Transition to B -> A ends, B starts
        _, ps = seq(1.0, {}, ps)
        assert hooks_fired == ["A_start", "A_end", "B_start"]

    def test_sequence_as_policy(self):
        """ControlSequence is a ControlPolicy — can be used with | operator."""
        seq = ControlSequence([
            HoldField(frequency=10.0, field_strength=5.0, duration=1.0,
                      target_node="field"),
        ])
        other = ConstantPolicy("sensor", "gain", 2.0)
        combined = seq | other
        assert isinstance(combined, SequentialPolicy)

    def test_empty_sequence(self):
        seq = ControlSequence([])
        assert seq.total_duration == 0.0
        ps = seq.initial_policy_state()
        ps["sequence_start_t"] = 0.0
        ext, _ = seq(0.0, {}, ps)
        assert ext == {}
