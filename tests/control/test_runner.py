"""Tests for PolicyRunner — the observe-decide-actuate-step orchestrator.

These tests use a mock GraphManager to avoid depending on MADDENING's
full graph infrastructure. The mock captures step() calls and returns
predictable state dicts.
"""

import pytest
import jax
import jax.numpy as jnp

from mime.control.runner import (
    IdentityUncertainty,
    PolicyRunner,
    RunResult,
    StepObserver,
    UncertaintyModel,
)
from mime.control.input_source import (
    CallbackControlInput,
    ConstantControlInput,
    PolicyControlInput,
    RemoteControlInput,
)
from mime.control.policy import ControlPolicy, ExternalInputs, PolicyState


# -- Mock GraphManager -----------------------------------------------------

class MockGraphManager:
    """Mimics GraphManager.step() and get_state() for testing.

    Tracks all external_inputs received and returns a predictable
    state that increments a counter each step.
    """

    def __init__(self, initial_state: dict | None = None):
        self._state = initial_state or {
            "robot": {"position": jnp.array(0.0), "velocity": jnp.array(0.0)}
        }
        self.step_count = 0
        self.received_inputs: list[dict] = []

    def get_state(self) -> dict:
        return self._state

    def step(self, external_inputs=None) -> dict:
        self.received_inputs.append(external_inputs or {})
        self.step_count += 1
        # Advance position by velocity (trivial dynamics)
        pos = self._state["robot"]["position"]
        vel = self._state["robot"]["velocity"]
        self._state = {
            "robot": {
                "position": pos + vel * 0.001,
                "velocity": vel,
            }
        }
        return self._state


# -- Mock UncertaintyModel -------------------------------------------------

class NoisyUncertainty:
    """Adds a fixed offset to observations and actuation for testing."""

    def __init__(self, obs_offset: float = 0.1, act_offset: float = 0.01):
        self.obs_offset = obs_offset
        self.act_offset = act_offset

    def observe(self, true_state, t, rng_key):
        result = {}
        for node_name, fields in true_state.items():
            result[node_name] = {}
            for field_name, value in fields.items():
                if hasattr(value, '__add__'):
                    result[node_name][field_name] = value + self.obs_offset
                else:
                    result[node_name][field_name] = value
        return result

    def actuate(self, commanded_inputs, t, rng_key):
        result = {}
        for node_name, fields in commanded_inputs.items():
            result[node_name] = {}
            for field_name, value in fields.items():
                if isinstance(value, (int, float)):
                    result[node_name][field_name] = value + self.act_offset
                else:
                    result[node_name][field_name] = value
        return result


# -- PolicyRunner basic tests ----------------------------------------------

class TestPolicyRunnerBasic:
    def test_step_advances_time(self):
        gm = MockGraphManager()
        ci = ConstantControlInput({})
        runner = PolicyRunner(gm, ci)
        assert runner.t == 0.0
        runner.step(0.001)
        assert runner.t == pytest.approx(0.001)
        runner.step(0.001)
        assert runner.t == pytest.approx(0.002)

    def test_step_calls_graph_manager(self):
        gm = MockGraphManager()
        ci = ConstantControlInput({"field": {"freq": 10.0}})
        runner = PolicyRunner(gm, ci)
        runner.step(0.001)
        assert gm.step_count == 1
        assert gm.received_inputs[0] == {"field": {"freq": 10.0}}

    def test_step_returns_state(self):
        gm = MockGraphManager()
        ci = ConstantControlInput({})
        runner = PolicyRunner(gm, ci)
        state = runner.step(0.001)
        assert "robot" in state
        assert "position" in state["robot"]

    def test_run_multiple_steps(self):
        gm = MockGraphManager()
        ci = ConstantControlInput({})
        runner = PolicyRunner(gm, ci)
        result = runner.run(100, 0.001)
        assert isinstance(result, RunResult)
        assert result.n_steps == 100
        assert gm.step_count == 100
        assert result.sim_time_s == pytest.approx(0.1)
        assert result.wall_time_s > 0

    def test_reset(self):
        gm = MockGraphManager()
        ci = PolicyControlInput(
            type("P", (ControlPolicy,), {
                "__call__": lambda self, t, obs, ps: ({}, {"n": ps.get("n", 0) + 1}),
                "initial_policy_state": lambda self: {"n": 0},
            })()
        )
        runner = PolicyRunner(gm, ci)
        runner.step(0.001)
        runner.step(0.001)
        assert runner.t == pytest.approx(0.002)
        runner.reset(rng_seed=42)
        assert runner.t == 0.0


# -- Uncertainty integration ------------------------------------------------

class TestPolicyRunnerUncertainty:
    def test_identity_uncertainty_is_default(self):
        gm = MockGraphManager()
        ci = ConstantControlInput({"field": {"freq": 10.0}})
        runner = PolicyRunner(gm, ci)
        runner.step(0.001)
        # With identity, commands pass through unchanged
        assert gm.received_inputs[0] == {"field": {"freq": 10.0}}

    def test_noisy_uncertainty_modifies_actuation(self):
        gm = MockGraphManager()
        ci = ConstantControlInput({"field": {"freq": 10.0}})
        uncertainty = NoisyUncertainty(obs_offset=0.0, act_offset=0.5)
        runner = PolicyRunner(gm, ci, uncertainty=uncertainty)
        runner.step(0.001)
        # Actuation noise should add 0.5 to the commanded frequency
        assert gm.received_inputs[0]["field"]["freq"] == pytest.approx(10.5)

    def test_noisy_uncertainty_modifies_observation(self):
        """Policy sees noisy state, not true state."""
        observed_states = []

        class RecordingPolicy(ControlPolicy):
            def __call__(self, t, obs, ps):
                observed_states.append(obs)
                return {}, ps
            def initial_policy_state(self):
                return {}

        gm = MockGraphManager({
            "robot": {"position": jnp.array(1.0), "velocity": jnp.array(0.0)}
        })
        ci = PolicyControlInput(RecordingPolicy())
        uncertainty = NoisyUncertainty(obs_offset=0.1, act_offset=0.0)
        runner = PolicyRunner(gm, ci, uncertainty=uncertainty)
        runner.step(0.001)

        # Policy should see position = 1.0 + 0.1 = 1.1
        obs_pos = float(observed_states[0]["robot"]["position"])
        assert obs_pos == pytest.approx(1.1)


# -- Observer tests --------------------------------------------------------

class TestPolicyRunnerObservers:
    def test_observer_fires_each_step(self):
        calls = []

        def my_observer(t, dt, true_state, obs_state, ext_in, app_in):
            calls.append({"t": t, "dt": dt})

        gm = MockGraphManager()
        ci = ConstantControlInput({})
        runner = PolicyRunner(gm, ci, observers=[my_observer])
        runner.run(5, 0.001)
        assert len(calls) == 5

    def test_observer_receives_all_data(self):
        captured = {}

        def my_observer(t, dt, true_state, obs_state, ext_in, app_in):
            captured["t"] = t
            captured["dt"] = dt
            captured["true_state"] = true_state
            captured["obs_state"] = obs_state
            captured["ext_in"] = ext_in
            captured["app_in"] = app_in

        gm = MockGraphManager()
        ci = ConstantControlInput({"field": {"freq": 20.0}})
        runner = PolicyRunner(gm, ci, observers=[my_observer])
        runner.step(0.001)

        assert captured["dt"] == 0.001
        assert "robot" in captured["true_state"]
        assert captured["ext_in"] == {"field": {"freq": 20.0}}

    def test_multiple_observers(self):
        log_a, log_b = [], []
        gm = MockGraphManager()
        ci = ConstantControlInput({})
        runner = PolicyRunner(
            gm, ci,
            observers=[
                lambda t, dt, ts, os, ei, ai: log_a.append(t),
                lambda t, dt, ts, os, ei, ai: log_b.append(t),
            ],
        )
        runner.run(3, 0.001)
        assert len(log_a) == 3
        assert len(log_b) == 3

    def test_add_remove_observer(self):
        log = []
        obs = lambda t, dt, ts, os, ei, ai: log.append(t)

        gm = MockGraphManager()
        ci = ConstantControlInput({})
        runner = PolicyRunner(gm, ci)

        runner.add_observer(obs)
        runner.step(0.001)
        assert len(log) == 1

        runner.remove_observer(obs)
        runner.step(0.001)
        assert len(log) == 1  # No new entries


# -- Remote control integration tests -------------------------------------

class TestPolicyRunnerRemote:
    def test_remote_input_with_runner(self):
        """End-to-end: remote source pushes commands, runner consumes them."""
        gm = MockGraphManager()
        rci = RemoteControlInput(timeout_s=1.0)
        runner = PolicyRunner(gm, rci)

        # Push command before stepping
        rci.push_command({"field": {"freq": 25.0}})
        runner.step(0.001)
        assert gm.received_inputs[0] == {"field": {"freq": 25.0}}

    def test_remote_zero_order_hold_in_runner(self):
        """Runner gets same command when no new one is pushed."""
        gm = MockGraphManager()
        rci = RemoteControlInput(timeout_s=0.1)
        runner = PolicyRunner(gm, rci)

        rci.push_command({"field": {"freq": 15.0}})
        runner.step(0.001)
        runner.step(0.001)  # No new command
        assert gm.received_inputs[0] == {"field": {"freq": 15.0}}
        assert gm.received_inputs[1] == {"field": {"freq": 15.0}}

    def test_remote_command_update_mid_run(self):
        """Commands can be updated between steps."""
        gm = MockGraphManager()
        rci = RemoteControlInput(timeout_s=0.1)
        runner = PolicyRunner(gm, rci)

        rci.push_command({"field": {"freq": 10.0}})
        runner.step(0.001)
        assert gm.received_inputs[0]["field"]["freq"] == 10.0

        rci.push_command({"field": {"freq": 50.0}})
        runner.step(0.001)
        assert gm.received_inputs[1]["field"]["freq"] == 50.0

    def test_callback_input_with_runner(self):
        """CallbackControlInput works as a ControlInput with PolicyRunner."""
        gm = MockGraphManager()
        ci = CallbackControlInput(
            lambda t, dt, obs: {"field": {"freq": t * 100.0}}
        )
        runner = PolicyRunner(gm, ci)
        runner.step(0.001)
        runner.step(0.001)
        # At t=0.001 (after first step), freq = 0.001 * 100 = 0.1
        assert gm.received_inputs[1]["field"]["freq"] == pytest.approx(0.1)


# -- RNG determinism tests ------------------------------------------------

class TestPolicyRunnerRNG:
    def test_same_seed_same_results(self):
        """Two runners with the same seed and uncertainty produce identical results."""
        results = []
        for _ in range(2):
            gm = MockGraphManager({
                "robot": {"position": jnp.array(1.0), "velocity": jnp.array(0.5)}
            })
            ci = ConstantControlInput({"field": {"freq": 10.0}})
            runner = PolicyRunner(gm, ci, uncertainty=NoisyUncertainty(), rng_seed=42)
            result = runner.run(10, 0.001)
            results.append(float(result.final_state["robot"]["position"]))

        assert results[0] == pytest.approx(results[1])

    def test_different_seed_different_rng_keys(self):
        """Different seeds produce different RNG key sequences."""
        runner_a = PolicyRunner(MockGraphManager(), ConstantControlInput({}), rng_seed=0)
        runner_b = PolicyRunner(MockGraphManager(), ConstantControlInput({}), rng_seed=1)
        assert not jnp.array_equal(runner_a.rng_key, runner_b.rng_key)
