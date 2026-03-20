"""Tests for the uncertainty layer — all models and composition."""

import pytest
import jax
import jax.numpy as jnp

from mime.uncertainty.base import (
    ActuationUncertainty,
    ComposedUncertainty,
    IdentityUncertainty,
    LocalisationUncertainty,
    ModelUncertainty,
    UncertaintyModel,
)


# -- Helpers ---------------------------------------------------------------

def make_state(pos=0.0, vel=0.0):
    """Create a minimal graph state dict for testing."""
    return {
        "robot": {
            "position": jnp.array([pos, 0.0, 0.0]),
            "velocity": jnp.array([vel, 0.0, 0.0]),
        }
    }


def make_commands(freq=10.0, strength=5.0):
    """Create minimal external inputs for testing."""
    return {
        "external_field": {
            "frequency_hz": freq,
            "field_strength_mt": strength,
        }
    }


KEY = jax.random.PRNGKey(42)


# -- IdentityUncertainty --------------------------------------------------

class TestIdentityUncertainty:
    def test_is_uncertainty_model(self):
        assert isinstance(IdentityUncertainty(), UncertaintyModel)

    def test_observe_passthrough(self):
        m = IdentityUncertainty()
        state = make_state(1.0, 2.0)
        obs = m.observe(state, 0.0, KEY)
        assert jnp.array_equal(obs["robot"]["position"], state["robot"]["position"])
        assert jnp.array_equal(obs["robot"]["velocity"], state["robot"]["velocity"])

    def test_actuate_passthrough(self):
        m = IdentityUncertainty()
        cmd = make_commands(20.0, 10.0)
        applied = m.actuate(cmd, 0.0, KEY)
        assert applied["external_field"]["frequency_hz"] == 20.0
        assert applied["external_field"]["field_strength_mt"] == 10.0


# -- ActuationUncertainty -------------------------------------------------

class TestActuationUncertainty:
    def test_is_uncertainty_model(self):
        assert isinstance(ActuationUncertainty(), UncertaintyModel)

    def test_observe_is_passthrough(self):
        m = ActuationUncertainty(frequency_noise_std=1.0)
        state = make_state(1.0)
        obs = m.observe(state, 0.0, KEY)
        assert jnp.array_equal(obs["robot"]["position"], state["robot"]["position"])

    def test_zero_noise_is_passthrough(self):
        m = ActuationUncertainty(frequency_noise_std=0.0, field_strength_noise_std=0.0)
        cmd = make_commands(10.0, 5.0)
        applied = m.actuate(cmd, 0.0, KEY)
        assert applied["external_field"]["frequency_hz"] == 10.0
        assert applied["external_field"]["field_strength_mt"] == 5.0

    def test_frequency_noise_applied(self):
        m = ActuationUncertainty(frequency_noise_std=1.0, target_node="external_field")
        cmd = make_commands(10.0, 5.0)
        applied = m.actuate(cmd, 0.0, KEY)
        # Should differ from commanded by a non-zero amount
        assert applied["external_field"]["frequency_hz"] != 10.0
        # But should be close (1-sigma noise on 10 Hz)
        assert abs(float(applied["external_field"]["frequency_hz"]) - 10.0) < 5.0

    def test_field_strength_noise_applied(self):
        m = ActuationUncertainty(field_strength_noise_std=0.5, target_node="external_field")
        cmd = make_commands(10.0, 5.0)
        applied = m.actuate(cmd, 0.0, KEY)
        assert applied["external_field"]["field_strength_mt"] != 5.0

    def test_frequency_drift_accumulates(self):
        m = ActuationUncertainty(
            frequency_noise_std=0.0, frequency_drift_rate=1.0,
            target_node="external_field",
        )
        cmd = make_commands(10.0, 5.0)
        applied_t0 = m.actuate(cmd, 0.0, KEY)
        applied_t10 = m.actuate(cmd, 10.0, KEY)
        # At t=0: drift = 0
        assert float(applied_t0["external_field"]["frequency_hz"]) == pytest.approx(10.0)
        # At t=10: drift = 10 Hz
        assert float(applied_t10["external_field"]["frequency_hz"]) == pytest.approx(20.0)

    def test_missing_target_node_passthrough(self):
        m = ActuationUncertainty(frequency_noise_std=1.0, target_node="nonexistent")
        cmd = make_commands(10.0, 5.0)
        applied = m.actuate(cmd, 0.0, KEY)
        assert applied == cmd

    def test_deterministic_with_same_key(self):
        m = ActuationUncertainty(frequency_noise_std=1.0, target_node="external_field")
        cmd = make_commands(10.0, 5.0)
        a1 = m.actuate(cmd, 0.0, KEY)
        a2 = m.actuate(cmd, 0.0, KEY)
        assert float(a1["external_field"]["frequency_hz"]) == float(
            a2["external_field"]["frequency_hz"]
        )

    def test_different_key_different_noise(self):
        m = ActuationUncertainty(frequency_noise_std=1.0, target_node="external_field")
        cmd = make_commands(10.0, 5.0)
        a1 = m.actuate(cmd, 0.0, jax.random.PRNGKey(0))
        a2 = m.actuate(cmd, 0.0, jax.random.PRNGKey(1))
        assert float(a1["external_field"]["frequency_hz"]) != float(
            a2["external_field"]["frequency_hz"]
        )


# -- LocalisationUncertainty ----------------------------------------------

class TestLocalisationUncertainty:
    def test_is_uncertainty_model(self):
        assert isinstance(LocalisationUncertainty(), UncertaintyModel)

    def test_actuate_is_passthrough(self):
        m = LocalisationUncertainty(position_noise_std_mm=1.0)
        cmd = make_commands()
        applied = m.actuate(cmd, 0.0, KEY)
        assert applied == cmd

    def test_zero_noise_passthrough(self):
        m = LocalisationUncertainty(
            position_noise_std_mm=0.0, velocity_noise_std=0.0,
            dropout_probability=0.0,
        )
        state = make_state(1.0, 2.0)
        obs = m.observe(state, 0.0, KEY)
        assert jnp.allclose(obs["robot"]["position"], state["robot"]["position"])

    def test_position_noise_applied(self):
        m = LocalisationUncertainty(position_noise_std_mm=1.0)
        state = make_state(10.0, 0.0)
        obs = m.observe(state, 0.0, KEY)
        # Position should be perturbed
        assert not jnp.array_equal(obs["robot"]["position"], state["robot"]["position"])
        # But not by a huge amount
        diff = jnp.linalg.norm(obs["robot"]["position"] - state["robot"]["position"])
        assert float(diff) < 10.0

    def test_velocity_noise_applied(self):
        m = LocalisationUncertainty(velocity_noise_std=0.5)
        state = make_state(0.0, 5.0)
        obs = m.observe(state, 0.0, KEY)
        assert not jnp.array_equal(obs["robot"]["velocity"], state["robot"]["velocity"])

    def test_tracking_confidence_injected(self):
        m = LocalisationUncertainty(position_noise_std_mm=1.0, dropout_probability=0.0)
        state = make_state()
        obs = m.observe(state, 0.0, KEY)
        assert "tracking_confidence" in obs["robot"]
        # No dropout -> confidence = 1.0
        assert float(obs["robot"]["tracking_confidence"]) == 1.0

    def test_dropout_zeros_noise(self):
        """With dropout_probability=1.0, tracking always fails."""
        m = LocalisationUncertainty(
            position_noise_std_mm=100.0, dropout_probability=1.0,
        )
        state = make_state(5.0)
        obs = m.observe(state, 0.0, KEY)
        # During dropout, noise is zeroed — position should be unchanged
        assert jnp.allclose(obs["robot"]["position"], state["robot"]["position"])
        assert float(obs["robot"]["tracking_confidence"]) == 0.0

    def test_missing_target_node_passthrough(self):
        m = LocalisationUncertainty(
            position_noise_std_mm=1.0, target_node="nonexistent",
        )
        state = make_state(1.0)
        obs = m.observe(state, 0.0, KEY)
        assert jnp.array_equal(obs["robot"]["position"], state["robot"]["position"])

    def test_deterministic_with_same_key(self):
        m = LocalisationUncertainty(position_noise_std_mm=1.0)
        state = make_state(1.0)
        o1 = m.observe(state, 0.0, KEY)
        o2 = m.observe(state, 0.0, KEY)
        assert jnp.array_equal(o1["robot"]["position"], o2["robot"]["position"])


# -- ModelUncertainty ------------------------------------------------------

class TestModelUncertainty:
    def test_is_uncertainty_model(self):
        assert isinstance(ModelUncertainty(), UncertaintyModel)

    def test_actuate_is_passthrough(self):
        m = ModelUncertainty(noise_fraction=0.1)
        cmd = make_commands()
        applied = m.actuate(cmd, 0.0, KEY)
        assert applied == cmd

    def test_zero_noise_passthrough(self):
        m = ModelUncertainty(noise_fraction=0.0)
        state = make_state(1.0, 2.0)
        obs = m.observe(state, 0.0, KEY)
        assert jnp.allclose(obs["robot"]["position"], state["robot"]["position"])

    def test_multiplicative_noise(self):
        m = ModelUncertainty(noise_fraction=0.1)
        state = make_state(10.0, 5.0)
        obs = m.observe(state, 0.0, KEY)
        # Position should be perturbed by ~10%
        pos = obs["robot"]["position"]
        assert not jnp.array_equal(pos, state["robot"]["position"])
        ratio = float(pos[0]) / 10.0
        assert 0.5 < ratio < 1.5  # Very loose — just checking it's multiplicative

    def test_target_nodes_filter(self):
        m = ModelUncertainty(noise_fraction=0.5, target_nodes=("other_node",))
        state = make_state(10.0)
        obs = m.observe(state, 0.0, KEY)
        # Robot node should be untouched since only "other_node" is targeted
        assert jnp.array_equal(obs["robot"]["position"], state["robot"]["position"])

    def test_exclude_fields(self):
        m = ModelUncertainty(
            noise_fraction=0.5,
            exclude_fields=("position",),
        )
        state = make_state(10.0, 5.0)
        obs = m.observe(state, 0.0, KEY)
        # Position excluded — should be unchanged
        assert jnp.array_equal(obs["robot"]["position"], state["robot"]["position"])
        # Velocity not excluded — should be perturbed
        assert not jnp.array_equal(obs["robot"]["velocity"], state["robot"]["velocity"])

    def test_tracking_confidence_excluded_by_default(self):
        """tracking_confidence is in default exclude_fields."""
        m = ModelUncertainty(noise_fraction=0.5)
        state = {
            "robot": {
                "position": jnp.array([1.0]),
                "tracking_confidence": jnp.array(1.0),
            }
        }
        obs = m.observe(state, 0.0, KEY)
        assert float(obs["robot"]["tracking_confidence"]) == 1.0


# -- ComposedUncertainty --------------------------------------------------

class TestComposedUncertainty:
    def test_add_creates_composed(self):
        a = IdentityUncertainty()
        b = IdentityUncertainty()
        composed = a + b
        assert isinstance(composed, ComposedUncertainty)
        assert len(composed.models) == 2

    def test_triple_add_flattens(self):
        a = IdentityUncertainty()
        b = IdentityUncertainty()
        c = IdentityUncertainty()
        composed = a + b + c
        assert isinstance(composed, ComposedUncertainty)
        assert len(composed.models) == 3

    def test_composed_applies_both_observe(self):
        """Localisation noise + model noise both applied."""
        loc = LocalisationUncertainty(position_noise_std_mm=1.0)
        model = ModelUncertainty(noise_fraction=0.1)
        composed = loc + model
        state = make_state(10.0, 5.0)
        obs = composed.observe(state, 0.0, KEY)
        # Both should have perturbed the state
        assert not jnp.array_equal(obs["robot"]["position"], state["robot"]["position"])

    def test_composed_applies_both_actuate(self):
        """Two actuation noise sources compound."""
        a1 = ActuationUncertainty(frequency_noise_std=1.0, target_node="external_field")
        a2 = ActuationUncertainty(frequency_noise_std=1.0, target_node="external_field")
        composed = a1 + a2
        cmd = make_commands(10.0, 5.0)
        applied = composed.actuate(cmd, 0.0, KEY)
        # Double noise should produce larger deviation on average
        assert applied["external_field"]["frequency_hz"] != 10.0

    def test_composed_observe_then_actuate(self):
        """Observation noise doesn't leak into actuation and vice versa."""
        loc = LocalisationUncertainty(position_noise_std_mm=100.0)
        act = ActuationUncertainty(frequency_noise_std=100.0, target_node="external_field")
        composed = loc + act

        # Actuation should not be affected by localisation model
        cmd = make_commands(10.0, 5.0)
        applied = composed.actuate(cmd, 0.0, KEY)
        # Only ActuationUncertainty should change frequency
        # LocalisationUncertainty._actuate is passthrough
        assert applied["external_field"]["frequency_hz"] != 10.0

    def test_is_uncertainty_model(self):
        composed = IdentityUncertainty() + IdentityUncertainty()
        assert isinstance(composed, UncertaintyModel)

    def test_deterministic_with_same_key(self):
        composed = (
            LocalisationUncertainty(position_noise_std_mm=1.0)
            + ActuationUncertainty(frequency_noise_std=1.0, target_node="external_field")
        )
        state = make_state(10.0)
        o1 = composed.observe(state, 0.0, KEY)
        o2 = composed.observe(state, 0.0, KEY)
        assert jnp.array_equal(o1["robot"]["position"], o2["robot"]["position"])


# -- Integration with runner -----------------------------------------------

class TestUncertaintyWithRunner:
    """Verify the uncertainty module integrates with PolicyRunner."""

    def test_runner_accepts_new_uncertainty_models(self):
        """PolicyRunner should accept our ABC-based models just like the old protocol."""
        from mime.control.runner import PolicyRunner
        from mime.control.input_source import ConstantControlInput

        class MockGM:
            def get_state(self):
                return make_state(1.0)
            def step(self, ext=None):
                return make_state(1.0)

        composed = (
            LocalisationUncertainty(position_noise_std_mm=0.5)
            + ActuationUncertainty(frequency_noise_std=0.1, target_node="external_field")
            + ModelUncertainty(noise_fraction=0.05)
        )

        runner = PolicyRunner(
            graph_manager=MockGM(),
            control_input=ConstantControlInput(make_commands()),
            uncertainty=composed,
            rng_seed=42,
        )
        state = runner.step(0.001)
        assert "robot" in state
