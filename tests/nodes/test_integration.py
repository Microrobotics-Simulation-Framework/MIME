"""Integration test: full magnetic actuation chain.

Tests the coupled system:
ExternalMagneticFieldNode -> MagneticResponseNode -> RigidBodyNode
                                                  -> PhaseTrackingNode

This validates that nodes can be wired together and produce physically
sensible results (robot rotates in response to rotating field).
"""

import pytest
import jax.numpy as jnp
import math

from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.robot.magnetic_response import MagneticResponseNode
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.nodes.robot.phase_tracking import PhaseTrackingNode
from mime.core.quaternion import identity_quat


def step_chain(field_node, mag_node, body_node, phase_node,
               field_state, mag_state, body_state, phase_state,
               field_bi, dt):
    """Step the full chain manually (without GraphManager)."""
    # 1. External field
    field_state = field_node.update(field_state, field_bi, dt)
    B = field_state["field_vector"]
    grad_B = field_state["field_gradient"]

    # 2. Magnetic response
    mag_bi = {
        "field_vector": B,
        "field_gradient": grad_B,
        "orientation": body_state["orientation"],
    }
    mag_state = mag_node.update(mag_state, mag_bi, dt)

    # 3. Rigid body
    body_bi = {
        "magnetic_force": mag_state["magnetic_force"],
        "magnetic_torque": mag_state["magnetic_torque"],
    }
    body_state = body_node.update(body_state, body_bi, dt)

    # 4. Phase tracking
    phase_bi = {
        "orientation": body_state["orientation"],
        "field_vector": B,
    }
    phase_state = phase_node.update(phase_state, phase_bi, dt)

    return field_state, mag_state, body_state, phase_state


class TestFullChainIntegration:
    def setup_method(self):
        self.dt = 0.0001  # 0.1 ms — fine enough for 10 Hz rotation
        self.field = ExternalMagneticFieldNode("field", self.dt)
        self.mag = MagneticResponseNode(
            "mag", self.dt,
            volume_m3=1e-15,
            n_axi=0.2,
            n_rad=0.4,
        )
        self.body = RigidBodyNode(
            "body", self.dt,
            semi_major_axis_m=100e-6,
            semi_minor_axis_m=50e-6,
            fluid_viscosity_pa_s=8.5e-4,
        )
        self.phase = PhaseTrackingNode("phase", self.dt)

    def run_steps(self, n_steps, freq=10.0, strength=10.0):
        fs = self.field.initial_state()
        ms = self.mag.initial_state()
        bs = self.body.initial_state()
        ps = self.phase.initial_state()
        field_bi = {"frequency_hz": freq, "field_strength_mt": strength}

        for _ in range(n_steps):
            fs, ms, bs, ps = step_chain(
                self.field, self.mag, self.body, self.phase,
                fs, ms, bs, ps, field_bi, self.dt,
            )
        return fs, ms, bs, ps

    @pytest.mark.slow
    def test_robot_rotates_with_field(self):
        """Robot should rotate when a rotating field is applied."""
        _, _, bs, _ = self.run_steps(1000, freq=10.0, strength=10.0)
        # After 1000 steps at 0.1ms = 0.1s, robot should have rotated
        assert not jnp.allclose(bs["orientation"], identity_quat())
        assert jnp.linalg.norm(bs["angular_velocity"]) > 0

    @pytest.mark.slow
    def test_phase_error_stays_small_below_stepout(self):
        """At low frequency, robot tracks the field — phase error stays small."""
        _, _, _, ps = self.run_steps(5000, freq=5.0, strength=10.0)
        # After 0.5s at 5 Hz, phase error should be small (synchronised)
        # Note: this depends on the balance of magnetic torque vs. viscous drag
        # With the default parameters, the robot should synchronise
        pe = float(ps["phase_error"])
        # Allow up to ~1 radian — depends on exact parameters
        assert pe < 1.5

    def test_torque_is_nonzero_during_rotation(self):
        """Magnetic torque should be non-zero when field and moment are misaligned."""
        fs = self.field.initial_state()
        ms = self.mag.initial_state()
        bs = self.body.initial_state()
        ps = self.phase.initial_state()
        field_bi = {"frequency_hz": 10.0, "field_strength_mt": 10.0}

        # After one step, field has rotated but body hasn't yet
        fs, ms, bs, ps = step_chain(
            self.field, self.mag, self.body, self.phase,
            fs, ms, bs, ps, field_bi, self.dt,
        )
        # Run a few more steps so misalignment develops
        for _ in range(10):
            fs, ms, bs, ps = step_chain(
                self.field, self.mag, self.body, self.phase,
                fs, ms, bs, ps, field_bi, self.dt,
            )
        assert jnp.linalg.norm(ms["magnetic_torque"]) > 0

    def test_no_field_no_motion(self):
        """With zero field strength, robot should not move."""
        _, _, bs, _ = self.run_steps(100, freq=10.0, strength=0.0)
        assert jnp.allclose(bs["position"], jnp.zeros(3))
        assert jnp.allclose(bs["orientation"], identity_quat())

    @pytest.mark.slow
    def test_all_states_finite(self):
        """After many steps, all state values should be finite (no NaN/Inf)."""
        _, ms, bs, ps = self.run_steps(1000, freq=20.0, strength=15.0)
        for state_dict in [ms, bs, ps]:
            for key, val in state_dict.items():
                if hasattr(val, 'shape'):
                    assert jnp.isfinite(val).all(), f"Non-finite in {key}: {val}"
