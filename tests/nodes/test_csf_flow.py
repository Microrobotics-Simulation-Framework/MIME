"""Tests for CSFFlowNode."""

import pytest
import jax
import jax.numpy as jnp
import math

from mime.nodes.environment.csf_flow import CSFFlowNode


class TestCSFFlowBasic:
    def test_initial_state(self):
        node = CSFFlowNode("csf", 0.001)
        state = node.initial_state()
        assert jnp.allclose(state["drag_force"], jnp.zeros(3))
        assert jnp.allclose(state["drag_torque"], jnp.zeros(3))

    def test_stationary_robot_quiescent_no_drag(self):
        """Stationary robot in quiescent fluid: zero drag."""
        node = CSFFlowNode("csf", 0.001, pulsatile=False)
        state = node.initial_state()
        bi = {
            "position": jnp.zeros(3),
            "velocity": jnp.zeros(3),
            "angular_velocity": jnp.zeros(3),
        }
        new_state = node.update(state, bi, 0.001)
        assert jnp.allclose(new_state["drag_force"], jnp.zeros(3))
        assert jnp.allclose(new_state["drag_torque"], jnp.zeros(3))

    def test_moving_robot_gets_drag(self):
        """Moving robot in quiescent fluid: F = -6*pi*mu*a*V."""
        a = 100e-6
        mu = 8.5e-4
        node = CSFFlowNode("csf", 0.001, robot_radius_m=a,
                           fluid_viscosity_pa_s=mu, pulsatile=False)
        state = node.initial_state()
        V = jnp.array([1e-3, 0.0, 0.0])  # 1 mm/s
        bi = {"position": jnp.zeros(3), "velocity": V, "angular_velocity": jnp.zeros(3)}
        new_state = node.update(state, bi, 0.001)

        # Expected: F = -6*pi*mu*a*V
        F_expected = -6 * math.pi * mu * a * 1e-3
        assert jnp.abs(new_state["drag_force"][0] - F_expected) / abs(F_expected) < 0.01

    def test_drag_opposes_motion(self):
        node = CSFFlowNode("csf", 0.001, pulsatile=False)
        state = node.initial_state()
        V = jnp.array([1e-3, 0.0, 0.0])
        bi = {"position": jnp.zeros(3), "velocity": V, "angular_velocity": jnp.zeros(3)}
        new_state = node.update(state, bi, 0.001)
        # Drag should oppose velocity
        assert new_state["drag_force"][0] < 0

    def test_rotational_drag(self):
        """Rotating robot gets rotational drag: T = -8*pi*mu*a^3*omega."""
        a = 100e-6
        mu = 8.5e-4
        node = CSFFlowNode("csf", 0.001, robot_radius_m=a,
                           fluid_viscosity_pa_s=mu, pulsatile=False)
        state = node.initial_state()
        omega = jnp.array([0.0, 0.0, 10.0])  # 10 rad/s
        bi = {"position": jnp.zeros(3), "velocity": jnp.zeros(3),
              "angular_velocity": omega}
        new_state = node.update(state, bi, 0.001)

        T_expected = -8 * math.pi * mu * a**3 * 10.0
        assert jnp.abs(new_state["drag_torque"][2] - T_expected) / abs(T_expected) < 0.01

    def test_drag_scales_linearly_with_velocity(self):
        node = CSFFlowNode("csf", 0.001, pulsatile=False)
        state = node.initial_state()

        V1 = jnp.array([1e-3, 0.0, 0.0])
        V2 = jnp.array([2e-3, 0.0, 0.0])
        bi1 = {"position": jnp.zeros(3), "velocity": V1, "angular_velocity": jnp.zeros(3)}
        bi2 = {"position": jnp.zeros(3), "velocity": V2, "angular_velocity": jnp.zeros(3)}

        s1 = node.update(state, bi1, 0.001)
        s2 = node.update(state, bi2, 0.001)

        # Drag should double when velocity doubles
        ratio = float(s2["drag_force"][0] / s1["drag_force"][0])
        assert abs(ratio - 2.0) < 0.01


class TestCSFFlowPulsatile:
    def test_pulsatile_nonzero_background(self):
        node = CSFFlowNode("csf", 0.001, pulsatile=True,
                           peak_velocity_m_s=0.04, cardiac_freq_hz=1.1)
        state = node.initial_state()
        bi = {"position": jnp.zeros(3), "velocity": jnp.zeros(3),
              "angular_velocity": jnp.zeros(3)}
        # Step to a time when sin(omega*t) != 0
        for _ in range(250):  # ~0.25s, well into a cardiac cycle
            state = node.update(state, bi, 0.001)
        # Background velocity should be non-zero at some point
        assert jnp.linalg.norm(state["background_velocity"]) > 0

    def test_stationary_robot_in_flow_gets_drag(self):
        """Stationary robot in flowing CSF should experience drag."""
        node = CSFFlowNode("csf", 0.001, pulsatile=True,
                           peak_velocity_m_s=0.04, cardiac_freq_hz=1.1)
        state = node.initial_state()
        bi = {"position": jnp.zeros(3), "velocity": jnp.zeros(3),
              "angular_velocity": jnp.zeros(3)}
        # Advance to peak flow
        for _ in range(250):
            state = node.update(state, bi, 0.001)
        # Drag should be non-zero (flow pushes stationary robot)
        assert jnp.linalg.norm(state["drag_force"]) > 0


class TestCSFFlowMetadata:
    def test_meta_set(self):
        assert CSFFlowNode.meta is not None
        assert CSFFlowNode.meta.algorithm_id == "MIME-NODE-004"

    def test_mime_meta_set(self):
        assert CSFFlowNode.mime_meta is not None
        assert CSFFlowNode.mime_meta.role.value == "environment"
        assert len(CSFFlowNode.mime_meta.anatomical_regimes) > 0

    def test_validate_consistency(self):
        node = CSFFlowNode("csf", 0.001)
        errors = node.validate_mime_consistency()
        assert errors == [], f"Consistency errors: {errors}"


class TestCSFFlowJAX:
    def test_jit_traceable(self):
        node = CSFFlowNode("csf", 0.001)
        state = node.initial_state()
        bi = {"position": jnp.zeros(3), "velocity": jnp.array([1e-3, 0., 0.]),
              "angular_velocity": jnp.zeros(3)}
        jitted = jax.jit(node.update)
        new_state = jitted(state, bi, 0.001)
        assert jnp.isfinite(new_state["drag_force"]).all()

    def test_grad_drag_wrt_velocity(self):
        node = CSFFlowNode("csf", 0.001, pulsatile=False)
        state = node.initial_state()

        def drag_x(vx):
            bi = {"position": jnp.zeros(3),
                  "velocity": jnp.array([vx, 0.0, 0.0]),
                  "angular_velocity": jnp.zeros(3)}
            s = node.update(state, bi, 0.001)
            return s["drag_force"][0]

        g = jax.grad(drag_x)(jnp.array(1e-3))
        # dF/dV = -6*pi*mu*a
        a = 100e-6
        mu = 8.5e-4
        expected = -6 * jnp.pi * mu * a
        assert jnp.abs(g - expected) / abs(expected) < 0.01
