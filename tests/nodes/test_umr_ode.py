"""Tests for UMR ODE model (pure JAX, not a MimeNode)."""

import math

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.robot.umr_ode import (
    umr_ode_rhs,
    umr_euler_integrate,
    umr_speed_curve,
    compute_step_out_frequency,
    sweep_frequency,
    pack_params,
    unpack_params,
    params_dict_to_array,
    params_array_to_dict,
)


# ---------------------------------------------------------------------------
# Default test parameters
# ---------------------------------------------------------------------------

def _default_params_dict():
    # C_rot chosen so step-out freq = n*m*B/(2*pi*C_rot) >> 128 Hz
    # i.e. C_rot = n*m*B / (2*pi*f_step) with f_step=128 gives C_rot ~ 4e-9
    # We use omega_field = 50 Hz (well below step-out) for stable tests
    n_mag = 1.0
    m_single = 1.07e-3
    B = 3e-3
    f_step = 128.0
    C_rot = n_mag * m_single * B / (2.0 * math.pi * f_step)
    return {
        "omega_field": 2.0 * math.pi * 50.0,  # 50 Hz, well below step-out
        "n_mag": n_mag,
        "m_single": m_single,
        "B": B,
        "I_eff": 1e-10,
        "m_eff": 1e-5,
        "C_rot": C_rot,
        "C_prop": 1e-4,
        "C_trans": 1e-3,
    }


def _default_params():
    d = _default_params_dict()
    return params_dict_to_array(d)


# ---------------------------------------------------------------------------
# ODE RHS tests
# ---------------------------------------------------------------------------

class TestOdeRhs:
    def test_rhs_shape(self):
        state = jnp.zeros(3)
        params = _default_params()
        d = umr_ode_rhs(state, params)
        assert d.shape == (3,)

    def test_rhs_at_zero_state(self):
        """At state = [0, 0, 0], dtheta/dt = omega_field, dOmega/dt = 0, dU/dt = 0."""
        state = jnp.zeros(3)
        params = _default_params()
        d = umr_ode_rhs(state, params)
        p = _default_params_dict()
        # dtheta/dt = omega_field - 0 = omega_field
        assert jnp.allclose(d[0], p["omega_field"])
        # dOmega/dt = (n*m*B*sin(0) - C_rot*0) / I_eff = 0
        assert jnp.allclose(d[1], 0.0)
        # dU/dt = (C_prop*0 - C_trans*0) / m_eff = 0
        assert jnp.allclose(d[2], 0.0)

    def test_steady_state_equilibrium(self):
        """At equilibrium, d/dt should be approximately zero.

        At steady state (below step-out):
          - theta = const => dtheta/dt = 0 => Omega = omega_field
          - dOmega/dt = 0 => n*m*B*sin(theta_eq) = C_rot*omega_field
          - dU/dt = 0 => U_eq = (C_prop/C_trans)*omega_field
        """
        p = _default_params_dict()
        omega = p["omega_field"]
        n_m_B = p["n_mag"] * p["m_single"] * p["B"]
        C_rot = p["C_rot"]

        # Check we're below step-out (sin(theta) <= 1)
        sin_theta_eq = C_rot * omega / n_m_B
        if abs(sin_theta_eq) > 1.0:
            pytest.skip("Default params are above step-out for this test")

        theta_eq = jnp.arcsin(sin_theta_eq)
        Omega_eq = omega
        U_eq = (p["C_prop"] / p["C_trans"]) * omega

        state_eq = jnp.array([theta_eq, Omega_eq, U_eq])
        params = _default_params()
        d = umr_ode_rhs(state_eq, params)

        assert jnp.allclose(d[0], 0.0, atol=1e-6), f"dtheta/dt = {d[0]}"
        assert jnp.allclose(d[1], 0.0, atol=1e2), f"dOmega/dt = {d[1]}"
        assert jnp.allclose(d[2], 0.0, atol=1e2), f"dU/dt = {d[2]}"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_integrate_shape(self):
        params = _default_params()
        n_steps = 100
        history = umr_euler_integrate(params, 1e-5, n_steps)
        assert history.shape == (n_steps + 1, 3)

    def test_speed_curve_shape(self):
        params = _default_params()
        t, U = umr_speed_curve(params, 1e-5, 0.001)
        assert t.shape == U.shape
        assert len(t) == 101  # 0.001/1e-5 = 100 steps + 1

    def test_speed_curve_starts_at_zero(self):
        params = _default_params()
        t, U = umr_speed_curve(params, 1e-5, 0.001)
        assert jnp.allclose(U[0], 0.0)

    def test_speed_monotonically_increases_initially(self):
        """Speed should generally increase from zero (before any oscillation)."""
        params = _default_params()
        t, U = umr_speed_curve(params, 1e-6, 0.001)
        # Check that U at end > U at start
        assert U[-1] > U[0]


# ---------------------------------------------------------------------------
# Step-out frequency
# ---------------------------------------------------------------------------

class TestStepOut:
    def test_step_out_formula(self):
        n_mag = 1.0
        m = 1.07e-3
        B = 3e-3
        C_rot = 1e-6
        f_step = compute_step_out_frequency(n_mag, m, B, C_rot)
        expected = n_mag * m * B / (2.0 * math.pi * C_rot)
        assert abs(f_step - expected) / expected < 1e-10

    def test_step_out_scales_with_n_mag(self):
        m = 1.07e-3
        B = 3e-3
        C_rot = 1e-6
        f1 = compute_step_out_frequency(1, m, B, C_rot)
        f2 = compute_step_out_frequency(2, m, B, C_rot)
        assert abs(f2 - 2 * f1) / f1 < 1e-10


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------

class TestParamPacking:
    def test_roundtrip(self):
        d = _default_params_dict()
        arr = params_dict_to_array(d)
        d2 = params_array_to_dict(arr)
        for k in d:
            # Allow float32 precision loss (rtol ~1e-6)
            assert abs(d[k] - d2[k]) < max(abs(d[k]) * 1e-5, 1e-10), (
                f"Mismatch for {k}: {d[k]} vs {d2[k]}"
            )

    def test_pack_unpack(self):
        arr = pack_params(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        d = unpack_params(arr)
        assert d["omega_field"] == 1.0
        assert d["C_trans"] == 9.0


# ---------------------------------------------------------------------------
# JAX traceability
# ---------------------------------------------------------------------------

class TestJAXTraceability:
    def test_grad_through_integrate(self):
        """jax.grad should work through umr_euler_integrate."""
        def final_speed(omega_field):
            p = pack_params(
                omega_field, 1.0, 1.07e-3, 3e-3,
                1e-10, 1e-5, 1e-6, 1e-4, 1e-3,
            )
            history = umr_euler_integrate(p, 1e-5, 100)
            return history[-1, 2]

        g = jax.grad(final_speed)(jnp.array(100.0))
        assert jnp.isfinite(g)

    def test_vmap_over_params(self):
        """jax.vmap should work over batched parameter arrays."""
        omegas = jnp.array([100.0, 200.0, 400.0])

        def final_speed(omega):
            p = pack_params(
                omega, 1.0, 1.07e-3, 3e-3,
                1e-10, 1e-5, 1e-6, 1e-4, 1e-3,
            )
            history = umr_euler_integrate(p, 1e-5, 100)
            return history[-1, 2]

        speeds = jax.vmap(final_speed)(omegas)
        assert speeds.shape == (3,)
        assert jnp.all(jnp.isfinite(speeds))
