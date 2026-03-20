"""Tests for D2Q9 LBM core — lattice constants, equilibrium, collision, streaming."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import math

from mime.nodes.environment.lbm.d2q9 import (
    E, W, OPP, CS2, Q,
    equilibrium, compute_macroscopic, stream, collide_bgk,
    guo_forcing, lbm_step, init_equilibrium,
    bounce_back_mask, create_channel_walls, create_cylinder_mask,
    tau_from_viscosity, physical_to_lattice_force,
)


class TestLatticeConstants:
    def test_velocities_shape(self):
        assert E.shape == (9, 2)

    def test_weights_sum_to_one(self):
        assert abs(W.sum() - 1.0) < 1e-6

    def test_weights_shape(self):
        assert W.shape == (9,)

    def test_opposite_involution(self):
        """OPP[OPP[i]] == i for all i."""
        for i in range(9):
            assert OPP[OPP[i]] == i

    def test_opposite_reverses_velocity(self):
        """e[OPP[i]] == -e[i] for i > 0."""
        for i in range(1, 9):
            assert np.array_equal(E[OPP[i]], -E[i])

    def test_rest_velocity_zero(self):
        assert np.array_equal(E[0], [0, 0])

    def test_momentum_conservation_symmetry(self):
        """Sum of e_i * w_i == 0 (no net lattice momentum)."""
        momentum = (W[:, None] * E).sum(axis=0)
        assert np.allclose(momentum, 0.0, atol=1e-10)


class TestEquilibrium:
    def test_shape(self):
        rho = jnp.ones((10, 20))
        u = jnp.zeros((10, 20, 2))
        f_eq = equilibrium(rho, u)
        assert f_eq.shape == (10, 20, 9)

    def test_at_rest_recovers_weights(self):
        """f_eq at zero velocity should be w_i * rho."""
        rho = jnp.ones((5, 5)) * 2.0
        u = jnp.zeros((5, 5, 2))
        f_eq = equilibrium(rho, u)
        for q in range(9):
            expected = W[q] * 2.0
            assert jnp.allclose(f_eq[..., q], expected, atol=1e-6)

    def test_density_conservation(self):
        """Sum of f_eq over directions == rho."""
        rho = jnp.ones((10, 10)) * 1.5
        u = jnp.zeros((10, 10, 2)).at[..., 0].set(0.1)
        f_eq = equilibrium(rho, u)
        rho_recovered = jnp.sum(f_eq, axis=-1)
        assert jnp.allclose(rho_recovered, rho, atol=1e-5)

    def test_momentum_conservation(self):
        """Sum of f_eq * e_i == rho * u."""
        rho = jnp.ones((10, 10)) * 1.0
        u = jnp.zeros((10, 10, 2)).at[..., 0].set(0.05)
        f_eq = equilibrium(rho, u)
        momentum = f_eq @ jnp.array(E, dtype=jnp.float32)
        expected = rho[..., None] * u
        assert jnp.allclose(momentum, expected, atol=1e-5)

    def test_non_negative_at_low_velocity(self):
        """f_eq should be non-negative for |u| << cs."""
        rho = jnp.ones((5, 5))
        u = jnp.zeros((5, 5, 2)).at[..., 0].set(0.01)
        f_eq = equilibrium(rho, u)
        assert jnp.all(f_eq >= 0)


class TestMacroscopic:
    def test_roundtrip_from_equilibrium(self):
        """compute_macroscopic(equilibrium(rho, u)) recovers rho and u."""
        rho = jnp.ones((10, 10)) * 1.2
        u = jnp.zeros((10, 10, 2)).at[..., 0].set(0.03)
        f = equilibrium(rho, u)
        rho2, u2 = compute_macroscopic(f)
        assert jnp.allclose(rho2, rho, atol=1e-5)
        assert jnp.allclose(u2, u, atol=1e-5)


class TestStreaming:
    def test_shape_preserved(self):
        f = jnp.ones((10, 10, 9))
        f_s = stream(f)
        assert f_s.shape == (10, 10, 9)

    def test_rest_unchanged(self):
        """Rest population (q=0) should not move."""
        f = jnp.zeros((10, 10, 9)).at[5, 5, 0].set(1.0)
        f_s = stream(f)
        assert f_s[5, 5, 0] == 1.0

    def test_positive_x_shifts_right(self):
        """q=1 (+x) should shift the distribution one cell in +x."""
        f = jnp.zeros((10, 10, 9)).at[3, 5, 1].set(1.0)
        f_s = stream(f)
        assert f_s[4, 5, 1] == 1.0
        assert f_s[3, 5, 1] == 0.0

    def test_diagonal_shifts_correctly(self):
        """q=5 (+x,+y) should shift diagonally."""
        f = jnp.zeros((10, 10, 9)).at[3, 3, 5].set(1.0)
        f_s = stream(f)
        assert f_s[4, 4, 5] == 1.0

    def test_periodic_boundary(self):
        """Streaming wraps around (periodic BC via jnp.roll)."""
        f = jnp.zeros((10, 10, 9)).at[9, 5, 1].set(1.0)
        f_s = stream(f)
        assert f_s[0, 5, 1] == 1.0  # Wrapped from x=9 to x=0

    def test_mass_conservation(self):
        """Total mass should be conserved by streaming."""
        key = jax.random.PRNGKey(42)
        f = jax.random.uniform(key, (20, 20, 9)) * 0.1 + W[None, None, :]
        mass_before = jnp.sum(f)
        f_s = stream(f)
        mass_after = jnp.sum(f_s)
        assert jnp.abs(mass_before - mass_after) / mass_before < 1e-5  # relative


class TestCollision:
    def test_equilibrium_is_fixed_point(self):
        """Colliding an equilibrium distribution should return itself."""
        rho = jnp.ones((5, 5))
        u = jnp.zeros((5, 5, 2))
        f_eq = equilibrium(rho, u)
        f_out = collide_bgk(f_eq, rho, u, tau=1.0)
        assert jnp.allclose(f_out, f_eq, atol=1e-6)

    def test_relaxation_toward_equilibrium(self):
        """After collision, f should be closer to f_eq."""
        rho = jnp.ones((5, 5))
        u = jnp.zeros((5, 5, 2))
        f_eq = equilibrium(rho, u)
        # Perturb away from equilibrium
        f = f_eq + 0.01 * jax.random.normal(jax.random.PRNGKey(0), f_eq.shape)
        f_out = collide_bgk(f, rho, u, tau=1.0)
        dist_before = jnp.sum((f - f_eq) ** 2)
        dist_after = jnp.sum((f_out - f_eq) ** 2)
        assert dist_after < dist_before


class TestBounceBack:
    def test_wall_reflects_directions(self):
        """At wall nodes, directions should be swapped with opposites."""
        f = jnp.zeros((5, 5, 9))
        # Put mass in direction 1 (+x) at wall node (0, 2)
        f = f.at[0, 2, 1].set(1.0)
        wall = jnp.zeros((5, 5), dtype=bool).at[0, 2].set(True)
        f_b = bounce_back_mask(f, wall)
        # Should now be in direction 2 (-x) at (0, 2)
        assert f_b[0, 2, 2] == 1.0
        assert f_b[0, 2, 1] == 0.0

    def test_non_wall_unchanged(self):
        f = jnp.ones((5, 5, 9))
        wall = jnp.zeros((5, 5), dtype=bool)
        f_b = bounce_back_mask(f, wall)
        assert jnp.allclose(f_b, f)


class TestChannelWalls:
    def test_shape(self):
        mask = create_channel_walls(100, 50)
        assert mask.shape == (100, 50)

    def test_walls_at_boundaries(self):
        mask = create_channel_walls(20, 10)
        assert mask[5, 0]    # Bottom wall
        assert mask[5, 9]    # Top wall
        assert not mask[5, 5]  # Interior


class TestLBMStep:
    def test_returns_correct_shapes(self):
        f = init_equilibrium(20, 10)
        f_new, rho, u = lbm_step(f, tau=0.8)
        assert f_new.shape == (20, 10, 9)
        assert rho.shape == (20, 10)
        assert u.shape == (20, 10, 2)

    def test_mass_conservation_no_walls(self):
        """Without walls, total mass should be conserved."""
        f = init_equilibrium(20, 20, density=1.5)
        mass_0 = jnp.sum(f)
        for _ in range(10):
            f, _, _ = lbm_step(f, tau=0.8, wall_mask=None)
        mass_f = jnp.sum(f)
        assert jnp.abs(mass_0 - mass_f) / mass_0 < 1e-5

    def test_quiescent_stays_quiescent(self):
        """Zero-velocity fluid should remain at rest."""
        f = init_equilibrium(20, 20)
        for _ in range(50):
            f, rho, u = lbm_step(f, tau=0.8, wall_mask=None)
        assert jnp.max(jnp.abs(u)) < 1e-6


class TestPoiseuille:
    """Poiseuille flow validation — the gold-standard LBM test."""

    def test_poiseuille_profile(self):
        """Pressure-driven channel flow should develop a parabolic profile.

        Analytical solution: u(y) = (F/(2*nu)) * y * (H - y)
        where F is the body force, nu = (tau-0.5)/3, H = ny - 2.
        """
        nx, ny = 3, 42   # Short channel, 40 fluid cells
        tau = 0.8
        nu = (tau - 0.5) / 3.0
        F_body = 1e-5  # Small force to stay in linear regime

        # Force field: uniform in x-direction
        force = jnp.zeros((nx, ny, 2)).at[..., 0].set(F_body)
        wall_mask = create_channel_walls(nx, ny)

        # Initialise and run to steady state
        f = init_equilibrium(nx, ny)
        for _ in range(5000):
            f, rho, u = lbm_step(f, tau, wall_mask=wall_mask, force=force)

        # Extract centreline velocity profile (x-component, middle of channel)
        u_profile = u[nx // 2, :, 0]

        # Analytical Poiseuille: u(y) = F/(2*nu) * y * (H-y)
        # y ranges from 0.5 (first fluid cell above wall) to H-0.5
        H = ny - 2  # channel width in lattice units
        y_fluid = jnp.arange(ny) - 0.5  # distance from bottom wall
        u_analytical = (F_body / (2.0 * nu)) * y_fluid * (H - y_fluid)
        # Zero at walls
        u_analytical = jnp.where(wall_mask[0], 0.0, u_analytical)

        # Compare in the interior (skip wall cells)
        interior = ~wall_mask[0]
        u_sim = u_profile[interior]
        u_ana = u_analytical[interior]

        # Relative error at channel centre (peak velocity)
        peak_sim = float(jnp.max(u_sim))
        peak_ana = float(jnp.max(u_ana))
        rel_error = abs(peak_sim - peak_ana) / peak_ana

        assert rel_error < 0.05, (
            f"Poiseuille peak velocity error: {rel_error:.4f} "
            f"(sim={peak_sim:.6e}, analytical={peak_ana:.6e})"
        )


class TestUnitConversion:
    def test_tau_from_viscosity(self):
        """CSF: nu=7e-7 m^2/s, dx=10um, dt computed."""
        dx = 10e-6
        nu = 7e-7
        # Choose dt such that tau is reasonable (0.5 < tau < 2)
        dt = 0.5 * dx**2 / nu  # nu_lattice = 0.5, tau = 0.5/cs2 + 0.5 = 2.0
        tau = tau_from_viscosity(nu, dx, dt)
        assert 0.5 < tau < 5.0
        # Check: nu_lattice = (tau - 0.5) * cs2
        nu_lattice = (tau - 0.5) * CS2
        nu_recovered = nu_lattice * dx**2 / dt
        assert abs(nu_recovered - nu) / nu < 1e-10


class TestJAX:
    def test_lbm_step_jit(self):
        """lbm_step should be JIT-compilable."""
        f = init_equilibrium(10, 10)
        jitted = jax.jit(lbm_step, static_argnums=(1,))
        f_new, rho, u = jitted(f, 0.8)
        assert jnp.isfinite(f_new).all()

    def test_equilibrium_grad(self):
        """Equilibrium should be differentiable w.r.t. velocity."""
        rho = jnp.ones((5, 5))

        def sum_feq(ux):
            u = jnp.zeros((5, 5, 2)).at[..., 0].set(ux)
            return jnp.sum(equilibrium(rho, u))

        g = jax.grad(sum_feq)(0.01)
        assert jnp.isfinite(g)
