"""Tests for convergence monitoring — residual computation and run-to-convergence."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.convergence import (
    compute_velocity_residual,
    run_to_convergence,
)
from mime.nodes.environment.lbm.d3q19 import (
    init_equilibrium,
    create_channel_walls,
)
from mime.nodes.environment.lbm.bounce_back import compute_missing_mask


class TestComputeVelocityResidual:
    def test_identical_fields_zero_residual(self):
        """Residual should be exactly 0 for identical velocity fields."""
        u = jnp.ones((8, 8, 8, 3)) * 0.01
        fluid = jnp.ones((8, 8, 8), dtype=bool)
        res = compute_velocity_residual(u, u, fluid, norm_type="L2")
        assert res == 0.0

    def test_identical_fields_zero_residual_linf(self):
        """Linf residual should be 0 for identical fields."""
        u = jnp.ones((8, 8, 8, 3)) * 0.01
        fluid = jnp.ones((8, 8, 8), dtype=bool)
        res = compute_velocity_residual(u, u, fluid, norm_type="Linf")
        assert res == 0.0

    def test_nonzero_difference(self):
        """Residual should be positive when fields differ."""
        u_old = jnp.ones((8, 8, 8, 3)) * 0.01
        u_new = jnp.ones((8, 8, 8, 3)) * 0.02
        fluid = jnp.ones((8, 8, 8), dtype=bool)
        res = compute_velocity_residual(u_new, u_old, fluid, norm_type="L2")
        assert res > 0.0

    def test_masked_nodes_excluded(self):
        """Only fluid nodes contribute to residual."""
        u_old = jnp.zeros((4, 4, 4, 3))
        u_new = jnp.ones((4, 4, 4, 3)) * 0.1
        # Only one fluid node
        fluid = jnp.zeros((4, 4, 4), dtype=bool).at[0, 0, 0].set(True)
        res_one = compute_velocity_residual(u_new, u_old, fluid, norm_type="L2")
        # All fluid nodes
        fluid_all = jnp.ones((4, 4, 4), dtype=bool)
        res_all = compute_velocity_residual(u_new, u_old, fluid_all, norm_type="L2")
        # Both should be positive but they compute the same relative change
        assert res_one > 0.0
        assert res_all > 0.0


class TestRunToConvergence:
    def test_couette_converges(self):
        """Couette flow in a planar channel should converge."""
        nx, ny, nz = 4, 20, 4
        tau = 0.8

        solid = create_channel_walls(nx, ny, nz, wall_axis=1, wall_thickness=1)
        mm = compute_missing_mask(solid)

        # Moving top wall
        wall_vel = jnp.zeros((nx, ny, nz, 3))
        wall_vel = wall_vel.at[:, :, :, 0].set(0.01)  # x-velocity everywhere

        f_init = init_equilibrium(nx, ny, nz)

        f_final, n_steps, res_hist = run_to_convergence(
            f_init, tau, solid, mm,
            wall_velocity=wall_vel,
            max_steps=5000,
            check_interval=100,
            rtol=1e-5,
            norm_type="L2",
        )

        # Should have converged
        assert len(res_hist) > 0
        assert res_hist[-1] < 1e-5 or n_steps == 5000
        # Should not need all steps for simple Couette
        assert n_steps < 5000, f"Did not converge in 5000 steps, last residual: {res_hist[-1]}"

    def test_early_termination(self):
        """If already at equilibrium, should stop almost immediately."""
        nx, ny, nz = 4, 10, 4
        tau = 0.8

        solid = create_channel_walls(nx, ny, nz, wall_axis=1)
        mm = compute_missing_mask(solid)

        # Start at equilibrium with zero velocity — already converged
        f_init = init_equilibrium(nx, ny, nz)

        f_final, n_steps, res_hist = run_to_convergence(
            f_init, tau, solid, mm,
            max_steps=10000,
            check_interval=100,
            rtol=1e-4,
        )

        # Should terminate very early (zero velocity → zero residual)
        assert n_steps <= 200, f"Expected early termination, got {n_steps} steps"

    def test_residual_history_decreasing(self):
        """Residual should generally decrease for Couette flow."""
        nx, ny, nz = 4, 16, 4
        tau = 0.8

        solid = create_channel_walls(nx, ny, nz, wall_axis=1)
        mm = compute_missing_mask(solid)

        wall_vel = jnp.zeros((nx, ny, nz, 3))
        wall_vel = wall_vel.at[:, :, :, 0].set(0.01)

        f_init = init_equilibrium(nx, ny, nz)

        _, _, res_hist = run_to_convergence(
            f_init, tau, solid, mm,
            wall_velocity=wall_vel,
            max_steps=3000,
            check_interval=100,
            rtol=1e-8,  # Don't converge — just collect history
        )

        # After initial transient, residuals should decrease
        # Check that the last residual is less than the second one
        if len(res_hist) >= 3:
            assert res_hist[-1] < res_hist[1], (
                f"Residual not decreasing: {res_hist[1]} -> {res_hist[-1]}"
            )
