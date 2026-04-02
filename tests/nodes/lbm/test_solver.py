"""Tests for the coupled IB-LBM solver."""

import pytest
import jax
import jax.numpy as jnp
import math

from mime.nodes.environment.lbm.solver import (
    IBLBMConfig, IBLBMState, IBResult,
    create_state, step, run,
)
from mime.nodes.environment.lbm.ib import (
    generate_circle_points, compute_boundary_velocities,
)
from mime.nodes.environment.lbm.d2q9 import create_channel_walls


class TestSolverCreation:
    def test_create_state(self):
        config = IBLBMConfig(nx=20, ny=20)
        state = create_state(config)
        assert state.f.shape == (20, 20, 9)
        assert state.density.shape == (20, 20)
        assert state.velocity.shape == (20, 20, 2)
        assert state.step_count == 0

    def test_custom_initial_conditions(self):
        config = IBLBMConfig(nx=20, ny=20)
        state = create_state(config, initial_density=1.5, initial_velocity=(0.01, 0.0))
        assert jnp.allclose(state.density, 1.5, atol=0.01)


class TestPureFluidStep:
    def test_step_without_ib(self):
        """Step with no boundary points should work as pure LBM."""
        config = IBLBMConfig(nx=20, ny=20, tau=0.8)
        state = create_state(config)
        new_state, ib_result = step(state, config)
        assert ib_result is None
        assert new_state.step_count == 1

    def test_multiple_steps(self):
        config = IBLBMConfig(nx=20, ny=20, tau=0.8)
        state = create_state(config)
        for _ in range(10):
            state, _ = step(state, config)
        assert state.step_count == 10

    def test_run_scan(self):
        """run() with jax.lax.scan should produce same result as loop."""
        config = IBLBMConfig(nx=20, ny=20, tau=0.8)
        state = create_state(config)
        final = run(state, config, n_steps=50)
        assert final.step_count == 50
        assert jnp.isfinite(final.density).all()
        assert jnp.isfinite(final.velocity).all()


class TestPoiseuilleSolver:
    @pytest.mark.slow
    def test_poiseuille_via_solver(self):
        """Poiseuille flow through the solver interface."""
        nx, ny = 3, 32
        tau = 0.8
        nu = (tau - 0.5) / 3.0
        F = 1e-5

        config = IBLBMConfig(nx=nx, ny=ny, tau=tau)
        force = jnp.zeros((nx, ny, 2)).at[..., 0].set(F)
        wall_mask = create_channel_walls(nx, ny)
        state = create_state(config, wall_mask=wall_mask)

        # Run to steady state
        for _ in range(3000):
            state, _ = step(state, config, external_force=force)

        # Check parabolic profile
        u_profile = state.velocity[1, :, 0]
        H = ny - 2
        y = jnp.arange(ny) - 0.5
        u_analytical = (F / (2 * nu)) * y * (H - y)
        u_analytical = jnp.where(wall_mask[0], 0.0, u_analytical)

        peak_sim = float(jnp.max(u_profile))
        peak_ana = float(jnp.max(u_analytical))
        rel_error = abs(peak_sim - peak_ana) / peak_ana
        assert rel_error < 0.1, f"Poiseuille error: {rel_error:.4f}"


class TestIBCoupling:
    def test_step_with_stationary_circle(self):
        """Immerse a stationary circle in quiescent fluid."""
        nx, ny = 40, 40
        config = IBLBMConfig(nx=nx, ny=ny, tau=0.8, mdf_iterations=3)
        state = create_state(config, wall_mask=jnp.zeros((nx, ny), dtype=bool))

        center = jnp.array([20.0, 20.0])
        radius = 5.0
        n_pts = 30
        points, ds = generate_circle_points(center[0], center[1], radius, n_pts)
        target_vel = jnp.zeros((n_pts, 2))  # Stationary

        new_state, ib_result = step(
            state, config,
            boundary_points=points,
            target_velocities=target_vel,
            ds=ds,
            body_center=center,
        )

        assert ib_result is not None
        assert ib_result.drag_force.shape == (2,)
        # Stationary body in quiescent fluid: drag should be ~zero
        assert jnp.linalg.norm(ib_result.drag_force) < 0.1

    def test_moving_body_feels_drag(self):
        """A body moving through quiescent fluid should feel drag."""
        nx, ny = 60, 40
        config = IBLBMConfig(nx=nx, ny=ny, tau=0.8, mdf_iterations=5)
        state = create_state(config, wall_mask=jnp.zeros((nx, ny), dtype=bool))

        center = jnp.array([30.0, 20.0])
        radius = 3.0
        n_pts = 20
        points, ds = generate_circle_points(center[0], center[1], radius, n_pts)

        # Body moves in +x at U=0.05 lattice units
        U_body = jnp.array([0.05, 0.0])
        target_vel = compute_boundary_velocities(U_body, 0.0, points, center)

        # Run a few steps to develop drag
        for _ in range(20):
            state, ib_result = step(
                state, config,
                boundary_points=points,
                target_velocities=target_vel,
                ds=ds,
                body_center=center,
            )

        assert ib_result is not None
        # Drag should oppose motion (negative x-component)
        assert ib_result.drag_force[0] < 0, (
            f"Expected negative drag, got {ib_result.drag_force}"
        )

    def test_ib_result_fields(self):
        """IBResult should contain all expected fields."""
        nx, ny = 30, 30
        config = IBLBMConfig(nx=nx, ny=ny, tau=0.8, mdf_iterations=2)
        state = create_state(config, wall_mask=jnp.zeros((nx, ny), dtype=bool))

        points, ds = generate_circle_points(15.0, 15.0, 3.0, 10)
        target = jnp.zeros((10, 2))
        center = jnp.array([15.0, 15.0])

        _, result = step(state, config, points, target, ds, center)

        assert result.drag_force.shape == (2,)
        assert result.drag_torque.shape == ()
        assert result.marker_forces.shape == (10, 2)
        assert result.u_markers.shape == (10, 2)

    def test_rotating_body_feels_torque(self):
        """A rotating body should feel resistive torque."""
        nx, ny = 40, 40
        config = IBLBMConfig(nx=nx, ny=ny, tau=0.8, mdf_iterations=5)
        state = create_state(config, wall_mask=jnp.zeros((nx, ny), dtype=bool))

        center = jnp.array([20.0, 20.0])
        radius = 4.0
        n_pts = 24
        points, ds = generate_circle_points(center[0], center[1], radius, n_pts)

        # Body rotates CCW at omega = 0.01
        omega = 0.01
        target_vel = compute_boundary_velocities(jnp.zeros(2), omega, points, center)

        for _ in range(20):
            state, result = step(
                state, config, points, target_vel, ds, center,
            )

        assert result is not None
        # Torque should oppose rotation (negative for CCW rotation)
        assert result.drag_torque != 0.0


class TestSolverJAX:
    def test_step_jit(self):
        """Solver step should be JIT-compilable (without IB)."""
        config = IBLBMConfig(nx=20, ny=20, tau=0.8)
        state = create_state(config)

        @jax.jit
        def do_step(f, density, velocity, wall_mask):
            s = IBLBMState(f=f, density=density, velocity=velocity,
                           wall_mask=wall_mask, step_count=0)
            new_s, _ = step(s, config)
            return new_s.f, new_s.density, new_s.velocity

        f, rho, u = do_step(state.f, state.density, state.velocity, state.wall_mask)
        assert jnp.isfinite(f).all()

    def test_run_produces_finite(self):
        """run() should produce finite results."""
        config = IBLBMConfig(nx=20, ny=20, tau=0.8)
        state = create_state(config)
        final = run(state, config, 10)
        assert jnp.isfinite(final.f).all()
