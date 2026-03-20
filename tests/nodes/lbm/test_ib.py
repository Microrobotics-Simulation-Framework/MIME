"""Tests for the Immersed Boundary module — kernels, interpolation, MDF."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import math

from mime.nodes.environment.lbm.ib import (
    peskin_phi_4, peskin_phi_3,
    compute_kernels,
    interpolate_velocity, spread_force,
    multi_direct_forcing,
    compute_drag_force, compute_drag_torque,
    generate_circle_points, compute_boundary_velocities,
)


class TestPeskinKernel:
    def test_phi4_at_zero(self):
        """phi(0) = 0.25 * (1 + cos(0)) = 0.5."""
        assert peskin_phi_4(jnp.array(0.0)) == pytest.approx(0.5)

    def test_phi4_at_boundary(self):
        """phi(2) = 0.25 * (1 + cos(pi)) = 0."""
        assert peskin_phi_4(jnp.array(2.0)) == pytest.approx(0.0, abs=1e-6)

    def test_phi4_outside_support(self):
        """phi(r) = 0 for |r| > 2."""
        assert peskin_phi_4(jnp.array(3.0)) == 0.0
        assert peskin_phi_4(jnp.array(-3.0)) == 0.0

    def test_phi4_symmetric(self):
        """phi(-r) = phi(r)."""
        r = jnp.linspace(-3, 3, 100)
        assert jnp.allclose(peskin_phi_4(r), peskin_phi_4(-r))

    def test_phi4_integrates_to_one(self):
        """Integral of phi over [-2, 2] should be 1."""
        r = jnp.linspace(-2, 2, 10000)
        dr = r[1] - r[0]
        integral = jnp.sum(peskin_phi_4(r)) * dr
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_phi4_non_negative(self):
        r = jnp.linspace(-3, 3, 1000)
        assert jnp.all(peskin_phi_4(r) >= 0)

    def test_phi3_at_zero(self):
        val = peskin_phi_3(jnp.array(0.0))
        expected = (1.0 / 3.0) * (1.0 + math.sqrt(1.0))  # = 2/3
        assert val == pytest.approx(expected, abs=1e-5)

    def test_phi3_outside_support(self):
        assert peskin_phi_3(jnp.array(2.0)) == 0.0


class TestKernelComputation:
    def test_shape(self):
        points = jnp.array([[5.0, 5.0], [10.0, 10.0]])
        kernels = compute_kernels(points, 20, 20)
        assert kernels.shape == (2, 20, 20)

    def test_non_negative(self):
        points = jnp.array([[5.0, 5.0]])
        kernels = compute_kernels(points, 20, 20)
        assert jnp.all(kernels >= 0)

    def test_localised_support(self):
        """Kernel should be zero far from the point."""
        points = jnp.array([[10.0, 10.0]])
        kernels = compute_kernels(points, 30, 30)
        # Far corner should be zero
        assert kernels[0, 0, 0] == 0.0
        assert kernels[0, 29, 29] == 0.0

    def test_peak_at_point(self):
        """Kernel should have its peak near the point location."""
        points = jnp.array([[10.0, 10.0]])
        kernels = compute_kernels(points, 20, 20)
        peak_idx = jnp.unravel_index(jnp.argmax(kernels[0]), (20, 20))
        assert peak_idx[0] == 10
        assert peak_idx[1] == 10


class TestInterpolation:
    def test_uniform_velocity(self):
        """Interpolation of a uniform field should return that field."""
        nx, ny = 20, 20
        u = jnp.ones((nx, ny, 2)) * 0.1
        points = jnp.array([[10.0, 10.0]])
        kernels = compute_kernels(points, nx, ny)
        u_interp = interpolate_velocity(u, kernels)
        assert u_interp.shape == (1, 2)
        # Should be close to 0.1 (not exact due to finite kernel support)
        assert jnp.allclose(u_interp[0], 0.1, atol=0.02)

    def test_zero_velocity(self):
        """Interpolation of zero field returns zero."""
        nx, ny = 20, 20
        u = jnp.zeros((nx, ny, 2))
        points = jnp.array([[10.0, 10.0]])
        kernels = compute_kernels(points, nx, ny)
        u_interp = interpolate_velocity(u, kernels)
        assert jnp.allclose(u_interp, 0.0, atol=1e-10)


class TestSpreading:
    def test_shape(self):
        points = jnp.array([[10.0, 10.0]])
        forces = jnp.array([[1.0, 0.0]])
        kernels = compute_kernels(points, 20, 20)
        f_grid = spread_force(forces, kernels)
        assert f_grid.shape == (20, 20, 2)

    def test_zero_force_zero_spread(self):
        points = jnp.array([[10.0, 10.0]])
        forces = jnp.zeros((1, 2))
        kernels = compute_kernels(points, 20, 20)
        f_grid = spread_force(forces, kernels)
        assert jnp.allclose(f_grid, 0.0)

    def test_total_force_conservation(self):
        """Total spread force should equal the applied boundary force * ds."""
        points = jnp.array([[10.0, 10.0]])
        forces = jnp.array([[1.0, 0.5]])
        ds = 1.0
        kernels = compute_kernels(points, 30, 30)
        f_grid = spread_force(forces, kernels, ds=ds)
        total = jnp.sum(f_grid, axis=(0, 1))
        assert jnp.allclose(total, forces[0] * ds, atol=0.1)


class TestMultiDirectForcing:
    def test_reduces_velocity_deficit(self):
        """MDF should reduce the velocity error at boundary points."""
        nx, ny = 30, 30
        velocity = jnp.zeros((nx, ny, 2))
        points = jnp.array([[15.0, 15.0]])
        target = jnp.array([[0.1, 0.0]])
        kernels = compute_kernels(points, nx, ny)

        u_before = interpolate_velocity(velocity, kernels)
        error_before = jnp.linalg.norm(target - u_before)

        corrected_u, _ = multi_direct_forcing(
            velocity, kernels, target, n_iter=5,
        )

        u_after = interpolate_velocity(corrected_u, kernels)
        error_after = jnp.linalg.norm(target - u_after)

        assert error_after < error_before

    def test_more_iterations_better(self):
        """More MDF iterations should give smaller error."""
        nx, ny = 30, 30
        velocity = jnp.zeros((nx, ny, 2))
        points = jnp.array([[15.0, 15.0]])
        target = jnp.array([[0.1, 0.0]])
        kernels = compute_kernels(points, nx, ny)

        errors = []
        for n_iter in [1, 3, 5, 10]:
            corrected_u, _ = multi_direct_forcing(
                velocity, kernels, target, n_iter=n_iter,
            )
            u_interp = interpolate_velocity(corrected_u, kernels)
            error = float(jnp.linalg.norm(target - u_interp))
            errors.append(error)

        # Each should be <= previous (monotonically decreasing error)
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i-1] + 1e-6


class TestCircleGeometry:
    def test_point_count(self):
        points, ds = generate_circle_points(10.0, 10.0, 5.0, 20)
        assert points.shape == (20, 2)

    def test_radius_correct(self):
        cx, cy, r = 10.0, 10.0, 5.0
        points, ds = generate_circle_points(cx, cy, r, 50)
        distances = jnp.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        assert jnp.allclose(distances, r, atol=1e-5)

    def test_ds_correct(self):
        r = 5.0
        n = 50
        _, ds = generate_circle_points(10.0, 10.0, r, n)
        expected = 2 * math.pi * r / n
        assert ds == pytest.approx(expected, rel=1e-5)


class TestBoundaryVelocities:
    def test_pure_translation(self):
        """Translation only: all points should have the same velocity."""
        points = jnp.array([[11.0, 10.0], [10.0, 11.0], [9.0, 10.0]])
        center = jnp.array([10.0, 10.0])
        v_center = jnp.array([0.1, 0.0])
        v_target = compute_boundary_velocities(v_center, 0.0, points, center)
        for i in range(3):
            assert jnp.allclose(v_target[i], v_center, atol=1e-6)

    def test_pure_rotation(self):
        """Rotation only: velocity should be perpendicular to radius."""
        center = jnp.array([10.0, 10.0])
        # Point at (11, 10) — radius along +x
        points = jnp.array([[11.0, 10.0]])
        omega = 1.0  # 1 rad/s CCW
        v_target = compute_boundary_velocities(jnp.zeros(2), omega, points, center)
        # omega x r: omega * [-r_y, r_x] = 1.0 * [0, 1] = [0, 1]
        assert jnp.allclose(v_target[0], jnp.array([0.0, 1.0]), atol=1e-6)


class TestDragComputation:
    def test_drag_opposite_to_motion(self):
        """For a body moving in +x, drag should be in -x."""
        # Simulate: uniform flow = 0, body moves in +x
        # The IB forces push fluid in +x -> drag on body is -x
        marker_forces = jnp.array([[0.1, 0.0], [0.1, 0.0]])
        drag = compute_drag_force(marker_forces, ds=1.0)
        assert drag[0] < 0  # Drag opposes the applied force direction

    def test_zero_force_zero_drag(self):
        marker_forces = jnp.zeros((10, 2))
        drag = compute_drag_force(marker_forces)
        assert jnp.allclose(drag, 0.0)

    def test_torque_sign(self):
        """CCW rotation should produce CW drag torque (negative)."""
        # Forces pushing fluid CCW -> torque on body is CW (negative)
        center = jnp.array([0.0, 0.0])
        points = jnp.array([[1.0, 0.0]])
        forces = jnp.array([[0.0, 0.1]])  # Force in +y at +x position
        torque = compute_drag_torque(forces, points, center)
        # r x F = 1*0.1 - 0*0 = 0.1 -> drag torque = -0.1
        assert torque < 0


class TestJAXCompatibility:
    def test_kernel_jit(self):
        points = jnp.array([[10.0, 10.0]])
        jitted = jax.jit(compute_kernels, static_argnums=(1, 2))
        kernels = jitted(points, 20, 20)
        assert jnp.isfinite(kernels).all()

    def test_mdf_jit(self):
        nx, ny = 20, 20
        velocity = jnp.zeros((nx, ny, 2))
        points = jnp.array([[10.0, 10.0]])
        target = jnp.array([[0.1, 0.0]])
        kernels = compute_kernels(points, nx, ny)

        @jax.jit
        def run_mdf(u, k, t):
            return multi_direct_forcing(u, k, t, n_iter=3)

        corrected, force = run_mdf(velocity, kernels, target)
        assert jnp.isfinite(corrected).all()

    def test_interpolation_grad(self):
        """Interpolation should be differentiable."""
        nx, ny = 20, 20
        points = jnp.array([[10.0, 10.0]])
        kernels = compute_kernels(points, nx, ny)

        def interp_sum(ux_val):
            u = jnp.zeros((nx, ny, 2)).at[10, 10, 0].set(ux_val)
            return jnp.sum(interpolate_velocity(u, kernels))

        g = jax.grad(interp_sum)(1.0)
        assert jnp.isfinite(g)
