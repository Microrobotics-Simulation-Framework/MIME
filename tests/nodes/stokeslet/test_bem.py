"""Tests for BEM assembly and solve."""

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.stokeslet.bem import (
    assemble_system_matrix,
    assemble_rhs_rigid_motion,
    compute_force_torque,
    solve_bem,
)
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh


class TestBEMAssembly:
    @pytest.fixture
    def sphere_mesh(self):
        mesh = sphere_surface_mesh(radius=1.0, n_refine=2)
        return (
            jnp.array(mesh.points),
            jnp.array(mesh.weights),
            mesh.n_points,
        )

    def test_matrix_shape(self, sphere_mesh):
        points, weights, N = sphere_mesh
        A = assemble_system_matrix(points, weights, epsilon=0.1, mu=1.0)
        assert A.shape == (3 * N, 3 * N)

    def test_matrix_finite(self, sphere_mesh):
        points, weights, N = sphere_mesh
        A = assemble_system_matrix(points, weights, epsilon=0.1, mu=1.0)
        assert jnp.all(jnp.isfinite(A))

    def test_rhs_pure_translation(self, sphere_mesh):
        points, weights, N = sphere_mesh
        center = jnp.zeros(3)
        U = jnp.array([1.0, 0.0, 0.0])
        omega = jnp.zeros(3)
        rhs = assemble_rhs_rigid_motion(points, center, U, omega)
        assert rhs.shape == (3 * N,)
        # For pure translation, all x-components should be 1.0
        rhs_reshaped = rhs.reshape(N, 3)
        np.testing.assert_allclose(rhs_reshaped[:, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(rhs_reshaped[:, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(rhs_reshaped[:, 2], 0.0, atol=1e-10)

    def test_force_torque_integration(self):
        """Uniform traction on a sphere should give correct total force."""
        mesh = sphere_surface_mesh(radius=1.0, n_refine=3)
        points = jnp.array(mesh.points)
        weights = jnp.array(mesh.weights)
        center = jnp.zeros(3)

        # Uniform traction f = [1, 0, 0] everywhere
        N = mesh.n_points
        traction = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (N, 1))
        F, T = compute_force_torque(points, weights, traction, center)

        # F should be [total_area, 0, 0]
        expected_area = 4 * np.pi  # radius=1
        np.testing.assert_allclose(float(F[0]), expected_area, rtol=0.05)
        np.testing.assert_allclose(float(F[1]), 0.0, atol=0.01)
        np.testing.assert_allclose(float(F[2]), 0.0, atol=0.01)


class TestBEMSolve:
    def test_solve_small_system(self):
        """Solve a small BEM system and verify solution satisfies Af=rhs."""
        mesh = sphere_surface_mesh(radius=1.0, n_refine=1)
        points = jnp.array(mesh.points)
        weights = jnp.array(mesh.weights)

        A = assemble_system_matrix(points, weights, epsilon=0.3, mu=1.0)
        U = jnp.array([1.0, 0.0, 0.0])
        rhs = assemble_rhs_rigid_motion(points, jnp.zeros(3), U, jnp.zeros(3))

        f = solve_bem(A, rhs)

        # Verify Af ≈ rhs
        residual = jnp.linalg.norm(A @ f - rhs) / jnp.linalg.norm(rhs)
        assert float(residual) < 1e-6, f"Residual too large: {residual}"
