"""Tests for surface mesh generators."""

import numpy as np
import pytest

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh,
    cylinder_surface_mesh,
)


class TestSphereMesh:
    def test_points_at_correct_radius(self):
        """Centroids should be near the sphere surface."""
        mesh = sphere_surface_mesh(center=(0, 0, 0), radius=1.0, n_refine=2)
        dists = np.linalg.norm(mesh.points, axis=1)
        # Centroids of triangles on a sphere are slightly inside
        # (at r = R * cos(half_angle)), but should be within a few %
        np.testing.assert_allclose(dists, 1.0, atol=0.05)

    def test_normals_outward(self):
        """Normals should point away from center."""
        mesh = sphere_surface_mesh(center=(0, 0, 0), radius=1.0, n_refine=2)
        dots = np.sum(mesh.points * mesh.normals, axis=1)
        assert np.all(dots > 0), "Some normals point inward"

    def test_normals_unit_length(self):
        """Normals should be unit vectors."""
        mesh = sphere_surface_mesh(center=(0, 0, 0), radius=1.0, n_refine=2)
        norms = np.linalg.norm(mesh.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_total_area(self):
        """Sum of weights should approximate 4πr²."""
        for n_refine in [2, 3, 4]:
            mesh = sphere_surface_mesh(radius=2.0, n_refine=n_refine)
            expected = 4 * np.pi * 2.0 ** 2
            assert abs(mesh.total_area - expected) / expected < 0.05, (
                f"n_refine={n_refine}: area={mesh.total_area:.3f}, "
                f"expected={expected:.3f}"
            )

    def test_point_count_scaling(self):
        """N should scale as ~10 * 4^n_refine."""
        for n_refine in [1, 2, 3]:
            mesh = sphere_surface_mesh(n_refine=n_refine)
            expected = 20 * 4 ** n_refine  # 20 faces, each subdivided
            assert mesh.n_points == expected

    def test_center_offset(self):
        """Points should be centered at the given center."""
        center = (1.0, 2.0, 3.0)
        mesh = sphere_surface_mesh(center=center, radius=1.0, n_refine=2)
        centroid = np.mean(mesh.points, axis=0)
        np.testing.assert_allclose(centroid, center, atol=0.1)


class TestCylinderMesh:
    def test_points_at_correct_radius(self):
        """Points should be at the cylinder radius."""
        mesh = cylinder_surface_mesh(radius=2.0, length=4.0, n_circ=16, n_axial=10)
        # For Z-axis cylinder, radial distance in XY plane
        r_xy = np.sqrt(mesh.points[:, 0]**2 + mesh.points[:, 1]**2)
        np.testing.assert_allclose(r_xy, 2.0, atol=1e-10)

    def test_normals_inward(self):
        """Normals should point inward (toward axis) for interior flow."""
        mesh = cylinder_surface_mesh(radius=1.0, length=2.0)
        # Inward normal: dot(normal, radial_direction) < 0
        r_hat = mesh.points.copy()
        r_hat[:, 2] = 0  # zero out axial component
        r_hat /= np.linalg.norm(r_hat, axis=1, keepdims=True) + 1e-30
        dots = np.sum(mesh.normals * r_hat, axis=1)
        assert np.all(dots < 0), "Some normals point outward"

    def test_total_area(self):
        """Sum of weights should approximate 2πr*L."""
        mesh = cylinder_surface_mesh(radius=1.5, length=3.0, n_circ=32, n_axial=20)
        expected = 2 * np.pi * 1.5 * 3.0
        assert abs(mesh.total_area - expected) / expected < 0.1

    def test_axis_permutation(self):
        """Cylinder along X axis should have points in YZ plane."""
        mesh = cylinder_surface_mesh(radius=1.0, length=2.0, axis=0,
                                      n_circ=8, n_axial=4)
        # For X-axis cylinder, radial distance in YZ plane
        r_yz = np.sqrt(mesh.points[:, 1]**2 + mesh.points[:, 2]**2)
        np.testing.assert_allclose(r_yz, 1.0, atol=1e-10)
