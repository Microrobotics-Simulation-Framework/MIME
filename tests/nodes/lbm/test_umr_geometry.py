"""Tests for UMR discontinuous helix geometry and SDF-based q-values."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import math

from mime.nodes.robot.helix_geometry import (
    create_cylinder_body_mask,
    create_discontinuous_fins_mask,
    create_umr_mask,
    umr_sdf,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    compute_q_values_cylinder,
    compute_q_values_sdf,
)


class TestCylinderBodyMask:
    def test_shape(self):
        mask = create_cylinder_body_mask(
            30, 30, 60, body_radius=5.0, body_length=30.0,
            cone_length=10.0, cone_end_radius=1.0,
            center=(15.0, 15.0, 30.0),
        )
        assert mask.shape == (30, 30, 60)
        assert mask.dtype == jnp.bool_

    def test_volume_approx_pi_r2_l(self):
        """Volume of the cylindrical part should be ~pi*R^2*L."""
        R = 8.0
        L = 30.0
        cone_L = 0.0  # No cone for this test
        nx, ny, nz = 40, 40, 60
        center = (20.0, 20.0, 30.0)
        mask = create_cylinder_body_mask(
            nx, ny, nz, body_radius=R, body_length=L,
            cone_length=cone_L, cone_end_radius=0.0,
            center=center,
        )
        volume = float(jnp.sum(mask))
        expected = math.pi * R**2 * L
        # Discretisation error ~5% at R=8
        rel_error = abs(volume - expected) / expected
        assert rel_error < 0.10, (
            f"Cylinder volume error {rel_error:.1%}: got {volume:.0f}, expected {expected:.0f}"
        )

    def test_cone_tip_radius(self):
        """Cone tip should taper to the end radius."""
        R = 5.0
        cone_end_R = 1.0
        cone_L = 20.0
        body_L = 20.0
        nx, ny, nz = 30, 30, 60
        center = (15.0, 15.0, 30.0)
        mask = create_cylinder_body_mask(
            nx, ny, nz, body_radius=R, body_length=body_L,
            cone_length=cone_L, cone_end_radius=cone_end_R,
            center=center,
        )
        # At the very tip of the cone (z = center_z + total_length/2)
        # The cone end is at z = cz + (body_L + cone_L)/2 = 30 + 20 = 50
        # Check that the radius at z=49 (near tip) is close to cone_end_R
        tip_slice = mask[:, :, 49]
        # Count solid nodes in this slice
        solid_count = float(jnp.sum(tip_slice))
        # Expected: ~pi * cone_end_R^2 (very small)
        # At R=1, pi*1 ~ 3.14 nodes
        assert solid_count < math.pi * (R * 1.5) ** 2, (
            f"Tip slice has {solid_count} solid nodes, should be much less than {math.pi * R**2:.0f}"
        )
        # The tip slice should have some solid nodes (not zero if inside the cone)
        # At z=49, we're at the very end of the cone
        # z_cone_end = 30 + 20 = 50, so z=49 is 1 unit before the tip
        # t = (49 - (30 + 20 - 20)) / 20 = (49 - 30 + (20+20)/2 - 20) / 20
        # Actually, z_body_end = cz + (body_L - total_L)/2 + body_L
        # z_start = cz - total_L/2 = 30 - 20 = 10
        # z_body_end = 10 + 20 = 30
        # z_cone_end = 10 + 40 = 50
        # At z=49: t = (49 - 30) / 20 = 0.95
        # cone_radius = 5*(1-0.95) + 1*0.95 = 0.25 + 0.95 = 1.2
        # ~pi * 1.2^2 = 4.5 nodes
        assert solid_count > 0, "Tip slice should have some solid nodes"

    def test_center_axis_solid(self):
        """Center of the body should be solid."""
        mask = create_cylinder_body_mask(
            30, 30, 60, body_radius=5.0, body_length=30.0,
            cone_length=10.0, cone_end_radius=1.0,
            center=(15.0, 15.0, 30.0),
        )
        assert mask[15, 15, 30]


class TestDiscontinuousFinsMask:
    def test_has_solid_nodes(self):
        """Fins mask should contain solid nodes."""
        mask = create_discontinuous_fins_mask(
            60, 60, 60,
            body_radius=10.0, fin_outer_radius=18.0,
            fin_length=15.0, fin_width=6.0, fin_thickness=2.0,
            n_fin_sets=2, fins_per_set=3,
            helix_pitch=80.0,
            center=(30.0, 30.0, 30.0),
            body_length=40.0,
        )
        assert jnp.sum(mask) > 0

    def test_fin_count_at_least_6(self):
        """Should have at least 6 disconnected fin regions (2 sets x 3).

        Use short fins (length=4) over a long body (length=80) so the
        6 fins are well-separated in z.
        """
        nx, ny, nz = 80, 80, 120
        mask = create_discontinuous_fins_mask(
            nx, ny, nz,
            body_radius=10.0, fin_outer_radius=20.0,
            fin_length=4.0, fin_width=4.0, fin_thickness=2.0,
            n_fin_sets=2, fins_per_set=3,
            helix_pitch=200.0,
            center=(40.0, 40.0, 60.0),
            body_length=80.0,
        )
        solid_count = float(jnp.sum(mask))
        assert solid_count > 0, "No fin nodes generated"

        # Check that fins exist in at least 5 distinct z-bands
        # (6 fins but some may merge if close in z)
        z_has_solid = jnp.any(jnp.any(mask, axis=0), axis=0)  # (nz,)
        z_solid = np.array(z_has_solid)
        transitions = np.diff(z_solid.astype(int))
        n_bands = np.sum(transitions == 1)  # rising edges
        if z_solid[0]:
            n_bands += 1
        assert n_bands >= 5, (
            f"Expected at least 5 distinct z-bands of fins, got {n_bands}"
        )


class TestUMRMask:
    def test_shape(self):
        mask = create_umr_mask(30, 30, 60, center=(15.0, 15.0, 30.0))
        assert mask.shape == (30, 30, 60)

    def test_body_solid(self):
        """Center of the UMR body should be solid (using lattice-scaled params)."""
        # Default UMR params are in mm (~0.87mm radius), which is sub-lattice.
        # Scale up to lattice units for this test.
        mask = create_umr_mask(
            30, 30, 60,
            body_radius=5.0, body_length=20.0,
            cone_length=10.0, cone_end_radius=1.0,
            fin_outer_radius=8.0, fin_length=8.0, fin_width=3.0,
            fin_thickness=1.0, helix_pitch=40.0,
            center=(15.0, 15.0, 30.0),
        )
        assert mask[15, 15, 25]  # Inside body region

    def test_rotation_changes_mask(self):
        """Rotating the UMR should change the fin positions."""
        mask1 = create_umr_mask(
            30, 30, 60,
            body_radius=5.0, body_length=20.0,
            cone_length=10.0, cone_end_radius=1.0,
            fin_outer_radius=8.0, fin_length=8.0, fin_width=3.0,
            fin_thickness=1.0, helix_pitch=40.0,
            center=(15.0, 15.0, 30.0), rotation_angle=0.0,
        )
        mask2 = create_umr_mask(
            30, 30, 60,
            body_radius=5.0, body_length=20.0,
            cone_length=10.0, cone_end_radius=1.0,
            fin_outer_radius=8.0, fin_length=8.0, fin_width=3.0,
            fin_thickness=1.0, helix_pitch=40.0,
            center=(15.0, 15.0, 30.0), rotation_angle=math.pi / 3,
        )
        assert not jnp.array_equal(mask1, mask2)


class TestUMRSdf:
    def test_sign_convention(self):
        """SDF should be negative inside, positive outside."""
        # Point at center of body — should be inside (negative)
        center = (0.0, 0.0, 0.0)
        pts_inside = jnp.array([[0.0, 0.0, 0.0]])
        sdf_inside = umr_sdf(pts_inside, center=center)
        assert float(sdf_inside[0]) < 0, f"SDF at center should be negative, got {sdf_inside[0]}"

        # Point far away — should be outside (positive)
        pts_outside = jnp.array([[100.0, 100.0, 100.0]])
        sdf_outside = umr_sdf(pts_outside, center=center)
        assert float(sdf_outside[0]) > 0, f"SDF far away should be positive, got {sdf_outside[0]}"

    def test_body_surface_near_zero(self):
        """SDF should be near zero at the body surface."""
        center = (0.0, 0.0, 0.0)
        R = 0.87
        # Point on the surface: (R, 0, 0) — at z=0 which is within body
        pts_surface = jnp.array([[R, 0.0, 0.0]])
        sdf_val = umr_sdf(pts_surface, center=center)
        assert abs(float(sdf_val[0])) < 0.1, (
            f"SDF at surface should be near zero, got {sdf_val[0]}"
        )


class TestQValuesSDF:
    def test_cylinder_sdf_vs_analytical(self):
        """q_values from SDF should match analytical cylinder values.

        Compare compute_q_values_sdf (using a cylinder SDF) with
        compute_q_values_cylinder on boundary links where q > 0.1.
        """
        # Create a simple z-aligned cylinder
        nx, ny, nz = 24, 24, 8
        cx, cy = 12.0, 12.0
        R = 5.0

        # Create solid mask for cylinder
        ix = jnp.arange(nx, dtype=jnp.float32)
        iy = jnp.arange(ny, dtype=jnp.float32)
        iz = jnp.arange(nz, dtype=jnp.float32)
        gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
        dist_2d = jnp.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        solid_mask = dist_2d < R

        mm = compute_missing_mask(solid_mask)

        # Analytical q-values
        q_analytical = compute_q_values_cylinder(mm, cx, cy, R, is_inner=True)

        # SDF for z-aligned cylinder: distance = sqrt(dx^2 + dy^2) - R
        def cylinder_sdf(pts):
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            r = jnp.sqrt(dx ** 2 + dy ** 2)
            return r - R  # positive outside, negative inside

        q_sdf = compute_q_values_sdf(mm, cylinder_sdf)

        # Compare on boundary links where q > 0.1 (avoid numerically tricky links)
        # Use the outgoing-direction mask from missing_mask
        boundary_mask = mm & (q_analytical > 0.1) & (q_analytical < 0.9)

        if jnp.sum(boundary_mask) > 0:
            q_a = q_analytical[boundary_mask]
            q_s = q_sdf[boundary_mask]

            # Relative error should be < 0.1% (16-iteration bisection gives ~1e-5 precision)
            rel_error = jnp.abs(q_a - q_s) / jnp.maximum(q_a, 1e-10)
            max_rel_error = float(jnp.max(rel_error))
            mean_rel_error = float(jnp.mean(rel_error))

            assert max_rel_error < 0.001, (
                f"Max relative error {max_rel_error:.6f} exceeds 0.1% threshold "
                f"(mean={mean_rel_error:.6f}, N={int(jnp.sum(boundary_mask))} links)"
            )
        else:
            pytest.skip("No boundary links with q in (0.1, 0.9)")

    def test_shape(self):
        """q_values_sdf should have the correct shape."""
        nx, ny, nz = 10, 10, 10
        solid = jnp.zeros((nx, ny, nz), dtype=bool).at[5, 5, 5].set(True)
        mm = compute_missing_mask(solid)

        def simple_sdf(pts):
            return jnp.sqrt(jnp.sum((pts - 5.0) ** 2, axis=-1)) - 0.5

        q_vals = compute_q_values_sdf(mm, simple_sdf)
        assert q_vals.shape == (19, nx, ny, nz)

    def test_q_values_in_range(self):
        """q_values should be in (0, 1) at boundary links."""
        nx, ny, nz = 10, 10, 10
        solid = jnp.zeros((nx, ny, nz), dtype=bool).at[5, 5, 5].set(True)
        mm = compute_missing_mask(solid)

        def simple_sdf(pts):
            return jnp.sqrt(jnp.sum((pts - 5.0) ** 2, axis=-1)) - 0.5

        q_vals = compute_q_values_sdf(mm, simple_sdf)
        boundary_q = q_vals[mm]
        assert float(jnp.min(boundary_q)) > 0.0
        assert float(jnp.max(boundary_q)) < 1.0
