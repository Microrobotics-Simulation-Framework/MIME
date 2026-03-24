"""Tests for IBLBMFluidNode.

Tests 1-9: unit tests on small grids (16-24^3).
Test 10: Bouzidi path regression against T2.6b reference at 64^3.

(Integration tests 10-11 from the plan are in
tests/verification/test_iblbm_graph_wiring.py)
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.lbm.fluid_node import (
    IBLBMFluidNode,
    make_iblbm_rigid_body_edges,
)


# -- Helpers ------------------------------------------------------------------

VESSEL_DIAMETER_MM = 9.4
UMR_GEOM_MM = dict(
    body_radius=0.87, body_length=4.1, cone_length=1.9,
    cone_end_radius=0.255, fin_outer_radius=1.42,
    fin_length=2.03, fin_width=0.55, fin_thickness=0.15,
    helix_pitch=8.0,
)


def _make_node(N=24, ratio=0.30, use_bouzidi=False):
    """Create an IBLBMFluidNode for testing at resolution N."""
    dx = VESSEL_DIAMETER_MM / N
    geom_lu = {k: v / dx for k, v in UMR_GEOM_MM.items()}
    R_umr_lu = geom_lu["body_radius"]
    R_vessel_lu = R_umr_lu / ratio

    body_geometry_params = dict(nx=N, ny=N, nz=N, **geom_lu)
    return IBLBMFluidNode(
        name="lbm",
        timestep=1.0,
        nx=N, ny=N, nz=N,
        tau=0.8,
        vessel_radius_lu=R_vessel_lu,
        body_geometry_params=body_geometry_params,
        use_bouzidi=use_bouzidi,
    )


def _make_omega(N=24, ratio=0.30):
    """Compute angular velocity matching the sweep script convention."""
    dx = VESSEL_DIAMETER_MM / N
    geom_lu = {k: v / dx for k, v in UMR_GEOM_MM.items()}
    R_fin_lu = geom_lu["fin_outer_radius"]
    cs = 1.0 / math.sqrt(3)
    return 0.05 * cs / R_fin_lu


# -- Test 1: initial_state shapes --------------------------------------------

class TestInitialState:
    def test_initial_state_shapes(self):
        node = _make_node(N=16)
        state = node.initial_state()

        assert state["f"].shape == (16, 16, 16, 19)
        assert state["f"].dtype == jnp.float32
        assert state["solid_mask"].shape == (16, 16, 16)
        assert state["body_angle"].shape == ()
        assert state["drag_force"].shape == (3,)
        assert state["drag_torque"].shape == (3,)
        assert set(state.keys()) == {
            "f", "solid_mask", "body_angle", "drag_force", "drag_torque",
        }


# -- Test 2: update matches sweep script logic -------------------------------

class TestUpdateMatchesSweep:
    def test_update_matches_sweep_logic(self):
        """5 steps via node must match the same operations done manually."""
        from mime.nodes.environment.lbm.d3q19 import (
            init_equilibrium, lbm_step_split,
        )
        from mime.nodes.environment.lbm.bounce_back import (
            compute_missing_mask, apply_bounce_back,
            compute_momentum_exchange_force, compute_momentum_exchange_torque,
        )
        from mime.nodes.environment.lbm.rotating_body import (
            _rotation_velocity_field,
        )
        from mime.nodes.robot.helix_geometry import create_umr_mask

        N = 24
        node = _make_node(N=N, use_bouzidi=False)
        omega = _make_omega(N=N)
        bi = {"body_angular_velocity": jnp.array([0.0, 0.0, omega])}

        # Node path
        state = node.initial_state()
        for _ in range(5):
            state = node.update(state, bi, 1.0)

        # Manual path (replicating run_single logic)
        dx = VESSEL_DIAMETER_MM / N
        geom_lu = {k: v / dx for k, v in UMR_GEOM_MM.items()}
        R_umr_lu = geom_lu["body_radius"]
        R_vessel_lu = R_umr_lu / 0.30
        cx, cy, cz = N / 2.0, N / 2.0, N / 2.0
        center = (cx, cy, cz)

        ix = jnp.arange(N, dtype=jnp.float32)
        iy = jnp.arange(N, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        dist_2d = jnp.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        pipe_wall = jnp.broadcast_to(
            (dist_2d >= R_vessel_lu)[..., None], (N, N, N),
        )
        pipe_missing = compute_missing_mask(pipe_wall)

        f = init_equilibrium(N, N, N)
        angle = 0.0
        for _ in range(5):
            angle_new = angle + omega
            umr_mask = create_umr_mask(
                center=center, rotation_angle=angle_new,
                nx=N, ny=N, nz=N, **geom_lu,
            )
            umr_missing = compute_missing_mask(umr_mask)
            solid_mask = pipe_wall | umr_mask
            wall_vel = _rotation_velocity_field(
                (N, N, N), omega, (0, 0, 1), center,
            )

            f_pre, f_post, rho, u = lbm_step_split(f, 0.8)
            f = apply_bounce_back(
                f_post, f_pre, pipe_missing, solid_mask, wall_velocity=None,
            )
            f = apply_bounce_back(
                f, f_pre, umr_missing, solid_mask, wall_velocity=wall_vel,
            )
            angle = angle_new

        assert jnp.allclose(state["f"], f, atol=1e-6), (
            f"Max diff: {float(jnp.max(jnp.abs(state['f'] - f)))}"
        )


# -- Test 3: stationary body produces zero torque ----------------------------

class TestStationaryBody:
    def test_stationary_body_zero_torque(self):
        node = _make_node(N=16)
        state = node.initial_state()
        bi = {"body_angular_velocity": jnp.zeros(3)}

        for _ in range(10):
            state = node.update(state, bi, 1.0)

        assert float(jnp.max(jnp.abs(state["drag_torque"]))) < 1e-4


# -- Test 4: rotating body produces resistive torque -------------------------

class TestRotatingBody:
    def test_rotating_body_resistive_torque(self):
        N = 24
        node = _make_node(N=N)
        omega = _make_omega(N=N)
        state = node.initial_state()
        bi = {"body_angular_velocity": jnp.array([0.0, 0.0, omega])}

        for _ in range(50):
            state = node.update(state, bi, 1.0)

        tz = float(state["drag_torque"][2])
        # Torque should be nonzero and positive (body drives fluid,
        # fluid resists with positive torque in momentum exchange convention)
        assert abs(tz) > 1e-6, f"Torque too small: {tz}"


# -- Test 5: mass conservation -----------------------------------------------

class TestMassConservation:
    def test_mass_conservation(self):
        N = 16
        node = _make_node(N=N)
        state = node.initial_state()
        bi = {"body_angular_velocity": jnp.array([0.0, 0.0, 0.01])}

        rho_initial = float(jnp.sum(state["f"]))

        for _ in range(5):
            state = node.update(state, bi, 1.0)

        rho_final = float(jnp.sum(state["f"]))
        rel_change = abs(rho_final - rho_initial) / rho_initial
        assert rel_change < 1e-4, f"Mass conservation violated: {rel_change}"


# -- Test 6: boundary fluxes match state -------------------------------------

class TestBoundaryFluxes:
    def test_boundary_fluxes_match_state(self):
        node = _make_node(N=16)
        state = node.initial_state()
        bi = {"body_angular_velocity": jnp.array([0.0, 0.0, 0.01])}

        state = node.update(state, bi, 1.0)
        fluxes = node.compute_boundary_fluxes(state, bi, 1.0)

        assert jnp.allclose(fluxes["drag_force"], state["drag_force"])
        assert jnp.allclose(fluxes["drag_torque"], state["drag_torque"])


# -- Test 7: boundary_input_spec keys ----------------------------------------

class TestBoundaryInputSpec:
    def test_boundary_input_spec_keys(self):
        node = _make_node(N=16)
        spec = node.boundary_input_spec()

        assert "body_angular_velocity" in spec
        assert "body_orientation" in spec
        assert spec["body_angular_velocity"].shape == (3,)
        assert spec["body_orientation"].shape == (4,)


# -- Test 8: requires_halo ---------------------------------------------------

class TestRequiresHalo:
    def test_requires_halo(self):
        node = _make_node(N=16)
        assert node.requires_halo is True


# -- Test 9: MIME metadata consistency ----------------------------------------

class TestMetadataConsistency:
    def test_mime_metadata_consistency(self):
        node = _make_node(N=16)
        errors = node.validate_mime_consistency()
        assert errors == [], f"Metadata errors: {errors}"


# -- Test 10: Bouzidi path matches T2.6b reference ---------------------------

class TestBouzidiRegression:
    @pytest.mark.slow
    def test_bouzidi_path_matches_reference(self):
        """Run at 64^3, ratio 0.30, 200 steps with Bouzidi.

        Mean drag_torque_z must match the T2.6b sanity test reference
        value (21.3916 lu at 64^3 with sparse Bouzidi, 8 bisection iters)
        within 0.1%.
        """
        N = 64
        node = _make_node(N=N, ratio=0.30, use_bouzidi=True)
        omega = _make_omega(N=N)
        state = node.initial_state()
        bi = {"body_angular_velocity": jnp.array([0.0, 0.0, omega])}

        n_steps = 200
        torques_z = []
        for step in range(n_steps):
            state = node.update(state, bi, 1.0)
            torques_z.append(float(state["drag_torque"][2]))

        period = int(round(2 * math.pi / omega))
        if len(torques_z) >= period:
            mean_tz = np.mean(torques_z[-period:])
        else:
            mean_tz = np.mean(torques_z)

        reference_tz = 21.3916
        rel_error = abs(mean_tz - reference_tz) / abs(reference_tz)
        assert rel_error < 0.001, (
            f"Bouzidi regression failed: mean_tz={mean_tz:.4f}, "
            f"reference={reference_tz}, rel_error={rel_error:.4f}"
        )


# -- Test 11: edge wiring helper ---------------------------------------------

class TestEdgeWiringHelper:
    def test_make_iblbm_rigid_body_edges_returns_correct_transforms(self):
        """make_iblbm_rigid_body_edges returns edges with correct unit transforms."""
        edges = make_iblbm_rigid_body_edges(
            "lbm", "body",
            dx_physical=1e-4, dt_physical=1e-6, rho_physical=1060.0,
        )
        assert len(edges) == 4

        # Force edge should have a transform
        force_edge = next(e for e in edges if e.source_field == "drag_force")
        assert force_edge.transform is not None
        assert force_edge.additive is True
        assert force_edge.source_units == "lattice"
        assert force_edge.target_units == "N"

        # Torque edge should have a transform
        torque_edge = next(e for e in edges if e.source_field == "drag_torque")
        assert torque_edge.transform is not None
        assert torque_edge.additive is True
        assert torque_edge.source_units == "lattice"
        assert torque_edge.target_units == "N*m"

        # Transform should be JAX-traceable and change the value
        dummy = jnp.ones(3)
        force_result = force_edge.transform(dummy)
        assert force_result.shape == (3,)
        assert not jnp.allclose(force_result, dummy)

        torque_result = torque_edge.transform(dummy)
        assert torque_result.shape == (3,)
        assert not jnp.allclose(torque_result, dummy)

        # Back-edges should have no transform
        back_edges = [e for e in edges if e.target_node == "lbm"]
        assert len(back_edges) == 2
        assert all(e.transform is None for e in back_edges)

    def test_boundary_flux_spec_declared(self):
        """IBLBMFluidNode declares boundary_flux_spec with output_units."""
        node = _make_node(N=16)
        flux_spec = node.boundary_flux_spec()
        assert "drag_force" in flux_spec
        assert "drag_torque" in flux_spec
        assert flux_spec["drag_force"].output_units == "lattice"
        assert flux_spec["drag_torque"].output_units == "lattice"

    def test_boundary_input_spec_expected_units(self):
        """IBLBMFluidNode declares expected_units on boundary inputs."""
        node = _make_node(N=16)
        spec = node.boundary_input_spec()
        assert spec["body_angular_velocity"].expected_units == "lattice"
        assert spec["body_orientation"].expected_units == "lattice"
