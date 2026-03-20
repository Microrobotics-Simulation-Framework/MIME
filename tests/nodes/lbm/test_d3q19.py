"""Tests for D3Q19 LBM core — 3D lattice constants, equilibrium, Poiseuille."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.d3q19 import (
    E, W, OPP, CS2, Q,
    equilibrium, compute_macroscopic, stream, collide_bgk,
    lbm_step, init_equilibrium, bounce_back_mask,
    create_channel_walls, create_pipe_walls,
    tau_from_viscosity,
)


# ── Lattice constants ────────────────────────────────────────────────────

class TestD3Q19Constants:
    def test_velocity_count(self):
        assert E.shape == (19, 3)

    def test_weights_sum_to_one(self):
        assert abs(W.sum() - 1.0) < 1e-6

    def test_opposite_involution(self):
        for i in range(19):
            assert OPP[OPP[i]] == i

    def test_opposite_reverses_velocity(self):
        for i in range(1, 19):
            assert np.array_equal(E[OPP[i]], -E[i])

    def test_rest_velocity_zero(self):
        assert np.array_equal(E[0], [0, 0, 0])

    def test_momentum_symmetry(self):
        """Sum of w_i * e_i == 0."""
        momentum = (W[:, None] * E).sum(axis=0)
        assert np.allclose(momentum, 0.0, atol=1e-10)

    def test_second_moment_isotropic(self):
        """Sum of w_i * e_ia * e_ib == cs^2 * delta_ab."""
        # This is the isotropy condition for correct Navier-Stokes recovery
        tensor = np.zeros((3, 3))
        for q in range(19):
            tensor += W[q] * np.outer(E[q], E[q])
        expected = CS2 * np.eye(3)
        assert np.allclose(tensor, expected, atol=1e-10)


# ── Equilibrium ──────────────────────────────────────────────────────────

class TestD3Q19Equilibrium:
    def test_shape(self):
        rho = jnp.ones((4, 4, 4))
        u = jnp.zeros((4, 4, 4, 3))
        f_eq = equilibrium(rho, u)
        assert f_eq.shape == (4, 4, 4, 19)

    def test_at_rest_gives_weights(self):
        rho = jnp.ones((3, 3, 3)) * 2.0
        u = jnp.zeros((3, 3, 3, 3))
        f_eq = equilibrium(rho, u)
        for q in range(19):
            assert jnp.allclose(f_eq[..., q], W[q] * 2.0, atol=1e-6)

    def test_density_conservation(self):
        rho = jnp.ones((4, 4, 4)) * 1.3
        u = jnp.zeros((4, 4, 4, 3)).at[..., 0].set(0.05)
        f_eq = equilibrium(rho, u)
        assert jnp.allclose(jnp.sum(f_eq, axis=-1), rho, atol=1e-5)

    def test_momentum_conservation(self):
        rho = jnp.ones((4, 4, 4))
        u = jnp.zeros((4, 4, 4, 3)).at[..., 2].set(0.03)
        f_eq = equilibrium(rho, u)
        momentum = f_eq @ jnp.array(E, dtype=jnp.float32)
        assert jnp.allclose(momentum, rho[..., None] * u, atol=1e-5)


# ── Streaming ────────────────────────────────────────────────────────────

class TestD3Q19Streaming:
    def test_shape_preserved(self):
        f = jnp.ones((6, 6, 6, 19))
        assert stream(f).shape == (6, 6, 6, 19)

    def test_rest_unchanged(self):
        f = jnp.zeros((6, 6, 6, 19)).at[3, 3, 3, 0].set(1.0)
        assert stream(f)[3, 3, 3, 0] == 1.0

    def test_positive_x_shifts(self):
        f = jnp.zeros((6, 6, 6, 19)).at[2, 3, 3, 1].set(1.0)
        f_s = stream(f)
        assert f_s[3, 3, 3, 1] == 1.0
        assert f_s[2, 3, 3, 1] == 0.0

    def test_diagonal_shifts(self):
        """q=7 (+x+y) should shift diagonally in xy."""
        f = jnp.zeros((8, 8, 8, 19)).at[3, 3, 4, 7].set(1.0)
        f_s = stream(f)
        assert f_s[4, 4, 4, 7] == 1.0

    def test_mass_conservation(self):
        key = jax.random.PRNGKey(42)
        f = jax.random.uniform(key, (8, 8, 8, 19)) * 0.01 + W[None, None, None, :]
        mass_before = jnp.sum(f)
        mass_after = jnp.sum(stream(f))
        assert jnp.abs(mass_before - mass_after) / mass_before < 1e-5


# ── Collision ────────────────────────────────────────────────────────────

class TestD3Q19Collision:
    def test_equilibrium_fixed_point(self):
        rho = jnp.ones((4, 4, 4))
        u = jnp.zeros((4, 4, 4, 3))
        f_eq = equilibrium(rho, u)
        f_out = collide_bgk(f_eq, rho, u, tau=1.0)
        assert jnp.allclose(f_out, f_eq, atol=1e-6)


# ── Bounce-back ──────────────────────────────────────────────────────────

class TestD3Q19BounceBack:
    def test_wall_reflects(self):
        f = jnp.zeros((4, 4, 4, 19)).at[0, 2, 2, 1].set(1.0)
        wall = jnp.zeros((4, 4, 4), dtype=bool).at[0, 2, 2].set(True)
        f_b = bounce_back_mask(f, wall)
        assert f_b[0, 2, 2, 2] == 1.0  # OPP[1] = 2
        assert f_b[0, 2, 2, 1] == 0.0


# ── Wall masks ───────────────────────────────────────────────────────────

class TestD3Q19Walls:
    def test_channel_walls_shape(self):
        mask = create_channel_walls(10, 20, 10, wall_axis=1)
        assert mask.shape == (10, 20, 10)

    def test_channel_walls_at_boundaries(self):
        mask = create_channel_walls(10, 20, 10, wall_axis=1)
        assert mask[5, 0, 5]     # bottom
        assert mask[5, 19, 5]    # top
        assert not mask[5, 10, 5]  # interior

    def test_pipe_walls_shape(self):
        mask = create_pipe_walls(10, 20, 20)
        assert mask.shape == (10, 20, 20)

    def test_pipe_center_is_fluid(self):
        mask = create_pipe_walls(10, 20, 20)
        assert not mask[5, 10, 10]  # center should be fluid

    def test_pipe_corner_is_wall(self):
        mask = create_pipe_walls(10, 20, 20)
        assert mask[5, 0, 0]  # corner is outside pipe


# ── Full LBM step ────────────────────────────────────────────────────────

class TestD3Q19LBMStep:
    def test_shapes(self):
        f = init_equilibrium(8, 8, 8)
        f_new, rho, u = lbm_step(f, tau=0.8)
        assert f_new.shape == (8, 8, 8, 19)
        assert rho.shape == (8, 8, 8)
        assert u.shape == (8, 8, 8, 3)

    def test_quiescent_stays_quiescent(self):
        f = init_equilibrium(8, 8, 8)
        for _ in range(20):
            f, rho, u = lbm_step(f, tau=0.8)
        assert jnp.max(jnp.abs(u)) < 1e-6

    def test_mass_conservation_no_walls(self):
        f = init_equilibrium(8, 8, 8, density=1.2)
        mass_0 = jnp.sum(f)
        for _ in range(10):
            f, _, _ = lbm_step(f, tau=0.8)
        assert jnp.abs(jnp.sum(f) - mass_0) / mass_0 < 1e-5


# ── Poiseuille flow (3D channel) ────────────────────────────────────────

class TestD3Q19Poiseuille:
    def test_channel_poiseuille(self):
        """Pressure-driven flow in a 3D channel between parallel plates.

        Walls at y=0 and y=ny-1. Flow in x-direction.
        Analytical: u(y) = F/(2*nu) * y * (H - y)
        """
        nx, ny, nz = 3, 22, 3  # Thin channel, 20 fluid cells in y
        tau = 0.8
        nu = (tau - 0.5) / 3.0
        F_body = 1e-5

        force = jnp.zeros((nx, ny, nz, 3)).at[..., 0].set(F_body)
        wall_mask = create_channel_walls(nx, ny, nz, wall_axis=1)

        f = init_equilibrium(nx, ny, nz)
        for _ in range(5000):
            f, rho, u = lbm_step(f, tau, wall_mask=wall_mask, force=force)

        # Profile along y at (nx//2, :, nz//2)
        u_profile = u[nx // 2, :, nz // 2, 0]

        H = ny - 2
        y = jnp.arange(ny) - 0.5
        u_analytical = (F_body / (2.0 * nu)) * y * (H - y)
        u_analytical = jnp.where(wall_mask[0, :, 0], 0.0, u_analytical)

        # Compare peak velocities
        # 3D channel with bounce-back has ~5-10% error at ny=22 due to
        # the half-way bounce-back location uncertainty (wall at y=0.5, not y=0)
        interior = ~wall_mask[0, :, 0]
        peak_sim = float(jnp.max(u_profile[interior]))
        peak_ana = float(jnp.max(u_analytical[interior]))
        rel_error = abs(peak_sim - peak_ana) / peak_ana

        assert rel_error < 0.10, (
            f"3D Poiseuille error: {rel_error:.4f} "
            f"(sim={peak_sim:.6e}, ana={peak_ana:.6e})"
        )

    def test_pipe_poiseuille(self):
        """Pressure-driven flow in a cylindrical pipe.

        Analytical: u(r) = F/(4*nu) * (R^2 - r^2)
        """
        nx, ny, nz = 3, 22, 22  # Short pipe, R ~ 9 lattice units
        tau = 0.8
        nu = (tau - 0.5) / 3.0
        F_body = 1e-5

        force = jnp.zeros((nx, ny, nz, 3)).at[..., 0].set(F_body)
        wall_mask = create_pipe_walls(nx, ny, nz)

        f = init_equilibrium(nx, ny, nz)
        for _ in range(5000):
            f, rho, u = lbm_step(f, tau, wall_mask=wall_mask, force=force)

        # Centreline velocity
        cy, cz = ny // 2, nz // 2
        u_center = float(u[nx // 2, cy, cz, 0])

        R = (ny - 2) / 2.0
        u_center_analytical = F_body * R ** 2 / (4.0 * nu)

        rel_error = abs(u_center - u_center_analytical) / u_center_analytical
        assert rel_error < 0.10, (
            f"Pipe Poiseuille centreline error: {rel_error:.4f} "
            f"(sim={u_center:.6e}, ana={u_center_analytical:.6e})"
        )


# ── JAX compatibility ───────────────────────────────────────────────────

class TestD3Q19JAX:
    def test_jit_step(self):
        f = init_equilibrium(6, 6, 6)
        jitted = jax.jit(lbm_step, static_argnums=(1,))
        f_new, rho, u = jitted(f, 0.8)
        assert jnp.isfinite(f_new).all()

    def test_equilibrium_grad(self):
        rho = jnp.ones((4, 4, 4))

        def sum_feq(ux):
            u = jnp.zeros((4, 4, 4, 3)).at[..., 0].set(ux)
            return jnp.sum(equilibrium(rho, u))

        g = jax.grad(sum_feq)(0.01)
        assert jnp.isfinite(g)
