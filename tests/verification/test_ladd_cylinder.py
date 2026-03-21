"""MIME-VER-008: Couette flow torque benchmark.

Validates the bounce-back + momentum exchange implementation against
the analytical Couette solution for a rotating inner cylinder inside
a static outer cylindrical wall:

    T_Couette = -4 * pi * nu * Omega * R1^2 * R2^2 / (R2^2 - R1^2)
                (per unit length, in lattice units with dx=dt=1)

This is the correct reference for a bounded (pipe) domain, replacing
the infinite-domain Stokes formula which doesn't apply to finite domains.

The benchmark runs at multiple resolutions:
- 32x32x3: debug resolution (fast, catches implementation bugs)
- 64x64x3: CI resolution (quantitative accuracy <15%)
- 128x128x3: validation resolution (MIME-VER-008 pass criterion: <5% error)

The 3D domain is periodic in z with nz=3 (thin slab), making it
effectively 2D. Both cylinder axes are along z.

Reference:
- Ladd, A.J.C. (1994). "Numerical simulations of particulate suspensions
  via a discretized Boltzmann equation." J. Fluid Mech. 271, 285-309.
- Mei, R., Yu, D., Shyy, W., Luo, L.S. (1999). "Force evaluation in the
  lattice Boltzmann method involving curved geometry." Phys. Rev. E 65, 041203.
"""

import math
import pytest
import jax
import jax.numpy as jnp

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)
from mime.nodes.environment.lbm.d3q19 import (
    init_equilibrium, lbm_step_split, CS2,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    apply_bounce_back,
    compute_momentum_exchange_torque,
    compute_momentum_exchange_force,
)


# ── Geometry helpers ─────────────────────────────────────────────────────

def create_cylinder_mask_3d(
    nx: int, ny: int, nz: int,
    center_x: float, center_y: float,
    radius: float,
) -> jnp.ndarray:
    """Create a solid mask for an infinite cylinder along z.

    Nodes with distance < radius from (center_x, center_y) are solid.
    """
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(ix, iy, indexing='ij')
    dist_2d = jnp.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    mask_2d = dist_2d < radius
    return jnp.broadcast_to(mask_2d[:, :, None], (nx, ny, nz))


def create_outer_wall_mask_3d(
    nx: int, ny: int, nz: int,
    center_x: float, center_y: float,
    radius: float,
) -> jnp.ndarray:
    """Create a solid mask for a cylindrical outer wall along z.

    Nodes with distance >= radius from (center_x, center_y) are solid.
    """
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(ix, iy, indexing='ij')
    dist_2d = jnp.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    mask_2d = dist_2d >= radius
    return jnp.broadcast_to(mask_2d[:, :, None], (nx, ny, nz))


def compute_cylinder_wall_velocity(
    nx: int, ny: int, nz: int,
    center_x: float, center_y: float,
    angular_velocity: float,
) -> jnp.ndarray:
    """Compute wall velocity for a cylinder rotating about the z-axis.

    u_wall = Omega x r = Omega * (-ry, rx, 0)
    """
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(ix, iy, indexing='ij')

    rx = xx - center_x
    ry = yy - center_y

    ux = -angular_velocity * ry
    uy = angular_velocity * rx
    uz = jnp.zeros_like(ux)

    vel_2d = jnp.stack([ux, uy, uz], axis=-1)  # (nx, ny, 3)
    return jnp.broadcast_to(vel_2d[:, :, None, :], (nx, ny, nz, 3))


# ── Couette flow simulation ─────────────────────────────────────────────

def run_couette_cylinder(
    n_grid: int,
    R_inner: float,
    R_outer: float,
    omega_lattice: float,
    tau: float,
    n_steps: int,
) -> tuple[float, float]:
    """Run Couette flow: spinning inner cylinder inside static outer wall.

    Parameters
    ----------
    n_grid : int
        Grid size (n_grid x n_grid x 3).
    R_inner : float
        Inner cylinder radius in lattice units.
    R_outer : float
        Outer wall radius in lattice units.
    omega_lattice : float
        Angular velocity of inner cylinder (rad per timestep).
    tau : float
        BGK relaxation time.
    n_steps : int
        Number of LBM steps to reach steady state.

    Returns
    -------
    torque_sim : float
        Simulated torque on the inner cylinder (z-component, lattice units).
    torque_analytical : float
        Analytical Couette torque (lattice units).
    """
    nx, ny, nz = n_grid, n_grid, 3
    nu = (tau - 0.5) * CS2
    center_x = nx / 2.0
    center_y = ny / 2.0
    center = jnp.array([center_x, center_y, nz / 2.0])

    # Inner cylinder (solid, rotating)
    inner_solid = create_cylinder_mask_3d(
        nx, ny, nz, center_x, center_y, R_inner,
    )
    # Outer wall (solid, static)
    outer_solid = create_outer_wall_mask_3d(
        nx, ny, nz, center_x, center_y, R_outer,
    )
    # Combined solid mask
    solid = inner_solid | outer_solid

    # Missing masks: combined (for bounce-back) and inner-only (for torque)
    mm_inner = compute_missing_mask(inner_solid)
    mm_outer = compute_missing_mask(outer_solid)
    mm = mm_inner | mm_outer

    # Wall velocity: rotating on inner cylinder boundary, zero on outer wall.
    # We compute Omega x r everywhere, then zero it out at nodes that are
    # NOT adjacent to the inner cylinder. Since the outer wall is static,
    # its boundary nodes get zero correction (equivalent to static BB).
    wall_vel_full = compute_cylinder_wall_velocity(
        nx, ny, nz, center_x, center_y, omega_lattice,
    )
    # Mask: True at fluid nodes adjacent to inner cylinder
    inner_boundary = jnp.any(mm_inner, axis=0)  # (nx, ny, nz)
    # Zero wall velocity except at inner-cylinder boundary nodes
    wall_vel = wall_vel_full * inner_boundary[..., None]

    # Initialise at rest
    f = init_equilibrium(nx, ny, nz)

    # Run to steady state
    for _ in range(n_steps):
        f_pre, f_post, rho, u = lbm_step_split(f, tau)
        f = apply_bounce_back(f_post, f_pre, mm, solid, wall_velocity=wall_vel)

    # Compute torque on INNER CYLINDER ONLY via momentum exchange.
    # Using mm_inner (not mm) isolates the inner cylinder's contribution.
    # Using mm would give ~zero net torque since inner + outer cancel at steady state.
    f_pre, f_post, _, _ = lbm_step_split(f, tau)
    f_bb = apply_bounce_back(f_post, f_pre, mm, solid, wall_velocity=wall_vel)
    torque = compute_momentum_exchange_torque(f_pre, f_bb, mm_inner, center)
    torque_z = float(torque[2])

    # Analytical Couette torque per unit length:
    #   T = -4 * pi * nu * Omega * R1^2 * R2^2 / (R2^2 - R1^2)
    # times nz slices (periodic in z, each slice contributes equally)
    R1, R2 = R_inner, R_outer
    torque_analytical = (
        -4.0 * math.pi * nu * omega_lattice
        * R1**2 * R2**2 / (R2**2 - R1**2) * nz
    )

    return torque_z, torque_analytical


# ── Debug resolution tests (32x32, fast) ─────────────────────────────────

class TestCouetteDebug:
    """Quick Couette flow at 32x32 — catches implementation bugs."""

    # Small domain: R_inner=4, R_outer=13 gives ~9 cells in the gap
    R_INNER = 4.0
    R_OUTER = 13.0
    TAU = 0.8
    OMEGA = 0.01
    N_STEPS = 3000

    def test_torque_nonzero(self):
        """Torque should be nonzero for a rotating inner cylinder."""
        torque_z, _ = run_couette_cylinder(
            32, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        assert abs(torque_z) > 0, f"Expected non-zero torque, got {torque_z}"

    def test_torque_sign_consistent(self):
        """Torque sign should be consistent between positive and negative Omega.

        If Omega flips, torque should flip too. We don't enforce a specific
        sign convention (momentum exchange convention may differ from
        analytical), but the physics must be antisymmetric in Omega.
        """
        t_pos, _ = run_couette_cylinder(
            32, self.R_INNER, self.R_OUTER, +self.OMEGA, self.TAU, self.N_STEPS,
        )
        t_neg, _ = run_couette_cylinder(
            32, self.R_INNER, self.R_OUTER, -self.OMEGA, self.TAU, self.N_STEPS,
        )
        assert t_pos * t_neg < 0, (
            f"Torque should flip with Omega: "
            f"+Omega→{t_pos:.4e}, -Omega→{t_neg:.4e}"
        )

    def test_torque_proportional_to_omega(self):
        """Doubling omega should approximately double the torque."""
        t1, _ = run_couette_cylinder(
            32, self.R_INNER, self.R_OUTER,
            0.005, self.TAU, self.N_STEPS,
        )
        t2, _ = run_couette_cylinder(
            32, self.R_INNER, self.R_OUTER,
            0.010, self.TAU, self.N_STEPS,
        )
        ratio = abs(t2 / t1)
        assert 1.5 < ratio < 2.5, f"Torque ratio: {ratio:.2f} (expected ~2.0)"

    def test_torque_order_of_magnitude(self):
        """Torque magnitude should be within 50% of analytical at 32x32."""
        torque_z, torque_ana = run_couette_cylinder(
            32, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        error = abs(abs(torque_z) - abs(torque_ana)) / abs(torque_ana)
        assert error < 0.50, (
            f"Couette torque error {error:.1%} > 50% "
            f"(|sim|={abs(torque_z):.4e}, |ana|={abs(torque_ana):.4e})"
        )


# ── CI resolution tests (64x64, moderate accuracy) ───────────────────────

class TestCouetteCI:
    """Couette flow at 64x64 — quantitative CI validation."""

    # Larger domain: R_inner=8, R_outer=27 gives ~19 cells in the gap
    R_INNER = 8.0
    R_OUTER = 27.0
    TAU = 0.8
    OMEGA = 0.005
    N_STEPS = 5000

    def test_torque_accuracy(self):
        """Couette torque should be within 15% of analytical at 64x64."""
        torque_z, torque_ana = run_couette_cylinder(
            64, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        error = abs(abs(torque_z) - abs(torque_ana)) / abs(torque_ana)
        assert error < 0.15, (
            f"Couette torque error {error:.1%} > 15% "
            f"(|sim|={abs(torque_z):.4e}, |ana|={abs(torque_ana):.4e})"
        )

    def test_torque_R_squared_scaling(self):
        """Torque should scale as R_inner^2 (Couette scaling).

        For fixed R_outer, T ~ R1^2 * R2^2 / (R2^2 - R1^2).
        Doubling R1 from 6 to 12 with R_outer=27:
          ratio = (12^2 * 27^2 / (27^2 - 12^2)) / (6^2 * 27^2 / (27^2 - 6^2))
                = (144 / 585) / (36 / 693) = 0.2462 / 0.05195 = 4.74
        """
        t1, _ = run_couette_cylinder(
            64, 6.0, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        t2, _ = run_couette_cylinder(
            64, 12.0, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        ratio_sim = abs(t2 / t1)
        # Analytical ratio
        R2 = self.R_OUTER
        ratio_ana = (12.0**2 * R2**2 / (R2**2 - 12.0**2)) / (
            6.0**2 * R2**2 / (R2**2 - 6.0**2)
        )
        rel_err = abs(ratio_sim - ratio_ana) / ratio_ana
        assert rel_err < 0.25, (
            f"R^2 scaling ratio: sim={ratio_sim:.2f}, "
            f"ana={ratio_ana:.2f}, error={rel_err:.1%}"
        )


# ── Formal verification benchmark (128x128) ─────────────────────────────

@pytest.mark.slow
@verification_benchmark(
    benchmark_id="MIME-VER-008",
    description="Couette flow torque — D3Q19 bounce-back at 128x128",
    node_type="D3Q19 LBM + bounce-back",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Couette torque error < 5% at 128x128",
    references=("Ladd1994", "Mei1999"),
)
def test_couette_benchmark():
    """MIME-VER-008: Couette flow at validation resolution.

    Inner cylinder (R1=16) rotating inside static outer wall (R2=55)
    on a 128x128x3 grid. Gap = 39 lattice units — well-resolved.

    Analytical reference:
        T = -4*pi*nu*Omega*R1^2*R2^2 / (R2^2 - R1^2) * nz
    """
    torque_z, torque_ana = run_couette_cylinder(
        n_grid=128,
        R_inner=16.0,
        R_outer=55.0,
        omega_lattice=0.002,
        tau=0.8,
        n_steps=10000,
    )
    error = abs(abs(torque_z) - abs(torque_ana)) / abs(torque_ana)
    assert error < 0.05, (
        f"MIME-VER-008 FAIL: Couette torque error {error:.1%} > 5% "
        f"(|sim|={abs(torque_z):.6e}, |ana|={abs(torque_ana):.6e})"
    )
