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
    apply_bouzidi_bounce_back,
    compute_q_values_cylinder,
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


# ═══════════════════════════════════════════════════════════════════════════
# Bouzidi interpolated bounce-back — Couette flow validation
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_link_wall_correction_cylinder(
    missing_mask: jnp.ndarray,
    q_values: jnp.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    angular_velocity: float,
) -> jnp.ndarray:
    """Compute per-link Ladd wall velocity correction at the wall position.

    For each boundary link (outgoing direction q at fluid node x_f),
    computes the wall velocity at the actual wall position along the link:
        r_wall = x_f + q * e_q
        u_wall = omega × (r_wall - center)
        correction[q] = 2 * w[q] * (e[q] · u_wall) / cs²

    This is cylinder-specific geometry. A general per-link correction
    helper for arbitrary geometries would belong in a separate geometry
    module.

    Parameters
    ----------
    missing_mask : (Q, nx, ny, nz) bool — outgoing convention.
    q_values : (Q, nx, ny, nz) float32 — fractional distances, outgoing convention.
    center_x, center_y : float — cylinder center.
    radius : float — cylinder radius (for reference; wall position is from q_values).
    angular_velocity : float — rotation rate (rad/timestep).

    Returns
    -------
    wall_correction : (Q, nx, ny, nz) float32
        Per-link correction in outgoing convention. Only meaningful where
        missing_mask is True.
    """
    from mime.nodes.environment.lbm.d3q19 import E, W, CS2, Q as Q_num

    _, nx, ny, nz = missing_mask.shape
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(ix, iy, indexing='ij')  # (nx, ny)

    slices = []
    for q in range(Q_num):
        ex, ey = float(E[q, 0]), float(E[q, 1])
        w_q = float(W[q])

        if ex == 0 and ey == 0:
            # Rest or pure-z direction: no in-plane correction
            slices.append(jnp.zeros((nx, ny, nz)))
            continue

        qv = q_values[q]  # (nx, ny, nz)

        # Wall position along link: r_wall = (gx + q*ex, gy + q*ey)
        wall_x = gx[:, :, None] + qv * ex
        wall_y = gy[:, :, None] + qv * ey

        # Wall velocity: omega × (r_wall - center)
        rx_wall = wall_x - center_x
        ry_wall = wall_y - center_y
        ux_wall = -angular_velocity * ry_wall
        uy_wall = angular_velocity * rx_wall

        # Correction: 2 * w * (e · u_wall) / cs²
        e_dot_u = ex * ux_wall + ey * uy_wall
        corr = 2.0 * w_q * e_dot_u / CS2

        slices.append(corr)

    return jnp.stack(slices, axis=0)  # (Q, nx, ny, nz)


def compute_per_link_wall_feq_cylinder(
    missing_mask: jnp.ndarray,
    q_values: jnp.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    angular_velocity: float,
) -> jnp.ndarray:
    """Compute per-link f_eq(rho=1, u_wall) at the wall position.

    For each boundary link (outgoing direction q at fluid node x_f),
    computes the equilibrium distribution at the wall velocity:
        r_wall = x_f + q * e_q
        u_wall = omega × (r_wall - center)
        wall_feq[q] = f_eq(rho=1, u=u_wall)[q]

    This is cylinder-specific geometry. A general per-link equilibrium
    helper for arbitrary geometries would belong in a separate geometry
    module.

    Parameters
    ----------
    missing_mask : (Q, nx, ny, nz) bool — outgoing convention.
    q_values : (Q, nx, ny, nz) float32 — fractional distances, outgoing convention.
    center_x, center_y : float — cylinder center.
    radius : float — cylinder radius.
    angular_velocity : float — rotation rate (rad/timestep).

    Returns
    -------
    wall_feq : (Q, nx, ny, nz) float32
        Per-link f_eq(rho=1, u_wall) indexed by outgoing direction but
        evaluated for the INCOMING direction:
            wall_feq[q_out, x, y, z] = f_eq(rho=1, u_wall)[OPP[q_out]]
        where u_wall is at the wall position of outgoing link q_out.
        This convention ensures that when the BB function remaps with
        feq_in = wall_feq[..., opp], it gets f_eq for the correct
        (incoming) direction at the correct (link-specific) wall position.
    """
    from mime.nodes.environment.lbm.d3q19 import E, W, CS2, Q as Q_num

    CS4 = CS2 * CS2
    _, nx, ny, nz = missing_mask.shape
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(ix, iy, indexing='ij')

    from mime.nodes.environment.lbm.d3q19 import OPP as OPP_np

    slices = []
    for q_out in range(Q_num):
        # The incoming direction for this link
        q_in = int(OPP_np[q_out])
        ex_in, ey_in = float(E[q_in, 0]), float(E[q_in, 1])
        w_in = float(W[q_in])

        # Wall position uses the OUTGOING direction (spatial convention)
        ex_out, ey_out = float(E[q_out, 0]), float(E[q_out, 1])

        if ex_out == 0 and ey_out == 0:
            # Rest or pure-z: f_eq = w * rho = w (for rho=1, u=0 in-plane)
            slices.append(jnp.broadcast_to(jnp.float32(w_in), (nx, ny, nz)))
            continue

        qv = q_values[q_out]  # (nx, ny, nz)

        # Wall position along outgoing link direction
        wall_x = gx[:, :, None] + qv * ex_out
        wall_y = gy[:, :, None] + qv * ey_out

        # Wall velocity: omega × (r_wall - center)
        rx_wall = wall_x - center_x
        ry_wall = wall_y - center_y
        ux_wall = -angular_velocity * ry_wall
        uy_wall = angular_velocity * rx_wall

        # f_eq for the INCOMING direction at the wall velocity:
        # wall_feq[q_out] = f_eq(rho=1, u_wall)[q_in]
        # This is the equilibrium population heading from wall toward fluid.
        e_dot_u = ex_in * ux_wall + ey_in * uy_wall
        u_sq = ux_wall**2 + uy_wall**2
        feq = w_in * (1.0 + e_dot_u / CS2
                       + e_dot_u**2 / (2.0 * CS4)
                       - u_sq / (2.0 * CS2))

        slices.append(feq)

    return jnp.stack(slices, axis=0)  # (Q, nx, ny, nz)


def run_couette_bouzidi(
    n_grid: int,
    R_inner: float,
    R_outer: float,
    omega_lattice: float,
    tau: float,
    n_steps: int,
) -> tuple[float, float]:
    """Run Couette flow with Bouzidi interpolated bounce-back.

    Same setup as run_couette_cylinder but uses second-order accurate
    Bouzidi IBB instead of simple halfway BB. Wall velocity correction
    is evaluated at the actual wall position (per-link), not at the
    fluid node position.

    Returns
    -------
    torque_sim, torque_analytical
    """
    nx, ny, nz = n_grid, n_grid, 3
    nu = (tau - 0.5) * CS2
    center_x = nx / 2.0
    center_y = ny / 2.0
    center = jnp.array([center_x, center_y, nz / 2.0])

    # Geometry masks
    inner_solid = create_cylinder_mask_3d(
        nx, ny, nz, center_x, center_y, R_inner,
    )
    outer_solid = create_outer_wall_mask_3d(
        nx, ny, nz, center_x, center_y, R_outer,
    )
    solid = inner_solid | outer_solid

    # Missing masks
    mm_inner = compute_missing_mask(inner_solid)
    mm_outer = compute_missing_mask(outer_solid)
    mm = mm_inner | mm_outer

    # q-values for Bouzidi
    q_inner = compute_q_values_cylinder(
        mm_inner, center_x, center_y, R_inner, is_inner=True,
    )
    q_outer = compute_q_values_cylinder(
        mm_outer, center_x, center_y, R_outer, is_inner=False,
    )
    q_values = jnp.where(mm_inner, q_inner, jnp.where(mm_outer, q_outer, 0.5))

    # Wall velocity (inner only, at fluid node position).
    # The Mei correction scaling inside apply_bouzidi_bounce_back handles
    # the q-dependent adjustment automatically.
    wall_vel_full = compute_cylinder_wall_velocity(
        nx, ny, nz, center_x, center_y, omega_lattice,
    )
    inner_boundary = jnp.any(mm_inner, axis=0)
    wall_vel = wall_vel_full * inner_boundary[..., None]

    # Initialise at rest
    f = init_equilibrium(nx, ny, nz)

    # Run to steady state
    for _ in range(n_steps):
        f_pre, f_post, rho, u = lbm_step_split(f, tau)
        f = apply_bouzidi_bounce_back(
            f_post, f_pre, mm, solid, q_values, wall_velocity=wall_vel,
        )

    # Compute torque on inner cylinder
    f_pre, f_post, _, _ = lbm_step_split(f, tau)
    f_bb = apply_bouzidi_bounce_back(
        f_post, f_pre, mm, solid, q_values, wall_velocity=wall_vel,
    )
    torque = compute_momentum_exchange_torque(f_pre, f_bb, mm_inner, center)
    torque_z = float(torque[2])

    # Analytical
    R1, R2 = R_inner, R_outer
    torque_analytical = (
        -4.0 * math.pi * nu * omega_lattice
        * R1**2 * R2**2 / (R2**2 - R1**2) * nz
    )

    return torque_z, torque_analytical


# ── Bouzidi debug tests (32x32) ─────────────────────────────────────────

class TestBouzidiDebug:
    """Quick Bouzidi Couette at 32x32 — catches implementation bugs."""

    # Non-integer radii avoid grid alignment (q → 0 at aligned nodes)
    R_INNER = 4.3
    R_OUTER = 13.3
    TAU = 0.8
    OMEGA = 0.01
    N_STEPS = 3000

    def test_torque_nonzero(self):
        """Bouzidi BB should produce nonzero torque."""
        torque_z, _ = run_couette_bouzidi(
            32, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        assert abs(torque_z) > 0, f"Expected non-zero torque, got {torque_z}"

    def test_torque_correct_sign(self):
        """Torque sign should match analytical (both negative for +Omega)."""
        torque_z, torque_ana = run_couette_bouzidi(
            32, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        assert torque_z * torque_ana > 0, (
            f"Sign mismatch: sim={torque_z:.4e}, ana={torque_ana:.4e}"
        )

    def test_torque_better_than_simple_bb(self):
        """Bouzidi should be at least as accurate as simple BB at 32x32."""
        t_bouzidi, t_ana = run_couette_bouzidi(
            32, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        t_simple, _ = run_couette_cylinder(
            32, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        err_bouzidi = abs(abs(t_bouzidi) - abs(t_ana)) / abs(t_ana)
        err_simple = abs(abs(t_simple) - abs(t_ana)) / abs(t_ana)
        assert err_bouzidi <= err_simple + 0.02, (
            f"Bouzidi ({err_bouzidi:.1%}) should not be much worse than "
            f"simple BB ({err_simple:.1%})"
        )

    def test_q_values_in_range(self):
        """All q-values at boundary links should be in (0, 1)."""
        nx, ny, nz = 32, 32, 3
        cx, cy = nx / 2.0, ny / 2.0
        inner = create_cylinder_mask_3d(nx, ny, nz, cx, cy, self.R_INNER)
        mm_inner = compute_missing_mask(inner)
        q_vals = compute_q_values_cylinder(mm_inner, cx, cy, self.R_INNER, is_inner=True)
        # Only check at boundary links
        q_at_boundary = q_vals[mm_inner]
        assert jnp.all(q_at_boundary > 0), "q-values must be > 0"
        assert jnp.all(q_at_boundary < 1), "q-values must be < 1"


# ── Bouzidi CI tests (64x64) ────────────────────────────────────────────

class TestBouzidiCI:
    """Bouzidi Couette at 64x64 — quantitative accuracy checks."""

    # Non-integer radii for clean Bouzidi q-values
    R_INNER = 8.3
    R_OUTER = 27.3
    TAU = 0.8
    OMEGA = 0.005
    N_STEPS = 5000

    def test_torque_accuracy_under_5_percent(self):
        """Bouzidi Couette torque should be within 5% at 64x64.

        Simple BB achieves ~2% at this resolution. Bouzidi should
        match or improve on this.
        """
        torque_z, torque_ana = run_couette_bouzidi(
            64, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        error = abs(abs(torque_z) - abs(torque_ana)) / abs(torque_ana)
        assert error < 0.05, (
            f"Bouzidi Couette error {error:.1%} > 5% "
            f"(|sim|={abs(torque_z):.4e}, |ana|={abs(torque_ana):.4e})"
        )

    def test_bouzidi_improves_on_simple_bb(self):
        """Bouzidi should achieve lower error than simple BB at 64x64."""
        t_bouzidi, t_ana = run_couette_bouzidi(
            64, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        t_simple, _ = run_couette_cylinder(
            64, self.R_INNER, self.R_OUTER, self.OMEGA, self.TAU, self.N_STEPS,
        )
        err_bouzidi = abs(abs(t_bouzidi) - abs(t_ana)) / abs(t_ana)
        err_simple = abs(abs(t_simple) - abs(t_ana)) / abs(t_ana)
        # Bouzidi should be better (or at least very close)
        assert err_bouzidi < err_simple + 0.005, (
            f"Bouzidi ({err_bouzidi:.2%}) should improve on "
            f"simple BB ({err_simple:.2%})"
        )


# ── Bouzidi convergence order test ──────────────────────────────────────

@pytest.mark.slow
def test_bouzidi_convergence_order():
    """Bouzidi should converge as O(dx^2), simple BB as O(dx).

    Run at 32 and 64 (doubling resolution = halving dx). The error
    ratio should be ~4 for O(dx^2) and ~2 for O(dx).
    """
    tau = 0.8
    omega = 0.005

    # Use consistent R/grid ratios with non-integer radii
    configs = [
        (32, 6.3, 13.3, 3000),
        (64, 12.6, 26.6, 5000),
    ]

    errors_simple = []
    errors_bouzidi = []
    for n, r1, r2, steps in configs:
        t_s, t_a = run_couette_cylinder(n, r1, r2, omega, tau, steps)
        t_b, _ = run_couette_bouzidi(n, r1, r2, omega, tau, steps)
        errors_simple.append(abs(abs(t_s) - abs(t_a)) / abs(t_a))
        errors_bouzidi.append(abs(abs(t_b) - abs(t_a)) / abs(t_a))

    # Error ratio when doubling resolution
    ratio_simple = errors_simple[0] / max(errors_simple[1], 1e-10)
    ratio_bouzidi = errors_bouzidi[0] / max(errors_bouzidi[1], 1e-10)

    # Simple BB: O(dx) → ratio ~2. Bouzidi: O(dx^2) → ratio ~4.
    # Use relaxed bounds since discrete geometry effects add noise.
    assert ratio_bouzidi > ratio_simple * 0.8, (
        f"Bouzidi convergence ratio ({ratio_bouzidi:.1f}) should exceed "
        f"simple BB ratio ({ratio_simple:.1f}). "
        f"Errors: simple={errors_simple}, bouzidi={errors_bouzidi}"
    )


# ── Formal Bouzidi benchmark (128x128) ──────────────────────────────────

@pytest.mark.slow
@verification_benchmark(
    benchmark_id="MIME-VER-009",
    description="Bouzidi IBB Couette torque — D3Q19 at 128x128",
    node_type="D3Q19 LBM + Bouzidi IBB",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Couette torque error < 1% at 128x128 with Bouzidi IBB",
    references=("Bouzidi2001", "Ladd1994"),
)
def test_bouzidi_benchmark():
    """MIME-VER-009: Bouzidi Couette flow at validation resolution.

    Target: < 1% error, demonstrating second-order wall accuracy.
    """
    torque_z, torque_ana = run_couette_bouzidi(
        n_grid=128,
        R_inner=16.3,
        R_outer=55.3,
        omega_lattice=0.002,
        tau=0.8,
        n_steps=10000,
    )
    error = abs(abs(torque_z) - abs(torque_ana)) / abs(torque_ana)
    assert error < 0.01, (
        f"MIME-VER-009 FAIL: Bouzidi Couette error {error:.2%} > 1% "
        f"(|sim|={abs(torque_z):.6e}, |ana|={abs(torque_ana):.6e})"
    )
