"""Immersed Boundary (IB) method — Multi-Direct Forcing with Peskin delta.

Couples Lagrangian boundary points (robot surface) to the Eulerian
fluid grid (LBM lattice). Uses the Multi-Direct Forcing (MDF) method
(Luo et al. 2007) which iteratively corrects the velocity at boundary
points for better no-slip enforcement than single-step penalty methods.

Two core operations (adjoint pair):
1. **Interpolation**: u_markers = einsum("dxy,nxy->nd", u, kernels)
2. **Spreading**:     f_grid   = einsum("nd,nxy->dxy", f_markers, kernels)

The delta function kernels are precomputed once per timestep (when
boundary points move) and reused across MDF iterations.

Inspired by vivsim (Zhang et al.) for the einsum-based kernel patterns.

All operations are pure JAX, differentiable, and JIT-compilable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Delta function kernels ──────────────────────────────────────────────

def peskin_phi_4(r: jnp.ndarray) -> jnp.ndarray:
    """Peskin 4-point regularised delta function kernel (range ±2).

    phi(r) = (1/4)(1 + cos(pi*r/2))  for |r| <= 2
    phi(r) = 0                         for |r| > 2

    Smooth, non-negative, integrates to 1.

    Parameters
    ----------
    r : jnp.ndarray
        Normalised distance (x - X) / h.

    Returns
    -------
    phi : jnp.ndarray, same shape as r.
    """
    abs_r = jnp.abs(r)
    phi = 0.25 * (1.0 + jnp.cos(jnp.pi * abs_r / 2.0))
    return jnp.where(abs_r <= 2.0, phi, 0.0)


def peskin_phi_3(r: jnp.ndarray) -> jnp.ndarray:
    """3-point delta kernel (range ±1.5). More compact support.

    phi(r) = (1/3)(1 + sqrt(1 - 3r^2))      for |r| <= 0.5
    phi(r) = (1/6)(5 - 3|r| - sqrt(-2 + 6|r| - 3r^2))  for 0.5 < |r| <= 1.5
    phi(r) = 0                                 for |r| > 1.5
    """
    abs_r = jnp.abs(r)
    r2 = r * r

    inner = (1.0 / 3.0) * (1.0 + jnp.sqrt(jnp.maximum(1.0 - 3.0 * r2, 0.0)))
    outer = (1.0 / 6.0) * (5.0 - 3.0 * abs_r - jnp.sqrt(
        jnp.maximum(-2.0 + 6.0 * abs_r - 3.0 * r2, 0.0)
    ))

    phi = jnp.where(abs_r <= 0.5, inner,
          jnp.where(abs_r <= 1.5, outer, 0.0))
    return phi


# ── Kernel precomputation ───────────────────────────────────────────────

def compute_kernels(
    boundary_points: jnp.ndarray,
    nx: int,
    ny: int,
    h: float = 1.0,
    kernel_fn=peskin_phi_4,
) -> jnp.ndarray:
    """Precompute delta function kernels for all boundary points.

    Returns a (N, nx, ny) array of kernel weights. This is computed
    once per timestep when boundary points move, then reused across
    MDF iterations.

    Parameters
    ----------
    boundary_points : (N, 2)
        Lagrangian point positions in lattice units.
    nx, ny : int
        Eulerian grid dimensions.
    h : float
        Lattice spacing.
    kernel_fn : callable
        Delta function kernel (peskin_phi_4 or peskin_phi_3).

    Returns
    -------
    kernels : (N, nx, ny) float32
    """
    x_grid = jnp.arange(nx, dtype=jnp.float32)
    y_grid = jnp.arange(ny, dtype=jnp.float32)
    # (nx, ny) grids
    xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')

    def kernel_for_one_point(point):
        rx = (xx - point[0]) / h
        ry = (yy - point[1]) / h
        return kernel_fn(rx) * kernel_fn(ry)

    # vmap over boundary points: (N, nx, ny)
    return jax.vmap(kernel_for_one_point)(boundary_points)


# ── Interpolation (Eulerian → Lagrangian) ────────────────────────────────

def interpolate_velocity(
    velocity: jnp.ndarray,
    kernels: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate fluid velocity at Lagrangian boundary points.

    u_markers = einsum("dxy,nxy->nd", u, kernels)

    Parameters
    ----------
    velocity : (nx, ny, 2)
        Eulerian fluid velocity field.
    kernels : (N, nx, ny)
        Precomputed delta function kernels.

    Returns
    -------
    u_markers : (N, 2)
        Interpolated velocity at each boundary point.
    """
    # Transpose velocity to (2, nx, ny) for einsum
    u = jnp.moveaxis(velocity, -1, 0)  # (2, nx, ny)
    return jnp.einsum("dxy,nxy->nd", u, kernels)


# ── Force spreading (Lagrangian → Eulerian) ──────────────────────────────

def spread_force(
    marker_forces: jnp.ndarray,
    kernels: jnp.ndarray,
    ds: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Spread Lagrangian forces onto the Eulerian grid.

    f_grid = einsum("nd,nxy->dxy", f_markers * ds, kernels)

    Parameters
    ----------
    marker_forces : (N, 2)
        Forces at boundary points.
    kernels : (N, nx, ny)
        Precomputed delta function kernels.
    ds : (N,) or float
        Arc length element per boundary point.

    Returns
    -------
    force_field : (nx, ny, 2)
        Eulerian force density.
    """
    if isinstance(ds, (int, float)):
        scaled_forces = marker_forces * ds
    else:
        scaled_forces = marker_forces * ds[:, None]

    # einsum: (N, 2) x (N, nx, ny) -> (2, nx, ny)
    f_grid = jnp.einsum("nd,nxy->dxy", scaled_forces, kernels)
    # Transpose back to (nx, ny, 2)
    return jnp.moveaxis(f_grid, 0, -1)


# ── Multi-Direct Forcing ─────────────────────────────────────────────────

def multi_direct_forcing(
    velocity: jnp.ndarray,
    kernels: jnp.ndarray,
    target_velocities: jnp.ndarray,
    ds: jnp.ndarray | float = 1.0,
    n_iter: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-Direct Forcing IB method (Luo et al. 2007).

    Iteratively corrects the fluid velocity at boundary points to
    enforce the no-slip condition. Much better convergence than
    single-step penalty methods.

    Algorithm per iteration:
        1. Interpolate fluid velocity at markers: u_markers
        2. Compute correction: g = (v_target - u_markers) * 2 * ds
        3. Spread correction onto grid
        4. Update velocity: u += correction / rho

    Parameters
    ----------
    velocity : (nx, ny, 2)
        Current fluid velocity field.
    kernels : (N, nx, ny)
        Precomputed delta function kernels.
    target_velocities : (N, 2)
        Desired velocity at boundary points (from rigid body).
    ds : (N,) or float
        Arc length element.
    n_iter : int
        Number of MDF iterations (default 5).

    Returns
    -------
    corrected_velocity : (nx, ny, 2)
        Velocity field with IB correction applied.
    total_force : (nx, ny, 2)
        Total IB body force spread onto the grid.
    """
    def mdf_iteration(carry, _):
        u, total_f = carry

        # Interpolate current velocity at markers
        u_markers = interpolate_velocity(u, kernels)

        # Velocity deficit -> correction force
        g_markers = (target_velocities - u_markers) * 2.0

        # Spread correction onto grid
        g_grid = spread_force(g_markers, kernels, ds)

        # Update velocity (assuming rho ≈ 1)
        u = u + g_grid * 0.5
        total_f = total_f + g_grid

        return (u, total_f), None

    total_f_init = jnp.zeros_like(velocity)
    (corrected_u, total_f), _ = jax.lax.scan(
        mdf_iteration,
        (velocity, total_f_init),
        None,
        length=n_iter,
    )

    return corrected_u, total_f


# ── Drag force / torque computation ──────────────────────────────────────

def compute_marker_forces(
    velocity: jnp.ndarray,
    kernels: jnp.ndarray,
    target_velocities: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the IB force at each marker point.

    F_k = 2 * (v_target_k - u_interpolated_k)

    Parameters
    ----------
    velocity : (nx, ny, 2)
    kernels : (N, nx, ny)
    target_velocities : (N, 2)

    Returns
    -------
    marker_forces : (N, 2)
    """
    u_markers = interpolate_velocity(velocity, kernels)
    return 2.0 * (target_velocities - u_markers)


def compute_drag_force(marker_forces: jnp.ndarray, ds: float = 1.0) -> jnp.ndarray:
    """Total drag force on body = -sum(F_k * ds) (Newton's third law).

    Parameters
    ----------
    marker_forces : (N, 2)
    ds : float

    Returns
    -------
    drag : (2,)
    """
    return -jnp.sum(marker_forces * ds, axis=0)


def compute_drag_torque(
    marker_forces: jnp.ndarray,
    boundary_points: jnp.ndarray,
    center: jnp.ndarray,
    ds: float = 1.0,
) -> jnp.ndarray:
    """Total drag torque on body (2D scalar).

    T = -sum(r_k x F_k * ds)

    Parameters
    ----------
    marker_forces : (N, 2)
    boundary_points : (N, 2)
    center : (2,)
    ds : float

    Returns
    -------
    torque : scalar
    """
    r = boundary_points - center[None, :]
    cross = r[:, 0] * marker_forces[:, 1] - r[:, 1] * marker_forces[:, 0]
    return -jnp.sum(cross * ds)


# ── Geometry helpers ─────────────────────────────────────────────────────

def generate_circle_points(
    center_x: float,
    center_y: float,
    radius: float,
    n_points: int,
) -> tuple[jnp.ndarray, float]:
    """Generate evenly-spaced Lagrangian points on a circle.

    Returns
    -------
    points : (n_points, 2)
    ds : float
        Arc length element = 2*pi*radius / n_points.
    """
    theta = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
    x = center_x + radius * jnp.cos(theta)
    y = center_y + radius * jnp.sin(theta)
    ds = 2.0 * jnp.pi * radius / n_points
    return jnp.stack([x, y], axis=-1), float(ds)


def compute_boundary_velocities(
    center_velocity: jnp.ndarray,
    angular_velocity: float,
    boundary_points: jnp.ndarray,
    center: jnp.ndarray,
) -> jnp.ndarray:
    """Target velocities for a rigid body: U_k = V + omega x r_k.

    Parameters
    ----------
    center_velocity : (2,)
    angular_velocity : float (scalar, positive = CCW)
    boundary_points : (N, 2)
    center : (2,)

    Returns
    -------
    target_velocities : (N, 2)
    """
    r = boundary_points - center[None, :]
    omega_cross_r = angular_velocity * jnp.stack([-r[:, 1], r[:, 0]], axis=-1)
    return center_velocity[None, :] + omega_cross_r
