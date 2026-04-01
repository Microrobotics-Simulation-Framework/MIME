"""Immersed boundary operators for BEM-LBM Schwarz coupling.

Implements Peskin's 4-point discrete delta function and the
force spreading / velocity interpolation operators for transferring
data between Lagrangian (BEM surface) and Eulerian (LBM grid) meshes.

The spreading operator distributes Lagrangian point forces onto the
Eulerian grid as a body force density (for Guo forcing in the LBM).
The interpolation operator samples the Eulerian velocity field at
Lagrangian points (for BEM background flow).

Both operators use precomputed stencils: the delta function weights
and grid indices are computed once at init time (fixed Lagrangian
positions) and reused every Schwarz iteration. This avoids JIT
retracing and ensures O(64 × N_lag) cost per operation.

References:
    Peskin (2002), Acta Numerica 11:479-517 — Eq. 6.27 (delta), 4.17/4.19
    Tian et al. (2011), J. Comput. Phys. 230:7266-7283 — Eqs. 9, 13
    vivsim (github.com/haimingz/vivsim) — JAX IB implementation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def peskin_delta_4pt(r: jnp.ndarray) -> jnp.ndarray:
    """Peskin 4-point discrete delta kernel (1D).

    φ(r) = (3 - 2|r| + √(1 + 4|r| - 4r²)) / 8   for |r| < 1
    φ(r) = (5 - 2|r| - √(-7 + 12|r| - 4r²)) / 8  for 1 ≤ |r| < 2
    φ(r) = 0                                        for |r| ≥ 2

    Reference: Peskin (2002) Eq. 6.27
    """
    abs_r = jnp.abs(r)
    # Clamp arguments to sqrt to avoid NaN in JIT (branches not taken)
    inner_arg = jnp.maximum(1.0 + 4.0 * abs_r - 4.0 * abs_r**2, 0.0)
    outer_arg = jnp.maximum(-7.0 + 12.0 * abs_r - 4.0 * abs_r**2, 0.0)

    phi_inner = (3.0 - 2.0 * abs_r + jnp.sqrt(inner_arg)) * 0.125
    phi_outer = (5.0 - 2.0 * abs_r - jnp.sqrt(outer_arg)) * 0.125

    return jnp.where(
        abs_r >= 2.0, 0.0,
        jnp.where(abs_r < 1.0, phi_inner, phi_outer),
    )


def precompute_ib_stencil(
    lagrangian_points: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute IB stencil indices and delta weights.

    For each Lagrangian point, identifies the 4×4×4 = 64 Eulerian
    nodes in its support and computes the tensor-product delta weights.

    Call once at init time. The returned arrays are static and can be
    passed directly to spread_forces / interpolate_velocity.

    Parameters
    ----------
    lagrangian_points : (N_lag, 3) float64
        Lagrangian positions in lattice units (not physical).
    grid_shape : (nx, ny, nz)

    Returns
    -------
    stencil_indices : (N_lag, 64) int32
        Flat indices into the (nx, ny, nz) grid for each stencil node.
    stencil_weights : (N_lag, 64) float64
        Tensor-product delta weights δ_h(x - X_k) for each stencil node.
    """
    N_lag = len(lagrangian_points)
    nx, ny, nz = grid_shape

    # Stencil offsets: -1, 0, 1, 2 (4-point support centered on floor)
    offsets_1d = np.array([-1, 0, 1, 2], dtype=np.int32)

    # 3D tensor product: 4³ = 64 offset combinations
    di, dj, dk = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
    di = di.ravel()  # (64,)
    dj = dj.ravel()
    dk = dk.ravel()

    # Reference node (floor of each Lagrangian position)
    ref = np.floor(lagrangian_points).astype(np.int32)  # (N_lag, 3)

    # Stencil grid indices with periodic wrapping
    si = (ref[:, 0:1] + di[None, :]) % nx  # (N_lag, 64)
    sj = (ref[:, 1:2] + dj[None, :]) % ny
    sk = (ref[:, 2:3] + dk[None, :]) % nz

    stencil_indices = (si * ny * nz + sj * nz + sk).astype(np.int32)

    # Delta weights: tensor product of 1D deltas
    # Distance from each Lagrangian point to each stencil node
    rx = lagrangian_points[:, 0:1] - (ref[:, 0:1] + di[None, :])  # (N_lag, 64)
    ry = lagrangian_points[:, 1:2] - (ref[:, 1:2] + dj[None, :])
    rz = lagrangian_points[:, 2:3] - (ref[:, 2:3] + dk[None, :])

    # Evaluate 1D Peskin delta (numpy for init-time computation)
    def _phi(r):
        abs_r = np.abs(r)
        inner = np.maximum(1.0 + 4.0 * abs_r - 4.0 * abs_r**2, 0.0)
        outer = np.maximum(-7.0 + 12.0 * abs_r - 4.0 * abs_r**2, 0.0)
        phi_in = (3.0 - 2.0 * abs_r + np.sqrt(inner)) * 0.125
        phi_out = (5.0 - 2.0 * abs_r - np.sqrt(outer)) * 0.125
        return np.where(abs_r >= 2.0, 0.0, np.where(abs_r < 1.0, phi_in, phi_out))

    wx = _phi(rx)  # (N_lag, 64)
    wy = _phi(ry)
    wz = _phi(rz)
    stencil_weights = (wx * wy * wz).astype(np.float64)

    return stencil_indices, stencil_weights


def spread_forces(
    lagrangian_forces: jnp.ndarray,
    stencil_indices: jnp.ndarray,
    stencil_weights: jnp.ndarray,
    grid_shape: tuple[int, int, int],
) -> jnp.ndarray:
    """Spread Lagrangian forces to Eulerian grid via precomputed stencil.

    g(x) = Σ_k F_k · δ_h(x - X_k)

    Parameters
    ----------
    lagrangian_forces : (N_lag, 3) point forces in lattice units
    stencil_indices : (N_lag, 64) from precompute_ib_stencil
    stencil_weights : (N_lag, 64) from precompute_ib_stencil
    grid_shape : (nx, ny, nz)

    Returns
    -------
    force_field : (nx, ny, nz, 3) Eulerian force density
    """
    nx, ny, nz = grid_shape
    N_lag = len(lagrangian_forces)

    # Weighted forces at each stencil point: (N_lag, 64, 3)
    weighted = stencil_weights[..., None] * lagrangian_forces[:, None, :]

    # Flatten grid for scatter-add
    flat_force = jnp.zeros((nx * ny * nz, 3))

    # Scatter-add all stencil contributions
    flat_indices = stencil_indices.ravel()  # (N_lag * 64,)
    flat_values = weighted.reshape(-1, 3)   # (N_lag * 64, 3)

    flat_force = flat_force.at[flat_indices].add(flat_values)

    return flat_force.reshape(nx, ny, nz, 3)


def interpolate_velocity(
    velocity_field: jnp.ndarray,
    stencil_indices: jnp.ndarray,
    stencil_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate Eulerian velocity to Lagrangian points via precomputed stencil.

    U(X_k) = Σ_x u(x) · δ_h(x - X_k) · h³

    In lattice units h=1, so the h³ factor is 1.

    Parameters
    ----------
    velocity_field : (nx, ny, nz, 3)
    stencil_indices : (N_lag, 64) from precompute_ib_stencil
    stencil_weights : (N_lag, 64) from precompute_ib_stencil

    Returns
    -------
    velocity_at_points : (N_lag, 3)
    """
    flat_vel = velocity_field.reshape(-1, 3)  # (nx*ny*nz, 3)

    # Gather velocity at stencil nodes: (N_lag, 64, 3)
    stencil_vel = flat_vel[stencil_indices]

    # Weighted sum: (N_lag, 3)
    return jnp.sum(stencil_weights[..., None] * stencil_vel, axis=1)
