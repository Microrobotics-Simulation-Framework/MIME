"""Precomputed lookup table for the cylindrical wall Green's function.

The image field G_image(x, x₀) inside an infinite no-slip cylinder
depends on only four cylindrical scalars: (ρ, ρ₀, Δφ, Δz). This
module precomputes G_image on a regular (or tanh-clustered) grid and
provides fast interpolated assembly for arbitrary body meshes.

Workflow:
    table = precompute_wall_table(R_cyl, mu)
    save_wall_table(table, "wall_R2.0.npz")
    # later:
    table = load_wall_table("wall_R2.0.npz")
    G_wall = assemble_image_correction_matrix_from_table(
        body_pts, body_wts, R_cyl, mu, table)
    A_confined = A_body_BEM + G_wall
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ── Parity table for G_image in cylindrical coords ────────────────────
# (Δφ_parity, Δz_parity): +1 = even, -1 = odd
# Verified empirically against direct Fourier-Bessel code.

PARITY_DPHI = np.array([
    [+1, -1, +1],   # alpha=rho
    [-1, +1, -1],   # alpha=phi
    [+1, -1, +1],   # alpha=z
])

PARITY_DZ = np.array([
    [+1, +1, -1],
    [+1, +1, -1],
    [-1, -1, +1],
])


# ── Data structure ─────────────────────────────────────────────────────

@dataclass
class WallTable:
    """Precomputed cylindrical wall Green's function image tensor."""
    R_cyl: float
    mu: float
    rho_grid: np.ndarray      # (n_rho,) values in [ε, R_cyl - ε]
    dphi_grid: np.ndarray     # (n_dphi,) values in [0, π]
    dz_grid: np.ndarray       # (n_dz,) values in [0, L_max]
    data: np.ndarray          # (n_rho, n_rho, n_dphi, n_dz, 3, 3)


def _compute_slice(args, params):
    """Worker for one (ρ_target, ρ_source) table slice. Module-level for pickling."""
    from .cylinder_greens_function_v2 import _assemble_image_only, _rotation_matrices

    ir, is_, rho_t, rho_s = args
    dphi_flat = params['dphi_flat']
    dz_flat = params['dz_flat']
    n_dphi = params['n_dphi']
    n_dz = params['n_dz']

    tgt = np.column_stack([
        rho_t * np.cos(dphi_flat),
        rho_t * np.sin(dphi_flat),
        dz_flat,
    ])
    src = np.array([[rho_s, 0.0, 0.0]])

    G_img = _assemble_image_only(
        tgt, src, params['R_cyl'], params['mu'],
        n_max=params['n_max'], n_k=params['n_k'], n_phi=params['n_phi'],
    )

    # Rotate Cartesian → cylindrical at each target
    R_tgts = _rotation_matrices(dphi_flat)
    n_pts = len(dphi_flat)

    result = np.zeros((n_dphi, n_dz, 3, 3))
    for p in range(n_pts):
        G_cart_p = G_img[3*p:3*p+3, :]
        result[p // n_dz, p % n_dz] = R_tgts[p] @ G_cart_p

    return ir, is_, result


def save_wall_table(table: WallTable, path: str):
    np.savez_compressed(path,
                        R_cyl=table.R_cyl, mu=table.mu,
                        rho_grid=table.rho_grid,
                        dphi_grid=table.dphi_grid,
                        dz_grid=table.dz_grid,
                        data=table.data)


def load_wall_table(path: str) -> WallTable:
    d = np.load(path)
    return WallTable(
        R_cyl=float(d['R_cyl']), mu=float(d['mu']),
        rho_grid=d['rho_grid'], dphi_grid=d['dphi_grid'],
        dz_grid=d['dz_grid'], data=d['data'])


# ── Table precomputation ──────────────────────────────────────────────

def precompute_wall_table(
    R_cyl: float,
    mu: float,
    n_rho: int = 30,
    n_dphi: int = 64,
    n_dz: int = 128,
    L_max_factor: float = 5.0,
    n_max: int = 15,
    n_k: int = 80,
    n_phi: int = 64,
    n_jobs: int = 0,
) -> WallTable:
    """Precompute the 4D wall image Green's function table.

    Parameters
    ----------
    R_cyl : cylinder radius
    mu : viscosity
    n_rho : grid points in ρ and ρ₀, range [ε, R_cyl - ε]
    n_dphi : grid points in Δφ, range [0, π]
    n_dz : grid points in Δz, range [0, L_max] (tanh-clustered)
    L_max_factor : L_max = L_max_factor × R_cyl
    n_max, n_k, n_phi : Fourier-Bessel parameters
    n_jobs : parallel workers (0 = os.cpu_count())

    Returns
    -------
    WallTable
    """
    from .cylinder_greens_function_v2 import _assemble_image_only

    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    # ── Grid construction ─────────────────────────────────────────
    eps_rho = 0.02 * R_cyl
    rho_grid = np.linspace(eps_rho, R_cyl - eps_rho, n_rho)

    # Δφ: uniform on [0, π] (the image varies smoothly in φ)
    dphi_grid = np.linspace(0, np.pi, n_dphi)

    # Δz: tanh-clustered so ~half the points cover |Δz| < R_cyl
    L_max = L_max_factor * R_cyl
    beta_z = 2.0  # clustering strength
    s = np.linspace(0, 1, n_dz)
    dz_grid = L_max * np.tanh(beta_z * s) / np.tanh(beta_z)

    # ── Prepare target grid for one (ρ, ρ₀) slice ────────────────
    # Targets at (ρ, Δφ_j, Δz_k) in Cartesian:
    #   x = ρ cos(Δφ), y = ρ sin(Δφ), z = Δz
    # Source at (ρ₀, 0, 0)
    DPHI, DZ = np.meshgrid(dphi_grid, dz_grid, indexing='ij')
    dphi_flat = DPHI.ravel()  # (n_dphi × n_dz,)
    dz_flat = DZ.ravel()

    n_pts_per_slice = len(dphi_flat)

    # Allocate output
    data = np.zeros((n_rho, n_rho, n_dphi, n_dz, 3, 3))

    # ── Build work list ───────────────────────────────────────────
    work = []
    for ir in range(n_rho):
        for is_ in range(n_rho):
            work.append((ir, is_, rho_grid[ir], rho_grid[is_]))

    # Shared parameters for worker
    slice_params = dict(
        dphi_flat=dphi_flat, dz_flat=dz_flat,
        n_dphi=n_dphi, n_dz=n_dz,
        R_cyl=R_cyl, mu=mu,
        n_max=n_max, n_k=n_k, n_phi=n_phi,
    )

    # ── Execute (parallel or serial) ──────────────────────────────
    if n_jobs == 1:
        for idx, item in enumerate(work):
            ir, is_, result = _compute_slice(item, slice_params)
            data[ir, is_] = result
    else:
        from multiprocessing import Pool
        # Use starmap with (args, params) tuples for reliable pickling
        work_with_params = [(item, slice_params) for item in work]
        with Pool(n_jobs) as pool:
            results = pool.starmap(_compute_slice, work_with_params)
        for ir, is_, result in results:
            data[ir, is_] = result

    return WallTable(R_cyl=R_cyl, mu=mu,
                     rho_grid=rho_grid, dphi_grid=dphi_grid,
                     dz_grid=dz_grid, data=data)


# ── Table-interpolated assembly ───────────────────────────────────────

def assemble_image_correction_matrix_from_table(
    body_pts: np.ndarray,
    body_wts: np.ndarray,
    R_cyl: float,
    mu: float,
    table: WallTable,
) -> np.ndarray:
    """Assemble (3N, 3N) wall correction matrix by table interpolation.

    Drop-in replacement for ``assemble_image_correction_matrix`` from
    ``cylinder_greens_function_v2``. Same inputs, same output shape.
    """
    from .cylinder_greens_function_v2 import _cart_to_cyl, _rotation_matrices

    N = len(body_pts)
    rho, phi, z = _cart_to_cyl(body_pts)

    # All (target, source) pairs
    rho_i = rho[:, None].repeat(N, axis=1)          # (N, N)
    rho_j = rho[None, :].repeat(N, axis=0)
    dphi_raw = phi[:, None] - phi[None, :]
    dz_raw = z[:, None] - z[None, :]

    # Map to half-domain with sign tracking
    dphi_mod = np.mod(dphi_raw + np.pi, 2 * np.pi) - np.pi
    dphi_sign = np.sign(dphi_mod)
    dphi_sign[dphi_sign == 0] = 1.0
    dphi_abs = np.abs(dphi_mod)

    dz_sign = np.sign(dz_raw)
    dz_sign[dz_sign == 0] = 1.0
    dz_abs = np.abs(dz_raw)

    # ── Fast 4D interpolation (manual, vectorized) ────────────────
    # For each axis: find bracket indices and weights
    def _interp_weights(grid, vals_flat):
        """Find indices and lerp weights for 1D interpolation."""
        idx = np.searchsorted(grid, vals_flat, side='right') - 1
        idx = np.clip(idx, 0, len(grid) - 2)
        lo = grid[idx]
        hi = grid[idx + 1]
        w = np.where(hi > lo, (vals_flat - lo) / (hi - lo), 0.0)
        w = np.clip(w, 0.0, 1.0)
        return idx, w

    shape = rho_i.shape  # (N, N)
    i0, w0 = _interp_weights(table.rho_grid, rho_i.ravel())
    i1, w1 = _interp_weights(table.rho_grid, rho_j.ravel())
    i2, w2 = _interp_weights(table.dphi_grid, dphi_abs.ravel())
    i3, w3 = _interp_weights(table.dz_grid, dz_abs.ravel())

    # 4D trilinear: 16 corners, weighted sum
    # data shape: (n_rho, n_rho, n_dphi, n_dz, 3, 3)
    data = table.data
    G_cyl_flat = np.zeros((N * N, 3, 3))

    for d0 in range(2):
        for d1 in range(2):
            for d2 in range(2):
                for d3 in range(2):
                    wt = ((1 - w0) if d0 == 0 else w0) * \
                         ((1 - w1) if d1 == 0 else w1) * \
                         ((1 - w2) if d2 == 0 else w2) * \
                         ((1 - w3) if d3 == 0 else w3)
                    G_cyl_flat += wt[:, None, None] * data[
                        i0 + d0, i1 + d1, i2 + d2, i3 + d3]

    # Apply parity signs
    dphi_s = dphi_sign.ravel()
    dz_s = dz_sign.ravel()
    for a in range(3):
        for b in range(3):
            sign = np.ones(N * N)
            if PARITY_DPHI[a, b] == -1:
                sign *= dphi_s
            if PARITY_DZ[a, b] == -1:
                sign *= dz_s
            G_cyl_flat[:, a, b] *= sign

    G_cyl_2d = G_cyl_flat.reshape(N, N, 3, 3)

    # Rotate cylindrical → Cartesian: G_cart = R_tgt^T @ G_cyl @ R_src
    R_tgt = _rotation_matrices(phi)
    R_src = _rotation_matrices(phi)
    R_tgt_T = R_tgt.transpose(0, 2, 1)

    G_tmp = np.matmul(G_cyl_2d, R_src[np.newaxis, :, :, :])
    G_cart = np.matmul(R_tgt_T[:, np.newaxis, :, :], G_tmp)

    # Apply BEM quadrature weights (per source)
    wts = np.asarray(body_wts)
    G_cart *= wts[np.newaxis, :, np.newaxis, np.newaxis]

    return G_cart.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
