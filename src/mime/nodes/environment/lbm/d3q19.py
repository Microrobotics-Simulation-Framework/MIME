"""D3Q19 Lattice Boltzmann Method — 3D core operations.

Extends the D2Q9 patterns to 3D. All functions are pure JAX,
fully JIT-compilable and differentiable.

Lattice units: dx = dt_lbm = 1.
    nu = (tau - 0.5) / 3

Conventions:
    - State shape: (nx, ny, nz, 19) for distribution functions
    - Density shape: (nx, ny, nz)
    - Velocity shape: (nx, ny, nz, 3)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ── D3Q19 lattice constants ─────────────────────────────────────────────

# 19 velocity vectors: (ex, ey, ez)
E = np.array([
    [0,  0,  0],   # 0: rest
    [1,  0,  0],   # 1: +x
    [-1, 0,  0],   # 2: -x
    [0,  1,  0],   # 3: +y
    [0, -1,  0],   # 4: -y
    [0,  0,  1],   # 5: +z
    [0,  0, -1],   # 6: -z
    [1,  1,  0],   # 7: +x+y
    [-1, 1,  0],   # 8: -x+y
    [1, -1,  0],   # 9: +x-y
    [-1,-1,  0],   # 10: -x-y
    [1,  0,  1],   # 11: +x+z
    [-1, 0,  1],   # 12: -x+z
    [1,  0, -1],   # 13: +x-z
    [-1, 0, -1],   # 14: -x-z
    [0,  1,  1],   # 15: +y+z
    [0, -1,  1],   # 16: -y+z
    [0,  1, -1],   # 17: +y-z
    [0, -1, -1],   # 18: -y-z
], dtype=np.int32)

# Weights
W = np.array([
    1.0 / 3.0,                                     # rest
    1.0 / 18.0, 1.0 / 18.0,                       # ±x
    1.0 / 18.0, 1.0 / 18.0,                       # ±y
    1.0 / 18.0, 1.0 / 18.0,                       # ±z
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,  # xy edges
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,  # xz edges
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,  # yz edges
], dtype=np.float32)

# Opposite direction indices
OPP = np.array([
    0,       # rest
    2, 1,    # ±x
    4, 3,    # ±y
    6, 5,    # ±z
    10, 9, 8, 7,       # xy edges
    14, 13, 12, 11,    # xz edges
    18, 17, 16, 15,    # yz edges
], dtype=np.int32)

CS2 = 1.0 / 3.0
CS4 = CS2 * CS2
Q = 19


# ── Equilibrium distribution ────────────────────────────────────────────

def equilibrium(
    density: jnp.ndarray,
    velocity: jnp.ndarray,
) -> jnp.ndarray:
    """Compute equilibrium distribution f_eq for D3Q19.

    Parameters
    ----------
    density : (...) float32
    velocity : (..., 3) float32

    Returns
    -------
    f_eq : (..., 19) float32
    """
    e_dot_u = velocity @ jnp.array(E, dtype=jnp.float32).T  # (..., 19)
    u_sq = jnp.sum(velocity ** 2, axis=-1, keepdims=True)    # (..., 1)

    f_eq = jnp.array(W) * density[..., None] * (
        1.0
        + e_dot_u / CS2
        + e_dot_u ** 2 / (2.0 * CS4)
        - u_sq / (2.0 * CS2)
    )
    return f_eq


# ── Macroscopic quantities ──────────────────────────────────────────────

def compute_macroscopic(
    f: jnp.ndarray,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract density and velocity from distribution functions.

    Parameters
    ----------
    f : (..., 19) float32
    force : (..., 3) float32, optional

    Returns
    -------
    density : (...) float32
    velocity : (..., 3) float32
    """
    density = jnp.sum(f, axis=-1)
    momentum = f @ jnp.array(E, dtype=jnp.float32)

    if force is not None:
        momentum = momentum + 0.5 * force

    velocity = momentum / jnp.maximum(density[..., None], 1e-10)
    return density, velocity


# ── Streaming step ──────────────────────────────────────────────────────

def stream(f: jnp.ndarray) -> jnp.ndarray:
    """Streaming: shift each f_q by its lattice velocity e_q.

    Parameters
    ----------
    f : (nx, ny, nz, 19) float32

    Returns
    -------
    f_streamed : (nx, ny, nz, 19) float32
    """
    slices = []
    for q in range(Q):
        fq = f[..., q]
        ex, ey, ez = int(E[q, 0]), int(E[q, 1]), int(E[q, 2])
        if ex != 0:
            fq = jnp.roll(fq, ex, axis=0)
        if ey != 0:
            fq = jnp.roll(fq, ey, axis=1)
        if ez != 0:
            fq = jnp.roll(fq, ez, axis=2)
        slices.append(fq)
    return jnp.stack(slices, axis=-1)


# ── Guo forcing ─────────────────────────────────────────────────────────

def guo_forcing(
    velocity: jnp.ndarray,
    force: jnp.ndarray,
    tau: float,
) -> jnp.ndarray:
    """Guo et al. (2002) forcing term for D3Q19.

    Parameters
    ----------
    velocity : (..., 3) float32
    force : (..., 3) float32
    tau : float

    Returns
    -------
    S : (..., 19) float32
    """
    e_f = jnp.array(E, dtype=jnp.float32)  # (19, 3)

    # Reshape for broadcasting: add Q dimension
    # velocity: (..., 1, 3), e_f: (19, 3)
    v_expanded = velocity[..., None, :]  # (..., 1, 3)
    e_minus_u = e_f - v_expanded         # (..., 19, 3)

    e_dot_u = velocity @ e_f.T           # (..., 19)
    e_scaled = e_f * (e_dot_u / CS4)[..., None]  # (..., 19, 3)

    bracket = e_minus_u / CS2 + e_scaled  # (..., 19, 3)

    f_expanded = force[..., None, :]     # (..., 1, 3)
    S = (1.0 - 0.5 / tau) * jnp.array(W) * jnp.sum(
        bracket * f_expanded, axis=-1
    )
    return S


# ── BGK collision ───────────────────────────────────────────────────────

def collide_bgk(
    f: jnp.ndarray,
    density: jnp.ndarray,
    velocity: jnp.ndarray,
    tau: float,
    force: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """BGK collision: f_out = f - (f - f_eq)/tau + S.

    Parameters
    ----------
    f : (..., 19) float32
    density : (...) float32
    velocity : (..., 3) float32
    tau : float
    force : (..., 3) float32, optional

    Returns
    -------
    f_out : (..., 19) float32
    """
    f_eq = equilibrium(density, velocity)
    f_out = f - (f - f_eq) / tau

    if force is not None:
        f_out = f_out + guo_forcing(velocity, force, tau)

    return f_out


# ── Boundary conditions ─────────────────────────────────────────────────

def bounce_back_mask(
    f: jnp.ndarray,
    wall_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Half-way bounce-back at wall nodes.

    Parameters
    ----------
    f : (nx, ny, nz, 19) float32
    wall_mask : (nx, ny, nz) bool

    Returns
    -------
    f_bounced : (nx, ny, nz, 19) float32
    """
    f_reflected = f[..., OPP]
    mask = wall_mask[..., None]
    return jnp.where(mask, f_reflected, f)


def create_channel_walls(
    nx: int,
    ny: int,
    nz: int,
    wall_axis: int = 1,
    wall_thickness: int = 1,
) -> jnp.ndarray:
    """Create wall mask for a 3D channel.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    wall_axis : int
        Axis perpendicular to walls (0=x, 1=y, 2=z).
        Default 1: walls at y=0 and y=ny-1 (flow in x, channel in y).
    wall_thickness : int
        Wall thickness in lattice units.

    Returns
    -------
    wall_mask : (nx, ny, nz) bool
    """
    mask = np.zeros((nx, ny, nz), dtype=bool)
    if wall_axis == 0:
        mask[:wall_thickness, :, :] = True
        mask[-wall_thickness:, :, :] = True
    elif wall_axis == 1:
        mask[:, :wall_thickness, :] = True
        mask[:, -wall_thickness:, :] = True
    else:
        mask[:, :, :wall_thickness] = True
        mask[:, :, -wall_thickness:] = True
    return jnp.array(mask)


def create_pipe_walls(
    nx: int,
    ny: int,
    nz: int,
    radius: float | None = None,
    center_y: float | None = None,
    center_z: float | None = None,
) -> jnp.ndarray:
    """Create wall mask for a cylindrical pipe (flow in x-direction).

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    radius : float, optional
        Pipe radius in lattice units. Default: (ny-2)/2.
    center_y, center_z : float, optional
        Pipe center. Default: (ny-1)/2, (nz-1)/2.

    Returns
    -------
    wall_mask : (nx, ny, nz) bool
    """
    if radius is None:
        radius = (ny - 2) / 2.0
    if center_y is None:
        center_y = (ny - 1) / 2.0
    if center_z is None:
        center_z = (nz - 1) / 2.0

    y = jnp.arange(ny, dtype=jnp.float32)
    z = jnp.arange(nz, dtype=jnp.float32)
    yy, zz = jnp.meshgrid(y, z, indexing='ij')
    dist = jnp.sqrt((yy - center_y) ** 2 + (zz - center_z) ** 2)
    pipe_mask_2d = dist > radius  # True outside pipe = wall
    # Broadcast to 3D: same cross-section for all x
    return jnp.broadcast_to(pipe_mask_2d[None, :, :], (nx, ny, nz))


# ── Full LBM step ───────────────────────────────────────────────────────

def lbm_step(
    f: jnp.ndarray,
    tau: float,
    wall_mask: jnp.ndarray | None = None,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One complete D3Q19 LBM step: collision → streaming → BC.

    Parameters
    ----------
    f : (nx, ny, nz, 19)
    tau : float
    wall_mask : (nx, ny, nz) bool, optional
    force : (nx, ny, nz, 3) float32, optional

    Returns
    -------
    f_new, density, velocity
    """
    density, velocity = compute_macroscopic(f, force=force)
    f_new = collide_bgk(f, density, velocity, tau, force=force)
    f_new = stream(f_new)

    if wall_mask is not None:
        f_new = bounce_back_mask(f_new, wall_mask)
        velocity = jnp.where(wall_mask[..., None], 0.0, velocity)

    return f_new, density, velocity


# ── Initialisation ──────────────────────────────────────────────────────

def init_equilibrium(
    nx: int,
    ny: int,
    nz: int,
    density: float = 1.0,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> jnp.ndarray:
    """Initialise distribution functions to equilibrium.

    Parameters
    ----------
    nx, ny, nz : int
    density : float
    velocity : (float, float, float)

    Returns
    -------
    f : (nx, ny, nz, 19) float32
    """
    rho = jnp.ones((nx, ny, nz)) * density
    u = jnp.zeros((nx, ny, nz, 3))
    u = u.at[..., 0].set(velocity[0])
    u = u.at[..., 1].set(velocity[1])
    u = u.at[..., 2].set(velocity[2])
    return equilibrium(rho, u)


# ── Unit conversion (same formulas as 2D) ───────────────────────────────

def tau_from_viscosity(nu_physical: float, dx: float, dt: float) -> float:
    """Compute BGK tau from physical viscosity."""
    nu_lattice = nu_physical * dt / (dx ** 2)
    return nu_lattice / CS2 + 0.5
