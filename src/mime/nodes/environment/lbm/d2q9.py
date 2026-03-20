"""D2Q9 Lattice Boltzmann Method — core operations.

All functions are pure JAX (jax.numpy), fully JIT-compilable and
differentiable. No Python control flow depends on array values.

Lattice units: dx = dt_lbm = 1. Physical units are mapped via:
    nu_physical = nu_lattice * (dx_physical^2 / dt_physical)
    nu_lattice = (tau - 0.5) / 3

The BGK (single relaxation time) collision operator is used:
    f_out = f - (f - f_eq) / tau + S

where S is the Guo forcing term for external body forces.

Conventions:
    - Lattice velocities e_i are indexed 0..8
    - Direction 0 is rest
    - State shape: (nx, ny, 9) for distribution functions
    - Density shape: (nx, ny)
    - Velocity shape: (nx, ny, 2)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ── D2Q9 lattice constants ──────────────────────────────────────────────
# Use numpy for concrete arrays (JAX auto-promotes in jnp operations).

# 9 velocity vectors: (ex, ey)
E = np.array([
    [0,  0],   # 0: rest
    [1,  0],   # 1: +x
    [-1, 0],   # 2: -x
    [0,  1],   # 3: +y
    [0, -1],   # 4: -y
    [1,  1],   # 5: +x+y
    [-1, 1],   # 6: -x+y
    [1, -1],   # 7: +x-y
    [-1,-1],   # 8: -x-y
], dtype=np.int32)

# Weights
W = np.array([
    4.0 / 9.0,                         # rest
    1.0 / 9.0, 1.0 / 9.0,             # ±x
    1.0 / 9.0, 1.0 / 9.0,             # ±y
    1.0 / 36.0, 1.0 / 36.0,           # diagonals
    1.0 / 36.0, 1.0 / 36.0,
], dtype=np.float32)

# Opposite direction indices (for bounce-back)
OPP = np.array([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=np.int32)

# Speed of sound squared
CS2 = 1.0 / 3.0
CS4 = CS2 * CS2

# Number of lattice directions
Q = 9


# ── Equilibrium distribution ────────────────────────────────────────────

def equilibrium(
    density: jnp.ndarray,
    velocity: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the equilibrium distribution f_eq.

    Parameters
    ----------
    density : (nx, ny) float32
    velocity : (nx, ny, 2) float32

    Returns
    -------
    f_eq : (nx, ny, 9) float32
    """
    # e · u: (nx, ny, 9)
    e_dot_u = velocity @ jnp.array(E, dtype=jnp.float32).T
    # |u|^2: (nx, ny, 1)
    u_sq = jnp.sum(velocity ** 2, axis=-1, keepdims=True)

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
    tau: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract density and velocity from distribution functions.

    When force is provided, applies the Guo velocity correction:
        u = (sum f_i e_i + F*dt/2) / rho

    Parameters
    ----------
    f : (nx, ny, 9) float32
    force : (nx, ny, 2) float32, optional
        External body force per unit volume (in lattice units).
    tau : float
        BGK relaxation time (needed for Guo correction).

    Returns
    -------
    density : (nx, ny) float32
    velocity : (nx, ny, 2) float32
    """
    density = jnp.sum(f, axis=-1)
    momentum = f @ jnp.array(E, dtype=jnp.float32)  # (nx, ny, 2)

    if force is not None:
        # Guo velocity correction: u = (momentum + F*dt/2) / rho
        # In lattice units dt = 1
        momentum = momentum + 0.5 * force

    velocity = momentum / jnp.maximum(density[..., None], 1e-10)
    return density, velocity


# ── Streaming step ──────────────────────────────────────────────────────

def stream(f: jnp.ndarray) -> jnp.ndarray:
    """Streaming: shift each f_i by its lattice velocity e_i.

    Uses jnp.roll. The Python loop over 9 directions unrolls at
    JAX trace time (no dynamic control flow).

    Parameters
    ----------
    f : (nx, ny, 9) float32

    Returns
    -------
    f_streamed : (nx, ny, 9) float32
    """
    slices = []
    for q in range(Q):
        fq = f[..., q]
        ex, ey = int(E[q, 0]), int(E[q, 1])
        if ex != 0:
            fq = jnp.roll(fq, ex, axis=0)
        if ey != 0:
            fq = jnp.roll(fq, ey, axis=1)
        slices.append(fq)
    return jnp.stack(slices, axis=-1)


# ── BGK collision with Guo forcing ──────────────────────────────────────

def guo_forcing(
    velocity: jnp.ndarray,
    force: jnp.ndarray,
    tau: float,
) -> jnp.ndarray:
    """Guo et al. (2002) forcing term for the BGK operator.

    S_i = (1 - 1/(2*tau)) * w_i * [
        (e_i - u)/cs^2 + (e_i · u)/cs^4 * e_i
    ] · F

    Parameters
    ----------
    velocity : (nx, ny, 2) float32
    force : (nx, ny, 2) float32
        Body force per unit volume (lattice units).
    tau : float

    Returns
    -------
    S : (nx, ny, 9) float32
    """
    e_f = jnp.array(E, dtype=jnp.float32)  # (9, 2)

    # (e_i - u): (nx, ny, 9, 2)
    e_minus_u = e_f[None, None, :, :] - velocity[..., None, :]

    # (e_i · u): (nx, ny, 9)
    e_dot_u = velocity @ e_f.T

    # e_i * (e·u / cs^4): (nx, ny, 9, 2)
    e_scaled = e_f[None, None, :, :] * (e_dot_u / CS4)[..., None]

    # bracket: (nx, ny, 9, 2)
    bracket = e_minus_u / CS2 + e_scaled

    # Dot with force: (nx, ny, 9)
    S = (1.0 - 0.5 / tau) * jnp.array(W) * jnp.sum(
        bracket * force[..., None, :], axis=-1
    )
    return S


def collide_bgk(
    f: jnp.ndarray,
    density: jnp.ndarray,
    velocity: jnp.ndarray,
    tau: float,
    force: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """BGK collision step with optional Guo forcing.

    f_out = f - (f - f_eq) / tau + S

    Parameters
    ----------
    f : (nx, ny, 9) float32
    density : (nx, ny) float32
    velocity : (nx, ny, 2) float32
    tau : float
        Relaxation time. nu = (tau - 0.5) * cs^2 = (tau - 0.5) / 3.
    force : (nx, ny, 2) float32, optional
        External body force (lattice units).

    Returns
    -------
    f_out : (nx, ny, 9) float32
    """
    f_eq = equilibrium(density, velocity)
    f_out = f - (f - f_eq) / tau

    if force is not None:
        S = guo_forcing(velocity, force, tau)
        f_out = f_out + S

    return f_out


# ── Boundary conditions ─────────────────────────────────────────────────

def bounce_back_mask(
    f: jnp.ndarray,
    wall_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply half-way bounce-back at wall nodes.

    At wall nodes, incoming distributions are reflected to the
    opposite direction. This implements no-slip walls.

    Parameters
    ----------
    f : (nx, ny, 9) float32
        Post-streaming distributions.
    wall_mask : (nx, ny) bool
        True at wall nodes.

    Returns
    -------
    f_bounced : (nx, ny, 9) float32
    """
    # At wall nodes, swap each direction with its opposite
    f_reflected = f[..., OPP]  # (nx, ny, 9) with swapped directions
    mask = wall_mask[..., None]  # (nx, ny, 1) broadcast to (nx, ny, 9)
    return jnp.where(mask, f_reflected, f)


def create_channel_walls(
    nx: int,
    ny: int,
    wall_thickness: int = 1,
) -> jnp.ndarray:
    """Create a wall mask for a 2D channel (walls at y=0 and y=ny-1).

    Parameters
    ----------
    nx : int
        Grid points in x (flow direction).
    ny : int
        Grid points in y (channel width).
    wall_thickness : int
        Thickness of wall in lattice units.

    Returns
    -------
    wall_mask : (nx, ny) bool
    """
    mask = np.zeros((nx, ny), dtype=bool)
    mask[:, :wall_thickness] = True
    mask[:, -wall_thickness:] = True
    return jnp.array(mask)


def create_cylinder_mask(
    nx: int,
    ny: int,
    center_x: float,
    center_y: float,
    radius: float,
) -> jnp.ndarray:
    """Create a wall mask for a circular cylinder obstacle.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    center_x, center_y : float
        Cylinder center (lattice units).
    radius : float
        Cylinder radius (lattice units).

    Returns
    -------
    mask : (nx, ny) bool
    """
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx))
    dist = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return dist <= radius


# ── Full LBM step ───────────────────────────────────────────────────────

def lbm_step(
    f: jnp.ndarray,
    tau: float,
    wall_mask: jnp.ndarray | None = None,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform one complete LBM step: collision → streaming → BC.

    Parameters
    ----------
    f : (nx, ny, 9)
        Current distribution functions.
    tau : float
        BGK relaxation time.
    wall_mask : (nx, ny) bool, optional
        Wall nodes for bounce-back.
    force : (nx, ny, 2) float32, optional
        External body force (lattice units).

    Returns
    -------
    f_new : (nx, ny, 9)
        Updated distribution functions.
    density : (nx, ny)
        Macroscopic density.
    velocity : (nx, ny, 2)
        Macroscopic velocity.
    """
    # Compute macroscopic quantities (with Guo correction if force present)
    density, velocity = compute_macroscopic(f, force=force, tau=tau)

    # Collision
    f_new = collide_bgk(f, density, velocity, tau, force=force)

    # Streaming
    f_new = stream(f_new)

    # Bounce-back at walls
    if wall_mask is not None:
        f_new = bounce_back_mask(f_new, wall_mask)
        # Zero out velocity at wall nodes
        velocity = jnp.where(wall_mask[..., None], 0.0, velocity)

    return f_new, density, velocity


# ── Initialisation helpers ──────────────────────────────────────────────

def init_equilibrium(
    nx: int,
    ny: int,
    density: float = 1.0,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> jnp.ndarray:
    """Initialise distribution functions to equilibrium.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    density : float
        Initial uniform density.
    velocity : (float, float)
        Initial uniform velocity (ux, uy).

    Returns
    -------
    f : (nx, ny, 9) float32
    """
    rho = jnp.ones((nx, ny)) * density
    u = jnp.zeros((nx, ny, 2))
    u = u.at[..., 0].set(velocity[0])
    u = u.at[..., 1].set(velocity[1])
    return equilibrium(rho, u)


# ── Unit conversion helpers ─────────────────────────────────────────────

def tau_from_viscosity(
    nu_physical: float,
    dx: float,
    dt: float,
) -> float:
    """Compute BGK relaxation time tau from physical viscosity.

    Parameters
    ----------
    nu_physical : float
        Kinematic viscosity [m^2/s].
    dx : float
        Lattice spacing [m].
    dt : float
        LBM timestep [s].

    Returns
    -------
    tau : float
        BGK relaxation time (must be > 0.5 for stability).
    """
    nu_lattice = nu_physical * dt / (dx ** 2)
    tau = nu_lattice / CS2 + 0.5
    return tau


def physical_to_lattice_force(
    force_physical: float,
    dx: float,
    dt: float,
    rho0: float = 1.0,
) -> float:
    """Convert physical force density to lattice units.

    Parameters
    ----------
    force_physical : float
        Force per unit volume [N/m^3] (2D: [N/m^2]).
    dx : float
        Lattice spacing [m].
    dt : float
        LBM timestep [s].
    rho0 : float
        Reference density in lattice units.

    Returns
    -------
    force_lattice : float
    """
    return force_physical * dx ** 2 * dt ** 2 / rho0
