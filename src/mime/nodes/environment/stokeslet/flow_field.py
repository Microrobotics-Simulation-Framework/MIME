"""Post-processing: evaluate velocity field from BEM solution.

Given the solved surface traction f(y_n) on the body (and optionally
wall) surface, the fluid velocity at any point x is:

    u_j(x) = (1/8πμ) Σ_n S^ε_jk(x, y_n) · f_k(y_n) · w_n

This is an O(N_surface × N_eval) operation — essentially free
compared to the BEM solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernel import stokeslet_tensor


def evaluate_velocity_field(
    eval_points: jnp.ndarray,
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    traction: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Evaluate fluid velocity at arbitrary points from BEM solution.

    Parameters
    ----------
    eval_points : (M, 3) evaluation points
    surface_points : (N, 3) source points
    surface_weights : (N,) quadrature weights
    traction : (N, 3) solved force densities
    epsilon : float
    mu : float

    Returns
    -------
    velocity : (M, 3) velocity at evaluation points
    """
    prefactor = 1.0 / (8.0 * jnp.pi * mu)

    def _velocity_at_point(x):
        # Sum contributions from all source points
        def _contribution(y, w, f):
            S = stokeslet_tensor(x, y, epsilon)
            return prefactor * w * S @ f

        contributions = jax.vmap(_contribution)(
            surface_points, surface_weights, traction,
        )
        return jnp.sum(contributions, axis=0)

    return jax.vmap(_velocity_at_point)(eval_points)


def composite_velocity_field(
    lbm_velocity: jnp.ndarray,
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    traction: jnp.ndarray,
    interface_center: jnp.ndarray,
    interface_radius: float,
    dx: float,
    origin: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Composite BEM near-field onto LBM far-field velocity grid.

    For lattice nodes inside the interface sphere, evaluate BEM velocity
    (Stokeslet sum from body traction) and overwrite the unphysical
    LBM values there. Outside the sphere, keep LBM values.

    Parameters
    ----------
    lbm_velocity : (nx, ny, nz, 3) LBM velocity field
    surface_points : (N, 3) body surface mesh points
    surface_weights : (N,) quadrature weights
    traction : (N, 3) solved body traction from BEM
    interface_center : (3,) interface sphere center in physical coords
    interface_radius : float, interface sphere radius
    dx : float, lattice spacing in physical units
    origin : (3,) physical coordinates of lattice node (0,0,0)
    epsilon : float, BEM regularisation parameter
    mu : float, dynamic viscosity

    Returns
    -------
    composited : (nx, ny, nz, 3) unified velocity field
    """
    nx, ny, nz = lbm_velocity.shape[:3]

    # Generate lattice node positions
    ix = jnp.arange(nx) * dx + origin[0]
    iy = jnp.arange(ny) * dx + origin[1]
    iz = jnp.arange(nz) * dx + origin[2]
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    lattice_pts = jnp.stack([gx, gy, gz], axis=-1)  # (nx, ny, nz, 3)

    # Distance from interface center
    r_vec = lattice_pts - interface_center[None, None, None, :]
    dist = jnp.sqrt(jnp.sum(r_vec ** 2, axis=-1))  # (nx, ny, nz)

    # Mask: inside the interface sphere
    inside = dist < interface_radius  # (nx, ny, nz)

    # Evaluate BEM velocity at inside points
    inside_pts = lattice_pts[inside]  # (M, 3) where M = num inside points
    n_inside = inside_pts.shape[0]

    if n_inside > 0:
        bem_vel = evaluate_velocity_field(
            inside_pts, surface_points, surface_weights,
            traction, epsilon, mu,
        )  # (M, 3)

        # Scatter BEM values back into the grid
        composited = lbm_velocity.at[inside].set(bem_vel)
    else:
        composited = lbm_velocity

    return composited
