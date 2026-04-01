"""Wall correction via multi-radius polynomial extrapolation.

Computes the vessel wall's contribution to the flow at the body surface
by subtracting the BEM free-space Stokeslet velocity from the walled LBM
velocity at multiple evaluation spheres, fitting a polynomial in a/R,
and extrapolating to the body surface (a/R = 1).

The IB smoothing error cancels in the subtraction because both the
LBM (Peskin-smoothed Stokeslet) and the BEM (analytical Stokeslet)
agree at distances >> the smoothing width (2h from body surface).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.immersed_boundary import interpolate_velocity
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field


def compute_wall_correction(
    u_lbm: jnp.ndarray,
    traction: jnp.ndarray,
    body_pts: jnp.ndarray,
    body_wts: jnp.ndarray,
    eval_stencils: list[dict],
    x_vals: jnp.ndarray,
    epsilon: float,
    mu: float,
    dx: float,
    dt: float,
) -> jnp.ndarray:
    """Multi-radius polynomial extrapolation of wall correction Δu.

    For each eval sphere:
      1. IB-interpolate walled LBM velocity at eval points
      2. BEM-evaluate free-space Stokeslet velocity at same points
      3. Δu = mean_over_sphere(u_walled - u_freespace)

    Fit cubic polynomial Δu(a/R) and extrapolate to a/R = 1 (body surface).

    Parameters
    ----------
    u_lbm : (nx, ny, nz, 3) LBM velocity field in lattice units
    traction : (N_body, 3) current BEM surface traction [Pa]
    body_pts : (N_body, 3) body surface points [m]
    body_wts : (N_body,) BEM quadrature weights [m²]
    eval_stencils : list of dicts with keys:
        'pts_phys': (N_eval, 3) eval points [m]
        'idx': (N_eval, 64) stencil indices (jnp)
        'wts': (N_eval, 64) stencil weights (jnp)
    x_vals : (N_radii,) values of a/R for each eval radius
    epsilon : BEM regularisation parameter
    mu : dynamic viscosity [Pa·s]
    dx : lattice spacing [m]
    dt : LBM timestep [s]

    Returns
    -------
    delta_u : (3,) wall correction at body surface [m/s]
    """
    n_radii = len(eval_stencils)
    du_means = []

    for es in eval_stencils:
        # LBM velocity at eval points (lattice → physical)
        u_walled = interpolate_velocity(u_lbm, es['idx'], es['wts']) * dx / dt

        # BEM free-space Stokeslet velocity at same points (exact)
        u_freespace = evaluate_velocity_field(
            es['pts_phys'], body_pts, body_wts, traction, epsilon, mu,
        )

        # Wall correction = walled - free-space, averaged over sphere
        du = u_walled - u_freespace
        du_means.append(jnp.mean(du, axis=0))

    du_means = jnp.stack(du_means)  # (N_radii, 3)

    # Fit polynomial in a/R: Δu(x) = c₀ + c₁x + c₂x² [+ c₃x³]
    n_coeffs = min(4, n_radii)
    V = jnp.stack([x_vals ** k for k in range(n_coeffs)], axis=1)

    delta_u = jnp.zeros(3)
    for j in range(3):
        coeffs = jnp.linalg.lstsq(V, du_means[:, j])[0]
        # Extrapolate to body surface: x = a/R = 1 → sum of all coefficients
        delta_u = delta_u.at[j].set(jnp.sum(coeffs))

    return delta_u
