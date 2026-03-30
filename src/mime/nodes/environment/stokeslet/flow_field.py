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
