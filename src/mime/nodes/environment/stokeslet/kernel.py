"""Regularised Stokeslet tensor for 3D Stokes flow.

The regularised Stokeslet replaces the singular Dirac delta forcing
with a smooth blob function φ_ε, yielding a smooth exact solution
to the Stokes equations everywhere (including at x = x₀).

Reference:
    Cortez, Fauci & Medovikov (2005), "The method of regularized
    Stokeslets in three dimensions", Phys. Fluids 17:031504.
    Eq. 10b: Stokeslet tensor. Eq. 9: blob function.
    PDF: http://dumkaland.org/publications/CortezFauciMedovikov1.pdf
"""

from __future__ import annotations

import jax.numpy as jnp


def rotlet_tensor(
    x: jnp.ndarray,
    x0: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Regularised rotlet tensor R_ij(x, x₀).

    R_ij = ε_ijl (x_l - x₀_l) / (4π(r² + ε²)^(3/2))

    The rotlet gives the velocity due to a point torque.
    Used in the Power & Miranda completion for the CDL formulation.

    Reference: Cortez & Varela (2015), Section 2.3.1.

    Parameters
    ----------
    x : (3,) target point
    x0 : (3,) source point (interior to body)
    epsilon : float

    Returns
    -------
    R : (3, 3) tensor (without prefactor — matches Stokeslet convention)
    """
    r_vec = x - x0
    r_sq = jnp.dot(r_vec, r_vec)
    eps_sq = epsilon ** 2
    denom = (r_sq + eps_sq) ** 1.5

    # ε_ijl r_l / denom  →  antisymmetric matrix
    # [R]_ij = ε_ijl r_l / denom
    # This is equivalent to: R @ L = (r × L) / denom
    R = jnp.array([
        [0.0, -r_vec[2], r_vec[1]],
        [r_vec[2], 0.0, -r_vec[0]],
        [-r_vec[1], r_vec[0], 0.0],
    ]) / denom

    return R


def stokeslet_tensor(
    x: jnp.ndarray,
    x0: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Regularised Stokeslet tensor S^ε_jk(x, x₀).

    S^ε_jk = [δ_jk(r² + 2ε²) + (x_j - x₀_j)(x_k - x₀_k)]
             / (r² + ε²)^(3/2)

    The 1/(8πμ) prefactor is NOT included — it is applied during
    matrix assembly to keep the kernel pure geometry.

    Parameters
    ----------
    x : (3,) target point
    x0 : (3,) source point
    epsilon : float, regularisation parameter

    Returns
    -------
    S : (3, 3) tensor (without the 1/(8πμ) prefactor)
    """
    r_vec = x - x0
    r_sq = jnp.dot(r_vec, r_vec)
    eps_sq = epsilon ** 2
    denom = (r_sq + eps_sq) ** 1.5

    S = (jnp.eye(3) * (r_sq + 2.0 * eps_sq)
         + jnp.outer(r_vec, r_vec)) / denom

    return S
