"""Regularised stresslet (double-layer) kernel for 3D Stokes flow.

The stresslet T^ε_ijk is the stress tensor associated with the
regularised Stokeslet. It appears in the double-layer potential
(DLP) of the boundary integral equation.

Reference:
    Smith, Gallagher, Schuech & Montenegro-Johnson (2021),
    "The role of the double-layer potential in regularised stokeslet
    models of self-propulsion", Fluids 6(11):411.
    Eqs. (12)-(13): regularised stresslet tensor.

    Cortez & Varela (2015), "A general system of images for
    regularized Stokeslets", J. Comput. Phys. 285:41-54.
    Eq. (6): stresslet from differentiation of the blob function.
"""

from __future__ import annotations

import jax.numpy as jnp


def stresslet_tensor_contracted(
    x: jnp.ndarray,
    y: jnp.ndarray,
    n: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Regularised stresslet T^ε_ijk contracted with normal n_k.

    Returns the 3×3 matrix K_ij = T^ε_ijk(x, y) · n_k(y), which
    maps velocity u_i at source y to the double-layer contribution
    at target x.

    T^ε_ijk(x,y) = -6(x_i-y_i)(x_j-y_j)(x_k-y_k) / (r²+ε²)^(5/2)
                 - 3ε²[(x_i-y_i)δ_jk + (x_j-y_j)δ_ki + (x_k-y_k)δ_ij]
                   / (r²+ε²)^(5/2)

    Parameters
    ----------
    x : (3,) target point
    y : (3,) source point
    n : (3,) outward normal at y
    epsilon : float

    Returns
    -------
    K : (3, 3) — K_ij = T^ε_ijk · n_k  (without 1/(8π) prefactor)
    """
    r = x - y
    r_sq = jnp.dot(r, r)
    eps_sq = epsilon ** 2
    denom = (r_sq + eps_sq) ** 2.5  # (r²+ε²)^(5/2)

    r_dot_n = jnp.dot(r, n)

    # Term 1: -6 r_i r_j (r_k n_k) / denom
    K1 = -6.0 * jnp.outer(r, r) * r_dot_n / denom

    # Term 2: -3ε² [r_i n_j + r_j n_i + (r·n) δ_ij] / denom
    # This comes from contracting:
    #   -3ε² [r_i δ_jk + r_j δ_ki + r_k δ_ij] n_k
    # = -3ε² [r_i n_j + n_i r_j + r_dot_n δ_ij]
    K2 = -3.0 * eps_sq * (
        jnp.outer(r, n) + jnp.outer(n, r) + r_dot_n * jnp.eye(3)
    ) / denom

    return K1 + K2
