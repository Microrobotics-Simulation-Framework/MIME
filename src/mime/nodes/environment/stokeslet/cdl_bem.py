"""Completed double-layer BEM (Power & Miranda augmented system).

Augments the ½I + K double-layer system with Stokeslet + rotlet
completion at an interior point x_c. The augmented system is
(3N+6) × (3N+6). Force = α, torque = β (completion strengths).

System:
    [½I + K  |  S_col  R_col] [q]   [u]
    [W_F     |    0      0  ] [α] = [0]
    [W_T     |    0      0  ] [β]   [0]

References:
    Power & Miranda (1987), SIAM J. Appl. Math. 47(4):689-698.
    Gonzalez (2009), SIAM J. Appl. Math. 69(4):933-966.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernel import stokeslet_tensor, rotlet_tensor
from .stresslet import stresslet_tensor_contracted


def assemble_cdl_system(
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
    surface_weights: jnp.ndarray,
    x_star: jnp.ndarray,
    epsilon: float,
    theta: float = 0.5,
) -> jnp.ndarray:
    """Assemble the augmented CDL system matrix (3N+6) × (3N+6).

    Top-left: ½I + K  (DLP operator with jump condition)
    Top-right: S_col | R_col  (Stokeslet + rotlet at x_star)
    Bottom-left: W_F | W_T  (compatibility: ∫q dA = 0, ∫(y-xc)×q dA = 0)
    Bottom-right: zeros

    Parameters
    ----------
    surface_points : (N, 3)
    surface_normals : (N, 3)
    surface_weights : (N,)
    x_star : (3,) interior completion point
    epsilon : float
    theta : float, unused (kept for API compatibility)

    Returns
    -------
    M : (3N+6, 3N+6)
    """
    N = len(surface_points)
    prefactor = 1.0 / (8.0 * jnp.pi)

    # --- Top-left block: ½I + K (3N × 3N) ---

    def _dlp_block(m, n):
        x_m = surface_points[m]
        x_n = surface_points[n]
        n_n = surface_normals[n]
        w_n = surface_weights[n]

        K = prefactor * w_n * stresslet_tensor_contracted(x_n, x_m, n_n, epsilon)
        return jnp.where(m == n, jnp.zeros((3, 3)), K)

    blocks = jax.vmap(
        jax.vmap(_dlp_block, in_axes=(None, 0)),
        in_axes=(0, None),
    )(jnp.arange(N), jnp.arange(N))

    K_mat = blocks.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    half_I_plus_K = 0.5 * jnp.eye(3 * N) + K_mat

    # --- Top-right block: S_col | R_col (3N × 6) ---
    # Column j (j=0,1,2): Stokeslet S^ε(x_m, x_star) · e_j at each x_m
    # Column j+3 (j=0,1,2): Rotlet R^ε(x_m, x_star) · e_j at each x_m

    def _completion_row(m):
        x_m = surface_points[m]
        S = stokeslet_tensor(x_m, x_star, epsilon)   # (3, 3)
        R = rotlet_tensor(x_m, x_star, epsilon)       # (3, 3)
        # S_col: each column is S · e_j = S[:, j]
        # R_col: each column is R · e_j = R[:, j]
        return jnp.concatenate([S, R], axis=1)  # (3, 6)

    completion_blocks = jax.vmap(_completion_row)(jnp.arange(N))  # (N, 3, 6)
    # Reshape to (3N, 6)
    top_right = completion_blocks.reshape(3 * N, 6)

    # --- Bottom-left block: W_F | W_T (6 × 3N) ---
    # W_F[j, 3n+l] = δ_jl · w_n  →  enforces ∫ q_j dA = 0
    # W_T[j, 3n+l] = ε_jkl · (y_n - x_c)_k · w_n  →  enforces ∫(y-xc) × q dA = 0

    # W_F: 3 × 3N
    W_F = jnp.zeros((3, 3 * N))
    for n in range(N):
        w = surface_weights[n]
        for j in range(3):
            W_F = W_F.at[j, 3 * n + j].set(w)

    # W_T: 3 × 3N  (Levi-Civita contraction)
    W_T = jnp.zeros((3, 3 * N))
    r = surface_points - x_star  # (N, 3)
    for n in range(N):
        w = surface_weights[n]
        rn = r[n]
        # ε_jkl r_k w → cross product structure
        # j=0: ε_0kl r_k = r_1 δ_l2 - r_2 δ_l1
        W_T = W_T.at[0, 3 * n + 1].set(-rn[2] * w)  # ε_012 = +1, but ε_0,k,l: (0,1,2)→+1, (0,2,1)→-1
        W_T = W_T.at[0, 3 * n + 2].set(rn[1] * w)
        # j=1: ε_1kl r_k = r_2 δ_l0 - r_0 δ_l2
        W_T = W_T.at[1, 3 * n + 0].set(rn[2] * w)
        W_T = W_T.at[1, 3 * n + 2].set(-rn[0] * w)
        # j=2: ε_2kl r_k = r_0 δ_l1 - r_1 δ_l0
        W_T = W_T.at[2, 3 * n + 0].set(-rn[1] * w)
        W_T = W_T.at[2, 3 * n + 1].set(rn[0] * w)

    bottom_left = jnp.concatenate([W_F, W_T], axis=0)  # (6, 3N)

    # --- Bottom-right block: zeros (6 × 6) ---
    bottom_right = jnp.zeros((6, 6))

    # --- Assemble augmented system ---
    top = jnp.concatenate([half_I_plus_K, top_right], axis=1)      # (3N, 3N+6)
    bottom = jnp.concatenate([bottom_left, bottom_right], axis=1)   # (6, 3N+6)
    M = jnp.concatenate([top, bottom], axis=0)                      # (3N+6, 3N+6)

    return M


def compute_cdl_force_torque(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    psi: jnp.ndarray,
    center: jnp.ndarray,
    theta: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract force and torque from the augmented CDL solution.

    NOT USED — force/torque come from α, β directly.
    Kept for API compatibility.
    """
    # This function is superseded by extracting α, β from the
    # solution vector. See compute_cdl_resistance_matrix.
    weighted_psi = psi * surface_weights[:, None]
    force = -jnp.sum(weighted_psi, axis=0)
    r = surface_points - center
    torque = -jnp.sum(jnp.cross(r, weighted_psi), axis=0)
    return force, torque
