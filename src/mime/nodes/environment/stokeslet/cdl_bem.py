"""Completed double-layer BEM (Power & Miranda augmented system).

Augments the ½I + K double-layer system with Stokeslet + rotlet
completion singularities. For multiply-connected domains (body
inside vessel), each closed surface gets its own completion.

System for body-only (unconfined):
    [½I + K  |  S_b  R_b] [q  ]   [u]
    [W_Fb    |   0    0 ] [α_b] = [0]
    [W_Tb    |   0    0 ] [β_b]   [0]

System for body + wall (confined):
    [½I + K  |  S_b  R_b  S_w  R_w] [q  ]   [u]
    [W_Fb    |   0    0    0    0  ] [α_b] = [0]
    [W_Tb    |   0    0    0    0  ] [β_b]   [0]
    [W_Fw    |   0    0    0    0  ] [α_w]   [0]
    [W_Tw    |   0    0    0    0  ] [β_w]   [0]

Force = 8π · α_b, Torque = -8π · β_b.

References:
    Power & Miranda (1987), SIAM J. Appl. Math. 47(4):689-698.
    Gonzalez (2009), SIAM J. Appl. Math. 69(4):933-966.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import stokeslet_tensor, rotlet_tensor
from .stresslet import stresslet_tensor_contracted


def _assemble_half_I_plus_K(
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
    surface_weights: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Assemble the ½I + K DLP matrix (3N × 3N)."""
    N = len(surface_points)
    prefactor = 1.0 / (8.0 * jnp.pi)

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
    return 0.5 * jnp.eye(3 * N) + K_mat


def _completion_columns(
    surface_points: jnp.ndarray,
    x_star: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Build Stokeslet + rotlet completion columns (3N × 6)."""
    def _row(m):
        x_m = surface_points[m]
        S = stokeslet_tensor(x_m, x_star, epsilon)
        R = rotlet_tensor(x_m, x_star, epsilon)
        return jnp.concatenate([S, R], axis=1)
    blocks = jax.vmap(_row)(jnp.arange(len(surface_points)))
    return blocks.reshape(3 * len(surface_points), 6)


def _compatibility_rows(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    x_ref: jnp.ndarray,
    n_total: int,
    offset: int,
) -> jnp.ndarray:
    """Build W_F and W_T compatibility rows (6 × 3*n_total).

    Only covers points [offset : offset + len(surface_points)].
    """
    N = len(surface_points)
    W_F = jnp.zeros((3, 3 * n_total))
    W_T = jnp.zeros((3, 3 * n_total))
    r = surface_points - x_ref

    for n in range(N):
        idx = offset + n
        w = float(surface_weights[n])
        rn = r[n]
        for j in range(3):
            W_F = W_F.at[j, 3 * idx + j].set(w)

        # Cross product: ε_jkl r_k → torque compatibility
        W_T = W_T.at[0, 3 * idx + 1].set(-float(rn[2]) * w)
        W_T = W_T.at[0, 3 * idx + 2].set(float(rn[1]) * w)
        W_T = W_T.at[1, 3 * idx + 0].set(float(rn[2]) * w)
        W_T = W_T.at[1, 3 * idx + 2].set(-float(rn[0]) * w)
        W_T = W_T.at[2, 3 * idx + 0].set(-float(rn[1]) * w)
        W_T = W_T.at[2, 3 * idx + 1].set(float(rn[0]) * w)

    return jnp.concatenate([W_F, W_T], axis=0)


def assemble_cdl_system(
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
    surface_weights: jnp.ndarray,
    x_star: jnp.ndarray,
    epsilon: float,
    theta: float = 0.5,
    n_body: int | None = None,
    x_wall_star: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Assemble the augmented CDL system.

    For unconfined (n_body=None): (3N+6) × (3N+6) with body completion.
    For confined (n_body set, x_wall_star set): (3N+12) × (3N+12) with
    body + wall completion.

    Parameters
    ----------
    surface_points : (N, 3) — body points first, then wall points
    surface_normals : (N, 3)
    surface_weights : (N,)
    x_star : (3,) body completion point (inside body)
    epsilon : float
    theta : float, unused
    n_body : int or None — if set, first n_body points are body
    x_wall_star : (3,) or None — wall completion point (outside wall)
    """
    N = len(surface_points)
    confined = (n_body is not None and x_wall_star is not None)
    N_b = n_body if confined else N
    N_w = N - N_b if confined else 0
    n_completion = 12 if confined else 6

    # ½I + K (3N × 3N)
    half_I_K = _assemble_half_I_plus_K(
        surface_points, surface_normals, surface_weights, epsilon,
    )

    # Completion columns
    # Body completion: Stokeslet + rotlet at x_star for ALL surface points
    S_b_R_b = _completion_columns(surface_points, x_star, epsilon)  # (3N, 6)

    if confined:
        # Wall completion: Stokeslet + rotlet at x_wall_star for ALL points
        S_w_R_w = _completion_columns(surface_points, x_wall_star, epsilon)
        top_right = jnp.concatenate([S_b_R_b, S_w_R_w], axis=1)  # (3N, 12)
    else:
        top_right = S_b_R_b  # (3N, 6)

    # Compatibility rows — body DOFs only
    W_body = _compatibility_rows(
        surface_points[:N_b], surface_weights[:N_b],
        x_star, N, offset=0,
    )  # (6, 3N)

    if confined:
        # Compatibility rows — wall DOFs only
        W_wall = _compatibility_rows(
            surface_points[N_b:], surface_weights[N_b:],
            x_wall_star, N, offset=N_b,
        )  # (6, 3N)
        bottom_left = jnp.concatenate([W_body, W_wall], axis=0)  # (12, 3N)
    else:
        bottom_left = W_body  # (6, 3N)

    # Bottom-right: zeros
    bottom_right = jnp.zeros((n_completion, n_completion))

    # Assemble
    top = jnp.concatenate([half_I_K, top_right], axis=1)
    bottom = jnp.concatenate([bottom_left, bottom_right], axis=1)
    M = jnp.concatenate([top, bottom], axis=0)

    return M


def compute_cdl_force_torque(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    psi: jnp.ndarray,
    center: jnp.ndarray,
    theta: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unused — force/torque from α_b, β_b directly."""
    weighted_psi = psi * surface_weights[:, None]
    force = -jnp.sum(weighted_psi, axis=0)
    r = surface_points - center
    torque = -jnp.sum(jnp.cross(r, weighted_psi), axis=0)
    return force, torque
