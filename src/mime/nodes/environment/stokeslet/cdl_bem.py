"""Completed double-layer BEM formulation (Power & Miranda 1987).

Uses a combination of double-layer potential + Stokeslet/rotlet
completion to give a second-kind Fredholm equation with bounded
condition number. Superior to the single-layer formulation for
confined flows.

The flow representation (Gonzalez 2009, Eq. 6.1):
    u = θ·Y[Γ,ψ] + (1-θ)·W[Γ,ψ]

where Y is the point-force+rotlet potential at x_* (inside body),
W is the double-layer potential, θ ∈ (0,1) is a mixing parameter.

The boundary integral equation (Gonzalez 2009, Eq. 6.9):
    ∫_Γ K_θ(x,y) ψ(y) dA_y + c_θ ψ(x) = v(x)

where c_θ = (1-θ)·α, α = 1/2 for smooth surface.

Force and torque (Gonzalez 2009, Eq. 6.6):
    F = -8πθ ∫_Γ ψ(y) dA_y
    T = -8πθ ∫_Γ (y-c) × ψ(y) dA_y

References:
    Power & Miranda (1987), SIAM J. Appl. Math. 47(4):689-698.
    Gonzalez (2009), SIAM J. Appl. Math. 69(4):933-966.
    Smith et al. (2021), Fluids 6(11):411 — stresslet kernel.
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
    """Assemble the CDL system matrix (3N × 3N).

    M[3m+j, 3n+l] = K_θ^jl(x_m, y_n) · w_n    [m ≠ n]
    M[3m+j, 3m+l] = c_θ · δ_jl               [m = n, diagonal]

    where K_θ (Gonzalez Eq. 6.10):
        K_θ^jl(x,y) = θ·S_jl(x, x_*) + θ·R_jk(x, x_*) ε_kpl (y_p - x_{*p})
                     + (1-θ)·T_jlk(x, y) n_k(y)

    The ½I term comes from c_θ = (1-θ)·(1/2).

    Parameters
    ----------
    surface_points : (N, 3)
    surface_normals : (N, 3)
    surface_weights : (N,)
    x_star : (3,) interior point for completion
    epsilon : float
    theta : float, mixing parameter (0 < θ < 1)

    Returns
    -------
    M : (3N, 3N)
    """
    N = len(surface_points)
    c_theta = (1.0 - theta) * 0.5  # jump condition coefficient

    def _kernel_block(m, n):
        """Compute the 3×3 kernel block K_θ(x_m, y_n) · w_n."""
        x_m = surface_points[m]
        y_n = surface_points[n]
        n_n = surface_normals[n]
        w_n = surface_weights[n]

        # Term 1: θ · Stokeslet at x_* (completion)
        # S_jl(x_m, x_*) — note: same for all n (depends on x_m, x_* only)
        S = stokeslet_tensor(x_m, x_star, epsilon)

        # Term 2: θ · Rotlet at x_* contracted with (y_n - x_*)
        # R_jk(x_m, x_*) · ε_kpl (y_p - x_{*p})
        # This gives a 3×3 matrix mapping ψ_l → velocity_j
        R = rotlet_tensor(x_m, x_star, epsilon)
        r_yn = y_n - x_star
        # R_jk ε_kpl r_p = R @ cross_matrix(r)
        # where cross_matrix(r) maps l → ε_kpl r_p = (r × e_l)_k
        cross_r = jnp.array([
            [0.0, -r_yn[2], r_yn[1]],
            [r_yn[2], 0.0, -r_yn[0]],
            [-r_yn[1], r_yn[0], 0.0],
        ])
        rotlet_contrib = R @ cross_r

        # Term 3: (1-θ) · Stresslet T_jlk · n_k (double-layer)
        # Sign: The DLP identity (Smith et al. Eq. 29) requires
        # DLP = -(1/8π) ∫ T(x,y) n(x) u(x) dS ≈ (1/2)u(y)
        # Our stresslet_tensor_contracted(x,y,n,ε) computes T_ijk(x,y)n_k
        # with the Smith et al. sign (leading -6). To get the correct
        # DLP sign, we negate here.
        T_contracted = -stresslet_tensor_contracted(x_m, y_n, n_n, epsilon)

        # Combined kernel (Gonzalez Eq. 6.10)
        # Prefactor: Stokeslet/rotlet use 1/(8πμ) convention but here
        # we work in dimensionless form (μ absorbed into ψ interpretation)
        K = theta * (S + rotlet_contrib) + (1.0 - theta) * T_contracted

        # Zero out stresslet self-interaction (m == n)
        # The S and rotlet terms are nonzero at m == n (they use x_*)
        K_no_self_T = theta * (S + rotlet_contrib)
        K = jnp.where(m == n, K_no_self_T, K)

        return w_n * K

    # Assemble using double vmap
    blocks = jax.vmap(
        jax.vmap(_kernel_block, in_axes=(None, 0)),
        in_axes=(0, None),
    )(jnp.arange(N), jnp.arange(N))
    # blocks shape: (N, N, 3, 3)

    # Reshape to (3N, 3N)
    M = blocks.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)

    # Add diagonal c_θ · I
    M = M + c_theta * jnp.eye(3 * N)

    return M


def compute_cdl_force_torque(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    psi: jnp.ndarray,
    center: jnp.ndarray,
    theta: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract force and torque from CDL density ψ.

    From Gonzalez (2009) Eq. (6.6):
        F = -8π θ ∫_Γ ψ(y) dA_y
        T = -8π θ ∫_Γ (y-c) × ψ(y) dA_y

    Parameters
    ----------
    surface_points : (N, 3)
    surface_weights : (N,)
    psi : (N, 3) CDL density
    center : (3,) moment reference point
    theta : float

    Returns
    -------
    force : (3,)
    torque : (3,)
    """
    prefactor = -8.0 * jnp.pi * theta
    weighted_psi = psi * surface_weights[:, None]

    force = prefactor * jnp.sum(weighted_psi, axis=0)

    r = surface_points - center
    torque = prefactor * jnp.sum(jnp.cross(r, weighted_psi), axis=0)

    return force, torque
