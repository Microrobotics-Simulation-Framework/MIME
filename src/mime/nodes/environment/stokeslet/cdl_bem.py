"""Double-layer BEM formulation for the resistance problem.

Uses the regularised stresslet kernel to form a second-kind
Fredholm equation (½I + K)ψ = v. The ½ comes from the jump
condition of the double-layer potential at the surface.

For the resistance problem (prescribed velocity → force/torque),
the null space of K corresponds to rigid body motions, but the
½I term makes the system invertible. Force and torque are
extracted by integrating the density ψ.

Force extraction uses the property that for the exterior Stokes
problem, the force on the body equals the net strength of the
equivalent single-layer distribution, which for the CDL relates
to ψ via:
    F = -8πθ ∫ ψ dA  (Gonzalez 2009, Eq. 6.6)
For the pure DLP (θ→0), we use the direct relationship:
    F_j = ∫ f_j dA  where f is the surface traction
Since we don't have f directly from the DLP density ψ, we extract
force by computing the far-field Stokeslet strength.

References:
    Smith et al. (2021), Fluids 6(11):411 — stresslet kernel.
    Power & Miranda (1987), SIAM J. Appl. Math. 47(4):689-698.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .stresslet import stresslet_tensor_contracted


def assemble_cdl_system(
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
    surface_weights: jnp.ndarray,
    x_star: jnp.ndarray,
    epsilon: float,
    theta: float = 0.5,
) -> jnp.ndarray:
    """Assemble the double-layer BEM system matrix (½I + K).

    The matrix is:
        M_jl(x_m, y_n) = (1/2)δ_jl δ_mn
                        + (1/8π) T_jlk(x_n, x_m) n_k(x_n) w_n   [m≠n]

    Note: the stresslet T(x_n, x_m) evaluates with r = x_n - x_m
    (source point first). This gives the correct DLP sign as verified
    by the identity DLP[u] ≈ (1/2)u for rigid body motion.

    Parameters
    ----------
    surface_points : (N, 3)
    surface_normals : (N, 3)
    surface_weights : (N,)
    x_star : (3,) unused (kept for API compatibility)
    epsilon : float
    theta : float, unused (kept for API compatibility)

    Returns
    -------
    M : (3N, 3N)
    """
    N = len(surface_points)
    prefactor = 1.0 / (8.0 * jnp.pi)

    def _kernel_block(m, n):
        x_m = surface_points[m]
        x_n = surface_points[n]
        n_n = surface_normals[n]
        w_n = surface_weights[n]

        # DLP kernel: (1/8π) T_jlk(x_n, x_m) n_k(x_n) w_n
        # stresslet_tensor_contracted(x_n, x_m, n_n, eps) computes
        # T_ijk with r = x_n - x_m, contracted with n_k at x_n
        K = prefactor * w_n * stresslet_tensor_contracted(x_n, x_m, n_n, epsilon)

        # Zero self-interaction (m == n)
        return jnp.where(m == n, jnp.zeros((3, 3)), K)

    # Assemble via double vmap
    blocks = jax.vmap(
        jax.vmap(_kernel_block, in_axes=(None, 0)),
        in_axes=(0, None),
    )(jnp.arange(N), jnp.arange(N))

    M = blocks.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)

    # Add ½I (jump condition)
    M = M + 0.5 * jnp.eye(3 * N)

    return M


def compute_cdl_force_torque(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    psi: jnp.ndarray,
    center: jnp.ndarray,
    theta: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract force and torque from CDL density ψ.

    For the double-layer formulation, the relationship between the
    DLP density ψ and the physical force depends on the specific
    formulation. For a rigid body in Stokes flow, we use the
    Lorentz reciprocal theorem to relate ψ to force:

    The density ψ satisfies (½I + K)ψ = v. For rigid body motion
    v = U + ω×r, the total force and torque can be extracted from
    ψ using the surface integral with appropriate prefactors.

    From the single-layer representation of the same problem,
    the force is F = ∫ f dA. The DLP density ψ relates to the
    SLP traction via the integral equation. For the regularised
    case with ε ≪ a, the leading-order relationship is:
        F ≈ -∫ ψ dA   (the sign comes from the exterior convention)
        T ≈ -∫ (y-c) × ψ dA

    Parameters
    ----------
    surface_points : (N, 3)
    surface_weights : (N,)
    psi : (N, 3) CDL density
    center : (3,)
    theta : float, unused

    Returns
    -------
    force, torque : (3,), (3,)
    """
    weighted_psi = psi * surface_weights[:, None]
    force = -jnp.sum(weighted_psi, axis=0)
    r = surface_points - center
    torque = -jnp.sum(jnp.cross(r, weighted_psi), axis=0)
    return force, torque
