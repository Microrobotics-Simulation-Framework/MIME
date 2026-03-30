"""BEM assembly, solve, and force/torque extraction.

Assembles the dense Nyström BEM system matrix from regularised
Stokeslet evaluations, solves via LU factorization, and extracts
total hydrodynamic force and torque from the surface traction.

Upgrade paths (not implemented, noted for future):
    - Smith (2018) nearest-neighbour discretization (10x cost reduction):
      DOI: 10.1016/j.jcp.2017.12.008
    - Gallagher & Smith (2021) Richardson extrapolation in ε:
      DOI: 10.1098/rsos.210108
      Local PDF: tmp/stokelet_403_papers/rsos.210108.pdf
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from .kernel import stokeslet_tensor


def assemble_system_matrix(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Assemble the 3N × 3N BEM system matrix.

    A[3m:3m+3, 3n:3n+3] = (w_n / 8πμ) · S^ε(x_m, y_n)

    Parameters
    ----------
    surface_points : (N, 3) collocation points
    surface_weights : (N,) quadrature weights
    epsilon : float
    mu : float, dynamic viscosity

    Returns
    -------
    A : (3N, 3N) dense matrix
    """
    N = len(surface_points)
    prefactor = 1.0 / (8.0 * jnp.pi * mu)

    # Double vmap: for each target m, for each source n, compute S^ε
    def _row_of_blocks(x_m):
        def _single_block(x_n, w_n):
            return prefactor * w_n * stokeslet_tensor(x_m, x_n, epsilon)
        return jax.vmap(_single_block)(surface_points, surface_weights)

    # blocks shape: (N, N, 3, 3)
    blocks = jax.vmap(_row_of_blocks)(surface_points)

    # Reshape to (3N, 3N)
    A = blocks.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    return A


def assemble_rhs_rigid_motion(
    surface_points: jnp.ndarray,
    center: jnp.ndarray,
    U: jnp.ndarray,
    omega: jnp.ndarray,
) -> jnp.ndarray:
    """Compute RHS vector for prescribed rigid body motion.

    u(x_m) = U + ω × (x_m - center)

    Parameters
    ----------
    surface_points : (N, 3)
    center : (3,)
    U : (3,) translational velocity
    omega : (3,) angular velocity

    Returns
    -------
    rhs : (3N,) velocity vector
    """
    r = surface_points - center
    u = U + jnp.cross(omega, r)
    return u.ravel()


def assemble_rhs_confined(
    body_points: jnp.ndarray,
    n_wall: int,
    center: jnp.ndarray,
    U: jnp.ndarray,
    omega: jnp.ndarray,
) -> jnp.ndarray:
    """Compute RHS for confined system (body + wall).

    u_body = U + ω × (x_m - center)  for body points
    u_wall = 0                        for wall points

    Parameters
    ----------
    body_points : (N_body, 3)
    n_wall : int, number of wall points
    center : (3,)
    U : (3,) translational velocity
    omega : (3,) angular velocity

    Returns
    -------
    rhs : (3*(N_body + N_wall),) velocity vector
    """
    r = body_points - center
    u_body = U + jnp.cross(omega, r)
    u_wall = jnp.zeros((n_wall, 3))
    return jnp.concatenate([u_body.ravel(), u_wall.ravel()])


def assemble_confined_system(
    body_points: jnp.ndarray,
    body_weights: jnp.ndarray,
    wall_points: jnp.ndarray,
    wall_weights: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Assemble the combined body+wall BEM system matrix.

    [A_bb  A_bw] [f_body]   [u_body]
    [A_wb  A_ww] [f_wall] = [u_wall]

    Parameters
    ----------
    body_points : (N_b, 3)
    body_weights : (N_b,)
    wall_points : (N_w, 3)
    wall_weights : (N_w,)
    epsilon : float
    mu : float

    Returns
    -------
    A : (3*(N_b+N_w), 3*(N_b+N_w)) dense matrix
    """
    all_points = jnp.concatenate([body_points, wall_points], axis=0)
    all_weights = jnp.concatenate([body_weights, wall_weights])
    return assemble_system_matrix(all_points, all_weights, epsilon, mu)


def solve_bem(
    A: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve Af = rhs via direct solve.

    Parameters
    ----------
    A : (3N, 3N) system matrix
    rhs : (3N,) or (3N, K) right-hand side(s)

    Returns
    -------
    f : (3N,) or (3N, K) surface traction densities
    """
    return jnp.linalg.solve(A, rhs)


def solve_bem_multi_rhs(
    A: jnp.ndarray,
    rhs_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Solve Af = b for multiple RHS vectors via LU factorization.

    Factorizes once, then solves for each column of rhs_matrix.

    JIT COMPILATION NOTE: LU factorization of large matrices (e.g.
    39k×39k for N=13,000) triggers XLA compilation that takes several
    minutes on first call. This is intentional — it should be called
    at init time, not inside update().

    Parameters
    ----------
    A : (3N, 3N) system matrix
    rhs_matrix : (3N, K) multiple right-hand sides

    Returns
    -------
    solutions : (3N, K) solutions
    """
    lu, piv = jax.scipy.linalg.lu_factor(A)
    return jax.scipy.linalg.lu_solve((lu, piv), rhs_matrix)


def compute_force_torque(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    traction: jnp.ndarray,
    center: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate surface traction to get total force and torque.

    F_j = Σ_n f_j(y_n) · w_n
    T_j = Σ_n (y_n - center) × f(y_n) · w_n

    Parameters
    ----------
    surface_points : (N, 3)
    surface_weights : (N,)
    traction : (N, 3) force density at each point
    center : (3,) moment reference point

    Returns
    -------
    force : (3,)
    torque : (3,)
    """
    weighted_f = traction * surface_weights[:, None]
    force = jnp.sum(weighted_f, axis=0)

    r = surface_points - center
    torque = jnp.sum(jnp.cross(r, weighted_f), axis=0)

    return force, torque
