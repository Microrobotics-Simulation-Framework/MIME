"""6×6 resistance matrix computation for rigid bodies in Stokes flow.

The resistance matrix R relates rigid body motion [U, ω] to
hydrodynamic force/torque [F, T]:

    [F]   [R_FU  R_Fω] [U]
    [T] = [R_TU  R_Tω] [ω]

R is computed by solving 6 BEM problems (one per column): 3 unit
translations and 3 unit rotations. The resulting force/torque from
each gives one column of R.

No separate rotlet kernel is needed. The rotation-coupling blocks
(R_Fω, R_Tω) are obtained by setting the BEM RHS to the rigid
rotation velocity field u(x) = ω × (x - x_c) and solving the
standard Stokeslet system Af = u. The integrated force/torque
from the solved traction directly gives the rotation columns.

Reference values for validation:
    Sphere drag: F = 6πμaU, T = 8πμa³ω (Stokes 1851)
    Sphere in cylinder: Haberman & Sayre (1958), DTMB Report 1143
    f = (1 − 2.105λ + 2.0865λ³ − 1.7068λ⁵ + 0.72603λ⁶)
        / (1 − 0.75857λ⁵)
    where λ = d_particle / D_cylinder
"""

from __future__ import annotations

import jax.numpy as jnp

from .bem import (
    assemble_system_matrix,
    assemble_confined_system,
    assemble_rhs_rigid_motion,
    assemble_rhs_confined,
    compute_force_torque,
    solve_bem_multi_rhs,
)


def compute_resistance_matrix(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Compute the 6×6 resistance matrix by solving 6 BEM problems.

    Factorizes the system matrix once, then solves 6 RHS vectors
    (3 unit translations + 3 unit rotations). Each solution gives
    one column of R.

    Parameters
    ----------
    surface_points : (N, 3) collocation points
    surface_weights : (N,) quadrature weights
    center : (3,) body center of mass
    epsilon : float, regularisation parameter
    mu : float, dynamic viscosity

    Returns
    -------
    R : (6, 6) resistance matrix [F; T] = R @ [U; ω]
    """
    N = len(surface_points)

    # Assemble system matrix (same for all 6 problems)
    A = assemble_system_matrix(surface_points, surface_weights, epsilon, mu)

    # Build 6 RHS vectors: 3 unit translations + 3 unit rotations
    e = jnp.eye(3)
    zero = jnp.zeros(3)

    rhs_columns = []
    for i in range(3):
        # Unit translation along axis i
        rhs = assemble_rhs_rigid_motion(surface_points, center, e[i], zero)
        rhs_columns.append(rhs)
    for i in range(3):
        # Unit rotation about axis i
        rhs = assemble_rhs_rigid_motion(surface_points, center, zero, e[i])
        rhs_columns.append(rhs)

    rhs_matrix = jnp.stack(rhs_columns, axis=1)  # (3N, 6)

    # Solve all 6 at once via LU factorization
    solutions = solve_bem_multi_rhs(A, rhs_matrix)  # (3N, 6)

    # Extract force and torque for each column → build R
    R = jnp.zeros((6, 6))
    for col in range(6):
        traction = solutions[:, col].reshape(N, 3)
        F, T = compute_force_torque(
            surface_points, surface_weights, traction, center,
        )
        R = R.at[:3, col].set(F)
        R = R.at[3:, col].set(T)

    return R


def compute_confined_resistance_matrix(
    body_points: jnp.ndarray,
    body_weights: jnp.ndarray,
    wall_points: jnp.ndarray,
    wall_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Compute 6×6 resistance matrix with vessel wall confinement.

    Same as compute_resistance_matrix but includes wall surface
    points with no-slip BCs. The combined system is larger but
    only body-surface tractions contribute to force/torque.

    Parameters
    ----------
    body_points : (N_b, 3)
    body_weights : (N_b,)
    wall_points : (N_w, 3)
    wall_weights : (N_w,)
    center : (3,)
    epsilon : float
    mu : float

    Returns
    -------
    R : (6, 6) confined resistance matrix
    """
    N_b = len(body_points)
    N_w = len(wall_points)

    # Assemble combined system matrix
    A = assemble_confined_system(
        body_points, body_weights, wall_points, wall_weights, epsilon, mu,
    )

    # Build 6 RHS vectors with wall no-slip
    e = jnp.eye(3)
    zero = jnp.zeros(3)

    rhs_columns = []
    for i in range(3):
        rhs = assemble_rhs_confined(body_points, N_w, center, e[i], zero)
        rhs_columns.append(rhs)
    for i in range(3):
        rhs = assemble_rhs_confined(body_points, N_w, center, zero, e[i])
        rhs_columns.append(rhs)

    rhs_matrix = jnp.stack(rhs_columns, axis=1)  # (3*(N_b+N_w), 6)

    # Solve
    solutions = solve_bem_multi_rhs(A, rhs_matrix)

    # Extract force/torque from BODY portion only
    R = jnp.zeros((6, 6))
    for col in range(6):
        # Only the first 3*N_b entries are body tractions
        body_traction = solutions[:3 * N_b, col].reshape(N_b, 3)
        F, T = compute_force_torque(body_points, body_weights, body_traction, center)
        R = R.at[:3, col].set(F)
        R = R.at[3:, col].set(T)

    return R
