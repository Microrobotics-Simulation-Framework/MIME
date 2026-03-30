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
    compute_dlp_rhs_correction,
    compute_force_torque,
    solve_bem_multi_rhs,
)
from .cdl_bem import assemble_cdl_system, compute_cdl_force_torque


def compute_resistance_matrix(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
    surface_normals: jnp.ndarray | None = None,
    use_dlp: bool = True,
) -> jnp.ndarray:
    """Compute the 6×6 resistance matrix by solving 6 BEM problems.

    Factorizes the system matrix once, then solves 6 RHS vectors
    (3 unit translations + 3 unit rotations). Each solution gives
    one column of R.

    When use_dlp=True (default), the RHS includes the double-layer
    potential correction from Smith et al. (2021), which significantly
    improves accuracy for confined problems. Requires surface_normals.

    Parameters
    ----------
    surface_points : (N, 3) collocation points
    surface_weights : (N,) quadrature weights
    center : (3,) body center of mass
    epsilon : float, regularisation parameter
    mu : float, dynamic viscosity
    surface_normals : (N, 3) outward unit normals (needed if use_dlp=True)
    use_dlp : bool
        If True, include double-layer potential correction in the RHS.

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
        U, omega = e[i], zero
        if use_dlp and surface_normals is not None:
            # Compute velocity field for this rigid motion
            r = surface_points - center
            vel = U + jnp.cross(omega, r)
            rhs = compute_dlp_rhs_correction(
                surface_points, surface_normals, surface_weights,
                vel, epsilon,
            )
        else:
            rhs = assemble_rhs_rigid_motion(surface_points, center, U, omega)
        rhs_columns.append(rhs)
    for i in range(3):
        U, omega = zero, e[i]
        if use_dlp and surface_normals is not None:
            r = surface_points - center
            vel = U + jnp.cross(omega, r)
            rhs = compute_dlp_rhs_correction(
                surface_points, surface_normals, surface_weights,
                vel, epsilon,
            )
        else:
            rhs = assemble_rhs_rigid_motion(surface_points, center, U, omega)
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


def compute_cdl_resistance_matrix(
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
    surface_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    theta: float = 0.5,
) -> jnp.ndarray:
    """Compute 6×6 resistance matrix using augmented CDL BEM.

    The augmented system is (3N+6) × (3N+6):
        [½I+K | S_col R_col] [q]   [u]
        [W_F  |   0     0  ] [α] = [0]
        [W_T  |   0     0  ] [β]   [0]

    Force = α (Stokeslet completion strength).
    Torque = β (rotlet completion strength).

    The prefactor on α, β depends on how the Stokeslet kernel
    is normalised. Our stokeslet_tensor omits the 1/(8πμ) prefactor,
    but the completion columns in assemble_cdl_system DO NOT include
    it either. The completion represents: u(x_m) += S(x_m, x_c) · α.
    Since S is the bare tensor (no 1/(8πμ)), α has units of
    [velocity · (r²+ε²)^(3/2)]. The physical force is obtained by
    calibrating against VER-020 (Stokes drag).

    Parameters
    ----------
    surface_points : (N, 3)
    surface_normals : (N, 3) outward normals
    surface_weights : (N,)
    center : (3,) body center (= x_* interior point)
    epsilon : float
    theta : float, unused

    Returns
    -------
    R : (6, 6) resistance matrix
    """
    N = len(surface_points)

    # Assemble augmented system (3N+6) × (3N+6)
    M = assemble_cdl_system(
        surface_points, surface_normals, surface_weights,
        center, epsilon,
    )

    # Build 6 RHS vectors: [velocity (3N), force_constraint (3), torque_constraint (3)]
    e = jnp.eye(3)
    zero = jnp.zeros(3)

    rhs_columns = []
    for i in range(3):
        # Unit translation along axis i
        r = surface_points - center
        vel = e[i] + jnp.cross(zero, r)  # = e[i] for translation
        rhs_top = vel.ravel()                    # (3N,)
        rhs_bottom = jnp.zeros(6)                # force/torque constraints = 0
        rhs_columns.append(jnp.concatenate([rhs_top, rhs_bottom]))
    for i in range(3):
        # Unit rotation about axis i
        r = surface_points - center
        vel = zero + jnp.cross(e[i], r)
        rhs_top = vel.ravel()
        rhs_bottom = jnp.zeros(6)
        rhs_columns.append(jnp.concatenate([rhs_top, rhs_bottom]))

    rhs_matrix = jnp.stack(rhs_columns, axis=1)  # (3N+6, 6)

    # Solve
    solutions = solve_bem_multi_rhs(M, rhs_matrix)  # (3N+6, 6)

    # Extract α (force) and β (torque) from the last 6 entries
    # The Stokeslet kernel is the bare tensor (no 1/(8πμ) prefactor),
    # so Force = 8πμ · α. The rotlet convention gives Torque = -8πμ · β.
    # μ is not passed to this function — it cancels in the resistance
    # matrix (R maps velocity to force, both scale with μ). The factor
    # 8π comes from the Stokeslet normalisation.
    scale_F = 8.0 * jnp.pi
    scale_T = -8.0 * jnp.pi

    R = jnp.zeros((6, 6))
    for col in range(6):
        alpha = solutions[3 * N: 3 * N + 3, col]
        beta = solutions[3 * N + 3: 3 * N + 6, col]
        R = R.at[:3, col].set(scale_F * alpha)
        R = R.at[3:, col].set(scale_T * beta)

    return R


def compute_confined_resistance_matrix(
    body_points: jnp.ndarray,
    body_weights: jnp.ndarray,
    wall_points: jnp.ndarray,
    wall_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
    body_normals: jnp.ndarray | None = None,
    wall_normals: jnp.ndarray | None = None,
    use_dlp: bool = True,
) -> jnp.ndarray:
    """Compute 6×6 resistance matrix with vessel wall confinement.

    Same as compute_resistance_matrix but includes wall surface
    points with no-slip BCs. The combined system is larger but
    only body-surface tractions contribute to force/torque.

    When use_dlp=True, includes the double-layer potential correction
    on the combined (body+wall) RHS.

    Parameters
    ----------
    body_points : (N_b, 3)
    body_weights : (N_b,)
    wall_points : (N_w, 3)
    wall_weights : (N_w,)
    center : (3,)
    epsilon : float
    mu : float
    body_normals : (N_b, 3) or None
    wall_normals : (N_w, 3) or None
    use_dlp : bool

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

    can_dlp = (use_dlp and body_normals is not None and wall_normals is not None)

    rhs_columns = []
    for i in range(3):
        U, omega = e[i], zero
        if can_dlp:
            # Compute velocity on all surfaces
            r_body = body_points - center
            vel_body = U + jnp.cross(omega, r_body)
            vel_wall = jnp.zeros((N_w, 3))
            all_pts = jnp.concatenate([body_points, wall_points])
            all_normals = jnp.concatenate([body_normals, wall_normals])
            all_weights = jnp.concatenate([body_weights, wall_weights])
            all_vel = jnp.concatenate([vel_body, vel_wall])
            rhs = compute_dlp_rhs_correction(
                all_pts, all_normals, all_weights, all_vel, epsilon,
            )
        else:
            rhs = assemble_rhs_confined(body_points, N_w, center, U, omega)
        rhs_columns.append(rhs)
    for i in range(3):
        U, omega = zero, e[i]
        if can_dlp:
            r_body = body_points - center
            vel_body = U + jnp.cross(omega, r_body)
            vel_wall = jnp.zeros((N_w, 3))
            all_pts = jnp.concatenate([body_points, wall_points])
            all_normals = jnp.concatenate([body_normals, wall_normals])
            all_weights = jnp.concatenate([body_weights, wall_weights])
            all_vel = jnp.concatenate([vel_body, vel_wall])
            rhs = compute_dlp_rhs_correction(
                all_pts, all_normals, all_weights, all_vel, epsilon,
            )
        else:
            rhs = assemble_rhs_confined(body_points, N_w, center, U, omega)
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
