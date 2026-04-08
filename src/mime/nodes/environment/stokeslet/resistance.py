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
from .nearest_neighbour import (
    compute_nearest_neighbour_map,
    assemble_nn_system_matrix,
    assemble_nn_confined_system,
)

import numpy as np_cpu  # for Richardson weights (not JAX)


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
    n_body: int | None = None,
    x_wall_star: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute 6×6 resistance matrix using augmented CDL BEM.

    For unconfined: pass all points as body (n_body=None).
    For confined: set n_body = number of body points, provide
    x_wall_star = completion point outside the wall.

    Force = 8π · α_b, Torque = -8π · β_b.

    Parameters
    ----------
    surface_points : (N, 3) — body points first, then wall
    surface_normals : (N, 3) outward normals
    surface_weights : (N,)
    center : (3,) body center (= x_b interior point)
    epsilon : float
    theta : float, unused
    n_body : int or None
    x_wall_star : (3,) or None

    Returns
    -------
    R : (6, 6) resistance matrix
    """
    N = len(surface_points)
    confined = (n_body is not None and x_wall_star is not None)
    N_b = n_body if confined else N
    n_completion = 12 if confined else 6

    M = assemble_cdl_system(
        surface_points, surface_normals, surface_weights,
        center, epsilon, n_body=n_body, x_wall_star=x_wall_star,
    )

    # Build 6 RHS vectors
    e = jnp.eye(3)
    zero = jnp.zeros(3)

    rhs_columns = []
    for i in range(3):
        r = surface_points - center
        vel = e[i] + jnp.cross(zero, r)
        # Body points get rigid motion, wall points get zero
        if confined:
            vel = vel.at[N_b:].set(0.0)
        rhs_top = vel.ravel()
        rhs_bottom = jnp.zeros(n_completion)
        rhs_columns.append(jnp.concatenate([rhs_top, rhs_bottom]))
    for i in range(3):
        r = surface_points - center
        vel = zero + jnp.cross(e[i], r)
        if confined:
            vel = vel.at[N_b:].set(0.0)
        rhs_top = vel.ravel()
        rhs_bottom = jnp.zeros(n_completion)
        rhs_columns.append(jnp.concatenate([rhs_top, rhs_bottom]))

    rhs_matrix = jnp.stack(rhs_columns, axis=1)
    solutions = solve_bem_multi_rhs(M, rhs_matrix)

    # Extract α_b, β_b (body completion strengths)
    scale_F = 8.0 * jnp.pi
    scale_T = -8.0 * jnp.pi

    R = jnp.zeros((6, 6))
    for col in range(6):
        alpha_b = solutions[3 * N: 3 * N + 3, col]
        beta_b = solutions[3 * N + 3: 3 * N + 6, col]
        R = R.at[:3, col].set(scale_F * alpha_b)
        R = R.at[3:, col].set(scale_T * beta_b)

    return R


def compute_cdl_resistance_matrix_richardson(
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
    surface_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    theta: float = 0.5,
    n_body: int | None = None,
    x_wall_star: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """CDL resistance matrix with Richardson extrapolation in ε.

    Solves at three ε values (ε, √2·ε, 2ε) and linearly combines
    to cancel O(ε) and O(ε²) regularisation error. Costs 3× the
    single CDL solve but dramatically improves accuracy.

    Reference: Gallagher & Smith (2021), R. Soc. Open Sci. 8:210108.
    Eq. 3.6-3.8, Section 5.

    Parameters
    ----------
    Same as compute_cdl_resistance_matrix.

    Returns
    -------
    R : (6, 6) Richardson-extrapolated resistance matrix
    """
    # Richardson weights from Vandermonde inverse (Eq. 3.8)
    # For (ε₁, ε₂, ε₃) = (ε, √2·ε, 2·ε):
    r1, r2, r3 = 1.0, np_cpu.sqrt(2), 2.0
    B = np_cpu.array([[1, r1, r1**2], [1, r2, r2**2], [1, r3, r3**2]])
    weights = np_cpu.linalg.inv(B)[0]  # first row of B⁻¹

    eps_values = [epsilon * r1, epsilon * r2, epsilon * r3]

    R_sum = jnp.zeros((6, 6))
    for i, eps_i in enumerate(eps_values):
        R_i = compute_cdl_resistance_matrix(
            surface_points, surface_normals, surface_weights,
            center, eps_i, theta, n_body, x_wall_star,
        )
        R_sum = R_sum + float(weights[i]) * R_i

    return R_sum


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


def compute_nn_resistance_matrix(
    force_points: jnp.ndarray,
    force_weights: jnp.ndarray,
    quad_points: jnp.ndarray,
    quad_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Compute 6×6 resistance matrix using nearest-neighbour method.

    Uses two-level discretization: coarse N force points for the
    unknown traction, fine Q quadrature points for integration.
    ε is tied to the fine spacing h_q, giving better accuracy than
    standard Nyström at the same DOF count.

    Parameters
    ----------
    force_points : (N, 3) coarse force/collocation points
    force_weights : (N,) coarse weights (for force integration)
    quad_points : (Q, 3) fine quadrature points
    quad_weights : (Q,) fine quadrature weights
    center : (3,) body center
    epsilon : float — should be ~ h_q / 2
    mu : float

    Returns
    -------
    R : (6, 6) resistance matrix
    """
    N = len(force_points)

    # Compute nearest-neighbour map
    nn_indices = compute_nearest_neighbour_map(
        np_cpu.array(force_points), np_cpu.array(quad_points),
    )

    # Assemble system matrix A = K @ P
    A = assemble_nn_system_matrix(
        force_points, quad_points, quad_weights,
        nn_indices, epsilon, mu,
    )

    # Build 6 RHS: rigid body velocities at force points
    e = jnp.eye(3)
    zero = jnp.zeros(3)

    rhs_columns = []
    for i in range(3):
        rhs = assemble_rhs_rigid_motion(force_points, center, e[i], zero)
        rhs_columns.append(rhs)
    for i in range(3):
        rhs = assemble_rhs_rigid_motion(force_points, center, zero, e[i])
        rhs_columns.append(rhs)

    rhs_matrix = jnp.stack(rhs_columns, axis=1)  # (3N, 6)
    solutions = solve_bem_multi_rhs(A, rhs_matrix)  # (3N, 6)

    # Extract force/torque using coarse force weights
    R = jnp.zeros((6, 6))
    for col in range(6):
        traction = solutions[:, col].reshape(N, 3)
        F, T = compute_force_torque(
            force_points, force_weights, traction, center,
        )
        R = R.at[:3, col].set(F)
        R = R.at[3:, col].set(T)

    return R


def compute_nn_confined_resistance_matrix(
    body_force_pts: jnp.ndarray,
    body_force_wts: jnp.ndarray,
    body_quad_pts: jnp.ndarray,
    body_quad_wts: jnp.ndarray,
    wall_force_pts: jnp.ndarray,
    wall_force_wts: jnp.ndarray,
    wall_quad_pts: jnp.ndarray,
    wall_quad_wts: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Compute 6×6 confined resistance matrix using NN method.

    Two-level body + wall discretization. Force extraction uses
    only the body portion of the solution.

    Parameters
    ----------
    body_force_pts : (N_b, 3) coarse body force points
    body_force_wts : (N_b,) coarse body weights
    body_quad_pts : (Q_b, 3) fine body quadrature points
    body_quad_wts : (Q_b,)
    wall_force_pts : (N_w, 3) coarse wall force points
    wall_force_wts : (N_w,) coarse wall weights
    wall_quad_pts : (Q_w, 3) fine wall quadrature points
    wall_quad_wts : (Q_w,)
    center : (3,)
    epsilon : float
    mu : float

    Returns
    -------
    R : (6, 6) confined resistance matrix
    """
    N_b = len(body_force_pts)
    N_w = len(wall_force_pts)

    # NN maps for body and wall separately
    body_nn = compute_nearest_neighbour_map(
        np_cpu.array(body_force_pts), np_cpu.array(body_quad_pts),
    )
    wall_nn = compute_nearest_neighbour_map(
        np_cpu.array(wall_force_pts), np_cpu.array(wall_quad_pts),
    )

    # Assemble confined system
    A = assemble_nn_confined_system(
        body_force_pts, body_quad_pts, body_quad_wts, body_nn,
        wall_force_pts, wall_quad_pts, wall_quad_wts, wall_nn,
        epsilon, mu,
    )

    # Build 6 RHS at coarse force points
    e = jnp.eye(3)
    zero = jnp.zeros(3)

    rhs_columns = []
    for i in range(3):
        rhs = assemble_rhs_confined(body_force_pts, N_w, center, e[i], zero)
        rhs_columns.append(rhs)
    for i in range(3):
        rhs = assemble_rhs_confined(body_force_pts, N_w, center, zero, e[i])
        rhs_columns.append(rhs)

    rhs_matrix = jnp.stack(rhs_columns, axis=1)
    solutions = solve_bem_multi_rhs(A, rhs_matrix)

    # Extract force/torque from body portion only
    R = jnp.zeros((6, 6))
    for col in range(6):
        body_traction = solutions[:3 * N_b, col].reshape(N_b, 3)
        F, T = compute_force_torque(
            body_force_pts, body_force_wts, body_traction, center,
        )
        R = R.at[:3, col].set(F)
        R = R.at[3:, col].set(T)

    return R


def compute_nn_confined_resistance_richardson(
    body_force_pts: jnp.ndarray,
    body_force_wts: jnp.ndarray,
    body_quad_pts: jnp.ndarray,
    body_quad_wts: jnp.ndarray,
    wall_force_pts: jnp.ndarray,
    wall_force_wts: jnp.ndarray,
    wall_quad_pts: jnp.ndarray,
    wall_quad_wts: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """NN confined resistance with Richardson extrapolation in ε.

    Solves at three ε values (ε, √2·ε, 2ε) and linearly combines
    to cancel O(ε) and O(ε²) regularisation error.

    Reference: Gallagher & Smith (2021), R. Soc. Open Sci. 8:210108.
    """
    r1, r2, r3 = 1.0, np_cpu.sqrt(2), 2.0
    B = np_cpu.array([[1, r1, r1**2], [1, r2, r2**2], [1, r3, r3**2]])
    weights = np_cpu.linalg.inv(B)[0]

    R_sum = jnp.zeros((6, 6))
    for i, ri in enumerate([r1, r2, r3]):
        R_i = compute_nn_confined_resistance_matrix(
            body_force_pts, body_force_wts,
            body_quad_pts, body_quad_wts,
            wall_force_pts, wall_force_wts,
            wall_quad_pts, wall_quad_wts,
            center, epsilon * ri, mu,
        )
        R_sum = R_sum + float(weights[i]) * R_i

    return R_sum


def compute_gcyl_confined_resistance_matrix(
    body_points: jnp.ndarray,
    body_normals: jnp.ndarray,
    body_weights: jnp.ndarray,
    center: jnp.ndarray,
    epsilon: float,
    mu: float,
    R_cyl: float,
    n_max: int = 15,
    n_k: int = 80,
    n_phi: int = 64,
    use_dlp: bool = True,
) -> jnp.ndarray:
    """Compute 6×6 confined resistance matrix using the Liron-Shahar
    cylindrical Green's function for the wall.

    The system matrix is:
        A_confined = A_body_BEM + G_wall

    where A_body_BEM is the free-space regularised BEM matrix and
    G_wall is the analytical wall correction from the Fourier-Bessel
    image system. No wall mesh is needed.

    The wall correction interface is a pure function
    (``assemble_image_correction_matrix``) that can be swapped for
    FMM or H-matrix compression without touching the body solver.

    Parameters
    ----------
    body_points : (N, 3) collocation points on body surface
    body_normals : (N, 3) outward unit normals
    body_weights : (N,) quadrature weights
    center : (3,) body center of mass / rotation reference
    epsilon : float, BEM regularisation parameter
    mu : float, dynamic viscosity
    R_cyl : float, cylinder wall radius
    n_max, n_k, n_phi : int, Fourier-Bessel truncation parameters
    use_dlp : bool, include Smith et al. (2021) DLP correction

    Returns
    -------
    R : (6, 6) resistance matrix [F; T] = R @ [U; ω]
    """
    from .cylinder_greens_function_v2 import assemble_image_correction_matrix

    N = len(body_points)
    pts_np = np_cpu.asarray(body_points)
    wts_np = np_cpu.asarray(body_weights)

    # Sanity check: body must be inside the cylinder and approximately
    # centered on its axis, otherwise direction-independence breaks.
    rho_body = np_cpu.sqrt(pts_np[:, 0]**2 + pts_np[:, 1]**2)
    assert np_cpu.all(rho_body < R_cyl), (
        f"Body extends outside cylinder: max(ρ)={rho_body.max():.3f} ≥ R={R_cyl:.3f}"
    )
    centroid_offset = np_cpu.sqrt(
        np_cpu.mean(pts_np[:, 0])**2 + np_cpu.mean(pts_np[:, 1])**2
    )
    if centroid_offset > 0.05 * R_cyl:
        import warnings
        warnings.warn(
            f"Body centroid is {centroid_offset:.3f} off cylinder axis "
            f"(R_cyl={R_cyl:.3f}). Direction-independence may be poor."
        )

    # Free-space BEM body matrix (regularised Stokeslet)
    A_free = assemble_system_matrix(body_points, body_weights, epsilon, mu)

    # Analytical wall correction (Liron-Shahar)
    G_wall = assemble_image_correction_matrix(
        pts_np, wts_np, R_cyl, mu,
        n_max=n_max, n_k=n_k, n_phi=n_phi,
    )

    # Combined confined system
    A = A_free + jnp.array(G_wall)

    # Build 6 RHS vectors (3 translations + 3 rotations)
    e = jnp.eye(3)
    zero = jnp.zeros(3)

    rhs_columns = []
    for i in range(3):
        U, omega = e[i], zero
        if use_dlp and body_normals is not None:
            r = body_points - center
            vel = U + jnp.cross(omega, r)
            rhs = compute_dlp_rhs_correction(
                body_points, body_normals, body_weights, vel, epsilon,
            )
        else:
            rhs = assemble_rhs_rigid_motion(body_points, center, U, omega)
        rhs_columns.append(rhs)
    for i in range(3):
        U, omega = zero, e[i]
        if use_dlp and body_normals is not None:
            r = body_points - center
            vel = U + jnp.cross(omega, r)
            rhs = compute_dlp_rhs_correction(
                body_points, body_normals, body_weights, vel, epsilon,
            )
        else:
            rhs = assemble_rhs_rigid_motion(body_points, center, zero, e[i])
        rhs_columns.append(rhs)

    rhs_matrix = jnp.stack(rhs_columns, axis=1)  # (3N, 6)

    # Solve all 6 at once
    solutions = solve_bem_multi_rhs(A, rhs_matrix)  # (3N, 6)

    # Extract force/torque per column → build R
    R = jnp.zeros((6, 6))
    for col in range(6):
        traction = solutions[:, col].reshape(N, 3)
        F, T = compute_force_torque(
            body_points, body_weights, traction, center,
        )
        R = R.at[:3, col].set(F)
        R = R.at[3:, col].set(T)

    return R
