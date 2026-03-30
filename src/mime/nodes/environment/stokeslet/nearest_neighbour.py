"""Nearest-neighbour regularised Stokeslet method (Smith 2018).

Decouples force discretization (N coarse points) from quadrature
(Q fine points). The Stokeslet integral is evaluated at Q quadrature
points but the force density is piecewise-constant on Voronoi cells
defined by N force points. This allows ε to be tied to the fine
quadrature spacing h_q while the linear system is only 3N × 3N.

The system matrix is A = K @ P where:
    K : (3N, 3Q) — Stokeslet evaluations from N field to Q source points
    P : (3Q, 3N) — nearest-neighbour interpolation (sparse)

Reference:
    Smith (2018), "A nearest-neighbour discretisation of the regularized
    stokeslet boundary integral equation", J. Comput. Phys. 358:88-102.
    Eq. 20-26, Appendix A.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import stokeslet_tensor


def compute_nearest_neighbour_map(
    force_points: np.ndarray,
    quad_points: np.ndarray,
) -> np.ndarray:
    """Compute nearest-neighbour assignment: quad → force.

    For each quadrature point q, find the nearest force point n.
    Uses chunked computation to avoid memory issues with large meshes.

    Parameters
    ----------
    force_points : (N, 3) coarse force/collocation points
    quad_points : (Q, 3) fine quadrature points

    Returns
    -------
    nn_indices : (Q,) int — nn_indices[q] = index of nearest force point
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(force_points)
    _, nn_indices = tree.query(quad_points)
    return nn_indices


def build_nearest_neighbour_matrix(
    nn_indices: np.ndarray,
    quad_weights: np.ndarray,
    n_force: int,
) -> jnp.ndarray:
    """Build the (3Q, 3N) nearest-neighbour interpolation matrix P.

    P maps force densities at N coarse points to weighted contributions
    at Q fine quadrature points. Each row q has a single nonzero block:
    P[3q:3q+3, 3n:3n+3] = w_q · I_3  where n = nn_indices[q].

    Parameters
    ----------
    nn_indices : (Q,) nearest force point for each quad point
    quad_weights : (Q,) quadrature weights at fine points
    n_force : int, number of force points N

    Returns
    -------
    P : (3Q, 3N) dense matrix (sparse structure but stored dense for JAX)
    """
    Q = len(nn_indices)
    N = n_force

    # Build as numpy first (index assignment), convert to JAX
    P = np.zeros((3 * Q, 3 * N), dtype=np.float64)
    for q in range(Q):
        n = nn_indices[q]
        w = quad_weights[q]
        for j in range(3):
            P[3 * q + j, 3 * n + j] = w

    return jnp.array(P)


def assemble_stokeslet_field_matrix(
    field_points: jnp.ndarray,
    source_points: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Assemble (3M, 3Q) Stokeslet matrix from M field to Q source points.

    K[3m:3m+3, 3q:3q+3] = (1/8πμ) · S^ε(x[m], X[q])

    Note: NO quadrature weights here — those are in the P matrix.

    Parameters
    ----------
    field_points : (M, 3) target/evaluation points
    source_points : (Q, 3) source/quadrature points
    epsilon : float
    mu : float

    Returns
    -------
    K : (3M, 3Q) dense matrix
    """
    prefactor = 1.0 / (8.0 * jnp.pi * mu)

    def _row_of_blocks(x_m):
        def _single_block(x_q):
            return prefactor * stokeslet_tensor(x_m, x_q, epsilon)
        return jax.vmap(_single_block)(source_points)

    # blocks shape: (M, Q, 3, 3)
    blocks = jax.vmap(_row_of_blocks)(field_points)

    M = len(field_points)
    Q = len(source_points)
    K = blocks.transpose(0, 2, 1, 3).reshape(3 * M, 3 * Q)
    return K


def assemble_nn_system_matrix(
    force_points: jnp.ndarray,
    quad_points: jnp.ndarray,
    quad_weights: jnp.ndarray,
    nn_indices: np.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Assemble the 3N × 3N nearest-neighbour system matrix.

    A = K @ P  where:
        K : (3N, 3Q) Stokeslet from force points to quad points
        P : (3Q, 3N) nearest-neighbour interpolation

    Parameters
    ----------
    force_points : (N, 3) coarse collocation/force points
    quad_points : (Q, 3) fine quadrature points
    quad_weights : (Q,) fine quadrature weights
    nn_indices : (Q,) int, nearest force point for each quad point
    epsilon : float — should be tied to fine spacing h_q
    mu : float

    Returns
    -------
    A : (3N, 3N) system matrix
    """
    N = len(force_points)

    K = assemble_stokeslet_field_matrix(
        force_points, quad_points, epsilon, mu,
    )  # (3N, 3Q)

    P = build_nearest_neighbour_matrix(
        nn_indices, quad_weights, N,
    )  # (3Q, 3N)

    return K @ P  # (3N, 3N)


def assemble_nn_confined_system(
    body_force_pts: jnp.ndarray,
    body_quad_pts: jnp.ndarray,
    body_quad_wts: jnp.ndarray,
    body_nn: np.ndarray,
    wall_force_pts: jnp.ndarray,
    wall_quad_pts: jnp.ndarray,
    wall_quad_wts: jnp.ndarray,
    wall_nn: np.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Assemble confined NN system with body + wall.

    The combined system is:
        [A_bb  A_bw] [f_body]   [u_body]
        [A_wb  A_ww] [f_wall] = [0     ]

    Each block uses the nearest-neighbour discretization:
        A_bb = K(body_field → body_quad) @ P_body
        A_bw = K(body_field → wall_quad) @ P_wall
        A_wb = K(wall_field → body_quad) @ P_body
        A_ww = K(wall_field → wall_quad) @ P_wall

    Parameters
    ----------
    body_force_pts : (N_b, 3) body force/collocation points
    body_quad_pts : (Q_b, 3) body fine quadrature points
    body_quad_wts : (Q_b,)
    body_nn : (Q_b,) int
    wall_force_pts : (N_w, 3) wall force/collocation points
    wall_quad_pts : (Q_w, 3) wall fine quadrature points
    wall_quad_wts : (Q_w,)
    wall_nn : (Q_w,) int
    epsilon : float
    mu : float

    Returns
    -------
    A : (3*(N_b+N_w), 3*(N_b+N_w))
    """
    N_b = len(body_force_pts)
    N_w = len(wall_force_pts)

    P_body = build_nearest_neighbour_matrix(body_nn, body_quad_wts, N_b)
    P_wall = build_nearest_neighbour_matrix(wall_nn, wall_quad_wts, N_w)

    prefactor = 1.0 / (8.0 * jnp.pi * mu)

    # K_bb: body field → body quad sources
    K_bb = assemble_stokeslet_field_matrix(
        body_force_pts, body_quad_pts, epsilon, mu)
    # K_bw: body field → wall quad sources
    K_bw = assemble_stokeslet_field_matrix(
        body_force_pts, wall_quad_pts, epsilon, mu)
    # K_wb: wall field → body quad sources
    K_wb = assemble_stokeslet_field_matrix(
        wall_force_pts, body_quad_pts, epsilon, mu)
    # K_ww: wall field → wall quad sources
    K_ww = assemble_stokeslet_field_matrix(
        wall_force_pts, wall_quad_pts, epsilon, mu)

    A_bb = K_bb @ P_body  # (3N_b, 3N_b)
    A_bw = K_bw @ P_wall  # (3N_b, 3N_w)
    A_wb = K_wb @ P_body  # (3N_w, 3N_b)
    A_ww = K_ww @ P_wall  # (3N_w, 3N_w)

    top = jnp.concatenate([A_bb, A_bw], axis=1)
    bottom = jnp.concatenate([A_wb, A_ww], axis=1)
    return jnp.concatenate([top, bottom], axis=0)
