"""BEM assembly, solve, and force/torque extraction.

Assembles the dense Nyström BEM system matrix from regularised
Stokeslet evaluations, solves via LU factorization, and extracts
total hydrodynamic force and torque from the surface traction.

Two assembly backends are available:

    ``assemble_system_matrix``          — double-vmap, best on GPU
    ``assemble_system_matrix_chunked``  — chunked numpy/JAX, CPU- or
                                           GPU-friendly; peak memory
                                           capped by ``chunk_rows``

Use ``assemble_system_matrix_auto(backend='auto')`` for automatic
selection: GPU when JAX has a GPU device, chunked-CPU otherwise.

Upgrade paths (not implemented, noted for future):
    - Smith (2018) nearest-neighbour discretization (10x cost reduction):
      DOI: 10.1016/j.jcp.2017.12.008
    - Gallagher & Smith (2021) Richardson extrapolation in ε:
      DOI: 10.1098/rsos.210108
      Local PDF: tmp/stokelet_403_papers/rsos.210108.pdf
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from .kernel import stokeslet_tensor
from .stresslet import stresslet_tensor_contracted

logger = logging.getLogger(__name__)

# Stable one-time sanity check between backends — set by first GPU call
_GPU_VALIDATED_AGAINST_CPU: bool = False


def _has_gpu() -> bool:
    """Return True when JAX reports at least one GPU/TPU device."""
    try:
        devices = jax.devices()
    except Exception:
        return False
    return any(d.platform in ("gpu", "cuda", "tpu") for d in devices)


def assemble_system_matrix(
    surface_points: jnp.ndarray,
    surface_weights: jnp.ndarray,
    epsilon: float,
    mu: float,
) -> jnp.ndarray:
    """Assemble the 3N × 3N BEM system matrix (double-vmap, unchunked).

    A[3m:3m+3, 3n:3n+3] = (w_n / 8πμ) · S^ε(x_m, y_n)

    For very large N (N > 5000) on memory-constrained devices prefer
    :func:`assemble_system_matrix_chunked`.

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


@jax.jit
def _stokeslet_chunk(r, eps_sq):
    """(nc, N, 3) r vector → (nc, N, 3, 3) Stokeslet tensor blocks."""
    r_sq = jnp.sum(r ** 2, axis=-1)
    denom = (r_sq + eps_sq) ** 1.5
    S = (jnp.eye(3)[None, None, :, :] * (r_sq + 2 * eps_sq)[:, :, None, None]
         + r[:, :, :, None] * r[:, :, None, :]) / denom[:, :, None, None]
    return S


def assemble_system_matrix_chunked(
    surface_points,
    surface_weights,
    epsilon: float,
    mu: float,
    chunk_rows: int = 500,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Chunked-row BEM assembly, memory-capped and GPU-friendly.

    Processes ``chunk_rows`` target rows at a time. Peak device memory
    ≈ ``chunk_rows × N × 9 × sizeof(dtype)``. For N=3120, chunk=500,
    float32 that's ≈ 112 MB — comfortable on a 4 GB GPU. Per-call JIT
    compile cost is amortised across many calls (same shape).

    Returns (3N, 3N) JAX array.
    """
    pts_j = jnp.asarray(surface_points, dtype=dtype)
    wts_j = jnp.asarray(surface_weights, dtype=dtype)
    N = int(pts_j.shape[0])
    prefactor = 1.0 / (8.0 * np.pi * mu)
    eps_sq = float(epsilon ** 2)

    blocks = []
    for i0 in range(0, N, chunk_rows):
        i1 = min(i0 + chunk_rows, N)
        nc = i1 - i0
        r = pts_j[i0:i1, None, :] - pts_j[None, :, :]      # (nc, N, 3)
        S = _stokeslet_chunk(r, eps_sq)                     # (nc, N, 3, 3)
        S = S * (prefactor * wts_j[None, :, None, None])
        blocks.append(S.transpose(0, 2, 1, 3).reshape(3 * nc, 3 * N))
    return jnp.concatenate(blocks, axis=0)


def assemble_system_matrix_auto(
    surface_points,
    surface_weights,
    epsilon: float,
    mu: float,
    backend: str = "auto",
    chunk_rows: int = 500,
    dtype=jnp.float32,
    validate_gpu_once: bool = True,
) -> jnp.ndarray:
    """Dispatch to CPU or GPU assembly based on ``backend``.

    ``backend``:
        ``'auto'`` — GPU when JAX sees one; otherwise CPU chunked.
        ``'gpu'``  — force GPU (chunked, JIT-friendly).
        ``'cpu'``  — force CPU chunked path.

    On first GPU call this function cross-checks a small sample
    against the double-vmap path and logs a warning if relative
    error > 1e-10 (set ``validate_gpu_once=False`` to skip).
    """
    global _GPU_VALIDATED_AGAINST_CPU

    if backend == "auto":
        backend = "gpu" if _has_gpu() else "cpu"

    if backend == "cpu":
        return assemble_system_matrix_chunked(
            surface_points, surface_weights, epsilon, mu,
            chunk_rows=chunk_rows, dtype=dtype,
        )

    # GPU path — same chunked implementation, runs on the GPU device.
    A_gpu = assemble_system_matrix_chunked(
        surface_points, surface_weights, epsilon, mu,
        chunk_rows=chunk_rows, dtype=dtype,
    )

    if validate_gpu_once and not _GPU_VALIDATED_AGAINST_CPU:
        # Cheap subset validation: first 100 rows vs unchunked double-vmap reference
        N = int(len(surface_points))
        n_ref = min(100, N)
        pts_ref = jnp.asarray(surface_points[:n_ref], dtype=jnp.float64)
        wts_ref = jnp.asarray(surface_weights[:n_ref], dtype=jnp.float64)
        A_ref = assemble_system_matrix(pts_ref, wts_ref, float(epsilon), float(mu))
        A_sub = np.asarray(A_gpu[:3 * n_ref, :3 * n_ref].astype(jnp.float64))
        A_ref_np = np.asarray(A_ref)
        denom = max(float(np.max(np.abs(A_ref_np))), 1e-30)
        rel = float(np.max(np.abs(A_sub - A_ref_np)) / denom)
        if rel > 1e-6:  # float32 GPU vs float64 CPU: ~1e-7 expected
            logger.warning(
                "GPU BEM vs CPU reference relative error %.2e exceeds 1e-6 "
                "tolerance. Using float64 on GPU may be required.", rel,
            )
        else:
            logger.info("GPU BEM validated vs CPU reference (rel err %.2e)", rel)
        _GPU_VALIDATED_AGAINST_CPU = True

    return A_gpu


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


def compute_dlp_rhs_correction(
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
    surface_weights: jnp.ndarray,
    velocity: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Compute the double-layer potential correction to the BEM RHS.

    From Smith et al. (2021) Eq. (17), the full boundary integral
    equation is:

        -SLP(f) - DLP(u) = (1/2)u + O(κε)

    Rearranging for the SLP system Af = rhs:

        rhs = (1/2)u + DLP(u)

    where DLP_j(y) = (1/8π) Σ_{n≠m} T^ε_ijk(x[n], x[m]) n_k(x[n])
                     A[n] u_i(x[n])

    The existing code uses rhs = u (no 1/2, no DLP). This function
    computes the corrected RHS.

    Parameters
    ----------
    surface_points : (N, 3)
    surface_normals : (N, 3) outward unit normals
    surface_weights : (N,) quadrature weights
    velocity : (N, 3) prescribed velocity at each point
    epsilon : float

    Returns
    -------
    rhs_corrected : (3N,) corrected right-hand side
    """
    N = len(surface_points)
    prefactor = 1.0 / (8.0 * jnp.pi)

    def _dlp_at_point(m):
        x_m = surface_points[m]
        u_m = velocity[m]

        def _contribution(n):
            x_n = surface_points[n]
            n_n = surface_normals[n]
            w_n = surface_weights[n]
            u_n = velocity[n]

            # T^ε_ijk(x[n], x[m]) contracted with n_k(x[n])
            K = stresslet_tensor_contracted(x_n, x_m, n_n, epsilon)
            dlp_contrib = -prefactor * w_n * (K @ u_n)

            # Zero out self-interaction (n == m)
            return jnp.where(n == m, jnp.zeros(3), dlp_contrib)

        contributions = jax.vmap(_contribution)(jnp.arange(N))
        dlp_m = jnp.sum(contributions, axis=0)

        return 0.5 * u_m + dlp_m

    rhs_corrected = jax.vmap(_dlp_at_point)(jnp.arange(N))
    return rhs_corrected.ravel()


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
