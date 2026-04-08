"""Fast Stokeslet matrix-vector product for BEM and MFS.

Computes u(x_m) = Σ_n (1/8πμ) S^ε(x_m, y_n) · σ(y_n) without
forming the full (3M × 3N) dense matrix.

Three backends:
  1. numpy — Dense reference, O(MN), CPU. For validation.
  2. jax   — JAX vmap, O(MN), GPU-friendly, differentiable.
  3. fmm   — FMM3D (Flatiron), O(M+N), CPU. For large problems.

Used by:
  - GMRES-based discrete wall MFS (Phase 2)
  - Fast force/torque evaluation from known source strengths
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import fmm3dpy
    FMM_AVAILABLE = True
except ImportError:
    FMM_AVAILABLE = False


# ── Numpy dense reference ─────────────────────────────────────────────

def stokeslet_matvec_numpy(
    target_pts: np.ndarray,
    source_pts: np.ndarray,
    strengths: np.ndarray,
    mu: float,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Dense O(MN) Stokeslet matvec. CPU numpy. For validation only.

    Parameters
    ----------
    target_pts : (M, 3)
    source_pts : (N, 3)
    strengths : (N, 3) force densities (no quadrature weights)
    mu : viscosity
    epsilon : regularisation (0 = singular)

    Returns
    -------
    velocity : (M, 3) at targets
    """
    r = target_pts[:, None, :] - source_pts[None, :, :]  # (M, N, 3)
    r_sq = np.sum(r**2, axis=-1)                           # (M, N)
    eps_sq = epsilon**2
    denom = (r_sq + eps_sq) ** 1.5                         # (M, N)
    inv_denom = np.where(denom > 1e-30, 1.0 / denom, 0.0)

    pf = 1.0 / (8.0 * np.pi * mu)

    # G_jk σ_k = pf * [(δ_jk(r²+2ε²) + r_j r_k) / denom] σ_k
    # = pf * [(r²+2ε²)/denom * σ + (r·σ)/denom * r]
    r_dot_s = np.sum(r * strengths[None, :, :], axis=-1)   # (M, N)

    vel = pf * np.sum(
        (r_sq + 2 * eps_sq)[..., None] * inv_denom[..., None] *
        strengths[None, :, :]
        + r_dot_s[..., None] * inv_denom[..., None] * r,
        axis=1,
    )  # (M, 3)
    return vel


# ── JAX vmap ──────────────────────────────────────────────────────────

if JAX_AVAILABLE:
    def stokeslet_matvec_jax(
        target_pts: jnp.ndarray,
        source_pts: jnp.ndarray,
        strengths: jnp.ndarray,
        mu: float,
        epsilon: float = 0.0,
    ) -> jnp.ndarray:
        """JAX O(MN) Stokeslet matvec. GPU-friendly, differentiable.

        Same API as numpy version but uses JAX arrays.
        """
        r = target_pts[:, None, :] - source_pts[None, :, :]
        r_sq = jnp.sum(r**2, axis=-1)
        eps_sq = epsilon**2
        denom = (r_sq + eps_sq) ** 1.5
        inv_denom = jnp.where(denom > 1e-30, 1.0 / denom, 0.0)

        pf = 1.0 / (8.0 * jnp.pi * mu)
        r_dot_s = jnp.sum(r * strengths[None, :, :], axis=-1)

        vel = pf * jnp.sum(
            (r_sq + 2 * eps_sq)[..., None] * inv_denom[..., None] *
            strengths[None, :, :]
            + r_dot_s[..., None] * inv_denom[..., None] * r,
            axis=1,
        )
        return vel


# ── FMM3D (Stokes FMM via fmm3dpy) ───────────────────────────────────

if FMM_AVAILABLE:
    def stokeslet_matvec_fmm(
        target_pts: np.ndarray,
        source_pts: np.ndarray,
        strengths: np.ndarray,
        mu: float,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """FMM O(M+N) Stokeslet matvec via fmm3dpy.

        Only supports the singular Stokeslet (epsilon=0).
        Uses the Stokes single-layer FMM.
        """
        if epsilon != 0.0:
            raise ValueError("FMM backend only supports epsilon=0 (singular)")

        # fmm3dpy wants Fortran-order (3, N) arrays
        sources = np.asfortranarray(source_pts.T, dtype=np.float64)
        targets = np.asfortranarray(target_pts.T, dtype=np.float64)
        stoklet = np.asfortranarray(strengths.T, dtype=np.float64)

        # Stokes FMM: stoklet at targets
        # ifppreg=0: no eval at sources, ifppregtarg=1: velocity at targets
        out = fmm3dpy.stfmm3d(
            eps=1e-10,
            sources=sources,
            targets=targets,
            stoklet=stoklet,
            ifppreg=0,
            ifppregtarg=1,
        )

        # out.pottarg is (3, M) or (3, 1, M) — velocity at targets
        # fmm3dpy uses 1/(8π) convention, need 1/(8πμ)
        pot = np.asarray(out.pottarg).squeeze()  # ensure (3, M)
        return pot.T / mu  # (M, 3)


# ── Dispatcher ────────────────────────────────────────────────────────

def stokeslet_matvec(
    target_pts,
    source_pts,
    strengths,
    mu: float,
    epsilon: float = 0.0,
    backend: str = 'auto',
):
    """Compute Stokeslet matvec with automatic backend selection.

    Parameters
    ----------
    target_pts : (M, 3)
    source_pts : (N, 3)
    strengths : (N, 3)
    mu : viscosity
    epsilon : regularisation (0 = singular)
    backend : 'auto', 'numpy', 'jax', 'fmm'

    Returns
    -------
    velocity : (M, 3) at targets
    """
    if backend == 'auto':
        if FMM_AVAILABLE and epsilon == 0.0 and len(source_pts) > 2000:
            backend = 'fmm'
        elif JAX_AVAILABLE:
            backend = 'jax'
        else:
            backend = 'numpy'

    if backend == 'numpy':
        return stokeslet_matvec_numpy(
            np.asarray(target_pts), np.asarray(source_pts),
            np.asarray(strengths), mu, epsilon)
    elif backend == 'jax':
        return stokeslet_matvec_jax(
            jnp.asarray(target_pts), jnp.asarray(source_pts),
            jnp.asarray(strengths), mu, epsilon)
    elif backend == 'fmm':
        return stokeslet_matvec_fmm(
            np.asarray(target_pts), np.asarray(source_pts),
            np.asarray(strengths), mu, epsilon)
    else:
        raise ValueError(f"Unknown backend: {backend}")
