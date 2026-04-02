"""Gather-based fused LBM step for fast compilation.

Replaces the 19 separate jnp.roll streaming calls with a single
vectorized gather, eliminating XLA's autotuning overhead on H100.
Compiles in <3s vs 30-40 min for the roll-based version.

Optimizations:
- Gather-based streaming (no jnp.roll, single indexed read)
- On-the-fly source coordinate computation (no precomputed index array)
- Fused Guo forcing (no 5D intermediate arrays)

Usage:
    from mime.nodes.environment.lbm.pallas_lbm import lbm_full_step_pallas
    f_new, u = lbm_full_step_pallas(f, force, tau, pipe_wall, pipe_missing, 2)
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.d3q19 import E, W, OPP, CS2, CS4, Q


@functools.lru_cache(maxsize=4)
def _build_stream_indices(nx: int, ny: int, nz: int) -> jnp.ndarray:
    """Precompute gather indices for streaming (cached per grid size).

    Returns flat indices: (nx*ny*nz, 19) int32.
    Memory: N × 19 × 4 bytes (e.g. 48³ = 6.6MB, 128³ = 160MB).
    """
    e_np = np.array(E, dtype=np.int32)
    ix = np.arange(nx).reshape(-1, 1, 1)
    iy = np.arange(ny).reshape(1, -1, 1)
    iz = np.arange(nz).reshape(1, 1, -1)

    indices = np.zeros((nx, ny, nz, Q), dtype=np.int32)
    for q in range(Q):
        ex, ey, ez = e_np[q]
        indices[:, :, :, q] = (
            ((ix - ex) % nx) * ny * nz +
            ((iy - ey) % ny) * nz +
            ((iz - ez) % nz)
        )
    return jnp.array(indices.reshape(nx * ny * nz, Q))


def _stream_onthefly(f_post, nx, ny, nz):
    """Stream via on-the-fly coordinate computation (no index array).

    Computes source node for each direction using modular arithmetic
    instead of reading from a precomputed index array. Trades ~160MB
    memory bandwidth (at 128³) for integer arithmetic per node.
    """
    e_int = jnp.array(E, dtype=jnp.int32)  # (19, 3)

    # Build 3D coordinate arrays
    ix = jnp.arange(nx, dtype=jnp.int32)
    iy = jnp.arange(ny, dtype=jnp.int32)
    iz = jnp.arange(nz, dtype=jnp.int32)

    f_flat = f_post.reshape(-1, Q)

    # For each q, compute source flat index and gather
    # Vectorize over q: compute all 19 source indices at once
    # src_x[i,j,k,q] = (i - e[q,0]) % nx
    gx = ix[:, None, None, None] - e_int[None, None, None, :, 0]  # (nx,1,1,19)
    gy = iy[None, :, None, None] - e_int[None, None, None, :, 1]  # (1,ny,1,19)
    gz = iz[None, None, :, None] - e_int[None, None, None, :, 2]  # (1,1,nz,19)

    src_flat = (gx % nx) * ny * nz + (gy % ny) * nz + (gz % nz)  # (nx,ny,nz,19)
    src_flat = src_flat.reshape(-1, Q)  # (N, 19)

    f_streamed = f_flat[src_flat, jnp.arange(Q)]
    return f_streamed.reshape(nx, ny, nz, Q)


def lbm_full_step_pallas(
    f: jnp.ndarray,
    force: jnp.ndarray,
    tau: float,
    pipe_wall: jnp.ndarray,
    pipe_missing: jnp.ndarray,
    open_bc_axis: int | None = None,
    use_precomputed_indices: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fused LBM step: collision + forcing + streaming + BB + open BCs.

    Drop-in replacement for _lbm_full_step. Uses gather-based streaming
    for fast JIT compilation on H100.

    No custom_vjp needed — LBM step is forward-only (no AD through
    the simulation). If adjoint optimization is needed in the future,
    wrap with jax.custom_vjp and define a backward pass.

    Parameters
    ----------
    f : (nx, ny, nz, 19)
    force : (nx, ny, nz, 3)
    tau : float
    pipe_wall : (nx, ny, nz) bool
    pipe_missing : (19, nx, ny, nz) bool
    open_bc_axis : int or None
    use_precomputed_indices : bool
        If True, use cached index array (faster per step, more memory).
        If False, compute source coords on-the-fly (less memory).

    Returns
    -------
    f_new : (nx, ny, nz, 19)
    u : (nx, ny, nz, 3)
    """
    nx, ny, nz, _ = f.shape
    N = nx * ny * nz

    e = jnp.array(E, dtype=jnp.float32)
    w = jnp.array(W, dtype=jnp.float32)
    opp = jnp.array(OPP, dtype=jnp.int32)

    # ── 1. Macroscopic: ρ, u ─────────────────────────────────────
    rho = jnp.sum(f, axis=-1)
    momentum = f @ e
    if force is not None:
        momentum = momentum + 0.5 * force
    u = momentum / jnp.maximum(rho[..., None], 1e-10)

    # ── 2. Collision: BGK + fused Guo forcing ────────────────────
    e_dot_u = u @ e.T  # (nx,ny,nz,19)
    u_sq = jnp.sum(u ** 2, axis=-1, keepdims=True)
    f_eq = w * rho[..., None] * (
        1.0 + e_dot_u / CS2 + e_dot_u ** 2 / (2.0 * CS4) - u_sq / (2.0 * CS2)
    )
    f_post = f - (f - f_eq) / tau

    if force is not None:
        # Fused Guo forcing: avoid 5D intermediates (nx,ny,nz,19,3)
        # S_q = (1-1/(2τ)) w_q Σ_α F_α [(e_qα - u_α)/c_s² + e_qα(e_q·u)/c_s⁴]
        #     = (1-1/(2τ)) w_q [F·e_q/c_s² - F·u/c_s² + (e_q·u)(F·e_q)/c_s⁴]
        pref = 1.0 - 0.5 / tau
        F_dot_e = force @ e.T  # (nx,ny,nz,19) — force projected onto each direction
        F_dot_u = jnp.sum(force * u, axis=-1, keepdims=True)  # (nx,ny,nz,1)
        S = pref * w * (F_dot_e / CS2 - F_dot_u / CS2 + e_dot_u * F_dot_e / CS4)
        f_post = f_post + S

    # ── 3. Streaming ─────────────────────────────────────────────
    if use_precomputed_indices:
        stream_idx = _build_stream_indices(nx, ny, nz)
        f_flat = f_post.reshape(N, Q)
        f_streamed = f_flat[stream_idx, jnp.arange(Q)].reshape(nx, ny, nz, Q)
    else:
        f_streamed = _stream_onthefly(f_post, nx, ny, nz)

    # ── 4. Bounce-back ───────────────────────────────────────────
    f_pre_opp = f_post[..., opp]
    mm = jnp.moveaxis(pipe_missing, 0, -1)
    mm_in = mm[..., opp]
    f_bb = jnp.where(mm_in, f_pre_opp, f_streamed)

    # ── 5. Open BCs ──────────────────────────────────────────────
    if open_bc_axis is not None:
        f_bb = _apply_open_bc(f_bb, open_bc_axis)

    return f_bb, u


def _apply_open_bc(f, axis):
    """Open BCs: outlet extrapolation, inlet equilibrium at rest."""
    rho_0 = 1.0
    w = jnp.array(W, dtype=jnp.float32)
    if axis == 0:
        f = f.at[-1, :, :, :].set(f[-2, :, :, :])
        f = f.at[0, :, :, :].set(jnp.broadcast_to(w * rho_0, (f.shape[1], f.shape[2], Q)))
    elif axis == 1:
        f = f.at[:, -1, :, :].set(f[:, -2, :, :])
        f = f.at[:, 0, :, :].set(jnp.broadcast_to(w * rho_0, (f.shape[0], f.shape[2], Q)))
    elif axis == 2:
        f = f.at[:, :, -1, :].set(f[:, :, -2, :])
        f = f.at[:, :, 0, :].set(jnp.broadcast_to(w * rho_0, (f.shape[0], f.shape[1], Q)))
    return f
