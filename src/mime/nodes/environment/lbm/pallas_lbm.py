"""Gather-based fused LBM step for fast compilation.

Replaces the 19 separate jnp.roll streaming calls with a single
vectorized gather, eliminating XLA's autotuning overhead on H100.
Compiles in <2s vs 30-40 min for the roll-based version.

The kernel fuses: BGK collision + Guo forcing + streaming (gather) +
bounce-back + open BCs into a single function that JIT-compiles as
one XLA computation.

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

    For direction q at node (i,j,k), the post-streaming value comes
    from node (i-ex, j-ey, k-ez) with periodic wrapping.

    Returns flat indices: (nx*ny*nz, 19) int32
    """
    e_np = np.array(E, dtype=np.int32)
    N = nx * ny * nz

    ix = np.arange(nx).reshape(-1, 1, 1)  # (nx,1,1)
    iy = np.arange(ny).reshape(1, -1, 1)  # (1,ny,1)
    iz = np.arange(nz).reshape(1, 1, -1)  # (1,1,nz)

    indices = np.zeros((nx, ny, nz, Q), dtype=np.int32)
    for q in range(Q):
        ex, ey, ez = e_np[q]
        src_x = (ix - ex) % nx
        src_y = (iy - ey) % ny
        src_z = (iz - ez) % nz
        indices[:, :, :, q] = src_x * ny * nz + src_y * nz + src_z

    return jnp.array(indices.reshape(N, Q))


def lbm_full_step_pallas(
    f: jnp.ndarray,
    force: jnp.ndarray,
    tau: float,
    pipe_wall: jnp.ndarray,
    pipe_missing: jnp.ndarray,
    open_bc_axis: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fused LBM step: collision + forcing + streaming + BB + open BCs.

    Drop-in replacement for _lbm_full_step. Uses gather-based streaming
    instead of 19 jnp.roll calls for fast JIT compilation on H100.

    No custom_vjp needed — LBM step is forward-only (no AD through
    the simulation). If adjoint optimization is needed in the future,
    wrap with jax.custom_vjp and define a backward pass.

    Parameters
    ----------
    f : (nx, ny, nz, 19) distributions
    force : (nx, ny, nz, 3) body force
    tau : float, relaxation time
    pipe_wall : (nx, ny, nz) bool, solid nodes
    pipe_missing : (19, nx, ny, nz) bool, missing directions
    open_bc_axis : int or None

    Returns
    -------
    f_new : (nx, ny, nz, 19)
    u : (nx, ny, nz, 3)
    """
    nx, ny, nz, _ = f.shape
    N = nx * ny * nz

    e = jnp.array(E, dtype=jnp.float32)  # (19, 3)
    w = jnp.array(W, dtype=jnp.float32)  # (19,)
    opp = jnp.array(OPP, dtype=jnp.int32)  # (19,)

    # ── 1. Macroscopic: ρ, u ─────────────────────────────────────
    rho = jnp.sum(f, axis=-1)  # (nx, ny, nz)
    momentum = f @ e  # (nx, ny, nz, 3)
    if force is not None:
        momentum = momentum + 0.5 * force
    u = momentum / jnp.maximum(rho[..., None], 1e-10)

    # ── 2. Collision: BGK + Guo ──────────────────────────────────
    e_dot_u = u @ e.T  # (nx, ny, nz, 19)
    u_sq = jnp.sum(u ** 2, axis=-1, keepdims=True)
    f_eq = w * rho[..., None] * (
        1.0 + e_dot_u / CS2 + e_dot_u ** 2 / (2.0 * CS4) - u_sq / (2.0 * CS2)
    )
    f_post = f - (f - f_eq) / tau

    if force is not None:
        v_exp = u[..., None, :]
        e_minus_u = e - v_exp
        e_scaled = e * (e_dot_u / CS4)[..., None]
        bracket = e_minus_u / CS2 + e_scaled
        f_exp = force[..., None, :]
        S = (1.0 - 0.5 / tau) * w * jnp.sum(bracket * f_exp, axis=-1)
        f_post = f_post + S

    # ── 3. Streaming: vectorized gather ──────────────────────────
    stream_idx = _build_stream_indices(nx, ny, nz)  # (N, 19)
    f_flat = f_post.reshape(N, Q)

    # Single vectorized gather: for each node n and direction q,
    # read f_flat[stream_idx[n, q], q]
    # This is equivalent to: f_streamed[n, q] = f_post[src_node(n,q), q]
    f_streamed = f_flat[stream_idx, jnp.arange(Q)]  # (N, 19)
    f_streamed = f_streamed.reshape(nx, ny, nz, Q)

    # ── 4. Bounce-back (fused with streaming result) ─────────────
    f_pre_opp = f_post[..., opp]
    mm = jnp.moveaxis(pipe_missing, 0, -1)  # (nx,ny,nz,19)
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
