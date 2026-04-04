"""Triton GPU kernels for D3Q19 LBM step.

Three-kernel approach (prevents compiler reordering of rho/u accumulation):
  Kernel 1 (_macroscopic_kernel): Kahan-compensated rho, u from f and force
  Kernel 2 (_collision_forcing_kernel): BGK collision + Guo forcing from rho, u
  Kernel 3 (_stream_bb_kernel): Pull-stream + bounce-back

Splitting macroscopic from collision forces rho/u through global memory,
so the Triton compiler cannot fuse/reorder the accumulation with the
equilibrium computation. Proven zero error vs JAX in isolation.

Open BCs applied in JAX after kernel 3 (trivial cost, avoids
Triton ordering issues at boundary faces).

Compiles in <1s on Ampere/Hopper GPUs. Bypasses XLA autotuning
that causes 60+ min JIT on H100.

Requirements: triton >= 3.1.0, jax-triton >= 0.2.0
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

try:
    import triton
    import triton.language as tl
    import jax_triton as jt
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from mime.nodes.environment.lbm.d3q19 import E, W, OPP, Q


if TRITON_AVAILABLE:

    E_NP = np.array(E, dtype=np.int32)
    W_NP = np.array(W, dtype=np.float32)
    OPP_NP = np.array(OPP, dtype=np.int32)

    # Lazily cached JAX arrays for D3Q19 lattice constants.
    # Created once on first use — avoids per-call jnp.array() overhead
    # (5 host-to-device copies + allocations per LBM step).
    _D3Q19_JAX_CACHE = {}

    def _get_d3q19_jax():
        if not _D3Q19_JAX_CACHE:
            _D3Q19_JAX_CACHE['ex'] = jnp.array(E_NP[:, 0])
            _D3Q19_JAX_CACHE['ey'] = jnp.array(E_NP[:, 1])
            _D3Q19_JAX_CACHE['ez'] = jnp.array(E_NP[:, 2])
            _D3Q19_JAX_CACHE['w'] = jnp.array(W_NP)
            _D3Q19_JAX_CACHE['opp'] = jnp.array(OPP_NP)
        return _D3Q19_JAX_CACHE

    @triton.jit
    def _macroscopic_kernel(
        f_ptr, force_ptr, ex_ptr, ey_ptr, ez_ptr,
        rho_ptr, ux_ptr, uy_ptr, uz_ptr,
        N_FLAT: tl.constexpr, QQ: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Kahan-compensated rho and velocity from f and force.

        Separate kernel so Triton cannot reorder the accumulation
        with downstream equilibrium computation.
        """
        pid = tl.program_id(0)
        nids = pid * BLOCK + tl.arange(0, BLOCK)
        mask = nids < N_FLAT

        rho = tl.zeros((BLOCK,), tl.float32)
        rho_c = tl.zeros((BLOCK,), tl.float32)
        mx = tl.zeros((BLOCK,), tl.float32)
        mx_c = tl.zeros((BLOCK,), tl.float32)
        my = tl.zeros((BLOCK,), tl.float32)
        my_c = tl.zeros((BLOCK,), tl.float32)
        mz = tl.zeros((BLOCK,), tl.float32)
        mz_c = tl.zeros((BLOCK,), tl.float32)
        for q in range(19):
            fq = tl.load(f_ptr + nids*QQ + q, mask=mask, other=0.0)
            ex = tl.load(ex_ptr+q).to(tl.float32)
            ey = tl.load(ey_ptr+q).to(tl.float32)
            ez = tl.load(ez_ptr+q).to(tl.float32)
            y = fq - rho_c; t = rho + y; rho_c = (t - rho) - y; rho = t
            val = fq * ex; y = val - mx_c; t = mx + y; mx_c = (t - mx) - y; mx = t
            val = fq * ey; y = val - my_c; t = my + y; my_c = (t - my) - y; my = t
            val = fq * ez; y = val - mz_c; t = mz + y; mz_c = (t - mz) - y; mz = t

        fx = tl.load(force_ptr + nids*3+0, mask=mask, other=0.0)
        fy = tl.load(force_ptr + nids*3+1, mask=mask, other=0.0)
        fz = tl.load(force_ptr + nids*3+2, mask=mask, other=0.0)
        mx += 0.5*fx; my += 0.5*fy; mz += 0.5*fz
        rs = tl.maximum(rho, 1e-10)
        ux = mx/rs; uy = my/rs; uz = mz/rs

        tl.store(rho_ptr+nids, rho, mask=mask)
        tl.store(ux_ptr+nids, ux, mask=mask)
        tl.store(uy_ptr+nids, uy, mask=mask)
        tl.store(uz_ptr+nids, uz, mask=mask)

    @triton.jit
    def _collision_forcing_kernel(
        f_ptr, rho_ptr, ux_ptr, uy_ptr, uz_ptr,
        force_ptr, ex_ptr, ey_ptr, ez_ptr, w_ptr,
        f_out_ptr,
        N_FLAT: tl.constexpr, QQ: tl.constexpr, BLOCK: tl.constexpr,
        TAU: tl.constexpr,
    ):
        """BGK collision + Guo forcing from pre-computed rho, u.

        Reads rho/u from global memory (written by _macroscopic_kernel),
        so the compiler cannot fuse/reorder the accumulation.
        """
        pid = tl.program_id(0)
        nids = pid * BLOCK + tl.arange(0, BLOCK)
        mask = nids < N_FLAT
        cs2 = 1.0/3.0; cs4 = cs2*cs2
        inv_tau = 1.0/TAU; guo_pref = 1.0 - 0.5*inv_tau

        rho = tl.load(rho_ptr+nids, mask=mask, other=0.0)
        ux = tl.load(ux_ptr+nids, mask=mask, other=0.0)
        uy = tl.load(uy_ptr+nids, mask=mask, other=0.0)
        uz = tl.load(uz_ptr+nids, mask=mask, other=0.0)
        usq = ux*ux + uy*uy + uz*uz

        fx = tl.load(force_ptr + nids*3+0, mask=mask, other=0.0)
        fy = tl.load(force_ptr + nids*3+1, mask=mask, other=0.0)
        fz = tl.load(force_ptr + nids*3+2, mask=mask, other=0.0)

        for q in range(19):
            fq = tl.load(f_ptr + nids*QQ+q, mask=mask, other=0.0)
            ex = tl.load(ex_ptr+q).to(tl.float32)
            ey = tl.load(ey_ptr+q).to(tl.float32)
            ez = tl.load(ez_ptr+q).to(tl.float32)
            wq = tl.load(w_ptr+q)
            edu = ux*ex+uy*ey+uz*ez
            feq = wq*rho*(1.0+edu/cs2+edu*edu/(2.0*cs4)-usq/(2.0*cs2))
            fp = fq-(fq-feq)*inv_tau
            Fde = fx*ex+fy*ey+fz*ez
            Fdu = fx*ux+fy*uy+fz*uz
            sq = guo_pref*wq*(Fde/cs2-Fdu/cs2+edu*Fde/cs4)
            tl.store(f_out_ptr + nids*QQ+q, fp+sq, mask=mask)

    @triton.jit
    def _stream_bb_kernel(
        f_post_ptr, missing_ptr, opp_ptr,
        ex_ptr, ey_ptr, ez_ptr,
        f_out_ptr,
        NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
        N_FLAT: tl.constexpr, QQ: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Pull-stream + bounce-back. Open BCs done outside."""
        pid = tl.program_id(0)
        nids = pid * BLOCK + tl.arange(0, BLOCK)
        mask = nids < N_FLAT
        ny_nz = NY * NZ

        ix = nids // ny_nz
        iy = (nids % ny_nz) // NZ
        iz = nids % NZ

        for q in range(19):
            exi = tl.load(ex_ptr + q)
            eyi = tl.load(ey_ptr + q)
            ezi = tl.load(ez_ptr + q)
            opp_q = tl.load(opp_ptr + q)

            # Pull-stream
            sx = (ix - exi + NX) % NX
            sy = (iy - eyi + NY) % NY
            sz = (iz - ezi + NZ) % NZ
            src = sx * ny_nz + sy * NZ + sz
            f_streamed = tl.load(f_post_ptr + src * QQ + q, mask=mask, other=0.0)

            # Bounce-back
            is_missing = tl.load(missing_ptr + opp_q * N_FLAT + nids, mask=mask, other=0)
            is_bb = is_missing > 0
            f_bb = tl.load(f_post_ptr + nids * QQ + opp_q, mask=mask, other=0.0)
            f_val = tl.where(is_bb, f_bb, f_streamed)

            tl.store(f_out_ptr + nids * QQ + q, f_val, mask=mask)


def lbm_full_step_triton(
    f: jnp.ndarray,
    force: jnp.ndarray,
    tau: float,
    pipe_wall: jnp.ndarray,
    pipe_missing: jnp.ndarray,
    open_bc_axis: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Full D3Q19 LBM step via three Triton kernels + JAX open BCs.

    Compiles in <1s on Ampere/Hopper GPUs.
    """
    if not TRITON_AVAILABLE:
        raise ImportError("triton and jax-triton required for Triton backend")

    nx, ny, nz, _ = f.shape
    N = nx * ny * nz
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)

    f_flat = f.reshape(N, Q)
    force_flat = force.reshape(N, 3)

    # Accept pre-flattened int32 missing mask (avoids per-step astype)
    if pipe_missing.ndim == 1:
        missing_flat = pipe_missing
    else:
        missing_flat = pipe_missing.reshape(Q * N).astype(jnp.int32)

    c = _get_d3q19_jax()
    ex, ey, ez, w, opp = c['ex'], c['ey'], c['ez'], c['w'], c['opp']

    # Kernel 1: macroscopic (Kahan-compensated rho, u)
    rho_flat, ux, uy, uz = jt.triton_call(
        f_flat, force_flat, ex, ey, ez,
        kernel=_macroscopic_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((N,), jnp.float32),
            jax.ShapeDtypeStruct((N,), jnp.float32),
            jax.ShapeDtypeStruct((N,), jnp.float32),
            jax.ShapeDtypeStruct((N,), jnp.float32),
        ],
        grid=grid, N_FLAT=N, QQ=Q, BLOCK=BLOCK,
    )

    # Kernel 2: collision + Guo forcing (reads rho, u from memory)
    f_post = jt.triton_call(
        f_flat, rho_flat, ux, uy, uz, force_flat, ex, ey, ez, w,
        kernel=_collision_forcing_kernel,
        out_shape=jax.ShapeDtypeStruct((N, Q), jnp.float32),
        grid=grid, N_FLAT=N, QQ=Q, BLOCK=BLOCK, TAU=tau,
    )

    # Kernel 3: pull-stream + bounce-back
    f_out = jt.triton_call(
        f_post, missing_flat, opp, ex, ey, ez,
        kernel=_stream_bb_kernel,
        out_shape=jax.ShapeDtypeStruct((N, Q), jnp.float32),
        grid=grid, NX=nx, NY=ny, NZ=nz,
        N_FLAT=N, QQ=Q, BLOCK=BLOCK,
    )

    u = jnp.stack([ux, uy, uz], axis=-1).reshape(nx, ny, nz, 3)
    f_result = f_out.reshape(nx, ny, nz, Q)

    # Open BCs in JAX (trivial cost, avoids Triton ordering issues)
    if open_bc_axis is not None:
        w_bc = c['w']
        if open_bc_axis == 0:
            f_result = f_result.at[-1, :, :, :].set(f_result[-2, :, :, :])
            f_result = f_result.at[0, :, :, :].set(
                jnp.broadcast_to(w_bc, (ny, nz, Q)))
        elif open_bc_axis == 1:
            f_result = f_result.at[:, -1, :, :].set(f_result[:, -2, :, :])
            f_result = f_result.at[:, 0, :, :].set(
                jnp.broadcast_to(w_bc, (nx, nz, Q)))
        elif open_bc_axis == 2:
            f_result = f_result.at[:, :, -1, :].set(f_result[:, :, -2, :])
            f_result = f_result.at[:, :, 0, :].set(
                jnp.broadcast_to(w_bc, (nx, ny, Q)))

    return f_result, u
