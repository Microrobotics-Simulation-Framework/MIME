"""FFT-diagonalised pressure Poisson solver for uniform Cartesian meshes.

The discrete cell-centred Laplacian on a uniform Cartesian grid with
all-Neumann boundary conditions is diagonalised exactly by the type-II
discrete cosine transform (DCT-II). With Dirichlet boundary conditions
the type-II discrete sine transform (DST-II) plays the same role.

Eigenvalues for DCT-II on N cells with spacing dx are

    λ_k = (2 / dx²) * (cos(k π / N) − 1)  =  −(4 / dx²) sin²(k π / (2N))

for k = 0..N−1. The 1D Laplacian operator on cell-centred values with
Neumann BCs is the symmetric three-point stencil ``[1, −2, 1] / dx²``
with mirrored ghost cells, whose eigenvectors are exactly the DCT-II
basis (Strang 1999, Trefethen & Bau §39).

For 2D / 3D Cartesian, the Laplacian is separable:

    L = L_x ⊗ I_y ⊗ I_z + I_x ⊗ L_y ⊗ I_z + I_x ⊗ I_y ⊗ L_z

so DCT along each axis simultaneously diagonalises the whole operator;
the eigenvalues sum.

This module exposes a single function :func:`solve_pressure_poisson` that
takes a flat ``[N_cells]`` right-hand side, reshapes to the Cartesian
layout, applies the DCT, divides by eigenvalues, and applies the inverse
DCT — all as a single jit-fusible operation.

References
----------
- Strang (1999) "The discrete cosine transform". SIAM Review 41(1).
- Trefethen & Bau (1997) Numerical Linear Algebra, §39.
- jax-cfd ``fast_diagonalization.py`` (referenced for algorithmic
  pattern only; this implementation is independent and uses
  jax.scipy.fft directly).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.fvm.mesh import FVMMesh


def _dct_eigenvalues_neumann(N: int, dx: float, dtype) -> jnp.ndarray:
    """Eigenvalues of the 1D cell-centred Neumann Laplacian under DCT-II."""
    k = jnp.arange(N, dtype=dtype)
    return -(4.0 / (dx * dx)) * jnp.sin(k * jnp.pi / (2.0 * N)) ** 2


def _dst_matrix(N: int, dtype) -> jnp.ndarray:
    """Orthonormal cell-centred DST-II matrix (basis ``sin((2j+1) k π / 2N)``).

    Diagonalises the cell-centred 1D Laplacian with homogeneous
    Dirichlet boundary conditions at both ends — i.e. ``u_b = 0`` at the
    physical boundary located ``dx/2`` away from the first/last cell
    centres. ``k`` runs ``1..N`` and the rows of ``M`` are normalised
    to be orthonormal.

    Note the special normalisation for the Nyquist row ``k=N``: the
    basis vector there is ``(-1)^j``, whose squared sum is ``N`` (twice
    the sum for k<N), so its row needs the smaller factor ``1/√N``
    instead of ``√(2/N)``. Missing this gave a factor-2 eigenvalue
    inflation that drove a Nyquist instability in 3D body-force flows.
    """
    n = np.arange(N)
    k = np.arange(1, N + 1)
    M = np.sin(np.pi * (2 * n[None, :] + 1) * k[:, None] / (2 * N))
    M *= np.sqrt(2.0 / N)
    M[-1, :] /= np.sqrt(2.0)
    return jnp.asarray(M, dtype=dtype)


def _dst_eigenvalues_dirichlet(N: int, dx: float, dtype) -> jnp.ndarray:
    """Eigenvalues of the cell-centred Dirichlet Laplacian under DST-II."""
    k = jnp.arange(1, N + 1, dtype=dtype)
    return -(4.0 / (dx * dx)) * jnp.sin(k * jnp.pi / (2.0 * N)) ** 2


def _dct_matrix(N: int, dtype) -> jnp.ndarray:
    """Orthonormal DCT-II matrix M of shape ``(N, N)``.

    ``X = M @ x`` computes the 1D type-II DCT with ``norm='ortho'``. The
    inverse (DCT-III) is the transpose: ``x = M.T @ X``. Implemented as a
    dense matmul so that the entire pressure solve fits inside a single
    XLA fusion (cuFFT batched plans were observed to fail inside
    ``jax.lax.fori_loop`` on this hardware/driver combination).

    For grid sizes used in this solver (≲256 per axis) the O(N²) dense
    matmul is dominated by other costs in the PISO loop and avoids a
    fragile dependency on cuFFT plan caching.
    """
    n = np.arange(N)
    k = np.arange(N)
    M = np.cos(np.pi * (2 * n[None, :] + 1) * k[:, None] / (2 * N))
    M *= np.sqrt(2.0 / N)
    M[0, :] /= np.sqrt(2.0)
    return jnp.asarray(M, dtype=dtype)


def _periodic_real_dft_matrix(N: int, dtype) -> jnp.ndarray:
    """Orthonormal real-valued basis that diagonalises the periodic
    second-difference operator (``Lap = circulant([-2, 1, 0, ..., 1])``).

    Returns ``M[k, n]`` of shape ``(N, N)`` with rows:
      * k = 0       :  constant mode, normalised
      * k = 1..N/2-1: (cos, sin) pairs, normalised
      * k = N/2     : Nyquist (only if N even)

    These are eigenvectors of the symmetric circulant Laplacian, so
    ``L = M.T @ diag(λ) @ M`` and the inverse transform is ``M.T``.
    """
    n = np.arange(N)
    rows = []
    rows.append(np.full(N, 1.0 / np.sqrt(N)))           # k=0 constant
    half = N // 2
    if N % 2 == 0:
        odd_k_max = half - 1
    else:
        odd_k_max = half
    for k in range(1, odd_k_max + 1):
        c = np.sqrt(2.0 / N) * np.cos(2 * np.pi * k * n / N)
        s = np.sqrt(2.0 / N) * np.sin(2 * np.pi * k * n / N)
        rows.append(c)
        rows.append(s)
    if N % 2 == 0:
        rows.append((1.0 / np.sqrt(N)) * np.cos(np.pi * n))  # Nyquist
    M = np.stack(rows, axis=0)
    return jnp.asarray(M, dtype=dtype)


def _periodic_eigenvalues(N: int, dx: float, dtype) -> jnp.ndarray:
    """Eigenvalues of the periodic 1D Laplacian ordered to match
    :func:`_periodic_real_dft_matrix`.

    The continuous eigenvalue is ``λ_k = −(4/dx²) sin²(π k / N)``. The
    cos and sin partners share the same eigenvalue, so we list each twice
    (except for k = 0 and the Nyquist k = N/2 if N is even).
    """
    half = N // 2
    eigs = [0.0]
    if N % 2 == 0:
        odd_k_max = half - 1
    else:
        odd_k_max = half
    for k in range(1, odd_k_max + 1):
        lam = -(4.0 / (dx * dx)) * np.sin(np.pi * k / N) ** 2
        eigs.append(lam)
        eigs.append(lam)
    if N % 2 == 0:
        eigs.append(-(4.0 / (dx * dx)) * np.sin(np.pi * half / N) ** 2)
    return jnp.asarray(np.array(eigs), dtype=dtype)


def _apply_dct_along_axis(x: jnp.ndarray, M: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Apply DCT (or its transpose) along one axis via ``jnp.tensordot``.

    ``tensordot(M, x, axes=([1], [axis]))`` produces an array whose first
    axis is the new (transformed) axis and remaining axes are ``x``'s
    other axes in their original order. ``moveaxis`` puts the transformed
    axis back where it belongs — using ``swapaxes`` here is wrong for
    ``ndim ≥ 4`` and was the cause of a subtle 3D pressure-coupling
    bug that manifested only when the mesh was anisotropic.

    ``precision="highest"`` — the default GPU matmul precision (TF32,
    ~10-bit mantissa) gives this spectral Poisson/Helmholtz solver a
    ~0.5% error: the eigenvalue division 1/lambda (dynamic range ~N^2)
    mildly amplifies TF32 noise across the chained forward+inverse
    transforms. A direct linear solver should be near-exact, and the
    dense transform is not the PISO bottleneck, so full precision here
    is essentially free.
    """
    return jnp.moveaxis(
        jnp.tensordot(M, x, axes=([1], [axis]), precision="highest"),
        0, axis,
    )


def make_pressure_solver(
    mesh: FVMMesh,
    *,
    bc: str | tuple[str, ...] = "neumann",
    pin_zero_mode: bool = True,
):
    """Construct a JIT-friendly pressure Poisson solver closure.

    Parameters
    ----------
    mesh : FVMMesh
        Must be Cartesian-structured.
    bc : str or tuple
        BC per axis. Pass a single string (``"neumann"`` or ``"periodic"``)
        to use the same on every axis, or a tuple of length ``mesh.dim``
        for axis-specific. Currently supported:
          * ``"neumann"`` — zero-gradient cell-centred pressure (used
            with closed walls / prescribed-flux inlet/outlet).
          * ``"periodic"`` — periodic in that axis. Requires the mesh
            to have been built with ``periodic_x``/``periodic_y``.
    pin_zero_mode : bool
        If True, pin the constant mode to zero (gauge fix for pure
        Neumann/periodic problems).

    Returns
    -------
    solver : Callable[[jnp.ndarray], jnp.ndarray]
        Function taking a flat ``rhs[N_cells]`` (the integrated source,
        ∫ ∇·u* dV / dt) and returning a flat ``p[N_cells]``.

    Notes
    -----
    The convention is that the right-hand side is the *cell-integrated*
    source ``b_P = ∫_P ∇·u* dV``. The discrete equation solved is

        Σ_f (p_N − p_P) |Sf| / |d|  =  b_P                       (*)

    whose eigenvalue under DCT-II is ``λ_k * V_P`` (since both sides have
    a hidden factor of ``V_P``). Concretely we divide ``rhs / V_P`` first
    to get the cell-averaged divergence, transform, divide by ``λ``, and
    inverse-transform.
    """
    if mesh.cartesian_shape is None:
        raise ValueError("FFT pressure solver requires a Cartesian mesh")

    shape = mesh.cartesian_shape
    spacing = mesh.cartesian_spacing
    dim = len(shape)
    dtype = mesh.V.dtype

    if isinstance(bc, str):
        bcs = (bc,) * dim
    else:
        bcs = tuple(bc)
    if len(bcs) != dim:
        raise ValueError(f"bc must have length {dim}; got {bcs}")
    for b in bcs:
        if b not in ("neumann", "periodic"):
            raise NotImplementedError(f"bc={b!r} not yet supported")

    eig_axes = []
    Ms = []
    for a in range(dim):
        if bcs[a] == "neumann":
            eig_axes.append(_dct_eigenvalues_neumann(shape[a], spacing[a], dtype))
            Ms.append(_dct_matrix(shape[a], dtype))
        elif bcs[a] == "periodic":
            eig_axes.append(_periodic_eigenvalues(shape[a], spacing[a], dtype))
            Ms.append(_periodic_real_dft_matrix(shape[a], dtype))
        else:
            raise NotImplementedError(f"bc={bcs[a]!r} not supported by pressure solver")

    # Sum eigenvalues with broadcasting
    lam = jnp.zeros(shape, dtype=dtype)
    for a in range(dim):
        bshape = [1] * dim
        bshape[a] = shape[a]
        lam = lam + eig_axes[a].reshape(bshape)
    # Avoid division by zero at the constant mode.
    lam_safe = jnp.where(jnp.abs(lam) < 1e-30, 1.0, lam)
    inv_lam = jnp.where(jnp.abs(lam) < 1e-30, 0.0, 1.0 / lam_safe)

    cell_volume = float(np.prod(spacing))

    def solver(rhs_flat: jnp.ndarray) -> jnp.ndarray:
        # rhs_flat is integrated divergence per cell.
        b = rhs_flat.reshape(shape) / cell_volume   # cell-averaged
        # Forward transform along all axes
        bhat = b
        for a in range(dim):
            bhat = _apply_dct_along_axis(bhat, Ms[a], a)
        phat = bhat * inv_lam
        if pin_zero_mode:
            zero_idx = tuple([0] * dim)
            phat = phat.at[zero_idx].set(0.0)
        # Inverse transform (transpose of orthonormal forward)
        p = phat
        for a in range(dim):
            p = _apply_dct_along_axis(p, Ms[a].T, a)
        return p.reshape(-1)

    return solver


def make_helmholtz_solver(
    mesh: FVMMesh,
    *,
    bc: str | tuple[str, ...] = "dirichlet",
    pin_zero_mode: bool = False,
):
    """Construct an FFT-diagonalised solver for ``(I − α ∇²) x = b``.

    ``α`` is supplied at solve time so the same solver instance can be
    reused for multiple α values (e.g. ``α = ν dt``).

    Boundary modes per axis: ``"dirichlet"`` (cell-centred zero at the
    physical face), ``"neumann"`` (zero gradient), or ``"periodic"``.
    Default ``"dirichlet"`` is correct for no-slip walls.

    Returns ``solver(b_flat, alpha) -> x_flat``.
    """
    if mesh.cartesian_shape is None:
        raise ValueError("Helmholtz solver requires a Cartesian mesh")

    shape = mesh.cartesian_shape
    spacing = mesh.cartesian_spacing
    dim = len(shape)
    dtype = mesh.V.dtype

    if isinstance(bc, str):
        bcs = (bc,) * dim
    else:
        bcs = tuple(bc)

    eig_axes = []
    Ms = []
    for a in range(dim):
        if bcs[a] == "dirichlet":
            eig_axes.append(_dst_eigenvalues_dirichlet(shape[a], spacing[a], dtype))
            Ms.append(_dst_matrix(shape[a], dtype))
        elif bcs[a] == "neumann":
            eig_axes.append(_dct_eigenvalues_neumann(shape[a], spacing[a], dtype))
            Ms.append(_dct_matrix(shape[a], dtype))
        elif bcs[a] == "periodic":
            eig_axes.append(_periodic_eigenvalues(shape[a], spacing[a], dtype))
            Ms.append(_periodic_real_dft_matrix(shape[a], dtype))
        else:
            raise NotImplementedError(f"bc={bcs[a]!r} not supported")

    lam = jnp.zeros(shape, dtype=dtype)
    for a in range(dim):
        bshape = [1] * dim
        bshape[a] = shape[a]
        lam = lam + eig_axes[a].reshape(bshape)

    has_const_mode = all(bcs[a] in ("neumann", "periodic") for a in range(dim))

    def solver(b_flat: jnp.ndarray, alpha: jnp.ndarray | float):
        b = b_flat.reshape(shape + b_flat.shape[1:])
        bhat = b
        for a in range(dim):
            bhat = _apply_dct_along_axis(bhat, Ms[a], a)
        denom = 1.0 - alpha * lam
        # For pure Neumann/periodic with α=0 the constant mode has denom=1
        # (eig=0); for α≠0 also =1 (no scaling). Inversion is well-defined.
        bhat_shape = bhat.shape
        denom_b = denom.reshape(shape + (1,) * (bhat.ndim - dim))
        xhat = bhat / denom_b
        if pin_zero_mode and has_const_mode:
            zero_idx = tuple([0] * dim)
            xhat = xhat.at[zero_idx].set(0.0)
        x = xhat
        for a in range(dim):
            x = _apply_dct_along_axis(x, Ms[a].T, a)
        return x.reshape((-1,) + b_flat.shape[1:])

    return solver


# ---------------------------------------------------------------------------
# FFT-based DCT/DST helpers (preferred — O(N log N) per axis, dispatches
# to cuFFT inside JIT). Replace the dense O(N²) matmul path above.
#
# Identity used for DST-II (no native dstn in jax.scipy.fft):
#   DST-II[k]  =  flip(DCT-II((-1)^j x))[k]
# Verified to float32 noise (max err 5e-7) at N=8 against scipy.fft.dst.
# ---------------------------------------------------------------------------

import jax.scipy.fft as _jsf


def _alternating_sign(N: int, dtype) -> jnp.ndarray:
    return jnp.asarray((-1.0) ** np.arange(N), dtype=dtype)


def _apply_dct2_axis(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """DCT-II along ``axis`` (orthonormal)."""
    return _jsf.dct(x, type=2, norm="ortho", axis=axis)


def _apply_idct2_axis(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Inverse DCT-II = DCT-III along ``axis`` (orthonormal)."""
    return _jsf.idct(x, type=2, norm="ortho", axis=axis)


def _apply_dst2_axis(x: jnp.ndarray, sign: jnp.ndarray, axis: int) -> jnp.ndarray:
    """DST-II via the DCT-II identity ``flip(DCT-II((-1)^j x))``."""
    sign_shape = [1] * x.ndim
    sign_shape[axis] = -1
    s = sign.reshape(sign_shape)
    return jnp.flip(_apply_dct2_axis(x * s, axis), axis=axis)


def _apply_idst2_axis(x: jnp.ndarray, sign: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Inverse DST-II — same identity reversed (DST-II is self-inverse with
    orthonormal scaling, so this reapplies the same sequence)."""
    sign_shape = [1] * x.ndim
    sign_shape[axis] = -1
    s = sign.reshape(sign_shape)
    flipped = jnp.flip(x, axis=axis)
    return _apply_idct2_axis(flipped, axis) * s


def _apply_rfft_periodic_axis(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Forward periodic DFT along ``axis`` (returns complex)."""
    return jnp.fft.fft(x, axis=axis, norm="ortho")


def _apply_irfft_periodic_axis(x: jnp.ndarray, axis: int, N: int) -> jnp.ndarray:
    """Inverse periodic DFT along ``axis``."""
    return jnp.fft.ifft(x, n=N, axis=axis, norm="ortho")


def make_pressure_solver_fft(
    mesh: FVMMesh,
    *,
    bc: str | tuple[str, ...] = "neumann",
    pin_zero_mode: bool = True,
):
    """FFT-based pressure Poisson solver.

    Same interface as :func:`make_pressure_solver` but the per-axis
    transforms dispatch to cuFFT via ``jax.scipy.fft.dct`` (O(N log N))
    instead of a dense N×N matmul. ~10× faster per step on RTX 2060.
    """
    if mesh.cartesian_shape is None:
        raise ValueError("FFT pressure solver requires a Cartesian mesh")
    shape = mesh.cartesian_shape
    spacing = mesh.cartesian_spacing
    dim = len(shape)
    dtype = mesh.V.dtype

    if isinstance(bc, str):
        bcs = (bc,) * dim
    else:
        bcs = tuple(bc)
    for b in bcs:
        if b not in ("neumann", "periodic"):
            raise NotImplementedError(f"pressure bc={b!r} not supported")

    # Eigenvalues per axis
    eig_axes = []
    for a in range(dim):
        if bcs[a] == "neumann":
            eig_axes.append(_dct_eigenvalues_neumann(shape[a], spacing[a], dtype))
        else:
            eig_axes.append(_periodic_eigenvalues_complex(shape[a], spacing[a], dtype))
    lam = jnp.zeros(shape, dtype=dtype)
    for a in range(dim):
        bshape = [1] * dim; bshape[a] = shape[a]
        lam = lam + eig_axes[a].reshape(bshape)
    lam_safe = jnp.where(jnp.abs(lam) < 1e-30, 1.0, lam)
    inv_lam = jnp.where(jnp.abs(lam) < 1e-30, 0.0, 1.0 / lam_safe)
    cell_volume = float(np.prod(spacing))

    def solver(rhs_flat: jnp.ndarray) -> jnp.ndarray:
        b = rhs_flat.reshape(shape) / cell_volume
        # Forward transforms axis-by-axis.
        bhat = b.astype(jnp.complex64) if any(c == "periodic" for c in bcs) else b
        for a in range(dim):
            if bcs[a] == "neumann":
                bhat = _apply_dct2_axis(bhat, a)
            else:
                bhat = _apply_rfft_periodic_axis(bhat, a)
        phat = bhat * inv_lam
        if pin_zero_mode:
            zero_idx = tuple([0] * dim)
            phat = phat.at[zero_idx].set(0.0)
        # Inverse transforms in reverse order
        p = phat
        for a in reversed(range(dim)):
            if bcs[a] == "neumann":
                p = _apply_idct2_axis(p, a)
            else:
                p = _apply_irfft_periodic_axis(p, a, shape[a])
        if any(c == "periodic" for c in bcs):
            p = p.real
        return p.reshape(-1)

    return solver


def _periodic_eigenvalues_complex(N: int, dx: float, dtype) -> jnp.ndarray:
    """Eigenvalues of the 1D periodic discrete Laplacian for full DFT.

    For a circulant 3-point stencil, ``λ_k = -(4/dx²) sin²(π k / N)``
    for k=0..N-1 (the same for both halves of the spectrum, since the
    discrete Laplacian is symmetric).
    """
    k = jnp.arange(N, dtype=dtype)
    return -(4.0 / (dx * dx)) * jnp.sin(jnp.pi * k / N) ** 2


def make_helmholtz_solver_fft(
    mesh: FVMMesh,
    *,
    bc: str | tuple[str, ...] = "dirichlet",
):
    """FFT-based Helmholtz solver: ``(I − α ∇²) x = b``.

    Per-axis BC: ``"dirichlet"`` (DST-II via DCT-II identity),
    ``"neumann"`` (DCT-II), ``"periodic"`` (DFT). α is supplied at
    solve time so the same closure handles many time steps.
    """
    if mesh.cartesian_shape is None:
        raise ValueError("Helmholtz solver requires a Cartesian mesh")
    shape = mesh.cartesian_shape
    spacing = mesh.cartesian_spacing
    dim = len(shape)
    dtype = mesh.V.dtype

    if isinstance(bc, str):
        bcs = (bc,) * dim
    else:
        bcs = tuple(bc)

    eig_axes = []
    signs_for_dst = []   # one per axis; None for non-Dirichlet
    for a in range(dim):
        if bcs[a] == "dirichlet":
            eig_axes.append(_dst_eigenvalues_dirichlet(shape[a], spacing[a], dtype))
            signs_for_dst.append(_alternating_sign(shape[a], dtype))
        elif bcs[a] == "neumann":
            eig_axes.append(_dct_eigenvalues_neumann(shape[a], spacing[a], dtype))
            signs_for_dst.append(None)
        elif bcs[a] == "periodic":
            eig_axes.append(_periodic_eigenvalues_complex(shape[a], spacing[a], dtype))
            signs_for_dst.append(None)
        else:
            raise NotImplementedError(f"Helmholtz bc={bcs[a]!r} not supported")

    lam = jnp.zeros(shape, dtype=dtype)
    for a in range(dim):
        bshape = [1] * dim; bshape[a] = shape[a]
        lam = lam + eig_axes[a].reshape(bshape)

    has_periodic = any(c == "periodic" for c in bcs)

    def solver(b_flat: jnp.ndarray, alpha):
        b = b_flat.reshape(shape + b_flat.shape[1:])
        # Forward transforms
        bhat = b.astype(jnp.complex64) if has_periodic else b
        for a in range(dim):
            if bcs[a] == "dirichlet":
                bhat = _apply_dst2_axis(bhat, signs_for_dst[a], a)
            elif bcs[a] == "neumann":
                bhat = _apply_dct2_axis(bhat, a)
            else:
                bhat = _apply_rfft_periodic_axis(bhat, a)
        denom = 1.0 - alpha * lam
        denom_b = denom.reshape(shape + (1,) * (bhat.ndim - dim))
        xhat = bhat / denom_b
        # Inverse transforms in reverse axis order
        x = xhat
        for a in reversed(range(dim)):
            if bcs[a] == "dirichlet":
                x = _apply_idst2_axis(x, signs_for_dst[a], a)
            elif bcs[a] == "neumann":
                x = _apply_idct2_axis(x, a)
            else:
                x = _apply_irfft_periodic_axis(x, a, shape[a])
        if has_periodic:
            x = x.real
        return x.reshape((-1,) + b_flat.shape[1:])

    return solver


# ─────────────────────────────────────────────────────────────────────────
# Iterative (matrix-free CG) solvers — the unstructured / body-fitted path.
#
# The FFT/dense solvers above diagonalise the Laplacian on a *Cartesian* grid.
# A body-fitted / unstructured mesh has no such basis, so the pressure-Poisson
# and implicit-diffusion (Helmholtz) systems are solved matrix-free with
# Jacobi-preconditioned conjugate gradients on the *same* orthogonal FVM
# operator (``laplacian_orthogonal``) the Cartesian solvers represent — so on a
# Cartesian mesh the iterative path reproduces the FFT/dense result (verified in
# ``tests/verification/test_fvm_iterative_pressure.py``), and on an unstructured
# mesh it is the only applicable path. Sharded scale-out of this CG (MADDENING
# §A5 ``sharded_cg``) is the M2 H1b / F follow-up; this is the single-device core.
# ─────────────────────────────────────────────────────────────────────────

from mime.nodes.environment.fvm.operators import (  # noqa: E402
    laplacian_orthogonal,
)


def _jacobi_pcg(A, b, M, *, rtol, atol, maxiter):
    """Self-contained preconditioned conjugate gradient (``lax.fori_loop``).

    Used instead of ``jax.scipy.sparse.linalg.cg``: the latter builds on
    ``custom_linear_solve``, and a data-dependent ``while_loop`` nested inside
    the large jitted PISO step is mis-serialised by JAX's *persistent
    compilation cache* — the cached executable silently returns the zero
    initial guess (so a production run with the on-disk cache enabled would get
    a wrong, all-zero solve). A fixed-trip-count ``fori_loop`` avoids the
    data-dependent loop entirely and is cache-robust.

    Iterations past convergence are frozen by a residual-norm mask, so the
    extra trips are cheap no-ops on the *state* (the matvec still runs, but the
    solution stops changing) — correctness does not depend on the exact count,
    only on running *enough* iterations. ``A`` / ``M`` are matvec callables; the
    initial guess is zero so the initial residual is ``b``.
    """
    bnorm = jnp.sqrt(jnp.sum(b * b))
    thresh = jnp.maximum(rtol * bnorm, atol)
    x0 = jnp.zeros_like(b)
    r0 = b
    z0 = M(r0)
    rz0 = jnp.sum(r0 * z0)

    def body(_k, c):
        x, r, z, p, rz = c
        active = (jnp.sqrt(jnp.sum(r * r)) > thresh).astype(b.dtype)
        Ap = A(p)
        pAp = jnp.sum(p * Ap)
        alpha = rz / jnp.where(pAp == 0, 1.0, pAp)
        x_n = x + alpha * p
        r_n = r - alpha * Ap
        z_n = M(r_n)
        rz_n = jnp.sum(r_n * z_n)
        beta = rz_n / jnp.where(rz == 0, 1.0, rz)
        p_n = z_n + beta * p
        # Freeze the whole state once converged (active==0): correctness does
        # not depend on the exact trip count, only on running *enough* trips.
        def blend(new, old):
            return active * new + (1.0 - active) * old
        return (blend(x_n, x), blend(r_n, r), blend(z_n, z),
                blend(p_n, p), blend(rz_n, rz))

    x = jax.lax.fori_loop(0, int(maxiter), body, (x0, r0, z0, z0, rz0))[0]
    return x


def _orthogonal_face_coeff(mesh: FVMMesh) -> jnp.ndarray:
    """Per-interior-face geometric coefficient gA_f = |Sf| / |d| of the
    orthogonal Laplacian."""
    return mesh.area / mesh.d_mag


def _interior_laplacian_diagonal(mesh: FVMMesh) -> jnp.ndarray:
    """Diagonal magnitude of the interior orthogonal Laplacian (= sum of gA
    over the faces touching each cell). This is ``diag(-L)`` for the pure-
    interior operator."""
    gA = _orthogonal_face_coeff(mesh)
    return (jax.ops.segment_sum(gA, mesh.owner, num_segments=mesh.N_cells)
            + jax.ops.segment_sum(gA, mesh.neighbour, num_segments=mesh.N_cells))


def _dirichlet_boundary_diagonal(mesh: FVMMesh) -> jnp.ndarray:
    """Extra diagonal magnitude from zero-Dirichlet boundary faces (gA_bf per
    owner cell) — used by the Helmholtz no-slip-wall operator."""
    diag = jnp.zeros((mesh.N_cells,))
    for patch in mesh.patches:
        d_mag_b = jnp.linalg.norm(patch.d, axis=-1)
        gA_bf = patch.area / d_mag_b
        diag = diag + jax.ops.segment_sum(
            gA_bf, patch.owner, num_segments=mesh.N_cells)
    return diag


def make_pressure_solver_iterative(
    mesh: FVMMesh,
    *,
    bc: str | tuple[str, ...] = "neumann",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    maxiter: int | None = None,
):
    """Matrix-free, Jacobi-preconditioned CG pressure-Poisson solver.

    Solves ``laplacian_orthogonal(p) = rhs`` with zero-gradient (Neumann)
    boundaries — the same system the Cartesian FFT/dense pressure solvers
    invert. The pure-Neumann operator is singular (constant null-space), so the
    right-hand side is projected to its range (mean-zero) and the solution is
    gauge-fixed to mean-zero, matching the FFT solver's ``pin_zero_mode``.
    """
    if bc not in ("neumann",) and not (
        isinstance(bc, (tuple, list)) and all(b == "neumann" for b in bc)
    ):
        raise NotImplementedError(
            f"iterative pressure solver supports Neumann pressure BCs; got {bc!r}"
        )
    if maxiter is None:
        maxiter = min(int(mesh.N_cells), 1000)
    diag = _interior_laplacian_diagonal(mesh)            # diag(-L), positive
    inv_diag = 1.0 / jnp.where(diag > 0, diag, 1.0)

    def neg_L(p):
        # Cast to the iterate's dtype: the mesh coefficients may be a different
        # float width (float32 mesh vs an x64 solve), and jax.scipy CG requires
        # the matvec output dtype to match its input exactly.
        return (-laplacian_orthogonal(p, mesh)).astype(p.dtype)

    def precond(r):
        return (inv_diag.reshape((-1,) + (1,) * (r.ndim - 1)) * r).astype(r.dtype)

    def solve(rhs: jnp.ndarray) -> jnp.ndarray:
        b = -(rhs - jnp.mean(rhs, axis=0, keepdims=True))
        p = _jacobi_pcg(neg_L, b, precond, rtol=rtol, atol=atol, maxiter=maxiter)
        return p - jnp.mean(p, axis=0, keepdims=True)

    return solve


def make_helmholtz_solver_iterative(
    mesh: FVMMesh,
    *,
    bc: str | tuple[str, ...] = "dirichlet",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    maxiter: int | None = None,
):
    """Matrix-free, Jacobi-preconditioned CG implicit-diffusion solver.

    Solves ``u - alpha * laplacian_orthogonal(u, dirichlet0) / V = rhs`` with
    zero-Dirichlet (no-slip) wall boundaries — the same Helmholtz system the
    Cartesian DST solver inverts. The operator is symmetric positive-definite,
    so CG converges without a null-space projection.
    """
    if bc not in ("dirichlet",) and not (
        isinstance(bc, (tuple, list)) and all(b == "dirichlet" for b in bc)
    ):
        raise NotImplementedError(
            f"iterative Helmholtz solver supports Dirichlet wall BCs; got {bc!r}"
        )
    if maxiter is None:
        maxiter = min(int(mesh.N_cells), 1000)
    invV = 1.0 / mesh.V
    lap_diag = _interior_laplacian_diagonal(mesh) + _dirichlet_boundary_diagonal(mesh)

    def solve(rhs: jnp.ndarray, alpha: float) -> jnp.ndarray:
        comp = rhs.shape[1:]
        zeros_specs = {
            p.name: {"type": "dirichlet",
                     "value": jnp.zeros((int(p.owner.size),) + comp, dtype=rhs.dtype)}
            for p in mesh.patches
        }
        invV_b = invV.reshape((-1,) + (1,) * (rhs.ndim - 1))

        def A(u):
            # Cast to the iterate's dtype (float32 mesh coeffs vs an x64 solve);
            # jax.scipy CG requires matvec output dtype == input dtype.
            lap = laplacian_orthogonal(u, mesh, boundary_specs=zeros_specs)
            return (u - alpha * lap * invV_b).astype(u.dtype)

        inv_diagA = (1.0 / (1.0 + alpha * lap_diag * invV)).reshape(
            (-1,) + (1,) * (rhs.ndim - 1))

        def precond(r):
            return (inv_diagA * r).astype(r.dtype)

        return _jacobi_pcg(A, rhs, precond, rtol=rtol, atol=atol, maxiter=maxiter)

    return solve
