"""Vectorized cylindrical Stokeslet Green's function (Liron-Shahar 1978).

Computes G_cyl = G_free + G_image for a Stokeslet inside an infinite
no-slip cylinder. The image is computed via Fourier-Bessel series:
  - Fourier integral in z (Gauss-Legendre with t² substitution)
  - Fourier series in φ (truncated at n_max modes)
  - Toroidal-poloidal Stokes solution with z-parity-dependent signs

Assembly is fully vectorized over sources and targets. Sources are
binned by cylindrical radius ρ₀ — sources with the same ρ₀ share
identical Bessel evaluations and mode coefficients, so only the
(cheap) phase factors and rotations differ per source.

Uses scipy Bessel functions (CPU). The resulting matrix is converted
to JAX for GPU LU factorization.
"""

from __future__ import annotations

import numpy as np
from scipy import special


# ── Coordinate helpers ─────────────────────────────────────────────────

def _cart_to_cyl(pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z


def _rotation_matrices(phi):
    """(N, 3, 3) rotation from Cartesian to cylindrical at each φ."""
    c, s = np.cos(phi), np.sin(phi)
    N = len(phi)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = c;  R[:, 0, 1] = s
    R[:, 1, 0] = -s; R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def _ive_ivpe(n, s):
    """Scaled I_n and I_n' (derivative w.r.t. s). Accepts arrays."""
    In_e = special.ive(n, s)
    if n == 0:
        Inp_e = special.ive(1, s)
    else:
        Inp_e = special.ive(n - 1, s) - np.where(
            s > 1e-30, n / s, 0.0) * In_e
    return In_e, Inp_e


# ── Public interface ───────────────────────────────────────────────────

def assemble_image_correction_matrix(
    body_pts: np.ndarray,
    body_wts: np.ndarray,
    R_cyl: float,
    mu: float,
    n_max: int = 15,
    n_k: int = 80,
    n_phi: int = 64,
) -> np.ndarray:
    """Return (3N, 3N) wall correction matrix in BEM-weighted form.

    The returned matrix G_wall satisfies:
        A_confined = A_body_BEM + G_wall

    where A_body_BEM is the free-space regularised BEM system matrix
    and G_wall accounts for the no-slip cylinder wall at radius R_cyl.

    This is a pure function with no class state. The wall correction
    can be swapped for FMM or H-matrix later without touching the
    body solver.
    """
    N = len(body_pts)
    G_image = _assemble_image_only(
        body_pts, body_pts, R_cyl, mu,
        n_max=n_max, n_k=n_k, n_phi=n_phi,
    )
    # Apply per-source quadrature weights to match BEM convention.
    # G_image already includes 1/(8πμ) from the Green's function.
    wts = np.asarray(body_wts)
    for j in range(N):
        G_image[:, 3*j:3*j+3] *= wts[j]
    return G_image


def assemble_cylinder_stokeslet_matrix(
    colloc_pts: np.ndarray,
    source_pts: np.ndarray,
    R_cyl: float,
    mu: float,
    n_max: int = 15,
    n_k: int = 80,
    n_phi: int = 64,
) -> np.ndarray:
    """Assemble (3Nc × 3Ns) confined Stokeslet matrix G_cyl.

    G_cyl = G_free + G_image (no BEM weights — raw Green's function).
    """
    G_free = _assemble_free_space(colloc_pts, source_pts, mu)
    G_image = _assemble_image_only(
        colloc_pts, source_pts, R_cyl, mu,
        n_max=n_max, n_k=n_k, n_phi=n_phi,
    )
    return G_free + G_image


# ── Core image assembly (ρ₀-binned + source-chunked) ──────────────────

def _assemble_image_only(
    colloc_pts, source_pts, R_cyl, mu,
    n_max=15, n_k=80, n_phi=64,
):
    """Compute (3Nc × 3Ns) image correction matrix G_image.

    Sources are binned by cylindrical radius ρ₀. The expensive Bessel
    evaluations and mode coefficients depend only on ρ₀, so sources
    sharing the same ρ₀ reuse these. Only the cheap phase factors and
    rotations differ per source.

    For large meshes, the target-source outer product is chunked to
    keep peak memory under ~500 MB.
    """
    Nc = len(colloc_pts)
    Ns = len(source_pts)

    rho_c, phi_c, z_c = _cart_to_cyl(colloc_pts)
    rho_s, phi_s, z_s = _cart_to_cyl(source_pts)

    # ── ρ₀ binning ────────────────────────────────────────────────
    # Round ρ₀ to 4 significant figures to identify unique bins.
    rho_tol = 1e-6 * R_cyl
    rho_rounded = np.round(rho_s / rho_tol) * rho_tol
    unique_rho, inverse_idx = np.unique(rho_rounded, return_inverse=True)
    n_bins = len(unique_rho)
    # inverse_idx[j] = bin index for source j

    # ── Quadrature setup ──────────────────────────────────────────
    max_rho_s = np.max(rho_s) if Ns > 0 else 0.0
    min_gap = max(R_cyl - max_rho_s, 0.1 * R_cyl)
    t_max = np.sqrt(30.0 * R_cyl / min_gap)
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_k)
    t_nodes = 0.5 * t_max * (t_nodes + 1)
    t_weights = 0.5 * t_max * t_weights
    k_nodes = t_nodes**2 / R_cyl
    k_weights = t_weights * 2 * t_nodes / R_cyl

    phi_q = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    cos_phi_q = np.cos(phi_q)
    sin_phi_q = np.sin(phi_q)

    R_tgt = _rotation_matrices(phi_c)               # (Nc, 3, 3)
    R_tgt_T = R_tgt.transpose(0, 2, 1).copy()       # (Nc, 3, 3) — R^T
    R_src = _rotation_matrices(phi_s)               # (Ns, 3, 3)

    # ── Precompute phase-factor matrices ──────────────────────────
    dphi_mat = phi_c[:, None] - phi_s[None, :]       # (Nc, Ns)
    dz_mat = z_c[:, None] - z_s[None, :]             # (Nc, Ns)
    cn_mats = np.stack([np.cos(nn * dphi_mat) for nn in range(n_max + 1)])
    sn_mats = np.stack([np.sin(nn * dphi_mat) for nn in range(n_max + 1)])

    # ── Precompute wall-distance array per bin ────────────────────
    # a_bins[b, phi] = transverse distance at wall for bin b
    a_bins = np.sqrt(R_cyl**2 + unique_rho[:, None]**2
                     - 2 * R_cyl * unique_rho[:, None] * cos_phi_q[None, :])
    a_bins = np.maximum(a_bins, 1e-30)               # (n_bins, N_phi)

    pf = 1.0 / (8.0 * np.pi * mu)

    # ── Source chunking (for the Nc × Ns_chunk outer product) ─────
    max_bytes = 500_000_000
    chunk_ns = max(1, max_bytes // (Nc * 9 * 8))
    chunk_ns = min(chunk_ns, Ns)

    G_image = np.zeros((3 * Nc, 3 * Ns))

    for ik in range(len(k_nodes)):
        k_val = k_nodes[ik]
        wk = k_weights[ik]
        if k_val < 1e-15:
            continue

        # ── Per-bin Bessel evals + wall BCs (EXPENSIVE, done n_bins times) ──
        ka_bins = k_val * a_bins                     # (n_bins, N_phi)
        K0_bins = special.kv(0, ka_bins)
        K1_bins = special.kv(1, ka_bins)

        koa_bins = np.where(a_bins > 1e-20, k_val / a_bins, 0.0)

        # Cartesian displacement at wall per bin
        dx_bins = R_cyl * cos_phi_q[None, :] - unique_rho[:, None]
        dy_bins = R_cyl * sin_phi_q[None, :]         # broadcast (1, N_phi)

        # 6 Cartesian wall-BC components per bin, each (n_bins, N_phi)
        Gxx_c = pf * (2*K0_bins + 2*koa_bins * dx_bins**2 * K1_bins)
        Gxy_c = pf * (2*koa_bins * dx_bins * dy_bins * K1_bins)
        Gyy_c = pf * (2*K0_bins + 2*koa_bins * dy_bins**2 * K1_bins)
        Gxz_s = pf * (2*k_val * dx_bins * K0_bins)
        Gyz_s = pf * (2*k_val * dy_bins * K0_bins)
        Gzz_c = pf * (4*K0_bins - 2*k_val*a_bins*K1_bins)

        cphi = cos_phi_q[None, :]
        sphi = sin_phi_q[None, :]

        # 9 cylindrical wall-BC arrays per bin, each (n_bins, N_phi)
        bc0_rho_cos = Gxx_c * cphi + Gxy_c * sphi
        bc0_phi_cos = -Gxx_c * sphi + Gxy_c * cphi
        bc0_uz_sin = Gxz_s
        bc1_rho_cos = Gxy_c * cphi + Gyy_c * sphi
        bc1_phi_cos = -Gxy_c * sphi + Gyy_c * cphi
        bc1_uz_sin = Gyz_s
        bc2_rho_sin = Gxz_s * cphi + Gyz_s * sphi
        bc2_phi_sin = -Gxz_s * sphi + Gyz_s * cphi
        bc2_uz_cos = Gzz_c

        # Axial phase factors for this k (precomputed, Nc × Ns)
        ckz_mat = np.cos(k_val * dz_mat)
        skz_mat = np.sin(k_val * dz_mat)

        for n in range(n_max + 1):
            # ── System matrices (scalar 3×3) ──────────────────────
            sR = k_val * R_cyl
            In_e_R, Inp_e_R = _ive_ivpe(n, sR)
            qn_R = (k_val**2 * R_cyl**2 + n**2) / (k_val * R_cyl) \
                if sR > 1e-30 else 0.0

            def _build_M(p):
                return np.array([
                    [n * In_e_R / R_cyl,
                     p * k_val**2 * Inp_e_R,
                     p * qn_R * In_e_R],
                    [-k_val * Inp_e_R,
                     -p * n * k_val * In_e_R / R_cyl,
                     -p * n * Inp_e_R],
                    [0.0,
                     -k_val**2 * In_e_R,
                     -(2 * In_e_R + k_val * R_cyl * Inp_e_R)],
                ])

            M_t = _build_M(+1)
            M_a = _build_M(-1)
            try:
                Mi_t = np.linalg.inv(M_t)
                Mi_a = np.linalg.inv(M_a)
            except np.linalg.LinAlgError:
                continue

            # ── Target basis (Nc, 3, 3) ───────────────────────────
            s_tgt = k_val * rho_c
            In_e_t, Inp_e_t = _ive_ivpe(n, s_tgt)
            qn_t = np.where(s_tgt > 1e-30,
                            (k_val**2 * rho_c**2 + n**2) / (k_val * rho_c),
                            0.0)
            nIr = np.where(rho_c > 1e-30, n * In_e_t / rho_c, 0.0)
            nkIr = np.where(rho_c > 1e-30, n * k_val * In_e_t / rho_c, 0.0)
            exp_fac = np.exp(k_val * (rho_c - R_cyl))

            def _build_B(p):
                B = np.zeros((Nc, 3, 3))
                B[:, 0, 0] = nIr
                B[:, 0, 1] = p * k_val**2 * Inp_e_t
                B[:, 0, 2] = p * qn_t * In_e_t
                B[:, 1, 0] = -k_val * Inp_e_t
                B[:, 1, 1] = -p * nkIr
                B[:, 1, 2] = -p * n * Inp_e_t
                B[:, 2, 1] = -k_val**2 * In_e_t
                B[:, 2, 2] = -(2 * In_e_t + k_val * rho_c * Inp_e_t)
                return B

            Bt_t = _build_B(+1)
            Bt_a = _build_B(-1)

            # ── Per-bin mode extraction + coefficient solve ────────
            cn_q = np.cos(n * phi_q)
            sn_q = np.sin(n * phi_q)
            w0 = 1.0 / n_phi if n == 0 else 2.0 / n_phi

            # wall_rhs per bin: (n_bins, 3, 3)
            wr = np.zeros((n_bins, 3, 3))
            wr[:, 0, 0] = w0 * (bc0_rho_cos @ cn_q)
            wr[:, 1, 0] = w0 * (bc0_phi_cos @ sn_q)
            wr[:, 2, 0] = w0 * (bc0_uz_sin  @ cn_q)
            wr[:, 0, 1] = w0 * (bc1_rho_cos @ sn_q)
            wr[:, 1, 1] = w0 * (bc1_phi_cos @ cn_q)
            wr[:, 2, 1] = w0 * (bc1_uz_sin  @ sn_q)
            wr[:, 0, 2] = w0 * (bc2_rho_sin @ cn_q)
            wr[:, 1, 2] = w0 * (bc2_phi_sin @ sn_q)
            wr[:, 2, 2] = w0 * (bc2_uz_cos  @ cn_q)

            # coeffs per bin: (n_bins, 3, 3)
            coeffs_bins = np.zeros((n_bins, 3, 3))
            coeffs_bins[:, :, :2] = -np.einsum(
                'ab,sbc->sac', Mi_t, wr[:, :, :2])
            coeffs_bins[:, :, 2] = -np.einsum(
                'ab,sb->sa', Mi_a, wr[:, :, 2])

            # Scatter bin coefficients to all sources: (Ns, 3, 3)
            coeffs_all = coeffs_bins[inverse_idx]

            # ── Phase + eval + rotation + accumulate (chunked) ────
            cn = cn_mats[n]                          # (Nc, Ns)
            sn = sn_mats[n]
            ckz = ckz_mat
            skz = skz_mat
            scale = wk / np.pi

            for j0 in range(0, Ns, chunk_ns):
                j1 = min(j0 + chunk_ns, Ns)
                _cs = slice(j0, j1)
                Ns_c = j1 - j0

                # Target eval: (Nc, Ns_c, 3, 2) and (Nc, Ns_c, 3)
                u_trans = exp_fac[:, None, None, None] * np.matmul(
                    Bt_t[:, np.newaxis, :, :],
                    coeffs_all[np.newaxis, _cs, :, :2])
                u_axial = exp_fac[:, None, None] * np.matmul(
                    Bt_a[:, np.newaxis, :, :],
                    coeffs_all[np.newaxis, _cs, :, 2:3]).squeeze(-1)

                # Phase + G_nk assembly: (Nc, Ns_c, 3, 3)
                cn_c = cn[:, _cs]
                sn_c = sn[:, _cs]
                ckz_c = ckz[:, _cs]
                skz_c = skz[:, _cs]

                G_nk = np.empty((Nc, Ns_c, 3, 3))

                G_nk[:, :, 0, 0] = u_trans[:, :, 0, 0] * cn_c * ckz_c
                G_nk[:, :, 1, 0] = u_trans[:, :, 1, 0] * sn_c * ckz_c
                G_nk[:, :, 2, 0] = u_trans[:, :, 2, 0] * cn_c * skz_c

                G_nk[:, :, 0, 1] = u_trans[:, :, 0, 1] * sn_c * ckz_c
                G_nk[:, :, 1, 1] = u_trans[:, :, 1, 1] * cn_c * ckz_c
                G_nk[:, :, 2, 1] = u_trans[:, :, 2, 1] * sn_c * skz_c

                G_nk[:, :, 0, 2] = u_axial[:, :, 0] * cn_c * skz_c
                G_nk[:, :, 1, 2] = u_axial[:, :, 1] * sn_c * skz_c
                G_nk[:, :, 2, 2] = u_axial[:, :, 2] * cn_c * ckz_c

                G_nk *= scale

                # Rotation: G_cart = R_tgt^T @ G_nk @ R_src
                G_tmp = np.matmul(G_nk, R_src[np.newaxis, _cs, :, :])
                G_cart = np.matmul(
                    R_tgt_T[:, np.newaxis, :, :], G_tmp)

                # Accumulate: (Nc, Ns_c, 3, 3) → (3Nc, 3Ns_c)
                G_image[:, 3*j0:3*j1] += G_cart.transpose(
                    0, 2, 1, 3).reshape(3 * Nc, 3 * Ns_c)

    return G_image


# ── Free-space Stokeslet (unchanged) ──────────────────────────────────

def _assemble_free_space(colloc, source, mu):
    Nc, Ns = len(colloc), len(source)
    r = colloc[:, None, :] - source[None, :, :]
    dist = np.sqrt(np.sum(r**2, axis=-1))
    inv_d = 1.0 / np.maximum(dist, 1e-30)
    inv_d3 = inv_d**3
    I3 = np.eye(3)[None, None, :, :]
    rr = r[..., :, None] * r[..., None, :]
    pf = 1.0 / (8.0 * np.pi * mu)
    S = pf * (I3 * inv_d[..., None, None] + rr * inv_d3[..., None, None])
    return S.transpose(0, 2, 1, 3).reshape(3 * Nc, 3 * Ns)
