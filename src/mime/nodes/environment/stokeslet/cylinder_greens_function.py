"""Cylindrical Stokeslet Green's function (Liron-Shahar 1978).

Computes the velocity at x due to a point force at x₀, both inside an
infinite cylinder of radius R with no-slip walls:

    G_cyl(x, x₀) = G_free(x, x₀) + G_image(x, x₀)

The image part is computed via Fourier-Bessel series:
  - Fourier integral in the axial direction z (Gauss-Legendre quadrature)
  - Fourier series in the azimuthal angle φ (truncated at n_max modes)
  - Modified Bessel functions I_n for the radial structure

The image field is decomposed using the toroidal-poloidal representation
of Stokes flow in cylindrical coordinates. For each (n, k) Fourier mode,
a 3×3 linear system determines the image coefficients from the wall BC.

Assembly uses scipy (CPU) for Bessel functions. The resulting matrix is
converted to JAX for GPU-accelerated LU factorization and backsolves.

References:
    Liron & Shahar (1978), J. Fluid Mech. 86:727-744.
    Pozrikidis (1992), Ch. 7 — Stokes flow in tubes.
"""

from __future__ import annotations

import numpy as np
from scipy import special


# ── Coordinate conversions ─────────────────────────────────────────────

def _cart_to_cyl(pts):
    """(N, 3) Cartesian → (rho, phi, z) arrays."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z


def _rotation_matrices(phi):
    """Cartesian ↔ cylindrical rotation matrices at azimuthal angle φ.

    Returns (N, 3, 3) array R such that u_cyl = R @ u_cart.
    """
    c, s = np.cos(phi), np.sin(phi)
    N = len(phi)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = c;  R[:, 0, 1] = s   # rho  = +x cos + y sin
    R[:, 1, 0] = -s; R[:, 1, 1] = c   # phi  = -x sin + y cos
    R[:, 2, 2] = 1.0                    # z    = z
    return R


# ── Wall boundary condition in Fourier space ───────────────────────────

def _wall_bc_k_transform(R, rho0, phi_q, k, mu):
    """z-Fourier transform of the free-space Stokeslet at the cylinder wall.

    For a source at (rho0, 0, 0), evaluates the Stokeslet velocity at
    wall points (R, phi_q[j], ·) and returns the k-transform.

    Returns
    -------
    bc_cos : (N_phi, 3_cyl, 3_cyl_force) — cosine-transform components
    bc_sin : (N_phi, 3_cyl, 3_cyl_force) — sine-transform components

    Force directions are cylindrical at the SOURCE (rho-hat, phi-hat, z-hat).
    Velocity components are cylindrical at the TARGET (wall).

    Parity:
      rho-hat force → u_rho,u_z even (cos), u_phi even (cos), u_z_z odd (sin)
      phi-hat force → same structure but sin/cos in phi swap
      z-hat force   → u_rho,u_phi odd (sin), u_z even (cos)
    """
    N_phi = len(phi_q)

    # Transverse distance from source to wall point
    a = np.sqrt(R**2 + rho0**2 - 2*R*rho0*np.cos(phi_q))  # (N_phi,)
    a = np.maximum(a, 1e-30)  # avoid division by zero

    K0 = special.kv(0, k * a)  # (N_phi,)
    K1 = special.kv(1, k * a)

    # Cartesian displacement at z=0: dx = Rcosφ - ρ₀, dy = Rsinφ
    dx = R * np.cos(phi_q) - rho0
    dy = R * np.sin(phi_q)

    # The z-Fourier transforms of the nine Cartesian Stokeslet components
    # at the wall (ρ = R) for source at (ρ₀, 0, 0):
    #
    # Cosine transform (even in z):
    #   G̃_ij^cos = (1/8πμ) × [2δ_ij K₀ + 2(k/a) Δx_i Δx_j K₁]
    #   for i,j ∈ {x,y} (the transverse-transverse block)
    #
    # Sine transform (odd in z, from z/r³ terms):
    #   G̃_iz^sin = (1/8πμ) × 2k Δx_i K₀   (force in z, vel in x/y)
    #   G̃_zi^sin = same (reciprocal)
    #
    # Cosine transform for zz:
    #   G̃_zz^cos = (1/8πμ) × [4K₀ - 2kaK₁]

    pf = 1.0 / (8.0 * np.pi * mu)
    koa = np.where(a > 1e-20, k / a, 0.0)  # k/a, safe

    # Transverse-transverse block (cosine in z)
    Gxx_c = pf * (2*K0 + 2*koa * dx**2 * K1)
    Gxy_c = pf * (2*koa * dx * dy * K1)
    Gyx_c = Gxy_c
    Gyy_c = pf * (2*K0 + 2*koa * dy**2 * K1)

    # Transverse-z block (sine in z, from z × Δx_i / r³)
    Gxz_s = pf * (2*k * dx * K0)   # force z, velocity x
    Gyz_s = pf * (2*k * dy * K0)
    Gzx_s = Gxz_s                   # reciprocal
    Gzy_s = Gyz_s

    # zz block (cosine in z)
    Gzz_c = pf * (4*K0 - 2*k*a*K1)

    # Convert Cartesian → cylindrical at the wall point
    # u_rho = u_x cos(φ) + u_y sin(φ)
    # u_phi = -u_x sin(φ) + u_y cos(φ)
    cphi = np.cos(phi_q)
    sphi = np.sin(phi_q)

    # Now build bc_cos[phi, alpha_cyl, beta_cyl_src] and bc_sin[...]
    # where beta_cyl_src ∈ {rho-hat, phi-hat, z-hat} at the source.
    # Since source is at φ₀=0: rho-hat = x-hat, phi-hat = y-hat.

    bc_cos = np.zeros((N_phi, 3, 3))
    bc_sin = np.zeros((N_phi, 3, 3))

    # ── rho-hat force (= x-force at φ₀=0) ──
    # u_x, u_y are EVEN in z → cosine; u_z is ODD → sine
    # Cylindrical target: u_ρ = u_x cosφ + u_y sinφ, etc.
    bc_cos[:, 0, 0] = Gxx_c * cphi + Gyx_c * sphi   # u_rho, cos
    bc_cos[:, 1, 0] = -Gxx_c * sphi + Gyx_c * cphi   # u_phi, cos
    bc_sin[:, 2, 0] = Gzx_s                           # u_z, sin

    # ── phi-hat force (= y-force at φ₀=0) ──
    bc_cos[:, 0, 1] = Gxy_c * cphi + Gyy_c * sphi
    bc_cos[:, 1, 1] = -Gxy_c * sphi + Gyy_c * cphi
    bc_sin[:, 2, 1] = Gzy_s

    # ── z-hat force ──
    # u_x, u_y are ODD in z → sine; u_z is EVEN → cosine
    bc_sin[:, 0, 2] = Gxz_s * cphi + Gyz_s * sphi
    bc_sin[:, 1, 2] = -Gxz_s * sphi + Gyz_s * cphi
    bc_cos[:, 2, 2] = Gzz_c

    return bc_cos, bc_sin


def _azimuthal_modes(bc_phi, phi_q, n_max):
    """Compute azimuthal Fourier coefficients from wall BC at φ-grid.

    bc_phi : (N_phi, ...) — values at quadrature points
    Returns : (n_max+1, ...) — Fourier cosine coefficients a_n

    Uses trapezoidal rule (exact for trig polynomials on uniform grid).
    a_n = (2/N) Σ_j bc(φ_j) cos(nφ_j)  for n ≥ 1
    a_0 = (1/N) Σ_j bc(φ_j)
    """
    N = len(phi_q)
    modes = np.zeros((n_max + 1,) + bc_phi.shape[1:])
    for n in range(n_max + 1):
        weight = 1.0 / N if n == 0 else 2.0 / N
        modes[n] = weight * np.sum(
            bc_phi * np.cos(n * phi_q)[:, None, None], axis=0
        )
    return modes


def _azimuthal_modes_sin(bc_phi, phi_q, n_max):
    """Same but for sine coefficients b_n."""
    N = len(phi_q)
    modes = np.zeros((n_max + 1,) + bc_phi.shape[1:])
    for n in range(1, n_max + 1):
        modes[n] = (2.0 / N) * np.sum(
            bc_phi * np.sin(n * phi_q)[:, None, None], axis=0
        )
    return modes


# ── Image coefficient solver ───────────────────────────────────────────

def _ive_ivpe(n, s):
    """Exponentially-scaled I_n and I_n' at argument s.

    Returns (ive, ivpe) where:
      ive  = exp(-s) I_n(s)
      ivpe = exp(-s) I_n'(s)  [derivative w.r.t. s]

    Using the recurrence: I_n'(s) = I_{n-1}(s) - (n/s) I_n(s)
    → ivpe = ive(n-1, s) - (n/s) ive(n, s)
    """
    In_e = special.ive(n, s)
    if n == 0:
        Inp_e = special.ive(1, s)  # I_0' = I_1
    else:
        In_minus_e = special.ive(n - 1, s)
        Inp_e = In_minus_e - (n / s) * In_e if s > 1e-30 else 0.0
    return In_e, Inp_e


def _image_system_matrix_scaled(n, k, R, z_parity):
    """Build the 3×3 system matrix M_s = M × exp(-kR).

    Uses exponentially-scaled Bessel functions to avoid overflow.

    The sign of the poloidal contribution to u_ρ and u_φ depends on
    the z-parity of χ:
      - z_parity = +1 (transverse force, χ ~ sin(kz)): ∂χ/∂z = +k cos
      - z_parity = -1 (axial force, χ ~ cos(kz)):      ∂χ/∂z = -k sin

    Row 0 (u_ρ): toroidal + z_parity × poloidal
    Row 1 (u_φ): toroidal - z_parity × poloidal  (opposite sign convention)
    Row 2 (u_z): poloidal only (no z-derivative → sign-independent)
    """
    if k < 1e-15:
        return np.eye(3) * 1e-30

    s = k * R
    In_e, Inp_e = _ive_ivpe(n, s)

    qn = (k**2 * R**2 + n**2) / (k * R) if s > 1e-30 else 0.0
    p = z_parity  # +1 or -1

    M_s = np.array([
        [n * In_e / R,         p * k**2 * Inp_e,        p * qn * In_e],
        [-k * Inp_e,          -p * n * k * In_e / R,    -p * n * Inp_e],
        [0.0,                  -k**2 * In_e,            -(2 * In_e + k * R * Inp_e)],
    ])
    return M_s


def _image_basis_at_rho_scaled(n, k, rho, z_parity):
    """Evaluate scaled image basis: B_s = B × exp(-kρ).

    z_parity: +1 for transverse force, -1 for axial force.
    """
    if k < 1e-15:
        return np.zeros((3, 3))

    s = k * rho
    In_e, Inp_e = _ive_ivpe(n, s)

    qn = (k**2 * rho**2 + n**2) / (k * rho) if s > 1e-30 else 0.0
    p = z_parity

    B_s = np.array([
        [n * In_e / rho if rho > 1e-30 else 0.0,
         p * k**2 * Inp_e,
         p * qn * In_e],
        [-k * Inp_e,
         -p * n * k * In_e / rho if rho > 1e-30 else 0.0,
         -p * n * Inp_e],
        [0.0,
         -k**2 * In_e,
         -(2 * In_e + k * rho * Inp_e)],
    ])
    return B_s


# ── Full matrix assembly ──────────────────────────────────────────────

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

    G_cyl = G_free + G_image

    Uses the Liron-Shahar cylindrical Green's function for the image.
    All computation on CPU (scipy Bessel functions).

    Parameters
    ----------
    colloc_pts : (Nc, 3) target points (Cartesian)
    source_pts : (Ns, 3) source points (Cartesian)
    R_cyl : cylinder radius
    mu : dynamic viscosity
    n_max : max azimuthal Fourier mode
    n_k : number of k-quadrature points
    n_phi : number of φ-quadrature points for wall BC

    Returns
    -------
    G : (3Nc, 3Ns) dense matrix (numpy float64)
    """
    Nc = len(colloc_pts)
    Ns = len(source_pts)

    # Cylindrical coordinates
    rho_c, phi_c, z_c = _cart_to_cyl(colloc_pts)
    rho_s, phi_s, z_s = _cart_to_cyl(source_pts)

    # Free-space Stokeslet (singular, no regularisation)
    G_free = _assemble_free_space(colloc_pts, source_pts, mu)

    # k-quadrature: use substitution k = t²/R to smooth the log singularity
    # of K₀(kR) near k=0. Integrand in t is smooth.
    # ∫₀^∞ f(k) dk = ∫₀^∞ f(t²/R) × (2t/R) dt
    # Truncate at t_max where the Bessel function decay makes the integrand negligible.
    max_rho_s = np.max(rho_s) if Ns > 0 else 0.0
    min_gap = R_cyl - max_rho_s
    min_gap = max(min_gap, 0.1 * R_cyl)
    # At k = t²/R, the integrand decays ~ exp(-t² × min_gap / R)
    # Need t² × min_gap/R ~ 30 → t_max ~ sqrt(30 R / min_gap)
    t_max = np.sqrt(30.0 * R_cyl / min_gap)
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_k)
    t_nodes = 0.5 * t_max * (t_nodes + 1)       # map [-1,1] → [0, t_max]
    t_weights = 0.5 * t_max * t_weights
    # Convert to k-space: k = t²/R, dk = 2t/R dt
    k_nodes = t_nodes**2 / R_cyl
    k_weights = t_weights * 2 * t_nodes / R_cyl  # Jacobian 2t/R

    # φ-quadrature grid (uniform, for trapezoidal rule = exact for trig)
    phi_q = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Rotation matrices for cylindrical ↔ Cartesian at targets and sources
    R_tgt = _rotation_matrices(phi_c)   # (Nc, 3, 3): u_cyl = R @ u_cart
    R_src = _rotation_matrices(phi_s)   # (Ns, 3, 3)

    # Accumulate image matrix in Cartesian
    G_image = np.zeros((3 * Nc, 3 * Ns))

    for ik, (k_val, wk) in enumerate(zip(k_nodes, k_weights)):
        if k_val < 1e-15:
            continue

        for n in range(n_max + 1):
            # Two system matrices: transverse (z_parity=+1) and axial (-1)
            M_s_trans = _image_system_matrix_scaled(n, k_val, R_cyl, z_parity=+1)
            M_s_axial = _image_system_matrix_scaled(n, k_val, R_cyl, z_parity=-1)

            try:
                M_inv_trans = np.linalg.inv(M_s_trans)
                M_inv_axial = np.linalg.inv(M_s_axial)
            except np.linalg.LinAlgError:
                continue

            # Two target bases
            B_tgt_trans = np.array([
                _image_basis_at_rho_scaled(n, k_val, rho_c[i], z_parity=+1)
                for i in range(Nc)
            ])  # (Nc, 3, 3)
            B_tgt_axial = np.array([
                _image_basis_at_rho_scaled(n, k_val, rho_c[i], z_parity=-1)
                for i in range(Nc)
            ])  # (Nc, 3, 3)

            exp_factor = np.exp(k_val * (rho_c - R_cyl))  # (Nc,)

            # For each source: compute wall BC, solve for coefficients
            for js in range(Ns):
                rho0 = rho_s[js]
                if rho0 >= R_cyl:
                    continue

                # Wall BC in Fourier space
                bc_cos, bc_sin = _wall_bc_k_transform(
                    R_cyl, rho0, phi_q, k_val, mu,
                )
                # bc_cos: (N_phi, 3_cyl, 3_cyl_force)
                # bc_sin: (N_phi, 3_cyl, 3_cyl_force)

                # Azimuthal mode n coefficients
                # For rho-hat and phi-hat forces: different cos/sin structure
                # rho-hat, z-hat: u_rho,u_z ~ cos(nφ), u_phi ~ sin(nφ)
                # phi-hat:        u_rho,u_z ~ sin(nφ), u_phi ~ cos(nφ)

                # Wall BC for each cylindrical force direction:
                wall_rhs = np.zeros((3, 3))  # (alpha_cyl, beta_cyl_force)

                # z-parity per (force direction, velocity component):
                #   rho/phi-force: u_rho,u_phi EVEN(cos), u_z ODD(sin)
                #   z-force:       u_rho,u_phi ODD(sin),  u_z EVEN(cos)
                #
                # Azimuthal mode type per force direction:
                #   rho/z-force (Type A): u_rho,u_z ~ cos(nΔφ), u_phi ~ sin(nΔφ)
                #   phi-force   (Type B): u_rho,u_z ~ sin(nΔφ), u_phi ~ cos(nΔφ)

                for beta_force in range(3):
                    if beta_force == 0:  # rho-hat force
                        # u_rho,u_phi from bc_cos (even-z), u_z from bc_sin (odd-z)
                        wall_rhs[0, 0] = _azimuthal_mode_single(
                            bc_cos[:, 0, 0], phi_q, n, 'cos')
                        wall_rhs[1, 0] = _azimuthal_mode_single(
                            bc_cos[:, 1, 0], phi_q, n, 'sin')
                        wall_rhs[2, 0] = _azimuthal_mode_single(
                            bc_sin[:, 2, 0], phi_q, n, 'cos')

                    elif beta_force == 1:  # phi-hat force (Type B)
                        # u_rho,u_phi from bc_cos (even-z), u_z from bc_sin (odd-z)
                        wall_rhs[0, 1] = _azimuthal_mode_single(
                            bc_cos[:, 0, 1], phi_q, n, 'sin')
                        wall_rhs[1, 1] = _azimuthal_mode_single(
                            bc_cos[:, 1, 1], phi_q, n, 'cos')
                        wall_rhs[2, 1] = _azimuthal_mode_single(
                            bc_sin[:, 2, 1], phi_q, n, 'sin')

                    else:  # z-hat force (beta_force == 2)
                        # u_rho,u_phi from bc_sin (odd-z), u_z from bc_cos (even-z)
                        wall_rhs[0, 2] = _azimuthal_mode_single(
                            bc_sin[:, 0, 2], phi_q, n, 'cos')
                        wall_rhs[1, 2] = _azimuthal_mode_single(
                            bc_sin[:, 1, 2], phi_q, n, 'sin')
                        wall_rhs[2, 2] = _azimuthal_mode_single(
                            bc_cos[:, 2, 2], phi_q, n, 'cos')

                # Solve for each force type with the appropriate M:
                # Transverse (beta=0,1): z_parity=+1
                # Axial (beta=2): z_parity=-1
                coeffs = np.zeros((3, 3))
                coeffs[:, 0] = -M_inv_trans @ wall_rhs[:, 0]  # rho-hat
                coeffs[:, 1] = -M_inv_trans @ wall_rhs[:, 1]  # phi-hat
                coeffs[:, 2] = -M_inv_axial @ wall_rhs[:, 2]  # z-hat

                # Evaluate at each target and accumulate
                dphi = phi_c - phi_s[js]  # (Nc,)
                dz = z_c - z_s[js]        # (Nc,)

                for it in range(Nc):
                    # Scaled image velocity for each force type
                    u_trans = exp_factor[it] * (B_tgt_trans[it] @ coeffs[:, :2])  # (3, 2)
                    u_axial = exp_factor[it] * (B_tgt_axial[it] @ coeffs[:, 2:])  # (3, 1)
                    u_cyl = np.column_stack([u_trans, u_axial])  # (3, 3)

                    # Phase factors depend on force type and component
                    # Type A (rho, z forces): u_rho,u_z ~ cos(n dphi), u_phi ~ sin(n dphi)
                    # Type B (phi force):     u_rho,u_z ~ sin(n dphi), u_phi ~ cos(n dphi)
                    cn = np.cos(n * dphi[it])
                    sn = np.sin(n * dphi[it])

                    # z-phase: depends on force direction and component
                    # rho-force: u_rho,u_phi cos(kdz), u_z sin(kdz)
                    # phi-force: same
                    # z-force:   u_rho,u_phi sin(kdz), u_z cos(kdz)
                    ckz = np.cos(k_val * dz[it])
                    skz = np.sin(k_val * dz[it])

                    # Build full phase-modulated image in cyl-cyl
                    # G_image_cyl[alpha, beta_cyl]
                    G_cyl_nk = np.zeros((3, 3))

                    # beta = 0 (rho-hat force): z-parity = cos for u_rho,u_phi; sin for u_z
                    G_cyl_nk[0, 0] = u_cyl[0, 0] * cn * ckz
                    G_cyl_nk[1, 0] = u_cyl[1, 0] * sn * ckz
                    G_cyl_nk[2, 0] = u_cyl[2, 0] * cn * skz

                    # beta = 1 (phi-hat force): phi-parity flipped
                    G_cyl_nk[0, 1] = u_cyl[0, 1] * sn * ckz
                    G_cyl_nk[1, 1] = u_cyl[1, 1] * cn * ckz
                    G_cyl_nk[2, 1] = u_cyl[2, 1] * sn * skz

                    # beta = 2 (z-hat force): z-parity flipped
                    G_cyl_nk[0, 2] = u_cyl[0, 2] * cn * skz
                    G_cyl_nk[1, 2] = u_cyl[1, 2] * sn * skz
                    G_cyl_nk[2, 2] = u_cyl[2, 2] * cn * ckz

                    # Quadrature weight: (1/π) × ∫ dk
                    scale = wk / np.pi

                    # Convert cyl-cyl → cart-cart
                    # G_cart = R_tgt^T @ G_cyl @ R_src
                    R_t = R_tgt[it]   # (3,3): u_cyl = R_t @ u_cart
                    R_s = R_src[js]

                    # G_cyl maps cyl-force → cyl-vel
                    # G_cart = R_t^T @ G_cyl @ R_s
                    # (since u_cart = R_t^T @ u_cyl and f_cyl = R_s @ f_cart)
                    G_cart = R_t.T @ (scale * G_cyl_nk) @ R_s

                    # Accumulate into image matrix
                    G_image[3*it:3*it+3, 3*js:3*js+3] += G_cart

    return G_free + G_image


def _azimuthal_mode_single(f_phi, phi_q, n, mode_type):
    """Compute single azimuthal Fourier coefficient.

    mode_type: 'cos' or 'sin'
    """
    N = len(phi_q)
    if mode_type == 'cos':
        weight = 1.0 / N if n == 0 else 2.0 / N
        return weight * np.sum(f_phi * np.cos(n * phi_q))
    else:
        if n == 0:
            return 0.0
        return (2.0 / N) * np.sum(f_phi * np.sin(n * phi_q))


def _assemble_free_space(colloc, source, mu):
    """Free-space singular Stokeslet matrix (3Nc × 3Ns)."""
    Nc, Ns = len(colloc), len(source)
    r = colloc[:, None, :] - source[None, :, :]  # (Nc, Ns, 3)
    dist = np.sqrt(np.sum(r**2, axis=-1))  # (Nc, Ns)

    inv_d = 1.0 / np.maximum(dist, 1e-30)
    inv_d3 = inv_d**3

    I3 = np.eye(3)[None, None, :, :]
    rr = r[..., :, None] * r[..., None, :]

    pf = 1.0 / (8.0 * np.pi * mu)
    S = pf * (I3 * inv_d[..., None, None]
              + rr * inv_d3[..., None, None])

    return S.transpose(0, 2, 1, 3).reshape(3 * Nc, 3 * Ns)
