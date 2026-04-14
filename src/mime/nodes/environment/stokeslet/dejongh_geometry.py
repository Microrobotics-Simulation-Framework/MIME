"""Parametric surface mesh for de Jongh et al. (2025) screw-shaped UMRs.

Generates BEM-quality surface meshes from the parametric Eq. 1:

    r(θ, ζ) = ζ ẑ + ρ(θ) [cos(α) x̂ + sin(α) ŷ]
    α = ν̃ ζ / R_cyl + θ
    ρ(θ) = R_cyl [1 + ε sin(Nθ)]

where ν̃ is the normalized wavenumber, R_cyl is the UMR cylinder radius,
ε is the modulation amplitude, and N is the number of starts.

Reference: de Jongh et al. (2025), Nonlinear Dyn. 113:29197-29213.
    Eq. 1, Tables 1-2.
"""

from __future__ import annotations

import numpy as np
from .surface_mesh import SurfaceMesh


# ── FL group parameters (Table 1) ────────────────────────────────────
# All share: L_UMR = 7.47 mm, N = 2, ε = 0.33, R_cyl = 1.56 mm

FL_TABLE = {
    1:  {"nu": 0.25},
    2:  {"nu": 0.5},
    3:  {"nu": 1.0},
    4:  {"nu": 1.25},
    5:  {"nu": 1.4},
    6:  {"nu": 1.55},
    7:  {"nu": 1.8},
    8:  {"nu": 2.0},
    9:  {"nu": 2.33},
    10: {"nu": 3.0},
    11: {"nu": 3.5},
}

FL_L_UMR = 7.47  # mm, fixed for all FL designs

# ── FW group parameters (Table 2) ────────────────────────────────────
# All share: N_t = 1 turn, N = 2, ε = 0.33, R_cyl = 1.56 mm

FW_TABLE = {
    1: {"nu": 1.0,  "L_UMR": 9.73},
    2: {"nu": 1.25, "L_UMR": 7.77},
    3: {"nu": 1.4,  "L_UMR": 6.88},
    4: {"nu": 1.55, "L_UMR": 6.29},
    5: {"nu": 2.0,  "L_UMR": 4.82},
    6: {"nu": 2.33, "L_UMR": 4.13},
}

# ── Common geometry constants ────────────────────────────────────────

R_CYL_DEFAULT = 1.56   # mm, UMR cylinder radius
EPSILON_DEFAULT = 0.33  # modulation amplitude
N_STARTS_DEFAULT = 2    # number of starts


def dejongh_umr_surface(
    nu: float,
    L_UMR: float,
    R_cyl: float = R_CYL_DEFAULT,
    epsilon: float = EPSILON_DEFAULT,
    N: int = N_STARTS_DEFAULT,
    n_theta: int = 80,
    n_zeta: int = 120,
) -> SurfaceMesh:
    """Generate BEM surface mesh for a de Jongh screw-shaped UMR.

    Triangulates the (θ, ζ) parameter domain directly. Each quad
    cell is split into 2 triangles. Collocation points are triangle
    centroids; normals from analytical ∂r/∂θ × ∂r/∂ζ; weights from
    triangle areas.

    Parameters
    ----------
    nu : float
        Normalized wavenumber (dimensionless).
    L_UMR : float
        Total UMR length [mm].
    R_cyl : float
        UMR cylinder radius [mm].
    epsilon : float
        Modulation amplitude.
    N : int
        Number of starts (lobes in cross-section).
    n_theta : int
        Grid points in θ direction (circumferential).
    n_zeta : int
        Grid points in ζ direction (axial).

    Returns
    -------
    SurfaceMesh
        points (M, 3), normals (M, 3), weights (M,) in mm.
        M ≈ 2 × n_theta × (n_zeta - 1).
    """
    # Parameter grids
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0, L_UMR, n_zeta)
    TH, ZE = np.meshgrid(theta, zeta, indexing='ij')  # (n_theta, n_zeta)

    # Modulated radius
    rho = R_cyl * (1.0 + epsilon * np.sin(N * TH))

    # Helical angle
    alpha = nu * ZE / R_cyl + TH

    # Surface positions: (n_theta, n_zeta, 3)
    X = rho * np.cos(alpha)
    Y = rho * np.sin(alpha)
    Z = ZE
    verts = np.stack([X, Y, Z], axis=-1)  # (n_theta, n_zeta, 3)

    # Analytical partial derivatives for normals
    drho_dtheta = R_cyl * epsilon * N * np.cos(N * TH)
    dalpha_dtheta = 1.0  # ∂α/∂θ = 1
    dalpha_dzeta = nu / R_cyl  # ∂α/∂ζ = ν̃/R_cyl

    # ∂r/∂θ
    dr_dth_x = drho_dtheta * np.cos(alpha) - rho * np.sin(alpha) * dalpha_dtheta
    dr_dth_y = drho_dtheta * np.sin(alpha) + rho * np.cos(alpha) * dalpha_dtheta
    dr_dth_z = np.zeros_like(TH)
    dr_dth = np.stack([dr_dth_x, dr_dth_y, dr_dth_z], axis=-1)

    # ∂r/∂ζ
    dr_dze_x = -rho * np.sin(alpha) * dalpha_dzeta
    dr_dze_y = +rho * np.cos(alpha) * dalpha_dzeta
    dr_dze_z = np.ones_like(TH)
    dr_dze = np.stack([dr_dze_x, dr_dze_y, dr_dze_z], axis=-1)

    # Build triangles from grid quads
    # Each quad (i, j)-(i+1, j)-(i+1, j+1)-(i, j+1) → 2 triangles
    # θ wraps: index (n_theta) → 0
    centroids = []
    normals = []
    areas = []

    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        for j in range(n_zeta - 1):
            # Four corners of the quad
            v00 = verts[i, j]
            v10 = verts[i_next, j]
            v11 = verts[i_next, j + 1]
            v01 = verts[i, j + 1]

            # Triangle 1: v00, v10, v11
            c1 = (v00 + v10 + v11) / 3.0
            e1a = v10 - v00
            e1b = v11 - v00
            n1 = np.cross(e1a, e1b)
            a1 = np.linalg.norm(n1) / 2.0

            # Triangle 2: v00, v11, v01
            c2 = (v00 + v11 + v01) / 3.0
            e2a = v11 - v00
            e2b = v01 - v00
            n2 = np.cross(e2a, e2b)
            a2 = np.linalg.norm(n2) / 2.0

            if a1 > 1e-30:
                centroids.append(c1)
                normals.append(n1 / (2.0 * a1))  # unit normal
                areas.append(a1)
            if a2 > 1e-30:
                centroids.append(c2)
                normals.append(n2 / (2.0 * a2))
                areas.append(a2)

    centroids = np.array(centroids, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)
    areas = np.array(areas, dtype=np.float64)

    # Ensure outward normals: the centroid vector from the axis should
    # point outward. Check dot product with radial direction.
    radial = centroids.copy()
    radial[:, 2] = 0.0  # project to xy plane
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.maximum(radial_norm, 1e-30)
    dots = np.sum(normals * radial, axis=1)
    flip_mask = dots < 0
    normals[flip_mask] *= -1.0

    return SurfaceMesh(
        points=centroids,
        normals=normals,
        weights=areas,
    )


def dejongh_fl_mesh(design_id: int, **kwargs) -> SurfaceMesh:
    """Generate mesh for an FL-group UMR design.

    Parameters
    ----------
    design_id : int
        FL design number (1–11, from Table 1).
    **kwargs
        Override n_theta, n_zeta, etc.

    Returns
    -------
    SurfaceMesh
    """
    if design_id not in FL_TABLE:
        raise ValueError(f"FL-{design_id} not in Table 1. Valid: 1-11")
    nu = FL_TABLE[design_id]["nu"]
    return dejongh_umr_surface(nu, FL_L_UMR, **kwargs)


def dejongh_fw_mesh(design_id: int, **kwargs) -> SurfaceMesh:
    """Generate mesh for an FW-group UMR design.

    Parameters
    ----------
    design_id : int
        FW design number (1–6, from Table 2).
    **kwargs
        Override n_theta, n_zeta, etc.

    Returns
    -------
    SurfaceMesh
    """
    if design_id not in FW_TABLE:
        raise ValueError(f"FW-{design_id} not in Table 2. Valid: 1-6")
    params = FW_TABLE[design_id]
    return dejongh_umr_surface(params["nu"], params["L_UMR"], **kwargs)
