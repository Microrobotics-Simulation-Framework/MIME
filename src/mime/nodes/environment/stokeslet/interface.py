"""Interface mesh utilities for Schwarz domain decomposition coupling.

Creates the shared interface surface mesh between BEM (near-field) and
LBM (far-field) solvers, plus interpolation maps for transferring
velocity data between the two discretizations.

The interface sphere serves triple duty:
1. BEM outer boundary surface (prescribed velocity from LBM)
2. LBM inner boundary surface (Bouzidi IBB with prescribed velocity from BEM)
3. Velocity evaluation/transfer grid between solvers
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .surface_mesh import SurfaceMesh, sphere_surface_mesh


def create_interface_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    n_refine: int = 3,
) -> SurfaceMesh:
    """Create a spherical interface mesh for Schwarz coupling.

    Reuses the icosahedral sphere mesh generator. The interface mesh
    points serve as both BEM boundary collocation points and
    LBM velocity evaluation/BC points.

    Parameters
    ----------
    center : (3,) sphere center
    radius : float, interface sphere radius
    n_refine : int, icosahedral subdivision level
        n_refine=2 → 320 points, n_refine=3 → 1280 points

    Returns
    -------
    SurfaceMesh with normals pointing INWARD (toward the body).
        Inward normals are used because the BEM fluid domain is
        between the body (outward normals) and the interface sphere
        (inward normals).
    """
    mesh = sphere_surface_mesh(center=center, radius=radius, n_refine=n_refine)
    # Flip normals inward — the BEM fluid is INSIDE the interface sphere
    return SurfaceMesh(
        points=mesh.points,
        normals=-mesh.normals,
        weights=mesh.weights,
    )


def interface_to_lattice_map(
    interface_points: np.ndarray,
    lattice_shape: tuple[int, int, int],
    dx: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map interface mesh points to lattice nodes for trilinear interpolation.

    For each interface point, find the 8 surrounding lattice nodes and
    compute trilinear interpolation weights. Used by the LBM solver to
    evaluate velocity at interface points (for sending to BEM as
    background flow).

    Parameters
    ----------
    interface_points : (N, 3) interface surface points in physical coords
    lattice_shape : (nx, ny, nz) lattice dimensions
    dx : float, lattice spacing in physical units
    origin : (3,) physical coordinates of lattice node (0,0,0)

    Returns
    -------
    indices : (N, 8) int — flat indices into (nx, ny, nz) lattice
    weights : (N, 8) float — trilinear interpolation weights
    """
    N = len(interface_points)
    nx, ny, nz = lattice_shape

    # Convert to fractional lattice coordinates
    frac = (interface_points - origin) / dx  # (N, 3)

    # Integer base indices (floor)
    i0 = np.floor(frac).astype(int)
    # Fractional part
    f = frac - i0  # (N, 3)

    # Clamp to valid range
    i0 = np.clip(i0, 0, np.array([nx - 2, ny - 2, nz - 2]))
    f = np.clip(f, 0.0, 1.0)

    # 8 corner offsets: (0,0,0), (1,0,0), (0,1,0), ...
    offsets = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ])

    indices = np.zeros((N, 8), dtype=int)
    weights_out = np.zeros((N, 8), dtype=np.float64)

    for c, (di, dj, dk) in enumerate(offsets):
        ii = i0[:, 0] + di
        jj = i0[:, 1] + dj
        kk = i0[:, 2] + dk
        indices[:, c] = ii * ny * nz + jj * nz + kk

        # Trilinear weight
        wx = f[:, 0] if di else (1.0 - f[:, 0])
        wy = f[:, 1] if dj else (1.0 - f[:, 1])
        wz = f[:, 2] if dk else (1.0 - f[:, 2])
        weights_out[:, c] = wx * wy * wz

    return indices, weights_out


def lattice_to_interface_map(
    lattice_shape: tuple[int, int, int],
    dx: float,
    origin: np.ndarray,
    interface_points: np.ndarray,
) -> np.ndarray:
    """Map lattice boundary nodes to nearest interface mesh points.

    For each lattice node near the interface sphere, find the nearest
    interface surface point. Used by the LBM to spread interface
    velocity as Bouzidi BC.

    Parameters
    ----------
    lattice_shape : (nx, ny, nz)
    dx : float, lattice spacing
    origin : (3,) physical coordinates of lattice node (0,0,0)
    interface_points : (N, 3) interface surface points

    Returns
    -------
    nearest_indices : (nx*ny*nz,) int
        For each lattice node, index of nearest interface point.
        Only meaningful for nodes near the interface sphere.
    """
    tree = cKDTree(interface_points)

    nx, ny, nz = lattice_shape
    # Generate all lattice node positions
    ix, iy, iz = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij',
    )
    lattice_pts = np.stack([
        origin[0] + ix.ravel() * dx,
        origin[1] + iy.ravel() * dx,
        origin[2] + iz.ravel() * dx,
    ], axis=1)

    _, nearest_indices = tree.query(lattice_pts)
    return nearest_indices


def interpolate_lattice_to_interface(
    velocity_field: np.ndarray,
    interp_indices: np.ndarray,
    interp_weights: np.ndarray,
) -> np.ndarray:
    """Interpolate lattice velocity field at interface points.

    Parameters
    ----------
    velocity_field : (nx, ny, nz, 3) or (nx*ny*nz, 3)
    interp_indices : (N_interface, 8) from interface_to_lattice_map
    interp_weights : (N_interface, 8) from interface_to_lattice_map

    Returns
    -------
    velocity_at_interface : (N_interface, 3)
    """
    vel_flat = velocity_field.reshape(-1, 3)
    N = len(interp_indices)

    result = np.zeros((N, 3), dtype=np.float64)
    for c in range(8):
        idx = interp_indices[:, c]
        w = interp_weights[:, c:c + 1]  # (N, 1) for broadcasting
        result += w * vel_flat[idx]

    return result
