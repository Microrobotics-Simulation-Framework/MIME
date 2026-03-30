"""Surface mesh dataclass and generators for BEM quadrature.

Provides SurfaceMesh (collocation points + normals + quadrature weights)
and factory functions for standard geometries (sphere, cylinder).

These are BEM-quality meshes for numerical integration, NOT visualization
meshes. The key requirements are: uniform point spacing, accurate normals,
and quadrature weights that sum to the correct surface area.

Reference for sphere discretization:
    Nguyen et al. (2025), Phys. Rev. Fluids 10:033101, Section III.
    Local PDF: tmp/stokelet_403_papers/PhysRevFluids.10.033101.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceMesh:
    """Quadrature-quality surface discretization for BEM.

    Attributes
    ----------
    points : (N, 3) float64
        Collocation/quadrature points on the surface.
    normals : (N, 3) float64
        Outward unit normals at each point.
    weights : (N,) float64
        Quadrature weights (surface area elements).
    """
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def total_area(self) -> float:
        return float(np.sum(self.weights))

    @property
    def mean_spacing(self) -> float:
        """Average inter-point spacing, estimated from area per point."""
        return float(np.sqrt(self.total_area / self.n_points))


def sphere_surface_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    n_refine: int = 3,
) -> SurfaceMesh:
    """Generate sphere surface mesh via icosahedral subdivision.

    Recursively subdivides an icosahedron and projects vertices onto
    the sphere. Produces near-uniform triangles with analytically
    computable areas.

    Parameters
    ----------
    center : (3,) float
    radius : float
    n_refine : int
        Number of subdivision levels. N_points ≈ 10 * 4^n_refine + 2.
        n_refine=2 → 162 points, n_refine=3 → 642, n_refine=4 → 2562.

    Returns
    -------
    SurfaceMesh
    """
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])  # normalize to unit sphere

    # Icosahedron faces (20 triangles)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    # Subdivide
    for _ in range(n_refine):
        verts, faces = _subdivide_icosphere(verts, faces)

    # Project onto sphere and scale
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius

    # Compute triangle centroids, areas, and normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)

    # Shift to center
    center_arr = np.array(center, dtype=np.float64)
    centroids = centroids + center_arr

    return SurfaceMesh(
        points=centroids,
        normals=normals,
        weights=areas,
    )


def _subdivide_icosphere(
    verts: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One level of icosphere subdivision."""
    edge_midpoints = {}
    new_verts = list(verts)

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)  # project onto unit sphere
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    new_faces = []
    for tri in faces:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c, ca, bc],
            [ab, bc, ca],
        ])

    return np.array(new_verts), np.array(new_faces, dtype=np.int32)


def cylinder_surface_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    length: float = 2.0,
    axis: int = 2,
    n_circ: int = 32,
    n_axial: int = 20,
) -> SurfaceMesh:
    """Generate cylinder inner wall surface mesh for BEM.

    The cylinder is centered at `center` with the given radius and
    length along the specified axis. Normals point INWARD (toward
    the cylinder axis) for interior flow problems.

    Parameters
    ----------
    center : (3,)
    radius : float
    length : float
    axis : int
        0=X, 1=Y, 2=Z.
    n_circ : int
        Circumferential point count.
    n_axial : int
        Axial point count.

    Returns
    -------
    SurfaceMesh
    """
    # Generate on a Z-axis cylinder, then rotate if needed
    thetas = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    dtheta = 2 * np.pi / n_circ
    zs = np.linspace(-length / 2, length / 2, n_axial)
    dz = length / max(n_axial - 1, 1)

    points = []
    normals = []
    weights = []

    for z in zs:
        for theta in thetas:
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
            # Inward normal (toward axis)
            normals.append([-np.cos(theta), -np.sin(theta), 0.0])
            # Panel area: dtheta * radius * dz
            weights.append(dtheta * radius * dz)

    points = np.array(points, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    # Rotate if axis != 2
    if axis != 2:
        # Permute axes: Z-cylinder → desired axis
        perm = [0, 1, 2]
        perm[2], perm[axis] = perm[axis], perm[2]
        points = points[:, perm]
        normals = normals[:, perm]

    # Shift to center
    center_arr = np.array(center, dtype=np.float64)
    points = points + center_arr

    return SurfaceMesh(points=points, normals=normals, weights=weights)
