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


def sdf_surface_mesh(
    sdf_func,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    mc_resolution: int = 64,
) -> SurfaceMesh:
    """Generate BEM surface mesh from a signed distance function.

    Evaluates the SDF on a regular grid, extracts the isosurface
    via marching cubes, then computes centroid quadrature
    (triangle centroids as collocation points, areas as weights).

    Parameters
    ----------
    sdf_func : callable
        (N, 3) -> (N,) signed distance function. Negative inside.
    bbox_min, bbox_max : (3,)
        Bounding box for the SDF evaluation grid.
    mc_resolution : int
        Grid resolution for marching cubes.

    Returns
    -------
    SurfaceMesh
    """
    from skimage.measure import marching_cubes

    # Build evaluation grid
    xs = np.linspace(bbox_min[0], bbox_max[0], mc_resolution)
    ys = np.linspace(bbox_min[1], bbox_max[1], mc_resolution)
    zs = np.linspace(bbox_min[2], bbox_max[2], mc_resolution)
    dx = (bbox_max[0] - bbox_min[0]) / (mc_resolution - 1)
    dy = (bbox_max[1] - bbox_min[1]) / (mc_resolution - 1)
    dz = (bbox_max[2] - bbox_min[2]) / (mc_resolution - 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    points_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Evaluate SDF (may use JAX — convert to writable numpy)
    import jax.numpy as jnp
    sdf_values = np.array(sdf_func(jnp.array(points_grid)))  # np.array makes writable copy
    sdf_3d = sdf_values.reshape(mc_resolution, mc_resolution, mc_resolution)

    # Marching cubes with per-axis spacing
    verts_mc, faces_mc, normals_mc, _ = marching_cubes(
        sdf_3d, level=0.0, spacing=(dx, dy, dz),
    )

    # Shift vertices to world coordinates
    verts_mc = verts_mc + np.array(bbox_min)

    # Compute triangle centroids, areas, normals
    v0 = verts_mc[faces_mc[:, 0]]
    v1 = verts_mc[faces_mc[:, 1]]
    v2 = verts_mc[faces_mc[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) / 2.0

    # Use marching cubes normals at centroids (average vertex normals)
    n0 = normals_mc[faces_mc[:, 0]]
    n1 = normals_mc[faces_mc[:, 1]]
    n2 = normals_mc[faces_mc[:, 2]]
    tri_normals = (n0 + n1 + n2) / 3.0
    tri_normals = tri_normals / (np.linalg.norm(tri_normals, axis=1, keepdims=True) + 1e-30)

    # Flip normals outward (SDF gradient points outward = positive)
    # marching_cubes normals should already point outward

    return SurfaceMesh(
        points=centroids.astype(np.float64),
        normals=tri_normals.astype(np.float64),
        weights=areas.astype(np.float64),
    )


def cylinder_surface_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    length: float = 2.0,
    axis: int = 2,
    n_circ: int = 32,
    n_axial: int = 20,
    cluster_center: bool = False,
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
    cluster_center : bool
        If True, cluster more points near z=0 (where the body is)
        using a tanh distribution. Improves accuracy for confined
        BEM where the body is at the cylinder center.

    Returns
    -------
    SurfaceMesh
    """
    # Generate on a Z-axis cylinder, then rotate if needed
    thetas = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    dtheta = 2 * np.pi / n_circ

    if cluster_center:
        # Tanh clustering: denser near z=0, sparser at ends
        beta = 2.0  # clustering strength
        s = np.linspace(-1, 1, n_axial)
        zs = (length / 2) * np.tanh(beta * s) / np.tanh(beta)
    else:
        zs = np.linspace(-length / 2, length / 2, n_axial)

    # Compute per-row dz for non-uniform spacing
    dz_arr = np.zeros(n_axial)
    for i in range(n_axial):
        if i == 0:
            dz_arr[i] = (zs[1] - zs[0]) if n_axial > 1 else length
        elif i == n_axial - 1:
            dz_arr[i] = zs[-1] - zs[-2]
        else:
            dz_arr[i] = (zs[i + 1] - zs[i - 1]) / 2.0

    points = []
    normals = []
    weights = []

    for i, z in enumerate(zs):
        for theta in thetas:
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
            normals.append([-np.cos(theta), -np.sin(theta), 0.0])
            weights.append(dtheta * radius * dz_arr[i])

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
