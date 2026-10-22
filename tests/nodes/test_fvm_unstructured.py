"""Unstructured FVM mesh builder (H1/H2) — face-graph from simplices.

Pins the geometric correctness of ``build_face_graph``: cell volumes, the
owner/neighbour graph, face normals oriented owner→neighbour, outward boundary
normals, and the divergence-closure invariant (sum of outward face-area
vectors over any closed cell is zero) — the property the FVM operators rely on.
"""

from __future__ import annotations

import numpy as np
import pytest

from mime.nodes.environment.fvm.unstructured import build_face_graph


# ── helpers ──────────────────────────────────────────────────────────────

def _cube_into_6_tets():
    """Unit cube [0,1]^3 split into 6 tets around the 0–6 diagonal."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    cells = np.array([
        [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
        [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
    ], dtype=int)
    return pts, cells


def _outward_face_sum_per_cell(mesh):
    """Sum of outward face-area vectors per cell (should be ~0 for closed
    cells, by the divergence theorem)."""
    N = mesh.N_cells
    dim = mesh.dim
    acc = np.zeros((N, dim))
    owner = np.asarray(mesh.owner)
    nei = np.asarray(mesh.neighbour)
    Sf = np.asarray(mesh.Sf)
    for f in range(mesh.N_faces):
        acc[owner[f]] += Sf[f]    # Sf points owner -> neighbour (outward of owner)
        acc[nei[f]] -= Sf[f]      # ... and inward of neighbour
    for p in mesh.patches:
        po = np.asarray(p.owner)
        pSf = np.asarray(p.Sf)
        for k in range(po.size):
            acc[po[k]] += pSf[k]  # boundary Sf already outward
    return acc


# ── single / paired simplices ────────────────────────────────────────────

def test_single_tet_volume_and_boundary():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    m = build_face_graph(pts, np.array([[0, 1, 2, 3]]), dim=3)
    assert m.N_cells == 1
    assert m.N_faces == 0  # no interior faces
    assert np.isclose(float(m.V[0]), 1.0 / 6.0)
    assert sum(int(p.owner.size) for p in m.patches) == 4
    # closed-cell divergence closure
    assert np.allclose(_outward_face_sum_per_cell(m), 0.0, atol=1e-6)


def test_two_tets_share_one_face_oriented():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], float)
    m = build_face_graph(pts, np.array([[0, 1, 2, 3], [0, 1, 2, 4]]), dim=3)
    assert m.N_faces == 1
    assert int(m.owner[0]) < int(m.neighbour[0])  # owner < neighbour convention
    # normal points owner -> neighbour
    assert float(np.dot(np.asarray(m.n[0]), np.asarray(m.d[0]))) > 0
    assert 0.0 <= float(m.w[0]) <= 1.0
    # Sf magnitude == area; n is unit
    assert np.isclose(float(np.linalg.norm(m.Sf[0])), float(m.area[0]))
    assert np.isclose(float(np.linalg.norm(m.n[0])), 1.0)


# ── full closed mesh (the strong test) ───────────────────────────────────

def test_cube_6_tets_volume_closure_and_surface():
    pts, cells = _cube_into_6_tets()
    m = build_face_graph(pts, cells, dim=3)
    assert m.N_cells == 6
    # total volume = 1
    assert np.isclose(float(np.sum(m.V)), 1.0, atol=1e-6)
    # every cell is closed
    assert np.allclose(_outward_face_sum_per_cell(m), 0.0, atol=1e-6)
    # total boundary area = cube surface area = 6
    tot_bnd = sum(float(np.sum(p.area)) for p in m.patches)
    assert np.isclose(tot_bnd, 6.0, atol=1e-6)
    # owner/neighbour indices valid
    assert int(m.owner.min()) >= 0 and int(m.neighbour.max()) < m.N_cells


def test_cube_interior_faces_each_shared_by_two_cells():
    # If every interior face has exactly two cells, the owner/neighbour arrays
    # have no duplicates of the same unordered pair beyond the shared faces.
    pts, cells = _cube_into_6_tets()
    m = build_face_graph(pts, cells, dim=3)
    # 6 tets * 4 faces = 24 face-incidences; interior faces counted twice.
    # boundary faces counted once. So 2*N_interior + N_boundary = 24.
    n_bnd = sum(int(p.owner.size) for p in m.patches)
    assert 2 * m.N_faces + n_bnd == 24


# ── 2D ───────────────────────────────────────────────────────────────────

def test_two_triangles_share_an_edge():
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    m = build_face_graph(pts, np.array([[0, 1, 2], [1, 3, 2]]), dim=2)
    assert m.dim == 2
    assert m.N_faces == 1           # the shared edge (1,2)
    assert np.isclose(float(np.sum(m.V)), 1.0)  # two right triangles = unit square
    assert np.allclose(_outward_face_sum_per_cell(m), 0.0, atol=1e-6)


# ── boundary tags ─────────────────────────────────────────────────────────

def test_boundary_face_tags_name_patches():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    # tag the face opposite vertex 3 (the base z=0 face, verts {0,1,2}) "inlet"
    tags = {frozenset({0, 1, 2}): "inlet"}
    m = build_face_graph(pts, np.array([[0, 1, 2, 3]]), dim=3,
                         boundary_face_tags=tags)
    names = {p.name for p in m.patches}
    assert "inlet" in names and "boundary" in names


# ── validation ───────────────────────────────────────────────────────────

def test_wrong_cell_shape_raises():
    pts = np.zeros((4, 3))
    with pytest.raises(ValueError, match="must have 4 vertices"):
        build_face_graph(pts, np.array([[0, 1, 2]]), dim=3)


def test_bad_dim_raises():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        build_face_graph(np.zeros((3, 3)), np.array([[0, 1, 2]]), dim=4)


def test_non_manifold_face_raises():
    # three tets all sharing the face {0,1,2} → non-manifold
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [0, 0, 1], [0, 0, -1], [1, 1, 1]], float)
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]])
    with pytest.raises(ValueError, match="non-manifold"):
        build_face_graph(pts, cells, dim=3)


# ── read_gmsh optional-dep behaviour ─────────────────────────────────────

def test_read_gmsh_without_meshio_gives_clear_error():
    try:
        import meshio  # noqa: F401
        pytest.skip("meshio installed; the ImportError path can't be exercised")
    except ImportError:
        from mime.nodes.environment.fvm.unstructured import read_gmsh
        with pytest.raises(ImportError, match="mime-engine\\[mesh\\]"):
            read_gmsh("nonexistent.msh")


def test_read_gmsh_round_trips_a_tet_mesh(tmp_path):
    meshio = pytest.importorskip("meshio")
    from mime.nodes.environment.fvm.unstructured import read_gmsh

    pts, cells = _cube_into_6_tets()
    path = str(tmp_path / "cube.msh")
    meshio.write(
        path,
        meshio.Mesh(points=pts, cells=[("tetra", cells)]),
        file_format="gmsh",
    )
    m = read_gmsh(path)
    assert m.N_cells == 6
    assert np.isclose(float(np.sum(m.V)), 1.0, atol=1e-6)
    assert np.allclose(_outward_face_sum_per_cell(m), 0.0, atol=1e-6)
