"""Unstructured-mesh face-graph construction for the FVM node.

H1/H2 (Against-the-Current): the pilot FVM is Cartesian-only (``mesh.py``
builds structured grids). A body-fitted / unstructured solver needs the same
``FVMMesh`` face-graph built from a *simplicial* mesh (triangles in 2D,
tetrahedra in 3D) — e.g. a Gmsh ``.msh`` of a vessel + helix.

The face-graph *operators* (PISO, gradients, fluxes) are already mesh-agnostic;
only the *builder* was missing. This module derives the owner/neighbour graph
and the face/cell geometry from cell connectivity, matching the conventions
``make_cartesian_mesh_3d`` establishes exactly:

* ``Sf = area · n`` with ``n`` the unit face normal oriented **owner → neighbour**
  (``n · (x_neighbour − x_owner) > 0``);
* ``d = x_neighbour − x_owner``; ``d_mag = |d|``;
* ``w`` the linear-interpolation weight for the *owner* value
  (``phi_f = w·phi_O + (1−w)·phi_N``), via the face-normal projection so a
  uniform Cartesian split gives exactly ``0.5``;
* boundary faces carry an **outward** normal and group into named patches.

``build_face_graph`` is pure NumPy and the unit of test. ``read_gmsh`` is a thin
lazy-``meshio`` wrapper (an *optional* dependency: ``pip install mime-engine[mesh]``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import jax.numpy as jnp

from mime.nodes.environment.fvm.mesh import BoundaryPatch, FVMMesh

# Local face templates for a simplex cell: which local vertex indices form each
# face. A triangle (2D) has 3 edge-faces; a tetrahedron (3D) has 4 tri-faces.
_FACE_TEMPLATES = {
    2: ((0, 1), (1, 2), (2, 0)),
    3: ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
}
_CELL_VERTS = {2: 3, 3: 4}


def _cell_centroids(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return points[cells].mean(axis=1)


def _cell_volumes(points: np.ndarray, cells: np.ndarray, dim: int) -> np.ndarray:
    v = points[cells]  # [N_cells, dim+1, dim]
    e = v[:, 1:, :] - v[:, :1, :]  # edges from vertex 0, [N, dim, dim]
    if dim == 2:
        # triangle area = 0.5 |e0 x e1|
        cross = e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]
        return 0.5 * np.abs(cross)
    # tetra volume = |det(e0, e1, e2)| / 6
    return np.abs(np.linalg.det(e)) / 6.0


def _face_area_normal(face_pts: np.ndarray, dim: int):
    """Return (area, unit_normal) for one face given its vertex coordinates.
    The normal sign is arbitrary here; the caller orients it."""
    if dim == 2:
        p0, p1 = face_pts[0], face_pts[1]
        e = p1 - p0
        length = float(np.linalg.norm(e))
        # in-plane perpendicular
        n = np.array([e[1], -e[0]], dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-300)
        return length, n
    p0, p1, p2 = face_pts[0], face_pts[1], face_pts[2]
    cr = np.cross(p1 - p0, p2 - p0)
    mag = float(np.linalg.norm(cr))
    area = 0.5 * mag
    n = cr / (mag + 1e-300)
    return area, n


def build_face_graph(
    points: np.ndarray,
    cells: np.ndarray,
    *,
    dim: int,
    boundary_face_tags: Optional[dict[frozenset, str]] = None,
    dtype=jnp.float32,
) -> FVMMesh:
    """Build an ``FVMMesh`` from a simplicial mesh.

    Parameters
    ----------
    points : (N_points, dim) float
        Vertex coordinates.
    cells : (N_cells, dim+1) int
        Simplex connectivity (triangles for ``dim==2``, tetrahedra for
        ``dim==3``), each row the global vertex indices of one cell.
    dim : int
        2 or 3.
    boundary_face_tags : dict[frozenset[int], str], optional
        Maps a boundary face's global-vertex set to a patch name (from Gmsh
        physical groups). Untagged boundary faces go to a ``"boundary"`` patch.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    points = np.asarray(points, dtype=np.float64)[:, :dim]
    cells = np.asarray(cells, dtype=np.int64)
    if cells.shape[1] != _CELL_VERTS[dim]:
        raise ValueError(
            f"{dim}D cells must have {_CELL_VERTS[dim]} vertices, "
            f"got {cells.shape[1]}"
        )
    N_cells = cells.shape[0]
    x = _cell_centroids(points, cells)              # [N_cells, dim]
    V = _cell_volumes(points, cells, dim)           # [N_cells]

    # Map each face (by its sorted global-vertex key) to the cells touching it.
    face_to_cells: dict[tuple, list[tuple[int, tuple]]] = defaultdict(list)
    for c in range(N_cells):
        cell = cells[c]
        for local in _FACE_TEMPLATES[dim]:
            verts = tuple(int(cell[i]) for i in local)
            face_to_cells[tuple(sorted(verts))].append((c, verts))

    owner_l, nei_l, Sf_l, n_l, area_l, d_l = [], [], [], [], [], []
    w_l = []
    boundary: dict[str, dict[str, list]] = defaultdict(
        lambda: {"owner": [], "Sf": [], "n": [], "area": [], "d": [], "face_x": []}
    )
    tags = boundary_face_tags or {}

    for key, entries in face_to_cells.items():
        face_pts = points[list(key)]
        face_centroid = face_pts.mean(axis=0)
        area, n_raw = _face_area_normal(face_pts, dim)
        if len(entries) == 2:
            c0, c1 = entries[0][0], entries[1][0]
            owner, nei = (c0, c1) if c0 < c1 else (c1, c0)
            d = x[nei] - x[owner]
            n = n_raw if np.dot(n_raw, d) > 0 else -n_raw
            denom = np.dot(x[nei] - x[owner], n)
            w = float(np.dot(x[nei] - face_centroid, n) / denom) if denom != 0 else 0.5
            w = min(1.0, max(0.0, w))
            owner_l.append(owner); nei_l.append(nei)
            Sf_l.append(area * n); n_l.append(n); area_l.append(area)
            d_l.append(d); w_l.append(w)
        elif len(entries) == 1:
            c = entries[0][0]
            outward = face_centroid - x[c]
            n = n_raw if np.dot(n_raw, outward) > 0 else -n_raw
            name = tags.get(frozenset(key), "boundary")
            p = boundary[name]
            p["owner"].append(c); p["Sf"].append(area * n); p["n"].append(n)
            p["area"].append(area); p["d"].append(face_centroid - x[c])
            p["face_x"].append(face_centroid)
        else:
            raise ValueError(
                f"face {key} shared by {len(entries)} cells (non-manifold mesh)"
            )

    owner = np.asarray(owner_l, dtype=np.int32)
    neighbour = np.asarray(nei_l, dtype=np.int32)
    Sf = np.asarray(Sf_l, dtype=np.float64).reshape(-1, dim)
    n = np.asarray(n_l, dtype=np.float64).reshape(-1, dim)
    area = np.asarray(area_l, dtype=np.float64)
    d = np.asarray(d_l, dtype=np.float64).reshape(-1, dim)
    d_mag = np.linalg.norm(d, axis=1)
    w = np.asarray(w_l, dtype=np.float64)

    patches = tuple(
        BoundaryPatch(
            name=name,
            owner=jnp.asarray(p["owner"], dtype=jnp.int32),
            Sf=jnp.asarray(np.asarray(p["Sf"]).reshape(-1, dim), dtype=dtype),
            n=jnp.asarray(np.asarray(p["n"]).reshape(-1, dim), dtype=dtype),
            area=jnp.asarray(p["area"], dtype=dtype),
            d=jnp.asarray(np.asarray(p["d"]).reshape(-1, dim), dtype=dtype),
            face_x=jnp.asarray(np.asarray(p["face_x"]).reshape(-1, dim), dtype=dtype),
        )
        for name, p in sorted(boundary.items())
    )

    return FVMMesh(
        owner=jnp.asarray(owner, dtype=jnp.int32),
        neighbour=jnp.asarray(neighbour, dtype=jnp.int32),
        Sf=jnp.asarray(Sf, dtype=dtype),
        n=jnp.asarray(n, dtype=dtype),
        area=jnp.asarray(area, dtype=dtype),
        d=jnp.asarray(d, dtype=dtype),
        d_mag=jnp.asarray(d_mag, dtype=dtype),
        w=jnp.asarray(w, dtype=dtype),
        V=jnp.asarray(V, dtype=dtype),
        x=jnp.asarray(x, dtype=dtype),
        V_owner=jnp.asarray(V[owner], dtype=dtype),
        V_neighbour=jnp.asarray(V[neighbour], dtype=dtype),
        patches=patches,
        N_cells=int(N_cells),
        N_faces=int(owner.size),
        dim=dim,
        cartesian_shape=None,
        cartesian_spacing=None,
        cartesian_origin=None,
    )


def read_gmsh(path: str, *, dim: Optional[int] = None, dtype=jnp.float32) -> FVMMesh:
    """Read a Gmsh ``.msh`` file into an ``FVMMesh`` (optional ``meshio`` dep).

    Volume cells (``tetra`` for 3D, ``triangle`` for 2D) become the FVM cells;
    tagged boundary facets (``triangle`` / ``line`` blocks carrying Gmsh
    physical-group names) become named boundary patches.

    Requires ``meshio`` — install the optional extra ``pip install
    mime-engine[mesh]``.
    """
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "read_gmsh needs the optional 'meshio' dependency; install it with "
            "`pip install mime-engine[mesh]` (or `pip install meshio`)."
        ) from exc

    m = meshio.read(path)
    cells_dict = m.cells_dict
    if dim is None:
        dim = 3 if "tetra" in cells_dict else 2
    vol_type = "tetra" if dim == 3 else "triangle"
    facet_type = "triangle" if dim == 3 else "line"
    if vol_type not in cells_dict:
        raise ValueError(
            f"no {vol_type!r} cells in {path!r} (is it a {dim}D mesh?)"
        )
    cells = cells_dict[vol_type]

    # Boundary-facet physical-group names → per-facet vertex-set tags.
    tags: dict[frozenset, str] = {}
    names_by_id = {v[0]: k for k, v in getattr(m, "field_data", {}).items()}
    facet_blocks = [
        (i, cb) for i, cb in enumerate(m.cells) if cb.type == facet_type
    ]
    phys = m.cell_data.get("gmsh:physical") if m.cell_data else None
    for bi, cb in facet_blocks:
        ids = phys[bi] if phys is not None and bi < len(phys) else None
        for fi, facet in enumerate(cb.data):
            name = names_by_id.get(int(ids[fi])) if ids is not None else None
            if name:
                tags[frozenset(int(v) for v in facet)] = name

    return build_face_graph(
        m.points, cells, dim=dim,
        boundary_face_tags=tags or None, dtype=dtype,
    )
