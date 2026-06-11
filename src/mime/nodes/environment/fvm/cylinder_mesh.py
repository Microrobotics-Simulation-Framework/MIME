"""Procedural body-fitted cylindrical-vessel tet mesh (B2).

Generates a conforming tetrahedral mesh of a solid cylinder (a vessel segment)
directly as points + tet connectivity, with no Gmsh dependency — the cylinder
wall is body-fitted (cells align with the curved boundary, faceted to the
azimuthal resolution). Feeds :func:`build_face_graph`; the inlet (``z=0``),
outlet (``z=L``) and ``wall`` (``r=R``) patches are tagged.

The cross-section is a centre-fan + concentric-ring triangulation; it is
extruded in ``z`` into triangular prisms, each split into 3 tetrahedra by a
**global-index conforming rule** (the diagonal of every lateral quad is chosen
through its lower-indexed bottom vertex, so adjacent prisms agree on the shared
face → a watertight, divergence-closed mesh).
"""

from __future__ import annotations

import numpy as np

from mime.nodes.environment.fvm.unstructured import build_face_graph


def _prism_to_tets(b0, b1, b2, npl):
    """Split a triangular prism into 3 conforming tets.

    ``b0,b1,b2`` are the bottom-layer global vertex indices; the top-layer
    vertices are ``bi + npl``. The split rotates the smallest bottom index to
    the front and picks the lateral-quad diagonals through the lower-indexed
    vertex — a rule that depends only on shared-vertex indices, so neighbouring
    prisms produce a matching diagonal on the shared quad (conformity).
    """
    b = (b0, b1, b2)
    amin = int(np.argmin(b))
    a, p, q = b[amin], b[(amin + 1) % 3], b[(amin + 2) % 3]
    A, P, Q = a + npl, p + npl, q + npl
    if p < q:
        return [(a, p, q, Q), (a, p, Q, P), (a, P, Q, A)]
    return [(a, q, p, P), (a, q, P, Q), (a, Q, P, A)]


def make_cylinder_tet_mesh(
    radius: float,
    length: float,
    *,
    n_radial: int = 3,
    n_theta: int = 12,
    n_axial: int = 6,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: int = 2,
    dtype=np.float64,
):
    """Body-fitted tet mesh of a solid cylinder of ``radius`` × ``length``.

    Parameters
    ----------
    radius, length : float
        Vessel radius and axial length (m).
    n_radial, n_theta, n_axial : int
        Ring count (radial resolution), azimuthal sectors, axial layers.
    origin : (3,) float
        Position of the inlet-face centre.
    axis : int
        Axial direction (0/1/2 → x/y/z). Default 2 (z).

    Returns
    -------
    FVMMesh with patches ``"inlet"`` (axial = 0), ``"outlet"`` (axial = length),
    ``"wall"`` (r = radius).
    """
    oz = origin[axis]            # axial origin
    # in-plane axes (the two that aren't `axis`)
    planar = [i for i in range(3) if i != axis]

    # ---- cross-section nodes: centre + concentric rings ----
    npl = 1 + n_radial * n_theta             # nodes per layer
    cross = np.zeros((npl, 2))               # (planar coords) at one layer
    idx = 1
    for j in range(1, n_radial + 1):
        r = radius * j / n_radial
        for k in range(n_theta):
            th = 2 * np.pi * k / n_theta + 0.5 * np.pi * j / n_radial  # ring stagger
            cross[idx] = (r * np.cos(th), r * np.sin(th))
            idx += 1

    def ring(j, k):                          # global in-layer index of ring node
        return 1 + (j - 1) * n_theta + (k % n_theta)

    # ---- cross-section triangulation ----
    tris = []
    for k in range(n_theta):                 # centre fan to ring 1
        tris.append((0, ring(1, k), ring(1, k + 1)))
    for j in range(1, n_radial):             # ring gaps → quad → 2 tris
        for k in range(n_theta):
            a, b = ring(j, k), ring(j, k + 1)
            c, d = ring(j + 1, k + 1), ring(j + 1, k)
            tris.append((a, b, c))
            tris.append((a, c, d))

    # ---- nodes for all axial layers ----
    n_layers = n_axial + 1
    pts = np.zeros((npl * n_layers, 3))
    zs = np.linspace(0.0, length, n_layers)
    org = np.asarray(origin, dtype=float)
    for i, z in enumerate(zs):
        layer = np.zeros((npl, 3))
        layer[:, planar[0]] = cross[:, 0]
        layer[:, planar[1]] = cross[:, 1]
        layer[:, axis] = z
        pts[i * npl:(i + 1) * npl] = layer + org      # shift by inlet centre

    # ---- extrude prisms → tets ----
    cells = []
    for i in range(n_axial):
        off = i * npl
        for (t0, t1, t2) in tris:
            cells.extend(_prism_to_tets(off + t0, off + t1, off + t2, npl))
    cells = np.array(cells, dtype=int)

    # ---- boundary face tags by vertex predicates ----
    r_planar = np.sqrt(
        (pts[:, planar[0]] - origin[planar[0]]) ** 2
        + (pts[:, planar[1]] - origin[planar[1]]) ** 2)
    ax = pts[:, axis]
    tol_r = 1e-9 + 1e-6 * radius
    tol_z = 1e-9 + 1e-6 * length
    on_inlet = ax < oz + tol_z
    on_outlet = ax > oz + length - tol_z
    on_wall = r_planar > radius - tol_r

    tags: dict[frozenset, str] = {}
    face_local = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for c in cells:
        for fl in face_local:
            vs = (int(c[fl[0]]), int(c[fl[1]]), int(c[fl[2]]))
            key = frozenset(vs)
            if all(on_inlet[v] for v in vs):
                tags[key] = "inlet"
            elif all(on_outlet[v] for v in vs):
                tags[key] = "outlet"
            elif all(on_wall[v] for v in vs):
                tags[key] = "wall"

    return build_face_graph(pts, cells, dim=3, boundary_face_tags=tags,
                            dtype=dtype)
