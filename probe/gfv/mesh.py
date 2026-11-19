"""A face-based polyhedral mesh (the collocated graph's static arrays), its builders and its quality measures.

``own``/``nei``: the two cells of each face, the area vector ``S`` pointing owner to neighbour; boundary faces come last
and have ``nei == n`` (a ghost). Meshing is outside JAX: gmsh builds the tube's tetrahedra, numpy builds the rest."""
from dataclasses import dataclass, field

import numpy as np

LOCAL_FACES = {
    "tetra": [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
    "hexahedron": [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "wedge": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "pyramid": [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}


@dataclass
class FaceMesh:
    n: int
    own: np.ndarray
    nei: np.ndarray
    S: np.ndarray
    xf: np.ndarray
    xc: np.ndarray
    V: np.ndarray
    d: np.ndarray          # owner centre to neighbour centre (to the face centre on the boundary)
    w: np.ndarray          # the owner's weight in a linear interpolation to the face
    patch: np.ndarray      # per face: -1 inside, else a boundary patch id
    info: dict = field(default_factory=dict)

    @property
    def m(self):
        return self.own.size

    @property
    def inner(self):
        return self.nei < self.n

    def coefficients(self):
        """The over-relaxed split of each face: ``g = |S|^2 / (S.d)`` (the orthogonal coefficient) and ``k = S - g d``."""
        g = np.einsum("fi,fi->f", self.S, self.S) / np.einsum("fi,fi->f", self.S, self.d)
        return g, self.S - g[:, None] * self.d

    def quality(self):
        inner = self.inner
        area = np.linalg.norm(self.S, axis=1)
        dn = np.linalg.norm(self.d, axis=1)
        align = np.einsum("fi,fi->f", self.S, self.d) / (area * dn)
        # skewness: the face centre's distance from where the centre-to-centre line crosses the face, over |d|
        t = np.einsum("fi,fi->f", self.xf - self.xc[self.own], self.S) / np.einsum("fi,fi->f", self.d, self.S)
        cross = self.xc[self.own] + t[:, None] * self.d
        skew = np.linalg.norm(self.xf - cross, axis=1) / dn
        # aspect ratio: per cell, the largest over the smallest centre-to-face normal distance
        normal = np.abs(np.einsum("fi,fi->f", self.xf - self.xc[self.own], self.S)) / area
        hi, lo = np.zeros(self.n), np.full(self.n, np.inf)
        np.maximum.at(hi, self.own, normal), np.minimum.at(lo, self.own, normal)
        other = np.abs(np.einsum("fi,fi->f", (self.xf - self.xc[self.own] - self.d)[inner], self.S[inner])) / area[inner]
        np.maximum.at(hi, self.nei[inner], other), np.minimum.at(lo, self.nei[inner], other)
        aspect = hi / lo
        faces_a_cell = (self.m + int(inner.sum())) / self.n
        closed = np.zeros((self.n, 3))
        np.add.at(closed, self.own, self.S), np.add.at(closed, self.nei[inner], -self.S[inner])
        scale = np.zeros(self.n)
        np.add.at(scale, self.own, area), np.add.at(scale, self.nei[inner], area[inner])
        return {"cells": int(self.n), "faces": int(self.m), "boundary_faces": int((~inner).sum()),
                "faces_a_cell": float(faces_a_cell),
                "alignment_min": float(align[inner].min()), "alignment_p1": float(np.percentile(align[inner], 1)),
                "alignment_mean": float(align[inner].mean()),
                "skewness_max": float(skew[inner].max()), "skewness_p99": float(np.percentile(skew[inner], 99)),
                "aspect_max": float(aspect.max()), "aspect_p99": float(np.percentile(aspect, 99)),
                "volume": float(self.V.sum()), "closedness": float(np.abs(closed).max() / scale.max())}

    def reordered(self, perm):
        """The same mesh with cell ``perm[i]`` renumbered ``i``, faces sorted by owner then neighbour (inner first)."""
        inv = np.empty(self.n + 1, np.int64)
        inv[perm] = np.arange(self.n)
        inv[self.n] = self.n
        own, nei = inv[self.own], inv[self.nei]
        flip = self.inner & (nei < own)
        S, d, w = self.S.copy(), self.d.copy(), self.w.copy()
        own2, nei2 = np.where(flip, nei, own), np.where(flip, own, nei)
        S[flip], d[flip], w[flip] = -S[flip], -d[flip], 1.0 - w[flip]
        e = np.lexsort((nei2, own2, ~self.inner))
        return FaceMesh(self.n, own2[e], nei2[e], S[e], self.xf[e], self.xc[perm], self.V[perm], d[e], w[e],
                        self.patch[e], dict(self.info))


def finish(n, own, nei, S, xf, xc, V, patch, d=None, info=None):
    inner = nei < n
    if d is None:
        d = np.where(inner[:, None], xc[np.where(inner, nei, 0)] - xc[own], xf - xc[own])
    to_nei = np.einsum("fi,fi->f", S, np.where(inner[:, None], d - (xf - xc[own]), 0.0))
    w = np.where(inner, to_nei / np.einsum("fi,fi->f", S, d), 0.0)
    e = np.lexsort((nei, own, ~inner))
    return FaceMesh(n, own[e].astype(np.int64), nei[e].astype(np.int64), S[e], xf[e], xc, V, d[e], w[e], patch[e], info or {})


def cartesian(cells, h):
    """The periodic Cartesian box as a collocated face mesh (every cell six inner faces, none on a boundary)."""
    n = int(np.prod(cells))
    grid = np.arange(n).reshape(cells)
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in cells], indexing="ij"), -1).reshape(-1, 3).astype(float)
    xc = (idx + 0.5) * h
    own, nei, S, xf, d = [], [], [], [], []
    for a in range(3):
        e = np.eye(3)[a]
        own.append(np.roll(grid, 1, a).ravel()), nei.append(grid.ravel())
        S.append(np.tile(e * h * h, (n, 1))), xf.append(xc - 0.5 * h * e), d.append(np.tile(e * h, (n, 1)))
    own, nei, S, xf, d = (np.concatenate(x) for x in (own, nei, S, xf, d))
    mesh = FaceMesh(n, own, nei, S, xf, xc, np.full(n, h ** 3), d, np.full(own.size, 0.5), np.full(own.size, -1), {})
    return mesh


def from_cells(points, blocks, patch_of=None):
    """A face mesh from vertex-based cells (``blocks``: ``{type: (cells, vertices) array}``, VTK vertex order)."""
    points = np.asarray(points, float)
    keys, loops, owner, offset = [], [], [], 0
    centres = []
    for kind, conn in blocks.items():
        conn = np.asarray(conn)
        centres.append(points[conn].mean(axis=1))
        for local in LOCAL_FACES[kind]:
            loop = conn[:, local]
            if loop.shape[1] == 3:
                loop = np.concatenate([loop, loop[:, 2:3]], axis=1)
            loops.append(loop), owner.append(np.arange(len(conn)) + offset)
        offset += len(conn)
    loops, owner = np.concatenate(loops), np.concatenate(owner)
    n, centre0 = offset, np.concatenate(centres)
    key = np.sort(np.where((loops[:, 2] == loops[:, 3])[:, None] & (np.arange(4) == 3), -1, loops), axis=1)
    _, first, inverse, count = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    assert count.max() <= 2
    faces = len(first)
    order = np.argsort(inverse, kind="stable")
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    own = owner[order[start]]
    nei = np.where(count == 2, owner[order[np.minimum(start + 1, len(order) - 1)]], n)
    loop = loops[order[start]]
    S, xf = polygon_geometry(points[loop], loop[:, 2] == loop[:, 3])
    patch = np.where(nei < n, -1, 0)
    if patch_of is not None:
        patch = np.where(nei < n, -1, patch_of(xf))
    return assemble(n, own, nei, S, xf, centre0, patch)


def polygon_geometry(p, is_tri):
    """Area vectors and centroids of planar polygons ``p (faces, 4, 3)`` (a triangle's last vertex repeated)."""
    faces = len(p)
    c0 = np.where(is_tri[:, None], p[:, :3].mean(axis=1), p.mean(axis=1))
    S, xf_num, area_sum = np.zeros((faces, 3)), np.zeros((faces, 3)), np.zeros(faces)
    for k in range(4):
        a, b = p[:, k], p[:, (k + 1) % 4]
        tri = 0.5 * np.cross(a - c0, b - c0)
        S += tri
        area = np.linalg.norm(tri, axis=1)
        xf_num += area[:, None] * (c0 + a + b) / 3.0
        area_sum += area
    return S, xf_num / area_sum[:, None]


def assemble(n, own, nei, S, xf, centre0, patch, info=None):
    """Orient each face owner to neighbour, then the cells' volumes and centroids from pyramids on their faces."""
    S = S.copy()
    flip = np.einsum("fi,fi->f", S, xf - centre0[own]) < 0
    S[flip] *= -1
    inner = nei < n
    V, xc_num = np.zeros(n), np.zeros((n, 3))
    for cell, sign, sel in ((own, 1.0, slice(None)), (nei[inner], -1.0, inner)):
        vol = sign * np.einsum("fi,fi->f", S[sel], xf[sel] - centre0[cell]) / 3.0
        np.add.at(V, cell, vol)
        np.add.at(xc_num, cell, vol[:, None] * (0.75 * xf[sel] + 0.25 * centre0[cell]))
    return finish(n, own, nei, S, xf, xc_num / V[:, None], V, patch, info=info)


def gmsh_tube(radius, length, size, layers=0, algorithm3d=1, seed=1):
    """Tetrahedra in a cylinder along z (gmsh, outside JAX); with ``layers > 0`` the cross-section's triangles and wall
    quadrilateral layers are extruded instead (prisms and hexahedra). Returns ``(points, blocks, gmsh version)``."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.RandomSeed", seed)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.add("tube")
    gmsh.option.setNumber("Mesh.MeshSizeMin", size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    if layers == 0:
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length, radius)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm3d)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
    else:
        disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
        gmsh.model.occ.synchronize()
        curves = [c[1] for c in gmsh.model.getBoundary([(2, disk)])]
        f = gmsh.model.mesh.field.add("BoundaryLayer")
        gmsh.model.mesh.field.setNumbers(f, "CurvesList", curves)
        gmsh.model.mesh.field.setNumber(f, "Size", size / 4)
        gmsh.model.mesh.field.setNumber(f, "Ratio", 1.3)
        gmsh.model.mesh.field.setNumber(f, "Quads", 1)
        gmsh.model.mesh.field.setNumber(f, "Thickness", size / 4 * (1.3 ** layers - 1) / 0.3)
        gmsh.model.mesh.field.setAsBoundaryLayer(f)
        gmsh.model.occ.extrude([(2, disk)], 0, 0, length, numElements=[max(1, int(round(length / size)))], recombine=True)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = np.zeros(int(tags.max()) + 1, np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    points = coords.reshape(-1, 3)
    names = {4: ("tetra", 4), 5: ("hexahedron", 8), 6: ("wedge", 6), 7: ("pyramid", 5)}
    blocks = {}
    types, _, nodes = gmsh.model.mesh.getElements(3)
    for t, nd in zip(types, nodes):
        name, k = names[int(t)]
        blocks[name] = index[nd.astype(np.int64)].reshape(-1, k)
    version = gmsh.__version__
    gmsh.finalize()
    return points, blocks, version


def tube_mesh(radius, length, target_cells, layers=0, path=""):
    """A gmsh tube with about ``target_cells`` cells (the size found by two corrections of a first guess). With ``path``
    the vertices and cells are read from that file if it exists (and written to it otherwise): the faces are rebuilt."""
    import json
    import os

    def patch_of(xf):
        return np.where(xf[:, 2] < 1e-9 * length, 1, np.where(xf[:, 2] > length * (1 - 1e-9), 2, 0))

    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        blocks = {k[6:]: z[k] for k in z.files if k.startswith("block_")}
        mesh = from_cells(z["points"], blocks, patch_of)
        mesh.info.update(json.loads(str(z["info"])))
        return mesh
    volume = np.pi * radius ** 2 * length
    per = 8.5 if layers == 0 else 2.2          # cells per size^3, about: tetrahedra; extruded prisms
    size = (volume * per / target_cells) ** (1 / 3)
    for _ in range(3):
        points, blocks, version = gmsh_tube(radius, length, size, layers)
        cells = sum(len(v) for v in blocks.values())
        if abs(cells / target_cells - 1) < 0.08:
            break
        size *= (cells / target_cells) ** (1 / 3)

    info = {"gmsh": version, "size": float(size), "kinds": {k: int(len(v)) for k, v in blocks.items()}, "layers": layers}
    if path:
        used = np.unique(np.concatenate([v.ravel() for v in blocks.values()]))
        remap = np.zeros(len(points), np.int64)
        remap[used] = np.arange(len(used))
        np.savez(path, points=points[used], info=json.dumps(info), **{"block_" + k: remap[v].astype(np.int32) for k, v in blocks.items()})
        points, blocks = points[used], {k: remap[v] for k, v in blocks.items()}
    mesh = from_cells(points, blocks, patch_of)
    mesh.info.update(info)
    return mesh


# ---------------------------------------------------------------------------------------------------------------------
# bodies of revolution: a meridional mesh (gmsh, two-dimensional) revolved into prisms and hexahedra
# ---------------------------------------------------------------------------------------------------------------------
BODY, CONTAINER = 1, 2


def meridional(aspect, container, h_surface, h_rim, h_far, growth=0.25, baffle=False, quads=False, seed=1):
    """Triangles (or mostly quadrilaterals) in the half-plane ``rho >= 0`` between a body of revolution of equatorial
    radius 1 and half-thickness ``aspect`` (``baffle``: the zero-thickness disk, an embedded segment) and a concentric
    spherical container. Sizes: ``h_surface`` on the body, ``h_rim`` at its rim, growing by ``growth`` per unit distance
    to ``h_far``. Returns ``(nodes (rho, z), cells (lists of 3 or 4 nodes), gmsh version)``."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.RandomSeed", seed)
    gmsh.option.setNumber("General.NumThreads", 1)
    occ = gmsh.model.occ
    half = np.pi / 2
    outer = occ.addCircle(0, 0, 0, container, angle1=-half, angle2=half)
    if baffle:
        p_lo, p_hi = occ.addPoint(0, -container, 0), occ.addPoint(0, container, 0)
        axis = occ.addLine(p_hi, p_lo)
        loop = occ.addCurveLoop([outer, axis])
        surface = occ.addPlaneSurface([loop])
        p0, p1 = occ.addPoint(0, 0, 0), occ.addPoint(1, 0, 0)
        segment = occ.addLine(p0, p1)
        occ.synchronize()
        out, _ = occ.fragment([(2, surface)], [(1, segment)])
        occ.synchronize()
        body_curves = [c[1] for c in gmsh.model.getEntities(1)
                       if abs(gmsh.model.getBoundingBox(1, c[1])[4]) < 1e-9 and abs(gmsh.model.getBoundingBox(1, c[1])[1]) < 1e-9
                       and gmsh.model.getBoundingBox(1, c[1])[3] < 1 + 1e-6 and gmsh.model.getBoundingBox(1, c[1])[3] > 0.5]
    else:
        if aspect >= 1.0 - 1e-12:
            inner_curve = occ.addCircle(0, 0, 0, 1.0, angle1=-half, angle2=half)
        else:
            inner_curve = occ.addEllipse(0, 0, 0, 1.0, aspect, angle1=-half, angle2=half)
        lo = occ.addLine(occ.addPoint(0, -container, 0), occ.addPoint(0, -aspect, 0))
        hi = occ.addLine(occ.addPoint(0, aspect, 0), occ.addPoint(0, container, 0))
        occ.synchronize()
        out, _ = occ.fragment([(1, outer)], [(1, inner_curve), (1, lo), (1, hi)])
        occ.synchronize()
        curves = [c[1] for c in gmsh.model.getEntities(1)]
        loop = gmsh.model.geo.addCurveLoop(curves, reorient=True)
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        body_curves = [c for c in curves if gmsh.model.getBoundingBox(1, c)[3] < 1 + 1e-6 and gmsh.model.getBoundingBox(1, c)[3] > 0.5]
    field = gmsh.model.mesh.field
    f_body = field.add("Distance")
    field.setNumbers(f_body, "CurvesList", body_curves)
    field.setNumber(f_body, "Sampling", 400)
    rim = gmsh.model.occ.addPoint(1, 0, 0)
    gmsh.model.occ.synchronize()
    f_rim = field.add("Distance")
    field.setNumbers(f_rim, "PointsList", [rim])
    f = field.add("MathEval")
    field.setString(f, "F", f"Min(Min({h_surface} + {growth} * F{f_body}, {h_rim} + {growth} * F{f_rim}), {h_far})")
    field.setAsBackgroundMesh(f)
    for name in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber("Mesh." + name, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8 if quads else 6)
    if quads:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = np.full(int(tags.max()) + 1, -1, np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    nodes = coords.reshape(-1, 3)[:, :2].copy()
    cells = []
    types, _, conn = gmsh.model.mesh.getElements(2)
    for t, nd in zip(types, conn):
        k = {2: 3, 3: 4}[int(t)]
        cells.extend(index[nd.astype(np.int64)].reshape(-1, k).tolist())
    version = gmsh.__version__
    gmsh.finalize()
    used = np.unique(np.concatenate([np.asarray(c) for c in cells]))
    remap = np.full(len(nodes), -1, np.int64)
    remap[used] = np.arange(len(used))
    return nodes[used], [remap[np.asarray(c)].tolist() for c in cells], version


def revolved(nodes, cells2d, sectors, aspect, container, baffle=False):
    """The meridional mesh revolved through ``sectors`` equal sectors into a face mesh of planar-faced prisms and
    hexahedra. Radii are scaled so each revolved polygon has its circle's area (``sqrt(dphi / sin dphi)``). Patches:
    :data:`BODY` (the ellipse, or both sides of the baffle's segment) and :data:`CONTAINER`."""
    nodes = np.asarray(nodes, float)
    dphi = 2 * np.pi / sectors
    scale = np.sqrt(dphi / np.sin(dphi))
    rho, z = nodes[:, 0].clip(min=0.0), nodes[:, 1]
    on_axis = rho < 1e-9
    radius = np.hypot(rho, z)
    on_container = np.abs(radius - container) < 1e-6 * container
    if baffle:
        on_body = (np.abs(z) < 1e-9) & (rho < 1 + 1e-9)
    else:
        on_body = np.abs(rho ** 2 + (z / aspect) ** 2 - 1.0) < 1e-6
    n2 = len(cells2d)
    # the two-dimensional cells' areas and centroids, and their edges
    area2, cen2 = np.zeros(n2), np.zeros((n2, 2))
    edges = {}
    for c, poly in enumerate(cells2d):
        p = np.stack([rho[poly], z[poly]], axis=1)
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        area2[c] = 0.5 * cross.sum()
        cen2[c] = ((p + q) * cross[:, None]).sum(axis=0) / (6.0 * area2[c])
        for a, b in zip(poly, poly[1:] + poly[:1]):
            edges.setdefault((min(a, b), max(a, b)), []).append(c)
    area2 = np.abs(area2)
    k = np.arange(sectors)
    phi = k * dphi
    n = n2 * sectors
    cell = lambda c, kk: c * sectors + (kk % sectors)
    own, nei, S, xf, patch = [], [], [], [], []
    # faces in the planes phi_k, between sector k - 1 and sector k
    for c in range(n2):
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros(sectors)], axis=1)
        own.append(cell(c, k - 1)), nei.append(cell(c, k))
        S.append(scale * area2[c] * e_phi)
        xf.append(np.stack([scale * cen2[c, 0] * np.cos(phi), scale * cen2[c, 0] * np.sin(phi), np.full(sectors, cen2[c, 1])], axis=1))
        patch.append(np.full(sectors, -1))
    own, nei, S, xf, patch = (np.concatenate(x) for x in (own, nei, S, xf, patch))
    # faces swept by the two-dimensional edges
    e_a, e_b, e_own, e_nei, e_patch = [], [], [], [], []
    for (a, b), cs in edges.items():
        if on_axis[a] and on_axis[b]:
            continue
        if len(cs) == 2:
            if baffle and on_body[a] and on_body[b]:
                for c in cs:
                    e_a.append(a), e_b.append(b), e_own.append(c), e_nei.append(-1), e_patch.append(BODY)
            else:
                e_a.append(a), e_b.append(b), e_own.append(cs[0]), e_nei.append(cs[1]), e_patch.append(-1)
        else:
            kind = BODY if (on_body[a] and on_body[b]) else (CONTAINER if (on_container[a] and on_container[b]) else 0)
            assert kind != 0, "an edge on no boundary"
            e_a.append(a), e_b.append(b), e_own.append(cs[0]), e_nei.append(-1), e_patch.append(kind)
    e_a, e_b, e_own, e_nei, e_patch = (np.asarray(x) for x in (e_a, e_b, e_own, e_nei, e_patch))
    ph0, ph1 = phi[None, :], phi[None, :] + dphi

    def point(node, ph):
        return np.stack([scale * rho[node][:, None] * np.cos(ph), scale * rho[node][:, None] * np.sin(ph),
                         np.broadcast_to(z[node][:, None], (len(node), sectors))], axis=-1)

    quad = np.stack([point(e_a, ph0), point(e_b, ph0), point(e_b, ph1), point(e_a, ph1)], axis=2).reshape(-1, 4, 3)
    S2, xf2 = polygon_geometry(quad, np.zeros(len(quad), bool))
    own2 = (e_own[:, None] * sectors + k[None, :]).ravel()
    nei2 = np.where(e_nei[:, None] >= 0, e_nei[:, None] * sectors + k[None, :], n).ravel()
    patch2 = np.repeat(e_patch, sectors)
    own, nei = np.concatenate([own, own2]), np.concatenate([nei, nei2])
    S, xf, patch = np.concatenate([S, S2]), np.concatenate([xf, xf2]), np.concatenate([patch, patch2])
    mid = (k + 0.5) * dphi
    inside = scale * np.cos(dphi / 2)
    centre0 = np.stack([(inside * cen2[:, 0:1] * np.cos(mid)[None]).ravel(), (inside * cen2[:, 0:1] * np.sin(mid)[None]).ravel(),
                        np.repeat(cen2[:, 1], sectors)], axis=1)
    return assemble(n, own, nei, S, xf, centre0, patch, info={"cells2d": n2, "sectors": sectors, "scale": float(scale)})
