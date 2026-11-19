"""(w2f copy of CORE spike 2's mesher; ``zc`` puts the domain's centre at ``z = zc``, the body staying at the origin.)
A periodic tube's lumen round a held body of revolution on its axis, as a face mesh: a gmsh triangle mesh of the
meridional half-plane (outside JAX), its two ends meshed as a periodic pair, revolved in numpy through equal sectors into
planar-faced prisms. Every boundary face is a wall (the body or the tube); the two ends are joined by inner faces.

Lengths are in metres. ``BODY`` and ``WALL`` are the patch ids."""
import numpy as np

from gfv import mesh as meshmod

BODY, WALL = 1, 2


def meridional(radius, length, a, aspect, h_body, h_rim, h_far, growth=0.3, seed=1, zc=0.0):
    """Triangles in ``0 <= rho <= radius``, ``|z| <= length / 2`` outside the ellipse of semi-axes ``a`` (rho) and
    ``aspect * a`` (z). Returns ``(nodes (rho, z), triangles, gmsh version)``; the nodes on ``z = +length/2`` are the
    periodic images of those on ``z = -length/2``."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.RandomSeed", seed)
    gmsh.option.setNumber("General.NumThreads", 1)
    occ = gmsh.model.occ
    s = 1.0 / a                       # mesh in units of the body's radius (gmsh's tolerances are absolute)
    R, L = radius * s, length * s
    C = zc * s
    rect = occ.addRectangle(0.0, -L / 2 + C, 0.0, R, L)
    disk = occ.addDisk(0.0, 0.0, 0.0, 1.0, aspect)
    out, _ = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()
    bottom, top, body = [], [], []
    for _, c in gmsh.model.getEntities(1):
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
        if abs(y0 + L / 2 - C) < 1e-6 and abs(y1 + L / 2 - C) < 1e-6:
            bottom.append(c)
        elif abs(y0 - L / 2 - C) < 1e-6 and abs(y1 - L / 2 - C) < 1e-6:
            top.append(c)
        elif x1 < 1 + 1e-6 and x1 > 0.5 and abs(y0) < aspect + 1e-6 and abs(y1) < aspect + 1e-6:
            body.append(c)
    assert len(bottom) == 1 and len(top) == 1 and body, (bottom, top, body)
    gmsh.model.mesh.setPeriodic(1, top, bottom, [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1])
    field = gmsh.model.mesh.field
    f_body = field.add("Distance")
    field.setNumbers(f_body, "CurvesList", body)
    field.setNumber(f_body, "Sampling", 400)
    rim = occ.addPoint(1.0, 0.0, 0.0)
    occ.synchronize()
    f_rim = field.add("Distance")
    field.setNumbers(f_rim, "PointsList", [rim])
    f = field.add("MathEval")
    field.setString(f, "F", f"Min(Min({h_body * s} + {growth} * F{f_body}, {h_rim * s} + {growth} * F{f_rim}), {h_far * s})")
    field.setAsBackgroundMesh(f)
    for name in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber("Mesh." + name, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = np.full(int(tags.max()) + 1, -1, np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    nodes = coords.reshape(-1, 3)[:, :2].copy()
    types, _, conn = gmsh.model.mesh.getElements(2)
    tris = np.concatenate([index[nd.astype(np.int64)].reshape(-1, 3) for t, nd in zip(types, conn) if int(t) == 2])
    version = gmsh.__version__
    gmsh.finalize()
    used = np.unique(tris)
    remap = np.full(len(nodes), -1, np.int64)
    remap[used] = np.arange(len(used))
    return nodes[used] / s, remap[tris], version


def section_order(tris):
    """Reverse Cuthill-McKee order of the section's triangles (neighbours share an edge)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    edges = {}
    for c, t in enumerate(tris):
        for i in range(3):
            edges.setdefault((min(t[i], t[(i + 1) % 3]), max(t[i], t[(i + 1) % 3])), []).append(c)
    pairs = np.array([cs for cs in edges.values() if len(cs) == 2])
    n = len(tris)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    return np.asarray(reverse_cuthill_mckee((g + g.T).tocsr(), symmetric_mode=True))


def revolved(nodes, tris, sectors, radius, length, a, aspect, zc=0.0):
    """The section revolved through ``sectors`` equal sectors; cell ``(c, k)`` is ``c * sectors + k``. Radii are scaled
    so each revolved polygon has its circle's area. Returns a :class:`gfv.mesh.FaceMesh` whose ``info`` holds
    ``cells2d``, ``sectors`` and the body's volume as meshed."""
    nodes = np.asarray(nodes, float)
    tris = np.asarray(tris)[section_order(tris)]
    dphi = 2 * np.pi / sectors
    scale = np.sqrt(dphi / np.sin(dphi))
    rho, z = nodes[:, 0].clip(min=0.0), nodes[:, 1]
    tol = 1e-6 * a
    on_axis = rho < tol
    on_wall = np.abs(rho - radius) < tol
    on_top, on_bottom = np.abs(z - zc - length / 2) < tol, np.abs(z - zc + length / 2) < tol
    on_body = np.abs((rho / a) ** 2 + (z / (aspect * a)) ** 2 - 1.0) < 1e-5
    n2 = len(tris)
    p = np.stack([rho[tris], z[tris]], axis=2)                       # (n2, 3, 2)
    q = np.roll(p, -1, axis=1)
    cross = p[:, :, 0] * q[:, :, 1] - q[:, :, 0] * p[:, :, 1]
    area2 = 0.5 * cross.sum(axis=1)
    cen2 = ((p + q) * cross[:, :, None]).sum(axis=1) / (6.0 * area2[:, None])
    area2 = np.abs(area2)
    edges = {}
    for c, t in enumerate(tris.tolist()):
        for i in range(3):
            edges.setdefault((min(t[i], t[(i + 1) % 3]), max(t[i], t[(i + 1) % 3])), []).append(c)
    k = np.arange(sectors)
    phi = k * dphi
    n = n2 * sectors
    zero3 = np.zeros(3)
    # faces in the planes phi_k, between sector k - 1 and sector k
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros(sectors)], axis=1)
    own = (np.arange(n2)[:, None] * sectors + ((k - 1) % sectors)[None, :]).ravel()
    nei = (np.arange(n2)[:, None] * sectors + k[None, :]).ravel()
    S = (scale * area2[:, None, None] * e_phi[None]).reshape(-1, 3)
    xf = np.stack([scale * cen2[:, 0:1] * np.cos(phi)[None], scale * cen2[:, 0:1] * np.sin(phi)[None],
                   np.broadcast_to(cen2[:, 1:2], (n2, sectors))], axis=-1).reshape(-1, 3)
    patch = np.full(n, -1)
    shift = np.zeros((n, 3))
    # faces swept by the section's edges; the two ends are paired by radius
    bottom_of = {}
    for (i, j), cs in edges.items():
        if len(cs) == 1 and on_bottom[i] and on_bottom[j]:
            bottom_of[round(float(rho[i] + rho[j]) / a * 1e6)] = cs[0]
    e_a, e_b, e_own, e_nei, e_patch, e_per = [], [], [], [], [], []
    for (i, j), cs in edges.items():
        if on_axis[i] and on_axis[j]:
            continue
        if len(cs) == 2:
            e_a.append(i), e_b.append(j), e_own.append(cs[0]), e_nei.append(cs[1]), e_patch.append(-1), e_per.append(0)
        elif on_bottom[i] and on_bottom[j]:
            continue
        elif on_top[i] and on_top[j]:
            key = round(float(rho[i] + rho[j]) / a * 1e6)
            other = bottom_of.get(key, bottom_of.get(key + 1, bottom_of.get(key - 1)))
            assert other is not None, "an end edge with no periodic image"
            e_a.append(i), e_b.append(j), e_own.append(cs[0]), e_nei.append(other), e_patch.append(-1), e_per.append(1)
        else:
            kind = BODY if (on_body[i] and on_body[j]) else (WALL if (on_wall[i] and on_wall[j]) else 0)
            assert kind != 0, ("an edge on no boundary", nodes[i], nodes[j])
            e_a.append(i), e_b.append(j), e_own.append(cs[0]), e_nei.append(-1), e_patch.append(kind), e_per.append(0)
    e_a, e_b, e_own, e_nei, e_patch, e_per = (np.asarray(x) for x in (e_a, e_b, e_own, e_nei, e_patch, e_per))
    ph0, ph1 = phi[None, :], phi[None, :] + dphi

    def point(node, ph):
        return np.stack([scale * rho[node][:, None] * np.cos(ph), scale * rho[node][:, None] * np.sin(ph),
                         np.broadcast_to(z[node][:, None], (len(node), sectors))], axis=-1)

    quad = np.stack([point(e_a, ph0), point(e_b, ph0), point(e_b, ph1), point(e_a, ph1)], axis=2).reshape(-1, 4, 3)
    S2, xf2 = meshmod.polygon_geometry(quad, np.zeros(len(quad), bool))
    own2 = (e_own[:, None] * sectors + k[None, :]).ravel()
    nei2 = np.where(e_nei[:, None] >= 0, e_nei[:, None] * sectors + k[None, :], n).ravel()
    shift2 = np.repeat(e_per, sectors)[:, None] * np.array([0.0, 0.0, length])[None]
    own, nei = np.concatenate([own, own2]), np.concatenate([nei, nei2])
    S, xf = np.concatenate([S, S2]), np.concatenate([xf, xf2])
    patch, shift = np.concatenate([patch, np.repeat(e_patch, sectors)]), np.concatenate([shift, shift2])
    mid = (k + 0.5) * dphi
    inside = scale * np.cos(dphi / 2)
    c0 = np.stack([(inside * cen2[:, 0:1] * np.cos(mid)[None]).ravel(), (inside * cen2[:, 0:1] * np.sin(mid)[None]).ravel(),
                   np.repeat(cen2[:, 1], sectors)], axis=1)
    # orient owner to neighbour; volumes and centroids from pyramids on the faces (a periodic face seen from its
    # neighbour sits one length back)
    S = S.copy()
    flip = np.einsum("fi,fi->f", S, xf - c0[own]) < 0
    S[flip] *= -1
    inner = nei < n
    V, xc_num = np.zeros(n), np.zeros((n, 3))
    vol = np.einsum("fi,fi->f", S, xf - c0[own]) / 3.0
    np.add.at(V, own, vol)
    np.add.at(xc_num, own, vol[:, None] * (0.75 * xf + 0.25 * c0[own]))
    xn = xf[inner] - shift[inner]
    vol = -np.einsum("fi,fi->f", S[inner], xn - c0[nei[inner]]) / 3.0
    np.add.at(V, nei[inner], vol)
    np.add.at(xc_num, nei[inner], vol[:, None] * (0.75 * xn + 0.25 * c0[nei[inner]]))
    xc = xc_num / V[:, None]
    d = np.where(inner[:, None], xc[np.where(inner, nei, 0)] + shift - xc[own], xf - xc[own])
    to_nei = np.einsum("fi,fi->f", S, np.where(inner[:, None], d - (xf - xc[own]), 0.0))
    w = np.where(inner, to_nei / np.einsum("fi,fi->f", S, d), 0.0)
    e = np.lexsort((nei, own, ~inner))
    body = patch == BODY
    body_volume = float(-np.einsum("fi,fi->", xf[body], S[body]) / 3.0)
    info = {"cells2d": int(n2), "sectors": int(sectors), "scale": float(scale), "body_volume": body_volume,
            "lumen_volume": float(V.sum()), "periodic_faces": int(e_per.sum() * sectors)}
    return meshmod.FaceMesh(n, own[e].astype(np.int64), nei[e].astype(np.int64), S[e], xf[e], xc, V, d[e], w[e], patch[e], info)


def build(spec):
    """``spec``: radius, length, a, aspect, h_body, h_rim, h_far, growth, sectors. The mesh and its facts."""
    nodes, tris, version = meridional(spec["radius"], spec["length"], spec["a"], spec["aspect"], spec["h_body"], spec["h_rim"],
                                      spec["h_far"], spec.get("growth", 0.3))
    m = revolved(nodes, tris, spec["sectors"], spec["radius"], spec["length"], spec["a"], spec["aspect"])
    m.info["gmsh"] = version
    return m
