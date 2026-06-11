"""Non-orthogonal diffusion correction + least-squares gradient (B1).

Body-fitted vessel meshes are non-orthogonal: the face area vector ``Sf`` is not
parallel to the owner→neighbour vector ``d``. The orthogonal Laplacian is then
*inconsistent* — its error does not vanish under refinement. These tests pin:

* ``grad_least_squares`` reproduces a linear field exactly on a skewed (sliver)
  tet mesh — the property Green-Gauss lacks and the correction relies on.
* ``laplacian_orthogonal(..., non_orthogonal=True)`` is **consistent**: on a
  sheared (well-shaped, non-orthogonal) simplex mesh its manufactured-solution
  error *decreases* under refinement, whereas the orthogonal-only error grows.
* The correction vanishes on a Cartesian mesh (``k_f → 0``), so existing
  Cartesian solvers are unaffected.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.fvm.unstructured import build_face_graph
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.operators import (
    grad_least_squares, grad_green_gauss, laplacian_orthogonal,
)

pytestmark = pytest.mark.x64


# ── meshes ─────────────────────────────────────────────────────────────────

def _kuhn_tet(n, L=1.0):
    lin = np.linspace(0, L, n + 1)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vid = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
    corner = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v = [vid(i + a, j + b, k + c) for a, b, c in corner]
                for t in tets:
                    cells.append([v[a] for a in t])
    return build_face_graph(pts, np.array(cells, int), dim=3)


def _sheared_tri(n, shear=0.4, L=1.0):
    lin = np.linspace(0, L, n + 1)
    X, Y = np.meshgrid(lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1).astype(float)
    pts[:, 0] += shear * pts[:, 1]            # non-orthogonal, well-shaped
    vid = lambda i, j: i * (n + 1) + j
    cells = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cells.append([a, b, c])
            cells.append([a, c, d])
    return build_face_graph(pts, np.array(cells, int), dim=2)


# ── least-squares gradient: linear-exact on a sliver mesh ───────────────────

def test_lsq_gradient_linear_exact_on_tets():
    mesh = _kuhn_tet(4)
    g0 = jnp.array([0.7, -1.3, 0.4])
    phi = mesh.x @ g0                          # linear field
    bvals = {p.name: (mesh.x[p.owner] + p.d) @ g0 for p in mesh.patches}
    grad = grad_least_squares(phi, mesh, bvals)
    # interior cells (boundary stencils are one-sided but still linear-exact)
    assert np.allclose(np.asarray(grad), np.asarray(g0), atol=1e-9)
    # Green-Gauss is NOT linear-exact on these slivers — confirm the contrast.
    gg = grad_green_gauss(phi, mesh, bvals)
    assert not np.allclose(np.asarray(gg), np.asarray(g0), atol=1e-3)


# ── non-orthogonal correction: consistency on a sheared mesh ────────────────

def _manufactured_error(mesh):
    phi_f = lambda x: jnp.sin(np.pi * x[:, 0]) * jnp.sin(np.pi * x[:, 1])
    lap_f = lambda x: -2 * np.pi ** 2 * phi_f(x)
    phi = phi_f(mesh.x)
    bspec, gbv = {}, {}
    for p in mesh.patches:
        fc = mesh.x[p.owner] + p.d
        val = phi_f(fc)
        bspec[p.name] = {"type": "dirichlet", "value": val, "mu": 1.0}
        gbv[p.name] = val
    xx = np.asarray(mesh.x)
    intr = ((xx[:, 1] > 0.25) & (xx[:, 1] < 0.75)
            & (xx[:, 0] > 0.25 + 0.4 * xx[:, 1]) & (xx[:, 0] < 0.75 + 0.4 * xx[:, 1]))
    le = np.asarray(lap_f(mesh.x))

    def l2(non_orthogonal):
        out = np.asarray(laplacian_orthogonal(
            phi, mesh, boundary_specs=bspec, non_orthogonal=non_orthogonal,
            grad_boundary_values=gbv) / mesh.V)
        return float(np.sqrt(np.mean((out - le)[intr] ** 2)))

    return l2(False), l2(True)


def test_nonorthogonal_correction_is_consistent():
    err_o, err_c = {}, {}
    for n in (16, 32, 64):
        m = _sheared_tri(n)
        err_o[n], err_c[n] = _manufactured_error(m)
    # corrected: error decreases under refinement (consistent / convergent)
    assert err_c[64] < err_c[32] < err_c[16]
    order = np.log2(err_c[32] / err_c[64])
    assert order > 0.8
    # orthogonal-only: error does NOT decrease (inconsistent on skewed mesh)
    assert err_o[64] > err_o[16]
    # and the correction is a large win at every resolution
    assert err_c[16] < 0.1 * err_o[16]


def test_correction_vanishes_on_cartesian():
    mesh = make_cartesian_mesh_3d(12, 12, 12, 1.0, 1.0, 1.0, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.standard_normal(mesh.N_cells))
    bspec = {p.name: {"type": "dirichlet",
                      "value": jnp.asarray(rng.standard_normal(p.owner.size)),
                      "mu": 1.0} for p in mesh.patches}
    gbv = {k: v["value"] for k, v in bspec.items()}
    out_o = laplacian_orthogonal(phi, mesh, boundary_specs=bspec)
    out_c = laplacian_orthogonal(phi, mesh, boundary_specs=bspec,
                                 non_orthogonal=True, grad_boundary_values=gbv)
    assert np.allclose(np.asarray(out_o), np.asarray(out_c), atol=1e-9)
