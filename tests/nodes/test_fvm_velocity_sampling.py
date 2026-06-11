"""FVM velocity-at-points sampling (A1) — the far-field interface output.

Pins :func:`sample_velocity_at_points` (the two-scale Schwarz coupling's
far→near transfer): a cell-centred velocity field sampled at arbitrary world
points, on both the Cartesian (trilinear) and unstructured (nearest-cell
gradient-reconstruction) backends. Both must reproduce a uniform field exactly
and a linear-shear field to 2nd-order accuracy.

Also pins the ``FVMFluidNode`` wiring: the ``n_sample_points`` constructor arg
exposes a ``sample_points`` input and a ``velocity_at_points`` output.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.fvm.integrator import sample_velocity_at_points
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.unstructured import build_face_graph


# ── meshes ─────────────────────────────────────────────────────────────────

def _tet_grid(n: int, L: float = 1.0):
    """An ``n×n×n`` grid of cubes over ``[0,L]^3``, each split into 6 tets."""
    lin = np.linspace(0.0, L, n + 1)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def vid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    cells = []
    # 6-tet (Kuhn) decomposition of each cube around the 0–6 main diagonal.
    corner = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v = [vid(i + dx, j + dy, k + dz) for (dx, dy, dz) in corner]
                for t in tets:
                    cells.append([v[a] for a in t])
    return build_face_graph(pts, np.array(cells, dtype=int), dim=3)


# Analytic fields, evaluated at cell centroids to make a cell-centred field.
def _uniform(x):
    return jnp.broadcast_to(jnp.array([0.3, -0.7, 1.1]), x.shape)


def _linear_shear(x):
    # u = (a·x, b·x, c·x) — a general linear field; gradient is constant.
    return jnp.stack([
        0.5 * x[:, 0] + 0.2 * x[:, 1] - 0.1 * x[:, 2],
        -0.3 * x[:, 0] + 0.4 * x[:, 2],
        0.15 * x[:, 1] - 0.25 * x[:, 2],
    ], axis=1)


_QUERY = jnp.array([
    [0.5, 0.5, 0.5], [0.3, 0.7, 0.2], [0.62, 0.41, 0.88], [0.1, 0.9, 0.55],
])


# ── Cartesian backend (trilinear) ───────────────────────────────────────────

# The sampling-precision tests build float64 meshes; opt them into x64.
pytestmark = pytest.mark.x64


def test_cartesian_uniform_exact():
    mesh = make_cartesian_mesh_3d(8, 8, 8, 1.0, 1.0, 1.0, dtype=jnp.float64)
    field = _uniform(mesh.x)
    got = sample_velocity_at_points(field, _QUERY, mesh)
    assert np.allclose(np.asarray(got), np.asarray(_uniform(_QUERY)), atol=1e-10)


def test_cartesian_linear_shear():
    mesh = make_cartesian_mesh_3d(16, 16, 16, 1.0, 1.0, 1.0, dtype=jnp.float64)
    field = _linear_shear(mesh.x)
    got = sample_velocity_at_points(field, _QUERY, mesh)
    # Trilinear reproduces a linear field exactly in the interior.
    assert np.allclose(np.asarray(got), np.asarray(_linear_shear(_QUERY)), atol=1e-6)


# ── unstructured backend (nearest-cell gradient reconstruction) ─────────────

def test_tet_uniform_exact():
    mesh = _tet_grid(4)
    field = _uniform(mesh.x)
    got = sample_velocity_at_points(field, _QUERY, mesh)
    assert np.allclose(np.asarray(got), np.asarray(_uniform(_QUERY)), atol=1e-8)


def test_tet_linear_shear_second_order():
    # Gradient reconstruction reproduces a linear field up to the Green-Gauss
    # gradient error, which is O(h²) in the reconstructed value — so it should
    # converge as the tet mesh refines.
    err = []
    for n in (4, 8):
        mesh = _tet_grid(n)
        field = _linear_shear(mesh.x)
        got = sample_velocity_at_points(field, _QUERY, mesh)
        err.append(float(np.max(np.abs(
            np.asarray(got) - np.asarray(_linear_shear(_QUERY))))))
    assert err[0] < 5e-2          # already small on the coarse mesh
    assert err[1] < err[0] * 0.6  # and shrinks under refinement


def test_scalar_field_shape():
    mesh = make_cartesian_mesh_3d(8, 8, 8, 1.0, 1.0, 1.0, dtype=jnp.float64)
    field = mesh.x[:, 0]                     # scalar field [N_cells]
    got = sample_velocity_at_points(field, _QUERY, mesh)
    assert got.shape == (_QUERY.shape[0],)
    assert np.allclose(np.asarray(got), np.asarray(_QUERY[:, 0]), atol=1e-6)


# ── FVMFluidNode wiring ─────────────────────────────────────────────────────

def test_node_exposes_sampling_ports():
    from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
    from mime.nodes.environment.fvm.boundary import VelocityBC
    from mime.nodes.environment.fvm.piso import PisoConfig

    mesh = make_cartesian_mesh_3d(8, 8, 8, 1.0, 1.0, 1.0, dtype=jnp.float32)
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=1e-3, rho=1000.0)
    node = FVMFluidNode("fvm", 1e-3, mesh=mesh, bcs=bcs, cfg=cfg,
                        n_sample_points=4)

    assert "sample_points" in node.boundary_input_spec()
    assert node.boundary_input_spec()["sample_points"].shape == (4, 3)
    assert "velocity_at_points" in node.boundary_flux_spec()
    assert node.boundary_flux_spec()["velocity_at_points"].shape == (4, 3)
    assert "velocity_at_points" in node.initial_state()
