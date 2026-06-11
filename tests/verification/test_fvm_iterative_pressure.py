"""Iterative (matrix-free CG) FVM pressure/diffusion path (M2 H1b).

The body-fitted / unstructured FVM solver cannot use the Cartesian FFT/dense
diagonalised Poisson + Helmholtz solvers; it solves the *same* orthogonal FVM
operators matrix-free with Jacobi-preconditioned CG. This pins:

* **No regression** — on a Cartesian mesh, a full PISO trajectory with
  ``transform_backend="iterative"`` reproduces the ``"dense"`` trajectory (the
  iterative solvers invert the identical operator the diagonalised solvers do).
* **The unstructured path runs** — a PISO step on an unstructured tetrahedral
  mesh (where FFT/dense do not apply, and ``"auto"`` selects ``"iterative"``)
  produces a finite velocity/pressure field.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.piso import (
    PisoConfig, initial_state, make_piso_step,
)
from mime.nodes.environment.fvm.unstructured import build_face_graph


def _wall_bcs(mesh):
    bcs = {}
    for p in mesh.patches:
        nbf = int(p.owner.size)
        bcs[p.name] = VelocityBC(
            u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)))
    return bcs


def _run(mesh, backend, *, n_steps=5, dt=0.01):
    cfg = PisoConfig(
        nu=0.05, rho=1.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        transform_backend=backend,
    )
    step = make_piso_step(
        mesh, _wall_bcs(mesh), cfg,
        body_force_fn=lambda t: jnp.array([1.0, 0.0, 0.0]),
    )
    state = initial_state(mesh)
    for _ in range(n_steps):
        state = step(state, dt)
    return state


def _cube_6_tets():
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    cells = np.array([
        [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
        [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
    ], dtype=int)
    return pts, cells


def test_iterative_matches_dense_on_cartesian():
    """Full PISO trajectory: the iterative backend reproduces the dense backend
    on a Cartesian box (no regression — same operator, matrix-free)."""
    mesh = make_cartesian_mesh_3d(6, 6, 6, 1.0, 1.0, 1.0, origin=(0.0, 0.0, 0.0))
    s_dense = _run(mesh, "dense")
    s_iter = _run(mesh, "iterative")
    for key in ("u", "p"):
        a, b = np.asarray(s_iter[key]), np.asarray(s_dense[key])
        rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
        assert rel < 1e-3, f"{key} iterative-vs-dense rel {rel:.2e}"


def test_auto_backend_selects_iterative_for_unstructured():
    pts, cells = _cube_6_tets()
    mesh = build_face_graph(pts, cells, dim=3)
    assert mesh.cartesian_shape is None
    # auto must not crash trying FFT/dense on a mesh with no Cartesian shape.
    state = _run(mesh, "auto", n_steps=2)
    assert np.all(np.isfinite(np.asarray(state["u"])))
    assert np.all(np.isfinite(np.asarray(state["p"])))


def test_iterative_runs_on_unstructured_tet_mesh():
    """The matrix-free path runs end-to-end on an unstructured tet mesh and
    produces a finite field — the body-fitted solver path exists."""
    pts, cells = _cube_6_tets()
    mesh = build_face_graph(pts, cells, dim=3)
    state = _run(mesh, "iterative", n_steps=3)
    u = np.asarray(state["u"])
    p = np.asarray(state["p"])
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(p))
    # the x-body-force drives a non-zero x-velocity
    assert np.abs(u[:, 0]).max() > 0.0


def test_iterative_pressure_bc_periodic_rejected():
    mesh = make_cartesian_mesh_3d(4, 4, 4, 1.0, 1.0, 1.0, origin=(0.0, 0.0, 0.0))
    from mime.nodes.environment.fvm.pressure import (
        make_pressure_solver_iterative,
    )
    with pytest.raises(NotImplementedError, match="Neumann"):
        make_pressure_solver_iterative(mesh, bc=("neumann", "neumann", "periodic"))
