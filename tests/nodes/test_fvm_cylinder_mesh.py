"""Body-fitted cylindrical-vessel tet mesh (B2).

Pins the procedural (gmsh-free) cylinder mesh generator: divergence-closure and
positive volumes (the FVM operators rely on these), faceted-volume convergence
toward π R² L, correctly-named inlet/outlet/wall patches, and that the mesh
builds a compilable ``FVMFluidNode`` (auto-selecting the unstructured iterative
backend).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.fvm.cylinder_mesh import make_cylinder_tet_mesh


def _div_closure_residual(m):
    acc = np.zeros((m.N_cells, m.dim))
    ow, ne, Sf = np.asarray(m.owner), np.asarray(m.neighbour), np.asarray(m.Sf)
    np.add.at(acc, ow, Sf)
    np.add.at(acc, ne, -Sf)
    for p in m.patches:
        np.add.at(acc, np.asarray(p.owner), np.asarray(p.Sf))
    return np.max(np.abs(acc))


def test_invariants_and_patches():
    R, L = 1e-3, 4e-3
    m = make_cylinder_tet_mesh(R, L, n_radial=4, n_theta=16, n_axial=8,
                               dtype=jnp.float64)
    V = np.asarray(m.V)
    assert (V > 0).all()                                   # no degenerate tets
    assert _div_closure_residual(m) < 1e-7 * R ** 2        # watertight
    names = {p.name for p in m.patches}
    assert names == {"inlet", "outlet", "wall"}
    # faceted cap area ≈ π R² cos-deficit; wall ≈ 2π R L
    area = {p.name: float(np.sum(p.area)) for p in m.patches}
    assert area["inlet"] == pytest.approx(np.pi * R ** 2, rel=0.06)
    assert area["outlet"] == pytest.approx(np.pi * R ** 2, rel=0.06)
    assert area["wall"] == pytest.approx(2 * np.pi * R * L, rel=0.06)


def test_volume_converges_with_azimuthal_refinement():
    R, L = 1e-3, 2e-3
    exact = np.pi * R ** 2 * L
    err = []
    for nth in (8, 16, 32):
        m = make_cylinder_tet_mesh(R, L, n_radial=3, n_theta=nth, n_axial=4,
                                   dtype=jnp.float64)
        err.append(abs(float(np.sum(m.V)) - exact) / exact)
    # inscribed-polygon deficit ~ (π/n)²/3 → roughly 4× smaller per doubling
    assert err[2] < err[1] < err[0]
    assert err[2] < 0.01


def test_builds_compilable_fluid_node():
    from mime.nodes.environment.fvm.boundary import VelocityBC
    from mime.nodes.environment.fvm.piso import PisoConfig
    from mime.nodes.environment.fvm.fluid_node import FVMFluidNode

    m = make_cylinder_tet_mesh(1e-3, 4e-3, n_radial=3, n_theta=12, n_axial=6,
                               dtype=jnp.float32)
    assert m.cartesian_shape is None                       # → iterative backend
    bcs = {"wall": VelocityBC(u_wall=jnp.zeros(3)),
           "inlet": VelocityBC(), "outlet": VelocityBC()}
    cfg = PisoConfig(nu=1e-3, rho=1000.0, transform_backend="auto")
    node = FVMFluidNode("vessel", 1e-4, mesh=m, bcs=bcs, cfg=cfg)
    st = node.initial_state()
    ns = node.update(st, {}, 1e-4)               # compiles + runs one PISO step
    assert np.isfinite(np.asarray(ns["u"])).all()
