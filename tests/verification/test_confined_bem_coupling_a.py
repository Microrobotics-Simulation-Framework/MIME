"""Confined BEM in the coupled solve + double-counting check (TASK A).

The two-scale wall-confinement argument (Q4) rests on the confined Liron–Shahar
BEM (``wall_table`` + ``R_cyl`` → ``_init_confined_schwarz``) actually being
active. These tests pin that and guard against two silent failures:

* **Reversion to free-space** — if the confined path is accidentally dropped, the
  drag falls back to ~free-space Stokes; ``test_confined_path_active`` catches it.
* **Double-counting** — if the FVM IBM wall and the BEM ``G_wall`` *both* added
  confinement independently, the coupled confined drag would be ≈ 2× the BEM-alone
  confined value. ``test_no_gross_double_counting`` asserts it is not (the drag is a
  single confined-BEM computation, so the co-flow background reduces it instead).

Units are the wall-table's non-dimensional convention (body radius = 1,
``R_cyl`` = table value, ``mu`` = 1).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.interface import create_interface_mesh

pytestmark = pytest.mark.x64

_TABLE = (Path(__file__).resolve().parents[2]
          / "data" / "dejongh_benchmark" / "wall_tables" / "wall_R2.500.npz")


def _bem(confined, body, iface, table):
    kw = dict(mu=float(table.mu), body_mesh=body, interface_mesh=iface)
    if confined:
        kw.update(wall_table=table, R_cyl=float(table.R_cyl))
    return StokesletFluidNode("bem", 1e-2, **kw)


def _drag_z(node, body, V=1.0, bg=None):
    nb = body.n_points
    bg = jnp.zeros((nb, 3)) if bg is None else bg
    st = node.initial_state()
    ns = node.update(st, {"body_velocity": jnp.array([0.0, 0.0, V]),
                          "body_angular_velocity": jnp.zeros(3),
                          "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0]),
                          "background_flow": bg}, 1e-2)
    return float(ns["drag_force"][2])


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_confined_path_active():
    table = load_wall_table(str(_TABLE))           # R_cyl=2.5, mu=1, body radius 1
    body = sphere_surface_mesh(radius=1.0, n_refine=2)
    iface = create_interface_mesh(radius=1.8, n_refine=2)
    stokes = 6 * np.pi * float(table.mu) * 1.0 * 1.0
    D_free = _drag_z(_bem(False, body, iface, table), body)
    D_conf = _drag_z(_bem(True, body, iface, table), body)
    # free-space recovers Stokes; confined is strongly enhanced (λ=1/2.5=0.4,
    # Haberman–Sayre centreline factor ≈ 3.4–3.6). A reversion to free-space
    # would drop the enhancement to ~1 and fail here.
    assert abs(D_free / stokes - 1.0) < 0.05
    assert D_conf / D_free > 2.0


@pytest.mark.slow
@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_no_gross_double_counting():
    """Coupled confined drag must NOT be ~2× the BEM-alone confined drag.

    A ratio near 2.0 would mean the FVM IBM wall and the BEM G_wall each added the
    wall confinement independently (double-counting) — every confinement result
    would then be silently inflated. The drag is a single confined-BEM solve
    (A_body+G_wall), so the FVM only sets the background; the co-flow background
    reduces the drag below the zero-background confined value. A FAILURE here
    (ratio ∈ [1.8, 2.2]) is a blocker: confinement results must not be trusted
    until the coupling decomposition is fixed.
    """
    from maddening.core.graph_manager import GraphManager
    from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
    from mime.nodes.environment.fvm.boundary import VelocityBC
    from mime.nodes.environment.fvm.piso import PisoConfig
    from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
    from mime.nodes.environment.fvm.ibm import IBMBody
    from mime.nodes.environment.two_scale import make_two_scale_coupling

    table = load_wall_table(str(_TABLE))
    Rcyl, mu, a, V, rho = float(table.R_cyl), float(table.mu), 1.0, 1.0, 1.0
    body = sphere_surface_mesh(radius=a, n_refine=2)
    nb = body.n_points
    iface = create_interface_mesh(radius=1.8 * a, n_refine=2)

    D_alone = _drag_z(_bem(True, body, iface, table), body, V=V)

    dx = a / 4
    Lxy, Lz = 2 * 1.25 * Rcyl, 8 * a
    nxy, nz = int(np.ceil(Lxy / dx)), int(np.ceil(Lz / dx))
    mesh = make_cartesian_mesh_3d(nxy, nxy, nz, Lxy, Lxy, Lz,
                                  origin=(-Lxy / 2, -Lxy / 2, -Lz / 2),
                                  dtype=jnp.float64, periodic_z=True)
    wall = IBMBody(name="wall", sdf=lambda x: Rcyl - jnp.sqrt(
        x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30))
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max")}
    cfg = PisoConfig(nu=mu / rho, rho=rho, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("neumann", "neumann", "periodic"),
                     velocity_bc=("dirichlet", "dirichlet", "periodic"),
                     ibm_alpha=1e5, ibm_eps=1.0 * dx, transform_backend="dense")
    far = FVMFluidNode("fvm", 1e-2, mesh=mesh, bcs=bcs, cfg=cfg, static_bodies=[wall],
                       n_sample_points=nb, n_forcing_points=nb, forcing_sigma=1.5 * dx)
    near = _bem(True, body, iface, table)
    gm = GraphManager()
    info = make_two_scale_coupling(gm, far, near,
                                   body_points=jnp.asarray(body.points),
                                   body_weights=jnp.asarray(body.weights))
    for fld, sh in (("body_velocity", (3,)), ("body_angular_velocity", (3,)),
                    ("body_orientation", (4,))):
        gm.add_external_input("bem", fld, shape=sh, dtype=jnp.float64)
    ext = dict(info["geometry_inputs"])
    ext["bem"] = {"body_velocity": jnp.array([0.0, 0.0, V]),
                  "body_angular_velocity": jnp.zeros(3),
                  "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])}
    for _ in range(80):
        s = gm.step(ext)
    D_coupled = float(s["bem"]["drag_force"][2])
    ratio = D_coupled / D_alone
    assert not (1.8 <= ratio <= 2.2), f"double-counting signature: ratio={ratio:.3f}"
    assert ratio < 1.3                              # co-flow reduces, not inflates
    assert D_coupled > 0.0
