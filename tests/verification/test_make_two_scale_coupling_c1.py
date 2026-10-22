"""make_two_scale_coupling (C1) — the reusable two-scale Schwarz builder.

Rebuilds the A3 de-risk slice (Cartesian FVM far-field ⊗ free-space BEM sphere
near-field) through the ``make_two_scale_coupling`` helper instead of hand-wiring,
and checks it reproduces the A3 physics (Stokes-drag recovery, sign-correct
transfers) and wires the expected edges / coupling group / geometry inputs. The
helper is far-field-agnostic: the same call serves the Cartesian+IBM far-field
that C2 uses.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.two_scale import make_two_scale_coupling

pytestmark = [pytest.mark.x64, pytest.mark.slow]

_A, _V, _RHO, _NU, _DT, _L, _N = 1e-3, 1e-3, 1000.0, 1e-3, 1e-4, 8e-3, 24
_MU = _RHO * _NU
_STOKES = 6 * np.pi * _MU * _A * _V


def _build(coupling_kwargs=None):
    mesh = make_cartesian_mesh_3d(
        _N, _N, _N, _L, _L, _L,
        origin=(-_L / 2, -_L / 2, -_L / 2), dtype=jnp.float64)
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=_NU, rho=_RHO, transform_backend="dense")
    body = sphere_surface_mesh(center=(0, 0, 0), radius=_A, n_refine=2)
    iface = create_interface_mesh(center=(0, 0, 0), radius=2 * _A, n_refine=2)
    nb = body.n_points
    far = FVMFluidNode("fvm", _DT, mesh=mesh, bcs=bcs, cfg=cfg,
                       n_sample_points=nb, n_forcing_points=nb,
                       forcing_sigma=1.5 * (_L / _N))
    near = StokesletFluidNode("bem", _DT, mu=_MU, body_mesh=body,
                              interface_mesh=iface)
    gm = GraphManager()
    info = make_two_scale_coupling(
        gm, far, near,
        body_points=jnp.asarray(body.points),
        body_weights=jnp.asarray(body.weights),
        coupling_kwargs=coupling_kwargs)
    # The near-field body kinematics are the caller's to declare (they could
    # instead come from a rigid-body node); here we drive them externally.
    for fld, sh in (("body_velocity", (3,)), ("body_angular_velocity", (3,)),
                    ("body_orientation", (4,))):
        gm.add_external_input("bem", fld, shape=sh, dtype=jnp.float64)
    ext = dict(info["geometry_inputs"])
    ext["bem"] = {"body_velocity": jnp.array([0.0, 0.0, _V]),
                  "body_angular_velocity": jnp.zeros(3),
                  "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])}
    c = int(np.argmin(np.linalg.norm(np.asarray(mesh.x), axis=1)))
    return gm, ext, c


def test_helper_wires_edges_and_group():
    gm, _, _ = _build()
    # the two interface transfer edges exist
    assert any(e.source_node == "fvm" and e.target_node == "bem"
               and e.source_field == "velocity_at_points"
               and e.target_field == "background_flow" for e in gm._edges)
    assert any(e.source_node == "bem" and e.target_node == "fvm"
               and e.source_field == "body_traction"
               and e.target_field == "forcing_values" for e in gm._edges)
    # a coupling group over both nodes, with the int-state-safe IQN defaults
    assert len(gm._coupling_groups) == 1
    grp = gm._coupling_groups[0]
    assert grp.acceleration == "iqn-ils"
    assert grp.convergence_norm == "interface"
    assert grp.accelerated_fields == {"fvm": ("u",), "bem": ("body_traction",)}


def test_helper_reproduces_a3_physics():
    gm, ext, c = _build()
    Fz0 = None
    for k in range(20):
        st = gm.step(ext)
        if k == 0:
            Fz0 = float(st["bem"]["drag_force"][2])
    Fz = float(st["bem"]["drag_force"][2])
    u_c = np.asarray(st["fvm"]["u"][c])
    # Stokes-drag recovery on the first (near-uncoupled) step
    assert abs(Fz0 / _STOKES - 1.0) < 0.1
    # sign-correct near→far (body drags fluid +z) and far→near feedback (drag ↓)
    assert u_c[2] > 0.0
    assert 0.0 < Fz < Fz0
    # converged within the iteration cap
    diag = gm.coupling_diagnostics()["bem+fvm"]
    assert diag["iterations"] <= 8 and np.isfinite(diag["residual"])
