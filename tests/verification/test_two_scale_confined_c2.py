"""Two-scale on a confined vessel (C2) — Cartesian + IBM-wall far-field ⊗ BEM.

The headline integration: a Cartesian FVM far-field with the vessel wall as an
IBM cylinder (the validated ``make_pipe_mesh``/Brinkman config) ⊗ a Stokeslet-BEM
near-field, coupled via ``make_two_scale_coupling`` (C1). This is the
single-GPU, gmsh-free realisation of the two-scale solver chosen after the
body-fitted tet path hit the skewness-stability wall (see
``plans/MIME_v0.3.0_PLAN.md`` B3 + memory ``fvm-skewness-stability-threshold``).

Pins that the coupled confined solve **runs end-to-end, the Schwarz iteration
converges, and the confinement is active** — the IBM vessel wall measurably
changes the swimmer drag vs the unconfined coupling. The confined-drag
*magnitude* (vs Bohlin / de Jongh) is E-phase validation, not asserted here.
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
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.two_scale import make_two_scale_coupling

pytestmark = [pytest.mark.x64, pytest.mark.slow]

_A, _RPIPE, _V = 1e-3, 2.5e-3, 1e-3
_RHO, _NU, _DT, _CPR = 1000.0, 1e-3, 2e-4, 4
_MU = _RHO * _NU
_STOKES = 6 * np.pi * _MU * _A * _V


def _build(with_wall: bool):
    dx = _A / _CPR
    Lxy = 2 * 1.2 * _RPIPE
    Lz = 6 * _A
    nxy = int(np.ceil(Lxy / dx))
    nz = int(np.ceil(Lz / dx))
    mesh = make_cartesian_mesh_3d(nxy, nxy, nz, Lxy, Lxy, Lz,
                                  origin=(-Lxy / 2, -Lxy / 2, -Lz / 2),
                                  dtype=jnp.float64, periodic_z=True)
    bcs = {nm: VelocityBC(u_wall=jnp.zeros(3))
           for nm in ("x_min", "x_max", "y_min", "y_max")}
    cfg = PisoConfig(nu=_NU, rho=_RHO, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("neumann", "neumann", "periodic"),
                     velocity_bc=("dirichlet", "dirichlet", "periodic"),
                     ibm_alpha=1e5, ibm_eps=1.0 * dx, transform_backend="dense")
    static = []
    if with_wall:
        static = [IBMBody(name="pipe_wall",
                          sdf=lambda x: _RPIPE - jnp.sqrt(
                              x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30))]
    body = sphere_surface_mesh(center=(0, 0, 0), radius=_A, n_refine=2)
    nb = body.n_points
    far = FVMFluidNode("fvm", _DT, mesh=mesh, bcs=bcs, cfg=cfg,
                       static_bodies=static, n_sample_points=nb,
                       n_forcing_points=nb, forcing_sigma=1.5 * dx)
    near = StokesletFluidNode("bem", _DT, mu=_MU, body_mesh=body,
                              interface_mesh=create_interface_mesh(
                                  radius=2 * _A, n_refine=2))
    gm = GraphManager()
    info = make_two_scale_coupling(gm, far, near,
                                   body_points=jnp.asarray(body.points),
                                   body_weights=jnp.asarray(body.weights))
    for fld, sh in (("body_velocity", (3,)), ("body_angular_velocity", (3,)),
                    ("body_orientation", (4,))):
        gm.add_external_input("bem", fld, shape=sh, dtype=jnp.float64)
    ext = dict(info["geometry_inputs"])
    ext["bem"] = {"body_velocity": jnp.array([0.0, 0.0, _V]),
                  "body_angular_velocity": jnp.zeros(3),
                  "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])}
    return gm, ext


def _run(gm, ext, n=100):
    for _ in range(n):
        st = gm.step(ext)
    Fz = float(st["bem"]["drag_force"][2])
    diag = gm.coupling_diagnostics()["bem+fvm"]
    return Fz, diag


def test_confined_two_scale_runs_converges_and_confinement_is_active():
    gm_c, ext_c = _build(with_wall=True)
    Fz_c, diag_c = _run(gm_c, ext_c)
    gm_f, ext_f = _build(with_wall=False)
    Fz_f, _ = _run(gm_f, ext_f)

    # runs end-to-end, finite, converges within the iteration cap
    assert np.isfinite(Fz_c) and Fz_c > 0.0
    assert diag_c["iterations"] <= 8 and np.isfinite(diag_c["residual"])
    # the coupled drag is a physically sane fraction of the free Stokes scale
    assert 0.1 < Fz_c / _STOKES < 3.0
    # the IBM vessel wall measurably changes the drag — confinement is active
    assert abs(Fz_c - Fz_f) / Fz_f > 0.05
