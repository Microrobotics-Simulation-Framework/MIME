"""HydrodynamicModel.TwoScale composite effect (C3).

Pins that the two-scale Schwarz coupling attaches as a SINGLE EffectModel
(not a hand-wired far/near pair): the registry lists it, and ``build()``
materialises the full coupled graph — the FVM far node, the BEM near node, the
far↔near interface edges, a coupling group over both, and the near→body drag /
body→near kinematics edges.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mime.effects import (
    Body, Experiment, HydrodynamicModel, Medium,
    get_effect, list_registered_effects,
)
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.interface import create_interface_mesh

_DT, _A, _MU = 1e-4, 1e-3, 1.0


def _rigid_body(name="body"):
    from mime.nodes.robot.rigid_body import RigidBodyNode
    return RigidBodyNode(
        name=name, timestep=_DT,
        semi_major_axis_m=_A, semi_minor_axis_m=_A, density_kg_m3=1100.0,
        fluid_viscosity_pa_s=1e-3, fluid_density_kg_m3=1000.0,
        use_analytical_drag=False)


def _two_scale_effect():
    L, N = 8 * _A, 12
    mesh = make_cartesian_mesh_3d(N, N, N, L, L, L,
                                  origin=(-L / 2, -L / 2, -L / 2))
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=1e-3, rho=1000.0, transform_backend="dense")
    body = sphere_surface_mesh(center=(0, 0, 0), radius=_A, n_refine=2)
    nb = body.n_points
    far = FVMFluidNode("fvm", _DT, mesh=mesh, bcs=bcs, cfg=cfg,
                       n_sample_points=nb, n_forcing_points=nb)
    near = StokesletFluidNode("bem", _DT, mu=_MU, body_mesh=body,
                              interface_mesh=create_interface_mesh(
                                  radius=2 * _A, n_refine=2))
    return HydrodynamicModel.TwoScale(
        far, near,
        body_points=jnp.asarray(body.points),
        body_weights=jnp.asarray(body.weights))


def _experiment():
    exp = Experiment(name="two_scale")
    exp.set_body(Body("body", node=_rigid_body(), properties={"hydrodynamic": {}}))
    exp.set_medium(Medium({"density": 1000.0, "viscosity": 1e-3}))
    exp.attach(_two_scale_effect())
    return exp


def test_registered():
    assert "HydrodynamicModel.TwoScale" in list_registered_effects()
    assert get_effect("HydrodynamicModel.TwoScale") is HydrodynamicModel.TwoScale


def test_attach_one_builds_the_full_coupled_graph():
    gm, handles = _experiment().build()
    names = set(n for h in handles for n in h.node_names)
    assert {"fvm", "bem"} <= names

    # far↔near interface edges
    assert any(e.source_node == "fvm" and e.target_node == "bem"
               and e.source_field == "velocity_at_points"
               and e.target_field == "background_flow" for e in gm._edges)
    assert any(e.source_node == "bem" and e.target_node == "fvm"
               and e.source_field == "body_traction"
               and e.target_field == "forcing_values" for e in gm._edges)
    # near→body drag + body→near kinematics
    assert any(e.source_node == "bem" and e.target_node == "body"
               and e.source_field == "drag_force" for e in gm._edges)
    assert any(e.source_node == "body" and e.target_node == "bem"
               and e.target_field == "body_velocity" for e in gm._edges)
    # a Schwarz coupling group over both fluid nodes
    grp = [g for g in gm._coupling_groups
           if {"fvm", "bem"} <= set(g.nodes)]
    assert len(grp) == 1
    assert grp[0].convergence_norm == "interface"
