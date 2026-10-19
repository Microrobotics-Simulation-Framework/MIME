"""The two-scale transfers on MADDENING 0.4 interface mappings (make_two_scale_coupling(transfers=...)).

On C1's small slice (Cartesian FVM far-field, free-space BEM sphere): the sampling on a multilinear_grid gather
reproduces the hand path's trilinear sampler, and the spreading on a multilinear_grid conservative scatter deposits
exactly the near field's point forces (its kernel is trilinear, not the hand path's Gaussian), and the slice
converges only while the IQN accelerates point_force, the field that edge reads: scoped to body_traction, as for the
hand path, the loop is a plain Gauss-Seidel sweep, which diverges for a trilinear kernel (drag -2.5e16 N at step 0;
port_step2/runs/kernel_width_control.py).
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

pytestmark = [pytest.mark.x64]

_A, _V, _RHO, _NU, _DT, _L, _N = 1e-3, 1e-3, 1000.0, 1e-3, 1e-4, 8e-3, 16
_MU = _RHO * _NU
_STOKES = 6 * np.pi * _MU * _A * _V
_TIGHT = dict(convergence_norm="l2", tolerance=1e-12, max_iterations=60)


def _build(transfers):
    mesh = make_cartesian_mesh_3d(_N, _N, _N, _L, _L, _L, origin=(-_L / 2, -_L / 2, -_L / 2),
                                  dtype=jnp.float64)
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3)) for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=_NU, rho=_RHO, transform_backend="dense")
    body = sphere_surface_mesh(center=(0.1 * _A, -0.2 * _A, 0.05 * _A), radius=_A, n_refine=2)
    iface = create_interface_mesh(center=(0, 0, 0), radius=2 * _A, n_refine=2)
    nb = body.n_points
    ports = {"hand": dict(n_sample_points=nb, n_forcing_points=nb),
             "sampling_mapped": dict(coupling_points=nb, n_forcing_points=nb),
             "mapped": dict(coupling_points=nb, cell_forcing=True)}[transfers]
    far = FVMFluidNode("fvm", _DT, mesh=mesh, bcs=bcs, cfg=cfg, forcing_sigma=1.5 * (_L / _N), **ports)
    near = StokesletFluidNode("bem", _DT, mu=_MU, body_mesh=body, interface_mesh=iface,
                              coupling_points=transfers == "mapped")
    gm = GraphManager()
    info = make_two_scale_coupling(gm, far, near, body_points=jnp.asarray(body.points),
                                   body_weights=jnp.asarray(body.weights), coupling_kwargs=_TIGHT,
                                   transfers=transfers)
    for fld, sh in (("body_velocity", (3,)), ("body_angular_velocity", (3,)), ("body_orientation", (4,))):
        gm.add_external_input("bem", fld, shape=sh, dtype=jnp.float64)
    ext = {n: dict(v) for n, v in info["geometry_inputs"].items()}
    ext.setdefault("bem", {}).update({"body_velocity": jnp.array([0.0, 0.0, _V]),
                                      "body_angular_velocity": jnp.zeros(3),
                                      "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])})
    return gm, ext


def _run(transfers, n=4):
    gm, ext = _build(transfers)
    drag, sampled = [], []
    for _ in range(n):
        st = gm.step(ext)
        drag.append(np.asarray(st["bem"]["drag_force"]))
        sampled.append(np.asarray(gm.resolve_boundary_inputs("bem")["background_flow"]))
    return gm, np.asarray(drag), np.asarray(sampled)


def test_mapped_edges_carry_their_geometry():
    for transfers, mapped_fields in (("sampling_mapped", {"u_phys"}), ("mapped", {"u_phys", "point_force"})):
        gm, _ = _build(transfers)
        mapped = {e.source_field: e for e in gm._edges if getattr(e, "mapping", None) is not None}
        assert set(mapped) == mapped_fields
        for e in mapped.values():
            assert e.mapping.kind == "multilinear_grid"
            assert e.geometry == ("source", "coupling_points")
        gm.compile()
        assert all(v == {} for v in gm.params["mappings"].values())


def test_sampling_on_a_mapping_reproduces_the_hand_path():
    _, d_hand, s_hand = _run("hand")
    _, d_map, s_map = _run("sampling_mapped")
    scale = np.abs(s_hand).max()
    assert scale > 0
    assert np.abs(s_map - s_hand).max() <= 1e-9 * scale
    assert np.abs(d_map - d_hand).max() <= 1e-9 * np.abs(d_hand).max()


def test_mapped_forcing_is_accelerated_through_the_field_its_edge_reads():
    gm, _ = _build("mapped")
    (group,) = gm._coupling_groups
    assert group.accelerated_fields == {"fvm": ("u",), "bem": ("point_force",)}


def test_mapped_forcing_deposits_the_point_forces():
    gm, drag, _ = _run("mapped")
    deposited = np.asarray(gm.resolve_boundary_inputs("fvm")["cell_force"]).sum(axis=0)
    point_total = np.asarray(gm.get_node_state("bem")["point_force"]).sum(axis=0)
    np.testing.assert_allclose(deposited, point_total, rtol=1e-12, atol=1e-12 * np.abs(point_total).max())
    np.testing.assert_allclose(point_total, np.asarray(gm.get_node_state("bem")["drag_force"]), rtol=1e-10)
    assert abs(drag[0, 2] / _STOKES - 1.0) < 0.1          # Stokes recovery on the first step, as C1


def test_group_reports_its_solve_without_diagnostics():
    # diagnostics=True asks for ~117 GB on schwarz_vessel_helix's first step on MADDENING 0.4.1 (its spectral and
    # gradient bounds); the solve's own report does not need it.
    gm, ext = _build("hand")
    (group,) = gm._coupling_groups
    assert group.diagnostics is False
    gm.step(ext)
    report = gm.coupling_diagnostics()["bem+fvm"]
    assert int(report["iterations"]) >= 1 and bool(report["converged"])
