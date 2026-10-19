"""FVMFluidNode reads its PisoConfig's float constants from gm.params (MADDENING 0.4).

A write through gm.params must reach the compiled step and give the number the constructor gives for the same value;
the leaves are the config's floats, cast to the mesh dtype in the step.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from mime.nodes.environment.fvm import FVMFluidNode, make_cartesian_mesh_3d, make_sphere_body_factory
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.fvm.piso import PisoConfig

V = jnp.asarray([1e-4, 0.0, 0.0], jnp.float32)


def _graph(nu=1e-6, **cfg_kwargs):
    mesh = make_cartesian_mesh_3d(8, 8, 6, 1.2, 1.2, 1.0, origin=(-0.6, -0.6, 0.0), periodic_z=True)
    wall = IBMBody(name="pipe_wall", sdf=lambda x: 0.5 - jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30))
    bcs = {}
    for nm in ("x_min", "x_max", "y_min", "y_max"):
        nbf = int(mesh.patch(nm).owner.size)
        bcs[nm] = VelocityBC(u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)))
    cfg = PisoConfig(nu=nu, rho=1000.0, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("neumann", "neumann", "periodic"),
                     velocity_bc=("dirichlet", "dirichlet", "periodic"),
                     ibm_alpha=1e5, **({"ibm_eps": 0.15} | cfg_kwargs))
    node = FVMFluidNode("fluid", 1e-3, mesh=mesh, bcs=bcs, cfg=cfg, static_bodies=[wall],
                        dynamic_body_factories=[("sphere", make_sphere_body_factory("sphere", radius=1e-3))])
    gm = GraphManager()
    gm.add_node(node)
    gm.add_external_input("fluid", "body_velocity", shape=(3,), dtype=jnp.float32)
    gm.compile()
    return gm


def _drag(gm, n=5):
    for _ in range(n):
        gm.step({"fluid": {"body_velocity": V}})
    return np.asarray(gm.get_node_state("fluid")["drag_force"])


def test_constants_are_leaves():
    gm = _graph()
    leaves = gm.params["nodes"]["fluid"]
    for key in ("nu", "rho", "gamma_conv", "ibm_alpha", "ibm_eps", "forcing_sigma", "force_shell"):
        assert key in leaves, key
    assert "ibm_eps_cells" not in leaves
    assert "fluid" not in gm.nodes_without_params()


def test_write_reaches_the_step_as_the_constructor_does():
    reference = _drag(_graph())
    written = _graph()
    leaf = written.params["nodes"]["fluid"]["nu"]
    written.params["nodes"]["fluid"]["nu"] = jnp.asarray(1e-3, dtype=leaf.dtype)
    by_write = _drag(written)
    by_constructor = _drag(_graph(nu=1e-3))
    assert not np.array_equal(by_write, reference)        # the write is not ignored
    np.testing.assert_array_equal(by_write, by_constructor)


def test_band_in_cells_is_the_leaf_when_the_config_asks_for_it():
    physical = _graph(ibm_eps=0.15)
    in_cells = _graph(ibm_eps=0.0, ibm_eps_cells=1.0)      # dx = 0.15
    assert "ibm_eps_cells" in in_cells.params["nodes"]["fluid"]
    assert "ibm_eps" not in in_cells.params["nodes"]["fluid"]
    np.testing.assert_array_equal(_drag(physical), _drag(in_cells))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _drag(_graph(ibm_eps=0.15, ibm_eps_cells=1.0), n=1)
