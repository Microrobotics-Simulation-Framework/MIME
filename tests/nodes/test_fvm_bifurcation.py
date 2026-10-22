"""Bifurcation vessel geometry via CSG + IBM (Cartesian far-field).

The Cartesian+IBM far-field handles a Y-junction with no body-fitted meshing: the
vessel lumen is a CSG union of capsules (``make_bifurcation_ibm_sdf``) used as
the Brinkman IBM body — the same machinery as the straight pipe. Pins the SDF
geometry (fluid lumen vs solid wall) and that the FVM far-field runs **stably**
on the bifurcation (the payoff of the Cartesian+IBM decision over body-fitted
tets), confining flow to the lumen and developing flow in *both* branches.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.fvm.sdf import make_bifurcation_ibm_sdf

_A = 1e-3
_S = np.sqrt(0.5)
_B1 = np.array([_S, 0.0, _S])     # branch 1: +x, +z (45°)
_B2 = np.array([-_S, 0.0, _S])    # branch 2: -x, +z (45°)


def _sdf():
    return make_bifurcation_ibm_sdf(
        junction=(0.0, 0.0, 0.0), parent_axis=(0.0, 0.0, 1.0),
        branch_axes=[_B1, _B2], radius=_A,
        parent_length=4 * _A, branch_length=4 * _A)


def test_bifurcation_sdf_geometry():
    sdf = _sdf()
    # fluid (sdf>0) along the parent axis and along each branch axis
    on_parent = jnp.array([[0.0, 0.0, -2 * _A]])
    on_b1 = jnp.array([2 * _A * _B1])
    on_b2 = jnp.array([2 * _A * _B2])
    for p in (on_parent, on_b1, on_b2):
        assert float(sdf(p)[0]) > 0.0
    # solid (sdf<0) well off all axes
    off = jnp.array([[3 * _A, 3 * _A, 0.0]])
    assert float(sdf(off)[0]) < 0.0
    # the two branches are distinct lumens: the midpoint between their tips is
    # solid (the Y opens up)
    mid_tips = jnp.array([0.5 * (4 * _A * _B1 + 4 * _A * _B2)])  # on the z-axis high up
    # that midpoint lies on the parent/z-axis extension above the junction —
    # outside both branch capsules → solid
    assert float(sdf(mid_tips)[0]) < 0.0


@pytest.mark.slow
def test_far_field_runs_and_flows_into_both_branches():
    from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
    from mime.nodes.environment.fvm.boundary import VelocityBC
    from mime.nodes.environment.fvm.piso import PisoConfig
    from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
    from mime.nodes.environment.fvm.ibm import IBMBody
    import jax

    jax.config.update("jax_enable_x64", True)
    dx = _A / 4
    Lo = 5 * _A
    n = int(np.ceil(2 * Lo / dx))
    mesh = make_cartesian_mesh_3d(n, n, n, 2 * Lo, 2 * Lo, 2 * Lo,
                                  origin=(-Lo, -Lo, -Lo), dtype=jnp.float64)
    wall = IBMBody(name="vessel", sdf=_sdf())
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=1e-2, rho=1000.0, ibm_alpha=1e5, ibm_eps=dx,
                     transform_backend="dense")
    node = FVMFluidNode("vessel_flow", 1e-4, mesh=mesh, bcs=bcs, cfg=cfg,
                        static_bodies=[wall],
                        body_force_fn=lambda t: jnp.array([0.0, 0.0, 1.0]))
    step = jax.jit(lambda s: node.update(s, {}, 1e-4))
    st = node.initial_state()
    for _ in range(150):
        st = step(st)
    u = np.asarray(jax.block_until_ready(st)["u"])
    x = np.asarray(mesh.x)
    assert np.isfinite(u).all()                       # stable on the bifurcation

    sdf_vals = np.asarray(_sdf()(mesh.x))
    lumen = sdf_vals > 0.25 * dx
    solid = sdf_vals < -2 * dx
    # IBM confines flow to the lumen: solid region is ~quiescent vs the lumen
    assert np.max(np.abs(u[solid])) < 0.1 * np.max(np.abs(u[lumen]))
    # flow develops in BOTH branches (cells near each branch axis, above junction)
    for b in (_B1, _B2):
        axdist = np.linalg.norm(np.cross(x - 0, b), axis=1)  # dist to branch line
        along = x @ b
        sel = lumen & (axdist < 1.2 * _A) & (along > 1.5 * _A)
        assert sel.sum() > 0
        assert np.mean(u[sel] @ b) > 0.0              # outward flow along branch
