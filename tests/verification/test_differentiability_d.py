"""Differentiability of the two-scale solver (Phase D).

On the Cartesian+IBM path the solver is differentiable **as-is** — the pressure
solve is the dense/FFT diagonalised (plain linear algebra, no ``fori_loop`` cache
issue), and the Schwarz coupling group's fixed-trip ``fori`` loop unrolls under
reverse-mode AD. So end-to-end gradients come straight from ``jax.grad`` through
``GraphManager._compiled_step`` (a pure ``(state, ext) → state`` function), with
no ``custom_vjp`` / IFT rewrite required. (``solver="ift"`` remains an *optional*
O(1)-memory adjoint for long horizons — a perf knob, not a correctness need.)

* **D1** — gradient through a forward Cartesian PISO solve (∂‖u‖²/∂body-force).
* **D2** — gradient through the *coupled* two-scale solve
  (∂drag/∂body-velocity), the gradient the dual-RPM controller (E4) needs.

Both match a central finite difference to ~machine precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.x64, pytest.mark.slow]


def test_d1_forward_piso_is_differentiable():
    from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
    from mime.nodes.environment.fvm.boundary import VelocityBC
    from mime.nodes.environment.fvm.piso import (
        PisoConfig, make_piso_step, initial_state)

    mesh = make_cartesian_mesh_3d(10, 10, 10, 1.0, 1.0, 1.0, dtype=jnp.float64)
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=1e-2, rho=1000.0, transform_backend="dense")

    def objective(gz, n=15):
        step = make_piso_step(mesh, bcs, cfg,
                              body_force_fn=lambda t: jnp.array([0.0, 0.0, gz]))
        st = initial_state(mesh)
        for _ in range(n):
            st = step(st, 1e-3)
        return jnp.mean(st["u"] ** 2)

    g0 = 1.0
    grad = float(jax.grad(objective)(g0))
    eps = 1e-3
    fd = float((objective(g0 + eps) - objective(g0 - eps)) / (2 * eps))
    assert abs(grad - fd) / abs(fd) < 1e-6


def test_d2_coupled_two_scale_is_differentiable():
    from maddening.core.graph_manager import GraphManager
    from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
    from mime.nodes.environment.fvm.boundary import VelocityBC
    from mime.nodes.environment.fvm.piso import PisoConfig
    from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
    from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
    from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
    from mime.nodes.environment.stokeslet.interface import create_interface_mesh
    from mime.nodes.environment.two_scale import make_two_scale_coupling

    a, rho, nu, dt, L, N = 1e-3, 1000.0, 1e-3, 1e-4, 8e-3, 20
    mu = rho * nu
    mesh = make_cartesian_mesh_3d(N, N, N, L, L, L,
                                  origin=(-L / 2, -L / 2, -L / 2), dtype=jnp.float64)
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=nu, rho=rho, transform_backend="dense")
    body = sphere_surface_mesh(center=(0, 0, 0), radius=a, n_refine=2)
    nb = body.n_points
    far = FVMFluidNode("fvm", dt, mesh=mesh, bcs=bcs, cfg=cfg,
                       n_sample_points=nb, n_forcing_points=nb,
                       forcing_sigma=1.5 * (L / N))
    near = StokesletFluidNode("bem", dt, mu=mu, body_mesh=body,
                              interface_mesh=create_interface_mesh(
                                  radius=2 * a, n_refine=2))
    gm = GraphManager()
    info = make_two_scale_coupling(gm, far, near,
                                   body_points=jnp.asarray(body.points),
                                   body_weights=jnp.asarray(body.weights))
    for fld, sh in (("body_velocity", (3,)), ("body_angular_velocity", (3,)),
                    ("body_orientation", (4,))):
        gm.add_external_input("bem", fld, shape=sh, dtype=jnp.float64)
    gm.compile()
    step, s0 = gm._compiled_step, gm._state
    pts = info["geometry_inputs"]["fvm"]

    def drag_after(Vz, nsteps=4):
        ext = {"fvm": {"sample_points": pts["sample_points"],
                       "forcing_points": pts["forcing_points"]},
               "bem": {"body_velocity": jnp.array([0.0, 0.0, Vz]),
                       "body_angular_velocity": jnp.zeros(3),
                       "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])}}
        s = s0
        for _ in range(nsteps):
            s = step(s, ext)
        return s["bem"]["drag_force"][2]

    V = 1e-3
    grad = float(jax.grad(drag_after)(V))
    eps = 1e-5
    fd = float((drag_after(V + eps) - drag_after(V - eps)) / (2 * eps))
    assert abs(grad - fd) / abs(fd) < 1e-5
