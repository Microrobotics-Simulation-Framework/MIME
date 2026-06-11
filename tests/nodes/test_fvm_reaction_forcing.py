"""FVM near→far reaction forcing (A2) — the near-field-reaction transfer.

Pins :func:`spread_point_forces` (point forces → regularized body
acceleration, momentum-conserving) and the ``FVMFluidNode`` wiring
(``n_forcing_points`` exposes ``forcing_points`` / ``forcing_values`` inputs).
The physics test confirms a point force drives a sign-correct flow and that the
response scales linearly with the force (Stokes linearity) — the property the
two-scale Schwarz coupling relies on.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.fvm.integrator import spread_point_forces
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.fluid_node import FVMFluidNode

pytestmark = pytest.mark.x64


def _box(n=16, L=1.0):
    return make_cartesian_mesh_3d(n, n, n, L, L, L, dtype=jnp.float64)


# ── spread_point_forces ─────────────────────────────────────────────────────

def test_spread_conserves_total_force():
    mesh = _box(16)
    rho = 1000.0
    points = jnp.array([[0.5, 0.5, 0.5], [0.3, 0.6, 0.4]])
    values = jnp.array([[0.0, 0.0, 2.0e-6], [1.0e-6, 0.0, 0.0]])
    accel = spread_point_forces(points, values, mesh, sigma=3 * (1.0 / 16),
                                rho=rho)
    # Total injected force = ∫ρ·a dV must equal Σ values (third-law balance).
    total = jnp.sum(rho * accel * mesh.V[:, None], axis=0)
    assert np.allclose(np.asarray(total), np.asarray(jnp.sum(values, axis=0)),
                       rtol=1e-6, atol=1e-12)


def test_spread_is_localized():
    mesh = _box(16)
    p = jnp.array([[0.5, 0.5, 0.5]])
    v = jnp.array([[0.0, 0.0, 1.0e-6]])
    accel = spread_point_forces(p, v, mesh, sigma=1.5 * (1.0 / 16), rho=1000.0)
    mag = jnp.linalg.norm(accel, axis=1)
    dist = jnp.linalg.norm(mesh.x - p[0], axis=1)
    near = mag[dist < 0.15]
    far = mag[dist > 0.4]
    assert float(jnp.max(near)) > 0.0
    assert float(jnp.max(far)) < 1e-2 * float(jnp.max(near))


# ── node wiring ─────────────────────────────────────────────────────────────

def _node(mesh, **kw):
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=1e-2, rho=1000.0)
    return FVMFluidNode("fvm", 1e-3, mesh=mesh, bcs=bcs, cfg=cfg, **kw)


def test_node_exposes_forcing_ports():
    node = _node(_box(8), n_forcing_points=2)
    spec = node.boundary_input_spec()
    assert spec["forcing_points"].shape == (2, 3)
    assert spec["forcing_values"].shape == (2, 3)


# ── physics: sign + Stokes linearity ────────────────────────────────────────

def _run(node, mesh, force_z, n_steps=120):
    state = node.initial_state()
    p = jnp.array([[0.5, 0.5, 0.5]])
    v = jnp.array([[0.0, 0.0, force_z]])
    bi = {"forcing_points": p, "forcing_values": v}
    for _ in range(n_steps):
        state = node.update(state, bi, 1e-3)
    # velocity at the centre cell
    c = int(jnp.argmin(jnp.linalg.norm(mesh.x - p[0], axis=1)))
    return np.asarray(state["u"][c])


@pytest.mark.slow
def test_point_force_drives_flow_sign_and_linearity():
    mesh = _box(20)
    node = _node(mesh, n_forcing_points=1, forcing_sigma=2.5 / 20)
    u1 = _run(node, mesh, 5.0e-4)
    u2 = _run(node, mesh, 1.0e-3)
    # Sign: +z force drives +z flow at the forcing point.
    assert u1[2] > 0.0
    # Stokes linearity: doubling the force doubles the response (low Re).
    assert np.allclose(u2, 2.0 * u1, rtol=0.05, atol=1e-9)
    # Transverse components stay tiny (axisymmetric forcing).
    assert abs(u1[0]) < 0.1 * u1[2] and abs(u1[1]) < 0.1 * u1[2]
