"""T2 — 2D Taylor-Green vortex energy decay.

Initial condition (period 2π in x, y):
    u(x,y,0) =  sin x cos y
    v(x,y,0) = -cos x sin y
    p(x,y,0) = (cos 2x + cos 2y) / 4

Analytical kinetic energy:  E(t) = E_0 * exp(-4 ν t).

Pass criteria:
  * E(t) is monotonically non-increasing (no spurious energy growth).
  * Final E(2.0) matches analytical to within 5%.

With ``gamma_conv=1.0`` (pure central convection) and the
implicit-diffusion Helmholtz step, the FVM gives 0.06% final error.
With ``gamma_conv=0.5`` (50% upwind) the upwind numerical viscosity
adds ~6% extra dissipation, which is the right qualitative behaviour
but fails the 5% bar — so this test fixes ``gamma_conv=1.0``.
"""
from __future__ import annotations

import jax
# Float64 is required: at gamma_conv=1.0 + N=64, the PISO + Helmholtz path
# accumulates ~21% spurious dissipation in float32 (GPU reduction-order
# noise compounds across n_steps≈158). Float64 reproduces the 0.06%
# analytical error quoted in the M0 brief.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.fvm import make_cartesian_mesh_2d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import (
    PisoConfig, run_piso_with_history, initial_state,
)


@pytest.mark.gpu
def test_taylor_green_energy_decay():
    N = 64
    L = 2 * np.pi
    nu = 0.01
    t_end = 2.0
    dx = L / N
    dt = 0.4 * dx                  # CFL ≈ 0.4 with U_max = 1
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps

    mesh = make_cartesian_mesh_2d(
        N, N, L, L, periodic_x=True, periodic_y=True,
        dtype=jnp.float64,
    )
    bcs = {}

    x = np.asarray(mesh.x[:, 0])
    y = np.asarray(mesh.x[:, 1])
    u0 = np.zeros((mesh.N_cells, 2), dtype=np.float64)
    u0[:, 0] = np.sin(x) * np.cos(y)
    u0[:, 1] = -np.cos(x) * np.sin(y)
    p0 = (np.cos(2 * x) + np.cos(2 * y)) / 4.0

    u_o = u0[np.asarray(mesh.owner)]
    u_n = u0[np.asarray(mesh.neighbour)]
    F0 = np.einsum("fd,fd->f",
                    0.5 * (u_o + u_n), np.asarray(mesh.Sf)).astype(np.float64)

    s0 = initial_state(mesh)
    s0 = {**s0,
          "u": jnp.asarray(u0),
          "p": jnp.asarray(p0.astype(np.float64)),
          "F": jnp.asarray(F0)}

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc=("periodic", "periodic"),
        velocity_bc=("periodic", "periodic"),
    )

    state, hist = run_piso_with_history(
        mesh, bcs, cfg, n_steps=n_steps, dt=dt,
        initial=s0, sample_every=1,
    )

    u_hist = np.asarray(hist["u"])
    t_hist = np.asarray(hist["t"])
    E_hist = 0.5 * np.mean(np.sum(u_hist ** 2, axis=-1), axis=-1)

    # Monotonicity (no spurious energy growth).
    diff = np.diff(E_hist)
    assert not np.any(diff > 1e-7), (
        f"E(t) increased at some step (max increase {diff.max():g})"
    )

    # Final energy vs analytical.
    E_final_ana = 0.25 * np.exp(-4 * nu * t_hist[-1])
    rel_err = abs(E_hist[-1] - E_final_ana) / E_final_ana
    assert rel_err < 0.05, (
        f"E({t_hist[-1]:.2f}) rel error {rel_err*100:.2f}% exceeds 5%"
    )
