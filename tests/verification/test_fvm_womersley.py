"""M1 deliverable — pulsatile (Womersley) channel flow benchmark.

Validates the transient PISO solver end-to-end on the canonical
analytical solution for body-force-driven oscillatory flow between
parallel plates (the "channel" form of the Womersley 1955 result).

Physics:
  * Channel between ``y = ±h`` (h = 1 m).
  * Periodic in x; ``Nx`` cells thick (≥ 2) so the periodic-mass-flux
    pathway is exercised but the dynamics is one-dimensional in y.
  * No-slip at ``y = ±h``.
  * Spatially uniform x-momentum body force per unit mass:
        f_x(t) = f_steady + f_osc · cos(ωt).

Dimensionless setup:
  * Womersley number ``Wo = h √(ω/ν) = 7``.
  * Reynolds number ``Re = U_mean · 2h / ν = 200`` (mean Poiseuille
    velocity from ``f_steady``).

Pass criteria (per the M1 brief):
  * Velocity at y = 0 and y = 0.8 h matches analytical Womersley
    amplitude and phase to within 2%.
  * Autodiff (``jax.grad``) continues to flow through the full PISO
    transient loop.

Implementation notes:
  * We initialise from the analytical solution at t = 0 to bypass the
    O(h²/ν) startup transient. The test then verifies that the solver
    *maintains* periodic Womersley flow over 3 full cycles.
  * The implicit-diffusion Helmholtz solve (DST + periodic FFT
    factorisation) makes the time step unconditionally stable for the
    diffusion part — necessary to resolve the Womersley boundary
    layer with reasonable dt at Wo = 7.
"""

from __future__ import annotations

import jax
# Float64 + tighter dt required: PISO's explicit convection step has a
# CFL constraint (only the diffusion step is unconditionally stable
# from the Helmholtz inverse). The original n_per_cycle=80 put CFL at
# ~7.7 — the scheme was unstable in float64 and only "stable" in
# float32 because reduction-order noise on GPU damped the unstable
# mode (with ~26% amplitude error). At n_per_cycle=640 CFL drops to
# ~0.96 and the result converges to <0.1% amplitude error.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.fft as sfft

from mime.nodes.environment.fvm import make_cartesian_mesh_2d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import (
    PisoConfig, run_piso, run_piso_with_history, initial_state,
)
from mime.nodes.environment.fvm.womersley import channel_velocity


def _build_channel(Nx=4, Ny=64, h=1.0):
    Lx = Nx * (2 * h / Ny)            # square cells
    Ly = 2 * h
    mesh = make_cartesian_mesh_2d(
        Nx, Ny, Lx, Ly, origin=(0.0, -h), periodic_x=True,
        dtype=jnp.float64,
    )
    zero_vel = jnp.zeros((Nx, 2), dtype=jnp.float64)
    zero_F = jnp.zeros((Nx,), dtype=jnp.float64)
    bcs = {
        "y_min": VelocityBC(u_wall=zero_vel, F_through=zero_F),
        "y_max": VelocityBC(u_wall=zero_vel, F_through=zero_F),
    }
    return mesh, bcs


def _flat_idx(ix, iy, Ny): return ix * Ny + iy


@pytest.mark.gpu
def test_pulsatile_channel_wo7_re200_matches_womersley():
    h, nu = 1.0, 0.001
    omega = 49.0 * nu / h ** 2          # Wo = 7
    T = 2 * np.pi / omega
    f_steady = 3e-4                     # gives U_mean = 0.1, Re = 200
    f_osc = f_steady

    Nx, Ny = 4, 64
    mesh, bcs = _build_channel(Nx, Ny, h)

    def body_force(t):
        fx = f_steady + f_osc * jnp.cos(omega * t)
        return jnp.array([fx, 0.0])

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=0.5, n_corrector=2,
        pressure_bc=("periodic", "neumann"),
        velocity_bc=("periodic", "dirichlet"),
    )

    # Initialise from analytical Womersley at t = 0
    y_cells = (np.arange(Ny) + 0.5) * (2 * h / Ny) - h
    u0_y = channel_velocity(y_cells, 0.0, h=h, nu=nu, omega=omega,
                            f_steady=f_steady, f_osc=f_osc)
    u_init = np.zeros((Nx * Ny, 2), dtype=np.float64)
    for ix in range(Nx):
        u_init[ix * Ny:(ix + 1) * Ny, 0] = u0_y
    s0 = initial_state(mesh)
    state0 = {**s0, "u": jnp.asarray(u_init, dtype=mesh.V.dtype)}

    # CFL constraint: U_max · dt / dy ≲ 1.  U_max ≈ 0.15, dy = 2h/64 =
    # 0.03125 ⇒ dt ≤ 0.21 s. n_per_cycle = 640 gives dt ≈ 0.2 s, CFL ≈
    # 0.96.  At the test's original n_per_cycle=80 the convection step
    # was at CFL ≈ 7.7 and the scheme was unstable in float64.
    n_per_cycle = 640
    dt = T / n_per_cycle
    n_cycles = 3
    state, hist = run_piso_with_history(
        mesh, bcs, cfg, n_steps=n_cycles * n_per_cycle, dt=dt,
        body_force_fn=body_force, initial=state0, sample_every=1,
    )

    ts = np.asarray(hist["t"])
    last = ts > (n_cycles - 1) * T  # last cycle for amplitude/phase

    for y_target_frac in (0.0, 0.8):
        iy = int(np.argmin(np.abs(y_cells - y_target_frac * h)))
        u_num = np.asarray(hist["u"][:, _flat_idx(0, iy, Ny), 0])
        u_ana = np.array([
            channel_velocity(np.array([y_cells[iy]]), tt,
                             h=h, nu=nu, omega=omega,
                             f_steady=f_steady, f_osc=f_osc)[0]
            for tt in ts
        ])

        amp_num = (u_num[last].max() - u_num[last].min()) / 2
        amp_ana = (u_ana[last].max() - u_ana[last].min()) / 2
        amp_rel_err = abs(amp_num - amp_ana) / amp_ana
        assert amp_rel_err < 0.02, (
            f"y/h={y_target_frac}: amplitude error {amp_rel_err*100:.2f}% "
            f"exceeds 2% (num={amp_num:g}, ana={amp_ana:g})"
        )

        # Phase via dominant FFT mode (oscillatory part only)
        osc_num = u_num[last] - u_num[last].mean()
        osc_ana = u_ana[last] - u_ana[last].mean()
        Fn = sfft.rfft(osc_num); Fa = sfft.rfft(osc_ana)
        k = int(np.argmax(np.abs(Fa)))
        phase_err = (np.angle(Fn[k]) - np.angle(Fa[k]))
        # wrap to [−π, π]
        phase_err = (phase_err + np.pi) % (2 * np.pi) - np.pi
        # 2% of one cycle = 7.2°
        assert abs(np.degrees(phase_err)) < 7.2, (
            f"y/h={y_target_frac}: phase error {np.degrees(phase_err):.2f}° "
            f"exceeds 2% of cycle (7.2°)"
        )


@pytest.mark.gpu
def test_pulsatile_channel_grad_through_piso():
    """jax.grad of a flow functional must be finite through PISO.

    Verifies autodiff transparency through the full transient PISO
    integration (FFT-diagonalised pressure + Helmholtz, projection
    correction).
    """
    h, nu = 1.0, 0.001
    omega = 49.0 * nu / h ** 2
    T = 2 * np.pi / omega

    Nx, Ny = 4, 32        # smaller grid for cheap FD reference
    mesh, bcs = _build_channel(Nx, Ny, h)

    def kinetic(f_amp: jnp.ndarray) -> jnp.ndarray:
        def body_force(t):
            return jnp.array([f_amp + f_amp * jnp.cos(omega * t), 0.0])
        cfg = PisoConfig(
            nu=nu, rho=1.0, gamma_conv=0.5, n_corrector=2,
            pressure_bc=("periodic", "neumann"),
            velocity_bc=("periodic", "dirichlet"),
        )
        dt = T / 40
        state = run_piso(mesh, bcs, cfg, n_steps=40, dt=dt,
                         body_force_fn=body_force)
        return 0.5 * jnp.sum(state["u"] ** 2)

    f0 = jnp.asarray(3e-4, dtype=jnp.float32)
    grad_ad = float(jax.grad(kinetic)(f0))
    eps = 1e-5
    grad_fd = (float(kinetic(f0 + eps)) - float(kinetic(f0 - eps))) / (2 * eps)
    rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-6)
    # float32 noise dominates; tight tolerance not realistic here
    assert rel < 5e-2, (
        f"jax.grad disagreed with FD: AD={grad_ad:g}, FD={grad_fd:g}, "
        f"rel={rel:g}"
    )
