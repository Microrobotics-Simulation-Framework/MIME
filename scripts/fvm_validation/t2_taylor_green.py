"""T2 — Taylor-Green vortex 2D energy decay.

Initial condition (period 2π in x, y):
    u(x,y,0) =  sin x cos y
    v(x,y,0) = -cos x sin y
    p(x,y,0) = (cos 2x + cos 2y) / 4

Analytical kinetic energy:  E(t) = E_0 * exp(-4 ν t)  with E_0 = 0.25.

Pass criteria:
  (1) E(t) is monotonically non-increasing at every step.
  (2) Final E(2) matches analytical to within 5%.
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_2d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import (
    PisoConfig, run_piso_with_history, initial_state,
)


def main():
    print("=" * 72)
    print("T2 — 2D Taylor-Green vortex energy decay")
    print("=" * 72)
    N = 64
    L = 2 * np.pi
    nu = 0.01
    t_end = 2.0
    # CFL with U_max = 1: dt < 0.5 * dx / U_max
    dx = L / N
    dt = 0.4 * dx
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    print(f"  N={N}, dx={dx:.4f}, dt={dt:.4f}, n_steps={n_steps}, t_end={t_end}")

    mesh = make_cartesian_mesh_2d(N, N, L, L,
                                  periodic_x=True, periodic_y=True)
    bcs = {}  # no boundary patches under double-periodic

    # Initial condition
    x = np.asarray(mesh.x[:, 0])
    y = np.asarray(mesh.x[:, 1])
    u0 = np.zeros((mesh.N_cells, 2), dtype=np.float32)
    u0[:, 0] = np.sin(x) * np.cos(y)
    u0[:, 1] = -np.cos(x) * np.sin(y)
    p0 = (np.cos(2 * x) + np.cos(2 * y)) / 4.0

    # Initial face mass flux (consistent with cell-centred velocity).
    # We compute it as Rhie-Chow average for the initial F.
    # Easier: initialise F from u_face via simple averaging.
    u_o = u0[np.asarray(mesh.owner)]
    u_n = u0[np.asarray(mesh.neighbour)]
    u_face = 0.5 * (u_o + u_n)
    F0 = np.einsum("fd,fd->f", u_face, np.asarray(mesh.Sf)).astype(np.float32)

    s0 = initial_state(mesh)
    s0 = {
        **s0,
        "u": jnp.asarray(u0),
        "p": jnp.asarray(p0.astype(np.float32)),
        "F": jnp.asarray(F0),
    }

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc=("periodic", "periodic"),
        velocity_bc=("periodic", "periodic"),
    )

    t0 = time.time()
    state, hist = run_piso_with_history(
        mesh, bcs, cfg, n_steps=n_steps, dt=dt,
        initial=s0, sample_every=1,
    )
    state["u"].block_until_ready()
    print(f"  wall time: {time.time()-t0:.1f}s")

    # Energy series
    u_hist = np.asarray(hist["u"])  # (n_steps, N_cells, 2)
    t_hist = np.asarray(hist["t"])
    E_hist = 0.5 * np.mean(np.sum(u_hist ** 2, axis=-1), axis=-1)
    E0 = 0.5 * np.mean(np.sum(u0 ** 2, axis=-1))
    E_ana = E0 * np.exp(-4 * nu * t_hist)

    # Monotonicity
    diff = np.diff(E_hist)
    n_increases = int(np.sum(diff > 1e-12))
    print(f"\n  E(0) initial            : {E0:.6f}")
    print(f"  E(0) analytical         : 0.25")
    print(f"  monotonicity violations : {n_increases} (out of {len(diff)} steps)")
    monotone_pass = n_increases == 0

    # Final
    E_final = E_hist[-1]
    E_final_ana = E_ana[-1]
    rel_err = abs(E_final - E_final_ana) / E_final_ana
    print(f"  E({t_hist[-1]:.2f}) numerical   : {E_final:.6e}")
    print(f"  E({t_hist[-1]:.2f}) analytical  : {E_final_ana:.6e}")
    print(f"  relative error          : {rel_err*100:.2f}%")
    final_pass = rel_err < 0.05

    # Sample E(t) at a few times
    print(f"\n  E(t) curve:")
    sample_idx = np.linspace(0, len(t_hist) - 1, 11).astype(int)
    for i in sample_idx:
        print(f"    t={t_hist[i]:.3f}  num={E_hist[i]:.6e}  ana={E_ana[i]:.6e}  "
              f"err={(E_hist[i]-E_ana[i])/E_ana[i]*100:+.2f}%")

    print(f"\n  PASS (monotone)        : {'PASS' if monotone_pass else 'FAIL'}")
    print(f"  PASS (final < 5%)      : {'PASS' if final_pass else 'FAIL'}")

    print(f"\nSummary:")
    print(f"  T2 Taylor-Green: monotone={monotone_pass}, final_rel_err={rel_err*100:.2f}%  "
          f"({'PASS' if (monotone_pass and final_pass) else 'FAIL'})")


if __name__ == "__main__":
    main()
