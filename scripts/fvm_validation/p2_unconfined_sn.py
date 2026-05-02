"""P2 — Unconfined sphere drag vs Schiller-Naumann at Re_p ∈ {1, 10, 100}.

Sphere of radius a in a periodic cubic box of side L = 20a (so wall
images are negligible). Uniform body force in +x drives flow. At
steady state the body force input balances sphere drag; we measure
both to cross-check.

Drag is extracted via the surface-integral force (clean shell at
1.5–3.5 dx outside the body — past the IBM diffuse band).

Schiller-Naumann correlation:  C_D = (24/Re)(1 + 0.15 Re^0.687).
Pass: < 10% error at all Re.
"""
from __future__ import annotations
import time
import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import (
    IBMBody, surface_integral_force,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf


def schiller_naumann(Re):
    return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)


def run_unconfined(*, Re_p, a=0.05, L_over_a=12.0, cells_per_radius=6,
                   nu_target=0.005, n_chunks=10, n_per_chunk=200,
                   dt=0.05, ibm_alpha=1e5):
    """Returns (F_si_x, U_inf_meas, F_balance, dx, mesh.N_cells, elapsed)."""
    L = L_over_a * a
    N = int(round(cells_per_radius * L / a))
    print(f"  Re_p={Re_p}: L={L}, N={N} ({N**3} cells)", flush=True)

    # nu chosen so target U for given Re_p
    # Re_p = U_inf * 2a / nu  ⇒  for chosen U_inf, nu = U_inf * 2a / Re_p
    # We need a starting U_inf to fix nu. Pick U_inf so it's neither
    # tiny (slow convergence) nor large (CFL).
    # Take U_inf_target = 0.1, then nu = 0.1 * 0.1 / Re_p = 0.01 / Re_p.
    U_target = 0.1
    nu = U_target * 2 * a / Re_p

    # Body force: at steady state, ρf*V_box ≈ 6πμa·U·K_inertial.
    # Pick f so U_inf converges to U_target.
    # For SN inertial: F = 0.5*ρ*U²*πa²*C_D
    # ρ f V_box = F  ⇒  f = F / V_box = 0.5*U²*πa²*C_D / V_box
    C_D = schiller_naumann(Re_p)
    V_box = L ** 3
    f = 0.5 * U_target**2 * np.pi * a**2 * C_D / V_box

    mesh = make_cartesian_mesh_3d(
        N, N, N, L, L, L, origin=(-L/2, -L/2, -L/2),
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]

    sphere_centre = jnp.zeros(3, dtype=jnp.float32)
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=a)
    sphere = IBMBody(name="sphere", sdf=sphere_sdf_fn,
                      ref_point=sphere_centre)

    bcs = {}    # all periodic, no boundary patches

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc="periodic", velocity_bc="periodic",
        ibm_alpha=ibm_alpha, ibm_eps=1.0 * dx,
        transform_backend="dense",
    )
    def body_force(t):
        return jnp.array([f, 0.0, 0.0])

    state = None
    t0 = time.time()
    for _ in range(n_chunks):
        state = run_piso(mesh, bcs, cfg, n_steps=n_per_chunk, dt=dt,
                          body_force_fn=body_force, ibm_bodies=[sphere],
                          initial=state)
    state["u"].block_until_ready()
    elapsed = time.time() - t0

    # Surface integral
    F_si, _ = surface_integral_force(
        state["u"], state["p"], mesh, sphere_sdf_fn,
        mu=cfg.rho * cfg.nu, dx=dx,
        shell_inner=1.5, shell_outer=3.5,
        ref_point=sphere_centre,
    )
    F_x = float(F_si[0])

    # Mean velocity (excluding sphere region) as proxy for U_inf
    phi = np.asarray(sphere_sdf_fn(mesh.x))
    far = phi > 3 * dx
    U_inf = float(np.mean(np.asarray(state["u"][:, 0])[far]))

    # Force balance: ρ f V_box ≈ F_drag (steady)
    F_balance = 1.0 * f * V_box
    return F_x, U_inf, F_balance, dx, mesh.N_cells, elapsed, nu


def main():
    print("=" * 78)
    print("P2 — Unconfined sphere drag vs Schiller-Naumann")
    print("=" * 78)
    for Re_p in (1.0, 10.0, 100.0):
        try:
            F_x, U_inf, F_bal, dx, n_cells, elapsed, nu = run_unconfined(
                Re_p=Re_p, cells_per_radius=6, L_over_a=12.0, n_chunks=10,
            )
        except Exception as e:
            print(f"  Re_p={Re_p}: FAILED ({type(e).__name__}: {e})")
            continue
        Re_actual = U_inf * 2 * 0.05 / nu
        rho = 1.0; a = 0.05
        C_D_FVM = F_x / (0.5 * rho * U_inf**2 * np.pi * a**2)
        C_D_SN = schiller_naumann(Re_actual)
        err = abs(C_D_FVM - C_D_SN) / C_D_SN
        print(f"\n  Re_p_target={Re_p}, Re_p_measured={Re_actual:.2f}")
        print(f"    U_inf measured = {U_inf:.4e}")
        print(f"    F_si = {F_x:.4e}, F_balance (ρfV_box) = {F_bal:.4e}")
        print(f"    C_D_FVM = {C_D_FVM:.3f}")
        print(f"    C_D_SN  = {C_D_SN:.3f}  (at measured Re)")
        print(f"    err = {err*100:.1f}%   {'PASS' if err < 0.10 else 'FAIL'}")
        print(f"    wall time = {elapsed:.0f}s")


if __name__ == "__main__":
    main()
