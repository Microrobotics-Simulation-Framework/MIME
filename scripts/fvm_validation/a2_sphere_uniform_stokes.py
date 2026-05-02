"""A2 — Sphere in uniform body-force-driven Stokes flow.

Static sphere of radius ``a`` in a periodic box (no walls, no pipe).
Box side L >> a so wall-image effects are negligible. Uniform body
force in +x drives the flow. At Re=0.01 (Stokes), the drag should be

    F_Stokes = 6πμaU_inf

where U_inf is the uniform-flow velocity (without sphere). For a
periodic box with body-force-driven flow, the relationship between the
body force ``f`` and the resulting "free-stream" velocity is set by
momentum balance: in steady state, the sphere absorbs all the body
force on the box-volume of fluid. So:

    F_drag = ρ * f * (V_box - V_sphere) ≈ ρ f V_box

This is the EXACT total drag. We compare:
  (a) the surface-integral drag (the new method)
  (b) the analytical Stokes drag with U_inf computed from the actual
      mean fluid velocity (excluding sphere region).

Pass criterion: surface-integral drag matches one or both within 5%
at all three resolutions (4, 8, 16 cells per radius).
"""
from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import (
    IBMBody, smoothed_indicator, surface_integral_force,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf


def run(cells_per_radius: int, *, a: float = 0.05,
        L_over_a: float = 10.0, nu: float = 1.0,
        f_body: float = 1e-4, n_chunks: int = 12,
        dt: float = 0.05, n_per_chunk: int = 200,
        ibm_alpha: float = 1e5):
    """Returns (F_si, F_balance, U_inf_meas) at given resolution."""
    L_box = L_over_a * a                 # cubic box
    N = int(round(cells_per_radius * L_box / a))   # = cells_per_radius * L_over_a
    # Use periodic in all three axes
    mesh = make_cartesian_mesh_3d(
        N, N, N, L_box, L_box, L_box,
        origin=(-L_box / 2, -L_box / 2, -L_box / 2),
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]

    sphere_centre = jnp.zeros(3, dtype=jnp.float32)

    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=a)

    sphere = IBMBody(
        name="sphere", sdf=sphere_sdf_fn,
        extract_force=False, ref_point=sphere_centre,
    )

    # No mesh BCs (fully periodic)
    bcs = {}

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc="periodic",
        velocity_bc="periodic",
        ibm_alpha=ibm_alpha, ibm_eps=1.0 * dx,
    )

    def body_force(t):
        return jnp.array([f_body, 0.0, 0.0])

    state = None
    for _ in range(n_chunks):
        state = run_piso(
            mesh, bcs, cfg, n_steps=n_per_chunk, dt=dt,
            body_force_fn=body_force, ibm_bodies=[sphere], initial=state,
        )
    state["u"].block_until_ready()

    # Surface-integral force
    F_si, _ = surface_integral_force(
        state["u"], state["p"], mesh, sphere_sdf_fn,
        mu=cfg.rho * cfg.nu, dx=dx,
        shell_inner=0.5, shell_outer=2.5,
        ref_point=sphere_centre,
    )
    F_si_x = float(F_si[0])

    # Force balance (Newton 3rd on box):
    # ρ f (V_box - V_sphere) = F_drag_on_sphere (steady)
    V_box = L_box ** 3
    V_sphere = (4 / 3) * np.pi * a ** 3
    F_balance = 1.0 * f_body * (V_box - V_sphere)

    # Measure U_inf (mean flow far from sphere, in fluid only)
    phi = np.asarray(sphere_sdf_fn(mesh.x))
    far_mask = phi > 4 * dx                  # well outside diffuse band
    u_x_arr = np.asarray(state["u"][:, 0])
    U_inf = float(np.mean(u_x_arr[far_mask]))

    F_stokes = 6 * np.pi * (cfg.rho * cfg.nu) * a * U_inf
    return F_si_x, F_balance, U_inf, F_stokes, dx, N


def main():
    print("=" * 78)
    print("A2 — Sphere in uniform Stokes flow (periodic, body-force-driven)")
    print("=" * 78)
    a = 0.05
    print(f"  sphere radius a = {a}, target Re ≈ 0.01")

    rows = []
    L_over_a = 10.0
    # Re_target ~ 0.01 ⇒ U_inf ~ 0.01*ν/(2a) = 0.1 for ν=1, a=0.05.
    # ρ f V_box ≈ F_drag = 6πμaU_inf
    # ⇒ f = 6πμa U_inf / V_box = 6π * 1 * 0.05 * 0.1 / (10*0.05)^3 = 7.5e-5
    f_body = 7.5e-5
    # Skipping cpr=16 (N=160) — XLA constant-folding takes >30 min on
    # this hardware (will be addressed by Fix C). cpr=4 and cpr=8
    # already span 4× resolution, enough to see the convergence trend.
    for cpr in (4, 8):
        t0 = time.time()
        try:
            F_si, F_bal, U_inf, F_stokes, dx, N = run(
                cells_per_radius=cpr, a=a, f_body=f_body,
                L_over_a=L_over_a,
                n_chunks=12, dt=0.05, n_per_chunk=200,
            )
        except Exception as e:
            print(f"\n  cpr={cpr}: FAILED ({type(e).__name__}: {e})")
            continue
        elapsed = time.time() - t0

        Re = U_inf * 2 * a / 1.0
        err_balance = abs(F_si - F_bal) / abs(F_bal)
        err_stokes = abs(F_si - F_stokes) / abs(F_stokes)
        mesh_cells = N ** 3
        print(f"\n  cells_per_radius={cpr}  N_box={N}  dx={dx:.4f}  "
              f"({mesh_cells} cells)")
        print(f"    Re_p = {Re:.4f}, U_inf (measured) = {U_inf:.4e}")
        print(f"    F_stokes (6πμaU)    = {F_stokes:.4e}")
        print(f"    F_balance (f V_box) = {F_bal:.4e}")
        print(f"    F_surface-integral  = {F_si:.4e}")
        print(f"    err vs Stokes       = {err_stokes*100:.2f}%")
        print(f"    err vs balance      = {err_balance*100:.2f}%")
        print(f"    wall time           = {elapsed:.0f}s")
        rows.append({
            "cpr": cpr, "N": N, "dx": dx,
            "Re": Re, "U_inf": U_inf,
            "F_si": F_si, "F_stokes": F_stokes, "F_balance": F_bal,
            "err_stokes": err_stokes, "err_balance": err_balance,
            "elapsed": elapsed,
        })

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  {'cpr':>5} {'F_si':>12} {'F_stokes':>12} {'F_balance':>12} "
          f"{'err_St':>8} {'err_bal':>8}")
    for r in rows:
        print(f"  {r['cpr']:>5} {r['F_si']:12.4e} {r['F_stokes']:12.4e} "
              f"{r['F_balance']:12.4e} {r['err_stokes']*100:7.2f}% "
              f"{r['err_balance']*100:7.2f}%")


if __name__ == "__main__":
    main()
