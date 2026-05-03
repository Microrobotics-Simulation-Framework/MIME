"""T3 — confined-Stokes drag at λ ∈ {0.1, 0.3} on isotropic-cpr mesh.

Re-run after Fix 1 (isotropic mesh) and Fix 2 (BC clearance).

Setup
-----
* Steady Poiseuille (no oscillation) driven by lifting at U_dc.
* Stokes regime: Re_R = U·R/ν ≪ 1 → Re_R = 0.001 here (U=1e-3, R=10·r_b).
* Spherical body radius r_b at the centerline.
* Pipe length L = 22·r_b (Fix 2 minimum at sphere_margin=5, bc_margin=5).
* Mesh isotropic ``dx = r_b/cpr``.

Outputs
-------
* K_FVM = F_md / F_unconfined_Stokes  vs  K_Happel(λ) for each λ.
* Acceptance per the brief:
    λ=0.1 — K_FVM > 0 and converges toward K_Happel(0.1)=1.27 from below
    λ=0.3 — K_FVM within 5% of K_Happel(0.3)=1.75
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_pipe_mesh
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import IBMBody, momentum_deficit_drag
from mime.nodes.environment.fvm.sdf import sphere_sdf
from mime.nodes.environment.fvm.lifting import (
    make_poiseuille_lift, make_poiseuille_p_lift,
)


def happel_brenner(lam: float) -> float:
    return 1.0 / (1.0 - 2.10443*lam + 2.08877*lam**3
                  - 0.94813*lam**5 - 1.372*lam**6
                  + 3.87*lam**8 - 4.19*lam**10)


def run_one(lam: float, cpr: int, U_dc: float = 1e-3, n_warmup: int = 800):
    print("=" * 78)
    print(f"T3 — λ = {lam}, cpr = {cpr}, U_dc = {U_dc} m/s")
    print("=" * 78)

    r_b = 1e-3
    R_pipe = r_b / lam
    sphere_margin = 5.0
    bc_margin = 5.0
    L_pipe = 2.0 * (sphere_margin + bc_margin) * r_b + 2.0 * r_b   # = 22 r_b
    nu = 1e-3
    rho = 1.0
    mu = rho * nu
    Re = U_dc * R_pipe / nu
    K_h = happel_brenner(lam)
    print(f"  R_pipe = {R_pipe*1e3:.2f} mm, r_b = {r_b*1e3} mm, "
          f"L_pipe = {L_pipe*1e3:.2f} mm")
    print(f"  Re(R) = {Re:.3e}  (Stokes regime),  K_Happel({lam}) = {K_h:.4f}")

    mesh = make_pipe_mesh(
        pipe_radius=R_pipe, pipe_length=L_pipe,
        robot_radius=r_b, cpr=cpr,
        periodic_x=False, periodic_y=False, periodic_z=False,
    )
    dx = mesh.cartesian_spacing[0]
    Nz = mesh.cartesian_shape[2]
    L_actual = Nz * dx
    print(f"  mesh {mesh.cartesian_shape} ({mesh.N_cells} cells, "
          f"dx = {dx*1e3:.4f} mm, cpr = {r_b/dx:.1f})")
    print(f"  L_pipe actual = {L_actual*1e3:.3f} mm")

    sphere_centre = jnp.array([0.0, 0.0, L_actual / 2], dtype=mesh.V.dtype)
    def pipe_wall_sdf(x):
        rxy = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-30)
        return R_pipe - rxy
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_b)
    bodies = [
        IBMBody(name="pipe_wall", sdf=pipe_wall_sdf),
        IBMBody(name="sphere",    sdf=sphere_sdf_fn),
    ]

    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        nb = int(mesh.patch(name).owner.size)
        bcs[name] = VelocityBC(
            u_wall=jnp.zeros((nb, 3)), F_through=jnp.zeros((nb,)),
        )

    cfg = PisoConfig(
        nu=nu, rho=rho, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )

    L_lift = make_poiseuille_lift(mesh, R_pipe=R_pipe, U_mean=U_dc, axis=2)

    print(f"  Running PISO ({n_warmup} steps)...", flush=True)
    t0 = time.time()
    dt = min(0.5, 0.5 * dx / max(2*U_dc, 1e-30))   # CFL-bounded but big
    state = run_piso(
        mesh, bcs, cfg, n_steps=n_warmup, dt=dt,
        body_force_fn=None, ibm_bodies=bodies, lifting=L_lift,
    )
    state["u"].block_until_ready()
    wall = time.time() - t0
    print(f"    PISO {n_warmup} steps in {wall:.0f}s "
          f"({wall/n_warmup*1e3:.1f} ms/step), dt = {dt:.2e} s")

    u_phys = state["u"] + L_lift.u_lift_static
    p_lift_fn = make_poiseuille_p_lift(mu=mu, U_mean=U_dc, pipe_radius=R_pipe)
    F_md = float(momentum_deficit_drag(
        u_phys, state["p"], mesh,
        sphere_centre=sphere_centre, sphere_radius=r_b,
        pipe_radius=R_pipe, pipe_axis=2, rho=rho,
        sphere_margin=sphere_margin, bc_margin=bc_margin,
        body_force=0.0, mu=mu,
        p_lift_fn=p_lift_fn, U_mean_analytical=U_dc,
    ))

    # Also report the centerline velocity at a plane upstream of the sphere
    u_arr = np.asarray(u_phys).reshape(mesh.cartesian_shape + (3,))
    Nx, Ny, Nz_ = mesh.cartesian_shape
    iz_far = Nz_ // 4   # well upstream of sphere
    U_centre_meas = float(u_arr[Nx//2, Ny//2, iz_far, 2])

    F_stokes_unconfined = 6.0 * np.pi * mu * r_b * U_centre_meas
    K_FVM = F_md / F_stokes_unconfined if abs(F_stokes_unconfined) > 1e-30 else 0.0

    print(f"  U_centre measured (z = L/4) = {U_centre_meas:.4e} m/s "
          f"(target {2*U_dc:.4e})")
    print(f"  F_md             = {F_md:.4e} N")
    print(f"  F_Stokes (uncon) = {F_stokes_unconfined:.4e} N")
    print(f"  K_FVM            = {K_FVM:.4f}")
    print(f"  K_Happel({lam})  = {K_h:.4f}")
    print(f"  err vs Happel    = {abs(K_FVM-K_h)/K_h*100:.2f}%")
    return {"lam": lam, "cpr": cpr, "K_FVM": K_FVM, "K_Happel": K_h,
            "F_md": F_md, "U_centre": U_centre_meas, "wall_s": wall}


def main():
    results = []
    # Try cpr=6 if memory allows; fall back to cpr=4 on OOM
    for lam in (0.1, 0.3):
        for cpr in (4,):
            try:
                r = run_one(lam, cpr=cpr)
                results.append(r)
            except Exception as e:
                print(f"  FAILED (λ={lam}, cpr={cpr}): {type(e).__name__}: {e}")
                results.append({"lam": lam, "cpr": cpr, "FAILED": str(e)})

    print("\n" + "=" * 78)
    print("T3 SUMMARY")
    print("=" * 78)
    print(f"{'λ':>6} {'cpr':>4} {'K_FVM':>10} {'K_Happel':>10} {'err %':>8}")
    for r in results:
        if "FAILED" in r:
            print(f"  {r['lam']:.2f}  {r['cpr']:>4d}     FAILED")
        else:
            err = abs(r["K_FVM"] - r["K_Happel"]) / r["K_Happel"] * 100
            print(f"  {r['lam']:.2f}  {r['cpr']:>4d}  "
                  f"{r['K_FVM']:>10.4f}  {r['K_Happel']:>10.4f}  {err:>7.2f}%")


if __name__ == "__main__":
    main()
