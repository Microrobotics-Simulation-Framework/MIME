"""T3 — confined-Stokes drag via surface_integral_force.

The momentum_deficit_drag estimator under-samples the Stokes pressure
dipole at ±5 r_b (the dipole decays as 1/r²; only ~3 % of signal at
the integration planes — see drag-diagnostic sprint). Surface integral
samples Cauchy stress on a 2-cell shell just outside the IBM body,
where the dipole is large.
"""
from __future__ import annotations
import time
import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_pipe_mesh, make_poiseuille_lift, make_poiseuille_p_lift,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import (
    IBMBody, surface_integral_force,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf


def happel_brenner(lam):
    return 1.0/(1.0-2.10443*lam+2.08877*lam**3-0.94813*lam**5
                -1.372*lam**6+3.87*lam**8-4.19*lam**10)


def shell_geometry_check(R_pipe, r_b, dx, label=""):
    gap_cells           = (R_pipe - r_b) / dx
    shell_outer_axis    = r_b/dx + 3.5
    wall_ibm_inner_axis = R_pipe/dx - 2.0
    clearance           = wall_ibm_inner_axis - shell_outer_axis
    print(f"  shell geom {label}: gap_cells={gap_cells:.1f}, "
          f"shell_outer={shell_outer_axis:.1f}, "
          f"wall_inner={wall_ibm_inner_axis:.1f}, "
          f"clearance={clearance:.1f} cells")
    return clearance


def run_one(*, lam, cpr, U_dc=1e-3, n_steps=800,
            shell=(1.5, 3.5), label_extra=""):
    print("=" * 78)
    print(f"T3 surface_integral — λ={lam}, cpr={cpr}, "
          f"shell={shell} {label_extra}")
    print("=" * 78)
    r_b = 1e-3
    R_pipe = r_b/lam
    sphere_margin = 5.0; bc_margin = 5.0
    L_pipe = 2.0*(sphere_margin+bc_margin)*r_b + 2.0*r_b   # 22 r_b
    nu = 1e-3; rho = 1.0
    mu = rho*nu
    K_h = happel_brenner(lam)

    mesh = make_pipe_mesh(pipe_radius=R_pipe, pipe_length=L_pipe,
                          robot_radius=r_b, cpr=cpr)
    dx = mesh.cartesian_spacing[0]
    Nx, Ny, Nz = mesh.cartesian_shape
    L_actual = Nz * dx
    sphere_centre = jnp.array([0.0, 0.0, L_actual/2], dtype=mesh.V.dtype)
    print(f"  mesh {mesh.cartesian_shape} = {mesh.N_cells} cells, "
          f"dx={dx*1e3:.4f}mm")
    clearance = shell_geometry_check(R_pipe, r_b, dx, label=f"λ={lam}")
    if clearance < 2:
        raise RuntimeError(
            f"Shell clearance {clearance:.1f} < 2 cells — pipe-wall IBM "
            f"would contaminate the extraction shell. Increase cpr."
        )

    def pipe_wall_sdf(x):
        rxy = jnp.sqrt(x[..., 0]**2+x[..., 1]**2+1e-30)
        return R_pipe - rxy
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_b)
    bodies = [
        IBMBody(name="pipe_wall", sdf=pipe_wall_sdf),
        IBMBody(name="sphere",    sdf=sphere_sdf_fn),
    ]

    bcs = {}
    for name in ("x_min","x_max","y_min","y_max","z_min","z_max"):
        nb = int(mesh.patch(name).owner.size)
        bcs[name] = VelocityBC(u_wall=jnp.zeros((nb,3)),
                                F_through=jnp.zeros((nb,)))

    cfg = PisoConfig(
        nu=nu, rho=rho, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0*dx,
    )
    L_lift = make_poiseuille_lift(mesh, R_pipe=R_pipe, U_mean=U_dc, axis=2)

    dt = min(0.5, 0.5*dx/max(2*U_dc, 1e-30))
    print(f"  PISO {n_steps} steps × dt={dt:.2e}s ...", flush=True)
    t0 = time.time()
    state = run_piso(mesh, bcs, cfg, n_steps=n_steps, dt=dt,
                     body_force_fn=None, ibm_bodies=bodies, lifting=L_lift)
    state["u"].block_until_ready()
    wall = time.time() - t0
    print(f"    done in {wall:.0f}s ({wall/n_steps*1e3:.1f} ms/step)")

    u_phys = state["u"] + L_lift.u_lift_static
    p_lift_fn = make_poiseuille_p_lift(mu=mu, U_mean=U_dc, pipe_radius=R_pipe)
    F_vec, _ = surface_integral_force(
        u_phys, state["p"], mesh, sphere_sdf_fn,
        mu=mu, dx=dx,
        shell_inner=shell[0], shell_outer=shell[1],
        ref_point=sphere_centre, p_lift_fn=p_lift_fn, pipe_axis=2,
    )
    F_z = float(F_vec[2])
    F_uncon = 6.0*np.pi*mu*r_b*(2*U_dc)
    K_FVM = F_z / F_uncon
    err = abs(K_FVM - K_h) / K_h * 100
    print(f"  F_z         = {F_z:.4e} N")
    print(f"  F_stokes(uncon, U=2U_dc) = {F_uncon:.4e} N")
    print(f"  K_FVM       = {K_FVM:.4f}")
    print(f"  K_Happel    = {K_h:.4f}   err = {err:.2f}%")
    return dict(lam=lam, cpr=cpr, shell=shell, K_FVM=K_FVM, K_Happel=K_h,
                err_pct=err, F_z=F_z, wall_s=wall)


def main():
    results = []
    # Primary runs
    for lam, cpr in [(0.1, 4), (0.1, 6), (0.3, 4), (0.3, 6), (0.3, 8)]:
        try:
            r = run_one(lam=lam, cpr=cpr, n_steps=800)
            results.append(r)
        except Exception as e:
            print(f"  FAILED λ={lam} cpr={cpr}: {type(e).__name__}: {e}")
            results.append(dict(lam=lam, cpr=cpr, FAILED=str(e)))

    # Shell sensitivity at λ=0.3 cpr=8 (or fall back to cpr that worked)
    print("\n" + "#"*78)
    print("Shell sensitivity at λ=0.3, cpr=6")
    print("#"*78)
    for shell in [(0.5, 2.5), (1.5, 3.5), (2.5, 4.5)]:
        try:
            r = run_one(lam=0.3, cpr=6, n_steps=800, shell=shell,
                        label_extra="(sensitivity)")
            results.append({**r, "sensitivity": True})
        except Exception as e:
            print(f"  FAILED shell={shell}: {e}")

    print("\n" + "=" * 78)
    print("T3 SURFACE_INTEGRAL SUMMARY")
    print("=" * 78)
    print(f"{'λ':>5} {'cpr':>4} {'shell':>14} {'K_FVM':>10} "
          f"{'K_Happel':>10} {'err %':>8}")
    for r in results:
        if "FAILED" in r:
            print(f"  {r['lam']:.2f}  {r['cpr']:>4d}  FAILED")
        else:
            shell_lbl = f"({r['shell'][0]},{r['shell'][1]})"
            sens = " [sens]" if r.get("sensitivity") else ""
            print(f"  {r['lam']:.2f}  {r['cpr']:>4d}  {shell_lbl:>14}  "
                  f"{r['K_FVM']:>10.4f}  {r['K_Happel']:>10.4f}  "
                  f"{r['err_pct']:>7.2f}%{sens}")


if __name__ == "__main__":
    main()
