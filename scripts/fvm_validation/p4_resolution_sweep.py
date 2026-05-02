"""P4 — Resolution sweep for confined Stokes drag at λ=0.3.

Sphere on the centreline of a body-force-driven pipe, at Re_pipe=0.01.
Sweep cells_per_radius ∈ {4, 6, 8, 12} and report:
  * K_FVM = F_si / (6πμaU_centre)
  * K_Happel (Happel-Brenner analytical correction)
  * relative error
  * gap_cells = (R_pipe − a) / dx — number of cells between sphere
    and pipe wall. The IBM diffuse band around each body has half-
    width eps=1*dx, so gap_cells must be ≥ ~3 for the bands to NOT
    overlap. If gap_cells < 5, the surface-integral shell may sit
    in the wall-IBM diffuse zone.
  * wall time per case
"""
from __future__ import annotations
import time
import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import IBMBody, surface_integral_force
from mime.nodes.environment.fvm.sdf import sphere_sdf


def happel_brenner(lam):
    return 1.0 / (1.0 - 2.10443*lam + 2.08877*lam**3
                  - 0.94813*lam**5 - 1.372*lam**6
                  + 3.87*lam**8 - 4.19*lam**10)


def fvm_drag(*, lam, R_pipe=0.5, L_pipe=1.0, cells_per_radius=8,
             N_axial=12, nu=1.0, n_chunks=12, n_per_chunk=200,
             dt=0.05, ibm_alpha=1e5):
    r_s = lam * R_pipe
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    N_cross = int(np.ceil(Lx / (r_s / cells_per_radius)))
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    cpr_actual = r_s / dx
    gap_cells = (R_pipe - r_s) / dx
    print(f"    mesh {N_cross}x{N_cross}x{N_axial}, dx={dx:.4f}, "
          f"cpr={cpr_actual:.1f}, gap_cells={gap_cells:.1f}, "
          f"({mesh.N_cells} cells)", flush=True)

    U_centre = 0.01 * nu / R_pipe   # Re_pipe=0.01 ⇒ U_centre = 2*U_mean
    f_steady = U_centre * 4 * nu / R_pipe**2

    sphere_centre = jnp.array([0.0, 0.0, L_pipe/2], dtype=jnp.float32)
    def pipe_wall_sdf(x):
        rho = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-30)
        return R_pipe - rho
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_s)
    wall = IBMBody(name="pipe_wall", sdf=pipe_wall_sdf)
    sphere = IBMBody(name="sphere", sdf=sphere_sdf_fn)

    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max"):
        p = mesh.patch(name); nbf = int(p.owner.size)
        bcs[name] = VelocityBC(u_wall=jnp.zeros((nbf, 3)),
                                F_through=jnp.zeros((nbf,)))

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc=("neumann", "neumann", "periodic"),
        velocity_bc=("dirichlet", "dirichlet", "periodic"),
        ibm_alpha=ibm_alpha, ibm_eps=1.0*dx,
    )
    def body_force(t):
        return jnp.array([0.0, 0.0, f_steady])

    state = None
    t0 = time.time()
    for _ in range(n_chunks):
        state = run_piso(mesh, bcs, cfg, n_steps=n_per_chunk, dt=dt,
                         body_force_fn=body_force,
                         ibm_bodies=[wall, sphere], initial=state)
    state["u"].block_until_ready()
    elapsed = time.time() - t0

    F_si, _ = surface_integral_force(
        state["u"], state["p"], mesh, sphere_sdf_fn,
        mu=cfg.rho * cfg.nu, dx=dx,
        shell_inner=1.5, shell_outer=3.5,
        ref_point=sphere_centre,
    )
    F_z = float(F_si[2])
    F_stokes_unbounded = 6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre
    return F_z, U_centre, F_stokes_unbounded, dx, cpr_actual, gap_cells, elapsed


def main():
    print("=" * 78)
    print("P4 — Resolution sweep for confined Stokes drag")
    print("=" * 78)
    rows = []
    for lam in (0.1, 0.2, 0.3):
        K_h = happel_brenner(lam)
        print(f"\n>> λ = {lam}, K_Happel = {K_h:.3f}", flush=True)
        for cpr_t in (4, 6, 8):
            try:
                F_z, U_c, F_s, dx, cpr, gap, t_e = fvm_drag(
                    lam=lam, cells_per_radius=cpr_t, n_chunks=8,
                )
            except Exception as e:
                print(f"    cpr={cpr_t}: FAILED ({type(e).__name__}: {e})")
                continue
            K_fvm = F_z / F_s
            err = abs(K_fvm - K_h) / K_h
            print(f"    cpr={cpr_t}: K_FVM={K_fvm:.3f}  K_Happel={K_h:.3f}  "
                  f"err={err*100:.1f}%  gap={gap:.1f}  ({t_e:.0f}s)",
                  flush=True)
            rows.append((lam, cpr_t, K_fvm, K_h, err, gap, t_e))

    print("\n" + "=" * 78)
    print("Summary table")
    print("=" * 78)
    print(f"  {'λ':>5} {'cpr':>4} {'K_FVM':>8} {'K_Happel':>9} "
          f"{'err':>7} {'gap':>5}")
    for lam, cpr, K, Kh, err, gap, _ in rows:
        flag = "PASS" if err < 0.05 else ("close" if err < 0.10 else "FAIL")
        print(f"  {lam:>5} {cpr:>4} {K:>8.3f} {Kh:>9.3f} "
              f"{err*100:>6.1f}% {gap:>5.1f}  {flag}")


if __name__ == "__main__":
    main()
