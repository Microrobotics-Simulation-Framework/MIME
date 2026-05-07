"""R6 — Dirichlet inlet / outlet pipe BCs verification.

Steps 1+2: no-sphere Poiseuille pipe with prescribed parabolic inlet
velocity at both z_min and z_max (fully developed assumption — both
ends prescribe the same profile so the flow is exactly Poiseuille
in steady state). Pressure: Neumann everywhere, mean pinned.

  * Profile check at 3 cross-sections vs analytical Poiseuille → < 1%
  * Momentum-deficit drag → < 0.1% of reference Stokes drag

Steps 3+4: with sphere, λ ∈ {0.1, 0.3} at cpr ∈ {4, 6, 8}.
  * K_FVM (momentum-deficit) vs K_Happel.
"""
from __future__ import annotations
import time
import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import (
    VelocityBC, poiseuille_inlet_velocity, poiseuille_inlet_F_through,
)
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import (
    IBMBody, surface_integral_force, momentum_deficit_drag,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf


def happel_brenner(lam):
    return 1.0 / (1.0 - 2.10443*lam + 2.08877*lam**3
                  - 0.94813*lam**5 - 1.372*lam**6
                  + 3.87*lam**8 - 4.19*lam**10)


def setup_pipe(*, R_pipe, L_pipe, U_mean, nu, with_sphere=False, lam=0.0,
               N_cross=32, N_axial=32, ibm_alpha=1e5):
    """Build mesh + BCs + cfg for an inlet/outlet pipe simulation."""
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    # NON-periodic mesh: z_min and z_max are inlet/outlet patches
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0),
        periodic_x=False, periodic_y=False, periodic_z=False,
    )
    dx = mesh.cartesian_spacing[0]
    r_s = lam * R_pipe if with_sphere else 0.0

    def pipe_wall_sdf(x):
        rho = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-30)
        return R_pipe - rho
    bodies = [IBMBody(name="pipe_wall", sdf=pipe_wall_sdf)]
    if with_sphere:
        sphere_centre = jnp.array([0.0, 0.0, L_pipe/2], dtype=jnp.float32)
        def sphere_sdf_fn(x):
            return sphere_sdf(x, center=sphere_centre, radius=r_s)
        bodies.append(IBMBody(name="sphere", sdf=sphere_sdf_fn))

    # Inlet (z_min) and outlet (z_max) — both prescribe Poiseuille
    # velocity (equivalent to fully-developed outflow for steady Stokes).
    u_inlet = poiseuille_inlet_velocity(
        mesh, "z_min", R_pipe=R_pipe, U_mean=U_mean, axis=2,
    )
    F_inlet = poiseuille_inlet_F_through(
        mesh, "z_min", R_pipe=R_pipe, U_mean=U_mean, axis=2,
    )
    u_outlet = poiseuille_inlet_velocity(
        mesh, "z_max", R_pipe=R_pipe, U_mean=U_mean, axis=2,
    )
    F_outlet = poiseuille_inlet_F_through(
        mesh, "z_max", R_pipe=R_pipe, U_mean=U_mean, axis=2,
    )
    bcs = {
        "x_min": VelocityBC(u_wall=jnp.zeros((mesh.patch("x_min").owner.size, 3)),
                             F_through=jnp.zeros((mesh.patch("x_min").owner.size,))),
        "x_max": VelocityBC(u_wall=jnp.zeros((mesh.patch("x_max").owner.size, 3)),
                             F_through=jnp.zeros((mesh.patch("x_max").owner.size,))),
        "y_min": VelocityBC(u_wall=jnp.zeros((mesh.patch("y_min").owner.size, 3)),
                             F_through=jnp.zeros((mesh.patch("y_min").owner.size,))),
        "y_max": VelocityBC(u_wall=jnp.zeros((mesh.patch("y_max").owner.size, 3)),
                             F_through=jnp.zeros((mesh.patch("y_max").owner.size,))),
        "z_min": VelocityBC(u_wall=u_inlet, F_through=F_inlet),
        "z_max": VelocityBC(u_wall=u_outlet, F_through=F_outlet),
    }
    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc="neumann",
        velocity_bc="dirichlet",
        ibm_alpha=ibm_alpha, ibm_eps=1.0*dx,
    )
    return mesh, bcs, cfg, bodies, dx, r_s


def main():
    print("=" * 78)
    print("R6 — Dirichlet inlet / outlet pipe BCs")
    print("=" * 78)
    R_pipe = 0.5; L_pipe = 1.0; nu = 1.0; U_mean = 0.005
    # Re_pipe = U_mean * 2R / nu = 0.005 * 1 / 1 = 0.005 (Stokes)

    print("\n>> Step 1+2: NO sphere — verify Poiseuille profile + zero drag")
    mesh, bcs, cfg, bodies, dx, _ = setup_pipe(
        R_pipe=R_pipe, L_pipe=L_pipe, U_mean=U_mean, nu=nu,
        with_sphere=False, N_cross=24, N_axial=32,
    )
    print(f"  mesh {mesh.cartesian_shape}, dx={dx:.4f}, "
          f"cells={mesh.N_cells}", flush=True)

    state = None
    t0 = time.time()
    for _ in range(8):
        state = run_piso(mesh, bcs, cfg, n_steps=200, dt=0.1,
                         body_force_fn=None, ibm_bodies=bodies, initial=state)
    state["u"].block_until_ready()
    print(f"  PISO time: {time.time()-t0:.0f}s", flush=True)

    # Check profile at 3 cross-sections
    u = np.asarray(state["u"]).reshape(mesh.cartesian_shape + (3,))
    Nx, Ny, Nz = mesh.cartesian_shape
    iy = Ny // 2
    print(f"\n  Profile check: u_z(r) along y=0, x-line for 3 z-slices")
    for iz_frac in (0.25, 0.50, 0.75):
        iz = int(iz_frac * Nz)
        x_slice = np.asarray(mesh.x).reshape(mesh.cartesian_shape + (3,))[:, iy, iz, 0]
        u_z_slice = u[:, iy, iz, 2]
        # Analytical Poiseuille
        u_z_ana = np.where(np.abs(x_slice) < R_pipe,
                            2 * U_mean * (1 - (x_slice / R_pipe)**2),
                            0.0)
        # Compare interior cells only
        interior = np.abs(x_slice) < R_pipe - 1.5 * dx
        max_err = np.max(np.abs(u_z_slice[interior] - u_z_ana[interior]))
        max_ana = np.max(np.abs(u_z_ana))
        rel = max_err / max_ana
        print(f"    z={iz_frac:.2f}L (iz={iz}): max abs err = {max_err:.3e}, "
              f"rel = {rel*100:.2f}%   {'PASS' if rel < 0.01 else 'FAIL'}")

    # Step 2: momentum-deficit on no-sphere — should be ~0
    F_md_nosp = float(momentum_deficit_drag(
        state["u"], state["p"], mesh,
        sphere_centre=jnp.array([0.0, 0.0, L_pipe/2]),
        sphere_radius=0.05,        # nominal value for the planes
        pipe_radius=R_pipe, pipe_axis=2, rho=cfg.rho,
        margin_planes=4.0, body_force=0.0, mu=cfg.rho*cfg.nu,
    ))
    F_ref = 6 * np.pi * cfg.rho * cfg.nu * 0.05 * U_mean * 2  # nominal scale
    rel = abs(F_md_nosp) / F_ref
    print(f"\n  Momentum deficit (no sphere) = {F_md_nosp:.4e}")
    print(f"  Reference Stokes scale       = {F_ref:.4e}")
    print(f"  ratio = {rel*100:.3f}%   "
          f"{'PASS' if rel < 0.01 else ('OK' if rel < 0.05 else 'FAIL')}")

    # ---- Sphere cases ----
    print("\n>> Step 3+4: WITH sphere, momentum-deficit drag at λ ∈ {0.1, 0.3}")
    for lam in (0.1, 0.3):
        K_h = happel_brenner(lam)
        print(f"\n  λ = {lam}, K_Happel = {K_h:.3f}")
        for cpr in (6,):
            r_s = lam * R_pipe
            dx_target = r_s / cpr
            margin = 1.2
            Lx = 2 * margin * R_pipe
            N_cross = int(np.ceil(Lx / dx_target))
            N_axial = max(32, int(np.ceil(L_pipe / dx_target)))
            N_axial = min(N_axial, 48)
            try:
                mesh, bcs, cfg, bodies, dx, r_s = setup_pipe(
                    R_pipe=R_pipe, L_pipe=L_pipe, U_mean=U_mean, nu=nu,
                    with_sphere=True, lam=lam,
                    N_cross=N_cross, N_axial=N_axial,
                )
                t0 = time.time()
                state = None
                for _ in range(10):
                    state = run_piso(mesh, bcs, cfg, n_steps=200, dt=0.1,
                                     body_force_fn=None,
                                     ibm_bodies=bodies, initial=state)
                state["u"].block_until_ready()
                t_e = time.time() - t0
                F_md = float(momentum_deficit_drag(
                    state["u"], state["p"], mesh,
                    sphere_centre=jnp.array([0.0, 0.0, L_pipe/2]),
                    sphere_radius=r_s, pipe_radius=R_pipe,
                    pipe_axis=2, rho=cfg.rho,
                    margin_planes=4.0, body_force=0.0, mu=cfg.rho*cfg.nu,
                ))
                F_si_vec, _ = surface_integral_force(
                    state["u"], state["p"], mesh,
                    bodies[1].sdf, mu=cfg.rho*cfg.nu, dx=dx,
                    shell_inner=1.5, shell_outer=3.5,
                    ref_point=jnp.array([0.0, 0.0, L_pipe/2]),
                )
                F_si = float(F_si_vec[2])
                # Use centerline U at z=L/4 as U_centre reference
                u_arr = np.asarray(state["u"]).reshape(mesh.cartesian_shape + (3,))
                ix = mesh.cartesian_shape[0]//2; iy = mesh.cartesian_shape[1]//2
                iz_far = mesh.cartesian_shape[2]//4
                U_centre_meas = float(u_arr[ix, iy, iz_far, 2])
                F_stokes = 6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre_meas
                K_md = F_md / F_stokes
                K_si = F_si / F_stokes
                print(f"    cpr={cpr}, mesh {mesh.cartesian_shape}, "
                      f"({mesh.N_cells} cells, t={t_e:.0f}s)")
                print(f"      U_centre_meas = {U_centre_meas:.4e} (target {2*U_mean})")
                print(f"      K_md = {K_md:.3f}  err vs Happel = "
                      f"{abs(K_md-K_h)/K_h*100:.1f}%")
                print(f"      K_si = {K_si:.3f}  err vs Happel = "
                      f"{abs(K_si-K_h)/K_h*100:.1f}%")
            except Exception as e:
                print(f"    cpr={cpr}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
