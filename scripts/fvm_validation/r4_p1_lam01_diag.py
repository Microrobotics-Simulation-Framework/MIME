"""R4-P1 — Standalone diagnostic for λ=0.1 confined Stokes drag.

Steps:
  1) Print all geometric and non-dimensional quantities explicitly
  2) Verify Happel-Brenner formula against tabulated K(0.1) ≈ 1.270
  3) Verify U_mean is the actual cross-sectional mean
  4) Sweep cpr ∈ {4, 6, 8, 12} and check K_FVM(1/cpr) trend
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


def run(cpr):
    R_pipe = 0.5; L_pipe = 1.0; nu = 1.0; lam = 0.1
    r_s = lam * R_pipe                          # 0.05
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe               # 1.2
    dx_target = r_s / cpr
    N_cross = int(np.ceil(Lx / dx_target))
    # Cap N_axial at 24 to keep memory bounded for the cpr=12 case
    N_axial = min(24, max(16, int(np.ceil(L_pipe / dx_target))))
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    cpr_actual = r_s / dx
    pipe_radius_cells = R_pipe / dx
    gap_cells = (R_pipe - r_s) / dx
    print(f"\n  --- cpr_target = {cpr} ---")
    print(f"  N_cross={N_cross}, N_axial={N_axial}, dx={dx:.5f}, cells={mesh.N_cells}")
    print(f"  sphere radius cells   = {cpr_actual:.2f}")
    print(f"  pipe   radius cells   = {pipe_radius_cells:.2f}")
    print(f"  gap (sphere→pipe)     = {gap_cells:.2f} cells")
    shell_lo = 1.5 * dx; shell_hi = 3.5 * dx
    print(f"  shell radial range    = [{r_s+shell_lo:.4f}, {r_s+shell_hi:.4f}]")
    print(f"  pipe wall location    = {R_pipe:.4f}  (gap to shell outer = "
          f"{(R_pipe - (r_s+shell_hi))/dx:.2f} cells)")

    U_centre = 0.01 * nu / R_pipe          # Re_pipe = U_mean*2R/ν = 0.01
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
        ibm_alpha=1e5, ibm_eps=1.0*dx,
    )
    def body_force(t):
        return jnp.array([0.0, 0.0, f_steady])

    state = None
    t0 = time.time()
    for _ in range(12):
        state = run_piso(mesh, bcs, cfg, n_steps=200, dt=0.05,
                         body_force_fn=body_force,
                         ibm_bodies=[wall, sphere], initial=state)
    state["u"].block_until_ready()
    t_sim = time.time() - t0

    # --- diagnostic prints ---
    u = np.asarray(state["u"]).reshape(mesh.cartesian_shape + (3,))
    x_arr = np.asarray(mesh.x).reshape(mesh.cartesian_shape + (3,))
    iz_far = N_axial // 4   # axial slice well away from sphere
    # Cross-section velocity profile through y=0 column
    iy = N_cross // 2
    u_z_radial = u[:, iy, iz_far, 2]
    radial = x_arr[:, iy, iz_far, 0]

    # U_mean from mass flux through the cross-section (only fluid cells)
    phi_wall = R_pipe - np.sqrt(x_arr[..., 0]**2 + x_arr[..., 1]**2)
    fluid_mask = phi_wall > 0   # inside pipe bore
    Q = float(np.sum(u[..., 2] * fluid_mask) * dx**2 * dx)  # volumetric flow
    pipe_xs_area = np.pi * R_pipe**2
    U_mean_meas = Q / (pipe_xs_area * L_pipe)
    U_max_meas = float(np.max(u_z_radial))

    # Analytical U_mean if Poiseuille was achieved
    U_mean_analytic = U_centre / 2     # Poiseuille: U_mean = U_max/2
    print(f"  U_centre target       = {U_centre:.5e}")
    print(f"  U_max  measured       = {U_max_meas:.5e}  ratio {U_max_meas/U_centre:.3f}")
    print(f"  U_mean target (Poise.)= {U_mean_analytic:.5e}")
    print(f"  U_mean measured       = {U_mean_meas:.5e}")

    F_si, _ = surface_integral_force(
        state["u"], state["p"], mesh, sphere_sdf_fn,
        mu=cfg.rho * cfg.nu, dx=dx,
        shell_inner=1.5, shell_outer=3.5,
        ref_point=sphere_centre,
    )
    F_z = float(F_si[2])
    F_stokes = 6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre
    K_FVM = F_z / F_stokes
    K_h = happel_brenner(0.1)
    print(f"  μ                     = {cfg.rho * cfg.nu}")
    print(f"  a (sphere radius)     = {r_s}")
    print(f"  6πμa·U_centre         = {F_stokes:.5e}")
    print(f"  F_FVM (z)             = {F_z:.5e}")
    print(f"  K_FVM                 = {K_FVM:.4f}")
    print(f"  K_Happel              = {K_h:.4f}")
    print(f"  err vs Happel         = {abs(K_FVM - K_h)/K_h*100:.1f}%")
    print(f"  wall time             = {t_sim:.0f}s")
    return K_FVM, K_h, cpr_actual, t_sim


def main():
    print("=" * 78)
    print("R4-P1 — λ=0.1 diagnostic")
    print("=" * 78)
    print(f"\n  Happel-Brenner formula at λ=0.1: K = {happel_brenner(0.1):.4f}")
    print(f"  Tabulated value from H&B 1983 Table 6-4.1: K(0.1) ≈ 1.270")
    print(f"  Match within tabulated precision: "
          f"{'OK' if abs(happel_brenner(0.1) - 1.270) < 0.01 else 'OFF'}")

    rows = []
    for cpr in (4, 6, 8, 12):
        try:
            K_FVM, K_h, cpr_actual, t_sim = run(cpr)
        except Exception as e:
            print(f"\n  cpr={cpr}: FAILED ({type(e).__name__}: {e})")
            continue
        rows.append((cpr, cpr_actual, K_FVM, K_h, t_sim))

    print("\n" + "=" * 78)
    print("Trend: K_FVM vs 1/cpr")
    print("=" * 78)
    print(f"  {'cpr':>5} {'1/cpr':>8} {'K_FVM':>8} {'err vs Happel':>15}")
    for cpr, cpr_a, K, Kh, t in rows:
        err = abs(K - Kh) / Kh
        print(f"  {cpr_a:>5.1f} {1/cpr_a:>8.4f} {K:>8.4f} {err*100:>14.1f}%")


if __name__ == "__main__":
    main()
