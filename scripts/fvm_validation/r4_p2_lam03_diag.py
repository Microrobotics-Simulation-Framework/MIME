"""R4-P2 — Diagnose λ=0.3 confined Stokes drag.

Steps:
  1) At cpr=6, sweep shell positions and report K_FVM
  2) Split drag into pressure vs viscous components and check ratio
     (Stokes prediction: F_v / F_p = 2)
  3) Verify shell cells aren't inside pipe wall IBM
  4) Implement and compare momentum-deficit drag
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
from mime.nodes.environment.fvm.operators import grad_green_gauss


def happel_brenner(lam):
    return 1.0 / (1.0 - 2.10443*lam + 2.08877*lam**3
                  - 0.94813*lam**5 - 1.372*lam**6
                  + 3.87*lam**8 - 4.19*lam**10)


def split_pressure_viscous(u, p, mesh, sdf_fn, mu, dx, ref_point,
                           shell_inner=1.5, shell_outer=3.5):
    """Return (F_pressure, F_viscous, F_total) using surface integral.

    F = ∮_S σ·n dA with σ = -p I + 2μ ε.
    """
    dim = mesh.dim
    phi = sdf_fn(mesh.x)
    grad_phi = grad_green_gauss(phi, mesh)
    norm_g = jnp.sqrt(jnp.sum(grad_phi ** 2, axis=-1) + 1e-30)
    n_hat = grad_phi / norm_g[:, None]

    grad_u = grad_green_gauss(u, mesh)
    eps_strain = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))
    sigma_p = -p[:, None, None] * jnp.eye(dim, dtype=u.dtype)[None, :, :]
    sigma_v = 2.0 * mu * eps_strain
    t_p = jnp.einsum("Pij,Pj->Pi", sigma_p, n_hat)
    t_v = jnp.einsum("Pij,Pj->Pi", sigma_v, n_hat)

    shell_mask = (phi > shell_inner * dx) & (phi < shell_outer * dx)
    shell_thickness = (shell_outer - shell_inner) * dx
    weight = (mesh.V / shell_thickness) * shell_mask
    F_p = jnp.sum(t_p * weight[:, None], axis=0)
    F_v = jnp.sum(t_v * weight[:, None], axis=0)
    return F_p, F_v, F_p + F_v


def momentum_deficit(u, p, mesh, R_pipe, U_centre, mu, *,
                     z_inlet, z_outlet, rho=1.0):
    """Momentum-deficit drag on body in periodic-z pipe.

    For periodic-z, take a control volume between two axial planes
    z_inlet (just upstream of sphere) and z_outlet (just downstream).
    Steady state:
        F_drag = ρ ∫∫_inlet u_z (U_∞ - u_z) dA -  ρ ∫∫_outlet u_z (U_∞ - u_z) dA
                + (p_inlet - p_outlet) * A_cross + viscous terms

    For Poiseuille reference U_∞(r) we use the analytical parabola
    at U_centre. The viscous term ∫∫ μ ∂u/∂z dA at inlet/outlet is
    typically zero for fully-developed flow but here we include it.

    Returns F_drag_z (axial drag on body).
    """
    shape = mesh.cartesian_shape
    spacing = mesh.cartesian_spacing
    Nx, Ny, Nz = shape
    dx, dy, dz = spacing
    u_arr = u.reshape(shape + (3,))
    p_arr = p.reshape(shape)
    x_arr = mesh.x.reshape(shape + (3,))

    # Find indices closest to z_inlet and z_outlet
    z_cells = (jnp.arange(Nz) + 0.5) * dz
    iz_in = int(jnp.argmin(jnp.abs(z_cells - z_inlet)))
    iz_out = int(jnp.argmin(jnp.abs(z_cells - z_outlet)))

    # Cross-section mass flux and momentum flux at each plane.
    # Only fluid cells (inside pipe).
    rho_xy = jnp.sqrt(x_arr[..., 0]**2 + x_arr[..., 1]**2)
    fluid = (rho_xy < R_pipe).astype(u.dtype)

    def slab_quantities(iz):
        u_slab = u_arr[:, :, iz, :]
        p_slab = p_arr[:, :, iz]
        f_slab = fluid[:, :, iz]
        # Momentum flux ρ u_z²
        mom_flux = float(rho * jnp.sum(u_slab[..., 2]**2 * f_slab) * dx * dy)
        # Pressure × area
        p_int = float(jnp.sum(p_slab * f_slab) * dx * dy)
        # Mean velocity
        Q = float(jnp.sum(u_slab[..., 2] * f_slab) * dx * dy)
        return mom_flux, p_int, Q

    M_in, P_in, Q_in = slab_quantities(iz_in)
    M_out, P_out, Q_out = slab_quantities(iz_out)

    # Net momentum flux out - in = -F (force on fluid = -F_drag_on_body)
    # F_drag_on_body = (M_in - M_out) + (P_in - P_out) * A_eff
    # A_eff: assume same cross-section; the pressure force cancels if the
    # planes are equivalent. Use A = π R²
    A_pipe = jnp.pi * R_pipe**2
    F_drag = (M_in - M_out)        # from momentum flux change
    # For a periodic-z box with sphere at midplane, P_in ≈ P_out by symmetry
    # so pressure term ~ 0. Keep it for completeness:
    F_drag_p = (P_in - P_out)       # not multiplied by area since P_in is
                                     # already pressure-integrated over A
    print(f"      iz_in={iz_in}, iz_out={iz_out}")
    print(f"      Q_in={Q_in:.4e}, Q_out={Q_out:.4e}")
    print(f"      M_in={M_in:.4e}, M_out={M_out:.4e}, ΔM={M_in-M_out:+.4e}")
    print(f"      P_in*A={P_in:.4e}, P_out*A={P_out:.4e}, ΔP={P_in-P_out:+.4e}")
    return F_drag + F_drag_p


def run():
    R_pipe = 0.5; L_pipe = 1.0; nu = 1.0; lam = 0.3
    r_s = lam * R_pipe                          # 0.15
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    cpr = 6
    dx_target = r_s / cpr
    N_cross = int(np.ceil(Lx / dx_target))
    N_axial = 24
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    print(f"  N_cross={N_cross}, N_axial={N_axial}, dx={dx:.5f}")
    print(f"  cells={mesh.N_cells}, sphere/dx={r_s/dx:.1f}")

    U_centre = 0.01 * nu / R_pipe
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
    print(f"  PISO time: {time.time()-t0:.0f}s")

    # Get measured U_centre far from sphere
    u_arr = np.asarray(state["u"]).reshape(mesh.cartesian_shape + (3,))
    iy = N_cross // 2; ix = N_cross // 2
    iz_far = N_axial // 8     # well upstream of sphere at L/2
    U_centre_meas = float(u_arr[ix, iy, iz_far, 2])
    print(f"\n  U_centre_target = {U_centre:.4e}")
    print(f"  U_centre measured (z=L/8) = {U_centre_meas:.4e}")

    K_h = happel_brenner(lam)
    F_stokes_target = 6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre
    F_stokes_meas = 6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre_meas
    print(f"  K_Happel(λ=0.3) = {K_h:.3f}")

    print("\n  --- Shell sensitivity sweep ---")
    print(f"  {'shell':>14} {'K_FVM(target)':>14} {'K_FVM(meas)':>14} "
          f"{'F_p_z':>11} {'F_v_z':>11} {'F_v/F_p':>9}")
    for shell in [(0.5, 2.5), (1.5, 3.5), (2.5, 4.5), (3.5, 5.5)]:
        F_p, F_v, F_tot = split_pressure_viscous(
            state["u"], state["p"], mesh, sphere_sdf_fn,
            mu=cfg.rho * cfg.nu, dx=dx, ref_point=sphere_centre,
            shell_inner=shell[0], shell_outer=shell[1],
        )
        F_z = float(F_tot[2])
        K_target = F_z / F_stokes_target
        K_meas = F_z / F_stokes_meas
        ratio = float(F_v[2]) / (float(F_p[2]) + 1e-30)
        print(f"  ({shell[0]},{shell[1]}) {K_target:>14.4f} {K_meas:>14.4f} "
              f"{float(F_p[2]):>11.4e} {float(F_v[2]):>11.4e} {ratio:>9.3f}")

    print("\n  --- Momentum-deficit drag ---")
    F_md = momentum_deficit(
        state["u"], state["p"], mesh, R_pipe, U_centre_meas, cfg.rho * cfg.nu,
        z_inlet=L_pipe * 0.05, z_outlet=L_pipe * 0.95,
    )
    K_md_target = F_md / F_stokes_target
    K_md_meas = F_md / F_stokes_meas
    print(f"\n  F_md = {F_md:.4e}")
    print(f"  K_md (vs target U_c) = {K_md_target:.3f}")
    print(f"  K_md (vs measured U_c) = {K_md_meas:.3f}")
    print(f"  K_Happel              = {K_h:.3f}")


if __name__ == "__main__":
    run()
