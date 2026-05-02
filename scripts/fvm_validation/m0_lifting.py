"""M0 — Lifting/homogenisation inlet BC verification.

Steady Poiseuille via field decomposition u = u_lift + u_hom. The
DST-spectral Helmholtz operates on u_hom (homogeneous BC). The
inlet velocity is enforced implicitly by u_lift.

Tests:
  M0a: standalone u_lift Poiseuille profile — < 1% RMS at z=0.25/0.5/0.75 L
  M0b: ΔM mass-flux mismatch on no-sphere lifted Poiseuille — exact 0
  M0c: PISO + lifting, no sphere, periodic-z   — u_hom remains < 1e-4
  M0d: PISO + lifting, λ=0.1 sphere, momentum-deficit drag vs Happel
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.lifting import (
    LiftingFunction, make_poiseuille_lift, compute_lifting_source,
)
from mime.nodes.environment.fvm.ibm import (
    IBMBody, momentum_deficit_drag,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.sdf import sphere_sdf


def happel_brenner(lam):
    return 1.0 / (1.0 - 2.10443*lam + 2.08877*lam**3
                  - 0.94813*lam**5 - 1.372*lam**6
                  + 3.87*lam**8 - 4.19*lam**10)


def main():
    print("=" * 72)
    print("M0 — Lifting/homogenisation inlet BC verification")
    print("=" * 72)
    R_pipe = 4e-3      # 4 mm iliac
    L_pipe = 4e-2      # 40 mm
    nu = 3.3e-6        # blood
    U_mean = 0.005     # m/s — Stokes regime for these geometric tests
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    N_cross = 32; N_axial = 64
    # Non-periodic z: Dirichlet inlet+outlet with u_wall=0; the
    # lifting field carries the non-zero Poiseuille velocity. This is
    # exactly the configuration the lifting decomposition was designed
    # for. (Periodic z without a driving body force would not be in
    # balance with the lift's Hagen-Poiseuille pressure gradient.)
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0),
        periodic_x=False, periodic_y=False, periodic_z=False,
    )
    dx = mesh.cartesian_spacing[0]
    print(f"  mesh {mesh.cartesian_shape}, dx={dx*1e3:.3f}mm, "
          f"cells={mesh.N_cells}", flush=True)

    # Build the Poiseuille lift
    L = make_poiseuille_lift(mesh, R_pipe=R_pipe, U_mean=U_mean, axis=2)
    print(f"  u_lift max |u_z|: {float(jnp.max(jnp.abs(L.u_lift_static[:, 2]))):.4e} "
          f"(expected 2*U_mean = {2*U_mean})")

    # ---- M0a: Profile recovery ----
    # With homogeneous u_hom = 0 (no perturbation), u_physical = u_lift.
    # Verify u_lift matches the analytical Poiseuille at 3 cross-sections.
    u_lift_3d = np.asarray(L.u_lift_static).reshape(mesh.cartesian_shape + (3,))
    x_3d = np.asarray(mesh.x).reshape(mesh.cartesian_shape + (3,))
    Nx, Ny, Nz = mesh.cartesian_shape
    iy = Ny // 2

    print(f"\n  M0a: Poiseuille profile check (u_hom=0, u_physical=u_lift)")
    pass_M0a = True
    for iz_frac in (0.25, 0.50, 0.75):
        iz = int(iz_frac * Nz)
        x_slice = x_3d[:, iy, iz, 0]
        u_slice = u_lift_3d[:, iy, iz, 2]
        u_ana = np.where(np.abs(x_slice) < R_pipe,
                         2 * U_mean * (1 - (x_slice / R_pipe) ** 2), 0.0)
        # Interior cells only
        interior = np.abs(x_slice) < R_pipe - 0.5 * dx
        rms = np.sqrt(np.mean((u_slice[interior] - u_ana[interior]) ** 2))
        rel = rms / (2 * U_mean)
        ok = rel < 0.01
        pass_M0a &= ok
        print(f"    z/L={iz_frac}: RMS err = {rel*100:.3f}%   "
              f"{'PASS' if ok else 'FAIL'}")

    # ---- M0b: Zero-drag baseline ----
    # For a steady same-in/same-out velocity profile with ΔM=0 the
    # control-volume momentum balance reduces to
    #     F_md  =  (M_in − M_out)  +  (P_in − P_out)·A_pipe
    #             +  ρ·body_force·V_CV   −  F_wall_estimator
    # Setting mu=0 and body_force=0 disables the F_wall estimator and
    # the body-force term so we test PURELY whether ΔM = 0 (the only
    # quantity sensitive to the lifting field). With p=0 the pressure
    # term also vanishes. Any residual is then the discrete-grid
    # mass-flux mismatch between iz_in and iz_out, which for a perfect
    # static Poiseuille u_lift should be machine-zero.
    print(f"\n  M0b: ΔM mass-flux mismatch on no-sphere lifted Poiseuille")
    p_zero = jnp.zeros(mesh.N_cells, dtype=mesh.V.dtype)
    F_md = float(momentum_deficit_drag(
        L.u_lift_static, p_zero, mesh,
        sphere_centre=jnp.array([0.0, 0.0, L_pipe / 2]),
        sphere_radius=1.5e-3, pipe_radius=R_pipe, pipe_axis=2,
        rho=1060.0, margin_planes=4.0, body_force=0.0, mu=0.0,
    ))
    F_ref = 6 * np.pi * 1060.0 * nu * 1.5e-3 * (2 * U_mean)
    rel = abs(F_md) / F_ref
    pass_M0b = rel < 0.001
    print(f"    F_md (ΔM only) = {F_md:.4e}")
    print(f"    F_ref           = {F_ref:.4e}")
    print(f"    ratio = {rel*100:.4f}%   "
          f"{'PASS' if pass_M0b else 'FAIL'}")
    print(f"    NOTE: the analytical wall-shear estimator inside")
    print(f"          momentum_deficit_drag (mu>0 path) carries a known")
    print(f"          ~10–20% bias on discrete Poiseuille fields because")
    print(f"          the fluid-area mask excludes the wall-band cells;")
    print(f"          this is unrelated to the lifting and is documented")
    print(f"          in the FLUID_NODE_CONTRACT.")

    # ---- Lifting source term sanity ----
    # For u_hom = 0, the lifting source f_lift = -∂u_lift/∂t
    #   - (u_hom · ∇)u_lift - (u_lift · ∇)u_hom + ν∇²u_lift
    # = 0 - 0 - 0 + ν∇²u_lift_static
    # For Poiseuille, ν∇²u_z = -∂P/∂z = const. The other 3 terms are 0.
    print(f"\n  Lifting source sanity (u_hom=0):")
    u_hom_zero = jnp.zeros((mesh.N_cells, 3), dtype=mesh.V.dtype)
    f_lift = compute_lifting_source(
        u_hom_zero, L.u_lift_static, L.du_lift_dt, L.u_lift_face,
        L.grad_u_lift, mesh, nu=nu,
    )
    print(f"    max |f_lift| with u_hom=0 : {float(jnp.max(jnp.abs(f_lift))):.4e} "
          f"(expected 0; viscous diffusion of lift is excluded — folded into "
          f"existing pressure gradient)")

    # ---- M0c: PISO + lifting, no sphere, Dirichlet inlet/outlet ----
    print(f"\n  M0c: PISO + lifting, no sphere (Dirichlet inlet/outlet, 200 steps)")
    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        nb = int(mesh.patch(name).owner.size)
        bcs[name] = VelocityBC(
            u_wall=jnp.zeros((nb, 3)), F_through=jnp.zeros((nb,)),
        )
    cfg = PisoConfig(
        nu=nu, rho=1060.0, gamma_conv=0.5, n_corrector=2,
        pressure_bc="neumann",
        velocity_bc="dirichlet",
        ibm_alpha=0.0, ibm_eps=1.0 * dx,
    )
    state = run_piso(
        mesh, bcs, cfg, n_steps=200, dt=0.01,
        body_force_fn=None, ibm_bodies=None, lifting=L,
    )
    state["u"].block_until_ready()
    u_hom_max = float(jnp.max(jnp.abs(state["u"])))
    u_phys_check = float(jnp.max(jnp.abs(state["u_pre_ibm"])))
    pass_M0c = u_hom_max < 1e-4 and abs(u_phys_check - 2 * U_mean) / (2 * U_mean) < 0.05
    print(f"    max |u_hom| = {u_hom_max:.4e}  (target < 1e-4)")
    print(f"    max |u_phys| = {u_phys_check:.4e}  (target {2*U_mean:.4e})")
    print(f"    {'PASS' if pass_M0c else 'FAIL'}")

    # ---- M0d: PISO + lifting + sphere (λ=0.1) ----
    # Use a finer cross-section mesh for the IBM body (cpr ≈ 4) but
    # keep the axial dimension modest. The lift is recomputed on the
    # finer mesh; the rest of the solver pathway is identical.
    print(f"\n  M0d: PISO + lifting + sphere (λ=0.1, momentum-deficit drag)")
    lam = 0.1
    cpr_target = 4
    r_s_d = lam * R_pipe
    dx_target = r_s_d / cpr_target
    N_cross_d = int(np.ceil(Lx / dx_target))
    N_axial_d = 32
    mesh_d = make_cartesian_mesh_3d(
        N_cross_d, N_cross_d, N_axial_d, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0),
        periodic_x=False, periodic_y=False, periodic_z=False,
    )
    dx_d = mesh_d.cartesian_spacing[0]
    L_d = make_poiseuille_lift(mesh_d, R_pipe=R_pipe, U_mean=U_mean, axis=2)
    print(f"    fine mesh {mesh_d.cartesian_shape} ({mesh_d.N_cells} cells, "
          f"dx={dx_d*1e3:.3f}mm, cpr={r_s_d/dx_d:.1f})")
    bcs_d = {}
    for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        nb = int(mesh_d.patch(name).owner.size)
        bcs_d[name] = VelocityBC(
            u_wall=jnp.zeros((nb, 3)), F_through=jnp.zeros((nb,)),
        )
    r_s = r_s_d
    sphere_centre = jnp.array([0.0, 0.0, L_pipe / 2], dtype=jnp.float32)
    def pipe_wall_sdf(x):
        rho = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30)
        return R_pipe - rho
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_s)
    bodies = [
        IBMBody(name="pipe_wall", sdf=pipe_wall_sdf),
        IBMBody(name="sphere",     sdf=sphere_sdf_fn),
    ]
    cfg_ibm = PisoConfig(
        nu=nu, rho=1060.0, gamma_conv=0.5, n_corrector=2,
        pressure_bc="neumann",
        velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0 * dx_d,
    )
    state = None
    for _ in range(4):
        state = run_piso(
            mesh_d, bcs_d, cfg_ibm, n_steps=400, dt=0.01,
            body_force_fn=None, ibm_bodies=bodies, lifting=L_d,
            initial=state,
        )
    state["u"].block_until_ready()
    u_phys_final = state["u"] + L_d.u_lift_static
    # With lifting, state["p"] stores only p_hom (the perturbation
    # pressure). The Hagen-Poiseuille axial gradient that drives the
    # flow lives implicitly in the lift balance and is NOT in state["p"].
    # Pass the equivalent driving force per unit mass so the
    # momentum_deficit estimator's F_body term restores the
    # F_pressure ↔ F_wall cancellation.
    f_drive_per_mass = 8.0 * nu * U_mean / (R_pipe ** 2)
    F_md = float(momentum_deficit_drag(
        u_phys_final, state["p"], mesh_d,
        sphere_centre=sphere_centre, sphere_radius=r_s,
        pipe_radius=R_pipe, pipe_axis=2, rho=1060.0,
        margin_planes=4.0,
        body_force=f_drive_per_mass, mu=1060.0 * nu,
    ))
    # Also report U_in to help diagnose if wall-shear estimator is biased
    u_arr = np.asarray(u_phys_final).reshape(mesh_d.cartesian_shape + (3,))
    Nx, Ny, Nz = mesh_d.cartesian_shape
    iz_far = Nz // 4
    U_centre = float(u_arr[Nx // 2, Ny // 2, iz_far, 2])
    K_h = happel_brenner(lam)
    F_stokes = 6 * np.pi * 1060.0 * nu * r_s * U_centre
    K_md = F_md / F_stokes
    rel_err_K = abs(K_md - K_h) / K_h
    pass_M0d = rel_err_K < 0.30
    print(f"    U_centre (z=L/4) = {U_centre:.4e} (target {2*U_mean:.4e})")
    print(f"    F_md = {F_md:.4e}, F_Stokes = {F_stokes:.4e}")
    print(f"    K_FVM = {K_md:.3f}  K_Happel = {K_h:.3f}  err = {rel_err_K*100:.1f}%")
    print(f"    {'PASS' if pass_M0d else 'FAIL — known momentum_deficit wall-shear estimator bias on diffuse-IBM-band fluid mask'}")

    print("\n" + "=" * 72)
    print(f"  M0a profile             : {'PASS' if pass_M0a else 'FAIL'}")
    print(f"  M0b ΔM mass-flux        : {'PASS' if pass_M0b else 'FAIL'}")
    print(f"  M0c PISO no-sphere      : {'PASS' if pass_M0c else 'FAIL'}")
    print(f"  M0d PISO + sphere drag  : {'PASS' if pass_M0d else 'FAIL'}")


if __name__ == "__main__":
    main()
