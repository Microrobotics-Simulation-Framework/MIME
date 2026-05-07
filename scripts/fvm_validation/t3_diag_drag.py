"""T3/M1 drag diagnostics — print numbers, no fixes.

D1 slip ratio
D2 IBM penalty α vs viscous scale
D3 |f_IBM| total vs F_stokes
D4 p_hom dipole on pipe axis
D5 effective blocked cross-section vs physical
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
from mime.nodes.environment.fvm import (
    make_pipe_mesh, make_poiseuille_lift, make_poiseuille_p_lift,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.fvm.sdf import sphere_sdf


def happel_brenner(lam):
    return 1.0/(1.0-2.10443*lam+2.08877*lam**3-0.94813*lam**5
                -1.372*lam**6+3.87*lam**8-4.19*lam**10)


def diag(*, lam, cpr, U_dc, n_warmup, label):
    print("=" * 78)
    print(f"DIAGNOSTICS: {label}  (λ={lam}, cpr={cpr}, U_dc={U_dc})")
    print("=" * 78)
    r_b = 1e-3
    R_pipe = r_b/lam
    L_pipe = 22*r_b
    nu = 1e-3; rho = 1.0
    mu = rho*nu
    K_h = happel_brenner(lam)

    mesh = make_pipe_mesh(pipe_radius=R_pipe, pipe_length=L_pipe,
                          robot_radius=r_b, cpr=cpr)
    dx = mesh.cartesian_spacing[0]
    Nx, Ny, Nz = mesh.cartesian_shape
    L_actual = Nz * dx
    sphere_centre = jnp.array([0.0, 0.0, L_actual/2], dtype=mesh.V.dtype)
    print(f"  mesh {mesh.cartesian_shape} = {mesh.N_cells} cells, dx={dx*1e3:.4f}mm")

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
        bcs[name] = VelocityBC(u_wall=jnp.zeros((nb,3)), F_through=jnp.zeros((nb,)))

    cfg = PisoConfig(
        nu=nu, rho=rho, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0*dx,
    )
    L_lift = make_poiseuille_lift(mesh, R_pipe=R_pipe, U_mean=U_dc, axis=2)

    dt = min(0.5, 0.5*dx/max(2*U_dc, 1e-30))
    state = run_piso(mesh, bcs, cfg, n_steps=n_warmup, dt=dt,
                     body_force_fn=None, ibm_bodies=bodies, lifting=L_lift)
    state["u"].block_until_ready()
    print(f"  PISO converged: {n_warmup} steps × dt={dt:.2e}s")

    u_phys = np.asarray(state["u"] + L_lift.u_lift_static)   # [N_cells, 3]
    p_hom  = np.asarray(state["p"])                          # [N_cells]
    x      = np.asarray(mesh.x)
    V      = np.asarray(mesh.V)

    # SDF for sphere
    rxy_sph = np.sqrt((x[:,0]-0)**2 + (x[:,1]-0)**2 + (x[:,2]-L_actual/2)**2)
    phi_sphere = rxy_sph - r_b   # < 0 inside

    # ---------------- D1 ----------------
    sphere_mask = phi_sphere < 0
    shell_mask  = (phi_sphere > 0) & (phi_sphere < 2*dx)
    far_mask    = phi_sphere > 5*dx
    u_norm      = np.linalg.norm(u_phys, axis=-1)
    u_inside_mean = float(np.mean(u_norm[sphere_mask])) if sphere_mask.any() else 0.0
    u_shell_mean  = float(np.mean(u_norm[shell_mask])) if shell_mask.any() else 0.0
    u_far_uz_mean = float(np.mean(np.abs(u_phys[far_mask, 2]))) if far_mask.any() else 0.0
    slip_ratio    = u_inside_mean / max(u_far_uz_mean, 1e-30)
    print()
    print(f"D1: Cells inside sphere = {int(sphere_mask.sum())}, "
          f"in shell = {int(shell_mask.sum())}, far = {int(far_mask.sum())}")
    print(f"    Mean |u| inside sphere : {u_inside_mean:.4e} m/s")
    print(f"    Mean |u| in shell      : {u_shell_mean:.4e} m/s")
    print(f"    Mean |u_z| far field   : {u_far_uz_mean:.4e} m/s "
          f"(target 2*U_dc = {2*U_dc:.4e})")
    print(f"    Slip ratio             : {slip_ratio:.4e}  "
          f"(target <0.01)")

    # ---------------- D2 ----------------
    alpha = cfg.ibm_alpha
    visc_scale = nu / dx**2
    print()
    print(f"D2: ibm_alpha = {alpha:.4e}  (hardcoded in PisoConfig)")
    print(f"    ν/dx²    = {visc_scale:.4e} s⁻¹")
    print(f"    α/(ν/dx²) = {alpha/visc_scale:.2f}  "
          f"(threshold >100 for reliable no-slip)")
    # Brinkman: u_new = (u + α·dt·u_body)/(1 + α·dt·χ)
    # at α·dt = {alpha*dt:.2e}, fraction suppressed per step ≈ {alpha*dt/(1+alpha*dt)}
    print(f"    α·dt = {alpha*dt:.2e}; per-step Brinkman suppression "
          f"= {alpha*dt/(1+alpha*dt):.6f}")

    # ---------------- D3 ----------------
    # Brinkman force per unit volume = α * H_eps(-φ) * (u_phys - 0)
    # using a smooth Heaviside with width ibm_eps:
    eps = cfg.ibm_eps
    chi = 0.5 * (1.0 - np.tanh(phi_sphere / eps))
    f_IBM = alpha * chi[:, None] * u_phys                    # [N_cells, 3]
    F_IBM_total = (f_IBM * V[:, None]).sum(axis=0)           # [3]
    F_stokes_unconfined = 6 * np.pi * mu * r_b * (2*U_dc)
    F_stokes_confined   = F_stokes_unconfined * K_h
    print()
    print(f"D3: Total IBM force on sphere (vector): "
          f"[{F_IBM_total[0]:+.3e}, {F_IBM_total[1]:+.3e}, {F_IBM_total[2]:+.3e}] N")
    print(f"    F_stokes unconfined  = 6πμr·U  = {F_stokes_unconfined:.4e} N")
    print(f"    F_stokes·K_Happel    = {F_stokes_confined:.4e} N")
    print(f"    |F_IBM_z|/F_stokes(uncon) = {abs(F_IBM_total[2])/F_stokes_unconfined:.4f}")
    print(f"    |F_IBM_z|/F_stokes(conf)  = {abs(F_IBM_total[2])/F_stokes_confined:.4f}")

    # ---------------- D4 ----------------
    # Cells closest to the pipe axis. Cartesian cell centres are at
    # (i+0.5)*dx - Lx/2; the nearest pair is at ±dx/2, so use |x|<dx
    # to capture the four cells at the axis (i=Nx/2-1, Nx/2 in x and y).
    axis_mask = (np.abs(x[:,0]) < dx) & (np.abs(x[:,1]) < dx)
    z_axis = x[axis_mask, 2]
    p_axis = p_hom[axis_mask]
    sort_idx = np.argsort(z_axis)
    z_axis_s = z_axis[sort_idx]
    p_axis_s = p_axis[sort_idx]
    print()
    print(f"D4: p_hom along pipe axis (x≈y≈0), sphere centre at z={L_actual/2*1e3:.2f}mm:")
    # Print a sparse subset of axis points
    n_axis = len(z_axis_s)
    print_every = max(1, n_axis // 16)
    for i in range(0, n_axis, print_every):
        marker = "  ←sphere" if abs(z_axis_s[i] - L_actual/2) < r_b else ""
        print(f"    z={z_axis_s[i]*1e3:6.2f}mm  p_hom={p_axis_s[i]:+.4e} Pa{marker}")
    p_expected = mu * U_dc / r_b
    p_range_axis = float(p_axis_s.max() - p_axis_s.min())
    print(f"    Expected p_hom dipole magnitude ~ μU/r_b = {p_expected:.4e} Pa")
    print(f"    Measured p_hom range on axis     = {p_range_axis:.4e} Pa")
    print(f"    p_range / p_expected             = {p_range_axis/p_expected:.4f}")

    # ---------------- D5 (cross-section blockage) ----------------
    iz_eq = Nz // 2
    z_3d = x[:, 2].reshape(mesh.cartesian_shape)
    iz_mask = np.abs(z_3d - L_actual/2)[:,:,iz_eq] < 1e-9
    # Compute on the equatorial slab
    rxy_eq = np.sqrt(x[:,0]**2 + x[:,1]**2).reshape(mesh.cartesian_shape)
    z_diff = np.abs(x[:,2] - L_actual/2).reshape(mesh.cartesian_shape)
    in_eq_slab = z_diff[:,:,iz_eq] < dx/2  # mask for cells at sphere equator
    phi_eq = (np.sqrt(rxy_eq[:,:,iz_eq]**2 + z_diff[:,:,iz_eq]**2) - r_b)
    A_blocked = float(np.sum(phi_eq < 0.5*dx) * dx*dx)
    A_phys = np.pi * r_b**2
    A_pipe = np.pi * R_pipe**2
    lam_eff = np.sqrt(A_blocked/np.pi) / R_pipe
    print()
    print(f"D5: Physical sphere cross-section A_phys = {A_phys*1e6:.4f} mm²")
    print(f"    Effective blocked area (cpr={cpr})    = {A_blocked*1e6:.4f} mm²")
    print(f"    Blockage ratio  A_eff/A_phys          = {A_blocked/A_phys:.4f}  (1.0 = correct)")
    print(f"    Pipe cross-section A_pipe              = {A_pipe*1e6:.4f} mm²")
    print(f"    λ_eff vs λ_physical                   = {lam_eff:.4f} vs {lam:.4f}")

    # ---------------- Verdict ----------------
    print()
    print(f"VERDICT for {label}:")
    print(f"    D1 slip ratio                  : {slip_ratio:.4f}   "
          f"({'PASS' if slip_ratio < 0.01 else 'FAIL'})")
    print(f"    D2 α/(ν/dx²)                   : {alpha/visc_scale:.2f}     "
          f"({'PASS' if alpha/visc_scale > 100 else 'FAIL'})")
    print(f"    D3 |F_IBM_z|/F_stokes(conf)    : {abs(F_IBM_total[2])/F_stokes_confined:.4f}   "
          f"({'PASS' if abs(F_IBM_total[2])/F_stokes_confined > 0.9 else 'FAIL'})")
    print(f"    D4 p_hom dipole/expected       : {p_range_axis/p_expected:.4f}   "
          f"(expected ~1)")
    print(f"    D5 blockage ratio              : {A_blocked/A_phys:.4f}   "
          f"(expected ~1.0)")

    return dict(label=label, slip=slip_ratio, alpha_ratio=alpha/visc_scale,
                F_IBM=float(F_IBM_total[2]), F_stokes=F_stokes_confined,
                p_dipole=p_range_axis, blockage=A_blocked/A_phys)


def main():
    diag(lam=0.3, cpr=8, U_dc=1e-3, n_warmup=800, label="T3 λ=0.3 cpr=8 (Stokes)")
    print()
    diag(lam=0.1, cpr=4, U_dc=1e-3, n_warmup=400, label="T3 λ=0.1 cpr=4 (Stokes)")


if __name__ == "__main__":
    main()
