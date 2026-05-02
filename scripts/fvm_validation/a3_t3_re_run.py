"""A3 — Re-run T3 (BEM cross-validation) with surface-integral force.

Sphere at the centreline of a body-force-driven pipe (IBM cylinder
wall). Drag is now extracted via the Cauchy-stress surface integral
:func:`surface_integral_force`. Compared against:
  * BEM (Stokeslet) at same geometry
  * Haberman-Sayre wall correction
  * Schiller-Naumann (unconfined inertial)
"""
from __future__ import annotations
import time
import numpy as np
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import (
    IBMBody, surface_integral_force,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh, cylinder_surface_mesh,
)
from mime.nodes.environment.stokeslet.resistance import (
    compute_resistance_matrix, compute_confined_resistance_matrix,
)


def haberman_sayre(lam):
    num = (1.0 - 2.105*lam + 2.0865*lam**3
           - 1.7068*lam**5 + 0.72603*lam**6)
    den = 1.0 - 0.75857*lam**5
    return 1.0 / (num / den)


def schiller_naumann(Re):
    return (24.0/Re) * (1.0 + 0.15 * Re**0.687)


def fvm_drag(*, lam: float, Re_pipe: float,
             R_pipe: float = 0.5, L_pipe: float = 1.0,
             cells_per_radius_target: int = 8,
             N_axial: int = 16,
             nu: float = 1.0, n_chunks: int = 12, n_per_chunk: int = 200,
             dt: float = 0.05, ibm_alpha: float = 1e5):
    """``cells_per_radius_target`` selects mesh resolution; N_cross
    is sized so the sphere has ≥ that many cells per radius."""
    r_s = lam * R_pipe
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    # Pick N_cross so dx ≤ r_s / cells_per_radius_target.
    N_cross = int(np.ceil(Lx / (r_s / cells_per_radius_target)))
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    cells_per_radius = r_s / dx
    print(f"    mesh {N_cross}x{N_cross}x{N_axial}, dx={dx:.4f}, "
          f"sphere_radius/dx = {cells_per_radius:.1f}, "
          f"({mesh.N_cells} cells)", flush=True)

    U_centre = Re_pipe * nu / R_pipe
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
    for _ in range(n_chunks):
        state = run_piso(mesh, bcs, cfg, n_steps=n_per_chunk, dt=dt,
                         body_force_fn=body_force,
                         ibm_bodies=[wall, sphere], initial=state)
    state["u"].block_until_ready()

    # Try several shells to assess sensitivity. Shell_inner must be
    # > 1*dx (the IBM diffuse-band half-width) so the integration
    # surface sits in clean fluid past the penalty contamination.
    print("    shell sensitivity: ", end="", flush=True)
    F_z_dict = {}
    for shell_in, shell_out in [(0.5, 2.5), (1.5, 3.5), (2.0, 4.0), (2.5, 4.5)]:
        F_si, _ = surface_integral_force(
            state["u"], state["p"], mesh, sphere_sdf_fn,
            mu=cfg.rho * cfg.nu, dx=dx,
            shell_inner=shell_in, shell_outer=shell_out,
            ref_point=sphere_centre,
        )
        F_z_dict[(shell_in, shell_out)] = float(F_si[2])
        print(f"({shell_in},{shell_out})={float(F_si[2]):.3e} ", end="", flush=True)
    print(flush=True)
    # Use the (1.5, 3.5) shell as the canonical answer (past diffuse band).
    F_z = F_z_dict[(1.5, 3.5)]
    F_stokes_unbounded = 6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre
    return F_z, U_centre, F_stokes_unbounded, dx, cells_per_radius


def bem_K(*, lam: float, R_pipe: float = 0.5, mu: float = 1.0,
          n_refine_sphere: int = 2, L_factor: float = 6.0):
    a = lam * R_pipe
    sphere_mesh = sphere_surface_mesh(radius=a, n_refine=n_refine_sphere)
    L_cyl = L_factor * R_pipe
    n_circ = max(24, int(2*np.pi*R_pipe/sphere_mesh.mean_spacing))
    n_axial = max(12, int(L_cyl/sphere_mesh.mean_spacing))
    n_circ = min(n_circ, 48); n_axial = min(n_axial, 40)
    wall_mesh = cylinder_surface_mesh(
        radius=R_pipe, length=L_cyl, n_circ=n_circ, n_axial=n_axial,
    )
    eps = sphere_mesh.mean_spacing / 2.0
    R_free = compute_resistance_matrix(
        jnp.array(sphere_mesh.points), jnp.array(sphere_mesh.weights),
        jnp.zeros(3), eps, mu,
        surface_normals=jnp.array(sphere_mesh.normals),
    )
    R_conf = compute_confined_resistance_matrix(
        jnp.array(sphere_mesh.points), jnp.array(sphere_mesh.weights),
        jnp.array(wall_mesh.points), jnp.array(wall_mesh.weights),
        jnp.zeros(3), eps, mu,
        body_normals=jnp.array(sphere_mesh.normals),
        wall_normals=jnp.array(wall_mesh.normals),
    )
    return float(R_conf[2, 2]) / float(R_free[2, 2])


def main():
    print("=" * 78)
    print("A3 — T3 re-run with surface-integral force")
    print("=" * 78)

    rows = []
    print("\n>> Stokes regime (Re_pipe=0.01)", flush=True)
    for lam in (0.1, 0.2, 0.3):
        print(f"\n  λ = {lam}", flush=True)
        t0 = time.time()
        # Pick cells_per_radius=8 if the resulting mesh fits in 6GB,
        # else step down. For lam=0.1 with R=0.5, sphere=0.05, we'd
        # need dx<=0.00625 ⇒ N_cross=192 — that OOMs. Fall back to
        # cpr_target=6 for the smallest λ.
        cpr_target = 6 if lam <= 0.1 else 8
        F_z, U_c, F_s, dx, cpr = fvm_drag(
            lam=lam, Re_pipe=0.01,
            cells_per_radius_target=cpr_target, N_axial=16,
            n_chunks=12,
        )
        t_fvm = time.time() - t0
        K_fvm = F_z / F_s
        K_b = bem_K(lam=lam)
        K_h = haberman_sayre(lam)
        eb = abs(K_fvm - K_b) / K_b
        eh = abs(K_fvm - K_h) / K_h
        print(f"    F_FVM = {F_z:.4e}, K_FVM = {K_fvm:.3f}  (t={t_fvm:.0f}s)")
        print(f"    K_BEM = {K_b:.3f}, err_BEM = {eb*100:.1f}%")
        print(f"    K_HS  = {K_h:.3f}, err_HS  = {eh*100:.1f}%")
        rows.append(dict(name=f"λ={lam},Re=0.01", K_fvm=K_fvm, K_b=K_b,
                         K_h=K_h, err_b=eb, err_h=eh))

    print("\n>> Inertial regime (Re_p ∈ {1, 10}, λ=0.1)", flush=True)
    for Re_p in (1.0, 10.0):
        lam = 0.1
        Re_pipe = Re_p / lam
        target_U = 0.2
        r_s = lam * 0.5
        nu = target_U * 2 * r_s / Re_p
        print(f"\n  Re_p={Re_p}", flush=True)
        F_z, U_c, F_s, dx, cpr = fvm_drag(
            lam=lam, Re_pipe=Re_pipe, nu=nu,
            cells_per_radius_target=6, N_axial=16,
            n_chunks=10,
        )
        rho = 1.0
        C_D_fvm = F_z / (0.5*rho*U_c**2 * np.pi * r_s**2)
        C_D_SN = schiller_naumann(Re_p)
        e = abs(C_D_fvm - C_D_SN) / C_D_SN
        print(f"    C_D_FVM = {C_D_fvm:.3f}, C_D_SN = {C_D_SN:.3f}, "
              f"err = {e*100:.1f}%")
        rows.append(dict(name=f"Re_p={Re_p},λ={lam}",
                         C_D_fvm=C_D_fvm, C_D_SN=C_D_SN, err=e))

    print("\nSummary:")
    for r in rows:
        print(f"  {r}")


if __name__ == "__main__":
    main()
