"""T3 — BEM cross-validation: sphere drag in pipe at low and moderate Re.

For each confinement λ ∈ {0.1, 0.2, 0.3} and Re ∈ {0.01, 1, 10}:
  * Run FVM (IBM sphere on the centreline of a body-force-driven pipe).
  * Extract drag force from IBM penalty (Brinkman-aware formula).
  * Compute K(λ) = F_FVM / (6πμaU_centre).
  * Compare to BEM (existing Stokeslet node) and to Haberman-Sayre
    analytical correction (existing reference in test_confined_validation).

For Stokes regime (Re=0.01) BEM is the reference. For inertial regimes
(Re=1, 10) compare to Schiller-Naumann (unconfined-inertial) and
verify that FVM > BEM (BEM has no inertial correction).
"""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import IBMBody, compute_ibm_forces
from mime.nodes.environment.fvm.sdf import sphere_sdf

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh, cylinder_surface_mesh,
)
from mime.nodes.environment.stokeslet.resistance import (
    compute_resistance_matrix, compute_confined_resistance_matrix,
)


def haberman_sayre(lam: float) -> float:
    num = (1.0 - 2.105 * lam + 2.0865 * lam ** 3 - 1.7068 * lam ** 5
           + 0.72603 * lam ** 6)
    den = 1.0 - 0.75857 * lam ** 5
    return 1.0 / (num / den)


def schiller_naumann(Re: float) -> float:
    return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)


def fvm_sphere_drag(
    *, lam: float, Re_pipe: float, R_pipe: float = 0.5,
    L_pipe: float = 1.0, N_cross: int = 32, N_axial: int = 16,
    nu: float = 1.0, n_chunks: int = 12, n_per_chunk: int = 200,
    dt: float = 0.05, ibm_alpha: float = 1e5,
):
    """Run FVM and return (F_drag_z, U_centre, K_FVM, F_stokes_unbounded)."""
    r_s = lam * R_pipe
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx / 2, -Ly / 2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]

    # Choose body force for desired Re_pipe
    # U_mean = Re_pipe * nu / (2 * R_pipe), U_centre = 2 * U_mean
    U_centre = Re_pipe * nu / R_pipe   # = 2 U_mean
    f_steady = U_centre * 4 * nu / R_pipe ** 2

    sphere_centre = jnp.array([0.0, 0.0, L_pipe / 2], dtype=jnp.float32)

    def pipe_wall_sdf(x):
        rho = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30)
        return R_pipe - rho

    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_s)

    wall = IBMBody(name="pipe_wall", sdf=pipe_wall_sdf)
    sphere = IBMBody(
        name="sphere", sdf=sphere_sdf_fn,
        extract_force=True, ref_point=sphere_centre,
    )

    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max"):
        p = mesh.patch(name); nbf = int(p.owner.size)
        bcs[name] = VelocityBC(
            u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)),
        )

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc=("neumann", "neumann", "periodic"),
        velocity_bc=("dirichlet", "dirichlet", "periodic"),
        ibm_alpha=ibm_alpha, ibm_eps=1.0 * dx,
    )

    def body_force(t):
        return jnp.array([0.0, 0.0, f_steady])

    state = None
    for _ in range(n_chunks):
        state = run_piso(
            mesh, bcs, cfg, n_steps=n_per_chunk, dt=dt,
            body_force_fn=body_force, ibm_bodies=[wall, sphere],
            initial=state,
        )
    state["u"].block_until_ready()

    forces = compute_ibm_forces(
        state["u_after_explicit"], mesh.x, mesh.V, [wall, sphere],
        alpha=cfg.ibm_alpha, eps=cfg.ibm_eps, rho=cfg.rho, dt=dt,
    )
    F_z = float(forces["sphere"]["force"][2])
    F_stokes = 6 * np.pi * 1.0 * nu * r_s * U_centre
    K_fvm = F_z / F_stokes
    return F_z, U_centre, K_fvm, F_stokes


def bem_sphere_drag(*, lam: float, R_pipe: float = 0.5,
                    n_refine_sphere: int = 2, mu: float = 1.0,
                    L_factor: float = 6.0):
    """Return K_BEM = F_confined / F_unbounded for a unit-velocity sphere."""
    a = lam * R_pipe
    sphere_mesh = sphere_surface_mesh(radius=a, n_refine=n_refine_sphere)
    L_cyl = L_factor * R_pipe
    n_circ = max(24, int(2 * np.pi * R_pipe / sphere_mesh.mean_spacing))
    n_axial = max(12, int(L_cyl / sphere_mesh.mean_spacing))
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
    F_free = float(R_free[2, 2])
    F_conf = float(R_conf[2, 2])
    K_bem = F_conf / F_free
    F_stokes_analytic = 6 * np.pi * mu * a
    return K_bem, F_free / F_stokes_analytic, F_free, F_conf


def main():
    print("=" * 78)
    print("T3 — BEM cross-validation: sphere drag in pipe")
    print("=" * 78)

    results = []

    # Stokes regime (Re_pipe = 0.01)
    print("\n>> Stokes regime (Re_pipe = 0.01)")
    for lam in (0.1, 0.2, 0.3):
        print(f"\n  λ = {lam:.2f}:")

        # FVM
        t0 = time.time()
        F_fvm, U_centre, K_fvm, F_stokes = fvm_sphere_drag(
            lam=lam, Re_pipe=0.01,
            N_cross=32, N_axial=16, nu=1.0, n_chunks=12,
        )
        t_fvm = time.time() - t0

        # BEM
        t0 = time.time()
        K_bem, F_free_norm, F_free, F_conf = bem_sphere_drag(lam=lam)
        t_bem = time.time() - t0

        K_hs = haberman_sayre(lam)
        err_fvm_bem = abs(K_fvm - K_bem) / K_bem
        err_fvm_hs = abs(K_fvm - K_hs) / K_hs

        print(f"    U_centre={U_centre:.4e}, F_stokes_unbounded={F_stokes:.4e}")
        print(f"    K_FVM = {K_fvm:.3f}  (F={F_fvm:.4e}, t={t_fvm:.0f}s)")
        print(f"    K_BEM = {K_bem:.3f}  (F_free_norm={F_free_norm:.3f}, t={t_bem:.0f}s)")
        print(f"    K_HS  = {K_hs:.3f}  (Haberman-Sayre)")
        print(f"    FVM vs BEM error: {err_fvm_bem*100:.1f}%")
        print(f"    FVM vs H&S error: {err_fvm_hs*100:.1f}%")
        results.append({
            "name": f"Stokes λ={lam}",
            "K_FVM": K_fvm, "K_BEM": K_bem, "K_HS": K_hs,
            "err_BEM": err_fvm_bem, "err_HS": err_fvm_hs,
            "pass": (err_fvm_bem < 0.05 and err_fvm_hs < 0.05),
        })

    # Inertial regime
    print("\n>> Inertial regime (Re_p = 1 and 10)")
    for Re_p in (1.0, 10.0):
        lam = 0.1
        # Re_p = U_centre * 2a / nu, so for given lam and Re_p:
        # Re_pipe = U_centre * 2R / nu = Re_p / lam
        Re_pipe = Re_p / lam
        # Choose nu so U_centre is moderate. Set R=0.5, nu chosen for stability.
        # Take U_centre = 0.2 → nu = U_centre*2*r_s/Re_p = 0.2*2*0.05/Re_p
        target_U = 0.2
        r_s = lam * 0.5
        nu = target_U * 2 * r_s / Re_p

        print(f"\n  Re_p = {Re_p:.1f}, λ = {lam}:")
        F_fvm, U_centre, K_fvm, F_stokes = fvm_sphere_drag(
            lam=lam, Re_pipe=Re_pipe, nu=nu,
            N_cross=32, N_axial=16, n_chunks=10,
            ibm_alpha=1e5,
        )
        # BEM at same geometry
        K_bem, _, _, _ = bem_sphere_drag(lam=lam)

        # F_FVM_z normalised vs Stokes-unbounded gives K_fvm (which
        # includes both confinement AND inertial correction).
        # Compare drag coefficient C_D
        rho = 1.0
        C_D_fvm = F_fvm / (0.5 * rho * U_centre ** 2 * np.pi * r_s ** 2)
        C_D_SN = schiller_naumann(Re_p)
        C_D_BEM_unconfined = (24.0 / Re_p) * K_bem  # confined Stokes C_D
        err_fvm_SN = abs(C_D_fvm - C_D_SN) / C_D_SN

        print(f"    U_centre={U_centre:.4f}, nu={nu:.4e}")
        print(f"    F_FVM = {F_fvm:.4e}  (K_FVM/Stokes = {K_fvm:.3f})")
        print(f"    F_BEM/F_Stokes (confined) = {K_bem:.3f}")
        print(f"    FVM C_D = {C_D_fvm:.3f}")
        print(f"    Schiller-Naumann C_D = {C_D_SN:.3f}  (unconfined inertial)")
        print(f"    err_FVM_vs_SN = {err_fvm_SN*100:.1f}%")
        print(f"    Sanity: F_FVM/F_BEM_eq = {K_fvm/K_bem:.3f}  "
              f"(>1 expected as Re→1 has inertial enhancement)")
        results.append({
            "name": f"Re_p={Re_p:.0f} λ={lam}",
            "C_D_FVM": C_D_fvm, "C_D_SN": C_D_SN,
            "err_SN": err_fvm_SN,
            "pass": err_fvm_SN < 0.10,
        })

    # Summary
    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    for r in results:
        print(f"  {r['name']:30s}  {r}")


if __name__ == "__main__":
    main()
