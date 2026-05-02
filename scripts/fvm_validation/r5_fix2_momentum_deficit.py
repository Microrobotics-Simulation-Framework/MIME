"""R5-Fix2 — Verify momentum-deficit drag method.

Tests:
  1) No-sphere Poiseuille pipe — momentum deficit must be ≈ 0 (< 0.1%
     of typical sphere drag).
  2) λ=0.2 — momentum-deficit must agree with surface-integral (which
     passed at 2.4% in round 3) within 5%.
  3) λ=0.3 — momentum-deficit vs Happel-Brenner.
"""
from __future__ import annotations
import time
import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import (
    IBMBody, surface_integral_force, momentum_deficit_drag,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf


def happel_brenner(lam):
    return 1.0 / (1.0 - 2.10443*lam + 2.08877*lam**3
                  - 0.94813*lam**5 - 1.372*lam**6
                  + 3.87*lam**8 - 4.19*lam**10)


def run(*, lam, R_pipe=0.5, L_pipe=1.5, cells_per_radius=6,
        with_sphere=True, n_chunks=12, n_per_chunk=200, dt=0.05,
        nu=1.0, ibm_alpha=1e5):
    r_s = lam * R_pipe
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    dx_target = (r_s if with_sphere else 0.05) / cells_per_radius
    N_cross = int(np.ceil(Lx / dx_target))
    N_axial = max(32, int(np.ceil(L_pipe / dx_target)))
    N_axial = min(N_axial, 48)   # cap memory
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    print(f"  mesh {N_cross}²×{N_axial}, dx={dx:.4f}, cells={mesh.N_cells}",
          flush=True)

    U_centre = 0.01 * nu / R_pipe
    f_steady = U_centre * 4 * nu / R_pipe**2
    sphere_centre = jnp.array([0.0, 0.0, L_pipe/2], dtype=jnp.float32)

    def pipe_wall_sdf(x):
        rho = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-30)
        return R_pipe - rho
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_s)
    bodies = [IBMBody(name="pipe_wall", sdf=pipe_wall_sdf)]
    if with_sphere:
        bodies.append(IBMBody(name="sphere", sdf=sphere_sdf_fn))

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
                         ibm_bodies=bodies, initial=state)
    state["u"].block_until_ready()
    elapsed = time.time() - t0
    print(f"  PISO time: {elapsed:.0f}s", flush=True)

    F_md = float(momentum_deficit_drag(
        state["u"], state["p"], mesh,
        sphere_centre=sphere_centre, sphere_radius=r_s,
        pipe_radius=R_pipe, pipe_axis=2, rho=cfg.rho,
        margin_planes=4.0,   # planes at z_sphere ± 4a
        body_force=f_steady,
        mu=cfg.rho * cfg.nu,
    ))
    F_si = None
    if with_sphere:
        F_si_vec, _ = surface_integral_force(
            state["u"], state["p"], mesh, sphere_sdf_fn,
            mu=cfg.rho * cfg.nu, dx=dx,
            shell_inner=1.5, shell_outer=3.5,
            ref_point=sphere_centre,
        )
        F_si = float(F_si_vec[2])

    F_stokes_unbounded = (6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre
                          if with_sphere else 1.0)
    return dict(F_md=F_md, F_si=F_si, F_stokes=F_stokes_unbounded,
                U_centre=U_centre, elapsed=elapsed)


def main():
    print("=" * 78)
    print("R5-Fix2 — Momentum-deficit drag verification")
    print("=" * 78)

    print("\n>> Test 1: NO sphere (Poiseuille only)")
    out = run(lam=0.1, with_sphere=False, cells_per_radius=8, n_chunks=10)
    F_typical = 6 * np.pi * 1.0 * 1.0 * 0.05 * 0.02   # nominal Stokes drag
    rel = abs(out["F_md"]) / F_typical
    print(f"  F_md (no sphere)   = {out['F_md']:.4e}")
    print(f"  Ref Stokes (λ=0.1) = {F_typical:.4e}")
    print(f"  ratio = {rel*100:.2f}%   "
          f"{'PASS' if rel < 0.001 else ('OK' if rel < 0.05 else 'FAIL')}")

    print("\n>> Test 2: λ=0.2 (cross-validate against surface integral)")
    out = run(lam=0.2, with_sphere=True, cells_per_radius=6, n_chunks=12)
    K_md = out["F_md"] / out["F_stokes"]
    K_si = out["F_si"] / out["F_stokes"]
    K_h = happel_brenner(0.2)
    print(f"  K_md       = {K_md:.3f}")
    print(f"  K_si       = {K_si:.3f}")
    print(f"  K_Happel   = {K_h:.3f}")
    err_md = abs(K_md - K_h) / K_h
    err_si = abs(K_si - K_h) / K_h
    print(f"  err_md vs Happel = {err_md*100:.1f}%")
    print(f"  err_si vs Happel = {err_si*100:.1f}%")
    print(f"  md vs si consistency = "
          f"{abs(K_md - K_si)/abs(K_si)*100:.1f}%")

    print("\n>> Test 3: λ=0.3 (the hard case)")
    out = run(lam=0.3, with_sphere=True, cells_per_radius=8, n_chunks=12)
    K_md = out["F_md"] / out["F_stokes"]
    K_si = out["F_si"] / out["F_stokes"]
    K_h = happel_brenner(0.3)
    print(f"  K_md       = {K_md:.3f}")
    print(f"  K_si       = {K_si:.3f}")
    print(f"  K_Happel   = {K_h:.3f}")
    err_md = abs(K_md - K_h) / K_h
    err_si = abs(K_si - K_h) / K_h
    print(f"  err_md vs Happel = {err_md*100:.1f}%")
    print(f"  err_si vs Happel = {err_si*100:.1f}%")


if __name__ == "__main__":
    main()
