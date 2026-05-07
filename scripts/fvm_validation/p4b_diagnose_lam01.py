"""P4b — Diagnose why λ=0.1 confined Stokes regressed from 13% to 66%.

A3 v2 (commit 0ba6b5e) at λ=0.1 cpr=6 N_axial=16 n_chunks=12: K_FVM=0.957
P4   (current code)   at λ=0.1 cpr=6 N_axial=12 n_chunks=8 : K_FVM=0.422

Try the A3 configuration with the CURRENT code (after inline RC, V_owner
precompute, etc.) to determine if the regression is from the inline RC
change or from the test-config differences (N_axial / n_chunks).
"""
from __future__ import annotations
import time
import numpy as np
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import IBMBody, surface_integral_force
from mime.nodes.environment.fvm.sdf import sphere_sdf


def run_lam01(*, N_axial, n_chunks, cells_per_radius=6):
    R_pipe = 0.5; L_pipe = 1.0; nu = 1.0; lam = 0.1
    r_s = lam * R_pipe
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    N_cross = int(np.ceil(Lx / (r_s / cells_per_radius)))
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    print(f"  mesh {N_cross}x{N_cross}x{N_axial}, dx={dx:.4f}, "
          f"({mesh.N_cells} cells)", flush=True)

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
    for _ in range(n_chunks):
        state = run_piso(mesh, bcs, cfg, n_steps=200, dt=0.05,
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
    F_stokes = 6 * np.pi * cfg.rho * cfg.nu * r_s * U_centre
    K = F_z / F_stokes
    return K, elapsed


def main():
    print("=" * 78)
    print("P4b — diagnose λ=0.1 regression")
    print("=" * 78)
    K_happel = 1.263
    print(f"  K_Happel = {K_happel:.3f}\n")

    print(">> A3-like config: N_axial=16, n_chunks=12, cpr=6")
    K, t = run_lam01(N_axial=16, n_chunks=12, cells_per_radius=6)
    err = abs(K - K_happel) / K_happel
    print(f"  K_FVM = {K:.3f}, err = {err*100:.1f}%, time {t:.0f}s\n")

    print(">> P4 config: N_axial=12, n_chunks=8, cpr=6")
    K, t = run_lam01(N_axial=12, n_chunks=8, cells_per_radius=6)
    err = abs(K - K_happel) / K_happel
    print(f"  K_FVM = {K:.3f}, err = {err*100:.1f}%, time {t:.0f}s\n")

    print(">> Long: N_axial=16, n_chunks=20, cpr=6 (5x A3 length)")
    K, t = run_lam01(N_axial=16, n_chunks=20, cells_per_radius=6)
    err = abs(K - K_happel) / K_happel
    print(f"  K_FVM = {K:.3f}, err = {err*100:.1f}%, time {t:.0f}s")


if __name__ == "__main__":
    main()
