"""T4 (static variant) — sphere held at fixed r/R, measure lateral force.

Instead of dynamic migration (which suffers from integrator-overshoot
at this Re), hold the sphere at a sequence of fixed r/R positions and
measure the steady-state lateral component F_x. Equilibrium = where
F_x changes sign.

This isolates the FORCE BALANCE physics from the integrator stability.
The Segré-Silberberg equilibrium at Re=100, λ=0.3 should appear as a
zero-crossing of F_x(r/R) somewhere in (0, 1).
"""
from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_cartesian_mesh_3d, FVMFluidNode, make_sphere_body_factory,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.ibm import IBMBody


def build_node(R_pipe=0.5, L_pipe=1.5, nu=0.005, lam=0.3,
               N_cross=48, N_axial=24,
               ibm_alpha=1e5):
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    r_s = lam * R_pipe

    def pipe_wall_sdf(x):
        rho = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-30)
        return R_pipe - rho
    wall = IBMBody(name="pipe_wall", sdf=pipe_wall_sdf)

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
    U_mean = 100 * nu / (2 * R_pipe)
    U_centre = 2 * U_mean
    body_force_amp = U_centre * 4 * nu / R_pipe**2

    def body_force(t):
        return jnp.array([0.0, 0.0, body_force_amp])

    sphere_factory = make_sphere_body_factory("sphere", radius=r_s)
    node = FVMFluidNode(
        name="fluid", timestep=0.01,
        mesh=mesh, bcs=bcs, cfg=cfg,
        static_bodies=[wall],
        dynamic_body_factories=[("sphere", sphere_factory)],
        body_force_fn=body_force,
        force_method="surface_integral", force_shell=(1.5, 3.5),
    )
    return node, mesh, R_pipe, L_pipe, r_s


def force_at(r_over_R: float, node, mesh, R_pipe, L_pipe, r_s,
             dt=0.05, n_steady=2000):
    """Hold sphere at (r_over_R * R, 0, L/2) and run to steady state.
    Return (F_lat_x, F_axial_z, F_y_residual)."""
    pos = jnp.array([r_over_R * R_pipe, 0.0, L_pipe / 2], dtype=jnp.float32)
    inputs = {
        "sphere_position": pos,
        "sphere_linear_velocity": jnp.zeros(3),
        "sphere_angular_velocity": jnp.zeros(3),
    }

    @jax.jit
    def run(state):
        def body(s, i):
            return node.update(s, inputs, dt), None
        s, _ = jax.lax.scan(body, state, jnp.arange(n_steady))
        return s

    state = node.initial_state()
    state = run(state)
    state["u"].block_until_ready()
    F = np.asarray(state["force_sphere"])
    return float(F[0]), float(F[2]), float(F[1])


def main():
    print("=" * 78)
    print("T4 (static) — F_lat vs r/R at Re_pipe=100, λ=0.3")
    print("=" * 78)
    node, mesh, R_pipe, L_pipe, r_s = build_node()
    dx = mesh.cartesian_spacing[0]
    print(f"  mesh: N_cross=48, N_axial=24, dx={dx:.4f}, "
          f"sphere_radius/dx = {r_s/dx:.1f}", flush=True)

    print(f"\n  {'r/R':>6} {'F_x (lat)':>13} {'F_z (drag)':>13} {'F_y':>13}", flush=True)
    rows = []
    for r in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        t0 = time.time()
        Fx, Fz, Fy = force_at(r, node, mesh, R_pipe, L_pipe, r_s)
        elapsed = time.time() - t0
        sign = "+" if Fx > 0 else "-"
        print(f"  {r:6.2f} {Fx:13.4e} {Fz:13.4e} {Fy:13.4e}  "
              f"({sign}, {elapsed:.0f}s)", flush=True)
        rows.append((r, Fx, Fz, Fy))

    print("\n>> Equilibrium location: where F_x(r/R) crosses zero")
    for i in range(len(rows) - 1):
        r1, F1, _, _ = rows[i]
        r2, F2, _, _ = rows[i + 1]
        if F1 * F2 < 0:
            r_eq = r1 - F1 * (r2 - r1) / (F2 - F1)   # linear interp
            print(f"   sign change between r/R={r1} ({F1:+.2e}) and "
                  f"r/R={r2} ({F2:+.2e}) ⇒ equilibrium ≈ r/R={r_eq:.3f}")


if __name__ == "__main__":
    main()
