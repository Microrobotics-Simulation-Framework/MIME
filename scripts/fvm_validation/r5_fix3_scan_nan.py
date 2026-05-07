"""R5-Fix3 — Diagnose the jax.lax.scan NaN with jax_debug_nans=True.

Run a short Segré-Silberberg scan with NaN debugging enabled. JAX
will raise on the first NaN-producing operation, pinpointing the
root cause.
"""
from __future__ import annotations
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
jax.config.update("jax_debug_nans", True)

import time
import numpy as np
import jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_cartesian_mesh_3d, FVMFluidNode, make_sphere_body_factory,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.fvm.integrator import (
    ParticleState, implicit_drag_step, trilinear_interp,
)


def main():
    R_pipe = 0.5; L_pipe = 1.5; nu = 0.005; lam = 0.3
    N_cross = 32; N_axial = 20
    margin = 1.2; Lx = Ly = 2 * margin * R_pipe
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
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )
    U_centre = 0.1; body_force_amp = U_centre * 4 * nu / R_pipe**2
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
    drag_coeff = 6 * np.pi * 1.0 * nu * r_s
    m_p = (4/3) * np.pi * r_s**3

    pos0 = jnp.array([0.2 * R_pipe, 0.0, L_pipe / 2], dtype=jnp.float32)
    vel0 = jnp.zeros(3, dtype=jnp.float32)
    state0 = node.initial_state()

    # Warm fluid (no NaN expected)
    static_inputs = {
        "sphere_position": pos0,
        "sphere_linear_velocity": jnp.zeros(3),
        "sphere_angular_velocity": jnp.zeros(3),
    }
    @jax.jit
    def warm(s):
        def b(s, i): return node.update(s, static_inputs, 0.05), None
        s, _ = jax.lax.scan(b, s, jnp.arange(400))
        return s
    print("warming up..."); s = warm(state0); s["u"].block_until_ready()
    print(f"after warm: |u|max={float(jnp.max(jnp.abs(s['u']))):.3e}")

    # Now jit a SHORT scan that should NaN; capture the trace.
    @jax.jit
    def short_scan(state, particle):
        def stride(carry, i):
            s, p_state = carry
            for _ in range(20):
                inputs = {
                    "sphere_position": p_state.position,
                    "sphere_linear_velocity": p_state.velocity,
                    "sphere_angular_velocity": jnp.zeros(3),
                }
                new_s = node.update(s, inputs, 0.05)
                F = new_s["force_sphere"]
                u_f_at_p = trilinear_interp(new_s["u"], p_state.position, mesh)
                u_dir = u_f_at_p / (jnp.linalg.norm(u_f_at_p) + 1e-30)
                F_axial = jnp.dot(F, u_dir) * u_dir
                F_lat = F - F_axial
                p_state = implicit_drag_step(
                    p_state, F_external=F_lat,
                    u_fluid_at_particle=u_f_at_p,
                    m_p=m_p, drag_coeff=drag_coeff, dt=0.05, n_sub=20,
                )
                s = new_s
            return (s, p_state), p_state.position
        (final_s, final_p), traj = jax.lax.scan(
            stride, (state, ParticleState(pos0, vel0)), jnp.arange(50),
        )
        return final_s, final_p, traj

    print("running scan with NaN debug...", flush=True)
    try:
        final_s, final_p, traj = short_scan(s, ParticleState(pos0, vel0))
        final_s["u"].block_until_ready()
        print("Scan completed without NaN!")
        print(f"final pos: {final_p.position}")
        traj_np = np.asarray(traj)
        for i in [0, 10, 25, 49]:
            r = np.linalg.norm(traj_np[i, :2])
            print(f"  sample={i}: pos={traj_np[i]}, r/R={r/R_pipe:.3f}")
    except Exception as e:
        print(f"Scan triggered exception: {type(e).__name__}")
        # Print last lines of traceback
        import traceback
        tb = traceback.format_exc().split("\n")
        for line in tb[-30:]:
            print(line)


if __name__ == "__main__":
    main()
