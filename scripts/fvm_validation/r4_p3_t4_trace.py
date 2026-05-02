"""R4-P3 — Trace T4 NaN: per-step diagnostic on Segré-Silberberg.

Print at each step: sphere r/R, max|u|, max|p|, IBM force magnitude.
Identify when and why NaN appears.

Also reports the literature-expected equilibrium for our (Re, λ).
"""
from __future__ import annotations
import time
import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_cartesian_mesh_3d, FVMFluidNode, make_sphere_body_factory,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.fvm.integrator import (
    ParticleState, implicit_drag_step, trilinear_interp,
)


def build_node(R_pipe=0.5, L_pipe=1.5, nu=0.005, lam=0.3,
               N_cross=32, N_axial=20, ibm_alpha=1e5, n_corrector=2):
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
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=n_corrector,
        pressure_bc=("neumann", "neumann", "periodic"),
        velocity_bc=("dirichlet", "dirichlet", "periodic"),
        ibm_alpha=ibm_alpha, ibm_eps=1.0 * dx,
    )
    U_mean = 100 * nu / (2 * R_pipe); U_centre = 2 * U_mean
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
    return node, mesh, R_pipe, L_pipe, r_s, body_force_amp, U_centre, cfg


def main():
    print("=" * 78)
    print("R4-P3 — T4 NaN trace (per-step diagnostic)")
    print("=" * 78)
    print("""
  Literature equilibrium for sphere in pipe (Segré-Silberberg):
    Re_p           : equilibrium r/R
      10           : ~0.50
      30           : ~0.55
      100 (small λ): ~0.60-0.63   (Schonberg & Hinch 1989, JFM)
      100 (λ=0.3) : ~0.45-0.55     (finite-size corrections; Asmolov 1999)
""")

    # Try with n_corrector=4 to see if it helps stability (Route A from brief)
    for ncorr in (2, 4):
        print(f"\n>> n_corrector={ncorr}")
        node, mesh, R_pipe, L_pipe, r_s, f_amp, U_c, cfg = build_node(
            n_corrector=ncorr,
        )
        nu = node._cfg.nu; rho = node._cfg.rho
        drag_coeff = 6 * np.pi * rho * nu * r_s
        m_p = (4/3) * np.pi * r_s**3 * rho

        # Warm up fluid
        pos0 = jnp.array([0.2 * R_pipe, 0.0, L_pipe / 2], dtype=jnp.float32)
        static_inputs = {
            "sphere_position": pos0,
            "sphere_linear_velocity": jnp.zeros(3),
            "sphere_angular_velocity": jnp.zeros(3),
        }
        @jax.jit
        def warm(state):
            def b(s, i): return node.update(s, static_inputs, 0.05), None
            s, _ = jax.lax.scan(b, state, jnp.arange(800))
            return s
        s = warm(node.initial_state())
        s["u"].block_until_ready()
        print(f"  warm-up done", flush=True)

        # One JIT-compiled step
        step = jax.jit(lambda state, p_state: node.update(state, {
            "sphere_position": p_state.position,
            "sphere_linear_velocity": p_state.velocity,
            "sphere_angular_velocity": jnp.zeros(3),
        }, 0.05))

        # Manual Python loop with per-step diagnostics
        p_state = ParticleState(pos0, jnp.zeros(3))
        n_steps_total = 200
        nan_step = None
        for i in range(n_steps_total):
            new_s = step(s, p_state)
            new_s["u"].block_until_ready()
            F = new_s["force_sphere"]
            u_max = float(jnp.max(jnp.abs(new_s["u"])))
            p_max = float(jnp.max(jnp.abs(new_s["p"])))
            F_mag = float(jnp.linalg.norm(F))
            u_f = trilinear_interp(new_s["u"], p_state.position, mesh)
            u_dir = u_f / (jnp.linalg.norm(u_f) + 1e-30)
            F_axial = jnp.dot(F, u_dir) * u_dir
            F_lat = F - F_axial
            p_state = implicit_drag_step(
                p_state, F_external=F_lat, u_fluid_at_particle=u_f,
                m_p=m_p, drag_coeff=drag_coeff, dt=0.05, n_sub=20,
            )
            r = float(jnp.linalg.norm(p_state.position[:2]))
            if i % 10 == 0 or np.isnan(u_max) or np.isnan(F_mag):
                print(f"    step {i:3d}: r/R={r/R_pipe:.3f}  "
                      f"|u|max={u_max:.2e}  |p|max={p_max:.2e}  "
                      f"|F|={F_mag:.2e}  |F_lat|={float(jnp.linalg.norm(F_lat)):.2e}",
                      flush=True)
            if np.isnan(u_max) or np.isnan(F_mag):
                nan_step = i
                break
            s = new_s
        if nan_step is not None:
            print(f"  NaN at step {nan_step}, breaking")
        else:
            print(f"  no NaN in {n_steps_total} steps; final r/R={r/R_pipe:.3f}")


if __name__ == "__main__":
    main()
