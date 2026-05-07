"""P3 — Segré-Silberberg with semi-implicit Maxey-Riley integrator.

Sphere is tracked with implicit-drag damping so the position update
is unconditionally stable in dt. Sub-stepping (n_sub > 1) lets us
take many small mechanical steps per fluid step without re-running
PISO.

Run two cases (r/R = 0.2 inner, r/R = 0.8 outer) and report the
trajectory r(t).
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
               N_cross=32, N_axial=20, ibm_alpha=1e5):
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
    return node, mesh, R_pipe, L_pipe, r_s, body_force_amp, U_centre


def run_case(initial_r_over_R: float, *,
             n_steps=8000, dt=0.05, sample_every=80,
             n_sub=20, n_warm=1000):
    node, mesh, R_pipe, L_pipe, r_s, f_amp, U_centre = build_node()
    nu = node._cfg.nu
    rho = node._cfg.rho
    drag_coeff = 6 * np.pi * rho * nu * r_s          # 6πμa
    m_p = (4 / 3) * np.pi * r_s ** 3 * rho           # neutrally buoyant

    initial_x = initial_r_over_R * R_pipe
    pos0 = jnp.array([initial_x, 0.0, L_pipe / 2], dtype=jnp.float32)
    vel0 = jnp.zeros(3, dtype=jnp.float32)
    state0 = node.initial_state()

    # Warm-up: hold sphere fixed, develop Poiseuille
    static_inputs = {
        "sphere_position": pos0,
        "sphere_linear_velocity": jnp.zeros(3),
        "sphere_angular_velocity": jnp.zeros(3),
    }
    @jax.jit
    def warmup(state):
        def body(s, i): return node.update(s, static_inputs, dt), None
        s, _ = jax.lax.scan(body, state, jnp.arange(n_warm))
        return s
    t0 = time.time()
    state = warmup(state0)
    state["u"].block_until_ready()
    t_warm = time.time() - t0
    print(f"  warm-up {n_warm} steps: {t_warm:.0f}s", flush=True)

    @jax.jit
    def coupled_run(state, particle):
        def stride(carry, i):
            s, p_state = carry
            for _ in range(sample_every):
                inputs = {
                    "sphere_position": p_state.position,
                    "sphere_linear_velocity": p_state.velocity,
                    "sphere_angular_velocity": jnp.zeros(3),
                }
                new_s = node.update(s, inputs, dt)
                F = new_s["force_sphere"]
                # Interpolate fluid u at sphere centre
                u_f_at_p = trilinear_interp(
                    new_s["u"], p_state.position, mesh,
                )
                # The IBM surface integral F includes BOTH linear
                # axial drag and the Segré-Silberberg lift. The
                # implicit-drag integrator already absorbs the linear
                # drag (it drives v → u_f). To avoid double-counting,
                # subtract the projected component of F along the
                # local fluid direction — what remains is the lift,
                # which is what we want to drive lateral migration.
                u_dir = u_f_at_p / (jnp.linalg.norm(u_f_at_p) + 1e-30)
                F_axial = jnp.dot(F, u_dir) * u_dir
                F_lateral = F - F_axial
                p_state = implicit_drag_step(
                    p_state, F_external=F_lateral,
                    u_fluid_at_particle=u_f_at_p,
                    m_p=m_p, drag_coeff=drag_coeff,
                    dt=dt, n_sub=n_sub,
                )
                s = new_s
            sample = jnp.concatenate([p_state.position, p_state.velocity])
            return (s, p_state), sample
        n_samples = n_steps // sample_every
        (final_s, final_p), traj = jax.lax.scan(
            stride, (state, ParticleState(pos0, vel0)), jnp.arange(n_samples),
        )
        return final_s, final_p, traj

    t0 = time.time()
    final_state, final_p, traj = coupled_run(state, ParticleState(pos0, vel0))
    final_state["u"].block_until_ready()
    elapsed = time.time() - t0
    return {
        "traj": np.asarray(traj),
        "final_pos": np.asarray(final_p.position),
        "final_vel": np.asarray(final_p.velocity),
        "elapsed": elapsed,
        "warmup": t_warm,
        "R_pipe": R_pipe, "r_s": r_s, "U_centre": U_centre,
    }


def main():
    print("=" * 78)
    print("P3 — Segré-Silberberg with semi-implicit drag (Maxey-Riley)")
    print("=" * 78)
    cases = [("inner", 0.2), ("outer", 0.8)]
    results = {}
    for label, r0 in cases:
        print(f"\n>> Case {label}: r/R = {r0}")
        out = run_case(r0)
        traj = out["traj"]
        R = out["R_pipe"]
        r_traj = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2) / R
        z_traj = traj[:, 2]
        v_lat = np.linalg.norm(traj[:, 3:5], axis=1)

        print(f"  wall time     : {out['elapsed']:.0f}s ({out['warmup']:.0f}s warm-up)")
        print(f"  initial r/R   : {r_traj[0]:.3f}")
        print(f"  final   r/R   : {r_traj[-1]:.3f}")
        print(f"  final |v_lat| : {v_lat[-1]:.3e}")

        n = len(r_traj)
        sample_idx = np.linspace(0, n - 1, 11).astype(int)
        for i in sample_idx:
            print(f"    sample={i:3d}  r/R={r_traj[i]:.3f}  z={z_traj[i]:.3f}  "
                  f"|v_lat|={v_lat[i]:.3e}", flush=True)
        results[label] = (r_traj, z_traj, v_lat)

    print("\n" + "=" * 78)
    print("Summary (target: r/R ≈ 0.60 ± 0.05, both sides)")
    print("=" * 78)
    for label, (r, z, v) in results.items():
        print(f"  case {label}: r/R {r[0]:.3f} -> {r[-1]:.3f}  "
              f"|v_lat|={v[-1]:.2e}")


if __name__ == "__main__":
    main()
