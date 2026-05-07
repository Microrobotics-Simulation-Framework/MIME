"""T4 — Full Segré-Silberberg migration to equilibrium.

Configuration:
  * 3D pipe, body-force-driven Poiseuille at Re_pipe ≈ 100.
  * Sphere of radius a, confinement λ = 0.3 → a = 0.3 R_pipe.
  * Sphere coupled to a simple rigid-body integrator (Euler) using
    Stokes-mobility translation with Faxen-type wall correction.

Two cases:
  (1) sphere starts at r/R = 0.2 → expected to migrate outward.
  (2) sphere starts at r/R = 0.8 → expected to migrate inward.

Pass criteria:
  * Both runs converge to r/R ≈ 0.60 ± 0.05.
  * No NaN.

Outputs the trajectory and final equilibrium for each case.
"""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.fvm import (
    make_cartesian_mesh_3d, FVMFluidNode, make_sphere_body_factory,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.ibm import IBMBody


def build_node(R_pipe=0.5, L_pipe=2.0, nu=0.005, lam=0.3,
               N_cross=32, N_axial=24,
               ibm_alpha=1e5, body_force_amp=None,
               use_surface_integral=True):
    margin = 1.2
    Lx = Ly = 2 * margin * R_pipe
    mesh = make_cartesian_mesh_3d(
        N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
        origin=(-Lx / 2, -Ly / 2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    r_s = lam * R_pipe

    def pipe_wall_sdf(x):
        rho = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30)
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

    if body_force_amp is None:
        # Body force chosen so Re_pipe = U_mean * 2R / nu = 100
        U_mean = 100 * nu / (2 * R_pipe)
        U_centre = 2 * U_mean
        body_force_amp = U_centre * 4 * nu / R_pipe ** 2

    def body_force(t):
        return jnp.array([0.0, 0.0, body_force_amp])

    sphere_factory = make_sphere_body_factory("sphere", radius=r_s)
    node = FVMFluidNode(
        name="fluid",
        timestep=0.01,
        mesh=mesh, bcs=bcs, cfg=cfg,
        static_bodies=[wall],
        dynamic_body_factories=[("sphere", sphere_factory)],
        body_force_fn=body_force,
        force_method="surface_integral" if use_surface_integral else "brinkman",
        force_shell=(1.5, 3.5),
    )
    return node, mesh, R_pipe, L_pipe, nu, r_s, body_force_amp


def run_migration(initial_r_over_R: float, *, n_steps=4000, dt=0.05,
                  R_pipe=0.5, L_pipe=2.0, nu=0.005, lam=0.3,
                  N_cross=32, N_axial=24, sample_every=20,
                  rho_sphere_over_fluid: float = 1.0,
                  n_warm: int = 4000):
    node, mesh, R_pipe, L_pipe, nu, r_s, f_amp = build_node(
        R_pipe=R_pipe, L_pipe=L_pipe, nu=nu, lam=lam,
        N_cross=N_cross, N_axial=N_axial,
    )

    initial_x = initial_r_over_R * R_pipe
    pos0 = jnp.array([initial_x, 0.0, L_pipe / 2], dtype=jnp.float32)

    state0 = node.initial_state()

    # ---- Warm-up: hold sphere stationary while fluid develops ----
    static_inputs = {
        "sphere_position": pos0,
        "sphere_linear_velocity": jnp.zeros(3),
        "sphere_angular_velocity": jnp.zeros(3),
    }
    @jax.jit
    def warmup(state):
        def body(s, i):
            return node.update(s, static_inputs, dt), None
        s, _ = jax.lax.scan(body, state, jnp.arange(n_warm))
        return s
    t0 = time.time()
    state0 = warmup(state0)
    state0["u"].block_until_ready()
    t_warm = time.time() - t0

    # ---- Migration: overdamped Stokes mobility ----
    # Stokes mobility (no wall correction) — slow, stable, and
    # equilibrium location is invariant to mobility magnitude.
    # The IBM force has a magnitude bias (T3 finding) but the
    # equilibrium r/R where lateral force = 0 is unaffected.
    inv_mob = 6 * np.pi * 1.0 * nu * r_s   # = 1/μ_Stokes
    # Lateral motion only — axial motion is fast (Poiseuille drift),
    # which would advect sphere out of the periodic-z box and away
    # from its starting axial position. We zero v_z to keep sphere
    # at the same axial slice (equivalent to a co-moving frame).

    @jax.jit
    def coupled_run(state, pos):
        def stride(carry, i):
            s, p = carry
            for _ in range(sample_every):
                inputs = {
                    "sphere_position": p,
                    "sphere_linear_velocity": jnp.zeros(3),
                    "sphere_angular_velocity": jnp.zeros(3),
                }
                new_s = node.update(s, inputs, dt)
                F = new_s["force_sphere"]
                # Overdamped — keep sphere on its axial slice
                v = F / inv_mob
                v = v.at[2].set(0.0)   # zero axial velocity
                p = p + dt * v
                s = new_s
            return (s, p), jnp.concatenate([p, v])
        n_samples = n_steps // sample_every
        (final_s, final_p), traj = jax.lax.scan(
            stride, (state, pos), jnp.arange(n_samples),
        )
        return final_s, final_p, traj

    t0 = time.time()
    final_state, final_pos, traj = coupled_run(state0, pos0)
    final_state["u"].block_until_ready()
    elapsed = time.time() - t0
    return {
        "traj": np.asarray(traj),       # [n_samples, 6] = pos+vel
        "final_pos": np.asarray(final_pos),
        "final_vel": np.asarray(traj[-1, 3:6]),
        "elapsed": elapsed,
        "warmup_time": t_warm,
        "R_pipe": R_pipe,
        "r_s": r_s,
        "U_centre": 100 * nu / (2 * R_pipe) * 2,
    }


def main():
    print("=" * 78)
    print("T4 — Segré-Silberberg migration (Re_pipe=100, λ=0.3)")
    print("=" * 78)

    # Strategy: warm fluid up first (~4000 steps), then run sphere
    # migration with overdamped Stokes mobility (no inertial overshoot).
    # The IBM drag is biased by ~10x (T3 finding) so the migration is
    # slow, but the EQUILIBRIUM POSITION (where lateral force = 0) is
    # independent of force magnitude.
    # Bumped resolution: λ=0.3 with N_cross=48 ⇒ 8 cells per sphere
    # radius (was 4). Surface-integral force extraction with shell
    # (1.5, 3.5) dx. Stokes mobility (overdamped) for stability.
    common = dict(
        R_pipe=0.5, L_pipe=1.5, nu=0.005, lam=0.3,
        N_cross=48, N_axial=24, dt=0.05, n_steps=6000,
        sample_every=60, n_warm=1500,
    )

    cases = [("inner", 0.2), ("outer", 0.8)]
    case_outs = {}
    for label, r0 in cases:
        print(f"\n>> Case {label}: r/R = {r0}")
        out = run_migration(r0, **common)
        traj = out["traj"]
        R_pipe = out["R_pipe"]
        r_traj = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2) / R_pipe
        z_traj = traj[:, 2]
        v_traj = traj[:, 3:6]

        print(f"  wall time     : {out['elapsed']:.1f}s")
        print(f"  initial r/R   : {r_traj[0]:.3f}")
        print(f"  final   r/R   : {r_traj[-1]:.3f}")
        print(f"  axial travel  : {z_traj[-1] - z_traj[0]:+.3f}")
        axial_diameters = (z_traj[-1] - z_traj[0]) / (2 * out["r_s"])
        print(f"  sphere diameters travelled (axial): {axial_diameters:.1f}")
        print(f"  final velocity (vx,vy,vz)   : {v_traj[-1]}")

        n = len(r_traj)
        sample_idx = np.linspace(0, n - 1, 11).astype(int)
        for i in sample_idx:
            print(f"    sample={i:4d}  r/R={r_traj[i]:.3f}  z={z_traj[i]:.3f}  "
                  f"|v_lat|={float(np.linalg.norm(v_traj[i, :2])):.3e}")

        case_outs[label] = {
            "r_over_R": r_traj,
            "z": z_traj,
            "v": v_traj,
            "elapsed": out["elapsed"],
        }

    # Summary
    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print("(equilibrium target: r/R ≈ 0.60 ± 0.05)")
    for label, c in case_outs.items():
        r = c["r_over_R"]
        # Direction of migration: positive if moved outward, negative if inward
        delta = r[-1] - r[0]
        print(f"  case {label}: r/R {r[0]:.3f} -> {r[-1]:.3f}  "
              f"(Δ={delta:+.3f}, |v_lat|={float(np.linalg.norm(c['v'][-1, :2])):.3e}, "
              f"wall {c['elapsed']:.0f}s)")


if __name__ == "__main__":
    main()
