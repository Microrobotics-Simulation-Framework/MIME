"""GPU performance check: JIT compile time + per-step wall time on 128³ PISO."""
from __future__ import annotations
import time
import jax
import jax.numpy as jnp
from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, make_piso_step, initial_state


def main():
    print("=" * 72)
    print("Perf — 128³ PISO step JIT compile + per-step wall time")
    print("=" * 72)
    N = 128
    L = 1.0
    nu = 0.001
    mesh = make_cartesian_mesh_3d(N, N, N, L, L, L,
                                  origin=(-L/2, -L/2, 0.0),
                                  periodic_z=True)
    print(f"  mesh: {mesh.N_cells} cells ({N}^3), {mesh.N_faces} faces")

    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max"):
        p = mesh.patch(name); nbf = int(p.owner.size)
        bcs[name] = VelocityBC(u_wall=jnp.zeros((nbf, 3)),
                               F_through=jnp.zeros((nbf,)))

    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc=("neumann", "neumann", "periodic"),
        velocity_bc=("dirichlet", "dirichlet", "periodic"),
    )
    step_unjit = make_piso_step(mesh, bcs, cfg, body_force_fn=None)
    step = jax.jit(step_unjit)

    s0 = initial_state(mesh)

    # First call → compile
    t0 = time.time()
    s1 = step(s0, 0.01)
    s1["u"].block_until_ready()
    t_compile = time.time() - t0
    print(f"  First call (compile + run) : {t_compile:.2f}s")

    # Subsequent calls
    t0 = time.time()
    for _ in range(20):
        s1 = step(s1, 0.01)
    s1["u"].block_until_ready()
    t_step_avg = (time.time() - t0) / 20
    print(f"  Per-step wall time (20-avg): {t_step_avg*1000:.2f}ms")
    print(f"  Throughput: {mesh.N_cells / t_step_avg / 1e6:.2f} Mcells/s")


if __name__ == "__main__":
    main()
