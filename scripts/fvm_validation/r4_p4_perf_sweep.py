"""R4-P4 — Perf crossover sweep: dense vs FFT at 32³, 48³, 64³, 96³, 128³.

Time 20 PISO steps for each (mesh size, backend) and report Mcells/s.
Identifies the crossover mesh size where FFT becomes faster than dense.
"""
from __future__ import annotations
import time
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, make_piso_step, initial_state


def run(N, backend):
    L = 1.0
    nu = 0.001
    mesh = make_cartesian_mesh_3d(N, N, N, L, L, L,
                                   origin=(-L/2, -L/2, 0.0),
                                   periodic_z=True)
    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max"):
        p = mesh.patch(name); nbf = int(p.owner.size)
        bcs[name] = VelocityBC(u_wall=jnp.zeros((nbf, 3)),
                                F_through=jnp.zeros((nbf,)))
    cfg = PisoConfig(
        nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
        pressure_bc=("neumann", "neumann", "periodic"),
        velocity_bc=("dirichlet", "dirichlet", "periodic"),
        transform_backend=backend,
    )
    step = jax.jit(make_piso_step(mesh, bcs, cfg, body_force_fn=None))
    s = initial_state(mesh)

    # Compile + warmup
    t0 = time.time()
    s = step(s, 0.01); s["u"].block_until_ready()
    compile_time = time.time() - t0

    # Time 20 steps
    t0 = time.time()
    for _ in range(20):
        s = step(s, 0.01)
    s["u"].block_until_ready()
    per_step = (time.time() - t0) / 20
    throughput = mesh.N_cells / per_step / 1e6
    return compile_time, per_step, throughput


def main():
    print("=" * 78)
    print("R4-P4 — Perf crossover (FFT vs dense, RTX 2060)")
    print("=" * 78)
    print(f"  {'N':>4} {'cells':>10} "
          f"{'dense_compile':>14} {'dense_step_ms':>14} {'dense_M/s':>10}  "
          f"{'fft_compile':>12} {'fft_step_ms':>12} {'fft_M/s':>9}  "
          f"{'fft/dense':>10}", flush=True)
    for N in (32, 48, 64, 96, 128):
        try:
            d_c, d_s, d_t = run(N, "dense")
        except Exception as e:
            print(f"  {N:>4}: dense FAILED: {type(e).__name__}: {e}")
            continue
        try:
            f_c, f_s, f_t = run(N, "fft")
        except Exception as e:
            print(f"  {N:>4}: fft FAILED: {type(e).__name__}: {e}")
            continue
        ratio = f_t / d_t
        print(f"  {N:>4} {N**3:>10} "
              f"{d_c:>14.2f} {d_s*1000:>14.2f} {d_t:>10.2f}  "
              f"{f_c:>12.2f} {f_s*1000:>12.2f} {f_t:>9.2f}  "
              f"{ratio:>10.2f}",
              flush=True)


if __name__ == "__main__":
    main()
