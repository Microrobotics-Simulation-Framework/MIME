"""Camera spike R1, item 5 (R3): the in-loop camera batched under ``vmap``, forward and reverse, against the unbatched run.

    python probe/r1_batch.py --batch B --kind stacked|theta --mode fwd|rev --out OUT.json [--width 320] [--steps 20]
        [--period 5]

``stacked``: B copies of the whole state stacked and mapped, as a batched study stacks its members' states, so the frame
clock is batched too and the frame's cond becomes a select (both branches every step). ``theta``: only the members'
starting speeds are mapped; the clock, which no member changes, stays unbatched, and JAX keeps the cond. ``B = 1`` runs
unmapped (the baseline). One configuration a process, so the peak (VmHWM) is that configuration's. The render runs with
the spike's sub-sample and cache-remake checkpoints (``MIME_SPIKE_SUBSAMPLE_CHECKPOINT``, ``MIME_SPIKE_CACHE_CHECKPOINT``)
so that reverse mode fits a runner.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def vm_hwm_gib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--kind", choices=("stacked", "theta"), default="stacked")
    parser.add_argument("--mode", choices=("fwd", "rev"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    os.environ["MIME_SPIKE_SUBSAMPLE_CHECKPOINT"] = "1"
    os.environ["MIME_SPIKE_CACHE_CHECKPOINT"] = "1"

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import r1_core as core

    built = core.case(width=arguments.width, period_steps=arguments.period, x0_um=-40.0, z0_um=15.0,
                      speed=60e-6 / (arguments.steps * 1e-3)).build()
    graph = built.graph
    state0, params0, theta0 = core.start_of(built)
    run = core.scalar_run(graph, arguments.steps, core.per_step_objective(graph), checkpoint=True)
    b = arguments.batch
    speeds = theta0[0] * (1.0 + 0.1 * jnp.arange(max(b, 1), dtype=theta0.dtype) / max(b, 1))

    def value(theta, state):
        return run(*core.inserted(theta, state, params0))[0]

    one = jax.grad(value) if arguments.mode == "rev" else value
    if b == 1:
        function, args = jax.jit(one), (theta0, state0)
    elif arguments.kind == "stacked":
        states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * b), state0)
        thetas = jnp.stack([theta0.at[0].set(v) for v in speeds])
        function, args = jax.jit(jax.vmap(one)), (thetas, states)
    else:
        function, args = jax.jit(jax.vmap(lambda v, theta, state: one(theta.at[0].set(v), state),
                                          in_axes=(0, None, None))), (speeds, theta0, state0)
    t0 = time.perf_counter()
    jax.block_until_ready(function(*args))
    first = time.perf_counter() - t0
    times = []
    for _ in range(arguments.repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(function(*args))
        times.append(time.perf_counter() - t0)
    lowered = function.lower(*args).compile().memory_analysis()
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "")
    report = {"batch": b, "kind": arguments.kind, "mode": arguments.mode, "width": arguments.width,
              "steps": arguments.steps, "period": arguments.period, "first_seconds": first, "steady_seconds": times,
              "seconds_per_step_min": min(times) / arguments.steps,
              "seconds_per_member_step_min": min(times) / arguments.steps / b,
              "peak_gib": vm_hwm_gib(), "xla_temp_gib": lowered.temp_size_in_bytes / 2**30 if lowered else None,
              "cpu": cpu, "threads": os.cpu_count()}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
