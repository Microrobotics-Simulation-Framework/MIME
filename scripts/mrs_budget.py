"""The MRS rung's performance budget: the cost and memory of one solve of the 9,360-unknown system (the
0.2 surrogate's training mesh of de Jongh's FL-9, 3,120 samples, frozen in ``tests/differential/references/``),
forward and differentiated, and of a step of ``Drag.MRS`` in a graph, by MADDENING 0.4's profiler.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/mrs_budget.py <out.json>
    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run --no-sync python \
        scripts/mrs_budget.py <out.json>

Not collected by pytest. Each time is the median of ``REPEATS`` runs after the first (which compiles), each blocked
until its result is ready; the memory is XLA's own account of the compiled program (its arguments, outputs and
temporaries) and, on a GPU, the device's peak. What it measures:

- ``forward``: the resistance from the samples (the matrix, its LU, six solves, the force sums);
- ``tangent``: the resistance and its derivative along one direction (forward mode, the samples scaled);
- ``gradient``: the reverse-mode derivative of a scalar of the resistance in that scale, checkpointed as the node
  is, and not, for the memory the checkpoint saves;
- ``factor``: the LU alone, and a Cholesky factorisation of the same matrix (symmetric positive definite on every
  case measured), for what a symmetric solve would save;
- ``graph``: ``profile_graph`` on a free FL-9 body with ``Drag.MRS`` sampled at about the same count, its steps,
  compile and kernel counts."""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from mime.nodes.hydro.mrs import blob_squared, resistance, rigid_motions, stokeslet_matrix  # noqa: E402

REPEATS = 5
WEIGHTS = jnp.asarray(np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


def timed(function, *args) -> dict[str, float]:
    started = time.perf_counter()
    jax.block_until_ready(function(*args))
    first = time.perf_counter() - started
    times = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(function(*args))
        times.append(time.perf_counter() - started)
    return {"first_s": first, "median_s": statistics.median(times), "min_s": min(times), "max_s": max(times)}


def memory(function, *args) -> dict[str, int]:
    analysis = jax.jit(function).lower(*args).compile().memory_analysis()
    if analysis is None:
        return {}
    return {name: int(getattr(analysis, name)) for name in ("argument_size_in_bytes", "output_size_in_bytes",
                                                             "temp_size_in_bytes", "generated_code_size_in_bytes")
            if hasattr(analysis, name)}


def device_peak() -> int | None:
    stats = jax.devices()[0].memory_stats()
    return None if not stats else int(stats.get("peak_bytes_in_use", 0))


def main(out: Path) -> None:
    held = np.load(ROOT / "tests" / "differential" / "references" / "port_mrs_full.npz")
    points = jnp.asarray(held["helix_training/points"])
    weights = jnp.asarray(held["helix_training/weights"])
    count = int(points.shape[0])

    def forward(scale):
        return resistance(points * scale, blob_squared(weights * scale ** 2, 0.5), 1.0, jnp.zeros(3))

    def scalar(scale):
        found = forward(scale)
        return jnp.sum(WEIGHTS * found / jnp.sqrt(jnp.outer(jnp.diag(found), jnp.diag(found))))

    def tangent(scale):
        return jax.jvp(forward, (scale,), (jnp.ones_like(scale),))

    checkpointed = jax.grad(jax.checkpoint(scalar))
    plain = jax.grad(scalar)
    scale = jnp.asarray(1.0)
    matrix = jax.jit(lambda: stokeslet_matrix(points, blob_squared(weights, 0.5), 1.0))()
    motions = rigid_motions(points, jnp.zeros(3))
    found: dict[str, object] = {
        "samples": count, "unknowns": 3 * count, "device": str(jax.devices()[0]), "backend": jax.default_backend(),
        "x64": bool(jax.config.jax_enable_x64), "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax": jax.__version__, "python": platform.python_version(), "machine": platform.machine(),
        "repeats": REPEATS,
        "forward": {**timed(jax.jit(forward), scale), **memory(forward, scale)},
        "tangent": {**timed(jax.jit(tangent), scale), **memory(tangent, scale)},
        "gradient_checkpointed": {**timed(jax.jit(checkpointed), scale), **memory(checkpointed, scale)},
        "gradient_plain": {**timed(jax.jit(plain), scale), **memory(plain, scale)},
        "factor_lu": timed(jax.jit(lambda m: jax.scipy.linalg.lu_factor(m)), matrix),
        "factor_cholesky": timed(jax.jit(lambda m: jax.scipy.linalg.cho_factor(m)), matrix),
        "solve_six": timed(jax.jit(jnp.linalg.solve), matrix, motions),
        "smallest_eigenvalue": float(np.linalg.eigvalsh(np.asarray(matrix))[0]) if count <= 4000 else None,
    }
    found["graph"] = graph_profile()
    found["device_peak_bytes"] = device_peak()
    out.write_text(json.dumps(found, indent=1, default=str) + "\n")
    print(json.dumps(found, indent=1, default=str))


def graph_profile() -> dict[str, object]:
    from mrs_cases import mrs, free_space
    from maddening.core.simulation.profiler import profile_graph
    from script_a import fl9

    from mime.units import mm

    built = free_space("budget", {"screw": fl9()}, mrs(mm(0.178), patches=("lateral",))).build()
    node = built.graph.get_node("drag:screw")
    report = profile_graph(built.graph, n_steps=5, n_warmup=1, measure_coupling=False)
    return {"counts": node.constants()["counts"], "samples": node.sample_count,
            "jit_compile_ms": report.jit_compile_ms, "mean_step_ms": report.mean_step_ms,
            "median_step_ms": report.median_step_ms, "dispatch_floor_ms": report.dispatch_floor_ms,
            "node_times_ms": report.node_times_ms, "counts_report": str(report.counts)}


if __name__ == "__main__":
    main(Path(sys.argv[1]))
