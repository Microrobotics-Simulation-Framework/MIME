"""The budget of the tube's correction, CPU, run by hand on a full table after ``mime assets fetch`` (its values do
not change what a read costs, so the 2026-05-21 table serves although it fails its check):

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/differential/tube_budget.py [along,around ...]

prints, at each grid of FL-9's lateral samples (80 by 39 by default, 3,120 samples: the 9,360-unknown system, the
surrogate's training mesh's count), the table's load and
compile time, the correction's forward time (median of three after the compiling run) and XLA's memory analysis of
its forward program and of the reverse program of the whole resistance (the correction inside the node's
checkpointed solve), with the table's read alone beside the assembly it is part of. Not collected by pytest."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

TABLE = "cylinder_wall_r40_R2.500"
BODY_RADIUS, VISCOSITY, RATIO = 1.56e-3, 1e-3, 2.5


def memory(compiled: Any) -> float:
    held = compiled.memory_analysis()
    return (held.temp_size_in_bytes + held.output_size_in_bytes) / 1e9


def main(grids: list[tuple[int, int]]) -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from script_a import fl9

    from mime.assets import locate, reference
    from mime.core.tables import evaluate
    from mime.nodes.hydro.mrs import blob_squared, resistance, tube_correction, tube_table_from

    started = time.perf_counter()
    table = tube_table_from(str(locate(reference(TABLE))))
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    arrays = table.arrays(jnp.float64)
    compiled_table = time.perf_counter() - started
    print(f"table: read and mirrored {loaded:.2f} s, compiled for float64 {compiled_table:.2f} s, "
          f"{arrays.values.nbytes / 1e9:.2f} GB of values")
    tube = RATIO * BODY_RADIUS
    frame = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    kind = fl9()
    settings = {"radius": BODY_RADIUS, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2}
    for grid in grids:
        held = kind.samples({k: jnp.asarray(v) if isinstance(v, float) else v for k, v in settings.items()},
                            "lateral", grid)
        points, weights = jnp.asarray(held.points), jnp.asarray(held.weights)
        n = points.shape[0]
        parameters = {"centre": (0.0, 0.0, 0.0), "frame": frame, "radius": tube, "table": arrays}

        def correction(points: Any) -> Any:
            return tube_correction(points, parameters, VISCOSITY).matrix

        def read_alone(points: Any) -> Any:
            local = points[:, [1, 2, 0]] / tube
            rho = jnp.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
            angle = jnp.abs(jnp.arctan2(local[:, None, 1], local[:, None, 0])
                            - jnp.arctan2(local[None, :, 1], local[None, :, 0])) % np.pi
            along = jnp.abs(local[:, None, 2] - local[None, :, 2])
            return evaluate(arrays, rho[:, None], rho[None, :], angle, along, owner="tube.table").value

        eps2 = blob_squared(weights, 0.5)
        weights_6 = jnp.asarray(np.random.default_rng(0).normal(size=(6, 6)))

        def loss(points: Any) -> Any:
            solved = jax.checkpoint(lambda p: resistance(p, eps2, VISCOSITY, p.mean(axis=0), correction(p)))
            return jnp.sum(solved(points) * weights_6)

        report = {}
        for name, function in (("correction", correction), ("the table's read alone", read_alone),
                               ("reverse, the whole resistance", jax.grad(loss))):
            program = jax.jit(function).lower(points).compile()
            times = []
            for _ in range(4):
                started = time.perf_counter()
                jax.block_until_ready(program(points))
                times.append(time.perf_counter() - started)
            report[name] = (statistics.median(times[1:]), memory(program))
        print(f"{n} samples ({3 * n} unknowns): " + "; ".join(f"{name} {t:.2f} s, {gb:.2f} GB"
                                                             for name, (t, gb) in report.items()))


if __name__ == "__main__":
    main([(int(a), int(b)) for a, b in (arg.split(",") for arg in sys.argv[1:])] or [(80, 39)])
