"""The table evaluator's per-lookup cost against the 0.2 wall-table lookup's, on the 0.2 table's own grid (40 x 40 x
64 x 128 knots in rho, rho0, the angle and the axial offset, from ``references/port_table.npz``, a 3 x 3 value at each
node), in float32, compiled, on the CPU:

    JAX_PLATFORMS=cpu uv run --no-sync python tests/differential/table_budget.py [--queries N] [--repeats R]

The baseline is the 0.2 JAX twin's lookup, its lines copied below unchanged (``cylinder_wall_table.py``, lines 369
to 392 at MIME-port041 6485516: a cell index and a weight per axis, then the sixteen corners summed), compiled with
``jax.jit``; the 0.2 code ran them eagerly inside its assembly, so compiled is the faster form of it. MIME reads the
same table, its values drawn at random (a read's cost depends on the grid's shape, not its values), at the same random
queries inside the grid, linear and, for comparison, cubic. Each is compiled once, then timed over repeats; the median
and the spread (quartiles, least and most) are printed per lookup, with the machine's load. ``--baseline-only`` times
the baseline alone and imports nothing of MIME's, so that it can run in the port's own environment as a cross-check.
Not collected by pytest."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REFERENCE = Path(__file__).parent / "references" / "port_table.npz"


def baseline_lookup(table_rho, table_dphi, table_dz, table_data, rho_i, rho_j, dphi_abs, dz_abs):
    """The 0.2 JAX twin's lookup (``cylinder_wall_table.py:369-392``), unchanged but for its inputs, which the twin
    computed from points just above it."""

    def _interp_idx_weight(grid, vals):
        idx = jnp.searchsorted(grid, vals, side='right') - 1
        idx = jnp.clip(idx, 0, len(grid) - 2)
        lo = grid[idx]
        hi = grid[idx + 1]
        w = jnp.where(hi > lo, (vals - lo) / (hi - lo), 0.0)
        return idx, jnp.clip(w, 0.0, 1.0)

    i0, w0 = _interp_idx_weight(table_rho, rho_i.ravel())
    i1, w1 = _interp_idx_weight(table_rho, rho_j.ravel())
    i2, w2 = _interp_idx_weight(table_dphi, dphi_abs.ravel())
    i3, w3 = _interp_idx_weight(table_dz, dz_abs.ravel())

    G_cyl = jnp.zeros((rho_i.size, 3, 3))
    for d0 in range(2):
        for d1 in range(2):
            for d2 in range(2):
                for d3 in range(2):
                    wt = ((1 - w0) if d0 == 0 else w0) * \
                         ((1 - w1) if d1 == 0 else w1) * \
                         ((1 - w2) if d2 == 0 else w2) * \
                         ((1 - w3) if d3 == 0 else w3)
                    G_cyl = G_cyl + wt[:, None, None] * table_data[
                        i0 + d0, i1 + d1, i2 + d2, i3 + d3]
    return G_cyl


def timed(function, arguments, repeats: int) -> dict:
    """``function`` compiled (its first call), then timed over ``repeats`` calls: seconds per call, by statistic."""
    started = time.perf_counter()
    jax.block_until_ready(function(*arguments))
    compiled = time.perf_counter() - started
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(function(*arguments))
        times.append(time.perf_counter() - started)
    held = np.asarray(times)
    return {"first_call_s": compiled, "median_s": float(np.median(held)), "q1_s": float(np.quantile(held, 0.25)),
            "q3_s": float(np.quantile(held, 0.75)), "min_s": float(held.min()), "max_s": float(held.max())}


def per_lookup(stats: dict, queries: int) -> str:
    ns = {key: value / queries * 1e9 for key, value in stats.items() if key != "first_call_s"}
    return (f"{ns['median_s']:.1f} ns per lookup (quartiles {ns['q1_s']:.1f} to {ns['q3_s']:.1f}, least "
            f"{ns['min_s']:.1f}, most {ns['max_s']:.1f}); first call {stats['first_call_s']:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1 << 16)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--cubic", action="store_true", help="also time MIME's cubic read")
    options = parser.parse_args()

    reference = np.load(REFERENCE)
    grid = (reference["full_rho"], reference["full_rho"], reference["full_angle"], reference["full_offset"])
    rng = np.random.default_rng(options.seed)
    data = rng.standard_normal((*(len(g) for g in grid), 3, 3), dtype=np.float32)
    query = np.stack([rng.uniform(g[0], g[-1], options.queries) for g in grid], axis=-1).astype(np.float32)
    results: dict = {"queries": options.queries, "repeats": options.repeats, "jax": jax.__version__,
                     "numpy": np.__version__, "python": platform.python_version(), "device": jax.default_backend(),
                     "cpus": os.cpu_count(), "load_before": os.getloadavg()}

    knots = [jnp.asarray(g, dtype=jnp.float32) for g in grid]
    held = jnp.asarray(data)
    columns = [jnp.asarray(query[:, a]) for a in range(4)]
    compiled_baseline = jax.jit(baseline_lookup)
    results["baseline"] = timed(compiled_baseline, [knots[0], knots[2], knots[3], held, *columns], options.repeats)
    print(f"0.2 lookup, compiled: {per_lookup(results['baseline'], options.queries)}")

    if not options.baseline_only:
        from mime.core.tables import Axis, Table, evaluate
        import mime.core.tables._evaluate as evaluate_module

        axes = tuple(Axis(name, "1", tuple(float(v) for v in g))
                     for name, g in zip(("rho", "rho0", "angle", "offset"), grid, strict=True))
        schemes = ("linear", "cubic") if options.cubic else ("linear",)
        at = jnp.asarray(query)
        expected = np.asarray(compiled_baseline(knots[0], knots[2], knots[3], held, *columns))
        for scheme in schemes:
            arrays = Table(axes, data, scheme=scheme).arrays(np.float32)
            for method in ("scan", "compare_all"):
                evaluate_module.SEARCH_METHOD = method
                read = jax.jit(lambda table, q: evaluate(table, q, owner="toy.wall_table").value)
                key = f"mime_{scheme}_{method}"
                results[key] = timed(read, [arrays, at], options.repeats)
                if scheme == "linear":
                    found = np.asarray(read(arrays, at))
                    results[key]["largest_difference_from_baseline_relative"] = float(
                        np.abs(found - expected).max() / np.abs(data).max())
                print(f"MIME {scheme}, search {method}: {per_lookup(results[key], options.queries)}")
    results["load_after"] = os.getloadavg()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
