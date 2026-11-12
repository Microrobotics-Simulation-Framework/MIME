"""Where the closed loop's gradient breaks (camera spike R2, item 6): the objective and its reverse-mode derivative
along a sweep of one parameter (``limit`` across the currents the run reaches, so that a limit switches on and off;
``kp`` across the gains where the active sets along the run change), with the pattern of coils at their limit at every
tick, so that each kink is located and the derivative's jump there measured against the neighbours' differences.

``--arm``, ``--sweep limit|kp``, ``--low``, ``--high``, ``--points``, ``--steps``, ``--seed`` (the draw of gradcheck.py);
writes JSON to ``--output``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_case  # noqa: E402
from gradcheck import NAMES, drawn  # noqa: E402

jnp: Any = jax.numpy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="camera")
    parser.add_argument("--sweep", default="limit")
    parser.add_argument("--low", type=float, default=10.0)
    parser.add_argument("--high", type=float, default=40.0)
    parser.add_argument("--points", type=int, default=121)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    from mime.build._maddening import differentiated_run, full_state, graph_params

    dtype = np.dtype(jnp.zeros(()).dtype)
    gains, step = drawn(args.seed)
    loop_case.STEP = step
    graph = loop_case.experiment(args.arm, **gains).build().graph
    graph.reset_state()
    state = {node: dict(fields) for node, fields in full_state(graph).items()}
    params = graph_params(graph)
    times = (np.arange(args.steps) + 1) * loop_case.DT
    reference = jnp.asarray(np.asarray(loop_case.reference(jnp.asarray(times))).T, dtype=dtype)
    run = differentiated_run(graph, args.steps, (), (("sphere", "pose"), ("controller", "write_currents")),
                             checkpoint=True)

    def objective(value: Any) -> tuple[Any, Any]:
        nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
        for name in NAMES:
            nodes["controller"][f"setting_{name}"] = value if name == args.sweep else jnp.asarray(gains[name], dtype)
        _, _, out = run(state, {**params, "nodes": nodes})
        miss = out["sphere"]["pose"][:, :3] - reference
        return jnp.sum(miss * miss) / 1e-6 / args.steps, out["controller"]["write_currents"]

    both = jax.jit(jax.value_and_grad(objective, has_aux=True))
    grid = np.linspace(args.low, args.high, args.points)
    rows = []
    for value in grid:
        (j, currents), g = both(jnp.asarray(value, dtype=dtype))
        currents = np.asarray(currents)[:: loop_case.PERIOD_STEPS]
        limit = value if args.sweep == "limit" else gains["limit"]
        signs = np.where(np.abs(currents) >= limit * (1 - 1e-6), np.sign(currents), 0).astype(int)
        rows.append({"value": float(value), "J": float(j), "dJ": float(g), "finite": bool(np.isfinite(float(g))),
                     "ticks_at_limit": int(np.any(signs != 0, axis=1).sum()),
                     "pattern": "".join("".join({-1: "-", 0: ".", 1: "+"}[int(c)] for c in tick) for tick in signs[:12]),
                     "peak_current": float(np.max(np.abs(currents)))})
    for i in range(1, len(rows) - 1):
        rows[i]["neighbour_cd"] = (rows[i + 1]["J"] - rows[i - 1]["J"]) / (grid[i + 1] - grid[i - 1])
    summary = {"arm": args.arm, "sweep": args.sweep, "dtype": str(dtype), "steps": args.steps, "seed": args.seed,
               "gains": gains, "step": step, "rows": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    for row in rows[:: max(1, len(rows) // 20)]:
        print(json.dumps(row))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
