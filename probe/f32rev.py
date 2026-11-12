"""Float32 reverse mode on a built tube tree at the trees' float32 default tolerance (1e-4; the maintainer's item 63.3:
measured before it is offered, or refused by name): a drawn two-outlet tree (tests/flow/region_cases.py's cost case,
IMEX Euler with convection, a pulsatile inflow at Re about 10, a shared distal law), its node's own step under one
scan for a few steps, the loss the inlet's pressure and the first outlet's share of the flow; its derivative with
respect to the viscosity and the three resistances of the shared distal law, in reverse and forward mode, in the run's
dtype, against float64's reverse mode at float64's default tolerance (1e-8) on the same draw. One JSON line a draw and
interior; run once in float32 (JAX_ENABLE_X64=0) and once in float64. Predictions: tests/flow/
predictions_prescribed_inlet.json (gate_float32_tree_reverse).

``python probe/f32rev.py --seeds 1 4 --interior 0.1 --steps 4 --output probe/out/f32rev_f32_01.jsonl``"""
import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "tests/flow")
sys.path.insert(0, "tests")
import region_cases as rc  # noqa: E402

LEAVES = ("viscosity", "first", "second", "distal")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs=2, required=True)
    parser.add_argument("--interior", type=float, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = open(args.output, "a")
    dtype = "float64" if jax.config.jax_enable_x64 else "float32"
    for seed in range(*args.seeds):
        jax.clear_caches()
        start = time.time()
        built = rc.cost_tree(args.interior, seed).build()
        node = built.graph.get_node("water.flow")
        p = node._step_constants(None)  # noqa: SLF001
        law = next(i for i, name in enumerate(node.constants()["opening_laws"]) if name == "shared_distal")

        def values_of(q):
            row = jnp.asarray(q["opening_parameters"])[law]
            return jnp.stack([jnp.asarray(q["viscosity"]), row[0], row[1], row[2]])

        def with_values(values):
            parameters = jnp.asarray(p["opening_parameters"]).at[law, 0:3].set(values[1:])
            return {**p, "viscosity": values[0], "opening_parameters": parameters}

        def loss(values):
            q = with_values(values)
            state, _ = jax.lax.scan(lambda s, _: (node.advance(q, s, {}, node.delta_t), None),
                                    dict(node.initial_state()), None, length=args.steps)
            flux = state["opening_flux"]
            return jnp.stack([state["opening_pressure"][0], flux[1] / (flux[1] + flux[2])])

        values = values_of(p)
        scale = jnp.abs(values)

        def scaled(u):  # each leaf on its own scale: d loss / d log(leaf)
            return loss(values * (1.0 + u))

        zero = jnp.zeros_like(values)
        reverse = np.asarray(jax.jit(jax.jacrev(scaled))(zero), dtype=np.float64)
        forward = np.asarray(jax.jit(jax.jacfwd(scaled))(zero), dtype=np.float64)
        final = jax.jit(lambda v: loss(v))(values)
        state, _ = jax.jit(lambda: jax.lax.scan(lambda s, _: (node.advance(p, s, {}, node.delta_t), None),
                                                dict(node.initial_state()), None, length=args.steps))()
        row = {"seed": seed, "interior": args.interior, "dtype": dtype, "steps": args.steps,
               "tolerance": float(node.constants()["tolerance"]), "at_floor": bool(node.constants()["at_floor"]),
               "cells": [int(node.constants()[f"cells_{x}"]) for x in "xyz"],
               "loss": np.asarray(final, dtype=np.float64).tolist(), "reverse": reverse.tolist(),
               "forward": forward.tolist(), "leaves": LEAVES, "losses": ["inlet_pressure", "first_share"],
               "solve_flagged_steps": float(np.asarray(state["solve_report_sum"])[2:5].sum()),
               "seconds": time.time() - start}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
