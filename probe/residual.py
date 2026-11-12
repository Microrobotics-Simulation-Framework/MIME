"""The prescribed profile's own convective residual over the open rig's draws (decision (A)): at each step's start, the
convective power on the prescribed half cells (half weight) over the net kinetic energy flux into the box (the face
formula), and the scheme's whole convective power against that formula (tests/flow/open_cases.py's draws, 30 steps).
Predictions: tests/flow/predictions_prescribed_inlet.json (gate_convective_residual).

``python probe/residual.py --seeds 3000 3040 --output probe/out/residual.jsonl``"""
import argparse
import json
import math
import os
import sys

import jax

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "tests/flow")
sys.path.insert(0, "tests")
from open_cases import draw, point_node, scan  # noqa: E402


def reading(node):
    def read(p, before, after):
        grid = node.open_grid(p)
        u = tuple(jnp.asarray(before[f"velocity_{x}"]) for x in "xyz")
        conv = grid.convection(u)
        full = grid.inner(conv, u)
        face = 0.5 * grid.cell_volume * jnp.sum(conv[2][..., 0] * u[2][..., 0])
        kinetic, _ = grid.opening_kinetic_fluxes(u)
        formula = grid.spacing ** 2 * jnp.sum(0.5 * u[2][..., 0] ** 3)
        return jnp.stack([face, full, -kinetic[0], formula])
    return read


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs=2, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = open(args.output, "w")
    for seed in range(*args.seeds):
        jax.clear_caches()
        point = draw(seed)
        node = point_node(point, ledger=False)
        final, rows = scan(node, args.steps, reading(node))
        rows = np.asarray(rows, dtype=np.float64)[1:]  # the first step starts from the projected start, its face zero
        residual = np.abs(rows[:, 0]) / np.maximum(np.abs(rows[:, 1]), 1e-300)
        books = np.abs(rows[:, 2] - rows[:, 3]) / np.maximum(np.abs(rows[:, 3]), 1e-300)
        c = node.constants()
        frequency = float(np.asarray(c["opening_parameters"])[0][2])
        alpha = 0.5 * math.sqrt(2.0 * math.pi * abs(frequency))
        row = {"seed": seed, "cells": point.cells, "reynolds": point.reynolds, "alpha": alpha,
               "stopped_at": int(final["stopped_at"]), "residual_median": float(np.median(residual)),
               "residual_largest": float(np.max(residual)), "books_vs_formula_largest": float(np.max(books))}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
