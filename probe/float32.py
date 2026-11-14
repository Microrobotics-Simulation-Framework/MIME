"""AP2 gate 4: the exact allocation in float32 against float64, per array MIME ships.

Per array, ``--count`` problems of the swap test's draws (``draws.draw``, each kind), each rounded to float32; the law
in float32 on it and in float64 on the same rounded problem, so that the difference is the law's float32 arithmetic and
not the problem's rounding. The share of the limit, ``max |u32 - u64| / limit``, over every problem and over those
whose minimiser is unique (``allocation_draws.unique``), with the scaled rows' condition at the worst; and the swap
test's verdict on the float32 commands against BVLS on the same rounded problem (``allocation_draws.verdict``).
``python probe/float32.py --count 3000 --output probe/out/float32.json``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))
sys.path.insert(0, str(HERE.parent / "tests" / "control"))

from allocation_cases import OWNER  # noqa: E402
from allocation_draws import SYSTEMS, Problem, unique, verdict  # noqa: E402
from draws import draw  # noqa: E402

from mime.control import exact_allocation  # noqa: E402

jnp = jax.numpy
#: The coil array provider's declared float32 guard, a share of the field (``mime.fields._coil_array.FLOAT32_BUDGET``):
#: the line read here against the share of the limit.
BUDGET = 1e-4


def rounded(problem: Problem) -> Problem:
    as32 = (lambda v: np.asarray(v, dtype=np.float32).astype(np.float64))
    return Problem(problem.system, problem.kind, as32(problem.matrix), as32(problem.target), as32(problem.scales),
                   float(np.float32(problem.limit)))


def solve(problems: list[Problem], dtype: str) -> list[np.ndarray]:
    jax.config.update("jax_enable_x64", dtype == "float64")
    law = jax.jit(partial(exact_allocation, owner=OWNER))
    found = []
    for p in problems:
        a, b, s = (jnp.asarray(v, dtype=dtype) for v in (p.matrix, p.target, p.scales))
        limit = jnp.asarray(p.limit, dtype=dtype)
        found.append(np.asarray(law(a, b, lower=-limit, upper=limit, scales=s).commands, dtype=np.float64))
    return found


def condition(problem: Problem) -> float:
    a, _ = problem.scaled("float64")
    singular = np.linalg.svd(a / np.linalg.norm(a, axis=0), compute_uv=False)
    nonzero = singular[singular > 1e-12 * singular[0]]
    return float(singular[0] / nonzero[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, object] = {"count": args.count, "seed": args.seed, "budget": BUDGET}
    for index, system in enumerate(SYSTEMS):
        start = time.time()
        rng = np.random.default_rng(args.seed + index)
        problems = [rounded(draw(system, rng)) for _ in range(args.count)]
        u64, u32 = solve(problems, "float64"), solve(problems, "float32")
        jax.config.update("jax_enable_x64", True)
        share = np.asarray([float(np.max(np.abs(a - b))) / p.limit for a, b, p in zip(u32, u64, problems, strict=True)])
        alone = np.asarray([unique(p, "float32") for p in problems])
        kinds = np.asarray([p.kind for p in problems])
        failed = [(i, v) for i, (p, u) in enumerate(zip(problems, u32, strict=True)) if (v := verdict(p, u, "float32"))]
        worst = int(np.argmax(share))
        worst_unique = int(np.flatnonzero(alone)[np.argmax(share[alone])]) if alone.any() else None
        report[system] = {
            "problems": args.count, "unique": int(alone.sum()),
            "max_share_of_limit": float(share.max()), "worst_kind": str(kinds[worst]),
            "worst_condition": condition(problems[worst]),
            "max_share_unique": None if worst_unique is None else float(share[worst_unique]),
            "worst_unique_kind": None if worst_unique is None else str(kinds[worst_unique]),
            "worst_unique_condition": None if worst_unique is None else condition(problems[worst_unique]),
            "p99_share_unique": float(np.quantile(share[alone], 0.99)) if alone.any() else None,
            "median_share": float(np.median(share)),
            "max_share_by_kind": {k: float(share[kinds == k].max()) for k in sorted(set(kinds))},
            "past_budget": int((share > BUDGET).sum()),
            "swap_verdict_failures_float32": len(failed),
            "first_failures": [f"{i}: {v}" for i, v in failed[:3]],
            "seconds": time.time() - start,
        }
        print(system, json.dumps(report[system]))
    args.output.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
