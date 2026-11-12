#!/usr/bin/env python3
"""discriminating_range.py: the smallest planted error an oracle catches on every draw (its resolution).

A gate that says "the oracle catches a planted error" says nothing until it says how small: below its resolution the
oracle misses some draws, because the harness's rounding floor moves with the draw (the contact oracle missed a 1 um
error one run in eight; its floor was about 15 um on a typical float32 draw). This finds that size, so that a gate states
the range it can discriminate and plants above it.

    discriminating_range.py --oracle my_oracles:contact_flags --path tests/oracles --lo 1e-9 --hi 1e-1 \\
        --draws 200 --output-json resolution.json
    discriminating_range.py --selftest

The oracle is `function(size: float, seed: int) -> bool`: True when it flags a plant of `size` (any units, one
positive number) at the draw of `seed`; `size == 0.0` is the clean call, which must be False (it is run on every draw: the
false-alarm rate). The scan is a grid of `--points` sizes, log-spaced (`--scale linear` for a linear grid) between `--lo`
and `--hi`, `--draws` draws each (seeds `--seed-base` + 0 .. draws-1, the same at every size, so a size's misses are
the draws' own). Reported:

  * `curve`: for each size, caught/draws and the missed seeds (up to ten);
  * `resolution`: the smallest grid size from which every larger grid size is caught on every draw; None if even `--hi` has
    misses (the oracle does not discriminate in this range, which is a result, not a failure of the script);
  * `resolution_refined`: a bisection (`--refine` steps) between the largest grid size with a miss and the resolution, which
    ASSUMES the oracle is monotone between those two neighbours (it is not checked inside the gap);
  * `non_monotone`: sizes below the resolution that were caught on every draw (a hole in the curve), so the resolution is
    read from the top and not from the first all-caught size;
  * `plant_at`: `--safety` (default 10) times the refined resolution: the size a gate plants, so the control is above the
    resolution by a stated factor;
  * `false_alarms`: size 0 on every draw;
  * `bound_95`: with `draws` all caught at the resolution the rate is no lower than 0.05^(1/draws) at 95 % (200 draws: 0.985);
    a rate of exactly 1 is never shown by draws.

JSON out; exit 0 whatever it found, 1 with `--strict` if there are false alarms or no resolution in the range.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

Oracle = Callable[[float, int], bool]


def grid(lo: float, hi: float, points: int, scale: str) -> list[float]:
    if points < 2 or not 0 < lo < hi:
        raise ValueError("a grid needs 0 < lo < hi and at least 2 points")
    if scale == "log":
        ratio = (hi / lo) ** (1.0 / (points - 1))
        return [lo * ratio**index for index in range(points - 1)] + [hi]
    return [lo + (hi - lo) * index / (points - 1) for index in range(points)]


def curve(oracle: Oracle, sizes: list[float], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for size in sizes:
        missed = [seed for seed in seeds if not oracle(size, seed)]
        rows.append({"size": size, "caught": len(seeds) - len(missed), "draws": len(seeds), "missed_seeds": missed[:10]})
    return rows


def resolution_of(rows: list[dict[str, Any]]) -> tuple[int | None, list[float]]:
    """The index of the smallest size from which every larger grid size is caught on every draw, and the sizes below it
    that were caught on every draw anyway (holes)."""
    index: int | None = None
    for position in range(len(rows) - 1, -1, -1):
        if rows[position]["caught"] != rows[position]["draws"]:
            break
        index = position
    holes = [rows[position]["size"] for position in range(0, index if index is not None else 0)
             if rows[position]["caught"] == rows[position]["draws"]]
    return index, holes


def refine(oracle: Oracle, low: float, high: float, seeds: list[int], steps: int, scale: str) -> float:
    """Bisect between a size with a miss (low) and one caught on every draw (high); the caught side is returned."""
    for _ in range(steps):
        middle = math.sqrt(low * high) if scale == "log" else (low + high) / 2.0
        if all(oracle(middle, seed) for seed in seeds):
            high = middle
        else:
            low = middle
    return high


def measure(oracle: Oracle, lo: float, hi: float, *, points: int = 25, draws: int = 100, seed_base: int = 0,
            scale: str = "log", refine_steps: int = 12, safety: float = 10.0) -> dict[str, Any]:
    seeds = list(range(seed_base, seed_base + draws))
    sizes = grid(lo, hi, points, scale)
    rows = curve(oracle, sizes, seeds)
    index, holes = resolution_of(rows)
    clean = [seed for seed in seeds if oracle(0.0, seed)]
    report: dict[str, Any] = {"lo": lo, "hi": hi, "points": points, "draws": draws, "seed_base": seed_base, "scale": scale,
                              "curve": rows, "non_monotone": holes,
                              "false_alarms": {"flagged": len(clean), "draws": draws, "seeds": clean[:10]},
                              "resolution": None, "resolution_refined": None, "plant_at": None, "bound_95": None}
    if index is not None:
        resolution = sizes[index]
        refined = refine(oracle, sizes[index - 1], resolution, seeds, refine_steps, scale) if index > 0 else resolution
        report.update(resolution=resolution, resolution_refined=refined, plant_at=refined * safety, safety=safety,
                      bound_95=0.05 ** (1.0 / draws))
    return report


def selftest() -> int:
    """A toy oracle with a known, draw-dependent floor: it flags a plant of size s at seed k iff s > floor(k), the floors
    spread over [1e-6, 2e-6] by the seed. The resolution must come out at the largest floor, 2e-6; a plant below the
    smallest floor is never caught, and the clean call never flags."""
    def floor(seed: int) -> float:
        return 1e-6 * (1.0 + random.Random(seed).random())

    def toy(size: float, seed: int) -> bool:
        return size > floor(seed)

    draws = 200
    largest = max(floor(seed) for seed in range(draws))
    report = measure(toy, 1e-8, 1e-3, points=26, draws=draws, refine_steps=20)
    refined = report["resolution_refined"]
    ok = (refined is not None and largest <= refined <= largest * 1.001 and report["false_alarms"]["flagged"] == 0
          and report["curve"][0]["caught"] == 0 and not report["non_monotone"]
          and any(0 < row["caught"] < draws for row in report["curve"]))
    # A control that must fail: a plant at the first grid size above the smallest floor but below the largest is caught
    # on some draws and missed on others, which is what the scan exists to show.
    middle = [row for row in report["curve"] if 0 < row["caught"] < draws]
    print(f"selftest: largest floor {largest:.4g}, refined resolution {refined}, {len(middle)} grid sizes partly caught: "
          f"{'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="check the script on a toy oracle with a known floor")
    parser.add_argument("--oracle", help="module:function(size, seed) -> bool, True when the oracle flags the plant")
    parser.add_argument("--path", help="a directory to put on sys.path before importing the oracle")
    parser.add_argument("--lo", type=float)
    parser.add_argument("--hi", type=float)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--scale", choices=("log", "linear"), default="log")
    parser.add_argument("--refine", type=int, default=12, help="bisection steps for the refined resolution")
    parser.add_argument("--safety", type=float, default=10.0, help="plant_at is this many times the resolution")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.selftest:
        return selftest()
    if not (arguments.oracle and arguments.lo and arguments.hi and arguments.output_json):
        parser.error("--oracle, --lo, --hi and --output-json are required (or --selftest)")
    if arguments.path:
        sys.path.insert(0, str(Path(arguments.path).resolve()))
    module, _, name = arguments.oracle.partition(":")
    oracle: Oracle = getattr(importlib.import_module(module), name)
    report = measure(oracle, arguments.lo, arguments.hi, points=arguments.points, draws=arguments.draws,
                     seed_base=arguments.seed_base, scale=arguments.scale, refine_steps=arguments.refine,
                     safety=arguments.safety)
    report["oracle"] = arguments.oracle
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["resolution"] is None:
        print(f"discriminating_range: misses remain even at {arguments.hi}: no resolution in this range; wrote "
              f"{arguments.output_json}")
    else:
        print(f"discriminating_range: resolution {report['resolution_refined']:.4g} (grid {report['resolution']:.4g}), "
              f"plant at {report['plant_at']:.4g} ({arguments.safety:g}x); {report['false_alarms']['flagged']}/"
              f"{arguments.draws} false alarms; non-monotone holes {len(report['non_monotone'])}; wrote {arguments.output_json}")
    bad = report["resolution"] is None or report["false_alarms"]["flagged"] > 0
    return 1 if arguments.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
