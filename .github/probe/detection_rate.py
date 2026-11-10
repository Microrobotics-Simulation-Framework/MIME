# SPDX-License-Identifier: Apache-2.0
"""Measure how often a control catches its planted error: "a control must fail every run", measured and not assumed.

    python .github/probe/detection_rate.py --node tests/x/test_oracle.py::test_catches --plant-var MIME_PLANT_ERROR \
        --runs 40 --output probe/out/detection_rate.json
    python .github/probe/detection_rate.py --script probe/oracle.py --plant-var MIME_PLANT_ERROR --plant-value 1e-6 \
        --runs 40 --output probe/out/detection_rate.json -- --size 3

The target is a pytest node id (`--node`) or a script (`--script`, with its arguments after `--`). It reads the
environment variable named by `--plant-var` and, when it is set, plants its error (a test of the oracle that a planted
error is detected, a probe script that runs an oracle over a planted draw). Each run is a fresh process with its own seed
in `--seed-var` (`--seed-base` plus the run's index; the target takes the seed or ignores it), once with the plant on
(`--plant-value`) and once, at the same seeds, with the variable removed from the environment (the clean arm, which says how
often the control fires with nothing planted).

A run is **caught** when the target exits 1 (a failing test, a script that found the error and says so by exiting 1),
**not caught** when it exits 0, and **inconclusive** otherwise: a crash, a timeout, a pytest collection error, and a pytest
run in which no test passed or failed (all skipped, none collected), because a test that did not run is not a test that
missed. Inconclusive runs count against `caught/N`. With `--hypothesis-seed` a node also gets `--hypothesis-seed=<seed>`.

The JSON written holds, for the planted arm: `caught`, `runs`, `detection_rate` (`caught/runs`), `lower_bound_95` (the exact
one-sided 95 % lower confidence bound on the rate, which for N runs all caught is 0.05^(1/N): 40 clean catches cannot
show a rate above 0.93), the `missed_seeds` and the `inconclusive_seeds` with the last lines of their output; and for the
clean arm `flagged` and its seeds. `every_run` is true when every planted run was caught and no clean run was flagged or
inconclusive. The exit status is 0 when the measurement ran; `--strict` makes anything but `every_run` exit 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CAUGHT, NOT_CAUGHT, INCONCLUSIVE = "caught", "not_caught", "inconclusive"
_SUMMARY_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)")


def pytest_counts(output: str) -> dict[str, int]:
    """The counts on pytest's last summary line ("1 failed, 2 passed, 1 skipped in 0.1s"); empty where there is none."""
    for line in reversed(output.strip().splitlines()):
        found = _SUMMARY_COUNT.findall(line)
        if found and " in " in line:
            counts: dict[str, int] = {}
            for number, word in found:
                counts[word.rstrip("s") if word.startswith("error") else word] = int(number)
            return counts
    return {}


def classify(exit_code: int | None, output: str, *, pytest: bool) -> str:
    """CAUGHT, NOT_CAUGHT or INCONCLUSIVE for one finished run (`exit_code` None: it timed out)."""
    if exit_code is None or exit_code not in (0, 1):
        return INCONCLUSIVE
    if not pytest:
        return CAUGHT if exit_code == 1 else NOT_CAUGHT
    counts = pytest_counts(output)
    if exit_code == 1:
        return CAUGHT if counts.get("failed", 0) > 0 else INCONCLUSIVE  # an error in setup or collection is not a catch
    return NOT_CAUGHT if counts.get("passed", 0) > 0 else INCONCLUSIVE  # all skipped, or none collected: nothing ran


def lower_bound_95(caught: int, runs: int) -> float:
    """The exact (Clopper-Pearson) one-sided 95 % lower bound on a detection rate after `caught` of `runs`."""
    if caught <= 0 or runs <= 0:
        return 0.0
    if caught >= runs:
        return 0.05 ** (1.0 / runs)

    def upper_tail(p: float) -> float:  # P(X >= caught) at rate p
        return sum(math.comb(runs, k) * p**k * (1 - p) ** (runs - k) for k in range(caught, runs + 1))

    low, high = 0.0, 1.0
    for _ in range(60):  # the lowest p at which `caught` or more is still reachable at 5 %
        middle = (low + high) / 2
        low, high = (middle, high) if upper_tail(middle) < 0.05 else (low, middle)
    return (low + high) / 2


def target_command(arguments: argparse.Namespace, seed: int) -> list[str]:
    if arguments.node:
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", arguments.node]
        if arguments.hypothesis_seed:
            command.append(f"--hypothesis-seed={seed}")
        return command
    return [sys.executable, str(arguments.script), *arguments.script_arguments]


def run_once(arguments: argparse.Namespace, seed: int, planted: bool) -> dict[str, object]:
    environment = {key: value for key, value in os.environ.items() if key != arguments.plant_var}
    environment[arguments.seed_var] = str(seed)
    if planted:
        environment[arguments.plant_var] = arguments.plant_value
    started = time.monotonic()
    try:
        done = subprocess.run(target_command(arguments, seed), cwd=arguments.cwd, env=environment, text=True,
                              capture_output=True, timeout=arguments.timeout, check=False)
        exit_code: int | None = done.returncode
        output = (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired as expired:
        exit_code = None
        raw = (expired.stdout or b"") + (expired.stderr or b"")
        output = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    outcome = classify(exit_code, output, pytest=bool(arguments.node))
    record: dict[str, object] = {"seed": seed, "outcome": outcome, "exit_code": exit_code,
                                 "seconds": round(time.monotonic() - started, 3)}
    if outcome != (CAUGHT if planted else NOT_CAUGHT):  # what a reader of a miss or a false alarm needs
        record["output_tail"] = output.strip().splitlines()[-6:]
    return record


def arm_summary(records: list[dict[str, object]], planted: bool) -> dict[str, object]:
    runs = len(records)
    caught = sum(1 for record in records if record["outcome"] == CAUGHT)
    summary: dict[str, object] = {
        "runs": runs, "caught" if planted else "flagged": caught,
        "inconclusive": sum(1 for record in records if record["outcome"] == INCONCLUSIVE),
        "inconclusive_seeds": [record["seed"] for record in records if record["outcome"] == INCONCLUSIVE],
    }
    if planted:
        summary["detection_rate"] = caught / runs if runs else None
        summary["caught_of_runs"] = f"{caught}/{runs}"
        summary["lower_bound_95"] = lower_bound_95(caught, runs)
        summary["missed_seeds"] = [record["seed"] for record in records if record["outcome"] == NOT_CAUGHT]
    else:
        summary["flagged_seeds"] = [record["seed"] for record in records if record["outcome"] == CAUGHT]
    summary["records"] = [record for record in records if "output_tail" in record]  # the runs worth reading; the rest are counts
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--node", help="a pytest node id, run as `python -m pytest -q <node>`")
    target.add_argument("--script", type=Path, help="a Python script; its arguments follow `--`")
    parser.add_argument("--plant-var", required=True, help="the environment variable the target reads to plant its error")
    parser.add_argument("--plant-value", default="1", help="its value on the planted arm (default 1)")
    parser.add_argument("--runs", type=int, default=30, help="runs on each arm, each a fresh process (default 30)")
    parser.add_argument("--clean-runs", type=int, default=None, help="runs on the clean arm (default: --runs)")
    parser.add_argument("--seed-var", default="DETECTION_SEED", help="the environment variable that carries the seed")
    parser.add_argument("--seed-base", type=int, default=0, help="run i gets seed base + i, on both arms")
    parser.add_argument("--hypothesis-seed", action="store_true", help="also pass --hypothesis-seed=<seed> to a node")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds a run may take (a timeout is inconclusive)")
    parser.add_argument("--cwd", type=Path, default=None, help="the directory the target runs in (default: the current one)")
    parser.add_argument("--output", required=True, type=Path, help="where the JSON goes")
    parser.add_argument("--strict", action="store_true", help="exit 1 unless every planted run was caught and the clean arm is quiet")
    parser.add_argument("script_arguments", nargs=argparse.REMAINDER, help="after `--`: the script's arguments")
    arguments = parser.parse_args(argv)
    if arguments.script_arguments and arguments.script_arguments[0] == "--":
        arguments.script_arguments = arguments.script_arguments[1:]
    if arguments.node and arguments.script_arguments:
        parser.error("arguments after `--` go to a --script; a --node takes none")
    if arguments.runs < 1:
        parser.error("--runs must be at least 1")
    clean_runs = arguments.runs if arguments.clean_runs is None else arguments.clean_runs

    planted = [run_once(arguments, arguments.seed_base + index, True) for index in range(arguments.runs)]
    clean = [run_once(arguments, arguments.seed_base + index, False) for index in range(clean_runs)]
    planted_summary, clean_summary = arm_summary(planted, True), arm_summary(clean, False)
    every_run = (planted_summary["caught"] == arguments.runs and clean_summary["flagged"] == 0
                 and clean_summary["inconclusive"] == 0)
    report = {
        "target": arguments.node or str(arguments.script), "script_arguments": arguments.script_arguments,
        "plant_var": arguments.plant_var, "plant_value": arguments.plant_value, "seed_var": arguments.seed_var,
        "seed_base": arguments.seed_base, "python": sys.version.split()[0],
        "planted": planted_summary, "clean": clean_summary, "every_run": every_run,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"detection_rate: caught {planted_summary['caught_of_runs']} (rate {planted_summary['detection_rate']:.3f}, "
          f"95% lower bound {planted_summary['lower_bound_95']:.3f}); clean arm flagged {clean_summary['flagged']}/{clean_runs}; "
          f"inconclusive {planted_summary['inconclusive']} planted, {clean_summary['inconclusive']} clean; "
          f"{'every run' if every_run else 'NOT every run'}; wrote {arguments.output}")
    return 0 if every_run or not arguments.strict else 1


if __name__ == "__main__":
    sys.exit(main())
