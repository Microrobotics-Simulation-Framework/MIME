"""What leaving the compiled program once per control period costs, and what keeping the controller inside it costs:
A0's rig at its 0.5 ms step, run on the CPU for ``CHUNKS`` periods of 67 steps (about 30 Hz) of simulated time, with
no controller (the arm's joint positions a constant drive, one compiled scan), with the same graph chunked for a hook
that does nothing (the cost of the chunks alone), and closed by A0's controller every ``period`` steps (every step,
as the 0.2 rig called it, and every 67), as a callback (mode 2) and inside the compiled step (mode 1). Each is built
and run once to compile, the two timed apart, then timed over ``REPEATS`` further runs, each continuing from where
the last left off. The conditions alternate within a repeat, so that the machine's load drifts across all of them
alike, and the ratio of each to the plain run is taken within the repeat; the table gives medians and quartiles of the
ratios first and of the times second. The time spent moving every node's state to NumPy after each chunk (the run's
``_to_numpy``, which grows with the graph's total state), once the chunk's scan is finished, is measured in the same
runs. The load average is recorded
at the start and end. Not collected by pytest:

    JAX_PLATFORMS=cpu python tests/differential/controller_budget.py [chunks [repeats]]

Absolute times on a machine other sessions share are indicative; the ratios within a repeat are less so.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import chain_cases  # noqa: E402
import mime.control._run as run_module  # noqa: E402
from chain_cases import kinematic_rpm, tracked_rpm, tracking  # noqa: E402

from mime.experiment import Experiment  # noqa: E402
from mime.scene import Scene, Z  # noqa: E402
from mime.units import ms, standard_gravity  # noqa: E402

DT = 0.5
CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
STEPS = 67 * CHUNKS
SECONDS = STEPS * DT * 1e-3
PLAIN = "no controller, one scan"
MODES = {"callback": "a callback (mode 2)", "graph": "inside the compiled step (mode 1)"}
NUMPY_SECONDS = [0.0]


def at_half_a_millisecond(name: str) -> Experiment:
    return Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(DT))


def timing_numpy() -> None:
    """Wraps the run's move of a state to NumPy so that the time it takes is added up."""
    held = run_module._to_numpy

    def timed_move(state):
        jax.block_until_ready(state)  # JAX dispatches asynchronously: wait for the scan so that only the move is timed
        start = time.perf_counter()
        found = held(state)
        NUMPY_SECONDS[0] += time.perf_counter() - start
        return found

    run_module._to_numpy = timed_move


def quartiles(values: list[float]) -> dict[str, float]:
    low, median, high = np.percentile(values, [25, 50, 75])
    return {"q1": float(low), "median": float(median), "q3": float(high)}


def main() -> None:
    warnings.simplefilter("ignore")
    chain_cases.experiment = at_half_a_millisecond
    timing_numpy()
    runs = {PLAIN: (kinematic_rpm().build(), {}),
            "no controller, chunks of 67 for a hook": (kinematic_rpm().build(), {"hook": lambda c: None,
                                                                                  "chunk": ms(DT * 67)})}
    first: dict[str, dict[str, float]] = {}
    for period in (67, 1):
        for mode in ("callback", "graph"):
            start = time.perf_counter()
            built = tracked_rpm(controller=tracking(), rate=ms(DT * period), latency=None, mode=mode).build()
            runs[f"controller every {period} steps, {MODES[mode]}"] = (built, {})
            first[f"controller every {period} steps, {MODES[mode]}"] = {"build_s": time.perf_counter() - start}
    for name, (built, options) in runs.items():
        start = time.perf_counter()
        built.run(ms(DT * STEPS), **options)
        first.setdefault(name, {})["first_run_s"] = time.perf_counter() - start
    walls: dict[str, list[float]] = {name: [] for name in runs}
    numpy_share: dict[str, list[float]] = {name: [] for name in runs}
    loads = [os.getloadavg()[0]]
    for _ in range(REPEATS):
        for name, (built, options) in runs.items():
            NUMPY_SECONDS[0] = 0.0
            start = time.perf_counter()
            built.run(ms(DT * STEPS), **options)
            walls[name].append(time.perf_counter() - start)
            numpy_share[name].append(NUMPY_SECONDS[0])
        loads.append(os.getloadavg()[0])
    ratios = {name: quartiles([w / p for w, p in zip(walls[name], walls[PLAIN], strict=True)]) for name in runs}
    inside_over_callback = {
        period: quartiles([g / c for g, c in zip(walls[f"controller every {period} steps, {MODES['graph']}"],
                                                  walls[f"controller every {period} steps, {MODES['callback']}"],
                                                  strict=True)]) for period in (67, 1)}
    table = {name: {"ratio_to_plain_within_repeat": ratios[name],
                    "wall_per_simulated_s": quartiles([w / SECONDS for w in walls[name]]),
                    "to_numpy_s_per_simulated_s": quartiles([t / SECONDS for t in numpy_share[name]]),
                    "to_numpy_share_of_wall": quartiles([t / w for t, w in zip(numpy_share[name], walls[name],
                                                                              strict=True)])}
             for name in runs}
    print(json.dumps({"simulated_s": SECONDS, "steps": STEPS, "dt_ms": DT, "repeats": REPEATS,
                      "load_average_1min_at_start_and_after_each_repeat": loads,
                      "mode_1_over_mode_2_within_repeat_by_period": inside_over_callback,
                      "build_and_first_run_s": first, "runs": table}, indent=1))


if __name__ == "__main__":
    main()
