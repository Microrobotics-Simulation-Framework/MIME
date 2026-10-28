"""What leaving the compiled program once per control period costs: A0's rig at its 0.5 ms step, run on the CPU for
``CHUNKS`` periods of 67 steps (about 30 Hz) of simulated time, with no controller (the arm's joint positions a
constant drive, one compiled scan), with the same graph chunked for a hook that does nothing (the cost of the chunks
alone), and closed by A0's controller as a callback every ``period`` steps (every step, as the 0.2 rig called it, and
every 67). Each is built, run once to compile, then timed over a second run from where the first left it. Not
collected by pytest:

    JAX_PLATFORMS=cpu python tests/differential/controller_budget.py [chunks]
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import chain_cases  # noqa: E402
from chain_cases import kinematic_rpm, tracked_rpm, tracking  # noqa: E402

from mime.experiment import Experiment  # noqa: E402
from mime.scene import Scene, Z  # noqa: E402
from mime.units import ms, standard_gravity  # noqa: E402

DT = 0.5
CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
STEPS = 67 * CHUNKS
SECONDS = STEPS * DT * 1e-3


def at_half_a_millisecond(name: str) -> Experiment:
    return Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(DT))


def timed(built, **run) -> float:
    built.run(ms(DT * STEPS), **run)
    start = time.perf_counter()
    built.run(ms(DT * STEPS), **run)
    return time.perf_counter() - start


def main() -> None:
    warnings.simplefilter("ignore")
    chain_cases.experiment = at_half_a_millisecond
    steps = STEPS
    found: dict[str, dict[str, float]] = {}
    plain = timed(kinematic_rpm().build())
    found["no controller, one scan"] = {"wall_s": plain}
    found["no controller, chunks of 67 for a hook"] = {"wall_s": timed(kinematic_rpm().build(), hook=lambda c: None,
                                                                       chunk=ms(DT * 67))}
    for period in (67, 1):
        built = tracked_rpm(controller=tracking(), rate=ms(DT * period), latency=None).build()
        wall = timed(built)
        found[f"controller every {period} steps"] = {"wall_s": wall, "periods": steps / period,
                                                     "extra_per_period_ms": (wall - plain) / (steps / period) * 1e3}
    for row in found.values():
        row["wall_per_simulated_s"] = row["wall_s"] / SECONDS
    print(json.dumps({"simulated_s": SECONDS, "steps": steps, "dt_ms": DT, "runs": found}, indent=1))


if __name__ == "__main__":
    main()
