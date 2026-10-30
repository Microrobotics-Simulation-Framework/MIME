"""The surrogate's inference cost per step through its type, against the node before it (phase 7i's budget): the drag
node of a revision before the type (its module read from git, by default ``e20704c``, the node 4c1 ported with 4c3's
and 7d's changes) against this tree's, each with script A's body and tube, the step compiled once, then called on one
pose ``CALLS`` times per repeat, the two alternating within each repeat so that the machine's load drifts across both
alike. Both read the same weights; the type adds a lookup at construction and nothing to the compiled step, which this
measures. Not collected by pytest:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python tests/differential/surrogate_budget.py [revision [calls [repeats]]]

It prints the median and quartiles of each node's time per call and of their ratio within a repeat, and checks that
both give the same resistance. Absolute times on a machine other sessions share are indicative."""

import json
import subprocess
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mime.data import asset
from mime.nodes.hydro import SurrogateDragNode

ROOT = Path(__file__).resolve().parents[2]
REVISION = sys.argv[1] if len(sys.argv) > 1 else "e20704c"
CALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
SETTINGS = dict(weights="dejongh_mlp_v2", weights_sha256=asset("dejongh_mlp_v2").sha256, viscosity=1e-3,
                helix_wavenumber=2.33, length=7.47e-3, radius=1.56e-3, outer_radius=1.33 * 1.56e-3,
                vessel_shape="cylinder", vessel_centre=(0.0, 0.0, 0.0), vessel_axis=(1.0, 0.0, 0.0),
                vessel_radius=3.175e-3, lubrication_epsilon=1.5e-4)
POSE = jnp.asarray((0.0, 0.0, -1.0e-3, 1.0, 0.0, 0.0, 0.0))


def node_of(revision: str) -> type:
    """The drag node's class as the module at ``revision`` defines it, imported under another name."""
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{revision}:src/mime/nodes/hydro/_surrogate_drag.py"],
                            capture_output=True, text=True, check=True).stdout
    module = types.ModuleType(f"surrogate_drag_{revision}")
    module.__dict__["__name__"] = module.__name__
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:_surrogate_drag.py", "exec"), module.__dict__)  # noqa: S102 - a tracked revision
    return module.SurrogateDragNode


def compiled(node: object):
    state = node.initial_state()
    return jax.jit(lambda pose: node.update(state, {"pose": pose}, 5e-4)["resistance"])


def per_call(step) -> float:
    step(POSE).block_until_ready()
    start = time.perf_counter()
    for _ in range(CALLS):
        out = step(POSE)
    out.block_until_ready()
    return (time.perf_counter() - start) / CALLS


def main() -> None:
    before = node_of(REVISION)("drag", 5e-4, **SETTINGS)
    after = SurrogateDragNode("drag", 5e-4, **SETTINGS)
    steps = {"before": compiled(before), "after": compiled(after)}
    same = float(np.max(np.abs(np.asarray(steps["before"](POSE)) - np.asarray(steps["after"](POSE)))))
    times: dict[str, list[float]] = {"before": [], "after": []}
    for repeat in range(REPEATS):
        for name in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
            times[name].append(per_call(steps[name]))
    ratio = np.array(times["after"]) / np.array(times["before"])

    def quartiles(values: np.ndarray) -> list[float]:
        return [float(v) for v in np.percentile(values, [25, 50, 75])]

    print(json.dumps({"revision": REVISION, "calls": CALLS, "repeats": REPEATS, "x64": bool(jax.config.jax_enable_x64),
                      "device": str(jax.devices()[0]), "largest_difference": same,
                      "before_us": [1e6 * v for v in quartiles(np.array(times["before"]))],
                      "after_us": [1e6 * v for v in quartiles(np.array(times["after"]))],
                      "ratio": quartiles(ratio)}, indent=1))


if __name__ == "__main__":
    main()
