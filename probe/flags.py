"""The open box's two ledger flags under decision (A) (the Inflow's opening prescribed): their clean readings over the
open rig's draws (``--lines``: tests/flow/open_cases.py's draws, 60 steps metered, every step read, the worst a draw)
and, as detection_rate.py's target (``--flag``), one draw with the plant OPEN_PLANT (a name of open_cases.PLANTS) at the
seed DETECTION_SEED: exit 1 where the flag reads above ``--line``. Predictions: tests/flow/predictions_prescribed_inlet.json
(gate_open_flags).

``python probe/flags.py --lines 0 50 --output probe/out/flags_f64_0.jsonl``"""
import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "tests/flow")
sys.path.insert(0, "tests")
from open_cases import PLANTS, STEPS, draw, point_node, scan  # noqa: E402

from mime.nodes.flow.fvm._flags import OPEN_FLAGS  # noqa: E402

NAMES = ("open_face_projection", "opening_kinetic_flux")


def readings(seed: int) -> dict:
    node = point_node(draw(seed))
    flags = [next(f for f in OPEN_FLAGS if f.name == name) for name in NAMES]
    final, found = scan(node, STEPS, lambda p, b, a: [f.reading(node, p, b, a) for f in flags])
    worst = {name: float(np.max(np.abs(np.asarray(values)))) for name, values in zip(NAMES, found, strict=True)}
    return {"seed": seed, "stopped_at": int(final["stopped_at"]), **worst}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, nargs=2)
    parser.add_argument("--flag", choices=NAMES)
    parser.add_argument("--line", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.lines:
        out = open(args.output, "w")
        for seed in range(*args.lines):
            jax.clear_caches()
            row = readings(seed)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
        return 0
    plant = os.environ.get("OPEN_PLANT")
    if plant:
        from _pytest.monkeypatch import MonkeyPatch

        PLANTS[plant](MonkeyPatch().setattr)
    row = readings(int(os.environ.get("DETECTION_SEED", "0")))
    print(json.dumps(row), flush=True)
    return 1 if row[args.flag] > args.line else 0


if __name__ == "__main__":
    sys.exit(main())
