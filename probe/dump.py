"""Bits (gate 13): one switch-scan configuration built and run three base steps in the tree this runs in, every node's
state written to an NPZ (one CPU, one thread: bit_compare.py pins them)."""
import sys
from pathlib import Path

import numpy as np

name, out = sys.argv[1], sys.argv[2]
sys.path[:0] = [str(Path("tests").resolve()), str(Path("tests/saved").resolve())]
from switch_configurations import configurations  # noqa: E402

from mime.build import _maddening as maddening  # noqa: E402
from mime.units import s  # noqa: E402

built = configurations()[name]().build()
dt = float(built.report.dt)
period = built._runs.chunk or 1
whole = -(-3 // period) * period
if built._runs.chunk is None:
    built.run(s(whole * dt), hook=lambda chunk: None, chunk=s(dt))
else:
    built.run(s(whole * dt))
arrays = {}
for node, fields in maddening.full_state(built.graph).items():
    for field, value in fields.items():
        arrays[f"{node}/{field}"] = np.asarray(value)
Path(out).parent.mkdir(parents=True, exist_ok=True)
np.savez(out, **arrays)
