"""G19: the structural cost estimate against the measured frame: node_measure.py's steady time per step (a frame a step)
beside the estimate the build report states, at one size, through the real node.
python scripts/camera_route2/g19.py WIDTH DTYPE OUT.json NODE_MEASURE, NODE_MEASURE the gate kit's node_measure.py
(a probe branch carries a copy)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

width, dtype, out, node_measure = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
here = os.path.dirname(os.path.abspath(__file__))
measured = out.replace(".json", "_measure.json")
subprocess.run([sys.executable, node_measure, "--case", "cases:coil_camera", "--path", here, "--arg",
                f"width={width}", "--steps", "20", "--repeats", "3", "--dtype", dtype, "--output-json", measured],
               check=True)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", dtype == "float64")
sys.path.insert(0, here)
import cases  # noqa: E402

built = cases.coil_camera(width).build()
notes = [line for lines in built.report.checked.values() for line in lines]
note = next(line for line in notes if "structural cost estimate" in line)
estimate = float(re.search(r"estimate ([0-9.e+]+) M flops", note).group(1)) * 1e6
held = json.load(open(measured))
seconds = held["seconds_per_step_steady_min"]
json.dump({"width": width, "dtype": dtype, "estimate_flop_equivalents": estimate, "seconds_per_frame": seconds,
           "rate": estimate / seconds, "note": note}, open(out, "w"), indent=1)
print(f"{width} {dtype}: estimate {estimate / 1e6:.4g} M, frame {seconds * 1e3:.3g} ms, rate {estimate / seconds / 1e9:.3g} G/s")
