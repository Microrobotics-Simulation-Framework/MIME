"""Writes ``references/lubrication_mfs.json``: the unit sphere's 6 x 6 resistance above a plane wall at the gaps
the near-contact hand-over gate judges against it (``tests/oracles/test_lubrication.py``), by the method of fundamental
solutions (``plane_oracles.mfs``, sharing no code with MIME), at its coarse and fine settings, with Brenner's series
beside the normal entry. Below a gap of about 0.05 radii the method's rings grow past a few GiB, so the gate judges the
normal entry alone there, against Brenner. Not collected; run once, in float64, where a few GiB are free:

    python tests/differential/generate_lubrication_mfs.py"""

import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
from plane_oracles import MFS_SETTINGS, brenner, mfs  # noqa: E402

#: The gaps, in the sphere's radius.
GAPS = (0.3, 0.2, 0.15, 0.1, 0.075, 0.06)


def main() -> None:
    rows = {}
    for setting, values in MFS_SETTINGS.items():
        for gap in GAPS:
            started = time.time()
            resistance, residual, sources = mfs(gap, **values)
            rows[f"{setting} {gap}"] = {
                "resistance": resistance.tolist(), "residual": residual, "sources": sources,
                "seconds": round(time.time() - started, 1), "brenner": brenner(gap),
                "normal_over_brenner": float(resistance[2, 2] / (6 * math.pi) / brenner(gap)),
                "asymmetry": float(np.abs(resistance - resistance.T).max() / np.abs(resistance).max())}
            print(setting, gap, {k: v for k, v in rows[f"{setting} {gap}"].items() if k != "resistance"}, flush=True)
    record = {"gaps": list(GAPS), "settings": MFS_SETTINGS, "units": "unit radius, unit viscosity; force and torque "
              "on the fluid per unit motion about the centre", "numpy": np.__version__,
              "python": platform.python_version(), "rows": rows}
    (HERE / "references" / "lubrication_mfs.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
