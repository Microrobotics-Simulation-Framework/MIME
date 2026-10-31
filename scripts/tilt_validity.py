"""The surrogate drag's tilt validity, measured against the boundary-element rung: how far the swim speed of FL-9 on
the axis of the walk's tube moves when the body is tilted, which the surrogate cannot see.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/tilt_validity.py <out.json> [spacing_mm]

The surrogate's resistance is built in the tube's frame from the body's centre and which way its axis points along the
tube, so it gives a tilted body the resistance of one along the axis. The boundary-element rung (the tube wall kind)
solves for the body as it lies. For FL-9 on the axis, at a tilt of 0, 2, 5, 10 and 15 degrees about each of two axes
across the tube (the helix's section is not round, so the direction of the tilt matters), this takes the rung's 6 x 6
resistance in float64 and the force-free speed along the tube per turn about it, ``-R_FU^-1 R_FOmega (2 pi, 0, 0)``,
and records each against the aligned body's. The tilt stops at 15 degrees: past about 19 the body no longer fits the
bore, and the wall table is not defined outside it. One spacing (0.5 mm by default, which is not converged for this
helix: ``docs/choosing_spacing.md``), one offset (on the axis), one body: it compares a tilted body with an aligned
one at the same spacing, which cancels none of the discretisation error and is not a bound on it. The output is
``docs/data/tilt_validity.json``, which ``tests/oracles/test_tilt_flag.py`` holds the declared limit
(``mime.nodes.hydro.TILT_VALIDITY``) to. Not collected by pytest."""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TILTS_DEGREES = (0.0, 2.0, 5.0, 10.0, 15.0)
#: The axes across the tube (x) the body is tilted about: the world's y and z.
AXES = ("y", "z")


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(spacing_m: float) -> dict[str, Any]:
    tables = load("default_spacing_table_for_the_tilt", ROOT / "scripts" / "default_spacing_table.py")
    rows: list[dict[str, Any]] = []
    for axis in AXES:
        for degrees in TILTS_DEGREES:
            half = math.radians(degrees) / 2.0
            quaternion = (math.cos(half), 0.0, math.sin(half) if axis == "y" else 0.0,
                          math.sin(half) if axis == "z" else 0.0)
            tables.at_rest = lambda position, q=quaternion: tables.State(
                position=position, orientation=q, velocity=tables.m_per_s((0.0, 0.0, 0.0)),
                angular_velocity=tables.rad_per_s((0.0, 0.0, 0.0)))
            r, samples = tables.resistance(tables.screw_task(spacing_m, tube=True).build(), "umr")
            speed = float(-np.linalg.solve(r[:3, :3], r[:3, 3:] @ np.array([2.0 * math.pi, 0.0, 0.0]))[0])
            rows.append({"axis": axis, "tilt_degrees": degrees, "samples": samples, "speed_per_turn_m": speed})
            print(rows[-1], flush=True)
    aligned = next(row["speed_per_turn_m"] for row in rows if row["tilt_degrees"] == 0.0)
    for row in rows:
        row["change_vs_aligned"] = row["speed_per_turn_m"] / aligned - 1.0
    return {"spacing_m": spacing_m, "x64": True, "python": platform.python_version(), "rows": rows,
            "note": "FL-9 on the axis of the 1/4 inch tube, boundary-element rung with the tube wall kind, float64"}


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print(__doc__, file=sys.stderr)
        return 2
    spacing = float(argv[1]) * 1e-3 if len(argv) == 2 else 0.5e-3
    Path(argv[0]).write_text(json.dumps(measure(spacing), indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
