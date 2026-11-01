"""Phase 7j-a's preregistration: every input of the phase's computations with its hash, written before any of them
runs (the sim-to-real plan's section 13): the commit the scripts run on, the scripts themselves, the budget's declared
ranges, the tube's table and the surrogate's weights, the samples (meshes) of every BEM discretisation the phase uses,
and, for the Modal runs, the wheel and the hashed requirements. Building the experiments to hash their samples solves
nothing.

    JAX_ENABLE_X64=1 python scripts/sim2real/prereg.py --out scripts/sim2real/preregistration_7ja.json \
        [--wheel WHEEL --requirements REQUIREMENTS]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import operating_point as op  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRIPTS = ("operating_point.py", "bem.py", "surrogate.py", "modal_app.py", "assemble.py", "budget_inputs.json",
           "prereg.py", "predictions_7ja.json")


def mesh_hashes() -> dict[str, Any]:
    """The samples of each discretisation in the body's frame (they move with the pose, not with it), hashed."""
    import bem

    out: dict[str, Any] = {}
    for patches in op.PATCHES:
        for spacing in (op.SPACING_MM, bem.FINER_SPACING_MM):
            built = op.experiment("bem", patches=patches, spacing_mm=spacing).build()
            node = built.graph.get_node(op.DRAG_NODE)
            op.require_quadrature(node)
            held = node.samples(op.leaves_of(built)[op.DRAG_NODE])
            points = np.ascontiguousarray(np.asarray(held.points, dtype=np.float64))
            out[f"{patches}/{spacing:.3f}mm"] = {
                "sha256_points": hashlib.sha256(points.tobytes()).hexdigest(), "samples": int(points.shape[0]),
                "unknowns": 3 * int(points.shape[0]), "counts": [list(c) for c in node.constants()["counts"]],
                "area_m2": float(np.sum(np.asarray(held.weights)))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--wheel", default=None, help="the MIME wheel the Modal image installs")
    parser.add_argument("--requirements", default=None, help="the hashed requirements the Modal image installs")
    args = parser.parse_args()
    op.require_float64()

    from mime import assets
    from mime.data import asset

    table = assets.reference("unit_tube_R0.997")
    weights = asset(op.DEFAULT_WEIGHTS)
    payload: dict[str, Any] = {
        "written": "before any of phase 7j-a's computations (dry runs of the code paths at a coarse mesh and a margin "
                   "outside the plan excepted, none of whose speeds was read)",
        "environment": op.environment(),
        "scripts_sha256": {name: op.sha256_file(HERE / name) for name in SCRIPTS if (HERE / name).exists()},
        "tests_read_sha256": {name: op.sha256_file(op.ROOT / name) for name in
                              ("tests/script_a.py", "tests/differential/script_a_steady.py")},
        "table": {"name": "unit_tube_R0.997", "sha256_declared": table.sha256,
                  "sha256_file": op.sha256_file(assets.locate(table))},
        "weights": {"name": weights.name, "sha256": weights.sha256},
        "meshes": mesh_hashes(),
        "operating_point": {"frequency_hz": op.FREQUENCY_HZ, "margin_mm": op.MARGIN_MM, "dt_ms": op.DT_MS,
                            "duration_s": op.DURATION_S, "spacing_mm": op.SPACING_MM, "blob": op.BLOB,
                            "phases_rad": list(op.PHASES), "variants": list(op.VARIANTS),
                            "primary_variant": "axial", "tube_radius_m": op.TUBE_RADIUS_M},
    }
    if args.wheel:
        payload["modal"] = {"wheel": Path(args.wheel).name, "wheel_sha256": op.sha256_file(args.wheel),
                            "requirements_sha256": op.sha256_file(args.requirements) if args.requirements else None,
                            "device": "A100-40GB (static, dynamics), A100-80GB (budget); the device served is "
                                      "recorded with each result"}
    op.write_json(args.out, payload)


if __name__ == "__main__":
    main()
