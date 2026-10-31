"""Post hoc mechanism test (chosen after route (b)'s records were read): the node's own method (point forces at the
samples, ``eps = blob x mean spacing``) on A's body in free space at about the node's sample count, the lateral patch
resampled with its axial count raised and its round lowered by ``sqrt(ratio)`` each, so that the flanks' axial spacing
is about the round's; the caps unchanged. The node's own samples at the same spacing are the control (the frozen
point's 2.7979 mm/s). Predictions: ``predictions_helix_ref_iso.json``, committed before it ran.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python helix_ref_iso.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import helix_ref as hr  # noqa: E402
import operating_point as op  # noqa: E402

RATIO = 2.3
SPACINGS_MM = (0.2, 0.16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    op.require_float64()
    out: dict = {"ratio": RATIO, "levels": {}}
    for spacing in SPACINGS_MM:
        body = hr.Body("helix", spacing)
        (around, along), caps = body.node.constants()["counts"]
        iso = [(max(9, round(around / math.sqrt(RATIO))), round(along * math.sqrt(RATIO))), tuple(caps)]
        for arm, counts in (("node_samples", None), ("isotropic", iso)):
            if counts is None:
                points, weights = body.points, body.weights
            else:
                points, weights = body.resampled(counts)
            mean_h = math.sqrt(weights.sum() / len(weights))
            row = {"counts": counts or [list(c) for c in body.node.constants()["counts"]], "samples": len(weights),
                   "mean_spacing_m": mean_h, "area_m2": float(weights.sum()),
                   "close_pairs": hr.close_pairs(points, mean_h), "blobs": {}}
            for blob in hr.BLOBS:
                start = time.perf_counter()
                r, method = hr.solve_in_place(points, (blob * mean_h) ** 2, body.viscosity, body.centre)
                row["blobs"][f"{blob}"] = hr.reading(r, "helix", method=method, seconds=time.perf_counter() - start)
            out["levels"][f"{spacing}"] = out["levels"].get(f"{spacing}", {})
            out["levels"][f"{spacing}"][arm] = row
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "iso.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
            print(spacing, arm, len(weights), {b: round(v["axial_mm_s"], 4) for b, v in row["blobs"].items()},
                  flush=True)


if __name__ == "__main__":
    main()
