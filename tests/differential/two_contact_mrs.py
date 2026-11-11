"""The two-contact gate's reference (``tests/oracles/test_elliptic_gates.py``, gate 5): an ellipsoid of semi-axes (0.96,
0.65, 0.65) mm lying across a straight tube of radius 1 mm (its long axis along x, the tube's along z), its two ends a
gap ``h1`` and ``h2`` from the wall on either side (``lubrication_cases.across_tube``), by MIME's own MRS at a finer
force spacing with the hand-over off (drag 0.5.0's mrs rung, the tube's wall kind, blob 0.05), at gaps it resolves
(above 0.75 of its spacing). A finer model of the same family: the gate it serves bounds the coarse rung's two-contact
composition against it and says nothing about hardware. Not collected; run from the repository's root:

    python tests/differential/two_contact_mrs.py <spacing_mm> <out.json> <h1_mm,h2_mm> [...]

Each row: the 6 x 6 about the body's centre in SI, the gaps, the force points' count and the least scaled eigenvalue."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, where lubrication_cases lives


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    from lubrication_cases import across, across_tube, stepper

    spacing, out, pairs = float(sys.argv[1]) * 1e-3, Path(sys.argv[2]), sys.argv[3:]
    start = time.perf_counter()
    node, state, leaves, step = stepper(across_tube(spacing, False).build())
    rows = {}
    for pair in pairs:
        h1, h2 = (float(v) * 1e-3 for v in pair.split(","))
        found = step(across(h1, h2), leaves)
        rows[pair] = {"resistance": np.asarray(found["resistance"]).tolist(), "h1": h1, "h2": h2,
                      "least_eigenvalue": float(found["least_eigenvalue"])}
        print(pair, rows[pair]["least_eigenvalue"], flush=True)
    points = int(np.asarray(node.body({k: leaves[k] for k in node.body_leaf_names()}).points).shape[0])
    out.write_text(json.dumps({"spacing": spacing, "force_points": points, "rows": rows}, indent=1, sort_keys=True)
                   + "\n")
    print(f"{points} force points, {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
