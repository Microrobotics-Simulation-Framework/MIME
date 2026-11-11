"""The two-contact gate's reference (``tests/oracles/test_elliptic_gates.py``, gate 5): an ellipsoid of semi-axes (0.95,
0.65, 0.65) mm lying across a straight tube of radius 1 mm (its long axis along x, the tube's along z), its two ends a
gap ``h1`` and ``h2`` from the wall on either side (``lubrication_cases.across_tube``), by MIME's own MRS at a finer
force spacing and blob with the hand-over off (drag 0.5.0's mrs rung, the tube's wall kind), at gaps it resolves
(above 0.75 of its spacing). A finer model of the same family: the gate it serves bounds the coarse rung's two-contact
composition against it and says nothing about hardware. Not collected; run from the repository's root:

    python tests/differential/two_contact_mrs.py <spacing_mm> <blob> <out.json> <h1_mm,h2_mm> [...]

Each row: the 6 x 6 about the body's centre in SI, the gaps and the least scaled eigenvalue; the file the force
points' and the quadrature's counts."""

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

    spacing, blob, out, pairs = float(sys.argv[1]) * 1e-3, float(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    start = time.perf_counter()
    node, state, leaves, step = stepper(across_tube(spacing, False, blob).build())
    rows = {}
    for pair in pairs:
        h1, h2 = (float(v) * 1e-3 for v in pair.split(","))
        found = step(across(h1, h2), leaves)
        rows[pair] = {"resistance": np.asarray(found["resistance"]).tolist(), "h1": h1, "h2": h2,
                      "least_eigenvalue": float(found["least_eigenvalue"])}
        print(pair, rows[pair]["least_eigenvalue"], flush=True)
    body = node.body({k: leaves[k] for k in node.body_leaf_names()})
    forces, points = int(np.asarray(body.force).shape[0]), int(np.asarray(body.points).shape[0])
    out.write_text(json.dumps({"spacing": spacing, "blob": blob, "force_points": forces, "quadrature_points": points,
                               "rows": rows}, indent=1, sort_keys=True) + "\n")
    print(f"{forces} force points, {points} quadrature points, {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
