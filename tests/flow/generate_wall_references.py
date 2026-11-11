"""The references FA5 and FA6 read (``data/general_wall_references.json``): the stenosis's flux and the loop's split
(daughters 0.4 and 0.3 D, filleted at 0.1), each through the node at 32 and 64 cells a diameter, steady Stokes, float64
at 1e-10, at four of ``wall_cases.draw``'s turns and offsets, and extrapolated to zero spacing at first order from the
two resolutions' means over the draws (the band's error is a shift constant in cells, so first order in ``h``).

    python tests/flow/generate_wall_references.py run <stenosis|loop> <dh> <seed>    # one JSON line
    python tests/flow/generate_wall_references.py assemble <rows.jsonl>...            # writes the data file

The runs are heavy (the loop at 64 cells a diameter holds 2.8 million cells): they ran on GitHub's runners."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from fvm_cases import run  # noqa: E402
from wall_cases import REFERENCES, box_flux, draw, loop, mid_plane_split, stenosis, tree_node  # noqa: E402

#: The draws the references average over.
SEEDS = (11, 12, 13, 14)


def measured(geometry: str, dh: int, seed: int) -> dict[str, float]:
    angle, offset = draw(seed)
    tree = {"stenosis": stenosis(), "loop": loop()}[geometry]
    began = time.time()
    placed = tree_node(tree, dh, angle=angle if geometry == "loop" else 0.0, offset=offset, tolerance=1e-10)
    state = run(placed.node, 1)
    row = {"geometry": geometry, "dh": dh, "seed": seed, "spacing": placed.layout.spacing,
           "flux": box_flux(state), "iterations": float(state["solve_report"][0]),
           "residual": float(state["solve_report"][1]), "inexact": float(state["wall_inexact"]),
           "seconds": time.time() - began}
    if geometry == "loop":
        first, second = mid_plane_split(placed, state)
        row["split"] = first / (first + second)
    return row


def assemble(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    found: dict[str, dict[str, float]] = {}
    for geometry, quantity in (("stenosis", "flux"), ("loop", "split")):
        means = {}
        for dh in (32, 64):
            held = [row for row in rows if row["geometry"] == geometry and row["dh"] == dh]
            assert len(held) == len(SEEDS) and all(row["inexact"] == 0.0 for row in held), (geometry, dh)
            means[dh] = (sum(row[quantity] for row in held) / len(held), held[0]["spacing"])
        (coarse, h_coarse), (fine, h_fine) = means[32], means[64]
        found[geometry] = {quantity: (fine * h_coarse - coarse * h_fine) / (h_coarse - h_fine),
                           f"{quantity}_32": coarse, f"{quantity}_64": fine}
    return found


if __name__ == "__main__":
    if sys.argv[1] == "run":
        print(json.dumps(measured(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))), flush=True)
    else:
        rows = [json.loads(line) for name in sys.argv[2:] for line in Path(name).read_text().splitlines() if line]
        REFERENCES.parent.mkdir(exist_ok=True)
        REFERENCES.write_text(json.dumps(assemble(rows), indent=2) + "\n", encoding="utf-8")
