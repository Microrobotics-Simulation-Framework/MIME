"""Item 1's pairs, summarised: per bin of the nearer sample's height over the wall, the table's error E against the
direct series, the plane's departure from the tube (``|exact - blake0|``) and the image's regularisation
(``|blake_eps - blake0|``), each over the exact block's largest entry; the self pairs on their own. No solve.

    python bem_budget_wall_summary.py --pairs PAIRS.npz --out OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS_MM = (0.0, 0.12, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8)
BLOBS = ("0.25", "0.35", "0.5", "0.7", "1.0")


def relative(miss: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.max(np.abs(miss), axis=(-2, -1)) / np.max(np.abs(reference), axis=(-2, -1))


def stats(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {"count": int(values.size), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    held = np.load(args.pairs)
    heights = held["heights"] * 1e3
    exact, table, blake0 = held["exact"], held["table"], held["blake0"]
    n = len(heights)
    lower = np.minimum(heights[:, None], heights[None, :])
    local = held["local"]
    separation = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1) * 3.175
    e_table = relative(table - exact, exact)
    e_plane = relative(exact - blake0, exact)
    e_reg = {b: relative(held[f"blake_{b}"] - blake0, exact) for b in BLOBS}
    out = {"near": n, "bins_mm": list(BINS_MM), "all_pairs": {}, "self_pairs": {}, "close_pairs": {},
           "whole": {"E_table": stats(e_table.ravel()), "plane_departure": stats(e_plane.ravel())}}
    diagonal = np.arange(n)
    close = (separation < 0.3) & ~np.eye(n, dtype=bool)
    for lo, hi in zip(BINS_MM[:-1], BINS_MM[1:], strict=True):
        key = f"{lo:.2f}-{hi:.2f}"
        chosen = (lower >= lo) & (lower < hi)
        out["all_pairs"][key] = {"E_table": stats(e_table[chosen]), "plane_departure": stats(e_plane[chosen]),
                                 **{f"regularisation_{b}": stats(e_reg[b][chosen]) for b in BLOBS}}
        own = (heights >= lo) & (heights < hi)
        out["self_pairs"][key] = {"E_table": stats(e_table[diagonal[own], diagonal[own]]),
                                  "plane_departure": stats(e_plane[diagonal[own], diagonal[own]]),
                                  **{f"regularisation_{b}": stats(e_reg[b][diagonal[own], diagonal[own]])
                                     for b in BLOBS}}
        both = chosen & close
        out["close_pairs"][key] = {"E_table": stats(e_table[both]), "plane_departure": stats(e_plane[both]),
                                   **{f"regularisation_{b}": stats(e_reg[b][both]) for b in BLOBS}}
    out["lowest_sample_height_mm"] = float(heights.min())
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
