"""The corner gate's truth re-measured at four times its samples on the worst clean draw: from the record the gate's
runs append (``CORNER_RECORD``), each case's clean draw of largest error, its two-branch pixels' truth at 128 x 128
jittered samples a pixel and 2048 signs (four times each), against the gate's (64 x 64, 512).
``python scripts/camera_route2/corner_recheck.py RECORD.jsonl OUT.json``, from the repository's root."""

from __future__ import annotations

import json
import sys

import jax

sys.path[:0] = ["tests/sensing", "tests"]
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import test_sampled as T  # noqa: E402


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    worst: dict = {}
    for row in rows:
        if row["plant"]:
            continue
        key = (row["coil"], row["width"], row["dtype"])
        if key not in worst or row["largest"] > worst[key]["largest"]:
            worst[key] = row
    out = []
    monkeypatch = pytest.MonkeyPatch()
    for (coil, width, dtype), row in sorted(worst.items()):
        jax.config.update("jax_enable_x64", dtype == "float64")
        monkeypatch.setattr(T, "planted", lambda name: False)
        base, count = T.corner_errors(row["seed"], monkeypatch, width, coil)
        fine, count_fine = T.corner_errors(row["seed"], monkeypatch, width, coil, side=2 * T.TRUTH_SIDE, signs=2048)
        monkeypatch.undo()
        entry = {"coil": coil, "width": width, "dtype": dtype, "seed": row["seed"], "pixels": count,
                 "largest_at_the_gate_s_truth": float(base.max()), "largest_at_four_times": float(fine.max()),
                 "truth_moved_by_at_most": float(np.max(np.abs(base - fine))), "line": row["line"],
                 "truth_floor": T.TRUTH_FLOOR}
        assert count == count_fine
        out.append(entry)
        print(json.dumps(entry), flush=True)
    with open(sys.argv[2], "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
