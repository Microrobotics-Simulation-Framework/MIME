"""Copy the rotor's recorded pose from phase 0's mixed-precision A0 runs into ``references/a0/``, beside the float32
runs' arrays ``freeze_a0_reference.py`` copied:

    .venv/bin/python tests/differential/freeze_a0_rotor_poses.py FOLDER

``FOLDER`` is the folder phase 0 wrote the axis-fixed A0 reference to (``A0_axisfix``, beside the checkouts). For each
mixed run (``mixed_f64_3hz``, ``mixed_f64_10hz``) this writes ``<label>_rotor_every_10th_step.npz``: the times, the
motor's recorded ``rotor_pose_world`` and the arm's joint angles, at every tenth step, as the run saved them (float64).
The pose comparison of phase 7c2 reads them for the reference's own float32-to-mixed gap in the magnet's pose, its
rounding floor. Run once. Not collected by pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUTPUT = HERE / "references" / "a0"
LABELS = ("mixed_f64_3hz", "mixed_f64_10hz")
KEPT = ("time_s", "state/motor/rotor_pose_world", "state/arm/joint_angles")
EVERY = 10


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    for label in LABELS:
        found = np.load(folder / f"{label}.npz")
        kept = {key.replace("/", "__"): np.asarray(found[key][::EVERY], dtype=np.float64) for key in KEPT}
        np.savez_compressed(OUTPUT / f"{label}_rotor_every_10th_step.npz", **kept)
        print(label, "done")


if __name__ == "__main__":
    main()
