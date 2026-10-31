"""Copy what the second declaration's phase gates read from phase 0's A0 runs into ``references/a0/``, beside the arrays
``freeze_a0_reference.py`` and ``freeze_a0_rotor_poses.py`` copied, and record every copied file's sha256:

    .venv/bin/python tests/differential/freeze_a0_phase_arrays.py FOLDER

``FOLDER`` is the folder phase 0 wrote the axis-fixed A0 reference to (``A0_axisfix``, beside the checkouts). For each
run this writes ``<label>_phases_every_10th_step.npz`` at every tenth step, as the run saved them: for the float32 runs
the motor's ``velocity_integral_error`` (the PI loop's integral, which their trajectory files do not hold); for the
mixed runs that and the motor's angle, the field at the body, the body's position and orientation (their rotor poses
are in ``<label>_rotor_every_10th_step.npz``). It then writes ``copied_arrays.json``: the sha256 of each copied file in
the folder, ``freeze_a0_rotor_poses.py``'s two included, and of the run each came from, which
``test_a0_dynamic_b.py`` holds the files to. Run once. Not collected by pytest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUTPUT = HERE / "references" / "a0"
KEPT = {
    "f32_3hz": ("time_s", "state/motor/velocity_integral_error"),
    "f32_10hz": ("time_s", "state/motor/velocity_integral_error"),
    "mixed_f64_3hz": ("time_s", "state/motor/angle", "state/motor/velocity_integral_error",
                      "state/ext_magnet/field_vector", "state/body/position", "state/body/orientation"),
    "mixed_f64_10hz": ("time_s", "state/motor/angle", "state/motor/velocity_integral_error",
                       "state/ext_magnet/field_vector", "state/body/position", "state/body/orientation"),
}
#: The rotor poses ``freeze_a0_rotor_poses.py`` copied from the same runs, hashed here beside the new files.
ROTOR_POSES = ("mixed_f64_3hz", "mixed_f64_10hz")
EVERY = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    record: dict[str, dict[str, object]] = {}
    for label, kept in KEPT.items():
        found = np.load(folder / f"{label}.npz")
        arrays = {key.replace("/", "__"): np.asarray(found[key][::EVERY], dtype=found[key].dtype) for key in kept}
        name = f"{label}_phases_every_10th_step.npz"
        np.savez_compressed(OUTPUT / name, **arrays)
        record[name] = {"from": f"{label}.npz", "from_sha256": sha256(folder / f"{label}.npz"), "kept": list(kept),
                        "every": EVERY, "sha256": sha256(OUTPUT / name)}
        print(label, "done")
    for label in ROTOR_POSES:
        name = f"{label}_rotor_every_10th_step.npz"
        record[name] = {"from": f"{label}.npz", "from_sha256": sha256(folder / f"{label}.npz"),
                        "kept": ["time_s", "state/motor/rotor_pose_world", "state/arm/joint_angles"], "every": EVERY,
                        "sha256": sha256(OUTPUT / name)}
    (OUTPUT / "copied_arrays.json").write_text(json.dumps(record, indent=1) + "\n")
    print("wrote copied_arrays.json")


if __name__ == "__main__":
    main()
