"""Copy phase 0's axis-fixed A0 reference into ``references/a0/``, with no private path in it:

    .venv/bin/python tests/differential/freeze_a0_reference.py FOLDER

``FOLDER`` is the folder phase 0 wrote the reference to (``A0_axisfix``, beside the checkouts). The reference is the
pre-0.3 tag's A0 (the ``ar4_helical_drive`` experiment, MADDENING 0.3.1, CPU) with one declared patch giving its drag
surrogate the tube's axis, run for 10 s at 3 Hz and 10 Hz, in float32 and in mixed float64 (the state cast to float64,
the controller still in float32). For each of those four runs this writes

- ``<label>_steady.json``: the steady quantities phase 0 computed from every step of the run;
- ``<label>_provenance.json``: how the run was made, with what it names of a private path left out;
- for the float32 runs, ``<label>_trajectory_every_10th_step.npz``: the arrays the emulator and the comparison read, at
  every tenth step, as float32.

Run once. Not collected by pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "a0"
LABELS = ("f32_3hz", "f32_10hz", "mixed_f64_3hz", "mixed_f64_10hz")
#: The arrays kept at every tenth step, under the names the reference saved them by.
KEPT = ("time_s", "state/body/position", "state/body/orientation", "state/body/velocity",
        "state/body/angular_velocity", "state/ext_magnet/field_vector", "state/motor/angle",
        "state/motor/angular_velocity", "state/motor/rotor_pose_world", "state/arm/joint_angles",
        "state/arm/end_effector_pose_world")
EVERY = 10
#: Provenance entries that name where the run was made, and are left out.
PLACE = ("repo", "mime_file", "uv_pip_freeze", "command", "saved_arrays", "state_dtypes")


def node_summary(nodes: dict[str, Any]) -> dict[str, Any]:
    """Each node's class and parameters; its attributes hold paths and weight arrays, and are left out."""
    return {name: {"class": node["class"], "timestep": node["timestep"],
                   "params": {key: value for key, value in node["params"].items() if key != "urdf_path"}}
            for name, node in nodes.items()}


def provenance(source: dict[str, Any]) -> dict[str, Any]:
    kept = {key: value for key, value in source.items() if key not in PLACE}
    patch = dict(source["patch"])
    for key in ("patched_copy", "original", "diff"):
        patch.pop(key, None)
    kept["patch"] = patch
    kept["nodes"] = node_summary(source["nodes"])
    kept["command"] = {"env": {key: value for key, value in source["command"]["env"].items()
                               if key in ("JAX_PLATFORMS", "JAX_ENABLE_X64")},
                       "arguments": [argument for argument in source["command"]["argv"][2:]
                                     if "/" not in argument and not argument.startswith("MIME")]}
    kept["written"] = ("by freeze_a0_reference.py from phase 0's A0_axisfix reference; the node attributes (paths "
                       "and the surrogate's normalisation arrays), the saved array list and the environment's "
                       "package list are left out")
    return kept


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        steady = json.loads((folder / f"{label}_steady.json").read_text())
        steady.pop("npz", None)
        (OUTPUT / f"{label}_steady.json").write_text(json.dumps(steady, indent=1) + "\n")
        source = json.loads((folder / f"{label}_provenance.json").read_text())
        (OUTPUT / f"{label}_provenance.json").write_text(json.dumps(provenance(source), indent=1) + "\n")
        if label.startswith("f32"):
            found = np.load(folder / f"{label}.npz")
            kept = {key.replace("/", "__"): found[key][::EVERY].astype(np.float32 if key != "time_s" else np.float64)
                    for key in KEPT}
            np.savez_compressed(OUTPUT / f"{label}_trajectory_every_10th_step.npz", **kept)
        print(label, "done")


if __name__ == "__main__":
    main()
