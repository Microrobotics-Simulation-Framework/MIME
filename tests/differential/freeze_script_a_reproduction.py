"""Freeze script A's reproduction arm as phase 4 left it, before phase 5 changes the path its values take:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/freeze_script_a_reproduction.py

writes ``references/script_a_phase4_reproduction.json``: for float64 and float32, the derived hash, the steady
quantities to every digit, and the SHA-256 of every array of a 5 s run (the state at ``t = 0`` and after each
step). Run once, on the commit phase 4 merged (``v0.3`` at db006fb); ``test_script_a_records.py`` holds the arm, now
read through the materials seam, to these bits. Not collected by pytest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "script_a_phase4_reproduction.json"


def array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(f"{array.dtype.str}{array.shape}".encode() + array.tobytes()).hexdigest()


def frozen(x64: bool) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", x64)
    from script_a import reproduction_arm
    from script_a_runs import quantities

    from mime.units import Hz, s

    built = reproduction_arm(frequency=Hz(10.0), dt=s(5e-4)).build()
    body, field = built.graph.get_node_state("umr"), built.graph.get_node_state("field.magnetic")
    history = built.run(s(5.0), history=True).history
    assert history is not None
    arrays = {"umr/pose": np.concatenate([np.asarray(body["pose"])[None], np.asarray(history["umr"]["pose"])]),
              "umr/twist": np.concatenate([np.asarray(body["twist"])[None], np.asarray(history["umr"]["twist"])]),
              "field/flux_density": np.concatenate([
                  np.asarray(field["flux_density_magnetic_torque_umr"])[None],
                  np.asarray(history["field.magnetic"]["flux_density_magnetic_torque_umr"])])}
    from script_a_steady import History

    pose, twist = arrays["umr/pose"], arrays["umr/twist"]
    found = quantities(History(time=np.arange(len(pose)) * 5e-4, position=pose[:, :3], velocity=twist[:, :3],
                               orientation=pose[:, 3:], angular_velocity=twist[:, 3:],
                               field=arrays["field/flux_density"]), 10.0)
    return {"derived_hash": built.record.hashes()["derived"],
            "steady": {name: float(value) for name, value in sorted(found.items())},
            "arrays": {name: array_digest(value) for name, value in arrays.items()},
            "steps": int(len(pose) - 1), "dtype": str(pose.dtype)}


def environment() -> dict[str, str]:
    """What the run's bits depend on beyond the code: the versions, the platform and the processor."""
    import platform

    from mime.saved._record import environment as recorded

    held = recorded()
    cpu = platform.processor()
    try:
        cpu = next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"python": held["python"], "jax": held["jax"], "jaxlib": held["jaxlib"], "platform": held["platform"],
            "cpu": cpu}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("f64", "f32"):
        print(json.dumps(frozen(sys.argv[1] == "f64")))
        return
    if sys.argv[1:] == ["--4c3"]:
        raise SystemExit("references/script_a_4c3_reproduction.json was written by this script, with its output named "
                         "so, on phase 4c3's head 8b30128 before it merged phase 5; rerun it there, not here")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    runs = {label: json.loads(subprocess.run([sys.executable, __file__, label], capture_output=True, text=True,
                                             check=True).stdout) for label in ("f64", "f32")}
    OUTPUT.write_text(json.dumps({
        "written": "2026-10-23, by freeze_script_a_reproduction.py, before phase 5 changed the path script A's values "
                   "take",
        "commit": commit,
        "experiment": "tests/script_a.py's reproduction_arm at 10 Hz, dt 0.5 ms, 5 s, CPU",
        "environment": environment(),
        "runs": runs,
    }, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
