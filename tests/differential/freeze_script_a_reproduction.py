"""Freeze script A's reproduction arm as phase 4 left it, before phase 5 changes the path its values take:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/freeze_script_a_reproduction.py

With ``--7d``, the same freeze of the arm as phase 7d left it, in ``references/script_a_7d_reproduction.json``.

With ``--refreeze "<reason>"``, the digests of the arm as the tree in force runs it are added to that file as a dated
entry of ``refreezes`` with the reason (the 7d freeze's own entries are kept, and the derived hash still has to match
them); ``--cut-sample`` cuts ``references/script_a_7d_sample.npz`` and its record from a run of the arm whose digests
are the 7d freeze's, which only the commits before 22fd495 give (that commit moved the last bits: MIME-ANO-133).

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


def arm_arrays(x64: bool) -> tuple[Any, dict[str, np.ndarray]]:
    """The built arm and the arrays of its 5 s run, on the law the freezes were made with (``rigid_inertial`` 0.2.0)
    where the tree keeps it (``tests/differential/test_script_a_records.py`` builds the arm so)."""
    import jax

    jax.config.update("jax_enable_x64", x64)
    import script_a
    from script_a import reproduction_arm

    from mime.units import Hz, s

    try:
        from mime.dynamics.body import RigidInertialExplicit
    except ImportError:
        pass
    else:
        script_a.RigidInertial = RigidInertialExplicit
    built = reproduction_arm(frequency=Hz(10.0), dt=s(5e-4)).build()
    body, field = built.graph.get_node_state("umr"), built.graph.get_node_state("field.magnetic")
    history = built.run(s(5.0), history=True).history
    assert history is not None
    arrays = {"umr/pose": np.concatenate([np.asarray(body["pose"])[None], np.asarray(history["umr"]["pose"])]),
              "umr/twist": np.concatenate([np.asarray(body["twist"])[None], np.asarray(history["umr"]["twist"])]),
              "field/flux_density": np.concatenate([
                  np.asarray(field["flux_density_magnetic_torque_umr"])[None],
                  np.asarray(history["field.magnetic"]["flux_density_magnetic_torque_umr"])])}
    return built, arrays


def frozen(x64: bool) -> dict[str, Any]:
    from script_a_runs import quantities
    from script_a_steady import History

    built, arrays = arm_arrays(x64)
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


SEVEN_D = HERE / "references" / "script_a_7d_reproduction.json"
SAMPLE = HERE / "references" / "script_a_7d_sample.npz"
SAMPLE_RECORD = HERE / "references" / "script_a_7d_sample.json"
#: Every 100th row of the 10001 (the first and the last among them), per array.
SAMPLE_EVERY = 100
#: The scale-relative tolerances the comparison asserts on every machine: float64's is the bound the test already
#: holds the steady quantities to; float32's is the same size in float32 roundings' terms, about eighty times what
#: 22fd495 moved. Defaults, declared for the maintainer to set.
SAMPLE_TOLERANCE = {"f64": 1e-12, "f32": 1e-4}
#: The move the control plants, a hundred times the tolerance.
SAMPLE_CONTROL = {"f64": 1e-10, "f32": 1e-2}


def run_label(label: str) -> dict[str, Any]:
    return json.loads(subprocess.run([sys.executable, __file__, label], capture_output=True, text=True,
                                     check=True).stdout)


def refreeze(reason: str) -> None:
    """Add the digests of the tree in force to the 7d freeze's file, with the reason, the date and the commit."""
    import datetime

    record = json.loads(SEVEN_D.read_text())
    runs = {label: run_label(label) for label in ("f64", "f32")}
    for label, run in runs.items():
        assert run["derived_hash"] == record["runs"][label]["derived_hash"], f"{label}: the derived graph moved"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    record.setdefault("refreezes", []).append({
        "date": datetime.date.today().isoformat(), "commit": commit, "reason": reason, "environment": environment(),
        "runs": {label: {"derived_hash": run["derived_hash"], "arrays": run["arrays"]} for label, run in runs.items()}})
    SEVEN_D.write_text(json.dumps(record, indent=1) + "\n")
    print(f"refroze {SEVEN_D}")


def cut_sample() -> None:
    """Cut the sample from a run whose digests are the 7d freeze's own: refused on any tree that does not give them."""
    import hashlib

    record = json.loads(SEVEN_D.read_text())
    cut: dict[str, np.ndarray] = {}
    scales: dict[str, dict[str, float]] = {}
    for label in ("f64", "f32"):
        # a fresh process per arm, as the freeze ran them
        file = HERE / f"_{label}_full.npz"
        subprocess.run([sys.executable, __file__, "--dump", label, str(file)], check=True)
        full = np.load(file)
        file.unlink()
        for name in ("umr/pose", "umr/twist", "field/flux_density"):
            array = full[name.replace("/", "__")]
            if array_digest(array) != record["runs"][label]["arrays"][name]:
                raise SystemExit(f"{label} {name}: this tree's run is not the 7d freeze's bits; cut from a commit "
                                 f"before 22fd495")
            assert array.shape[0] == 10001
            cut[f"{label}/{name}"] = array[::SAMPLE_EVERY]
            scales.setdefault(label, {})[name] = float(np.max(np.abs(array[::SAMPLE_EVERY])))
    np.savez_compressed(SAMPLE, **cut)
    SAMPLE_RECORD.write_text(json.dumps({
        "what": ("every 100th row (steps 0, 100, ..., 10000: the first and the last among them) of the three arrays "
                 "of script A's reproduction arm, float64 and float32, as the 7d freeze (script_a_7d_reproduction.json) "
                 "froze them, cut from a run whose sha256 of each full array is that freeze's"),
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        check=True).stdout.strip(),
        "environment": environment(),
        "file": SAMPLE.name, "sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
        "every": SAMPLE_EVERY,
        "full_array_digests": {label: record["runs"][label]["arrays"] for label in ("f64", "f32")},
        "tolerance_of_scale": SAMPLE_TOLERANCE,
        "tolerance_status": ("a default, declared for the maintainer to set: scale is each array's largest absolute "
                             "value in the sample; float64's is the bound the steady quantities are held to, about "
                             "300 times what 22fd495 moved; float32's is about 80 times what 22fd495 moved there"),
        "control_move_of_scale": SAMPLE_CONTROL,
        "scale": scales,
    }, indent=1) + "\n")
    print(f"wrote {SAMPLE} and {SAMPLE_RECORD}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("f64", "f32"):
        print(json.dumps(frozen(sys.argv[1] == "f64")))
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--dump":
        np.savez(sys.argv[3], **{name.replace("/", "__"): value for name, value in
                                 arm_arrays(sys.argv[2] == "f64")[1].items()})
        return
    if sys.argv[1:2] == ["--refreeze"] and len(sys.argv) == 3:
        refreeze(sys.argv[2])
        return
    if sys.argv[1:] == ["--cut-sample"]:
        cut_sample()
        return
    if sys.argv[1:] == ["--4c3"]:
        raise SystemExit("references/script_a_4c3_reproduction.json was written by this script, with its output named "
                         "so, on phase 4c3's head 8b30128 before it merged phase 5; rerun it there, not here")
    output, written = OUTPUT, ("2026-10-23, by freeze_script_a_reproduction.py, before phase 5 changed the path script "
                               "A's values take")
    if sys.argv[1:] == ["--7d"]:
        output = HERE / "references" / "script_a_7d_reproduction.json"
        written = ("2026-10-28, by freeze_script_a_reproduction.py --7d, on phase 7d's head: the arm with its body's "
                   "shape read from its hull's kind (FL-9's de Jongh helix), its extent the kind's radial reach and "
                   "the drag's tube its region's shape")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    runs = {label: run_label(label) for label in ("f64", "f32")}
    output.write_text(json.dumps({
        "written": written,
        "commit": commit,
        "experiment": "tests/script_a.py's reproduction_arm at 10 Hz, dt 0.5 ms, 5 s, CPU",
        "environment": environment(),
        "runs": runs,
    }, indent=1) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
