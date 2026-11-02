"""What script A's frozen comparison holds on every machine: a sample of the frozen arrays, within a scale-relative
tolerance, beside the digests that hold only where the freeze ran.

**Why two.** A digest is the bits of one compiled program on one environment. Phase 7l-c's unused output moved script
A's last bits (22fd495, MIME-ANO-133) with the derived graph's hash unchanged, and the digest was asserted only where
``this_environment()`` is the freeze's, so CI printed it and nothing failed. The sample, cut from the run whose
digests the 7d freeze holds (``references/script_a_7d_sample.json`` says how, and holds its own sha256), is asserted
everywhere against the run's arrays at ``tolerance_of_scale`` of each array's scale (its largest absolute value in
the sample): a rounding-sized move passes, a physics-sized one does not.

**What it bounds.** The sampled rows (every 100th step, the first and the last): a move on an unsampled row alone is
not seen by it, and the tolerance is a default for the maintainer to set, not a fitted number. The digests, where the
environment is the freeze's, still hold every row to the bit.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from mime.saved._record import environment

HERE = Path(__file__).parent
REFERENCES = HERE / "references"
SEVEN_D = json.loads((REFERENCES / "script_a_7d_reproduction.json").read_text())
RECORD = json.loads((REFERENCES / "script_a_7d_sample.json").read_text())
ARRAYS = ("umr/pose", "umr/twist", "field/flux_density")


def array_digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(f"{array.dtype.str}{array.shape}".encode() + array.tobytes()).hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor()


def this_environment() -> dict[str, str]:
    held = environment()
    return {"python": held["python"], "jax": held["jax"], "jaxlib": held["jaxlib"], "platform": held["platform"],
            "cpu": cpu_model()}


def frozen_digests(label: str) -> tuple[dict[str, str], dict[str, str], str]:
    """The digests in force for one arm, the environment they were taken in, and where they come from: the latest
    refreeze (a dated entry with its reason) where there is one, else the 7d freeze's own."""
    refreezes = SEVEN_D.get("refreezes", [])
    if refreezes:
        latest = refreezes[-1]
        return (latest["runs"][label]["arrays"], latest["environment"],
                f"the refreeze of {latest['date']} (its reason is in the file)")
    return SEVEN_D["runs"][label]["arrays"], SEVEN_D["environment"], "the 7d freeze"


def digests_where_frozen(label: str, found: dict[str, np.ndarray], here: dict[str, str] | None = None) -> bool:
    """Assert the run's digests are the frozen bits where ``here`` (this machine's environment, by default) is the one
    they were taken in, and say which arrays moved. Returns whether it asserted: where it did not, the run's digests
    were only printed by the caller, and the sample is what holds."""
    held, frozen_in, _ = frozen_digests(label)
    if frozen_in != (this_environment() if here is None else here):
        return False
    digests = digests_of(found)
    moved = [name for name in ARRAYS if digests[name] != held[name]]
    assert not moved, (f"{label} {moved} are not the frozen bits in the environment that froze them: a deliberate "
                       f"change is re-frozen with freeze_script_a_reproduction.py --refreeze and its reason "
                       f"(MIME-ANO-133)")
    return True


def arrays_of(history) -> dict[str, np.ndarray]:
    """The three arrays the freezes digest, from a run's history (the state at ``t = 0`` and after each step)."""
    return {"umr/pose": np.concatenate([history.position, history.orientation], axis=1),
            "umr/twist": np.concatenate([history.velocity, history.angular_velocity], axis=1),
            "field/flux_density": np.asarray(history.field)}


def digests_of(found: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: array_digest(value) for name, value in found.items()}


def frozen_sample(label: str) -> dict[str, np.ndarray]:
    with np.load(REFERENCES / RECORD["file"]) as held:
        return {name: held[f"{label}/{name}"] for name in ARRAYS}


def scale_of(label: str, name: str) -> float:
    return float(np.max(np.abs(frozen_sample(label)[name])))


def sampled(found: np.ndarray) -> np.ndarray:
    return found[:: RECORD["every"]]


def moves(found: dict[str, np.ndarray], label: str) -> dict[str, float]:
    """Per array, the largest absolute difference on the sampled rows from the frozen sample, as a fraction of the
    frozen array's scale."""
    held = frozen_sample(label)
    return {name: float(np.max(np.abs(sampled(found[name]).astype(np.float64) - held[name].astype(np.float64))))
            / scale_of(label, name) for name in ARRAYS}


def report(record_property, label: str, misses: dict[str, float]) -> None:
    """Put the largest ratios and the processor into the JUnit report, where CI's passing tests keep no output."""
    record_property("cpu", cpu_model())
    for name, miss in misses.items():
        record_property(f"{label} {name}: largest move / scale", f"{miss:.3e}")
    record_property(f"{label} tolerance of scale", f"{tolerance(label):.0e}")


def tolerance(label: str) -> float:
    return RECORD["tolerance_of_scale"][label]


def planted(found: dict[str, np.ndarray], label: str, name: str) -> dict[str, np.ndarray]:
    """``found`` with one element of ``name`` on a sampled row moved by the control's fraction of that array's scale."""
    moved = {key: np.array(value) for key, value in found.items()}
    moved[name][RECORD["every"] * 50, 0] += RECORD["control_move_of_scale"][label] * scale_of(label, name)
    return moved
