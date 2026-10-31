"""Measure the finding of A0's comparison on its dynamic rungs that is not on its list, and the control on its cause:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/measure_a0_dynamic_findings.py

**The finding.** At 10 Hz the body's and the field's rotation rates by phase, and the magnet's phase through its mount,
miss their declared tolerances, and the motor's angle rate sits at its own. Each reads the rotor's angle, which 0.3's
dynamic motor and the emulation both accumulate in single precision (``angle + dt (Omega - e . omega)``): the two differ
by units in the last place from the first step of the spin-up, where 0.3 computes the flange's turning in single
precision throughout and the emulation computes it in double and rounds it once. A large angle's unit absorbs most of
that, but not all: where the angles reach the window a unit apart, the window crosses a power of two a step apart, holds
one step more of one increment's quantum than the other, and the field's phase at the window's ends moves by a unit.

**Where 0.3 lands.** 0.3's float32 run is the same to the bit run to run, but XLA's CPU configuration changes it at
this level: with ``--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`` its angle leaves the default
run's from the third step and reaches 4.95 s one unit further from the emulation's (measured while this was written).
So the test measures how many units apart the two angles are at the window's start and end, against the emulation's
angles recorded here, and holds each quantity the finding moves to its tolerance and one unit's move for each unit
apart. Run this with the default flags, as CI runs.

**The control.** The emulation is rerun with its angle set, one step before the window opens (4.95 s), to the angle
0.3's run held there, and to the emulation's own (the control's control); and to its own moved by one unit either way,
which is how far one unit of the angle moves each quantity (the finding's bound in ``test_a0_dynamic.py``: the declared
tolerance and one unit's move). For each frequency it records the angles' difference in units at the window's start
and end, 0.3's quantities, and each emulated run's. Writes ``references/a0_dynamic_findings.json``. Not collected by
pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "a0_dynamic_findings.json"
#: The step one before the steady window opens (it opens at 5 s, step 10000, at both frequencies).
BEFORE = 9900
#: The quantities the rotor's angle moves.
MOVED = ("body_rotation_rate_hz", "field_rotation_rate_hz", "rotor_angle_rate_mean_hz", "magnet_phase_rad")


def units(value: float) -> float:
    """One unit in the last place of a single-precision ``value``."""
    return float(np.spacing(np.float32(abs(value))))


def phase_miss(poses: np.ndarray, key: str) -> float:
    from a0_dynamic_compare import FREQUENCY, reference_poses
    from a0_poses import difference, misses

    time, reference = reference_poses(key)
    emulated = np.load(HERE / "a0_dynamic_poses.npz")
    predicted = difference(emulated[f"{key}_new"], emulated[f"{key}_tag"])
    return misses(poses, reference, predicted, time, FREQUENCY[key])["magnet_phase_rad"]


def main() -> None:
    import jax

    from a0_compare import difference
    from a0_dynamic_compare import FREQUENCY, measure
    from a0_runs import window

    record: dict[str, Any] = {}
    for key, frequency in FREQUENCY.items():
        jax.config.update("jax_enable_x64", False)
        measured, run = measure(key)
        measured["magnet_phase_rad"] = phase_miss(run.rotor_poses[::10], key)
        angle = np.asarray(run.history.rotor_angle, dtype=np.float64)
        jax.config.update("jax_enable_x64", True)
        import predict_a0_dynamic as D

        own, _, own_poses = D.emulate(D.NEW_D, frequency)
        emulated_angle = np.asarray(own.rotor_angle)
        inside = window(np.asarray(own.body.time), frequency)
        first, last = int(np.argmax(inside)), len(inside) - 1
        runs: dict[str, Any] = {}
        unit = units(emulated_angle[BEFORE])
        for label, value in (("0.3's angle", float(angle[BEFORE])), ("the emulation's own", float(emulated_angle[BEFORE])),
                             ("one unit more", float(emulated_angle[BEFORE]) + unit),
                             ("one unit less", float(emulated_angle[BEFORE]) - unit)):
            history, _, poses = D.emulate(D.NEW_D, frequency, angle_at=(BEFORE, value))
            found = D.steady(history, frequency, D.NEW_D)
            found["magnet_phase_rad"] = phase_miss(poses, key)
            runs[label] = {"angle_at_4.95_s": value,
                           "quantities": {name: found[name] for name in MOVED},
                           "0.3_against_it": {name: (measured[name] - found[name] if name == "magnet_phase_rad"
                                                     else difference(name, measured[name], found[name]))
                                              for name in MOVED}}
        one_unit = {name: max(abs(difference(name, runs[f"one unit {way}"]["quantities"][name],
                                             runs["the emulation's own"]["quantities"][name]))
                              if name != "magnet_phase_rad" else
                              abs(runs[f"one unit {way}"]["quantities"][name]
                                  - runs["the emulation's own"]["quantities"][name])
                              for way in ("more", "less")) for name in MOVED}
        record[key] = {
            "emulated_angle": {"window_start": {"step": first, "angle": float(emulated_angle[first])},
                               "window_end": {"step": last, "angle": float(emulated_angle[last])}},
            "angles_apart_in_units": {"at_4.95_s": float((angle[BEFORE] - emulated_angle[BEFORE]) / unit),
                                      "window_start": float((angle[first] - emulated_angle[first])
                                                            / units(emulated_angle[first])),
                                      "window_end": float((angle[last] - emulated_angle[last])
                                                          / units(emulated_angle[last]))},
            "first_step_apart": int(np.argmax(angle != emulated_angle)),
            "measured": {name: measured[name] for name in MOVED}, "runs": runs, "one_unit_moves": one_unit}
        print(key, json.dumps(record[key]["angles_apart_in_units"]), json.dumps(one_unit), flush=True)
    OUTPUT.write_text(json.dumps({
        "finding": "the rotor's single-precision angle on 0.3's dynamic motor is reproduced by the emulation to about "
                   "one unit in its last place, not bit for bit: where the two reach the steady window a unit "
                   "apart, the rotation rates by phase, the motor's angle rate and the magnet's phase move by that "
                   "unit's effect, which at 10 Hz is beyond the declared tolerances",
        "not_on_the_declaration": "the declaration's tolerances for these quantities took 7c1's rule, whose emulation "
                                  "resolution came from the tag's scheme, where the emulation reproduces the tag's "
                                  "single-precision angle bit for bit; on the dynamic motor it does not",
        "control": "the emulation with its angle set to 0.3's at 4.95 s, against the same with its own: see runs",
        "by_frequency": record}, indent=1) + "\n")
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
