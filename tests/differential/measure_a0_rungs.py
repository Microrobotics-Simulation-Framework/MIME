"""Measure A0 on its kinematic rungs against A0 on its dynamic rungs, the data the kinematic rung's validity line will
cite (design #99), and A0 with its magnet's mass, reported as data:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/measure_a0_rungs.py

At 1, 2, 3, 5, 7 and 10 Hz, 10 s each in float32, it runs 0.3's A0 on its kinematic rungs (7c1's, ``script_a0``) and
on its dynamic rungs (``script_a0_dynamic``, as declared), and records per gated quantity the difference between the
rungs, against the commanded joint speeds and accelerations (the kinematic rung's joint positions, which are the
controller's commands, differenced per step: their largest over the steady window and over the whole run), beside the
description's velocity limits; the dynamic rung's own joint motion (its largest speeds, joint 6's kick); and where each
quantity's difference, on a line through zero against the window's largest commanded joint speed, crosses its
threshold, the comparison's tolerance (``a0_dynamic_expected_differences.json``, ``kinematic_against_dynamic``). Then
A0 with its magnet's mass at 3 and 10 Hz, spun about a diameter (its declared axis) and about its own axis (the
symmetric case), never compared with the 0.2 reference. Writes ``references/a0_rungs_measured.json``. Not collected by
pytest."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "a0_rungs_measured.json"
FREQUENCIES = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
STEP = 5e-4
DURATION = 10.0
#: The description's velocity limits (rad/s) of joints 1 to 6; it gives no acceleration limit.
VELOCITY_LIMITS = (1.0472, 1.0472, 1.0472, 1.0472, 1.0472, 1.5708)
#: The commanded joint speed the crossing is fitted against.
CROSSING_SPEED = "speed_over_20_ms_max_window_rad_s"


#: The span the commanded motion is resolved over, in steps (20 ms): per step, the single-precision commands' rounding
#: (a unit in the last place of a joint near 1.5 rad over a step is 2.4e-4 rad/s, and their noise is some tens of
#: units) is as large as the tracking's own joint speeds, and its second difference larger by far than any acceleration.
SPAN = 40


def joint_motion(joints: np.ndarray, time: np.ndarray, frequency: float) -> dict[str, Any]:
    """The joints' largest rate and second difference per step, as declared, over the steady window and over the whole
    run; and their largest rate and its change over ``SPAN`` steps, which resolve the commanded motion."""
    from a0_runs import window

    def largest(values: np.ndarray, inside: np.ndarray | None = None) -> list[float]:
        return [float(v) for v in np.max(np.abs(values if inside is None else values[inside]), axis=0)]

    speed = np.diff(joints, axis=0) / STEP
    acceleration = np.diff(joints, n=2, axis=0) / STEP ** 2
    spanned = (joints[SPAN:] - joints[:-SPAN]) / (SPAN * STEP)
    changed = (spanned[SPAN:] - spanned[:-SPAN]) / (SPAN * STEP)
    inside = window(time, frequency)
    return {"speed_max_window_rad_s": largest(speed, inside[1:]), "speed_max_run_rad_s": largest(speed),
            "acceleration_max_window_rad_s2": largest(acceleration, inside[2:]),
            "acceleration_max_run_rad_s2": largest(acceleration),
            "speed_over_20_ms_max_window_rad_s": largest(spanned, inside[SPAN:]),
            "speed_over_20_ms_max_run_rad_s": largest(spanned),
            "acceleration_over_20_ms_max_window_rad_s2": largest(changed, inside[2 * SPAN:]),
            "acceleration_over_20_ms_max_run_rad_s2": largest(changed)}


def kinematic(frequency: float) -> tuple[dict[str, float], dict[str, Any]]:
    """7c1's A0 on its kinematic rungs: its steady quantities, read as ``a0_runs.run_built`` reads them, and its
    commanded joint motion (the kinematic arm's joint positions are the controller's commands)."""
    from a0_runs import SCRIPT_A_FRAME, A0History, quantities
    from script_a0 import a0
    from script_a_steady import History

    from mime.units import Hz, ms, s

    built = a0(frequency=Hz(frequency), dt=ms(STEP * 1e3)).build()
    now = {node: built.graph.get_node_state(node) for node in ("umr", "rpm.magnetic", "rpm.motor", "rpm.arm")}
    history = built.run(s(DURATION), history=True).history
    assert history is not None

    def stacked(node: str, field: str) -> np.ndarray:
        return np.concatenate([np.asarray(now[node][field])[None], np.asarray(history[node][field])])

    steps = round(DURATION / STEP)
    pose, twist = stacked("umr", "pose"), stacked("umr", "twist")
    field = stacked("rpm.magnetic", "flux_density_magnetic_torque_umr")
    rotor = stacked("rpm.motor", "frame_poses")[:, 1, :3]
    rate = np.concatenate([[0.0], np.asarray(history["drive:rpm.motor.velocity"]["value"])])
    body = History(time=np.arange(steps + 1) * STEP, position=pose[:, :3], velocity=twist[:, :3],
                   orientation=pose[:, 3:], angular_velocity=twist[:, 3:], field=field)
    found = A0History(body, rotor, rate, stacked("rpm.motor", "angle"))
    joints = stacked("rpm.arm", "joint_positions").astype(np.float64)
    return quantities(found, frequency, SCRIPT_A_FRAME), joint_motion(joints, body.time, frequency)


def dynamic(frequency: float, **options: Any) -> tuple[dict[str, float], dict[str, Any], dict[str, float]]:
    from a0_dynamic_compare import run, transient
    from a0_runs import SCRIPT_A_FRAME, quantities

    found = run(frequency, **options)
    return (quantities(found.history, frequency, SCRIPT_A_FRAME),
            joint_motion(found.joints, found.history.body.time, frequency), transient(found))


def main() -> None:
    import jax

    from a0_compare import difference
    from a0_dynamic_compare import DECLARED, GATED, TOLERANCES

    jax.config.update("jax_enable_x64", False)
    rows = []
    for frequency in FREQUENCIES:
        print(f"{frequency:g} Hz ...", flush=True)
        kin, commanded = kinematic(frequency)
        dyn, moved, kick = dynamic(frequency)
        rows.append({"frequency_hz": frequency, "kinematic": kin, "dynamic": dyn,
                     "difference": {name: difference(name, dyn[name], kin[name]) for name in GATED
                                    if name in kin and name in dyn},
                     "commanded": commanded, "dynamic_joint_motion": moved, "transient": kick})
    crossing = {}
    for name in GATED:
        speeds = np.array([max(row["commanded"][CROSSING_SPEED]) for row in rows])
        moved = np.array([row["difference"].get(name, math.nan) for row in rows])
        usable = np.isfinite(moved)
        slope = float(np.sum(speeds[usable] * moved[usable]) / np.sum(speeds[usable] ** 2))
        crossing[name] = {"slope_per_rad_s": slope, "threshold": TOLERANCES[name],
                          "crossing_joint_speed_rad_s": TOLERANCES[name] / abs(slope) if slope else math.inf,
                          "above_the_threshold_at": [row["frequency_hz"] for row in rows
                                                     if abs(row["difference"].get(name, 0.0)) > TOLERANCES[name]]}
    massive = []
    for frequency in (3.0, 10.0):
        for case, axis in (("spun about a diameter (declared)", (1.0, 0.0, 0.0)),
                           ("spun about its own axis (symmetric)", (0.0, 0.0, 1.0))):
            print(f"{frequency:g} Hz, the magnet's mass, {case} ...", flush=True)
            dyn, moved, kick = dynamic(frequency, density=7500.0, magnet_axis=axis, top_speed=2 * math.pi * 10)
            massive.append({"frequency_hz": frequency, "case": case, "quantities": dyn, "joint_motion": moved,
                            "transient": kick})
    OUTPUT.write_text(json.dumps({
        "what": DECLARED["kinematic_against_dynamic"]["what"],
        "crossing_rule": f"the declared line through zero, fitted against the window's largest commanded joint speed over "
                         f"20 ms ({CROSSING_SPEED}); per step, the speed is the commands' rounding",
        "read_with_it": "the rung differences do not vanish with the commanded speed: the tracking error's is a fixed "
                        "share, the joint loop's 20 ms trail against the controller's whole lag, so a line through "
                        "zero places a crossing no run shows; above_the_threshold_at is what the runs show",
        "conditions": DECLARED["kinematic_against_dynamic"]["conditions"],
        "velocity_limits_rad_s": VELOCITY_LIMITS, "acceleration_limits": "none in the description",
        "runs": rows, "crossing": crossing,
        "with_the_magnets_mass": {"what": DECLARED["with_the_magnets_mass"]["what"],
                                  "inferences": DECLARED["with_the_magnets_mass"]["inferences"],
                                  "runs": massive},
    }, indent=1) + "\n")
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
