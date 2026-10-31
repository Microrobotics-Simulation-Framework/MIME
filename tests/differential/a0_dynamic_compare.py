"""A0 on its dynamic rungs in float32 against the frozen float32 reference, with the differences declared before the
measurement (``a0_dynamic_expected_differences.json``, written by ``predict_a0_dynamic.py``): a run of 0.3's A0 on
``Arm.Dynamic`` and ``Motor.Dynamic`` closed by its loops, and the comparison of its gated quantities with the
declared prediction. Shared by ``test_a0_dynamic.py``, ``test_a0_dynamic_controls.py`` and the measurement scripts.

0.3's dynamic motor publishes its rotor's absolute spin and its frames' angular velocities, not a speed relative to
its stator; ``rotor_rate_mean_hz`` reads that relative speed as the motor's state gives it, its spin less the flange's
turning along the axis (the reference's motor recorded its own speed, relative to its stator, which nothing turned
there), and ``rotor_angle_rate_mean_hz`` the motor's output, its angle, as 7c1's comparison does."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from a0_compare import Row, difference, missed, table
from a0_runs import SCRIPT_A_FRAME, A0History, quantities, reference_history, reference_steady
from script_a_steady import History

from mime.units import Hz, ms, s

HERE = Path(__file__).parent
DECLARED: dict[str, Any] = json.loads((HERE / "a0_dynamic_expected_differences.json").read_text())
PREDICTED: dict[str, dict[str, float]] = DECLARED["predicted"]
TOLERANCES: dict[str, float] = DECLARED["tolerances"]["values"]
GATED = tuple(TOLERANCES)
FREQUENCY = {"3hz": 3.0, "10hz": 10.0}
REFERENCE = {key: reference_steady(f"f32_{key}") for key in FREQUENCY}
STEP = 5e-4
DURATION = 10.0


@dataclass(frozen=True)
class DynamicRun:
    """A run of A0 on its dynamic rungs: its history in the reference's form, and what the comparison reads beside it
    (the rotor's pose and the arm's joints and rates at every step, the controller's state)."""

    history: A0History
    rotor_poses: np.ndarray
    joints: np.ndarray
    joint_rates: np.ndarray


def _rotate_z(quaternions: np.ndarray) -> np.ndarray:
    """The z axis of each frame whose orientation is the unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = np.asarray(quaternions, dtype=np.float64).T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


def run_built(built: Any, dt: float = STEP, duration: float = DURATION) -> DynamicRun:
    """``built`` (A0 on its dynamic rungs) run for ``duration`` seconds, its history as the reference's: the state at
    ``t = 0`` before the first step's."""
    body_now, field_now = built.graph.get_node_state("umr"), built.graph.get_node_state("rpm.magnetic")
    motor_now, arm_now = built.graph.get_node_state("rpm.motor"), built.graph.get_node_state("rpm.arm")
    steps = round(duration / dt)
    history = built.run(s(duration), history=True).history
    assert history is not None

    def stacked(first: Any, rest: Any) -> np.ndarray:
        return np.concatenate([np.asarray(first)[None], np.asarray(rest)])

    pose = stacked(body_now["pose"], history["umr"]["pose"])
    twist = stacked(body_now["twist"], history["umr"]["twist"])
    field = stacked(field_now["flux_density_magnetic_torque_umr"],
                    history["rpm.magnetic"]["flux_density_magnetic_torque_umr"])
    frames = stacked(motor_now["frame_poses"], history["rpm.motor"]["frame_poses"])
    rates = stacked(motor_now["frame_angular_velocities"], history["rpm.motor"]["frame_angular_velocities"])
    spin = stacked(motor_now["spin"], history["rpm.motor"]["spin"]).astype(np.float64)
    axis = _rotate_z(frames[:, 0, 3:])
    relative = spin - np.sum(axis * rates[:, 0].astype(np.float64), axis=1)
    angle = stacked(motor_now["angle"], history["rpm.motor"]["angle"])
    joints = stacked(arm_now["joint_positions"], history["rpm.arm"]["joint_positions"])
    joint_rates = stacked(arm_now["joint_velocities"], history["rpm.arm"]["joint_velocities"])
    assert len(pose) == steps + 1
    body = History(time=np.arange(steps + 1) * dt, position=pose[:, :3], velocity=twist[:, :3],
                   orientation=pose[:, 3:], angular_velocity=twist[:, 3:], field=field)
    return DynamicRun(A0History(body, frames[:, 1, :3], relative, angle), frames[:, 1].astype(np.float64),
                      joints.astype(np.float64), joint_rates.astype(np.float64))


def run(frequency: float, dt: float = STEP, duration: float = DURATION, **options: Any) -> DynamicRun:
    """0.3's A0 on its dynamic rungs (``options`` are ``script_a0_dynamic.a0_dynamic``'s)."""
    from script_a0_dynamic import a0_dynamic

    built = a0_dynamic(frequency=Hz(frequency), dt=ms(dt * 1e3), **options).build()
    return run_built(built, dt, duration)


def measure(key: str, **options: Any) -> tuple[dict[str, float], DynamicRun]:
    """The steady quantities of 0.3's A0 on its dynamic rungs at ``key``'s frequency, the reference's window rule, in
    the process's precision; and the run."""
    found = run(FREQUENCY[key], **options)
    return quantities(found.history, FREQUENCY[key], SCRIPT_A_FRAME), found


def compare(measured: dict[str, float], key: str, against: str | None = None) -> list[Row]:
    """One row per gated quantity: the measurement against the declared prediction at ``against`` (``key``'s)."""
    predicted = PREDICTED[against or key]
    return [Row(name, REFERENCE[key].get(name, math.nan), measured[name], predicted[name], TOLERANCES[name])
            for name in GATED]


def transient(found: DynamicRun) -> dict[str, float]:
    """What the rotor's spin-up does to the arm: joint 6's largest excursion from its home and its largest speed."""
    from chain_cases import AR4_HOME

    return {"joint_6_excursion_max_rad": float(np.max(np.abs(found.joints[:, 5] - AR4_HOME[5]))),
            "joint_6_speed_max_rad_s": float(np.max(np.abs(found.joint_rates[:, 5])))}


def reference_poses(key: str) -> tuple[np.ndarray, np.ndarray]:
    """The reference's sampled times and recorded rotor poses (every tenth step)."""
    found = np.load(HERE / "references" / "a0" / f"f32_{key}_trajectory_every_10th_step.npz")
    return found["time_s"], found["state__motor__rotor_pose_world"].astype(np.float64)


__all__ = ["DECLARED", "FREQUENCY", "GATED", "PREDICTED", "REFERENCE", "TOLERANCES", "DynamicRun", "Row", "compare",
           "difference", "measure", "missed", "reference_history", "reference_poses", "run", "run_built", "table",
           "transient"]
