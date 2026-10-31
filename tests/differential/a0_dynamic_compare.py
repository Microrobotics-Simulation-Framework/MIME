"""A0 on its dynamic rungs in float32 against the frozen float32 reference, with the differences declared before the
measurement, twice: a run of 0.3's A0 on ``Arm.Dynamic`` and ``Motor.Dynamic`` closed by its loops, and the comparison
of its quantities with each declaration's prediction. Shared by ``test_a0_dynamic.py``, the controls' tests and the
measurement scripts.

**The first declaration** (``a0_dynamic_expected_differences.json``, ``predict_a0_dynamic.py``) is as committed: its
gates that read the rotor's single-precision angle (:data:`ANGLE_READING`, and the magnet's phase) meet the finding at
10 Hz, and at 3 Hz on some runners (``references/a0_dynamic_findings.json``), and are reported, not gated. **The second**
(``a0_dynamic_expected_differences_b.json``, ``predict_a0_dynamic_b.py``), written before its run, reads every phase
against the angle of the same run (``a0_phases.py``) and is the gate (:func:`second`).

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
    integral: np.ndarray | None = None


def _rotate_z(quaternions: np.ndarray) -> np.ndarray:
    """The z axis of each frame whose orientation is the unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = np.asarray(quaternions, dtype=np.float64).T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


def run_built(built: Any, dt: float = STEP, duration: float = DURATION) -> DynamicRun:
    """``built`` (A0 on its dynamic rungs) run for ``duration`` seconds, its history as the reference's: the state at
    ``t = 0`` before the first step's."""
    body_now, field_now = built.graph.get_node_state("umr"), built.graph.get_node_state("rpm.magnetic")
    motor_now, arm_now = built.graph.get_node_state("rpm.motor"), built.graph.get_node_state("rpm.arm")
    controller_now = built.graph.get_node_state("controller")
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
    integral = stacked(controller_now["state_integral"], history["controller"]["state_integral"]) \
        if "state_integral" in history.get("controller", {}) else None
    assert len(pose) == steps + 1
    body = History(time=np.arange(steps + 1) * dt, position=pose[:, :3], velocity=twist[:, :3],
                   orientation=pose[:, 3:], angular_velocity=twist[:, 3:], field=field)
    return DynamicRun(A0History(body, frames[:, 1, :3], relative, angle), frames[:, 1].astype(np.float64),
                      joints.astype(np.float64), joint_rates.astype(np.float64),
                      None if integral is None else integral.astype(np.float64))


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


#: The finding not on the declaration, with its control (``measure_a0_dynamic_findings.py``).
FINDINGS: dict[str, Any] = json.loads((HERE / "references" / "a0_dynamic_findings.json").read_text())
#: The first declaration's gates that read the rotor's single-precision angle: at 10 Hz the finding moves them past
#: their declared tolerances, and they are reported there (the magnet's phase, the pose's third gate, likewise).
ANGLE_READING = ("body_rotation_rate_hz", "field_rotation_rate_hz", "rotor_angle_rate_mean_hz")


def units_apart(key: str, found: DynamicRun) -> tuple[float, float]:
    """How many units in its last place 0.3's rotor angle is from the emulation's at the window's start and end
    (reported: the finding's size; no gate reads it)."""
    held = FINDINGS["by_frequency"][key]["emulated_angle"]
    apart = []
    for end in ("window_start", "window_end"):
        angle = held[end]["angle"]
        apart.append(abs(float(found.history.rotor_angle[held[end]["step"]]) - angle)
                     / float(np.spacing(np.float32(abs(angle)))))
    return apart[0], apart[1]


def failures(rows: list[Row]) -> list[str]:
    """The quantities beyond their declared tolerance."""
    return [row.quantity for row in rows if not row.holds]


def transient(found: DynamicRun) -> dict[str, float]:
    """What the rotor's spin-up does to the arm: joint 6's largest excursion from its home and its largest speed."""
    from chain_cases import AR4_HOME

    return {"joint_6_excursion_max_rad": float(np.max(np.abs(found.joints[:, 5] - AR4_HOME[5]))),
            "joint_6_speed_max_rad_s": float(np.max(np.abs(found.joint_rates[:, 5])))}


def reference_poses(key: str) -> tuple[np.ndarray, np.ndarray]:
    """The reference's sampled times and recorded rotor poses (every tenth step)."""
    found = np.load(HERE / "references" / "a0" / f"f32_{key}_trajectory_every_10th_step.npz")
    return found["time_s"], found["state__motor__rotor_pose_world"].astype(np.float64)


# -- the second declaration --------------------------------------------------------------------------------------------

SECOND: dict[str, Any] = json.loads((HERE / "a0_dynamic_expected_differences_b.json").read_text())
#: The emulation's predicted turn of the rotor frame against its angle, sample by sample in the window.
PREDICTED_TURNS = np.load(HERE / "a0_dynamic_phases_b.npz")
#: The steps the motor's angle offset is read at (``predict_a0_dynamic_b``).
OFFSET_STEPS = (9999, 20000)


@dataclass(frozen=True)
class Gate:
    """One of the second declaration's gates: ``relative`` misses are over the prediction's magnitude."""

    quantity: str
    measured: float
    predicted: float
    tolerance: float
    relative: bool = False

    @property
    def miss(self) -> float:
        return (self.measured - self.predicted) / abs(self.predicted) if self.relative else \
            self.measured - self.predicted

    @property
    def holds(self) -> bool:
        return math.isfinite(self.miss) and abs(self.miss) <= self.tolerance


def phases(found: DynamicRun) -> Any:
    """``found`` as ``a0_phases`` reads a run, at every tenth step, in script A's frame."""
    from a0_phases import EVERY, Phases

    body = found.history.body
    assert found.integral is not None and found.history.rotor_angle is not None
    return Phases(np.asarray(body.time)[::EVERY], np.asarray(body.field)[::EVERY],
                  np.asarray(body.orientation)[::EVERY], np.asarray(body.position)[::EVERY],
                  np.asarray(found.history.rotor_angle)[::EVERY], found.rotor_poses[::EVERY],
                  found.integral[::EVERY], SCRIPT_A_FRAME)


def second(key: str, measured: dict[str, float], found: DynamicRun, *, read: Any = None,
           against: str | None = None) -> list[Gate]:
    """Every gate of the second declaration at ``key``'s frequency, against ``against``'s prediction (``key``'s):
    the first declaration's gates that never read the angle, carried; the phases read against the angle (``read``,
    ``found`` read by :func:`phases` when not given); the rotor frame's phase against its angle, sample by sample; the
    motor's angle offset from the emulation's across the window; the angle at the start; joint 6's kick."""
    from a0_phases import phase_against_angle_miss, quantities, reference_phases
    from a0_poses import difference, misses

    other = against or key
    carried = SECOND["carried"]
    gates = [Gate(name, measured[name], carried["predicted"][other][name], carried["tolerances"][name],
                  relative=name != "phase_lag_field_minus_moment_rad") for name in carried["tolerances"]]
    run_phases = read if read is not None else phases(found)
    on_run = quantities(run_phases, FREQUENCY[key])
    gates += [Gate(name, on_run[name], gate["predicted"][other], gate["tolerance"]["value"])
              for name, gate in SECOND["gates"].items()]
    reference = reference_phases(f"f32_{key}")
    turned = phase_against_angle_miss(run_phases, reference, PREDICTED_TURNS[f"{other}_predicted_turn"],
                                      FREQUENCY[key])
    gates.append(Gate("magnet_phase_against_rotor_angle_rad", turned, 0.0,
                      SECOND["magnet_phase_against_rotor_angle_rad"]["tolerance"]["value"]))
    time, poses = reference_poses(key)
    pose_misses = misses(found.rotor_poses[::10], poses, difference(np.load(HERE / "a0_dynamic_poses.npz")[
        f"{other}_new"], np.load(HERE / "a0_dynamic_poses.npz")[f"{other}_tag"]), time, FREQUENCY[key])
    gates += [Gate(name, pose_misses[name], 0.0, value) for name, value in carried["magnet_pose"].items()]
    offset = SECOND["motor_angle_offset_change_rad"]
    emulated = offset["emulated_angles"][other]
    angle = np.asarray(found.history.rotor_angle)
    change = ((np.float64(angle[OFFSET_STEPS[1]]) - emulated[1]) - (np.float64(angle[OFFSET_STEPS[0]]) - emulated[0]))
    gates.append(Gate("motor_angle_offset_change_rad", float(change), 0.0, offset["tolerance"][other]))
    gates.append(Gate("motor_initial_angle_rad", float(angle[0]), SECOND["motor_initial_angle_rad"]["predicted"],
                      SECOND["motor_initial_angle_rad"]["tolerance"]))
    kick = transient(found)
    gates += [Gate(name, kick[name], SECOND["transient"]["predicted"][other][name], SECOND["transient"]["tolerance"],
                   relative=True) for name in kick]
    return gates


def second_failures(gates: list[Gate]) -> list[str]:
    return [gate.quantity for gate in gates if not gate.holds]


def second_table(gates: list[Gate]) -> str:
    lines = [f"{'quantity':44s} {'0.3':>16s} {'predicted':>16s} {'miss':>10s} {'tolerance':>9s}"]
    lines += [f"{g.quantity:44s} {g.measured:16.10g} {g.predicted:16.10g} {g.miss:+10.2e} {g.tolerance:9.2e} "
              f"{'ok' if g.holds else 'MISS'}" for g in gates]
    return "\n".join(lines)


__all__ = ["ANGLE_READING", "DECLARED", "FINDINGS", "FREQUENCY", "GATED", "OFFSET_STEPS", "PREDICTED", "REFERENCE",
           "SECOND", "TOLERANCES", "DynamicRun", "Gate", "Row", "compare", "difference", "failures", "measure",
           "missed", "phases", "reference_history", "reference_poses", "run", "run_built", "second",
           "second_failures", "second_table", "table", "transient", "units_apart"]
