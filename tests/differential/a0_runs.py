"""A0's steady quantities, by the rule phase 0 fixed before its reference runs (the same as script A's, with the
tracking quantities added), read from the frozen reference or from a run of 0.3's A0 through the public API.

The reference's frame has the tube along x, as 0.3's A0 does; its body's moment is body x, 0.3's body y (as script
A's). Each history starts with the state at ``t = 0`` and has one row per step after it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from script_a0 import a0
from script_a_steady import SCRIPT_A_FRAME, Frame, History, steady

from mime.units import Hz, s

HERE = Path(__file__).parent
REFERENCES = HERE / "references" / "a0"
#: The frozen reference's frame: the tube along x, the angle measured from y, the moment body x.
REFERENCE_FRAME = Frame(axis=(1.0, 0.0, 0.0), angle_zero=(0.0, 1.0, 0.0), moment_in_body=(1.0, 0.0, 0.0))
STANDOFF = 0.15
DURATION = 10.0
LABELS = {"f32": "f32", "mixed_f64": "mixed_f64"}

#: What the comparison gates, and the tracking quantities beside them. ``radial_offset_mean_m`` sits on the projection's
#: constraint (its minimum equals its maximum in the reference) and ``rotation_ratio_body_over_field`` is 1 at any
#: frequency while the body is locked: neither can tell a change, so both are reported and never gated.
BODY = ("swim_speed_axial_m_s", "swim_speed_axial_mean_velocity_m_s", "body_rotation_rate_hz", "field_rotation_rate_hz",
        "phase_lag_field_minus_moment_rad")
TRACKING = ("tracking_error_axial_mean_m", "tracking_error_axial_rms_m", "magnet_body_distance_minus_standoff_mean_m",
            "rotor_rate_mean_hz")
REPORTED = {
    "radial_offset_mean_m": "the body rests on the projection's constraint: it measures the constraint, not the "
                            "dynamics",
    "rotation_ratio_body_over_field": "locked, the body turns with the field whatever the frequency",
    "field_inplane_magnitude_mean_T": "the field's in-plane magnitude at the body, a function of the standoff, which "
                                      "the distance already gates",
    "tracking_error_lateral_y_mean_m": "about a micrometre, at the rounding of the controller's single precision",
}


@dataclass(frozen=True)
class A0History:
    """The body's history in script A's form, the magnet's (rotor's) position, and the rotor's angular velocity."""

    body: History
    rotor_position: np.ndarray
    rotor_angular_velocity: np.ndarray


def quantities(history: A0History, frequency: float, frame: Frame) -> dict[str, float]:
    """The steady quantities the reference's ``_steady.json`` holds, under the same names."""
    found = steady(history.body, frame, frequency)
    time = np.asarray(history.body.time, dtype=np.float64)
    inside = time >= float(time[-1]) - int(np.floor((time[-1] / 2) * frequency + 1e-9)) / frequency - 1e-12
    delta = np.asarray(history.rotor_position, dtype=np.float64)[inside] - np.asarray(history.body.position,
                                                                                      dtype=np.float64)[inside]
    along = delta[:, 0]
    found.update({
        "tracking_error_axial_mean_m": float(along.mean()),
        "tracking_error_axial_rms_m": float(np.sqrt((along ** 2).mean())),
        "tracking_error_axial_max_abs_m": float(np.abs(along).max()),
        "tracking_error_lateral_y_mean_m": float(delta[:, 1].mean()),
        "magnet_body_distance_mean_m": float(np.linalg.norm(delta, axis=1).mean()),
        "magnet_body_distance_minus_standoff_mean_m": float(np.linalg.norm(delta, axis=1).mean() - STANDOFF),
        "rotor_rate_mean_hz": float(np.asarray(history.rotor_angular_velocity, dtype=np.float64)[inside].mean()
                                    / (2 * np.pi)),
    })
    return found


def reference_steady(label: str) -> dict[str, float]:
    """The reference's steady quantities from every step of its run, as phase 0 computed them."""
    return {name: value for name, value in json.loads((REFERENCES / f"{label}_steady.json").read_text()).items()
            if isinstance(value, float)}


def reference_history(label: str) -> A0History:
    """The reference's arrays at every tenth step."""
    found = np.load(REFERENCES / f"{label}_trajectory_every_10th_step.npz")
    body = History(time=found["time_s"], position=found["state__body__position"].astype(np.float64),
                   velocity=found["state__body__velocity"].astype(np.float64),
                   orientation=found["state__body__orientation"].astype(np.float64),
                   angular_velocity=found["state__body__angular_velocity"].astype(np.float64),
                   field=found["state__ext_magnet__field_vector"].astype(np.float64))
    return A0History(body, found["state__motor__rotor_pose_world"][:, :3].astype(np.float64),
                     found["state__motor__angular_velocity"].astype(np.float64))


def run(frequency: float, dt: float = 5e-4, duration: float = DURATION, **options: Any) -> A0History:
    """0.3's A0 for ``duration`` seconds, its history as the reference's: the state at ``t = 0`` before the first
    step's. ``options`` are ``script_a0.a0``'s."""
    from mime.units import ms

    built = a0(frequency=Hz(frequency), dt=ms(dt * 1e3), **options).build()
    return run_built(built, dt, duration)


def run_built(built: Any, dt: float, duration: float = DURATION) -> A0History:
    body_now, field_now = built.graph.get_node_state("umr"), built.graph.get_node_state("rpm.magnetic")
    motor_now = built.graph.get_node_state("rpm.motor")
    steps = round(duration / dt)
    history = built.run(s(duration), history=True).history
    assert history is not None

    def stacked(first: Any, rest: Any) -> np.ndarray:
        return np.concatenate([np.asarray(first)[None], np.asarray(rest)])

    pose = stacked(body_now["pose"], history["umr"]["pose"])
    twist = stacked(body_now["twist"], history["umr"]["twist"])
    field = stacked(field_now["flux_density_magnetic_torque_umr"],
                    history["rpm.magnetic"]["flux_density_magnetic_torque_umr"])
    rotor = stacked(motor_now["frame_poses"], history["rpm.motor"]["frame_poses"])[:, 1, :3]
    rate = stacked(0.0, history["drive:rpm.motor.velocity"]["value"])
    assert len(pose) == steps + 1
    return A0History(History(time=np.arange(steps + 1) * dt, position=pose[:, :3], velocity=twist[:, :3],
                             orientation=pose[:, 3:], angular_velocity=twist[:, 3:], field=field), rotor, rate)


__all__ = ["A0History", "BODY", "REFERENCE_FRAME", "REPORTED", "SCRIPT_A_FRAME", "TRACKING", "quantities",
           "reference_history", "reference_steady", "run", "run_built"]
