"""The phases of A0's rotor, its field and its body, read so that the rotor's single-precision angle cancels: the
quantities the second declaration of A0 on its dynamic rungs gates (``a0_dynamic_expected_differences_b.json``),
computed alike from the frozen reference, from the emulation and from a run of 0.3, at every tenth step.

Why they cancel the angle. The rotor's angle is a single-precision sum, and two runs whose increments agree to their
last bits still reach the steady window a few units of the angle's last place apart (the first declaration's finding,
``references/a0_dynamic_findings.json``). Everything downstream of the angle follows it: the rotor's frame is turned
by it, the magnet on that frame, and the field at the body is the magnet's. So each quantity here is read against the
angle of the same run: the field's turning and phase less the motor's angle, the body's turning less the field's, the
rotor frame's phase less its angle's own difference, the magnet's direction against the rotor frame it sits on. A unit
of the angle moves the field and the motor's angle alike, and drops out. What the angle's rounding still moves is the
motor's angle rate, gated on its own (``angle_offset_change``), and the absolute phase, which the run carries in its
start (``motor_initial_angle_rad``) and its rotor loop's integral (``rotor_loop_integral_rad``): the angle the loop
holds the rotor behind its command, ``theta = theta_0 + Omega t - I`` in exact arithmetic.

Signs. Angles about the tube's axis are script A's (``script_a_steady``); the field turns about it against the motor's
angle (the motor's axis points down the tube), so the field's phase against the angle is their sum."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from a0_poses import parts, phase
from a0_runs import REFERENCE_FRAME, window
from script_a_steady import Frame, _angle, rotate

REFERENCES = Path(__file__).parent / "references" / "a0"

#: The reference's arrays, and so every array these quantities read, are at every tenth step.
EVERY = 10
#: The quantities a run gives on its own, and their units.
SCALARS = {"field_turning_against_motor_angle_hz": "Hz", "body_turning_against_field_hz": "Hz",
           "field_phase_against_motor_angle_rad": "rad", "magnet_mount_phase_rad": "rad",
           "rotor_loop_integral_rad": "rad"}
#: The rotor frame's phase against its angle, which is read against the reference's sample by sample.
AGAINST_REFERENCE = {"magnet_phase_against_rotor_angle_rad": "rad"}


@dataclass(frozen=True)
class Phases:
    """A run at every tenth step: the field at the body, the body's orientation and position, the motor's angle, the
    rotor's pose ``[x, y, z, w, qx, qy, qz]`` (its x the magnet's moment, its z the motor's axis) and the rotor loop's
    integral of its speed error, in ``frame`` (where the body's moment sits in the body)."""

    time: np.ndarray
    field: np.ndarray
    orientation: np.ndarray
    position: np.ndarray
    angle: np.ndarray
    rotor: np.ndarray
    integral: np.ndarray
    frame: Frame


def reference_phases(label: str) -> Phases:
    """The frozen reference's run ``label`` (``f32_3hz``, ``mixed_f64_10hz``, ...) at every tenth step, from its
    trajectory or rotor-pose file and the arrays ``freeze_a0_phase_arrays.py`` copied beside it."""
    copied = np.load(REFERENCES / f"{label}_phases_every_10th_step.npz")
    integral = copied["state__motor__velocity_integral_error"]
    if label.startswith("f32"):
        found = np.load(REFERENCES / f"{label}_trajectory_every_10th_step.npz")
        return Phases(found["time_s"], found["state__ext_magnet__field_vector"], found["state__body__orientation"],
                      found["state__body__position"], found["state__motor__angle"],
                      found["state__motor__rotor_pose_world"], integral, REFERENCE_FRAME)
    rotor = np.load(REFERENCES / f"{label}_rotor_every_10th_step.npz")["state__motor__rotor_pose_world"]
    return Phases(copied["time_s"], copied["state__ext_magnet__field_vector"], copied["state__body__orientation"],
                  copied["state__body__position"], copied["state__motor__angle"], rotor, integral, REFERENCE_FRAME)


def _slope_hz(time: np.ndarray, angle: np.ndarray) -> float:
    return float(np.polyfit(time, angle, 1)[0] / (2 * math.pi))


def _circular_mean(angle: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * angle))))


def inferred_moment(field: np.ndarray, at: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """The direction of the point dipole at ``centre`` whose field at ``at`` is ``field``: from
    ``B = k (3 (m . r) r - m)`` with ``r`` the unit vector from the dipole, ``k m = 3/2 (B . r) r - B``."""
    r = np.asarray(at, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)
    b = np.asarray(field, dtype=np.float64)
    moment = 1.5 * np.sum(b * r, axis=1, keepdims=True) * r - b
    return moment / np.linalg.norm(moment, axis=1, keepdims=True)


def quantities(run: Phases, frequency: float) -> dict[str, float]:
    """The scalars of :data:`SCALARS` over the steady window (the reference's rule): the slope of the field's angle
    about the tube's axis plus the motor's angle, and of the body's moment's angle less the field's, in Hz; the
    circular mean of the field's angle plus the motor's; the circular mean of the turn about the rotor's axis from the
    rotor's x to the moment of the dipole whose field the body felt, at the rotor's centre; the mean of the loop's
    integral."""
    time = np.asarray(run.time, dtype=np.float64)
    inside = window(time, frequency)
    t = time[inside]
    field_angle = _angle(np.asarray(run.field, dtype=np.float64)[inside], run.frame)
    moment = rotate(np.asarray(run.orientation, dtype=np.float64)[inside], np.asarray(run.frame.moment_in_body))
    body_angle = _angle(moment, run.frame)
    motor = np.asarray(run.angle, dtype=np.float64)[inside]
    centre, axis, x = parts(np.asarray(run.rotor, dtype=np.float64)[inside])
    magnet = inferred_moment(np.asarray(run.field)[inside], np.asarray(run.position)[inside], centre)
    return {"field_turning_against_motor_angle_hz": _slope_hz(t, field_angle + motor),
            "body_turning_against_field_hz": _slope_hz(t, body_angle - field_angle),
            "field_phase_against_motor_angle_rad": _circular_mean(field_angle + motor),
            "magnet_mount_phase_rad": _circular_mean(phase(magnet, x, axis)),
            "rotor_loop_integral_rad": float(np.mean(np.asarray(run.integral, dtype=np.float64)[inside]))}


def phase_against_angle(rotor: np.ndarray, angle: np.ndarray, against_rotor: np.ndarray,
                        against_angle: np.ndarray) -> np.ndarray:
    """At each sample, the turn of ``rotor``'s x from ``against_rotor``'s about ``against_rotor``'s axis, less the turn
    the two angles account for: how far the rotor frame's phase is from where its own angle puts it, against the
    other's (the flange's turn about the axis, and the mount's)."""
    _, _, x = parts(rotor)
    _, w, y = parts(against_rotor)
    turn = phase(x, y, w) - (np.asarray(angle, dtype=np.float64) - np.asarray(against_angle, dtype=np.float64))
    return np.angle(np.exp(1j * turn))


def phase_against_angle_miss(run: Phases, reference: Phases, predicted: np.ndarray, frequency: float) -> float:
    """The largest miss over the steady window of ``run``'s rotor phase against its angle, read against
    ``reference``'s, from the ``predicted`` turn (the emulation's 0.3 against its tag, sample by sample in the
    window)."""
    inside = window(np.asarray(reference.time, dtype=np.float64), frequency)
    found = phase_against_angle(run.rotor, run.angle, reference.rotor, reference.angle)[inside]
    return float(np.max(np.abs(np.angle(np.exp(1j * (found - predicted))))))


def angle_offset_change(angle: np.ndarray, against: np.ndarray, first: int, last: int) -> float:
    """How far the offset of ``angle`` from ``against`` (two single-precision angles, at every step) moves between
    step ``first`` and step ``last``, in radians, in exact arithmetic (each is a difference of single-precision values,
    which double precision holds exactly)."""
    a, b = np.asarray(angle), np.asarray(against)
    return float((np.float64(a[last]) - np.float64(b[last])) - (np.float64(a[first]) - np.float64(b[first])))


def crossings_bound(start: float, end: float) -> tuple[float, list[float]]:
    """The most a single-precision angle's offset from another's can change while both turn from ``start`` to ``end``
    on increments that round alike: one unit of the last place below each power of two the angle crosses (where each
    is rounded onto a grid twice as coarse, an odd offset of the finer units becomes an even one), and nothing
    between. The bound and the powers crossed."""
    crossed = [2.0 ** n for n in range(-20, 40) if start < 2.0 ** n <= end]
    return float(sum(np.spacing(np.float32(power / 2)) for power in crossed)), crossed


def midpoint_margin(angle: np.ndarray, increment: np.ndarray) -> float:
    """The least distance, over the steps given, of each step's exact sum ``angle + increment`` from the nearest
    point where single precision rounds it the other way (a midpoint between two neighbours), in units of the
    increment's own last place: how much larger the increment would have to be for that step's sum to round
    otherwise."""
    a, i = np.asarray(angle, dtype=np.float32), np.asarray(increment, dtype=np.float32)
    exact = a.astype(np.float64) + i.astype(np.float64)
    rounded = (a + i).astype(np.float64)
    unit = np.spacing(np.abs(rounded.astype(np.float32))).astype(np.float64)
    below = rounded - unit / 2
    above = rounded + unit / 2
    distance = np.minimum(np.abs(exact - below), np.abs(above - exact))
    return float(np.min(distance / np.spacing(np.abs(i)).astype(np.float64)))


__all__ = ["AGAINST_REFERENCE", "EVERY", "SCALARS", "Phases", "angle_offset_change", "crossings_bound",
           "inferred_moment", "midpoint_margin", "phase_against_angle", "phase_against_angle_miss", "quantities",
           "reference_phases"]
