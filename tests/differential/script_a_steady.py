"""Script A's steady quantities, by the rule phase 0 fixed before its reference runs (its ``analyze.py``, in any frame):
the window is the second half of the run, trimmed at its start to a whole number of nominal field periods ending at
the last step, ``t_end = T``, ``n = floor((T / 2) f)``, ``t_start = T - n / f``.

Angles about the tube's axis ``a`` are right-handed, measured from ``e1`` towards ``a x e1``: the frozen reference's
tube runs along z with ``e1 = x``, script A's in 0.3 along x with ``e1 = y``, so both read the same angle for the same
physical turn. Each history starts with the state at ``t = 0`` and has one row per step after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Where script A's directions point in one world: the tube's axis and a point on it, the direction angles about
    it are measured from, and the body-frame axis of the body's moment."""

    axis: tuple[float, float, float]
    angle_zero: tuple[float, float, float]
    moment_in_body: tuple[float, float, float]
    axis_point: tuple[float, float, float] = (0.0, 0.0, 0.0)


#: The frozen reference's frame: the tube along z, gravity along -y, the moment along body x.
REFERENCE_FRAME = Frame(axis=(0.0, 0.0, 1.0), angle_zero=(1.0, 0.0, 0.0), moment_in_body=(1.0, 0.0, 0.0))
#: Script A's frame in 0.3: the tube along x, up along z (gravity along -z), the moment along body y. It is the
#: reference's turned by the cyclic permutation x -> y -> z -> x, a proper rotation.
SCRIPT_A_FRAME = Frame(axis=(1.0, 0.0, 0.0), angle_zero=(0.0, 1.0, 0.0), moment_in_body=(0.0, 1.0, 0.0))


@dataclass(frozen=True)
class History:
    """The run's arrays, one row per sample from ``t = 0``: position, velocity and the field in the world frame,
    orientation as unit quaternions ``(w, x, y, z)`` body to world."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    field: np.ndarray


def rotate(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = (quaternions[:, i] for i in range(4))
    matrix = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
    return matrix @ np.asarray(vector, dtype=np.float64)


def _angle(vectors: np.ndarray, frame: Frame) -> np.ndarray:
    axis, first = np.asarray(frame.axis), np.asarray(frame.angle_zero)
    second = np.cross(axis, first)
    return np.unwrap(np.arctan2(vectors @ second, vectors @ first))


def _rate_hz(time: np.ndarray, angle: np.ndarray) -> float:
    if not (np.isfinite(time).all() and np.isfinite(angle).all()):
        return math.nan
    return float(np.polyfit(time, angle, 1)[0] / (2 * math.pi))


def window(time: np.ndarray, frequency: float) -> tuple[float, float, int]:
    """``(t_start, t_end, n_periods)`` of the window."""
    end = float(time[-1])
    periods = int(math.floor((end / 2) * frequency + 1e-9))
    return end - periods / frequency, end, periods


def steady(history: History, frame: Frame, frequency: float) -> dict[str, float]:
    """The steady quantities the reference's ``_steady.json`` holds, under the same names."""
    time = np.asarray(history.time, dtype=np.float64)
    start, _, periods = window(time, frequency)
    inside = time >= start - 1e-12
    t = time[inside]
    axis = np.asarray(frame.axis)
    position = np.asarray(history.position, dtype=np.float64)
    along = position[inside] @ axis
    relative = position[inside] - np.asarray(frame.axis_point)
    radial = np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1)
    moment = rotate(np.asarray(history.orientation, dtype=np.float64)[inside], np.asarray(frame.moment_in_body))
    field = np.asarray(history.field, dtype=np.float64)[inside]
    body_angle, field_angle = _angle(moment, frame), _angle(field, frame)
    body, turning = _rate_hz(t, body_angle), _rate_hz(t, field_angle)
    in_plane = np.linalg.norm(field - np.outer(field @ axis, axis), axis=1)
    return {
        "n_field_periods": float(periods),
        "swim_speed_axial_m_s": float((along[-1] - along[0]) / (t[-1] - t[0])),
        "swim_speed_axial_mean_velocity_m_s": float(np.mean(np.asarray(history.velocity, dtype=np.float64)[inside]
                                                            @ axis)),
        "radial_offset_mean_m": float(radial.mean()),
        "radial_offset_min_m": float(radial.min()),
        "radial_offset_max_m": float(radial.max()),
        "body_rotation_rate_hz": body,
        "body_axial_angular_velocity_mean_hz": float(np.mean(np.asarray(history.angular_velocity,
                                                                         dtype=np.float64)[inside] @ axis)
                                                     / (2 * math.pi)),
        "field_rotation_rate_hz": turning,
        "rotation_ratio_body_over_field": body / turning if turning else math.nan,
        "phase_lag_field_minus_moment_rad": float(np.angle(np.mean(np.exp(1j * (field_angle - body_angle))))),
        "field_inplane_magnitude_mean_T": float(in_plane.mean()),
    }


def speed_between(history: History, frame: Frame, start: float, end: float) -> float:
    """The speed along the tube by displacement between two sample times."""
    time = np.asarray(history.time, dtype=np.float64)
    first, last = int(np.searchsorted(time, start - 1e-12)), int(np.searchsorted(time, end - 1e-12))
    along = np.asarray(history.position, dtype=np.float64) @ np.asarray(frame.axis)
    return float((along[last] - along[first]) / (time[last] - time[first]))


#: The quantities gated against the reference, and those reported only, with the reason.
GATED = ("swim_speed_axial_m_s", "swim_speed_axial_mean_velocity_m_s", "body_rotation_rate_hz",
         "field_rotation_rate_hz", "phase_lag_field_minus_moment_rad")
REPORTED = {
    "radial_offset_mean_m": "the body rests on the projection's constraint (its minimum equals its maximum in the "
                            "reference): it measures the constraint, not the dynamics",
    "rotation_ratio_body_over_field": "locked, the body turns with the field whatever the frequency: it is blind to a "
                                      "change in frequency (the +10% control leaves it at 1)",
}
