"""The faults planted in each part the A0 emulator was extended to, for the comparison's controls: the arm's pose, the
dipole's moment and its position, the controller's latency and its law. Each is a size in units of the comparison's
resolution for the quantity it moves, so that a planted fault is above what the comparison can see by a stated
factor (``test_a0_controls.py`` plants three resolutions; ``measure_a0_controls.py`` sweeps the factor).

The resolution of a quantity is its declared tolerance. For the tracking error's mean it is
``tolerance * prediction`` metres; for the lag, ``tolerance`` radians; the controller's latency can only be planted
in whole steps, the smallest of which already moves the tracking error by several tolerances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from a0_compare import PREDICTED, TOLERANCES

from chain_cases import AR4_BASE, MAGNET_MOMENT
from mime.units import m, ms

KEY = "3hz"
TRACKING = "tracking_error_axial_mean_m"
LAG = "phase_lag_field_minus_moment_rad"
#: Metres: the tracking error's mean at 3 Hz moved by its own tolerance.
LENGTH_RESOLUTION = TOLERANCES[TRACKING] * abs(PREDICTED[KEY][TRACKING])
#: Metres: a dipole shifted along the tube moves no frame the comparison reads (the tracking error is the rotor's), but
#: moves the axial gradient force on the body and so its speed: by 3.97e-4 of itself per micrometre, by the emulation
#: (``predict_a0.emulate`` with ``altered={"magnet_at": (1e-6, 0, 0)}``), so the shift that moves the speed by its
#: tolerance is the tolerance over that.
SPEED = "swim_speed_axial_mean_velocity_m_s"
DIPOLE_POSITION_RESOLUTION = TOLERANCES[SPEED] / 3.9716e-4 * 1e-6
#: Relative: the dipole moment's change that moves the lag (which goes as 1 / moment) by its tolerance.
MOMENT_RESOLUTION = TOLERANCES[LAG] / abs(PREDICTED[KEY][LAG])
#: Relative: the filter's smoothing change that moves the tracking error (its lag, a time constant dt / alpha, times
#: the speed) by its tolerance.
LAW_RESOLUTION = TOLERANCES[TRACKING]
DT = 5e-4


@dataclass(frozen=True)
class Plant:
    """``apply(size, sign)`` is the options that plant a fault of ``size`` resolutions on ``script_a0.a0``;
    ``quantity`` the gated quantity it moves most."""

    name: str
    part: str
    quantity: str
    apply: Callable[[float, float], dict[str, Any]]
    resolution: str


def _arm_pose(size: float, sign: float) -> dict[str, Any]:
    x, y, z = AR4_BASE
    return {"arm_at": (x + sign * size * LENGTH_RESOLUTION, y, z)}


def _dipole_position(size: float, sign: float) -> dict[str, Any]:
    # along the rotor's axis, which is the tube's: a shift across it would turn with the rotor
    return {"magnet_at": m((0.0, 0.0, sign * size * DIPOLE_POSITION_RESOLUTION))}


def _dipole_moment(size: float, sign: float) -> dict[str, Any]:
    return {"moment": MAGNET_MOMENT * (1.0 + sign * size * MOMENT_RESOLUTION)}


def _latency(size: float, sign: float) -> dict[str, Any]:
    return {"latency": ms(DT * 1e3 * max(round(size), 1))}


def _law(size: float, sign: float) -> dict[str, Any]:
    return {"smoothing": 0.005 * (1.0 + sign * size * LAW_RESOLUTION)}


PLANTS = (
    Plant("arm base shifted along the tube", "the arm's pose", TRACKING, _arm_pose,
          f"{LENGTH_RESOLUTION:.3g} m"),
    Plant("magnet shifted along the rotor's axis, the tube's", "the dipole's position", SPEED, _dipole_position,
          f"{DIPOLE_POSITION_RESOLUTION:.3g} m"),
    Plant("magnet moment scaled", "the dipole's moment", LAG, _dipole_moment, f"{MOMENT_RESOLUTION:.3g} of itself"),
    Plant("controller latency in whole steps", "the controller's latency", TRACKING, _latency,
          "one step (the least that can be planted)"),
    Plant("filter smoothing scaled", "the controller's law", TRACKING, _law, f"{LAW_RESOLUTION:.3g} of itself"),
)
