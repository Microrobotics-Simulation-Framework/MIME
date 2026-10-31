"""The faults the second declaration's controls plant in A0 on its dynamic rungs (``a0_dynamic_expected_differences_b.json``,
``controls``), at 3 Hz and 10 Hz, each with the gate it must fail:

- **the angle offset**: the motor's angle at its start offset by 2^-14 rad (6.10e-5: two units of the angle's last
  place at 10 Hz's window start, 5.2 of the first declaration's phase tolerances; the review of PR #77 found it passed
  the first comparison's rule), failing ``motor_initial_angle_rad``. Every phase read against the angle is built not
  to see it;
- **the mount**: the magnet turned 1e-4 rad in its mount about the motor's axis (8.6 of the first declaration's phase
  tolerances, which every gate passed), failing ``magnet_mount_phase_rad`` and
  ``field_phase_against_motor_angle_rad``;
- **the downstream drift**: on the unplanted run's arrays, the field and the body turned by ``r t`` about the tube's
  axis with the motor's angle untouched, ``r`` three field-turning tolerances, failing
  ``field_turning_against_motor_angle_hz``;
- **the rotor's J** 4 % large, failing joint 6's kick (``joint_6_excursion_max_rad``);
- **the rotor loop's Ki** scaled by three of the integral gate's resolutions either way, failing
  ``rotor_loop_integral_rad``.

Each is also planted at a second size (``SECOND_SIZE``), from which ``measure_a0_dynamic_b_controls.py`` measures the
gate's response and so its resolution (its tolerance over the response), recorded in
``references/a0_dynamic_b_controls_measured.json``: each plant above is at least 1.5 of those resolutions."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from a0_dynamic_compare import FREQUENCY, SECOND, DynamicRun, measure, phases, second

ANGLE_OFFSET = 2.0 ** -14
MOUNT_TURN = 1e-4
DRIFT_IN_TOLERANCES = 3.0
J_FACTOR = 1.04
KI_SIZE = 3.0
#: The second size each plant is made at, for its gate's measured response: by plant, its factor on the first size.
SECOND_SIZE = {"angle offset": 16.0, "mount": 10.0, "drift": 10.0, "J": 0.25, "Ki": 10.0}
#: The gate each plant must fail.
GATE = {"angle offset": "motor_initial_angle_rad", "mount": "magnet_mount_phase_rad",
        "drift": "field_turning_against_motor_angle_hz", "J": "joint_6_excursion_max_rad",
        "Ki": "rotor_loop_integral_rad"}
#: The kick is over within the first half second; the J plant runs that long.
KICK_DURATION = 0.5


def ki_step(key: str) -> float:
    """One of the integral gate's resolutions in Ki's relative change, as the declaration predicted it."""
    return float(SECOND["controls"][4]["predicted_resolution"][key])


def drift_rate() -> float:
    """``r``, rad/s: three of the field-turning gate's tolerances."""
    return 2 * math.pi * DRIFT_IN_TOLERANCES * SECOND["gates"]["field_turning_against_motor_angle_hz"]["tolerance"][
        "value"]


def planted(key: str, plant: str, size: float = 1.0, sign: float = 1.0) -> tuple[dict[str, float], DynamicRun]:
    """0.3's A0 on its dynamic rungs at ``key`` with ``plant`` at ``size`` times its declared size, of ``sign``."""
    if plant == "angle offset":
        return measure(key, initial_angle=sign * size * ANGLE_OFFSET)
    if plant == "mount":
        turn = sign * size * MOUNT_TURN
        return measure(key, magnet_direction=(math.cos(turn), math.sin(turn), 0.0))
    if plant == "J":
        from chain_cases import ROTOR_INERTIA
        from a0_dynamic_compare import run

        found = run(FREQUENCY[key], duration=KICK_DURATION,
                    rotor_inertia=ROTOR_INERTIA * (1.0 + sign * size * (J_FACTOR - 1.0)))
        return {}, found
    if plant == "Ki":
        return measure(key, integral=1e-2 * (1.0 + sign * size * KI_SIZE * ki_step(key)))
    raise ValueError(plant)


def _turned_about_x(vectors: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = (np.asarray(vectors, dtype=np.float64)[:, i] for i in range(3))
    return np.stack([x, c * y - s * z, s * y + c * z], axis=1)


def _quaternion_turned_about_x(q: np.ndarray, angle: np.ndarray) -> np.ndarray:
    h = 0.5 * angle
    w1, x1 = np.cos(h), np.sin(h)
    w2, x2, y2, z2 = np.asarray(q, dtype=np.float64).T
    return np.stack([w1 * w2 - x1 * x2, w1 * x2 + x1 * w2, w1 * y2 - x1 * z2, w1 * z2 + x1 * y2], axis=1)


def drifted(found: DynamicRun, rate: float) -> Any:
    """``found`` read as ``a0_phases`` reads it, with the field and the body turned by ``rate t`` about the tube's
    axis (script A's x) and the motor's angle untouched: a downstream drift between the rotor and the field."""
    read = phases(found)
    angle = rate * np.asarray(read.time, dtype=np.float64)
    return replace(read, field=_turned_about_x(read.field, angle),
                   orientation=_quaternion_turned_about_x(read.orientation, angle))


def drifted_gates(key: str, measured: dict[str, float], found: DynamicRun, size: float = 1.0) -> Any:
    return second(key, measured, found, read=drifted(found, size * drift_rate()))


__all__ = ["ANGLE_OFFSET", "DRIFT_IN_TOLERANCES", "GATE", "J_FACTOR", "KICK_DURATION", "KI_SIZE", "MOUNT_TURN",
           "SECOND_SIZE", "drift_rate", "drifted", "drifted_gates", "ki_step", "planted"]
