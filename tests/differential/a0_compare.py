"""A0 in float32 against the frozen float32 reference, with the differences declared before the measurement: the
declaration, a run of 0.3's A0, and the comparison of its gated quantities with the declared prediction. Shared by
``test_a0.py`` and ``test_a0_controls.py``."""

from __future__ import annotations

import json
import math
from functools import cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from a0_runs import SCRIPT_A_FRAME, motor_output_rate, quantities, reference_steady, run

HERE = Path(__file__).parent
DECLARED: dict[str, Any] = json.loads((HERE / "a0_expected_differences.json").read_text())
PREDICTED: dict[str, dict[str, float]] = DECLARED["predicted"]
TOLERANCES: dict[str, float] = DECLARED["tolerances"]["values"]
GATED = tuple(TOLERANCES)
LAG = "phase_lag_field_minus_moment_rad"
FREQUENCY = {"3hz": 3.0, "10hz": 10.0}
REFERENCE = {key: reference_steady(f"f32_{key}") for key in FREQUENCY}
#: The motor's output, gated beside the rotor's commanded speed (``rotor_rate_mean_hz``, which reads what the drive
#: commanded and so cannot fail on a motor fault). The kinematic motor reports no speed; its output is its angle, and
#: this is that angle's mean rate over the steady window. Added in the review fixes of 7c1, not declared before the
#: comparison: its prediction is the declared emulation's own accumulator line (the angle is a single-precision sum of
#: the commanded step, ``predict_a0.emulate``), and its tolerance the rotor's declared one (one float32 epsilon).
MOTOR_OUTPUT = "rotor_angle_rate_mean_hz"
MOTOR_OUTPUT_TOLERANCE = TOLERANCES["rotor_rate_mean_hz"]
STEP = 5e-4
DURATION = 10.0


def difference(name: str, value: float, against: float) -> float:
    """Relative, except the lag's (rad)."""
    return value - against if name == LAG else (value - against) / abs(against)


@dataclass(frozen=True)
class Row:
    quantity: str
    reference: float
    measured: float
    predicted: float
    tolerance: float

    @property
    def miss(self) -> float:
        return difference(self.quantity, self.measured, self.predicted)

    @property
    def holds(self) -> bool:
        return math.isfinite(self.miss) and abs(self.miss) <= self.tolerance


def measure(key: str, **options: Any) -> dict[str, float]:
    """The steady quantities of 0.3's A0 at ``key``'s frequency (``options`` are ``script_a0.a0``'s), the
    reference's window rule, in the process's precision."""
    return quantities(run(FREQUENCY[key], **options), FREQUENCY[key], SCRIPT_A_FRAME)


@cache
def predicted_motor_output(frequency: float) -> float:
    """The mean rate the motor's angle advances over the window if it adds the commanded step in single precision at
    every step, from an angle of zero: the declared emulation's line for 0.3's motor
    (``angle = float32(angle + float32(command) * float32(dt))``), read as the measurement reads 0.3's angle."""
    command, step = np.float32(2.0 * math.pi * frequency), np.float32(STEP)
    steps = round(DURATION / STEP)
    angle = np.zeros(steps + 1, dtype=np.float32)
    for index in range(steps):
        angle[index + 1] = np.float32(angle[index] + command * step)
    return motor_output_rate(angle, np.arange(steps + 1) * STEP, frequency)


def compare(measured: dict[str, float], key: str, against: str | None = None) -> list[Row]:
    """One row per gated quantity: the measurement against the declared prediction at ``against`` (``key``'s)."""
    predicted = PREDICTED[against or key]
    rows = [Row(name, REFERENCE[key].get(name, math.nan), measured[name], predicted[name], TOLERANCES[name])
            for name in GATED]
    return [*rows, Row(MOTOR_OUTPUT, math.nan, measured[MOTOR_OUTPUT],
                       predicted_motor_output(FREQUENCY[against or key]), MOTOR_OUTPUT_TOLERANCE)]


def table(rows: list[Row]) -> str:
    lines = [f"{'quantity':44s} {'reference':>14s} {'0.3':>14s} {'predicted':>14s} {'miss':>10s} {'tolerance':>9s}"]
    lines += [f"{r.quantity:44s} {r.reference:14.9g} {r.measured:14.9g} {r.predicted:14.9g} {r.miss:+10.2e} "
              f"{r.tolerance:9.1e} {'ok' if r.holds else 'MISS'}" for r in rows]
    return "\n".join(lines)


def missed(rows: list[Row]) -> list[str]:
    return [row.quantity for row in rows if not row.holds]
