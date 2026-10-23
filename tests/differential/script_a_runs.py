"""Script A's reproduction arm, run through 0.3's public API, its history in the form the steady-quantity rule reads,
and the comparison against the declared predictions. Shared by the float64 and float32 tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from script_a import reproduction_arm
from script_a_steady import GATED, REPORTED, SCRIPT_A_FRAME, History, speed_between, steady

from mime.units import Hz, s

HERE = Path(__file__).parent
DECLARED: dict[str, Any] = json.loads((HERE / "script_a_expected_differences.json").read_text())
#: The declaration 0.3 is held to: 4c3's declared changes land one at a time, each declared (``declared_changes``,
#: committed before its measurement) and then made, so the one in force is the last change made.
IN_FORCE = "contact-in-the-solve"
CURRENT: dict[str, Any] = next(entry for entry in DECLARED["declared_changes"] if entry["id"] == IN_FORCE)
PREDICTED: dict[str, dict[str, float]] = CURRENT["predicted"]
REFERENCES = HERE / "references" / "script_a"
REFERENCE = {label: json.loads((REFERENCES / f"{label}_steady.json").read_text())
             for label in ("f64", "f32", "f64_freq11hz")}
DURATION = 5.0
TRANSIENT = "transient_speed_0_to_0.5_s_m_s"
FIELD = "flux_density_magnetic_torque_umr"


def run(frequency: float = 10.0, dt: float = 5e-4) -> History:
    """Script A's reproduction arm for 5 s, with the state at ``t = 0`` before the first step's."""
    built = reproduction_arm(frequency=Hz(frequency), dt=s(dt)).build()
    body, field = built.graph.get_node_state("umr"), built.graph.get_node_state("field.magnetic")
    history = built.run(s(DURATION), history=True).history
    assert history is not None

    def stacked(first: Any, rest: Any) -> np.ndarray:
        return np.concatenate([np.asarray(first)[None], np.asarray(rest)])

    pose = stacked(body["pose"], history["umr"]["pose"])
    twist = stacked(body["twist"], history["umr"]["twist"])
    return History(time=np.arange(len(pose)) * dt, position=pose[:, :3], velocity=twist[:, :3],
                   orientation=pose[:, 3:], angular_velocity=twist[:, 3:],
                   field=stacked(field[FIELD], history["field.magnetic"][FIELD]))


def quantities(history: History, frequency: float) -> dict[str, float]:
    found = steady(history, SCRIPT_A_FRAME, frequency)
    found[TRANSIENT] = speed_between(history, SCRIPT_A_FRAME, 0.0, 0.5)
    return found


def difference(name: str, value: float, against: float) -> float:
    """Relative, except the lag's (rad)."""
    return value - against if name == "phase_lag_field_minus_moment_rad" else (value - against) / abs(against)


@dataclass(frozen=True)
class Row:
    quantity: str
    reference: float
    measured: float
    predicted: float
    tolerance: float

    @property
    def measured_difference(self) -> float:
        return difference(self.quantity, self.measured, self.reference)

    @property
    def predicted_difference(self) -> float:
        return difference(self.quantity, self.predicted, self.reference)

    @property
    def miss(self) -> float:
        return difference(self.quantity, self.measured, self.predicted)

    @property
    def holds(self) -> bool:
        return math.isfinite(self.miss) and abs(self.miss) <= self.tolerance


def compare(measured: dict[str, float], reference: dict[str, float], predicted: dict[str, float],
            tolerances: dict[str, float]) -> list[Row]:
    """One row per quantity the tolerances name: the measured value against the reference moved by the declared
    differences."""
    return [Row(name, reference.get(name, math.nan), measured[name], predicted[name], tolerance)
            for name, tolerance in tolerances.items()]


def table(rows: list[Row]) -> str:
    lines = [f"{'quantity':38s} {'reference':>16s} {'0.3':>16s} {'difference':>11s} {'predicted':>11s} "
             f"{'miss':>10s} {'tolerance':>9s}"]
    lines += [f"{r.quantity:38s} {r.reference:16.10g} {r.measured:16.10g} {r.measured_difference:+11.3e} "
              f"{r.predicted_difference:+11.3e} {r.miss:+10.2e} {r.tolerance:9.0e} {'ok' if r.holds else 'MISS'}"
              for r in rows]
    return "\n".join(lines)


def reported(measured: dict[str, float]) -> str:
    return "\n".join(f"{name}: {measured[name]!r} (reported, not gated: {why})" for name, why in REPORTED.items())


__all__ = ["CURRENT", "DECLARED", "GATED", "IN_FORCE", "PREDICTED", "REFERENCE", "TRANSIENT", "Row", "compare",
           "quantities", "reported", "run", "table"]
