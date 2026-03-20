"""Standard control primitives for magnetic actuation.

These are time-bounded open-loop commands that compose into
ControlSequences. They produce boundary inputs for an
ExternalMagneticFieldNode (or similar actuation node).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mime.control.policy import ControlPrimitive


@dataclass
class RotateField(ControlPrimitive):
    """Fixed-frequency field rotation.

    Parameters
    ----------
    frequency : float
        Rotation frequency in Hz.
    field_strength : float
        Field magnitude in mT.
    duration : float
        Duration in seconds.
    target_node : str
        Name of the actuation node to command.
    """
    frequency: float = 10.0
    field_strength: float = 10.0
    duration: float = 1.0
    target_node: str = "external_field"

    def external_inputs(
        self, t_local: float, dt: float, observed_state: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency,
            "field_strength_mt": self.field_strength,
        }


@dataclass
class SweepFrequency(ControlPrimitive):
    """Linear frequency ramp — step-out characterisation.

    Ramps frequency linearly from f_start to f_end over duration.
    """
    f_start: float = 10.0
    f_end: float = 50.0
    field_strength: float = 10.0
    duration: float = 5.0
    target_node: str = "external_field"

    def external_inputs(
        self, t_local: float, dt: float, observed_state: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        alpha = min(t_local / self.duration, 1.0) if self.duration > 0 else 1.0
        freq = self.f_start + alpha * (self.f_end - self.f_start)
        return {
            "frequency_hz": freq,
            "field_strength_mt": self.field_strength,
        }


@dataclass
class HoldField(ControlPrimitive):
    """Hold at fixed frequency and field strength."""
    frequency: float = 10.0
    field_strength: float = 10.0
    duration: float = 1.0
    target_node: str = "external_field"

    def external_inputs(
        self, t_local: float, dt: float, observed_state: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency,
            "field_strength_mt": self.field_strength,
        }


@dataclass
class RampField(ControlPrimitive):
    """Ramp field strength from zero — gentle startup.

    Linearly increases field strength from 0 to target over duration
    while maintaining constant frequency.
    """
    frequency: float = 10.0
    target_strength: float = 10.0
    duration: float = 1.0
    target_node: str = "external_field"

    def external_inputs(
        self, t_local: float, dt: float, observed_state: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        alpha = min(t_local / self.duration, 1.0) if self.duration > 0 else 1.0
        return {
            "frequency_hz": self.frequency,
            "field_strength_mt": alpha * self.target_strength,
        }


@dataclass
class StepDown(ControlPrimitive):
    """Drop frequency instantly — step-out recovery.

    Sets frequency to the target value immediately and holds it.
    """
    frequency: float = 5.0
    field_strength: float = 10.0
    duration: float = 1.0
    target_node: str = "external_field"

    def external_inputs(
        self, t_local: float, dt: float, observed_state: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency,
            "field_strength_mt": self.field_strength,
        }
