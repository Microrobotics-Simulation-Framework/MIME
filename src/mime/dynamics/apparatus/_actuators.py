"""Actuators: datasheet records of the motors a description's transmissions name, each holding its rotor's moment of
inertia about its axis with the source of that number. A simple transmission of reduction ``N`` reflects ``N^2`` times
that inertia onto its joint as armature (``Arm.Dynamic(actuators={"motor_6": "stepperonline.17hs19-2004s1"})``).

A record is a datasheet's value, typical of the part, not a measurement of any one motor, and says so; a lab registers
its own (:func:`register_actuator`), or an experiment gives a rotor's inertia directly, with its reason."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from mime.things import did_you_mean
from mime.units import MOMENT_OF_INERTIA, Quantity, kg_m2


@dataclass(frozen=True)
class Actuator:
    """A motor as its datasheet gives it: ``name`` (lowercase, ``vendor.part``), its rotor's moment of inertia about
    its axis (``rotor_inertia``, in kg m^2, carrying its source as its reason), and what the record is (``label``)."""

    name: str
    rotor_inertia: Quantity
    label: str

    def __post_init__(self) -> None:
        if self.name != self.name.lower() or "." not in self.name:
            raise ValueError(f"an actuator's name is lowercase, its vendor and its part: 'vendor.part', not "
                             f"{self.name!r}")
        value = self.rotor_inertia.value
        if self.rotor_inertia.dimension != MOMENT_OF_INERTIA or not isinstance(value, float) or \
                not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"actuator {self.name!r}'s rotor_inertia is {self.rotor_inertia}, not one moment of "
                             f"inertia above zero: kg_m2(...)")
        if self.rotor_inertia.reason is None:
            raise ValueError(f"actuator {self.name!r}'s rotor_inertia carries no source: kg_m2(..., reason=\"the "
                             f"datasheet it comes from\")")
        if not self.label.strip():
            raise ValueError(f"actuator {self.name!r} has no label saying what its record is")

    @property
    def inertia(self) -> float:
        """Its rotor's moment of inertia about its axis, kg m^2."""
        return cast(float, self.rotor_inertia.value)


_ACTUATORS: dict[str, Actuator] = {}


def register_actuator(actuator: Actuator) -> Actuator:
    """Make ``actuator`` known by its name. Registering a different record under a name already taken is refused."""
    held = _ACTUATORS.get(actuator.name)
    if held is not None and held != actuator:
        raise ValueError(f"an actuator named {actuator.name!r} is registered already, with another record")
    _ACTUATORS[actuator.name] = actuator
    return actuator


def actuator(name: str) -> Actuator:
    """The actuator registered as ``name``; ``KeyError`` names the candidates."""
    if name not in _ACTUATORS:
        raise KeyError(f"no actuator {name!r} is registered. {did_you_mean(name, _ACTUATORS)}")
    return _ACTUATORS[name]


def registered_actuators() -> tuple[Actuator, ...]:
    return tuple(_ACTUATORS[name] for name in sorted(_ACTUATORS))


#: A NEMA 17 hybrid stepper of 0.59 N m holding torque, from the vendor's drawing.
STEPPERONLINE_17HS19_2004S1 = register_actuator(Actuator(
    "stepperonline.17hs19-2004s1",
    kg_m2(82.0e-7, reason="StepperOnline's drawing of the 17HS19-2004S1 (NEMA 17 bipolar stepper, 0.59 N m holding "
                          "torque, 2.0 A a phase), DWG A0842 rev 0 dated 8.1.2025: rotor inertia 82.00 g cm^2"),
    "the datasheet's typical value for the motor's rotor, not a measurement of any one motor; a gearbox fitted to it "
    "adds its own inertia, which this record does not hold"))


__all__ = ["STEPPERONLINE_17HS19_2004S1", "Actuator", "actuator", "register_actuator", "registered_actuators"]
