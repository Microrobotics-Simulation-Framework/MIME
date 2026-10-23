"""SI quantities for declarations: a number with a dimension, checked when the experiment is planned.

A script writes ``mm(3.175)`` or ``ms(20)``; phase 1 of the build checks each quantity's dimension against what the
setting needs and turns it into a plain SI float. Nothing past phase 1 sees a :class:`Quantity`: node constructors,
``params`` and JAX arrays hold SI floats only.

An entry declares each dimensioned setting's dimension once, in its annotation: ``margin: Annotated[Quantity,
Length]``. The markers (:data:`Length`, :data:`Force` and the rest) are the SI dimensions themselves, so they combine:
``stiffness: Annotated[Quantity, Force / Length]``. A dimensionless setting is a plain number and needs no marker.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import cast, override

_BASE_SYMBOLS = ("m", "kg", "s", "A", "K", "mol", "cd")


@dataclass(frozen=True)
class Dimension:
    """Exponents of the seven SI base units, in the order m, kg, s, A, K, mol, cd."""

    exponents: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)

    @classmethod
    def of(cls, exponents: Iterable[int]) -> Dimension:
        e = tuple(exponents)
        if len(e) != len(_BASE_SYMBOLS):
            raise ValueError(f"a dimension has {len(_BASE_SYMBOLS)} exponents, not {len(e)}")
        return cls((e[0], e[1], e[2], e[3], e[4], e[5], e[6]))

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension.of(a + b for a, b in zip(self.exponents, other.exponents, strict=True))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension.of(a - b for a, b in zip(self.exponents, other.exponents, strict=True))

    def __pow__(self, power: int) -> Dimension:
        return Dimension.of(a * power for a in self.exponents)

    @override
    def __str__(self) -> str:
        if self in _NAMED:
            return _NAMED[self]
        terms = [symbol if power == 1 else f"{symbol}^{power}"
                 for symbol, power in zip(_BASE_SYMBOLS, self.exponents, strict=False) if power]
        return " ".join(terms) or "1"


def _base(index: int) -> Dimension:
    exponents = [0] * 7
    exponents[index] = 1
    return Dimension.of(exponents)


DIMENSIONLESS = Dimension()
LENGTH = _base(0)
MASS = _base(1)
TIME = _base(2)
CURRENT = _base(3)
TEMPERATURE = _base(4)
FREQUENCY = DIMENSIONLESS / TIME
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME**2
FORCE = MASS * ACCELERATION
MAGNETIC_FLUX_DENSITY = MASS / (CURRENT * TIME**2)
VOLUME = LENGTH**3
DENSITY = MASS / VOLUME
MOMENT_OF_INERTIA = MASS * LENGTH**2
MAGNETIC_MOMENT = CURRENT * LENGTH**2
DYNAMIC_VISCOSITY = MASS / (LENGTH * TIME)

_NAMED = {
    DIMENSIONLESS: "dimensionless",
    LENGTH: "length (m)",
    MASS: "mass (kg)",
    TIME: "time (s)",
    CURRENT: "current (A)",
    TEMPERATURE: "temperature (K)",
    FREQUENCY: "frequency (Hz)",
    VELOCITY: "velocity (m/s)",
    ACCELERATION: "acceleration (m/s^2)",
    FORCE: "force (N)",
    MAGNETIC_FLUX_DENSITY: "magnetic flux density (T)",
    VOLUME: "volume (m^3)",
    DENSITY: "density (kg/m^3)",
    MOMENT_OF_INERTIA: "moment of inertia (kg m^2)",
    MAGNETIC_MOMENT: "magnetic moment (A m^2)",
    DYNAMIC_VISCOSITY: "dynamic viscosity (Pa s)",
}

Length = LENGTH
Mass = MASS
Time = TIME
Current = CURRENT
Temperature = TEMPERATURE
Frequency = FREQUENCY
Velocity = VELOCITY
Acceleration = ACCELERATION
Force = FORCE
MagneticFluxDensity = MAGNETIC_FLUX_DENSITY
Volume = VOLUME
Density = DENSITY
MomentOfInertia = MOMENT_OF_INERTIA
MagneticMoment = MAGNETIC_MOMENT
DynamicViscosity = DYNAMIC_VISCOSITY

type Magnitude = float | tuple[float, ...]


class UnitError(ValueError):
    """A quantity of the wrong dimension, or a bare number where a quantity is needed."""


def _real(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real)


def _magnitude(value: object) -> Magnitude:
    """A script's value as SI floats: a real number, or a sequence of them. Checked here, since a script is not
    type-checked."""
    if _real(value):
        return float(cast(float, value))
    if isinstance(value, Iterable) and not isinstance(value, str):
        components = tuple(cast(Iterable[object], value))
        if components and all(_real(c) for c in components):
            return tuple(float(cast(float, c)) for c in components)
    raise TypeError(f"a quantity's value is a real number or a sequence of them, not {value!r}")


def _scaled(value: Magnitude, factor: float) -> Magnitude:
    if isinstance(value, tuple):
        return tuple(v * factor for v in value)
    return value * factor


@dataclass(frozen=True)
class Quantity:
    """A value in SI units and its dimension. A vector quantity holds a tuple of components."""

    value: Magnitude
    dimension: Dimension

    def __mul__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            if isinstance(self.value, tuple) and isinstance(other.value, tuple):
                raise TypeError("the product of two vector quantities is not defined here")
            if isinstance(other.value, tuple):
                return Quantity(_scaled(other.value, cast(float, self.value)), self.dimension * other.dimension)
            return Quantity(_scaled(self.value, cast(float, other.value)), self.dimension * other.dimension)
        return Quantity(_scaled(self.value, float(other)), self.dimension)

    __rmul__ = __mul__

    def __truediv__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            if isinstance(other.value, tuple):
                raise TypeError("division by a vector quantity is not defined")
            return Quantity(_scaled(self.value, 1.0 / other.value), self.dimension / other.dimension)
        return Quantity(_scaled(self.value, 1.0 / float(other)), self.dimension)

    def __rtruediv__(self, other: float) -> Quantity:
        if isinstance(self.value, tuple):
            raise TypeError("division by a vector quantity is not defined")
        return Quantity(float(other) / self.value, DIMENSIONLESS / self.dimension)

    def __neg__(self) -> Quantity:
        return Quantity(_scaled(self.value, -1.0), self.dimension)

    def _same_dimension(self, other: object, operation: str) -> Quantity:
        if not isinstance(other, Quantity):
            raise UnitError(f"cannot {operation} a {self.dimension} quantity and the bare number {other!r}")
        if other.dimension != self.dimension:
            raise UnitError(f"cannot {operation} {self.dimension} and {other.dimension}")
        return other

    def __add__(self, other: Quantity) -> Quantity:
        other = self._same_dimension(other, "add")
        if isinstance(self.value, tuple) != isinstance(other.value, tuple):
            raise TypeError("cannot add a scalar and a vector quantity")
        if isinstance(self.value, tuple) and isinstance(other.value, tuple):
            return Quantity(tuple(a + b for a, b in zip(self.value, other.value, strict=True)), self.dimension)
        return Quantity(cast(float, self.value) + cast(float, other.value), self.dimension)

    def __sub__(self, other: Quantity) -> Quantity:
        return self + (-self._same_dimension(other, "subtract"))

    def _scalars(self, other: Quantity) -> tuple[float, float]:
        held = self._same_dimension(other, "compare")
        if isinstance(self.value, tuple) or isinstance(held.value, tuple):
            raise TypeError("vector quantities are not ordered")
        return self.value, held.value

    def __lt__(self, other: Quantity) -> bool:
        a, b = self._scalars(other)
        return a < b

    def __le__(self, other: Quantity) -> bool:
        a, b = self._scalars(other)
        return a <= b

    def __gt__(self, other: Quantity) -> bool:
        a, b = self._scalars(other)
        return a > b

    def __ge__(self, other: Quantity) -> bool:
        a, b = self._scalars(other)
        return a >= b

    @override
    def __str__(self) -> str:
        return f"{self.value!r} {_SYMBOLS.get(self.dimension, str(self.dimension))}".rstrip()


_SYMBOLS = {
    DIMENSIONLESS: "", LENGTH: "m", MASS: "kg", TIME: "s", CURRENT: "A", TEMPERATURE: "K", FREQUENCY: "Hz",
    VELOCITY: "m/s", ACCELERATION: "m/s^2", FORCE: "N", MAGNETIC_FLUX_DENSITY: "T", VOLUME: "m^3",
    DENSITY: "kg/m^3", MOMENT_OF_INERTIA: "kg m^2", MAGNETIC_MOMENT: "A m^2", DYNAMIC_VISCOSITY: "Pa s",
}


def unit_symbol(dimension: Dimension) -> str:
    """How the saved form writes a dimension: its SI symbol (``m``, ``m/s^2``), ``1`` when dimensionless, and
    otherwise a product of base units (``m^2 kg s^-1``), which :func:`parse_unit` reads back."""
    if dimension == DIMENSIONLESS:
        return "1"
    if dimension in _SYMBOLS:
        return _SYMBOLS[dimension]
    return " ".join(symbol if power == 1 else f"{symbol}^{power}"
                    for symbol, power in zip(_BASE_SYMBOLS, dimension.exponents, strict=True) if power)


def parse_unit(symbol: str) -> Dimension:
    """The dimension :func:`unit_symbol` wrote as ``symbol``, or :class:`UnitError`."""
    if symbol == "1":
        return DIMENSIONLESS
    for dimension, text in _SYMBOLS.items():
        if text and text == symbol:
            return dimension
    exponents = [0] * len(_BASE_SYMBOLS)
    terms = symbol.split()
    for term in terms:
        base, _, power = term.partition("^")
        if base not in _BASE_SYMBOLS or (power and not re.fullmatch(r"-?[1-9][0-9]*", power)):
            raise UnitError(f"{symbol!r} is not a unit MIME writes: an SI symbol such as m, s, N or m/s^2, 1, or "
                            f"base units with integer powers such as 'm^2 kg s^-1'")
        exponents[_BASE_SYMBOLS.index(base)] += int(power) if power else 1
    if not terms:
        raise UnitError("an empty unit; write 1 for a dimensionless value")
    return Dimension.of(exponents)


def si(quantity: object, dimension: Dimension, *, what: str) -> Magnitude:
    """The SI magnitude of ``quantity``, or :class:`UnitError` naming ``what`` when it is not a quantity of
    ``dimension``. A bare number is refused for any dimension but :data:`DIMENSIONLESS`: units are never assumed."""
    if not isinstance(quantity, Quantity):
        if dimension == DIMENSIONLESS:
            return _magnitude(quantity)
        raise UnitError(f"{what} needs units of {dimension}; it was given the bare value {quantity!r}")
    if quantity.dimension != dimension:
        raise UnitError(f"{what} needs units of {dimension}; it was given {quantity.dimension} ({quantity})")
    return quantity.value


def _unit(factor: float, dimension: Dimension, doc: str) -> Callable[[float | Iterable[float]], Quantity]:
    def construct(value: float | Iterable[float]) -> Quantity:
        return Quantity(_scaled(_magnitude(value), factor), dimension)

    construct.__doc__ = doc
    return construct


m = _unit(1.0, LENGTH, "Metres.")
cm = _unit(1e-2, LENGTH, "Centimetres.")
mm = _unit(1e-3, LENGTH, "Millimetres.")
um = _unit(1e-6, LENGTH, "Micrometres.")
s = _unit(1.0, TIME, "Seconds.")
ms = _unit(1e-3, TIME, "Milliseconds.")
us = _unit(1e-6, TIME, "Microseconds.")
Hz = _unit(1.0, FREQUENCY, "Hertz.")
kHz = _unit(1e3, FREQUENCY, "Kilohertz.")
kg = _unit(1.0, MASS, "Kilograms.")
g = _unit(1e-3, MASS, "Grams.")
N = _unit(1.0, FORCE, "Newtons.")
A = _unit(1.0, CURRENT, "Amperes.")
K = _unit(1.0, TEMPERATURE, "Kelvin.")
T = _unit(1.0, MAGNETIC_FLUX_DENSITY, "Tesla.")
mT = _unit(1e-3, MAGNETIC_FLUX_DENSITY, "Millitesla.")
uT = _unit(1e-6, MAGNETIC_FLUX_DENSITY, "Microtesla.")
m_per_s = _unit(1.0, VELOCITY, "Metres per second.")
m_per_s2 = _unit(1.0, ACCELERATION, "Metres per second squared.")
rad = _unit(1.0, DIMENSIONLESS, "Radians.")
deg = _unit(math.pi / 180.0, DIMENSIONLESS, "Degrees, held in radians.")
rad_per_s = _unit(1.0, FREQUENCY, "Radians per second: an angular velocity. Its dimension is a frequency's, so Hz(1) "
                                  "written for an angular velocity is 1 rad/s, not one turn per second.")
m3 = _unit(1.0, VOLUME, "Cubic metres.")
mm3 = _unit(1e-9, VOLUME, "Cubic millimetres.")
kg_per_m3 = _unit(1.0, DENSITY, "Kilograms per cubic metre.")
kg_m2 = _unit(1.0, MOMENT_OF_INERTIA, "Kilogram square metres: a moment of inertia.")
A_m2 = _unit(1.0, MAGNETIC_MOMENT, "Ampere square metres: a magnetic moment.")
Pa_s = _unit(1.0, DYNAMIC_VISCOSITY, "Pascal seconds: a dynamic viscosity.")
mPa_s = _unit(1e-3, DYNAMIC_VISCOSITY, "Millipascal seconds (water near 20 degC is about 1).")
dimensionless = _unit(1.0, DIMENSIONLESS, "A pure number, such as a ratio or a normalised wavenumber.")


def standard_gravity() -> Quantity:
    """The standard acceleration of gravity, 9.80665 m/s^2 (a defined constant)."""
    return m_per_s2(9.80665)


__all__ = [
    "ACCELERATION", "CURRENT", "DENSITY", "DIMENSIONLESS", "DYNAMIC_VISCOSITY", "FORCE", "FREQUENCY", "LENGTH",
    "MAGNETIC_FLUX_DENSITY", "MAGNETIC_MOMENT", "MASS", "MOMENT_OF_INERTIA", "TEMPERATURE", "TIME", "VELOCITY",
    "VOLUME",
    "Acceleration", "Current", "Density", "DynamicViscosity", "Force", "Frequency", "Length", "MagneticFluxDensity",
    "MagneticMoment", "Mass", "MomentOfInertia", "Temperature", "Time", "Velocity", "Volume",
    "Dimension", "Quantity", "UnitError", "parse_unit", "si", "unit_symbol",
    "A", "A_m2", "Hz", "K", "N", "Pa_s", "T", "cm", "deg", "dimensionless", "g", "kHz", "kg", "kg_m2", "kg_per_m3",
    "m", "m3", "m_per_s", "m_per_s2", "mPa_s", "mT", "mm", "mm3", "ms", "rad", "rad_per_s", "s", "standard_gravity",
    "uT", "um", "us",
]
