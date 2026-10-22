"""SI quantities for declarations: a number with a dimension, checked when the experiment is planned.

A script writes ``mm(3.175)`` or ``ms(20)``; phase 1 of the build checks each quantity's dimension against what the
setting needs and turns it into a plain SI float. Nothing past phase 1 sees a :class:`Quantity`: node constructors,
``params`` and JAX arrays hold SI floats only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

_BASE_SYMBOLS = ("m", "kg", "s", "A", "K", "mol", "cd")


@dataclass(frozen=True)
class Dimension:
    """Exponents of the seven SI base units, in the order m, kg, s, A, K, mol, cd."""

    exponents: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents, strict=False)))  # type: ignore[arg-type]

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents, strict=False)))  # type: ignore[arg-type]

    def __pow__(self, power: int) -> Dimension:
        return Dimension(tuple(a * power for a in self.exponents))  # type: ignore[arg-type]

    def __str__(self) -> str:
        if self in _NAMED:
            return _NAMED[self]
        terms = [symbol if power == 1 else f"{symbol}^{power}"
                 for symbol, power in zip(_BASE_SYMBOLS, self.exponents, strict=False) if power]
        return " ".join(terms) or "1"


def _base(index: int) -> Dimension:
    exponents = [0] * 7
    exponents[index] = 1
    return Dimension(tuple(exponents))  # type: ignore[arg-type]


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
}

type Magnitude = float | tuple[float, ...]


class UnitError(ValueError):
    """A quantity of the wrong dimension, or a bare number where a quantity is needed."""


def _magnitude(value: Real | Iterable[Real]) -> Magnitude:
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Iterable) and not isinstance(value, str):
        components = tuple(value)
        if components and all(isinstance(c, Real) and not isinstance(c, bool) for c in components):
            return tuple(float(c) for c in components)
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

    def __mul__(self, other: Quantity | Real) -> Quantity:
        if isinstance(other, Quantity):
            if isinstance(self.value, tuple) and isinstance(other.value, tuple):
                raise TypeError("the product of two vector quantities is not defined here")
            if isinstance(other.value, tuple):
                return Quantity(_scaled(other.value, self.value),  # type: ignore[arg-type]
                                self.dimension * other.dimension)
            return Quantity(_scaled(self.value, other.value), self.dimension * other.dimension)
        return Quantity(_scaled(self.value, float(other)), self.dimension)

    __rmul__ = __mul__

    def __truediv__(self, other: Quantity | Real) -> Quantity:
        if isinstance(other, Quantity):
            if isinstance(other.value, tuple):
                raise TypeError("division by a vector quantity is not defined")
            return Quantity(_scaled(self.value, 1.0 / other.value), self.dimension / other.dimension)
        return Quantity(_scaled(self.value, 1.0 / float(other)), self.dimension)

    def __rtruediv__(self, other: Real) -> Quantity:
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
        if isinstance(self.value, tuple):
            summed = tuple(a + b for a, b in zip(self.value, other.value, strict=True))  # type: ignore[arg-type]
            return Quantity(summed, self.dimension)
        return Quantity(self.value + other.value, self.dimension)  # type: ignore[operator]

    def __sub__(self, other: Quantity) -> Quantity:
        return self + (-self._same_dimension(other, "subtract"))

    def __lt__(self, other: Quantity) -> bool:
        return self.value < self._same_dimension(other, "compare").value  # type: ignore[operator]

    def __le__(self, other: Quantity) -> bool:
        return self.value <= self._same_dimension(other, "compare").value  # type: ignore[operator]

    def __gt__(self, other: Quantity) -> bool:
        return self.value > self._same_dimension(other, "compare").value  # type: ignore[operator]

    def __ge__(self, other: Quantity) -> bool:
        return self.value >= self._same_dimension(other, "compare").value  # type: ignore[operator]

    def __str__(self) -> str:
        return f"{self.value!r} {_SYMBOLS.get(self.dimension, str(self.dimension))}".rstrip()


_SYMBOLS = {
    DIMENSIONLESS: "", LENGTH: "m", MASS: "kg", TIME: "s", CURRENT: "A", TEMPERATURE: "K", FREQUENCY: "Hz",
    VELOCITY: "m/s", ACCELERATION: "m/s^2", FORCE: "N", MAGNETIC_FLUX_DENSITY: "T",
}


def si(quantity: object, dimension: Dimension, *, what: str) -> Magnitude:
    """The SI magnitude of ``quantity``, or :class:`UnitError` naming ``what`` when it is not a quantity of
    ``dimension``. A bare number is refused for any dimension but :data:`DIMENSIONLESS`: units are never assumed."""
    if not isinstance(quantity, Quantity):
        if dimension == DIMENSIONLESS:
            return _magnitude(quantity)  # type: ignore[arg-type]
        raise UnitError(f"{what} needs units of {dimension}; it was given the bare value {quantity!r}")
    if quantity.dimension != dimension:
        raise UnitError(f"{what} needs units of {dimension}; it was given {quantity.dimension} ({quantity})")
    return quantity.value


def _unit(factor: float, dimension: Dimension, doc: str):
    def construct(value: Real | Iterable[Real]) -> Quantity:
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


def standard_gravity() -> Quantity:
    """The standard acceleration of gravity, 9.80665 m/s^2 (a defined constant)."""
    return m_per_s2(9.80665)


__all__ = [
    "ACCELERATION", "CURRENT", "DIMENSIONLESS", "FORCE", "FREQUENCY", "LENGTH", "MAGNETIC_FLUX_DENSITY", "MASS",
    "TEMPERATURE", "TIME", "VELOCITY",
    "Dimension", "Quantity", "UnitError", "si",
    "A", "Hz", "K", "N", "T", "cm", "deg", "g", "kHz", "kg", "m", "m_per_s", "m_per_s2", "mT", "mm", "ms", "rad", "s",
    "standard_gravity", "uT", "um", "us",
]
