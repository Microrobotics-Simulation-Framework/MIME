"""SI quantities for declarations: a number with a dimension, checked when the experiment is planned.

A script writes ``mm(3.175)`` or ``ms(20)``; phase 1 of the build checks each quantity's dimension against what the
setting needs and turns it into a plain SI float. Nothing past phase 1 sees a :class:`Quantity`: node constructors,
``params`` and JAX arrays hold SI floats only.

A declared quantity may carry its reason, beside its value: ``mm(3.175, reason="the tube's datasheet")``, with a text,
``chosen(...)`` or ``pending()`` (:mod:`mime.validity`'s basis, one type for a value's source and a waiver's basis). The
saved form writes it beside the value. Equality and arithmetic ignore it, and a derived quantity carries none.

A run's results come back as quantities too, and these may hold arrays (a series of a run's speeds): read one in the
unit you want with ``.to(mm_per_s)``, which refuses a unit of another dimension. An array quantity is a result, never a
declaration, so phase 1 refuses one.

An entry declares each dimensioned setting's dimension once, in its annotation: ``margin: Annotated[Quantity,
Length]``. The markers (:data:`Length`, :data:`Force` and the rest) are the SI dimensions themselves, so they combine:
``stiffness: Annotated[Quantity, Force / Length]``. A dimensionless setting is a plain number and needs no marker.

Two limits of a check by dimension alone:

- A dimension names one quantity. :data:`TORQUE` (N m) is dimensionally an energy (J), and the names and symbols a
  quantity prints with are keyed by its dimension, so an energy unit added beside it would collide with it there, and
  no check here tells a torque from an energy.
- A command declares one dimension for every component. The dynamic arm's ``joint_torques`` is declared
  :data:`TORQUE` for every joint, so a prismatic joint's force (N) cannot be given in newtons; the kinematic arm's
  ``joint_positions`` is dimensionless the same way, radians and a prismatic joint's metres alike.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal, cast, overload, override

import numpy as np
from numpy.typing import NDArray

from mime.validity._bases import Basis

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
TORQUE = FORCE * LENGTH
ROTATIONAL_DAMPING = TORQUE * TIME

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
    TORQUE: "torque (N m)",
    ROTATIONAL_DAMPING: "rotational damping (N m s)",
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
Torque = TORQUE
RotationalDamping = ROTATIONAL_DAMPING

type Magnitude = float | tuple[float, ...]
#: What an angular rate counts: turns (Hz) or radians (rad/s).
type Angle = Literal["turn", "radian"]
_ANGLE_UNIT: dict[Angle, str] = {"turn": "Hz", "radian": "rad/s"}
#: What a quantity holds: a declaration's magnitude, or a result's array of SI floats (read-only).
type Values = float | tuple[float, ...] | NDArray[np.float64]


class UnitError(ValueError):
    """A quantity of the wrong dimension, or a bare number where a quantity is needed: ``dimension`` is the one needed
    and ``given`` what was given, where :func:`si` knows them."""

    def __init__(self, message: str, *, dimension: Dimension | None = None, given: object = None) -> None:
        super().__init__(message)
        self.dimension = dimension
        self.given = given


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


def _scaled(value: Values, factor: float) -> Values:
    if isinstance(value, tuple):
        return tuple(v * factor for v in value)
    if isinstance(value, np.ndarray):
        return _frozen(value * factor)
    return value * factor


def _frozen(values: NDArray[Any]) -> NDArray[np.float64]:
    """``values`` as a read-only float64 array: a quantity is frozen, so its array is too. The array views an immutable
    copy of the bytes, so nothing that holds it, or its ``base``, can make it writeable again (an array that owns its
    buffer can: ``flags.writeable = True`` undoes the freeze)."""
    held = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(held.tobytes(), dtype=np.float64).reshape(held.shape)


def _array_or_magnitude(value: Values) -> NDArray[np.float64] | float:
    if isinstance(value, tuple):
        return np.asarray(value, dtype=np.float64)
    return value


@dataclass(frozen=True, eq=False)
class Quantity:
    """A value in SI units and its dimension. A vector quantity holds a tuple of components. A result's quantity may
    hold a read-only array of SI floats (:meth:`of`), its last axis a vector's components where it has them; arithmetic
    on one is elementwise, and :meth:`to` reads it in a unit."""

    value: Values
    dimension: Dimension
    #: What an angular rate counts, where the quantity knows: ``"turn"`` for ``Hz`` and ``kHz``, ``"radian"`` for
    #: ``rad_per_s``, which share a dimension and differ by 2 pi. ``None`` where it is not a rate of an angle or is not
    #: known (a result an observation declares only by dimension).
    angle: Angle | None = field(default=None, compare=False)
    #: Where the value comes from, or why it was chosen (``mm(3.175, reason="...")``): equality, hashing and
    #: arithmetic ignore it, and a quantity derived from others carries none.
    reason: Basis | None = field(default=None, compare=False)

    @classmethod
    def of(cls, values: object, dimension: Dimension) -> Quantity:
        """An array quantity: ``values`` in SI, copied into a read-only float64 array."""
        return cls(_frozen(np.asarray(values, dtype=np.float64)), dimension)

    def to(self, unit: Unit) -> Values:
        """The magnitude in ``unit`` (a float, a tuple or an array, as the quantity holds it), or :class:`UnitError`
        when ``unit`` is of another dimension: ``speed.to(mm_per_s)``."""
        if cast(object, unit) is degC:
            raise UnitError("cannot read a quantity in degC: it is an offset unit (0 degC is 273.15 K), so it converts "
                            "by an offset as well as a factor, which a unit here does not hold; read the temperature "
                            "in K and subtract ZERO_CELSIUS (mime.units.ZERO_CELSIUS) yourself")
        if not isinstance(cast(object, unit), Unit):
            raise TypeError(f"to() takes a unit of mime.units, such as mm or mm_per_s, not {unit!r}")
        if unit.dimension != self.dimension:
            raise UnitError(f"cannot read a quantity of {self.dimension} ({self}) in {unit.symbol}, a unit of "
                            f"{unit.dimension}")
        if self.angle is not None and unit.angle is not None and self.angle != unit.angle:
            raise UnitError(f"cannot read {self}, an angular rate in {_ANGLE_UNIT[self.angle]}, in {unit.symbol}: "
                            f"{_ANGLE_UNIT[self.angle]} and {unit.symbol} have the same dimension and differ by 2 pi "
                            f"(1 Hz is 2 pi rad/s), which a conversion here would silently drop; multiply or divide "
                            f"by 2 pi where you mean it")
        if isinstance(self.value, tuple):
            return tuple(v / unit.factor for v in self.value)
        if isinstance(self.value, np.ndarray):
            return _frozen(self.value / unit.factor)
        return self.value / unit.factor

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dimension != other.dimension:
            return False
        if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
            return (isinstance(self.value, np.ndarray) and isinstance(other.value, np.ndarray)
                    and self.value.shape == other.value.shape and bool(np.array_equal(self.value, other.value)))
        return self.value == other.value

    @override
    def __hash__(self) -> int:
        if isinstance(self.value, np.ndarray):
            raise TypeError("an array quantity is not hashable")
        return hash((self.value, self.dimension))

    def _array_with(self, other: Quantity | float, operation: str) -> NDArray[np.float64]:
        a = _array_or_magnitude(self.value)
        b = _array_or_magnitude(other.value if isinstance(other, Quantity) else float(other))
        held = a * b if operation == "*" else a / b if operation == "/" else a + b
        return _frozen(np.asarray(held))

    def __mul__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
                return Quantity(self._array_with(other, "*"), self.dimension * other.dimension)
            if isinstance(self.value, tuple) and isinstance(other.value, tuple):
                raise TypeError("the product of two vector quantities is not defined here")
            if isinstance(other.value, tuple):
                return Quantity(_scaled(other.value, cast(float, self.value)), self.dimension * other.dimension)
            return Quantity(_scaled(self.value, cast(float, other.value)), self.dimension * other.dimension)
        return Quantity(_scaled(self.value, float(other)), self.dimension, self.angle)

    __rmul__ = __mul__

    def __truediv__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
                return Quantity(self._array_with(other, "/"), self.dimension / other.dimension)
            if isinstance(other.value, tuple):
                raise TypeError("division by a vector quantity is not defined")
            return Quantity(_scaled(self.value, 1.0 / other.value), self.dimension / other.dimension)
        return Quantity(_scaled(self.value, 1.0 / float(other)), self.dimension, self.angle)

    def __rtruediv__(self, other: float) -> Quantity:
        if isinstance(self.value, np.ndarray):
            return Quantity(_frozen(float(other) / self.value), DIMENSIONLESS / self.dimension)
        if isinstance(self.value, tuple):
            raise TypeError("division by a vector quantity is not defined")
        return Quantity(float(other) / self.value, DIMENSIONLESS / self.dimension)

    def __neg__(self) -> Quantity:
        return Quantity(_scaled(self.value, -1.0), self.dimension, self.angle)

    def _same_dimension(self, other: object, operation: str) -> Quantity:
        if not isinstance(other, Quantity):
            raise UnitError(f"cannot {operation} a {self.dimension} quantity and the bare number {other!r}")
        if other.dimension != self.dimension:
            raise UnitError(f"cannot {operation} {self.dimension} and {other.dimension}")
        return other

    def __add__(self, other: Quantity) -> Quantity:
        other = self._same_dimension(other, "add")
        angle = self.angle if self.angle == other.angle else None
        if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
            return Quantity(self._array_with(other, "+"), self.dimension, angle)
        if isinstance(self.value, tuple) != isinstance(other.value, tuple):
            raise TypeError("cannot add a scalar and a vector quantity")
        if isinstance(self.value, tuple) and isinstance(other.value, tuple):
            return Quantity(tuple(a + b for a, b in zip(self.value, other.value, strict=True)), self.dimension, angle)
        return Quantity(cast(float, self.value) + cast(float, other.value), self.dimension, angle)

    def __sub__(self, other: Quantity) -> Quantity:
        return self + (-self._same_dimension(other, "subtract"))

    def _scalars(self, other: Quantity) -> tuple[float, float]:
        held = self._same_dimension(other, "compare")
        if isinstance(self.value, np.ndarray) or isinstance(held.value, np.ndarray):
            raise TypeError("array quantities are not ordered: compare what .to(unit) gives")
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
        symbol = _SYMBOLS.get(self.dimension, str(self.dimension))
        if isinstance(self.value, np.ndarray):
            return f"{np.array2string(self.value, threshold=8, precision=6)} {symbol}".rstrip()
        return f"{self.value!r} {symbol}".rstrip()


_SYMBOLS = {
    DIMENSIONLESS: "", LENGTH: "m", MASS: "kg", TIME: "s", CURRENT: "A", TEMPERATURE: "K", FREQUENCY: "Hz",
    VELOCITY: "m/s", ACCELERATION: "m/s^2", FORCE: "N", MAGNETIC_FLUX_DENSITY: "T", VOLUME: "m^3",
    DENSITY: "kg/m^3", MOMENT_OF_INERTIA: "kg m^2", MAGNETIC_MOMENT: "A m^2", DYNAMIC_VISCOSITY: "Pa s",
    TORQUE: "N m", ROTATIONAL_DAMPING: "N m s",
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
        raise UnitError(f"{what} needs units of {dimension}; it was given the bare value {quantity!r}",
                        dimension=dimension, given=quantity)
    if quantity.dimension != dimension:
        raise UnitError(f"{what} needs units of {dimension}; it was given {quantity.dimension} ({quantity})",
                        dimension=dimension, given=quantity)
    if isinstance(quantity.value, np.ndarray):
        raise UnitError(f"{what} is given an array quantity ({quantity}): an array is a run's result, and a "
                        f"declaration takes a number or a tuple of them")
    return quantity.value


@dataclass(frozen=True)
class Unit:
    """A unit: ``mm(3.175)`` makes a quantity of it, held in SI, and ``quantity.to(mm)`` reads one in it. ``factor``
    is the SI value of one of it, ``symbol`` how MIME writes it."""

    symbol: str
    factor: float
    dimension: Dimension
    description: str
    angle: Angle | None = None

    def __call__(self, value: float | Iterable[float], *, reason: Basis | None = None) -> Quantity:
        """A quantity of this unit, with its reason when given (``mm(3.175, reason="the tube's datasheet")``)."""
        return Quantity(_scaled(_magnitude(value), self.factor), self.dimension, self.angle, reason)

    @override
    def __str__(self) -> str:
        return self.symbol


m = Unit("m", 1.0, LENGTH, "Metres.")
cm = Unit("cm", 1e-2, LENGTH, "Centimetres.")
mm = Unit("mm", 1e-3, LENGTH, "Millimetres.")
um = Unit("um", 1e-6, LENGTH, "Micrometres.")
s = Unit("s", 1.0, TIME, "Seconds.")
ms = Unit("ms", 1e-3, TIME, "Milliseconds.")
us = Unit("us", 1e-6, TIME, "Microseconds.")
Hz = Unit("Hz", 1.0, FREQUENCY, "Hertz: turns per second.", "turn")
kHz = Unit("kHz", 1e3, FREQUENCY, "Kilohertz.", "turn")
kg = Unit("kg", 1.0, MASS, "Kilograms.")
g = Unit("g", 1e-3, MASS, "Grams.")
N = Unit("N", 1.0, FORCE, "Newtons.")
A = Unit("A", 1.0, CURRENT, "Amperes.")
K = Unit("K", 1.0, TEMPERATURE, "Kelvin.")
T = Unit("T", 1.0, MAGNETIC_FLUX_DENSITY, "Tesla.")
mT = Unit("mT", 1e-3, MAGNETIC_FLUX_DENSITY, "Millitesla.")
uT = Unit("uT", 1e-6, MAGNETIC_FLUX_DENSITY, "Microtesla.")
m_per_s = Unit("m/s", 1.0, VELOCITY, "Metres per second.")
m_per_s2 = Unit("m/s^2", 1.0, ACCELERATION, "Metres per second squared.")
mm_per_s = Unit("mm/s", 1e-3, VELOCITY, "Millimetres per second.")
um_per_s = Unit("um/s", 1e-6, VELOCITY, "Micrometres per second.")
rad = Unit("rad", 1.0, DIMENSIONLESS, "Radians.")
deg = Unit("deg", math.pi / 180.0, DIMENSIONLESS, "Degrees, held in radians.")
rad_per_s = Unit(
    "rad/s", 1.0, FREQUENCY,
    "Radians per second: an angular velocity. Its dimension is a frequency's, so Hz(1) written for an angular velocity "
    "is 1 rad/s, not one turn per second. A quantity made from it is refused by ``.to(Hz)``, and one made from Hz by "
    "``.to(rad_per_s)``: they differ by 2 pi.", "radian")
m3 = Unit("m^3", 1.0, VOLUME, "Cubic metres.")
mm3 = Unit("mm^3", 1e-9, VOLUME, "Cubic millimetres.")
kg_per_m3 = Unit("kg/m^3", 1.0, DENSITY, "Kilograms per cubic metre.")
kg_m2 = Unit("kg m^2", 1.0, MOMENT_OF_INERTIA, "Kilogram square metres: a moment of inertia.")
A_m2 = Unit("A m^2", 1.0, MAGNETIC_MOMENT, "Ampere square metres: a magnetic moment.")
Pa_s = Unit("Pa s", 1.0, DYNAMIC_VISCOSITY, "Pascal seconds: a dynamic viscosity.")
mPa_s = Unit("mPa s", 1e-3, DYNAMIC_VISCOSITY, "Millipascal seconds (water near 20 degC is about 1).")
N_m = Unit("N m", 1.0, TORQUE, "Newton metres: a torque.")
N_m_s = Unit("N m s", 1.0, ROTATIONAL_DAMPING,
             "Newton metre seconds: a torque per angular velocity, a rotational damping.")
dimensionless = Unit("1", 1.0, DIMENSIONLESS, "A pure number, such as a ratio or a normalised wavenumber.")


#: The kelvin temperature of 0 degC.
ZERO_CELSIUS = 273.15


@dataclass(frozen=True)
class Condition:
    """An operating condition with its tolerance: the value an experiment runs at, and how far from it a record's
    stated condition may lie and still apply (``degC(20, plus_minus=0.5)``). Both in the condition's dimension."""

    value: Quantity
    plus_minus: Quantity

    @override
    def __str__(self) -> str:
        return f"{self.value} +- {self.plus_minus}"


@overload
def degC(value: float, *, reason: Basis | None = None) -> Quantity: ...
@overload
def degC(value: float, *, plus_minus: float, reason: Basis | None = None) -> Condition: ...
def degC(value: float, *, plus_minus: float | None = None, reason: Basis | None = None) -> Quantity | Condition:
    """Degrees Celsius, held in kelvin; with ``plus_minus`` (a difference, in kelvin or degC alike), an operating
    condition with its tolerance. ``reason`` is the temperature's, as on any quantity."""
    celsius = _magnitude(value)
    if isinstance(celsius, tuple):
        raise TypeError(f"a temperature in degC is one number, not {value!r}")
    kelvin = K(celsius + ZERO_CELSIUS, reason=reason)
    if plus_minus is None:
        return kelvin
    return Condition(kelvin, K(plus_minus))


#: The vacuum magnetic permeability, mu_0, in N A^-2 (henry per metre): CODATA 2022's recommended value,
#: 1.25663706127(20)e-6, no longer exactly 4 pi 1e-7 since the 2019 SI.
VACUUM_PERMEABILITY = 1.25663706127e-6


def standard_gravity(*, reason: Basis | None = None) -> Quantity:
    """The standard acceleration of gravity, 9.80665 m/s^2 (a defined constant): an experiment's gravity by convention
    when it states none."""
    return m_per_s2(9.80665, reason=reason)


__all__ = [
    "ACCELERATION", "CURRENT", "DENSITY", "DIMENSIONLESS", "DYNAMIC_VISCOSITY", "FORCE", "FREQUENCY", "LENGTH",
    "MAGNETIC_FLUX_DENSITY", "MAGNETIC_MOMENT", "MASS", "MOMENT_OF_INERTIA", "ROTATIONAL_DAMPING", "TEMPERATURE",
    "TIME", "TORQUE", "VELOCITY", "VOLUME", "Acceleration", "Current", "Density", "DynamicViscosity", "Force",
    "Frequency", "Length", "MagneticFluxDensity", "MagneticMoment", "Mass", "MomentOfInertia", "RotationalDamping",
    "Temperature", "Time", "Torque", "Velocity", "Volume", "Condition", "Dimension", "Quantity", "Unit", "UnitError",
    "VACUUM_PERMEABILITY", "Values", "ZERO_CELSIUS", "parse_unit", "si", "unit_symbol", "A", "A_m2", "Hz", "K", "N",
    "N_m", "N_m_s", "Pa_s", "T", "cm", "deg", "degC", "dimensionless", "g", "kHz", "kg", "kg_m2", "kg_per_m3", "m",
    "m3", "m_per_s", "m_per_s2", "mPa_s", "mT", "mm", "mm3", "mm_per_s", "ms", "rad", "rad_per_s", "s",
    "standard_gravity", "uT", "um", "um_per_s", "us",
]
