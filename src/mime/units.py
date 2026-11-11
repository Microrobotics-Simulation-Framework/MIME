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

A frequency and an angular velocity share a dimension, a rate per second, and differ by 2 pi: ``Hz(1)`` is one turn
a second, ``rad_per_s(1)`` one radian. A quantity knows which it counts (``Quantity.angle``), and so does a site that
takes one: :data:`FREQUENCY` (``Frequency``) counts turns, in Hz, and :data:`ANGULAR_VELOCITY` (``AngularVelocity``)
radians, in rad/s. :func:`si` refuses a quantity of the other kind with the conversion to write, and an entry that
declares a rate per second that says neither (``DIMENSIONLESS / TIME``) is refused at registration.

Arithmetic keeps what a quantity counts. A number or a quantity that counts nothing multiplies or divides it and leaves
it counting turns or radians: ``Hz(5.0) * (2 * pi)`` is 31.4 Hz, never rad/s, so write ``rad_per_s(2 * pi * 5.0)`` for
an angular velocity. A sum, a difference or an order of a turn and a radian is refused, as ``.to`` refuses one, and
they are never equal; so is their product or ratio. ``rad`` and ``deg`` count radians, so ``rad(2 * pi) / s(1)`` is an
angular velocity, and a rate times a time counts what the rate counts: ``Hz(5.0) * s(1.0)`` is five turns, which a
site that takes an angle in radians refuses. A quantity that knows neither (a result, or one loaded from a saved
folder) takes what the other operand counts.

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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, ClassVar, Literal, cast, overload, override

import numpy as np
from numpy.typing import NDArray

from mime.validity._bases import Basis

_BASE_SYMBOLS = ("m", "kg", "s", "A", "K", "mol", "cd")
#: What an angular rate counts: turns (Hz) or radians (rad/s).
type Angle = Literal["turn", "radian"]
_ANGLE_UNIT: dict[Angle, str] = {"turn": "Hz", "radian": "rad/s"}


@dataclass(frozen=True, eq=False)
class Dimension:
    """Exponents of the seven SI base units, in the order m, kg, s, A, K, mol, cd, which equality and hashing compare
    alone. ``angle`` is what a rate per second counts where a site declares it: ``"turn"`` for :data:`FREQUENCY`,
    ``"radian"`` for :data:`ANGULAR_VELOCITY`, ``None`` for any other dimension; :func:`si` checks it against the
    quantity's own. Each of the two is an instance of its own class: ``typing`` keeps one ``Annotated[Quantity, ...]``
    for markers that are equal and of one type, so a marker's angle survives only as its type."""

    exponents: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    angle: ClassVar[Angle | None] = None

    @classmethod
    def of(cls, exponents: Iterable[int]) -> Dimension:
        e = tuple(exponents)
        if len(e) != len(_BASE_SYMBOLS):
            raise ValueError(f"a dimension has {len(_BASE_SYMBOLS)} exponents, not {len(e)}")
        return Dimension((e[0], e[1], e[2], e[3], e[4], e[5], e[6]))

    @override
    def __eq__(self, other: object) -> bool:
        return self.exponents == other.exponents if isinstance(other, Dimension) else NotImplemented

    @override
    def __hash__(self) -> int:
        return hash(self.exponents)

    @override
    def __repr__(self) -> str:
        return f"Dimension(exponents={self.exponents!r})"

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension.of(a + b for a, b in zip(self.exponents, other.exponents, strict=True))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension.of(a - b for a, b in zip(self.exponents, other.exponents, strict=True))

    def __pow__(self, power: int) -> Dimension:
        return Dimension.of(a * power for a in self.exponents)

    @override
    def __str__(self) -> str:
        if self.angle == "radian":
            return "angular velocity (rad/s)"
        if self in _NAMED:
            return _NAMED[self]
        terms = [symbol if power == 1 else f"{symbol}^{power}"
                 for symbol, power in zip(_BASE_SYMBOLS, self.exponents, strict=False) if power]
        return " ".join(terms) or "1"


class _Turns(Dimension):
    """A rate per second that counts turns: :data:`FREQUENCY`'s class."""

    angle = "turn"


class _Radians(Dimension):
    """A rate per second that counts radians: :data:`ANGULAR_VELOCITY`'s class."""

    angle = "radian"


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
#: A frequency, in Hz: turns, cycles or events per second. A site declared with it refuses an angular velocity.
FREQUENCY: Dimension = _Turns((DIMENSIONLESS / TIME).exponents)
#: An angular velocity, in rad/s: radians per second, the dimension of a frequency. A site declared with it refuses a
#: frequency in Hz.
ANGULAR_VELOCITY: Dimension = _Radians(FREQUENCY.exponents)
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME**2
FORCE = MASS * ACCELERATION
MAGNETIC_FLUX_DENSITY = MASS / (CURRENT * TIME**2)
#: A magnetic field's gradient, ``dB_i/dx_j``, in T/m.
FIELD_GRADIENT = MAGNETIC_FLUX_DENSITY / LENGTH
#: An area, and a porous material's Darcy permeability (m^2).
AREA = LENGTH**2
VOLUME = LENGTH**3
DENSITY = MASS / VOLUME
MOMENT_OF_INERTIA = MASS * LENGTH**2
MAGNETIC_MOMENT = CURRENT * LENGTH**2
DYNAMIC_VISCOSITY = MASS / (LENGTH * TIME)
TORQUE = FORCE * LENGTH
ROTATIONAL_DAMPING = TORQUE * TIME
PRESSURE = FORCE / LENGTH**2
VOLUME_FLUX = VOLUME / TIME
HYDRAULIC_RESISTANCE = PRESSURE * TIME / VOLUME
COMPLIANCE = VOLUME / PRESSURE

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
    FIELD_GRADIENT: "magnetic field gradient (T/m)",
    AREA: "area (m^2)",
    VOLUME: "volume (m^3)",
    DENSITY: "density (kg/m^3)",
    MOMENT_OF_INERTIA: "moment of inertia (kg m^2)",
    MAGNETIC_MOMENT: "magnetic moment (A m^2)",
    DYNAMIC_VISCOSITY: "dynamic viscosity (Pa s)",
    TORQUE: "torque (N m)",
    ROTATIONAL_DAMPING: "rotational damping (N m s)",
    PRESSURE: "pressure (Pa)",
    VOLUME_FLUX: "volume flux (m^3/s)",
    HYDRAULIC_RESISTANCE: "hydraulic resistance (Pa s/m^3)",
    COMPLIANCE: "compliance (m^3/Pa)",
}

Length = LENGTH
Mass = MASS
Time = TIME
Current = CURRENT
Temperature = TEMPERATURE
Frequency = FREQUENCY
AngularVelocity = ANGULAR_VELOCITY
Velocity = VELOCITY
Acceleration = ACCELERATION
Force = FORCE
MagneticFluxDensity = MAGNETIC_FLUX_DENSITY
FieldGradient = FIELD_GRADIENT
Area = AREA
Volume = VOLUME
Density = DENSITY
MomentOfInertia = MOMENT_OF_INERTIA
MagneticMoment = MAGNETIC_MOMENT
DynamicViscosity = DYNAMIC_VISCOSITY
Torque = TORQUE
RotationalDamping = ROTATIONAL_DAMPING
Pressure = PRESSURE
VolumeFlux = VOLUME_FLUX
HydraulicResistance = HYDRAULIC_RESISTANCE
Compliance = COMPLIANCE

type Magnitude = float | tuple[float, ...]
#: What a quantity holds: a declaration's magnitude, or a result's array of SI floats (read-only).
type Values = float | tuple[float, ...] | NDArray[np.float64]


class UnitError(ValueError):
    """A quantity of the wrong dimension, or a bare number where a quantity is needed: ``dimension`` is the one needed
    and ``given`` what was given, where :func:`si` knows them. ``alternatives`` are the quantities to write in its place
    where the error knows them, each with when it is the one (an angular rate of the other kind: ``Hz(5.0)`` if turns
    per second were meant), which :meth:`instead` puts in a site's own line."""

    def __init__(self, message: str, *, dimension: Dimension | None = None, given: object = None,
                 alternatives: tuple[tuple[str, str], ...] = ()) -> None:
        super().__init__(message)
        self.dimension = dimension
        self.given = given
        self.alternatives = alternatives

    def instead(self, line: Callable[[str], str]) -> str:
        """What to write instead, after a colon, each alternative quantity in the site's ``line`` (``lambda q:
        f"State(spin={q})"``), for a site's action to end with; empty where the error knows no alternative."""
        if not self.alternatives:
            return ""
        return ": " + ", or ".join(f"{line(quantity)} {when}" for quantity, when in self.alternatives)


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
    #: The number a script's arithmetic multiplied the quantity by since its unit made it (``Hz(5.0) * (2 * pi)`` holds
    #: 2 pi), so that a refusal tells a frequency multiplied by 2 pi, which stays in Hz, from one written as it is.
    scale: float = field(default=1.0, compare=False, repr=False)

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
            kind = f"an angular rate in {_ANGLE_UNIT[self.angle]}" if self.dimension == FREQUENCY else \
                f"which counts {_COUNTS[self.angle]}"
            raise UnitError(f"cannot read {self}, {kind}, in {unit.symbol}: {_ANGLE_UNIT[self.angle]} and "
                            f"{_ANGLE_UNIT[unit.angle]} have the same dimension and differ by 2 pi (1 Hz is 2 pi "
                            f"rad/s), which a conversion here would silently drop; multiply or divide by 2 pi where "
                            f"you mean it")
        if isinstance(self.value, tuple):
            return tuple(v / unit.factor for v in self.value)
        if isinstance(self.value, np.ndarray):
            return _frozen(self.value / unit.factor)
        return self.value / unit.factor

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dimension != other.dimension or _differ(self, other):
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
            angle, scale = _product_angle(self, other, "multiply"), _factor(self) * _factor(other)
            dimension = self.dimension * other.dimension
            if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
                return Quantity(self._array_with(other, "*"), dimension, angle, scale=scale)
            if isinstance(self.value, tuple) and isinstance(other.value, tuple):
                raise TypeError("the product of two vector quantities is not defined here")
            if isinstance(other.value, tuple):
                return Quantity(_scaled(other.value, cast(float, self.value)), dimension, angle, scale=scale)
            return Quantity(_scaled(self.value, cast(float, other.value)), dimension, angle, scale=scale)
        return Quantity(_scaled(self.value, float(other)), self.dimension, self.angle, scale=self.scale * float(other))

    __rmul__ = __mul__

    def __truediv__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            angle, scale = _product_angle(self, other, "divide"), _factor(self) / _factor(other)
            dimension = self.dimension / other.dimension
            if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
                return Quantity(self._array_with(other, "/"), dimension, angle, scale=scale)
            if isinstance(other.value, tuple):
                raise TypeError("division by a vector quantity is not defined")
            return Quantity(_scaled(self.value, 1.0 / other.value), dimension, angle, scale=scale)
        return Quantity(_scaled(self.value, 1.0 / float(other)), self.dimension, self.angle,
                        scale=self.scale / float(other))

    def __rtruediv__(self, other: float) -> Quantity:
        scale = float(other) / self.scale
        if isinstance(self.value, np.ndarray):
            return Quantity(_frozen(float(other) / self.value), DIMENSIONLESS / self.dimension, self.angle,
                            scale=scale)
        if isinstance(self.value, tuple):
            raise TypeError("division by a vector quantity is not defined")
        return Quantity(float(other) / self.value, DIMENSIONLESS / self.dimension, self.angle, scale=scale)

    def __neg__(self) -> Quantity:
        return Quantity(_scaled(self.value, -1.0), self.dimension, self.angle, scale=self.scale)

    def _same_dimension(self, other: object, operation: str) -> Quantity:
        if not isinstance(other, Quantity):
            raise UnitError(f"cannot {operation} a {self.dimension} quantity and the bare number {other!r}")
        if other.dimension != self.dimension:
            raise UnitError(f"cannot {operation} {self.dimension} and {other.dimension}")
        return other

    def __add__(self, other: Quantity) -> Quantity:
        return self._sum(self._same_dimension(other, "add"), "add")

    def __sub__(self, other: Quantity) -> Quantity:
        held = self._same_dimension(other, "subtract")
        _sum_angle(self, held, "subtract")
        return self._sum(-held, "subtract")

    def _sum(self, other: Quantity, operation: str) -> Quantity:
        angle = _sum_angle(self, other, operation)
        scale = self.scale if self.scale == other.scale else 1.0
        if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
            return Quantity(self._array_with(other, "+"), self.dimension, angle, scale=scale)
        if isinstance(self.value, tuple) != isinstance(other.value, tuple):
            raise TypeError("cannot add a scalar and a vector quantity")
        if isinstance(self.value, tuple) and isinstance(other.value, tuple):
            return Quantity(tuple(a + b for a, b in zip(self.value, other.value, strict=True)), self.dimension, angle,
                            scale=scale)
        return Quantity(cast(float, self.value) + cast(float, other.value), self.dimension, angle, scale=scale)

    def _scalars(self, other: Quantity) -> tuple[float, float]:
        held = self._same_dimension(other, "compare")
        _sum_angle(self, held, "compare")
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
        symbol = _ANGLE_UNIT[self.angle] if self.angle is not None and self.dimension == FREQUENCY else \
            _SYMBOLS.get(self.dimension, str(self.dimension))
        if isinstance(self.value, np.ndarray):
            return f"{np.array2string(self.value, threshold=8, precision=6)} {symbol}".rstrip()
        return f"{self.value!r} {symbol}".rstrip()


_COUNTS: dict[Angle, str] = {"turn": "turns", "radian": "radians"}


def _differ(a: Quantity, b: Quantity) -> bool:
    """Whether ``a`` and ``b`` both know what they count, and count different things (a turn and a radian)."""
    return a.angle is not None and b.angle is not None and a.angle != b.angle


def _sum_angle(a: Quantity, b: Quantity, operation: str) -> Angle | None:
    """What a sum, difference or order of ``a`` and ``b`` counts: what either knows, or :class:`UnitError` where they
    know different things."""
    if _differ(a, b):
        raise UnitError(f"cannot {operation} {a} and {b}: one counts {_COUNTS[cast(Angle, a.angle)]} and the other "
                        f"{_COUNTS[cast(Angle, b.angle)]}, which share a dimension and differ by 2 pi (1 Hz is 2 pi "
                        f"rad/s), which MIME does not drop silently; write both in one unit, multiplying or dividing "
                        f"one by 2 pi where you mean it", dimension=a.dimension, given=b)
    return a.angle if a.angle is not None else b.angle


def _product_angle(a: Quantity, b: Quantity, operation: str) -> Angle | None:
    """What a product or ratio of ``a`` and ``b`` counts: what the one that knows counts, nothing for a ratio of two
    that count the same (turns over turns), or :class:`UnitError` where both know (a product would count its angle
    twice, and a ratio of a turn and a radian is off by 2 pi)."""
    if a.angle is None or b.angle is None:
        return a.angle if a.angle is not None else b.angle
    if operation == "divide" and a.angle == b.angle:
        return None
    raise UnitError(f"cannot {operation} {a} by {b}: both count an angle ({_COUNTS[a.angle]} and {_COUNTS[b.angle]}), "
                    f"and a quantity here counts one at most; {operation} their magnitudes, each read with .to(...), "
                    f"where you mean it", dimension=a.dimension, given=b)


def _factor(quantity: Quantity) -> float:
    """What ``quantity`` adds to a product's :attr:`Quantity.scale`: its value where it is a number that counts
    nothing (``dimensionless(2 * pi)``), and its own scale otherwise."""
    if quantity.dimension == DIMENSIONLESS and quantity.angle is None and _real(quantity.value):
        return float(cast(float, quantity.value))
    return quantity.scale


def _times_two_pi(scale: float) -> int:
    """How many times arithmetic multiplied a quantity by 2 pi (``-1`` divided), from its :attr:`Quantity.scale`;
    zero where the scale is no power of 2 pi."""
    if scale == 0.0 or not math.isfinite(scale):
        return 0
    power = round(math.log(abs(scale)) / math.log(2.0 * math.pi))
    return power if power and math.isclose(abs(scale), (2.0 * math.pi) ** power, rel_tol=1e-9) else 0


_SYMBOLS = {
    DIMENSIONLESS: "", LENGTH: "m", MASS: "kg", TIME: "s", CURRENT: "A", TEMPERATURE: "K", FREQUENCY: "Hz",
    VELOCITY: "m/s", ACCELERATION: "m/s^2", FORCE: "N", MAGNETIC_FLUX_DENSITY: "T", AREA: "m^2", VOLUME: "m^3",
    DENSITY: "kg/m^3", MOMENT_OF_INERTIA: "kg m^2", MAGNETIC_MOMENT: "A m^2", DYNAMIC_VISCOSITY: "Pa s",
    TORQUE: "N m", ROTATIONAL_DAMPING: "N m s", PRESSURE: "Pa", VOLUME_FLUX: "m^3/s",
    HYDRAULIC_RESISTANCE: "Pa s/m^3", COMPLIANCE: "m^3/Pa",
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


def _number(value: float) -> str:
    return repr(float(value))


def _rate_written(value: Magnitude, unit: str, scale: str) -> str:
    """``value`` (a rate's magnitude) written in ``unit`` with each component not zero scaled by ``scale``, the
    conversion as an expression (``2 * pi * 5.0``), or as it is where ``scale`` is empty."""
    def one(component: float) -> str:
        if not scale or component == 0.0:
            return _number(component)
        return f"2 * pi * {_number(component)}" if scale == "*" else f"{_number(component)} / (2 * pi)"
    if isinstance(value, tuple):
        return f"{unit}(({', '.join(one(c) for c in value)}{',' if len(value) == 1 else ''}))"
    return f"{unit}({one(value)})"


def _other_angle(quantity: Quantity, dimension: Dimension, what: str) -> UnitError:
    """The refusal of an angular rate of the other kind than the site counts, with the conversions to write. Where
    arithmetic already multiplied a frequency by 2 pi (or divided an angular velocity by it), the number is the one
    meant, in the other unit, and that is the one line; otherwise both readings of the number written."""
    value = cast(Magnitude, quantity.value)
    turned = _times_two_pi(quantity.scale)
    if dimension.angle == "radian":
        given = "a frequency, in turns per second"
        if turned == 1:
            given += (", multiplied by 2 pi after it was written in Hz, which a number leaves counting turns "
                      "(Hz(5.0) * (2 * pi) is 31.4 Hz)")
            alternatives = ((_rate_written(value, "rad_per_s", ""),
                             "for the frequency in Hz multiplied by 2 pi, in radians per second"),)
        else:
            alternatives = ((_rate_written(value, "rad_per_s", "*"), "for the turns per second written (pi from math)"),
                            (_rate_written(value, "rad_per_s", ""), "if you meant radians per second"))
    else:
        given = "an angular velocity, in radians per second"
        if turned == -1:
            given += (", divided by 2 pi after it was written in rad/s, which a number leaves counting radians "
                      "(rad_per_s(31.4) / (2 * pi) is 5.0 rad/s)")
            alternatives = ((_rate_written(value, "Hz", ""),
                             "for the angular velocity in rad/s divided by 2 pi, in turns per second"),)
        else:
            alternatives = ((_rate_written(value, "Hz", ""), "if you meant turns per second"),
                            (_rate_written(value, "Hz", "/"), "for the radians per second written (pi from math)"))
    return UnitError(f"{what} needs units of {dimension}; it was given {quantity}, {given}: the two share a dimension "
                     f"and differ by 2 pi (1 Hz is 2 pi rad/s), which MIME does not assume either way",
                     dimension=dimension, given=quantity, alternatives=alternatives)


def _turns_counted(quantity: Quantity, what: str) -> UnitError:
    """The refusal of a number of turns (a frequency in Hz times a time) where a number goes, which counts an angle in
    radians, with the conversions to write."""
    value = cast(Magnitude, quantity.value)
    if _times_two_pi(quantity.scale) == 1:
        alternatives = ((_rate_written(value, "rad", ""), "for the turns multiplied by 2 pi, in radians"),)
        how = ", multiplied by 2 pi, which a number leaves counting turns"
    else:
        alternatives = ((_rate_written(value, "rad", "*"), "for the turns written (pi from math)"),
                        (_rate_written(value, "rad", ""), "if you meant radians"))
        how = ""
    return UnitError(f"{what} takes a number, which MIME counts in radians where it is an angle; it was given "
                     f"{_magnitude_text(value)} turns (a frequency in Hz times a time{how}): a turn is 2 pi radians, "
                     f"which MIME does not assume", dimension=DIMENSIONLESS, given=quantity, alternatives=alternatives)


def _magnitude_text(value: Magnitude) -> str:
    return f"({', '.join(_number(c) for c in value)})" if isinstance(value, tuple) else _number(value)


def si(quantity: object, dimension: Dimension, *, what: str) -> Magnitude:
    """The SI magnitude of ``quantity``, or :class:`UnitError` naming ``what`` when it is not a quantity of
    ``dimension``. A bare number is refused for any dimension but :data:`DIMENSIONLESS`: units are never assumed. An
    angular rate of the other kind than ``dimension`` counts (a frequency in Hz where an angular velocity goes, or one
    in rad/s where a frequency goes) is refused, with the conversions to write as its ``alternatives``, and so is a
    number of turns where a number goes (``Hz(5.0) * s(1.0)``), which MIME counts in radians; a quantity that does not
    know what it counts (a result, or one loaded from a saved folder) is taken as the site counts."""
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
    if dimension.angle is not None and quantity.angle is not None and quantity.angle != dimension.angle:
        raise _other_angle(quantity, dimension, what)
    if dimension == DIMENSIONLESS and quantity.angle == "turn":
        raise _turns_counted(quantity, what)
    return quantity.value


def rate_symbol(dimension: Dimension) -> str:
    """The unit a site declared with ``dimension`` takes its numbers in: as :func:`unit_symbol`, with rad/s for an
    angular velocity, which the saved form writes as Hz (the same SI number)."""
    return "rad/s" if dimension == FREQUENCY and dimension.angle == "radian" else unit_symbol(dimension)


def says_its_angle(dimension: Dimension) -> bool:
    """Whether a site declared with ``dimension`` says what it counts: any dimension but a rate per second, and a rate
    per second only as :data:`FREQUENCY` or :data:`ANGULAR_VELOCITY`."""
    return dimension != FREQUENCY or dimension.angle is not None


#: What a site declared with a rate per second that says neither is told to write instead.
SAY_THE_ANGLE = ("say which rate it is: mime.units.Frequency for a frequency in Hz (turns per second), or "
                 "mime.units.AngularVelocity for an angular velocity in rad/s; the two differ by 2 pi")


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
T_per_m = Unit("T/m", 1.0, FIELD_GRADIENT, "Tesla per metre: a magnetic field's gradient.")
mT_per_m = Unit("mT/m", 1e-3, FIELD_GRADIENT, "Millitesla per metre: an electromagnetic navigation system's gradients.")
m_per_s = Unit("m/s", 1.0, VELOCITY, "Metres per second.")
m_per_s2 = Unit("m/s^2", 1.0, ACCELERATION, "Metres per second squared.")
mm_per_s = Unit("mm/s", 1e-3, VELOCITY, "Millimetres per second.")
um_per_s = Unit("um/s", 1e-6, VELOCITY, "Micrometres per second.")
rad = Unit("rad", 1.0, DIMENSIONLESS, "Radians.", "radian")
deg = Unit("deg", math.pi / 180.0, DIMENSIONLESS, "Degrees, held in radians.", "radian")
rad_per_s = Unit(
    "rad/s", 1.0, ANGULAR_VELOCITY,
    "Radians per second: an angular velocity. Its dimension is a frequency's, and they differ by 2 pi: a quantity made "
    "from it is refused where a frequency goes (a field's turning frequency) and by ``.to(Hz)``, and one made from Hz "
    "where an angular velocity goes (a body's initial spin) and by ``.to(rad_per_s)``.", "radian")
m2 = Unit("m^2", 1.0, AREA, "Square metres: an area, or a porous material's Darcy permeability.")
um2 = Unit("um^2", 1e-12, AREA, "Square micrometres: a fibrin clot's Darcy permeability is about 1e-3 to 1.")
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
Pa = Unit("Pa", 1.0, PRESSURE, "Pascals: a pressure.")
m3_per_s = Unit("m^3/s", 1.0, VOLUME_FLUX, "Cubic metres per second: a volume flux.")
mL_per_s = Unit("mL/s", 1e-6, VOLUME_FLUX, "Millilitres per second: a vessel's volume flux.")
Pa_s_per_m3 = Unit("Pa s/m^3", 1.0, HYDRAULIC_RESISTANCE, "Pascal seconds per cubic metre: a hydraulic resistance.")
m3_per_Pa = Unit("m^3/Pa", 1.0, COMPLIANCE, "Cubic metres per pascal: a vessel's or a Windkessel's compliance.")
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
    "ACCELERATION", "ANGULAR_VELOCITY", "AREA", "AngularVelocity", "Area", "COMPLIANCE", "Compliance", "HYDRAULIC_RESISTANCE",
    "HydraulicResistance", "PRESSURE", "Pressure", "VOLUME_FLUX", "VolumeFlux", "Pa", "Pa_s_per_m3", "m3_per_Pa",
    "m3_per_s", "mL_per_s", "SAY_THE_ANGLE", "rate_symbol", "says_its_angle", "CURRENT",
    "DENSITY", "DIMENSIONLESS", "DYNAMIC_VISCOSITY", "FIELD_GRADIENT", "FORCE", "FREQUENCY", "LENGTH",
    "MAGNETIC_FLUX_DENSITY", "MAGNETIC_MOMENT", "MASS", "MOMENT_OF_INERTIA", "ROTATIONAL_DAMPING", "TEMPERATURE",
    "TIME", "TORQUE", "VELOCITY", "VOLUME", "Acceleration", "Current", "Density", "DynamicViscosity", "FieldGradient",
    "Force", "Frequency", "Length", "MagneticFluxDensity", "MagneticMoment", "Mass", "MomentOfInertia",
    "RotationalDamping", "Temperature", "Time", "Torque", "Velocity", "Volume", "Condition", "Dimension", "Quantity",
    "Unit", "UnitError", "VACUUM_PERMEABILITY", "Values", "ZERO_CELSIUS", "parse_unit", "si", "unit_symbol", "A",
    "A_m2", "Hz", "K", "N", "N_m", "N_m_s", "Pa_s", "T", "T_per_m", "cm", "deg", "degC", "dimensionless", "g", "kHz",
    "kg", "kg_m2", "kg_per_m3", "m", "m2", "m3", "m_per_s", "m_per_s2", "mPa_s", "mT", "mT_per_m", "mm", "mm3", "mm_per_s",
    "ms", "rad", "rad_per_s", "s", "standard_gravity", "uT", "um", "um2", "um_per_s", "us",
]
