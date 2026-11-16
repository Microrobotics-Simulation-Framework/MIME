"""SI quantities for declarations: a number with a dimension, checked when the experiment is planned.

A script writes ``mm(3.175)`` or ``ms(20)``; phase 1 of the build checks each quantity's dimension against what the
setting needs and turns it into a plain SI float. Nothing past phase 1 sees a :class:`Quantity`: node constructors,
``params`` and JAX arrays hold SI floats only.

A declared quantity may carry its reason, beside its value: ``mm(3.175, reason="the tube's datasheet")``, with a text,
``chosen(...)`` or ``pending()`` (:mod:`mime.validity`'s basis, one type for a value's source and a waiver's basis). The
saved form writes it beside the value. Equality and arithmetic ignore it, and a derived quantity carries none.

A run's results come back as quantities too, and these may hold arrays (a series of a run's speeds): read one in the
unit you want with ``.to(mm_per_s)`` (or ``.to("mm/s")``, a unit written as text), which refuses a unit of another
dimension. An array quantity is a result, never a
declaration, so phase 1 refuses one.

An entry declares each dimensioned setting's dimension once, in its annotation: ``margin: Annotated[Quantity,
Length]``. The markers (:data:`Length`, :data:`Force` and the rest) are the SI dimensions themselves, so they combine:
``stiffness: Annotated[Quantity, Force / Length]``. A dimensionless setting is a plain number and needs no marker.
An angle is the marker :data:`Angle` (radians, written ``deg(30)`` or ``rad(0.52)``): ``half_angle: Annotated[Quantity,
Angle]`` refuses a bare number, a length and a number of turns, where a dimensionless marker would take the bare number.
Registration refuses a setting that holds a ``Quantity`` in a form it does not read, never leaves it out.

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

import difflib
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, ClassVar, Literal, cast, overload, override

import numpy as np
from numpy.typing import NDArray

from mime._refusals import ArgumentRefused, Refusal
from mime.validity._bases import Basis

_BASE_SYMBOLS = ("m", "kg", "s", "A", "K", "mol", "cd")
#: The operator a NumPy ufunc with a quantity on its right stands for, as the method Python would have called on the
#: quantity (``numpy_scalar * quantity`` is ``quantity.__rmul__(numpy_scalar)``), by the ufunc's name.
_REFLECTED = {"add": "__radd__", "subtract": "__rsub__", "multiply": "__rmul__", "divide": "__rtruediv__",
              "less": "__gt__", "less_equal": "__ge__", "greater": "__lt__", "greater_equal": "__le__",
              "equal": "__eq__", "not_equal": "__ne__"}
#: What an angular rate counts: turns (Hz) or radians (rad/s).
type AngleCount = Literal["turn", "radian"]
_ANGLE_UNIT: dict[AngleCount, str] = {"turn": "Hz", "radian": "rad/s"}
_ANGLE_UNIT_NAME: dict[AngleCount, str] = {"turn": "Hz", "radian": "rad_per_s"}


@dataclass(frozen=True, eq=False)
class Dimension:
    """Exponents of the seven SI base units, in the order m, kg, s, A, K, mol, cd, which equality and hashing compare
    alone. ``angle`` is what a rate per second counts where a site declares it: ``"turn"`` for :data:`FREQUENCY`,
    ``"radian"`` for :data:`ANGULAR_VELOCITY`, ``None`` for any other dimension; :func:`si` checks it against the
    quantity's own. Each of the two is an instance of its own class: ``typing`` keeps one ``Annotated[Quantity, ...]``
    for markers that are equal and of one type, so a marker's angle survives only as its type."""

    exponents: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    angle: ClassVar[AngleCount | None] = None

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


class _PlaneAngle(Dimension):
    """A plane angle, counted in radians: :data:`Angle`'s class. It has no exponents, so it equals :data:`DIMENSIONLESS`,
    and it is its own class for the reason :class:`_Turns` is: ``typing`` keeps one ``Annotated[Quantity, ...]`` for markers
    that are equal and of one type, so a setting that says it is an angle is told from one that is dimensionless by its
    marker's type."""

    angle = "radian"

    @override
    def __str__(self) -> str:
        return "angle (rad)"


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
#: A plane angle, in radians: ``deg(30)`` and ``rad(0.5)``. A site declared with it takes a quantity of an angle and refuses
#: a bare number (30 of what?), a quantity of any other dimension, and a number of turns (a frequency times a time).
Angle: Dimension = _PlaneAngle(DIMENSIONLESS.exponents)
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME**2
FORCE = MASS * ACCELERATION
MAGNETIC_FLUX_DENSITY = MASS / (CURRENT * TIME**2)
#: A magnetic field's gradient, ``dB_i/dx_j``, in T/m.
FIELD_GRADIENT = MAGNETIC_FLUX_DENSITY / LENGTH
#: A magnetic field's rate of change, ``dB/dt``, in T/s.
FIELD_RATE = MAGNETIC_FLUX_DENSITY / TIME
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
    FIELD_RATE: "magnetic field rate of change (T/s)",
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
FieldRate = FIELD_RATE
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


class UnitError(ArgumentRefused):
    """A quantity of the wrong dimension, a bare number where a quantity is needed, or a unit label MIME cannot read:
    ``dimension`` is the one needed and ``given`` what was given, where :func:`si` knows them. ``alternatives`` are the
    quantities to write in its place where the error knows them, each with when it is the one (an angular rate of the
    other kind: ``Hz(5.0)`` if turns per second were meant), which :meth:`instead` puts in a site's own line.

    It is a ``ValueError`` and a :class:`~mime.errors.MimeError` with one refusal, which carries a code and what to do:
    ``call.units`` for a value (at the script's line where the error passed through one), and ``units.label`` for a label (``label`` true). An error given an ``action`` states what
    to do itself; one without it is given the quantity to write from ``dimension`` and ``given`` where it knows them. Its
    ``str`` is its message alone, so the sites that quote it in their own refusals read what they always did; the
    refusal's whole shape is ``refusals`` and :meth:`~mime.errors.MimeError.text`.

    Where it is raised decides the refusal it is. Raised in code MIME is checking or reading (an entry's methods, a
    controller's law, a saved form's manifest) it is that code's defect or the file's format, as a plain ``ValueError``
    from there would be (:attr:`~mime.errors.MimeError.kept_when_wrapped` is false). Reaching a script from its own
    expression, it keeps ``call.units``."""

    kept_when_wrapped = False

    def __init__(self, message: str, *, dimension: Dimension | None = None, given: object = None,
                 alternatives: tuple[tuple[str, str], ...] = (), action: str | None = None,
                 label: bool = False) -> None:
        # Not MimeError's own: the refusal is made when it is first read, since its line is the traceback's.
        Exception.__init__(self, message)
        self.original: BaseException | None = None
        self.chain: str | None = None
        self.traceback_file: str | None = None
        self.dimension = dimension
        self.given = given
        self.alternatives = alternatives
        self.action = action
        self.label = label
        # The refusal's lines are notes, which Python prints under the message: the notes are the exception's own, so
        # ``add_note`` and what Hypothesis adds to a failing example work as for any error. They are made here, not
        # held: ``refusals`` makes its own when read, with the script's line the traceback gives by then.
        try:
            from mime.errors._translate import unit_refusal

            for line in unit_refusal(self).lines():
                self.add_note(line)
        except Exception:  # noqa: BLE001 - making an error never fails for the sake of its note
            pass

    @property
    @override
    def refusals(self) -> tuple[Refusal, ...]:
        held = cast("tuple[Refusal, ...] | None", self.__dict__.get("_refusals"))
        if held is None:
            from mime.errors._translate import unit_refusal

            held = (unit_refusal(self),)
            self.__dict__["_refusals"] = held
        return held

    @override
    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""

    def instead(self, line: Callable[[str], str]) -> str:
        """What to write instead, after a colon, each alternative quantity in the site's ``line`` (``lambda q:
        f"State(spin={q})"``), for a site's action to end with; empty where the error knows no alternative."""
        if not self.alternatives:
            return ""
        return ": " + ", or ".join(f"{line(quantity)} {when}" for quantity, when in self.alternatives)


def _label_error(label: object, reason: str) -> UnitError:
    return UnitError(f"{label!r} is not a unit label MIME reads: {reason}", label=True)


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
    angle: AngleCount | None = field(default=None, compare=False)
    #: Where the value comes from, or why it was chosen (``mm(3.175, reason="...")``): equality, hashing and
    #: arithmetic ignore it, and a quantity derived from others carries none.
    reason: Basis | None = field(default=None, compare=False)
    #: The number a script's arithmetic multiplied the quantity by since its unit made it (``Hz(5.0) * (2 * pi)`` holds
    #: 2 pi), so that a refusal tells a frequency multiplied by 2 pi, which stays in Hz, from one written as it is.
    scale: float = field(default=1.0, compare=False, repr=False)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object) -> object:
        """NumPy's ufuncs leave a quantity to its own operators: ``numbers_array + quantity`` is :meth:`__radd__`'s
        refusal and not an array of one quantity per number, and ``numpy_scalar * quantity`` is :meth:`__mul__`'s (the
        operator the ufunc stands for, with the quantity on the right, as Python would have called it). The ufuncs
        that keep a unit are the quantity's own operators, as they were when NumPy took a quantity for an object
        (``np.negative``, ``np.positive``, ``np.absolute``, ``np.square``). Any other call (``np.sqrt(q)``) is a coded refusal that says what to write: ``__array_ufunc__ = None`` would
        have made it NumPy's own ``TypeError``, which has no code and names nothing to do."""
        if method == "__call__" and not kwargs and len(inputs) == 1 and inputs[0] is self:
            if ufunc is np.negative:
                return -self
            if ufunc is np.positive:
                return self
            if ufunc is np.absolute:
                return abs(self)
            if ufunc is np.square:
                return self * self
        reflected = _REFLECTED.get(ufunc.__name__)
        if (method == "__call__" and not kwargs and len(inputs) == 2 and inputs[1] is self
                and not isinstance(inputs[0], Quantity) and hasattr(self, reflected or "")):
            held = getattr(self, cast(str, reflected))(inputs[0])
            if held is not NotImplemented:
                return held
            if ufunc.__name__ in ("equal", "not_equal"):  # as Python answers ``a == b`` when neither knows the other
                return ufunc.__name__ == "not_equal"
        raise UnitError(f"NumPy's {ufunc.__name__} cannot be applied to a quantity ({self}): a quantity is not an array of "
                        f"numbers, and a function of NumPy would drop its unit or make an array of quantities",
                        action="use the quantity's own operations: abs(q) for the absolute value, q ** 2 for a power, "
                               "q.to(unit) for its number in a unit (q.value is the number in SI), and np functions on "
                               "those numbers")

    @classmethod
    def of(cls, values: object, dimension: Dimension) -> Quantity:
        """An array quantity: ``values`` in SI, copied into a read-only float64 array."""
        return cls(_frozen(np.asarray(values, dtype=np.float64)), dimension)

    def to(self, unit: Unit | str) -> Values:
        """The magnitude in ``unit`` (a float, a tuple or an array, as the quantity holds it), or :class:`UnitError`
        when ``unit`` is of another dimension: ``speed.to(mm_per_s)``. A unit may be written as text, as a script's
        ``--override`` writes one (a name of this module, its symbol, or a label of SI base units): ``force.to("mN")``;
        text that is none of them is refused (``units.label``)."""
        if isinstance(unit, str):
            unit = _unit_read(unit)
        if cast(object, unit) is degC:
            raise UnitError("cannot read a quantity in degC: it is an offset unit (0 degC is 273.15 K), so it converts "
                            "by an offset as well as a factor, which a unit here does not hold; read the temperature "
                            "in K and subtract ZERO_CELSIUS (mime.units.ZERO_CELSIUS) yourself",
                            action="read it in kelvin and subtract the offset yourself: "
                                   "`quantity.to(K) - ZERO_CELSIUS`, with `from mime.units import K, ZERO_CELSIUS`")
        if not isinstance(cast(object, unit), Unit):
            raise TypeError(f"to() takes a unit of mime.units, such as mm or mm_per_s, not {unit!r}")
        if unit.dimension != self.dimension:
            held = unit_name(self.dimension)
            raise UnitError(f"cannot read a quantity of {self.dimension} ({self}) in {unit.symbol}, a unit of "
                            f"{unit.dimension}", dimension=self.dimension,
                            action=f"read it in a unit of {self.dimension}" + (f", such as `quantity.to({held})` with "
                                                                              f"`from mime.units import {held}`" if held else ""))
        if self.angle is not None and unit.angle is not None and self.angle != unit.angle:
            kind = f"an angular rate in {_ANGLE_UNIT[self.angle]}" if self.dimension == FREQUENCY else \
                f"which counts {_COUNTS[self.angle]}"
            raise UnitError(f"cannot read {self}, {kind}, in {unit.symbol}: {_ANGLE_UNIT[self.angle]} and "
                            f"{_ANGLE_UNIT[unit.angle]} have the same dimension and differ by 2 pi (1 Hz is 2 pi "
                            f"rad/s), which a conversion here would silently drop; multiply or divide by 2 pi where "
                            f"you mean it",
                            action=f"read it in the unit it counts, `quantity.to({_ANGLE_UNIT_NAME[self.angle]})`, or "
                                   f"multiply or divide by 2 pi where you mean it")
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
        factor = _plain(other, "multiply")
        return Quantity(_scaled(self.value, factor), self.dimension, self.angle, scale=self.scale * factor)

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
        factor = _plain(other, "divide")
        return Quantity(_scaled(self.value, 1.0 / factor), self.dimension, self.angle, scale=self.scale / factor)

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

    def __abs__(self) -> Quantity:
        """The magnitude's absolute value, in the quantity's own dimension; a series' elementwise. A vector quantity has
        no one absolute value (its components', or its length), so it is refused."""
        if isinstance(self.value, tuple):
            raise UnitError("the absolute value of a vector quantity is not defined here: it has no one, its components' "
                            "or its length", action="take the absolute value of each component, read with `.to(unit)`, "
                                                    "and write the result in its own unit")
        value = np.abs(self.value) if isinstance(self.value, np.ndarray) else abs(self.value)
        return Quantity(_frozen(value) if isinstance(value, np.ndarray) else value, self.dimension, self.angle,
                        scale=abs(self.scale))

    def __pow__(self, power: int) -> Quantity:
        """An integer power (``m(2) ** 2`` is 4 m^2, ``m(2) ** -1`` is 0.5 1/m), its dimension raised with it. A fractional
        power is refused: a dimension's exponents are whole numbers here, so the square root of an area is the square
        root of its magnitude read in a unit, written as the length it is: ``m(sqrt(area.to(m2)))``. A quantity that counts
        an angle takes the powers 0, 1 and -1: a square of turns or radians is one a quantity here does not count."""
        if isinstance(cast(object, power), bool) or not isinstance(cast(object, power), Real) \
                or not float(power).is_integer():
            raise UnitError(f"a quantity takes an integer power, not {power!r}: its dimension's exponents are whole "
                            f"numbers",
                            action="write the integer power, `quantity ** 2`; for a root, read the magnitude with "
                                   "`.to(unit)`, take the root of the number and write it in its own unit")
        held = int(power)
        if self.angle is not None and abs(held) > 1:
            raise UnitError(f"cannot raise {self} to the power {held}: it counts an angle ({_COUNTS[self.angle]}), and "
                            f"a quantity here counts one at most", dimension=self.dimension, given=self,
                            action="raise its magnitude, read with `.to(unit)`, to the power, where you mean it")
        dimension = self.dimension**held
        if held < 0:
            return (1.0 / self) ** -held
        scale = self.scale**held
        if isinstance(self.value, np.ndarray):
            return Quantity(_frozen(self.value**held), dimension, self.angle if held else None, scale=scale)
        if isinstance(self.value, tuple):
            return Quantity(tuple(v**held for v in self.value), dimension, self.angle if held else None, scale=scale)
        return Quantity(self.value**held, dimension, self.angle if held else None, scale=scale)

    def __radd__(self, other: object) -> Quantity:
        """``0 + quantity`` is the quantity: what ``sum`` of quantities begins with. It is a Python ``int`` equal to 0
        alone, which is what ``sum`` starts at (``sum([mm(1.0), mm(2.0)])``) and nothing else is: any other number,
        ``0.0``, ``False`` and NumPy's zeros included, is refused as a bare number is on either side of a ``+``, and so
        is ``quantity + 0``. ``sum(items, start)`` with a start that is a quantity of the items' dimension works as ever."""
        if type(other) is int and other == 0:
            return self
        return self._same_dimension(other, "add")

    def _same_dimension(self, other: object, operation: str) -> Quantity:
        if not isinstance(other, Quantity):
            held = unit_name(self.dimension)
            raise UnitError(f"cannot {operation} a {self.dimension} quantity and the bare number {other!r}",
                            action="write the number with its unit" + (f": {held}({other!r}), with `from mime.units "
                                                                       f"import {held}`" if held and _real(other) else ""))
        if other.dimension != self.dimension:
            raise UnitError(f"cannot {operation} {self.dimension} and {other.dimension}",
                            action=f"{operation} quantities of one dimension: read one in the other's unit, "
                                   f"`quantity.to(unit)`, and write the result in its own")
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


_COUNTS: dict[AngleCount, str] = {"turn": "turns", "radian": "radians"}


def _plain(other: object, operation: str) -> float:
    """A number a quantity is multiplied or divided by: one number, as a Python ``float``. An array of several is refused
    by what it would be: an array quantity is a run's result, never a declaration."""
    if isinstance(other, np.ndarray) and other.ndim:
        raise UnitError(f"cannot {operation} a quantity by an array of {other.size}: an array quantity is a run's result "
                        f"(a series read from a result), never a declaration",
                        action="scale the numbers before the unit, mm(3.0 * numbers), or write a vector as "
                               "mm((1.0, 2.0))")
    return float(cast(float, other))


def _differ(a: Quantity, b: Quantity) -> bool:
    """Whether ``a`` and ``b`` both know what they count, and count different things (a turn and a radian)."""
    return a.angle is not None and b.angle is not None and a.angle != b.angle


def _sum_angle(a: Quantity, b: Quantity, operation: str) -> AngleCount | None:
    """What a sum, difference or order of ``a`` and ``b`` counts: what either knows, or :class:`UnitError` where they
    know different things."""
    if _differ(a, b):
        raise UnitError(f"cannot {operation} {a} and {b}: one counts {_COUNTS[cast(AngleCount, a.angle)]} and the other "
                        f"{_COUNTS[cast(AngleCount, b.angle)]}, which share a dimension and differ by 2 pi (1 Hz is 2 pi "
                        f"rad/s), which MIME does not drop silently; write both in one unit, multiplying or dividing "
                        f"one by 2 pi where you mean it", dimension=a.dimension, given=b,
                        action="write both in one unit, multiplying or dividing one by 2 pi where you mean it")
    return a.angle if a.angle is not None else b.angle


def _product_angle(a: Quantity, b: Quantity, operation: str) -> AngleCount | None:
    """What a product or ratio of ``a`` and ``b`` counts: what the one that knows counts, nothing for a ratio of two
    that count the same (turns over turns), or :class:`UnitError` where both know (a product would count its angle
    twice, and a ratio of a turn and a radian is off by 2 pi)."""
    if a.angle is None or b.angle is None:
        return a.angle if a.angle is not None else b.angle
    if operation == "divide" and a.angle == b.angle:
        return None
    raise UnitError(f"cannot {operation} {a} by {b}: both count an angle ({_COUNTS[a.angle]} and {_COUNTS[b.angle]}), "
                    f"and a quantity here counts one at most; {operation} their magnitudes, each read with .to(...), "
                    f"where you mean it", dimension=a.dimension, given=b,
                    action=f"{operation} their magnitudes, each read with `.to(unit)`, where you mean it")


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
    otherwise a product of base units (``m^2 kg s^-1``), which :func:`parse_unit` reads back. A plane angle
    (:data:`Angle`) is ``rad``, which reads back as dimensionless."""
    if isinstance(dimension, _PlaneAngle):
        return "rad"
    if dimension == DIMENSIONLESS:
        return "1"
    if dimension in _SYMBOLS:
        return _SYMBOLS[dimension]
    return " ".join(symbol if power == 1 else f"{symbol}^{power}"
                    for symbol, power in zip(_BASE_SYMBOLS, dimension.exponents, strict=True) if power)


#: An energy (J) and a power (W): dimensions of the label grammar's named units. Neither is in ``_NAMED`` or
#: ``_SYMBOLS``: an energy is dimensionally a torque (see the module's limits), which keeps its symbol, ``N m``.
ENERGY = FORCE * LENGTH
POWER = ENERGY / TIME

#: What a label's name stands for: the SI base symbols and the named derived units a label may write, each by its
#: exponents of the base units, with what an angular rate counts where the name says (``Hz`` counts turns, ``rad``
#: radians). A prefixed unit (``mm``, ``kPa``) is no name here: a label holds no scale factor, only a dimension.
_LABEL_NAMES: dict[str, tuple[tuple[int, ...], AngleCount | None]] = {
    **{symbol: (_base(i).exponents, None) for i, symbol in enumerate(_BASE_SYMBOLS)},
    "N": (FORCE.exponents, None), "Pa": (PRESSURE.exponents, None), "J": (ENERGY.exponents, None),
    "W": (POWER.exponents, None), "T": (MAGNETIC_FLUX_DENSITY.exponents, None),
    "Hz": (FREQUENCY.exponents, "turn"), "rad": (DIMENSIONLESS.exponents, "radian"),
}
_NO_EXPONENTS = DIMENSIONLESS.exponents
_POWER = "^"
_DIGITS = "0123456789"


@dataclass(frozen=True)
class _Term:
    """A label's term while it is read: exponents of the base units, and what its angular rate counts, if it names one:
    ``angle`` is turns (``Hz``) or radians (``rad``), and ``counted`` how many times over the term holds it (``rad/s``
    counts one, ``rad^2/s`` two, ``1/(rad s)`` minus one)."""

    exponents: tuple[int, ...]
    angle: AngleCount | None = None
    counted: int = 0


def _times(a: _Term, b: _Term, label: str) -> _Term:
    if a.counted and b.counted and a.angle != b.angle:
        raise _label_error(label, "it counts both turns (Hz) and radians (rad), which differ by 2 pi and are never mixed "
                                  "in one unit")
    counted = a.counted + b.counted
    return _Term(tuple(x + y for x, y in zip(a.exponents, b.exponents, strict=True)),
                 (a.angle if a.counted else b.angle) if counted else None, counted)


def _over(a: _Term, b: _Term, label: str) -> _Term:
    return _times(a, _Term(tuple(-x for x in b.exponents), b.angle, -b.counted), label)


def _raised(term: _Term, power: int) -> _Term:
    return _Term(tuple(x * power for x in term.exponents), term.angle if term.counted else None, term.counted * power)


class _Label:
    """A unit label's reader. The grammar:

    - a *factor* is a name (``m``, ``N``, ``Hz``), the 1, or a parenthesised expression, with an optional integer
      power written right after it (``s^-2``, ``(m s)^2``);
    - a *product* is factors separated by spaces (``kg m^2``);
    - an *expression* is a product, and then at most one ``/`` and a single factor, the denominator (``N s/m``,
      ``m/(s Pa)``). More than one term after a ``/`` is written in parentheses, so a label is never read two ways:
      ``J/mol K`` is refused, ``J/(mol K)`` is read.

    A label with a ``;`` is a composite (a pose is ``m; 1``): its parts are read apart (:func:`parse_label`)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.at = 0

    def unread(self, reason: str) -> UnitError:
        return _label_error(self.label, f"{reason} (at character {self.at + 1})" if self.at < len(self.label) else reason)

    def spaces(self) -> None:
        while self.at < len(self.label) and self.label[self.at].isspace():
            self.at += 1

    def expression(self) -> _Term:
        found = _Term(_NO_EXPONENTS)
        factors = 0
        while True:
            self.spaces()
            here = self.label[self.at] if self.at < len(self.label) else ""
            if here in ("", ")", "/"):
                break
            found = _times(found, self.factor(), self.label)
            factors += 1
            after = self.label[self.at] if self.at < len(self.label) else ""
            if after and not after.isspace() and after not in ")/":
                raise self.unread(
                    f"the factors of a product are separated by spaces, and {after!r} follows a factor directly "
                    f"('m1' is not 'm')" if after.isalnum() or after == "(" else
                    f"{after!r} is not a unit symbol, the number 1 or a parenthesis")
        if not factors:
            raise self.unread("an empty unit; write 1 for a dimensionless value" if not self.label.strip() else
                              "a term is missing")
        if here == "/":
            self.at += 1
            self.spaces()
            found = _over(found, self.factor(), self.label)
            self.spaces()
            if self.at < len(self.label) and self.label[self.at] != ")":
                raise self.unread("more than one term follows a '/': write the denominator in parentheses, "
                                  "'m/(s Pa)', or the quotient as a negative power, 'm s^-1 Pa^-1'")
        return found

    def factor(self) -> _Term:
        here = self.label[self.at] if self.at < len(self.label) else ""
        if here == "(":
            self.at += 1
            term = self.expression()
            if self.at >= len(self.label) or self.label[self.at] != ")":
                raise self.unread("a parenthesis is not closed")
            self.at += 1
        elif here == "1" and not self.label[self.at + 1:self.at + 2].isalnum():  # the 1 alone, not 10 or 1m
            self.at += 1
            term = _Term(_NO_EXPONENTS)
        elif here.isalpha():
            start = self.at
            while self.at < len(self.label) and self.label[self.at].isalpha():
                self.at += 1
            name = self.label[start:self.at]
            named = _LABEL_NAMES.get(name)
            if named is None:
                close = difflib.get_close_matches(name, _LABEL_NAMES, n=1)
                self.at = start
                raise self.unread(f"there is no unit symbol {name!r} (the names are {', '.join(_LABEL_NAMES)})"
                                  + (f"; did you mean {close[0]!r}?" if close else ""))
            term = _Term(named[0], named[1], 1 if named[1] else 0)
        elif here and here in _DIGITS:
            raise self.unread("the only number a label holds is 1, for a dimensionless value; a factor such as 1000 is "
                              "a prefix or a scale, which belongs to a quantity, not a label")
        else:
            raise self.unread(f"{here!r} is not a unit symbol, the number 1 or a parenthesis" if here else
                              "a term is missing")
        return self.power(term)

    def power(self, term: _Term) -> _Term:
        if self.label[self.at:self.at + 1] != _POWER:
            return term
        end = self.at + 1
        while end < len(self.label) and (self.label[end] in _DIGITS or (end == self.at + 1 and self.label[end] == "-")):
            end += 1
        written = self.label[self.at + 1:end]
        digits = written.lstrip("-")
        if not digits or digits.startswith("0"):
            self.at += 1
            raise self.unread("a power is an integer other than zero, written right after its term, as s^-2")
        self.at = end
        return _raised(term, int(written))


def _rate_counting(angle: AngleCount) -> Dimension:
    """The rate per second that counts ``angle`` once: :data:`FREQUENCY` for turns, :data:`ANGULAR_VELOCITY` for radians."""
    return FREQUENCY if angle == "turn" else ANGULAR_VELOCITY


def parse_unit(symbol: str) -> Dimension:
    """The dimension :func:`unit_symbol` wrote as ``symbol``, or :class:`UnitError`: every label MIME writes. The SI base
    symbols (``m kg s A K mol cd``) and the named units ``N Pa J W T Hz rad``, as products by spaces, one quotient by ``/``
    (more than one term in the denominator in parentheses), integer powers by ``^``, parentheses, and ``1``: ``m/s^2``,
    ``N s/m``, ``m/(s Pa)``, ``m^2 kg s^-1``.

    A turn and a radian stay apart: ``Hz`` is :data:`FREQUENCY` and ``rad/s`` :data:`ANGULAR_VELOCITY`, which are equal
    as dimensions and differ in what they count (:attr:`Dimension.angle`); ``rad`` alone is dimensionless, as
    ``Unit`` holds it. A prefixed unit (``mm``) has a scale, which a label does not hold: it is refused."""
    return _read(symbol)[0]


def _read(symbol: str) -> tuple[Dimension, _Term]:
    """A label's dimension and its term, which holds what the label counts."""
    if not isinstance(cast(object, symbol), str):
        raise _label_error(symbol, f"a unit label is text, such as 'N m', not a {type(symbol).__name__}")
    reader = _Label(symbol)
    term = reader.expression()
    if reader.at < len(symbol):
        raise reader.unread("a closing parenthesis with none open")
    if term.exponents == FREQUENCY.exponents and term.counted:
        if term.counted != 1:
            raise _label_error(symbol, f"it is a rate per second that counts {term.counted} of "
                                       f"{'turns (Hz)' if term.angle == 'turn' else 'radians (rad)'}, not one: a rate "
                                       f"counts turns or radians once, 'Hz' or 'rad/s'")
        return _rate_counting(cast(AngleCount, term.angle)), term
    return Dimension.of(term.exponents), term


def parse_value_unit(symbol: str) -> tuple[Dimension, AngleCount | None]:
    """The unit a value is written in, as :func:`parse_unit` reads it and with what it counts: the dimension and the
    quantity's :attr:`Quantity.angle`, so that ``10 rad s^-1`` is an angular velocity and ``2 Hz s`` is two turns, which
    a site that takes the other refuses and never reads as the number. A value counts an angle once or not at all: a
    label that counts it twice (``rad^2``) is not a value's unit."""
    dimension, term = _read(symbol)
    if term.counted not in (0, 1):
        raise _label_error(symbol, f"it counts {'turns (Hz)' if term.angle == 'turn' else 'radians (rad)'} {term.counted} "
                                   f"times over, and a value counts an angle once or not at all")
    return dimension, term.angle if term.counted else None


def parse_label(label: str) -> tuple[Dimension, ...]:
    """The dimension of each part of a unit label, split at its semicolons (``"m; 1"`` is a length and a number, a pose's
    parts); :class:`UnitError` for a part that is not a unit."""
    if not isinstance(cast(object, label), str):
        raise _label_error(label, f"a unit label is text, such as 'N m; 1', not a {type(label).__name__}")
    return tuple(parse_unit(part) for part in label.split(";"))


type Marker = Dimension | str


def unit_label(*parts: Marker) -> str:
    """A unit label written from dimension markers, the ones settings are declared with (``unit_label(Length,
    DIMENSIONLESS)`` is ``"m; 1"``, a pose's), or from labels already written; a part that is a label is read here, so
    that a typo fails where the label is written. A marker is written as :func:`rate_symbol` writes it, so an angular
    velocity is ``rad/s`` and a frequency ``Hz``."""
    if not parts:
        raise _label_error("", "a label has at least one part")
    return "; ".join(rate_symbol(part) if isinstance(part, Dimension) else _checked(part) for part in parts)


def _checked(label: str) -> str:
    parse_label(label)
    return label


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
        if dimension == DIMENSIONLESS and not isinstance(dimension, _PlaneAngle):
            return _magnitude(quantity)
        bare = isinstance(dimension, _PlaneAngle) and _real(quantity)
        raise UnitError(f"{what} needs units of {dimension}; it was given the bare value {quantity!r}"
                        f"{', which could be degrees or radians' if bare else ''}",
                        dimension=dimension, given=quantity,
                        alternatives=((f"deg({quantity!r})", "if degrees"), (f"rad({quantity!r})", "if radians"))
                        if bare else ())
    if quantity.dimension != dimension:
        raise UnitError(f"{what} needs units of {dimension}; it was given {quantity.dimension} ({quantity})",
                        dimension=dimension, given=quantity)
    if isinstance(quantity.value, np.ndarray):
        raise UnitError(f"{what} is given an array quantity ({quantity}): an array is a run's result, and a "
                        f"declaration takes a number or a tuple of them",
                        action="declare a number or a tuple of them with its unit: mm(3.0), mm((1.0, 2.0))")
    if dimension == DIMENSIONLESS and quantity.angle == "turn":
        raise _turns_counted(quantity, what)
    if dimension.angle is not None and quantity.angle is not None and quantity.angle != dimension.angle:
        raise _other_angle(quantity, dimension, what)
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
    angle: AngleCount | None = None

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
mN = Unit("mN", 1e-3, FORCE, "Millinewtons.")
A = Unit("A", 1.0, CURRENT, "Amperes.")
K = Unit("K", 1.0, TEMPERATURE, "Kelvin.")
T = Unit("T", 1.0, MAGNETIC_FLUX_DENSITY, "Tesla.")
mT = Unit("mT", 1e-3, MAGNETIC_FLUX_DENSITY, "Millitesla.")
uT = Unit("uT", 1e-6, MAGNETIC_FLUX_DENSITY, "Microtesla.")
T_per_m = Unit("T/m", 1.0, FIELD_GRADIENT, "Tesla per metre: a magnetic field's gradient.")
mT_per_m = Unit("mT/m", 1e-3, FIELD_GRADIENT, "Millitesla per metre: an electromagnetic navigation system's gradients.")
T_per_s = Unit("T/s", 1.0, FIELD_RATE, "Tesla per second: a magnetic field's rate of change.")
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
#: A turn counted per minute, and an angle's degree per second: rates that keep what they count (turns, radians).
rpm = Unit("rpm", 1.0 / 60.0, FREQUENCY, "Revolutions per minute: turns per minute, 1/60 Hz. It counts turns, as Hz does, "
           "and is refused where a site takes radians per second.", "turn")
deg_per_s = Unit("deg/s", math.pi / 180.0, ANGULAR_VELOCITY,
                 "Degrees per second: an angular velocity, held in rad/s (pi/180 of one each).", "radian")
m2 = Unit("m^2", 1.0, AREA, "Square metres: an area, or a porous material's Darcy permeability.")
mm2 = Unit("mm^2", 1e-6, AREA, "Square millimetres: an area.")
um2 = Unit("um^2", 1e-12, AREA, "Square micrometres: an area, or a porous material's Darcy permeability.")
m3 = Unit("m^3", 1.0, VOLUME, "Cubic metres.")
mm3 = Unit("mm^3", 1e-9, VOLUME, "Cubic millimetres.")
#: A millilitre is a cubic centimetre, 1e-6 m^3 exactly (the litre is 1e-3 m^3 by definition).
mL = Unit("mL", 1e-6, VOLUME, "Millilitres: a volume, 1e-6 m^3.")
kg_per_m3 = Unit("kg/m^3", 1.0, DENSITY, "Kilograms per cubic metre.")
kg_m2 = Unit("kg m^2", 1.0, MOMENT_OF_INERTIA, "Kilogram square metres: a moment of inertia.")
A_m2 = Unit("A m^2", 1.0, MAGNETIC_MOMENT, "Ampere square metres: a magnetic moment.")
Pa_s = Unit("Pa s", 1.0, DYNAMIC_VISCOSITY, "Pascal seconds: a dynamic viscosity.")
#: The centipoise, a hundredth of a poise (1 P = 0.1 Pa s): exactly one millipascal second.
cP = Unit("cP", 1e-3, DYNAMIC_VISCOSITY, "Centipoise: a dynamic viscosity, exactly 1 mPa s (water near 20 degC is about 1).")
mPa_s = Unit("mPa s", 1e-3, DYNAMIC_VISCOSITY, "Millipascal seconds (water near 20 degC is about 1).")
N_m = Unit("N m", 1.0, TORQUE, "Newton metres: a torque.")
N_m_s = Unit("N m s", 1.0, ROTATIONAL_DAMPING,
             "Newton metre seconds: a torque per angular velocity, a rotational damping.")
Pa = Unit("Pa", 1.0, PRESSURE, "Pascals: a pressure.")
kPa = Unit("kPa", 1e3, PRESSURE, "Kilopascals: a pressure.")
MPa = Unit("MPa", 1e6, PRESSURE, "Megapascals: a pressure.")
#: The conventional millimetre of mercury: a column of mercury of conventional density, 13 595.1 kg/m^3, at standard
#: gravity, 9.80665 m/s^2, one millimetre high, 13 595.1 * 9.80665 * 1e-3 = 133.322387415 Pa (NIST Special Publication
#: 811's value). The torr, 101 325/760 Pa = 133.32237 Pa, is a different unit that differs from it in the eighth significant figure.
mmHg = Unit("mmHg", 133.322387415, PRESSURE, "Millimetres of mercury, the conventional unit of a blood pressure: "
            "133.322387415 Pa.")
m3_per_s = Unit("m^3/s", 1.0, VOLUME_FLUX, "Cubic metres per second: a volume flux.")
mL_per_s = Unit("mL/s", 1e-6, VOLUME_FLUX, "Millilitres per second: a vessel's volume flux.")
Pa_s_per_m3 = Unit("Pa s/m^3", 1.0, HYDRAULIC_RESISTANCE, "Pascal seconds per cubic metre: a hydraulic resistance.")
m3_per_Pa = Unit("m^3/Pa", 1.0, COMPLIANCE, "Cubic metres per pascal: a vessel's or a Windkessel's compliance.")
dimensionless = Unit("1", 1.0, DIMENSIONLESS, "A pure number, such as a ratio or a normalised wavenumber.")


def units_by_name() -> dict[str, Unit]:
    """Every unit this module exports, by its Python name (``mPa_s``, ``mm_per_s``): what ``--override`` and a study's
    spec file read a unit as, beside the symbol each unit prints with (``mPa s``, ``mm/s``)."""
    return {name: value for name, value in globals().items() if isinstance(value, Unit) and not name.startswith("_")}


def unit_written(text: str) -> Unit | None:
    """The unit of this module that ``text`` writes, or ``None`` where it writes none: a unit's Python name (``mPa_s``,
    ``mm_per_s``), the symbol it prints with (``mPa s``, ``mm/s``, with or without spaces about its slash), or, in a token
    with no space in it, that symbol with its spaces left out (``mPas``). A unit written with spaces is its symbol as
    printed, spaces and all: it is never read with them removed, which would read ``m s`` as ``ms`` and ``2 m T`` as
    ``2 mT``; what is not a unit's symbol is read by the label grammar (``m s`` is a product)."""
    named = units_by_name()
    held = named.get(text)
    if held is not None:
        return held
    by_symbol = {unit.symbol: unit for unit in named.values()}
    if len(text.split()) > 1:
        # a slash is written with or without spaces about it (``mm / s``, ``rad /s``): it joins no letters, so the
        # symbol it spells is the same one
        return by_symbol.get(" ".join(re.sub(r"\s*/\s*", "/", text).split()))
    return by_symbol.get(text) or {unit.symbol.replace(" ", ""): unit for unit in named.values()}.get(text)


def _unit_read(text: str) -> Unit:
    """The unit a script wrote as text where a unit goes (``series.to("mN")``): :func:`unit_written`'s, or else the SI
    base units' label grammar's (``"kg m/s^2"``, a unit of factor one), or :class:`UnitError` coded ``units.label`` naming
    the nearest units."""
    held = unit_written(text.strip())
    if held is not None:
        return held
    try:
        dimension, angle = parse_value_unit(text)
    except UnitError:
        known = {name.lower(): name for name in (*units_by_name(), *(u.symbol for u in units_by_name().values()))}
        near = [known[found] for found in difflib.get_close_matches(text.strip().lower(), list(known), n=3)]
        raise UnitError(f"{text!r} is not a unit MIME reads: it is no unit's name or symbol in mime.units"
                        f"{' (nearest: ' + ', '.join(near) + ')' if near else ''}, and no label of SI base units "
                        f"such as 'N m' or 'kg m/s^2'", label=True,
                        action="name a unit of mime.units (`mm`, `mN`, `mm_per_s`) or its symbol ('mm/s', 'N m'), or write "
                               "SI base units ('kg m/s^2'); `quantity.to(mN)` takes the unit itself") from None
    return Unit(text.strip(), 1.0, dimension, "A unit written as a label of SI base units.", angle)


def unit_name(dimension: Dimension) -> str | None:
    """The name in this module of the SI unit of ``dimension`` (``s`` for a time, ``Hz`` for a frequency, ``rad_per_s`` for
    an angular velocity), where it has one."""
    held = (name for name, value in globals().items() if isinstance(value, Unit) and value.factor == 1.0
            and value.dimension == dimension and value.angle == dimension.angle and not name.startswith("_"))
    return next(held, None)


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


from mime.core._labels import install_label_reader  # noqa: E402  (``mime.core`` reads labels with this module's reader)

install_label_reader(parse_label, unit_label, UnitError)

__all__ = [
    "ACCELERATION", "ANGULAR_VELOCITY", "AREA", "Angle", "AngularVelocity", "Area", "COMPLIANCE", "Compliance", "HYDRAULIC_RESISTANCE",
    "HydraulicResistance", "PRESSURE", "Pressure", "VOLUME_FLUX", "VolumeFlux", "Pa", "Pa_s_per_m3", "m3_per_Pa",
    "m3_per_s", "mL_per_s", "SAY_THE_ANGLE", "rate_symbol", "says_its_angle", "CURRENT",
    "DENSITY", "DIMENSIONLESS", "DYNAMIC_VISCOSITY", "FIELD_GRADIENT", "FIELD_RATE", "FORCE", "FREQUENCY", "LENGTH",
    "MAGNETIC_FLUX_DENSITY", "MAGNETIC_MOMENT", "MASS", "MOMENT_OF_INERTIA", "ROTATIONAL_DAMPING", "TEMPERATURE",
    "TIME", "TORQUE", "VELOCITY", "VOLUME", "Acceleration", "Current", "Density", "DynamicViscosity", "FieldGradient",
    "FieldRate",
    "Force", "Frequency", "Length", "MagneticFluxDensity", "MagneticMoment", "Mass", "MomentOfInertia",
    "RotationalDamping", "Temperature", "Time", "Torque", "Velocity", "Volume", "Condition", "Dimension", "Quantity",
    "Unit", "UnitError", "VACUUM_PERMEABILITY", "Values", "ZERO_CELSIUS", "parse_unit", "parse_value_unit", "si", "unit_symbol", "A",
    "A_m2", "Hz", "K", "N", "mN", "mm2", "N_m", "N_m_s", "Pa_s", "T", "T_per_m", "T_per_s", "cm", "deg", "degC", "dimensionless", "g", "kHz",
    "kg", "kg_m2", "kg_per_m3", "m", "m2", "m3", "m_per_s", "m_per_s2", "mPa_s", "mT", "mT_per_m", "mm", "mm3", "mm_per_s",
    "ms", "rad", "rad_per_s", "s", "standard_gravity", "uT", "um", "um2", "um_per_s", "us",
    "ENERGY", "MPa", "Marker", "POWER", "cP", "deg_per_s", "kPa", "mL", "mmHg", "parse_label", "rpm", "unit_label",
    "unit_name", "units_by_name",
]
