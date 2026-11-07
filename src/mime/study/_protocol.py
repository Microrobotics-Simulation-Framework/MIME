"""A study's protocol: how each member's commands move in time. :class:`FromRest`, the default, holds every command at
its point's value from the start, the body starting from its declared initial state; :class:`Ramp` holds one
command at its point's value, then ramps it (``drives.Ramp``). A protocol is declared on each point, so every point is
still the experiment with its values declared, planned in full.

Step-out depends on the protocol: from rest a body is captured by the field or not; on a ramp it follows the field up
to a higher frequency than it can be captured at, and is caught again on the way down at a lower one. So each reading
here names the quantity it is (``from rest``, ``ramp up``, ``ramp down``) and never mixes them."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

from mime.build import Declaration, ThingDeclaration
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.drives import Constant
from mime.drives import Ramp as RampDrive
from mime.errors import public
from mime.results._series import Series
from mime.things import Apparatus, CommandPath
from mime.units import FREQUENCY, TIME, Quantity, s, si

if TYPE_CHECKING:
    from mime.results._study import StudyResult


@dataclass(frozen=True)
class FromRest:
    """Each member from the experiment's declared initial state, every command held at its point's value from the
    start: a sweep over a field's frequency reads the step-out from rest, where the body is captured by the field or
    not."""

    quantity: ClassVar[str] = "from rest"

    def text(self) -> str:
        return "from rest: every command held at its point's value from the start"


@dataclass(frozen=True)
class Ramp:
    """One command held at its point's value for ``hold``, then moved at ``rate`` towards ``to`` and held there
    (``drives.Ramp``): ``Ramp(field.commands.frequency, to=Hz(5.0), rate=Hz(0.25) / s(1.0), hold=s(3.0))``. The point's
    value of the command (its axis's value, or what the experiment declares) is where the ramp starts. A member whose
    ramp rises reads the ``ramp up`` quantity, one whose ramp falls the ``ramp down``."""

    command: object
    to: Quantity
    rate: Quantity
    hold: Quantity = s(0.0)

    @property
    def path(self) -> tuple[str, str]:
        """The apparatus and the command it ramps."""
        if isinstance(self.command, CommandPath):
            return self.command.owner, self.command.command
        held = str(self.command)
        owner, _, rest = held.partition(".")
        middle, _, command = rest.partition(".")
        if middle != "commands" or not command:
            raise ArgumentRefused([Refusal(
                "study.protocol-command", RefusalClass.USER,
                f"Ramp ramps an apparatus's command, named as a study's axis names one, not {held!r}",
                "name it by the apparatus's handle, Ramp(field.commands.frequency, to=..., rate=...), or its string, "
                "\"field.commands.frequency\"")])
        return owner, command

    def text(self) -> str:
        owner, command = self.path
        return (f"a ramp of {owner}.commands.{command}: held at the point's value for {self.hold}, then moved at "
                f"{self.rate} towards {self.to}")

    def drive(self, start: Quantity) -> RampDrive:
        return RampDrive(start, to=self.to, rate=self.rate, hold=self.hold)


type Protocol = FromRest | Ramp


def protocol_of(given: object) -> Protocol:
    if given is None:
        return FromRest()
    if isinstance(given, (FromRest, Ramp)):
        return given
    raise ArgumentRefused([Refusal(
        "call.wrong-type", RefusalClass.USER, f"protocol= takes FromRest() or Ramp(...), not {type(given).__name__}",
        "write protocol=FromRest(), or protocol=Ramp(field.commands.frequency, to=..., rate=...)")])


def declared_with(protocol: Protocol, declarations: Sequence[Declaration]) -> tuple[tuple[Declaration, ...],
                                                                                    RampDrive | None]:
    """The point's declarations with its protocol declared in them, and the drive the protocol gave (``None`` from
    rest): for a ramp, the command's constant value becomes the ramp's start. Refused where the experiment gives the
    command no constant value (``study.protocol-command``)."""
    if isinstance(protocol, FromRest):
        return tuple(declarations), None
    owner, command = protocol.path
    held: list[Declaration] = []
    made: RampDrive | None = None
    for declaration in declarations:
        if isinstance(declaration, ThingDeclaration) and isinstance(declaration.thing, Apparatus) and \
                declaration.thing.name == owner:
            thing = declaration.thing
            value = thing.commands.get(command)
            start = value.value if isinstance(value, Constant) else value
            if not isinstance(start, Quantity):
                said = "no value" if value is None else f"the drive {type(value).__name__}"
                raise ArgumentRefused([Refusal(
                    "study.protocol-command", RefusalClass.USER,
                    f"the protocol ramps {owner}.commands.{command} from its value at each point, and the experiment "
                    f"gives it {said}",
                    f"give the command a value, commands={{{command!r}: Hz(...)}}, or sweep it: "
                    f"grid({{{owner}.commands.{command}: [...]}})")])
            made = protocol.drive(start)
            declaration = ThingDeclaration(dataclasses.replace(thing, commands={**thing.commands, command: made}))
        held.append(declaration)
    if made is None:
        raise ArgumentRefused([Refusal(
            "study.protocol-command", RefusalClass.USER,
            f"the protocol ramps {owner}.commands.{command}, and the experiment declares no apparatus {owner!r}",
            "name an apparatus the experiment declares, by its handle: Ramp(field.commands.frequency, ...)")])
    return tuple(held), made


def _ramp_numbers(drive: RampDrive) -> tuple[float, float, float, float]:
    start, to, rate = (cast(float, si(value, value.dimension, what=name))
                       for name, value in (("start", drive.start), ("to", drive.to), ("rate", drive.rate)))
    return start, to, rate, cast(float, si(drive.hold, TIME, what="hold"))


def ramp_value(drive: RampDrive, time: np.ndarray) -> np.ndarray:
    """The ramp ``drive`` at each of ``time`` (s), in SI, in float64: held, then moved at its rate, then held."""
    start, to, rate, hold = _ramp_numbers(drive)
    return start + math.copysign(rate, to - start) * np.minimum(np.maximum(time - hold, 0.0), abs(to - start) / rate)


def commanded(drive: RampDrive, steps: int, dt: float) -> np.ndarray:
    """What ``drive`` commands at each of the first ``steps`` steps of ``dt`` (SI): its value at each step's middle, as
    its node computes it, in float64."""
    return ramp_value(drive, (np.arange(1, steps + 1, dtype=np.float64) - 0.5) * dt)


@dataclass(frozen=True)
class StepOut:
    """A from-rest step-out over a swept frequency, at one value of each other axis (``at``): ``last_locked``, the
    highest frequency below which every point is locked, and ``first_unlocked``, the frequency after it (``None`` where
    every point is locked); ``ratios``, each point's mean spin over its frequency in the order of ``frequencies``; a point is
    locked where its ratio is at least ``threshold``. Its ``quantity`` is ``from rest``."""

    quantity: ClassVar[str] = "from rest"
    at: tuple[tuple[str, str], ...]
    frequencies: tuple[float, ...]
    ratios: tuple[float, ...]
    threshold: float
    last_locked: float | None
    first_unlocked: float | None
    relocks: bool

    def text(self) -> str:
        where = f" at {', '.join(f'{path} = {value}' for path, value in self.at)}" if self.at else ""
        if self.first_unlocked is None:
            return f"step-out from rest{where}: every point locked, up to {self.frequencies[-1]:g} Hz"
        last = f"{self.last_locked:g} Hz" if self.last_locked is not None else "none"
        again = "; a point above it locks again" if self.relocks else ""
        return f"step-out from rest{where}: last locked {last}, first unlocked {self.first_unlocked:g} Hz{again}"


@dataclass(frozen=True)
class Crossing:
    """Where a ramp's member changes state: on a ramp up, the first window after the hold whose mean spin over its mean
    commanded frequency is below ``threshold`` (it steps out); on a ramp down, the first at or above it
    (it is caught). ``at`` is the commanded frequency at that window's centre (Hz), ``time`` the centre's time (s),
    ``None`` both where it never crosses; ``quantity`` is ``ramp up`` or ``ramp down``."""

    member: int
    quantity: str
    window: float
    threshold: float
    at: float | None
    time: float | None

    def text(self) -> str:
        if self.at is None or self.time is None:
            return f"member {self.member}, {self.quantity}: " + \
                ("never steps out in the run" if self.quantity == "ramp up" else "never caught in the run")
        what = "steps out" if self.quantity == "ramp up" else "is caught"
        return f"member {self.member}, {self.quantity}: {what} at {self.at:.2f} Hz ({self.time:.3f} s), by " \
               f"{self.window:g} s windows"


def _lag_id(lag: object) -> str:
    return str(getattr(lag, "id", lag))


def _seconds(value: Quantity, what: str) -> float:
    return cast(float, si(value, TIME, what=what))


@public
def step_out(result: StudyResult, lag: object, frequency: object, *, threshold: float = 0.99) -> tuple[StepOut, ...]:
    """The from-rest step-out of a study run from rest over the axis ``frequency`` (a field's frequency command), at
    each combination of its other axes: each member's ``mean_spin`` (``PhaseLag``'s, ``lag``, over its window, in turns
    per second) over its frequency, and the bracket where it first falls below ``threshold``. Refused for a study run on a ramp, whose
    quantity is another (``study.protocol-quantity``)."""
    from mime.study._axes import value_text

    if not isinstance(result.protocol, FromRest):
        raise ArgumentRefused([Refusal(
            "study.protocol-quantity", RefusalClass.USER,
            f"step_out reads the step-out from rest, and this study ran {result.protocol.text()}",
            "read a ramp's members with ramp_crossings(result, lag), each named ramp up or ramp down")])
    path = str(frequency)
    if path not in result.axes:
        raise ArgumentRefused([Refusal(
            "study.unknown-path", RefusalClass.USER,
            f"step_out reads the step-out over a swept frequency, and the study sweeps {', '.join(result.axes)}",
            "name the swept frequency: step_out(result, lag, field.commands.frequency)")])
    coordinates = result.coordinates[path]
    values = cast(Quantity, coordinates)
    if not isinstance(coordinates, Quantity) or values.dimension != FREQUENCY:
        raise ArgumentRefused([Refusal(
            "study.unknown-path", RefusalClass.USER, f"step_out reads a frequency, and {path} holds {coordinates!r}",
            "name the swept frequency: step_out(result, lag, field.commands.frequency)")])
    turns = np.asarray(values.value, dtype=np.float64)
    spin = np.asarray(cast(Quantity, result[_lag_id(lag)].mean_spin).value, dtype=np.float64)
    ratios = spin / turns
    others = [axis for axis in result.axes if axis != path]
    rows: dict[tuple[tuple[str, str], ...], list[int]] = {}
    for member in range(len(result)):
        key = tuple((axis, value_text(result.coordinate(axis, member))) for axis in others)
        rows.setdefault(key, []).append(member)
    found: list[StepOut] = []
    for key, members in rows.items():
        order = sorted(members, key=lambda m: turns[m])
        held = [float(turns[m]) for m in order]
        ratio = [float(ratios[m]) for m in order]
        out = next((i for i, r in enumerate(ratio) if not r >= threshold), None)
        found.append(StepOut(key, tuple(held), tuple(ratio), threshold,
                             held[out - 1] if out is not None and out > 0 else (held[-1] if out is None else None),
                             held[out] if out is not None else None,
                             out is not None and any(r >= threshold for r in ratio[out:])))
    return tuple(found)


@public
def ramp_crossings(result: StudyResult, lag: object, *, window: Quantity = s(0.2),
                   threshold: float = 0.99) -> tuple[Crossing, ...]:
    """Where each member of a study run on a ramp of a frequency (``Ramp``) changes state, read from ``PhaseLag``'s
    recorded ``spin`` (``lag``; the study records it every step): the windowed mean spin over the mean frequency the
    ramp commanded over the window, each window ``window`` long and centred at a step's end after the
    hold, against ``threshold``. A rising ramp's first window below it is where the body steps out (``ramp up``); a
    falling ramp's first window at or above it where it is caught (``ramp down``). Refused for a study run from rest
    (``study.protocol-quantity``)."""
    if not isinstance(result.protocol, Ramp):
        raise ArgumentRefused([Refusal(
            "study.protocol-quantity", RefusalClass.USER,
            "ramp_crossings reads a ramp's members, and this study ran from rest",
            "read it with step_out(result, lag, field.commands.frequency), the step-out from rest")])
    found: list[Crossing] = []
    width = _seconds(window, "window")
    for member in range(len(result)):
        run = result.member(member)
        spin = cast(Series, run[_lag_id(lag)].spin)
        series = np.asarray(spin.raw, dtype=np.float64)
        steps = np.asarray(spin.steps, dtype=np.int64)
        if not np.array_equal(steps, np.arange(1, len(steps) + 1)):
            raise ArgumentRefused([Refusal(
                "study.series-every-step", RefusalClass.USER,
                "ramp_crossings reads the spin at every step from the start, and the study recorded it more sparsely",
                "record it every step: sweep(..., record=[\"<lag>.spin\"]) with no every=")])
        dt = float(run.dt)
        drive = result.ramps[member]
        assert drive is not None
        turns = commanded(drive, len(steps), dt)
        count = round(width / dt)
        sums = np.concatenate([[0.0], np.cumsum(series)])
        commanded_sums = np.concatenate([[0.0], np.cumsum(turns)])
        ends = np.arange(count, len(series) + 1)
        ratio = (sums[ends] - sums[ends - count]) / (commanded_sums[ends] - commanded_sums[ends - count])
        centre = (ends - count / 2.0) * dt
        hold = _ramp_numbers(drive)[3]
        start, to, _, _ = _ramp_numbers(drive)
        rising = to > start
        after = centre > hold
        crossed = (ratio < threshold) if rising else (ratio >= threshold)
        hits = np.flatnonzero(after & crossed)
        at = time = None
        if hits.size:
            time = float(centre[hits[0]])
            at = float(ramp_value(drive, np.asarray([time]))[0])
        found.append(Crossing(member, "ramp up" if rising else "ramp down", width, threshold, at, time))
    return tuple(found)


__all__ = ["Crossing", "FromRest", "Protocol", "Ramp", "StepOut", "commanded", "declared_with", "protocol_of",
           "ramp_crossings", "ramp_value", "step_out"]
