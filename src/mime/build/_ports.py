"""An apparatus's declared ports, as phase 1 resolves them and phase 2 builds them (``mime.things.CommandPort``,
``mime.things.Measurement``).

**Phase 1.** Each command port is checked against the command it carries (its unit and shape restated, its bounds,
rest and slew in the command's dimension, its rate and latency whole numbers of base steps, a reason wherever a number
is given) and resolved to a :class:`CommandPortPlan`. A port is *staged* where anything of it is not ideal: a period
above one step, a latency, declared bounds or a slew. Where it declares no bounds, its receiver's stated ones are its
bounds (``Command.bounds``: a kinematic arm's joint limits), applied before the slew wherever the port is staged for
something else; they stage nothing by themselves. The command's drive, or the controller's write, then writes the
stage's input, and the stage writes the receiver's. Each measurement port
is resolved to its source (a state a dynamics publishes, or an observation's output) and to a
:class:`MeasurementPlan`; a controller reads it as ``<apparatus>.<port>`` and its age as ``<apparatus>.<port>.age``.

**Phase 2** (``mime.build._stages``) builds at most two nodes per apparatus (``ports:<apparatus>``,
``mime.nodes.control.CommandStageNode``; ``measure:<apparatus>``, ``MeasureStageNode``), each only where it holds a
staged port, with their edges: stage to receiver (a command), source to stage (a state). The schedule follows the
edges, so the command stage runs after its writer and before its receiver, and the measure stage after its source; a
controller inside the compiled step reads it at the start of the step, as it reads any state. No port, no node, no
edge: every program is the program it was."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from mime.build._refusals import RefusalClass
from mime.gradients import GradientStatus, composed
from mime.nodes.control import CLIPPED, COMMAND_STAGE, MEASURE_STAGE, TRIPPED, UNBOUNDED
from mime.things import Apparatus, CommandPort, Gaussian, Measurement
from mime.things._ports import CLIP, HOLDS, ON_BOUND, RESPONSE_BUILT_IN, SEAMS_BUILT_IN, TRIP
from mime.units import FREQUENCY, TIME, Dimension, Magnitude, Quantity, Unit, UnitError, si, unit_symbol

if TYPE_CHECKING:
    from mime.build._plan import EntryPlan
    from mime.registry import Command

#: A measurement port's name: lower-case words, as a controller's read label is.
_PORT_NAME = re.compile(r"[a-z][a-z0-9_]*")
#: The names a stage's own fields take, which no port may be named.
RESERVED = frozenset({"steps"})
_RTOL = 1e-9


def numbers(value: Magnitude | None) -> str:
    """A port's number, or one per element, as the report prints it."""
    if isinstance(value, tuple):
        return f"({', '.join(f'{c:.6g}' for c in value)})"
    return f"{value:.6g}" if value is not None else "none"


def stage_id(kind: str, apparatus: str) -> str:
    """A stage's node name and id: ``ports:<apparatus>`` or ``measure:<apparatus>``, as a drive's is
    ``drive:<apparatus>.<command>``."""
    return f"{kind}:{apparatus}"


def field_of(name: str) -> str:
    """A port's fields' stem in its stage: its command or name with each dot an underscore."""
    return name.replace(".", "_")


@dataclass(frozen=True)
class CommandPortPlan:
    """A command port as phase 1 resolved it: the apparatus and the command it carries, its receiver's node and input,
    the command's units and dimension, and its shape (``-1`` where the binding decides it) and the shape it restates;
    its period and latency in base steps, its bound (``""``, ``"clip"`` or ``"trip"``) and bounds, rest and slew in SI;
    its reason."""

    apparatus: str
    command: str
    receiver: tuple[str, str]
    units: str
    dimension: Dimension
    shape: tuple[int, ...]
    restated: tuple[int, ...] | None = None
    period: int = 1
    latency: int = 0
    bounded: str = UNBOUNDED
    low: Magnitude | None = None
    high: Magnitude | None = None
    rest: Magnitude = 0.0
    slew: Magnitude | None = None
    reason: str | None = None
    #: Whether the port declares its bounds; otherwise they are its receiver's stated ones (``Command.bounds``), which
    #: its stage applies where it is staged for something else, and which stage nothing by themselves.
    bounds_declared: bool = False

    @property
    def key(self) -> str:
        return field_of(self.command)

    @property
    def staged(self) -> bool:
        """Whether anything of it is not ideal in the stage: it then writes into the stage, which writes its
        receiver."""
        return (self.period > 1 or self.latency > 0 or (self.bounded != UNBOUNDED and self.bounds_declared)
                or self.slew is not None)

    @property
    def stage(self) -> str:
        return stage_id(COMMAND_STAGE, self.apparatus)

    def text(self, dt: float) -> str:
        unit = "" if self.units == "1" else f" {self.units}"
        parts = [f"every {self.period} step{'s' * (self.period > 1)}" if self.period > 1 else "",
                 f"latency {self.latency} step{'s' * (self.latency > 1)} ({self.latency * dt!r} s)"
                 if self.latency else "",
                 f"{'clipped' if self.bounded == CLIPPED else 'tripped to rest'} at {numbers(self.low)} to "
                 f"{numbers(self.high)}{unit}" + ("" if self.bounds_declared else " (its receiver's stated bounds)")
                 if self.bounded and (self.bounds_declared or self.staged) else "",
                 f"slew {numbers(self.slew)}{unit}/s" if self.slew is not None else ""]
        held = "; ".join(part for part in parts if part) or "ideal"
        return f"{self.apparatus}.{self.command} -> {self.receiver[0]}.{self.receiver[1]}: {held}"


@dataclass(frozen=True)
class MeasurementPlan:
    """A measurement port as phase 1 resolved it: the apparatus and the port's name, its source's node and field (and
    the field of the source's own age, for a source with a clock of its own), what it is, its units, dimension and
    the shape it restates; its period and latency in base steps; its noise's standard deviation and its quantum in SI;
    its seed; the model of the source's own noise it stands in for, where it adds noise to a value before it; its
    reason."""

    apparatus: str
    port: str
    source: tuple[str, str]
    of: str
    described: str
    units: str
    dimension: Dimension | None
    shape: tuple[int, ...] | None = None
    restated: tuple[int, ...] | None = None
    age: tuple[str, str] | None = None
    period: int = 1
    latency: int = 0
    sigma: Magnitude | None = None
    quantum: float | None = None
    seed: int = 0
    stands_in_for: str = ""
    reason: str | None = None
    #: The period of its source's own clock in base steps (a camera's frame period), ``None`` for a state.
    clock: int | None = None

    @property
    def stage(self) -> str:
        return stage_id(MEASURE_STAGE, self.apparatus)

    @property
    def ideal(self) -> bool:
        """Whether it delivers its source as it is: every value its source makes, at once, with no noise or quantum.
        An ideal port builds nothing, and a controller reading it reads its source, unless it reads its age, which
        the stage counts."""
        return (self.period == (self.clock or 1) and self.latency == 0 and self.sigma is None
                and self.quantum is None)

    @property
    def key(self) -> str:
        return self.port

    def text(self, dt: float) -> str:
        unit = "" if self.units == "1" else f" {self.units}"
        parts = [f"every {self.period} step{'s' * (self.period > 1)}" if self.period > 1 else "",
                 f"latency {self.latency} step{'s' * (self.latency > 1)} ({self.latency * dt!r} s)"
                 if self.latency else "",
                 f"Gaussian noise of sigma {numbers(self.sigma)}{unit}" if self.sigma is not None else "",
                 f"quantum {numbers(self.quantum)}{unit}" if self.quantum is not None else ""]
        held = "; ".join(part for part in parts if part) or "ideal (read from its source unless its age is read)"
        note = f" (in place of {self.stands_in_for}, which this read bypasses)" if self.stands_in_for else ""
        return f"{self.apparatus}.{self.port} <- {self.of} ({self.described}): {held}{note}"


@dataclass(frozen=True)
class StagePlan:
    """An apparatus's staged ports: its command ports that build its command stage, and its measurement ports, which
    build its measure stage."""

    apparatus: str
    commands: tuple[CommandPortPlan, ...] = ()
    measurements: tuple[MeasurementPlan, ...] = ()

    def node_names(self) -> list[str]:
        return ([stage_id(COMMAND_STAGE, self.apparatus)] if self.commands else []) + \
            ([stage_id(MEASURE_STAGE, self.apparatus)] if self.measurements else [])

    def status(self, node: str) -> GradientStatus:
        """How a derivative passes through the stage ``node``: the command stage's bounds and slew are kinks, its trip
        a jump; the measure stage's noise is exact with its keys held, its quantum cut (a rounding); the clock's samples
        and delays exact."""
        found: list[GradientStatus] = []
        if node == stage_id(COMMAND_STAGE, self.apparatus):
            for plan in self.commands:
                found += [GradientStatus.PIECEWISE_JUMP] if plan.bounded == TRIPPED else \
                    [GradientStatus.PIECEWISE_KINK] if plan.bounded == CLIPPED else []
                found += [GradientStatus.PIECEWISE_KINK] if plan.slew is not None else []
        else:
            for measured in self.measurements:
                found += [GradientStatus.CUT] if measured.quantum is not None else []
                found += [GradientStatus.FIXED_KEY] if measured.sigma is not None else []
        return composed(found)


class Planning(Protocol):
    """What phase 1's planner gives the ports' planning."""

    dt: float

    def refuse(self, code: str, category: RefusalClass, message: str, action: str) -> None: ...



def _seconds(value: float) -> str:
    return f"{value:.6g} s"


def steps(planner: Planning, value: object, what: str, *, period: bool) -> int | None:
    """``value`` (a time, or for a period a frequency) as a whole number of base steps, at least one for a period;
    ``None`` refused (``port.clock``), naming the nearest."""
    if value is None:
        return 1 if period else 0
    try:
        if period and isinstance(value, Quantity) and value.dimension == FREQUENCY:
            frequency = si(value, FREQUENCY, what=what)
            seconds = 1.0 / frequency if isinstance(frequency, float) and frequency > 0 else math.nan
        else:
            seconds = si(value, TIME, what=what)
    except UnitError as error:
        planner.refuse("port.clock", RefusalClass.USER, str(error),
                       "write a period or a latency as a time, such as ms(2)" + (", or a rate as a frequency, such as "
                                                                                 "Hz(500)" if period else ""))
        return None
    ratio = seconds / planner.dt if isinstance(seconds, float) and planner.dt else math.nan
    whole = round(ratio) if math.isfinite(ratio) else -1
    least = 1 if period else 0
    if whole >= least and abs(ratio - whole) <= _RTOL * max(ratio, 1.0):
        return whole
    nearest = sorted({max(least, math.floor(ratio)), max(least, math.ceil(ratio))}) if math.isfinite(ratio) else [least]
    planner.refuse("port.clock", RefusalClass.USER,
                   f"{what} is {value}, not a whole number of base steps dt = {_seconds(planner.dt)}"
                   + (" from one on" if period else " from zero on"),
                   "write rate= or latency= as a whole number of base steps; the nearest are " + " and ".join(
                       _seconds(k * planner.dt) for k in nearest))
    return None


def _seams(planner: Planning, port: CommandPort | Measurement, where: str) -> bool:
    named = [name for name in port.seams if getattr(port, name) is not None]
    for name in named:
        planner.refuse("port.seam", RefusalClass.USER,
                       f"{where} declares {name}=, a seam of the port that is not built yet: it lands in "
                       f"{SEAMS_BUILT_IN}", f"remove {name}= from the port")
    return not named


def _restated(planner: Planning, port: CommandPort | Measurement, dimension: Dimension, shape: tuple[int, ...] | None,
              units: str, where: str) -> bool:
    """Whether the port's restated unit and shape agree with what it carries (a shape of ``-1`` the binding decides,
    and a shape ``None`` known only once built, are checked there)."""
    held = True
    if port.unit is not None and (not isinstance(cast(object, port.unit), Unit) or port.unit.dimension != dimension):
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where} restates its unit as {port.unit}, and it carries {units} ({dimension})",
                       f"restate the unit it carries, unit={units}, or leave unit= out")
        held = False
    restated = port.shape
    if restated is not None and (not isinstance(cast(object, restated), tuple)
                                 or not all(isinstance(cast(object, n), int) and n >= 0 for n in restated)):
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where} restates its shape as {restated!r}, not a tuple of whole numbers",
                       "restate the shape as a tuple, shape=(3,), or leave shape= out")
        return False
    if restated is not None and shape is not None and (len(restated) != len(shape) or any(
            want not in (-1, got) for want, got in zip(shape, restated, strict=False))):
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where} restates its shape as {restated}, and it carries {shape}",
                       f"restate the shape it carries, shape={shape}, or leave shape= out")
        held = False
    return held


def _number(planner: Planning, value: object, dimension: Dimension, what: str) -> Magnitude | None:
    try:
        return si(value, dimension, what=what)
    except UnitError as error:
        planner.refuse("port.declaration", RefusalClass.USER, str(error),
                       f"write it as a quantity of {dimension}, from mime.units: A(25.0), A(4000.0) / s(1.0)")
    except TypeError as error:
        planner.refuse("port.declaration", RefusalClass.USER, f"{what}: {error}",
                       f"write it as a quantity of {dimension}, from mime.units: A(25.0), A(4000.0) / s(1.0)")
    return None


def _elements(value: Magnitude) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


def _reasoned(planner: Planning, port: CommandPort | Measurement, given: Sequence[str], where: str) -> bool:
    if given and not (isinstance(port.reason, str) and port.reason.strip()):
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where} declares {', '.join(given)} with no reason: a port's numbers are declared from a "
                       f"datasheet or a measurement, which its reason names",
                       "give the port reason=\"<where its numbers come from>\"")
        return False
    return True


def command_port(planner: Planning, apparatus: str, command: str, port: CommandPort, receiver: tuple[str, str],
                 spec: Command) -> CommandPortPlan | None:
    """``port``, on ``apparatus``'s ``command`` (taken by ``receiver``, as ``spec`` declares it), resolved; ``None``
    where it is refused."""
    where = f"apparatus {apparatus!r}'s port on {command!r}"
    units = unit_symbol(spec.dimension)
    held = _seams(planner, port, where)
    if port.response is not None:
        planner.refuse("port.seam", RefusalClass.USER,
                       f"{where} declares response=, the delivery's last rung (a coil's L/R lag, a servo), which is not "
                       f"built yet: it lands in {RESPONSE_BUILT_IN}, and a response declared now would be ignored",
                       "remove response= from the port; a kinematic arm's servo is its own setting meanwhile, "
                       "Arm.Kinematic(servo=...)")
        held = False
    held = _restated(planner, port, spec.dimension, spec.shape, units, where) and held
    if port.on_bound not in ON_BOUND or port.hold not in HOLDS:
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where}: on_bound={port.on_bound!r}, hold={port.hold!r}: a port's bound is one of "
                       f"{list(ON_BOUND)}, and its hold one of {list(HOLDS)}",
                       "write on_bound=\"clip\" or \"trip\", and leave hold= at its default")
        held = False
    given = [name for name in port.numbers if getattr(port, name) is not None]
    held = _reasoned(planner, port, given, where) and held
    low = high = None
    bounded = UNBOUNDED
    if port.bounds is not None:
        pair = port.bounds if isinstance(port.bounds, tuple) else (-port.bounds, port.bounds)
        if len(pair) != 2:
            planner.refuse("port.declaration", RefusalClass.USER, f"{where}: bounds={port.bounds!r} is not (low, high)",
                           "write bounds=(low, high), or one quantity q for (-q, q)")
            return None
        low = _number(planner, pair[0], spec.dimension, f"{where}: its low bound")
        high = _number(planner, pair[1], spec.dimension, f"{where}: its high bound")
        if low is None or high is None:
            return None
        lows, highs = _elements(low), _elements(high)
        if not (len(lows) == len(highs) or 1 in (len(lows), len(highs))) or any(
                not (math.isfinite(a) and math.isfinite(b) and a <= b)
                for a, b in zip(np.broadcast_to(lows, max(len(lows), len(highs))),
                                np.broadcast_to(highs, max(len(lows), len(highs))), strict=True)):
            planner.refuse("port.declaration", RefusalClass.USER,
                           f"{where}: its bounds {low!r} to {high!r} are not finite pairs, each low at most its high",
                           "write bounds=(low, high): one pair, or one pair per element")
            return None
        bounded = CLIPPED if port.on_bound == CLIP else TRIPPED
    elif spec.bounds is not None:
        low, high = spec.bounds
        bounded = CLIPPED if port.on_bound == CLIP else TRIPPED
    elif port.on_bound == TRIP:
        planner.refuse("port.declaration", RefusalClass.USER, f"{where}: on_bound=\"trip\" with no bounds to trip at",
                       "give the port bounds=(low, high), or leave on_bound= out")
        held = False
    rest = _number(planner, port.rest, spec.dimension, f"{where}: its rest") if port.rest is not None else \
        (spec.rest if spec.rest is not None else 0.0)
    slew = _number(planner, port.slew, spec.dimension / TIME, f"{where}: its slew") if port.slew is not None else None
    if slew is not None and not all(math.isfinite(c) and c > 0.0 for c in _elements(slew)):
        planner.refuse("port.declaration", RefusalClass.USER, f"{where}: its slew {slew!r} is not above zero",
                       "write slew= as the most the delivered value moves per second, above zero")
        held = False
    period = steps(planner, port.rate, f"the rate of {where}", period=True)
    latency = steps(planner, port.latency, f"the latency of {where}", period=False)
    if rest is None or period is None or latency is None or not held or (port.slew is not None and slew is None):
        return None
    if port.rest is not None and not (bounded == TRIPPED or latency > 0 or slew is not None):
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where} declares rest={port.rest}, which nothing it declares delivers: a rest acts before a "
                       f"delayed write arrives (latency=), where a slew starts from (slew=), and after a trip "
                       f"(on_bound=\"trip\")", "leave rest= out, or declare what delivers it")
        return None
    if port.rest is not None and low is not None and high is not None and any(
            not (a <= r <= b) for r, a, b in zip(*np.broadcast_arrays(
                np.asarray(_elements(rest)), np.asarray(_elements(low)), np.asarray(_elements(high))), strict=True)):
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where} declares rest={port.rest}, outside its bounds {numbers(low)} to {numbers(high)}: the "
                       f"port would deliver a value its bounds refuse", "give rest= a value within bounds=(low, high)")
        return None
    return CommandPortPlan(apparatus, command, receiver, units, spec.dimension, spec.shape, port.shape, period, latency,
                           bounded, low, high, rest, slew, port.reason, port.bounds is not None)


def measurement_port(planner: Planning, apparatus: Apparatus, name: str, port: Measurement, seed: int,
                     observations: Mapping[str, EntryPlan], states: Mapping[str, tuple[EntryPlan, str]],
                     segments: Sequence[str]) -> MeasurementPlan | None:
    """``port``, named ``name`` on ``apparatus``, resolved to its source (an observation's output by its id,
    ``observations``; a state a dynamics publishes by its path, ``states``); ``None`` where it is refused."""
    from mime import vocabulary

    where = f"apparatus {apparatus.name!r}'s measurement {name!r}"
    if not _PORT_NAME.fullmatch(name) or name in segments or name in RESERVED:
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where}: a measurement port's name is lower-case words joined by underscores, and none of its "
                       f"apparatus's chain segments' names" + (f" ({name!r} is one)" if name in segments else ""),
                       "name it otherwise: Apparatus(..., ports={\"frames\": Measurement(of=...)})")
        return None
    held = _seams(planner, port, where)
    if port.hold not in HOLDS:
        planner.refuse("port.declaration", RefusalClass.USER, f"{where}: hold={port.hold!r} is not one of {list(HOLDS)}",
                       "leave hold= at its default")
        held = False
    of = port.of if isinstance(cast(object, port.of), str) else str(port.of)
    clock: int | None = None
    age: tuple[str, str] | None = None
    stands_in_for = noise_model = ""
    if ":" in of:
        owner, _, output_name = of.rpartition(".")
        plan = observations.get(owner)
        from mime.registry import Observation

        outputs = {o.field: o for o in cast(type[Observation], plan.registered.cls).outputs} if plan is not None else {}
        output = outputs.get(output_name)
        if plan is None or output is None:
            planner.refuse("port.source-unknown", RefusalClass.USER,
                           f"{where} measures {of!r}, which is no observation's output (written <observation "
                           f"id>.<output>)", f"write of=\"<observation id>.<output>\": "
                                             f"{', '.join(sorted(f'{o}.<output>' for o in observations)) or 'none'}")
            return None
        entry = cast(Observation, plan.entry)
        units, dimension = unit_symbol(output.dimension), output.dimension
        shape: tuple[int, ...] | None = entry.output_shape(output)
        source = (plan.node_name(output.node), output.field)
        clock = entry.clock(plan.settings, planner.dt)
        if clock is not None and "age" in outputs:
            age = (plan.node_name(outputs["age"].node), outputs["age"].field)
        described = f"observation {owner}'s {output_name}"
        noise_model, stands_in_for = output.noise_model, output.before
    else:
        found = states.get(of)
        interface = vocabulary.lookup(found[1]) if found is not None else None
        if found is None or interface is None or interface.shape is None:
            planner.refuse("port.source-unknown", RefusalClass.USER,
                           f"{where} measures {of!r}, which is no state a dynamics publishes nor an observation's output",
                           "write of=\"<thing>.<state>\", a chain segment's \"<apparatus>.<segment>.<state>\", or "
                           "\"<observation id>.<output>\"")
            return None
        dynamics, published = found
        port_of = cast(Any, dynamics.record).publishes[published]
        source = (dynamics.node_name(port_of.node), port_of.field)
        units = interface.units
        from mime.units import parse_unit

        # A composite's units list its parts (a pose's "m; 1"): it has no one dimension.
        dimension = None if ";" in units else parse_unit(units)
        shape = tuple(interface.shape)
        described = f"{published} of {dynamics.owner if dynamics.segment is not None else dynamics.thing}"
    if dimension is None and (port.unit is not None or port.noise is not None or port.quantum is not None):
        planner.refuse("port.declaration", RefusalClass.USER,
                       f"{where} measures {of}, whose units are a composite's ({units}), and declares a unit, noise or "
                       f"a quantum, each of one dimension", "leave unit=, noise= and quantum= out of a port on a "
                                                           "composite")
        return None
    held = (dimension is None or _restated(planner, port, dimension, shape, units, where)) and held
    given = [n for n in port.numbers if getattr(port, n) is not None]
    held = _reasoned(planner, port, given, where) and held
    if (port.noise is not None or port.quantum is not None) and noise_model:
        planner.refuse("port.modelled-noise", RefusalClass.USER,
                       f"{where} declares {'noise' if port.noise is not None else 'a quantum'} on {of}, which already "
                       f"carries {noise_model}: the noise would be counted twice",
                       "measure the value before it instead (a camera's expected image, camera:.expected) and give the "
                       "port its noise there, or leave noise= and quantum= out")
        return None
    sigma: Magnitude | None = None
    if port.noise is not None:
        if not isinstance(cast(object, port.noise), Gaussian):
            planner.refuse("port.declaration", RefusalClass.USER, f"{where}: noise={port.noise!r} is not a Gaussian",
                           "write noise=Gaussian(sigma=...), in the units of what it measures")
            return None
        sigma = _number(planner, port.noise.sigma, cast(Dimension, dimension), f"{where}: its noise's sigma")
        if sigma is None:
            return None
        if not all(math.isfinite(c) and c > 0.0 for c in _elements(sigma)):
            planner.refuse("port.declaration", RefusalClass.USER, f"{where}: its noise's sigma {sigma!r} is not above "
                                                                   f"zero", "write Gaussian(sigma=...) above zero")
            held = False
    quantum: float | None = None
    if port.quantum is not None:
        found_quantum = _number(planner, port.quantum, cast(Dimension, dimension), f"{where}: its quantum")
        if found_quantum is None:
            return None
        if not (isinstance(found_quantum, float) and math.isfinite(found_quantum) and found_quantum > 0.0):
            planner.refuse("port.declaration", RefusalClass.USER,
                           f"{where}: its quantum {found_quantum!r} is not one step above zero",
                           "write quantum= as one quantity above zero")
            held = False
        else:
            quantum = found_quantum
    period = steps(planner, port.rate, f"the rate of {where}", period=True) if port.rate is not None else (clock or 1)
    latency = steps(planner, port.latency, f"the latency of {where}", period=False)
    if period is not None and clock is not None and period % clock:
        planner.refuse("port.clock", RefusalClass.USER,
                       f"the rate of {where} is {period} base steps, and {of} is made every {clock}: a port on a "
                       f"source with a clock of its own samples at a whole multiple of its period, which decimates it",
                       f"write rate= a whole multiple of {clock * planner.dt:.6g} s, or leave it out (every value)")
        return None
    if period is None or latency is None or not held:
        return None
    return MeasurementPlan(apparatus.name, name, source, of, described, units, dimension, shape, port.shape, age,
                           period, latency, sigma, quantum, seed,
                           stands_in_for if (sigma is not None or quantum is not None) else "", port.reason, clock)


__all__ = ["RESERVED", "CommandPortPlan", "MeasurementPlan", "Planning", "StagePlan", "command_port", "field_of",
           "measurement_port", "stage_id", "steps"]
