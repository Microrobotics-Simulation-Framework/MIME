"""The port stages: an apparatus's declared links, inside the compiled step. The build inserts at most two per apparatus,
each only where a declared port is not ideal: ``ports:<apparatus>`` between each command's writer and its receiver,
and ``measure:<apparatus>`` between each measured signal's source and whatever reads it.

**The clock.** Each stage steps at the base step and counts its own steps, an int32 from zero (``steps``). A port of
period ``k`` samples where the count is a multiple of ``k`` and holds the sample between; with a latency of ``L``
steps, what is read at count ``n`` is the sample held at ``n - L``, and before that the port's rest (a command's
``rest``, a measurement's source as it starts). The samples wait in a buffer of ``ceil(L / k) + 1``, never one per
step, so an image port holds that many frames.

**A command's rungs**, in order: sample (the rate), delay (the latency), bound (clipped to the bounds, flagged
``clipped_<command>``; or tripped, delivering the rest from that step on, flagged ``tripped_<command>``), then slew (the
most the value moves in a step, ``slew dt``, from the last value the amplifier drove), whose output is what the
receiver takes. What the writer wrote is kept as ``written_<command>``. Each port's fields carry its command's name (its
dots as underscores, ``arm_joint_positions``), so a flag names the port it was raised on, whichever port's switch it is:
the switches' owners (``port.bound.low``, ``port.trip``, ...) are the stage's, one for every port.

**A measurement's**, in order: sample, delay, then Gaussian noise drawn from the port's seed and the step the sample
was made at (so a held sample keeps its noise, and rendering the run again reproduces it), then the quantum, the value
rounded to its nearest multiple. ``<port>_age`` is the base steps since the value was made: for a source with a clock
of its own (a camera), counted from the step it made the value at (its ``age`` read beside it).

Neither stage decides anything: it carries, bounds, delays and shapes. A modelled trip is not any hardware's
protection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, cast, override

import jax
from jax import Array
from jaxtyping import Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Units, as_run_array, jnp, node_fields, node_input, switch

#: The local names of the two stages beside an apparatus's id: ``ports:<apparatus>``, ``measure:<apparatus>``.
COMMAND_STAGE, MEASURE_STAGE = "ports", "measure"
#: What a command port does at its bounds, as the stage holds it: nothing, clip, or trip.
UNBOUNDED, CLIPPED, TRIPPED = "", "clip", "trip"


#: The switches the command stage evaluates, declared once here: the stage is built by the build, never by an entry,
#: so its law is its own. Its clock (the period's sample, the latency's index) is integer arithmetic on its count,
#: which no derivative passes through, and no switch.
STAGE_SWITCHES = (
    switch.Declaration(
        owner="port.bound.low",
        indicator="a command port's delayed command less its low bound, in the command's units",
        orders={"delivered": 0},
        route="none: a command below the bound is delivered at the bound, whose derivative in the command is zero; a "
              "smooth limit would be another model of the amplifier"),
    switch.Declaration(
        owner="port.bound.high",
        indicator="a command port's high bound less its delayed command, in the command's units",
        orders={"delivered": 0},
        route="none: a command above the bound is delivered at the bound, as at the low one"),
    switch.Declaration(
        owner="port.trip",
        indicator="the most a command port's delayed command lies beyond its bounds, in the command's units: it trips "
                  "where this is above zero, and holds its rest for the rest of the run",
        orders={"delivered": -1},
        route="none: a trip latches the delivered value to the port's rest, a jump; a reset is a seam of the port"),
    switch.Declaration(
        owner="port.slew.low",
        indicator="the change a command port's amplifier would make in a step plus its slew times the step, in the "
                  "command's units",
        orders={"delivered": 0},
        route="none: a change faster than the slew is delivered at the slew, whose derivative in the command is zero; a "
              "smooth rate limit would be another model of the amplifier"),
    switch.Declaration(
        owner="port.slew.high",
        indicator="the slew times the step less the change a command port's amplifier would make in it, in the "
                  "command's units",
        orders={"delivered": 0},
        route="none: as at the low side"),
)


def depth(period: int, latency: int) -> int:
    """How many samples a port of ``period`` base steps and ``latency`` base steps holds: ``ceil(L / k) + 1``."""
    return math.ceil(latency / period) + 1


class StageState(TypedDict, extra_items=ReadOnly[Array]):
    """``steps``, the stage's own count, and each port's fields, named at construction (a measurement's stamps are
    integers, its values and every command's fields floats)."""

    steps: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class StageInputs(TypedDict, extra_items=ReadOnly[Array]):
    """Each port's input, named at construction (a sampled source's age an integer)."""


def _pushed(buffer: Array, value: Array) -> Array:
    return jnp.concatenate([buffer[1:], jnp.asarray(value, buffer.dtype)[None]])


def held_sample(buffer: Array, value: Array, count: Array, period: int, latency: int) -> tuple[Array, Array, Array]:
    """The buffer after this step's sample, and the index in it of the sample read at ``count``: the one held at
    ``count - latency``, the oldest slot (filled with the rest) before any has waited that long. Where ``period`` and
    ``latency`` are both one step and none, the value itself."""
    if period == 1 and latency == 0:
        return buffer, value, jnp.asarray(0, jnp.int32)
    if period == 1:
        buffer = _pushed(buffer, value)
    else:
        buffer = jnp.where((count % period) == 0, _pushed(buffer, value), buffer)
    offset = count // period - (count - latency) // period
    index = jnp.asarray(buffer.shape[0] - 1, jnp.int32) - offset.astype(jnp.int32)
    return buffer, buffer[index], index


class CommandStageNode(MimeNode):
    """An apparatus's command ports between each command's writer and its receiver: per command (its field name,
    ``commands``), sampled at its period and delayed by its latency (``periods``, ``latencies``, base steps), bounded
    (``bounded``: ``""``, ``"clip"`` or ``"trip"``, at ``low_<c>`` and ``high_<c>``), then slewed (``slewed``, at most
    ``slew_<c>`` per second). ``rest_<c>`` is what it delivers before a write arrives and after a trip. Every number is a
    leaf of its parameters."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-134",
        description="An apparatus's command ports: each command sampled, delayed, bounded (clipped or tripped) and "
                    "slew-limited between its writer and its receiver, on an integer clock.",
        governing_equations=(r"h_{n} = u_{k \lfloor n / k \rfloor}, \quad d_{n} = h_{n - L} \ (n \ge L), \ r "
                             r"\ (n < L), \quad b_{n} = \mathrm{clip}(d_{n}, l, h), \quad a_{n} = a_{n-1} + "
                             r"\mathrm{clip}(b_{n} - a_{n-1}, -s \, \mathrm{d}t, s \, \mathrm{d}t)"),
        assumptions=("The link's clock is the base step's: every period and latency is a whole number of base steps",
                     "A trip latches for the rest of the run: no reset is modelled",
                     "The stage carries a signal, outside the energy ledger: a command's work is booked under its "
                     "writer"),
    )

    def __init__(self, name: str, timestep: float, *, commands: Sequence[str], units: Mapping[str, str],
                 shapes: Mapping[str, Sequence[int]], periods: Mapping[str, int], latencies: Mapping[str, int],
                 bounded: Mapping[str, str], slewed: Mapping[str, bool], **leaves: Any) -> None:
        keys = sorted(commands)
        super().__init__(name, timestep, commands=tuple(keys),
                         units={k: str(units[k]) for k in keys}, shapes={k: tuple(int(n) for n in shapes[k]) for k in keys},
                         periods={k: int(periods[k]) for k in keys}, latencies={k: int(latencies[k]) for k in keys},
                         bounded={k: str(bounded[k]) for k in keys}, slewed={k: bool(slewed[k]) for k in keys},
                         **{name_: _leaf(value) for name_, value in sorted(leaves.items())})

    def _held(self, key: str) -> Any:
        return self.constants()[key]

    def _keys(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self._held("commands"))

    def _shape(self, key: str) -> tuple[int, ...]:
        return cast(Mapping[str, tuple[int, ...]], self._held("shapes"))[key]

    def _buffered(self, key: str) -> bool:
        return cast(Mapping[str, int], self._held("periods"))[key] > 1 or \
            cast(Mapping[str, int], self._held("latencies"))[key] > 0

    def _rest(self, key: str, p: Mapping[str, Any] | None = None) -> Array:
        held = p if p is not None else self.constants()
        return jnp.broadcast_to(as_run_array(held[f"rest_{key}"]), self._shape(key))

    def port_fields(self, key: str) -> tuple[str, ...]:
        """The state fields and inputs the port ``key`` holds in the stage: its delivered value (its input's name too),
        what was written, and where its rungs need them its buffer, its bound's flag and its amplifier's last value."""
        bounded = cast(Mapping[str, str], self._held("bounded"))[key]
        return (key, f"written_{key}", *((f"buffer_{key}",) if self._buffered(key) else ()),
                *((f"{'clipped' if bounded == CLIPPED else 'tripped'}_{key}",) if bounded else ()),
                *((f"amplified_{key}",) if cast(Mapping[str, bool], self._held("slewed"))[key] else ()))

    @override
    def initial_state(self) -> NodeFields:
        fields: dict[str, Array] = {"steps": jnp.asarray(0, dtype=jnp.int32)}
        for key in self._keys():
            rest = self._rest(key)
            fields[key] = rest
            fields[f"written_{key}"] = rest
            if self._buffered(key):
                periods, latencies = cast(Mapping[str, int], self._held("periods")), cast(Mapping[str, int],
                                                                                          self._held("latencies"))
                fields[f"buffer_{key}"] = jnp.stack([rest] * depth(periods[key], latencies[key]))
            bounded = cast(Mapping[str, str], self._held("bounded"))[key]
            if bounded == CLIPPED:
                fields[f"clipped_{key}"] = as_run_array(0.0)
            elif bounded == TRIPPED:
                fields[f"tripped_{key}"] = as_run_array(0.0)
            if cast(Mapping[str, bool], self._held("slewed"))[key]:
                fields[f"amplified_{key}"] = rest
        return node_fields(fields)

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        units = cast(Mapping[str, str], self._held("units"))
        return {key: node_input(units[key], self._shape(key)) for key in self._keys()}

    #: The rungs a command takes, in order: the link's sample and delay, then the amplifier's bound and slew. The order
    #: is the stage's law (an amplifier chain's physical order: the link carries what was sent, the amplifier bounds and
    #: slews what arrives).
    RUNGS: ClassVar[tuple[str, ...]] = ("delay", "bound", "slew")

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        count = state["steps"]
        out: dict[str, Array] = {"steps": count + 1}
        for key in self._keys():
            written = jnp.asarray(boundary_inputs[key])
            out[f"written_{key}"] = written
            value = written
            for rung in self.RUNGS:
                value = getattr(self, f"_{rung}")(key, value, p, state, out, count, dt)
            out[key] = jnp.asarray(value, self._rest(key, p).dtype)
        return node_fields(out)

    def _delay(self, key: str, value: Array, p: Mapping[str, Any], state: NodeFields, out: dict[str, Array],
               count: Array, dt: float) -> Array:
        """The link: the writer sampled at its period and held, then read its latency late."""
        if not self._buffered(key):
            return value
        periods, latencies = cast(Mapping[str, int], self._held("periods")), cast(Mapping[str, int],
                                                                                  self._held("latencies"))
        buffer, value, _ = held_sample(state[f"buffer_{key}"], value, count, periods[key], latencies[key])
        out[f"buffer_{key}"] = buffer
        return value

    def _bound(self, key: str, value: Array, p: Mapping[str, Any], state: NodeFields, out: dict[str, Array],
               count: Array, dt: float) -> Array:
        """The amplifier's bounds: the value clipped to them, or the rest delivered from the first step beyond them."""
        bounded = cast(Mapping[str, str], self._held("bounded"))[key]
        if not bounded:
            return value
        rest = self._rest(key, p)
        low, high = (jnp.broadcast_to(as_run_array(p[f"{side}_{key}"]), self._shape(key)) for side in ("low", "high"))
        if bounded == CLIPPED:
            inside = value
            value = switch.clip(inside, low, high, owner="port.bound", indicator=inside,
                                resolution=lambda: switch.rounding(inside))
            out[f"clipped_{key}"] = jnp.any(value != inside).astype(rest.dtype)
            return value
        beyond = jnp.max(jnp.maximum(low - value, value - high))
        tripped = jnp.maximum(state[f"tripped_{key}"], (beyond > 0).astype(rest.dtype))
        out[f"tripped_{key}"] = tripped
        return switch.where(tripped > 0, rest, value, owner="port.trip", indicator=lambda: beyond,
                            resolution=lambda: switch.rounding(value))

    def _slew(self, key: str, value: Array, p: Mapping[str, Any], state: NodeFields, out: dict[str, Array],
              count: Array, dt: float) -> Array:
        """The amplifier's slew: the most the value it drives moves in a step, from where it drove it last."""
        if not cast(Mapping[str, bool], self._held("slewed"))[key]:
            return value
        before = state[f"amplified_{key}"]
        most = as_run_array(p[f"slew_{key}"]) * dt
        change = value - before
        value = before + switch.clip(change, -most, most, owner="port.slew", indicator=change,
                                     resolution=lambda: switch.rounding(change))
        out[f"amplified_{key}"] = value
        return value


class MeasureStageNode(MimeNode):
    """An apparatus's measurement ports between each measured source and whatever reads it: per port (``ports``),
    sampled at its period and delayed by its latency (``periods``, ``latencies``, base steps), Gaussian noise of
    ``sigma_<p>`` drawn from ``seeds[<p>]`` and the step the sample was made at where ``noisy``, and rounded to
    ``quantum_<p>`` where ``quantised``. A source with a clock of its own (``aged``) gives its value's age beside it,
    ``age_<p>``. ``initial_<p>`` is the source as it starts, what is read before any sample has waited the latency."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-135",
        description="An apparatus's measurement ports: each measured signal sampled, delayed, given seeded Gaussian "
                    "noise and quantised between its source and its reader, with its age, on an integer clock.",
        governing_equations=(r"h_{n} = x_{k \lfloor n / k \rfloor}, \quad d_{n} = h_{n - L}, \quad y_{n} = q \, "
                             r"\mathrm{round}((d_{n} + \sigma \, \xi_{s}) / q), \quad a_{n} = n - s"),
        assumptions=("The sampling clock is the base step's: every period and latency is a whole number of base steps",
                     "The noise is additive and Gaussian, independent between samples, drawn from the port's seed and "
                     "the sample's step: no bias, drift or dropout is modelled"),
    )

    def __init__(self, name: str, timestep: float, *, ports: Sequence[str], units: Mapping[str, str],
                 shapes: Mapping[str, Sequence[int]], periods: Mapping[str, int], latencies: Mapping[str, int],
                 aged: Mapping[str, bool], seeds: Mapping[str, int], noisy: Mapping[str, bool],
                 quantised: Mapping[str, bool], **leaves: Any) -> None:
        keys = sorted(ports)
        super().__init__(name, timestep, ports=tuple(keys), units={k: str(units[k]) for k in keys},
                         shapes={k: tuple(int(n) for n in shapes[k]) for k in keys},
                         periods={k: int(periods[k]) for k in keys}, latencies={k: int(latencies[k]) for k in keys},
                         aged={k: bool(aged[k]) for k in keys}, seeds={k: int(seeds[k]) for k in keys},
                         noisy={k: bool(noisy[k]) for k in keys}, quantised={k: bool(quantised[k]) for k in keys},
                         **{name_: _leaf(value) for name_, value in sorted(leaves.items())})

    def _held(self, key: str) -> Any:
        return self.constants()[key]

    def _keys(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self._held("ports"))

    def _shape(self, key: str) -> tuple[int, ...]:
        return cast(Mapping[str, tuple[int, ...]], self._held("shapes"))[key]

    def _buffered(self, key: str) -> bool:
        return cast(Mapping[str, int], self._held("periods"))[key] > 1 or \
            cast(Mapping[str, int], self._held("latencies"))[key] > 0

    def _initial(self, key: str) -> Array:
        return jnp.broadcast_to(as_run_array(self.constants()[f"initial_{key}"]), self._shape(key))

    def _initial_age(self, key: str) -> int:
        return int(self.constants().get(f"initial_age_{key}", 0))

    def port_fields(self, key: str) -> tuple[str, ...]:
        """The state fields and inputs the port ``key`` holds in the stage: its value (its input's name too), its age,
        and where its rungs need them its buffer and stamps, and its source's own age."""
        return (key, f"{key}_age", *((f"buffer_{key}", f"stamps_{key}") if self._buffered(key) else ()),
                *((f"age_{key}",) if cast(Mapping[str, bool], self._held("aged"))[key] else ()))

    @override
    def initial_state(self) -> NodeFields:
        fields: dict[str, Array] = {"steps": jnp.asarray(0, dtype=jnp.int32)}
        periods, latencies = cast(Mapping[str, int], self._held("periods")), cast(Mapping[str, int],
                                                                                  self._held("latencies"))
        for key in self._keys():
            start = self._initial(key)
            fields[key] = start
            fields[f"{key}_age"] = as_run_array(float(self._initial_age(key)))
            if self._buffered(key):
                held = depth(periods[key], latencies[key])
                fields[f"buffer_{key}"] = jnp.stack([start] * held)
                fields[f"stamps_{key}"] = jnp.full((held,), -self._initial_age(key), dtype=jnp.int32)
        return node_fields(fields)

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        units = cast(Mapping[str, str], self._held("units"))
        aged = cast(Mapping[str, bool], self._held("aged"))
        found = {key: node_input(units[key], self._shape(key)) for key in self._keys()}
        found.update({f"age_{key}": BoundaryInputSpec(shape=(), dtype=jnp.int32, default=jnp.asarray(0, jnp.int32),
                                                      expected_units="1",
                                                      description="the source's own age of its value, base steps")
                      for key in self._keys() if aged[key]})
        return found

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        count = state["steps"]
        periods, latencies = cast(Mapping[str, int], self._held("periods")), cast(Mapping[str, int],
                                                                                  self._held("latencies"))
        aged, seeds = cast(Mapping[str, bool], self._held("aged")), cast(Mapping[str, int], self._held("seeds"))
        out: dict[str, Array] = {"steps": count + 1}
        for key in self._keys():
            value = jnp.asarray(boundary_inputs[key])
            made = count - jnp.asarray(boundary_inputs[f"age_{key}"], jnp.int32) if aged[key] else count
            if self._buffered(key):
                buffer, value, index = held_sample(state[f"buffer_{key}"], value, count, periods[key], latencies[key])
                stamps, _, _ = held_sample(state[f"stamps_{key}"], made, count, periods[key], latencies[key])
                made = stamps[index]
                out[f"buffer_{key}"], out[f"stamps_{key}"] = buffer, stamps
            if cast(Mapping[str, bool], self._held("noisy"))[key]:
                draw = jax.random.fold_in(jax.random.PRNGKey(seeds[key]), made)
                value = value + as_run_array(p[f"sigma_{key}"]) * jax.random.normal(draw, self._shape(key),
                                                                                     dtype=value.dtype)
            if cast(Mapping[str, bool], self._held("quantised"))[key]:
                quantum = as_run_array(p[f"quantum_{key}"])
                value = jnp.round(value / quantum) * quantum
            out[key] = jnp.asarray(value, self._initial(key).dtype)
            out[f"{key}_age"] = (count - made).astype(out[key].dtype)
        return node_fields(out)


def clashes(node: CommandStageNode | MeasureStageNode) -> dict[str, list[str]]:
    """Each field of ``node`` that more than one of its ports would hold (or a port and the stage's own count,
    ``steps``), with those ports: a port named ``tr_age`` beside a port ``tr`` would write ``tr``'s age, and the later
    write would win in silence."""
    holders: dict[str, list[str]] = {"steps": ["the stage's own count"]}
    keys = cast(tuple[str, ...], node.constants()["commands" if isinstance(node, CommandStageNode) else "ports"])
    for key in keys:
        for name in node.port_fields(key):
            holders.setdefault(name, []).append(key)
    return {name: held for name, held in holders.items() if len(held) > 1}


def _leaf(value: Any) -> Any:
    """A port's number as the stage's leaf: a float, or a tuple of floats."""
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in cast(Sequence[float], value))
    return float(value)


__all__ = ["CLIPPED", "COMMAND_STAGE", "MEASURE_STAGE", "STAGE_SWITCHES", "TRIPPED", "UNBOUNDED", "CommandStageNode",
           "MeasureStageNode", "StageInputs", "StageState", "clashes", "depth", "held_sample"]
