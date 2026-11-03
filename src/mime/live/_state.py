"""What a view of a live session shows, as plain data: no terminal, no ``rich`` and no session object here.

:class:`ServeState` is the state of a running ``mime serve`` as a person watching it needs it: the session's status as
a client reads it (``state``, ``step``, ``time``, ``frames``, ``pace``, ``recording``, ``activity``, ``pending``, and the
optional ``wall_time``, ``clients`` and ``recent``), the rows of the table of values, the flags seen raised, the
events not yet shown, and how the session stopped. It is fed from three places, none of which a client of ``mime.live/1``
could not also reach: :meth:`ServeState.note_status` (a status reply's fields), :meth:`ServeState.take` (a frame) and the
server's own account of who connected and what was asked (:meth:`ServeState.peer`, :meth:`ServeState.command`). A later
``mime watch`` fills the same state from a status poll and the frames it subscribes to, and the one renderer
(``mime.cli._serve_view``) draws either.

**The table of values.** A row is one name a person can pass to ``--watch``: a field of the frames and, optionally, an
index (``umr.pose[0:3]``, ``rpm.arm.frame_poses[26,0:3]``). A row that is one number keeps its minimum and maximum over
**every frame** :meth:`take` is given, not over the frames that happen to be drawn: a peak between two draws is a peak.
(A frame is the state at a chunk's end, so these are not the extremes over steps.) What is drawn of a row is sampled
from the latest frame, about four times a second; only the minimum and maximum are kept per frame. A row of two or three
numbers shows them as they stand and keeps no extremes, and a row of more shows the largest magnitude and the shape.

``--watch`` chooses what is shown. It never adds to or changes the frames, which are the schema's alone: this module
reads the schema and has nothing with which to write to it."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.live._protocol import FieldSpec, Frame, FrameSchema

#: The kinds of row: one number, two or three, more than three, and a flag.
SCALAR, VECTOR, ARRAY, FLAG = "scalar", "vector", "array", "flag"
#: A row of up to this many numbers shows them all.
VECTOR_MOST = 3
#: The seconds of drawn samples the sparkline shows, and how often a draw samples (the draw rate, about).
SPARK_SECONDS = 10.0
SAMPLES_PER_SECOND = 4
#: The span the speed and the frame rate are measured over (seconds), and the shortest span they are said over.
RATE_WINDOW = 2.5
RATE_SHORTEST = 0.4
#: The most commands the state keeps of what was asked recently, and the longest text kept of one.
RECENT_MOST = 5
TEXT_MOST = 48
#: Refused connections are said once together, not one by one (a port scanner would flood the log).
REFUSED_EVERY = 5.0
#: The most events held before the oldest are dropped.
EVENTS_MOST = 256

WATCH_ALL, WATCH_NONE = "all", "none"
_SPEC = re.compile(r"(?P<path>[^\[\]]+)(?:\[(?P<index>[^\[\]]*)\])?")
_INDEX = re.compile(r"(?:(?P<one>-?\d+)|(?P<low>-?\d*):(?P<high>-?\d*))")


@dataclass(frozen=True)
class FlagWatch:
    """A flag the frames carry: its name (``<node>.<name>``), the path of the frame field it is read from, and its limits
    where it is read against them (raised when any component's magnitude is above its limit); without limits it is raised
    when any component is not zero, as :class:`mime.results.FlagLayout` reads it."""

    name: str
    path: str
    limits: tuple[float, ...] | None = None

    def raised(self, value: Any) -> bool:
        held = np.asarray(value).reshape(-1)
        if self.limits is None:
            return bool((held != 0).any())
        return bool((np.abs(held) > np.asarray(self.limits, dtype=np.float64)).any())


@dataclass
class Row:
    """One line of the table of values: its ``label`` (the name ``--watch`` takes), the frame field ``path`` it reads, the
    index into it, its kind, and its unit where the schema gives one."""

    label: str
    path: str
    kind: str
    indexer: tuple[int | slice, ...] = ()
    #: The shape of the selection (its numbers' count is the product).
    shape: tuple[int, ...] = ()
    unit: str | None = None
    #: A one-number row: the index of its number in the field's flattened value.
    flat: int = 0
    #: A one-number row's minimum and maximum over every frame taken (infinite until one is).
    low: float = float("inf")
    high: float = float("-inf")
    #: The values drawn at, one each draw, newest last.
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=int(SPARK_SECONDS * SAMPLES_PER_SECOND)))
    flag: FlagWatch | None = None

    def numbers(self, value: Any) -> np.ndarray:
        """The row's numbers in a frame's value of its field, flat."""
        held = np.asarray(value)
        return held[self.indexer].reshape(-1) if self.indexer else held.reshape(-1)

    @property
    def has_range(self) -> bool:
        """Whether a minimum and a maximum are known: a one-number row that has seen a frame it could compare."""
        return self.kind == SCALAR and self.high >= self.low


@dataclass(frozen=True)
class Event:
    """A line for the scrollback: when it happened (the clock's seconds) and what."""

    at: float
    text: str


@dataclass(frozen=True)
class Recent:
    """A command a client asked, as ``recent`` in a status reply states it: its name, its arguments and the step it took
    effect at. Nothing of who sent it."""

    command: str
    arguments: Mapping[str, Any]
    step: int


@dataclass(frozen=True)
class Stopped:
    """How a session ended: why, and where it got to."""

    reason: str
    step: int
    time: float
    wall: float
    frames: int
    out: str | None


@dataclass
class ServeState:
    """The state of a running ``mime serve``, plain data; see the module. Every method takes the state's lock, and a
    renderer holds :attr:`lock` while it reads."""

    experiment: str
    #: The run folder the session writes when it stops.
    out: str | None = None
    rows: list[Row] = field(default_factory=lambda: [])
    #: The flags of the experiment, how many it declares and the ones the frames carry.
    flags_declared: int = 0
    flags: list[FlagWatch] = field(default_factory=lambda: [])
    #: Whether the table is hidden (``--watch none``).
    table: bool = True
    #: The status, as a client reads it.
    state: str = "starting"
    step: int = 0
    time: float = 0.0
    frames: int = 0
    pace: float | None = 1.0
    recording: bool = False
    activity: str = "idle"
    pending: tuple[str, ...] = ()
    non_finite: str | None = None
    #: The session's wall time and the clock reading it was read at; ``None`` from a server that does not say.
    wall_time: tuple[float, float] | None = None
    #: The peers connected: commanders, then viewers; ``None`` from a server that does not say.
    clients: tuple[int, int] | None = None
    recent: tuple[Recent, ...] = ()
    #: The frame most recently taken, and how many have been.
    latest: Frame | None = None
    frames_taken: int = 0
    #: The step each flag was first seen raised at (a frame is a chunk's end).
    raised: dict[str, int] = field(default_factory=lambda: {})
    #: Set when the session has begun to stop, and when it has stopped.
    stopping: str | None = None
    stopped: Stopped | None = None
    events: deque[Event] = field(default_factory=lambda: deque(maxlen=EVENTS_MOST))
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _rates: deque[tuple[float, float, int]] = field(default_factory=lambda: deque(), repr=False)
    _refused: int = field(default=0, repr=False)
    _refused_said: float = field(default=float("-inf"), repr=False)
    _scalars: list[Row] = field(default_factory=lambda: [], repr=False)
    _pending_flags: list[FlagWatch] = field(default_factory=lambda: [], repr=False)

    def watch(self, rows: Sequence[Row], flags: Sequence[FlagWatch], declared: int, *, table: bool = True) -> None:
        """Set the table's rows (:func:`watch_rows`) and the flags the frames carry (of ``declared``)."""
        with self.lock:
            self.rows = list(rows)
            self.table = table
            self.flags = list(flags)
            self.flags_declared = declared
            self._scalars = [row for row in self.rows if row.kind == SCALAR]
            self._pending_flags = list(flags)

    # -- what feeds it ------------------------------------------------------------------------------------------------

    def note_status(self, status: Mapping[str, Any], at: float) -> None:
        """A status as a client reads it (a reply's fields), read at ``at`` (a monotonic clock's seconds)."""
        with self.lock:
            self.state = str(status["state"])
            self.step = int(status["step"])
            self.time = float(status["time"])
            self.frames = int(status["frames"])
            pace = status.get("pace")
            self.pace = float(pace) if isinstance(pace, (int, float)) else None
            self.recording = bool(status.get("recording"))
            self.activity = str(status.get("activity", "idle"))
            self.pending = tuple(str(name) for name in status.get("pending", ()))
            held = status.get("non_finite")
            self.non_finite = str(held) if held is not None else None
            wall = status.get("wall_time")
            self.wall_time = (float(wall), at) if isinstance(wall, (int, float)) else None
            clients = status.get("clients")
            counts = cast(Mapping[str, Any], clients) if isinstance(clients, Mapping) else None
            self.clients = (int(counts.get("commands", 0)), int(counts.get("frames", 0))) if counts is not None else None
            self.recent = _recent_of(status.get("recent"))
            self._rates.append((at, self.time, self.frames))
            while self._rates and at - self._rates[0][0] > RATE_WINDOW:
                self._rates.popleft()

    def take(self, frame: Frame) -> None:
        """A frame, as it is published: the latest is kept, a one-number row's extremes are updated and a flag is looked
        at. This is the only work done per frame, and it is a few operations per row."""
        with self.lock:
            self.latest = frame
            self.frames_taken += 1
            values = frame.values
            for row in self._scalars:
                number = values[row.path].item(row.flat)
                if number < row.low:
                    row.low = number
                if number > row.high:
                    row.high = number
            if self._pending_flags:
                for flag in self._pending_flags:
                    if flag.raised(values[flag.path]):
                        self.raised[flag.name] = frame.step
                self._pending_flags = [flag for flag in self._pending_flags if flag.name not in self.raised]

    def sample(self) -> None:
        """Called at each draw: the latest frame's value of each one-number row joins its history."""
        with self.lock:
            if self.latest is None:
                return
            values = self.latest.values
            for row in self._scalars:
                row.samples.append(float(values[row.path].item(row.flat)))

    def event(self, text: str, at: float | None = None) -> None:
        with self.lock:
            self.events.append(Event(time.time() if at is None else at, text))

    def peer(self, what: str, role: str, address: str, dropped: int = 0) -> None:
        """A peer connected, left or was refused. A refused connection is counted and said with the others after a few
        seconds, never one by one."""
        who = "commander" if role == "commands" else "viewer"
        with self.lock:
            if what == "refused":
                self._refused += 1
            elif what == "connected":
                self.event(f"{who} connected ({address})")
            else:
                lost = f"; {dropped} frames dropped for it" if dropped else ""
                self.event(f"{who} left{lost}")

    def command(self, message: Mapping[str, Any], answer: Mapping[str, Any]) -> None:
        """A command a client asked and what came of it, as one event. ``hello`` and ``status`` are polled, and are not
        events. Nothing of the peer but what it asked: the name and its arguments."""
        name = message.get("command")
        if name in ("hello", "status"):
            return
        shown = _cut(str(name))
        if not answer.get("ok"):
            errors = cast(Sequence[Mapping[str, Any]], answer.get("errors") or [{}])
            code = _cut(str(errors[0].get("code", "refused")))
            self.event(f"refused: {shown} ({code})")
            return
        step = int(answer.get("step", self.step))
        if name in ("stop", "pause") and (name in answer.get("pending", ())):
            effect = "taken; it acts at the chunk's end"
        elif name == "stop":
            effect = "stopping"
        elif name == "pause":
            effect = f"paused at step {step:,}"
        elif name == "start":
            effect = f"running from step {step:,}"
        elif name == "step":
            effect = f"stepped to step {step:,}"
        else:
            effect = f"from step {step:,}"
        self.event(f"command: {_called(name, message)} ({effect})")

    def stop_begins(self, reason: str) -> None:
        with self.lock:
            if self.stopping is None:
                self.stopping = reason

    def stop_ends(self, reason: str, step: int, time_: float, wall: float, frames: int, out: str | None) -> None:
        with self.lock:
            self.stopped = Stopped(self.stopping or reason, step, time_, wall, frames, out)

    # -- what reads it ------------------------------------------------------------------------------------------------

    def drain(self, now: float) -> list[Event]:
        """The events not yet shown, oldest first, with the refused connections said as one line when a few seconds have
        passed since they last were."""
        with self.lock:
            if self._refused and now - self._refused_said >= REFUSED_EVERY:
                self.event(f"{self._refused} connection{'s' if self._refused != 1 else ''} refused (not ZMTP, or no "
                           "handshake)")
                self._refused, self._refused_said = 0, now
            held = list(self.events)
            self.events.clear()
            return held

    def wall(self, now: float) -> float | None:
        """The session's wall time at ``now`` (a monotonic clock's seconds), carried on from when it was last read."""
        with self.lock:
            return None if self.wall_time is None else self.wall_time[0] + max(0.0, now - self.wall_time[1])

    def _span(self) -> tuple[tuple[float, float, int], tuple[float, float, int]] | None:
        if len(self._rates) < 2:
            return None
        first, last = self._rates[0], self._rates[-1]
        return (first, last) if last[0] - first[0] >= RATE_SHORTEST else None

    def real_time_factor(self) -> float | None:
        """Simulated seconds per wall second over the last few seconds; ``None`` before there is a span to say it over."""
        with self.lock:
            span = self._span()
            return None if span is None else (span[1][1] - span[0][1]) / (span[1][0] - span[0][0])

    def frames_per_second(self) -> float | None:
        with self.lock:
            span = self._span()
            return None if span is None else (span[1][2] - span[0][2]) / (span[1][0] - span[0][0])


def _recent_of(held: Any) -> tuple[Recent, ...]:
    """``recent`` as a status states it: each command's name, arguments and step."""
    if not isinstance(held, Sequence) or isinstance(held, str):
        return ()
    found: list[Recent] = []
    for each in cast(Sequence[Any], held):
        if isinstance(each, Mapping):
            entry = cast(Mapping[str, Any], each)
            found.append(Recent(_cut(str(entry["command"])), dict(cast(Mapping[str, Any], entry.get("arguments", {}))),
                                int(entry["step"])))
    return tuple(found)


def _cut(text: str) -> str:
    return text if len(text) <= TEXT_MOST else text[:TEXT_MOST - 1] + "…"


def _called(name: object, message: Mapping[str, Any]) -> str:
    """A command as it was asked: its name and its arguments, ``set offset=0.001``."""
    words = [_cut(str(name))]
    if name == "set" and isinstance(message.get("values"), Mapping):
        values: Mapping[str, Any] = message["values"]
        words += [f"{_cut(str(label))}={_cut(str(value))}" for label, value in values.items()]
    elif name == "step":
        words.append(f"{_cut(str(message.get('chunks', 1)))} chunks")
    elif name == "record":
        words.append("on" if message.get("on", True) else "off")
    elif name == "pace":
        rtf = message.get("rtf")
        words.append("unpaced" if rtf is None else f"{_cut(str(rtf))}x")
    return " ".join(words)


# -- the rows ---------------------------------------------------------------------------------------------------------


def _unit_of(spec: FieldSpec, indexer: tuple[int | slice, ...]) -> str | None:
    """The unit of a selection: the field's, or for a pose (``m; 1``: metres, then unitless) the part that covers the
    columns selected, when they are all of one part."""
    if spec.unit is None:
        return None
    parts = [part.strip() for part in spec.unit.split(";")]
    if len(parts) != 2 or not spec.shape or spec.shape[-1] != 7:
        return spec.unit
    last = indexer[-1] if len(indexer) == len(spec.shape) else slice(None)
    columns = [range(7)[last]] if isinstance(last, int) else list(range(7)[last])
    covered = {0 if column < 3 else 1 for column in columns}
    return parts[covered.pop()] if len(covered) == 1 else spec.unit


def _parse(spec: str, schema: FrameSchema) -> Row:
    """``spec`` as a row of ``schema``'s frames: ``<node>.<field>`` and, in brackets, an integer or a ``low:high`` slice
    for each leading dimension, ``umr.pose[0:3]``, ``rpm.arm.frame_poses[26,0:3]``."""
    from mime.things import did_you_mean

    found = _SPEC.fullmatch(spec.strip())
    if found is None:
        raise ArgumentRefused([Refusal(
            "live.watch", RefusalClass.USER, f"--watch {spec!r} is not <node>.<field> or <node>.<field>[index, ...]",
            "name a field of the frames, with an index where it is not one number, as `--watch <node>.<field>[<low>:<high>]`")])
    path = found["path"].strip()
    fields = {each.path: each for each in schema.fields}
    if path not in fields:
        raise ArgumentRefused([Refusal(
            "live.watch", RefusalClass.USER, f"--watch names {path!r}, which the frames do not carry",
            f"{did_you_mean(path, fields)} Add a field to the frames with `--select <node>.<field>`",
            why="--watch chooses which of the frame's fields the table shows; it adds nothing to the frames")])
    held = fields[path]
    texts = [each.strip() for each in found["index"].split(",")] if found["index"] is not None else []
    if len(texts) > len(held.shape):
        raise ArgumentRefused([Refusal(
            "live.watch", RefusalClass.USER,
            f"--watch {spec!r} gives {len(texts)} indices to {path}, which has {len(held.shape)} "
            f"dimension{'s' if len(held.shape) != 1 else ''} (shape {held.shape})",
            f"give at most {len(held.shape)}: `--watch {path}[{','.join(['0'] * len(held.shape))}]`")])
    indexer: list[int | slice] = []
    for dimension, text in enumerate(texts):
        size = held.shape[dimension]
        part = _INDEX.fullmatch(text)
        if part is None:
            raise ArgumentRefused([Refusal(
                "live.watch", RefusalClass.USER, f"--watch {spec!r}: {text!r} is not an integer or a low:high slice",
                "write each index as an integer, a `<low>:<high>` slice, or `:` for all of the dimension")])
        if part["one"] is not None:
            at = int(part["one"])
            if not -size <= at < size:
                raise ArgumentRefused([Refusal(
                    "live.watch", RefusalClass.USER,
                    f"--watch {spec!r}: index {at} is outside dimension {dimension} of {path}, which has {size}",
                    f"use an index from {-size} to {size - 1}: `--watch {path}[{size - 1}]`")])
            indexer.append(at)
        else:
            cut = slice(int(part["low"]) if part["low"] else None, int(part["high"]) if part["high"] else None)
            if len(range(size)[cut]) == 0:
                raise ArgumentRefused([Refusal(
                    "live.watch", RefusalClass.USER,
                    f"--watch {spec!r}: the slice {text!r} selects nothing of dimension {dimension} of {path}, which "
                    f"has {size}", f"use a slice inside `0:{size}`: `--watch {path}[0:{size}]`")])
            indexer.append(cut)
    shape = np.broadcast_to(np.zeros((), dtype=np.int8), held.shape)[tuple(indexer)].shape if indexer else held.shape
    count = int(np.prod(shape, dtype=np.int64))
    label = path + (f"[{','.join(_written(each) for each in indexer)}]" if indexer else "")
    chosen = tuple(indexer)
    kind = SCALAR if count == 1 else VECTOR if count <= VECTOR_MOST else ARRAY
    row = Row(label, path, kind, chosen, shape, _unit_of(held, chosen))
    if kind == SCALAR:
        row.flat = _flat(held.shape, chosen)
    return row


def _written(each: int | slice) -> str:
    if isinstance(each, int):
        return str(each)
    return f"{'' if each.start is None else each.start}:{'' if each.stop is None else each.stop}"


def _flat(shape: tuple[int, ...], indexer: tuple[int | slice, ...]) -> int:
    """The position in the flattened field of the one number a selection of one number picks."""
    at: list[int] = []
    for dimension, size in enumerate(shape):
        each: int | slice = indexer[dimension] if dimension < len(indexer) else slice(None)
        at.append(each % size if isinstance(each, int) else range(size)[each][0])
    return int(np.ravel_multi_index(tuple(at), shape)) if shape else 0


def default_rows(schema: FrameSchema, bound: Sequence[Mapping[str, Any]], flags: Sequence[FlagWatch]) -> list[Row]:
    """The table when nothing is asked for: each one-number field that is not a flag (an observation's per-step outputs),
    each flag, and the position, the first three numbers, of each body the stage binds. ``bound`` is the stage binding's
    entries as ``hello`` gives them (``mime.binding/1``)."""
    flagged = {flag.path for flag in flags}
    fields = {each.path: each for each in schema.fields}
    rows = [_parse(each.path, schema) for each in schema.fields if not each.shape and each.path not in flagged]
    rows += [Row(flag.name, flag.path, FLAG, flag=flag, unit="1") for flag in flags]
    for entry in bound:
        path = f"{entry['node']}.{entry['field']}"
        if entry["kind"] == "pose" and entry["index"] is None and path in fields and fields[path].shape == (7,):
            rows.append(_parse(f"{path}[0:3]", schema))
    return rows


def watch_rows(schema: FrameSchema, specs: Sequence[str] | None, bound: Sequence[Mapping[str, Any]],
               flags: Sequence[FlagWatch]) -> tuple[list[Row], bool]:
    """The rows of the table, and whether it is shown, for what ``--watch`` was given (``specs``, repeatable).

    Nothing given: :func:`default_rows`. ``all``: those and every other one-number or three-number field. ``none``: no
    table. Anything else names the rows, in the order given. A name the frames do not carry, or an index outside a
    field, is refused (``live.watch``) with the nearest name. A flag is named by the frame field it is read from or by
    its own name, the label of its row. The schema is read, never written."""
    given = [each for each in (specs or ()) if each.strip()]
    if not given:
        return default_rows(schema, bound, flags), True
    if WATCH_NONE in given:
        if len(given) > 1:
            raise ArgumentRefused([Refusal(
                "live.watch", RefusalClass.USER, "--watch none cannot be given with other names",
                "give `--watch none` alone, or name the rows without it")])
        return [], False
    rows: list[Row] = []
    seen: set[str] = set()
    flagged = {flag.path: flag for flag in flags}
    named = {flag.name: flag for flag in flags}  # a flag row is labelled by its name, so that is a name --watch takes too
    for spec in given:
        if spec.strip() == WATCH_ALL:
            more = default_rows(schema, bound, flags) + [
                _parse(each.path, schema) for each in schema.fields
                if (not each.shape or each.shape == (VECTOR_MOST,)) and each.path not in flagged]
        elif spec.strip() in flagged or spec.strip() in named:
            flag = flagged.get(spec.strip()) or named[spec.strip()]
            more = [Row(flag.name, flag.path, FLAG, flag=flag, unit="1")]
        else:
            more = [_parse(spec, schema)]
        for row in more:
            if row.label not in seen:
                seen.add(row.label)
                rows.append(row)
    return rows, True


def flag_watches(schema: FrameSchema, flags: Sequence[Any]) -> list[FlagWatch]:
    """The flags of a built experiment (``built.layout.flags``) that the frames carry."""
    carried = {each.path for each in schema.fields}
    return [FlagWatch(flag.path, f"{flag.node}.{flag.field}", flag.limits) for flag in flags
            if f"{flag.node}.{flag.field}" in carried]


__all__ = ["ARRAY", "FLAG", "SCALAR", "VECTOR", "Event", "FlagWatch", "Recent", "Row", "ServeState", "Stopped",
           "default_rows", "flag_watches", "watch_rows"]
