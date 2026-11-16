"""The live session: a built experiment stepped in chunks of one fixed size, its frames handed to subscribers, its
controller's live inputs set between ticks, its commands logged with their steps so that it replays, and a minimal run
folder written when it stops.

A session given a run folder writes its command log there as it goes: the header when it opens, each command before it
acts, and after each chunk, before that chunk's frame is sent, the step it reached. Each line is flushed as it is
written, so a session that dies (a crash, a kill) leaves a log that replays to the last step it reached, and every frame
a client received is a step the log holds. The log is opened exclusively, before anything else is written into the
folder: a folder that holds one already is another session's, and is refused (``live.log-exists``), never written over.

A recording holds at most its budget of samples (``record_budget``; by default as many as fit in :data:`RECORD_BYTES`):
it stops by itself at the budget, keeping the samples it took, and a ``record`` that would start another stretch once
the budget is spent is refused (``live.record-budget``). The budget is in the log's header, so a replay stops where the
session did.

Two locks: one held by a chunk while it runs, which every command that changes what a run does takes first, so that it
acts between chunks; and one over the session's fields, held for moments, which is all ``status`` and ``hello`` take,
so that they are answered while a chunk runs (its compile included)."""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

import numpy as np

from mime.build import Mode
from mime.build._binding import SURFACE_KINDS
from mime.build._refusals import ArgumentRefused, MimeError, Refusal, RefusalClass
from mime.registry import Controller, LiveInput, live_default
from mime.live._protocol import COMMANDS, PROTOCOL, Frame, FrameSchema, refused, reply
from mime.units import TIME, Quantity, si

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment
    from mime.viz import StageSamples

type Subscriber = Callable[[Frame], None]
type History = dict[str, dict[str, np.ndarray]]

#: The frames a session sends per simulated second when no chunk is given: about a display's rate, rounded to whole
#: base steps.
FRAMES_PER_SECOND = 60.0
#: A run folder's command log: a header line, then one line per command with the step it took effect at
#: (``{"command": ..., "step": ...}``) and, after each chunk, the step the session reached (``{"reached": step}``).
COMMAND_LOG = "commands.jsonl"
LOG_FORMAT = "mime.commands/1"
#: The simulated time one ``step`` request may cover (:attr:`LiveSession.max_step_chunks`).
MAX_STEP_SECONDS = 3600.0
_MISSING: Any = object()
#: A log line that is no command: the step the session reached after a chunk.
REACHED = "reached"

#: How many of the commands taken a status reports as ``recent``.
RECENT_COMMANDS = 5
#: The memory a recording's samples may take by default (bytes): its budget is as many samples as fit, each the bound
#: fields' bytes and :data:`SAMPLE_OVERHEAD` for each field and for the sample's step.
RECORD_BYTES = 512 << 20
SAMPLE_OVERHEAD = 128

READY, RUNNING, PAUSED, STOPPED = "ready", "running", "paused", "stopped"
#: What the session is doing (``activity`` in every status): its first chunk running, which compiles the run's programs
#: (in a process that has not compiled them for an earlier session of the same built experiment), a later chunk
#: running, or neither, between chunks.
COMPILING, STEPPING, IDLE = "compiling", "stepping", "idle"


def _units(built: BuiltExperiment) -> dict[tuple[str, str], str]:
    """The unit of each state field MIME knows one for: an output an edge or an observation labels, a flag's."""
    from mime.build._report import graph_config

    found: dict[tuple[str, str], str] = {}
    for edge in graph_config(built.graph)["edges"]:
        if edge.get("source_units"):
            found.setdefault((str(edge["source_node"]), str(edge["source_field"])), str(edge["source_units"]))
    for entry in built.binding().entries:
        found[(entry.node, entry.field)] = entry.unit
    for observation in built.layout.observations:
        for output in observation.outputs:
            found[(output.node, output.field)] = output.unit
    for flag in built.layout.flags:
        found[(flag.node, flag.field)] = "1"
    return found


def _selection(built: BuiltExperiment, select: Sequence[str] | None, state: History) -> list[tuple[str, str]]:
    """The fields the frames carry: those ``select`` names, or every bound field but a surface's, per-step observation
    output and flag, each once. A surface's points and fields are carried when ``select`` names them: by default they
    would be most of every frame (a surface of 7,000 vertices, about 230 kB a frame)."""
    if select is None:
        chosen = dict.fromkeys((entry.node, entry.field) for entry in built.binding().entries
                               if entry.kind not in SURFACE_KINDS)
        chosen.update(dict.fromkeys((o.node, o.field) for o in built.layout.per_step()))
        chosen.update(dict.fromkeys((f.node, f.field) for f in built.layout.flags))
        return list(chosen)
    found: list[tuple[str, str]] = []
    paths = {f"{node}.{name}": (node, name) for node, fields in state.items() for name in fields}
    for path in select:
        if path not in paths:
            from mime.things._handles import did_you_mean

            raise ArgumentRefused([Refusal(
"live.select", RefusalClass.USER, f"select= names {path!r}, which is no field of the state",
                f"name fields as select=[\"<node>.<field>\", ...]. {did_you_mean(path, paths)}")])
        found.append(paths[path])
    return list(dict.fromkeys(found))


def frame_schema(built: BuiltExperiment, select: Sequence[str] | None = None) -> FrameSchema:
    """The schema of the frames a session of ``built`` sends with this ``select``: what the session's own is, before the
    session exists (``mime serve`` checks what ``--watch`` names against it before it opens a run folder)."""
    current = built.current_state()
    units = _units(built)
    return FrameSchema.of([(node, name, current[node][name], units.get((node, name)))
                           for node, name in _selection(built, select, current)])


class LiveSession:
    """A built experiment run live (``built.live(...)``). Its loop steps the compiled run ``chunk_steps`` base steps at
    a time, one length for the whole session, since each new length compiles anew; after each chunk it hands a frame
    of the selected fields to every subscriber. Its commands are those of ``mime.live/2`` (:meth:`command`), each also
    a method: :meth:`start`, :meth:`pause`, :meth:`step`, :meth:`stop` (which stops in every state), :meth:`set` (a
    live input of the controller in the callback mode), :meth:`record` (the recording layer, on or off) and
    :meth:`set_pace`. Every command is logged with the step it took effect at, so :func:`replay` reproduces the
    session's run folder.

    The session runs on the calling thread (:meth:`step`, :meth:`run_for`), on its own (:meth:`start`, which paces the
    loop to ``pace`` times real time on a thread of its own), or in a server's loop (:meth:`advance`, as
    ``mime serve`` drives it); a lock holds the commands that change a run to the gaps between chunks, and ``status``
    and ``hello`` are answered while a chunk runs. Its controller's live inputs are each at their declared default
    from its first tick until it sets them (:class:`~mime.registry.LiveInput`)."""

    def __init__(self, built: BuiltExperiment, *, chunk: Quantity | None = None, pace: float | None = 1.0,
                 select: Sequence[str] | None = None, out: Path | None = None,
                 chunk_steps: int | None = None, force: bool = False, record_budget: int | None = None) -> None:
        self.built = built
        dt = built.plans.dt
        period = built._runs.chunk  # pyright: ignore[reportPrivateUsage]
        if chunk_steps is not None:  # a replay's: the logged session's chunk, as it was
            steps = chunk_steps
        elif chunk is None:
            steps = max(1, round(1.0 / (FRAMES_PER_SECOND * dt)))
            if period is not None:
                steps = max(period, period * round(steps / period))
        else:
            steps = built.steps_in(chunk)
            if period is not None and steps % period:
                raise ArgumentRefused([Refusal(
"live.chunk", RefusalClass.USER, f"a chunk of {steps} base steps is not a whole number of the controller's control "
                                  f"periods of {period}",
                    f"give whole periods: built.live(chunk=s({max(1, steps // period) * period * dt!r}))",
                    why="a controller called back is called once per control period, between compiled chunks")])
        self.chunk_steps = steps
        self.dt = dt
        self.pace = self._pace(pace)
        self.out = out
        self.state = READY
        #: Whether an outer loop steps the session (``mime serve``'s), so that ``start`` starts no thread of its own.
        self.driven = False
        self.steps = built._runs.steps  # pyright: ignore[reportPrivateUsage]
        self.start_step = self.steps
        self.frames = 0
        self.log: list[dict[str, Any]] = []
        #: The chunks a step in hand has still to take (``mime serve`` takes its other requests between them).
        self.stepping = 0
        self.non_finite: str | None = None
        #: The chunks run so far, and whether one is running now.
        self.chunks = 0
        self._in_flight = False
        #: The commands a server took while the chunk in flight runs, in order, to act at its end (``mime serve``'s).
        self.pending: list[str] = []
        self._subscribers: list[Subscriber] = []
        self._lock = threading.RLock()
        self._chunk = threading.RLock()
        self._thread: threading.Thread | None = None
        self._wall: tuple[float, int] | None = None
        self._recording: StageSamples | None = None
        self._recorded: list[StageSamples] = []
        self._written: Path | None = None
        self._log_file: TextIO | None = None
        current = built.current_state()
        self._final: History = current
        self.schema = frame_schema(built, select)
        self.controller = built.plans.controller
        callback = built._runs.callback  # pyright: ignore[reportPrivateUsage]
        self._callback = callback
        declared = cast(type[Controller], self.controller.registered.cls) if self.controller is not None else None
        self.live_inputs: dict[str, LiveInput] = dict(declared.live) if declared is not None else {}
        #: Each live input at its declared default, as the step reads it until a session sets it.
        self.defaults = {label: cast(np.ndarray, live_default(held)) for label, held in self.live_inputs.items()}
        self._topology = built.topology()
        #: Who is connected, as ``clients`` in a status: set by a server, which alone knows (``(commands, frames)``).
        self.clients: Callable[[], tuple[int, int]] | None = None
        self._started = time.perf_counter()
        #: The samples a recording may hold, over every stretch of it.
        self.record_budget = self._budget(record_budget)
        self._refuse_an_older_stage()
        if out is not None:
            from mime.experiment._experiment import refuse_out_not_own, refuse_out_over

            saved = built._saved  # pyright: ignore[reportPrivateUsage]
            refuse_out_over(out, [saved.folder] if saved is not None else [])
            refuse_out_not_own(out, force)
            self._open_log(out)  # first, and exclusively: the folder's log is the session's claim on it
            self._write_experiment(out)
        self._open_inputs()  # last: an open that is refused or fails installs no live input in the built's callback

    def _open_inputs(self) -> None:
        """Every session starts from the declared inputs: its controller's step, called back, reads each live input at
        its default from the first tick, so that the step's program (a ``jax.jit``'s) is compiled with every input it
        will read, of the dtype and shape each ``set`` gives, and no ``set`` compiles it again."""
        if self._callback is not None:
            self._callback.live.clear()
            self._callback.live.update({label: value.copy() for label, value in self.defaults.items()})

    def _refuse_an_older_stage(self) -> None:
        """Refuse a saved folder whose stage an earlier stage writer wrote without prims the binding moves (a chain's
        frames, a vector): a client would find nothing to move."""
        from mime.scene import STAGE_FILE, STAGE_WRITER
        from mime.scene._pxr import Usd

        report, saved = self.built.load_report, self.built._saved  # pyright: ignore[reportPrivateUsage]
        if report is None or saved is None or report.stage_writer == STAGE_WRITER:
            return
        stage = Usd.Stage.Open(str(saved.folder / STAGE_FILE))
        held = {str(prim.GetPath()) for prim in stage.Traverse()}
        missing = [entry.name for entry in self.built.binding().entries if entry.prim not in held]
        if missing:
            raise ArgumentRefused([Refusal(
                "live.stage-writer", RefusalClass.USER,
                f"{saved.folder}'s stage was written by stage writer {report.stage_writer}, which wrote no prim for "
                f"{len(missing)} of the fields a live session moves ({', '.join(missing[:3])}"
                f"{', ...' if len(missing) > 3 else ''})",
                f"save it again with this MIME, whose stage writer {STAGE_WRITER} draws them: "
                f"Experiment.load(\"{saved.folder}\").save(\"<a new folder>\"), and serve that folder",
                why="a client moves the prims the binding names, in the stage it opens")])

    def _budget(self, given: int | None) -> int:
        """The samples a recording may hold: ``given``, or as many as fit in :data:`RECORD_BYTES`."""
        if given is not None:
            return checked_budget(given)
        fields = self.built.binding().fields()
        each = SAMPLE_OVERHEAD * (1 + len(fields)) + sum(
            np.asarray(self._final[node][name]).nbytes for node, name in fields)
        return max(1, RECORD_BYTES // each)

    def _samples(self) -> int:
        """The samples the recording holds, over every stretch of it."""
        return sum(len(part.steps) for part in self._recorded)

    @staticmethod
    def _pace(pace: float | None) -> float | None:
        if pace is None or (type(cast(object, pace)) in (int, float) and pace == 0):
            return None
        if not (type(cast(object, pace)) in (int, float) and math.isfinite(pace) and pace > 0):
            raise ArgumentRefused([Refusal(
"live.pace", RefusalClass.USER, f"pace is {pace!r}, not a real-time factor above zero",
                "give a factor of real time, pace=1.0 (real time) or pace=None (as fast as it runs)")])
        return float(pace)

    # -- subscribers and frames -------------------------------------------------------------------------------------

    def subscribe(self, subscriber: Subscriber) -> None:
        """Call ``subscriber`` with every frame from now on."""
        self._subscribers.append(subscriber)

    def _send(self, state: History, *, final: bool = False) -> Frame:
        frame = Frame(self.frames, self.steps, self.steps * self.dt, self.schema.values(state), final)
        self.frames += 1
        for subscriber in list(self._subscribers):
            subscriber(frame)
        return frame

    # -- stepping -----------------------------------------------------------------------------------------------------

    def advance(self) -> Frame | None:
        """One chunk, and its frame; ``None`` once the session has stopped or its state is not finite. A command that
        changes the run waits for it; ``status`` and ``hello`` are answered meanwhile."""
        with self._chunk:
            with self._lock:
                if self.state == STOPPED or self.non_finite is not None:
                    return None
                if self.frames == 0:
                    self._send(self._final)
                self._in_flight = True
            try:
                final, _ = self.built._runs.run(self.chunk_steps, history=False)  # pyright: ignore[reportPrivateUsage]
            except BaseException:
                with self._lock:
                    self._in_flight = False
                raise
            with self._lock:
                self._in_flight = False
                return self._ended(final)

    def _ended(self, final: History) -> Frame:
        """The bookkeeping after a chunk, its frame sent."""
        self.steps += self.chunk_steps
        self.chunks += 1
        self._final = final
        if self.stepping:
            self.stepping -= 1
        bad = [f"{node}.{name}" for node, fields in final.items() for name, value in fields.items()
               if np.issubdtype(np.asarray(value).dtype, np.inexact) and not np.isfinite(value).all()]
        if bad:
            self.non_finite = (f"the state stopped being finite numbers by step {self.steps} "
                               f"(t = {self.steps * self.dt:.6g} s), in {', '.join(bad[:4])}")
            self.state = PAUSED
            self.stepping = 0
        if self._recording is not None:
            self._recording.take(self.steps, final)
            if self._samples() >= self.record_budget:
                self._recording = None  # full: what it holds is kept, and status says it no longer records
        self._write_line({REACHED: self.steps})
        return self._send(final)

    def _logged(self, command: str, **arguments: Any) -> None:
        """Log a command at the step it takes effect at, durably, before it acts."""
        line = {"step": self.steps, "command": command, **arguments}
        self.log.append(line)
        self._write_line(line)

    def _open_log(self, out: Path) -> None:
        """Start the run folder's command log with its header, the file made by this call alone: one that is there
        already, another session's, is refused and left as it is."""
        header = {"format": LOG_FORMAT, "protocol": PROTOCOL, "experiment": self.built.plans.settings.experiment,
                  "input_hash": self.built.record.hashes()["input"], "dt": self.dt, "chunk_steps": self.chunk_steps,
                  "start_step": self.start_step, "record_budget": self.record_budget}
        out.mkdir(parents=True, exist_ok=True)
        try:
            self._log_file = (out / COMMAND_LOG).open("x", encoding="utf-8", newline="\n")
        except FileExistsError as error:
            raise ArgumentRefused([_log_exists(out)]) from error
        self._write_line(header)

    def _write_line(self, line: Mapping[str, Any]) -> None:
        """One line of the command log, flushed to the operating system, so that it outlives the process."""
        if self._log_file is not None:
            self._log_file.write(json.dumps(line, sort_keys=True) + "\n")
            self._log_file.flush()

    def _paced(self) -> float:
        """Seconds to wait before the next chunk, at the session's pace (0 unpaced). A session more than a chunk
        behind its pace (a chunk that compiled, a slow machine) takes the time it is at as its new start, rather than
        running unpaced to catch up."""
        if self.pace is None:
            return 0.0
        now = time.perf_counter()
        if self._wall is None:
            self._wall = (now, self.steps)
        began, at = self._wall
        due = began + (self.steps - at) * self.dt / self.pace
        if now - due > self.chunk_steps * self.dt / self.pace:
            self._wall = (now, self.steps)
            return 0.0
        return max(0.0, due - now)

    def due(self) -> float:
        """Seconds until the loop's next chunk is due: at once while a step is in hand, at the pace while running,
        ``math.inf`` otherwise."""
        if self.state == STOPPED or self.non_finite is not None:
            return math.inf
        if self.stepping:
            return 0.0
        return self._paced() if self.state == RUNNING else math.inf

    def start(self, *, thread: bool = True) -> None:
        """Start (or resume) stepping. With ``thread`` (in a script or a notebook) on a thread of the session's own,
        paced; without it, the caller's loop steps it (:meth:`advance` when :meth:`due` says)."""
        with self._chunk, self._lock:
            self._require_open("start")
            self._refuse_while_stepping("start")
            if self.state != RUNNING:
                self._logged("start")
                self.state = RUNNING
                self._wall = None
                if self.frames == 0:
                    self._send(self._final)
        if thread and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._loop, name="mime-live", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._lock:
                if self.state != RUNNING or self.non_finite is not None:
                    return
                wait = self._paced()
            if wait > 0:
                time.sleep(min(wait, 0.05))
                continue
            self.advance()

    def pause(self) -> None:
        """Stop stepping after the chunk in hand, running or taking a step; the session stays open."""
        with self._chunk, self._lock:
            self._require_open("pause")
            if self.state == RUNNING or self.stepping:
                self._logged("pause")
                self.state = PAUSED
                self.stepping = 0

    def begin_step(self, chunks: int = 1) -> None:
        """Take ``chunks`` chunks in hand, from a session that is not running, for the caller's loop to step one at a
        time (:meth:`advance` while :attr:`stepping`), taking a ``stop`` or a ``pause`` between them; it is paused
        after."""
        if not (type(cast(object, chunks)) is int and chunks >= 1):
            raise ArgumentRefused([Refusal(
                "live.step", RefusalClass.USER, f"step takes a whole number of chunks above zero, not {chunks!r}",
                "step(1), or step(n) for n chunks: `{\"command\": \"step\", \"chunks\": 3}`")])
        with self._chunk, self._lock:
            self._require_open("step")
            self._refuse_while_stepping("step")
            if self.state == RUNNING:
                raise ArgumentRefused([Refusal(
                    "live.state", RefusalClass.USER, "step is refused while the session runs",
                    "pause it first: session.pause(), then session.step(n)")])
            self._logged("step", chunks=chunks)
            self.state = PAUSED
            self.stepping = chunks

    def step(self, chunks: int = 1) -> list[Frame]:
        """Step ``chunks`` chunks now, on the calling thread, from a session that is not running; it is paused
        after. A ``stop`` or a ``pause`` from another thread is taken between chunks."""
        self.begin_step(chunks)
        found: list[Frame] = []
        while self.stepping:  # the lock is let go between chunks, so that a stop from another thread is taken
            frame = self.advance()
            if frame is None:
                break
            found.append(frame)
        return found

    def _refuse_while_stepping(self, command: str) -> None:
        if self.stepping:
            raise ArgumentRefused([Refusal(
                "live.state", RefusalClass.USER, f"{command} is refused while a step of {self.stepping} more chunks "
                                                 "is in hand",
                "wait for the step's reply, or end it: `{\"command\": \"pause\"}` stops it after the chunk in "
                "hand, `{\"command\": \"stop\"}` stops the session")])

    def run_for(self, duration: Quantity) -> list[Frame]:
        """Step a whole number of chunks covering ``duration`` on the calling thread, unpaced; its frames."""
        seconds = float(cast(float, si(duration, TIME, what="the duration")))
        return self.step(max(1, round(seconds / (self.chunk_steps * self.dt))))

    def set(self, label: str, value: Any) -> None:
        """Set the controller's live input ``label`` to ``value`` from its next tick on: refused unless the controller
        declares it and runs as a callback, or for a value not of its shape or not finite."""
        with self._chunk, self._lock:
            self._require_open("set")
            held = self._checked(label, value)
            self._logged("set", label=label, value=held.tolist())
            cast(Any, self._callback).live[label] = held

    def _checked(self, label: str, value: Any) -> np.ndarray:
        """``value`` as the live input ``label`` takes it, or the refusal: no controller called back, no such input,
        or a value not of finite numbers of its shape."""
        controller = self.controller
        declared = self.live_inputs.get(label)
        if controller is None or self._callback is None or controller.mode is not Mode.CALLBACK:
            mode = "no controller" if controller is None else f"its controller runs {controller.mode.name.lower()}"
            raise ArgumentRefused([Refusal(
"live.input-mode", RefusalClass.USER, f"a live input needs a controller called back between chunks, and this "
                                   f"experiment has {mode}",
                "declare the controller as a callback: exp.set_controller(..., mode=\"callback\"), and serve "
                "that experiment",
                why="the session sets a live input between ticks, on the host; inside the compiled step the "
                    "controller's state is in the graph")])
        if declared is None:
            raise ArgumentRefused([Refusal(
"live.input-unknown", RefusalClass.USER, f"the controller {controller.registered.cls.label()} has no live input "
                                      f"{label!r}; its live inputs are {', '.join(self.live_inputs) or 'none'}",
                "set one it declares: session.set(<label>, value), a label of Controller.live",
                why="a live session changes only what the controller declares live; a physical setting is a "
                    "rebuild")])
        held = _numbers(value)
        if held is None or held.shape != tuple(declared.shape) or not np.isfinite(held).all():
            raise ArgumentRefused([Refusal(
"live.input-value", RefusalClass.USER, f"{label} takes finite numbers of shape {tuple(declared.shape)} "
                                    f"({declared.units}), not {value!r}",
                f"give it one: session.set({label!r}, <a value of shape {tuple(declared.shape)}>)")])
        return held

    def record(self, on: bool) -> None:
        """Turn the recording layer on (from the state as it stands, then at every chunk's end) or off."""
        if type(cast(object, on)) is not bool:
            raise ArgumentRefused([Refusal(
                "live.record", RefusalClass.USER, f"record takes on, true or false, not {on!r}",
                "send `{\"command\": \"record\", \"on\": true}` or `false`; in Python, session.record(True)")])
        with self._chunk, self._lock:
            self._require_open("record")
            if on and self._recording is None and self._samples() >= self.record_budget:
                raise ArgumentRefused([Refusal(
                    "live.record-budget", RefusalClass.USER,
                    f"the recording holds {self._samples()} samples, its budget; this session records no more",
                    "serve it again with a larger budget, mime serve <folder> --record-budget <samples> "
                    "(built.live(record_budget=...)), or record a shorter stretch",
                    why="a recording is held in memory until the session stops, and the budget bounds it")])
            self._logged("record", on=on)
            if on and self._recording is None:
                from mime.viz import StageSamples

                self._recording = StageSamples(self.built.binding(), self.chunk_steps)
                self._recording.take(self.steps, self._final)
                self._recorded.append(self._recording)
                if self._samples() >= self.record_budget:
                    self._recording = None
            elif not on:
                self._recording = None

    def set_pace(self, pace: float | None) -> None:
        with self._chunk, self._lock:
            self._require_open("pace")
            held = self._pace(pace)
            self._logged("pace", pace=held)
            self.pace = held
            self._wall = None

    def stop(self) -> Path | None:
        """Stop, in whatever state the session is (ready, running, paused, or stepping on another thread, which
        finishes its chunk first); send the last frame and write the run folder. Stopping twice does nothing more. The
        folder written, or ``None`` without one."""
        with self._chunk, self._lock:
            if self.state == STOPPED:
                return self._written
            self._logged("stop")
            self.state = STOPPED
            self.stepping = 0
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._chunk, self._lock:
            if self._callback is not None:  # a live input never outlives its session
                self._callback.live.clear()
            self._send(self._final, final=True)
            if self.out is not None:
                self._written = self._write(self.out)
            return self._written

    def _require_open(self, command: str) -> None:
        if self.state == STOPPED:
            raise ArgumentRefused([Refusal(
                "live.stopped", RefusalClass.USER, f"{command} is refused: the session has stopped",
                "start another: built.live(...), or `mime serve <folder>` again")])

    # -- the protocol ------------------------------------------------------------------------------------------------

    def hello(self) -> dict[str, Any]:
        """The session as ``hello`` describes it: protocol, MIME's version, the experiment and its hashes, the base
        step and chunk, the controller and its live inputs, the stage, the binding, the topology, the build report's
        summary, the frame schema, the commands, the recording's budget, and the status. Answered while a chunk runs.
        It holds no path of this machine's: the stage is named relative to the run folder (``experiment/`` holds the
        saved folder's copy), and is ``None`` for a session with none."""
        from mime import __version__
        from mime.experiment._experiment import RUN_EXPERIMENT
        from mime.scene import STAGE_FILE

        controller = self.controller
        built = self.built
        stage = f"{RUN_EXPERIMENT}/{STAGE_FILE}" if self.out is not None else None
        return reply("hello", mime=__version__, experiment=built.plans.settings.experiment,
                     hashes=dict(built.record.hashes()), dt=self.dt, chunk_steps=self.chunk_steps,
                     controller=None if controller is None else {
                         "entry": controller.registered.cls.label(), "mode": controller.mode.name.lower(),
                         "every": controller.period, "latency": controller.latency},
                     live_inputs={label: {"units": held.units, "shape": list(held.shape), "meaning": held.meaning,
                                          "default": self.defaults[label].tolist()}
                                  for label, held in self.live_inputs.items()},
                     stage=stage, visuals=self._visuals(), binding=built.binding().record(),
                     topology=self._topology, report=built.report_summary(),
                     schema=self.schema.record(), commands=list(COMMANDS), record_budget=self.record_budget,
                     max_step_chunks=self.max_step_chunks,
                     **self.status_fields())

    def _visuals(self) -> dict[str, Any] | None:
        """What the asset cache draws of the stage's visuals: the overlay to add over the stage, by its id (its file
        name in the asset cache's ``overlays/``, never a folder's), the visual sets it lacks, which the stage draws as
        simple shapes, and the command that fetches each (``fetch``, run in the run folder: the folder it names is
        ``experiment/``, whose lock holds the set's manifest). ``None`` for a session with no run folder."""
        if self.out is None:
            return None
        from mime.experiment._experiment import RUN_EXPERIMENT
        from mime.scene import STAGE_FILE
        from mime.viz import overlay
        from mime.viz._visuals import UNKNOWN_FOLDER

        found = overlay(self.out / RUN_EXPERIMENT / STAGE_FILE)
        base = self.out.resolve()
        for name, held in found.fetch.items():
            if held != UNKNOWN_FOLDER and Path(held).is_absolute():
                inside = Path(held).resolve()
                found.fetch[name] = f"{inside.relative_to(base).as_posix()}/" if inside.is_relative_to(base) else \
                    UNKNOWN_FOLDER
        return {"overlay": found.path.name if found.path is not None else None, "drawn": found.drawn,
                "missing": found.missing, "fetch": found.fetch_commands(), "notice": found.notices}

    @property
    def wall_time(self) -> float:
        """The seconds since the session opened (the build before it not counted): ``wall_time`` in every status."""
        return time.perf_counter() - self._started

    @property
    def activity(self) -> str:
        """What the session is doing: :data:`COMPILING`, :data:`STEPPING` or :data:`IDLE`."""
        if not self._in_flight:
            return IDLE
        return COMPILING if self.chunks == 0 else STEPPING

    def _status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "state": self.state, "step": self.steps, "time": self.steps * self.dt, "frames": self.frames,
            "pace": self.pace, "recording": self._recording is not None, "non_finite": self.non_finite,
            "stepping": self.stepping, "activity": self.activity, "pending": list(self.pending),
            "wall_time": self.wall_time, "recent": self._recent(), "recorded": self._samples()}
        if self.clients is not None:
            commanders, viewers = self.clients()
            status["clients"] = {"commands": commanders, "frames": viewers}
        return status

    def _recent(self) -> list[dict[str, Any]]:
        """The last few commands taken: each its name, its arguments and the step it took effect at, from the log, which
        holds only commands the session accepted. Nothing of who sent one."""
        return [{"command": line["command"], "arguments": {key: value for key, value in line.items()
                                                           if key not in ("command", "step")}, "step": line["step"]}
                for line in self.log[-RECENT_COMMANDS:]]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return reply("status", **self._status())

    def status_fields(self) -> dict[str, Any]:
        """The status every reply carries: state, step, time, frames, pace, recording, non-finite, stepping, activity
        and pending. While a chunk runs, its step is the chunk's start."""
        with self._lock:
            return self._status()

    def command(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """One ``mime.live/2`` request, answered: the reply, or the refusal in MIME's coded shape. Never raises: a
        request a client got wrong is refused with its code, and any other error is answered in the coded shape of
        MIME's errors, the session going on. A ``step`` to a session a loop drives (``mime serve``'s) only takes its
        chunks in hand (:meth:`begin_step`): the loop steps them between its other requests and answers it when they
        are done."""
        name = message.get("command") if isinstance(cast(object, message), Mapping) else None
        try:
            return self._command(name, message)
        except MimeError as error:
            return refused(str(name) if name is not None else None, error.refusals)
        except Exception as error:  # noqa: BLE001 - a session never dies on one request
            from mime.errors._translate import translated

            return refused(str(name) if name is not None else None, translated(error).refusals)

    def _command(self, name: object, message: Mapping[str, Any]) -> dict[str, Any]:
        if message.get("protocol", PROTOCOL) != PROTOCOL:
            raise ArgumentRefused([Refusal(
"live.protocol", RefusalClass.USER, f"a request of {message.get('protocol')!r}; this session speaks {PROTOCOL}",
                f"send {PROTOCOL} requests, `{{\"protocol\": \"{PROTOCOL}\", \"command\": ...}}`, or use "
                "the MIME that speaks the client's protocol")])
        if name == "hello" and "protocol" not in message:
            raise ArgumentRefused([Refusal(
"live.protocol", RefusalClass.USER, f"a hello that does not say its protocol; this session speaks {PROTOCOL}",
                f"send `{{\"protocol\": \"{PROTOCOL}\", \"command\": \"hello\"}}`",
                why="hello's stage, visuals.overlay, visuals.fetch and stop's folder changed meaning in mime.live/2; a client "
                    "that does not say which protocol it reads would misread them, so hello names it")])
        if name not in COMMANDS:
            raise ArgumentRefused([Refusal(
"live.command", RefusalClass.USER, f"no command {name!r}; the commands are {', '.join(COMMANDS)}",
                "send one of them: `{\"command\": \"hello\"}`")])
        if name == "hello":
            return self.hello()
        if name == "status":
            return self.status()
        if name == "start":
            self.start(thread=not self.driven)
        elif name == "pause":
            self.pause()
        elif name == "step":
            chunks = self._checked_chunks(message.get("chunks", 1))
            if self.driven:
                self.begin_step(chunks)
            else:
                self.step(chunks)
        elif name == "stop":
            folder = self.stop()
            return reply("stop", folder=folder.name if folder is not None else None, **self._status())
        elif name == "set":
            values = message.get("values")
            if not isinstance(values, Mapping) or not values:
                raise ArgumentRefused([Refusal(
                    "live.input-value", RefusalClass.USER, "set takes values, a mapping of live inputs",
                    "send `{\"command\": \"set\", \"values\": {\"<label>\": <value>}}`")])
            given = cast(Mapping[str, Any], values)
            for label, value in given.items():  # every value checked before any is set
                self._checked(str(label), value)
            for label, value in given.items():
                self.set(str(label), value)
        elif name == "record":
            self.record(message.get("on", True))
        elif name == "pace":
            self.set_pace(self._checked_rtf(message))
        return reply(str(name), **self._status())

    @property
    def max_step_chunks(self) -> int:
        """The most chunks one ``step`` request may ask for: those that cover :data:`MAX_STEP_SECONDS` of simulated time
        (at least one). A step is answered when its chunks are stepped, so a request without a bound holds its client's
        reply for as long as it likes; the bound is the session's own, from its chunk and base step."""
        return max(1, int(MAX_STEP_SECONDS / (self.chunk_steps * self.dt)))

    def _checked_chunks(self, chunks: Any) -> Any:
        """``chunks`` of a ``step`` request, or the refusal when it is above :attr:`max_step_chunks`. What is not a whole
        number above zero is refused by :meth:`begin_step`."""
        if type(chunks) is int and chunks > self.max_step_chunks:
            raise ArgumentRefused([Refusal(
                "live.step", RefusalClass.USER,
                f"step takes at most {self.max_step_chunks} chunks in one request ({MAX_STEP_SECONDS:g} s of simulated "
                f"time at this session's chunk of {self.chunk_steps} steps), not {chunks}",
                "send several steps, or `start` the session and `pause` it when you have seen enough",
                why="a step is answered when its chunks are stepped, so its size is bounded by the session: hello's "
                    "`max_step_chunks`")])
        return chunks

    @staticmethod
    def _checked_rtf(message: Mapping[str, Any]) -> float | None:
        """The ``rtf`` of a ``pace`` request: present (a request without it is a mistyped key, which would leave the session
        unpaced without saying so), and null (unpaced) or a finite number above zero."""
        rtf = message.get("rtf", _MISSING)
        if rtf is _MISSING or (rtf is not None and not (type(rtf) in (int, float) and math.isfinite(rtf) and rtf > 0)):
            raise ArgumentRefused([Refusal(
                "live.pace", RefusalClass.USER,
                "pace takes `rtf`, " + ("which is missing" if rtf is _MISSING else f"not {rtf!r}") + ": a factor of real "
                "time above zero, or null for as fast as it runs",
                "send `{\"command\": \"pace\", \"rtf\": 1.0}` (real time) or `{\"command\": \"pace\", \"rtf\": null}` "
                "(unpaced)")])
        return rtf

    # -- the run folder ----------------------------------------------------------------------------------------------

    def _write_experiment(self, out: Path) -> None:
        """The run folder's ``experiment/``: the saved folder the session was loaded from, copied as it is, or this
        built experiment saved. Only the files of the saved form's names are written: nothing else in the folder is
        touched, as a run's rule keeps a forced folder's other files."""
        from mime.experiment._experiment import RUN_EXPERIMENT
        from mime.saved._folder import named_files

        target = out / RUN_EXPERIMENT
        saved = self.built._saved  # pyright: ignore[reportPrivateUsage]
        if saved is None:
            self.built.save(target)
            return
        for name in sorted(named_files(saved)):
            source = saved.folder / name
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target / name)

    def _write(self, out: Path) -> Path:
        from mime.experiment._experiment import RUN_OUTPUTS, RUN_RECORD, RUN_TOPOLOGY
        from mime.results import RUN_MANIFEST, RUN_REPORT

        out.mkdir(parents=True, exist_ok=True)
        result = self.built.result_of(self.start_step, self.steps - self.start_step, self._final)
        np.savez(out / RUN_OUTPUTS, allow_pickle=False, **result.arrays())
        manifest = result.manifest()
        if self.non_finite is not None:
            manifest["non_finite"] = self.non_finite
        _json(out / RUN_MANIFEST, manifest)
        _json(out / RUN_TOPOLOGY, self._topology)
        _json(out / RUN_REPORT, self.built.report_summary())
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        recording = None
        joined = _joined(self._recorded)
        if joined is not None:
            from mime.viz import write_recording

            recording = write_recording(out, self.built.binding(), joined, self.built.recording_settings())
        _json(out / RUN_RECORD, {
            "live": {"protocol": PROTOCOL, "chunk_steps": self.chunk_steps, "frames": self.frames,
                     "commands": COMMAND_LOG, "wall_seconds": round(time.perf_counter() - self._started, 3),
                     "record_budget": self.record_budget, "recorded": self._samples()},
            "steps": self.steps - self.start_step, "duration": (self.steps - self.start_step) * self.dt,
            "recording": recording.name if recording is not None else None, "non_finite": self.non_finite,
            "environment": self.built.record.environment})
        return out


def checked_budget(given: int) -> int:
    """A recording's budget as given, or its refusal (``live.record-budget``): a whole number of samples above zero."""
    if not (type(cast(object, given)) is int and given >= 1):
        raise ArgumentRefused([Refusal(
            "live.record-budget", RefusalClass.USER,
            f"a recording's budget is {given!r}, not a whole number of samples above zero",
            "give one, record_budget=3600 (or mime serve --record-budget 3600), or leave it out for as many as fit "
            f"in {RECORD_BYTES >> 20} MiB")])
    return given


def _log_exists(out: Path) -> Refusal:
    return Refusal(
        "live.log-exists", RefusalClass.USER,
        f"{out} holds a command log ({COMMAND_LOG}) already, another session's; a session never replaces one or writes "
        "into it",
        "write the session to a new folder or an empty one: --out <a new folder> (or out=); --force does not replace a "
        "command log",
        why="a run folder replays from its command log: replaced or written into, the run it records could not be "
            "replayed")


def refuse_a_log(out: Path) -> None:
    """Refuse a run folder that holds a command log already (``live.log-exists``): a session opens its own exclusively,
    and this says so before anything is built."""
    if (out / COMMAND_LOG).exists() or (out / COMMAND_LOG).is_symlink():
        raise ArgumentRefused([_log_exists(out)])


def _numbers(value: Any) -> np.ndarray | None:
    """``value`` as float64 numbers, or ``None`` when it is not a number or nested lists of numbers (a string, a
    boolean, a mapping, None)."""
    def plain(held: Any) -> bool:
        if isinstance(held, (list, tuple)):
            return all(plain(item) for item in cast(Sequence[Any], held))
        return type(held) in (int, float) or isinstance(held, (np.floating, np.integer)) or (
            isinstance(held, np.ndarray) and np.issubdtype(cast(np.ndarray, held).dtype, np.number))
    if not plain(value):
        return None
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):  # ragged lists
        return None


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _joined(parts: Sequence[StageSamples]) -> StageSamples | None:
    """The stretches a session recorded, as one series of samples (each sample's step is kept)."""
    held = [part for part in parts if part.steps]
    if not held:
        return None
    joined = type(held[0])(held[0].binding, held[0].every)
    for part in held:
        for index, step in enumerate(part.steps):
            if joined.steps and joined.steps[-1] == step:
                continue
            joined.steps.append(step)
            for key in joined.values:
                joined.values[key].append(part.values[key][index])
    return joined


def read_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A command log's header and its lines; a last line a dying session left cut short is dropped."""
    text = path.read_text(encoding="utf-8")
    rows = [row for row in text.splitlines() if row.strip()]
    lines: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            lines.append(json.loads(row))
        except ValueError:
            if index == len(rows) - 1 and not text.endswith("\n"):
                break
            lines = []
            break
    if not lines or not isinstance(cast(object, lines[0]), dict) or lines[0].get("format") != LOG_FORMAT:
        raise ArgumentRefused([Refusal(
"live.log", RefusalClass.USER, f"{path} is not a command log of {LOG_FORMAT}",
            "replay the commands.jsonl a live session wrote into its run folder: mime run <folder> --commands "
            "<run folder>/commands.jsonl --out <dir>")])
    return lines[0], lines[1:]


def replay(built: BuiltExperiment, log: Path, out: Path | None, *,
           subscriber: Subscriber | None = None, force: bool = False) -> LiveSession:
    """The live session a command log records, run again on ``built`` (built from the same saved folder): the same
    chunks, each live input and each recording turned on or off at the step it took effect at, stopped where it
    stopped (or, for a session that died, at the last step its log reached), its run folder written into ``out``,
    which may not be the folder the log or the experiment is in (``force`` as :class:`LiveSession` takes it); a
    recording stops at the budget the log's header states. ``subscriber`` takes each of its frames."""
    header, lines = read_log(log)
    if header["input_hash"] != built.record.hashes()["input"]:
        raise ArgumentRefused([Refusal(
"live.log", RefusalClass.USER, f"the log was written by a session of {header['experiment']} with input hash "
                        f"{header['input_hash']}, and this experiment's is {built.record.hashes()['input']}",
            "replay it on the saved folder the session served: mime run <that folder> --commands ...")])
    if out is not None:
        from mime.experiment._experiment import refuse_out_over

        refuse_out_over(out, [log])
    budget = header.get("record_budget")
    session = LiveSession(built, chunk=None, pace=None, out=out, chunk_steps=int(header["chunk_steps"]), force=force,
                          record_budget=int(budget) if budget is not None else None)
    if subscriber is not None:
        session.subscribe(subscriber)
    for line in lines:
        reached = line.get(REACHED)
        if reached is not None:
            while session.steps < int(reached):
                session.advance()
            continue
        while session.steps < int(line["step"]):
            session.advance()
        command = line["command"]
        if command == "set":
            session.set(str(line["label"]), line["value"])
        elif command == "record":
            session.record(bool(line["on"]))
        elif command == "pace":
            session.set_pace(cast(float | None, line.get("pace")))
        elif command == "stop":
            break
        else:  # start, pause and step: what the logged session's loop did is in the lines after them
            session.log.append(dict(line))
            session._write_line(line)  # pyright: ignore[reportPrivateUsage]
    session.stop()
    return session


__all__ = ["COMMAND_LOG", "COMPILING", "FRAMES_PER_SECOND", "IDLE", "LOG_FORMAT", "PAUSED", "READY", "REACHED",
           "RECORD_BYTES", "RUNNING", "SAMPLE_OVERHEAD", "STEPPING", "STOPPED", "LiveSession", "checked_budget",
           "frame_schema", "read_log", "refuse_a_log", "replay"]
