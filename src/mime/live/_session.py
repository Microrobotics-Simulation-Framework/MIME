"""The live session: a built experiment stepped in chunks of one fixed size, its frames handed to subscribers, its
controller's live inputs set between ticks, its commands logged with their steps so that it replays, and a minimal run
folder written when it stops.

A session given a run folder writes its command log there as it goes: the header when it opens, each command before it
acts, and after each chunk, before that chunk's frame is sent, the step it reached. Each line is flushed as it is
written, so a session that dies (a crash, a kill) leaves a log that replays to the last step it reached, and every frame
a client received is a step the log holds."""

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
from mime.build._refusals import ArgumentRefused, MimeError, Refusal, RefusalClass
from mime.registry import Controller, LiveInput
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
#: A log line that is no command: the step the session reached after a chunk.
REACHED = "reached"

READY, RUNNING, PAUSED, STOPPED = "ready", "running", "paused", "stopped"


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


class LiveSession:
    """A built experiment run live (``built.live(...)``). Its loop steps the compiled run ``chunk_steps`` base steps at
    a time, one length for the whole session, since each new length compiles anew; after each chunk it hands a frame
    of the selected fields to every subscriber. Its commands are those of ``mime.live/1`` (:meth:`command`), each also
    a method: :meth:`start`, :meth:`pause`, :meth:`step`, :meth:`stop` (which stops in every state), :meth:`set` (a
    live input of the controller in the callback mode), :meth:`record` (the recording layer, on or off) and
    :meth:`set_pace`. Every command is logged with the step it took effect at, so :func:`replay` reproduces the
    session's run folder.

    The session runs on the calling thread (:meth:`step`, :meth:`run_for`), on its own (:meth:`start`, which paces the
    loop to ``pace`` times real time on a thread of its own), or in a server's loop (:meth:`advance`, as
    ``mime serve`` drives it); a lock holds commands to the gaps between chunks."""

    def __init__(self, built: BuiltExperiment, *, chunk: Quantity | None = None, pace: float | None = 1.0,
                 select: Sequence[str] | None = None, out: Path | None = None,
                 chunk_steps: int | None = None) -> None:
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
        self._subscribers: list[Subscriber] = []
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._wall: tuple[float, int] | None = None
        self._recording: StageSamples | None = None
        self._recorded: list[StageSamples] = []
        self._written: Path | None = None
        self._log_file: TextIO | None = None
        current = built.current_state()
        self._final: History = current
        units = _units(built)
        chosen = self._selection(select, current)
        self.schema = FrameSchema.of([(node, name, current[node][name], units.get((node, name)))
                                      for node, name in chosen])
        self.controller = built.plans.controller
        callback = built._runs.callback  # pyright: ignore[reportPrivateUsage]
        self._callback = callback
        if callback is not None:  # every session starts from the declared inputs, none set
            callback.live.clear()
        declared = cast(type[Controller], self.controller.registered.cls) if self.controller is not None else None
        self.live_inputs: dict[str, LiveInput] = dict(declared.live) if declared is not None else {}
        self._started = time.perf_counter()
        self._refuse_an_older_stage()
        if out is not None:
            from mime.experiment._experiment import refuse_out_over

            saved = built._saved  # pyright: ignore[reportPrivateUsage]
            refuse_out_over(out, [saved.folder] if saved is not None else [])
            self._write_experiment(out)
            self._open_log(out)

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

    @staticmethod
    def _pace(pace: float | None) -> float | None:
        if pace is None or (type(cast(object, pace)) in (int, float) and pace == 0):
            return None
        if not (type(cast(object, pace)) in (int, float) and math.isfinite(pace) and pace > 0):
            raise ArgumentRefused([Refusal(
"live.pace", RefusalClass.USER, f"pace is {pace!r}, not a real-time factor above zero",
                "give a factor of real time, pace=1.0 (real time) or pace=None (as fast as it runs)")])
        return float(pace)

    def _selection(self, select: Sequence[str] | None, state: History) -> list[tuple[str, str]]:
        """The fields the frames carry: those ``select`` names, or every bound field, per-step observation output and
        flag, each once."""
        if select is None:
            chosen = dict.fromkeys(self.built.binding().fields())
            chosen.update(dict.fromkeys((o.node, o.field) for o in self.built.layout.per_step()))
            chosen.update(dict.fromkeys((f.node, f.field) for f in self.built.layout.flags))
            return list(chosen)
        found: list[tuple[str, str]] = []
        paths = {f"{node}.{name}": (node, name) for node, fields in state.items() for name in fields}
        for path in select:
            if path not in paths:
                from mime.things import did_you_mean

                raise ArgumentRefused([Refusal(
"live.select", RefusalClass.USER, f"select= names {path!r}, which is no field of the state",
                    f"name fields as select=[\"<node>.<field>\", ...]. {did_you_mean(path, paths)}")])
            found.append(paths[path])
        return list(dict.fromkeys(found))

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
        """One chunk, and its frame; ``None`` once the session has stopped or its state is not finite."""
        with self._lock:
            if self.state == STOPPED or self.non_finite is not None:
                return None
            if self.frames == 0:
                self._send(self._final)
            final, _ = self.built._runs.run(self.chunk_steps, history=False)  # pyright: ignore[reportPrivateUsage]
            self.steps += self.chunk_steps
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
            self._write_line({REACHED: self.steps})
            return self._send(final)

    def _logged(self, command: str, **arguments: Any) -> None:
        """Log a command at the step it takes effect at, durably, before it acts."""
        line = {"step": self.steps, "command": command, **arguments}
        self.log.append(line)
        self._write_line(line)

    def _open_log(self, out: Path) -> None:
        """Start the run folder's command log with its header."""
        header = {"format": LOG_FORMAT, "protocol": PROTOCOL, "experiment": self.built.plans.settings.experiment,
                  "input_hash": self.built.record.hashes()["input"], "dt": self.dt, "chunk_steps": self.chunk_steps,
                  "start_step": self.start_step}
        out.mkdir(parents=True, exist_ok=True)
        self._log_file = (out / COMMAND_LOG).open("w", encoding="utf-8", newline="\n")
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._require_open("record")
            self._logged("record", on=on)
            if on and self._recording is None:
                from mime.viz import StageSamples

                self._recording = StageSamples(self.built.binding(), self.chunk_steps)
                self._recording.take(self.steps, self._final)
                self._recorded.append(self._recording)
            elif not on:
                self._recording = None

    def set_pace(self, pace: float | None) -> None:
        with self._lock:
            self._require_open("pace")
            held = self._pace(pace)
            self._logged("pace", pace=held)
            self.pace = held
            self._wall = None

    def stop(self) -> Path | None:
        """Stop, in whatever state the session is (ready, running, paused, or stepping on another thread, which
        finishes its chunk first); send the last frame and write the run folder. Stopping twice does nothing more. The
        folder written, or ``None`` without one."""
        with self._lock:
            if self.state == STOPPED:
                return self._written
            self._logged("stop")
            self.state = STOPPED
            self.stepping = 0
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
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
        step and chunk, the controller, the stage, the binding, the topology, the frame schema, the commands."""
        from mime import __version__

        controller = self.controller
        built = self.built
        saved = built._saved  # pyright: ignore[reportPrivateUsage]
        stage = str((saved.folder / "scene" / "scene.usda").resolve()) if saved is not None else None
        return reply("hello", mime=__version__, experiment=built.plans.settings.experiment,
                     hashes=dict(built.record.hashes()), dt=self.dt, chunk_steps=self.chunk_steps,
                     controller=None if controller is None else {
                         "entry": controller.registered.cls.label(), "mode": controller.mode.name.lower(),
                         "every": controller.period, "latency": controller.latency},
                     live_inputs={label: {"units": held.units, "shape": list(held.shape), "meaning": held.meaning}
                                  for label, held in self.live_inputs.items()},
                     stage=stage, visuals=self._visuals(stage), binding=built.binding().record(),
                     topology=built.topology(),
                     schema=self.schema.record(), commands=list(COMMANDS), **self._status())

    def _visuals(self, stage: str | None) -> dict[str, Any] | None:
        """What the asset cache draws of the stage's visuals: the overlay to add over the stage (a layer in the cache,
        never in a folder), and the visual sets it lacks, which the stage draws as simple shapes."""
        if stage is None:
            return None
        from mime.viz import overlay

        found = overlay(Path(stage))
        return {"overlay": str(found.path) if found.path is not None else None, "drawn": found.drawn,
                "missing": found.missing, "notice": found.notices}

    def _status(self) -> dict[str, Any]:
        return {"state": self.state, "step": self.steps, "time": self.steps * self.dt, "frames": self.frames,
                "pace": self.pace, "recording": self._recording is not None, "non_finite": self.non_finite,
                "stepping": self.stepping}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return reply("status", **self._status())

    def status_fields(self) -> dict[str, Any]:
        """The status every reply carries: state, step, time, frames, pace, recording, non-finite and stepping."""
        with self._lock:
            return self._status()

    def command(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """One ``mime.live/1`` request, answered: the reply, or the refusal in MIME's coded shape. Never raises: a
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
            from mime.errors import translated

            return refused(str(name) if name is not None else None, translated(error).refusals)

    def _command(self, name: object, message: Mapping[str, Any]) -> dict[str, Any]:
        if message.get("protocol", PROTOCOL) != PROTOCOL:
            raise ArgumentRefused([Refusal(
"live.protocol", RefusalClass.USER, f"a request of {message.get('protocol')!r}; this session speaks {PROTOCOL}",
                f"send {PROTOCOL} requests, `{{\"protocol\": \"{PROTOCOL}\", \"command\": ...}}`, or use "
                "the MIME that speaks the client's protocol")])
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
            chunks = message.get("chunks", 1)
            if self.driven:
                self.begin_step(chunks)
            else:
                self.step(chunks)
        elif name == "stop":
            folder = self.stop()
            return reply("stop", folder=str(folder) if folder is not None else None, **self._status())
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
            self.set_pace(cast(float | None, message.get("rtf")))
        return reply(str(name), **self._status())

    # -- the run folder ----------------------------------------------------------------------------------------------

    def _write_experiment(self, out: Path) -> None:
        """The run folder's ``experiment/``: the saved folder the session was loaded from, copied as it is, or this
        built experiment saved."""
        from mime.experiment._experiment import RUN_EXPERIMENT
        from mime.saved._folder import named_files

        target = out / RUN_EXPERIMENT
        saved = self.built._saved  # pyright: ignore[reportPrivateUsage]
        if target.exists():
            shutil.rmtree(target)
        if saved is None:
            self.built.save(target)
            return
        for name in sorted(named_files(saved)):
            source = saved.folder / name
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target / name)

    def _write(self, out: Path) -> Path:
        from mime.experiment._experiment import RUN_OUTPUTS, RUN_RECORD, RUN_TOPOLOGY
        from mime.results import RUN_MANIFEST

        out.mkdir(parents=True, exist_ok=True)
        result = self.built.result_of(self.start_step, self.steps - self.start_step, self._final)
        np.savez(out / RUN_OUTPUTS, allow_pickle=False, **result.arrays())
        manifest = result.manifest()
        if self.non_finite is not None:
            manifest["non_finite"] = self.non_finite
        _json(out / RUN_MANIFEST, manifest)
        _json(out / RUN_TOPOLOGY, self.built.topology())
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
                     "commands": COMMAND_LOG, "wall_seconds": round(time.perf_counter() - self._started, 3)},
            "steps": self.steps - self.start_step, "duration": (self.steps - self.start_step) * self.dt,
            "recording": recording.name if recording is not None else None, "non_finite": self.non_finite,
            "environment": self.built.record.environment})
        return out


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
           subscriber: Subscriber | None = None) -> LiveSession:
    """The live session a command log records, run again on ``built`` (built from the same saved folder): the same
    chunks, each live input and each recording turned on or off at the step it took effect at, stopped where it
    stopped (or, for a session that died, at the last step its log reached), its run folder written into ``out``,
    which may not be the folder the log or the experiment is in. ``subscriber`` takes each of its frames."""
    header, lines = read_log(log)
    if header["input_hash"] != built.record.hashes()["input"]:
        raise ArgumentRefused([Refusal(
"live.log", RefusalClass.USER, f"the log was written by a session of {header['experiment']} with input hash "
                        f"{header['input_hash']}, and this experiment's is {built.record.hashes()['input']}",
            "replay it on the saved folder the session served: mime run <that folder> --commands ...")])
    if out is not None:
        from mime.experiment._experiment import refuse_out_over

        refuse_out_over(out, [log])
    session = LiveSession(built, chunk=None, pace=None, out=out, chunk_steps=int(header["chunk_steps"]))
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


__all__ = ["COMMAND_LOG", "FRAMES_PER_SECOND", "LOG_FORMAT", "PAUSED", "READY", "REACHED", "RUNNING", "STOPPED",
           "LiveSession", "read_log", "replay"]
