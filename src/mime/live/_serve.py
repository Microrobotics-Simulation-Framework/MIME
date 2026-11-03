"""``mime serve``: a saved experiment's live session in a process of its own, speaking ``mime.live/1`` over ZMQ, a PUB
for its frames and a REP for its commands, bound to localhost unless asked otherwise; and the Python client of it.

A served session does not outlive the process that launched it (:class:`LauncherWatch`): when that process is gone, the
session stops as on SIGTERM, writing its run folder, and the server exits. On Linux the kernel signals it
(``PR_SET_PDEATHSIG``, SIGTERM); on every platform the serve loop also looks between chunks at whether its launcher is
alive (on POSIX, a process whose launcher died is adopted and its parent's pid changes; on Windows, the launcher's
process handle is signalled when it exits). ``--outlive-launcher`` serves on regardless, for a server started to run
detached.

The server never goes silent: each chunk runs on a thread of its own while the loop moves bytes and answers. While a
chunk runs (a first chunk's compile included), ``hello`` and ``status`` are answered at once, the status saying what
the session is doing (``activity``: compiling, stepping); ``stop`` and ``pause`` are answered at once too, their
command in the status's ``pending``, and act at the chunk's end; every other command is answered at the chunk's end,
where it acts. The commands taken during a chunk act in the order they came, so the command log holds them in that
order at the chunk's end step. The frames a chunk sends are published by the loop, which alone touches the sockets."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import signal
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, cast

from mime.build._refusals import ArgumentRefused, BuildRefused, Refusal, RefusalClass
from mime.live._protocol import COMMANDS, PROTOCOL, Frame, FrameSchema, frame_from_json, is_frame, refused, reply
from mime.live._session import STOPPED, LiveSession, frame_schema
from mime.live._state import ServeState, flag_watches, watch_rows
from mime.live._zmtp import Client, Server, is_loopback, parse_endpoint
from mime.units import Quantity

DEFAULT_COMMANDS = "tcp://127.0.0.1:5555"
DEFAULT_FRAMES = "tcp://127.0.0.1:5556"
#: The longest the loop waits for a request before it looks again at whether a chunk is due or a signal came.
_POLL = 0.1
#: The commands answered at once while a chunk runs, and those answered at once that act at its end.
_ANSWERED = frozenset({"hello", "status"})
_TAKEN = frozenset({"stop", "pause"})
#: How often, at most, a view is told the session's status (seconds).
_STATUS_EVERY = 0.2


class ServeView(Protocol):
    """What ``mime serve`` tells a view of itself (``mime.cli._display.ServeDisplay``), if it is given one: what is under
    way while the experiment loads and builds, a pause before a line is printed in the view's place, and the state to
    draw once the sockets listen. The view reads :class:`~mime.live._state.ServeState`, which the loop writes; a view
    never calls the session."""

    def phase(self, text: str) -> None: ...

    def pause(self) -> None: ...

    def running(self, state: ServeState) -> None: ...


def _timeout(name: str, seconds: float | None) -> None:
    if seconds is not None and not (seconds > 0.0 and seconds < float("inf")):
        raise ArgumentRefused([Refusal(
            "live.timeout", RefusalClass.USER, f"{name} is {seconds!r}, not a time above zero",
            f"give a duration with a unit, such as `mime serve <folder> {name} 15min`",
            why="a limit of nothing or less would stop the session before it served anyone")])


def _bind_problems(endpoints: Sequence[str], remote: bool) -> list[Refusal]:
    found: list[Refusal] = []
    for endpoint in endpoints:
        try:
            host, _ = parse_endpoint(endpoint)
        except ValueError as error:
            found.append(Refusal("live.remote-bind", RefusalClass.USER, str(error),
                                 "give tcp://<host>:<port>, such as --commands tcp://127.0.0.1:5555"))
            continue
        if not remote and not is_loopback(host):
            found.append(Refusal(
                "live.remote-bind", RefusalClass.USER,
                f"{endpoint} is not a loopback address, and a live session has no authentication of its own",
                "keep it on localhost and reach it through an SSH tunnel (ssh -L 5555:localhost:5555 -L "
                "5556:localhost:5556 <host>), or pass --remote to bind it all the same",
                why="the session takes commands from whoever reaches its port; a tunnel authenticates and encrypts"))
    return found


class LauncherWatch:
    """Whether the process that launched this one is gone. The launcher is the parent when the watch is made; on Linux,
    with ``death_signal``, the kernel is also asked to send SIGTERM when it dies (``PR_SET_PDEATHSIG``), which the
    serve loop takes as a stop."""

    def __init__(self, *, death_signal: bool = True) -> None:
        self.launcher = os.getppid()
        self._handle: int | None = None
        if sys.platform == "win32":
            synchronize = 0x00100000
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel.OpenProcess(synchronize, False, self.launcher)
            self._handle = int(handle) if handle else None
        elif sys.platform.startswith("linux") and death_signal:
            try:
                ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
            except (OSError, AttributeError):  # no prctl to call: the watch below still sees it
                pass

    def gone(self) -> bool:
        """Whether the launcher has exited since the watch was made."""
        if sys.platform == "win32":
            if self._handle is None:
                return True  # it had exited before the watch could hold it
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            return int(kernel.WaitForSingleObject(self._handle, 0)) == 0  # WAIT_OBJECT_0: the process has exited
        return os.getppid() != self.launcher


def default_out(name: str) -> Path:
    """Where a session writes its run folder when not told: ``./<experiment>-live-<UTC time>/``."""
    return Path.cwd() / f"{name}-live-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def serve(folder: Path, *, commands: str = DEFAULT_COMMANDS, frames: str = DEFAULT_FRAMES,
          chunk: Quantity | None = None, pace: float | None = 1.0, out: Path | None = None,
          select: Sequence[str] | None = None, json_frames: bool = False, start: bool = False, remote: bool = False,
          ready_file: Path | None = None, say: Callable[[str], None] = print,
          outlive_launcher: bool = False, watch: Sequence[str] | None = None, view: ServeView | None = None,
          idle_timeout: float | None = None, max_wall: float | None = None) -> Path | None:
    """Load and build the saved experiment in ``folder``, open its live session and serve it until a ``stop`` command,
    SIGTERM or SIGINT: commands on ``commands`` (a REP), frames on ``frames`` (a PUB), binary or, with
    ``json_frames``, JSON. ``start`` starts stepping at once; otherwise a client's ``start`` does. ``ready_file``
    receives the bound endpoints as JSON once the sockets listen. It stops, as on SIGTERM, when the process that
    launched it is gone, unless ``outlive_launcher`` (:class:`LauncherWatch`). The run folder it wrote, or ``None``.

    ``watch`` names the rows of a view's table of values (:func:`~mime.live._state.watch_rows`; a name the frames do not
    carry is refused, ``live.watch``, before a run folder is opened) and changes nothing the frames carry. ``view``, if
    given, is told what is under way and handed the :class:`~mime.live._state.ServeState` the loop keeps; without one
    the loop does none of that work. ``idle_timeout`` stops the session when no client has been connected that long
    (from when it listens, or from the last to leave), and ``max_wall`` when the session's wall time (``wall_time`` in
    its status: from when it opened, the build before it not counted) reaches it; each stops it as SIGTERM does,
    writes the run folder and says why. A stop waits for the chunk in hand, so it comes at that chunk's end, later than the
    limit by what is left of the chunk."""
    _timeout("--idle-timeout", idle_timeout)
    _timeout("--max-wall", max_wall)
    problems = _bind_problems((commands, frames), remote)
    if problems:
        raise ArgumentRefused(problems)
    launcher = None if outlive_launcher else LauncherWatch()
    from mime.env import apply_lock_precision
    from mime.experiment import Experiment

    try:
        server = Server(commands, frames)
    except OSError as error:
        code = "live.address-in-use" if error.errno == errno.EADDRINUSE else "live.transport"
        raise BuildRefused([Refusal(
            code, RefusalClass.ENVIRONMENT, f"cannot listen on {commands} and {frames}: {error.strerror or error}",
            "serve on other ports: mime serve <folder> --commands tcp://127.0.0.1:<port> --frames "
            "tcp://127.0.0.1:<port> (port 0 takes any free one)")]) from error
    state: ServeState | None = None
    try:
        if view is not None:
            view.phase(f"loading {folder}: reading it, building it once and checking it against its lock")
        apply_lock_precision(folder)
        experiment = Experiment.load(folder)
        if view is not None:
            view.phase("building the graph to serve it (compiling it, which can take many seconds)")
        built = experiment.build()
        if view is not None:
            view.pause()
        if experiment.load_report is not None:
            say(experiment.load_report.text())
        target = out if out is not None else default_out(built.plans.settings.experiment)
        carried = frame_schema(built, select)
        flags = flag_watches(carried, built.layout.flags)
        rows, shown = watch_rows(carried, watch, cast(list[dict[str, Any]], built.binding().record()["entries"]), flags)
        session = built.live(chunk=chunk, pace=pace, select=select, out=target)
        if view is not None:
            state = ServeState(built.plans.settings.experiment, str(target))
            state.watch(rows, flags, len(built.layout.flags), table=shown)
    except BaseException:
        server.close()
        raise
    session.driven = True
    session.clients = server.counts
    schema = session.schema
    sent: deque[Frame] = deque()  # each frame a chunk sent, on its thread; published by the loop

    def publish() -> None:
        while sent:
            held = sent.popleft()
            if state is not None:
                state.take(held)
            if json_frames:
                server.publish(json.dumps(schema.json_frame(held.number, held.step, held.time, held.values,
                                                            final=held.final)).encode("utf-8"))
            else:
                server.publish(schema.encode(held.number, held.step, held.time, held.values, final=held.final))

    session.subscribe(sent.append)
    stopping: list[int] = []

    def stop_on(signum: int, _: FrameType | None) -> None:
        stopping.append(signum)

    previous = {signum: signal.signal(signum, stop_on) for signum in (signal.SIGTERM, signal.SIGINT)}
    endpoints = {"protocol": PROTOCOL, "commands": server.endpoints["REP"], "frames": server.endpoints["PUB"],
                 "out": str(target)}
    if ready_file is not None:
        ready_file.write_text(json.dumps(endpoints) + "\n", encoding="utf-8")
    steppers = _Steppers(server, session)  # the peers whose steps are in hand, each answered when its step ends
    running: Future[Frame | None] | None = None  # the chunk running on the worker's thread
    held: deque[tuple[int | None, bytes]] = deque()  # requests taken while it runs, acted on at its end, in order
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mime-serve-chunk")

    def act(peer: int | None, body: bytes) -> None:
        """A request acted on between chunks, answered unless it was answered when taken (``peer`` None) or is a
        step whose chunks are still to come."""
        answer = _answer(session, body)
        _noted(state, body, answer)
        if peer is not None:
            if answer.get("command") == "step" and answer["ok"] and session.stepping:
                steppers.add(peer)
            else:
                server.reply(peer, _encoded(answer))
        steppers.settle()

    reason: str | None = None  # why the loop ended, where the session did not end it
    last_client = last_status = time.monotonic()
    try:
        if state is not None:
            state.note_status(session.status_fields(), last_status)
            assert view is not None
            view.running(state)
        say(f"mime serve: {built.plans.settings.experiment} ready ({PROTOCOL}); commands on {endpoints['commands']}, "
            f"frames on {endpoints['frames']}; its run folder will be {target}")
        if start:
            session.start(thread=False)
        while session.state != STOPPED and not stopping:
            if launcher is not None and launcher.gone():
                reason = f"the process that launched it ({launcher.launcher}) is gone"
                say(f"mime serve: {reason}; stopping")
                break
            now = time.monotonic()
            if any(server.counts()):
                last_client = now
            elif idle_timeout is not None and now - last_client >= idle_timeout:
                reason = f"no client has been connected for {idle_timeout:g} s (--idle-timeout)"
                say(f"mime serve: {reason}; stopping")
                break
            if max_wall is not None and session.wall_time >= max_wall:
                reason = f"its wall time reached {max_wall:g} s (--max-wall)"
                say(f"mime serve: {reason}; stopping")
                break
            if state is not None:
                for event in server.take_events():
                    state.peer(event.what, event.role, event.address, event.dropped)
                if now - last_status >= _STATUS_EVERY:
                    state.note_status(session.status_fields(), now)
                    last_status = now
            for peer, body in server.poll(_POLL if running is not None else min(session.due(), _POLL)):
                if running is None:
                    act(peer, body)
                    continue
                answer = _while_running(session, body)
                if answer is not None:
                    _noted(state, body, answer)
                    server.reply(peer, _encoded(answer))
                if answer is None or (answer.get("command") in _TAKEN and answer["ok"]):
                    held.append((None if answer is not None else peer, body))
            publish()
            if running is not None and running.done():
                running.result()  # a chunk's error leaves the loop, and the session stops below
                running = None
                while held:
                    peer, body = held.popleft()
                    if peer is None:
                        session.pending.pop(0)
                    act(peer, body)
                publish()
            if running is None:
                steppers.answer_if_ended()
            if running is None and session.state != STOPPED and not stopping and session.due() <= 0.0:
                running = worker.submit(session.advance)  # one chunk, the requests answered while it runs
                running.add_done_callback(lambda _: server.wake())
        if reason is None:
            reason = ("stopped by " + ("Ctrl+C" if stopping[0] == signal.SIGINT else signal.Signals(stopping[0]).name)
                      if stopping else "a client's stop command")
        if state is not None:
            state.stop_begins(reason)
        written = session.stop()  # after the chunk in flight, if there is one
        publish()
        steppers.answer_all()
        for peer, body in held:  # requests a signal left unanswered: refused, the session stopped
            if peer is not None:
                server.reply(peer, _encoded(_answer(session, body)))
        server.flush(2.0)
        if running is not None:
            running.result()  # the error of a chunk a signal stopped after, if it raised one
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if session.state != STOPPED:  # an error of the server's own: the session still writes its run folder
            try:
                session.stop()
            except Exception:  # noqa: BLE001 - the first error is the one to raise
                pass
        worker.shutdown(wait=True)
        server.close()
    if state is not None:
        state.stop_ends(reason or "stopped", session.steps, session.steps * session.dt, session.wall_time, session.frames,
                        str(written) if written is not None else None)
    say(f"mime serve: stopped at step {session.steps}; wrote {written}")
    return written


class _Steppers:
    """The peers whose ``step`` is in hand, in the order they asked: each is answered, with the session's status then,
    once no step is in hand, whether its chunks were stepped or a ``pause`` or ``stop`` ended it. One slot would lose
    the reply of every step but the last when a batch of requests acted at a chunk's end holds more than one."""

    def __init__(self, server: Server, session: LiveSession) -> None:
        self.server, self.session = server, session
        self.waiting: deque[int] = deque()

    def add(self, peer: int) -> None:
        self.waiting.append(peer)

    def answer_if_ended(self) -> None:
        if not self.session.stepping:
            self.answer_all()

    def settle(self) -> None:
        """After each request acted on: a ``pause`` or ``stop`` that ended the steps in hand answers them before the
        next request acts, which may start another."""
        self.answer_if_ended()

    def answer_all(self) -> None:
        while self.waiting:
            self.server.reply(self.waiting.popleft(), _encoded(reply("step", **self.session.status_fields())))


def _encoded(answer: dict[str, Any]) -> bytes:
    return json.dumps(answer).encode("utf-8")


def _message(body: bytes) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A request's JSON object, or the refusal of one that is not a JSON object."""
    try:
        message = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError) as error:  # not UTF-8, not JSON, or nested past what JSON's reader takes
        return None, refused(None, [Refusal("live.command", RefusalClass.USER, f"a request that is not JSON: {error}",
                                            "send a JSON object: `{\"command\": \"hello\"}`")])
    if not isinstance(message, dict):
        return None, refused(None, [Refusal("live.command", RefusalClass.USER, "a request that is not a JSON object",
                                            "send a JSON object: `{\"command\": \"hello\"}`")])
    return cast(dict[str, Any], message), None


def _noted(state: ServeState | None, body: bytes, answer: dict[str, Any]) -> None:
    """What a client asked and what came of it, for a view (the name and arguments of the request, never who)."""
    if state is None:
        return
    message, _ = _message(body)
    if message is not None:
        state.command(message, answer)


def _answer(session: LiveSession, body: bytes) -> dict[str, Any]:
    message, refusal = _message(body)
    return session.command(message) if message is not None else cast(dict[str, Any], refusal)


def _while_running(session: LiveSession, body: bytes) -> dict[str, Any] | None:
    """The answer to a request taken while a chunk runs, or ``None`` for one answered at the chunk's end: ``hello``,
    ``status`` and every refusal of a request no session could take are answered at once, and so are ``stop`` and
    ``pause``, which the session takes (its ``pending``) to act at the chunk's end."""
    message, refusal = _message(body)
    if message is None:
        return refusal
    name = message.get("command")
    if message.get("protocol", PROTOCOL) != PROTOCOL or name not in COMMANDS or name in _ANSWERED:
        return session.command(message)  # answered without waiting: a refusal, a hello or a status
    if name not in _TAKEN:
        return None
    session.pending.append(str(name))
    fields = session.status_fields()
    if name == "stop":  # not the confirmation: no run folder yet, and the final frame says when it has stopped
        return reply("stop", taken=True, message="stop pending: the session stops at the end of the chunk in hand; "
                     "the final frame or the closed connection says it has, and no status reply will", **fields)
    return reply(str(name), **fields)


class LiveClient:
    """A Python client of ``mime serve``: its commands (:meth:`request`) and its frames (:meth:`frame`), decoded by
    the schema ``hello`` gives. The frames' socket connects first, so no frame sent after ``hello`` is missed."""

    def __init__(self, commands: str = DEFAULT_COMMANDS, frames: str = DEFAULT_FRAMES, *,
                 timeout: float = 30.0) -> None:
        self.frames = Client(frames, "SUB", timeout=timeout)
        self.commands = Client(commands, "REQ", timeout=timeout)
        self.hello = self.request("hello")
        self.schema = FrameSchema.from_record(self.hello["schema"])

    def request(self, command: str, **arguments: Any) -> dict[str, Any]:
        answer = self.commands.request(json.dumps({"protocol": PROTOCOL, "command": command,
                                                   **arguments}).encode("utf-8"))
        return cast(dict[str, Any], json.loads(answer.decode("utf-8")))

    def frame(self) -> Frame:
        data = self.frames.receive()
        if is_frame(data):
            return self.schema.decode(data)
        return frame_from_json(json.loads(data.decode("utf-8")), self.schema)

    def close(self) -> None:
        self.commands.close()
        self.frames.close()


__all__ = ["DEFAULT_COMMANDS", "DEFAULT_FRAMES", "LauncherWatch", "LiveClient", "default_out", "serve"]
