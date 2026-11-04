"""``mime serve`` as a person and as a launcher meet it, each in a child process of its own on A0's apparatus closed by
``ArmTracking`` (``live_cases.tracked``).

On a terminal (a pseudo-terminal sized 100 and 60 columns, and 0, which nobody sized and which reads as the fallback width)
a scripted client connects, starts the session, sets a live input, sends a command that is not one (with escapes in its
name), pauses and leaves, and Ctrl+C stops the server. The screen is read as a person sees it: each event once and above
the strip, the strip's step the status's, no line wider than the terminal, no escape of the client's reaching it, and, after
the stop, no live region and the block that says how it stopped.

Everywhere else it prints what it always printed: its two lines on standard output (after the load report), byte for byte
an oracle written out here from the objects the command holds, with a pipe, a pipe and ``FORCE_COLOR``/``CI``, standard
output a pipe with standard error a terminal (the view is drawn on standard error and standard output stays the
launcher's), and a terminal asked to be plain three ways (``NO_COLOR``, ``TERM=dumb``, ``--plain``), where nothing is drawn
at all. The control: the comparison finds a colour code planted in the oracle's own text."""

import json
import os
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from live_cases import saved

from _platform_support import needs_pty
from _terminal_support import fill_unread, is_full, screen_of, stripped
from mime.live import LiveClient

if sys.platform != "win32":
    import fcntl
    import pty
    import termios

pytestmark = [needs_pty, pytest.mark.own_child_cache]
HANG = 600
#: How long, once the child has exited, the reader has to take what it wrote and see the terminal report its end: a guard
#: against a descriptor that something else holds open, not a bound on the reading (which ends with the terminal's end).
END_WITHIN = 30.0


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("terminal"))


def child_environment(home: Path, **extra: str) -> dict[str, str]:
    """A user's environment for a child: its own temporary HOME, a terminal that is not dumb, none of the suite's own
    settings, and the CPU."""
    held = {key: value for key, value in os.environ.items()
            if key not in ("MIME_TRACEBACK", "NO_COLOR", "FORCE_COLOR", "TERM", "CI")}
    return {**held, "HOME": str(home), "USERPROFILE": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
            "MIME_TRACEBACK_DIR": str(home / "tracebacks"), "TERM": "xterm-256color", "JAX_PLATFORMS": "cpu",
            "MIME_OFFLINE": "1", **extra}


class Served:
    """``mime serve`` with some of its standard streams a terminal and the rest pipes, read as it runs: ``tty`` is what the
    terminal received, and ``out`` and ``err`` what the pipes did, once it has exited."""

    def __init__(self, folder: Path, tmp: Path, *, terminals: tuple[str, ...], columns: int = 100, rows: int = 30,
                 env: dict[str, str] | None = None, plain: bool = False, args: tuple[str, ...] = ()) -> None:
        self.tmp, self.ready, self.out_dir = tmp, tmp / "ready.json", tmp / "out"
        home = tmp / "home"
        home.mkdir(exist_ok=True)
        master, slave = pty.openpty()
        self.far = os.ttyname(slave)
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        self.master, self._held, self._lock = master, bytearray(), threading.Lock()
        piped = subprocess.PIPE
        self.process = subprocess.Popen(
            [sys.executable, "-m", "mime.cli", *(["--plain"] if plain else []), "serve", str(folder), "--commands",
             "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0", "--ready-file", str(self.ready), "--out",
             str(self.out_dir), *args],
            stdin=slave, stdout=slave if "out" in terminals else piped, stderr=slave if "err" in terminals else piped,
            env=child_environment(home, **(env or {})), cwd=tmp, close_fds=True, start_new_session=True)
        os.close(slave)
        self._open = True
        #: Whether ``finish`` saw the reader end at the terminal's end (False: it gave up after :data:`END_WITHIN`).
        self.reader_saw_the_end: bool | None = None
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.piped: tuple[str, str] | None = None
        deadline = time.monotonic() + HANG
        while not self.ready.exists() or not self.ready.read_text():
            assert self.process.poll() is None, self.finish()
            assert time.monotonic() < deadline, "mime serve never wrote its ready file"
            time.sleep(0.05)

    def stop_reading(self) -> None:
        """From here nobody reads the terminal (Ctrl+S, a stalled ssh session): the reader ends, and the terminal is filled
        so that a write to it blocks now, however much room a platform's terminal has."""
        self._open = False
        self._reader.join(timeout=5)
        fill_unread(self.far)
        assert is_full(self.far)

    def _read(self) -> None:
        while self._open:
            try:
                if not select.select([self.master], [], [], 0.05)[0]:
                    continue
                more = os.read(self.master, 65536)
            except (OSError, ValueError):
                return
            if not more:
                return
            with self._lock:
                self._held += more

    @property
    def tty(self) -> str:
        time.sleep(0.1)
        with self._lock:
            return bytes(self._held).decode("utf-8", "replace")

    def client(self) -> LiveClient:
        endpoints = json.loads(self.ready.read_text())
        return LiveClient(endpoints["commands"], endpoints["frames"], timeout=60.0)

    def until(self, wanted, what: str) -> str:
        """The terminal's screen once ``wanted`` is true of its lines: a guard against waiting for ever, not a bound."""
        deadline = time.monotonic() + HANG
        while True:
            screen = screen_of(self.tty)
            if wanted(screen):
                return "\n".join(screen)
            assert self.process.poll() is None or what.startswith("exit"), (what, "\n".join(screen))
            assert time.monotonic() < deadline, f"never saw: {what}\n" + "\n".join(screen)
            time.sleep(0.1)

    def finish(self, limit: float | None = None) -> tuple[str, str, int]:
        """Wait for the exit; what the pipes received (empty for a terminal) and the exit code."""
        out, err = self.process.communicate(timeout=HANG if limit is None else limit)
        # What the child wrote last is in the terminal, not yet in ``tty``, and a quiet spell does not say it has arrived (a
        # reader that a loaded machine held up for longer than the spell lost the end: "the live region is gone"). The
        # event is the terminal's own end, which the reader returns on once it has taken everything before it. A reader
        # that was stopped on purpose (``stop_reading``) is already over, and this returns at once.
        self._reader.join(timeout=END_WITHIN)
        self.reader_saw_the_end = not self._reader.is_alive()
        self._open = False
        self._reader.join(timeout=2)
        return (out or b"").decode("utf-8"), (err or b"").decode("utf-8"), self.process.returncode

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=60)
        self._open = False
        try:
            os.close(self.master)
        except OSError:
            pass


@pytest.fixture
def served(folder, tmp_path):
    made: list[Served] = []

    def start(**options) -> Served:
        child = Served(folder, tmp_path, **options)
        made.append(child)
        return child

    yield start
    for child in made:
        child.close()


@pytest.mark.parametrize("columns", [100, 60, 0])
def test_a_scripted_session_on_a_terminal_shows_each_event_once_above_a_strip_that_follows_the_status(served, columns):
    width = columns or 80
    child = served(terminals=("out", "err"), columns=columns)
    child.until(lambda s: any(line.startswith("READY") for line in s), "the strip, READY")
    live = child.client()
    child.until(lambda s: any("viewer connected" in line for line in s) and any("commander connected" in line for line in s),
                "both clients connected")
    assert live.request("start")["ok"]
    deadline = time.monotonic() + HANG
    while live.request("status")["activity"] != "stepping" or live.request("status")["step"] == 0:
        assert time.monotonic() < deadline, "never stepped"
        time.sleep(0.1)
    assert live.request("set", values={"offset": 0.002})["ok"]
    hostile = "bogus\x1b]0;pwned\x07\nSTOPPED forged"
    assert not live.request(hostile)["ok"]
    live.request("pause")
    status = live.request("status")
    while status["pending"] or status["state"] != "paused":
        assert time.monotonic() < deadline, "never paused"
        time.sleep(0.05)
        status = live.request("status")
    step = status["step"]
    screen = child.until(lambda s: any(line.startswith("PAUSED") and f"step {step:,}" in line for line in s),
                         f"PAUSED at step {step:,}")
    lines = screen.splitlines()
    from_strip = lines[next(i for i, line in enumerate(lines) if line.startswith("PAUSED")):]
    assert f"step {step:,}" in from_strip[0] and "wall " in from_strip[0]
    assert any("1 commander · 1 viewer" in line for line in from_strip) and any("Ctrl+C to stop" in line for line in from_strip)
    assert any(line.startswith("swim_speed:umr.speed") for line in from_strip)
    assert any(line.startswith("umr.pose[0:3]") for line in from_strip)
    header = next(line for line in from_strip if line.startswith("value"))
    assert "unit" in header and "now" in header
    assert max(len(line) for line in from_strip) <= width, (width, [line for line in from_strip if len(line) > width])
    live.close()
    child.until(lambda s: any("commander left" in line for line in s) and any("viewer left" in line for line in s),
                "both clients left")
    child.process.send_signal(signal.SIGINT)
    _, _, code = child.finish()
    raw = child.tty
    final = screen_of(raw)
    try:
        the_end_as_a_person_sees_it(child, code, raw, final, step)
    except AssertionError as failed:
        raise AssertionError(f"{failed}\n{what_the_terminal_had(child, raw, final)}") from None


def the_end_as_a_person_sees_it(child: Served, code: int, raw: str, final: list[str], step: int) -> None:
    """The session test's checks of the screen after the stop: each event once and in order, no escape of the client's,
    no live region, and the block that says how it stopped."""
    text = "\n".join(final)
    assert code == 0 and (child.out_dir / "outputs.json").is_file()
    events = [line[10:] for line in final if len(line) > 10 and line[2] == ":" and line[5] == ":" and line[8:10] == "  "]

    def count(part: str) -> int:
        return sum(event.startswith(part) for event in events)

    for part in ("viewer connected (127.0.0.1:", "commander connected (127.0.0.1:", "command: start (running from step 0)",
                 "command: set offset=0.002 (from step ", f"command: pause (paused at step {step:,})", "commander left",
                 "viewer left", "refused: bogus"):
        assert count(part) == 1, (part, events)
    assert count("command: pause (taken; it acts at the chunk's end)") <= 1, "taken while a chunk ran, or not"
    assert "\\x1b]0;pwned\\x07" in text and "\x1b]0;" not in raw, "the client's escape is written out, not sent"
    assert not any(line.startswith("STOPPED forged") for line in final), "a newline in a command's name forges no line"
    assert not any("Ctrl+C to stop" in line for line in final), "the live region is gone"
    assert any(line.startswith("STOPPED  stopped by Ctrl+C") for line in final)
    assert any("mime verify " in line for line in final) and str(child.out_dir) in text.replace("\n", "")
    places = {part: next(i for i, line in enumerate(final) if part in line) for part in (
        "viewer connected", "command: start", "commander left", "STOPPED  stopped by Ctrl+C")}
    assert list(places.values()) == sorted(places.values()), places


#: The most of each part of what the terminal had that a failure of the session test shows (characters).
SHOWN = 4000


def what_the_terminal_had(child: Served, raw: str, final: list[str]) -> str:
    """For a failure's message, so that one on a runner shows what happened: the screen after the stop, its lines
    numbered, and what the terminal received (its control sequences written out) where the strip was first drawn and at
    its end, each part bounded by :data:`SHOWN`."""
    screen = "\n".join(f"{index:3d} | {line}" for index, line in enumerate(final))
    first = max(0, raw.find("READY") - SHOWN // 4)
    return (f"the screen after the stop ({len(final)} lines; its last {SHOWN} characters):\n{screen[-SHOWN:]}\n"
            f"what the terminal received ({len(raw)} characters; the reader "
            f"{'saw' if child.reader_saw_the_end else 'did not see'} the terminal's end), where the strip was first "
            f"drawn:\n{raw[first:first + SHOWN]!r}\nand at its end:\n{raw[-SHOWN:]!r}")


# -- what a launcher and a plain terminal get ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def load_text(folder) -> str:
    """The load report as the command prints it, read once from the objects the command holds."""
    from mime.env import apply_lock_precision
    from mime.experiment import Experiment

    apply_lock_precision(folder)
    load = Experiment.load(folder).load_report
    assert load is not None
    return load.text()


def oracle(load_text: str, ready: dict) -> str:
    """Standard output as ``mime serve`` wrote it before the view existed, for a session stopped at step 0: the load
    report, the line when it listens, the line when it stops."""
    return (f"{load_text}\n"
            f"mime serve: tracked ready (mime.live/1); commands on {ready['commands']}, frames on {ready['frames']}; its run "
            f"folder will be {ready['out']}\n"
            f"mime serve: stopped at step 0; wrote {ready['out']}\n")


def stop_it(child: Served) -> None:
    live = child.client()
    assert live.request("stop")["ok"]
    live.close()


def planted_colour(text: str) -> bool:
    return "\x1b" in text


ARMS = {
    "piped": dict(terminals=(), env={}),
    "FORCE_COLOR+CI": dict(terminals=(), env={"FORCE_COLOR": "1", "CI": "true"}),
    "stdout-piped-stderr-terminal": dict(terminals=("err",), env={}),
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_standard_output_is_the_old_commands_byte_for_byte_wherever_it_is_a_pipe(served, load_text, arm):
    child = served(**ARMS[arm])
    ready = json.loads(child.ready.read_text())
    if arm.startswith("stdout-piped"):
        child.until(lambda s: any(line.startswith("READY") for line in s), "the view, on standard error")
    stop_it(child)
    out, err, code = child.finish()
    assert code == 0 and out == oracle(load_text, ready), "the launcher's lines, as they were"
    assert not planted_colour(out) and not planted_colour(err), "nothing styled on a pipe"
    if arm.startswith("stdout-piped"):
        assert "STOPPED" in stripped(child.tty), "and the view was drawn on the terminal that standard error is"


@pytest.mark.parametrize(("env", "plain"), [({"NO_COLOR": "1"}, False), ({"TERM": "dumb"}, False), ({}, True)],
                         ids=["NO_COLOR", "TERM=dumb", "--plain"])
def test_a_terminal_asked_to_be_plain_gets_what_a_pipe_gets_and_nothing_is_drawn(served, load_text, env, plain):
    child = served(terminals=("out", "err"), env=env, plain=plain)
    ready = json.loads(child.ready.read_text())
    stop_it(child)
    _, _, code = child.finish()
    raw = child.tty
    assert code == 0 and not planted_colour(raw), "no control sequence at all"
    printed = raw.replace("\r\n", "\n")
    expected = oracle(load_text, ready)
    assert expected in printed or all(line in printed for line in expected.splitlines()), printed
    assert "Ctrl+C to stop" not in printed and "READY" not in printed and "STOPPED" not in printed


def test_the_comparison_finds_a_colour_code_planted_in_the_old_output(load_text):
    ready = {"commands": "tcp://127.0.0.1:1", "frames": "tcp://127.0.0.1:2", "out": "/out"}
    clean = oracle(load_text, ready)
    assert not planted_colour(clean)
    planted = clean.replace("mime serve: tracked ready", "\x1b[1mmime serve:\x1b[0m tracked ready")
    assert planted_colour(planted) and planted != clean


# -- a terminal nobody reads -----------------------------------------------------------------------------------------------

#: How long, after a stop was asked for or its time came, the process has to have exited and written its run folder: a
#: bound that a hang fails and a slow machine does not (the first chunk compiles before a stop can act).
EXIT_WITHIN = 90.0
#: The ways a session is stopped that nobody asked for with a command: (the options, the signal sent once the terminal is full).
WAYS = {
    "max-wall": (("--max-wall", "6s"), None),
    "idle-timeout": (("--idle-timeout", "6s"), None),
    "SIGTERM": ((), signal.SIGTERM if sys.platform != "win32" else None),
    "Ctrl+C": ((), signal.SIGINT if sys.platform != "win32" else None),
}
#: Which standard streams are the terminal: both, as a user's is, and standard error alone, with standard output a pipe.
STREAMS = {"both": ("out", "err"), "stderr-only": ("err",)}


def run_to_its_stop(served, way: str, terminals: tuple[str, ...], reading: bool) -> tuple[Served, int]:
    """``mime serve --start --unpaced --watch all`` on a terminal, stopped the way ``way`` says; from the moment its ready
    file is written nobody reads the terminal unless ``reading``. The child and its exit code, waited for at most
    :data:`EXIT_WITHIN` seconds, a hang failing as one."""
    options, sent = WAYS[way]
    child = served(terminals=terminals, args=("--start", "--unpaced", "--watch", "all", *options))
    if not reading:
        child.stop_reading()
        time.sleep(1.5)  # the view draws four times a second: it is in a blocked write by now
        assert is_full(child.far), "the terminal was read by no one, and is still full"
    if sent is not None:
        child.process.send_signal(sent)
    try:
        _, _, code = child.finish(limit=EXIT_WITHIN + (6.0 if sent is None else 0.0))
    except subprocess.TimeoutExpired:
        pytest.fail(f"mime serve did not exit within {EXIT_WITHIN:.0f} s of {way} with a terminal "
                    f"{'read' if reading else 'nobody reads'} ({'+'.join(terminals)}): it is still running")
    return child, code


def written(child: Served) -> bool:
    return (child.out_dir / "outputs.json").is_file() and (child.out_dir / "run.json").is_file()


@pytest.mark.parametrize("way", list(WAYS))
def test_the_control_with_the_terminal_read_each_way_of_stopping_exits_and_writes_the_run_folder(served, way):
    child, code = run_to_its_stop(served, way, STREAMS["both"], reading=True)
    assert code == 0 and written(child), (way, code)
    assert any(line.startswith("STOPPED") for line in screen_of(child.tty)), "and the view said how it stopped"


@pytest.mark.parametrize("way", list(WAYS))
def test_a_terminal_nobody_reads_does_not_keep_serve_from_writing_the_run_folder_and_exiting(served, way):
    """F1 of the review of PR #122: with both standard streams on a terminal that stopped being read, ``--max-wall`` wrote no
    run folder and the process never exited, and SIGTERM wrote the folder and then hung in the last line it printed. The
    control above is the same run with the terminal read."""
    child, code = run_to_its_stop(served, way, STREAMS["both"], reading=False)
    assert code == 0 and written(child), (way, code)


@pytest.mark.parametrize("way", ["max-wall", "SIGTERM"])
def test_with_only_standard_error_unread_serve_still_exits_and_writes_the_run_folder(served, way):
    """The view's own end, with standard output a pipe: the process hung closing the view, whose lock the blocked draw held."""
    child, code = run_to_its_stop(served, way, STREAMS["stderr-only"], reading=False)
    assert code == 0 and written(child), (way, code)
