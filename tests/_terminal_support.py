"""A real pseudo-terminal for the tests of what ``mime`` draws on a terminal (POSIX; Windows has none, see
``_platform_support.needs_pty``): a stream in this process whose ``isatty()`` is true, and a child process started with
a terminal for its standard streams. What is drawn is read back as text with its control sequences; ``stripped`` removes
them (so that a line the display drew and erased is still in what is read: the tests look for what was drawn, not for
what is left on a screen)."""

from __future__ import annotations

import contextlib
import os
import re
import select
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO

if sys.platform != "win32":
    import fcntl
    import pty
    import termios

#: A control sequence: CSI (cursor, erase, colour, show/hide cursor) and OSC (title, link).
CONTROL = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Za-z0-9]")


def stripped(transcript: str) -> str:
    """The transcript without control sequences, line ends as ``\\n`` (a terminal writes ``\\r\\n``)."""
    return CONTROL.sub("", transcript).replace("\r\n", "\n").replace("\r", "\n")


def screen_of(transcript: str) -> list[str]:
    """The lines on a screen after ``transcript``: what a person sees, with what a live display erased and drew again
    resolved (``stripped`` keeps every frame that was ever drawn). It understands what ``rich`` writes to redraw a few
    lines (carriage return, line feed, cursor up, erase the line, and colour and cursor visibility, which draw nothing)
    and nothing else; a sequence it does not know raises, so a display that starts to write another is noticed here."""
    rows: list[list[str]] = [[]]
    row = col = 0
    index = 0
    while index < len(transcript):
        char = transcript[index]
        if char == "\x1b":
            match = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])").match(transcript, index)
            if match is None:
                raise ValueError(f"an escape sequence screen_of does not know: {transcript[index:index + 12]!r}")
            argument, final = match.group(1), match.group(2)
            count = int(argument) if argument.isdigit() else 1
            if final == "A":
                row = max(0, row - count)
            elif final == "K":
                rows[row] = [] if argument == "2" else rows[row][:col]
            elif final not in ("m", "h", "l"):
                raise ValueError(f"a control sequence screen_of does not know: {match.group(0)!r}")
            index = match.end()
            continue
        index += 1
        if char == "\r":
            col = 0
        elif char == "\n":
            row += 1
            col = 0
            while len(rows) <= row:
                rows.append([])
        else:
            line = rows[row]
            line.extend(" " * (col - len(line)))
            if col < len(line):
                line[col] = char
            else:
                line.append(char)
            col += 1
    return [text.rstrip() for text in ("".join(line) for line in rows)]


def _size(fd: int, columns: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


def _drain(master: int, wait: float = 0.0) -> bytes:
    held = b""
    while select.select([master], [], [], wait)[0]:
        try:
            more = os.read(master, 65536)
        except OSError:
            break
        if not more:
            break
        held += more
        wait = 0.0
    return held


def fill_unread(far: str) -> int:
    """Write to the far side of a pseudo-terminal, ``far`` its path (``os.ttyname`` of its slave: ``os.ptsname`` is Python
    3.13's), which nobody is reading, until it takes no more; the bytes it took. From then on a write to the terminal blocks until something reads it. The kernel moves
    what was written to the reading end after the write returns, which makes room again, so the filling goes on until
    several passes, a little apart, find no room."""
    fd = os.open(far, os.O_WRONLY | os.O_NONBLOCK | os.O_NOCTTY)
    taken, barren = 0, 0
    try:
        while barren < 4:
            before = taken
            for size in (1024, 64, 1):  # what a size leaves, the next smaller one fills
                while True:
                    try:
                        taken += os.write(fd, b"x" * size)
                    except BlockingIOError:
                        break
            barren = barren + 1 if taken == before else 0
            time.sleep(0.05)
        return taken
    finally:
        os.close(fd)


def is_full(far: str) -> bool:
    """Whether a write to the pseudo-terminal whose far side is the path ``far`` would block now."""
    fd = os.open(far, os.O_WRONLY | os.O_NONBLOCK | os.O_NOCTTY)
    try:
        os.write(fd, b"x")
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


class Terminal:
    """A pseudo-terminal whose ``stream`` a program in this process writes to as to a terminal. ``transcript()`` is
    everything written to it so far.

    A thread reads the terminal's far end all the while: a pseudo-terminal holds only a little unread (about a kilobyte
    on macOS), a write past that blocks, and a display drawing from its own thread while the test waits for it would
    block for good."""

    def __init__(self, columns: int = 100, rows: int = 30) -> None:
        self._master, slave = pty.openpty()
        _size(self._master, columns, rows)
        self.stream: IO[str] = os.fdopen(slave, "w", buffering=1, encoding="utf-8", newline="")
        self._held = bytearray()
        self._lock = threading.Lock()
        self._open = True
        self._reader = threading.Thread(target=self._read, name="terminal-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while self._open:
            try:
                if not select.select([self._master], [], [], 0.02)[0]:
                    continue
                more = os.read(self._master, 65536)
            except (OSError, ValueError):
                return
            if not more:
                return
            with self._lock:
                self._held += more

    def transcript(self) -> str:
        self.stream.flush()
        time.sleep(0.08)  # what was just written is on its way through the terminal to the reader
        with self._lock:
            return bytes(self._held).decode("utf-8", "replace")

    def close(self) -> None:
        self._open = False
        self._reader.join(timeout=2)
        with contextlib.suppress(OSError):
            self.stream.close()
        with contextlib.suppress(OSError):
            os.close(self._master)


@contextlib.contextmanager
def terminal(columns: int = 100, rows: int = 30) -> Iterator[Terminal]:
    held = Terminal(columns, rows)
    try:
        yield held
    finally:
        held.close()


def untranslated(fd: int) -> None:
    """The terminal ``fd`` with its output processing off (``OPOST``), so that what a program writes reaches the far end
    as written. On, a line end becomes a carriage return and a line feed, and macOS's terminal, where its output queue
    fills between the two (2048 bytes into one write), writes the carriage return twice."""
    attributes = termios.tcgetattr(fd)
    attributes[1] &= ~termios.OPOST
    termios.tcsetattr(fd, termios.TCSANOW, attributes)


def run_on_terminal(argv: Sequence[str], *, env: Mapping[str, str], cwd: Path, columns: int = 100,
                    timeout: float = 120.0, translated: bool = True) -> tuple[str, int]:
    """``argv`` run with a terminal for its standard input, output and error: everything it wrote, and its exit code.
    Without ``translated``, the terminal's output processing is off and the transcript is the bytes the program wrote
    (:func:`untranslated`): what a test comparing them with a pipe's byte for byte reads."""
    master, slave = pty.openpty()
    _size(master, columns, 30)
    if not translated:
        untranslated(slave)
    child = subprocess.Popen(list(argv), stdin=slave, stdout=slave, stderr=slave, env=dict(env), cwd=cwd,
                             close_fds=True)
    os.close(slave)
    held, deadline = b"", time.monotonic() + timeout
    try:
        while True:
            if time.monotonic() > deadline:
                child.kill()
                raise TimeoutError(f"{argv[0]} ran longer than {timeout} s; it wrote so far:\n"
                                   f"{held.decode('utf-8', 'replace')}")
            if select.select([master], [], [], 0.1)[0]:
                try:
                    more = os.read(master, 65536)
                except OSError:  # EIO: the child closed its end
                    break
                if not more:
                    break
                held += more
            elif child.poll() is not None:
                held += _drain(master)
                break
        return held.decode("utf-8", "replace"), child.wait()
    finally:
        os.close(master)


def run_with_streams(argv: Sequence[str], *, env: Mapping[str, str], cwd: Path, terminals: Sequence[str],
                     columns: int = 100, timeout: float = 120.0, typed: bytes = b"") -> tuple[dict[str, str], int]:
    """``argv`` run with some of its standard streams a terminal and the others pipes: ``terminals`` names which
    (``"out"``, ``"err"``; standard input is the terminal when there is one). What each stream received, by name (``"tty"``
    for the terminal, whichever streams it carries), and the exit code. The terminal is read while the child runs, so a
    child writing more than a pseudo-terminal holds does not block. ``typed`` is what the person types, waiting in the
    terminal's input for the child to read it (the terminal echoes it, so it is on the terminal's side of the transcript)."""
    master, slave = pty.openpty()
    _size(master, columns, 30)
    child = subprocess.Popen(list(argv), stdin=slave, stdout=slave if "out" in terminals else subprocess.PIPE,
                             stderr=slave if "err" in terminals else subprocess.PIPE, env=dict(env), cwd=cwd, close_fds=True)
    os.close(slave)
    if typed:
        os.write(master, typed)
    open_ends: dict[int, str] = {master: "tty"}
    for name, stream in (("out", child.stdout), ("err", child.stderr)):
        if stream is not None:
            open_ends[stream.fileno()] = name
    held = {"tty": b"", "out": b"", "err": b""}
    deadline = time.monotonic() + timeout
    try:
        while open_ends:
            if time.monotonic() > deadline:
                child.kill()
                raise TimeoutError(f"{argv[0]} ran longer than {timeout} s")
            ready = select.select(list(open_ends), [], [], 0.1)[0]
            for fd in ready:
                try:
                    more = os.read(fd, 65536)
                except OSError:  # EIO: the child closed its end of the terminal
                    more = b""
                if more:
                    held[open_ends[fd]] += more
                else:
                    del open_ends[fd]
            if not ready and child.poll() is not None:
                for fd in list(open_ends):
                    held[open_ends[fd]] += _drain(fd)
                break
        code = child.wait()
    finally:
        os.close(master)
        for stream in (child.stdout, child.stderr):
            if stream is not None:
                stream.close()
    return {name: data.decode("utf-8", "replace") for name, data in held.items()}, code
