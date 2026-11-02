"""Ctrl+C on a command that waits on compiled code. Python runs a signal handler between bytecodes, so while the main
thread is inside an XLA call (a compile, or a scan of hours) a handler written in Python waits for the call to return.
What does not wait is the signal's own C handler, which writes the signal's number to the wakeup file descriptor the
moment it arrives, and XLA releases the GIL for the whole of a compile and of a run, so a thread of ours reading that
descriptor answers at once.

The first press is acknowledged by that thread in one line. What the command does about it depends on where it is:
while it loads and builds, the handler raises ``KeyboardInterrupt`` in the main thread as soon as Python gets there,
and the command reports it and exits 130; once :meth:`Interrupt.running` is called, the press asks for the run to stop
at its next chunk (:meth:`Interrupt.stop`, passed as ``stop=``), which the run does, writing what it reached. The
second press is acted on by the thread alone, wherever the main thread is: it says so and ends the process with 130,
writing nothing. A run folder is written so that this leaves nothing that reads as a finished run (``outputs.json`` is
its last file)."""

from __future__ import annotations

import os
import signal
import socket
import threading
from collections.abc import Callable
from types import FrameType, TracebackType
from typing import Self

#: The exit code of a command ended by Ctrl+C, as a shell reports one (128 + SIGINT).
INTERRUPTED = 130
_STOPPING_AT_CHUNK = ("Ctrl+C: stopping at the end of this chunk and writing the run as far as it got; press Ctrl+C "
                      "again to quit now, writing nothing")
_STOPPING = "Ctrl+C: stopping; press Ctrl+C again to quit now"
_QUITTING = "Ctrl+C again: quitting now, nothing written"
#: The reason a run records when this stopped it.
USER_STOP = "stopped by the user (Ctrl+C)"


def _write_stderr(line: str) -> None:
    """One line on standard error, written straight to the descriptor: no buffer to lock, nothing a stopped main thread
    could be holding."""
    try:
        os.write(2, (line + "\n").encode("utf-8", "replace"))
    except OSError:  # no standard error to write to
        pass


class Interrupt:
    """Ctrl+C for the length of a ``with`` block (the main thread's SIGINT; in another thread it does nothing). ``leave``
    ends the process on the second press, and ``say`` writes the lines, both replaceable for a test."""

    def __init__(self, *, leave: Callable[[int], object] = os._exit,
                 say: Callable[[str], None] = _write_stderr) -> None:
        self._leave = leave
        self._say = say
        self._lock = threading.Lock()
        self._presses = 0
        self._first = threading.Event()
        self._at_chunk = False
        self._message = _STOPPING
        self._previous: signal.Handlers | Callable[[int, FrameType | None], object] | int | None = None
        self._channel: tuple[socket.socket, socket.socket] | None = None
        self._watcher: threading.Thread | None = None

    def __enter__(self) -> Self:
        if threading.current_thread() is not threading.main_thread():
            return self
        reader, writer = socket.socketpair()
        writer.setblocking(False)
        self._previous = signal.signal(signal.SIGINT, self._handler)
        signal.set_wakeup_fd(writer.fileno(), warn_on_full_buffer=False)
        self._channel = (reader, writer)
        self._watcher = threading.Thread(target=self._watch, args=(reader,), name="mime-interrupt", daemon=True)
        self._watcher.start()
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        if self._channel is None:
            return
        reader, writer = self._channel
        self._channel = None
        signal.set_wakeup_fd(-1)
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
        writer.close()
        if self._watcher is not None:
            self._watcher.join(timeout=2)
        reader.close()

    def running(self) -> None:
        """From here a first press asks the run to stop at its next chunk, and no longer raises ``KeyboardInterrupt``
        in the main thread."""
        with self._lock:
            self._at_chunk = True
            self._message = _STOPPING_AT_CHUNK

    def stop(self, progress: object = None) -> str | None:
        """The ``stop=`` of a run (given the chunk's progress, which it does not need): its reason once Ctrl+C has been
        pressed, else ``None``."""
        return USER_STOP if self._first.is_set() else None

    @property
    def pressed(self) -> bool:
        """Whether Ctrl+C has been pressed since the block began."""
        return self._first.is_set()

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        """Runs in the main thread, when Python gets to it: raises while nothing is waiting to be asked to stop."""
        if not self._at_chunk:
            raise KeyboardInterrupt

    def _watch(self, reader: socket.socket) -> None:
        while True:
            try:
                numbers = reader.recv(64)
            except OSError:
                return
            if not numbers:
                return
            for number in numbers:
                if number == signal.SIGINT:
                    self._pressed()

    def _pressed(self) -> None:
        with self._lock:
            self._presses += 1
            presses, message = self._presses, self._message
        if presses == 1:
            self._first.set()
            self._say(message)
        elif presses == 2:
            self._say(_QUITTING)
            self._leave(INTERRUPTED)
