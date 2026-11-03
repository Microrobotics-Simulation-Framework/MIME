"""A terminal nobody reads must not keep ``mime serve`` from stopping.

A write to a terminal that has stopped being read (Ctrl+S, a stalled ssh session, a launcher that gave serve a
pseudo-terminal it never drains) blocks once the terminal's buffer is full, and blocks for as long as nothing reads it.
:class:`BoundedWriter` is the stream ``mime serve`` writes to a terminal through: a write is queued and a thread of its own
makes it, so the callers (the serve loop's messages, the view's draws, the closing of the view) wait a bounded time
for it and never more. What cannot be written within that time is dropped, and counted in :attr:`BoundedWriter.dropped`.

The thread is a daemon and writes to the file descriptor directly, never through the interpreter's own buffered stream: a
thread blocked in a write holds that stream's lock, and the interpreter's flush at exit would wait on it for ever."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import IO, Any

#: How long a flush waits for the thread to have written what is queued, once (seconds). After a flush that gave up, the
#: next ones do not wait until the thread has caught up.
FLUSH_WAIT = 0.25
#: The longest the end of a command waits for the terminal to take what is left of what it wrote (seconds).
CLOSE_WAIT = 1.0
#: The most that is queued behind a write that is not finishing (bytes); a write that would pass it is dropped.
QUEUE_LIMIT = 64 * 1024


class BoundedWriter:
    """A text stream over ``stream``'s file descriptor that never blocks its caller for more than ``wait`` seconds once,
    and then not at all while the terminal does not take what is queued. A ``stream`` with no file descriptor (a
    capture, a buffer) is written to directly, as it is not a terminal that can stop being read."""

    def __init__(self, stream: IO[str], *, wait: float = FLUSH_WAIT, limit: int = QUEUE_LIMIT) -> None:
        self._stream = stream
        try:
            self._fd: int | None = stream.fileno()
        except (AttributeError, ValueError, OSError):
            self._fd = None
        self._wait, self._limit = wait, limit
        self._condition = threading.Condition()
        self._queue: deque[bytes] = deque()
        self._queued = 0  # the bytes queued or being written
        self._thread: threading.Thread | None = None
        self._behind = False  # a flush gave up, and the thread has not caught up since
        self._broken = False  # the terminal refused a write (hung up): nothing more is tried
        #: The writes dropped: queued behind a terminal that was not taking them, or after it hung up.
        self.dropped = 0
        self.encoding: str = getattr(stream, "encoding", None) or "utf-8"

    # -- what a console asks of a file ------------------------------------------------------------------------------------

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    def write(self, text: str) -> int:
        if self._fd is None:
            return self._stream.write(text)
        if os.linesep != "\n":
            text = text.replace("\n", os.linesep)
        data = text.encode(self.encoding, "replace")
        with self._condition:
            if self._broken or (self._queued and self._queued + len(data) > self._limit):
                self.dropped += 1
                return len(text)
            self._queue.append(data)
            self._queued += len(data)
            if self._thread is None:
                self._thread = threading.Thread(target=self._write_all, name="mime-serve-terminal", daemon=True)
                self._thread.start()
            self._condition.notify_all()
        return len(text)

    def flush(self) -> None:
        if self._fd is None:
            self._stream.flush()
        else:
            self.drain(0.0 if self._behind else self._wait)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    # -- the bound ------------------------------------------------------------------------------------------------------

    def drain(self, seconds: float) -> bool:
        """Wait up to ``seconds`` for what is queued to have been written; whether it was. A wait that gives up marks the
        writer as behind, so that the next flush does not wait again until the terminal has taken what is queued."""
        deadline = time.monotonic() + seconds
        with self._condition:
            while self._queued:
                left = deadline - time.monotonic()
                if left <= 0.0:
                    self._behind = True
                    return False
                self._condition.wait(left)
            self._behind = False
            return True

    def _write_all(self) -> None:
        assert self._fd is not None
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                data = self._queue.popleft()
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(self._fd, view):]
            except OSError:  # the terminal hung up, or the descriptor is closed: it takes nothing more
                with self._condition:
                    self._broken = True
                    self.dropped += 1 + len(self._queue)
                    self._queue.clear()
                    self._queued = 0
                    self._condition.notify_all()
                return
            with self._condition:
                self._queued -= len(data)
                self._condition.notify_all()


__all__ = ["CLOSE_WAIT", "BoundedWriter", "FLUSH_WAIT", "QUEUE_LIMIT"]
