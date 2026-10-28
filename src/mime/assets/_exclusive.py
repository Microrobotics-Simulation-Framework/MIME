"""A lock on one asset across processes: the cache is shared by every process on the machine that fetches, and by a
cluster's job array on a shared filesystem, so the one who stores a file holds the lock and the others wait, then find
it stored."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

LOCK_SUFFIX = ".lock"
_POLL_SECONDS = 0.05


def _acquire(descriptor: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        while True:  # msvcrt locks a byte range, and the blocking mode gives up after ten tries: poll instead
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(_POLL_SECONDS)
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _release(descriptor: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


@contextmanager
def exclusive(path: Path) -> Generator[None]:
    """Hold the lock file ``path`` (created if absent) until the block ends, waiting for whoever holds it.

    ``flock`` on POSIX (the lock goes with the open file, so a process that dies releases it) and ``msvcrt.locking`` on
    Windows; no other platform is supported. Two threads of one process exclude each other too: each opens its own
    descriptor. The block must not ask for the same lock again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _acquire(descriptor)
        try:
            yield
        finally:
            _release(descriptor)
    finally:
        os.close(descriptor)
