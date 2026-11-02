"""Keep a test's worker under a ceiling of resident memory by giving JAX's compiled programs back.

JAX keeps every program it compiles for as long as the function that made it can be called again, and a test that
makes a new closure for each example (MADDENING's ``params_effective`` check differentiates the dynamic arm's step
afresh for each example of its two hundred) never calls one twice: the arm battery's check held 7.7 GB at its end,
all of it programs (clearing JAX's caches every 15 s kept it between 0.5 and 1.6 GB, and the arrays it holds came to
under a megabyte). ``jax.clear_caches()`` changes no number, only what is compiled again, and a program that is
needed again comes from CI's persistent compilation cache.

The ceiling is checked by a thread, because the test is inside the check and calls nothing of ours between examples.
It is a headroom above where the worker stands when the block begins, not an absolute size: a worker that ran other
tests before holds their programs and arrays, and an absolute limit under what it holds would clear the caches at every
look.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

STATM = Path("/proc/self/statm")


def resident_mib() -> float | None:
    """This process's resident memory in MiB; ``None`` where there is no ``/proc`` (the ceiling then does nothing)."""
    try:
        return int(STATM.read_text(encoding="ascii").split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return None


def give_back_programs() -> None:
    """Clear JAX's caches, collect, and return glibc's free memory to the system."""
    import jax

    jax.clear_caches()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class Ceiling:
    """What a :func:`under_ceiling` block did: how often it gave the programs back."""

    def __init__(self) -> None:
        self.cleared = 0


@contextlib.contextmanager
def under_ceiling(headroom_mib: float, poll: float = 0.5, quiet: float = 5.0) -> Iterator[Ceiling]:
    """For the block, keep this process within ``headroom_mib`` of resident memory above where it stands after the
    programs earlier tests of its worker kept have been given back, which the block does first. Whenever the process
    is over that, JAX's caches are cleared (looked at every ``poll`` seconds, and not twice within ``quiet`` seconds),
    and the limit becomes where the process then stands plus the headroom: memory that is not programs and grows is
    not cleared for again and again."""
    record = Ceiling()
    if resident_mib() is None:
        yield record
        return
    give_back_programs()
    limit = (resident_mib() or 0.0) + headroom_mib
    done = threading.Event()

    def watch() -> None:
        nonlocal limit
        last = time.monotonic() - quiet
        while not done.wait(poll):
            now = resident_mib()
            if now is None or now <= limit or time.monotonic() - last < quiet:
                continue
            give_back_programs()
            record.cleared += 1
            last = time.monotonic()
            limit = (resident_mib() or now) + headroom_mib

    watcher = threading.Thread(target=watch, name="cache-ceiling", daemon=True)
    watcher.start()
    try:
        yield record
    finally:
        done.set()
        watcher.join()
