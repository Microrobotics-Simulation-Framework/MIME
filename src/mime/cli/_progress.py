"""What a terminal shows of a run in progress, as plain data: no terminal and no ``rich`` here, so the arithmetic is
tested without one and ``mime.cli._display`` only draws it.

A run is compiled chunks; after each but the last, ``stop`` is asked with the chunk's :class:`~mime.control.Progress`, which
is where this is told. Three things are honest about what it says:

* **Time remaining is an estimate.** It is the steps left over the speed of the chunks since the last one that
  compiled finished. A chunk compiles when its length is one the run has not had before (the first, and in a run of two
  lengths the second too: each length is compiled once). Until a chunk that did not compile has run there is no speed
  to extrapolate, and it says so.
* **A flag is known at a chunk's last step.** One raised and gone again inside a chunk is not seen, so it is said to be
  raised *by* a step, the end of the chunk it was first seen at, never *at* one.
* **The last chunk is not announced.** ``stop`` is not asked after it, so the display cannot tell the run's end from
  the chunk before it; the run folder is written after it, which can take longer than a chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


def duration_text(seconds: float) -> str:
    """A span of time as a person reads it: ``850 ms``, ``4.2 s``, ``12 s``, ``3 min 5 s``, ``2 h 10 min``."""
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 10.0:
        return f"{seconds:.1f} s"
    whole = round(seconds)
    if whole < 60:
        return f"{whole} s"
    minutes, rest = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes} min {rest} s" if rest else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def simulated_text(seconds: float) -> str:
    """A simulated time with a unit that keeps its digits few: ``0.8 ms``, ``41 ms``, ``2.5 s``."""
    magnitude = abs(seconds)
    if magnitude == 0.0:
        return "0 s"
    if magnitude >= 1000.0:
        return f"{seconds:,.0f} s"
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if magnitude >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds:.3g} s"


@dataclass
class RunState:
    """Where a run has got to: the steps it is asked for, the base step, and what each chunk reported."""

    steps_asked: int
    dt: float
    #: When the run began (``time.perf_counter()``), compiling included.
    started: float
    steps: int = 0
    first_chunk: tuple[float, int] | None = None
    #: When and at which step the last chunk that compiled (the first of a length not had before) ended: the speed is
    #: measured from here.
    compiled_chunk: tuple[float, int] | None = None
    last_chunk: tuple[float, int] | None = None
    chunk_steps: int = 0
    #: The chunk lengths the run has had, each compiled once.
    lengths: set[int] = field(default_factory=set[int])
    #: The flags seen raised so far, with the step of the chunk's end each was first seen at.
    flags: dict[str, int] = field(default_factory=dict[str, int])

    def chunk(self, steps: int, now: float, raised: Sequence[str] | None = None) -> None:
        """A chunk ended at ``steps`` (counted from the run's start) at ``now``; ``raised`` the flags raised at its
        last step, when they were looked at (``None``: not looked at this time)."""
        self.chunk_steps = steps - self.steps
        self.steps = steps
        if self.first_chunk is None:
            self.first_chunk = (now, steps)
        if self.chunk_steps not in self.lengths:
            self.lengths.add(self.chunk_steps)
            self.compiled_chunk = (now, steps)
        self.last_chunk = (now, steps)
        for path in raised or ():
            self.flags.setdefault(path, steps)

    @property
    def fraction(self) -> float:
        return min(1.0, self.steps / self.steps_asked) if self.steps_asked else 0.0

    @property
    def time(self) -> float:
        """The simulated time reached, in seconds."""
        return self.steps * self.dt

    def steps_per_second(self) -> float | None:
        """The speed since the last chunk that compiled finished, or ``None`` before a chunk that did not compile has."""
        if self.compiled_chunk is None or self.last_chunk is None or self.last_chunk[1] == self.compiled_chunk[1]:
            return None
        elapsed = self.last_chunk[0] - self.compiled_chunk[0]
        return (self.last_chunk[1] - self.compiled_chunk[1]) / elapsed if elapsed > 0.0 else None

    def remaining(self, now: float) -> float | None:
        """An estimate of the seconds left: the steps left at :meth:`steps_per_second`, less the time since the last
        chunk finished (the chunk under way has been running that long). ``None`` before there is a speed."""
        speed = self.steps_per_second()
        if speed is None or self.last_chunk is None:
            return None
        return max(0.0, (self.steps_asked - self.steps) / speed - (now - self.last_chunk[0]))

    @property
    def last_chunk_under_way(self) -> bool:
        """Whether the chunk under way is the last one: no more steps are left than the chunk before it took."""
        return self.chunk_steps > 0 and self.steps_asked - self.steps <= self.chunk_steps

    def estimate_line(self, now: float) -> str:
        """The second line, in plain words: the speed, and the time remaining as an estimate."""
        speed = self.steps_per_second()
        parts = [f"{speed:,.0f} steps/s" if speed is not None else "speed: after a chunk that did not compile"]
        if self.last_chunk_under_way:
            parts.append("finishing the last chunk, then writing the run folder")
        else:
            left = self.remaining(now)
            parts.append(f"about {duration_text(left)} left (an estimate)" if left is not None else
                         "time left: an estimate after a chunk that did not compile")
        return "  ·  ".join(parts)

    def flags_line(self) -> str:
        """The third line: the flags raised so far, and how to stop."""
        return f"{self.flags_text()}  ·  Ctrl+C to stop"

    def flags_text(self) -> str:
        if not self.flags:
            return "no flag raised yet (seen at chunk ends)"
        return "flags raised: " + ", ".join(f"{path} by step {step:,}" for path, step in self.flags.items())

    def head(self) -> str:
        """The first line: how far, in steps and in simulated time."""
        return (f"{self.fraction:4.0%}  step {self.steps:,} of {self.steps_asked:,}  ·  "
                f"{simulated_text(self.time)} of {simulated_text(self.steps_asked * self.dt)} simulated")


#: How often the flags are looked at, at most (seconds): reading them waits for the chunk's result on the device.
FLAG_INTERVAL = 0.25
