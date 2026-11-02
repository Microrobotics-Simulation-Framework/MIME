"""What ``mime`` shows on a terminal, and the only module that imports ``rich``.

**The rule.** When output is not going to a terminal it is byte for byte what it was before this module existed: a pipe,
a file, a CI log, Modal's, and tests' captured output are all read by programs. So is a terminal asked not to be
styled: ``NO_COLOR`` set, ``TERM=dumb``, or ``--plain``. Each stream is judged alone (:func:`styled`): progress is on
standard error and results on standard output, so ``mime run ... | tee log`` still shows its progress, and the pipe
gets what it always got. Colour never carries a meaning the words do not.

``rich`` is imported inside the functions that draw, so a command that draws nothing (``mime --help``, anything piped)
never loads it. Everything that reaches a terminal here is text from outside made plain first
(:func:`mime._terminal.for_terminal`) and drawn as :class:`rich.text.Text`, never as markup, so a name with brackets or
an escape in it is only text."""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, TextIO

from mime._terminal import for_terminal
from mime.cli._progress import FLAG_INTERVAL, RunState, duration_text

if TYPE_CHECKING:
    from rich.console import Console

    from mime.build import MimeError, Refusal
    from mime.cli._summary import RunSummary

#: Set by ``--plain``: no stream is styled.
_plain = False


def set_plain(on: bool) -> None:
    global _plain
    _plain = on


def styled(stream: IO[str]) -> bool:
    """Whether ``stream`` is drawn on: a terminal, not asked to be plain (``--plain``, ``NO_COLOR`` set and not empty,
    ``TERM=dumb``)."""
    if _plain or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _shares_a_terminal(other: IO[str], stream: IO[str]) -> bool:
    """Whether ``other`` is a terminal and the very one ``stream`` is. Only then do ``other``'s lines go through the
    display (above it, so they do not tear it); anywhere else, a pipe or a file or another terminal, they stay on
    ``other``: a line written to standard output is standard output's, whatever standard error is drawing."""
    try:
        if not other.isatty():
            return False
        here, there = os.fstat(other.fileno()), os.fstat(stream.fileno())
    except (AttributeError, ValueError, OSError):
        return False
    return (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino)


def _console(stream: IO[str]) -> Console:
    from rich.console import Console

    return Console(file=stream, highlight=False, markup=False, emoji=False)


def _text(line: str, style: str = "") -> Any:
    from rich.text import Text

    return Text(for_terminal(line), style=style, no_wrap=False, overflow="fold")


class RunDisplay:
    """The live line of ``mime run``: a spinner naming the phase while the experiment loads, builds and compiles, then the
    bar, the steps, the simulated time, the speed, the time remaining (an estimate), the flags raised so far and how to
    stop. It is drawn about four times a second whatever the run's speed; the run only stores its numbers into it."""

    def __init__(self, stream: IO[str]) -> None:
        from rich.live import Live

        self._stream = stream
        self._console = _console(stream)
        self._phase = ""
        self._phase_started = time.perf_counter()
        self.state: RunState | None = None
        self._live = Live(self, console=self._console, refresh_per_second=4, transient=True,
                          redirect_stdout=_shares_a_terminal(sys.stdout, stream), redirect_stderr=True)
        self._running = False

    def phase(self, text: str) -> None:
        """Name what is under way (a spinner and the seconds it has taken), starting the display if it is stopped."""
        self._phase, self._phase_started = text, time.perf_counter()
        self.state = None
        if not self._running:
            self._running = True
            self._live.start()

    def running(self, state: RunState) -> None:
        """From here the display shows the run's progress; until its first chunk ends it says it is compiling."""
        self.state = state
        self._phase, self._phase_started = "compiling the first chunk, which can take many seconds", state.started
        if not self._running:
            self._running = True
            self._live.start()

    def pause(self) -> None:
        """Take the display off the screen (before something is printed in its place)."""
        if self._running:
            self._running = False
            self._live.stop()

    close = pause

    def say(self, line: str) -> None:
        """A line above the display (Ctrl+C's acknowledgement), without breaking it."""
        self._console.print(_text(line))

    def leave(self, code: int) -> None:
        """The second Ctrl+C, which ends the process where it stands: the cursor the display hid is shown again first."""
        try:
            os.write(self._stream.fileno(), b"\x1b[?25h")
        except (OSError, ValueError, AttributeError):
            pass
        os._exit(code)

    def __rich_console__(self, console: Console, options: Any) -> Iterator[Any]:
        from rich.progress_bar import ProgressBar
        from rich.spinner import Spinner
        from rich.table import Table

        now = time.perf_counter()
        state = self.state
        unicode = (console.encoding or "").lower().startswith("utf")
        waiting = state is None or state.first_chunk is None
        if waiting:
            yield Spinner("dots" if unicode else "line",
                          text=_text(f"{self._phase}  ({duration_text(now - self._phase_started)})"))
            if state is not None:
                yield _text("Ctrl+C to stop", "dim")
            return
        assert state is not None
        grid = Table.grid(padding=(0, 1))
        grid.add_column()
        grid.add_column()
        grid.add_row(ProgressBar(total=state.steps_asked, completed=state.steps, width=28), _text(state.head()))
        yield grid
        yield _text(state.estimate_line(now), "dim")
        yield _text(state.flags_line(), "dim")


class FlagPoll:
    """Reads the flags raised at a chunk's end no more often than :data:`FLAG_INTERVAL`: reading them waits for the chunk's
    result on the device, so a run is not made to wait for it after every one of its many chunks."""

    def __init__(self, read: Any) -> None:
        self._read = read
        self._last = float("-inf")

    def __call__(self, now: float) -> Sequence[str] | None:
        if now - self._last < FLAG_INTERVAL:
            return None
        self._last = now
        return self._read()


#: JAX's words for an entry it could not read, which is what a damaged one is. Its words for one it could not *write* (a full
#: disk, a directory not writable) share the rest of the phrase, and are not damage: they get no note.
_READ_ERROR = "Error reading persistent compilation cache entry"


@contextmanager
def damaged_cache_note(stream: IO[str]) -> Generator[None]:
    """Beside JAX's warning of a damaged compilation cache entry (one it could not read), once, the command that clears it:
    JAX's words do not name one. A write error is not damage and gets nothing. Only where ``stream`` is styled; the warning itself is shown as JAX showed it."""
    if not styled(stream):
        yield
        return
    shown = warnings.showwarning
    said = False

    def show(message: Warning | str, category: type[Warning], filename: str, lineno: int, file: TextIO | None = None,
             line: str | None = None) -> None:
        nonlocal said
        shown(message, category, filename, lineno, file, line)
        if not said and _READ_ERROR in str(message):
            said = True
            (file or sys.stderr).write("mime: an entry of XLA's compilation cache is damaged; `mime cache clear` removes "
                                       "the cache (the next run builds it again), `--no-xla-cache` runs without it\n")

    warnings.showwarning = show
    try:
        yield
    finally:
        warnings.showwarning = shown


_WHO = {"user": "yours to fix", "environment": "the environment's to fix", "internal": "a defect, not yours",
        "unrecognised": "an error from below that MIME does not recognise"}


def show_refusal(error: MimeError, headline: str, footer: Sequence[str], stream: IO[str]) -> None:
    """A refused command as a block on ``stream``: for each refusal its code and who can act, what is wrong, what to do,
    why and where; then the docs address of each code, unbroken so that it can be copied, and the footer
    (the defect's owner and the full traceback's file) as the ordinary text gave it."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    console = _console(stream)
    parts: list[Any] = []
    for index, refusal in enumerate(error.refusals):
        if index:
            parts.append(Text(""))
        parts.append(_head(refusal))
        parts.append(_text(refusal.message))
        for label, held in (("what to do", refusal.action), ("why", refusal.why), ("where", refusal.where)):
            if held:
                parts.append(_labelled(label, held))
    count = len(error.refusals)
    title = headline if count == 1 else f"{headline} for {count} reasons"
    console.print(Panel(Group(*parts), title=for_terminal(title), title_align="left", border_style="red", expand=False,
                        width=min(console.width, 100)))
    for refusal in error.refusals:
        console.print(_text(f"docs for {refusal.code}: {refusal.page}"), soft_wrap=True)
    for line in footer:
        console.print(_text(line), soft_wrap=True)


def _head(refusal: Refusal) -> Any:
    from rich.text import Text

    head = Text(for_terminal(refusal.code), style="bold")
    head.append(f"   ({_WHO[refusal.category.value]})", style="dim")
    return head


def _labelled(label: str, held: str) -> Any:
    from rich.text import Text

    line = Text(f"{label}: ", style="bold")
    line.append(for_terminal(held))
    return line


def show_summary(summary: RunSummary, stream: IO[str], *, full: Sequence[tuple[str, str]] = ()) -> None:
    """The end of ``mime run`` on ``stream``: what needs attention first, then what ran, the readings, what was written,
    what to do next, and where the rest is. ``full`` adds the sections ``--full`` asks for, each a title and its text."""
    console = _console(stream)
    if summary.attention:
        console.print(_text(f"Needs attention ({len(summary.attention)})", "bold yellow"))
        for line in summary.attention:
            console.print(_text(f"  ! {line}"), soft_wrap=True)
    else:
        console.print(_text("Nothing needs attention: no flag raised, nothing computed at a limit.", "green"))
    for line in summary.ran:
        console.print(_text(line), soft_wrap=True)
    if summary.readings:
        console.print(_text("Readings", "bold"))
        for line in summary.readings:
            console.print(_text(f"  {line}"), soft_wrap=True)
    for line in summary.wrote:
        console.print(_text(line), soft_wrap=True)
    for title, text in full:
        console.print(_text(title, "bold"))
        console.print(_text(text), soft_wrap=True)
    console.print(_text("Next", "bold"))
    for line in summary.next_steps:
        console.print(_text(f"  {line}"), soft_wrap=True)
    console.print(_text(summary.more, "dim"), soft_wrap=True)
