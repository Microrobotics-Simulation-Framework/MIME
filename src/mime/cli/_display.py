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

import difflib
import functools
import os
import re
import sys
import time
import warnings
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Concatenate, TextIO

from mime._said import (
    DONE,
    EDIT_OURS,
    EDIT_SITE,
    LINK,
    PERMISSION,
    PLAN_FIELD,
    PLAN_HEAD,
    PLAN_LINE,
    PLAN_REFERENCE,
    PLAN_SECTION,
    PLAN_UP,
    PLAN_WARNING,
    STEP_HEAD,
    STEP_IRREVERSIBLE,
    STEP_LINE,
    STEP_TRIAL,
    STOPPED,
    ends_block,
    kind_of,
)
from mime._terminal import for_terminal
from mime.cli._progress import FLAG_INTERVAL, RunState, TransferState, duration_text

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


#: The width drawn at where the terminal's own cannot be read, or reads as nothing.
FALLBACK_COLUMNS = 80


def _columns_of(stream: IO[str]) -> int:
    """The width of the terminal ``stream`` is: its size; failing that ``COLUMNS``; failing that
    :data:`FALLBACK_COLUMNS`. A pseudo-terminal nobody sized (a harness's, ``script``'s, ``docker exec -t`` before its first
    resize) reads as 0 columns and not as an error, and a console told to draw 0 columns draws nothing: not the plan, and
    not a question."""
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, ValueError, OSError):
        columns = 0
    if columns > 0:
        return columns
    from_environment = os.environ.get("COLUMNS", "")
    return int(from_environment) if from_environment.isdigit() and int(from_environment) > 0 else FALLBACK_COLUMNS


def _console_of(stream: IO[str]) -> Console:
    """A console for ``stream`` as wide as the terminal ``stream`` is (rich asks the process's own standard streams, which
    are not ``stream`` when it is another terminal), and never 0 columns wide."""
    from rich.console import Console

    return Console(file=stream, highlight=False, markup=False, emoji=False, width=_columns_of(stream))


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
    (the defect's owner and the full traceback's file) as the ordinary text gave it. Refusals of one code that are
    about a subject and share their finding, action, why and where (:func:`refusal_groups`) are one block: the code
    once, what is wrong once, a row for each subject, what to do once."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    console = _console(stream)
    parts: list[Any] = []
    blocks = refusal_groups(error.refusals)
    for index, block in enumerate(blocks):
        if index:
            parts.append(Text(""))
        first = block[0]
        parts.append(_head(first))
        shared = first.subject if len(block) > 1 else None
        if shared is None:
            parts.append(_text(first.message))
        else:
            parts.append(_text(f"{len(block)} {shared.finding}:"))
            parts.extend((_subjects(block), Text("")))
        for label, held in (("what to do", first.action if shared is None else shared.action),
                            ("why", first.why), ("where", first.where)):
            if held:
                parts.append(_labelled(label, held))
    count = len(error.refusals)
    title = headline if count == 1 else f"{headline} for {count} reasons"
    console.print(Panel(Group(*parts), title=for_terminal(title), title_align="left", border_style="red", expand=False,
                        width=min(console.width, 100)))
    for block in blocks:
        console.print(_text(f"docs for {block[0].code}: {block[0].page}"), soft_wrap=True)
    for line in footer:
        console.print(_text(line), soft_wrap=True)


def refusal_groups(refusals: Sequence[Refusal]) -> list[list[Refusal]]:
    """The refusals as the blocks a terminal draws them in, in the order of each block's first refusal: refusals about a
    subject that share a code, a finding, the columns, the action and the why and where are one block, and every other
    refusal is a block of its own, so that a code whose actions differ per item is never merged. Every refusal is in
    exactly one block."""
    blocks: list[list[Refusal]] = []
    held: dict[tuple[object, ...], list[Refusal]] = {}
    for refusal in refusals:
        subject = refusal.subject
        if subject is None:
            blocks.append([refusal])
            continue
        key = (refusal.code, refusal.category, subject.finding, subject.columns, subject.action, refusal.why,
               refusal.where)
        if key not in held:
            held[key] = []
            blocks.append(held[key])
        held[key].append(refusal)
    return blocks


def _subjects(block: Sequence[Refusal]) -> Any:
    """The subjects of a block as a table, a row for each, the columns as the subject names them."""
    from rich.table import Table

    columns = block[0].subject.columns if block[0].subject is not None else ()
    table = Table(box=None, show_header=True, header_style="dim", pad_edge=False, padding=(0, 2, 0, 0), expand=False)
    for column in columns:
        table.add_column(for_terminal(column), overflow="fold")
    for refusal in block:
        if refusal.subject is not None:
            table.add_row(*(_text(cell) for cell in refusal.subject.cells))
    return table


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


#: How each kind of plan line is drawn (the words are the line's own; the style only sets it apart).
_PLAN_STYLE = {PLAN_HEAD: "bold", PLAN_UP: "green", PLAN_REFERENCE: "cyan", PLAN_FIELD: "", PLAN_LINE: "dim",
               PLAN_SECTION: "bold"}
_STEP_KINDS = frozenset({STEP_HEAD, STEP_IRREVERSIBLE, STEP_TRIAL, STEP_LINE})
#: The two values of a site edit: ``    on the site:  <value>`` and ``    <what writes it>: <value>``.
_EDIT_LINE = re.compile(r"\s+(?P<who>.+?):\s+(?P<value>.*)", re.DOTALL)


def _is_warning(line: str) -> bool:
    """A warning is a line that starts ``! ``: push's and pull's own convention, the same on a pipe."""
    return line.startswith("! ")


def _marked(a: str, b: str) -> tuple[Any, Any]:
    """Two values as text, the stretches where they differ in bold and underlined, so that the eye finds them; the values
    are whole, and what the marks say, the two lines beside them say too."""
    from rich.text import Text

    one, other = Text(for_terminal(a)), Text(for_terminal(b))
    matched = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matched.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            one.stylize("bold underline red", i1, i2)
        if j2 > j1:
            other.stylize("bold underline green", j1, j2)
    return one, other


def _too_wide(rows: Sequence[Any], width: int) -> bool:
    """Whether a word of any of ``rows`` (text) is longer than ``width`` columns: a hash or an address, which a frame
    narrower than it would fold across lines."""
    from rich.cells import cell_len

    return any(cell_len(word) > width for row in rows for word in str(getattr(row, "plain", row)).split())


class SaidDisplay:
    """What ``mime push`` and ``mime pull`` say, on a terminal: the plan as one block with what goes up and what travels
    as a reference told apart, a warning standing out, the publish step in a heavy frame with the irreversible sentence
    reversed, each permission in a frame of its own with its question, the two values of a site edit side by side, and
    each question set apart from the lines around it.

    A line arrives as push said it (a :class:`~mime._said.Said`, or a plain ``str``); lines of one block are held until
    the block's last line is said (``last``) or the next line is of another kind, so that the block is drawn whole, and
    drawn before whatever the command does next. The words are exactly push's: a frame, a colour or a bold changes how a
    line is set, never what it says, and the same lines on a pipe are the ones they were.

    A question, its statement and its hint are on one terminal, the one the person answers at: the question is drawn on
    ``stream`` and its hint (``[y/N] ``, ``> ``) written there too, never left to ``input``, which puts a prompt on
    standard error."""

    def __init__(self, stream: IO[str], transfers: TransferDisplay | None = None) -> None:
        self._transfers = transfers
        self._console = transfers.console if transfers is not None and _shares_a_terminal(stream, transfers.stream) \
            else _console_of(stream)
        self._plan: list[tuple[str, str]] = []
        self._warnings: list[str] = []
        self._step: list[tuple[str, str]] = []
        self._permission: list[str] = []
        self._site: str | None = None

    # --- lines -------------------------------------------------------------------------------------------------------

    def say(self, line: str) -> None:
        kind = kind_of(line)
        if self._transfers is not None and kind in (DONE, LINK, STOPPED):
            self._transfers.finish()  # the transfers are over: their last frame stays above what is said of them
        if kind == PLAN_WARNING:
            self._settle(keep="plan")
            self._warnings.append(line)
        elif kind is not None and kind.startswith("plan-"):
            self._settle(keep="plan")
            self._plan.append((kind, line))
        elif _is_warning(line) and self._plan:
            self._warnings.append(line)
        elif kind is not None and kind in _STEP_KINDS:
            self._settle(keep="step")
            self._step.append((kind, line))
        elif kind == PERMISSION:
            self._settle(keep="permission")
            self._permission.append(line)
        elif kind == EDIT_SITE:
            self._settle(keep="")
            self._site = line
        elif kind == EDIT_OURS and self._site is not None:
            site, self._site = self._site, None
            self._edit(site, line)
        else:
            self._settle(keep="")
            self._plain(line, kind)
        if ends_block(line) and kind is not None:
            if kind.startswith("plan-"):
                self._draw_plan()
            elif kind in _STEP_KINDS:
                self._draw_step()

    def close(self) -> None:
        """Whatever is held is drawn: the command ended."""
        self._settle(keep="")
        if self._transfers is not None:
            self._transfers.finish()

    # --- questions ---------------------------------------------------------------------------------------------------

    def ask(self, question: str) -> bool:
        """A yes-or-no question as a person reads it: set apart, then answered at ``[y/N]``; anything but y or yes is no,
        as it is where there is no display (:func:`mime.cli._ask`)."""
        self._put_question(question)
        try:
            return self._answer("[y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    def typed(self, prompt: str) -> str:
        """A line typed after ``prompt``, which is set apart as a question is."""
        self._put_question(prompt.rstrip())
        try:
            return self._answer("> ")
        except EOFError:
            return ""

    def _answer(self, hint: str) -> str:
        """The hint on the terminal the question is on, then the line typed. ``input`` is given no prompt of its own: it
        would put one on standard error, which may be a file while the question is on the screen."""
        stream = self._console.file
        stream.write(hint)
        stream.flush()
        return input()

    def _put_question(self, question: str) -> None:
        from rich.text import Text

        if self._transfers is not None:
            self._transfers.hide()
        self._settle(keep="")
        try:
            self._console.print()
            head = Text("? ", style="bold magenta")
            head.append(for_terminal(question), style="bold")
            self._console.print(head, soft_wrap=True)
        except Exception:  # noqa: BLE001 - a question is always asked, in the plainest way when it cannot be set apart
            stream = self._console.file
            stream.write(f"\n? {for_terminal(question)}\n")
            stream.flush()

    # --- blocks ------------------------------------------------------------------------------------------------------

    def _settle(self, keep: str) -> None:
        """Draw every held block but the one still being collected (``keep``)."""
        if keep != "plan":
            self._draw_plan()
        if keep != "step":
            self._draw_step()
        if keep != "permission":
            self._draw_permission()
        if self._site is not None:
            line, self._site = self._site, None
            self._plain(line, EDIT_SITE)

    def _frame(self, rows: Sequence[Any], *, style: str, heavy: bool = False, double: bool = False) -> None:
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.rule import Rule

        width = min(self._console.width, 100)
        if _too_wide(rows, width - 4):
            # A word wider than the frame's inside (a hash, an address) would be folded across lines, and a hash is
            # copied whole: the rows are drawn with no side to break them, between two rules, and the terminal wraps them.
            line = "\u2501" if heavy else "\u2550" if double else "\u2500"
            self._console.print(Rule(style=style, characters=line))
            for row in rows:
                self._console.print(row, soft_wrap=True)
            self._console.print(Rule(style=style, characters=line))
            return
        self._console.print(Panel(Group(*rows), box=box.HEAVY if heavy else box.DOUBLE if double else box.ROUNDED,
                                  border_style=style, expand=False, width=width))

    def _draw_plan(self) -> None:
        if self._plan:
            self._frame([_text(line, _PLAN_STYLE.get(kind, "")) for kind, line in self._plan], style="blue")
            self._plan = []
        for warning in self._warnings:
            self._warning(warning)
        self._warnings = []

    def _warning(self, line: str) -> None:
        self._console.print(_text(line, "bold yellow"), soft_wrap=True)

    def _draw_step(self) -> None:
        """The publish step, whole, and on the screen when this returns: it is drawn before the request it describes."""
        if not self._step:
            return
        production = any(kind == STEP_IRREVERSIBLE for kind, _ in self._step)
        styles = {STEP_HEAD: "bold", STEP_IRREVERSIBLE: "bold white on red", STEP_TRIAL: "bold yellow",
                  STEP_LINE: ""}
        self._frame([_text(line, styles[kind]) for kind, line in self._step], style="red" if production else "yellow",
                    heavy=True)
        self._step = []
        self._console.file.flush()

    def _draw_permission(self) -> None:
        if not self._permission:
            return
        self._frame([_text(line) for line in self._permission], style="magenta", double=True)
        self._permission = []

    def _edit(self, site: str, ours: str) -> None:
        """The two values of an edited field side by side, one under what the site holds and one under what ``sharing.yaml``
        (or the file that writes the field) would write, the differing stretches marked."""
        from rich import box
        from rich.table import Table
        from rich.text import Text

        held, writes = _EDIT_LINE.fullmatch(site), _EDIT_LINE.fullmatch(ours)
        if held is None or writes is None:
            self._plain(site, EDIT_SITE)
            self._plain(ours, EDIT_OURS)
            return
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold", expand=True,
                      width=min(self._console.width, 140))
        table.add_column(Text(for_terminal(held["who"])), overflow="fold", ratio=1)
        table.add_column(Text(for_terminal(writes["who"])), overflow="fold", ratio=1)
        one, other = _marked(held["value"], writes["value"])
        table.add_row(one, other)
        self._console.print(table)

    def _plain(self, line: str, kind: str | None) -> None:
        style = {DONE: "green", STOPPED: "bold yellow", LINK: "bold underline"}.get(kind or "", "")
        if _is_warning(line):
            self._warning(line)
        elif kind in (DONE, STOPPED) and "\n" in line:
            first, _, rest = line.partition("\n")
            self._console.print(_text(first, "bold green" if kind == DONE else "bold yellow"), soft_wrap=True)
            self._console.print(_text(rest), soft_wrap=True)
        else:
            self._console.print(_text(line, style), soft_wrap=True)


def _contained[**P](method: Callable[Concatenate[TransferDisplay, P], None]) -> Callable[Concatenate[TransferDisplay, P], None]:
    """A display's own fault never reaches what it displays: the library does not catch what a progress callback raises (a
    request reads it as a dropped connection, and the refusal blames the network), so the display does, switches itself
    off, and the command goes on as it would have without one."""

    @functools.wraps(method)
    def contained(self: TransferDisplay, *args: P.args, **kwargs: P.kwargs) -> None:
        if self.failed:
            return
        try:
            method(self, *args, **kwargs)
        except Exception:  # noqa: BLE001 - whatever the drawing raised; a KeyboardInterrupt is not an Exception
            self.give_up()

    return contained


class TransferDisplay:
    """The live lines of a push's uploads or a pull's downloads, drawn on standard error about four times a second.

    **One line for the whole**, which stays when the transfers are over: ``1/2 files uploaded``, the bytes of the total,
    and, while no bytes are moving, what is under way (``waiting between requests``, ``checking the md5``), so that the
    screen is never silent between one file and the next. **Beneath it the current file's bar**, its name, bytes, speed
    and an estimate of the time remaining; a file whole stays as that bar until the next file starts, and the last one
    stays at the end. Lines said meanwhile are drawn above (the display shares the terminal's console with the lines, when
    they share the terminal), and a question takes the display off the screen first and a later report draws it again.

    It is told through :class:`mime._transfers.Transfers`; any fault of its own turns it off (:func:`_contained`)."""

    def __init__(self, stream: IO[str], verb: str, past: str) -> None:
        from rich.live import Live

        self.stream = stream
        self.console = _console_of(stream)
        self.failed = False
        self._verb, self._past = verb, past
        self._clear()
        self._final = False
        self._live = Live(self, console=self.console, refresh_per_second=4, transient=False, redirect_stdout=False)
        self._running = False

    def _clear(self) -> None:
        self._expected: dict[str, int | None] = {}
        self._finished: set[str] = set()
        self._current: TransferState | None = None
        self._activity = ""

    # --- what is told ------------------------------------------------------------------------------------------------

    @_contained
    def files(self, files: Sequence[tuple[str, int | None]]) -> None:
        for key, size in files:
            self._expected.setdefault(key, size)
        self._wake()

    @_contained
    def sending(self, key: str, done: int, total: int | None) -> None:
        now = time.perf_counter()
        held = self._current
        if held is None or held.label != key or done < held.done:  # a new file, or the same one sent again from its start
            held = self._current = TransferState(key, total, now)
        held.done = done
        if total is not None:
            held.total = total
        if self._expected.get(key) is None:
            self._expected[key] = held.total
        self._wake()

    @_contained
    def sent(self, key: str) -> None:
        self._finished.add(key)
        self._expected.setdefault(key, None)
        held = self._current
        if held is not None and held.label == key:
            held.complete(time.perf_counter())
        self._wake()

    @_contained
    def doing(self, words: str) -> None:
        self._activity = words
        self._wake()

    # --- on and off the screen ---------------------------------------------------------------------------------------

    def _wake(self) -> None:
        if not self._running:
            self._running = True
            self._live.start(refresh=True)

    @_contained
    def hide(self) -> None:
        """Take the lines off the screen, erased, before a question is put: a later report draws them again."""
        if self._running:
            self._running = False
            self._live.transient = True
            try:
                self._live.stop()
            finally:
                self._live.transient = False

    @_contained
    def finish(self) -> None:
        """The transfers are over: the lines as they stand stay on the screen, with nothing said to be under way. If nothing
        moved, they are taken off. The next report starts them afresh."""
        moved = bool(self._finished) or self._current is not None
        if not moved:
            self.hide()
        else:
            self._final = True
            self._activity = ""
            if not self._running:
                self._live.start(refresh=False)  # a frame to leave
            self._running = False
            self._live.stop()
        self._clear()
        self._final = False

    close = finish

    def give_up(self) -> None:
        """A fault of the display's own: it is off for good, and the screen is left as it is."""
        self.failed = True
        self._running = False
        try:
            self._live.stop()
        except Exception:  # noqa: BLE001 - the display is already broken; nothing it does may reach the command
            pass

    # --- how it looks ------------------------------------------------------------------------------------------------

    def __rich_console__(self, console: Console, options: Any) -> Iterator[Any]:
        from rich.progress_bar import ProgressBar
        from rich.spinner import Spinner
        from rich.table import Table

        from mime.assets import size_text

        now = time.perf_counter()
        total, finished = len(self._expected), len(self._finished)
        sizes = list(self._expected.values())
        held = self._current
        done_bytes = sum(size or 0 for key, size in self._expected.items() if key in self._finished) + \
            (held.done if held is not None and held.label not in self._finished else 0)
        parts: list[str] = []
        if total:
            parts.append(f"{finished}/{total} file{'s' if total != 1 else ''} {self._past}")
            if all(size is not None for size in sizes):
                whole = sum(size or 0 for size in sizes)
                parts.append(size_text(done_bytes) if finished == total else f"{size_text(done_bytes)} of {size_text(whole)}")
            else:
                parts.append(size_text(done_bytes))
        if self._activity:
            parts.append(self._activity)
        overall = _text("  \u00b7  ".join(parts))
        if self._final or not parts:
            if parts:
                yield overall
        else:
            unicode = (console.encoding or "").lower().startswith("utf")
            yield Spinner("dots" if unicode else "line", text=overall)
        if held is None:
            return
        whole_now = held.label in self._finished
        grid = Table.grid(padding=(0, 1))
        grid.add_column()
        grid.add_column()
        bar = ProgressBar(total=held.total, completed=held.done, width=28, pulse=held.total is None and not whole_now)
        words = (f"{self._past} {held.label}" if whole_now else f"{self._verb} {held.label}" if not self._final else
                 held.label)
        tail = held.text(now) + ("" if whole_now or not self._final else ", not finished")
        grid.add_row(bar, _text(f"{words}  \u00b7  {tail}"))
        yield grid
