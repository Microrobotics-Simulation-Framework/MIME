"""``mime serve``'s live view, from a ``ServeState`` alone: the strip and the table of values as plain text, and the display
drawing them through a real pseudo-terminal at 100 and at 60 columns and at 0 (a pseudo-terminal nobody sized, which reads
as the fallback width and not as nothing).

What is held: a number is never wrapped (no line is wider than the terminal, at any width), the table's columns go in the
declared order, its rows are at most ``min(12, height - 10)`` with the rest said in one line, the unit comes from the
schema and says so where there is none; a draw is sampled, about four a second, whatever the frame rate; words from
a peer are plain text on the screen, never a command to the terminal; and the view leaves nothing of itself behind and
erases nothing of anyone else's, whenever a line is printed beside its draws.

The controls, each of which fails the same check on every run: a display that draws at every frame instead of four times a
second (the check is draws against frames, over frames fed with no pause), a drop order other than the declared one
(the check is that a column absent means every column declared to go before it is absent too), a display that does
not escape what a peer wrote (the check is that no escape of the peer's reaches the terminal), a console whose prints
are not ordered with the draws (a frame of the strip is left in the scrollback) and one live display kept across a
pause (the load report loses a line)."""

import re
import threading
import time

import numpy as np
import pytest

from _platform_support import needs_pty
from _terminal_support import screen_of, stripped, terminal
from mime.cli import _display, _serve_view
from mime.cli._display import SERVE_DRAW_EVERY, ServeDisplay
from mime.cli._serve_view import DROPPED, NO_UNIT, fit, rows_shown, strip_lines, table_lines
from mime.live import ARRAY, FlagWatch, Frame, FrameSchema, Row, ServeState, Stopped, watch_rows

SPEED = "swim_speed:umr.speed"
BOUND = [{"node": "umr", "field": "pose", "index": None, "kind": "pose"}]
FLAGS = [FlagWatch("guard.tripped", "guard.tripped")]
STATUS = {"state": "running", "step": 36704, "time": 36.704, "frames": 2295, "pace": 1.0, "recording": False,
          "activity": "stepping", "pending": [], "non_finite": None, "wall_time": 41.0,
          "clients": {"commands": 1, "frames": 1}}


def schema() -> FrameSchema:
    return FrameSchema.of([("umr", "pose", np.zeros(7, np.float32), "m; 1"),
                           ("rpm.magnetic", "flux", np.zeros(3, np.float32), "T"),
                           ("swim_speed:umr", "speed", np.zeros((), np.float32), "m/s"),
                           ("guard", "tripped", np.zeros((), np.float32), "1"),
                           ("bare", "level", np.zeros((), np.float32), None)])


def frame(number: int, speed: float = 2.31e-4) -> Frame:
    values = {spec.path: np.zeros(spec.shape, np.float32) for spec in schema().fields}
    values[SPEED] = np.asarray(speed, np.float32)
    values["umr.pose"][:3] = (1.2e-2, 0.0, 3.4e-3)
    return Frame(number, number * 16, number * 0.016, values)


def running_state(specs=None, status=None) -> ServeState:
    state = ServeState("tracked", "/runs/tracked-live")
    rows, shown = watch_rows(schema(), specs, BOUND, FLAGS)
    state.watch(rows, FLAGS, 1, table=shown)
    for second in range(4):
        state.note_status({**STATUS, **(status or {}), "step": 36704 + 1000 * second, "frames": 2295 + 62 * second,
                           "time": 36.7 + 1.0 * second, "wall_time": 41.0 + second}, 100.0 + second)
    state.take(frame(1, speed=-1.0))
    state.take(frame(2, speed=2.9e-4))
    state.take(frame(3))
    state.sample()
    return state


def test_the_strip_says_the_state_in_a_word_then_the_step_the_times_the_speed_the_clients_and_how_to_stop():
    lines = strip_lines(running_state(), 100, 103.5)
    assert lines[0].startswith("RUNNING · step 39,704 · 39.7 s simulated · wall 0:44 · 1.00x real time · 62 frames/s")
    assert lines[1] == "1 commander · 1 viewer · recording off · flags: none raised yet (seen at frames) · Ctrl+C to stop"
    assert len(lines) == 2 and max(map(len, lines)) <= 100


@pytest.mark.parametrize(("status", "word", "detail"), [
    ({"state": "ready", "frames": 0, "step": 0, "time": 0.0}, "READY", "waiting for a client's start"),
    ({"state": "running", "activity": "compiling"}, "COMPILING", "the first chunk compiles the run's programs"),
    ({"state": "paused"}, "PAUSED", ""), ({"state": "paused", "pending": ["pause"]}, "PAUSING", "pauses at the chunk's end"),
    ({"state": "running", "pending": ["stop"]}, "STOPPING", "a client's stop"),
    ({"state": "paused", "non_finite": "the state stopped being finite numbers by step 90"}, "PAUSED",
     "stopped being finite"),
])
def test_every_state_is_a_word_before_it_is_anything_else(status, word, detail):
    lines = strip_lines(running_state(status=status), 100, 103.0)
    assert lines[0].startswith(word) and detail in " ".join(lines)


def test_a_flag_seen_raised_is_named_with_the_step_it_was_seen_by():
    state = running_state()
    state.take(Frame(9, 144, 0.1, {**frame(9).values, "guard.tripped": np.asarray(1.0, np.float32)}))
    assert "flag guard.tripped raised by step 144" in " ".join(strip_lines(state, 100, 103.0))
    assert any(line.startswith("guard.tripped") and "RAISED (by step 144)" in line for line in table_lines(state, 100, 30))


@pytest.mark.parametrize("width", [100, 80, 60, 40])
def test_no_line_of_the_strip_or_the_table_is_wider_than_the_window_so_no_number_is_wrapped(width):
    state = running_state(["all", "umr.pose[0:3]"])
    for line in strip_lines(state, width, 103.0) + table_lines(state, width, 30):
        assert len(line) <= width, (width, line)
    assert re.search(r"step 39,704", " ".join(strip_lines(state, width, 103.0))), "a number is whole at every width"


def test_the_table_has_value_unit_now_min_max_and_the_last_ten_seconds_and_a_unit_the_schema_does_not_declare_says_so():
    lines = table_lines(running_state(["all"]), 100, 30)
    assert lines[0].split() == ["value", "unit", "now", "min", "max", "last", "10", "s"]
    speed = next(line for line in lines if line.startswith(SPEED))
    assert speed.split()[:5] == [SPEED, "m/s", "0.000231", "-1", "0.00029"] and len(speed.split()) == 6, "and its last ten seconds"
    assert next(line for line in lines if line.startswith("bare.level")).split()[1:4] == ["no", "unit", "declared"]
    assert NO_UNIT == "no unit declared"
    position = next(line for line in lines if line.startswith("umr.pose[0:3]"))
    assert position.split()[1:5] == ["m", "0.012", "0", "0.0034"], "a position's numbers, and no extremes kept for a row of three"
    assert position.split()[5:] == ["-", "-"], "min and max are over every frame or not shown"


def present(width: int) -> set[str]:
    """The optional columns the table has at ``width``; the minimum and the maximum are one, ``range``."""
    header = table_lines(running_state(["all"]), width, 30)[0].split()
    assert ("min" in header) == ("max" in header), "the minimum and the maximum go together"
    return {name for name, word in (("unit", "unit"), ("range", "min"), ("spark", "last")) if word in header}


#: The order the columns are declared to go in, written here and not read from the module: a check that read the order
#: from the code under test would hold for any order.
DECLARED = ("spark", "range", "unit")


def drop_order_holds(widths=range(30, 140)) -> bool:
    """Whether at every width a column that is absent has every column declared to go before it absent too."""
    for width in widths:
        held = present(width)
        for position, name in enumerate(DECLARED):
            if name not in held and any(earlier in held for earlier in DECLARED[:position]):
                return False
    return True


def test_the_columns_go_in_the_declared_order_as_the_window_narrows():
    assert DROPPED == DECLARED
    assert drop_order_holds()
    assert present(100) == {"unit", "range", "spark"} and present(70) == {"unit", "range"}
    assert present(60) == {"unit"} and present(54) == {"unit"} and present(46) == set()
    assert "now" in table_lines(running_state(["all"]), 34, 30)[0].split(), "the value and its name stay"


def test_the_control_another_order_breaks_the_drop_order_check_every_run(monkeypatch):
    monkeypatch.setattr(_serve_view, "DROPPED", ("unit", "range", "spark"))
    assert not drop_order_holds()


def test_a_table_too_narrow_for_its_numbers_is_one_line_that_says_so_never_a_wrapped_number():
    lines = table_lines(running_state(["all"]), 22, 30)
    assert len(lines) == 1 and "not shown: the window is too narrow for the table" in lines[0]


def test_the_rows_are_at_most_twelve_or_fewer_in_a_short_window_and_the_rest_is_one_line():
    specs = [f"umr.pose[{i}]" for i in range(7)] + [f"umr.pose[{i}:{i + 2}]" for i in range(5)] + [SPEED, "bare.level",
                                                                                                    "rpm.magnetic.flux"]
    state = running_state(specs)
    assert len(state.rows) == 15
    for height, expected in [(30, 12), (24, 12), (20, 10), (16, 6), (11, 1)]:
        assert rows_shown(height) == expected
        lines = table_lines(state, 100, height)
        assert len(lines) == 1 + expected + 1, (height, lines)
        assert lines[-1] == f"... {15 - expected} more (--watch all, or name rows)"
    assert "not shown: the window is too short" in table_lines(state, 100, 10)[0]
    assert table_lines(running_state(["none"]), 100, 30) == [], "--watch none hides the table"


def test_fit_breaks_only_between_segments():
    assert fit(["aaa", "bbb", "ccc"], 9) == ["aaa · bbb", "ccc"]
    assert fit(["a very long segment", "b"], 5) == ["a very long segment", "b"]


# -- through a pseudo-terminal -------------------------------------------------------------------------------------------

@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_the_display_draws_the_strip_and_the_table_with_events_above_and_leaves_the_scrollback_and_a_block(columns):
    width = columns or 80
    state = running_state(["all"])
    state.event("viewer connected (127.0.0.1:51822)", at=1000.0)
    state.event("command: set offset=0.001 (from step 36,800)", at=1004.0)
    with terminal(columns=columns, rows=30) as tty:
        display = ServeDisplay(tty.stream)
        display.phase("loading the folder")
        display.draw()
        assert "loading the folder" in stripped(tty.transcript())
        display.running(state)
        display.draw()
        screen = screen_of(tty.transcript())
        text = "\n".join(screen)
        assert "viewer connected (127.0.0.1:51822)" in text and "command: set offset=0.001" in text
        assert screen.index(next(l for l in screen if "viewer connected" in l)) < screen.index(
            next(l for l in screen if l.startswith("RUNNING"))), "events are above the strip"
        assert next(l for l in screen if l.startswith("RUNNING")).split(" · ")[1] == "step 39,704"
        assert any(l.startswith(SPEED) for l in screen) and any(l.startswith("value") for l in screen)
        assert max(len(l) for l in screen) <= width, (width, [l for l in screen if len(l) > width])
        state.stop_begins("stopped by Ctrl+C")
        state.stop_ends("stopped by Ctrl+C", 40000, 40.0, 49.0, 2400, "/runs/tracked-live")
        display.close()
        after = screen_of(tty.transcript())
    assert not any("Ctrl+C to stop" in l for l in after), "the live region is gone"
    assert any(l.startswith("STOPPED  stopped by Ctrl+C") for l in after) and any("viewer connected" in l for l in after)
    assert any("mime verify /runs/tracked-live/experiment" in l for l in after)
    assert sum("viewer connected" in l for l in after) == 1, "an event is printed once"


#: A phase that takes more than one line at every width drawn at here, as serve's own building phase does at 80 columns.
LONG_PHASE = "building the graph to serve it (compiling it, which can take many seconds); " * 2
READY = {**STATUS, "state": "ready", "step": 0, "time": 0.0, "frames": 0, "activity": "idle", "wall_time": 0.0,
         "clients": {"commands": 0, "frames": 0}}
#: The strip's first line, whatever the state: a frame of the view.
STRIP_HEAD = re.compile(r"[A-Z]+ · step ")
#: How long anything here waits for a thread: a guard against a hang, not a bound on the drawing.
HANG = 60.0


def ready_state() -> ServeState:
    state = ServeState("tracked", "/runs/tracked-live")
    rows, shown = watch_rows(schema(), ["all"], BOUND, FLAGS)
    state.watch(rows, FLAGS, 1, table=shown)
    state.note_status(READY, 100.0)
    return state


def back_from_the_load_report(monkeypatch, columns: int, *, draw_due: bool) -> tuple[list[str], list[str]]:
    """What serve does around its load report, on a terminal ``columns`` wide: its phase drawn, the view taken off the
    screen, the report printed, the view back with the session's state and serve's ready line printed at once, then the
    stop. With ``draw_due``, a draw comes while the ready line is being printed, after it has found the top of the
    region and before it draws the region again (where a loaded machine put the view's own draw: CI's "the live region
    is gone", PR #126's run 41839710392): it is made on another thread as the line itself is drawn, and the line waits
    until that draw has been made or is held back. The view's own thread draws nothing here. The screen once the phase
    was drawn, and after the stop."""
    monkeypatch.setattr(_display, "SERVE_DRAW_EVERY", 3600.0)
    state = ready_state()
    drawers: list[threading.Thread] = []
    armed = False  # while the ready line is printed
    with terminal(columns=columns, rows=30) as tty:
        display = ServeDisplay(tty.stream)
        text = _display._text

        def draw_now() -> None:
            decided = threading.Event()
            held_back: list[bool] = []

            def draw() -> None:
                free = display._guard.acquire(blocking=False)  # noqa: SLF001 - held by the line: this draw waits
                if free:
                    display._guard.release()  # noqa: SLF001
                held_back.append(not free)
                decided.set()
                display.draw()

            drawers.append(threading.Thread(target=draw, daemon=True))
            drawers[-1].start()
            assert decided.wait(HANG), "the draw never started"
            if not held_back[0]:
                drawers[-1].join(HANG)
                assert not drawers[-1].is_alive(), "a draw nothing held back never ended"

        class DrawDue:
            def __init__(self, held) -> None:
                self.held = held

            def __rich_console__(self, console, options):
                draw_now()
                yield self.held

        def text_with_a_draw_due(line: str, style: str = ""):
            held = text(line, style)
            return DrawDue(held) if draw_due and armed and not drawers else held

        monkeypatch.setattr(_display, "_text", text_with_a_draw_due)
        display.phase(LONG_PHASE)
        display.draw()
        phase = screen_of(tty.transcript())
        display.pause()
        display.say("load report: its first line\nload report: its last line")
        display.running(state)
        armed = True
        display.say("mime serve: tracked ready (mime.live/1)")
        armed = False
        for drawer in drawers:
            drawer.join(HANG)
        display.draw()
        state.stop_begins("stopped by Ctrl+C")
        state.stop_ends("stopped by Ctrl+C", 0, 0.0, 1.0, 0, "/runs/tracked-live")
        display.close()
        after = screen_of(tty.transcript())
    assert not draw_due or drawers, "no draw was made while the ready line drew the strip: this cannot fail"
    return phase, after


def keeps_one_live_display(monkeypatch) -> None:
    """The control of the view's coming back: one live display, kept across the pause, as the view had before."""
    made = ServeDisplay._new_live  # noqa: SLF001

    def kept(display: ServeDisplay):
        if "_kept" not in display.__dict__:
            display.__dict__["_kept"] = made(display)
        return display.__dict__["_kept"]

    monkeypatch.setattr(ServeDisplay, "_new_live", kept)


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_the_first_line_once_the_view_is_back_erases_nothing_printed_while_it_was_away(monkeypatch, columns):
    """The view kept one live display across the pause, which kept the height of the region it last drew (the phase's
    spinner), and the first line printed once the view was back moved up by it: it erased the load report's last line at
    80 columns, where serve's building phase takes two. Here the phase takes more than one line at every width (checked:
    a phase of one line is no move up at all, and the check could not fail)."""
    phase, after = back_from_the_load_report(monkeypatch, columns, draw_due=False)
    assert len([line for line in phase if line]) > 1, phase
    assert "load report: its first line" in after and "load report: its last line" in after, after


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_the_control_one_live_display_kept_across_the_pause_erases_a_line_of_the_report_every_run(monkeypatch, columns):
    keeps_one_live_display(monkeypatch)
    _, after = back_from_the_load_report(monkeypatch, columns, draw_due=False)
    assert "load report: its last line" not in after, "the check above would fail it"


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_a_line_printed_while_a_draw_comes_leaves_no_frame_of_the_strip_behind(monkeypatch, columns):
    """A line of serve's own and a draw on the view's thread are ordered: the draw comes after the line, never inside
    it. Inside it, the draw replaced the region and the line then moved up by the height it had found before: the top of
    a frame, the strip with "Ctrl+C to stop", was left in the scrollback (CI's "the live region is gone", which a loaded
    machine made by putting the view's draw there). The draw is put there on purpose here, every run."""
    _, after = back_from_the_load_report(monkeypatch, columns, draw_due=True)
    assert not any(STRIP_HEAD.match(line) for line in after), after
    assert not any("Ctrl+C to stop" in line for line in after), "the live region is gone"
    assert sum(line.startswith("mime serve: tracked ready") for line in after) == 1, after


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_the_control_prints_not_ordered_with_the_draws_leave_a_frame_of_the_strip_every_run(monkeypatch, columns):
    monkeypatch.setattr(_display, "_ordered_console", lambda stream, _guard: _display._console_of(stream))
    _, after = back_from_the_load_report(monkeypatch, columns, draw_due=True)
    assert any(STRIP_HEAD.match(line) for line in after), "the check above would fail it"


def feed(display: ServeDisplay, state: ServeState, seconds: float, per_frame: bool) -> tuple[int, int, float]:
    """Frames taken as fast as a loop could take them (no pause), for ``seconds``; the frames, the draws they caused or
    that were made meanwhile, and the time it took. ``per_frame`` is the control: a draw at every frame."""
    frames, started = 0, time.perf_counter()
    base = display.draws
    while time.perf_counter() - started < seconds:
        state.take(frame(frames))
        frames += 1
        if per_frame:
            display.draw()
    return frames, display.draws - base, time.perf_counter() - started


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_the_view_is_drawn_about_four_times_a_second_whatever_the_frame_rate(columns):
    state = running_state()
    with terminal(columns=columns) as tty:
        display = ServeDisplay(tty.stream)
        display.running(state)
        frames, draws, elapsed = feed(display, state, 2.0, per_frame=False)
        display.close()
    assert frames > 1000, "frames fed far faster than the view is drawn"
    assert 1 <= draws <= elapsed / SERVE_DRAW_EVERY + 2 and draws * 50 < frames, (frames, draws, elapsed)


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_the_control_a_draw_at_every_frame_is_far_over_four_a_second_every_run(columns):
    state = running_state()
    with terminal(columns=columns) as tty:
        display = ServeDisplay(tty.stream)
        display.running(state)
        frames, draws, elapsed = feed(display, state, 1.0, per_frame=True)
        display.close()
    assert draws >= frames, "a draw for every frame"
    assert not (draws <= elapsed / SERVE_DRAW_EVERY + 2), "which the check above refuses"


HOSTILE = "[bold red]x[/] \x1b]0;pwned\x07 \x1b[31mred\x1b[0m\nSTOPPED forged line"


def hostile_transcript(columns: int) -> str:
    """What a terminal was sent after a peer's words were handed to the view as a command's name, an argument, a refusal's
    code and an address."""
    state = running_state(["all"])
    state.command({"command": HOSTILE, "values": {HOSTILE: HOSTILE}}, {"ok": False, "errors": [{"code": HOSTILE}]})
    state.command({"command": "set", "values": {HOSTILE: HOSTILE}}, {"ok": True, "step": 9, "pending": []})
    state.event("viewer connected (" + HOSTILE + ")")
    with terminal(columns=columns, rows=30) as tty:
        display = ServeDisplay(tty.stream)
        display.running(state)
        display.draw()
        display.close()
        return tty.transcript()


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_words_from_a_peer_are_plain_text_on_the_screen_never_markup_and_never_a_command_to_the_terminal(columns):
    raw = hostile_transcript(columns)
    screen = "\n".join(screen_of(raw))
    assert "pwned" in raw and "\x1b]0;" not in raw and "\x1b[31mred" not in raw, "no escape of the peer's reached the terminal"
    assert "\\x1b]0;pwned\\x07" in screen and "[bold red]x[/]" in screen, "written out, and not markup"
    assert not any(line.startswith("STOPPED forged") for line in screen.splitlines()), "a newline does not forge a line"
    assert "\\nSTOPPED forged line" in screen


@needs_pty
@pytest.mark.parametrize("columns", [100, 60, 0])
def test_the_control_a_display_that_does_not_escape_what_a_peer_wrote_lets_its_escapes_through_every_run(monkeypatch, columns):
    monkeypatch.setattr(_display, "plain", lambda text: text)
    raw = hostile_transcript(columns)
    assert "\x1b]0;pwned" in raw or "\x1b[31mred" in raw, "the check above would fail it"


@needs_pty
def test_a_draw_at_a_resized_terminal_follows_the_new_width():
    from _terminal_support import _size

    state = running_state(["all"])
    with terminal(columns=100, rows=30) as tty:
        display = ServeDisplay(tty.stream)
        display.running(state)
        display.draw()
        _size(tty._master, 60, 30)
        mark = len(tty.transcript())
        display.draw()
        display.close()
        raw = tty.transcript()[mark:]
    drawn = [line for line in stripped(raw).splitlines() if line.startswith(("RUNNING", "value", SPEED))]
    assert drawn and max(len(line) for line in drawn) <= 60, drawn


def test_the_state_and_the_text_of_the_view_load_no_terminal_library_and_the_drawing_does():
    """``rich`` is imported by the module that draws and by nothing else: the state and the text made of it are plain
    data and strings, so that another feed (a ``mime watch``) renders the same view. The control is the drawing module,
    which loads it when it draws."""
    import subprocess
    import sys

    def loads_rich(code: str) -> bool:
        done = subprocess.run([sys.executable, "-c", f"import sys; {code}; print('rich' in sys.modules)"], check=True,
                              capture_output=True, text=True)
        return done.stdout.strip() == "True"

    assert not loads_rich("import mime.live, mime.live._state, mime.cli._serve_view")
    assert loads_rich("import mime.cli._display as d; d._line('x')"), "the control: drawing loads it"


class NoLoopOverIt(np.ndarray):
    """An array that cannot be iterated: a Python loop over it raises, a reduction in numpy does not."""

    def __iter__(self):
        raise AssertionError("a Python loop over the array, under the state's lock")


def test_a_large_watched_array_is_summarised_by_a_reduction_and_not_a_python_loop_under_the_lock():
    """At 10^6 numbers the loop held the state's lock 145 ms a draw (the review of PR #122, F2). Whether a loop runs is
    structural, so the check is that none can: the row's numbers are handed over as an array that raises when iterated.
    The control is a loop of the old form over the same array, which raises."""
    values = np.arange(-6, 6, dtype=np.float32).reshape(3, 4)
    numbers = values.reshape(-1).view(NoLoopOverIt)
    with pytest.raises(AssertionError):
        max(abs(float(each)) for each in numbers)  # the control: what the draw used to do
    state = running_state()
    state.latest = Frame(1, 16, 0.016, {"big.field": values})
    row = Row("big.field", "big.field", ARRAY, (), values.shape, "1")
    row.numbers = lambda _value: numbers  # type: ignore[method-assign]
    assert _serve_view._now_text(row, state) == "max |x| 6 of 3x4"
