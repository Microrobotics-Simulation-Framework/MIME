"""The text of ``mime serve``'s live view, as plain strings made from a :class:`~mime.live.ServeState`: no terminal and no
``rich`` here, so the layout is tested without one and ``mime.cli._display.ServeDisplay`` only draws what this makes.

The renderer takes only a ``ServeState``. A ``mime watch`` client that fills the same state from a status poll and the
frames it subscribes to draws the same view.

**Three rules, held at every width.** A number is never wrapped: a line is made of segments and a segment is never
split, and the table's columns are dropped, in the order :data:`DROPPED`, before a number would not fit; when even the
numbers do not fit, the table is replaced by one line that says so. A state is a word before it is a colour (the display
colours nothing that these words do not already say). Text from outside is escaped by the display, which draws every
line here as plain text, never as markup."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np

from mime.cli._progress import duration_text, simulated_text
from mime.cli._shell import quoted
from mime.live import ARRAY, FLAG, SCALAR, Event, Row, ServeState

#: The table's most rows, and the lines of a window it leaves free (the strip, the header, the summary and the
#: scrollback): rows are at most ``min(MOST_ROWS, height - RESERVED)``.
MOST_ROWS = 12
RESERVED = 10
#: The window's height when it cannot be read.
FALLBACK_ROWS = 24
#: The columns of the table that go, in this order, when the width is too small for them; the value and its name stay.
#: ``range`` is the minimum and the maximum, which go together: one without the other is half a statement.
DROPPED = ("spark", "range", "unit")
_GROUPS = {"spark": ("spark",), "range": ("min", "max"), "unit": ("unit",)}
#: What the unit column says where the schema declares none.
NO_UNIT = "no unit declared"
SEPARATOR = " · "
GAP = "  "
#: A row's name is cut to this many characters before a column goes, and to this many before the table is given up for the
#: line that says it does not fit.
LABEL_KEEP = 20
LABEL_LEAST = 8
SPARK_WIDTH = 12
SPARK_UNICODE = "▁▂▃▄▅▆▇█"
SPARK_ASCII = "_.-^"
SPINNER_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_ASCII = "-\\|/"
#: Below this many columns the strip is one flow of short segments, not three lines of long ones.
COMPACT_BELOW = 80


def clock(seconds: float) -> str:
    """A wall time as a clock reads it: ``0:41``, ``1:02:03``."""
    whole = max(0, round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def number(value: float) -> str:
    """A reading to three significant figures, never longer than nine characters."""
    return f"{value:.3g}"


def fit(segments: Sequence[str], width: int) -> list[str]:
    """``segments`` joined by :data:`SEPARATOR` into lines no wider than ``width``, breaking only between segments; one
    wider than the width stands alone, and is the terminal's to fold."""
    lines: list[str] = []
    line = ""
    for segment in segments:
        if line and len(line) + len(SEPARATOR) + len(segment) > width:
            lines.append(line)
            line = ""
        line = f"{line}{SEPARATOR}{segment}" if line else segment
    if line:
        lines.append(line)
    return lines


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' if count != 1 else ''}"


def headline(state: ServeState, now: float, unicode: bool = True) -> tuple[str, str]:
    """The state in a word, and what it is waiting for or doing, from the status as a client reads it."""
    if state.stopped is not None:
        return "STOPPED", state.stopped.reason
    if state.stopping is not None:
        return "STOPPING", f"{state.stopping}; it ends at the chunk's end, then writes the run folder"
    if "stop" in state.pending:
        return "STOPPING", "a client's stop; it acts at the chunk's end, then the run folder is written"
    if state.state == "ready":
        return "READY", "waiting for a client's start"
    if state.state == "paused":
        if state.non_finite:
            return "PAUSED", state.non_finite
        return ("PAUSING", "it pauses at the chunk's end") if "pause" in state.pending else ("PAUSED", "")
    if state.activity == "compiling":
        spinner = (SPINNER_UNICODE if unicode else SPINNER_ASCII)
        return "COMPILING", f"{spinner[int(now * 10) % len(spinner)]} the first chunk compiles the run's programs, " \
                            "which can take many seconds"
    return state.state.upper(), ""


def strip_lines(state: ServeState, width: int, now: float, unicode: bool = True) -> list[str]:
    """The live strip: the state, the step, the simulated time against the wall time, the speed against real time, the
    frames a second, the clients, whether it records and the flags, and how to stop. Two to four lines at 100
    columns, as many at 60; every number whole."""
    with state.lock:
        word, detail = headline(state, now, unicode)
        compact = width < COMPACT_BELOW
        wall = state.wall(now)
        running = state.state == "running" and state.stopping is None and state.stopped is None
        first = [word, f"step {state.step:,}", f"{simulated_text(state.time)}{'' if compact else ' simulated'}"]
        if wall is not None:
            first.append(f"wall {clock(wall)}")
        rates: list[str] = []
        speed, rate = state.real_time_factor(), state.frames_per_second()
        if running and state.activity != "compiling" and speed is not None:
            rates.append(f"{speed:.2f}x real time")
        if running and state.activity != "compiling" and rate is not None:
            rates.append(f"{rate:.0f} frames/s")
        if not compact:
            rates.append("unpaced" if state.pace is None else f"pace {state.pace:g}x")
        who: list[str] = []
        if state.clients is not None:
            who += [_plural(state.clients[0], "commander"), _plural(state.clients[1], "viewer")]
        who.append(f"{'rec' if compact else 'recording'} {'on' if state.recording else 'off'}")
        who += _flags(state, compact)
        if state.stopped is None:
            who.append("Ctrl+C to stop")
        lines = fit([*first, *rates], width)
        if detail:
            lines.append(detail)
        return lines + fit(who, width)


def _flags(state: ServeState, compact: bool) -> list[str]:
    """The flags, as segments: each raised one with the step it was first seen by, or that none is."""
    if not state.flags_declared:
        return ["no flags"] if compact else ["flags: none declared"]
    if not state.flags:
        return ["flags: none in the frames"]
    if state.raised:
        return [f"flag {name} raised by step {step:,}" for name, step in state.raised.items()]
    return ["flags: none yet"] if compact else ["flags: none raised yet (seen at frames)"]


# -- the table of values ----------------------------------------------------------------------------------------------


def rows_shown(height: int) -> int:
    """The most rows the table shows in a window ``height`` lines high: twelve, or fewer where the window is short."""
    return min(MOST_ROWS, height - RESERVED)


def _now_text(row: Row, state: ServeState) -> str:
    frame = state.latest
    if frame is None:
        return "-"
    if row.kind == FLAG:
        assert row.flag is not None
        first = state.raised.get(row.flag.name)
        raised = row.flag.raised(frame.values[row.path])
        if raised:
            return f"RAISED (by step {first:,})" if first is not None else "RAISED"
        return f"clear now (raised by step {first:,})" if first is not None else "clear"
    numbers = row.numbers(frame.values[row.path])
    if row.kind == ARRAY:
        size = "x".join(str(each) for each in row.shape)
        return f"max |x| {number(float(np.max(np.abs(numbers))))} of {size}"
    return " ".join(number(float(each)) for each in numbers)


_BLOCKS = {True: SPARK_UNICODE, False: SPARK_ASCII}


def spark(samples: Sequence[float], unicode: bool) -> str:
    """The last few seconds of a row's drawn values as :data:`SPARK_WIDTH` characters, scaled to their own range."""
    finite = [each for each in samples if math.isfinite(each)]
    if not finite:
        return ""
    blocks = _BLOCKS[unicode]
    per = max(1, math.ceil(len(samples) / SPARK_WIDTH))
    low, high = min(finite), max(finite)
    out = ""
    for start in range(0, len(samples), per):
        group = [each for each in samples[start:start + per] if math.isfinite(each)]
        if not group:
            out += " "
            continue
        mean = sum(group) / len(group)
        level = (len(blocks) - 1) // 2 if high == low else round((mean - low) / (high - low) * (len(blocks) - 1))
        out += blocks[level]
    return out


def table_lines(state: ServeState, width: int, height: int, unicode: bool = True) -> list[str]:
    """The table of values: a header and one line per row (at most :func:`rows_shown`), then one line for the rest. The
    columns are ``value``, ``unit``, ``now``, ``min``, ``max`` and the last ten seconds; they go in the order
    :data:`DROPPED` as the width asks, and no number is wrapped or cut. In a window too short or too narrow it is one line
    saying so."""
    with state.lock:
        if not state.table or not state.rows:
            return []
        limit = rows_shown(height)
        if limit < 1:
            return [f"{_plural(len(state.rows), 'value')} not shown: the window is too short for the table"]
        shown = state.rows[:limit]
        cells: list[dict[str, str]] = []
        for row in shown:
            range_ = row.has_range
            cells.append({
                "label": row.label, "unit": row.unit if row.unit is not None else NO_UNIT, "now": _now_text(row, state),
                "min": number(row.low) if range_ else "-", "max": number(row.high) if range_ else "-",
                "spark": spark(list(row.samples), unicode) if row.kind == SCALAR else ""})
        titles = {"label": "value", "unit": "unit", "now": "now", "min": "min", "max": "max", "spark": "last 10 s"}
        natural = {key: max(len(titles[key]), *(len(cell[key]) for cell in cells)) for key in titles}
        natural["label"] = min(natural["label"], 36)
        natural["spark"] = SPARK_WIDTH
        columns = ["label", "unit", "now", "min", "max", "spark"]
        keep = min(LABEL_KEEP, natural["label"])
        others = 0
        for dropped in (None, *DROPPED):
            for column in _GROUPS.get(dropped or "", ()):
                columns.remove(column)
            others = sum(natural[key] for key in columns if key != "label") + len(GAP) * (len(columns) - 1)
            if width - others >= keep:
                break
        label_width = min(natural["label"], width - others)
        if label_width < min(LABEL_LEAST, natural["label"]):
            return [f"{_plural(len(state.rows), 'value')} not shown: the window is too narrow for the table"]
        widths = {**natural, "label": label_width}

        def line(cell: dict[str, str]) -> str:
            parts: list[str] = []
            for key in columns:
                held = cell[key]
                if key == "label":
                    held = held if len(held) <= widths[key] else held[:widths[key] - 1] + "…"
                    parts.append(held.ljust(widths[key]))
                elif key == "unit":
                    parts.append(held.ljust(widths[key]))
                elif key == "spark":
                    parts.append(held.ljust(widths[key]))
                else:
                    parts.append(held.rjust(widths[key]))
            return GAP.join(parts).rstrip()

        out = [line(titles)] + [line(cell) for cell in cells]
        if len(state.rows) > limit:
            out.append(f"... {len(state.rows) - limit} more (--watch all, or name rows)")
        return out


# -- events and the end -----------------------------------------------------------------------------------------------


def event_line(event: Event) -> str:
    """An event for the scrollback: the time of day it happened at, then what."""
    return f"{time.strftime('%H:%M:%S', time.localtime(event.at))}  {event.text}"


def stopped_lines(state: ServeState) -> list[str]:
    """The block printed when the session has stopped: why, where it got to, the run folder and what to do with it."""
    with state.lock:
        end = state.stopped
        if end is None:
            return []
        lines = [f"STOPPED  {end.reason}",
                 f"at step {end.step:,} ({simulated_text(end.time)} simulated) · wall {duration_text(end.wall)} · "
                 f"{end.frames:,} frames"]
        if end.out is not None:
            lines += [f"run folder  {end.out}", "Next",
                      f"  check it again:  mime verify {quoted(end.out + '/experiment')}",
                      f"  replay it:       mime run {quoted(end.out + '/experiment')} --commands "
                      f"{quoted(end.out + '/commands.jsonl')} --out <another folder>",
                      f"  read it:         python -c \"from mime.results import RunResult; import sys; "
                      f"print(RunResult.read(sys.argv[1]).text())\" {quoted(end.out)}"]
        return lines


__all__ = ["DROPPED", "MOST_ROWS", "NO_UNIT", "RESERVED", "clock", "event_line", "fit", "number", "rows_shown", "spark",
           "strip_lines", "stopped_lines", "table_lines", "headline"]
