"""What ``mime push`` and ``mime pull`` draw on a terminal, through a real pseudo-terminal (POSIX; Windows has no ``pty``
module, ``_platform_support.needs_pty`` says what does not run there): the plan as one block with what goes up and what
travels as a reference apart and a warning outside it; the publish step in a heavy frame with the irreversible sentence
reversed, each permission in a frame of its own, each question set apart; the two values of a site edit side by side;
a file's progress while it goes up or comes down. The display is the only thing here that is not byte for byte what it was
(``tests/sharing/test_push_output.py`` holds that the rest is); a frame, a colour or a bold sets a line, and the words
are the line's own, which the last tests hold across a wrapped line and a stranger's escape."""

from __future__ import annotations

import io
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from _platform_support import needs_pty
from _terminal_support import run_with_streams, screen_of, stripped, terminal
from mime._said import (
    DONE,
    EDIT_OURS,
    EDIT_SITE,
    LINK,
    PERMISSION,
    PLAN_FIELD,
    PLAN_HEAD,
    PLAN_REFERENCE,
    PLAN_UP,
    PLAN_WARNING,
    STEP_HEAD,
    STEP_IRREVERSIBLE,
    STEP_LINE,
    STEP_TRIAL,
    STOPPED,
    Said,
)
from mime.cli import _display
from mime.cli._display import SaidDisplay, TransferDisplay
from mime.cli._progress import TransferState
from mime.push._publish import PRODUCTION_FACTS, SANDBOX_FACTS

pytestmark = needs_pty

FRAME = re.compile(r"[─-╿]")
SGR = re.compile(r"\x1b\[([0-9;]*)m")

PLAN = [Said("Pushing 'full_toy' to the Zenodo sandbox (a test service): a new draft, its DOI reserved", PLAN_HEAD),
        Said("  up: full_toy.mimebundle 31 kB, NOTICE 2 kB", PLAN_UP),
        Said("  by reference, never uploaded: scene/layers/arm_visual.usda (LicenseRef-Robot-Maker-NonCommercial, "
             "from https://example.org/robot-maker/6a3ebb1/arm_visual.usda)", PLAN_REFERENCE),
        Said("  title: mime_sandbox_toy (MIME push sample)", PLAN_FIELD),
        Said("! 1 permission declared: each becomes public, under your name, when the record is published", PLAN_WARNING,
             last=True)]


def words(text: str) -> str:
    """Text as its characters in order: the frames' lines removed and every space too, so a line a frame wrapped, or a
    word a narrow screen had to fold, is the line again."""
    return "".join(FRAME.sub(" ", text).split())


def drawn(lines, columns: int = 120) -> tuple[str, str]:
    """What ``lines`` draw on a terminal: the transcript with its control sequences, and without."""
    with terminal(columns=columns) as tty:
        shown = SaidDisplay(tty.stream)
        for line in lines:
            shown.say(line)
        shown.close()
        raw = tty.transcript()
    return raw, stripped(raw)


def last_style_before(raw: str, text: str) -> str:
    return SGR.findall(raw[:raw.index(text)])[-1]


def test_the_plan_is_one_block_with_what_goes_up_and_what_is_a_reference_in_their_own_rows_and_the_warning_outside():
    raw, plain = drawn(PLAN)
    lines = plain.splitlines()
    top = next(i for i, line in enumerate(lines) if "╭" in line)
    bottom = next(i for i, line in enumerate(lines) if "╰" in line)
    up = next(i for i, line in enumerate(lines) if "  up: full_toy.mimebundle" in line)
    reference = next(i for i, line in enumerate(lines) if "by reference, never uploaded" in line)
    warning = next(i for i, line in enumerate(lines) if "! 1 permission declared" in line)
    assert top < up < reference < bottom < warning, "the warning stands below the block, outside it"
    assert up != reference and not lines[warning].lstrip().startswith("│")
    assert last_style_before(raw, "  up: full_toy") != last_style_before(raw, "  by reference"), \
        "what goes up and what travels as a reference are set apart by more than the words"
    assert "1;33" in last_style_before(raw, "! 1 permission declared"), "a warning is bold yellow"


def test_the_plan_is_drawn_when_its_last_line_is_said_not_when_the_next_thing_is():
    """A push draws the plan, then makes the draft and uploads, saying nothing more until it is done: the plan must be on
    the screen before that, so the line that ends it draws it."""
    with terminal(columns=120) as tty:
        shown = SaidDisplay(tty.stream)
        for line in PLAN:
            shown.say(line)
        before_the_next_line = stripped(tty.transcript())
        shown.close()
    assert "\u256d" in before_the_next_line and "! 1 permission declared" in before_the_next_line
    held = [*PLAN[:-1], Said(PLAN[-2], PLAN_FIELD)]
    with terminal(columns=120) as tty:
        shown = SaidDisplay(tty.stream)
        for line in held:
            shown.say(line)
        assert stripped(tty.transcript()) == "", "the control: with no line marked as the last, nothing is drawn yet"
        shown.close()


def test_a_plain_line_that_starts_with_a_bang_is_a_warning_below_the_plan_too():
    raw, plain = drawn([*PLAN[:-1], "! a warning that arrived as an ordinary line"])
    lines = plain.splitlines()
    assert lines.index(next(line for line in lines if "\u2570" in line)) < \
        lines.index("! a warning that arrived as an ordinary line")


def test_the_control_the_same_lines_without_their_kinds_are_not_a_block():
    _, plain = drawn([str(line) for line in PLAN])
    assert "╭" not in plain and not FRAME.search(plain)


def test_the_publish_step_is_a_heavy_frame_whose_irreversible_sentence_is_reversed_and_the_sandbox_is_not_red():
    statement = [Said("Publishing 'full_toy' on zenodo as 10.5281/zenodo.1 (version 1 of a new concept).", STEP_HEAD),
                 Said(PRODUCTION_FACTS, STEP_IRREVERSIBLE),
                 Said("  You are publishing, under your name, 4 files as your own work (or your lab's), licensed "
                      "CC-BY-4.0.", STEP_LINE),
                 Said("  You state that everyone named as a creator or contributor agreed to be named.", STEP_LINE)]
    raw, plain = drawn(statement)
    assert "┏" in plain and "┗" in plain, "a heavy frame"
    assert last_style_before(raw, "This cannot be undone") == "1;37;41", "bold white on red"
    assert "31" in SGR.findall(raw)[0].split(";"), "a red border"
    sandbox = [statement[0], Said(SANDBOX_FACTS, STEP_TRIAL), *statement[2:]]
    raw_sandbox, _ = drawn(sandbox)
    assert last_style_before(raw_sandbox, "This is the sandbox") == "1;33", "bold yellow, not the production's"
    assert "1;37;41" not in raw_sandbox


def test_a_permission_is_a_frame_of_its_own_and_its_question_is_set_apart_below_it(monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "y")
    with terminal(columns=120) as tty:
        shown = SaidDisplay(tty.stream)
        shown.say(Said("  1. You are declaring publicly, under your name, that Example Maker permitted you to "
                       "redistribute this file.", PERMISSION))
        shown.say(Said("       scene/layers/arm_shell.usda", PERMISSION))
        assert shown.ask(Said("Declare permission 1 publicly, under your name?", "question-permission")) is True
        shown.close()
        plain = stripped(tty.transcript())
    lines = plain.splitlines()
    frame = next(i for i, line in enumerate(lines) if "╔" in line)
    closed = next(i for i, line in enumerate(lines) if "╚" in line)
    question = next(i for i, line in enumerate(lines) if "? Declare permission 1 publicly" in line)
    assert frame < closed < question and lines[question - 1].strip() == "", "a blank line before the question"
    assert lines[question + 1].strip() == "[y/N]", "the hint is on the screen under the question"
    assert asked == [""], "input is given no prompt of its own: it would put one on standard error"


@pytest.mark.parametrize(("typed", "expected"), [("y", True), ("YES", True), (" yes ", True), ("n", False), ("", False),
                                                  ("yep", False), ("1", False)])
def test_a_question_is_a_yes_only_to_y_or_yes_as_it_is_without_a_display(typed, expected, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": typed)
    with terminal() as tty:
        assert SaidDisplay(tty.stream).ask("Delete the draft?") is expected


def test_the_end_of_input_is_no_to_a_question_and_nothing_to_a_prompt(monkeypatch):
    def end(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", end)
    with terminal() as tty:
        shown = SaidDisplay(tty.stream)
        assert shown.ask("Delete the draft?") is False and shown.typed("Type the record number (1): ") == ""


def test_the_record_number_is_asked_as_a_question_and_typed_after_a_marker(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "629525")
    with terminal() as tty:
        shown = SaidDisplay(tty.stream)
        assert shown.typed(Said("Type the record number to publish (629525): ", "question-number")) == "629525"
        plain = stripped(tty.transcript())
        assert "? Type the record number to publish (629525):" in plain and plain.rstrip().endswith(">")


def test_the_two_values_of_a_site_edit_stand_side_by_side_with_what_differs_marked():
    raw, plain = drawn([Said("    on the site:  title: A title corrected on the site", EDIT_SITE),
                        Said("    sharing.yaml: title: mime_sandbox_toy (MIME push sample)", EDIT_OURS)])
    row = next(line for line in plain.splitlines() if "A title corrected" in line)
    assert "title: mime_sandbox_toy" in row, "both values begin on one row"
    header = next(line for line in plain.splitlines() if "on the site" in line)
    assert "sharing.yaml" in header and header.index("on the site") < header.index("sharing.yaml")
    assert any("4" in code.split(";") for code in SGR.findall(raw)), "the differing stretches are underlined"
    assert "     on the site:  title:" not in plain, "not the two lines the pipe gets"


def test_a_site_edit_whose_second_line_never_comes_is_drawn_as_the_line_it_was():
    _, plain = drawn([Said("    on the site:  title: A", EDIT_SITE), "kept what was edited on the site"])
    assert "on the site:  title: A" in plain and "kept what was edited" in plain


def test_what_was_done_what_was_stopped_and_the_link_are_set_apart_and_say_what_they_said():
    raw, plain = drawn([Said("made the draft 629525 (a first version) and reserved its DOI 10.5072/zenodo.629525", DONE),
                        Said("Not published: permission 1 was not declared.", STOPPED),
                        Said("https://sandbox.zenodo.org/uploads/629525", LINK)])
    assert "made the draft 629525" in plain and "Not published: permission 1" in plain
    assert plain.splitlines()[-1].strip() == "https://sandbox.zenodo.org/uploads/629525" or \
        "https://sandbox.zenodo.org/uploads/629525" in plain
    assert "4" in last_style_before(raw, "https://sandbox").split(";"), "the link is underlined"


def test_a_summary_that_says_the_unpacking_is_partial_is_set_as_a_stop_with_its_lines_below_it():
    raw, plain = drawn([Said("Bundle 'full_toy' unpacked to ./out/, but not complete: 1 asset could not be downloaded\n"
                             "  - Assets: 1 not downloaded: run `mime assets fetch ./out/`", STOPPED)])
    assert last_style_before(raw, "Bundle 'full_toy'") == "1;33", "bold yellow, not the green of done"
    assert "\u2713" not in plain and "  - Assets: 1 not downloaded" in plain


@pytest.mark.parametrize("columns", [40, 72, 140])
def test_every_character_of_every_line_is_on_the_screen_however_narrow_it_is(columns):
    lines = [*PLAN,
             Said("Publishing 'full_toy' on zenodo-sandbox as 10.5072/zenodo.629525 (version 1 of a new concept).",
                  STEP_HEAD), Said(SANDBOX_FACTS, STEP_TRIAL),
             Said("  You state that everyone named as a creator or contributor agreed to be named.", STEP_LINE),
             Said("  1. You are declaring publicly, under your name, that Example Maker Ltd (synthetic) permitted you "
                  "to redistribute this file (scope: redistribute, no modification, non-commercial).", PERMISSION),
             Said("published 'full_toy' as 10.5072/zenodo.629525 (version 1)\nsecond line", DONE),
             "an ordinary line of the sort every command says"]
    _, plain = drawn(lines, columns)
    for line in lines:
        assert words(str(line)) in words(plain), line


def test_text_from_a_stranger_reaches_the_screen_as_the_escape_that_names_it():
    raw, plain = drawn([Said("  title: \x1b]0;pwned\x07 fake", PLAN_FIELD), "tail\x1b[2J"])
    assert "\\x1b]0;pwned\\x07" in plain and "tail\\x1b[2J" in plain
    assert "\x1b]0;pwned" not in raw and "\x1b[2J" not in raw


# --- progress -------------------------------------------------------------------------------------------------------


def shown(display: TransferDisplay, tty) -> list[str]:  # noqa: ANN001
    """The screen as it is after a frame is drawn now (the display draws about four times a second by itself)."""
    display._live.refresh()  # noqa: SLF001
    return screen_of(tty.transcript())


def test_one_line_counts_the_files_and_the_current_files_bar_is_beneath_it():
    with terminal(columns=120) as tty:
        display = TransferDisplay(tty.stream, "uploading", "uploaded")
        display.files([("full_toy.mimebundle", 31_000), ("NOTICE", 2_000)])
        display.sending("full_toy.mimebundle", 16_000, 31_000)
        screen = shown(display, tty)
        display.close()
    assert "0/2 files uploaded" in screen[0] and "16 kB of 33 kB" in screen[0]
    assert "uploading full_toy.mimebundle" in screen[1] and "16 kB of 31 kB" in screen[1]
    assert "\u2501" in screen[1], "a bar"


def test_a_file_whole_stays_as_the_last_bar_and_so_does_the_count_when_it_is_over():
    with terminal(columns=120) as tty:
        display = TransferDisplay(tty.stream, "uploading", "uploaded")
        display.files([("a.bin", 1_000), ("b.bin", 2_000)])
        display.sending("a.bin", 1_000, 1_000)
        display.sent("a.bin")
        screen = shown(display, tty)
        assert "1/2 files uploaded" in screen[0] and "uploaded a.bin" in screen[1], "between the files, a is still there"
        display.sending("b.bin", 2_000, 2_000)
        display.sent("b.bin")
        display.finish()
        end = screen_of(tty.transcript())
        assert display._running is False  # noqa: SLF001
    lines = [line for line in end if line]
    assert len(lines) == 2, "the two lines, and nothing else, are left"
    assert lines[0] == "2/2 files uploaded  \u00b7  3 kB" and "uploaded b.bin" in lines[1]
    assert "uploading" not in "".join(lines) and "a.bin" not in "".join(lines), "only the most recent bar stays"


def test_between_two_files_the_line_says_what_is_under_way_and_the_screen_is_never_silent():
    with terminal(columns=120) as tty:
        display = TransferDisplay(tty.stream, "uploading", "uploaded")
        display.files([("a.bin", 1_000), ("b.bin", 2_000)])
        display.sending("a.bin", 1_000, 1_000)
        display.doing("Zenodo is committing a.bin, and its md5 is checked")
        committing = shown(display, tty)
        display.sent("a.bin")
        display.doing("waiting between requests (the service is asked gently)")
        waiting = shown(display, tty)
        display.close()
    assert "committing a.bin" in committing[0] and "waiting between requests" in waiting[0]
    assert all(frame and frame[0].strip() for frame in (committing, waiting))


def test_a_report_with_nothing_said_of_what_is_under_way_names_nothing_the_control():
    """The control of the test above: without ``doing`` the gap is a count and a bar, with no words about the wait."""
    with terminal(columns=120) as tty:
        display = TransferDisplay(tty.stream, "uploading", "uploaded")
        display.files([("a.bin", 1_000), ("b.bin", 2_000)])
        display.sending("a.bin", 1_000, 1_000)
        display.sent("a.bin")
        bare = shown(display, tty)
        display.close()
    assert "waiting" not in bare[0] and "committing" not in bare[0] and "checked" not in bare[0]


def test_what_is_under_way_before_any_file_is_a_line_of_its_own_and_nothing_is_left_if_no_file_moved():
    with terminal(columns=120) as tty:
        display = TransferDisplay(tty.stream, "uploading", "uploaded")
        display.doing("making or updating the draft on zenodo-sandbox, and reserving its DOI")
        assert "making or updating the draft" in shown(display, tty)[0]
        display.finish()
        assert [line for line in screen_of(tty.transcript()) if line] == [], "an unchanged push leaves no progress behind"


def test_a_file_of_no_stated_size_is_counted_without_a_total():
    with terminal(columns=120) as tty:
        display = TransferDisplay(tty.stream, "downloading", "downloaded")
        display.sending("bundle", 5_000, None)
        screen = shown(display, tty)
        display.close()
    assert "downloading bundle" in screen[1] and "5 kB" in screen[1] and " of " not in screen[1]
    assert screen[0].strip().startswith(("\u280b", "\u2819", "-", "\\", "|", "/")) or "0/1" in screen[0] or "files" in screen[0]


def test_the_lines_said_while_files_go_up_are_drawn_above_the_progress_which_goes_on():
    with terminal(columns=120) as tty:
        moving = TransferDisplay(tty.stream, "uploading", "uploaded")
        said = SaidDisplay(tty.stream, moving)
        moving.files([("NOTICE", 2_000), ("b.bin", 1)])
        moving.sending("NOTICE", 100, 2_000)
        said.say("! an upload was cut and is sent again")
        screen = shown(moving, tty)
        said.close()
    line = next(i for i, text in enumerate(screen) if "an upload was cut" in text)
    count = next(i for i, text in enumerate(screen) if "files uploaded" in text)
    assert line < count, "the line above, the progress below it"
    assert sum("files uploaded" in text for text in screen) == 1, "one copy of the progress, not one for each line said"


def test_a_question_takes_the_progress_off_the_screen_and_a_later_report_draws_it_again(monkeypatch):
    seen: list[list[str]] = []

    def answer(prompt: str = "") -> str:
        seen.append(screen_of(tty.transcript()))
        return "y"

    monkeypatch.setattr("builtins.input", answer)
    with terminal(columns=120) as tty:
        moving = TransferDisplay(tty.stream, "uploading", "uploaded")
        said = SaidDisplay(tty.stream, moving)
        moving.files([("a.bin", 1_000), ("b.bin", 1)])
        moving.sending("a.bin", 10, 1_000)
        assert "files uploaded" in "".join(shown(moving, tty))
        assert said.ask("Delete the draft?") is True
        moving.sending("a.bin", 500, 1_000)
        again = shown(moving, tty)
        said.close()
    assert not any("files uploaded" in text for text in seen[0]), "nothing of the progress is under a question"
    assert any("? Delete the draft?" in text for text in seen[0])
    assert any("files uploaded" in text for text in again)


#: How many reports the rate test makes at least, so the limit, not the reports' count, keeps the frames down.
REPORTS = 200


def test_the_progress_is_drawn_about_four_times_a_second_however_often_it_is_told():
    def drawn(display_type: type[TransferDisplay]) -> tuple[int, int, float]:
        """Told at least ``REPORTS`` times over at least 1.25 s, however coarse the host's sleep (macOS runners told it
        156 and 77 times in a fixed 1.25 s): the reports, the frames drawn, and the seconds it took."""
        with terminal(columns=120) as tty:
            display = display_type(tty.stream, "uploading", "uploaded")
            display.files([("a.bin", 1_000_000), ("b.bin", 1)])
            told, started = 0, time.perf_counter()
            while time.perf_counter() - started < 1.25 or told < REPORTS:
                told += 1
                display.sending("a.bin", told, 1_000_000)
                time.sleep(0.002)
            elapsed = time.perf_counter() - started
            display.close()
            frames = stripped(tty.transcript()).count("files uploaded")
        return told, frames, elapsed

    told, frames, elapsed = drawn(TransferDisplay)
    assert frames <= 4 * elapsed + 3, (told, frames, elapsed)

    class Every(TransferDisplay):
        """The control: a display that draws on every report."""

        def sending(self, key: str, done: int, total: int | None) -> None:
            super().sending(key, done, total)
            self._live.refresh()

    told, frames, elapsed = drawn(Every)
    assert frames > 4 * elapsed + 3, "the count would have caught a display drawing on every report"


def test_a_fault_of_the_displays_own_turns_it_off_and_raises_nothing(monkeypatch):
    with terminal(columns=120) as tty:
        display = TransferDisplay(tty.stream, "uploading", "uploaded")

        def broken(*_: object, **__: object) -> None:
            raise RuntimeError("the terminal went away")

        monkeypatch.setattr(display._live, "start", broken)  # noqa: SLF001
        display.files([("a.bin", 10)])
        display.sending("a.bin", 5, 10)
        display.sent("a.bin")
        display.doing("anything")
        display.hide()
        display.finish()
    assert display.failed is True


def test_the_arithmetic_of_a_transfer_leaves_the_first_second_out_and_never_goes_below_zero():
    held = TransferState("a", 10_000_000, 0.0)
    held.done = 1_000_000
    assert held.remaining(0.5) is None and held.speed(0.5) == 2_000_000
    assert held.remaining(2.0) == pytest.approx(18.0)
    assert "about 18 s left (an estimate)" in held.text(2.0) and "1 MB of 10 MB" in held.text(2.0)
    held.done = 10_000_000
    assert held.remaining(5.0) == 0.0 and "left" not in held.text(5.0)
    assert TransferState("a", None, 0.0).fraction is None and TransferState("a", 4, 0.0).fraction == 0.0


def test_a_file_that_is_whole_stops_its_clock_and_has_nothing_left():
    held = TransferState("a", 10_000_000, 0.0)
    held.done = 4_000_000
    held.complete(4.0)
    assert held.done == 10_000_000 and held.remaining(40.0) is None
    assert held.speed(40.0) == pytest.approx(10_000_000 / 4.0), "the speed it had, not one the waiting after it dilutes"
    assert held.text(40.0) == "10 MB  \u00b7  2 MB/s" and held.speed(4.0) == held.speed(400.0)
    quick = TransferState("b", 100, 0.0)
    quick.complete(0.01)
    assert "/s" not in quick.text(9.0), "a file that went in a blink has no speed worth stating"


# --- the voice: which stream is drawn on -------------------------------------------------------------------------------


@pytest.fixture
def streams(monkeypatch) -> Iterator[None]:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    _display.set_plain(False)
    yield
    _display.set_plain(False)


def voice_with(monkeypatch, out, err):
    import mime.cli as cli

    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return cli._Voice("uploading")  # noqa: SLF001


def test_each_stream_is_judged_alone_a_pipe_gets_the_old_voice_and_a_terminal_the_new(streams, monkeypatch):
    import mime.cli as cli

    with terminal() as one, terminal() as two:
        pipe = voice_with(monkeypatch, io.StringIO(), io.StringIO())
        assert pipe.say is cli._say and pipe.ask is cli._ask and pipe.typed is cli._typed and pipe.progress is None  # noqa: SLF001
        both = voice_with(monkeypatch, one.stream, two.stream)
        assert both.say != cli._say and both.progress is not None  # noqa: SLF001
        errors = voice_with(monkeypatch, io.StringIO(), two.stream)
        assert errors.say is cli._say and errors.progress is not None, "tee: lines to the pipe, progress to the terminal"  # noqa: SLF001
        lines = voice_with(monkeypatch, one.stream, io.StringIO())
        assert lines.say != cli._say and lines.progress is None  # noqa: SLF001
        for held in (both, errors, lines, pipe):
            held.close()


def test_the_control_a_terminal_asked_to_be_plain_gets_the_pipes_voice(streams, monkeypatch):
    import mime.cli as cli

    with terminal() as one, terminal() as two:
        for setting in (lambda: monkeypatch.setenv("NO_COLOR", "1"), lambda: monkeypatch.setenv("TERM", "dumb"),
                        lambda: _display.set_plain(True)):
            setting()
            held = voice_with(monkeypatch, one.stream, two.stream)
            assert held.say is cli._say and held.progress is None  # noqa: SLF001
            monkeypatch.delenv("NO_COLOR", raising=False)
            monkeypatch.setenv("TERM", "xterm-256color")
            _display.set_plain(False)


# --- a terminal that reads 0 columns ---------------------------------------------------------------------------------


def test_a_terminal_that_reads_0_columns_is_drawn_on_at_80_and_the_plan_and_the_question_are_on_the_screen(monkeypatch):
    """A pseudo-terminal nobody sized (a harness's, ``script``'s, ``docker exec -t``) reads 0 columns, which is not an
    error; a console told to draw 0 columns draws nothing, not the plan and not a question."""
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with terminal(columns=0) as tty:
        assert os.get_terminal_size(tty.stream.fileno()).columns == 0, "the precondition: the size reads 0"
        shown = SaidDisplay(tty.stream)
        for line in PLAN:
            shown.say(line)
        assert shown.ask(Said("Declare permission 1 publicly, under your name?", "question-permission")) is False
        shown.close()
        plain = stripped(tty.transcript())
    assert "Pushing 'full_toy'" in plain and "\u256d" in plain and "! 1 permission declared" in plain
    assert "? Declare permission 1 publicly, under your name?" in plain and plain.rstrip().endswith("[y/N]")
    assert max(len(line) for line in plain.splitlines() if line[:1] in "\u256d\u2502\u2570") <= 80, "the frame fits 80"


def test_a_terminal_that_reads_0_columns_takes_the_width_the_environment_names_before_80(monkeypatch):
    monkeypatch.setenv("COLUMNS", "60")
    with terminal(columns=0) as tty:
        assert _display._columns_of(tty.stream) == 60  # noqa: SLF001
        monkeypatch.setenv("COLUMNS", "0")
        assert _display._columns_of(tty.stream) == 80  # noqa: SLF001
        monkeypatch.setenv("COLUMNS", "wide")
        assert _display._columns_of(tty.stream) == 80  # noqa: SLF001
    assert _display._columns_of(io.StringIO()) == 80, "a stream with no size to read"  # noqa: SLF001


def test_the_control_a_console_told_the_size_the_terminal_reads_draws_nothing_at_0_columns(monkeypatch):
    """The fault this guards, planted: the width is the terminal's own reading, 0. The test above fails on this."""
    monkeypatch.setattr(_display, "_columns_of", lambda stream: os.get_terminal_size(stream.fileno()).columns)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with terminal(columns=0) as tty:
        shown = SaidDisplay(tty.stream)
        for line in PLAN:
            shown.say(line)
        shown.ask("Declare permission 1 publicly, under your name?")
        shown.close()
        plain = stripped(tty.transcript())
    assert "Pushing 'full_toy'" not in plain and "Declare permission" not in plain


# --- a hash or an address in a frame too narrow for it -------------------------------------------------------------


SHA = "a3f5c9e1" * 8


@pytest.mark.parametrize("columns", [40, 72, 140])
def test_a_sha256_in_a_permission_is_on_one_line_of_the_transcript_whatever_the_width(columns):
    """A frame narrower than the hash is left off, so that nothing folds it: the terminal wraps it for the eye, and a hash
    selected and copied is whole."""
    _, plain = drawn([Said(f"  1. You are declaring publicly, under your name, that Example Maker permitted you to "
                           f"redistribute this file (sha256 {SHA}).", PERMISSION)], columns)
    assert SHA in plain


def test_the_control_a_frame_that_folds_a_word_splits_a_sha256_at_40_columns(monkeypatch):
    monkeypatch.setattr(_display, "_too_wide", lambda rows, width: False)
    _, plain = drawn([Said(f"  1. You are declaring publicly, under your name, that Example Maker permitted you to "
                           f"redistribute this file (sha256 {SHA}).", PERMISSION)], 40)
    assert SHA not in plain, "the fault the test above guards"


def test_a_long_address_in_the_plan_is_not_folded_either():
    address = "https://example.org/robot-maker/6a3ebb1/arm_visual.usda"
    _, plain = drawn(PLAN, 40)
    assert address in plain


# --- the publish step is drawn when its last line is said ------------------------------------------------------------


def test_the_publish_step_is_drawn_when_its_last_line_is_said_not_when_the_next_thing_is():
    from mime._said import ending

    step = [Said("Publishing 'full_toy' on zenodo-sandbox as 10.5072/zenodo.629525 (version 1 of a new concept).",
                 STEP_HEAD), Said(SANDBOX_FACTS, STEP_TRIAL),
            Said("  You state that everyone named as a creator or contributor agreed to be named.", STEP_LINE)]
    with terminal(columns=120) as tty:
        shown = SaidDisplay(tty.stream)
        for line in ending(list(step)):
            shown.say(line)
        before_anything_else = stripped(tty.transcript())
        shown.close()
    assert "\u250f" in before_anything_else and "\u2517" in before_anything_else and "agreed to be named" in before_anything_else
    with terminal(columns=120) as tty:
        shown = SaidDisplay(tty.stream)
        for line in step:
            shown.say(line)
        assert stripped(tty.transcript()) == "", "the control: with no line marked as the last, nothing is drawn yet"
        shown.close()


# --- the questions go to the terminal the person answers at ----------------------------------------------------------


def child_environment(home: Path) -> dict[str, str]:
    held = {key: value for key, value in os.environ.items() if key not in ("NO_COLOR", "FORCE_COLOR", "TERM")}
    return {**held, "HOME": str(home), "USERPROFILE": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
            "TERM": "xterm-256color", "JAX_PLATFORMS": "cpu", "MIME_OFFLINE": "1"}


def asked_in_a_child(code: str, tmp_path: Path, terminals: list[str], typed: bytes = b"y\n") -> dict[str, str]:
    """``code`` in a child whose standard input is a terminal, and of whose output and error the ``terminals`` are: what
    arrived at each stream."""
    got, exit_code = run_with_streams([sys.executable, "-c", code], env=child_environment(tmp_path), cwd=tmp_path,
                                      terminals=terminals, typed=typed, timeout=100.0)
    assert exit_code == 0, got
    return got


QUESTION = "Delete the draft 629525?"


def test_a_question_with_the_output_piped_is_put_on_the_terminal_the_person_answers_at_and_not_into_the_pipe(tmp_path):
    """``mime push ... --publish | tee log``: the person answers at a terminal and the output is a pipe. Python's own prompt
    went into the pipe, and the person saw their own echo and nothing else."""
    got = asked_in_a_child(f"from mime.cli import _ask; print('ANSWER', _ask({QUESTION!r}))", tmp_path, ["err"])
    assert got["out"] == "ANSWER True\n", "the pipe holds what the command prints, and no question"
    assert f"{QUESTION} [y/N] " in stripped(got["tty"])


def test_the_control_input_with_a_prompt_puts_the_question_into_the_pipe(tmp_path):
    got = asked_in_a_child(f"print('ANSWER', input({QUESTION + ' [y/N] '!r}).strip())", tmp_path, ["err"])
    assert QUESTION in got["out"] and QUESTION not in stripped(got["tty"]), "the fault the test above guards"


def test_a_question_drawn_on_the_screen_has_its_hint_on_the_screen_too_when_standard_error_is_redirected(tmp_path):
    got = asked_in_a_child(
        f"import sys; from mime.cli._display import SaidDisplay; print('ANSWER', SaidDisplay(sys.stdout).ask({QUESTION!r}))",
        tmp_path, ["out"])
    screen = stripped(got["tty"])
    assert f"? {QUESTION}" in screen and "[y/N]" in screen and "ANSWER True" in screen
    assert got["err"] == "", "nothing of the question is in the file standard error is"


def test_the_control_input_with_a_hint_puts_the_hint_where_the_question_is_not(tmp_path):
    got = asked_in_a_child(
        f"import sys; from mime.cli._display import SaidDisplay; d = SaidDisplay(sys.stdout); d._put_question({QUESTION!r}); "
        f"print('ANSWER', input('[y/N] ').strip())", tmp_path, ["out"])
    assert "[y/N]" in got["err"], "the fault the test above guards: the hint went to the file, the question to the screen"


# --- a fault of the display never changes what the command does ------------------------------------------------------


def test_lines_fall_back_to_the_old_voice_when_the_display_cannot_draw_them(streams, monkeypatch, capsys):
    def broken(self: SaidDisplay, line: str) -> None:
        raise RuntimeError("the terminal went away")

    monkeypatch.setattr(SaidDisplay, "say", broken)
    with terminal() as one, terminal() as two:
        held = voice_with(monkeypatch, one.stream, two.stream)
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        held.say("the draft is made")
        held.say("the second line")
        written = sys.stdout.getvalue()
        held.close()
    assert written == "the draft is made\nthe second line\n", "said the way a pipe is told, from the first fault on"
