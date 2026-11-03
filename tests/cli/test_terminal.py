"""What ``mime`` draws on a terminal, through a real pseudo-terminal (POSIX; ``_platform_support.needs_pty`` says what
does not run on Windows): the live line of ``mime run`` (phases, bar, steps, speed, time remaining as an estimate, flags
by a step, how to stop, drawn about four times a second at most), a refusal as a block, the summary that leads with
what needs attention, the note beside JAX's damaged-cache warning, and ``--plain``, ``NO_COLOR`` and ``TERM=dumb``,
which give a terminal what a pipe gets.

The display is the only thing here that is not byte for byte what it was; ``test_pipes.py`` holds that the rest is."""

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import jax
import pytest

from _platform_support import needs_pty
from _terminal_support import run_on_terminal, run_with_streams, stripped, terminal
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.cli import _display
from mime.cli._display import RunDisplay, damaged_cache_note, set_plain, show_refusal, styled
from mime.cli._progress import RunState
from mime.saved import read_folder
from mime.testing.toys import full_toy

pytestmark = [needs_pty, pytest.mark.own_child_cache]


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("toy")
    full_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture(autouse=True)
def plain_off():
    yield
    set_plain(False)


def child_environment(home: Path, **extra: str) -> dict[str, str]:
    """A user's environment for a child: its own temporary HOME (the tracebacks go there, and the XLA cache where CI has
    set none; where it has, the test's own directory in its place, ``tests/_child_cache.py``), a terminal that is not
    dumb, none of the suite's own settings, and the CPU."""
    held = {key: value for key, value in os.environ.items()
            if key not in ("MIME_TRACEBACK", "NO_COLOR", "FORCE_COLOR", "TERM")}
    return {**held, "HOME": str(home), "USERPROFILE": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
            "MIME_TRACEBACK_DIR": str(home / "tracebacks"), "TERM": "xterm-256color", "JAX_PLATFORMS": "cpu",
            "MIME_OFFLINE": "1", **extra}


def dt_of(folder: Path) -> float:
    return read_folder(folder).lock["resolved"]["run"]["dt"]


def test_a_stream_is_drawn_on_only_when_it_is_a_terminal_not_asked_to_be_plain(monkeypatch):
    import io

    with terminal() as tty:
        assert styled(tty.stream)
        assert not styled(io.StringIO())  # a pipe, a file, a captured stream
        monkeypatch.setenv("NO_COLOR", "1")
        assert not styled(tty.stream)
        monkeypatch.delenv("NO_COLOR")
        monkeypatch.setenv("NO_COLOR", "")  # set and empty is not asking, as the convention says
        assert styled(tty.stream)
        monkeypatch.setenv("TERM", "dumb")
        assert not styled(tty.stream)
        monkeypatch.setenv("TERM", "xterm")
        set_plain(True)
        assert not styled(tty.stream)
        set_plain(False)
        assert styled(tty.stream)


def test_plain_is_for_one_command_line_and_does_not_outlive_the_call(tmp_path, monkeypatch):
    """``--plain`` sets a flag for the length of a ``main`` call: a later call, or a stream asked about after it, is not
    left plain by it."""
    from mime.cli import main as mime_main

    monkeypatch.chdir(tmp_path)
    with terminal() as tty:
        assert mime_main(["--plain"]) == 0
        assert styled(tty.stream)


def test_the_live_line_names_the_phase_then_shows_the_bar_the_estimate_the_flags_and_how_to_stop():
    with terminal() as tty:
        display = RunDisplay(tty.stream)
        display.phase("loading the folder")
        display._live.refresh()
        assert "loading the folder" in stripped(tty.transcript())
        now = time.perf_counter()
        state = RunState(1000, 0.001, now - 10.0)
        display.running(state)
        display._live.refresh()
        assert "compiling the first chunk" in stripped(tty.transcript()) and "Ctrl+C to stop" in stripped(tty.transcript())
        for steps, ago in ((100, 3.0), (200, 2.0), (300, 1.0)):
            state.chunk(steps, now - ago)
        display._live.refresh()
        text = stripped(tty.transcript())
        assert " 30%  step 300 of 1,000  ·  300 ms of 1 s simulated" in text
        assert "100 steps/s" in text and "no flag raised yet (seen at chunk ends)" in text
        # 700 steps left at 100 a second is 7 s, less the second since the last chunk ended, less the time the test took
        assert re.search(r"about (5\.[5-9]|6\.0) s left \(an estimate\)", text), text
        state.chunk(400, now - 0.5, ["swim.slip"])
        display._live.refresh()
        assert "flags raised: swim.slip by step 400" in stripped(tty.transcript())
        assert " at step" not in stripped(tty.transcript())
        display.close()
        assert tty.transcript().endswith("\x1b[?25h") or "\x1b[?25h" in tty.transcript()[-40:], "the cursor is shown again"


def test_the_live_line_is_drawn_about_four_times_a_second_at_most_whatever_the_run_reports():
    with terminal() as tty:
        display = RunDisplay(tty.stream)
        now = time.perf_counter()
        state = RunState(100_000, 0.001, now)
        display.running(state)
        started, steps = time.perf_counter(), 0
        while time.perf_counter() - started < 2.0:
            steps += 10
            state.chunk(steps, time.perf_counter(), ["a"] if steps == 10 else None)  # the run reporting, far oftener
            time.sleep(0.0005)
        display.close()
        frames = stripped(tty.transcript()).count("Ctrl+C to stop")
    assert 3 <= frames <= 4 * 2 + 3, f"{frames} frames in 2 s"


def test_a_refusal_is_a_block_with_its_code_what_is_wrong_what_to_do_why_and_where_the_docs_are(monkeypatch, tmp_path):
    from mime.errors._display import footer_lines

    monkeypatch.delenv("MIME_TRACEBACK")  # the suite shows errors in full; a user's command shows the short form
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path))

    refusal = Refusal("run.duration", RefusalClass.USER, "a run of 0.001 s is not a whole number of base steps of 0.02 s",
                      "run a whole number of steps: run(s(0.02))", why="a run steps the graph by whole base steps",
                      where="/work/script.py:12")
    error = BuildRefused([refusal])
    with terminal(columns=90) as tty:
        show_refusal(error, "mime run is refused", footer_lines(error, switch="--traceback"), tty.stream)
        raw = tty.transcript()
    text = stripped(raw)
    for part in ("mime run is refused", "run.duration", "(yours to fix)", "not a whole number of base steps",
                 "what to do: run a whole number of steps", "why: a run steps the graph by whole base steps",
                 "where: /work/script.py:12", "docs for run.duration: https://microrobotica.org/mime/",
                 "The full traceback is in", "--traceback shows it here"):
        assert part in text, part
    assert "╭" in text and "╰" in text, "a box"
    assert "[user]" not in text, "the plain text's shape is the pipe's"
    docs = [line for line in text.splitlines() if line.startswith("docs for run.duration")]
    assert len(docs) == 1 and docs[0].endswith("run.duration"), "the address is on one line, to be copied"


def installed_and_pinned(unpinned: int, *, missing: int = 0, mispinned: int = 0):
    """The refusals ``mime env check`` makes of an environment with ``unpinned`` distributions nothing pins, ``missing``
    pinned ones not installed and ``mispinned`` installed at another version, from the function that makes them."""
    from mime.env._check import Expected, distribution_refusals

    installed = {f"extra{n:02d}": [f"1.{n}.0"] for n in range(unpinned)}
    expected: dict[str, Expected] = {}
    for n in range(missing):
        expected[f"absent{n:02d}"] = Expected(f"2.{n}.0", "the cpu pins")
    for n in range(mispinned):
        installed[f"drift{n:02d}"] = [f"3.{n}.1"]
        expected[f"drift{n:02d}"] = Expected(f"3.{n}.0", "the base image")
    return distribution_refusals(installed, expected)


def every_subject_is_shown(text: str, refusals) -> None:
    """The count of subjects against the refusals: each refusal about a subject has one row of the drawn block whose
    words are its cells in order, and no row is drawn more than once."""
    rows = [line.strip("│ ╭╮╰╯─").split() for line in text.splitlines()]
    for refusal in refusals:
        assert refusal.subject is not None
        cells = list(refusal.subject.cells)
        words = [word for cell in cells for word in cell.split()]
        found = [row for row in rows if any(row[i:i + len(words)] == words for i in range(len(row)))]
        assert len(found) == 1, f"{refusal.code} {cells} is drawn on {len(found)} rows, not one"


def drawn(refusals, columns=100) -> str:
    error = BuildRefused(refusals)
    with terminal(columns=columns, rows=200) as tty:
        show_refusal(error, "mime env is refused", [], tty.stream)
        return stripped(tty.transcript())


def test_refusals_of_one_code_are_one_block_with_a_row_for_every_subject_and_the_action_once():
    refusals = installed_and_pinned(15)
    text = drawn(refusals)
    assert len(refusals) == 15 and text.count("env.unpinned   (") == 1, "the code is the head of one block"
    assert "15 installed distributions are pinned by nothing:" in text and "distribution" in text
    assert text.count("what to do:") == 1 and "pip uninstall <name>" in text
    assert text.count("docs for env.unpinned") == 1
    every_subject_is_shown(text, refusals)


def test_codes_are_blocks_in_the_order_of_their_first_refusal_and_each_has_its_own_columns():
    refusals = installed_and_pinned(3, missing=2, mispinned=2)
    assert [r.code for r in refusals] == ["env.missing", "env.missing", "env.pin", "env.pin", "env.unpinned",
                                          "env.unpinned", "env.unpinned"]
    text = drawn(refusals)
    assert text.index("env.missing") < text.index("env.pin   (") < text.index("env.unpinned")
    assert "pinned by" in text and "pinned at" in text
    every_subject_is_shown(text, refusals)


def test_refusals_whose_actions_differ_are_not_merged_and_a_lone_subject_is_the_old_sentence():
    from mime.build._refusals import Subject

    def refusal(name, action):
        subject = Subject("things are wrong", ("thing",), (name,), action)
        return Refusal("env.unpinned", RefusalClass.ENVIRONMENT, f"{name} is wrong", subject.acted(), subject=subject)

    different = [refusal("a", "mend <name> this way"), refusal("b", "mend <name> that way")]
    text = drawn(different)
    assert "things are wrong" not in text and "a is wrong" in text and "b is wrong" in text
    assert "what to do: mend a this way" in text and "what to do: mend b that way" in text
    assert "a is wrong" in drawn(different[:1]) and "1 things" not in drawn(different[:1])


def test_the_control_a_subject_dropped_from_a_group_is_found_by_the_count(monkeypatch):
    """The check above fails for the thing it guards, every run: the table with its last row left out is caught."""
    refusals = installed_and_pinned(6)
    every_subject_is_shown(drawn(refusals), refusals)
    original = _display._subjects
    monkeypatch.setattr(_display, "_subjects", lambda block: original(block[:-1]))
    with pytest.raises(AssertionError, match="is drawn on 0 rows"):
        every_subject_is_shown(drawn(refusals), refusals)


def test_every_refusal_is_in_exactly_one_block_whatever_the_mix():
    refusals = installed_and_pinned(5, missing=3, mispinned=4)
    plain = Refusal("env.lab", RefusalClass.USER, "the lab's file is not pins", "write pins")
    mixed = [plain, *refusals, plain]
    blocks = _display.refusal_groups(mixed)
    assert sorted(id(r) for block in blocks for r in block) == sorted(id(r) for r in mixed)
    assert [len(block) for block in blocks] == [1, 3, 4, 5, 1]


def test_words_from_outside_are_text_in_a_block_never_markup_or_a_command_to_the_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path))
    hostile = "[bold red]x[/] \x1b]0;pwned\x07 \x1b[31mred\x1b[0m"
    error = BuildRefused([Refusal("run.duration", RefusalClass.USER, hostile, "do [link=http://x]this[/link]")])
    with terminal(columns=120) as tty:
        show_refusal(error, "mime run is refused", [], tty.stream)
        raw = tty.transcript()
    text = stripped(raw)
    assert "[bold red]x[/]" in text and "[link=http://x]this[/link]" in text
    assert "\\x1b]0;pwned\\x07" in text and "\\x1b[31mred" in text
    # the block's own border is red (ESC [31m then a box character); the message's escape is not an escape here
    assert "\x1b]0;" not in raw and "\x1b[31mred" not in raw, "no escape from the message reached the terminal"


def write_warning(message, category, filename, lineno, file=None, line=None):
    """What a warning's display does, as the test's own: pytest has taken ``warnings.showwarning`` for its report."""
    file.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")


def test_a_damaged_cache_warning_gets_one_note_naming_the_command_that_clears_it_beside_it(monkeypatch):
    monkeypatch.setattr(warnings, "showwarning", write_warning)
    message = "Error reading persistent compilation cache entry for 'jit_step': ValueError: bad entry"
    with terminal(columns=200) as tty, warnings.catch_warnings():
        warnings.simplefilter("always")
        held = warnings.showwarning
        with damaged_cache_note(tty.stream):
            warnings.showwarning(UserWarning(message), UserWarning, "jax/_src/compiler.py", 806, tty.stream)
            warnings.showwarning(UserWarning(message), UserWarning, "jax/_src/compiler.py", 806, tty.stream)
            warnings.showwarning(UserWarning("an unrelated warning"), UserWarning, "somewhere.py", 1, tty.stream)
        assert warnings.showwarning is held, "put back"
        text = stripped(tty.transcript())
    assert text.count("mime cache clear") == 1, "once, not once a warning"
    assert text.index(message) < text.index("mime cache clear") < text.index("an unrelated warning")


def test_the_note_is_not_written_where_nothing_is_drawn(monkeypatch):
    """The control: a captured stream is not a terminal, so the warning is JAX's alone, as it always was."""
    import io

    monkeypatch.setattr(warnings, "showwarning", write_warning)
    caught = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with damaged_cache_note(caught):
            warnings.showwarning(UserWarning("Error reading persistent compilation cache entry for 'x'"), UserWarning,
                                 "compiler.py", 806, caught)
    assert "mime cache clear" not in caught.getvalue() and "persistent compilation cache entry" in caught.getvalue()


def run_command(saved, tmp_path, *words, env=None, columns=100):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return run_on_terminal([sys.executable, "-m", "mime.cli", *words], env=child_environment(home, **(env or {})),
                           cwd=tmp_path, columns=columns)


def test_mime_run_on_a_terminal_draws_its_phases_and_ends_with_what_needs_attention_then_the_next_steps(saved, tmp_path):
    out = tmp_path / "run"
    raw, code = run_command(saved, tmp_path, "run", str(saved), "--duration", f"{40 * dt_of(saved)!r}s", "--out", str(out))
    text = stripped(raw)
    assert code == 0, text
    assert "loading" in text and "Ctrl+C to stop" not in text.splitlines()[-1]
    assert "Needs attention" in text or "Nothing needs attention" in text
    assert "profile exploratory: 17 items pending, not sealed" in text, "the profile's wall of items is one line"
    assert "pending: bath.flow.coupling" not in text, "the items themselves are behind --full"
    for part in ("ran full_toy: 40 steps of 0.02 s", "Readings", "wrote ", "Next", f"mime verify {out / 'experiment'}",
                 "RunResult.read(", "mime serve ", "--full prints the load report"):
        assert part in text, part
    assert "running 0.8 s of simulated time" not in text, "the plain line is the pipe's"
    assert (out / "report.json").is_file() and json.loads((out / "outputs.json").read_text())["steps"] == 40


@pytest.mark.parametrize(("words", "env"), [
    (["--plain"], {}), ([], {"NO_COLOR": "1"}), ([], {"TERM": "dumb"}),
], ids=["--plain", "NO_COLOR", "TERM=dumb"])
def test_a_terminal_asked_to_be_plain_gets_exactly_what_a_pipe_gets(saved, tmp_path, words, env):
    import subprocess

    duration = f"{4 * dt_of(saved)!r}s"
    home = tmp_path / "home"
    home.mkdir()
    piped = subprocess.run([sys.executable, "-m", "mime.cli", "run", str(saved), "--duration", duration, "--out",
                            str(tmp_path / "piped")], capture_output=True, text=True, check=False,
                           env=child_environment(home), cwd=tmp_path)
    assert piped.returncode == 0 and piped.stderr == ""
    raw, code = run_command(saved, tmp_path, *words, "run", str(saved), "--duration", duration, "--out",
                            str(tmp_path / "terminal"), env=env)
    assert code == 0 and "\x1b" not in raw, "nothing is drawn: no control sequence at all"
    assert raw.replace("\r\n", "\n").replace(str(tmp_path / "terminal"), str(tmp_path / "piped")) == piped.stdout


def test_a_refused_duration_on_a_terminal_is_a_block_and_names_the_forms(saved, tmp_path):
    raw, code = run_command(saved, tmp_path, "run", str(saved), "--duration", "10x", "--out", str(tmp_path / "run"))
    text = stripped(raw)
    assert code == 2, text
    for part in ("mime run is refused", "run.duration", "--duration 10x is not a duration", "what to do: write it with a unit", "10s, 500ms, 250us, 2min, 1h", "docs for run.duration: https://"):
        assert part in text, part
    assert not (tmp_path / "run").exists()


def test_the_same_refusal_with_plain_is_the_text_a_pipe_gets(saved, tmp_path):
    raw, code = run_command(saved, tmp_path, "--plain", "run", str(saved), "--duration", "10x", "--out",
                            str(tmp_path / "run"))
    assert code == 2 and "\x1b" not in raw and "╭" not in raw
    assert "mime run is refused:\n[user] run.duration: --duration 10x is not a duration" in raw.replace("\r\n", "\n")


def test_a_write_error_of_the_cache_is_not_called_damage_and_does_not_use_up_the_note(monkeypatch):
    """JAX words a failed write (a full disk, a directory not writable) with the phrase of a failed read. It gets no note, and
    a read error after it still does."""
    monkeypatch.setattr(warnings, "showwarning", write_warning)
    writing = "Error writing persistent compilation cache entry for 'jit_step': OSError: [Errno 28] No space left on device"
    reading = "Error reading persistent compilation cache entry for 'jit_step': ValueError: bad entry"
    with terminal(columns=200) as tty, warnings.catch_warnings():
        warnings.simplefilter("always")
        with damaged_cache_note(tty.stream):
            warnings.showwarning(UserWarning(writing), UserWarning, "jax/_src/compiler.py", 871, tty.stream)
            assert "mime cache clear" not in stripped(tty.transcript())
            warnings.showwarning(UserWarning(reading), UserWarning, "jax/_src/compiler.py", 806, tty.stream)
        text = stripped(tty.transcript())
    assert text.count("mime cache clear") == 1 and text.index(writing) < text.index(reading) < text.index("mime cache clear")


#: A wrapper that makes the run print one line to standard output per chunk, as a controller's law or a node's host
#: callback might, and, when asked, plants the regression of redirecting standard output through the display.
STDOUT_WRAPPER = '''
import os, sys
import mime.cli
import mime.cli._display as display
from mime._interrupt import Interrupt

if os.environ.get("PLANT_REDIRECT"):
    display._shares_a_terminal = lambda *streams: True

asked = Interrupt.stop


def stop(self, progress=None):
    print("SIDE-LINE", flush=True)
    return asked(self, progress)


Interrupt.stop = stop
sys.exit(mime.cli.main(sys.argv[1:]))
'''


def lines_of_a_run_on(saved, tmp_path, terminals, *, plant=False, name="run"):
    """The streams of ``mime run`` of the toy for 40 steps with the stdout wrapper, the streams in ``terminals`` real
    terminals and the others pipes; what each received, and the exit code."""
    wrapper = tmp_path / "stdout_wrapper.py"
    wrapper.write_text(STDOUT_WRAPPER, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = child_environment(home, **({"PLANT_REDIRECT": "1"} if plant else {}))
    got, code = run_with_streams([sys.executable, str(wrapper), "run", str(saved), "--duration", f"{40 * dt_of(saved)!r}s",
                                  "--out", str(tmp_path / name)], env=environment, cwd=tmp_path, terminals=terminals)
    assert code == 0, got
    return got


@pytest.fixture(scope="module")
def side_lines(saved, tmp_path_factory) -> int:
    """How many lines the wrapper prints to standard output in a run of 40 steps with both streams piped: one a chunk."""
    count = lines_of_a_run_on(saved, tmp_path_factory.mktemp("piped"), ())["out"].count("SIDE-LINE")
    assert count >= 5
    return count


def test_standard_output_stays_standard_output_when_standard_error_is_the_terminal_that_is_drawn_on(saved, tmp_path,
                                                                                                    side_lines):
    """``mime run ... | tee log``: the display is on standard error, and a line the run's own code prints to standard output
    reaches the pipe, none of them the terminal."""
    expected = side_lines
    got = lines_of_a_run_on(saved, tmp_path, ("err",))
    assert "Ctrl+C to stop" in stripped(got["tty"]), "the display was drawn on the terminal, so this is the case under test"
    assert got["out"].count("SIDE-LINE") == expected and got["tty"].count("SIDE-LINE") == 0
    assert "ran 40 steps; wrote " in got["out"], "and the pipe still gets everything it always got"


def test_standard_output_stays_standard_output_when_it_is_the_terminal_and_standard_error_a_pipe(saved, tmp_path, side_lines):
    """The reverse: no display (it is standard error's to draw), the lines are on the terminal that standard output is
    along with the run's own words, and standard error, a pipe, is left empty."""
    expected = side_lines
    got = lines_of_a_run_on(saved, tmp_path, ("out",))
    assert got["tty"].count("SIDE-LINE") == expected and "Ctrl+C to stop" not in stripped(got["tty"])
    assert got["err"] == "" and "running 0.8 s of simulated time" in stripped(got["tty"])


def test_with_both_streams_on_one_terminal_a_line_printed_to_standard_output_arrives_whole_above_the_display(
        saved, tmp_path, side_lines):
    expected = side_lines
    got = lines_of_a_run_on(saved, tmp_path, ("out", "err"))
    assert got["tty"].count("SIDE-LINE") == expected and "Ctrl+C to stop" in stripped(got["tty"])


def test_the_control_redirecting_standard_output_through_the_display_empties_the_pipe_as_the_test_would_find(
        saved, tmp_path, side_lines):
    """The planted regression (``redirect_stdout=True`` whatever standard output is): the pipe loses every line to the terminal,
    and the check of the first test above sees both."""
    expected = side_lines
    got = lines_of_a_run_on(saved, tmp_path, ("err",), plant=True)
    assert got["out"].count("SIDE-LINE") == 0 != expected and got["tty"].count("SIDE-LINE") == expected


def refusals_of_env_check(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("[environment] env.")]


def traceback_file_removed(text: str) -> str:
    return re.sub(r"mime-\d{8}T\d{6}\.\d{9}-\d+-", "mime-STAMP-", text)


def test_env_check_on_a_terminal_is_one_block_per_code_and_every_package_the_pipe_names_is_a_row(saved, tmp_path):
    import subprocess

    home = tmp_path / "home"
    home.mkdir()
    words = [sys.executable, "-m", "mime.cli", "env", "check", "--variant", "cpu"]
    piped = subprocess.run(words, capture_output=True, text=True, check=False, env=child_environment(home), cwd=tmp_path)
    assert piped.returncode == 2
    listed = refusals_of_env_check(piped.stderr)
    assert len(listed) >= 2, "a development environment holds distributions nothing pins; the grouping needs two"
    raw, code = run_command(saved, tmp_path, "env", "check", "--variant", "cpu")
    text = stripped(raw)
    assert code == 2, text
    for line in listed:
        name = line.split(": ", 1)[1].split(" ")[0]
        assert any(row.strip("│ ").split()[:1] == [name] for row in text.splitlines()), f"{name} has no row"
    codes = {line.split(" ")[1].rstrip(":") for line in listed}
    assert len(codes) < len(listed), "the control: the pipe has more refusals than codes, so the blocks group"
    assert text.count("what to do:") == len(codes) and text.count("docs for env.") == len(codes)
    assert len(text.splitlines()) < len(piped.stderr.splitlines()), "grouped output is shorter than the pipe's"


@pytest.mark.parametrize(("words", "env"), [
    (["--plain"], {}), ([], {"NO_COLOR": "1"}), ([], {"TERM": "dumb"}),
], ids=["--plain", "NO_COLOR", "TERM=dumb"])
def test_env_check_asked_to_be_plain_gets_exactly_what_a_pipe_gets(saved, tmp_path, words, env):
    import subprocess

    home = tmp_path / "home"
    home.mkdir()
    command = ["env", "check", "--variant", "cpu"]
    piped = subprocess.run([sys.executable, "-m", "mime.cli", *command], capture_output=True, text=True, check=False,
                           env=child_environment(home), cwd=tmp_path)
    assert piped.returncode == 2 and len(refusals_of_env_check(piped.stderr)) >= 2
    raw, code = run_command(saved, tmp_path, *words, *command, env=env)
    assert code == 2 and "\x1b" not in raw and "╭" not in raw, "nothing is drawn"
    # a terminal has one stream for both; the pipe's standard error is what carries a refusal
    assert traceback_file_removed(raw.replace("\r\n", "\n")) == traceback_file_removed(piped.stdout + piped.stderr)
