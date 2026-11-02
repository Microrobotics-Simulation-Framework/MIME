"""When output is not going to a terminal it is byte for byte what it was before the display existed: a pipe, a file,
CI's log and Modal's are read by programs. Every platform runs this file (a pipe is a pipe on Windows too).

The oracle is the old command's own format, written out here from the objects the command holds (the load report, the
result, the run folder's files); the control is that the same run drawn for a terminal is not it, and that a colour code
planted in a piped line is found."""

import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import pytest

import mime.cli
from mime.cli import main as mime_main
from mime.env import apply_lock_precision
from mime.experiment import Experiment
from mime.results import RunResult
from mime.saved import read_folder
from mime.testing.toys import full_toy

#: A control sequence or any escape character: nothing of the kind is in text a pipe gets.
ESCAPE = re.compile("\x1b")


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("toy")
    full_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    held = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", held)


def dt_of(folder: Path) -> float:
    return read_folder(folder).lock["resolved"]["run"]["dt"]


def what_mime_run_printed_before(folder: Path, out: Path, duration: float, steps: int) -> str:
    """``mime run``'s standard output as the command wrote it before the display: the load report, the line before the run,
    the line after it, and the result."""
    apply_lock_precision(folder)
    load = Experiment.load(folder).load_report
    assert load is not None
    return (f"{load.text()}\n"
            f"running {duration!r} s of simulated time; Ctrl+C stops it at the next chunk and writes the steps it "
            "reached, twice quits at once\n"
            f"ran {steps} steps; wrote {out / 'outputs.npz'}, {out / 'outputs.json'}, {out / 'run.json'} and the run's "
            f"saved form, {out / 'experiment'}\n"
            f"{RunResult.read(out).text()}\n")


def planted_colour(text: str) -> bool:
    return ESCAPE.search(text) is not None


def test_piped_output_is_exactly_the_old_commands_and_holds_no_escape(saved, capsys, tmp_path):
    duration = 4 * dt_of(saved)
    out = tmp_path / "run"
    assert mime_main(["run", str(saved), "--duration", repr(duration), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == what_mime_run_printed_before(saved, out, duration, 4)
    assert not planted_colour(captured.out + captured.err)


def test_a_duration_with_a_unit_is_the_same_run_and_prints_the_same_seconds(saved, capsys, tmp_path):
    from mime.units import ms

    out = tmp_path / "run"
    assert dt_of(saved) == 0.02
    assert mime_main(["run", str(saved), "--duration", "80ms", "--out", str(out)]) == 0
    assert capsys.readouterr().out == what_mime_run_printed_before(saved, out, ms(80.0).value, 4)


def test_the_control_a_colour_code_planted_in_piped_output_is_found(saved, capsys, tmp_path, monkeypatch):
    """The check above fails for the thing it guards: a line that a regression coloured. (``_say`` writes a control
    character out as its escape, so the plant is in the result's text, which is printed as it is.)"""
    original = RunResult.text
    monkeypatch.setattr(RunResult, "text", lambda self: f"\x1b[1m{original(self)}\x1b[0m")
    out = tmp_path / "run"
    assert mime_main(["run", str(saved), "--duration", repr(2 * dt_of(saved)), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert planted_colour(printed)
    monkeypatch.undo()
    assert printed != what_mime_run_printed_before(saved, out, 2 * dt_of(saved), 2)


def test_the_control_output_drawn_for_a_terminal_is_not_the_old_commands(saved, capsys, tmp_path, monkeypatch):
    """The same run with the terminal's display forced on a captured stream: what it prints differs, so the equality
    above is a test that a pipe is not drawn for."""
    monkeypatch.setattr(mime.cli, "styled", lambda stream: True)
    out = tmp_path / "run"
    assert mime_main(["run", str(saved), "--duration", repr(2 * dt_of(saved)), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Needs attention" in printed or "Nothing needs attention" in printed
    assert printed != what_mime_run_printed_before(saved, out, 2 * dt_of(saved), 2)


def test_a_refusal_on_a_pipe_is_the_old_text_with_no_block(saved, capsys, tmp_path):
    assert mime_main(["run", str(saved), "--duration", repr(1.5 * dt_of(saved)), "--out", str(tmp_path / "run")]) == 2
    captured = capsys.readouterr()
    assert captured.out.startswith("load report: ") and not planted_colour(captured.out + captured.err)
    assert "mime run is refused:\n[user] run.duration: a run of " in captured.err
    assert "\n    do: run a whole number of steps" in captured.err and "\n    see: https://" in captured.err
    assert "docs for" not in captured.err and "╭" not in captured.err


def test_a_child_process_piped_prints_what_the_function_prints(saved, tmp_path):
    """Through the real command and real pipes, as a CI log or a script reads it (the one test of the file whose
    subprocess is the installed console script's module)."""
    duration = repr(2 * dt_of(saved))
    done = subprocess.run([sys.executable, "-m", "mime.cli", "run", str(saved), "--duration", duration, "--out",
                           str(tmp_path / "run")], capture_output=True, text=True, check=False,
                          env={**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
                               "XDG_CACHE_HOME": str(tmp_path / ".cache"), "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr
    assert done.stdout == what_mime_run_printed_before(saved, tmp_path / "run", 2 * dt_of(saved), 2)
    assert not planted_colour(done.stdout + done.stderr)


#: The settings that ask a program to colour a pipe or to say it is interactive: ``rich`` itself honours the first and the
#: third, and CI configurations set the first and the second. None of them makes a pipe a terminal for ``mime``.
FORCING = [{"FORCE_COLOR": "1"}, {"FORCE_COLOR": "3"}, {"TTY_COMPATIBLE": "1"}, {"TTY_INTERACTIVE": "1"}, {"CI": "true"}]


def printed_with(environment, saved, capsys, tmp_path, monkeypatch, *, plant=False):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    if plant:
        from mime.cli import _display

        real = _display.styled
        monkeypatch.setattr(mime.cli, "styled",
                            lambda stream: bool(os.environ.get("FORCE_COLOR")) or real(stream))
    out = tmp_path / "run"
    assert mime_main(["run", str(saved), "--duration", repr(2 * dt_of(saved)), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    return captured.out, captured.err, out


@pytest.mark.parametrize("environment", FORCING, ids=lambda each: "+".join(f"{k}={v}" for k, v in each.items()))
def test_a_pipe_told_to_force_colour_is_still_exactly_the_old_commands_and_holds_no_escape(environment, saved, capsys,
                                                                                         tmp_path, monkeypatch):
    printed, error, out = printed_with(environment, saved, capsys, tmp_path, monkeypatch)
    assert error == "" and not planted_colour(printed)
    assert printed == what_mime_run_printed_before(saved, out, 2 * dt_of(saved), 2)


def test_the_control_a_pipe_drawn_for_because_force_colour_is_set_is_found(saved, capsys, tmp_path, monkeypatch):
    """The arm above fails for the regression it guards: ``styled`` honouring ``FORCE_COLOR`` on a stream that is not a
    terminal draws the summary and the progress on the pipe, and the same comparison finds it."""
    printed, error, out = printed_with({"FORCE_COLOR": "1"}, saved, capsys, tmp_path, monkeypatch, plant=True)
    assert printed != what_mime_run_printed_before(saved, out, 2 * dt_of(saved), 2)
    assert "Needs attention" in printed or "Nothing needs attention" in printed or planted_colour(error)
