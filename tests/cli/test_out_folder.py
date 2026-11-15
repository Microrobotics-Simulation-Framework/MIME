"""Where a run may be written (UX-1's F2): ``mime run --out`` and ``BuiltExperiment.run_to`` wrote into any folder, and
replaced the files of a run's own names in it (``outputs.json``, ``run.json``, ``experiment/``'s saved form) with nothing
said: a user's folder with an ``outputs.json`` of its own lost it. A run never changes a file that is not its own now.

- ``out`` that is new, empty or MIME's own run folder (``run.json``, and only files of the names a run writes) is
  written into, as before: a re-run into its own folder is the notebook's and the script's habit;
- any other folder is refused **before the run starts** (``run.out-not-empty``), before the saved experiment is even
  loaded, and is left byte for byte, modification times included, as it was: UX-1's case (a folder holding
  ``experiment/notes.txt`` and an ``outputs.json`` of the user's), a folder with a file of its own and a folder that
  holds a run's files and a stranger's;
- ``--force`` (``force=True``) runs into it all the same: a warning at the start says what it holds and that Ctrl+C
  cancels until writing begins, an amber notice says when writing begins, and every file that is not of a run's names is
  kept byte for byte; Ctrl+C before writing starts cancels the run (130, the folder as it was, nothing written).

UX-1 reported that the run deleted a file of the user's (``experiment/notes.txt``); it did not here: in every arm below
the note survives, and what the base lost was ``outputs.json``, replaced by the run's own. The invariant tested is the
stronger one, that nothing which is not a run's name is touched, and that a refused folder is not touched at all."""

from __future__ import annotations

import io
import signal
import threading
from pathlib import Path

import jax
import pytest
from folder_state import state_of, untouched

from mime._interrupt import INTERRUPTED, Interrupt
from mime.build import ArgumentRefused
from mime.cli import main as mime_main
from mime.experiment import RUN_EXPERIMENT, RUN_OUTPUTS, RUN_RECORD, Experiment
from mime.live import COMMAND_LOG
from mime.results import RUN_MANIFEST, RUN_REPORT, RunResult
from mime.saved import read_folder
from mime.testing.toys import full_toy
from mime.units import s
from mime.viz._recording import RECORDING_FILE

CODE = "run.out-not-empty"


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


def run(saved: Path, out: Path, *more: str, steps: int = 4) -> int:
    return mime_main(["run", str(saved), "--duration", f"{steps * dt_of(saved)!r}s", "--out", str(out), *more])


def files(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*"))
            if path.is_file()}


def ux1_folder(root: Path) -> Path:
    """UX-1's case: a folder of the user's with a note in ``experiment/`` and an ``outputs.json`` of its own."""
    out = root / "mine"
    (out / "experiment").mkdir(parents=True)
    (out / "experiment" / "notes.txt").write_text("my notes\n", encoding="utf-8")
    (out / RUN_MANIFEST).write_text('{"mine": true}\n', encoding="utf-8")
    return out


def thesis_folder(root: Path) -> Path:
    out = root / "thesis"
    out.mkdir()
    (out / "chapter1.tex").write_text("\\chapter{One}\n", encoding="utf-8")
    return out


def own_folder_with_a_stranger(saved: Path, root: Path) -> Path:
    out = root / "own"
    assert run(saved, out) == 0
    (out / "chapter1.tex").write_text("mine\n", encoding="utf-8")
    return out


# -- what a run is written into without a flag ------------------------------------------------------------------------


def test_a_new_folder_an_empty_one_and_mimes_own_run_folder_are_written_into(saved, tmp_path, capsys):
    assert run(saved, tmp_path / "new") == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(saved, empty) == 0
    assert run(saved, empty, steps=5) == 0, "a second run into its own folder, as a script re-run does"
    assert RunResult.read(empty).steps == 5
    assert "warning" not in capsys.readouterr().err


@pytest.mark.parametrize("make", [ux1_folder, thesis_folder, own_folder_with_a_stranger],
                         ids=["UX-1's case", "a thesis folder", "a run folder with a stranger's file"])
def test_any_other_folder_is_refused_before_the_run_and_left_as_it_was(saved, tmp_path, capsys, make, monkeypatch):
    out = make(saved, tmp_path) if make is own_folder_with_a_stranger else make(tmp_path)
    before = files(out)
    monkeypatch.setattr(Experiment, "load", lambda folder: (_ for _ in ()).throw(AssertionError("loaded")))
    with untouched(out):
        assert run(saved, out) == 2
    err = capsys.readouterr().err
    assert CODE in err and "not an empty folder or a MIME run folder" in err and "--force" in err
    assert "Ctrl+C cancels the run until writing begins" in err and "amber notice" in err
    assert files(out) == before


def test_a_file_where_the_folder_should_be_is_refused(saved, tmp_path):
    out = tmp_path / "out"
    out.write_text("a file\n", encoding="utf-8")
    assert run(saved, out) == 2 and out.read_text(encoding="utf-8") == "a file\n"


def test_the_python_run_is_refused_alike_and_force_runs_into_it(saved, tmp_path):
    built = Experiment.load(saved).build()
    out = ux1_folder(tmp_path)
    with untouched(out), pytest.raises(ArgumentRefused) as refused:
        built.run_to(out, s(4 * dt_of(saved)))
    assert refused.value.codes() == [CODE] and "force=True" in refused.value.refusals[0].action
    written = built.run_to(out, s(4 * dt_of(saved)), force=True)
    assert written.steps == 4 and (out / "experiment" / "notes.txt").read_text(encoding="utf-8") == "my notes\n"


def test_what_a_run_folder_holds_is_what_this_module_says_a_run_writes(saved, tmp_path):
    """The names in ``RUN_FOLDER_NAMES`` are the constants of the writers (the recording layer and a live session's log
    are not imported by the run, so they are named here), and a real run folder holds nothing outside them."""
    from mime.experiment._experiment import RUN_FOLDER_NAMES, foreign_entries

    assert {RUN_OUTPUTS, RUN_RECORD, RUN_EXPERIMENT, RUN_MANIFEST, RUN_REPORT, RECORDING_FILE, COMMAND_LOG,
            "topology.json"} == RUN_FOLDER_NAMES
    out = tmp_path / "run"
    assert run(saved, out, "--history") == 0
    assert {path.name for path in out.iterdir()} <= RUN_FOLDER_NAMES and foreign_entries(out) == ()


# -- --force ----------------------------------------------------------------------------------------------------------


def test_force_runs_into_the_folder_warns_first_says_when_it_writes_and_keeps_every_file_not_a_runs(saved, tmp_path,
                                                                                                   capfd):
    out = ux1_folder(tmp_path)
    kept = {name: data for name, data in files(out).items() if name != RUN_MANIFEST}
    assert run(saved, out, "--force") == 0
    err = capfd.readouterr().err
    warning, notice = err.index("warning: --force"), err.index(f"writing into {out} now")
    assert warning < notice, "the warning first, the notice when writing begins"
    assert "mine holds experiment/, outputs.json" in err and "Ctrl+C cancels it until it begins to write" in err
    assert "Ctrl+C no longer cancels" in err
    after = files(out)
    assert all(after[name] == data for name, data in kept.items()), "a file that is not a run's is kept byte for byte"
    assert RunResult.read(out).steps == 4 and after[RUN_MANIFEST] != b'{"mine": true}\n', "a run's own name is replaced"


def test_force_into_an_empty_or_own_folder_says_nothing(saved, tmp_path, capfd):
    assert run(saved, tmp_path / "new", "--force") == 0 and run(saved, tmp_path / "new", "--force") == 0
    assert "warning" not in capfd.readouterr().err


def test_force_is_a_flag_of_run_and_listed_in_its_help(capsys):
    with pytest.raises(SystemExit):
        mime_main(["run", "--help"])
    assert "--force" in capsys.readouterr().out


# -- Ctrl+C before writing starts cancels ------------------------------------------------------------------------------


def press_after(seconds: float, monkeypatch) -> None:
    """A real SIGINT raised ``seconds`` after the run is under way."""
    started = Interrupt.running

    def running(self: Interrupt) -> None:
        started(self)
        timer = threading.Timer(seconds, signal.raise_signal, args=(signal.SIGINT,))
        timer.daemon = True
        timer.start()

    monkeypatch.setattr(Interrupt, "running", running)


def test_ctrl_c_during_a_forced_run_cancels_it_and_writes_nothing(saved, tmp_path, monkeypatch, capfd):
    out = ux1_folder(tmp_path)
    before = state_of(out)
    press_after(1.5, monkeypatch)
    steps = 64 * 4 * 10**6
    with untouched(out):
        assert run(saved, out, "--force", steps=steps) == INTERRUPTED
    captured = capfd.readouterr()
    assert captured.err.count("Ctrl+C: cancelling at the end of this chunk, and nothing will be written") == 1
    assert f"cancelled: the run ended without writing anything; {out} is as it was" in captured.out
    assert "writing into" not in captured.err
    assert state_of(out) == before


def test_control_the_same_press_on_a_run_that_is_not_forced_stops_it_and_writes_what_it_reached(saved, tmp_path,
                                                                                                 monkeypatch, capfd):
    """Without ``--force`` Ctrl+C is as it was: the run stops at its chunk and is written, marked stopped."""
    out = tmp_path / "run"
    press_after(1.5, monkeypatch)
    steps = 64 * 4 * 10**6
    assert run(saved, out, steps=steps) == INTERRUPTED
    assert "Ctrl+C: stopping at the end of this chunk" in capfd.readouterr().err
    assert RunResult.read(out).stopped is not None


def test_a_press_that_arrives_after_the_run_and_before_the_first_file_still_cancels(saved, tmp_path, capfd):
    """The window the run's last chunk leaves: the press is made while the run ends, and the cancel is judged at the
    moment writing would begin, where nothing has been written."""
    built = Experiment.load(saved).build()
    out = ux1_folder(tmp_path)

    def cancel() -> None:
        raise KeyboardInterrupt

    with untouched(out), pytest.raises(KeyboardInterrupt):
        built.run_to(out, s(4 * dt_of(saved)), force=True, before_writing=cancel)
    seen: list[dict[str, bytes]] = []
    built.run_to(out, s(4 * dt_of(saved)), force=True, before_writing=lambda: seen.append(files(out)))
    assert len(seen) == 1 and RUN_OUTPUTS not in seen[0], "called before anything of the run is in the folder"


def test_a_press_on_a_forced_run_after_writing_began_says_it_can_no_longer_cancel(saved, tmp_path, monkeypatch):
    said: list[str] = []
    interrupt = Interrupt(leave=lambda code: None, say=said.append)
    interrupt.cancels()
    interrupt.writing()
    interrupt._pressed()
    assert said == ["Ctrl+C: the run is being written and can no longer be cancelled; press Ctrl+C again to quit now, "
                    "which can leave the folder half written"]


# -- the amber notice -------------------------------------------------------------------------------------------------


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_an_amber_line_is_amber_on_a_terminal_and_plain_anywhere_else(monkeypatch):
    from mime.cli._display import AMBER, amber_line

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "200")
    terminal, plain = Terminal(), io.StringIO()
    amber_line(terminal, "writing now")
    amber_line(plain, "writing now")
    assert "38;5;214" in terminal.getvalue() and "writing now" in terminal.getvalue()
    assert plain.getvalue() == "writing now\n" and AMBER == "color(214)"
