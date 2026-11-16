"""Where a run may be written (UX-1's F2): ``mime run --out`` and ``BuiltExperiment.run_to`` wrote into any folder, and
replaced the files of a run's own names in it (``outputs.json``, ``run.json``, ``experiment/``'s saved form) with nothing
said: a user's folder with an ``outputs.json`` of its own lost it. A run never changes a file that is not its own now.

- ``out`` that is new or empty is written into; MIME's own run folder (``run.json``, and only files of the names a run
  writes) is too from Python (``BuiltExperiment.run_to``: a re-run into its own folder is the script's habit), but
  ``mime run`` refuses it like any other, and runs into it with ``--force`` (bundle 6: no silent overwrite anywhere);
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
import shutil
import signal
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


def test_a_new_folder_and_an_empty_one_are_written_into_and_a_second_run_over_the_first_needs_force(saved, tmp_path,
                                                                                                  capsys):
    assert run(saved, tmp_path / "new") == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(saved, empty) == 0
    assert "warning" not in capsys.readouterr().err
    with untouched(empty):
        assert run(saved, empty, steps=5) == 2, "mime run never replaces a run silently (his rule, item 81)"
    err = capsys.readouterr().err
    assert CODE in err and "a MIME run folder already" in err and "--force" in err
    assert run(saved, empty, "--force", steps=5) == 0
    assert RunResult.read(empty).steps == 5
    assert "warning: --force" in capsys.readouterr().err


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


def test_force_into_a_new_or_empty_folder_says_nothing_and_over_a_run_it_warns(saved, tmp_path, capfd):
    assert run(saved, tmp_path / "new", "--force") == 0
    assert "warning" not in capfd.readouterr().err
    assert run(saved, tmp_path / "new", "--force") == 0
    assert "warning: --force" in capfd.readouterr().err, "a second run replaces the first: said"


def test_force_is_a_flag_of_run_and_listed_in_its_help(capsys):
    with pytest.raises(SystemExit):
        mime_main(["run", "--help"])
    assert "--force" in capsys.readouterr().out


# -- Ctrl+C before writing starts cancels ------------------------------------------------------------------------------


def press_at_first_chunk(monkeypatch) -> list[int]:
    """A real SIGINT raised by the run itself at a known point, the end of its first chunk, where the run asks whether
    to stop: the press is made there and waited for until the interrupt's watcher has taken it (``Interrupt.pressed``),
    so the run's next question is answered by it on every run, whatever the machine's speed. The list is the chunks
    the run asked about (a press that was never made leaves it empty)."""
    asked: list[int] = []
    asks = Interrupt.stop

    def stop(self: Interrupt, progress: object = None) -> str | None:
        asked.append(len(asked))
        if len(asked) == 1:
            signal.raise_signal(signal.SIGINT)
            assert self._first.wait(timeout=60), "the press was not taken by the interrupt's watcher"
        return asks(self, progress)

    monkeypatch.setattr(Interrupt, "stop", stop)
    return asked


def test_ctrl_c_during_a_forced_run_cancels_it_and_writes_nothing(saved, tmp_path, monkeypatch, capfd):
    out = ux1_folder(tmp_path)
    before = state_of(out)
    asked = press_at_first_chunk(monkeypatch)
    with untouched(out):
        assert run(saved, out, "--force", steps=8) == INTERRUPTED
    captured = capfd.readouterr()
    assert asked == [0], "the run was ended at the chunk the press was made at, not run to its end"
    assert captured.err.count("Ctrl+C: cancelling at the end of this chunk, and nothing will be written") == 1
    assert f"cancelled: the run ended without writing anything; {out} is as it was" in captured.out
    assert "writing into" not in captured.err
    assert state_of(out) == before


def test_control_the_same_press_on_a_run_that_is_not_forced_stops_it_and_writes_what_it_reached(saved, tmp_path,
                                                                                                 monkeypatch, capfd):
    """Without ``--force`` Ctrl+C is as it was: the run stops at its chunk and is written, marked stopped."""
    out = tmp_path / "run"
    asked = press_at_first_chunk(monkeypatch)
    assert run(saved, out, steps=8) == INTERRUPTED
    assert asked == [0] and "Ctrl+C: stopping at the end of this chunk" in capfd.readouterr().err
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


# -- what the runner leaves, and what a run reads ---------------------------------------------------------------------


def test_a_folder_holding_only_the_runners_environment_check_is_written_into_and_a_stranger_beside_it_is_not(saved, tmp_path,
                                                                                                         capsys):
    """The Modal runner writes ``env-check.json`` into the run's folder before it runs (``mime env check --record``): that
    folder is a run's own, and a run is written into it (the base did, and a test of the run on a Volume holds it)."""
    out = tmp_path / "volume" / "run"
    out.mkdir(parents=True)
    (out / "env-check.json").write_text("{}\n", encoding="utf-8")
    assert run(saved, out) == 0 and (out / "env-check.json").is_file() and RunResult.read(out).steps == 4
    assert foreign_entries_of(out) == ()
    beside = tmp_path / "beside"
    beside.mkdir()
    (beside / "env-check.json").write_text("{}\n", encoding="utf-8")
    (beside / "notes.txt").write_text("mine\n", encoding="utf-8")
    with untouched(beside):
        assert run(saved, beside) == 2
    assert CODE in capsys.readouterr().err


def foreign_entries_of(folder: Path):
    from mime.experiment._experiment import foreign_entries

    return foreign_entries(folder)


def test_writing_over_the_folder_the_run_reads_is_still_that_refusal_and_not_this_one(saved, capsys):
    with untouched(saved):
        assert run(saved, saved) == 2
    err = capsys.readouterr().err
    assert "run.out-over-source" in err and CODE not in err


# -- a folder that changes while the run computes (the review's mid-run hook) -----------------------------------------


def test_a_folder_that_became_a_stranger_while_the_run_computed_is_refused_where_writing_begins(saved, tmp_path):
    """``out`` was empty when the run started (the first look let it through), and a chunk hook wrote an ``outputs.json``
    of the user's into it: the base replaced it with nothing said. The folder is judged again where writing begins and
    refused (``run.out-not-empty``, naming the change), the file left byte for byte and nothing of the run written."""
    built = Experiment.load(saved).build()
    out = tmp_path / "out"

    def hook(chunk) -> None:
        out.mkdir(exist_ok=True)
        (out / RUN_MANIFEST).write_text("USER DATA", encoding="utf-8")

    dt = dt_of(saved)
    with pytest.raises(ArgumentRefused) as refused:
        built.run_to(out, s(4 * dt), hook=hook, chunk=s(2 * dt))
    assert refused.value.codes() == [CODE] and "changed while the run computed" in refused.value.refusals[0].message
    assert files(out) == {RUN_MANIFEST: b"USER DATA"}


def test_control_a_hook_that_writes_nothing_into_the_folder_leaves_the_run_written(saved, tmp_path):
    built = Experiment.load(saved).build()
    dt = dt_of(saved)
    written = built.run_to(tmp_path / "out", s(4 * dt), hook=lambda chunk: None, chunk=s(2 * dt))
    assert written.steps == 4 and RunResult.read(tmp_path / "out").steps == 4


def test_control_the_same_hook_under_force_is_run_into_as_asked(saved, tmp_path):
    built = Experiment.load(saved).build()
    out = tmp_path / "out"
    dt = dt_of(saved)

    def hook(chunk) -> None:
        out.mkdir(exist_ok=True)
        (out / "mine.txt").write_text("mine", encoding="utf-8")

    built.run_to(out, s(4 * dt), hook=hook, chunk=s(2 * dt), force=True)
    assert (out / "mine.txt").read_text(encoding="utf-8") == "mine" and RunResult.read(out).steps == 4


# -- "MIME's own" is judged by what a folder holds, not by its names --------------------------------------------------


def run_folder(saved: Path, root: Path) -> Path:
    out = root / "genuine"
    assert run(saved, out, "--history") == 0
    return out


def test_a_folder_holding_a_strangers_run_json_alone_is_not_mimes_own(saved, tmp_path, capsys):
    out = tmp_path / "o"
    out.mkdir()
    (out / RUN_RECORD).write_text('{"other tool": 1}', encoding="utf-8")
    with untouched(out):
        assert run(saved, out) == 2
    assert CODE in capsys.readouterr().err


def test_a_strangers_run_json_with_a_users_outputs_npz_is_not_mimes_own(saved, tmp_path, capsys):
    out = tmp_path / "o"
    out.mkdir()
    (out / RUN_RECORD).write_text('{"other tool": 1}', encoding="utf-8")
    (out / RUN_OUTPUTS).write_text("my data", encoding="utf-8")
    with untouched(out):
        assert run(saved, out) == 2
    assert (out / RUN_OUTPUTS).read_text(encoding="utf-8") == "my data" and CODE in capsys.readouterr().err


@pytest.mark.parametrize("name, data", [(RUN_OUTPUTS, "my data"), (RUN_MANIFEST, '{"mine": 1}'), (RUN_REPORT, "notes"),
                                        ("topology.json", "[1, 2]")])
def test_a_run_folder_whose_file_of_a_runs_name_is_not_what_a_run_writes_is_not_mimes_own(saved, tmp_path, capsys, name,
                                                                                      data):
    out = run_folder(saved, tmp_path)
    (out / name).write_text(data, encoding="utf-8")
    with untouched(out):
        assert run(saved, out) == 2
    assert CODE in capsys.readouterr().err


def test_control_a_genuine_run_folder_is_run_into_again_with_force_whatever_it_holds_of_a_runs_own(saved, tmp_path):
    out = run_folder(saved, tmp_path)
    assert run(saved, out, "--force") == 0 and run(saved, out, "--history", "--force") == 0
    assert RunResult.read(out).steps == 4


# -- links ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [(), ("--force",)], ids=["without --force", "with --force"])
def test_a_link_to_nowhere_as_out_is_a_refusal_before_the_run_computes(saved, tmp_path, capsys, monkeypatch, flags):
    """It was ``error.uncoded`` (exit 3), after the run: ``mkdir`` met the link."""
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "nowhere")
    monkeypatch.setattr(Experiment, "load", lambda folder: (_ for _ in ()).throw(AssertionError("loaded")))
    assert run(saved, link, *flags) == 2
    err = capsys.readouterr().err
    assert CODE in err and "a link to nowhere" in err and "error.uncoded" not in err
    assert link.is_symlink() and not (tmp_path / "nowhere").exists()


@pytest.mark.parametrize("name", [RUN_REPORT, RUN_EXPERIMENT])
def test_a_link_inside_a_run_folder_makes_it_not_mimes_own_and_is_never_written_through(saved, tmp_path, capsys, name):
    out = run_folder(saved, tmp_path)
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("precious", encoding="utf-8")
    (out / name).unlink() if (out / name).is_file() else shutil.rmtree(out / name)
    (out / name).symlink_to(precious if name == RUN_EXPERIMENT else precious / "keep.txt")
    before = files(precious)
    assert run(saved, out) == 2
    assert files(precious) == before and CODE in capsys.readouterr().err


def test_a_link_to_a_folder_of_the_users_as_out_is_judged_by_what_it_leads_to(saved, tmp_path):
    real = thesis_folder(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(real)
    with untouched(real):
        assert run(saved, link) == 2
