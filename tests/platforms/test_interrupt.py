"""The machinery of Ctrl+C on ``mime run`` (``mime._interrupt``) in this process, on every platform: the signal is
raised with ``signal.raise_signal``, which Windows too runs through Python's own handler and wakeup descriptor, and the
subprocess tests of the same behaviour (``tests/env/test_run_interrupt.py``) need a child to send SIGINT to and so run
on POSIX only. What is not exercised here: a Windows console's CTRL_C_EVENT, and a terminal's keypress.

The first press is acknowledged by a thread at once and asks a run to stop at its next chunk; the second leaves
(``leave`` stands in for ``os._exit``); before the run, a press raises ``KeyboardInterrupt`` in the main thread and the
command ends 130, writing nothing; the handlers are put back. Through ``main``, a run stopped by a real signal is a run
folder of the steps it reached, marked stopped, and a run quit while it is written is not a folder that reads."""

import json
import signal
import threading
import time
from pathlib import Path

import jax
import pytest

from mime.cli import main as mime_main
from mime._interrupt import INTERRUPTED, USER_STOP, Interrupt
from mime.control._run import automatic_chunks
from mime.experiment import Experiment
from mime.results import RunResult
from mime.saved import read_folder
from mime.testing.toys import full_toy
from mime.units import s


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


def wait_until(condition, bound: float = 30.0) -> None:
    deadline = time.monotonic() + bound
    while not condition():
        assert time.monotonic() < deadline, "the watcher thread did not act"
        time.sleep(0.005)


def test_the_first_press_is_acknowledged_and_asks_for_a_stop_the_second_leaves():
    said: list[str] = []
    left: list[int] = []
    with Interrupt(leave=left.append, say=said.append) as interrupt:
        assert interrupt.stop() is None
        interrupt.running()
        signal.raise_signal(signal.SIGINT)
        wait_until(lambda: said)
        assert interrupt.stop() == USER_STOP and interrupt.pressed
        assert len(said) == 1 and said[0].startswith("Ctrl+C: stopping at the end of this chunk") and not left
        signal.raise_signal(signal.SIGINT)
        wait_until(lambda: left)
    assert left == [INTERRUPTED] and said[-1] == "Ctrl+C again: quitting now, nothing written"


def test_before_the_run_a_press_raises_in_the_main_thread_and_the_handlers_are_put_back():
    before = signal.getsignal(signal.SIGINT)
    said: list[str] = []
    with Interrupt(leave=lambda code: None, say=said.append):
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
            time.sleep(10)
        wait_until(lambda: said)
        assert said == ["Ctrl+C: stopping; press Ctrl+C again to quit now"]
    assert signal.getsignal(signal.SIGINT) is before
    assert signal.set_wakeup_fd(-1) == -1  # none left behind


def test_off_the_main_thread_it_does_nothing_and_raises_nothing():
    failures: list[BaseException] = []

    def inside() -> None:
        try:
            with Interrupt() as interrupt:
                assert interrupt.stop() is None
        except BaseException as error:  # noqa: BLE001 - the thread's outcome is what is asserted
            failures.append(error)

    thread = threading.Thread(target=inside)
    thread.start()
    thread.join()
    assert failures == []


def test_ctrl_c_in_process_stops_the_run_at_a_chunk_and_main_returns_130(saved, tmp_path, monkeypatch, capfd):
    """The whole path through ``main`` with a real signal raised from a timer once the run is under way."""
    out = tmp_path / "run"
    started = Interrupt.running

    def running(self: Interrupt) -> None:
        started(self)
        timer = threading.Timer(1.5, signal.raise_signal, args=(signal.SIGINT,))
        timer.daemon = True
        timer.start()

    monkeypatch.setattr(Interrupt, "running", running)
    steps = 10**9
    assert mime_main(["run", str(saved), "--duration", repr(steps * dt_of(saved)), "--out", str(out)]) == INTERRUPTED
    captured = capfd.readouterr()
    assert captured.err.count("Ctrl+C: stopping at the end of this chunk") == 1
    result = RunResult.read(out)
    assert result.stopped is not None and result.stopped.steps_asked == steps and 0 < result.steps < steps
    assert result.steps % automatic_chunks(steps)[0] == 0
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["stopped"]["steps_reached"] == result.steps


def test_ctrl_c_while_the_experiment_builds_ends_the_command_130_and_writes_nothing(saved, tmp_path, monkeypatch, capfd):
    def building(self):
        signal.raise_signal(signal.SIGINT)
        time.sleep(30)  # the handler raises in this thread long before
        raise AssertionError("the press did not interrupt the build")

    monkeypatch.setattr(Experiment, "build", building)
    out = tmp_path / "run"
    assert mime_main(["run", str(saved), "--duration", "1", "--out", str(out)]) == INTERRUPTED
    assert "interrupted before the run began; nothing was written" in capfd.readouterr().out and not out.exists()
    assert mime_main(["verify", str(saved)]) == INTERRUPTED


def test_a_run_quit_while_it_is_written_is_not_a_folder_that_reads(saved, tmp_path, monkeypatch):
    """``outputs.json`` is the last file of a run folder, so that the second press while the saved form is compiled and
    written (the longest part of the write) leaves nothing ``RunResult.read`` takes for a run."""
    from mime.experiment import BuiltExperiment
    from mime.results import NotARunFolder

    built = Experiment.load(saved).build()
    monkeypatch.setattr(BuiltExperiment, "topology", lambda self: (_ for _ in ()).throw(KeyboardInterrupt))
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        built.run_to(out, s(10 * dt_of(saved)))
    assert (out / "experiment").exists() and (out / "outputs.npz").exists() and not (out / "outputs.json").exists()
    with pytest.raises(NotARunFolder):
        RunResult.read(out)


