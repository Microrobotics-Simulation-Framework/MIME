"""Where a live session writes its run folder (his item 106, part 1, found by #347's agent): two ``mime serve`` sessions
started in the same second in one folder shared the default ``<experiment>-live-<UTC second>``, and the second deleted
the first's ``experiment/`` and truncated its ``commands.jsonl``.

- The session opens its command log **exclusively, before it writes anything else** into the folder: a second session
  into a folder whose log a first holds is refused (``live.log-exists``), ``force`` or not, and the first's log and
  ``experiment/`` are byte for byte as they were. The race is planted by judging the folder empty, as a second session
  that looked before the first wrote would have.
- ``experiment/`` is written file by file, never deleted first: a forced folder's other files are kept, as a run's are.
- The default run folder is made by the session that takes it, ``-2``, ``-3`` and on after it for a second and a third
  in the same second.
- ``replay`` (``mime run --commands``) takes the same rule: never into a folder that holds a log."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pytest
from live_cases import files_of, saved

from mime.build import ArgumentRefused
from mime.experiment import Experiment
from mime.live import COMMAND_LOG, default_out, replay


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return saved(tmp_path_factory.mktemp("run_folder"))


@pytest.fixture(scope="module")
def loaded(folder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Experiment.load(folder).build()


def test_a_second_session_into_a_folder_a_first_holds_is_refused_and_the_first_keeps_its_log_and_experiment(
        loaded, tmp_path, monkeypatch):
    out = tmp_path / "run"
    first = loaded.live(pace=None, out=out)
    log, experiment = (out / COMMAND_LOG).read_bytes(), files_of(out / "experiment")
    with pytest.raises(ArgumentRefused, match="live.log-exists"):
        loaded.live(pace=None, out=out, force=True)
    # the race: a second session that judged the folder before the first wrote into it
    monkeypatch.setattr("mime.experiment._experiment.refuse_out_not_own", lambda *_, **__: None)
    with pytest.raises(ArgumentRefused, match="live.log-exists"):
        loaded.live(pace=None, out=out)
    monkeypatch.undo()
    assert (out / COMMAND_LOG).read_bytes() == log and files_of(out / "experiment") == experiment
    written = first.stop()
    assert written == out and (out / COMMAND_LOG).read_bytes().startswith(log)
    other = tmp_path / "other"  # a replay into a folder holding a log, forced, is refused as well
    other.mkdir()
    (other / COMMAND_LOG).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArgumentRefused, match="live.log-exists"):
        replay(loaded, out / COMMAND_LOG, other, force=True)
    assert (other / COMMAND_LOG).read_text(encoding="utf-8") == "{}\n"
    assert sorted(other.iterdir()) == [other / COMMAND_LOG]


def test_a_forced_folders_other_files_are_kept_and_its_experiment_written_file_by_file(loaded, tmp_path):
    out = tmp_path / "mine"
    (out / "experiment").mkdir(parents=True)
    (out / "experiment" / "notes.txt").write_text("my notes\n", encoding="utf-8")
    (out / "mine.txt").write_text("mine\n", encoding="utf-8")
    with pytest.raises(ArgumentRefused, match="run.out-not-empty"):
        loaded.live(pace=None, out=out)
    loaded.live(pace=None, out=out, force=True).stop()
    assert (out / "experiment" / "notes.txt").read_text(encoding="utf-8") == "my notes\n"
    assert (out / "mine.txt").read_text(encoding="utf-8") == "mine\n"
    assert (out / "experiment" / "mime.lock").is_file() and (out / COMMAND_LOG).is_file()


def test_the_default_run_folder_is_the_sessions_own_within_one_second(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    second = time.struct_time((2026, 11, 16, 12, 0, 0, 0, 320, 0))
    monkeypatch.setattr(time, "gmtime", lambda *_: second)
    folders = [default_out("rig") for _ in range(3)]
    assert [held.name for held in folders] == ["rig-live-20261116T120000Z", "rig-live-20261116T120000Z-2",
                                               "rig-live-20261116T120000Z-3"]
    assert all(held.is_dir() and not any(held.iterdir()) for held in folders)
