"""A warning a public call raises shows the script's own line, not MIME's: ``Experiment.build``, ``Experiment.save`` and
``Experiment.load`` (which build inside themselves), and the view's output warning through ``RunWritten.view``.

A0 declares an arm whose description holds elements the build reads past, so that its build warns
(:class:`DescriptionWarning`); the ball is the cheap rig for the view's. ``BuiltExperiment.save`` and ``run_to`` build
nothing and warn of nothing, which the last test holds to, and under ``mime run`` (a subprocess of the installed console
script, where no frame is the script's) a range warning is a ``mime: warning:`` line with no source.

The control is the old fixed stack level put back (``warnings.warn(..., stacklevel=3)`` where the build's wrapper and
``build`` stand between the warning and a ``save``'s caller): the same assertions then name ``_experiment.py`` for
``save`` and ``load``, so a test that passed either way would show here."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
from chain_cases import kinematic_rpm
from gltf_cases import ball
from range_cases import TURNING, rotating_screw

import mime.experiment._experiment as experiment_module
import mime.viz._web as web_module
from mime.core.multibody import DescriptionWarning
from mime.experiment import Experiment
from mime.units import ms


def files_warned_from(call) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    warned = [w for w in caught if issubclass(w.category, (DescriptionWarning, UserWarning))]
    assert warned, "the call warned of nothing, so there is no line to look at"
    return sorted({w.filename for w in warned})


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("described") / "saved"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kinematic_rpm("described").save(folder, steps=2)
    return folder


def test_every_public_call_that_builds_warns_at_the_scripts_line(saved, tmp_path):
    assert files_warned_from(lambda: kinematic_rpm("described").build()) == [__file__]
    assert files_warned_from(lambda: kinematic_rpm("described").save(tmp_path / "again", steps=2)) == [__file__]
    assert files_warned_from(lambda: Experiment.load(saved)) == [__file__]


def test_control_the_old_fixed_stack_level_names_mimes_own_line_for_save_and_load(saved, tmp_path, monkeypatch):
    def at_the_old_level(text, category):
        warnings.warn(text, category, stacklevel=4)  # one more than the old 3: this function is a frame of its own

    monkeypatch.setattr(experiment_module, "warn_at_script", at_the_old_level)
    assert files_warned_from(lambda: kinematic_rpm("described").build()) == [__file__], "the old level was right here"
    assert files_warned_from(lambda: kinematic_rpm("described").save(tmp_path / "old", steps=2)) \
        == [experiment_module.__file__]
    assert files_warned_from(lambda: Experiment.load(saved)) == [experiment_module.__file__]


def test_the_views_output_warning_through_a_run_written_names_the_scripts_line(tmp_path, monkeypatch):
    built = ball().build()
    written = built.run_to(tmp_path / "run", ms(2), stage_every=ms(1))
    monkeypatch.setattr(web_module, "OUTPUT_WARNING_BYTES", 0)
    assert files_warned_from(lambda: written.view()) == [__file__]


def test_a_built_experiments_save_and_run_to_build_nothing_so_nothing_is_warned_from_them(tmp_path, monkeypatch):
    """Absence, held so that it can fail: the warning helper is wrapped to count its calls. ``Experiment.build`` calls it
    (the control, the first build here: the count is above zero), and a built experiment's ``save`` and ``run_to`` must
    not, as they would if either rebuilt. The control builds and does not save: the helper is called by the build, and
    a save of an experiment that had not been built would only compile the run once more (this test took 20 to 31 s on
    CI's cold runners with that second compile)."""
    calls: list[str] = []
    real = experiment_module.warn_at_script

    def counting(text, category):
        calls.append(text)
        real(text, category)

    monkeypatch.setattr(experiment_module, "warn_at_script", counting)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = kinematic_rpm("described").build()
    assert calls, "control: a build calls the helper"
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built.save(tmp_path / "saved", steps=2)
        built.run_to(tmp_path / "run", ms(2))
    assert calls == []


# Under ``mime run`` no frame is the script's: the stack is MIME's and the console script's stub. A range warning is
# shown as ``mime: warning: <text>``, once, with no source line, as a description's is.


@pytest.fixture(scope="module")
def turning_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("turning") / "saved"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rotating_screw().save(folder, steps=2)
    return folder


def console_script() -> str:
    found = shutil.which("mime", path=str(Path(sys.executable).parent))
    assert found is not None, "the console script is installed beside the interpreter, as CI installs it"
    return found


def run_in_a_process(command: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=900,
                          env={**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
                               "XDG_CACHE_HOME": str(tmp_path / ".cache"), "JAX_PLATFORMS": "cpu"})


def turning_lines(done: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in done.stderr.splitlines() if TURNING in line]


def test_under_mime_run_a_range_warning_is_a_line_with_no_source_and_not_the_console_scripts(turning_folder, tmp_path):
    done = run_in_a_process([console_script(), "run", str(turning_folder), "--duration", "0.01",
                             "--out", str(tmp_path / "run")], tmp_path)
    assert done.returncode == 0, done.stderr
    (line,) = turning_lines(done)
    assert line.startswith("mime: warning: "), line
    assert "UserWarning" not in done.stderr and "sys.exit(main())" not in done.stderr
    assert console_script() not in done.stderr, "no line of the console script's stub"


def test_control_the_same_warning_from_a_script_that_calls_the_command_names_that_script(turning_folder, tmp_path):
    """The assertions above fail for this run: a script of the user's that calls ``mime.cli.main`` is the script, and
    the warning keeps the usual form with its file and line."""
    script = tmp_path / "caller.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        from mime.cli import main
        sys.exit(main(["run", {str(turning_folder)!r}, "--duration", "0.01", "--out", {str(tmp_path / "run")!r}]))
        """), encoding="utf-8")
    done = run_in_a_process([sys.executable, str(script)], tmp_path)
    assert done.returncode == 0, done.stderr
    lines = turning_lines(done)
    assert lines and all(line.startswith(f"{script}:") and "UserWarning" in line for line in lines), done.stderr
