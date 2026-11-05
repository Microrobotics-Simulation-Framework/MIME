"""A warning a public call raises shows the script's own line, not MIME's: ``Experiment.build``, ``Experiment.save`` and
``Experiment.load`` (which build inside themselves), and the view's output warning through ``RunWritten.view``.

A0 declares an arm whose description holds elements the build reads past, so that its build warns
(:class:`DescriptionWarning`); the ball is the cheap rig for the view's. ``BuiltExperiment.save`` and ``run_to`` build
nothing and warn of nothing, which the last test holds to.

The control is the old fixed stack level put back (``warnings.warn(..., stacklevel=3)`` where the build's wrapper and
``build`` stand between the warning and a ``save``'s caller): the same assertions then name ``_experiment.py`` for
``save`` and ``load``, so a test that passed either way would show here."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from chain_cases import kinematic_rpm
from gltf_cases import ball

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


def test_a_built_experiments_save_and_run_to_build_nothing_and_warn_of_nothing(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = kinematic_rpm("described").build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built.save(tmp_path / "saved", steps=2)
        built.run_to(tmp_path / "run", ms(2))
    assert [str(w.message) for w in caught if issubclass(w.category, (DescriptionWarning, UserWarning))] == []
