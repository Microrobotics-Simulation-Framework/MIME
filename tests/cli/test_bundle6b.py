"""CLI bundle 6b, the rest of what the CLI tour found, each with the test that fails without the fix.

(p) a group of commands named alone, (q) ``--traceback``'s order, (r) and (s) an override's names and units read from the
document, once, (t) ``mime push --init`` with no service, (y) and (z) one frame for a mistake in the command line."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from mime.cli import main as mime_main
from mime.testing.toys import full_toy


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("toy") / "ball"
    full_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    held = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", held)


# -- (y) (z) one frame for a mistake in the command line --------------------------------------------------------------


@pytest.mark.parametrize("words", [["run", "x", "--duration", "3ms"], ["show"], ["run", "x", "--out", "y", "--bogus"],
                                   ["bogus"], ["credentials", "set", "nowhere"], ["push", "--defaults"]],
                         ids=lambda words: " ".join(words))
def test_a_mistake_in_the_command_line_is_a_coded_refusal_with_what_to_do_and_no_usage_dump(capsys, words):
    """argparse's frame (a usage block and a bare line) and the credentials command's own were two more styles of error
    beside the coded refusal: each is one, with a code, a ``do:`` and a ``see:``."""
    assert mime_main(words) == 2
    err = capsys.readouterr().err
    assert "cli.usage" in err and "    do: " in err and "    see: " in err
    assert not any(line.startswith("usage:") or ": error: " in line for line in err.splitlines())
    assert "The full traceback" not in err, "a mistake in the line has no traceback to point to"


def test_the_refusal_of_a_mistake_names_the_commands_own_help_and_its_example(capsys):
    assert mime_main(["show"]) == 2
    err = capsys.readouterr().err
    assert "`mime show --help`" in err and "mime show my_experiment --graph" in err


# -- (p) a group of commands given no command -------------------------------------------------------------------------


@pytest.mark.parametrize("group, commands", [("study", ("plan", "run")), ("assets", ("list", "fetch", "sources")),
                                             ("env", ("check",))])
def test_a_group_named_alone_lists_its_commands_and_not_the_parsers_name_for_them(capsys, group, commands):
    assert mime_main([group]) == 2
    err = capsys.readouterr().err
    assert "_command" not in err and "arguments are required" not in err
    assert all(f"`mime {group} {name}`" in err for name in commands)


# -- (q) --traceback: the refusal, the traceback, the refusal's line ------------------------------------------------


def test_traceback_shows_the_refusal_first_and_its_line_again_last(saved, capsys, tmp_path):
    assert mime_main(["--traceback", "run", str(saved), "--duration", "3", "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    first, traceback, last = err.index("run.duration"), err.index("Traceback (most recent call last)"), \
        err.rstrip().splitlines()[-1]
    assert first < traceback, "the refusal is above the traceback"
    assert last.startswith("mime run is refused: run.duration: --duration 3 is a bare number")


# -- (r) (s) an override's names and units are read from the document, before the load, and refused once -------------


@pytest.fixture
def loading_is_forbidden(monkeypatch):
    from mime.experiment import Experiment

    def load(*_args, **_kwargs):
        raise AssertionError("the experiment was loaded")

    monkeypatch.setattr(Experiment, "load", load)


@pytest.mark.parametrize("override, code", [("bath.viscosty=2mPa_s", "value.unknown"),
                                            ("bath.viscosity=2", "value.units"),
                                            ("bth.viscosity=2mPa_s", "study.unknown-path")])
def test_an_override_with_a_wrong_name_or_unit_is_refused_before_anything_loads(saved, capsys, tmp_path,
                                                                                loading_is_forbidden, override, code):
    assert mime_main(["run", str(saved), "--duration", "0.1s", "--out", str(tmp_path / "o"), "--override", override]) == 2
    err = capsys.readouterr().err
    assert code in err and "running " not in err and not (tmp_path / "o").exists()


def test_control_a_right_override_is_not_refused_by_the_check_on_the_document(saved):
    from mime.study._override import parse_override, refuse_unwritable

    refuse_unwritable(saved / "experiment.mime.yaml", [parse_override("bath.viscosity=2mPa_s")])


def test_a_declared_value_in_the_wrong_unit_is_refused_once_not_again_as_undeclared():
    """``water.viscosity`` declared as a density gave ``value.units`` and, for the same cause, ``value.undeclared``."""
    from mime import Body, Experiment, Medium, State
    from mime.build import BuildRefused
    from mime.dynamics import RigidInertial
    from mime.interactions import Drag, Gravity
    from mime.scene import Scene, Sphere, Z
    from mime.things import Part
    from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

    for wrong in (kg_per_m3(2.0), 2.0):
        exp = Experiment("settling_ball", scene=Scene.new(up=Z, units="m"), dt=ms(0.1))
        water = exp.add_medium(Medium("water"))
        exp.declare("water.density", kg_per_m3(998.0))
        exp.declare("water.viscosity", wrong)
        ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
            position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
            angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(10.0)))}))
        exp.declare("ball.parts.steel.density", kg_per_m3(7850.0))
        exp.attach(Drag.MRS(), body=ball, medium=water)
        exp.attach(Gravity(), body=ball, medium=water)
        with pytest.raises(BuildRefused) as refused:
            exp.plan()
        assert refused.value.codes() == ["value.units"]


# -- (t) mime push --init needs no service; push --defaults is gone -------------------------------------------------


def test_push_init_writes_the_manifest_with_no_service_and_names_the_choice_for_the_next_step(saved, tmp_path, capsys):
    import shutil

    folder = tmp_path / "ball"
    shutil.copytree(saved, folder)
    assert mime_main(["push", str(folder), "--init"]) == 0
    out = capsys.readouterr().out
    assert "wrote " in out and "--to <zenodo|zenodo-sandbox>" in out
    assert mime_main(["push", str(folder), "--status"]) == 2, "every other mode still needs --to"
    assert "--to" in capsys.readouterr().err


def test_push_defaults_is_gone_and_mime_defaults_is_the_one_spelling(capsys):
    assert mime_main(["push", "--defaults"]) == 2
    assert "cli.usage" in capsys.readouterr().err
    assert mime_main(["defaults"]) == 0
