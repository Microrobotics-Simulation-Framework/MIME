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


# -- (u) mime assets check always ends with a total ------------------------------------------------------------------


def test_assets_check_ends_with_a_total_even_when_nothing_is_cached(capsys, monkeypatch):
    """With nothing cached it printed nothing at all, which reads as a command that did not run."""
    from types import SimpleNamespace

    from mime import assets

    monkeypatch.setattr(assets, "known", lambda: {name: SimpleNamespace(name=name) for name in ("a", "b", "c")})
    outcomes = {"a": "not cached", "b": "not cached", "c": "not cached"}
    monkeypatch.setattr(assets, "check", lambda ref: outcomes[ref.name])
    assert mime_main(["assets", "check"]) == 0
    assert capsys.readouterr().out == "3 assets: 0 ok, 0 corrupt, 3 not cached (nothing to hash)\n"
    outcomes.update(a="ok", b="corrupt")
    assert mime_main(["assets", "check"]) == 1
    assert capsys.readouterr().out.splitlines() == ["a  ok", "b  corrupt",
                                                    "3 assets: 1 ok, 1 corrupt, 1 not cached (nothing to hash)"]


# -- (v) the AR4's manifest is shipped, references only ----------------------------------------------------------------


def test_the_shipped_visual_set_holds_references_and_no_mesh_bytes():
    """Confirmed before shipping (#198, the tiers): every record is a name, a sha256, a size, an upstream address and
    the terms (redistribution not permitted), the file is a few kilobytes beside meshes of megabytes, and nothing but
    JSON is in its folder. It is the example description's own manifest, byte for byte."""
    import importlib.resources
    import json

    root = importlib.resources.files("mime.data") / "visual_sets"
    names = sorted(each.name for each in root.iterdir() if each.name != "__pycache__")
    assert names == ["ar4_mk5.json"]
    shipped = (root / "ar4_mk5.json").read_bytes()
    assert len(shipped) < 50_000 and shipped == (Path(__file__).resolve().parents[2] / "examples" / "descriptions" / "ar4"
                                                  / "visuals.json").read_bytes()
    held = json.loads(shipped)
    assert held["licences"]["LicenseRef-Annin-Robotics-NonCommercial"]["redistribution"] == "not_permitted"
    for record in held["assets"]:
        assert set(record) <= {"name", "sha256", "size", "licence", "locations", "role", "kind"}, record
        assert record["size"] > 10_000
        assert all(location.startswith("https://raw.githubusercontent.com/Annin-Robotics/") for location in
                   record["locations"])
    assert len(json.dumps(held["assets"])) < 20_000 < sum(record["size"] for record in held["assets"]) / 1000, \
        "the records are a thousandth of the meshes they name: the bytes are at their upstream"


def test_visuals_list_names_the_shipped_set_with_nothing_installed_and_fetch_knows_it_by_name(capsys):
    from mime.viz import manifests_for

    assert mime_main(["visuals", "list"]) == 0
    assert "ar4_mk5  18 files" in capsys.readouterr().out
    assert [held.name for held in manifests_for("ar4_mk5")] == ["ar4_mk5"]
    from mime.scene import visual_sets

    assert "ar4_mk5" not in visual_sets(), "a description still finds its meshes' records beside it, never in MIME's data"


# -- (w) mime modal --gpu from an experiment saved without the highest matmul precision -------------------------------


@pytest.mark.parametrize("gpu, article", [("A100-40GB", "an"), ("H100", "an"), ("L4", "an"), ("T4", "a")])
def test_the_gpu_precision_refusal_has_its_grammar_and_says_how_to_save_for_a_gpu(gpu, article):
    from mime.env._check import gpu_precision_refusals

    (refusal,) = gpu_precision_refusals(None, gpu)
    assert f"a run on {article} {gpu} computes" in refusal.message
    assert "None (a CPU run records none)" in refusal.message
    assert "JAX_DEFAULT_MATMUL_PRECISION=highest python <script>.py" in refusal.action
    assert gpu_precision_refusals("highest", gpu) == [] and gpu_precision_refusals(None, None) == []


# -- (x) scaffold --yes chooses no licence and no namespace --------------------------------------------------------


def test_yes_writes_marked_placeholders_where_it_was_given_no_licence_or_namespace_and_check_refuses_them(tmp_path, capsys):
    """``--yes`` took CC-BY-4.0 and the namespace ``mylab`` from "scaffold's own" while its help said it accepts no
    licence: a block now carries a marker in each place, and ``mime check`` says what to choose."""
    block = tmp_path / "lb"
    assert mime_main(["scaffold", "block", str(block), "--yes"]) == 0
    text = (block / "mime.toml").read_text(encoding="utf-8")
    assert "LicenseRef-CHOOSE-A-LICENCE" in text and "choose_a_namespace.lb" in text
    assert "CC-BY-4.0" not in text and "mylab" not in text
    capsys.readouterr()
    assert mime_main(["check", str(block)]) == 1
    said = capsys.readouterr()
    assert "placeholder" in said.out + said.err and "choose" in said.out + said.err


def test_control_the_same_scaffold_with_a_licence_and_an_id_given_passes_its_check(tmp_path, capsys):
    block = tmp_path / "lb"
    assert mime_main(["scaffold", "block", str(block), "--yes", "--set", "block.id=labx.lb", "--set",
                      "block.licence=CC-BY-4.0"]) == 0
    capsys.readouterr()
    assert mime_main(["check", str(block)]) == 0


def test_an_experiment_scaffolded_with_yes_declares_no_licence_and_says_where_to_choose_one(tmp_path):
    folder = tmp_path / "ex"
    assert mime_main(["scaffold", "experiment", str(folder), "--yes"]) == 0
    text = (folder / "experiment.py").read_text(encoding="utf-8")
    assert "# experiment.licence(" in text and "CC-BY-4.0\")" not in text.replace("such as CC-BY-4.0 or MIT", "")
    assert not any(line.strip().startswith("experiment.licence(") for line in text.splitlines())


def test_the_placeholders_are_refused_wherever_a_licence_or_a_namespace_is_read():
    from mime.blocks._definition import id_problem
    from mime.push._defaults import namespace_problem
    from mime.terms import PLACEHOLDER_LICENCE, PLACEHOLDER_NAMESPACE, licence_problem

    assert "placeholder" in (licence_problem(PLACEHOLDER_LICENCE) or "")
    assert "placeholder" in (namespace_problem(PLACEHOLDER_NAMESPACE) or "")
    assert "placeholder" in (id_problem(f"{PLACEHOLDER_NAMESPACE}.probe") or "")
    assert licence_problem("LicenseRef-Mine") is None and id_problem("labx.probe") is None


# -- (ab) verify ends with an acceptance paragraph for an experiment, a run, a study and a block alike --------------------


def test_verify_of_an_experiment_ends_with_an_acceptance_as_a_study_and_a_block_do(saved, capsys):
    from mime.blocks._card import ACCEPTANCE as BLOCK
    from mime.saved._studies import ACCEPTANCE as STUDY

    assert mime_main(["verify", str(saved)]) == 0
    last = capsys.readouterr().out.rstrip().splitlines()[-1]
    assert last.startswith("An acceptance: the experiment's saved form loads and rebuilds here")
    for other in (BLOCK, STUDY):
        assert other.startswith("An acceptance: ") and "It bounds nothing about physics" in other
        assert "It bounds nothing about physics" in last


# -- (ac) every command's help documents its exit codes ----------------------------------------------------------------


def test_every_command_without_commands_under_it_documents_its_exit_codes_in_its_help(capsys):
    from mime.cli import _guide, build_parser

    parser = build_parser()
    paths = _guide.leaf_paths(parser)
    leaves = [path for path in paths if not any(other[:len(path)] == path and len(other) > len(path) for other in paths)]
    assert len(leaves) > 30
    assert [path for path in leaves if path not in _guide.EXIT_CODES] == []
    for path in leaves:
        child = parser
        for word in path:
            child = _guide._children(child)[word]  # pyright: ignore[reportPrivateUsage]
        text = child.format_help()
        assert "exit codes:" in text and "refused: the cause and what to do are printed" in text, path
        for code, meaning in _guide.EXIT_CODES[path]:
            assert f"  {code:<5}{meaning}" in text, (path, code)


def test_the_top_of_the_help_lists_the_codes_every_command_may_give(capsys):
    with pytest.raises(SystemExit):
        mime_main(["--help"])
    out = capsys.readouterr().out
    assert "2 refused; 3 a MIME defect; 130 Ctrl+C; 141 the reader of a pipe closed it" in out
