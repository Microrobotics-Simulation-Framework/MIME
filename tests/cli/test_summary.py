"""The end of ``mime run`` on a terminal leads with what needs attention and keeps the rest behind ``--full``, from the
same objects the files and a pipe hold in full. Run here with the terminal's drawing forced on a captured stream (Rich
writes plain text to one), so that what is said is read as text; ``test_terminal.py`` runs it on a real terminal.

**What this bounds.** That the summary names a post-run watch that is raised (the 50 um sphere's Reynolds number, past its
rung's range: ``examples/sphere_settling.py`` is the control for it) and says nothing needs attention for a run that has
none (the 10 um sphere). It does not say the model is right, only what MIME already flagged."""

import importlib.util
import sys
from pathlib import Path

import jax
import pytest

import mime.cli
from mime.cli import _summary
from mime.cli import main as mime_main

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("example_sphere_settling", ROOT / "examples" / "sphere_settling.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def folders(example, tmp_path_factory):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    root = tmp_path_factory.mktemp("spheres")
    for radius in (10.0, 50.0):
        example.sphere_settling(radius).save(root / f"r{int(radius)}")
    jax.config.update("jax_enable_x64", held)
    return root


@pytest.fixture(autouse=True)
def terminal_drawing(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(mime.cli, "styled", lambda stream: stream is not None)
    held = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", held)


def run(folders, name, duration, tmp_path, capsys, *more):
    assert mime_main(["run", str(folders / name), "--duration", duration, "--out", str(tmp_path / "run"), *more]) == 0
    return capsys.readouterr().out


def test_a_raised_post_run_watch_leads_the_summary_and_the_rest_is_behind_full(folders, tmp_path, capsys):
    out = run(folders, "r50", "75ms", tmp_path, capsys)
    lines = out.splitlines()
    assert lines[0].startswith("Needs attention (")
    assert any(line.startswith("  ! post-run watch raised: reynolds:drag:ball") and "ABOVE" in line for line in lines)
    assert lines.index(next(line for line in lines if line.startswith("ran sphere_settling"))) > 1
    assert "Readings" in out and "Next" in out and "load report:" not in out and "build report:" not in out
    assert "--full prints the load report, the build report's summary and every reading" in out
    assert str(tmp_path / "run" / "report.json") in out


def test_full_adds_the_load_report_the_build_reports_summary_and_the_whole_result(folders, tmp_path, capsys):
    out = run(folders, "r50", "75ms", tmp_path, capsys, "--full")
    for part in ("Load report", "load report: ", "input hash: reproduced exactly", "Build report", "build report: experiment",
                 "entries (", "refusals: none", "Result", "run of sphere_settling: 30 steps"):
        assert part in out, part


def test_a_run_with_nothing_raised_says_nothing_needs_attention(folders, tmp_path, capsys):
    out = run(folders, "r10", "3ms", tmp_path, capsys)
    assert out.startswith("Nothing needs attention"), out.splitlines()[0]
    assert "post-run watch raised" not in out and "ran sphere_settling" in out


def test_nothing_is_dropped_from_the_files_whatever_the_summary_leads_with(folders, tmp_path, capsys):
    import json

    run(folders, "r50", "75ms", tmp_path, capsys)
    manifest = json.loads((tmp_path / "run" / "outputs.json").read_text())
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert manifest["watches"], "the watch the summary led with is in the manifest, whole"
    assert report["summary"] and report["profile"]["profile"] == "exploratory"


#: Folders a user might have named oddly (or a pulled bundle's, unpacked under its own name): each is more than a word to a shell.
ODD_FOLDERS = ["my run", "x;touch${IFS}pwned;$(id)", "it's", '"q"', "$HOME", "a&b"]


def listed_by_a_shell(line: str, directory) -> tuple[list[str], bool]:
    """``line`` run by ``sh`` with ``mime`` and ``python`` standing for programs that list their arguments, one per line,
    and whether it made a file ``pwned``."""
    import subprocess

    standing_in = ("mime() { for word in \"$@\"; do printf '%s\\n' \"$word\"; done; }\n"
                   "python() { for word in \"$@\"; do printf '%s\\n' \"$word\"; done; }\n")
    done = subprocess.run(["sh", "-c", standing_in + line], capture_output=True, text=True, cwd=directory, check=False)
    return done.stdout.splitlines(), (directory / "pwned").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the lines are for a POSIX shell")
@pytest.mark.parametrize("name", ODD_FOLDERS)
def test_every_command_the_next_steps_offer_keeps_a_folder_s_name_one_word(name, tmp_path):
    folder, out = tmp_path / name, tmp_path / f"{name}_run"
    commands = [line.removeprefix("read the result:").removeprefix("check it again:").removeprefix("watch it live:").strip()
                for line in _summary.next_steps(folder, out) if not line.startswith("without MIME")]
    shell = tmp_path / "shell"
    shell.mkdir()
    heard = []
    for command in commands:
        words, ran = listed_by_a_shell(command, shell)
        assert not ran, command
        heard.append(words)
    read, verify, serve = heard
    assert read[-1] == str(out) and read[0] == "-c", "the result's folder is the program's one argument, never in its code"
    assert verify == ["verify", str(out / "experiment")] and serve == ["serve", str(folder), "--start"]


@pytest.mark.skipif(sys.platform == "win32", reason="the lines are for a POSIX shell")
def test_the_control_a_name_unquoted_in_a_pasted_line_does_run_what_it_holds(tmp_path):
    words, ran = listed_by_a_shell("mime serve x;touch${IFS}pwned;y --start", tmp_path)
    assert ran and words == ["serve", "x"]


def sphere_without_a_density(radius_um: float = 10.0):
    """``examples/sphere_settling.py``'s sphere in water whose density is declared nowhere: the build goes on and the
    drag's post-run watch cannot be computed (``not_computed``)."""
    from mime import Body, Experiment, Medium, State
    from mime.dynamics import RigidInertial
    from mime.interactions import Drag
    from mime.scene import Scene, Sphere, Z
    from mime.things import Part
    from mime.units import kg_per_m3, m, m_per_s, m_per_s2, mPa_s, ms, rad_per_s, um

    exp = Experiment("no_density", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(0.1))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, -1e-3)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(radius_um)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.attach(Drag.MRS(spacing=um(radius_um / 4), blob=0.5), body=ball, medium=water)
    return exp


@pytest.fixture(scope="module")
def watch_folders(folders, tmp_path_factory):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    root = tmp_path_factory.mktemp("watches")
    sphere_without_a_density().save(root / "no_density")
    jax.config.update("jax_enable_x64", held)
    return root


def test_a_watch_that_could_not_be_computed_is_listed_and_the_summary_does_not_say_nothing_needs_attention(
        watch_folders, tmp_path, capsys):
    """Run for real, not on a stub: the medium's density is declared nowhere, so the watch is ``not_computed``."""
    out = run(watch_folders, "no_density", "1ms", tmp_path, capsys)
    lines = out.splitlines()
    assert lines[0].startswith("Needs attention (1)") and not out.startswith("Nothing needs attention"), lines[0]
    assert any(line.startswith("  ! post-run watch could not be checked: reynolds:drag:ball") and "not computed" in line
               and "water.density" in line for line in lines)


def test_a_watch_with_no_range_to_compare_with_is_listed_the_same(folders, tmp_path, capsys, monkeypatch):
    from mime.interactions import Drag

    monkeypatch.setattr(Drag, "reynolds", ())
    out = run(folders, "r10", "3ms", tmp_path, capsys)
    assert out.splitlines()[0].startswith("Needs attention (")
    assert any(line.startswith("  ! post-run watch could not be checked: reynolds:drag:ball") and "declares no range" in line
               for line in out.splitlines())


@pytest.mark.parametrize("status", ["above", "within", "no_range", "not_computed"])
def test_every_status_but_within_reaches_the_attention_list(status):
    """Each status of a watch through ``attention_lines`` (the real run's cases are above; the control is ``within``,
    which is a pass and is silent)."""
    import types

    from mime.results._watch import WatchRead

    watch = WatchRead("drag:ball", "drag", "stokes", "ball", "water", status, 0.5, 0.1, 1.0, 1.0, 1e-4, 1000.0, 1e-3, 0.1, "b",
                      True, 0.1, "", ("the medium's density is not declared",) if status == "not_computed" else ())
    report = types.SimpleNamespace(checked={}, refusals=(), warnings=(), profile=None)
    written = types.SimpleNamespace(result=types.SimpleNamespace(flags=(), watches=(watch,)), stopped=None)
    lines = _summary.attention_lines(types.SimpleNamespace(report=report), written, None, Path("exp"))
    assert bool(lines) == (status != "within"), lines
