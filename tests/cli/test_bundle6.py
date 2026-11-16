"""CLI bundle 6: what the CLI tour found, each with the test that fails without the fix.

(a) env check, (b) a closed pipe, (c) one refusal for a folder that is not there, (d) and (e) hints, (f) a study's
output folder looked at before planning, (g) no silent overwrite, (h) the reduced docs build said, (i) the
subcommand's usage, (j) ``mime show`` reads a run, (k) a code for a bad address, (m) ``mime push --dry-run``."""

from __future__ import annotations

import subprocess
import sys
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


# -- (c) one refusal for a folder that does not exist ----------------------------------------------------------------


@pytest.mark.parametrize("words", [["verify"], ["pending"], ["show"], ["check"], ["modal"], ["scaffold", "study"],
                                   ["scaffold", "bundle"], ["push", "--to", "zenodo-sandbox"], ["pack"], ["card"],
                                   ["run", "--out", "o"], ["serve"]],
                         ids=lambda words: " ".join(words))
def test_a_folder_that_is_not_there_is_refused_alike_by_every_command(saved, capsys, words):
    """Each command named a missing folder by the first file it looked for (a lock, mime.toml, two files); each now
    says ``load.missing-folder`` and names the saved folder beside it that is the likely meant one."""
    missing = saved.parent / "bal"
    command, rest = words[0], words[1:]
    kinds = [word for word in rest if word in ("study", "bundle")]
    line = [command, *kinds, str(missing), *[word for word in rest if word not in kinds]]
    assert mime_main(line) == 2
    err = capsys.readouterr().err
    assert "load.missing-folder" in err and "there is no folder" in err
    assert "ball" in err, "the folder beside it is named"


# -- (b) a pipe the reader closed -------------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="a closed pipe is signalled differently there")
def test_a_pipe_closed_by_its_reader_ends_the_command_quietly():
    """``mime registry --json | head`` ended in an uncoded refusal that asked the user to report a bug. The reader's
    leaving is no bug: nothing is printed, and the exit code is the one a process ended by SIGPIPE shows."""
    process = subprocess.Popen([sys.executable, "-m", "mime.cli", "registry", "--json"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    assert len(process.stdout.read(100)) == 100
    process.stdout.close()
    err = process.stderr.read()
    assert process.wait(timeout=120) == 141
    process.stderr.close()
    assert err == b"", err.decode()


def test_a_broken_pipe_met_inside_a_command_is_not_a_mime_error(monkeypatch, capsys):
    """The control for the test above in-process: the same error raised by a command's print, wrapped or not."""
    from mime import cli

    def broken(*_: object) -> int:
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(cli, "_registry", broken)
    monkeypatch.setattr(cli, "_pipe_closed", lambda: 141)
    assert mime_main(["registry"]) == 141
    assert "error.uncoded" not in capsys.readouterr().err

    def wrapped(*_: object) -> int:
        try:
            raise BrokenPipeError(32, "Broken pipe")
        except BrokenPipeError as error:
            raise RuntimeError("write failed") from error

    monkeypatch.setattr(cli, "_registry", wrapped)
    assert mime_main(["registry"]) == 141


# -- (d) (e) (j) a run folder where a saved experiment is read; mime show reads a run -----------------------------------


@pytest.fixture(scope="module")
def ran(saved, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ran") / "run"
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        assert mime_main(["run", str(saved), "--duration", "0.1s", "--out", str(out)]) == 0
    finally:
        jax.config.update("jax_enable_x64", held)
    return out


@pytest.mark.parametrize("command, reads", [("verify", "--run"), ("pending", "run/experiment")])
def test_a_run_folder_given_to_a_saved_experiments_command_is_named_with_what_reads_it(ran, capsys, command, reads):
    assert mime_main([command, str(ran)]) == 2
    err = capsys.readouterr().err
    assert "load.run-folder" in err and "is a run folder" in err and reads in err
    assert "mime show" in err and "env.lock" not in err


def test_show_reads_a_run_folders_outputs(ran, capsys):
    from mime.results import RunResult

    assert mime_main(["show", str(ran)]) == 0
    out = capsys.readouterr().out
    assert out.strip() == RunResult.read(ran).text().strip()
    assert "5 steps" in out


def test_show_json_of_a_run_folder_is_its_manifest_and_the_tree_flags_are_refused(ran, capsys):
    import json

    assert mime_main(["show", str(ran), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == "mime-run-outputs"
    with pytest.raises(SystemExit) as stopped:
        mime_main(["show", str(ran), "--graph"])
    assert stopped.value.code == 2 and "a run folder shows its outputs" in capsys.readouterr().err


# -- (f) a study's output folder is looked at before the base loads and the plan is made ------------------------------


def test_mime_study_run_refuses_a_folder_that_holds_something_before_it_reads_the_spec(tmp_path, capsys):
    """The refusal came after about 29 s of loading and planning. The spec named here does not exist, and the
    refusal is still the folder's: the folder is looked at first."""
    out = tmp_path / "mine"
    out.mkdir()
    (out / "chapter1.tex").write_text("mine\n", encoding="utf-8")
    assert mime_main(["study", "run", str(tmp_path / "no_such.study.mime.yaml"), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "study.folder-not-empty" in err and "chapter1.tex" in err and "study.spec" not in err
    assert [path.name for path in out.iterdir()] == ["chapter1.tex"]


def test_a_study_run_to_a_folder_that_holds_something_does_not_plan(tmp_path, monkeypatch):
    from mime import Study
    from mime.build import BuildRefused

    study = Study(full_toy())  # no sweep: a plan would refuse differently, or not at all
    monkeypatch.setattr(Study, "plan", lambda self: (_ for _ in ()).throw(AssertionError("planned")))
    out = tmp_path / "mine"
    out.mkdir()
    (out / "chapter1.tex").write_text("mine\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        study.run_to(out)
    assert refused.value.codes() == ["study.folder-not-empty"]


# -- (g) no silent overwrite: mime run over a run, mime modal over a script ---------------------------------------------


def test_mime_run_refuses_a_run_folder_without_force_and_warns_with_it(saved, ran, tmp_path, capsys):
    """MIME's own run folder was exempt from ``run.out-not-empty``: a second run replaced the first, said nothing."""
    out = tmp_path / "again"
    import shutil

    shutil.copytree(ran, out)
    before = {path.name: path.read_bytes() for path in sorted(out.iterdir()) if path.is_file()}
    assert mime_main(["run", str(saved), "--duration", "0.1s", "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "run.out-not-empty" in err and "a MIME run folder already" in err and "--force" in err
    assert {path.name: path.read_bytes() for path in sorted(out.iterdir()) if path.is_file()} == before


def modal_words(saved: Path, out: Path, *more: str) -> list[str]:
    return ["modal", str(saved), "--out", str(out), *more]


def test_mime_modal_refuses_to_replace_a_script_without_force_and_names_what_it_replaced_with_it(saved, tmp_path,
                                                                                                 capsys):
    out = tmp_path / "script"
    assert mime_main(modal_words(saved, out)) == 0
    written = {path.name: path.read_bytes() for path in out.iterdir()}
    (out / "notes.txt").write_text("mine\n", encoding="utf-8")
    capsys.readouterr()
    assert mime_main(modal_words(saved, out)) == 2
    err = capsys.readouterr().err
    assert "modal.out-exists" in err and "modal_app.py" in err and "--force" in err
    assert {path.name: path.read_bytes() for path in out.iterdir() if path.name != "notes.txt"} == written
    assert mime_main(modal_words(saved, out, "--force")) == 0
    shown = capsys.readouterr().out
    assert "warning: --force: replaced" in shown and "modal_app.py" in shown
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine\n", "what is not the script's is never touched"


def test_mime_modal_into_a_folder_that_holds_only_other_files_writes_without_force(saved, tmp_path, capsys):
    out = tmp_path / "script"
    out.mkdir()
    (out / "notes.txt").write_text("mine\n", encoding="utf-8")
    assert mime_main(modal_words(saved, out)) == 0
    assert "warning" not in capsys.readouterr().out


# -- (i) an argument no command took shows that command's usage ---------------------------------------------------------


@pytest.mark.parametrize("words, usage", [(["run", "folder", "--out", "o", "--bogus"], "usage: mime run "),
                                          (["--plain", "study", "run", "spec", "--out", "o", "--bogus"],
                                           "usage: mime study run "),
                                          (["verify", "folder", "--bogus"], "usage: mime verify "),
                                          (["--bogus"], "usage: mime [-h]")])
def test_an_unrecognised_flag_shows_the_usage_of_its_command(capsys, words, usage):
    with pytest.raises(SystemExit) as stopped:
        mime_main(words)
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith(usage) and "unrecognized arguments: --bogus" in err


# -- (k) an address that is no tcp:// endpoint has a code that says so ----------------------------------------------------


def test_an_address_without_the_tcp_scheme_is_refused_as_an_endpoint_not_as_a_remote_bind(saved, capsys):
    assert mime_main(["serve", str(saved), "--commands", "127.0.0.1:5555"]) == 2
    err = capsys.readouterr().err
    assert "live.endpoint" in err and "is not a tcp:// endpoint" in err and "live.remote-bind" not in err


# -- (h) a reduced docs build says so, at its start and at its end ---------------------------------------------------------


def test_a_docs_build_outside_a_checkout_says_what_it_left_out_twice(tmp_path, capsys):
    """With no hand-written sources (no checkout at ``--root``) the build wrote only the generated pages, silently:
    132 of the full set, with nothing to say the guides were missing."""
    assert mime_main(["docs", "build", "--root", str(tmp_path / "empty"), "--out", str(tmp_path / "pages"),
                      "--skip-evidence"]) == 0
    lines = capsys.readouterr().out.splitlines()
    notes = [line for line in lines if "left out" in line]
    assert len(notes) == 2 and "--root <checkout>" in notes[0] and "--root <checkout>" in notes[1]
    assert lines.index(notes[0]) < len(lines) - 1 and lines[-1] == notes[1], "again in the closing summary"
    assert lines[-1].startswith("mime docs build: wrote ")


# -- (a) a plain install passes its own env check: a version inside the declared ranges is a note -------------------------


def pins_of(*names: str):
    from mime.env._pins import applicable, pins

    from mime.env._check import Expected, this_system

    held = applicable(pins("cpu"), this_system().markers)
    return {name: Expected(held[name].version, "the cpu pins") for name in names}


def newer(version: str) -> str:
    """A version above ``version`` by one in its last number: the next release a plain install may resolve to."""
    head, _, last = version.rpartition(".")
    return f"{head}.{int(last) + 1}" if last.isdigit() else version


def test_a_version_inside_the_declared_ranges_is_a_note_and_one_outside_is_a_failure():
    from mime.env._check import distribution_refusals, installed_distributions, outside_declared_ranges

    held = installed_distributions()
    expected = pins_of("numpy")
    pinned = expected["numpy"].version
    installed = {"numpy": (newer(pinned),)}
    notes: list[str] = []
    assert distribution_refusals(installed, expected, strict=False, notes=notes) == []
    assert len(notes) == 1 and f"tested with {pinned}" in notes[0] and "inside the ranges" in notes[0]
    (strictly,) = distribution_refusals(installed, expected)
    assert strictly.code == "env.pin", "strict, the default: exact pins"
    (broken,) = distribution_refusals({"numpy": ("0.0.1",)}, expected, strict=False, notes=notes)
    assert broken.code == "env.pin" and "outside" in broken.message and "declares" in broken.message
    assert outside_declared_ranges("numpy", held["numpy"][0]) is None
    assert outside_declared_ranges("numpy", "0.0.1") is not None


def test_a_missing_distribution_and_one_installed_twice_stay_failures_without_strict():
    from mime.env._check import distribution_refusals

    expected = pins_of("numpy")
    assert [r.code for r in distribution_refusals({}, expected, strict=False)] == ["env.missing"]
    assert [r.code for r in distribution_refusals({"numpy": ("2.5.5", "2.5.6")}, expected, strict=False)] == ["env.pin"]


def test_env_check_without_a_folder_notes_what_differs_and_passes_and_strict_refuses_it(monkeypatch, capsys):
    """The plain install of MIME failed its own check on five pins newer than CI's (markdown-it-py, numpy, openmaterica,
    platformdirs, wadler-lindig). The environment is stood in for; the ranges are the real installed ones."""
    from mime.env import _check

    expected = pins_of("numpy")
    ahead = {"numpy": (newer(expected["numpy"].version),)}
    monkeypatch.setattr(_check, "installed_distributions", lambda paths=None: ahead)
    monkeypatch.setattr(_check, "expected_distributions", lambda *a, **k: (expected, []))
    monkeypatch.setattr(_check, "environment_modules", lambda paths=None: {})
    words = ["env", "check", "--variant", "cpu"]
    assert mime_main(words) == 0
    out = capsys.readouterr().out
    assert "1 note, not failures" in out and "note: numpy is installed at" in out
    assert mime_main([*words, "--strict"]) == 2
    assert "env.pin" in capsys.readouterr().err
