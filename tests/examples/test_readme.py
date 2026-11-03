"""The README's install and quickstart, run as written: the quickstart's Python is extracted from the README and
executed in a temporary directory with a temporary home, its saved folder is then read by the ``mime pending`` the
README names, and the install section's names are held to the project's own (the distribution, the extras, the
branch). The README is the first thing a new user runs, so the test runs what it says and nothing else.

**What this bounds.** That the quickstart builds, runs, saves, and that the shell command the README gives prints what
the README says it prints; that the distribution and extras the README installs exist in the project. It does not
install from a network, and does not run the ``pip`` line."""

import re
import tomllib
from pathlib import Path

import pytest

from mime.cli import main as mime_main

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def fences(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", README, re.S)


@pytest.fixture(scope="module")
def quickstart(tmp_path_factory):
    folder = tmp_path_factory.mktemp("quickstart")
    code = fences("python")[0]
    namespace: dict[str, object] = {"__name__": "quickstart"}
    import contextlib
    import io
    import os

    previous = Path.cwd()
    os.chdir(folder)
    try:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(code, "README.md (quickstart)", "exec"), namespace)  # noqa: S102 - the README's own code
    finally:
        os.chdir(previous)
    return namespace, printed.getvalue(), folder


def test_the_quickstart_runs_as_written_and_its_speed_is_within_two_per_cent_of_stokess_law(quickstart):
    namespace, printed, _ = quickstart
    assert 1.46 < float(printed.split()[0]) < 1.50, "Stokes's law at standard gravity gives 1.493 mm/s for this ball; the default is a first look"
    result = namespace["result"]
    assert "post-run watches:" in printed and "within the" in printed and "ABOVE" not in printed
    assert result.watches[0].status == "within"  # type: ignore[attr-defined]


def test_the_shell_command_the_readme_gives_reads_the_folder_the_quickstart_saved(quickstart, capsys, monkeypatch):
    _, _, folder = quickstart
    commands = [line for block in fences("sh") for line in block.splitlines() if line.startswith("mime pending")]
    assert commands == ["mime pending settling_ball"]
    monkeypatch.chdir(folder)
    assert mime_main(commands[0].split()[1:]) == 1, "it exits with one while anything is pending"
    shown = capsys.readouterr().out
    assert "6 items pending" in shown and "exp.declare(\"water.viscosity\"" in shown
    assert "three reasons and three numerical defaults" in README


def test_the_install_section_names_the_distribution_the_extras_and_the_branch_that_exist():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    name, extras = project["project"]["name"], project["project"]["optional-dependencies"]
    install = [line for block in fences("sh") for line in block.splitlines() if "pip install" in line]
    assert len(install) == 1 and install[0].startswith(f'pip install "{name}[verify] @ git+https://')
    assert "verify" in extras and "cuda12" in extras and f"{name}[cuda12]" in README
    assert "@v0.3" in install[0] and "docs/platforms.md" in README and (ROOT / "docs" / "platforms.md").is_file()
    assert "uv run mime" in README and project["project"]["scripts"] == {"mime": "mime.cli:main"}
