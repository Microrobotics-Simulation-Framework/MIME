"""The README's install and quickstart, run as written: the quickstart's Python is extracted from the README and
executed in a temporary directory with a temporary home, its saved folder is then read by the ``mime pending`` the
README names, and the install section's names are held to the project's own (the distribution, the extras, the
pre-release flag a plain install would need). The README is the first thing a new user runs, so the test runs what it
says and nothing else.

The README is also the project's page on PyPI, which resolves no relative link, so every link in it is absolute: a file
of the repository at this release's tag (``blob/v<version>/`` or ``tree/v<version>/``, naming a path of this checkout),
a page of this release's documentation (``https://microrobotica.org/mime/<release>/``, naming a page whose source is in
``docs/``), a notebook of ``docs/notebooks/`` opened in Colab, or one of the few other addresses listed. The control:
the README as it was before the release, with its relative links and its install from the branch, fails both checks.

**What this bounds.** That the quickstart builds, runs, saves, and that the shell command the README gives prints what
the README says it prints; that the distribution and extras the README installs exist in the project; that each link
names a file or a page source this checkout has, at the tag and the release of its own version. It does not install
from a network, does not run the ``pip`` line, and does not fetch a link (the tag does not exist until the release)."""

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

import mime
from mime.build._refusals import DOCS, docs_version
from mime.cli import main as mime_main

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")


REPOSITORY = "https://github.com/Microrobotics-Simulation-Framework/MIME"
_SPEC = importlib.util.spec_from_file_location("colab_links", ROOT / "scripts" / "colab_links.py")
assert _SPEC is not None and _SPEC.loader is not None
COLAB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(COLAB)
#: The absolute addresses the README may link that are neither a file at this release's tag nor a page of its docs
#: (a notebook opened in Colab is held by ``scripts/colab_links.py``'s check, which tests/examples/test_notebooks.py runs).
OTHER_LINKS = {"https://microrobotica.org/maddening/", f"{REPOSITORY}/tree/pre-0.3", f"{REPOSITORY}/tree/v0.2.0",
               COLAB.BADGE}


def fences(language: str, text: str = README) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, re.S)


def link_problems(text: str, root: Path = ROOT) -> list[str]:
    """Every link of ``text`` that is relative, or names what this checkout's release does not have."""
    files = re.compile(rf"{re.escape(REPOSITORY)}/(?:blob|tree)/v{re.escape(mime.__version__)}/(?P<path>[^#]+)")
    pages = re.compile(rf"{re.escape(DOCS)}/{re.escape(docs_version())}/(?P<path>[^#]*)")
    found = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target in OTHER_LINKS:
            continue
        if (held := COLAB.LINK.fullmatch(target)) is not None:
            if not (root / "docs" / "notebooks" / held["name"]).is_file():
                found.append(f"{target}: no docs/notebooks/{held['name']} in this checkout")
            continue
        if (held := files.fullmatch(target)) is not None:
            if not (root / held["path"]).exists():
                found.append(f"{target}: no {held['path']} in this checkout")
        elif (held := pages.fullmatch(target)) is not None:
            page = held["path"].removeprefix("pages/")
            if held["path"] and not (held["path"].startswith("pages/") and (root / "docs" / f"{page}.md").is_file()):
                found.append(f"{target}: no page source docs/{page}.md")
        else:
            found.append(f"{target}: neither a file at the tag v{mime.__version__} nor a page of the release "
                         f"{docs_version()}'s documentation")
    return found


def install_problems(text: str) -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    name = project["project"]["name"]
    install = [line for block in fences("sh", text) for line in block.splitlines() if "pip install" in line]
    if install != [f"pip install --pre {name}"]:
        return [f"the install lines are {install}; one, `pip install --pre {name}`, since a plain install skips a "
                f"pre-release"]
    return []


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


def test_the_install_section_names_the_distribution_the_extras_and_the_pre_release_flag():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    name, extras = project["project"]["name"], project["project"]["optional-dependencies"]
    assert install_problems(README) == []
    assert "verify" in extras and "cuda12" in extras and f"{name}[cuda12]" in README
    assert f'pip install --pre "{name}[verify]"' in README and "mime --version" in README
    assert "mime assets fetch unit_tube_R0.997" in README
    assert "docs/platforms.md" in README and (ROOT / "docs" / "platforms.md").is_file()
    assert "uv run mime" in README and project["project"]["scripts"] == {"mime": "mime.cli:main"}


def test_every_link_is_absolute_and_names_a_file_or_a_page_of_this_release():
    assert link_problems(README) == []
    assert any(f"/blob/v{mime.__version__}/" in link for link in re.findall(r"\]\(([^)\s]+)\)", README))
    assert any(link.startswith(f"{DOCS}/{docs_version()}/pages/") for link in re.findall(r"\]\(([^)\s]+)\)", README))


#: The README's install and two of its links as they were before the release.
EARLIER = """```sh
pip install "mime-engine[verify] @ git+https://github.com/Microrobotics-Simulation-Framework/MIME@v0.3"
```
See [`docs/platforms.md`](docs/platforms.md) and [the profiles](https://microrobotica.org/mime/0.3.0/pages/nowhere).
"""


def test_control_the_readme_before_the_release_fails_both_checks():
    assert install_problems(EARLIER) != []
    problems = link_problems(EARLIER)
    assert len(problems) == 2 and problems[0].startswith("docs/platforms.md: neither a file at the tag")
    assert problems[1].endswith("/pages/nowhere: no page source docs/nowhere.md")
    assert link_problems(f"[a]({REPOSITORY}/blob/v{mime.__version__}/no_such_file.md)") != []
    assert link_problems(f"[n]({COLAB.link('no_such_notebook.ipynb')})") != [], "a Colab link to no notebook passes"
