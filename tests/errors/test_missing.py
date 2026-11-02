"""A module that one of MIME's own requirements provides, missing from the environment, is refused by name with the
command that installs it (``env.missing``), and never as ``error.uncoded``: the case is an environment installed in
place before a requirement was added, which an editable install does not pick up.

Each arm that expects the refusal has a control that plants the lookup away and sees ``error.uncoded`` from the very
same raise, so the refusal is shown to come from the lookup alone."""

import importlib.metadata
import importlib.util
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from _platform_support import needs_pty
from _terminal_support import run_on_terminal, stripped
from mime import __version__
from mime.build import BuildRefused, InternalError, MimeError
from mime.env import pins
from mime.errors import public, translated
from mime.errors import _missing
from mime.errors._missing import declared_requirements, declaring, missing_dependency, reinstall_command

ROOT = Path(__file__).resolve().parents[2]


def gap(module: str):
    """A function in a module of MIME's (by its name and file), importing ``module`` as MIME's own code would."""
    namespace: dict[str, Any] = {"__name__": "mime._planted_import"}
    source = f"import importlib\ndef gap():\n    importlib.import_module({module!r})\n"
    exec(compile(source, "/planted/mime/_planted_import.py", "exec"), namespace)  # noqa: S102
    return namespace["gap"]


def raised(action: Any) -> BaseException:
    try:
        action()
    except Exception as error:  # noqa: BLE001 - the test's own catch of what it planted
        return error
    raise AssertionError("the action did not raise")


def made_unimportable(monkeypatch, module: str) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    assert importlib.util.find_spec(module) is None


def test_the_declared_requirements_are_pyprojects_and_each_module_is_what_its_distribution_provides():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    written = {(canonicalize_name(Requirement(r).name), str(Requirement(r).specifier), None)
               for r in project["dependencies"]}
    for extra, requirements in project["optional-dependencies"].items():
        written |= {(canonicalize_name(Requirement(r).name), str(Requirement(r).specifier), extra) for r in requirements}
    shipped = {(canonicalize_name(d.distribution), str(Requirement(d.requirement).specifier), d.extra)
               for d in declared_requirements()}
    assert shipped == written, "regenerate src/mime/data/env/declared.json from pyproject.toml"
    provided = importlib.metadata.packages_distributions()
    for d in declared_requirements():
        installed = {canonicalize_name(name) for name in provided.get(d.module, ())}
        if d.extra is None:
            assert canonicalize_name(d.distribution) in installed, f"{d.module} is not what {d.distribution} provides"
        elif installed:  # an extra's package may be absent from this lane; where it is installed its module is held
            assert canonicalize_name(d.distribution) in installed


def test_every_core_requirement_is_pinned_so_that_mime_env_check_reports_it_missing():
    pinned = {pin.name for pin in pins("cpu")}
    assert {canonicalize_name(d.distribution) for d in declared_requirements() if d.extra is None} <= pinned


def test_a_missing_core_requirement_is_refused_by_name_with_the_command_and_the_control_sees_uncoded(monkeypatch):
    made_unimportable(monkeypatch, "rich")
    refusal = translated(raised(gap("rich"))).refusals[0]
    assert refusal.code == "env.missing" and refusal.category.value == "environment"
    assert "rich>=13.9" in refusal.message and f"mime-engine {__version__}" in refusal.message
    assert "predate this MIME" in (refusal.why or "") and "removed" in (refusal.why or "")
    assert refusal.action.count("pip install 'rich>=13.9'") == 1 and reinstall_command() in refusal.action
    monkeypatch.setattr(_missing, "declared_requirements", lambda: ())
    control = translated(raised(gap("rich")))
    assert isinstance(control, InternalError) and control.code == "error.uncoded"


def test_a_requirement_behind_an_extra_names_the_extra(monkeypatch):
    made_unimportable(monkeypatch, "hypothesis")
    refusal = translated(raised(gap("hypothesis"))).refusals[0]
    assert refusal.code == "env.missing" and "'verify' extra of mime-engine" in refusal.message
    assert "hypothesis>=6.165" in refusal.message and f"{reinstall_command('verify')}" in refusal.action
    assert "[verify]" in reinstall_command("verify") or "--extra verify" in reinstall_command("verify")


def test_a_module_no_requirement_names_stays_uncoded_and_a_submodule_of_a_present_package_is_not_missing(monkeypatch):
    assert declaring("no_such_module_planted") is None
    error = translated(raised(gap("no_such_module_planted")))
    assert isinstance(error, InternalError) and error.code == "error.uncoded"
    assert missing_dependency(raised(gap("rich.no_such_submodule_planted"))) is None
    assert missing_dependency(ValueError("rich")) is None


def test_the_command_is_the_installs_own(monkeypatch, tmp_path):
    monkeypatch.setattr(_missing, "_checkout", lambda: None)
    assert reinstall_command() == "pip install --upgrade mime-engine"
    assert reinstall_command("verify") == "pip install --upgrade 'mime-engine[verify]'"
    checkout = tmp_path / "a checkout's folder"
    checkout.mkdir()
    monkeypatch.setattr(_missing, "_checkout", lambda: checkout)
    assert reinstall_command() == f"pip install -e {shlex.quote(str(checkout))}"
    assert reinstall_command("verify") == f"pip install -e {shlex.quote(f'{checkout}[verify]')}"
    assert shlex.split(reinstall_command())[-1] == str(checkout), "a path with a space and a quote is one argument"
    (checkout / "uv.lock").write_text("")
    (checkout / ".venv").mkdir()
    monkeypatch.setattr(sys, "prefix", str(checkout / ".venv"))
    assert reinstall_command() == f"cd {shlex.quote(str(checkout))} && uv sync"
    assert reinstall_command("verify") == f"cd {shlex.quote(str(checkout))} && uv sync --extra verify"
    assert shlex.split(reinstall_command())[1] == str(checkout)


def test_a_public_call_raises_it_as_a_build_refusal_that_a_module_not_found_handler_still_catches(monkeypatch):
    made_unimportable(monkeypatch, "rich")
    wrapped = public(gap("rich"))
    with pytest.raises(ModuleNotFoundError) as caught:
        wrapped()
    assert isinstance(caught.value, MimeError) and isinstance(caught.value, BuildRefused)
    assert caught.value.codes() == ["env.missing"]


def mime_run(tmp_path: Path, plant_away: bool) -> tuple[str, int]:
    """``mime run`` on a terminal in a child where ``rich`` cannot be imported, as in an environment that lacks it."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    plant = "import mime.errors._missing as m; m.declared_requirements = lambda: (); " if plant_away else ""
    program = ("import sys; sys.modules['rich'] = None; " + plant +
               "from mime.cli import main; sys.exit(main(sys.argv[1:]))")
    environment = {"PATH": "", "HOME": str(home), "USERPROFILE": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
                   "MIME_TRACEBACK_DIR": str(home / "tracebacks"), "TERM": "xterm-256color", "JAX_PLATFORMS": "cpu",
                   "MIME_OFFLINE": "1", "MIME_TRACEBACK": "short"}
    raw, code = run_on_terminal([sys.executable, "-c", program, "run", "settling_ball", "--duration", "3ms", "--out",
                                 str(tmp_path / "out")], env=environment, cwd=tmp_path)
    return stripped(raw), code


@needs_pty
def test_mime_run_on_a_terminal_without_rich_prints_the_coded_refusal_in_plain_text(tmp_path):
    text, code = mime_run(tmp_path, plant_away=False)
    assert code == 2, text
    assert "env.missing" in text and "rich>=13.9" in text and "pip install" in text and "error.uncoded" not in text
    assert text.count("mime run is refused") == 1, "the plain fallback prints it, the block (which needs rich) cannot"
    control, _ = mime_run(tmp_path, plant_away=True)
    assert "error.uncoded" in control and "env.missing" not in control
