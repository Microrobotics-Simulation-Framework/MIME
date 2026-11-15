"""The control of ``mime scaffold`` (design §5): each kind scaffolded with ``--yes`` into a folder, then its own check.

``failures_of(kind, folder)`` scaffolds the kind and returns every way its check fails, empty when it passes: an error the
scaffold raises is one of them, since a scaffold that cannot write what its templates describe has failed the check. The
checks are the kind's own:

- an experiment is saved by running its file, and its saved folder verifies;
- a bundle and a study get a ``mime.toml`` that ``mime check`` finds nothing in, listing exactly what is still to write
  (no creator named, and for a block no case);
- a plugin is built into a wheel with the environment's own build backend and unpacked into a temporary target, which only a
  subprocess sees (``PYTHONPATH``): there ``mime registry --json`` lists its example entry and no namespace failed to load;
  its own tests pass (the default lane and the slow lane); with the planted error in (``MIME_PLANT_ERROR``) **only the planted
  property fails**; the ``bash ci`` blocks of its README run (its CI's own extraction line, read from its ``ci.yml``) and
  ``mime check`` of its benchmark finds nothing and lists exactly the creator; and its ``ci.yml`` names the four jobs. Nothing
  is installed into the environment this suite runs in, whose registry every later test reads.

A defect planted in a template (a misspelt key) must fail the check of its kind (``plant``)."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import hatchling.build
import pytest
import yaml

from mime import scaffold
from mime.cli import main as mime_main
from mime.push import _check, _draft
from mime.scaffold import _templates
from push_cases import saved_toy
from study_rigs import write_study

NAMESPACE = "scafx"
#: What each kind's check lists as still to write, in order, by the start of each item.
TO_DO = {"bundle": ("mime.toml names no creator",), "study": ("mime.toml names no creator",),
         "block": ("mime.toml names no creator", "block ")}
KINDS = ("experiment", "bundle", "study", "block", "plugin")


def child_environment(*paths: Path) -> dict[str, str]:
    """This process's environment, with ``paths`` ahead on the path the child reads and the CPU for JAX."""
    held = os.environ.get("PYTHONPATH")
    return {**os.environ, "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": os.pathsep.join([*(str(each) for each in paths), *([held] if held else [])])}


def unpacked(plugin: Path, target: Path) -> Path:
    """The plugin built into a wheel by hatchling (installed in the environment: no network, no pip) and unpacked into
    ``target``: the dist-info and the package, where ``importlib.metadata`` finds them for a process whose
    ``PYTHONPATH`` names ``target``, and for no other."""
    here = Path.cwd()
    os.chdir(plugin)
    try:
        wheel = hatchling.build.build_wheel(str(target.parent / f"{target.name}-wheel"))
    finally:
        os.chdir(here)
    with zipfile.ZipFile(target.parent / f"{target.name}-wheel" / wheel) as archive:
        archive.extractall(target)
    return target


def listed(target: Path) -> tuple[list[str], list[str]]:
    """The entries ``mime registry --json`` lists in a process that sees ``target``, and the namespaces it did not load."""
    done = subprocess.run([sys.executable, "-m", "mime.cli", "registry", "--json"], capture_output=True, text=True,
                          env=child_environment(target), check=False)
    if done.returncode != 0:
        return [], [f"mime registry --json exited {done.returncode}: {done.stderr[-400:]}"]
    held = json.loads(done.stdout)
    return [entry["name"] for entry in held["entries"]], list(held["not_loaded"])


#: The one test the planted error (tests/plants.py of the plugin) must fail, and no other.
PLANTED_PROPERTY = "tests/engine/test_properties.py::test_the_load_is_linear_in_the_force"
PLANT_VARIABLE = "MIME_PLANT_ERROR"
CI_JOBS = {"default", "slow", "release", "typing"}


def pytest_in(folder: Path, target: Path, *arguments: str, planted: bool = False) -> tuple[int, list[str], str]:
    """The plugin's own pytest, in a subprocess that sees ``target``: its exit code, the tests that failed (their ids) and the
    output's tail. ``planted`` sets the variable that has the plugin's conftest plant its error."""
    environment = child_environment(target)
    if planted:
        environment[PLANT_VARIABLE] = "1"
    done = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--no-header", "--tb=no", "-rf",
                           *arguments], cwd=folder, capture_output=True, text=True, env=environment, check=False)
    failed = [line.removeprefix("FAILED ").split(" - ")[0].strip() for line in done.stdout.splitlines()
              if line.startswith("FAILED ")]
    return done.returncode, failed, done.stdout[-1500:]


def readme_use(folder: Path, target: Path) -> list[str]:
    """The README's ``bash ci`` blocks run as its CI runs them: the extraction line is read from the ``ci.yml`` the scaffold
    wrote, and run here with this environment's ``mime`` and ``python`` first on the path. Every way it fails: a command that
    exits non-zero, and a benchmark whose ``mime check`` finds anything but the creator still to write."""
    line = next((each.split("run: ", 1)[1] for each in (folder / ".github/workflows/ci.yml").read_text(encoding="utf-8")
                 .splitlines() if "README.md | uv run" in each), None)
    if line is None:
        return ["ci.yml has no line that runs the README's `bash ci` blocks"]
    environment = child_environment(target)
    environment["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    done = subprocess.run(["bash", "-c", line.replace("uv run --no-sync ", "")], cwd=folder, capture_output=True,
                          text=True, env=environment, check=False)
    failures: list[str] = []
    if done.returncode != 0:
        failures.append(f"the README's Use section exits {done.returncode}:\n{(done.stdout + done.stderr)[-500:]}")
    elif "0 findings, 1 to do" not in done.stdout or "names no creator" not in done.stdout:
        failures.append(f"mime check of the benchmark does not find 0 findings and the creator alone to do:\n"
                        f"{done.stdout[-500:]}")
    return failures


def plugin_failures(folder: Path, target: Path) -> list[str]:
    """Every way the scaffolded plugin at ``folder`` fails: a file that does not compile, a ``pyproject.toml`` that does not
    read, a ``ci.yml`` without its four jobs, an example entry ``mime registry --json`` does not list once the plugin is
    unpacked into ``target``, a test of its own that fails there (the default lane and the slow lane), a planted error that
    fails anything but the one property it is planted for (or that property not failing), and a README that does not run."""
    failures: list[str] = []
    try:
        project = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        return [f"pyproject.toml does not read: {error}"]
    for path in sorted(folder.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            failures.append(f"{path.relative_to(folder)} does not compile: {error}")
    jobs = set(yaml.safe_load((folder / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"])
    if jobs != CI_JOBS:
        failures.append(f"ci.yml has the jobs {sorted(jobs)}, expected {sorted(CI_JOBS)}")
    if failures:
        return failures
    unpacked(folder, target)
    points = project["project"].get("entry-points", {})
    namespace = next(iter(points.get("mime.entries", {NAMESPACE: ""})))
    names, not_loaded = listed(target)
    if f"{namespace}.example_load" not in names:
        failures.append(f"mime registry --json does not list {namespace}.example_load (it lists {len(names)} entries)")
    failures += [f"namespace failed to load: {each}" for each in not_loaded]
    if failures:
        return failures
    code, _, tail = pytest_in(folder, target, "tests")
    if code != 0:
        return [f"the plugin's own tests fail:\n{tail}"]
    code, _, tail = pytest_in(folder, target, "-m", "slow", "tests")
    if code != 0:
        return [f"the plugin's slow lane fails:\n{tail}"]
    _, failed, tail = pytest_in(folder, target, "tests", planted=True)
    if failed != [PLANTED_PROPERTY]:
        return [f"with the planted error in, the tests that fail are {failed}, expected only {PLANTED_PROPERTY}:\n{tail}"]
    return readme_use(folder, target)


def check_of(kind: str, folder: Path) -> list[str]:
    """``mime check``'s verdict on ``folder``: no finding, and exactly the kind's to-do list."""
    found = _check.check(folder)
    failures = [f"finding: {item.what}" for item in found.findings]
    todo = [item.what for item in found.of(_check.TO_DO)]
    expected = TO_DO[kind]
    if len(todo) != len(expected) or not all(item.startswith(start) for item, start in zip(todo, expected, strict=False)):
        failures.append(f"to do: {todo}, expected items starting {list(expected)}")
    return failures


def verified(folder: Path) -> list[str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = mime_main(["verify", str(folder)])
    return [] if code == 0 else [f"mime verify exited {code}:\n{out.getvalue()[-400:]}"]


def made(kind: str, root: Path) -> Path:
    """The path to scaffold ``kind`` at, in ``root``: a saved or study folder for a bundle and a study."""
    if kind == "bundle":
        return saved_toy(root)
    if kind == "study":
        return write_study(root / "toy_study", figure=False)
    return root / ("scafx-probes" if kind == "plugin" else f"new_{kind}")


def failures_of(kind: str, root: Path) -> list[str]:
    """Scaffold ``kind`` with ``--yes`` at a path in ``root`` and run its check: every failure, empty when it passes."""
    path = made(kind, root)
    try:
        sets = [f"plugin.namespace={NAMESPACE}"] if kind == "plugin" else []
        scaffold.scaffold(kind, path, yes=True, sets=sets, say=lambda line: None)
        if kind == "experiment":
            return verified(path / "saved")
        if kind == "plugin":
            return plugin_failures(path, root / "target")
        return check_of(kind, path)
    except Exception as error:  # noqa: BLE001 - a scaffold that raises has failed its check, whatever it raised
        return [f"the scaffold raised {type(error).__name__}: {str(error)[:300]}"]


#: More misspelt keys and slips in the plugin's files, each of which must fail the plugin's check: the file's template, the
#: text to replace and what replaces it. The first of a kind is the one ``planted`` plants for the count of 5.
PLUGIN_PLANTS: dict[str, tuple[str, str, str, str]] = {
    "the mime.assets entry point's key": ("pyproject.toml.txt", '"mime.assets"', '"mime.assts"', "test_registry"),
    "the planted error that does nothing": ("plants.py.txt", "jnp.clip(real(force, gain), -SATURATION, SATURATION)",
                                            "real(force, gain)", "test_the_planted_error_is_caught_every_time"),
    "the variable that switches the planted error on, misspelt": ("plants.py.txt", 'PLANT_VARIABLE = "MIME_PLANT_ERROR"',
                                                                   'PLANT_VARIABLE = "MIME_PLANT_ERRRO"', "expected only"),
    "an import of mime in the engine": ("engine_load.py.txt", "from jax import Array", "from jax import Array\nimport mime",
                                        "test_engine_imports"),
    "a command of the README that is misspelt": ("README.md.txt", "python examples/example.py", "python examples/exampel.py",
                                                 "README"),
}


def planted(kind: str, which: str | None = None) -> Callable[[pytest.MonkeyPatch], None]:
    """Plant a misspelt key in ``kind``'s template (a bundle's and a study's text comes from push's one writer, whose
    output the plant edits), by a patch on the module that holds it. For a plugin, ``which`` names one of
    :data:`PLUGIN_PLANTS` instead."""
    def edit(patch: pytest.MonkeyPatch) -> None:
        if kind in ("bundle", "study"):
            original = _draft.manifest_text
            key = "\nlicence = " if kind == "bundle" else "\ntitle = "
            patch.setattr(_draft, "manifest_text", lambda *a, **k: original(*a, **k).replace(key, key.replace("e", "x", 1)))
            return
        name, old, new = ({"experiment": ("experiment.py.txt", "experiment.licence(", "experiment.licnce("),
                           "block": ("mime.toml.txt", "\nlicence = ", "\nlicnce = "),
                           "plugin": ("pyproject.toml.txt", '"mime.entries"', '"mime.entrys"')}[kind]
                          if which is None else PLUGIN_PLANTS[which][:3])
        real = _templates.template
        patch.setattr(_templates, "template", lambda k, n: real(k, n).replace(old, new) if (k, n) == (kind, name)
                      else real(k, n))
    return edit


__all__ = ["KINDS", "NAMESPACE", "PLUGIN_PLANTS", "check_of", "child_environment", "failures_of", "listed", "made", "planted", "unpacked",
           "verified"]


class Person:
    """A person at the terminal: each question answered in turn from ``answers``, the save-defaults question by ``save``;
    what was asked and said is kept."""

    def __init__(self, *answers: str, save: bool = False) -> None:
        self.answers = list(answers)
        self.save = save
        self.asked: list[str] = []
        self.said: list[str] = []

    def typed(self, prompt: str) -> str:
        self.asked.append(prompt)
        assert self.answers, f"asked more than was answered: {prompt!r}"
        return self.answers.pop(0)

    def ask(self, question: str) -> bool:
        self.asked.append(question)
        return self.save

    def say(self, line: str) -> None:
        self.said.append(line)


def tree_of(root: Path) -> dict[str, bytes | None]:
    """Every path below ``root`` (``None`` for a folder) with its bytes: what a scaffold may not change."""
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in sorted(root.rglob("*"))}


__all__ += ["Person", "tree_of"]
