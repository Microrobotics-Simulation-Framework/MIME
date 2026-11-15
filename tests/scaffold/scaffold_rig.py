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
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def plugin_static(folder: Path) -> tuple[dict[str, Any], list[str]]:
    """The plugin's ``pyproject.toml`` read, and every way its files fail before anything is built: a ``pyproject.toml`` that
    does not read, a file that does not compile, a ``ci.yml`` without its four jobs."""
    try:
        project = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        return {}, [f"pyproject.toml does not read: {error}"]
    failures: list[str] = []
    for path in sorted(folder.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            failures.append(f"{path.relative_to(folder)} does not compile: {error}")
    jobs = set(yaml.safe_load((folder / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"])
    if jobs != CI_JOBS:
        failures.append(f"ci.yml has the jobs {sorted(jobs)}, expected {sorted(CI_JOBS)}")
    return project, failures


def plugin_listing(folder: Path, target: Path, project: dict[str, Any]) -> list[str]:
    """The plugin unpacked into ``target`` from its wheel, and every way ``mime registry --json`` then fails to list its example
    entry or to load a namespace."""
    unpacked(folder, target)
    points = project["project"].get("entry-points", {})
    namespace = next(iter(points.get("mime.entries", {NAMESPACE: ""})))
    names, not_loaded = listed(target)
    failures = []
    if f"{namespace}.example_load" not in names:
        failures.append(f"mime registry --json does not list {namespace}.example_load (it lists {len(names)} entries)")
    return failures + [f"namespace failed to load: {each}" for each in not_loaded]


def plugin_tests(folder: Path, target: Path, *arguments: str) -> list[str]:
    """The plugin's own pytest on ``arguments`` in a process that sees ``target``: how it fails, empty when it passes."""
    code, _, tail = pytest_in(folder, target, *arguments)
    return [] if code == 0 else [f"the plugin's tests fail ({' '.join(arguments)}):\n{tail}"]


def planted_run_failures(folder: Path, target: Path) -> list[str]:
    """The tests the planted error fails when it is put in (``MIME_PLANT_ERROR``): only the one property it is planted for."""
    _, failed, tail = pytest_in(folder, target, "tests", planted=True)
    if failed != [PLANTED_PROPERTY]:
        return [f"with the planted error in, the tests that fail are {failed}, expected only {PLANTED_PROPERTY}:\n{tail}"]
    return []


def plugin_lanes(folder: Path, target: Path) -> list[str]:
    """The stages that run the plugin's own tests, in order, to the first that fails: the default lane, the slow lane, the
    planted-error run and the README's blocks."""
    for stage in (lambda: plugin_tests(folder, target, "tests"), lambda: plugin_tests(folder, target, "-m", "slow", "tests"),
                  lambda: planted_run_failures(folder, target), lambda: readme_use(folder, target)):
        failures = stage()
        if failures:
            return failures
    return []


def plugin_failures(folder: Path, target: Path, lanes: bool = True) -> list[str]:
    """Every way the scaffolded plugin at ``folder`` fails: a file that does not compile, a ``pyproject.toml`` that does not
    read, a ``ci.yml`` without its four jobs, an example entry ``mime registry --json`` does not list once the plugin is
    unpacked into ``target``, a test of its own that fails there (the default lane and the slow lane), a planted error that
    fails anything but the one property it is planted for (or that property not failing), and a README that does not run.
    With ``lanes`` false, only the stages up to the listing (what a namespace has to be to register)."""
    project, failures = plugin_static(folder)
    if failures:
        return failures
    failures = plugin_listing(folder, target, project)
    if failures or not lanes:
        return failures
    return plugin_lanes(folder, target)


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


def made(kind: str, root: Path, name: str | None = None) -> Path:
    """The path to scaffold ``kind`` at, in ``root``: a saved or study folder for a bundle and a study; ``name`` is the
    folder of the other kinds."""
    if kind == "bundle":
        return saved_toy(root)
    if kind == "study":
        return write_study(root / "toy_study", figure=False)
    return root / (name or ("scafx-probes" if kind == "plugin" else f"new_{kind}"))


def failures_of(kind: str, root: Path, name: str | None = None, sets: list[str] | None = None, lanes: bool = True) -> list[str]:
    """Scaffold ``kind`` with ``--yes`` at a path in ``root`` (``name`` its folder, ``sets`` its ``--set`` words: for a plugin
    the namespace ``scafx``) and run its check: every failure, empty when it passes. ``lanes`` false stops a plugin's check
    at its registry listing."""
    path = made(kind, root, name)
    try:
        sets = sets if sets is not None else [f"plugin.namespace={NAMESPACE}"] if kind == "plugin" else []
        scaffold.scaffold(kind, path, yes=True, sets=sets, say=lambda line: None)
        if kind == "experiment":
            return verified(path / "saved")
        if kind == "plugin":
            return plugin_failures(path, root / "target", lanes)
        return check_of(kind, path)
    except Exception as error:  # noqa: BLE001 - a scaffold that raises has failed its check, whatever it raised
        return [f"the scaffold raised {type(error).__name__}: {str(error)[:300]}"]


#: The package of the plugin the control scaffolds (``scafx-probes``).
PLUGIN_PACKAGE = "scafx_probes"


@dataclass(frozen=True)
class Plant:
    """A slip planted in one file of the scaffolded plugin: ``old`` replaced by ``new`` in ``path`` (below the plugin's folder),
    the one stage that must catch it (``stage`` runs on the plugin and the target it is unpacked in), whether the slip is in
    what the wheel holds (``rebuilds``: the plugin is unpacked again) and a text the failure must carry."""

    path: str
    old: str
    new: str
    stage: Callable[[Path, Path], list[str]]
    message: str
    rebuilds: bool = False


def registry_stage(folder: Path, target: Path) -> list[str]:
    return plugin_tests(folder, target, "tests/test_registry.py")


def planted_control_stage(folder: Path, target: Path) -> list[str]:
    return plugin_tests(folder, target, "tests/engine/test_properties.py::test_the_planted_error_is_caught_every_time")


def engine_imports_stage(folder: Path, target: Path) -> list[str]:
    return plugin_tests(folder, target, "tests/test_engine_imports.py")


def relative_imports_stage(folder: Path, target: Path) -> list[str]:
    return plugin_tests(folder, target, "tests/test_relative_imports.py")


def durations_stage(folder: Path, target: Path) -> list[str]:
    return plugin_tests(folder, target, "tests/test_durations.py")


#: More misspelt keys and slips in the plugin's files, each of which must fail the stage of the plugin's check that is there to
#: catch it, and only that stage runs (the folder is the clean one's copy, with the one file written again): the file, the text
#: to replace and what replaces it, the stage, and what its failure says. The first of a kind is the one ``planted`` plants for
#: the count of 5.
PLUGIN_PLANTS: dict[str, Plant] = {
    "the mime.assets entry point's key": Plant("pyproject.toml", '"mime.assets"', '"mime.assts"', registry_stage, "test_registry",
                                               rebuilds=True),
    "the planted error that does nothing": Plant("tests/plants.py", "jnp.clip(real(force, gain), -SATURATION, SATURATION)",
                                                 "real(force, gain)", planted_control_stage,
                                                 "test_the_planted_error_is_caught_every_time"),
    "the variable that switches the planted error on, misspelt": Plant(
        "tests/plants.py", 'PLANT_VARIABLE = "MIME_PLANT_ERROR"', 'PLANT_VARIABLE = "MIME_PLANT_ERRRO"', planted_run_failures,
        "expected only"),
    "an import of mime in the engine": Plant(f"src/{PLUGIN_PACKAGE}/engine/load.py", "from jax import Array",
                                             "from jax import Array\nimport mime", engine_imports_stage, "test_engine_imports"),
    "an importlib.import_module of mime in the engine": Plant(
        f"src/{PLUGIN_PACKAGE}/engine/load.py", "from jax import Array",
        'from importlib import import_module\nfrom jax import Array\n\nimport_module("mime.units")', engine_imports_stage,
        "test_engine_imports"),
    "a __import__ of mime in the engine": Plant(f"src/{PLUGIN_PACKAGE}/engine/load.py", "from jax import Array",
                                                'from jax import Array\n\n__import__("mime")', engine_imports_stage,
                                                "test_engine_imports"),
    "an import of mime in the fit package": Plant(f"src/{PLUGIN_PACKAGE}/fit/__init__.py", '"""A second package',
                                                  'import mime\n"""A second package', engine_imports_stage, "test_engine_imports"),
    "an absolute import of the package by its own name in the adapter": Plant(
        f"src/{PLUGIN_PACKAGE}/entries/__init__.py", "from .example_load import ExampleLoad",
        f"from {PLUGIN_PACKAGE}.entries.example_load import ExampleLoad", relative_imports_stage, "test_relative_imports"),
    "an absolute import of the package by its own name in the example entry": Plant(
        f"src/{PLUGIN_PACKAGE}/entries/example_load.py", "from ..engine import load as law",
        f"from {PLUGIN_PACKAGE}.engine import load as law", relative_imports_stage, "test_relative_imports"),
    "a relative import of the adapter in the engine": Plant(f"src/{PLUGIN_PACKAGE}/engine/load.py", "from jax import Array",
                                                            "from jax import Array\nfrom ..entries import ExampleLoad",
                                                            engine_imports_stage, "test_engine_imports"),
    "a relative import of the adapter's folder in the fit package": Plant(
        f"src/{PLUGIN_PACKAGE}/fit/__init__.py", '"""A second package', 'from .. import entries\n"""A second package',
        engine_imports_stage, "test_engine_imports"),
    "an aliased import_module of the package by its own name in the adapter": Plant(
        f"src/{PLUGIN_PACKAGE}/entries/__init__.py", "from .example_load import ExampleLoad",
        f'from importlib import import_module as im\n\nExampleLoad = im("{PLUGIN_PACKAGE}.entries.example_load").ExampleLoad',
        relative_imports_stage, "test_relative_imports"),
    "an allowlist row that names no test in the report": Plant("scripts/check_durations.py",
                                                               "for module, name in sorted(set(reasons) - set(seconds)):",
                                                               "for module, name in []:", durations_stage, "test_durations"),
    "a duration limit a hundred times too loose": Plant("scripts/check_durations.py", "if took <= limit:",
                                                        "if took <= limit * 100:", durations_stage, "test_durations"),
    "an allowlist line that needs no reason": Plant("scripts/check_durations.py", "if not separator or not reason.strip():", "if False:",
                                                    durations_stage, "test_durations"),
    "a command of the README that is misspelt": Plant("README.md", "python examples/example.py", "python examples/exampel.py",
                                                      readme_use, "README"),
}


def plugin_planted_failures(clean: Path, target: Path, plant: Plant, work: Path) -> list[str]:
    """``clean`` copied into ``work`` with ``plant`` written into its one file, and the stage that must catch it run on the
    copy (against ``target``, or a wheel of the copy unpacked again when the slip is in what the wheel holds): the stage's
    failures. The clean folder is never touched, so one scaffold serves every plant."""
    folder = work / clean.name
    shutil.copytree(clean, folder, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    text = (folder / plant.path).read_text(encoding="utf-8")
    assert plant.old in text, f"the text to plant over is not in {plant.path}: the plant would do nothing"
    (folder / plant.path).write_text(text.replace(plant.old, plant.new, 1), encoding="utf-8")
    return plant.stage(folder, unpacked(folder, work / "target") if plant.rebuilds else target)


@dataclass(frozen=True)
class PluginBase:
    """A plugin scaffolded with ``--yes``, built and unpacked once for a module's tests: the folder, the target it is unpacked
    into (only subprocesses see it), the failures of the stages up to its listing, and what this process's registry was before."""

    folder: Path
    target: Path
    failures: list[str]
    before: tuple[list[str], list[str], bool]


@dataclass(frozen=True)
class PluginClean:
    """The clean plugin's whole check, run once: every failure (empty when it passes), and this process's registry before the
    scaffold and after the check."""

    failures: list[str]
    before: tuple[list[str], list[str], bool]
    after: tuple[list[str], list[str], bool]


def entry_points() -> list[str]:
    return sorted(f"{each.name}={each.value}" for each in importlib.metadata.entry_points(group="mime.entries"))


def environment() -> tuple[list[str], list[str], bool]:
    """What every later test in this process would read of the plugin: the ``mime.entries`` entry points, the import path
    and whether the plugin's package is imported."""
    return entry_points(), list(sys.path), any(name.startswith("scafx_") for name in sys.modules)


def plugin_base(root: Path) -> PluginBase:
    """Scaffold the plugin with ``--yes`` in ``root``, unpack its wheel and run the stages up to the listing."""
    before = environment()
    folder = root / "scafx-probes"
    scaffold.scaffold("plugin", folder, yes=True, sets=[f"plugin.namespace={NAMESPACE}"], say=lambda line: None)
    project, failures = plugin_static(folder)
    target = root / "target"
    return PluginBase(folder, target, failures or plugin_listing(folder, target, project), before)


def plugin_clean(base: PluginBase) -> PluginClean:
    """The rest of the clean plugin's check, over a base: its lanes (none run when an earlier stage failed)."""
    failures = base.failures or plugin_lanes(base.folder, base.target)
    return PluginClean(failures, base.before, environment())


def planted(kind: str) -> Callable[[pytest.MonkeyPatch], None]:
    """Plant a misspelt key in ``kind``'s template (a bundle's and a study's text comes from push's one writer, whose
    output the plant edits), by a patch on the module that holds it."""
    def edit(patch: pytest.MonkeyPatch) -> None:
        if kind in ("bundle", "study"):
            original = _draft.manifest_text
            key = "\nlicence = " if kind == "bundle" else "\ntitle = "
            patch.setattr(_draft, "manifest_text", lambda *a, **k: original(*a, **k).replace(key, key.replace("e", "x", 1)))
            return
        name, old, new = {"experiment": ("experiment.py.txt", "experiment.licence(", "experiment.licnce("),
                          "block": ("mime.toml.txt", "\nlicence = ", "\nlicnce = "),
                          "plugin": ("pyproject.toml.txt", '"mime.entries"', '"mime.entrys"')}[kind]
        real = _templates.template
        patch.setattr(_templates, "template", lambda k, n: real(k, n).replace(old, new) if (k, n) == (kind, name)
                      else real(k, n))
    return edit


__all__ = ["KINDS", "NAMESPACE", "PLUGIN_PLANTS", "PluginBase", "PluginClean", "check_of", "child_environment", "entry_points",
           "environment", "failures_of", "listed", "made", "planted", "plugin_base", "plugin_clean",
           "plugin_planted_failures", "unpacked", "verified"]


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
