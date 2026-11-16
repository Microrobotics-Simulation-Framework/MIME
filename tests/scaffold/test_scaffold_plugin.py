"""The plugin kind's layout (P7b): what ``mime scaffold plugin`` writes, held file by file, without installing it.

The control (``test_scaffold_control.py``) runs the plugin; these hold what a run cannot see:

- the layout is the approved list, and nothing is left out or added (a file dropped from the table is caught);
- the namespace is filled in six places from one value, and no value is left as ``${...}`` or ``$$``;
- the record's sha256 and size are the data's, and fail to match when the data is changed after the filling;
- the creators and the content licence come from the person's ``defaults.toml``, for the citation, the records and the
  benchmark's ``mime.toml``; none of them, and the files hold a placeholder to replace;
- no file holds a machine path, a home folder or a token;
- a licence whose text MIME does not ship gets a file that says so, and Apache-2.0 gets its text;
- the engine's import rule is MIME's own at the scaffold: the AST test in the plugin is the one that holds it, over the
  engine and the second MIME-free package (``fit/``);
- the follow-ups of the clot port (P7c): the typing route (strict, with the unknown-type rules off for the packages that
  import no ``mime`` and for them alone), the duration guard in the default lane, the sdist's ``scripts`` and ``docs``,
  no stale case name in ``.gitignore``, the relative-import rule, the float64 reference advice.
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from typing import Any
from pathlib import Path

import pytest
import yaml
from scaffold_rig import NAMESPACE

from mime import scaffold
from mime.push import _defaults

PACKAGE = "scafx_probes"
#: Every file a plugin gets, below its folder. The approved list of the P7 plan, with the files of its package.
EXPECTED = {
    ".github/workflows/ci.yml", ".gitignore", "CITATION.cff", "LICENSE", "README.md", "pyproject.toml",
    "pyrightconfig.json",
    "benchmarks/build_case.py", "benchmarks/mime.toml", "examples/example.py",
    "scripts/check_durations.py",
    f"src/{PACKAGE}/__init__.py", f"src/{PACKAGE}/engine/__init__.py", f"src/{PACKAGE}/engine/load.py",
    f"src/{PACKAGE}/fit/__init__.py",
    f"src/{PACKAGE}/entries/__init__.py", f"src/{PACKAGE}/entries/_vocabulary.py",
    f"src/{PACKAGE}/entries/example_load.py",
    f"src/{PACKAGE}/records/__init__.py", f"src/{PACKAGE}/records/records.json",
    f"src/{PACKAGE}/records/data/example.json",
    "tests/assert_types.py", "tests/conftest.py", "tests/plants.py", "tests/test_engine_imports.py",
    "tests/import_rules.py", "tests/test_relative_imports.py", "tests/test_durations.py", "tests/duration_allowlist.txt",
    "tests/test_nodes.py", "tests/test_registry.py", "tests/test_validity.py",
    "tests/engine/test_properties.py", "tests/verification/test_example.py",
}


#: A home folder, a drive letter, a scratch or work folder: what a path on someone's machine looks like (the words are split so
#: that this file holds none of them itself).
MACHINE_PATH = "/ho" "me/|/Us" "ers/|[A-Za-z]:\\\\|coord" "_work|/t" "mp/"


def scaffolded(tmp_path: Path, *sets: str) -> Path:
    path = tmp_path / "scafx-probes"
    scaffold.scaffold("plugin", path, yes=True, sets=[f"plugin.namespace={NAMESPACE}", *sets], say=lambda line: None)
    return path


def files_of(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in sorted(root.rglob("*"))
            if path.is_file()}


def test_the_plugin_writes_the_approved_layout_and_nothing_else(tmp_path: Path):
    assert set(files_of(scaffolded(tmp_path))) == EXPECTED


def test_the_namespace_is_filled_in_the_six_places_it_appears(tmp_path: Path):
    held = files_of(scaffolded(tmp_path))
    project = tomllib.loads(held["pyproject.toml"])["project"]["entry-points"]
    assert project["mime.entries"] == {NAMESPACE: f"{PACKAGE}.entries"}  # 1
    assert project["mime.assets"] == {NAMESPACE: f"{PACKAGE}.records"}  # 2
    assert f'NAMESPACE, VERSION = "{NAMESPACE}", "0.1.0"' in held[f"src/{PACKAGE}/entries/_vocabulary.py"]  # 3
    assert f'name = "{NAMESPACE}.example_load"' in held[f"src/{PACKAGE}/entries/example_load.py"]  # 4
    assert json.loads(held[f"src/{PACKAGE}/records/records.json"])["assets"][0]["name"] == f"{NAMESPACE}_example_record"  # 5
    assert f"`{NAMESPACE}.example_load:probe`" in held["examples/example.py"]  # 6
    assert f"package:{NAMESPACE}/data/example.json" in held[f"src/{PACKAGE}/records/records.json"]


def test_the_naming_rule_is_written_as_comments(tmp_path: Path):
    text = files_of(scaffolded(tmp_path))[f"src/{PACKAGE}/entries/_vocabulary.py"]
    assert f"`{NAMESPACE}.example_load`" in text and f"`{NAMESPACE}_example_record`" in text
    assert f"`{NAMESPACE}.example_load:probe`" in text
    assert "with a dot" in text and "with an underscore" in text and "with a colon" in text


def test_no_placeholder_or_escape_is_left_in_any_file(tmp_path: Path):
    for path, text in files_of(scaffolded(tmp_path)).items():
        assert "${" not in text.replace("${{", "") and "$$" not in text, path


def test_the_records_sha256_and_size_are_the_datas(tmp_path: Path):
    root = scaffolded(tmp_path)
    data = (root / f"src/{PACKAGE}/records/data/example.json").read_bytes()
    record = json.loads((root / f"src/{PACKAGE}/records/records.json").read_text())["assets"][0]
    assert record["sha256"] == hashlib.sha256(data).hexdigest() and record["size"] == len(data)
    # The control of that test: a data file changed after the filling no longer matches, so the plugin's own test would fail.
    (root / f"src/{PACKAGE}/records/data/example.json").write_bytes(data + b" ")
    changed = (root / f"src/{PACKAGE}/records/data/example.json").read_bytes()
    assert record["sha256"] != hashlib.sha256(changed).hexdigest()


def test_the_creators_and_the_content_licence_come_from_the_defaults(tmp_path: Path):
    _defaults.write_defaults({"creators": [{"family": "Carberry", "given": "Josiah", "orcid": "0000-0002-1825-0097"}]},
                             licences={"content": "CC0-1.0", "code": "MIT"}, namespace=NAMESPACE)
    held = files_of(scaffolded(tmp_path))
    cff = yaml.safe_load(held["CITATION.cff"])
    assert cff["authors"] == [{"family-names": "Carberry", "given-names": "Josiah",
                               "orcid": "https://orcid.org/0000-0002-1825-0097"}] and cff["license"] == "MIT"
    assert json.loads(held[f"src/{PACKAGE}/records/records.json"])["assets"][0]["licence"] == "CC0-1.0"
    assert json.loads(held[f"src/{PACKAGE}/records/records.json"])["assets"][0]["credit"] == "Josiah Carberry"
    benchmark = tomllib.loads(held["benchmarks/mime.toml"])
    assert benchmark["bundle"]["licence"] == "CC0-1.0"
    assert benchmark["record"]["creators"] == [{"family": "Carberry", "given": "Josiah", "orcid": "0000-0002-1825-0097"}]
    assert 'experiment.licence("CC0-1.0")' in held["examples/example.py"]


def test_with_no_defaults_the_creator_is_a_placeholder_the_files_say_to_replace(tmp_path: Path):
    held = files_of(scaffolded(tmp_path))
    assert yaml.safe_load(held["CITATION.cff"])["authors"] == [{"family-names": "Author", "given-names": "A."}]
    assert "replace it" in held["CITATION.cff"]
    assert tomllib.loads(held["benchmarks/mime.toml"])["record"].get("creators", []) == []


def test_no_file_holds_a_machine_path(tmp_path: Path):
    for path, text in files_of(scaffolded(tmp_path)).items():
        assert not re.search(MACHINE_PATH, text), path


def test_a_licence_mime_does_not_ship_gets_a_file_that_says_so(tmp_path: Path):
    held = files_of(scaffolded(tmp_path, "plugin.licence=MIT"))
    assert held["LICENSE"].startswith("This plugin's code is licensed under MIT") and "full text" in held["LICENSE"]
    assert files_of(scaffolded(tmp_path / "a", "plugin.licence=Apache-2.0"))["LICENSE"].lstrip().startswith("Apache License")
    assert files_of(scaffolded(tmp_path / "b"))["LICENSE"].startswith("No licence has been chosen"), \
        "scaffold chooses no licence for the plugin's code"


def test_the_ci_names_the_lanes_and_a_label_gates_the_slow_one(tmp_path: Path):
    text = files_of(scaffolded(tmp_path))[".github/workflows/ci.yml"]
    ci = yaml.safe_load(text)
    assert set(ci["jobs"]) == {"default", "slow", "release", "typing"}
    assert "contains(github.event.pull_request.labels.*.name, 'slow')" in ci["jobs"]["slow"]["if"]
    assert ci["jobs"]["default"]["if"] == "github.event.action != 'labeled'"  # a label does not re-run the default matrix
    assert "labeled" in ci[True]["pull_request"]["types"], "the slow job runs on the label event, so the trigger names it"
    assert any(step.get("run", "").startswith("uv run --no-sync pytest -m slow") for step in ci["jobs"]["slow"]["steps"])
    assert "schedule" in ci[True] and ci[True]["schedule"][0]["cron"]  # PyYAML reads the key `on` as True
    release = " ".join(step.get("run", "") for step in ci["jobs"]["release"]["steps"])
    assert re.search(r"MIME@release/v\d+\.\d+\.0", release)
    assert "mime assets audit dist/*" in text and "mime registry --json" in text and "bash ci" in text
    assert 'pyright --pythonpath "$(which python)"' in text


def test_pyproject_marks_slow_and_deselects_it_by_default(tmp_path: Path):
    pytest_options = tomllib.loads(files_of(scaffolded(tmp_path))["pyproject.toml"])["tool"]["pytest"]["ini_options"]
    assert pytest_options["addopts"] == ["-ra", "--strict-markers", "-m", "not slow"]
    assert pytest_options["markers"][0].startswith("slow:")


def test_the_pin_to_one_core_note_and_the_jq_line_are_in_the_readme(tmp_path: Path):
    readme = files_of(scaffolded(tmp_path))["README.md"]
    assert "JAX_PLATFORMS=cpu taskset -c 0" in readme and "Pin to one core" in readme
    assert f'mime registry --json | jq \'.entries[] | select(.name | startswith("{NAMESPACE}."))\'' in readme


@pytest.mark.parametrize("path", ["tests/engine/test_properties.py", "tests/plants.py"])
def test_the_property_skeleton_tells_the_author_the_rules_that_cost_time(path: str, tmp_path: Path):
    text = files_of(scaffolded(tmp_path))[path]
    expected = ("detection_rate.py", "EVERY run") if "plants" in path else ("Never derandomise", "scale it came from",
                                                                         "dtype the code computes in")
    assert all(each in text for each in expected), path


def test_the_sdist_carries_scripts_and_docs_and_the_tests_find_scripts(tmp_path: Path):
    options = tomllib.loads(files_of(scaffolded(tmp_path))["pyproject.toml"])["tool"]
    assert {"scripts", "docs"} <= set(options["hatch"]["build"]["targets"]["sdist"]["include"])
    assert "scripts" in options["pytest"]["ini_options"]["pythonpath"]  # tests import scripts/check_durations.py


def test_gitignore_names_no_benchmark_script_and_ignores_what_a_benchmark_generates_and_not_an_authors_own_folders(tmp_path: Path):
    """The case's folder name was written in three files and went stale when the port renamed the case, so the ignore names no
    case and no script; but it must not swallow an author's own folder below ``benchmarks/`` (data, a table, notes) either:
    it ignores the files a saved experiment is made of, and the ``mime.toml`` the script copies beside them. Read by git itself,
    not by the text."""
    root = scaffolded(tmp_path)
    assert "build_case" not in (root / ".gitignore").read_text(encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for folder in ("case", "strip/scene", "data", "notes"):
        (root / "benchmarks" / folder).mkdir(parents=True)  # git matches a pattern ending in a slash against folders that exist

    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=root, check=False).returncode == 0

    generated = ("benchmarks/case/experiment.mime.yaml", "benchmarks/case/mime.lock", "benchmarks/case/reference.npz",
                 "benchmarks/case/mime.toml", "benchmarks/strip/scene/scene.usda", "benchmarks/strip/experiment.mime.yaml", "junit.xml")
    authors_own = ("benchmarks/build_case.py", "benchmarks/mime.toml", "benchmarks/data/table.csv", "benchmarks/data/reference.json",
                   "benchmarks/notes/README.md", "benchmarks/case/notes.md", "benchmarks/strip/fixture.json")
    assert [path for path in generated if not ignored(path)] == []
    assert [path for path in authors_own if ignored(path)] == []


def test_the_default_lane_runs_the_duration_guard_on_a_junit_report(tmp_path: Path):
    steps = [step.get("run", "") for step in yaml.safe_load(files_of(scaffolded(tmp_path))[".github/workflows/ci.yml"])
             ["jobs"]["default"]["steps"]]
    assert "uv run --no-sync pytest --junitxml=junit.xml" in steps
    assert steps.index("uv run --no-sync python scripts/check_durations.py junit.xml") > steps.index(
        "uv run --no-sync pytest --junitxml=junit.xml")


def test_the_second_mime_free_package_is_covered_by_the_import_rule(tmp_path: Path):
    held = files_of(scaffolded(tmp_path))
    assert 'MIME_FREE = ("engine", "fit")' in held["tests/test_engine_imports.py"]
    assert f"src/{PACKAGE}/fit/__init__.py" in held and "imports no `mime`" in held[f"src/{PACKAGE}/fit/__init__.py"]


def test_the_package_imports_itself_relatively_in_the_template(tmp_path: Path):
    """The template's own modules: no absolute import of the package, in any file of ``src/`` (the plugin's own test holds the
    rule from here on; this is the template keeping it)."""
    for path, text in files_of(scaffolded(tmp_path)).items():
        if path.startswith(f"src/{PACKAGE}/") and path.endswith(".py"):
            assert not re.search(rf"^\s*(from|import) {PACKAGE}\b", text, re.MULTILINE), path
    assert "from ..engine import load as law" in files_of(scaffolded(tmp_path / "again"))[f"src/{PACKAGE}/entries/example_load.py"]


def test_the_readme_says_when_a_float64_only_reference_is_fine(tmp_path: Path):
    readme = " ".join(files_of(scaffolded(tmp_path))["README.md"].split())
    assert "reference the engine is computed *against*" in readme and "float64 alone" in readme
    assert "Never let the law's own path be float64-only" in readme


def test_the_readme_does_not_suggest_a_suite_wide_x64_switch(tmp_path: Path):
    """CLAUDE.md: a float64-only rig says so by a declared guard, never by a blanket ``JAX_ENABLE_X64=1``. The README names the
    scoped route (``enable_x64`` in that test), says the law's own path is tested in both dtypes, and no longer offers
    ``tests/conftest.py`` as the place that turns x64 on; the template's conftest does not turn it on either."""
    held = files_of(scaffolded(tmp_path))
    readme = " ".join(held["README.md"].split())
    assert "jax.experimental.enable_x64" in readme and "never as a suite-wide switch" in readme
    assert "The law's own path is tested in both dtypes" in readme
    assert "or `tests/conftest.py` does" not in readme
    assert not re.search(r"enable_x64|JAX_ENABLE_X64|jax_enable_x64", held["tests/conftest.py"])


# -- the import rules and the duration guard, run as the plugin's own tests read them -----------------------------------

def template_module(root: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """``tests/<name>.py`` of the scaffolded plugin imported here (its ``import_rules`` beside it found), for a call of its
    functions on a planted source. The modules are dropped from ``sys.modules`` afterwards by the monkeypatch."""
    monkeypatch.syspath_prepend(str(root / "tests"))
    monkeypatch.delitem(sys.modules, "import_rules", raising=False)
    spec = importlib.util.spec_from_file_location(f"scaffolded_{name}", root / "tests" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delitem(sys.modules, "import_rules", raising=False)
    return module


#: Each way into the adapter that a MIME-free package can spell, written as the engine (or fit) would write it. The old rule
#: read ``mime`` by name only, and all of these pull ``mime`` in through the adapter without naming it.
ADAPTER_IMPORTS = ("from ..entries import ExampleLoad", "from .. import entries", "from ..entries.example_load import ExampleLoad",
                   "from ..records import data", "import PKG.entries", "from PKG.entries import ExampleLoad",
                   "from PKG import entries", "from ... import entries", 'import importlib\nimportlib.import_module("PKG.entries")',
                   'import importlib\nimportlib.import_module(".entries", package="PKG")',
                   'import importlib\nimportlib.import_module("..entries", package=__package__)')


@pytest.mark.parametrize("folder", ["engine", "fit"])
def test_the_mime_free_rule_refuses_every_spelling_of_an_import_of_the_adapter(folder: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The hole: the engine and fit imported no ``mime`` by name, yet ``from ..entries import ExampleLoad`` (the natural spelling
    once a package imports itself relatively), ``from .. import entries`` and the rest brought it in through the adapter, and both
    AST tests passed. Each spelling is refused from ``engine/`` and from ``fit/`` (and from a module one folder deeper), by the
    plugin's own function on the scaffolded package; the honest imports between the MIME-free packages are not."""
    root = scaffolded(tmp_path)
    module = template_module(root, "test_engine_imports", monkeypatch)
    base = f"{PACKAGE}.{folder}"
    for planted in ADAPTER_IMPORTS:
        source = planted.replace("PKG", PACKAGE)
        here = f"{base}.sub" if planted.startswith("from ... ") else base
        assert module.adapter_imports(source, here), f"{source!r} was not refused in {here}"
    for honest in ("from . import load", "from ..engine import load", "from ..fit import model", "from .. import fit, engine",
                   "import jax\nfrom jax import numpy", f"import {PACKAGE}_other", "import mimesis"):
        assert not module.adapter_imports(honest, base) and not module.mime_imports(honest), f"{honest!r} was refused"


@pytest.mark.parametrize("path", ["engine/leak.py", "fit/leak.py"])
@pytest.mark.parametrize("spelling", ["from ..entries import ExampleLoad", "from .. import entries",
                                      "from ..entries.example_load import ExampleLoad"])
def test_a_planted_relative_import_of_the_adapter_fails_the_plugins_own_test(spelling: str, path: str, tmp_path: Path):
    """The end to end: the plugin's own ``tests/test_engine_imports.py`` run by pytest on the scaffolded package, clean first
    (it passes), then with one module of ``engine/`` or ``fit/`` importing the adapter: it fails, and names the module."""
    root = scaffolded(tmp_path)
    run = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_engine_imports.py",
           "tests/test_relative_imports.py"]
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")}
    clean = subprocess.run(run, cwd=root, capture_output=True, text=True, env=environment, check=False)
    assert clean.returncode == 0, clean.stdout[-1500:]
    (root / "src" / PACKAGE / path).write_text(spelling + "\n", encoding="utf-8")
    planted = subprocess.run(run, cwd=root, capture_output=True, text=True, env=environment, check=False)
    assert planted.returncode == 1 and path in planted.stdout and "test_the_mime_free_packages_import_no_mime" in planted.stdout, \
        planted.stdout[-1500:]


def test_a_package_root_that_imports_mime_is_the_adapter_and_a_root_module_that_does_not_is_not(tmp_path: Path,
                                                                                                 monkeypatch: pytest.MonkeyPatch):
    root = scaffolded(tmp_path)
    module = template_module(root, "test_engine_imports", monkeypatch)
    package = root / "src" / PACKAGE
    (package / "_constants.py").write_text("PI = 3.14159\n", encoding="utf-8")
    assert not module.adapter_imports("from .. import _constants\nfrom .._constants import PI", f"{PACKAGE}.engine")
    (package / "_constants.py").write_text("from mime.units import N\n", encoding="utf-8")
    assert module.adapter_imports("from .._constants import PI", f"{PACKAGE}.engine")
    (package / "_constants.py").unlink()
    assert not module.adapter_imports("from .. import engine", f"{PACKAGE}.fit")
    (package / "__init__.py").write_text("import mime\n", encoding="utf-8")
    assert module.adapter_imports("from .. import engine", f"{PACKAGE}.fit")


#: What the import tests read that the first version missed (the review's list), each as a source that imports ``PKG``.
CHEAP_STATIC = ('from importlib import import_module as im\nim("PKG.x")', "import importlib\nimportlib.import_module(name='PKG.x')",
                "import importlib\nimportlib.import_module('.engine', package='PKG')",
                "import importlib\nimportlib.import_module('.engine', 'PKG')", 'import importlib\nimportlib.import_module(f"PKG.x")',
                "from importlib import import_module\nload = import_module\nload('PKG.x')", "load = __import__\nload('PKG')")
#: What a static check cannot see: the docstrings say so, and these hold that they are not found.
NOT_SEEN = ('import importlib\nimportlib.import_module("P" + "KG")', 'import importlib.util\nimportlib.util.find_spec("PKG.x")',
            'import sys\nsys.modules["PKG.x"]', 'import runpy\nrunpy.run_module("PKG.x")', 'exec("import PKG")',
            'import importlib\nimportlib.import_module(f"PKG.{x}")')


@pytest.mark.parametrize("test", ["test_relative_imports", "test_engine_imports"])
def test_the_import_tests_read_the_cheap_static_forms_and_say_plainly_what_they_cannot_see(test: str, tmp_path: Path,
                                                                                         monkeypatch: pytest.MonkeyPatch):
    """The relative-import test's docstring said it read ``importlib.import_module`` and ``__import__`` calls; an aliased name,
    ``name=``, the ``package=`` form and a placeholder-free f-string were missed, so the docstring overstated. The cheap static
    ones are now read (both tests share ``import_rules``); the rest are listed in each docstring, and held not found."""
    root = scaffolded(tmp_path)
    module = template_module(root, test, monkeypatch)
    package = PACKAGE if test == "test_relative_imports" else "mime"
    found = module.absolute_self_imports if test == "test_relative_imports" else module.mime_imports
    if test == "test_relative_imports":
        monkeypatch.setattr(module, "NAME", package)
    for planted in CHEAP_STATIC:
        assert found(planted.replace("PKG", package)), f"{planted!r} was not found"
    for unseen in NOT_SEEN:
        assert not found(unseen.replace("PKG", package)), f"{unseen!r} was found: the docstring's limit would be wrong"
    docstring = " ".join((module.__doc__ or "").split())
    for limit in ("find_spec", "sys.modules", "runpy", "exec", "computed"):
        assert limit in docstring, f"{test}'s docstring does not name {limit} as what it cannot see"
    assert "aliased" in docstring and "package=" in docstring


def test_the_duration_guard_fails_a_stale_row_and_takes_an_id_with_a_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = scaffolded(tmp_path)
    monkeypatch.syspath_prepend(str(root / "scripts"))
    monkeypatch.delitem(sys.modules, "check_durations", raising=False)
    guard = importlib.import_module("check_durations")
    try:
        report = ('<testsuites><testsuite><testcase classname="tests.test_a" name="test_fast" time="0.2"/>'
                  '<testcase classname="tests.test_a" name="test_p[a # b]" time="45.0"/></testsuite></testsuites>')
        failures, _ = guard.judge(report, "tests/test_a.py::test_gone # kept: it was slow once\n"
                                          "tests/test_a.py::test_p[a # b] # kept: the id holds a hash\n", limit=30.0)
        assert len(failures) == 1 and "test_gone" in failures[0] and "not in the report" in failures[0]
    finally:
        sys.modules.pop("check_durations", None)


# -- the typing check ---------------------------------------------------------------------------------------------------

#: pyright's PyPI package runs on node: the system's if there is one, else it downloads one from nodejs.org on first use. A
#: runner has node (GitHub's ``ubuntu-latest``); a machine without it would fail these tests on the network, not on the
#: plugin's types, so they are skipped there, with this reason, rather than made to depend on a download.
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="pyright's package would download node from nodejs.org: "
                                "no node on this machine's path, and these tests do not depend on the network")


def pyright_report(root: Path) -> dict[str, Any]:
    """pyright's JSON for the scaffolded plugin, run as its CI runs it: in the plugin's folder, the environment given by
    ``--pythonpath`` (this suite's own, which holds MIME), whether or not ``pyrightconfig.json``'s ``.venv`` exists (it does
    not here: pyright says so on stderr and goes on)."""
    pyright = Path(sys.executable).parent / "pyright"
    done = subprocess.run([str(pyright), "--outputjson", "--pythonpath", sys.executable], cwd=root, capture_output=True,
                          text=True, check=False)
    assert done.returncode in (0, 1), done.stderr[-400:]
    return json.loads(done.stdout)


@needs_node
def test_the_scaffolded_plugin_type_checks_strictly_at_zero_errors(tmp_path: Path):
    report = pyright_report(scaffolded(tmp_path))
    summary = report["summary"]
    assert summary["errorCount"] == 0, report["generalDiagnostics"]
    assert summary["filesAnalyzed"] >= 8  # the package's modules and the assert_type file: not a run over nothing


@needs_node
def test_a_changed_type_fails_the_assert_type_file(tmp_path: Path):
    """The control of the test above: the `assert_type` file can fail. A setting declared `float` and asserted `int` is
    reported, in that file, every time."""
    root = scaffolded(tmp_path)
    path = root / "tests" / "assert_types.py"
    text = path.read_text(encoding="utf-8")
    assert "assert_type(entry.gain, float)" in text
    path.write_text(text.replace("assert_type(entry.gain, float)", "assert_type(entry.gain, int)"), encoding="utf-8")
    report = pyright_report(root)
    messages = [(each["file"], each["message"]) for each in report["generalDiagnostics"]]
    assert any(file.endswith("assert_types.py") and "assert_type" in message for file, message in messages), messages


@needs_node
def test_the_unknown_type_rules_are_off_for_the_engine_and_fit_alone(tmp_path: Path):
    """The typing route's control. An unannotated function, in the adapter, is reported (strict stays strict there); in the
    engine and in ``fit`` the same function is not (jaxlib has no stubs, so untyped JAX reads as unknown), and every other
    strict rule still is: an unused variable in each is reported. The route is both halves at once, every run."""
    root = scaffolded(tmp_path)
    unannotated = "def untyped(a, b):\n    return a + b\n\n\ndef unused() -> None:\n    value = 1\n"
    for folder in ("engine", "fit", "entries"):
        (root / "src" / PACKAGE / folder / "_planted.py").write_text(unannotated, encoding="utf-8")
    rules: dict[str, set[str]] = {}
    for each in pyright_report(root)["generalDiagnostics"]:
        rules.setdefault(each["file"].split("/")[-2], set()).add(each["rule"])
    assert rules["engine"] == {"reportUnusedVariable"}, rules
    assert rules["fit"] == {"reportUnusedVariable"}, rules
    assert {"reportUnusedVariable", "reportMissingParameterType"} <= rules["entries"], rules
    assert set(rules) == {"engine", "fit", "entries"}, "nothing else is reported"


def test_the_pyright_config_relaxes_the_unknown_type_rules_for_the_mime_free_packages_only(tmp_path: Path):
    config = json.loads(files_of(scaffolded(tmp_path))["pyrightconfig.json"])
    assert config["typeCheckingMode"] == "strict"
    relaxed = {each["root"]: {rule for rule, value in each.items() if value is False} for each in config["executionEnvironments"]}
    assert set(relaxed) == {f"src/{PACKAGE}/engine", f"src/{PACKAGE}/fit"}, "the adapter and the records are not relaxed"
    assert all(rules == {"reportUnknownArgumentType", "reportUnknownMemberType", "reportUnknownVariableType",
                         "reportUnknownParameterType", "reportUnknownLambdaType", "reportMissingParameterType"}
               for rules in relaxed.values())


def test_the_typing_extra_pins_the_pyright_this_suite_checks_the_template_with(tmp_path: Path):
    """The template's pyright pin is the release the test above ran, not a number written once and left: a bump of MIME's
    own pin fails here until the template follows."""
    from importlib.metadata import version

    extras = tomllib.loads(files_of(scaffolded(tmp_path))["pyproject.toml"])["project"]["optional-dependencies"]
    assert extras["typing"] == [f"pyright=={version('pyright')}"]
