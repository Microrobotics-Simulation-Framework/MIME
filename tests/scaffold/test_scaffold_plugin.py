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
import json
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
    assert files_of(scaffolded(tmp_path / "a"))["LICENSE"].lstrip().startswith("Apache License")


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


def test_gitignore_names_no_benchmark_script_and_ignores_any_saved_case(tmp_path: Path):
    """The case's folder name was written in three files and went stale when the port renamed the case: the ignore is any
    folder below ``benchmarks/``, and names no script. Read by git itself, not by the text."""
    root = scaffolded(tmp_path)
    assert "build_case" not in (root / ".gitignore").read_text(encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for folder in ("case", "strip"):
        (root / "benchmarks" / folder).mkdir()  # git matches a pattern ending in a slash against folders that exist

    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=root, check=False).returncode == 0

    assert ignored("benchmarks/case/mime.toml") and ignored("benchmarks/strip/experiment.json") and ignored("junit.xml")
    assert not ignored("benchmarks/build_case.py") and not ignored("benchmarks/mime.toml")


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
