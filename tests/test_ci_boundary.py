"""Which lanes a pull request runs (``scripts/ci_boundary.py``, the ``shards`` job of ``.github/workflows/ci.yml``,
CONTRIBUTING.md, "CI: which lanes a pull request runs"; a decision of 2026-11-08 the maintainer can reverse).

``default`` on Python 3.13, ``maddening-main`` and ``openmaterica-main`` run on a push, a manual run and the nightly run,
and on a full pull request (the label ``ready``) only when its diff touches the pins, CI itself, or an adapter of MIME to
that upstream. This file holds the paths in one place against the repository: every path exists, the scripts the workflows
call are the scripts the filter names (a script a workflow comes to call, and the filter lacks, fails here), the adapters
are the files that import their upstream and MADDENING is not so widely imported that the adapters stand for little, and
the workflows keep the three lanes out of the compilation cache and their duration gate a warning. The workflows' side
of the boundary, the jobs each scenario starts, is played in ``tests/test_ci_ready_label.py``."""

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_ci_boundary", ROOT / "scripts" / "ci_boundary.py")
assert spec is not None and spec.loader is not None
boundary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = boundary  # a dataclass finds its module here
spec.loader.exec_module(boundary)
CI = ROOT / ".github" / "workflows" / "ci.yml"
SRC = ROOT / "src" / "mime"
PY313 = [{"python-version": "3.13"}]
PY313_314 = [{"python-version": "3.13"}, {"python-version": "3.14"}]
UPSTREAM_IMPORT = {"maddening": re.compile(r"^\s*(?:from|import) maddening\b", re.M),
                   "materica": re.compile(r"^\s*(?:from|import) materica\b", re.M)}


@pytest.mark.parametrize("event, full, changed, lanes, exclude, lane_exclude", [
    # Every lane on a push and a manual run, whatever the mode or the files.
    ("push", True, [], (True, True, True), [], []),
    ("workflow_dispatch", True, [], (True, True, True), [], []),
    # A pull request without ready: none, and only Python 3.12 runs its shards; 3.14's job stays, to refuse.
    ("pull_request", False, ["uv.lock"], (False, False, False), PY313_314, PY313),
    ("pull_request", False, [], (False, False, False), PY313_314, PY313),
    # A full pull request that touches none of them.
    ("pull_request", True, ["docs/a.md", "tests/test_a.py", "src/mime/nodes/body/_gravity.py"], (False, False, False),
     PY313, PY313),
    # The pins and CI start all three.
    ("pull_request", True, ["pyproject.toml"], (True, True, True), [], []),
    ("pull_request", True, ["uv.lock", "docs/a.md"], (True, True, True), [], []),
    ("pull_request", True, [".github/workflows/ci.yml"], (True, True, True), [], []),
    ("pull_request", True, [".github/actions/unit-tube/action.yml"], (True, True, True), [], []),
    ("pull_request", True, ["scripts/check_shards.py"], (True, True, True), [], []),
    ("pull_request", True, ["tests/test_ci_lane.py"], (True, True, True), [], []),
    # An adapter starts its upstream's lane alone, and Python 3.13 stays out.
    ("pull_request", True, ["src/mime/build/_maddening.py"], (True, False, False), PY313, PY313),
    ("pull_request", True, ["src/mime/core/_node.py"], (True, False, False), PY313, PY313),
    # The modules that call into MADDENING (build the graph, its edge specs, its verification machinery) are adapters too,
    # lab_fixture/ among them (installed in every lane).
    ("pull_request", True, ["src/mime/build/_construct.py"], (True, False, False), PY313, PY313),
    ("pull_request", True, ["src/mime/core/ledger.py", "src/mime/testing/_nodes.py"], (True, False, False), PY313, PY313),
    ("pull_request", True, ["lab_fixture/src/mime_lab_fixture/__init__.py"], (True, False, False), PY313, PY313),
    ("pull_request", True, ["src/mime/materials/_read.py"], (False, True, False), PY313, PY313),
    ("pull_request", True, ["src/mime/materials/_resolver.py", "src/mime/materials/_records.py"], (False, True, False),
     PY313, PY313),
    # Not an adapter: a file beside one, a script CI does not call, the shards' table.
    ("pull_request", True, ["src/mime/build/_plan.py", "scripts/mrs_budget.py", "tests/shard_weights.json",
                            "tests/duration_allowlist.txt"], (False, False, False), PY313, PY313),
    # Not an adapter either: a node's declaration (metadata, parameter specs) and a type annotation of its base class.
    ("pull_request", True, ["src/mime/nodes/body/_gravity.py", "src/mime/build/_ledger.py",
                            "src/mime/registry/_entries.py"], (False, False, False), PY313, PY313),
    # Unreadable paths count as everything changed.
    ("pull_request", True, None, (True, True, True), [], []),
])
def test_the_lanes_a_run_starts(event, full, changed, lanes, exclude, lane_exclude):
    made = boundary.decide(event, full, changed)
    assert (made.maddening, made.openmaterica, made.python313) == lanes
    assert (made.exclude, made.lane_exclude) == (exclude, lane_exclude)


def test_the_script_writes_the_four_outputs_the_count_job_names(tmp_path, monkeypatch):
    output = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(boundary, "changed_paths", lambda: ["src/mime/materials/_read.py"])
    assert boundary.main(["--event", "pull_request", "--full", "true"]) == 0
    written = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert set(written) == {"maddening", "openmaterica", "exclude", "lane_exclude"}
    assert (written["maddening"], written["openmaterica"]) == ("false", "true")
    assert json.loads(written["exclude"]) == PY313 == json.loads(written["lane_exclude"])


def test_every_path_the_filter_names_exists():
    for path in (*boundary.PINS, *boundary.CI, *boundary.MADDENING_ADAPTERS, *boundary.OPENMATERICA_ADAPTERS):
        assert (ROOT / path).exists(), f"{path} is named by scripts/ci_boundary.py and is not in the repository"


def _called_scripts() -> set[str]:
    """The scripts the workflows name outside comments (a comment may mention one that is called nowhere)."""
    found: set[str] = set()
    for path in sorted((ROOT / ".github").rglob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("#"):
                found |= set(re.findall(r"\bscripts/[A-Za-z0-9_]+\.(?:py|sh)\b", line))
    return found


def test_the_scripts_the_workflows_call_are_the_scripts_the_filter_names():
    """A script a workflow comes to call and the filter lacks would change CI without starting the lanes; one the filter
    names and no workflow calls would start them for nothing."""
    named = {path for path in boundary.CI_SCRIPTS if path.startswith("scripts/")}
    called = _called_scripts()
    assert called - named == set(), f"called by a workflow and not in CI_SCRIPTS: {sorted(called - named)}"
    assert named - called == set(), f"in CI_SCRIPTS and called by no workflow: {sorted(named - called)}"


def test_the_one_test_file_a_workflow_runs_as_a_script_is_named():
    text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / ".github").rglob("*.yml"))
    assert "tests/test_ci_lane.py" in text and "tests/test_ci_lane.py" in boundary.CI_SCRIPTS


@pytest.mark.parametrize("upstream, adapters", [("maddening", boundary.MADDENING_ADAPTERS),
                                                ("materica", boundary.OPENMATERICA_ADAPTERS)])
def test_each_adapter_imports_its_upstream(upstream, adapters):
    for path in adapters:
        files = sorted((ROOT / path).rglob("*.py")) if path.endswith("/") else [ROOT / path]
        assert any(UPSTREAM_IMPORT[upstream].search(each.read_text(encoding="utf-8")) for each in files), (
            f"{path} is named an adapter to {upstream} and imports none of it")


def test_every_file_that_imports_openmaterica_is_an_adapter():
    importers = {str(path.relative_to(ROOT)) for path in SRC.rglob("*.py")
                 if UPSTREAM_IMPORT["materica"].search(path.read_text(encoding="utf-8"))}
    assert importers == set(boundary.OPENMATERICA_ADAPTERS)


def _maddening_calls(path: Path) -> bool:
    """Whether the module imports one of the names by which MIME calls into MADDENING (``boundary.MADDENING_CALLS``):
    read from its syntax tree, so that a multi-line import and an import inside a function count."""
    calls = set(boundary.MADDENING_CALLS)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "maddening":
            if any(name.name in calls for name in node.names) or any(
                    node.module == call or node.module.startswith(call + ".") for call in calls):
                return True
        if isinstance(node, ast.Import) and any(
                name.name == call or name.name.startswith(call + ".") for name in node.names for call in calls):
            return True
    return False


def _covered(path: str) -> bool:
    return any(path == each or (each.endswith("/") and path.startswith(each)) for each in boundary.MADDENING_ADAPTERS)


def test_every_module_that_calls_into_maddening_is_an_adapter_and_each_adapter_is_one():
    """The rule in ``scripts/ci_boundary.py``: outside ``nodes/`` (a node's declaration: metadata, parameter specs, the
    boundary inputs it names), a module that imports MADDENING's graph manager, its specs, its warning, its compliance
    internals or its verification machinery is an adapter, and ``lab_fixture/`` is one wholesale. Both directions, so that a
    new caller is listed and a listed file that stops calling is dropped."""
    callers = {rel for path in [*SRC.rglob("*.py"), *(ROOT / "lab_fixture").rglob("*.py")]
               if not (rel := str(path.relative_to(ROOT))).startswith("src/mime/nodes/") and _maddening_calls(path)}
    unlisted = sorted(path for path in callers if not _covered(path))
    assert unlisted == [], f"call into MADDENING and are no adapter (add them to MADDENING_ADAPTERS): {unlisted}"
    stale = sorted(path for path in boundary.MADDENING_ADAPTERS if not path.endswith("/") and path not in callers)
    assert stale == [], f"named adapters that call into MADDENING no longer: {stale}"


def test_control_the_rule_reads_a_call_and_leaves_out_a_declaration(tmp_path):
    """A planted module of each kind: the calling one is read as a caller, the declaring one is not (so the test above
    can fail both ways)."""
    calls = tmp_path / "calls.py"
    calls.write_text("from maddening import (\n    GraphManager,\n)\n", encoding="utf-8")
    declares = tmp_path / "declares.py"
    declares.write_text("from maddening.core.compliance.metadata import NodeMeta\n"
                        "from maddening.core.node import SimulationNode\n", encoding="utf-8")
    verifies = tmp_path / "verifies.py"
    verifies.write_text("from maddening.testing import verification\n", encoding="utf-8")
    assert _maddening_calls(calls) and _maddening_calls(verifies) and not _maddening_calls(declares)


def test_maddening_is_not_imported_across_most_of_the_package():
    """The filter stands for the boundary with MADDENING because the modules that call into it are a minority of the
    package. If most of ``src/mime`` came to import it, the pins-and-CI-and-adapters filter would stand for little, and
    the choice would be reopened."""
    files = list(SRC.rglob("*.py"))
    importers = [path for path in files if UPSTREAM_IMPORT["maddening"].search(path.read_text(encoding="utf-8"))]
    assert 0 < len(importers) / len(files) < 0.5, f"{len(importers)} of {len(files)} files import maddening"
    adapters = {path for path in boundary.MADDENING_ADAPTERS if path.startswith("src/")}
    assert {str(path.relative_to(ROOT)) for path in importers} >= adapters
    assert len(adapters) / len(files) < 0.1, "the adapters are no small part of the package"


def workflow() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def test_the_count_job_runs_the_boundary_script_with_the_event_and_the_mode():
    job = workflow()["jobs"]["shards"]
    (step,) = [each for each in job["steps"] if each.get("id") == "boundary"]
    assert step["run"] == 'python3 scripts/ci_boundary.py --event "$EVENT" --full "$FULL"'
    assert step["env"] == {"EVENT": "${{ github.event_name }}", "FULL": "${{ steps.mode.outputs.full }}"}
    checkout = next(each for each in job["steps"] if str(each.get("uses", "")).startswith("actions/checkout@"))
    assert checkout["with"]["fetch-depth"] == 2, "the step diffs the merge commit against its base's tip"


@pytest.mark.parametrize("lane", ["maddening-main", "openmaterica-main"])
def test_the_two_lanes_keep_no_compilation_cache_and_their_duration_gate_warns(lane):
    jobs = workflow()["jobs"]
    steps = jobs[f"{lane}-tests"]["steps"]
    text = json.dumps(steps)
    for forbidden in ("actions/cache", "compilation_cache_mode", "prune_jax_cache"):
        assert forbidden not in text, f"{lane}'s shards use {forbidden}: the lane keeps no compilation cache"
    gate = next(each for each in steps if each.get("name") == "Test durations")
    assert " --warn-only " in gate["run"], f"{lane}'s duration gate fails instead of warning"
    # The programs the shard's own workers and child processes share stay in the run (tests/_child_cache.py).
    tests = next(each for each in steps if each.get("id") == "tests")
    assert "JAX_COMPILATION_CACHE_DIR" in tests["env"]


def test_python_3_13_keeps_no_compilation_cache_and_its_duration_gate_warns_and_the_others_do_not():
    steps = workflow()["jobs"]["default-tests"]["steps"]
    cc = next(each for each in steps if each.get("id") == "cc")
    assert '--keeps-cache "${{ matrix.python-version != \'3.13\' }}"' in cc["run"]
    gate = next(each for each in steps if each.get("name") == "Test durations")
    assert gate["env"]["WARN_ONLY"] == "${{ matrix.python-version == '3.13' && '--warn-only' || '' }}"
    assert "$WARN_ONLY" in gate["run"]


def moved_lane_problems(jobs: dict) -> list[str]:
    """How a red ``-main`` lane stays visible: the job under the lane's name is ``continue-on-error`` on a pull request
    alone (its shards and completion job are, always), so on a push and a manual run it turns the run red, and it reads
    the shard jobs' own conclusions (a shard that died before it uploaded a report is no report to miss)."""
    found: list[str] = []
    for lane in ("maddening-main", "openmaterica-main"):
        job = jobs[lane]
        if job.get("continue-on-error") != "${{ github.event_name == 'pull_request' }}":
            found.append(f"{lane}: its job's continue-on-error is {job.get('continue-on-error')!r}, not the pull request alone")
        if jobs[f"{lane}-tests"].get("continue-on-error") is not True:
            found.append(f"{lane}: its shards are not continue-on-error, so a red shard fails the run before the lane reports")
        reads = [step for step in job["steps"] if f'--prefix "{lane} tests (shard "' in str(step.get("run", ""))]
        if len(reads) != 1 or reads[0].get("if") != "${{ !cancelled() }}":
            found.append(f"{lane}: its job does not read its shard jobs' conclusions after a red step")
    return found


def test_a_red_moved_lane_turns_a_push_run_red():
    assert moved_lane_problems(workflow()["jobs"]) == []


@pytest.mark.parametrize("plant", ["job-always-continues", "shards-stop-continuing", "shard-read-dropped"])
def test_control_each_way_a_red_moved_lane_could_hide_is_found(plant):
    jobs = workflow()["jobs"]
    if plant == "job-always-continues":
        jobs["maddening-main"]["continue-on-error"] = True
    elif plant == "shards-stop-continuing":
        del jobs["openmaterica-main-tests"]["continue-on-error"]
    else:
        jobs["maddening-main"]["steps"] = [step for step in jobs["maddening-main"]["steps"]
                                           if "maddening-main tests (shard" not in str(step.get("run", ""))]
    assert moved_lane_problems(jobs) != []
