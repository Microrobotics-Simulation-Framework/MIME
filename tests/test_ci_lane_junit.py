"""The evidence counts page reads the default lane's JUnit reports, and CI uploads them.

A pytest control cannot run a workflow, so this reads ``ci.yml`` as PyYAML parses it (the model is
``test_ci_docs_artifact.py``): a function returns the workflow's problems, a test requires the real file to have none,
and a parametrised test applies a fault to a deep copy and requires the named problem. The workflow and the faults are
fixed, so every fault is reported on every run. What it holds, in the ``default`` job, in this order:

- the shards' reports are downloaded, their newest attempts kept, and the XML gathered into one directory;
- that directory is uploaded as the lane's JUnit artifact (``if-no-files-found: error``), so an empty lane fails;
- the steps that make the directory run after a red or failed shard too (``!cancelled()``, in a lane that ran), and
  they are **run**: the newest-attempt step and the gather step, over a downloaded directory that holds a shard's
  older and newer attempt, leave the newer attempt's report in the directory and the older one nowhere, so a glob
  that reads only an older attempt (or one directory) fails;
- the documentation pages are built after that, with ``--junit`` that same directory (the page then reads each cited
  test's outcome in the files the artifact holds; that no build step skips the evidence, or any other check, is the one
  check over every workflow, ``test_ci_wiring.py``);
- and the lane's device is the one the page prints, ``cpu``: ``JAX_PLATFORMS`` in the workflow's environment, with
  no job or step of the default lane setting it otherwise.

What it does not hold: that the gathered files hold the tests the page cites (the build fails on a cited test the
report does not hold only as "not run"), nor that a hosted runner has no GPU."""

import copy
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from mime.docs._evidence_page import LANE_DEVICE

CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
Workflow = dict[str, Any]
JUNIT = "$RUNNER_TEMP/junit"


def load() -> Workflow:
    loaded: Workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    return loaded


def directory(text: str) -> str:
    return re.sub(r"\$\{\{\s*runner\.temp\s*\}\}|\$\{RUNNER_TEMP\}", "$RUNNER_TEMP", text).strip().strip("\"'")


ROOT = CI.parents[2]
#: Run after a red or failed shard too, and, in a reduced run (the label ``ready`` absent), only for the lane that ran
#: (``test_ci_wiring.LANE_RAN``).
GATE = "${{ !cancelled() && env.MIME_CI_RUNS == 'true' }}"
LANE = "durations-default-python-3.12"
#: A downloaded directory: shard 1 ran twice (its second attempt's report is the newer), shard 2 once.
DOWNLOADED = {f"{LANE}-shard-1-attempt-1": ("test-results-shard1.xml", "older"),
              f"{LANE}-shard-1-attempt-2": ("test-results-shard1.xml", "newer"),
              f"{LANE}-shard-2-attempt-1": ("test-results-shard2.xml", "only")}
GATHERED = {"test-results-shard1.xml": "newer", "test-results-shard2.xml": "only"}


def gather_problems(steps: list[Workflow], scratch: Path) -> list[str]:
    """The newest-attempt step and the gather step of the default job, each under ``!cancelled()`` and then run, in
    bash as GitHub runs it, over a downloaded directory holding an older and a newer attempt of a shard: what the
    directory the page reads and the artifact uploads then holds."""
    pruning = [s for s in steps if "latest_attempt.py" in str(s.get("run", ""))]
    gathering = [s for s in steps if "test-results-shard" in str(s.get("run", "")) and "junit" in str(s.get("run", ""))]
    if len(pruning) != 1 or len(gathering) != 1:
        return [f"the default job has {len(pruning)} newest-attempt steps and {len(gathering)} gather steps; one each"]
    found = [f"the step '{step.get('name')}' does not run under !cancelled(), so a red shard leaves the page and the "
             f"artifact without the reports" for step in (pruning[0], gathering[0]) if step.get("if") != GATE]
    scratch = Path(tempfile.mkdtemp(dir=scratch))
    shards = scratch / "shards"
    for name, (report, content) in DOWNLOADED.items():
        (shards / name).mkdir(parents=True)
        (shards / name / report).write_text(content, encoding="utf-8")
    for step in (pruning[0], gathering[0]):
        done = subprocess.run([shutil.which("bash") or "bash", "-e", "-c", str(step["run"])], cwd=ROOT, check=False,
                              capture_output=True, text=True, env={"PATH": os.defpath, "RUNNER_TEMP": str(scratch)})
        if done.returncode != 0:
            found.append(f"the step '{step.get('name')}' fails over a downloaded directory of a re-run shard: "
                         f"{(done.stdout + done.stderr).strip()[-200:]}")
    kept = sorted(path.name for path in shards.iterdir())
    if kept != sorted(name for name in DOWNLOADED if not name.endswith("shard-1-attempt-1")):
        found.append(f"the downloaded directory still holds an older attempt, or lost a shard: {kept}")
    gathered = {path.name: path.read_text(encoding="utf-8") for path in sorted((scratch / "junit").glob("*"))}
    if gathered != GATHERED:
        found.append(f"the gathered directory holds {gathered}, not each shard's newest attempt's report {GATHERED}")
    return found


def junit_problems(workflow: Workflow, scratch: Path | None = None) -> list[str]:
    job = (workflow.get("jobs") or {}).get("default")
    if not isinstance(job, dict):
        return ["the workflow has no job 'default'"]
    steps: list[Workflow] = job.get("steps") or []

    def at(match: Any) -> int | None:
        return next((i for i, step in enumerate(steps) if match(step)), None)

    found: list[str] = []
    download = at(lambda s: str(s.get("uses", "")).startswith("actions/download-artifact@")
                  and "durations-default" in str((s.get("with") or {}).get("pattern", "")))
    newest = at(lambda s: "latest_attempt.py" in str(s.get("run", "")))
    gather = at(lambda s: "test-results-shard" in str(s.get("run", "")) and "junit" in str(s.get("run", "")))
    upload = at(lambda s: str(s.get("uses", "")).startswith("actions/upload-artifact@")
                and directory(str((s.get("with") or {}).get("path", ""))) == JUNIT)
    build = at(lambda s: "mime docs build" in str(s.get("run", "")))
    names = {"download": download, "newest-attempt": newest, "gather": gather, "upload": upload, "build": build}
    found += [f"the default job has no step that does '{name}'" for name, index in names.items() if index is None]
    if found:
        return found
    assert download is not None and newest is not None and gather is not None and upload is not None
    assert build is not None
    if not download < newest < gather < upload < build:
        found.append("the default job's steps are not in the order: download, newest attempt, gather, upload, build")
    uploaded = steps[upload]
    if (uploaded.get("with") or {}).get("if-no-files-found") != "error":
        found.append("the JUnit upload does not set if-no-files-found: error, so a lane with no report uploads nothing "
                     "and passes")
    run = str(steps[build].get("run", ""))
    given = re.search(r"mime docs build\b[^\n]*?--junit[ =](\S+)", run.replace("\\\n", " "))
    if given is None or directory(given.group(1)) != JUNIT:
        found.append(f"the build does not read the gathered reports: it passes --junit {given.group(1) if given else '(none)'}, "
                     f"and the upload takes {JUNIT}")
    if scratch is not None:
        found += gather_problems(steps, scratch)
    environments = [workflow.get("env") or {}] + [(j.get("env") or {}) for key, j in workflow["jobs"].items()
                                                 if key in ("default", "default-tests")]
    environments += [s.get("env") or {} for key in ("default", "default-tests") for s in workflow["jobs"][key]["steps"]]
    devices = [str(e["JAX_PLATFORMS"]) for e in environments if "JAX_PLATFORMS" in e]
    if (workflow.get("env") or {}).get("JAX_PLATFORMS") != LANE_DEVICE:
        found.append(f"the workflow's JAX_PLATFORMS is not {LANE_DEVICE!r}, the device the page prints for the lane")
    if any(device != LANE_DEVICE for device in devices):
        found.append(f"a job or step of the default lane sets JAX_PLATFORMS to "
                     f"{sorted({device for device in devices if device != LANE_DEVICE})}, not {LANE_DEVICE!r}")
    return found


def test_the_workflow_uploads_the_default_lanes_junit_and_the_build_reads_it_on_the_cpu(tmp_path):
    assert junit_problems(load(), tmp_path) == []


def drop(name_part: str):
    def fault(workflow: Workflow) -> None:
        steps = workflow["jobs"]["default"]["steps"]
        workflow["jobs"]["default"]["steps"] = [s for s in steps if name_part not in str(s.get("name", ""))]
    return fault


def build_first(workflow: Workflow) -> None:
    steps = workflow["jobs"]["default"]["steps"]
    build = next(s for s in steps if "mime docs build" in str(s.get("run", "")))
    steps.remove(build)
    steps.insert(0, build)


def edit_build(old: str, new: str):
    def fault(workflow: Workflow) -> None:
        step = next(s for s in workflow["jobs"]["default"]["steps"] if "mime docs build" in str(s.get("run", "")))
        assert old in step["run"]
        step["run"] = step["run"].replace(old, new)
    return fault


def step_of(workflow: Workflow, name_part: str) -> Workflow:
    return next(s for s in workflow["jobs"]["default"]["steps"] if name_part in str(s.get("name", "")))


def edit_step(name_part: str, old: str, new: str):
    def fault(workflow: Workflow) -> None:
        step = step_of(workflow, name_part)
        assert old in step["run"]
        step["run"] = step["run"].replace(old, new)
    return fault


def without_gate(name_part: str):
    def fault(workflow: Workflow) -> None:
        step_of(workflow, name_part).pop("if")
    return fault


def set_env(workflow: Workflow) -> None:
    workflow["env"]["JAX_PLATFORMS"] = "cuda"


def step_env(workflow: Workflow) -> None:
    workflow["jobs"]["default-tests"]["env"]["JAX_PLATFORMS"] = "cuda"


def tolerate_empty(workflow: Workflow) -> None:
    upload = next(s for s in workflow["jobs"]["default"]["steps"] if "upload-artifact" in str(s.get("uses", ""))
                  and "junit" in str(s["with"]["path"]))
    upload["with"]["if-no-files-found"] = "warn"


FAULTS = [
    (drop("Upload the default lane's JUnit"), "the default job has no step that does 'upload'"),
    (drop("Gather the default lane's JUnit"), "the default job has no step that does 'gather'"),
    (build_first, "the default job's steps are not in the order"),
    (edit_build('--junit "$RUNNER_TEMP/junit"', ""), "the build does not read the gathered reports: it passes --junit (none)"),
    (edit_build('"$RUNNER_TEMP/junit"', '"$RUNNER_TEMP/report"'), "the build does not read the gathered reports"),
    (edit_step("Gather", '"$RUNNER_TEMP"/shards/*/test-results', '"$RUNNER_TEMP"/shards/*-attempt-1/test-results'),
     "the gathered directory holds"),
    (edit_step("Gather", '"$RUNNER_TEMP"/shards/*/test-results', '"$RUNNER_TEMP"/shards/*-attempt-2/test-results'),
     "the gathered directory holds"),
    (edit_step("Keep each", "latest_attempt.py", "latest_attempt.py --help #"),
     "the downloaded directory still holds an older attempt"),
    (without_gate("Gather"), "the step 'Gather the default lane's JUnit reports' does not run under !cancelled()"),
    (without_gate("Keep each"), "the step 'Keep each shard's newest attempt' does not run under !cancelled()"),
    (tolerate_empty, "the JUnit upload does not set if-no-files-found: error"),
    (set_env, "the workflow's JAX_PLATFORMS is not 'cpu'"),
    (step_env, "a job or step of the default lane sets JAX_PLATFORMS to ['cuda']"),
]


@pytest.mark.parametrize(("fault", "problem"), FAULTS, ids=[f"{n}-{p[:40]}" for n, (_, p) in enumerate(FAULTS)])
def test_a_planted_fault_in_the_workflow_is_reported(fault, problem, tmp_path):
    workflow = copy.deepcopy(load())
    fault(workflow)
    assert any(problem in found for found in junit_problems(workflow, tmp_path)), junit_problems(workflow, tmp_path)
