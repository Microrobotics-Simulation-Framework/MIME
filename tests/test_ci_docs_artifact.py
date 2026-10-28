"""N6: the built documentation pages are uploaded only after they were scanned, and a finding fails the job.

Generated pages are never committed, so the confidentiality scan of the tracked tree does not see them, and they can
hold text found in no tracked file (a captured message, a runtime path). CI builds them, scans the built directory with
the same script, key and fork exception as the tree's scan, and uploads the directory as a run artifact. A pytest
control can show the scan's exit status on a directory (``test_check_confidential.py``, N5's output case) and cannot
show that a workflow step is skipped, so this reads the workflow, as PyYAML parses it:

- the build, the scan and the upload are steps of one job, in that order, over one directory;
- the scan's failure fails the job: no ``continue-on-error``, no ``|| true`` or ``set +e``, no shell of its own, and no
  condition that skips it when the upload would run;
- the upload runs only after every earlier step succeeded: no ``always()``, ``failure()`` or ``!cancelled()``, and no
  earlier step lets its failure through (``continue-on-error``);
- and the scan step's own script, run in bash as GitHub runs it on a directory, exits non-zero on a planted private
  path and zero on a clean one (X-3).

The model is OpenMaterica's ``tests/unit/test_ci_workflows.py``: a function returns the workflow's problems, a test
requires the real file to have none, and a parametrised test applies a fault to a deep copy and requires the named
problem. Each fault is reported on every run: the workflow and the faults are fixed, so there is no detection rate to
measure."""

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
Workflow = dict[str, Any]
RUNNER_TEMP = re.compile(r"\$\{RUNNER_TEMP\}|\$RUNNER_TEMP|\$\{\{\s*runner\.temp\s*\}\}")
NEUTERING = re.compile(r"\|\|\s*(?:true|:)(?![\w-])|\bset\s+\+e\b")
#: A step condition that runs when every earlier step succeeded and in no other case.
AFTER_SUCCESS = (None, "${{ success() }}", "success()")
#: A condition that also runs after a failure, or that never runs.
FAILURE_SIDE = re.compile(r"always\(\)|failure\(\)|!\s*cancelled\(\)|cancelled\(\)")
MIME = "Microrobotics-Simulation-Framework/MIME"


def load(path: Path = CI) -> Workflow:
    loaded: Workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def directory(text: str) -> str:
    """A path in a script or a ``with`` value, as one spelling: the runner's temporary directory as ``$RUNNER_TEMP``,
    quotes removed."""
    return RUNNER_TEMP.sub("$RUNNER_TEMP", text).strip().strip("\"'")


def scan_directory(step: Workflow) -> str | None:
    """The argument of ``--directory`` in the scan step's script, if it has one."""
    found = re.search(r"check_confidential\.py\b[^\n]*?--directory[ =](\S+)", str(step.get("run", "")))
    return directory(found.group(1)) if found else None


def build_directory(step: Workflow) -> str | None:
    found = re.search(r"mime docs build\b[^\n]*?--out[ =](\S+)", str(step.get("run", "")))
    return directory(found.group(1)) if found else None


def is_scan(step: Workflow) -> bool:
    return "check_confidential.py" in str(step.get("run", "")) and "--directory" in str(step.get("run", ""))


def is_upload(step: Workflow) -> bool:
    return str(step.get("uses", "")).startswith("actions/upload-artifact@")


def is_build(step: Workflow) -> bool:
    return "mime docs build" in str(step.get("run", ""))


def condition_problem(where: str, step: Workflow) -> str | None:
    """Why the step may not run when every earlier step succeeded (the scan), or may run when one failed (the
    upload), by its ``if``: ``None`` when it runs after success only."""
    held = step.get("if")
    if held in AFTER_SUCCESS:
        return None
    if re.fullmatch(r"\$\{\{\s*(?:false|0|'')\s*\}\}|false|0", str(held).strip()):
        return f"{where}: runs under 'if: {held}', which never runs it"
    if FAILURE_SIDE.search(str(held)):
        return f"{where}: runs under 'if: {held}', which runs it after an earlier step failed"
    return f"{where}: runs only if {held!r}, which is not 'every earlier step succeeded'"


def docs_artifact_problems(workflow: Workflow) -> list[str]:
    """Everything wrong with how the workflow builds, scans and uploads the pages."""
    found: list[str] = []
    placed: list[tuple[str, int, Workflow]] = []
    for job_id, job in (workflow.get("jobs") or {}).items():
        for number, step in enumerate(job.get("steps") or []):
            if is_scan(step) or is_upload(step) or is_build(step):
                placed.append((job_id, number, step))
    scans = [place for place in placed if is_scan(place[2])]
    uploads = [place for place in placed if is_upload(place[2])]
    builds = [place for place in placed if is_build(place[2])]
    if len(builds) != 1:
        found.append(f"{len(builds)} steps build the pages (mime docs build); one does")
    if len(scans) != 1:
        found.append(f"{len(scans)} steps scan a built directory (check_confidential.py --directory); one does")
    if len(uploads) != 1:
        found.append(f"{len(uploads)} steps upload an artifact; one uploads the built pages")
    if found:
        return found
    (scan_job, scan_at, scan), (up_job, up_at, upload), (build_job, build_at, build) = scans[0], uploads[0], builds[0]
    if len({scan_job, up_job, build_job}) != 1:
        found.append(f"the build ({build_job}), the scan ({scan_job}) and the upload ({up_job}) are steps of different "
                     f"jobs; the upload would not wait for the scan's verdict")
        return found
    if not build_at < scan_at < up_at:
        found.append(f"the steps are in the order build {build_at}, scan {scan_at}, upload {up_at}; the build, then "
                     f"the scan, then the upload")
    built, scanned, uploaded = build_directory(build), scan_directory(scan), \
        directory(str((upload.get("with") or {}).get("path", "")))
    if not (built and built == scanned == uploaded):
        found.append(f"the build writes {built!r}, the scan reads {scanned!r} and the upload takes {uploaded!r}; all "
                     f"three are the same directory")
    where = f"the scan step {scan.get('name')!r}"
    if scan.get("continue-on-error"):
        found.append(f"{where}: continue-on-error hides its failure from the job")
    if "shell" in scan:
        found.append(f"{where}: sets its own shell; it runs in GitHub's bash -e")
    if NEUTERING.search(str(scan.get("run", ""))):
        found.append(f"{where}: its script holds an '|| true' or a 'set +e'")
    problem = condition_problem(where, scan)
    if problem:
        found.append(problem)
    problem = condition_problem(f"the upload step {upload.get('name')!r}", upload)
    if problem:
        found.append(problem)
    for step in (workflow["jobs"][up_job].get("steps") or [])[:up_at]:
        if step.get("continue-on-error"):
            found.append(f"the step {step.get('name')!r} comes before the upload and has continue-on-error, so the "
                         f"upload runs after its failure")
    return found


def scan_behaviour_problems(workflow: Workflow, scratch: Path) -> list[str]:
    """The scan step's script, run in ``bash -e`` as GitHub runs it, on a directory of pages: a planted private path
    must fail it and a clean directory must pass, with the repository's own script and no key, as a pull request from a
    fork runs (the one run that may go without). A planted word cannot be shown here without the real key."""
    steps = [step for job in (workflow.get("jobs") or {}).values() for step in job.get("steps") or []
             if is_scan(step)]
    if len(steps) != 1:
        return [f"{len(steps)} scan steps to run; one"]
    event = scratch / "event.json"
    event.write_text(json.dumps({"pull_request": {"head": {"repo": {"full_name": "someone/MIME"}}}}), encoding="utf-8")
    found: list[str] = []
    for label, content, wanted in (("clean", "# Relative to docs/entries.\n", 0),
                                   ("a planted private path", "# In " + "/" + "home/someone/checkout/docs.\n", 1)):
        pages = scratch / label.replace(" ", "-") / "docs-pages"
        pages.mkdir(parents=True)
        (pages / "page.md").write_text(content, encoding="utf-8")
        script = scratch / f"{label.replace(' ', '-')}.sh"
        script.write_text(str(steps[0]["run"]), encoding="utf-8")
        done = subprocess.run([shutil.which("bash") or "bash", "-e", str(script)], capture_output=True, text=True,
                              check=False, cwd=ROOT,
                              env={"PATH": "/usr/bin:/bin:/usr/local/bin", "RUNNER_TEMP": str(pages.parent),
                                   "GITHUB_EVENT_NAME": "pull_request", "GITHUB_EVENT_PATH": str(event),
                                   "GITHUB_REPOSITORY": MIME})
        if done.returncode != wanted:
            found.append(f"the scan step on {label} exits {done.returncode}, not {wanted}: {done.stderr.strip()[:200]}")
    return found


def test_the_workflow_builds_scans_and_uploads_the_pages_in_that_order():
    assert docs_artifact_problems(load()) == []


def test_the_scan_step_fails_on_a_planted_private_path_and_passes_a_clean_directory(tmp_path):
    assert scan_behaviour_problems(load(), tmp_path) == []


def faulted(change) -> Workflow:
    workflow = copy.deepcopy(load())
    change(workflow)
    return workflow


def steps_of(workflow: Workflow) -> list[Workflow]:
    return workflow["jobs"]["default"]["steps"]


def scan_of(workflow: Workflow) -> Workflow:
    return next(step for step in steps_of(workflow) if is_scan(step))


def upload_of(workflow: Workflow) -> Workflow:
    return next(step for step in steps_of(workflow) if is_upload(step))


def move_scan_after_upload(workflow: Workflow) -> None:
    steps = steps_of(workflow)
    scan = scan_of(workflow)
    steps.remove(scan)
    steps.insert(steps.index(upload_of(workflow)) + 1, scan)


def scan_in_a_second_job(workflow: Workflow) -> None:
    scan = scan_of(workflow)
    steps_of(workflow).remove(scan)
    workflow["jobs"]["scan_elsewhere"] = {"runs-on": "ubuntu-latest", "steps": [scan]}


def set_on_scan(**changed: object):
    return lambda workflow: scan_of(workflow).update(changed)


def set_on_upload(**changed: object):
    return lambda workflow: upload_of(workflow).update(changed)


def suffix_scan(text: str):
    return lambda workflow: scan_of(workflow).update(run=str(scan_of(workflow)["run"]).rstrip("\n") + text + "\n")


def scan_another_directory(workflow: Workflow) -> None:
    scan = scan_of(workflow)
    scan["run"] = str(scan["run"]).replace("docs-pages", "somewhere-else")


def scan_without_a_directory(workflow: Workflow) -> None:
    scan = scan_of(workflow)
    scan["run"] = str(scan["run"]).replace(' --directory "$RUNNER_TEMP/docs-pages"', "")


def upload_another_directory(workflow: Workflow) -> None:
    upload_of(workflow)["with"]["path"] = "${{ runner.temp }}/somewhere-else"


def build_continues_on_error(workflow: Workflow) -> None:
    next(step for step in steps_of(workflow) if is_build(step))["continue-on-error"] = True


FAULTS = {
    "the-scan-moved-after-the-upload": (
        move_scan_after_upload, "the steps are in the order build"),
    "the-scan-removed": (
        lambda workflow: steps_of(workflow).remove(scan_of(workflow)), "0 steps scan a built directory"),
    "the-scan-in-a-second-job": (
        scan_in_a_second_job, "are steps of different jobs"),
    "continue-on-error-on-the-scan": (
        set_on_scan(**{"continue-on-error": True}), "continue-on-error hides its failure from the job"),
    "or-true-ending-the-scan": (
        suffix_scan(" || true"), "its script holds an '|| true' or a 'set +e'"),
    "set-plus-e-in-the-scan": (
        suffix_scan("; set +e"), "its script holds an '|| true' or a 'set +e'"),
    "the-scan-not-given-the-built-directory": (
        scan_without_a_directory, "0 steps scan a built directory"),
    "the-scan-given-another-directory": (
        scan_another_directory, "all three are the same directory"),
    "the-upload-takes-another-directory": (
        upload_another_directory, "all three are the same directory"),
    "always-on-the-upload": (
        set_on_upload(**{"if": "${{ always() }}"}), "which runs it after an earlier step failed"),
    "not-cancelled-on-the-upload": (
        set_on_upload(**{"if": "${{ !cancelled() }}"}), "which runs it after an earlier step failed"),
    "the-scan-under-if-false": (
        set_on_scan(**{"if": "${{ false }}"}), "which never runs it"),
    "the-scan-under-always": (
        set_on_scan(**{"if": "${{ always() }}"}), "which runs it after an earlier step failed"),
    "the-scan-in-its-own-shell": (
        set_on_scan(shell="sh"), "sets its own shell"),
    "the-build-continues-on-error": (
        build_continues_on_error, "has continue-on-error, so the upload runs after its failure"),
}


@pytest.mark.parametrize("fault", FAULTS, ids=list(FAULTS))
def test_a_fault_in_the_order_or_the_scan_is_reported_naming_it(fault):
    change, named = FAULTS[fault]
    problems = docs_artifact_problems(faulted(change))
    assert any(named in problem for problem in problems), problems


def test_a_scan_that_cannot_fail_is_caught_by_running_it_too(tmp_path):
    """X-3: the lexical faults are reported above; running the step's script finds a scan whose exit status is lost
    however it is written. ``|| true`` here, the clean directory passing and the planted path passing too."""
    problems = scan_behaviour_problems(faulted(suffix_scan(" || true")), tmp_path)
    assert problems and "exits 0, not 1" in problems[0]


def test_the_second_job_fault_leaves_a_scan_the_real_workflow_does_not_have():
    """The control is real: the faulted workflow has two jobs where the real one has none that scan."""
    real = load()
    assert not any(is_scan(step) for job_id, job in real["jobs"].items() if job_id != "default"
                   for step in job.get("steps") or [])
